"""Prepare vendor VLM artifacts from the same SmolVLM checkpoint as tinygrad.

The decoder is exported as its original Llama architecture. RKLLM's documented
embedding-input API accepts the projected image and text embeddings directly.
No weights, model layers, image tokens, or image-conditioned attention are dropped.
"""
import argparse, json, pickle, shutil
from pathlib import Path

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('stage',choices=['vision-onnx','vision-rknn','text-hf','text-rkllm'])
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--context',type=int,default=512,help='Vendor decoder capacity; toolkit supports multiples of 32 through 16384')
  parser.add_argument('--optimization-level',type=int,choices=(0,1),default=0,help='Published reference recipe is 0; 1 reproduces legacy exports')
  parser.add_argument('--flash-attention',action='store_true')
  parser.add_argument('--cpu-norm',action='store_true',help='Evaluate FP32 CPU LayerNorm for vision accuracy')
  parser.add_argument('--patch-only',action='store_true',help='Isolate the patch convolution when checking vendor accuracy')
  args = parser.parse_args()
  if not 32<=args.context<=16384 or args.context%32: parser.error('Vendor context must be a multiple of 32 between 32 and 16384')
  root = args.root.resolve()
  checkpoint, output = root/'hf',root/'vendor'
  output.mkdir(exist_ok=True)
  if args.stage == 'vision-onnx':
    import torch
    from transformers import Idefics3ForConditionalGeneration
    torch.set_num_threads(4)
    model = Idefics3ForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='eager').eval()
    class Encoder(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.vision, self.connector = model.model.vision_model,model.model.connector
      def forward(self,pixels): return self.connector(self.vision(pixels).last_hidden_state)
    torch.onnx.export(Encoder().eval(),torch.zeros(1,3,512,512),str(output/'vision.onnx'),
                      input_names=['pixels'],output_names=['image_embeddings'],opset_version=17,do_constant_folding=True)
  elif args.stage == 'vision-rknn':
    from rknn.api import RKNN
    op_target = None
    if args.cpu_norm:
      import onnx
      # RKNN op_target uses optimized graph operation names (reported in its
      # operation table), rather than the original ONNX LayerNormalization node.
      op_target = {node.name+'#1':'cpu' for node in onnx.load(str(output/'vision.onnx')).graph.node if node.op_type=='LayerNormalization'}
    rknn = RKNN(verbose=False)
    if rknn.config(target_platform='rk3588',enable_flash_attention=args.flash_attention,op_target=op_target):
      raise RuntimeError('RKNN config failed')
    outputs = ['/vision/embeddings/patch_embedding/Conv_output_0'] if args.patch_only else None
    if rknn.load_onnx(str(output/'vision.onnx'),outputs=outputs): raise RuntimeError('RKNN ONNX loading failed')
    if rknn.build(do_quantization=False): raise RuntimeError('RKNN FP16 vision build failed')
    path = output/('vision-patch.rknn' if args.patch_only else 'vision-cpu-norm.rknn' if args.cpu_norm else 'vision.rknn')
    if rknn.export_rknn(str(path)): raise RuntimeError('RKNN export failed')
    rknn.release()
  elif args.stage == 'text-hf':
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    dest = output/'text-hf'
    dest.mkdir(exist_ok=True)
    config = json.loads((checkpoint/'config.json').read_text())['text_config']
    # Keep the decoder's exact dimensions, RoPE, vocab, and normalization settings.
    config = {k:config[k] for k in ('hidden_size','intermediate_size','num_attention_heads','num_key_value_heads',
      'num_hidden_layers','rms_norm_eps','vocab_size','head_dim','rope_theta','max_position_embeddings','tie_word_embeddings',
      'bos_token_id','eos_token_id','attention_bias','mlp_bias','hidden_act')}
    config.update(model_type='llama',architectures=['LlamaForCausalLM'],torch_dtype='float16')
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json'):
      shutil.copyfile(checkpoint/name,dest/name)
    weights = {}
    with safe_open(checkpoint/'model.safetensors',framework='pt') as source:
      for name in source.keys():
        if name.startswith('model.text_model.'):
          weights[name.replace('model.text_model.','model.')] = source.get_tensor(name).half().contiguous()
        elif name == 'lm_head.weight': weights[name] = source.get_tensor(name).half().contiguous()
    save_file(weights,dest/'model.safetensors')
    fixture = root/'demo'
    ids = json.loads((fixture/'input_ids.json').read_text())
    embeddings = weights['model.embed_tokens.weight'][torch.tensor([ids])].float()
    image = torch.from_numpy(np.load(fixture/'image_embeddings.npy'))
    slots = torch.tensor(ids) == json.loads((checkpoint/'config.json').read_text())['image_token_id']
    embeddings[0,slots] = image[0]
    embeddings.numpy().tofile(fixture/'input_embeddings.f32')
    sample = {'inputs_embeds':embeddings,'attention_mask':torch.ones(1,len(ids),dtype=torch.int64),
              'position_ids':torch.arange(len(ids),dtype=torch.int64)[None]}
    with (output/'calibration-0.pkl').open('wb') as file: pickle.dump(sample,file)
    (output/'calibration.json').write_text(json.dumps([{'sample':'calibration-0.pkl','token_nums':len(ids)}])+'\n')
  else:
    from extra.rockchip.rkllm_reference import export_reference
    path = output/f'text-w8a8-opt{args.optimization_level}-ctx{args.context}.rkllm'
    metadata = export_reference(output/'text-hf',output/'calibration.json',path,context=args.context,
                                optimization_level=args.optimization_level,source_checkpoint=checkpoint/'model.safetensors')
    print(json.dumps({'event':'reference_exported',**metadata}),flush=True)

if __name__ == '__main__': main()
