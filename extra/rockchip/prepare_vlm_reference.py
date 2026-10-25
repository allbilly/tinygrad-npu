"""Create shared VLM inputs and FP32 Transformers references using installed tools.

Run with a Python that already has torch, transformers, safetensors and cv2.
This benchmark deliberately disables image splitting and uses one square 512px
RGB image. Both engines receive the exact same saved normalized pixels and IDs.
"""
import argparse, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--image',type=Path,required=True)
  parser.add_argument('--question',default='What is shown in this image?')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--tokens',type=int,default=32)
  parser.add_argument('--reuse-vision',action='store_true',help='Reuse independently computed reference image features')
  args = parser.parse_args()
  args.output.mkdir(parents=True,exist_ok=True)
  torch.set_num_threads(4)
  image = cv2.imread(str(args.image))
  if image is None: raise ValueError('Cannot decode image')
  rgb = cv2.cvtColor(cv2.resize(image,(512,512),interpolation=cv2.INTER_LANCZOS4),cv2.COLOR_BGR2RGB)
  pixels = ((rgb.astype(np.float32)/np.float32(255)-np.float32(.5))/np.float32(.5)).transpose(2,0,1)[None].copy()
  np.save(args.output/'pixels.npy',pixels)
  np.save(args.output/'rgb.npy',rgb)
  tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
  image_prompt = '<fake_token_around_image><global-img>'+'<image>'*64+'<fake_token_around_image>'
  prompt = '<|im_start|>User:'+image_prompt+args.question+'<end_of_utterance>\nAssistant:'
  ids = tokenizer.encode(prompt,add_special_tokens=False)
  (args.output/'input_ids.json').write_text(json.dumps(ids)+'\n')
  report = {'complete':False,'image':str(args.image.resolve()),'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
            'question':args.question,'prompt':prompt,'input_ids':ids,'image_tokens':64,'splitting':False,
            'preprocessing':'cv2 INTER_LANCZOS4, square 512x512, RGB, FP32 (uint8/255-.5)/.5',
            'tokens':args.tokens,'reference':'Transformers FP32, eager bidirectional vision and causal text attention'}
  start = time.perf_counter()
  model = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint,torch_dtype=torch.float32,attn_implementation='eager').eval()
  report['load_seconds'] = time.perf_counter()-start
  print('reference loaded',flush=True)
  with torch.inference_mode():
    start = time.perf_counter()
    vision = model.model.vision_model(torch.from_numpy(pixels)).last_hidden_state
    embeddings = model.model.connector(vision)
    report['vision_seconds'] = time.perf_counter()-start
    np.save(args.output/'vision.npy',vision.numpy())
    np.save(args.output/'image_embeddings.npy',embeddings.numpy())
    tokens = torch.tensor([ids])
    merged = model.model.text_model.embed_tokens(tokens).clone()
    merged[0,tokens[0]==model.config.image_token_id] = embeddings[0]
    merged.float().numpy().tofile(args.output/'input_embeddings.f32')
    # Compare the public multimodal path, not just the independently called pieces.
    result = model(input_ids=tokens,use_cache=True,**({'image_hidden_states':embeddings} if args.reuse_vision else
                                                    {'pixel_values':torch.from_numpy(pixels[:,None])}))
    np.save(args.output/'logits.npy',result.logits[:,-1].numpy())
    logits = [result.logits[0,-1].numpy().copy()]
    first = int(result.logits[0,-1].argmax())
    generated = [first]
    cache = result.past_key_values
    for _ in range(args.tokens-1):
      result = model(input_ids=torch.tensor([[generated[-1]]]),past_key_values=cache,use_cache=True)
      cache = result.past_key_values
      generated.append(int(result.logits[0,-1].argmax()))
      logits.append(result.logits[0,-1].numpy().copy())
    np.save(args.output/'teacher_logits.npy',np.stack(logits))
  eos = model.generation_config.eos_token_id
  stops = {eos} if isinstance(eos,int) else set(eos or [])
  quality_positions = next((i+1 for i,token in enumerate(generated) if token in stops),len(generated))
  report.update(complete=True,generated_ids=generated,generated_text=tokenizer.decode(generated),
                quality_positions_before_or_at_eos=quality_positions)
  (args.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:report[k] for k in ('vision_seconds','generated_text','complete')}),flush=True)

if __name__ == '__main__': main()
