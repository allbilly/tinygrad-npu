"""Check a Spark GGUF baseline's greedy outputs with an independent PyTorch decoder.

Uses the pinned official HF config from the benchmark catalog and streams GGUF
weights one block at a time. Mirrors modeling_spark.py's equations, without using
tinygrad's Transformer, rotary, masking or attention implementations. This checks
implementation agreement, not comparable quality against RKLLM W8A8.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path


def check(catalog:dict, key:str, baseline:dict, tokens:int) -> dict:
  os.environ['DEV'] = 'CPU'
  import torch
  from tinygrad import Device
  from tinygrad.helpers import prod
  from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor
  torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
  row = next(row for row in catalog['models'] if row['model']+'/'+row['size']==key)
  c = row['config']
  if c['model_type']!='spark2_5' or c['hidden_act']!='gelu' or not c['headwise_attn_output_gate'] or c['gate_attn_act_mode']!='sigmoid':
    raise ValueError('Expected the official Spark2_5 architecture')
  if not baseline.get('complete'):raise ValueError('Baseline is incomplete')
  path = Path(baseline['checkpoint'])
  with path.open('rb') as source:digest = hashlib.file_digest(source,'sha256').hexdigest()
  if digest!=baseline['checkpoint_sha256']:raise ValueError('Checkpoint differs from the measured baseline')
  metadata,entries = gguf_parse(path)
  fields = {'block_count':'num_hidden_layers','embedding_length':'hidden_size','feed_forward_length':'intermediate_size',
            'attention.head_count':'num_attention_heads','attention.head_count_kv':'num_key_value_heads','attention.key_length':'head_dim'}
  if metadata['general.architecture']!='spark2_5':raise ValueError('Expected Spark GGUF')
  if any(metadata['spark2_5.'+field]!=c[name] for field,name in fields.items()):raise ValueError('GGUF and pinned HF config differ')
  if metadata['spark2_5.attention.sliding_window_pattern']!=[kind=='sliding_attention' for kind in c['layer_types']]:
    raise ValueError('GGUF and HF layer policies differ')
  for kind,suffix in (('full_attention',''),('sliding_attention','_swa')):
    rope = c['rope_parameters'][kind]
    if (metadata['spark2_5.rope.freq_base'+suffix]!=rope['rope_theta'] or
        metadata['spark2_5.rope.dimension_count'+suffix]!=int(c['head_dim']*rope['partial_rotary_factor'])):
      raise ValueError('GGUF and HF rotary policies differ')
  def weight(name):
    data,shape,kind = entries[name]
    decoded = ggml_data_to_tensor(data.to('CPU'),prod(shape),kind).reshape(shape).half().contiguous().realize()
    return torch.from_numpy(decoded.numpy()).float()
  def norm(x,w):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c['rms_norm_eps'])*w
  prompt = baseline['input_ids']
  expected = [trial['generated_ids'] for trial in baseline['trials'] if trial['warmup'] is False]
  if not expected or not 1<=tokens<=len(expected[0]) or any(ids!=expected[0] for ids in expected):
    raise ValueError('Need deterministic baseline outputs and a valid check length')
  n_heads,n_kv,head = c['num_attention_heads'],c['num_key_value_heads'],c['head_dim']
  caches:dict[int,tuple] = {}
  actual,margins = [],[]
  start = 0
  begin = time.monotonic()
  with torch.inference_mode():
    for step in range(tokens):
      ids = prompt if step==0 else actual[-1:]
      x = weight('token_embd.weight')[torch.tensor([ids])]
      positions = torch.arange(start,start+len(ids),dtype=torch.float32)
      for index,kind in enumerate(c['layer_types']):
        prefix = f'blk.{index}.'
        # Only one block's dequantized weights are resident at a time.
        w = {name.removeprefix(prefix):weight(name) for name in entries if name.startswith(prefix)}
        h = norm(x,w['attn_norm.weight'])
        q,k,v = (h@w['attn_qkv.weight'].T).split((n_heads*head,n_kv*head,n_kv*head),dim=-1)
        q,k,v = (a.reshape(1,len(ids),heads,head).transpose(1,2) for a,heads in ((q,n_heads),(k,n_kv),(v,n_kv)))
        rope = c['rope_parameters'][kind]
        width = int(head*rope['partial_rotary_factor'])
        angles = positions[:,None]/rope['rope_theta']**(torch.arange(0,width,2,dtype=torch.float32)/width)[None,:]
        def rotate(a):
          first,second = a[...,:width].chunk(2,dim=-1)
          return torch.cat((first*angles.cos()-second*angles.sin(),second*angles.cos()+first*angles.sin(),a[...,width:]),dim=-1)
        q,k = rotate(q),rotate(k).half()
        v = v.half()
        if index in caches:
          k,v = torch.cat((caches[index][0],k),dim=2),torch.cat((caches[index][1],v),dim=2)
        caches[index] = k,v
        k,v = (a.float().repeat_interleave(n_heads//n_kv,dim=1) for a in (k,v))
        keys = torch.arange(start+len(ids))
        visible = keys[None,:]<=positions[:,None]
        if kind=='sliding_attention':visible &= keys[None,:]>positions[:,None]-c['sliding_window']
        attention = ((q@k.transpose(-1,-2)/math.sqrt(head)).masked_fill(~visible,float('-inf')).softmax(-1)@v)
        attention *= (h@w['attn_gate.weight'].T).sigmoid().transpose(1,2).unsqueeze(-1)
        x = x+attention.transpose(1,2).reshape(1,len(ids),-1)@w['attn_output.weight'].T
        h = norm(x,w['ffn_norm.weight'])
        x = x+(torch.nn.functional.gelu(h@w['ffn_gate.weight'].T,approximate='none')*(h@w['ffn_up.weight'].T))@w['ffn_down.weight'].T
        del w
        Device['CPU'].allocator.free_cache()
      logits = norm(x[:,-1:],weight('output_norm.weight'))@weight('output.weight' if 'output.weight' in entries else 'token_embd.weight').T
      top = logits[0,0].topk(2)
      actual.append(int(top.indices[0]))
      margins.append(float(top.values[0]-top.values[1]))
      start+=len(ids)
      print(json.dumps({'step':step,'actual':actual[-1],'expected':expected[0][step],'margin':margins[-1]}),flush=True)
  return {'complete':True,'passed':actual==expected[0][:tokens],'checked_tokens':tokens,'prompt_tokens':len(prompt),
          'actual_ids':actual,'expected_ids':expected[0][:tokens],'reference_top2_margins':margins,'checkpoint_sha256':digest,
          'hf_config_revision':row['revision'],'hf_config_url':row['config_url'],'seconds':time.monotonic()-begin,
          'cache_dtype':'FP16, matching the measured backend','reference_arithmetic':'FP32 PyTorch; FP16-decoded GGUF weights',
          'torch_version':torch.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'rkllm_quality_qualified':False}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('catalog','baseline','report'):parser.add_argument('--'+name,type=Path,required=True)
  parser.add_argument('--model',required=True,help='Catalog model/size key')
  parser.add_argument('--tokens',type=int,default=4)
  args = parser.parse_args()
  result = check(json.loads(args.catalog.read_text()),args.model,json.loads(args.baseline.read_text()),args.tokens)
  args.report.write_text(json.dumps(result,indent=2)+'\n')
  if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
