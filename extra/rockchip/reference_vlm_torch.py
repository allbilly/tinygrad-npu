"""Independent FP32 Torch reference using the unchanged checkpoint and saved API input.

Only installed Torch, NumPy, the safetensor reader and tokenizer are required.
Validate the implementation against the earlier Transformers fixture first.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tinygrad import dtypes
from tinygrad.nn.state import safe_load
from onpu.vlm import decode_image, load_tokenizer, render_messages


class Reference:
  def __init__(self, path):
    self.config = json.loads((path/'config.json').read_text())
    self.weights = {}
    for name,value in safe_load(path/'model.safetensors').items():
      array = ((value.bitcast(dtypes.uint16).numpy().astype(np.uint32)<<16).view(np.float32)
               if value.dtype is dtypes.bfloat16 else value.numpy().astype(np.float32))
      self.weights[name] = torch.from_numpy(array)
    self.cache = []

  def linear(self, x, name): return F.linear(x,self.weights[name+'.weight'],self.weights.get(name+'.bias'))

  def vision(self, pixels):
    cfg = self.config['vision_config']
    prefix = 'model.vision_model.'
    x = F.conv2d(pixels,self.weights[prefix+'embeddings.patch_embedding.weight'],
                 self.weights[prefix+'embeddings.patch_embedding.bias'],stride=cfg['patch_size']).flatten(2).transpose(1,2)
    x = x+self.weights[prefix+'embeddings.position_embedding.weight']
    def norm(x,name):return F.layer_norm(x,(cfg['hidden_size'],),self.weights[name+'.weight'],self.weights[name+'.bias'],cfg['layer_norm_eps'])
    heads,dim = cfg['num_attention_heads'],cfg['hidden_size']//cfg['num_attention_heads']
    for index in range(cfg['num_hidden_layers']):
      p = prefix+f'encoder.layers.{index}.'
      h = norm(x,p+'layer_norm1')
      q,k,v = (self.linear(h,p+'self_attn.'+name+'_proj').reshape(1,-1,heads,dim).transpose(1,2) for name in ('q','k','v'))
      attended = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)
      x = x+self.linear(attended,p+'self_attn.out_proj')
      x = x+self.linear(F.gelu(self.linear(norm(x,p+'layer_norm2'),p+'mlp.fc1'),approximate='tanh'),p+'mlp.fc2')
    x = norm(x,prefix+'post_layernorm')
    scale = self.config['scale_factor']
    side = math.isqrt(x.shape[1])
    x = x.reshape(1,side//scale,scale,side//scale,scale,cfg['hidden_size']).permute(0,1,3,2,4,5).reshape(1,-1,cfg['hidden_size']*scale*scale)
    return self.linear(x,'model.connector.modality_projection.proj')

  def text(self, x, start=0):
    cfg = self.config['text_config']
    heads,kv,dim = cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim']
    def norm(x,name): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+cfg['rms_norm_eps'])*self.weights[name+'.weight']
    angles = torch.arange(start,start+x.shape[1],dtype=torch.float32)[:,None]/(
      cfg['rope_theta']**(torch.arange(0,dim,2,dtype=torch.float32)/dim))[None]
    cosine,sine = (torch.cat((value,value),dim=-1)[None,None] for value in (angles.cos(),angles.sin()))
    def rope(value):return value*cosine+torch.cat((-value[...,dim//2:],value[...,:dim//2]),-1)*sine
    for index in range(cfg['num_hidden_layers']):
      p = f'model.text_model.layers.{index}.'
      h = norm(x,p+'input_layernorm')
      q = rope(self.linear(h,p+'self_attn.q_proj').reshape(1,-1,heads,dim).transpose(1,2))
      k = rope(self.linear(h,p+'self_attn.k_proj').reshape(1,-1,kv,dim).transpose(1,2))
      v = self.linear(h,p+'self_attn.v_proj').reshape(1,-1,kv,dim).transpose(1,2)
      if start:
        k,v = torch.cat((self.cache[index][0],k),2),torch.cat((self.cache[index][1],v),2)
        self.cache[index] = k,v
      elif index<len(self.cache): self.cache[index] = k,v
      else:self.cache.append((k,v))
      attended = F.scaled_dot_product_attention(q,k.repeat_interleave(heads//kv,1),v.repeat_interleave(heads//kv,1),is_causal=start==0)
      x = x+self.linear(attended.transpose(1,2).reshape_as(x),p+'self_attn.o_proj')
      h = norm(x,p+'post_attention_layernorm')
      x = x+self.linear(F.silu(self.linear(h,p+'mlp.gate_proj'))*self.linear(h,p+'mlp.up_proj'),p+'mlp.down_proj')
    return self.linear(norm(x[:,-1],'model.text_model.norm'),'lm_head')

  def merged(self, ids, image):
    x = F.embedding(torch.tensor([ids]),self.weights['model.text_model.embed_tokens.weight']).clone()
    selected = torch.tensor(ids)==self.config['image_token_id']
    x[0,selected] = image[0]
    return x


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--input',type=Path,required=True)
  parser.add_argument('--fixture',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  torch.set_num_interop_threads(1)
  model = Reference(args.checkpoint)
  report = {'complete':False,'reference':'independent Torch FP32 checkpoint math','trials':[]}
  def error(actual,expected):return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
  with torch.inference_mode():
    image = model.vision(torch.from_numpy(np.load(args.fixture/'pixels.npy')))
    image_error = error(image.numpy(),np.load(args.fixture/'image_embeddings.npy'))
    ids = json.loads((args.fixture/'input_ids.json').read_text())
    logits = model.text(model.merged(ids,image))
    logits_error = error(logits.numpy(),np.load(args.fixture/'logits.npy'))
    assert image_error<1e-4 and logits_error<1e-4,(image_error,logits_error)
    report['transformers_fixture_validation'] = {'image_nrmse':image_error,'logits_nrmse':logits_error}
    print(report['transformers_fixture_validation'],flush=True)
    payload = json.loads(args.input.read_text())
    rendered,url = render_messages(payload['messages'],64)
    tokenizer = load_tokenizer(args.checkpoint)
    ids = tokenizer.encode(rendered)
    pixels = np.frombuffer(decode_image(url),dtype=np.uint8).reshape(512,512,3).transpose(2,0,1)[None].astype(np.float32)
    pixels = (pixels/np.float32(255)-np.float32(.5))/np.float32(.5)
    image = model.vision(torch.from_numpy(pixels))
    start = time.perf_counter()
    logits = model.text(model.merged(ids,image))
    report['prefill_seconds'] = time.perf_counter()-start
    logits.numpy().tofile(args.report.with_suffix('.logits.f32'))
    tokens = [int(logits.argmax())]
    for position in range(len(ids),len(ids)+payload['max_tokens']-1):
      if tokenizer.is_end(tokens[-1]):break
      values = F.embedding(torch.tensor([[tokens[-1]]]),model.weights['model.text_model.embed_tokens.weight'])
      logits = model.text(values,position)
      tokens.append(int(logits.argmax()))
    report.update(complete=True,prompt_tokens=len(ids),tokens=tokens,answer=tokenizer.decode(tokens),
                  input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
