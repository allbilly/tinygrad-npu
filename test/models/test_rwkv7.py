"""Independent RWKV7 equations from BlinkDL's official CUDA-free demo recurrence."""
import itertools, os, unittest
from unittest.mock import patch
import numpy as np
import torch
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestRWKV7(unittest.TestCase):
  def model(self):
    c = TransformerConfig(2,32,64,4,4,1e-5,96,8,10000.,0,8,max_context=24,rwkv7=(8,8,4,12))
    model = Transformer(c)
    rng = np.random.default_rng(707)
    for name,p in nn.state.get_state_dict(model).items():
      values = rng.normal(0,.09,p.shape).astype(np.float32)
      if 'norm' in name and name.endswith('weight') or 'time_mix_ln.weight' in name:values+=1
      if 'lerp' in name:values = rng.uniform(0,1,p.shape).astype(np.float32)
      p.replace(Tensor(values,device='CPU').realize())
    return model

  def reference(self, model, ids):
    c = model.config
    w = {k:torch.from_numpy(v.numpy()).float() for k,v in nn.state.get_state_dict(model).items()
         if 'cache_' not in k and 'recurrent_state' not in k}
    def norm(x,prefix):return torch.nn.functional.layer_norm(x,(c.dim,),w[prefix+'.weight'],w[prefix+'.bias'],c.norm_eps)
    x = norm(w['token_embd.weight'][torch.tensor(ids)],'token_embd_norm')
    B,T,C = x.shape
    v_first = torch.zeros_like(x)
    for layer in range(c.num_blocks):
      p = f'blk.{layer}.'
      def linear(x,name):return x@w[p+name+'.weight'].T
      h = norm(x,p+'attn_norm')
      delta = torch.cat((torch.zeros(B,1,C),h[:,:-1]),dim=1)-h
      xr,xw,xk,xv,xa,xg = [h+delta*w[p+'time_mix_lerp_fused.weight'][i] for i in range(6)]
      r,k,v = linear(xr,'time_mix_receptance'),linear(xk,'time_mix_key'),linear(xv,'time_mix_value')
      decay = torch.exp(-np.exp(-.5)*torch.sigmoid(linear(linear(xw,'time_mix_w1').tanh(),'time_mix_w2')+w[p+'time_mix_w0.weight']))
      a = torch.sigmoid(linear(linear(xa,'time_mix_a1'),'time_mix_a2')+w[p+'time_mix_a0.weight'])
      g = linear(torch.sigmoid(linear(xg,'time_mix_g1')),'time_mix_g2')
      kk = torch.nn.functional.normalize((k*w[p+'time_mix_k_k.weight']).reshape(B,T,c.n_heads,c.head_dim),dim=-1)
      k = k*(1+(a-1)*w[p+'time_mix_k_a.weight'])
      if layer:
        blend = torch.sigmoid(linear(linear(xv,'time_mix_v1'),'time_mix_v2')+w[p+'time_mix_v0.weight'])
        v = v+(v_first-v)*blend
      else:v_first = v
      extra = ((r*k*w[p+'time_mix_r_k.weight']).reshape(B,T,c.n_heads,c.head_dim).sum(-1,keepdim=True)
               *v.reshape(B,T,c.n_heads,c.head_dim))
      r,k,v,decay,a = [z.reshape(B,T,c.n_heads,c.head_dim) for z in (r,k,v,decay,a)]
      state,outputs = torch.zeros(B,c.n_heads,c.head_dim,c.head_dim),[]
      for t in range(T):
        ab = -kk[:,t].unsqueeze(-1)@(kk[:,t]*a[:,t]).unsqueeze(-2)
        vk = v[:,t].unsqueeze(-1)@k[:,t].unsqueeze(-2)
        state = state*decay[:,t].unsqueeze(-2)+state@ab+vk
        outputs.append((state@r[:,t].unsqueeze(-1)).squeeze(-1))
      attention = torch.nn.functional.group_norm(torch.stack(outputs,dim=1).reshape(B*T,C),c.n_heads,
                                                 w[p+'time_mix_ln.weight'],w[p+'time_mix_ln.bias'],c.head_dim*c.norm_eps)
      x = x+linear((attention.reshape(B,T,c.n_heads,c.head_dim)+extra).reshape(B,T,C)*g,'time_mix_output')
      h = norm(x,p+'attn_norm_2')
      delta = torch.cat((torch.zeros(B,1,C),h[:,:-1]),dim=1)-h
      x = x+linear(linear(h+delta*w[p+'channel_mix_lerp_k.weight'],'channel_mix_key').relu().square(),'channel_mix_value')
    return (norm(x[:,-1:],'output_norm')@w['output.weight'].T)[:,-1].numpy()

  def compare(self, model, device):
    ids = [[3,8,12,4,7,6]]
    for start,part in ((0,ids),(6,[[9]]),(7,[[11,5]])):
      if start:ids[0].extend(part[0])
      actual = model.logits(Tensor(part,device=device),start).numpy()
      np.testing.assert_allclose(actual,self.reference(model,ids),rtol=5e-5,atol=1e-5)
    restart = [[7,2,8]]
    np.testing.assert_allclose(model.logits(Tensor(restart,device=device),0).numpy(),self.reference(model,restart),rtol=5e-5,atol=1e-5)

  def test_cpu_prefill_cached_chunks_and_reset(self):
    self.compare(self.model(),'CPU')

  def test_generation_and_recurrent_prefix_reuse(self):
    model = self.model()
    prompt = [3,8,12,4,7,6]
    self.assertEqual(model.prefill_chunk_size(128),1)
    actual = list(itertools.islice(model.generate(prompt.copy(),chunk_size=4),3))
    expected = []
    for _ in range(3):expected.append(int(self.reference(model,[prompt+expected]).argmax()))
    self.assertEqual(actual,expected)
    cached = model._cached_tokens.copy()
    self.assertEqual(model.get_start_pos(cached+[5]),len(cached))
    self.assertEqual(model.get_start_pos(cached[:-1]+[1,5]),0)

  def test_gguf_loader_uses_wkv_head_size_and_layer_norm(self):
    model = self.model()
    state = {k:v.half().clone() for k,v in nn.state.get_state_dict(model).items()}
    # The converter emits unused first-layer value-mix aliases with ICLR rank.
    for suffix in ('0','1','2'):state['blk.0.time_mix_v'+suffix+'.weight'] = state['blk.0.time_mix_a'+suffix+'.weight']
    values = {'context_length':24,'embedding_length':32,'block_count':2,'attention.head_count':0,
              'attention.layer_norm_epsilon':1e-5,'wkv.head_size':8,'feed_forward_length':64,
              'attention.decay_lora_rank':8,'attention.iclr_lora_rank':8,
              'attention.value_residual_mix_lora_rank':4,'attention.gate_lora_rank':12}
    kv = {'general.architecture':'rwkv7','tokenizer.ggml.tokens':[str(i) for i in range(96)],
          **{'rwkv7.'+k:v for k,v in values.items()}}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=24,realize=True)
    self.assertEqual((loaded.config.n_heads,loaded.config.head_dim),(4,8))
    self.assertIsInstance(loaded.output_norm,nn.LayerNorm)
    self.assertTrue(loaded.has_recurrent_block)
    self.compare(loaded,'CPU')

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_prefill_and_cached_chunks_against_torch(self):
    from tinygrad.llm.kernels.mali import configure
    model = self.model()
    for p in nn.state.get_parameters(model):p.replace(p.half().to('CL').realize())
    info = configure(model,pack_weights=False)
    self.assertEqual(info['linear_layers'],1)
    self.assertEqual(info['generic_recurrent_blocks'],2)
    self.compare(model,'CL')
    prompt = [3,8,12,4,7,6]
    actual = list(itertools.islice(model.generate(prompt.copy(),chunk_size=4),2))
    for index,token in enumerate(actual):
      self.assertEqual(token,int(self.reference(model,[prompt+actual[:index]]).argmax()))
