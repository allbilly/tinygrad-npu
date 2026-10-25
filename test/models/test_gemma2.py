"""Gemma2 decoder reference: HF modeling_gemma2.py and llama.cpp conversion/gemma.py.

GGUF norm multipliers already contain the HF +1 offset. KV storage here matches
the backend's FP16 cache; the independent PyTorch computation uses FP32 otherwise.
"""
import math, os, unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestGemma2(unittest.TestCase):
  def model(self, **overrides):
    c = TransformerConfig(2,40,96,2,1,1e-6,72,16,10000.,16,16,max_context=16,
                          embedding_scale=math.sqrt(40),ffn_activation='gelu_tanh',post_norm=True,
                          attn_logit_softcap=1.25,final_logit_softcap=.9,sliding_window=4,sliding_window_pattern=2)
    model = Transformer(replace(c,**overrides))
    rng = np.random.default_rng(17)
    for name,p in nn.state.get_state_dict(model).items():
      data = rng.normal(0,.17,p.shape).astype(np.float32)
      if 'norm' in name:data+=1
      p.replace(Tensor(data,device='CPU').realize())
    return model

  def reference(self, model, ids):
    c = model.config
    w = {k:torch.from_numpy(v.numpy()).float() for k,v in nn.state.get_state_dict(model).items() if 'cache' not in k}
    def norm(x,key):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c.norm_eps)*w[key]
    x = w['token_embd.weight'][torch.tensor(ids)]*c.embedding_scale
    length = x.shape[1]
    pos = torch.arange(length,dtype=torch.float32)
    angles = pos[:,None]/(c.rope_theta**(torch.arange(0,c.head_dim,2,dtype=torch.float32)/c.head_dim))[None,:]
    def rotate(a):
      first,second = a.chunk(2,dim=-1)
      return torch.cat((first*angles.cos()-second*angles.sin(),second*angles.cos()+first*angles.sin()),dim=-1)
    for i in range(c.num_blocks):
      prefix = f'blk.{i}.'
      h = norm(x,prefix+'attn_norm.weight')
      q = rotate((h@w[prefix+'attn_q.weight'].T).reshape(1,-1,c.n_heads,c.head_dim).transpose(1,2))
      k = rotate((h@w[prefix+'attn_k.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2)).half().float()
      v = (h@w[prefix+'attn_v.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2).half().float()
      k,v = (a.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1) for a in (k,v))
      scores = ((q@k.transpose(-1,-2)/math.sqrt(c.head_dim))/c.attn_logit_softcap).tanh()*c.attn_logit_softcap
      visible = pos[:,None]>=pos[None,:]
      if model.blk[i].config.sliding_window:visible &= pos[None,:]>pos[:,None]-c.sliding_window
      attention = (scores.masked_fill(~visible,float('-inf')).softmax(-1)@v).transpose(1,2).reshape(1,-1,c.n_heads*c.head_dim)
      x = x+norm(attention@w[prefix+'attn_output.weight'].T,prefix+'attn_post_norm.weight')
      h = norm(x,prefix+'ffn_norm.weight')
      activation = torch.nn.functional.gelu(h@w[prefix+'ffn_gate.weight'].T,approximate='tanh')
      x = x+norm((activation*(h@w[prefix+'ffn_up.weight'].T))@w[prefix+'ffn_down.weight'].T,prefix+'ffn_post_norm.weight')
    logits = norm(x[:,-1:],'output_norm.weight')@w['output.weight'].T
    return ((logits/c.final_logit_softcap).tanh()*c.final_logit_softcap)[:,-1].numpy()

  def test_full_decoder_and_sliding_cache_against_torch(self):
    model = self.model()
    ids = [[7,12,4,8,2,11]]
    for step in range(3):
      start = 0 if step==0 else len(ids[0])-1
      tokens = ids if step==0 else [[ids[0][-1]]]
      actual = model.logits(Tensor(tokens,device='CPU'),start).numpy()
      expected = self.reference(model,ids)
      np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=5e-6)
      self.assertGreater(np.max(np.abs(actual)),.8)
      self.assertLessEqual(np.max(np.abs(actual)),.900001)
      ids[0].append(5+step)

  def test_softcap_never_uses_uncapped_attention_adapter(self):
    model = self.model()
    def forbidden(*args):raise AssertionError('Uncapped attention adapter was called')
    for block in model.blk:block.prefill_attention=block.decode_attention=forbidden
    ids = [[3,4,5]]
    actual = model.logits(Tensor(ids,device='CPU'),0).numpy()
    np.testing.assert_allclose(actual,self.reference(model,ids),rtol=3e-5,atol=5e-6)

  def test_gguf_norm_offsets_and_gemma_metadata(self):
    model = self.model()
    c = model.config
    state = {k.replace('attn_post_norm','post_attention_norm').replace('ffn_post_norm','post_ffw_norm'):v.clone()
             for k,v in nn.state.get_state_dict(model).items()}
    kv = {'general.architecture':'gemma2','tokenizer.ggml.tokens':[str(i) for i in range(c.vocab_size)],
          'gemma2.attention.head_count':c.n_heads,'gemma2.attention.head_count_kv':c.n_kv_heads,
          'gemma2.attention.key_length':16,'gemma2.attention.value_length':16,
          'gemma2.block_count':2,'gemma2.embedding_length':40,'gemma2.feed_forward_length':96,
          'gemma2.attention.layer_norm_rms_epsilon':1e-6,'gemma2.context_length':16,
          'gemma2.attn_logit_softcapping':1.25,'gemma2.final_logit_softcapping':.9,'gemma2.attention.sliding_window':4}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    self.assertEqual((loaded.config.ffn_activation,loaded.config.post_norm),('gelu_tanh',True))
    self.assertEqual(loaded.config.embedding_scale,math.sqrt(40))
    self.assertEqual([b.config.sliding_window for b in loaded.blk],[4,0])
    np.testing.assert_array_equal(loaded.blk[0].attn_post_norm.weight.numpy(),model.blk[0].attn_post_norm.weight.half().numpy())
    ids = [[3,4,5]]
    np.testing.assert_allclose(loaded.logits(Tensor(ids,device='CPU'),0).numpy(),self.reference(loaded,ids),rtol=3e-5,atol=5e-6)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_decoder_softcaps_and_cached_token_against_torch(self):
    from tinygrad.llm.kernels.mali import configure
    model = self.model(dim=96,hidden_dim=192,n_heads=4,n_kv_heads=2,vocab_size=128,embedding_scale=math.sqrt(96))
    for p in nn.state.get_parameters(model):p.replace(p.half().to('CL').realize())
    info = configure(model)
    self.assertEqual(info['attention_blocks'],0)
    ids = [[7,12,4,8,2,11]]
    actual = model.logits(Tensor(ids,device='CL'),0).numpy()
    np.testing.assert_allclose(actual,self.reference(model,ids),rtol=5e-5,atol=1e-5)
    start = len(ids[0])
    ids[0].append(5)
    actual = model.logits(Tensor([[5]],device='CL'),start).numpy()
    np.testing.assert_allclose(actual,self.reference(model,ids),rtol=5e-5,atol=1e-5)
