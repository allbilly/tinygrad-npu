"""MiniCPM3 GGUF versus the published full K/V attention equations.

The native decoder absorbs K/V projections into latent attention; this reference
explicitly expands keys and values, including the shared rotary key.
"""
import math, os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestMiniCPM3(unittest.TestCase):
  def fixture(self):
    config = TransformerConfig(2,32,64,4,4,1e-5,64,12,10000.,4,8,max_context=16,q_lora_rank=16,kv_lora_rank=8)
    model = Transformer(config)
    rng = np.random.default_rng(303)
    state = {}
    for name,param in nn.state.get_state_dict(model).items():
      value = rng.normal(0,.1,param.shape).astype(np.float16)
      if 'norm' in name:value+=1
      state[name] = Tensor(value,device='CPU').realize()
    for i in range(2):
      p = f'blk.{i}.'
      k,v = state.pop(p+'attn_k_b.weight'),state.pop(p+'attn_v_b.weight')
      state[p+'attn_kv_b.weight'] = k.transpose(1,2).cat(v,dim=1).reshape(64,8).contiguous().realize()
    state['rope_factors_short.weight'] = Tensor([1.1,2.3],device='CPU').realize()
    state['rope_factors_long.weight'] = Tensor([2.,4.],device='CPU').realize()
    return state

  def load(self,state):
    # The real GGUF has factor tensors but no scaling.type, value_length, or MiniCPM scales.
    values = {'context_length':16,'block_count':2,'embedding_length':32,'feed_forward_length':64,
              'attention.head_count':4,'attention.head_count_kv':4,'attention.layer_norm_rms_epsilon':1e-5,
              'attention.key_length':12,'rope.dimension_count':4,'attention.q_lora_rank':16,'attention.kv_lora_rank':8}
    kv = {'general.architecture':'minicpm3','tokenizer.ggml.tokens':[str(i) for i in range(64)],
          **{'minicpm3.'+k:v for k,v in values.items()}}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})), \
         patch('tinygrad.llm.model.gguf_shard',return_value=dict(state)):
      return Transformer.from_gguf('fixture',max_context=16,realize=True)[0]

  def reference(self,state,tokens):
    w = {name:value.numpy().astype(np.float32) for name,value in state.items()}
    def norm(x,name):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*w[name]
    def linear(x,name):return x@w[name+'.weight'].T
    def softmax(x):
      values = np.exp(x-x.max(-1,keepdims=True))
      return values/values.sum(-1,keepdims=True)
    size = len(tokens)
    phase = np.arange(size,dtype=np.float32)[:,None]/w['rope_factors_short.weight']*10000.**(-np.arange(0,4,2,dtype=np.float32)/4)
    def rope(x):
      a,b = np.split(x,2,axis=-1)
      return np.concatenate((a*np.cos(phase)-b*np.sin(phase),b*np.cos(phase)+a*np.sin(phase)),axis=-1)
    x = w['token_embd.weight'][tokens]*12
    for layer in range(2):
      p = f'blk.{layer}.'
      h = norm(x,p+'attn_norm.weight')
      q = linear(norm(linear(h,p+'attn_q_a'),p+'attn_q_a_norm.weight'),p+'attn_q_b').reshape(size,4,12).transpose(1,0,2)
      compressed = linear(h,p+'attn_kv_a_mqa')
      latent = norm(compressed[:,:8],p+'attn_kv_a_norm.weight')
      expanded = linear(latent,p+'attn_kv_b').reshape(size,4,16).transpose(1,0,2)
      q = np.concatenate((q[:,:,:8],rope(q[:,:,8:])),axis=-1)
      shared = np.broadcast_to(rope(compressed[:,8:][None]),(4,size,4))
      keys = np.concatenate((expanded[:,:,:8],shared),axis=-1)
      scores = (q@keys.transpose(0,2,1))/math.sqrt(12)
      row,col = np.triu_indices(size,1)
      scores[:,row,col] = -np.inf
      attention = (softmax(scores)@expanded[:,:,8:]).transpose(1,0,2).reshape(size,32)
      x = x+linear(attention,p+'attn_output')*(1.4/math.sqrt(2))
      h = norm(x,p+'ffn_norm.weight')
      gate,up = linear(h,p+'ffn_gate'),linear(h,p+'ffn_up')
      x = x+linear(gate/(1+np.exp(-gate))*up,p+'ffn_down')*(1.4/math.sqrt(2))
    return linear(norm(x[-1],'output_norm.weight')/(32/256),'output')

  def compare(self,device):
    state = self.fixture()
    model = self.load(state)
    self.assertEqual(model.config.v_head_dim,8)
    self.assertEqual(model.config.embedding_scale,12)
    self.assertAlmostEqual(model.config.residual_scale,1.4/math.sqrt(2))
    self.assertAlmostEqual(model.config.logit_scale,32/256)
    np.testing.assert_array_equal(model.config.rope_factors,state['rope_factors_short.weight'].numpy())
    if device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for param in nn.state.get_parameters(model):param.to_('CL')
      configure(model,pack_weights=False)
    for ids,start in (([3,7,2],0),([9],3)):
      expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
      actual = model.logits(Tensor([ids],device=device),start).numpy().ravel()
      np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-4)

  def test_cpu_full_projection_prefill_and_cached_decode(self):self.compare('CPU')

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_full_projection_prefill_and_cached_decode(self):self.compare('CL')
