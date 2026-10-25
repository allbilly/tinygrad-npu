"""HF grouped value heads versus llama.cpp's tiled GGUF order.

Reference equations follow transformers' torch_recurrent_gated_delta_rule;
the value-head permutation follows conversion/qwen.py's V reorder contract.
"""
import math, unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TransformerConfig


class TestGatedDeltaLayout(unittest.TestCase):
  def fixture(self, value_heads, convert=True):
    ssm = SSMConfig(conv_kernel=3,state_size=8,group_count=2,time_step_rank=value_heads,inner_size=value_heads*8)
    c = TransformerConfig(1,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=24,ssm=ssm)
    block = GatedDeltaNetBlock(c,ssm)
    rng = np.random.default_rng(3504)
    weights = {name:rng.normal(0,.12,p.shape).astype(np.float32) for name,p in nn.state.get_state_dict(block).items()}
    for name in weights:
      if 'norm.weight' in name:weights[name]+=1
    weights['ssm_a'] = -rng.uniform(.5,2,value_heads).astype(np.float32)
    order = np.arange(value_heads).reshape(2,value_heads//2).T.reshape(-1)
    tiled = {name:value.copy() for name,value in weights.items()}
    if convert:
      for name in ('attn_qkv.weight','ssm_conv1d.weight'):
        # Q/K channels keep their original order; only the value suffix moves.
        qk = 2*ssm.group_count*ssm.state_size
        v = weights[name][qk:].reshape(value_heads,8,*weights[name].shape[1:])[order]
        tiled[name] = np.concatenate((weights[name][:qk],v.reshape(weights[name][qk:].shape)))
      tiled['attn_gate.weight'] = weights['attn_gate.weight'].reshape(value_heads,8,-1)[order].reshape(weights['attn_gate.weight'].shape)
      for name in ('ssm_alpha.weight','ssm_beta.weight','ssm_dt.bias','ssm_a'):tiled[name] = weights[name][order]
      tiled['ssm_out.weight'] = weights['ssm_out.weight'].reshape(32,value_heads,8)[:,order].reshape(weights['ssm_out.weight'].shape)
    for name,p in nn.state.get_state_dict(block).items():p.replace(Tensor(tiled[name],device='CPU').realize())
    return block,weights,ssm

  def reference(self, x, w, ssm, eps):
    B,T,_ = x.shape
    def sigmoid(x):return 1/(1+np.exp(-x))
    def silu(x):return x*sigmoid(x)
    # Native generic GDN rounds its normalized input and final gated output to FP16.
    h = x.astype(np.float16).astype(np.float32)
    qkv = h@w['attn_qkv.weight'].T
    window = np.concatenate((np.zeros((B,ssm.conv_kernel-1,qkv.shape[-1]),np.float32),qkv),axis=1)
    conv = silu(sum(window[:,i:i+T]*w['ssm_conv1d.weight'][:,i] for i in range(ssm.conv_kernel)))
    q,k,v = np.split(conv,[ssm.group_count*ssm.state_size,2*ssm.group_count*ssm.state_size],axis=-1)
    def normalized(a):
      a = a.reshape(B,T,ssm.group_count,ssm.state_size)
      return a/np.sqrt((a*a).sum(-1,keepdims=True)+1e-6)
    factor = ssm.time_step_rank//ssm.group_count
    q,k = np.repeat(normalized(q),factor,axis=2),np.repeat(normalized(k),factor,axis=2)
    q = q/math.sqrt(ssm.state_size)
    v = v.reshape(B,T,ssm.time_step_rank,-1)
    beta = sigmoid(h@w['ssm_beta.weight'].T)
    g = np.logaddexp(0,h@w['ssm_alpha.weight'].T+w['ssm_dt.bias'])*w['ssm_a']
    decay = np.exp(g)
    state,outputs = np.zeros((B,ssm.time_step_rank,v.shape[-1],ssm.state_size),np.float32),[]
    for index in range(T):
      state = state*decay[:,index,:,None,None]
      predicted = (state*k[:,index,:,None,:]).sum(-1)
      delta = (v[:,index]-predicted)*beta[:,index,:,None]
      state = state+delta[:,:,:,None]*k[:,index,:,None,:]
      outputs.append((state*q[:,index,:,None,:]).sum(-1))
    core = np.stack(outputs,axis=1)
    core = core/np.sqrt((core*core).mean(-1,keepdims=True)+eps)*w['ssm_norm.weight']
    gate = silu(h@w['attn_gate.weight'].T).reshape(core.shape)
    output = (core*gate).astype(np.float16).astype(np.float32).reshape(B,T,-1)@w['ssm_out.weight'].T
    return output

  def test_grouped_reference_matches_tiled_gguf_prefill_cache_and_reset(self):
    # Ratio one covers the 0.8B case; ratio two covers the 4B checkpoint.
    for heads in (2,4):
      with self.subTest(value_heads=heads):
        block,w,ssm = self.fixture(heads)
        x = (np.random.default_rng(35).integers(-8,9,(1,9,32))/16).astype(np.float32)
        block._init_state(Tensor(x[:,:1],device='CPU'))
        start = 0
        for end in (6,7,9):
          actual = block._attention(Tensor(x[:,start:end],device='CPU'),start).numpy()
          expected = self.reference(x[:,:end],w,ssm,block.config.norm_eps)[:,start:end]
          np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=1e-5)
          start = end
        actual = block._attention(Tensor(x[:,:3],device='CPU'),0).numpy()
        np.testing.assert_allclose(actual,self.reference(x[:,:3],w,ssm,block.config.norm_eps),rtol=5e-5,atol=1e-5)

  def test_fixture_detects_omitted_value_head_reorder(self):
    block,w,ssm = self.fixture(4,convert=False)
    x = (np.random.default_rng(35).integers(-8,9,(1,6,32))/16).astype(np.float32)
    block._init_state(Tensor(x[:,:1],device='CPU'))
    wrong = block._attention(Tensor(x,device='CPU'),0).numpy()
    expected = self.reference(x,w,ssm,block.config.norm_eps)
    self.assertGreater(float(np.max(np.abs(wrong-expected))),1e-3)
