"""Capped-score attention must cap before masking, including cached/JIT paths."""
import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention


def reference(query, cache, length, cap, window=0):
  _,heads,rows,dim = query.shape
  kv_heads = cache.shape[2]
  out = np.zeros_like(query)
  for head in range(heads):
    keys,values = cache[:,0,head//(heads//kv_heads)].astype(np.float64)
    for row in range(rows):
      used = length-rows+row+1
      first = max(0,used-window) if window else 0
      scores = keys[first:used] @ query[0,head,row].astype(np.float64) / np.sqrt(dim)
      scores = np.tanh(scores/cap)*cap
      weights = np.exp(scores-scores.max())
      weights /= weights.sum()
      out[0,head,row] = weights @ values[first:used]
  return out


class TestSoftcappedCPUAttention(unittest.TestCase):
  def test_blocked_tail_grouped_decode_saturation_and_poisoned_mask(self):
    rng = np.random.default_rng(507)
    for rows,heads,kv_heads,dim,window,cap in ((17,4,2,16,0,1.25),(17,4,1,256,8,50.),
                                             (1,8,1,256,0,.125),(1,4,2,16,8,128.)):
      length,context = 37,64
      q = rng.normal(0,4,(1,heads,rows,dim)).astype(np.float32)
      kv = rng.normal(0,2,(2,1,kv_heads,context,dim)).astype(np.float16)
      kv[:,:,:,length:] = np.nan
      if window:kv[:,:,:,:length-rows+1-window] = np.nan
      expected = reference(q,kv,length,cap,window)
      for fast in (False,True):
        tq,tc = Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
        actual = prefill_attention(tq,tc,length,fast_softmax=fast,softcap=cap,window=window).numpy()
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-4)
        if rows==1:
          decoded = decode_attention(tq,tc,length,fast_softmax=fast,softcap=cap,window=window).numpy()
          np.testing.assert_allclose(decoded,expected,atol=2e-5,rtol=3e-4)

  def test_dynamic_prefill_crosses_block_and_scalar_paths(self):
    rng = np.random.default_rng(509)
    q = rng.normal(0,2,(1,4,24,16)).astype(np.float32)
    kv = rng.normal(0,1,(2,1,2,64,16)).astype(np.float16)
    tq,tc = Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
    count,length = UOp.variable('capped_rows',1,24),UOp.variable('capped_length',1,64)
    for window in (0,8):
      run = TinyJit(lambda x,c,n,r:prefill_attention(x[:,:,:r],c,n,fast_softmax=True,softcap=1.25,window=window).realize())
      for rows in (17,1,16,15,8,17):
        end = 20+rows
        actual = run(tq,tc,length.bind(end),count.bind(rows)).pad_to(q.shape).numpy()[:,:,:rows]
        np.testing.assert_allclose(actual,reference(q[:,:,:rows],kv,end,1.25,window),atol=2e-5,rtol=3e-4)

  def test_cap_zero_preserves_uncapped_bits_and_invalid_caps_are_rejected(self):
    rng = np.random.default_rng(511)
    q = Tensor(rng.normal(size=(1,4,17,16)).astype(np.float32),device='CPU').realize()
    cache = Tensor(rng.normal(size=(2,1,2,32,16)).astype(np.float16),device='CPU').realize()
    ordinary = prefill_attention(q,cache,25,fast_softmax=True).numpy()
    explicit = prefill_attention(q,cache,25,fast_softmax=True,softcap=0.).numpy()
    np.testing.assert_array_equal(ordinary.view(np.uint32),explicit.view(np.uint32))
    for cap in (-1.,.124,128.1,float('nan'),float('inf'),True):
      with self.assertRaises(ValueError):prefill_attention(q,cache,25,softcap=cap)
      with self.assertRaises(ValueError):decode_attention(q[:,:,:1],cache,25,softcap=cap)
