import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention


def _reference(query, cache, length, *, score_scale=None, window=0):
  """Independent float64 grouped-query attention with explicit causal/window masking."""
  _,heads,rows,dim = query.shape
  _,_,kv_heads,context,_ = cache.shape
  scale = 1 / np.sqrt(dim) if score_scale is None else score_scale
  out = np.zeros((1,heads,rows,dim), dtype=np.float64)
  keys,values = cache[0,0].astype(np.float64),cache[1,0].astype(np.float64)
  for h in range(heads):
    kh = h // (heads // kv_heads)
    for r in range(rows):
      pos = length-rows+r
      first = max(0,pos-window+1) if window else 0
      ix = np.arange(first,pos+1)
      score = keys[kh,ix] @ query[0,h,r].astype(np.float64) * scale
      prob = np.exp(score-score.max())
      prob /= prob.sum()
      out[0,h,r] = prob @ values[kh,ix]
  return out.astype(np.float32)


class TestRockchipGemmaCPUAttention(unittest.TestCase):
  def test_gemma_scales_grouped_heads_and_cached_prefill_and_decode(self):
    rng = np.random.default_rng(20261026)
    for heads,kv_heads,dim in ((8,1,256),(8,2,256),(8,1,512),(8,2,512)):
      for fast in (False,True):
        for window in (0,8):
          length,rows,context = 37,17,64
          q = rng.normal(0,.08,(1,heads,rows,dim)).astype(np.float32)
          kv = rng.normal(0,.1,(2,1,kv_heads,context,dim)).astype(np.float16)
          # Old prefix and future rows are outside every query's legal window.
          if window:
            kv[:,:,:,:(length-rows-window+1)] = np.nan
          kv[:,:,:,length:] = np.nan
          tq,tc = Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
          for scale in (None,1.0):
            expected = _reference(q,kv,length,score_scale=scale,window=window)
            actual = prefill_attention(tq,tc,length,fast_softmax=fast,score_scale=scale,window=window).numpy()
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-4)

            dq = q[:,:,:1].copy()
            decode_length = length-rows+1
            dexpected = _reference(dq,kv,decode_length,score_scale=scale,window=window)
            dactual = decode_attention(Tensor(dq,device='CPU').realize(),tc,decode_length,fast_softmax=fast,
                                       score_scale=scale,window=window).numpy()
            self.assertTrue(np.isfinite(dactual).all())
            np.testing.assert_allclose(dactual,dexpected,atol=2e-5,rtol=3e-4)

  def test_dynamic_prefill_rows_cross_scalar_and_block_paths(self):
    rng = np.random.default_rng(6408)
    heads,kv_heads,dim,context,capacity = 8,2,256,64,24
    q = rng.normal(0,.12,(1,heads,capacity,dim)).astype(np.float32)
    kv = rng.normal(0,.1,(2,1,kv_heads,context,dim)).astype(np.float16)
    # For the sliding case, these keys/values are globally masked across the tested rows.
    kv[:,:,:,:13] = np.nan
    kv[:,:,:,44:] = np.nan
    tq,tc = Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
    rows = UOp.variable('rows',1,capacity)
    length = UOp.variable('length',1,context)
    for fast in (False,True):
      run = TinyJit(lambda x,c,n,r:prefill_attention(x[:,:,:r],c,n,fast_softmax=fast,
                                                       score_scale=1.0,window=8).realize())
      for count in (17,1,16,15,8,17):
        end = 20+count
        # Avoid poisoned future keys while preserving the full bounded cache geometry.
        actual = run(tq,tc,length.bind(end),rows.bind(count))
        got = actual.pad_to((1,heads,capacity,dim)).numpy()[:,:,:count]
        expected = _reference(q[:,:,:count],kv,end,score_scale=1.0,window=8)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got,expected,atol=2e-5,rtol=3e-4)

  def test_near_capacity_window_masks_old_and_future_cache(self):
    rng = np.random.default_rng(8192)
    heads,kv_heads,dim,context,end,rows,window = 8,1,512,8192,8191,17,512
    q = rng.normal(0,.025,(1,heads,rows,dim)).astype(np.float32)
    kv = rng.normal(0,.04,(2,1,kv_heads,context,dim)).astype(np.float16)
    first = end-rows-window+1
    kv[:,:,:,:first] = np.nan
    kv[:,:,:,end:] = np.nan
    tq,tc = Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
    for fast in (False,True):
      expected = _reference(q,kv,end,score_scale=1.0,window=window)
      actual = prefill_attention(tq,tc,end,fast_softmax=fast,score_scale=1.0,window=window).numpy()
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-4)

      decode_length = end-rows+1
      dq = q[:,:,:1].copy()
      dexpected = _reference(dq,kv,decode_length,score_scale=1.0,window=window)
      dactual = decode_attention(Tensor(dq,device='CPU').realize(),tc,decode_length,fast_softmax=fast,
                                 score_scale=1.0,window=window).numpy()
      self.assertTrue(np.isfinite(dactual).all())
      np.testing.assert_allclose(dactual,dexpected,atol=2e-5,rtol=3e-4)

  def test_full_context_window_is_bitwise_full_causal(self):
    rng = np.random.default_rng(512)
    heads,kv_heads,dim,context,rows = 8,2,512,64,17
    q = Tensor(rng.normal(0,.1,(1,heads,rows,dim)).astype(np.float32),device='CPU').realize()
    kv = Tensor(rng.normal(0,.1,(2,1,kv_heads,context,dim)).astype(np.float16),device='CPU').realize()
    for fast in (False,True):
      full = prefill_attention(q,kv,context,fast_softmax=fast,score_scale=1.0,window=0).numpy()
      bounded = prefill_attention(q,kv,context,fast_softmax=fast,score_scale=1.0,window=context).numpy()
      larger = prefill_attention(q,kv,context,fast_softmax=fast,score_scale=1.0,window=context+8).numpy()
      np.testing.assert_array_equal(bounded.view(np.uint32),full.view(np.uint32))
      np.testing.assert_array_equal(larger.view(np.uint32),full.view(np.uint32))

  def test_scale_and_window_validation(self):
    q = Tensor.zeros(1,4,1,256,device='CPU').realize()
    kv = Tensor.zeros(2,1,1,32,256,device='CPU').half().realize()
    for bad_scale in (0.0,-1.0,float('inf'),float('nan'),True):
      with self.assertRaises(ValueError): decode_attention(q,kv,1,score_scale=bad_scale)
      with self.assertRaises(ValueError): prefill_attention(q,kv,1,score_scale=bad_scale)
    for bad_window in (-1,1.5,True):
      with self.assertRaises(ValueError): decode_attention(q,kv,1,window=bad_window)
      with self.assertRaises(ValueError): prefill_attention(q,kv,1,window=bad_window)


if __name__=='__main__': unittest.main()
