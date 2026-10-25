import os, unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, function
from tinygrad.llm.kernels.rockchip_dot_attention import Int8CPUAttention, supports_dotprod


def quantize(x):
  scale = np.maximum(np.max(abs(x),axis=-1,keepdims=True)/np.float32(127),np.float32(1e-8))
  return np.clip(np.rint(x*(np.float32(1)/scale)),-127,127).astype(np.int32),scale[...,0]


def reference(query,cache,end):
  result = np.empty_like(query)
  for head in range(9):
    key,value = cache[:,0,head//3,:end].astype(np.float32)
    keys,ks = quantize(key)
    queries,qs = quantize(query[0,head])
    for row in range(query.shape[2]):
      used = end-query.shape[2]+row+1
      score = ((queries[row]@keys[:used].T).astype(np.float32)*qs[row])*ks[:used]*np.float32(.125)
      probability = np.exp(score-score.max())
      result[0,head,row] = (probability/probability.sum())@value[:used]
  return result


@unittest.skipUnless(supports_dotprod(),'Linux AArch64 SDOT required')
class TestDotCPUAttention(unittest.TestCase):
  def test_quantized_reference_and_future_mask_at_partial_max_context(self):
    rng = np.random.default_rng(115)
    for context,end,count in ((63,61,16),(8192,8191,65),(8192,8192,128)):
      values = rng.normal(0,.3,(2,1,3,context,64)).astype(np.float16)
      values[:,:,:,end:] = np.nan
      q = rng.normal(0,.3,(1,9,count,64)).astype(np.float32)
      actual = Int8CPUAttention(context)(Tensor(q,device='CPU'),Tensor(values,device='CPU'),end).numpy()
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual,reference(q,values,end),atol=4e-6,rtol=3e-4)

  def test_query_count_and_pooled_decode_have_identical_last_row(self):
    rng = np.random.default_rng(116)
    values = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
    values[:,:,:,8191:] = np.nan
    q = rng.normal(0,.3,(1,9,128,64)).astype(np.float32)
    state = Int8CPUAttention(8192)
    expected = state(Tensor(q,device='CPU'),Tensor(values,device='CPU'),8191).numpy()[:,:,-1:]
    query,cache = Tensor(q[:,:,-1:],device='CPU'),Tensor(values,device='CPU')
    @function(precompile=True,allow_implicit=True)
    def decode(q,k,end): return state.decode(q,k,end)
    actual = decode(query,cache,UOp.variable('end',1,8192).bind(8191)).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    if {4,5,6,7}.issubset(os.sched_getaffinity(0)):
      from tinygrad.llm.kernels.rockchip_pool import attention_pool
      pool = attention_pool()
      before = pool.lib.pool_calls()
      @function(precompile=True,allow_implicit=True)
      def pooled(q,k,end): return state.decode(q,k,end,pool=pool)
      with pool: actual = pooled(query,cache,UOp.variable('end',1,8192).bind(8191)).numpy()
      self.assertGreater(pool.lib.pool_calls(),before)
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_gaps_divergent_history_reset_and_small_queries(self):
    rng = np.random.default_rng(117)
    state = Int8CPUAttention(63)
    values = rng.normal(0,.3,(2,1,3,63,64)).astype(np.float16)
    for end,count,change in ((16,16,0),(32,16,16),(56,16,32),(32,16,16),(8,8,0),(31,3,8)):
      values[:,:,:,change:end] = rng.normal(0,.3,(2,1,3,end-change,64)).astype(np.float16)
      q = rng.normal(0,.3,(1,9,count,64)).astype(np.float32)
      cache = values.copy()
      cache[:,:,:,end:] = np.nan
      actual = state(Tensor(q,device='CPU'),Tensor(cache,device='CPU'),end).numpy()
      np.testing.assert_allclose(actual,reference(q,cache,end),atol=4e-6,rtol=3e-4)
      header = state.packed[:4].bitcast(dtypes.int).numpy().tolist()
      self.assertEqual(header[0],end)
      if end==56: self.assertEqual(header[1],24)
      if end==31: self.assertEqual(header[1],23)

  def test_causal_future_nan_values_and_invalid_query(self):
    rng = np.random.default_rng(118)
    values = rng.normal(0,.3,(2,1,3,32,64)).astype(np.float16)
    values[1,0,:,15,0] = np.nan
    q = rng.normal(0,.3,(1,9,16,64)).astype(np.float32)
    state = Int8CPUAttention(32)
    actual = state(Tensor(q,device='CPU'),Tensor(values,device='CPU'),16).numpy()
    self.assertTrue(np.isfinite(actual[:,:,:15]).all())
    self.assertTrue(np.isnan(actual[:,:,15,0]).all())
    np.testing.assert_allclose(actual,reference(q,values,16),atol=4e-6,rtol=3e-4,equal_nan=True)
    q[0,0,0,0] = np.inf
    actual = state(Tensor(q,device='CPU'),Tensor(values,device='CPU'),16).numpy()
    self.assertTrue(np.isnan(actual[0,0,0]).all())


if __name__=='__main__': unittest.main()
