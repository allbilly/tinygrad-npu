import platform, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, function
from tinygrad.llm.kernels.rockchip_packed_attention import PackedCPUAttention
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.llm.kernels.rockchip_cpu import prefill_attention


def reference(query,cache,end):
  rows = query.shape[2]
  out = np.empty_like(query)
  for head in range(9):
    for row in range(rows):
      used = end-rows+row+1
      key,value = cache[:,0,head//3,:used].astype(np.float32)
      score = query[0,head,row]@key.T/np.float32(8)
      p = np.exp(score-score.max())
      out[0,head,row] = (p/p.sum())@value
  return out


class TestPackedCPUAttention(unittest.TestCase):
  @unittest.skipUnless(platform.machine().lower() in ('aarch64','arm64'),'AArch64 FP16 SIMD required')
  def test_optional_fp16_qk_and_nonfinite_intermediate_fallback(self):
    rng = np.random.default_rng(108)
    for context,end,count,scale in ((64,63,16,1),(8192,8191,128,1),(8192,8190,65,1),(8192,8192,127,1),
                                    (64,63,7,1),(32,24,16,1e6)):
      values = rng.normal(0,.3,(2,1,3,context,64)).astype(np.float16)
      values[:,:,:,end:] = np.nan
      q = rng.normal(0,.3,(1,9,count,64)).astype(np.float32)*np.float32(scale)
      state = PackedCPUAttention(context,fp16_qk=True)
      @function(precompile=True,allow_implicit=True)
      def run(query,cache,length): return state(query,cache,length)
      actual = run(Tensor(q,device='CPU'),Tensor(values,device='CPU'),UOp.variable('end',1,context).bind(end)).numpy()
      expected = reference(q,values,end)
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=.02)

  def test_incremental_gaps_divergent_history_and_reset_with_cpu_replay_fallback(self):
    rng = np.random.default_rng(95)
    context,capacity = 63,16
    state = PackedCPUAttention(context)
    run = TinyJit(lambda q,k,end,rows:state(q[:,:,:rows],k,end).realize())
    values = rng.normal(0,.3,(2,1,3,context,64)).astype(np.float16)
    for index,(end,count,start_change) in enumerate(((16,16,0),(32,16,16),(56,16,32),(32,16,16),(8,8,0),(31,7,8))):
      values[:,:,:,start_change:end] = rng.normal(0,.3,(2,1,3,end-start_change,64)).astype(np.float16)
      masked = values.copy()
      masked[:,:,:,end:] = np.nan
      q = rng.normal(0,.3,(1,9,capacity,64)).astype(np.float32)
      actual = run(Tensor(q,device='CPU').realize(),Tensor(masked,device='CPU').realize(),
                   UOp.variable('end',1,context).bind(end),UOp.variable('rows',1,capacity).bind(count))
      result = actual.pad_to((1,9,capacity,64)).numpy()[:,:,:count]
      expected = reference(q[:,:,:count],masked,end)
      np.testing.assert_allclose(result,expected,atol=4e-6,rtol=3e-4)
      self.assertTrue(np.isfinite(result).all())
      header = state.packed[:4].bitcast(dtypes.int).numpy()
      if index==2:
        self.assertEqual(list(header),[56,24])
      if index==1:
        run.captured = RockchipCapturedJit(run.captured)
    # This CPU-only graph intentionally lacks the NPU work required by the
    # mixed replay implementation. Full-model hardware qualification covers it.
    self.assertIsNone(run.captured.native)
    self.assertEqual(run.captured.reason,'no NPU work')

  def test_decode_rewrite_invalidates_packed_prefix_before_next_prefill(self):
    rng = np.random.default_rng(98)
    state = PackedCPUAttention(64)
    values = rng.normal(0,.3,(2,1,3,64,64)).astype(np.float16)
    query = rng.normal(0,.3,(1,9,16,64)).astype(np.float32)
    state(Tensor(query,device='CPU'),Tensor(values,device='CPU'),32).realize()
    values[:,:,:,19:40] = rng.normal(0,.3,(2,1,3,21,64)).astype(np.float16)
    decode_query = rng.normal(0,.3,(1,9,1,64)).astype(np.float32)
    @function(precompile=True,allow_implicit=True)
    def decode(q,k,end): return state.decode(q,k,end)
    actual = decode(Tensor(decode_query,device='CPU'),Tensor(values,device='CPU'),UOp.variable('end',1,64).bind(20)).numpy()
    np.testing.assert_allclose(actual,reference(decode_query,values,20),atol=4e-6,rtol=3e-4)
    actual = state(Tensor(query,device='CPU'),Tensor(values,device='CPU'),40).numpy()
    np.testing.assert_allclose(actual,reference(query,values,40),atol=4e-6,rtol=3e-4)

  def test_near_maximum_full_partial_rows_and_masked_nonfinite_future(self):
    rng = np.random.default_rng(96)
    context = 8192
    for count,end in ((128,8192),(65,8190),(7,8184),(127,8192)):
      state = PackedCPUAttention(context)
      values = rng.normal(0,.3,(2,1,3,context,64)).astype(np.float16)
      values[:,:,:,end:] = np.nan
      query = rng.normal(0,.3,(1,9,count,64)).astype(np.float32)
      if count==65: query *= 100
      actual = state(Tensor(query,device='CPU'),Tensor(values,device='CPU'),end).numpy()
      expected = reference(query,values,end)
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
      baseline = prefill_attention(Tensor(query,device='CPU'),Tensor(values,device='CPU'),end,fast_softmax=True).numpy()
      np.testing.assert_array_equal(actual,baseline)

  def test_causal_tail_does_not_multiply_masked_nan_values_by_zero(self):
    rng = np.random.default_rng(97)
    values = rng.normal(0,.3,(2,1,3,32,64)).astype(np.float16)
    values[1,0,:,15,0] = np.nan
    query = rng.normal(0,.3,(1,9,16,64)).astype(np.float32)
    state = PackedCPUAttention(32)
    actual = state(Tensor(query,device='CPU'),Tensor(values,device='CPU'),16).numpy()
    expected = reference(query,values,16)
    self.assertTrue(np.isfinite(actual[:,:,:15]).all())
    self.assertTrue(np.isnan(actual[:,:,15,0]).all())
    np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4,equal_nan=True)


if __name__=='__main__':unittest.main()
