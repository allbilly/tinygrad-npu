import ctypes, os, unittest
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.llm.kernels.rockchip_attention import NPUVisionAttention

class TestNPUVisionAttention(unittest.TestCase):
  def test_unqualified_shapes_rejected_without_device(self):
    for heads,rows in [(1,1024),(6,1024),(12,512),(12,1023)]:
      with self.assertRaises(ValueError): NPUVisionAttention(heads,rows)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_outputs_rebinding_finite_guard_and_resource_release(self):
    rng = np.random.default_rng(41)
    cases = [(3,False),(12,False)]
    if os.environ.get('ROCKCHIP_GPU_TEST')=='1': cases.extend([(3,True),(12,True)])
    for heads,shared in cases:
      op = NPUVisionAttention(heads,reset_context=heads==3,gpu_softmax=shared)
      Device['ROCKCHIP'].reset()
      for iteration in range(4):
        values = [rng.normal(0,.4,(1,heads,1024,64)).astype(np.float32) for _ in range(3)]
        half = [a.astype(np.float16).astype(np.float32) for a in values]
        scores = half[0] @ half[1].swapaxes(-1,-2) / 8
        probs = np.exp(scores-scores.max(axis=-1,keepdims=True))
        probs /= probs.sum(axis=-1,keepdims=True)
        expected = probs.astype(np.float16).astype(np.float32) @ half[2]
        before = Device['ROCKCHIP'].submit_count
        actual = op(*(Tensor(a,device='CPU').realize() for a in values)).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-3)
        self.assertEqual(Device['ROCKCHIP'].submit_count-before,2 if shared else heads//3*2)
      for invalid in (float('nan'),float('inf'),float('-inf'),1e10):
        values[1][0,-1,-1,-1] = invalid
        before = Device['ROCKCHIP'].submit_count
        with self.assertRaises(ValueError): op(*(Tensor(a,device='CPU').realize() for a in values))
        self.assertEqual(Device['ROCKCHIP'].submit_count,before)
        self.assertFalse(Device['ROCKCHIP'].poisoned)
      values[1][:] = .2
      self.assertTrue(np.isfinite(op(*(Tensor(a,device='CPU').realize() for a in values)).numpy()).all())
      if shared:
        from tinygrad.runtime.autogen import opencl as cl, rockchip as rk
        from tinygrad.runtime.ops_cl import check
        helper = op.softmax
        # Device-output corruption must be detected by the GPU before any PV.
        scores = np.ctypeslib.as_array((ctypes.c_float*(op.scores.size//4)).from_address(op.scores.addr))
        scores.fill(0)
        scores[100] = np.nan
        ctypes.c_uint32.from_address(helper.flag.addr).value = 0
        op.dev.sync(op.scores,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        op.dev.sync(helper.flag,rk.RKNPU_MEM_SYNC_TO_DEVICE)
        before = op.dev.submit_count
        check(cl.clEnqueueNDRangeKernel(helper.gpu.queue,helper.kernel,1,None,(ctypes.c_size_t*1)(heads*1024*4),
                                        (ctypes.c_size_t*1)(64),0,None,None))
        helper.gpu.synchronize()
        op.dev.sync(helper.flag,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        self.assertEqual(ctypes.c_uint32.from_address(helper.flag.addr).value,1)
        self.assertEqual(op.dev.submit_count,before)
      op.close()
      self.assertFalse(op.owned)
      if shared:
        self.assertFalse(op.softmax.mem)
        self.assertFalse(op.softmax.fds)
        self.assertIsNone(op.softmax.flag)
      with self.assertRaises(RuntimeError): op(*(Tensor(a,device='CPU') for a in values))

if __name__ == '__main__': unittest.main()
