import ctypes, os, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.llm.kernels.rockchip import RockchipLinear
from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
from tinygrad.llm.kernels.rockchip_vision_mlp import FP16VisionMLP
from tinygrad.llm.kernels.rockchip_vision_pipeline import _Stage
from tinygrad.llm.kernels.rockchip_vision_prepared import PreparedVisionMLP


class TestPreparedVisionMLP(unittest.TestCase):
  def test_unqualified_shape_rejected_without_npu(self):
    layer = RockchipLinear(nn.Linear(32,32),'on','off')
    with self.assertRaises(ValueError):PreparedVisionMLP(layer,layer)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1' and os.environ.get('ROCKCHIP_GPU_TEST')=='1','explicit NPU/GPU test')
  def test_outputs_rebinding_guards_and_release(self):
    Tensor.manual_seed(57)
    sources = [nn.Linear(768,3072),nn.Linear(3072,768)]
    for layer in sources:
      for parameter in nn.state.get_parameters(layer):parameter.replace(parameter.half().contiguous().realize())
    fc1,fc2 = [RockchipLinear(layer,'on','off') for layer in sources]
    fc1.prepare()
    fc2.prepare()
    original = FP16VisionMLP(fc1,fc2)
    reference = _Stage(lambda x,r:(r+original(x)).realize(),True,ctypes.cast(cache_cleaner().clean_to,ctypes.c_void_p).value)
    op = PreparedVisionMLP(fc1,fc2)
    dev = Device['ROCKCHIP']
    try:
      for i in range(4):
        value = Tensor.randn(1,1024,768,device='CPU').realize()
        residual = Tensor.randn(1,1024,768,device='CPU').realize()
        expected = reference(value,residual).numpy().copy()
        before,tasks,allocations = dev.submit_count,dev.task_count,dev.allocation_count
        actual = op(value,residual).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-3)
        self.assertEqual(dev.submit_count-before,2)
        self.assertEqual(dev.task_count-tasks,96)
        self.assertEqual(dev.allocation_count,allocations)
      for bad in (float('nan'),float('inf'),70000.):
        invalid = Tensor.full((1,1024,768),bad,device='CPU').contiguous().realize()
        before = dev.submit_count
        with self.assertRaises(ValueError):op(invalid,residual)
        self.assertEqual(dev.submit_count,before)
        self.assertFalse(dev.poisoned)
      invalid = Tensor.full((1,1024,768),float('nan'),device='CPU').contiguous().realize()
      before = dev.submit_count
      with self.assertRaises(ValueError):op(value,invalid)
      self.assertEqual(dev.submit_count,before)
      maximum = Tensor.full((1,1024,768),65504.,device='CPU').contiguous().realize()
      before = dev.submit_count
      with self.assertRaises(ValueError):op(maximum,residual)
      self.assertEqual(dev.submit_count-before,1) # GPU overflow guard prevents the FC2 submission.
      self.assertFalse(dev.poisoned)
      self.assertTrue(np.isfinite(op(value,residual).numpy()).all())
      meta = op.weights[0].meta
      version = meta.version
      before = dev.submit_count
      try:
        meta.version+=1
        with self.assertRaises(RuntimeError):op(value,residual)
        self.assertEqual(dev.submit_count,before)
      finally:meta.version=version
      for buffer in (op.kept[0],op.bias1.uop.buffer,fc2.weight.uop.buffer):
        storage = buffer.get_storage()
        try:
          buffer._storage=replace(storage)
          with self.assertRaises(RuntimeError):op(value,residual)
          self.assertEqual(dev.submit_count,before)
        finally:buffer._storage=storage
      bias = fc1.bias
      try:
        fc1.bias=Tensor.zeros(3072,device='CPU',dtype=dtypes.half).realize()
        with self.assertRaises(RuntimeError):op(value,residual)
        self.assertEqual(dev.submit_count,before)
      finally:fc1.bias=bias
    finally:op.close()
    self.assertFalse(op.owned)
    self.assertFalse(op.packed_weights)
    self.assertFalse(op.softmax.mem)
    self.assertIsNone(op.softmax.flag)
    with self.assertRaises(RuntimeError):op(value,residual)


if __name__=='__main__':unittest.main()
