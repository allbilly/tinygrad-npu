"""Tiled row-major Mali GEMM against the unchanged packed accumulation path."""
import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.mali_linear import gemm, pack_weight


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliTiledGemm(unittest.TestCase):
  def test_storage_types_and_row_column_depth_tails_match_packed_bits(self):
    rng = np.random.default_rng(196)
    for dtype in (np.float16,np.float32):
      for rows,columns,depth in ((1,65,68),(7,97,132),(32,113,128),(33,67,96),(64,128,256),(32,64,13696)):
        weight = (rng.standard_normal((columns,depth),dtype=np.float32)/np.sqrt(depth)).astype(dtype)
        values = rng.normal(0,.3,(1,rows,depth)).astype(np.float32)
        stored,x = Tensor(weight,device='CL').realize(),Tensor(values,device='CL').realize()
        packed = pack_weight(stored).realize()
        expected,actual = gemm(x,stored,packed).numpy(),gemm(x,stored).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_allclose(actual,values@weight.astype(np.float32).T,atol=6e-6,rtol=3e-4)

  def test_symbolic_row_capacity_rebinds_without_changing_valid_outputs(self):
    rng = np.random.default_rng(197)
    weight = rng.normal(0,.1,(97,132)).astype(np.float16)
    stored = Tensor(weight,device='CL').realize()
    packed = pack_weight(stored).realize()
    run = TinyJit(lambda x,rows:gemm(x[:,:rows],stored).realize())
    for count in (32,17,1,31,7,32):
      values = rng.normal(0,.3,(1,32,132)).astype(np.float32)
      actual = run(Tensor(values,device='CL').realize(),UOp.variable('rows',1,32).bind(count)).pad_to((1,32,97)).numpy()[:,:count]
      expected = gemm(Tensor(values[:,:count],device='CL').realize(),stored,packed).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
