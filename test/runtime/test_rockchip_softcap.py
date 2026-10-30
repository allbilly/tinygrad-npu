import os, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_softcap import softcap
from tinygrad.llm.kernels.rockchip_greedy import greedy_argmax


class TestRockchipSoftcap(unittest.TestCase):
  def test_cpu_matches_generic_bitwise_and_numpy_float64(self):
    rng = np.random.default_rng(20261026)
    logits = rng.normal(0,30,(2,4099)).astype(np.float32)
    logits[0,:8] = [-0.0,0.0,-1.0,1.0,-1000.0,1000.0,-50.0,50.0]
    for cap in (1.0,2.5,3.0,3.5,30.0,50.0,1.1):
      tensor = Tensor(logits,device='CPU').realize()
      actual = softcap(tensor,cap).numpy()
      generic = (tensor/cap).tanh().mul(cap).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),generic.view(np.uint32))
      expected = (np.tanh(logits.astype(np.float64)/cap)*cap).astype(np.float32)
      # The existing sigmoid-based FP32 tanh loses a few ulps near zero;
      # its final multiplication amplifies that absolute error by cap.
      np.testing.assert_allclose(actual,expected,atol=4*np.finfo(np.float32).eps*cap,rtol=2e-6)

  def test_vector_tails_and_boundary_sizes(self):
    rng = np.random.default_rng(1031)
    for shape in ((2,512),(1,1024),(1,1025),(1,1031),(2,1031)):
      values = rng.normal(0,8,shape).astype(np.float32)
      actual = softcap(Tensor(values,device='CPU').realize(),3.0).numpy()
      expected = (Tensor(values,device='CPU')/3.0).tanh().mul(3.0).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_nonfinite_signed_zero_and_large_finite_values(self):
    values = np.array([[-np.inf,-1e30,-1000.0,-0.0,0.0,1000.0,1e30,np.inf,np.nan]],np.float32)
    values = np.tile(values,(2,115))
    actual = softcap(Tensor(values,device='CPU').realize(),2.5).numpy()
    generic = (Tensor(values,device='CPU')/2.5).tanh().mul(2.5).numpy()
    np.testing.assert_allclose(actual,generic,atol=1e-6,rtol=1e-6,equal_nan=True)
    self.assertTrue(np.isfinite(actual[0,[0,1,2,3,4,5,6,7]]).all())
    np.testing.assert_array_equal(actual[0,[0,7]],np.array([-2.5,2.5],np.float32))
    expected = (np.tanh(values.astype(np.float64)/2.5)*2.5).astype(np.float32)
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6,equal_nan=True)

  def test_noncontiguous_view_and_fallbacks(self):
    base = Tensor(np.arange(2*4099,dtype=np.float32).reshape(2,4099),device='CPU').realize()
    view = base[:,::2]
    self.assertIsNone(view.uop.contiguous_view())
    actual = softcap(view,3.0)
    generic = (view/3.0).tanh().mul(3.0)
    np.testing.assert_array_equal(actual.numpy().view(np.uint32),generic.numpy().view(np.uint32))

    half = Tensor(np.linspace(-10,10,1023,dtype=np.float16).reshape(1,1023),device='CPU').realize()
    half_actual = softcap(half,3.0)
    half_generic = (half/3.0).tanh().mul(3.0)
    self.assertEqual(half_actual.dtype,dtypes.half)
    np.testing.assert_array_equal(half_actual.numpy().view(np.uint16),half_generic.numpy().view(np.uint16))

    small = Tensor(np.array([[1.0,-2.0,3.0]],np.float32),device='CPU').realize()
    np.testing.assert_array_equal(softcap(small,3.0).numpy().view(np.uint32),
                                  ((small/3.0).tanh().mul(3.0)).numpy().view(np.uint32))
    scalar = Tensor(-4.0,device='CPU').realize()
    np.testing.assert_array_equal(softcap(scalar,2.0).numpy().view(np.uint32),
                                  ((scalar/2.0).tanh().mul(2.0)).numpy().view(np.uint32))
    for cap in (.124,128.1):
      np.testing.assert_array_equal(softcap(base,cap).numpy().view(np.uint32),
                                    ((base/cap).tanh()*cap).numpy().view(np.uint32))

  def test_threaded_vocabulary_and_softcap_greedy_ties(self):
    if hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))<4:
      self.skipTest('Threaded softcap needs four available CPUs')
    rng = np.random.default_rng(262144)
    logits = rng.normal(0,20,(1,262144)).astype(np.float32)
    actual = softcap(Tensor(logits,device='CPU').realize(),3.5).numpy()
    expected = (Tensor(logits,device='CPU')/3.5).tanh().mul(3.5).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    saturated = np.full((1,262144),-1000.0,np.float32)
    saturated[0,3] = saturated[0,131072] = 1000.0
    capped = softcap(Tensor(saturated,device='CPU').realize(),3.0)
    self.assertTrue(np.isfinite(capped.numpy()).all())
    self.assertEqual(greedy_argmax(capped).tolist(),[3])
    np.testing.assert_array_equal(capped.numpy()[0,[3,131072]],np.array([3.0,3.0],np.float32))

  def test_invalid_caps(self):
    logits = Tensor.zeros(1,1024,device='CPU').realize()
    for cap in (0.0,-1.0,float('inf'),float('-inf'),float('nan')):
      with self.subTest(cap=cap),self.assertRaises(ValueError): softcap(logits,cap)


if __name__=='__main__': unittest.main()
