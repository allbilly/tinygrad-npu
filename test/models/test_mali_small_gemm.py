"""Small-M Mali GEMMs preserve legacy per-output FMA order and symbolic capacity."""
import math, os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels import mali_linear


GEOMETRIES=((8,32,1),(4,64,1),(16,32,2))


def project(x,weight,packed,geometry=None):
  """Call the untouched legacy kernel directly, or force a test geometry."""
  columns,depth=weight.shape
  rows=math.prod(x.max_shape[:-1])
  output=Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  result=output.custom_kernel(x.pad_to(x.max_shape).contiguous(),packed if packed is not None else weight.contiguous(),
    fxn=lambda o,a,w:(mali_linear._gemm(o,a,w,rows,columns,depth,packed is not None) if geometry is None
                     else mali_linear._small_gemm(o,a,w,rows,columns,depth,packed is not None,geometry)))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


class TestMaliSmallGemmContract(unittest.TestCase):
  def test_large_symbolic_capacity_uses_legacy_even_when_bound_to_three(self):
    count=UOp.variable('large_small_m_rows',3,32).bind(3)
    x=Tensor.empty(1,32,2048,device='CL')[:,:count]
    weight=Tensor.empty(512,2048,device='CL',dtype=dtypes.half)
    with patch.object(mali_linear,'_small_gemm_supported',side_effect=AssertionError('unmeasured capacity must retain legacy')):
      result=mali_linear.gemm(x,weight)
    self.assertEqual(result.shape,(1,count,512))
    self.assertEqual(result.max_shape,(1,32,512))
    self.assertIsNone(mali_linear._small_gemm_geometry(1,512,2048,True))
    self.assertIsNone(mali_linear._small_gemm_geometry(3,248320,2048,True))

  def test_fp32_weights_keep_legacy_without_probing_the_device(self):
    x=Tensor.empty(1,3,2048,device='CL')
    weight=Tensor.empty(512,2048,device='CL',dtype=dtypes.float)
    with patch.object(mali_linear,'_small_gemm_supported',side_effect=AssertionError('FP32 weights must retain legacy')):
      self.assertEqual(mali_linear.gemm(x,weight).shape,(1,3,512))

  def test_full_head_dispatches_measured_rowmajor_capacities_without_allocating_weights(self):
    weight=Tensor.empty(248320,2048,device='CL',dtype=dtypes.half)
    for rows,geometry in ((3,(4,64,1)),(8,(16,32,2))):
      x=Tensor.empty(1,rows,2048,device='CL')
      with patch.object(mali_linear,'_small_gemm_supported',return_value=True), \
           patch.object(mali_linear,'_small_gemm',wraps=mali_linear._small_gemm) as selected:
        result=mali_linear.gemm(x,weight)
      self.assertEqual(result.shape,(1,rows,248320))
      self.assertTrue(selected.called)
      self.assertEqual(selected.call_args.args[-1],geometry)

  def test_full_head_keeps_unmeasured_packed_fp32_and_larger_symbolic_cases_on_legacy(self):
    weight=Tensor.empty(248320,2048,device='CL',dtype=dtypes.half)
    packed=Tensor.empty(7760,2048,32,device='CL',dtype=dtypes.half)
    fp32_weight=Tensor.empty(248320,2048,device='CL',dtype=dtypes.float)
    with patch.object(mali_linear,'_small_gemm_supported',side_effect=AssertionError('unqualified head path must retain legacy')):
      for rows in (5,7,16):
        self.assertEqual(mali_linear.gemm(Tensor.empty(1,rows,2048,device='CL'),weight).shape,(1,rows,248320))
      for rows in (3,8):
        self.assertEqual(mali_linear.gemm(Tensor.empty(1,rows,2048,device='CL'),weight,packed).shape,(1,rows,248320))
      self.assertEqual(mali_linear.gemm(Tensor.empty(1,3,2048,device='CL'),fp32_weight).shape,(1,3,248320))
      count=UOp.variable('full_head_rows',1,16).bind(3)
      result=mali_linear.gemm(Tensor.empty(1,16,2048,device='CL')[:,:count],weight)
      self.assertEqual(result.shape,(1,count,248320))
      self.assertEqual(result.max_shape,(1,16,248320))


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliSmallGemm(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not mali_linear._small_gemm_supported('CL'):raise unittest.SkipTest('small-M kernels qualify only the measured Mali device metadata')

  def test_forced_geometries_preserve_fma_bits_for_tails_batches_and_mixed_ranges(self):
    rng=np.random.default_rng(207)
    for mixed in (False,True):
      for shape,columns in (((1,3,68),65),((1,8,132),97),((1,16,2049),33),((1,7,31),67),((2,3,96),65)):
        depth=shape[-1]
        weights=rng.standard_normal((columns,depth),dtype=np.float32)
        values=rng.normal(0,.3,shape).astype(np.float32)
        if mixed:
          weights=np.ldexp(weights,rng.integers(-13,14,weights.shape))
          values=np.ldexp(values,rng.integers(-12,13,values.shape)).astype(np.float32)
        else:weights*=np.float32(1/np.sqrt(depth))
        weights=weights.astype(np.float16)
        self.assertTrue(np.isfinite(weights).all() and np.isfinite(values).all())
        weight,x=Tensor(weights,device='CL').realize(),Tensor(values,device='CL').realize()
        packed=mali_linear.pack_weight(weight).realize()
        expected=project(x,weight,packed).numpy()
        if not mixed:
          np.testing.assert_allclose(expected,values@weights.astype(np.float32).T,atol=6e-6,rtol=3e-4)
        for stored in (None,packed):
          np.testing.assert_array_equal(project(x,weight,stored).numpy().view(np.uint32),expected.view(np.uint32))
          for geometry in GEOMETRIES:
            with self.subTest(shape=shape,columns=columns,mixed=mixed,packed=stored is not None,geometry=geometry):
              actual=project(x,weight,stored,geometry).numpy()
              np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_measured_dispatch_matches_explicit_legacy_reference(self):
    rng=np.random.default_rng(208)
    for columns,depth in ((512,2048),(2048,2048)):
      weights=(rng.standard_normal((columns,depth),dtype=np.float32)*np.float32(1/np.sqrt(depth))).astype(np.float16)
      weight=Tensor(weights,device='CL').realize()
      packed=mali_linear.pack_weight(weight).realize()
      for rows in (3,8,16):
        x=Tensor(rng.normal(0,.3,(1,rows,depth)).astype(np.float32),device='CL').realize()
        for stored in (None,packed):
          expected=project(x,weight,stored).numpy()
          with patch.object(mali_linear,'_small_gemm',wraps=mali_linear._small_gemm) as selected:
            actual=mali_linear.gemm(x,weight,stored).numpy()
          self.assertTrue(selected.called)
          self.assertTrue(all(call.args[3]==rows for call in selected.call_args_list))
          np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_symbolic_rebind_uses_maximum_capacity_and_ignores_nan_hidden_rows(self):
    rng=np.random.default_rng(209)
    weights=(rng.standard_normal((512,2048),dtype=np.float32)*np.float32(1/np.sqrt(2048))).astype(np.float16)
    weight=Tensor(weights,device='CL').realize()
    packed=mali_linear.pack_weight(weight).realize()
    for stored in (None,packed):
      run=TinyJit(lambda x,count:mali_linear.gemm(x[:,:count],weight,stored).realize())
      with patch.object(mali_linear,'_small_gemm',wraps=mali_linear._small_gemm) as selected:
        for count in (16,8,3,1,15,16):
          values=rng.normal(0,.3,(1,16,2048)).astype(np.float32)
          values[:,count:]=np.nan
          actual=run(Tensor(values,device='CL').realize(),UOp.variable('small_m_rows',1,16).bind(count)).pad_to((1,16,512)).numpy()[:,:count]
          expected=project(Tensor(values[:,:count].copy(),device='CL').realize(),weight,stored).numpy()
          np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      self.assertTrue(selected.called)
      self.assertTrue(all(call.args[3]==16 for call in selected.call_args_list))


if __name__=='__main__':unittest.main()
