"""Native Q8 projections must preserve existing Mali FP16 projection operands."""
import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.kernels.mali_q8 import MaliQ8Linear, gemm_q8, gemv_q8, _gemm_geometry, _gemm_q8


def q8_fixture(columns:int, depth:int, seed:int=198) -> tuple[np.ndarray,np.ndarray]:
  rng = np.random.default_rng(seed)
  scales = rng.uniform(-.003,.003,(columns,depth//32)).astype(np.float16)
  scales.flat[:min(5,scales.size)] = np.array([0,2**-24,-2**-24,2**-14,-2**-14],np.float16)[:min(5,scales.size)]
  coefficients = rng.integers(-128,128,(columns,depth//32,32),dtype=np.int8)
  raw = np.empty((columns,depth//32,34),np.uint8)
  raw[:,:,:2] = scales.view(np.uint8).reshape(columns,depth//32,2)
  raw[:,:,2:] = coefficients.view(np.uint8)
  weight = (scales.astype(np.float32)[:,:,None]*coefficients.astype(np.float32)).astype(np.float16).reshape(columns,depth)
  return raw.flatten(),weight


class TestMaliQ8Contract(unittest.TestCase):
  """Construct only lazy CL graphs; these checks do not initialize a GPU."""
  def test_retains_only_quantized_storage_and_bias(self):
    raw = Tensor.empty(3*34,device='CL',dtype=dtypes.uint8)
    bias = Tensor.empty(3,device='CL',dtype=dtypes.float)
    layer = MaliQ8Linear(raw,3,32,bias)
    self.assertIs(layer.raw,raw)
    self.assertFalse(hasattr(layer,'weight'))
    self.assertEqual({id(value) for value in nn.state.get_parameters(layer)},{id(raw),id(bias)})
    for dtype in (dtypes.half,dtypes.float):
      for shape in ((32,),(1,32),(2,3,32)):
        result = layer(Tensor.empty(*shape,device='CL',dtype=dtype))
        self.assertEqual(result.shape,(*shape[:-1],3))
        self.assertIs(result.dtype,dtype)

  def test_rejects_unsupported_layouts_before_launch(self):
    raw = Tensor.empty(3*34,device='CL',dtype=dtypes.uint8)
    for columns,depth in ((0,32),(3,0),(3,31),(True,32),(3,True),(4,32)):
      with self.subTest(columns=columns,depth=depth), self.assertRaises(ValueError): MaliQ8Linear(raw,columns,depth)
    for storage in (Tensor.empty(3*34,device='CPU',dtype=dtypes.uint8),Tensor.empty(3*34,device='CL',dtype=dtypes.int8)):
      with self.assertRaises(ValueError): MaliQ8Linear(storage,3,32)
    for bias in (Tensor.empty(4,device='CL'),Tensor.empty(3,device='CPU'),Tensor.empty(3,device='CL',dtype=dtypes.int)):
      with self.assertRaises(ValueError): MaliQ8Linear(raw,3,32,bias)
    layer = MaliQ8Linear(raw,3,32)
    for x in (Tensor.empty(1,31,device='CL'),Tensor.empty(1,32,device='CPU'),Tensor.empty(1,32,device='CL',dtype=dtypes.int),
              Tensor.empty(0,32,device='CL'),Tensor.empty(2**27,32,device='CL'),Tensor.empty(device='CL')):
      with self.assertRaises(ValueError): layer(x)
    with self.assertRaises(ValueError): gemv_q8(Tensor.empty(2,32,device='CL'),raw,3,32)

  def test_symbolic_rows_keep_bounded_launch_and_valid_shape(self):
    rows = UOp.variable('q8_rows',1,32).bind(7)
    raw = Tensor.empty(67*3*34,device='CL',dtype=dtypes.uint8)
    result = MaliQ8Linear(raw,67,96)(Tensor.empty(2,32,96,device='CL')[:,:rows])
    self.assertEqual(result.shape,(2,rows,67))
    self.assertEqual(result.max_shape,(2,32,67))

  def test_prefill_geometry_is_limited_to_measured_complete_shapes(self):
    for columns,depth in ((3584,1024),(6144,1024),(1024,3584)):
      self.assertEqual(_gemm_geometry(128,columns,depth),(64,64,128))
      for rows in (1,17,32,64,127,129):self.assertEqual(_gemm_geometry(rows,columns,depth),(32,32,64))
    for columns,depth in ((67,64),(3583,1024),(3584,2048),(248320,1024)):
      self.assertEqual(_gemm_geometry(128,columns,depth),(32,32,64))


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliQ8Hardware(unittest.TestCase):
  def test_tuned_full_prefill_matches_legacy_q8_bits(self):
    rng=np.random.default_rng(3508128)
    for columns,depth in ((3584,1024),(6144,1024),(1024,3584)):
      raw_values,_=q8_fixture(columns,depth)
      raw=Tensor(raw_values,device='CL').realize()
      x=Tensor(rng.normal(0,.3,(128,depth)).astype(np.float32),device='CL').realize()
      legacy=Tensor.empty(128,columns,device='CL',dtype=dtypes.float).custom_kernel(x,raw,
        fxn=lambda o,a,w:_gemm_q8(o,a,w,128,columns,depth,(32,32,64)))[0].numpy().copy()
      actual=gemm_q8(x,raw,columns,depth).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),legacy.view(np.uint32))

  def test_tuned_prefill_rebinds_partial_rows(self):
    rng=np.random.default_rng(3508129)
    columns,depth=3584,1024
    raw_values,_=q8_fixture(columns,depth)
    raw=Tensor(raw_values,device='CL').realize()
    run=TinyJit(lambda x,rows:gemm_q8(x[:,:rows],raw,columns,depth).realize())
    for count in (128,63,1,127,128):
      values=rng.normal(0,.3,(1,128,depth)).astype(np.float32)
      values[:,count:]=np.nan
      actual=run(Tensor(values,device='CL').realize(),UOp.variable('q8_tuned_rows',1,128).bind(count))
      valid=actual.pad_to((1,128,columns)).numpy()[:,:count]
      expected=gemm_q8(Tensor(values[:,:count].copy(),device='CL').realize(),raw,columns,depth).numpy()
      np.testing.assert_array_equal(valid.view(np.uint32),expected.view(np.uint32))

  def test_tiled_prefill_matches_fp16_weight_gemm_bits_with_tails(self):
    from tinygrad.llm.kernels.mali_linear import gemm
    rng = np.random.default_rng(199)
    for shape,columns in (((1,32),65),((7,96),97),((32,128),113),((33,96),67),((64,256),128),((2,3,96),67)):
      raw_values,weights = q8_fixture(columns,shape[-1])
      raw,weight = Tensor(raw_values,device='CL').realize(),Tensor(weights,device='CL').realize()
      for dtype in (np.float32,np.float16):
        values = rng.normal(0,.3,shape).astype(dtype)
        x = Tensor(values,device='CL').realize()
        expected,actual = gemm(x.float(),weight).numpy(),gemm_q8(x,raw,columns,shape[-1]).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_allclose(actual,values.astype(np.float32)@weights.astype(np.float32).T,atol=2e-6,rtol=3e-4)
        np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))

  def test_decode_matches_fp16_gemv_bits_and_layer_dtype_with_bias(self):
    from tinygrad.llm.kernels.mali_linear import gemv
    rng = np.random.default_rng(200)
    for columns,depth in ((65,32),(97,96),(113,128),(129,1024)):
      raw_values,weights = q8_fixture(columns,depth)
      raw,weight = Tensor(raw_values,device='CL').realize(),Tensor(weights,device='CL').realize()
      bias_values = rng.normal(0,.1,columns).astype(np.float16)
      bias = Tensor(bias_values,device='CL').realize()
      layer = MaliQ8Linear(raw,columns,depth,bias)
      for dtype in (np.float32,np.float16):
        values = rng.normal(0,.3,(1,1,depth)).astype(dtype)
        x = Tensor(values,device='CL').realize()
        expected,actual = gemv(x.float(),weight).numpy(),gemv_q8(x,raw,columns,depth).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_array_equal(layer(x).numpy(),(expected+bias_values.astype(np.float32)).astype(dtype))

  def test_symbolic_batched_rows_rebind_without_reading_invalid_tail(self):
    from tinygrad.llm.kernels.mali_linear import gemm
    rng = np.random.default_rng(201)
    raw_values,weights = q8_fixture(67,96)
    raw,weight = Tensor(raw_values,device='CL').realize(),Tensor(weights,device='CL').realize()
    layer = MaliQ8Linear(raw,67,96)
    run = TinyJit(lambda x,rows:layer(x[:,:rows]).realize())
    for count in (32,17,1,31,7,32):
      values = rng.normal(0,.3,(2,32,96)).astype(np.float32)
      values[:,count:] = np.nan
      actual = run(Tensor(values,device='CL').realize(),UOp.variable('q8_rows',1,32).bind(count)).pad_to((2,32,67)).numpy()[:,:count]
      expected = gemm(Tensor(values[:,:count].copy(),device='CL').realize(),weight).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


if __name__=='__main__':unittest.main()
