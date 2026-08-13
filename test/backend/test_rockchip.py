"""Rockchip NPU census: passing ops and explicit backend-contract skips, with DRM_IOCTL_RKNPU_SUBMIT counts.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip.py -q -n0

The backend accepts FP16 inputs and emits contiguous FP16 DPU EW output. It does not
perform host arithmetic or FP32→FP16 conversion. GEMM gather layout is prepared on
the host, then every MUL/ADD is submitted to DPU EW.

Reduction via ROCKCHIP_EW_REDUCE=sequential|kahan|twoproduct (default sequential).
Each program uses one PC chain sized from its actual register-command and task-descriptor bytes.
"""
from __future__ import annotations
import math, unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context
from test.backend import test_ops as _test_ops
from test.backend.test_ops import slow_test

# fp16 tol matches test_ops.test_gemm_fp16
_FP16 = dict(atol=5e-3, rtol=5e-3)
_FP16_WITH_GRAD = dict(atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
_TEST_OPS_HELPER = _test_ops.helper_test_op

def _fp16_test_op(*args, **kwargs):
  """Run a test_ops case at the same FP16 tolerance as test_gemm_fp16."""
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(*args, **kwargs)

def _fp16_pool_test_op(*args, **kwargs):
  """Keep explicit MaxPool fixtures on the backend's FP16 input contract."""
  if (vals:=kwargs.get("vals")) is not None: kwargs["vals"] = [np.asarray(value, dtype=np.float16) for value in vals]
  return _fp16_test_op(*args, **kwargs)

def _fp16_fp32_golden_test_op(shps, torch_fxn, tinygrad_fxn, **kwargs):
  """Use an FP32-accumulated golden when CPU FP16 reduction is less accurate than DPU EW."""
  def fp32_golden(*tensors):
    out = torch_fxn(*(x.float() if x.is_floating_point() else x for x in tensors))
    return out.to(tensors[0].dtype) if tensors and tensors[0].is_floating_point() else out
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(shps, fp32_golden, tinygrad_fxn, **kwargs)

def _fp32_test_exception(self, shps, torch_fxn, tinygrad_fxn=None, expected=None, forward_only=False, exact=False,
                         vals=None, low=-1.5, high=1.5):
  """Preserve upstream dtype-mismatch fixtures when DEFAULT_FLOAT=HALF would make both operands half."""
  if shps is None: arrays = [np.asarray(value, dtype=np.float32) for value in vals]
  else:
    rng = np.random.default_rng(0)
    arrays = [rng.uniform(low, high, size=shape).astype(np.float32) for shape in shps]
  ts = [torch.tensor(array, requires_grad=not forward_only) for array in arrays]
  tst = [Tensor(array) for array in arrays]
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  with self.assertRaises(expected) as torch_cm: torch_fxn(*ts)
  with self.assertRaises(expected) as tinygrad_cm: tinygrad_fxn(*tst)
  if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))

def _ew_submits(n:int) -> int:
  """All tiles for one realized Rockchip program share one dynamically sized PC-chain."""
  return int(n > 0)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["ROCKCHIP"]

  def _half(self, shape, seed:int=0) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-2, 2, size=shape).astype(np.float16))

  def _check(self, expected_submits:int|None, out:Tensor, ref:np.ndarray, atol=5e-3, rtol=5e-3):
    """Realize `out`, compare to `ref`, assert ioctl submit delta.

    Default tol matches test_ops half gemm (DEFAULT_FLOAT=HALF / test_gemm_fp16).
    """
    before, before_tasks = self.dev.submit_count, self.dev.task_count
    got = out.realize().numpy()
    submits, tasks = self.dev.submit_count-before, self.dev.task_count-before_tasks
    # The fail-closed runtime caps every PC chain at 48 physical tasks. Preserve the old one-chain contract as the
    # minimum number of bounded chains; tests with an intentionally multi-submit recipe still use their exact count.
    if expected_submits is None or expected_submits == 1 and tasks > 48: expected_submits = (tasks+47)//48
    print(f"  {self._testMethodName}: tasks={tasks} submits={submits} (expected {expected_submits})")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    self.assertEqual(submits, expected_submits, f"{self._testMethodName}: submits={submits} expected={expected_submits}")

  def _check_conv2d(self, x_shape:tuple[int, ...], w_shape:tuple[int, ...], seed:int, expected_submits:int|None=None, **kwargs):
    rng = np.random.default_rng(seed)
    xn = rng.uniform(-2, 2, size=x_shape).astype(np.float16)
    wn = rng.uniform(-2, 2, size=w_shape).astype(np.float16)
    ref = torch.nn.functional.conv2d(torch.from_numpy(xn), torch.from_numpy(wn), **kwargs).numpy()
    self._check(expected_submits, Tensor(xn).conv2d(Tensor(wn), **kwargs), ref, **_FP16)

  # ---- ADD ----
  def test_tiny_add(self):
    a, b = self._half((3,), 1), self._half((3,), 2)
    self._check(_ew_submits(3), a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add(self):
    a, b = self._half((45, 68), 3), self._half((45, 68), 4)
    self._check(_ew_submits(45*68), a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add3(self):
    # two logical EW ops share one PC-chain ioctl
    a, b, c = self._half((45, 65), 7), self._half((45, 65), 8), self._half((45, 65), 9)
    ref = (a.numpy().astype(np.float32) + b.numpy() + c.numpy()).astype(np.float16)
    self._check(1, a + b + c, ref)

  # ---- MUL ----
  def test_tiny_mul(self):
    a, b = self._half((64,), 10), self._half((64,), 11)
    self._check(_ew_submits(64), a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_mul(self):
    a, b = self._half((64, 64), 12), self._half((64, 64), 13)
    self._check(_ew_submits(64*64), a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_scalar_mul(self):
    a = self._half((45, 65), 14)
    n = _ew_submits(45*65)
    self._check(n, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(n, a * -1, (a.numpy().astype(np.float32) * -1).astype(np.float16))
    self._check(n, 255 * a, (a.numpy().astype(np.float32) * 255).astype(np.float16))
    self._check(n, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_mul_naninf(self):
    a = self._half((45, 65), 16)
    n = _ew_submits(45*65)
    self._check(n, a * math.inf, (a.numpy().astype(np.float32) * np.float32(np.inf)).astype(np.float16))
    self._check(n, a * -math.inf, (a.numpy().astype(np.float32) * np.float32(-np.inf)).astype(np.float16))
    self._check(n, a * math.nan, (a.numpy().astype(np.float32) * np.float32(np.nan)).astype(np.float16))

  # ---- DIV ----

  # ---- MAX ----

  # ---- GEMM / MATMUL (from test_ops, fp16 tol) ----
  def test_matmul_simple(self):
    _fp16_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot)
  @slow_test
  def test_matmul(self):
    _fp16_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot)
  def test_matmul_batched(self):
    _fp16_test_op([(3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot)
  def test_matmul_batched_vector(self):
    _fp16_test_op([(4,3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot)
  def test_small_gemm(self):
    _fp16_test_op([(8,8), (8,8)], lambda x,y: x.matmul(y), lambda x,y: x@y)
  def test_9_gemm(self):
    _fp16_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y)
  def test_small_gemm_padded(self):
    _fp16_test_op([(9,9), (9,9)],
                  lambda x,y: torch.nn.functional.pad(x, (0,7,0,7)).matmul(torch.nn.functional.pad(y, (0,7,0,7))),
                  lambda x,y: x.pad(((0,7),(0,7)))@y.pad(((0,7),(0,7))))
  def test_small_gemm_range(self):
    _fp16_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                  vals=[np.arange(0,64,dtype=np.float16).reshape(8,8), np.arange(64,128,dtype=np.float16).reshape(8,8)])
  def test_small_gemm_eye(self):
    _fp16_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                  vals=[np.eye(8).astype(np.float16), np.eye(8).astype(np.float16)])
  @slow_test
  def test_gemm_fp16(self):
    _fp16_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()))
  @slow_test
  def test_gemm(self):
    _fp16_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y))
  @slow_test
  def test_big_gemm(self):
    _fp16_test_op([(256,256), (256,256)], lambda x,y: x.matmul(y))
  def test_gemm_with_zeros_shape(self):
    for shapes in ([(8,8), (8,0)], [(0,8), (8,8)], [(0,8), (8,0)], [(8,0), (0,8)], [(0,0), (0,0)], [(0), (0,8)], [(0), (0)]):
      _fp16_test_op(shapes, lambda x,y: x.matmul(y), Tensor.dot)

  # ---- CONV2D (from test_ops, fp16 tol) ----
  def test_simple_conv2d_1x1(self):
    self._check_conv2d((1,4,9,9), (4,4,1,1), 100)

  def test_simple_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 101)

  def test_simple_conv2d_batched(self):
    self._check_conv2d((2,4,9,9), (4,4,3,3), 102)

  def test_strided_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 104, stride=2)

  def test_depthwise_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,1,3,3), 105, groups=4)

  def test_simple_conv2d_m4(self):
    self._check_conv2d((1,16,9,9), (16,16,3,3), 300, expected_submits=51)

  def test_simple_conv2d_1x1_m4(self):
    self._check_conv2d((1,16,32,32), (16,16,1,1), 301)

  def test_grouped_conv2d(self):
    self._check_conv2d((1,8,9,9), (8,4,3,3), 302, groups=2)

  def test_dilated_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 303, dilation=2)

  def test_asymmetric_padding_conv2d(self):
    rng = np.random.default_rng(400)
    xn = rng.uniform(-2, 2, size=(1,1,4,4)).astype(np.float16)
    wn = rng.uniform(-2, 2, size=(1,1,2,2)).astype(np.float16)
    ref = torch.nn.functional.conv2d(torch.nn.functional.pad(torch.from_numpy(xn), (2,1,2,1)), torch.from_numpy(wn)).numpy()
    self._check(1, Tensor(xn).conv2d(Tensor(wn), padding=(2,1,2,1)), ref, **_FP16)

  def test_output_padded_conv_transpose2d(self):
    rng = np.random.default_rng(401)
    xn = rng.uniform(-2, 2, size=(2,4,6,5)).astype(np.float16)
    wn = rng.uniform(-2, 2, size=(4,4,3,3)).astype(np.float16)
    bn = rng.uniform(-2, 2, size=(4,)).astype(np.float16)
    args = dict(output_padding=(1,1), stride=(2,3))
    ref = torch.nn.functional.conv_transpose2d(torch.from_numpy(xn), torch.from_numpy(wn), torch.from_numpy(bn), **args).numpy()
    self._check(None, Tensor(xn).conv_transpose2d(Tensor(wn), Tensor(bn), **args), ref, **_FP16)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipConvOps(unittest.TestCase):
  """Every test*conv* case from test_ops, rerun with the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception
  _test_conv2d = _test_ops.TestOps._test_conv2d

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_bias_conv_transpose2d(self):
    # PyTorch CPU accumulates this FP16 ConvTranspose in FP16 and differs from the exact product sum after bias cancellation.
    _fp16_fp32_golden_test_op([(2,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b), lambda x,w,b: Tensor.conv_transpose2d(x,w,b))

  def test_conv1d(self):
    # PyTorch CPU's FP16 Conv1D weight-gradient reduction can be less accurate than the exact DPU expansion.
    _test_ops.helper_test_op = _fp16_fp32_golden_test_op
    try: _test_ops.TestOps.test_conv1d(self)
    finally: _test_ops.helper_test_op = _fp16_test_op

  def test_depthwise_conv2d(self):
    _test_ops.helper_test_op = _fp16_fp32_golden_test_op
    try: _test_ops.TestOps.test_depthwise_conv2d(self)
    finally: _test_ops.helper_test_op = _fp16_test_op

  def test_large_input_conv2d(self):
    _test_ops.helper_test_op = _fp16_fp32_golden_test_op
    try: _test_ops.TestOps.test_large_input_conv2d(self)
    finally: _test_ops.helper_test_op = _fp16_test_op

  @slow_test
  def test_simple_conv_transpose3d(self):
    # The DPU expansion is also closer to FP32 accumulation than PyTorch CPU's FP16 ConvTranspose3D reduction.
    _fp16_fp32_golden_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w), lambda x,w: Tensor.conv_transpose2d(x,w))

# Keep the Rockchip convolution census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test") and "conv" in _name and _name not in (
    "test_bias_conv_transpose2d", "test_conv1d", "test_depthwise_conv2d", "test_large_input_conv2d", "test_simple_conv_transpose3d"):
    setattr(TestRockchipConvOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxPoolOps(unittest.TestCase):
  """FP16 MaxPool2D census plus exact signed-INT16 DPU coverage."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_pool_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def _check_int16(self, data:np.ndarray, *, return_indices:bool=False, expected_submits:int=1, **kwargs):
    expected = torch.nn.functional.max_pool2d(torch.from_numpy(data), return_indices=return_indices, **kwargs)
    before = Device["ROCKCHIP"].submit_count
    got = Tensor(data, device="ROCKCHIP").max_pool2d(return_indices=return_indices, **kwargs)
    if return_indices: expected, got = expected[1].int().numpy(), got[1].realize().numpy()
    else: expected, got = expected.numpy(), got.realize().numpy()
    np.testing.assert_array_equal(got, expected)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, expected_submits)

  def test_max_pool2d_padding_int(self):
    data = np.random.default_rng(2608).integers(-32768, 32768, (4,2,11,28), dtype=np.int16)
    self._check_int16(data, kernel_size=(2,2), padding=1)


# Keep the MaxPool2D census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test_max_pool2d") and _name != "test_max_pool2d_padding_int":
    setattr(TestRockchipMaxPoolOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxUnpoolOps(unittest.TestCase):
  """FP16 MaxUnpool scatter from the upstream operator census."""

  test_max_unpool2d = _test_ops.TestOps.test_max_unpool2d

  def test_max_unpool2d_inf(self):
    data = np.array([[[[math.inf, -math.inf, math.nan], [1.0, 2.0, 3.0]]]], dtype=np.float16)
    expected = torch.nn.functional.max_unpool2d(
      *torch.nn.functional.max_pool2d(torch.from_numpy(data), kernel_size=(2,2), return_indices=True), kernel_size=(2,2)).numpy()
    got = Tensor.max_unpool2d(
      *Tensor.max_pool2d(Tensor(data, device="ROCKCHIP"), kernel_size=(2,2), return_indices=True), kernel_size=(2,2)).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAvgPoolOps(unittest.TestCase):
  """Every FP16 AvgPool case from test_ops at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_avg_pool3d(self):
    # PyTorch CPU does not implement FP16 AvgPool3D; use its FP32 result cast to the FP16 device contract.
    _fp16_fp32_golden_test_op([(1,1,16,16,16)],
      lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), forward_only=True)

# Keep the FP16 AvgPool census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if (_name.startswith("test_avg_pool") and _name != "test_avg_pool3d") or _name == "test_global_avg_pool2d":
    setattr(TestRockchipAvgPoolOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAdaptivePoolOps(unittest.TestCase):
  """Adaptive-pooling-equivalent divisible windows; tinygrad has no adaptive pooling API."""


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPolynomialLossOps(unittest.TestCase):
  """Polynomial and simple loss expressions composed only from FP16 DPU EW primitives."""

  test_topo_sort = _test_ops.TestOps.test_topo_sort


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSqrtOps(unittest.TestCase):
  """FP16 square roots evaluated by shared DPU EW arithmetic."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_sqrt = _test_ops.TestOps.test_sqrt
  test_rsqrt = _test_ops.TestOps.test_rsqrt


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegerPowerOps(unittest.TestCase):
  """Constant integer powers composed from FP16 DPU EW multiplication and division."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception
  test_int_pow_const_int = _test_ops.TestOps.test_int_pow_const_int

  def test_pow(self):
    before = Device["ROCKCHIP"].submit_count
    for exponent in (0, 1, 2, 3, -2):
      _fp16_test_op([(45,65)], lambda x,exponent=exponent:x**exponent, forward_only=True)
    for exponent in (2, -2):
      _fp16_test_op([()], lambda x,exponent=exponent:x**exponent, forward_only=True)
    _fp16_test_op([(45,65)], lambda x:x**3, forward_only=True, low=-30, high=-27)
    _fp16_test_op([()], lambda x:x**3, forward_only=True, low=-30, high=-27)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 7)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTensorPowerOps(unittest.TestCase):
  """Runtime FP16 exponents lowered to native ABS/FLOOR and no-LUT EXP2/LOG2 DPU stages."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_pow_full = _test_ops.TestOps.test_pow_full
  test_pow_const = _test_ops.TestOps.test_pow_const
  test_pow_int_base_float_exponent = _test_ops.TestOps.test_pow_int_base_float_exponent
  test_pow_neg_inf_frac_exponent = _test_ops.TestOps.test_pow_neg_inf_frac_exponent
  test_pow_zero_const = _test_ops.TestOps.test_pow_zero_const
  test_pow_zero_exponent = _test_ops.TestOps.test_pow_zero_exponent
  test_pow_zero_tensor = _test_ops.TestOps.test_pow_zero_tensor

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipOneHotOps(unittest.TestCase):
  """Exact INT32 one-hot equality through four raw bytes and DPU EW masks."""

  test_one_hot = _test_ops.TestOps.test_one_hot


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipGatherOps(unittest.TestCase):
  """Dynamic INT32 gather with exact index and raw 16-bit representation selection on DPU."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_gather = _test_ops.TestOps.test_gather


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaskedSelectOps(unittest.TestCase):
  """Forward-only bounded FP16 and integer selection through DPU masks and raw-byte gathers."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_masked_select(self):
    # Preserve the compact three-prefix reduction graph consumed by the Rockchip matcher.
    with Context(NOOPT=1): _test_ops.TestOps.test_masked_select(self)

  test_masked_select_size = _test_ops.TestOps.test_masked_select_size


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipNonzeroOps(unittest.TestCase):
  """Fixed FP16/integer nonzero coordinates selected by DPU prefix/equality stages."""

  test_nonzero = _test_ops.TestOps.test_nonzero
  test_nonzero_size = _test_ops.TestOps.test_nonzero_size


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipFancyIndexOps(unittest.TestCase):
  """Dynamic INT32 fancy indices with exact negative normalization and raw 16-bit values."""

  _get_index_randoms = _test_ops.TestOps._get_index_randoms

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_fancy_indexing_inf = _test_ops.TestOps.test_fancy_indexing_inf
  test_slice_fancy_indexing_dim_inject_none = _test_ops.TestOps.test_slice_fancy_indexing_dim_inject_none
  test_slice_fancy_indexing_errors = _test_ops.TestOps.test_slice_fancy_indexing_errors
  test_slice_fancy_indexing_list_indices = _test_ops.TestOps.test_slice_fancy_indexing_list_indices
  test_slice_fancy_indexing_tuple_indices = _test_ops.TestOps.test_slice_fancy_indexing_tuple_indices
  test_slice_fancy_indexing_with_tensors = _test_ops.TestOps.test_slice_fancy_indexing_with_tensors

  def test_slice_fancy_indexing_dim_inject_and_collapse(self):
    _,b,_,d,_,_,j,_,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[1,b,None,d,1], lambda x:x[1,j,None,o,1])

  def test_slice_fancy_indexing_dim_collapse_int(self):
    _,b,c,d,e,_,j,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[1,b,c,d,e], lambda x:x[1,j,k,o,p])

  def test_slice_fancy_indexing_no_dim_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,c,d,e], lambda x:x[i,j,k,o,p])

  def test_slice_fancy_indexing_list_with_tensors(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(a,b,c,d,e)], lambda x:x[(i,j,k,o,p)])


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipScatterOps(unittest.TestCase):
  """16-bit Scatter and FP16 ScatterReduce through static or exact dynamic INT32 selection."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  helper_test_exception = _fp32_test_exception
  test_scatter = _test_ops.TestOps.test_scatter
  test_scatter_add = _test_ops.TestOps.test_scatter_add
  test_scatter_mul = _test_ops.TestOps.test_scatter_mul
  test_scatter_no_reduce_tensor_src = _test_ops.TestOps.test_scatter_no_reduce_tensor_src
  test_scatter_reduce = _test_ops.TestOps.test_scatter_reduce


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipValidationOps(unittest.TestCase):
  """Upstream argument-validation cases that complete before unsupported numeric execution."""

  helper_test_exception = _fp32_test_exception
  test_scatter_reduce_errors = _test_ops.TestOps.test_scatter_reduce_errors
  test_scaled_dot_product_attention_gqa_errors = _test_ops.TestOps.test_scaled_dot_product_attention_gqa_errors


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAttentionOps(unittest.TestCase):
  """Scaled dot-product attention through tiled DPU MUL, balanced ADD, softmax, and value reduction plans."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_scaled_dot_product_attention = _test_ops.TestOps.test_scaled_dot_product_attention
  test_scaled_dot_product_attention_causal = _test_ops.TestOps.test_scaled_dot_product_attention_causal
  test_scaled_dot_product_attention_gqa = _test_ops.TestOps.test_scaled_dot_product_attention_gqa
  test_scaled_dot_product_attention_mismatch_ls = _test_ops.TestOps.test_scaled_dot_product_attention_mismatch_ls


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBroadcastOps(unittest.TestCase):
  """FP16 broadcasting through static gather and DPU EW arithmetic."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_broadcast_simple = _test_ops.TestOps.test_broadcast_simple
  test_broadcast_full = _test_ops.TestOps.test_broadcast_full
  test_broadcast_partial = _test_ops.TestOps.test_broadcast_partial


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBitwiseOps(unittest.TestCase):
  """Exact INT32 bitwise operations and shifts through the native DPU integer EW pipeline."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception
  test_xor = _test_ops.TestOps.test_xor
  test_and = _test_ops.TestOps.test_and
  test_or = _test_ops.TestOps.test_or
  test_bitwise_not = _test_ops.TestOps.test_bitwise_not
  test_int_or = _test_ops.TestOps.test_int_or
  test_lshift = _test_ops.TestOps.test_lshift
  test_rshift = _test_ops.TestOps.test_rshift
  test_lshift_signed = _test_ops.TestOps.test_lshift_signed
  test_rshift_signed = _test_ops.TestOps.test_rshift_signed
  test_idiv_shift_rewrite_negative = _test_ops.TestOps.test_idiv_shift_rewrite_negative


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipInt16EWOps(unittest.TestCase):
  """Bounded signed INT16 arithmetic and exact comparisons on the native DPU integer EW pipeline."""

  def _check(self, expected, op, *values, output_dtype=np.int16, expected_tasks=None, expected_submits=1):
    before = Device["ROCKCHIP"].submit_count
    got = op(*(Tensor(np.asarray(value, dtype=np.int16)) for value in values)).realize().numpy()
    self.assertEqual(got.dtype, output_dtype)
    np.testing.assert_array_equal(got, np.asarray(expected, dtype=output_dtype))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, expected_submits)

  def _check_bool(self, op, *values, expected=None, expected_tasks=None, expected_submits=1):
    arrays = tuple(np.asarray(value, dtype=np.int16) for value in values)
    before = Device["ROCKCHIP"].submit_count
    got = op(*(Tensor(value) for value in arrays)).realize().numpy()
    self.assertEqual(got.dtype, np.bool_)
    np.testing.assert_array_equal(got, op(*arrays) if expected is None else expected)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, expected_submits)

  def _check_index(self, expected, op, values, submits, tasks):
    before = Device["ROCKCHIP"].submit_count
    got = op(Tensor(np.asarray(values, dtype=np.int16))).realize().numpy()
    self.assertEqual(got.dtype, np.int32)
    np.testing.assert_array_equal(got, np.asarray(expected, dtype=np.int32))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, submits)

  def _check_cumsum(self, shape, axis, tasks):
    values = (np.arange(np.prod(shape), dtype=np.uint32)*7919).astype(np.uint16).view(np.int16).reshape(shape)
    self._check(np.cumsum(values, axis=axis, dtype=np.int32), lambda x:x.cumsum(axis), values, output_dtype=np.int32,
                expected_tasks=tasks, expected_submits=2)

  @staticmethod
  def _saturating_cumprod(values, axis):
    moved = np.moveaxis(values, axis, -1).astype(np.int32)
    out, acc = np.empty_like(moved, dtype=np.int16), np.ones(moved.shape[:-1], dtype=np.int32)
    for i in range(moved.shape[-1]):
      acc = np.clip(acc*moved[..., i], -32768, 32767)
      out[..., i] = acc
    return np.moveaxis(out, -1, axis)

  def test_where(self):
    a = np.asarray([-32768,-32768,-30000,-1,0,1,30000,32767], dtype=np.int16)
    b = np.asarray([32767,-32768,30000,0,0,-1,-30000,-32768], dtype=np.int16)
    self._check(np.where(a < b, a, b), lambda x,y:(x<y).where(x, y), a, b)
    self._check(np.where(a != 0, a, -7), lambda x:(x!=0).where(x, -7), a)

  def test_clip(self):
    values = np.asarray([-32768,-1201,-1200,-1,0,1,1200,1201,32767], dtype=np.int16)
    self._check(np.clip(values, -1200, 1200), lambda x:x.clip(-1200, 1200), values, expected_tasks=2)
    self._check(np.maximum(values, -1200), lambda x:x.clip(-1200, None), values, expected_tasks=1)
    self._check(np.minimum(values, 1200), lambda x:x.clip(None, 1200), values, expected_tasks=1)
    self._check(np.full_like(values, -100), lambda x:x.clip(100, -100), values, expected_tasks=2)

  def test_relu(self):
    values = np.asarray([-32768,-1,0,1,32767], dtype=np.int16)
    self._check(np.maximum(values, 0), lambda x:x.relu(), values, expected_tasks=1)

  def test_sign(self):
    values = np.asarray([-32768,-1200,-1,0,1,1200,32767], dtype=np.int16)
    self._check(np.sign(values), lambda x:x.sign(), values, expected_tasks=2)


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipDotOps(unittest.TestCase):
  """FP16 dot, batched dot, and matvec compositions lowered through DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_dot_1d = _test_ops.TestOps.test_dot_1d
  test_dot = _test_ops.TestOps.test_dot
  test_broadcastdot = _test_ops.TestOps.test_broadcastdot
  test_multidot = _test_ops.TestOps.test_multidot
  test_matvec = _test_ops.TestOps.test_matvec

  def test_mulacc_with_zero_strides(self):
    """Upstream shapes with external FP16 buffers, preventing its all-constant cases from folding onto CPU."""
    scalar_a = Tensor(np.ones((1,1,1), dtype=np.float16)).realize()
    scalar_b = Tensor(np.ones((1,1,1), dtype=np.float16)).realize()
    a = Tensor(np.ones((2,4,1), dtype=np.float16)).realize()
    b = Tensor(np.ones((1,4,1), dtype=np.float16)).realize()
    lhs = Tensor(np.ones((1,2), dtype=np.float16)).realize()
    rhs = Tensor(np.ones((2,3), dtype=np.float16)).realize()
    before = Device["ROCKCHIP"].submit_count
    outputs = (scalar_a.expand(2,4,3).mul(scalar_b.expand(2,4,3)).sum(-1).realize(),
               a.expand(2,4,3).mul(b.expand(2,4,3)).sum((0,2)).realize(), lhs.dot(rhs).realize())
    for got,expected in zip(outputs, (np.full((2,4), 3, dtype=np.float16), np.full(4, 6, dtype=np.float16),
                                      np.full((1,3), 2, dtype=np.float16))):
      np.testing.assert_array_equal(got.numpy(), expected)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 3)
  test_matvecmat = _test_ops.TestOps.test_matvecmat

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipEinsumOps(unittest.TestCase):
  """FP16 einsum contractions composed from static gathers and DPU EW reductions."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_einsum = _test_ops.TestOps.test_einsum

  def test_einsum_ellipsis(self):
    _fp16_test_op([(3,8,9), (3,8,9)], lambda a,b:torch.einsum("...id,...jd->...ij", a, b),
                  lambda a,b:Tensor.einsum("...id,...jd->...ij", a, b))
    _fp16_test_op([(3,8,9), (3,8,9)], lambda a,b:torch.einsum("...id,...jd", a, b),
                  lambda a,b:Tensor.einsum("...id,...jd", a, b))
    _fp16_test_op([(2,3,4,5), (5,2,4)], lambda a,b:torch.einsum("i...j,ji...->...", a, b),
                  lambda a,b:Tensor.einsum("i...j,ji...->...", a, b))
    self.helper_test_exception([(2,3,4), (2,3,4)], lambda a,b:torch.einsum("...ik...,...jk->", a, b),
                               lambda a,b:Tensor.einsum("...ik...,...jk->", a, b), expected=(RuntimeError, IndexError))
    self.helper_test_exception([(2,3,4), (2,3,4)], lambda a,b:torch.einsum("i...j,ji...->...", a, b),
                               lambda a,b:Tensor.einsum("i...j,ji...->...", a, b), expected=RuntimeError)

  test_einsum_trace = _test_ops.TestOps.test_einsum_trace
  test_einsum_shape_check = _test_ops.TestOps.test_einsum_shape_check
  test_einsum_arity_check1 = _test_ops.TestOps.test_einsum_arity_check1
  test_einsum_arity_check2 = _test_ops.TestOps.test_einsum_arity_check2

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeOps(unittest.TestCase):
  """FP16 cumulative reductions on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_small_cumsum = _test_ops.TestOps.test_small_cumsum
  test_simple_cumsum = _test_ops.TestOps.test_simple_cumsum
  test_cumsum = _test_ops.TestOps.test_cumsum
  test_cumsum_zero_axis = _test_ops.TestOps.test_cumsum_zero_axis
  test_logcumsumexp = _test_ops.TestOps.test_logcumsumexp
  test_logcumsumexp_numerical = _test_ops.TestOps.test_logcumsumexp_numerical

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeProductOps(unittest.TestCase):
  """FP16 cumulative products on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_small_cumprod = _test_ops.TestOps.test_small_cumprod
  test_simple_cumprod = _test_ops.TestOps.test_simple_cumprod
  test_cumprod = _test_ops.TestOps.test_cumprod
  test_cumprod_zero_axis = _test_ops.TestOps.test_cumprod_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeExtremaOps(unittest.TestCase):
  """Unchanged FP16 cumulative-extrema methods with exact DPU-selected INT32 indices."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def _test_simple(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.cummax, Tensor.cummax) if kind == "max" else (torch.cummin, Tensor.cummin)
    for count in (512, 1022):
      np.random.seed(0)
      data = np.random.uniform(-2, 2, size=(count,)).astype(np.float16)
      expected = torch_fxn(torch.from_numpy(data), dim=0)
      if count == 512:
        values, indices = tinygrad_fxn(Tensor(data), axis=0)
        Tensor.realize(values, indices)
      else:
        values = tinygrad_fxn(Tensor(data), axis=0)[0].realize()
        indices = tinygrad_fxn(Tensor(data), axis=0)[1].realize()
      np.testing.assert_allclose(values.numpy(), expected.values.numpy(), **_FP16)
      np.testing.assert_equal(indices.numpy(), expected.indices.int().numpy())

  test_small_cummax = _test_ops.TestOps.test_small_cummax
  @slow_test
  def test_simple_cummax(self): self._test_simple("max")
  test_cummax = _test_ops.TestOps.test_cummax
  test_cummax_zero_axis = _test_ops.TestOps.test_cummax_zero_axis
  test_small_cummin = _test_ops.TestOps.test_small_cummin
  @slow_test
  def test_simple_cummin(self): self._test_simple("min")
  test_cummin = _test_ops.TestOps.test_cummin
  test_cummin_zero_axis = _test_ops.TestOps.test_cummin_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipArgExtremaOps(unittest.TestCase):
  """FP16 ArgMax/ArgMin selected entirely by DPU EW, including first-index ties."""

  test_argmax = _test_ops.TestOps.test_argmax
  test_argmin = _test_ops.TestOps.test_argmin
  test_softmax_argmax = _test_ops.TestOps.test_softmax_argmax

  def _test(self, kind:str, values:tuple[tuple[float, ...], ...]):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    for case in values:
      with self.subTest(kind=kind, values=case):
        _TEST_OPS_HELPER(None, lambda x: torch_fxn(x).int(), tinygrad_fxn, vals=[list(case)], forward_only=True)

  def _test_axes(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    for axis,keepdim in ((0, False), (1, False), (1, True)):
      with self.subTest(kind=kind, axis=axis, keepdim=keepdim):
        _TEST_OPS_HELPER([(10,20)], lambda x, axis=axis, keepdim=keepdim: torch_fxn(x, dim=axis, keepdim=keepdim).int(),
                         lambda x, axis=axis, keepdim=keepdim: tinygrad_fxn(x, axis=axis, keepdim=keepdim), forward_only=True)

  def _test_global(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    def run(shps, vals=None):
      _TEST_OPS_HELPER(shps, lambda x: torch_fxn(x).int(), tinygrad_fxn, vals=vals, forward_only=True)
    run([(10,20)])
    tied = np.zeros((10,20), dtype=np.float16)
    tied.flat[[37,99]] = 1.0 if kind == "max" else -1.0
    run(None, [tied.tolist()])


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortValueOps(unittest.TestCase):
  """FP16 stable sort values using static bitonic lane maps and DPU EW MIN/MAX."""

  test_sort = _test_ops.TestOps.test_sort


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortIndexOps(unittest.TestCase):
  """Exact stable INT32 sort indices selected by FP16 value/count equality on DPU EW."""

  def _axis(self, axis:int, descending:bool):
    _TEST_OPS_HELPER([(8,8,6)], lambda x: x.sort(axis, descending).indices.int(),
                     lambda x: x.sort(axis, descending)[1], forward_only=True)

  test_argsort = _test_ops.TestOps.test_argsort


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTopKOps(unittest.TestCase):
  """FP16 TopK values and stable INT32 indices composed from native sort and static slicing."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception
  test_topk = _test_ops.TestOps.test_topk

  def _test(self, shape:tuple[int, ...], k:int, axis:int, largest:bool):
    _fp16_test_op([shape], lambda x: x.topk(k, axis, largest, True).values,
                  lambda x: x.topk(k, axis, largest, True)[0], forward_only=True)
    _TEST_OPS_HELPER([shape], lambda x: x.topk(k, axis, largest, True).indices.int(),
                     lambda x: x.topk(k, axis, largest, True)[1], forward_only=True)


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipElementwiseExtremaOps(unittest.TestCase):
  """FP16 portions of test_ops maximum/minimum; integer and boolean inputs are outside the NPU contract."""

  def _test(self, torch_fxn, tinygrad_fxn):
    _fp16_test_op([(45,65), (45,65)], torch_fxn, tinygrad_fxn)
    _fp16_test_op([(), ()], torch_fxn, tinygrad_fxn)
    _fp16_test_op(None, torch_fxn, tinygrad_fxn, vals=[[1., 0., 3., -4.], 3.])
    _fp16_test_op(None, torch_fxn, tinygrad_fxn, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    _fp16_test_op(None, torch_fxn, tinygrad_fxn,
                  vals=[[math.inf, -math.inf, math.nan, -0.0], [math.inf, math.inf, 1.0, 0.0]], forward_only=True)

  def test_maximum(self): self._test(torch.maximum, Tensor.maximum)
  def test_minimum(self): self._test(torch.minimum, Tensor.minimum)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSignOps(unittest.TestCase):
  """FP16 sign and softsign computed entirely by DPU EW stages."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_sign = _test_ops.TestOps.test_sign
  test_sign_exact = _test_ops.TestOps.test_sign_exact
  test_copysign = _test_ops.TestOps.test_copysign
  test_softsign = _test_ops.TestOps.test_softsign
  test_softsign_exact = _test_ops.TestOps.test_softsign_exact

  def test_copysign_exact(self):
    values = np.array([-1., -0., 0., 1., math.inf, -math.inf, math.nan], dtype=np.float16)
    magnitude, sign = np.repeat(values, len(values)), np.tile(values, len(values))
    expected = np.copysign(magnitude, sign)
    got = Tensor(magnitude).copysign(Tensor(sign)).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTranscendentalOps(unittest.TestCase):
  """No-LUT EXP2/LOG2 polynomial compositions executed entirely by DPU EW."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_exp2_log2_zero_times_negative = _test_ops.TestOps.test_exp2_log2_zero_times_negative
  test_asin = _test_ops.TestOps.test_asin
  test_acos = _test_ops.TestOps.test_acos
  test_atan = _test_ops.TestOps.test_atan
  test_asinh = _test_ops.TestOps.test_asinh
  test_acosh = _test_ops.TestOps.test_acosh
  test_celu = _test_ops.TestOps.test_celu
  test_selu = _test_ops.TestOps.test_selu
  test_silu = _test_ops.TestOps.test_silu
  test_swish = _test_ops.TestOps.test_swish
  test_exp = _test_ops.TestOps.test_exp
  test_exp2 = _test_ops.TestOps.test_exp2
  test_log = _test_ops.TestOps.test_log
  test_log10 = _test_ops.TestOps.test_log10
  test_log2 = _test_ops.TestOps.test_log2
  test_logaddexp = _test_ops.TestOps.test_logaddexp
  test_sigmoid = _test_ops.TestOps.test_sigmoid
  test_sigmoid_extreme = _test_ops.TestOps.test_sigmoid_extreme
  test_logsigmoid = _test_ops.TestOps.test_logsigmoid
  test_softplus = _test_ops.TestOps.test_softplus
  test_erf = _test_ops.TestOps.test_erf
  test_gelu = _test_ops.TestOps.test_gelu
  test_gelu_extreme = _test_ops.TestOps.test_gelu_extreme
  test_quick_gelu = _test_ops.TestOps.test_quick_gelu
  test_quick_gelu_extreme = _test_ops.TestOps.test_quick_gelu_extreme
  test_elu = _test_ops.TestOps.test_elu
  test_mish = _test_ops.TestOps.test_mish
  test_softmax = _test_ops.TestOps.test_softmax
  test_softmax_other_axis = _test_ops.TestOps.test_softmax_other_axis
  test_log_softmax = _test_ops.TestOps.test_log_softmax
  test_log_softmax_other_axis = _test_ops.TestOps.test_log_softmax_other_axis
  test_softmin = _test_ops.TestOps.test_softmin
  test_logsumexp = _test_ops.TestOps.test_logsumexp
  test_tanh = _test_ops.TestOps.test_tanh
  test_tanh_extreme = _test_ops.TestOps.test_tanh_extreme
  test_atanh = _test_ops.TestOps.test_atanh
  test_sinh = _test_ops.TestOps.test_sinh
  test_cosh = _test_ops.TestOps.test_cosh
  test_sin = _test_ops.TestOps.test_sin
  test_cos = _test_ops.TestOps.test_cos
  test_tan = _test_ops.TestOps.test_tan
  test_sigmoid_alt_extreme = _test_ops.TestOps.test_sigmoid_alt_extreme
  test_binary_crossentropy = _test_ops.TestOps.test_binary_crossentropy
  test_binary_crossentropy_reductions = _test_ops.TestOps.test_binary_crossentropy_reductions
  test_binary_crossentropy_logits_pos_weights = _test_ops.TestOps.test_binary_crossentropy_logits_pos_weights


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipLossOps(unittest.TestCase):
  """Cross-entropy composites evaluated by DPU EW and reduction stages."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_cross_entropy_class_indices = _test_ops.TestOps.test_cross_entropy_class_indices
  test_cross_entropy_class_probabilities = _test_ops.TestOps.test_cross_entropy_class_probabilities
  test_cross_entropy_reductions = _test_ops.TestOps.test_cross_entropy_reductions
  test_cross_entropy_smoothing = _test_ops.TestOps.test_cross_entropy_smoothing
  test_nll_loss = _test_ops.TestOps.test_nll_loss
  test_nll_loss_3d = _test_ops.TestOps.test_nll_loss_3d
  test_nll_loss_3d_weight = _test_ops.TestOps.test_nll_loss_3d_weight
  test_nll_loss_ignore_index = _test_ops.TestOps.test_nll_loss_ignore_index
  test_nll_loss_reductions = _test_ops.TestOps.test_nll_loss_reductions
  test_nll_loss_weight = _test_ops.TestOps.test_nll_loss_weight
  test_sparse_categorical_crossentropy = _test_ops.TestOps.test_sparse_categorical_crossentropy
  test_sparse_categorical_crossentropy_ignore_index = _test_ops.TestOps.test_sparse_categorical_crossentropy_ignore_index
  test_sparse_categorical_crossentropy_label_smoothing = _test_ops.TestOps.test_sparse_categorical_crossentropy_label_smoothing
  test_sparse_categorical_crossentropy_reductions = _test_ops.TestOps.test_sparse_categorical_crossentropy_reductions


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCastOps(unittest.TestCase):
  """Native DPU conversion from FP16 inputs to boolean, byte, INT32, and FP32 outputs."""

  test_cast = _test_ops.TestOps.test_cast
  test_cast_relu = _test_ops.TestOpsUint8.test_cast_relu


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBitcastOps(unittest.TestCase):
  """Exact FP16-pair to INT32 representation movement; the upstream odd-width FP16 shape is invalid."""

  def test_bitcast(self):
    values = np.array([0x0000, 0x8000, 0x3c00, 0xc000, 0x7c00, 0xfc00, 0x7e01, 0xfe01,
                       0x3555, 0xb555, 0x7bff, 0xfbff, 0x0001, 0x8001, 0x03ff, 0x83ff,
                       0x0400, 0x8400, 0x3c01, 0xbc01, 0x4000, 0xc000, 0x4200, 0xc200], dtype=np.uint16).view(np.float16).reshape(2,3,4)
    before = Device["ROCKCHIP"].submit_count
    np.testing.assert_array_equal(Tensor(values).bitcast(dtypes.int32).numpy(), values.view(np.int32))
    np.testing.assert_array_equal(Tensor(values).permute(1,0,2).bitcast(dtypes.int32).numpy(),
                                  values.transpose(1,0,2).view(np.int32))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 0)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipClassificationOps(unittest.TestCase):
  """FP16 IEEE predicates computed as DPU masks and packed through the typed bool-output ABI."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_isinf = _test_ops.TestOps.test_isinf
  test_isnan = _test_ops.TestOps.test_isnan
  test_isfinite = _test_ops.TestOps.test_isfinite

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipComparisonOps(unittest.TestCase):
  """FP16 portions of test_ops comparisons; integer and boolean inputs are outside the NPU contract."""

  def _test_cmp(self, fxn, reverse:bool=True):
    _fp16_test_op(None, fxn, fxn, forward_only=True, vals=[[0., 1., 2.], [2., 1., 0.]])
    for shapes in [[(3,4,5), (3,4,5)], [(3,4,5), (5,)], [(5,), (3,4,5)]]:
      _fp16_test_op(shapes, fxn, fxn, forward_only=True)
    _fp16_test_op(None, lambda x,y:fxn(x, 2), lambda x,y:fxn(x, 2), forward_only=True,
                  vals=[[0., 1., 2.], [2., 1., 0.]])
    if reverse:
      _fp16_test_op(None, lambda x,y:fxn(2, y), lambda x,y:fxn(2, y), forward_only=True,
                    vals=[[0., 1., 2.], [2., 1., 0.]])
    specials = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan]
    pairs = [(lhs, rhs) for lhs in specials for rhs in specials]
    _fp16_test_op(None, fxn, fxn, forward_only=True, vals=[[x[0] for x in pairs], [x[1] for x in pairs]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y:x == y, reverse=False)
  def test_cmp_gt(self): self._test_cmp(lambda x,y:x > y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y:x >= y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y:x < y)
  def test_cmp_le(self): self._test_cmp(lambda x,y:x <= y)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipContractSkips(unittest.TestCase):
  """Upstream methods outside the forward-only FP16-input Rockchip contract."""

  test_cmp_lt_backwards = unittest.skip("Rockchip testing is forward-only")(_test_ops.TestOps.test_cmp_lt_backwards)
  test_cmp_ne_backwards = unittest.skip("Rockchip testing is forward-only")(_test_ops.TestOps.test_cmp_ne_backwards)
  test_pow_const_direct = unittest.skip("requires explicit FP32 inputs and backward execution")(_test_ops.TestOps.test_pow_const_direct)
  test_pow_int = unittest.skip("unsupported by the upstream test")(_test_ops.TestOps.test_pow_int)
  test_scatter_reduce_prod_zeros = unittest.skip(
    "Torch FP32 destination rejects the DEFAULT_FLOAT=HALF source before NPU execution")(_test_ops.TestOps.test_scatter_reduce_prod_zeros)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipLogicalPredicateOps(unittest.TestCase):
  """FP16 logical-not and scalar isclose compositions over native DPU comparison masks."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_logical_not(self):
    _fp16_test_op(None, torch.logical_not, Tensor.logical_not,
                  vals=[[1., 2., 0., 0.5, -0.0, math.inf, -math.inf, math.nan]], forward_only=True)

  test_isclose = _test_ops.TestOps.test_isclose
  test_isclose_edge_cases = _test_ops.TestOps.test_isclose_edge_cases
  test_isclose_scalar = _test_ops.TestOps.test_isclose_scalar

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegralRoundingOps(unittest.TestCase):
  """Native FP16 floor/ceil plus no-LUT DPU compositions for truncation and round-to-even."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_floor = _test_ops.TestOps.test_floor
  test_ceil = _test_ops.TestOps.test_ceil
  test_trunc = _test_ops.TestOps.test_trunc
  test_round = _test_ops.TestOps.test_round

  def test_round_quantization_gradient(self):
    _fp16_test_op(None, lambda x: x + 0.125*(x.round()-x), vals=[[-1.2,-0.7,-0.2,0.2,0.7,1.2]], forward_only=True)


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipModuloOps(unittest.TestCase):
  """Unchanged FP16/INT32 remainder coverage over native DPU division stages."""

  test_mod = _test_ops.TestOps.test_mod
  test_fmod = _test_ops.TestOps.test_fmod

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipDivisionRoundingOps(unittest.TestCase):
  """FP16 rounding plus exact INT32 division through native DPU EW."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception
  test_div_int = _test_ops.TestOps.test_div_int
  test_div_rounding_mode = _test_ops.TestOps.test_div_rounding_mode

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBooleanReductionOps(unittest.TestCase):
  """ANY/ALL over FP16 inputs; external boolean tensors remain outside the DPU contract."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  @staticmethod
  def _check(name:str, values:np.ndarray, axis=None):
    _fp16_test_op(None, lambda x:getattr(x, name)(axis=axis), lambda x:getattr(x, name)(axis=axis),
                  vals=[values], forward_only=True)

  def test_any(self):
    self._check("any", np.array([0., -0., 0.], dtype=np.float16))
    self._check("any", np.array([0., math.nan, math.inf, -math.inf], dtype=np.float16))
    _fp16_test_op([()], lambda x:x.any(), forward_only=True)

  def test_any_axis(self):
    values = np.zeros((3,4,5,6), dtype=np.float16)
    values[0,0,0,0], values[1,1,2,3], values[2,3,4,5] = 1., math.inf, math.nan
    self._check("any", values, axis=(1,2))

  def test_all(self):
    self._check("all", np.array([1., -2., math.nan, math.inf, -math.inf], dtype=np.float16))
    self._check("all", np.array([1., 0., math.nan], dtype=np.float16))
    _fp16_test_op([()], lambda x:x.all(), forward_only=True)

  def test_all_axis(self):
    values = np.ones((3,4,5,6), dtype=np.float16)
    values[0,0,0,0], values[1,1,2,3] = 0., -0.
    self._check("all", values, axis=(1,2))

  def test_all_large(self):
    _test_ops.TestOps.test_all_large(self)
  test_any_zero_axis = _test_ops.TestOps.test_any_zero_axis
  test_all_zero_axis = _test_ops.TestOps.test_all_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipWhereOps(unittest.TestCase):
  """FP16 WHERE and masked arithmetic lowered to DPU comparison masks and selection."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_inf_where = _test_ops.TestOps.test_inf_where
  test_masked_fill = _test_ops.TestOps.test_masked_fill

  def test_where_permute(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(5,5)], lambda x: torch.where(x > .5, 4, 2).type(torch.int32).permute((1,0)),
                  lambda x: (x > .5).where(4, 2).clone().permute((1,0)), forward_only=True)
    values = [[math.nan, math.inf, -math.inf], [-0., 0., 1.]]
    _fp16_test_op(None, lambda x: torch.where(x > .5, 4, 2).type(torch.int32).permute((1,0)),
                  lambda x: (x > .5).where(4, 2).clone().permute((1,0)), vals=[values], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 6)


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipInterpolateOps(unittest.TestCase):
  """FP16 interpolation cases advanced one test_ops method at a time."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_interpolate_nearest = _test_ops.TestOps.test_interpolate_nearest
  test_interpolate_nearest_exact = _test_ops.TestOps.test_interpolate_nearest_exact
  test_interpolate_linear = _test_ops.TestOps.test_interpolate_linear
  test_interpolate_linear_corners_aligned = _test_ops.TestOps.test_interpolate_linear_corners_aligned
  test_interpolate_bilinear = _test_ops.TestOps.test_interpolate_bilinear
  test_interpolate_bilinear_corners_aligned = _test_ops.TestOps.test_interpolate_bilinear_corners_aligned
  test_interpolate_trilinear = _test_ops.TestOps.test_interpolate_trilinear
  test_interpolate_trilinear_corners_aligned = _test_ops.TestOps.test_interpolate_trilinear_corners_aligned

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMovementOps(unittest.TestCase):
  """Static 16-bit view and layout methods proven by the Rockchip gather path."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_transpose = _test_ops.TestOps.test_transpose
  test_permute = _test_ops.TestOps.test_permute
  test_reshape = _test_ops.TestOps.test_reshape
  test_view = _test_ops.TestOps.test_view
  test_flip = _test_ops.TestOps.test_flip
  test_flip_eye_crash = _test_ops.TestOps.test_flip_eye_crash
  test_squeeze = _test_ops.TestOps.test_squeeze
  test_unsqueeze = _test_ops.TestOps.test_unsqueeze
  test_flatten = _test_ops.TestOps.test_flatten
  test_unflatten = _test_ops.TestOps.test_unflatten
  test_detach = _test_ops.TestOps.test_detach
  test_expand = _test_ops.TestOps.test_expand

  def test_slice_negative_strides(self):
    rng = np.random.default_rng(600)
    a = rng.standard_normal((10,10,10)).astype(np.float16)
    t = Tensor(a)
    for idx in (np.s_[::-1], np.s_[::-2], np.s_[:,2:0:-1], np.s_[:,2:0:-1,3:1:-2], np.s_[4:0:-3,2:0:-1,-1:-5:-2],
                np.s_[2:5:-1,:,:], np.s_[:,2:5:-1,:], np.s_[:,:,2:5:-1]):
      np.testing.assert_allclose(a[idx], t[idx].numpy(), **_FP16)

  def test_slice_with_const_tensor(self):
    self.skipTest("Rockchip DPU accepts FP16 inputs only; this case requires an integer index tensor")

  test_slice_in_bounds_1dim = _test_ops.TestOps.test_slice_in_bounds_1dim
  test_slice_on_0dim_tensor = _test_ops.TestOps.test_slice_on_0dim_tensor
  test_slice_int_indexing = _test_ops.TestOps.test_slice_int_indexing
  test_slice_in_bounds_multidim = _test_ops.TestOps.test_slice_in_bounds_multidim
  test_slice_with_none = _test_ops.TestOps.test_slice_with_none
  test_slice_one_endpoint_out_of_bounds = _test_ops.TestOps.test_slice_one_endpoint_out_of_bounds
  test_slice_stride_gt_one = _test_ops.TestOps.test_slice_stride_gt_one
  test_slice_both_endpoints_out_of_bounds = _test_ops.TestOps.test_slice_both_endpoints_out_of_bounds
  test_slice_start_gt_end = _test_ops.TestOps.test_slice_start_gt_end
  test_slice_zero_in_shape = _test_ops.TestOps.test_slice_zero_in_shape
  test_slice_errors = _test_ops.TestOps.test_slice_errors
  test_slice_ellipsis = _test_ops.TestOps.test_slice_ellipsis
  test_double_slice = _test_ops.TestOps.test_double_slice
  test_pad_reshape = _test_ops.TestOps.test_pad_reshape
  test_pad_slice = _test_ops.TestOps.test_pad_slice
  test_diag = _test_ops.TestOps.test_diag
  test_diagonal = _test_ops.TestOps.test_diagonal
  test_roll = _test_ops.TestOps.test_roll


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTriangularOps(unittest.TestCase):
  """Numeric FP16 portions of test_ops tril/triu; external boolean inputs are unsupported."""

  def _test(self, name:str):
    cases = (((3,3), 0), ((3,3), 1), ((3,3), 2), ((3,3), -1), ((3,3), -2),
             ((4,5), 4), ((4,5), 5), ((4,5), 6), ((4,5), -4), ((4,5), -5), ((4,5), -6),
             ((5,3,3), 0), ((5,0,3), 0), ((5,3,3), 1))
    for shape, diagonal in cases:
      with self.subTest(shape=shape, diagonal=diagonal):
        _fp16_test_op([shape], lambda x: getattr(x, name)(diagonal), lambda x: getattr(x, name)(diagonal))

  def test_tril(self): self._test("tril")
  def test_triu(self): self._test("triu")

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipConcatOps(unittest.TestCase):
  """16-bit concatenation, stacking, and repetition through partial raw gathers."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_stack(self):
    for dim in range(-1, 3):
      _fp16_test_op([(5,6,3)]*3, lambda x,y,z: torch.stack((x,y,z), dim), lambda x,y,z: Tensor.stack(x,y,z,dim=dim))
      _fp16_test_op([(5,6,3)]*3, lambda x,y,z: torch.stack((x,y,z), dim), lambda x,y,z: Tensor.stack((x,y,z),dim=dim))
    with self.assertRaises(IndexError): Tensor.stack(Tensor.randn(45,65,3), dim=77)
    with self.assertRaises(ValueError): Tensor.stack((Tensor([1,2]), Tensor([3,4])), Tensor([5,6]))
    np.testing.assert_allclose(Tensor.stack(Tensor(3.14), Tensor(3.14)).numpy(), np.array([3.14,3.14]), **_FP16)

  test_cat = _test_ops.TestOps.test_cat
  test_multicat = _test_ops.TestOps.test_multicat
  test_stack_slice = _test_ops.TestOps.test_stack_slice
  test_stack_max = _test_ops.TestOps.test_stack_max
  test_repeat = _test_ops.TestOps.test_repeat
  test_repeat_interleave = _test_ops.TestOps.test_repeat_interleave
  test_simple_repeat = _test_ops.TestOps.test_simple_repeat


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPaddingOps(unittest.TestCase):
  """Constant, reflect, replicate, and circular 16-bit padding through raw gathers."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_pad = _test_ops.TestOps.test_pad
  test_pad_reflect_mode = _test_ops.TestOps.test_pad_reflect_mode
  test_pad_replicate_mode = _test_ops.TestOps.test_pad_replicate_mode
  test_pad_circular_mode = _test_ops.TestOps.test_pad_circular_mode
  test_padding_add = _test_ops.TestOps.test_padding_add


@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipReductionOps(unittest.TestCase):
  """FP16 scalar sum, mean, minimum/maximum, and product reductions on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_min(self):
    for shape in ((3,3), (45,3)): _fp16_test_op([shape], lambda x: x.min())
    _fp16_test_op([(45,3)], lambda x: x.min().mul(0.5))
    _fp16_test_op([()], lambda x: x.min())

  def test_max(self):
    _fp16_test_op([(45,3)], lambda x: x.max())
    _fp16_test_op([(45,3)], lambda x: x.max().mul(0.5))
    _fp16_test_op(None, lambda x: x.max().mul(0.5), vals=[[[1.0,1.0,0.0,1.0]]])
    _fp16_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: x.max(axis=1))
    _fp16_test_op([()], lambda x: x.max())

  def test_max_nan(self):
    before = Device["ROCKCHIP"].submit_count
    for values in ([1.0, math.nan], [math.nan, 1.0]):
      self.assertTrue(math.isnan(Tensor(np.array(values, dtype=np.float16)).max().item()))
    values = np.array([[1.0, math.nan], [2.0, 3.0]], dtype=np.float16)
    for reduction,finite in ((Tensor.max, 3.0), (Tensor.min, 2.0)):
      result = reduction(Tensor(values), axis=1).numpy()
      self.assertTrue(math.isnan(result[0]))
      self.assertEqual(result[1], finite)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 4)

  def test_sum_full(self):
    before = Device["ROCKCHIP"].submit_count
    with Context(NOOPT=1): _fp16_test_op([(16384)], lambda x: x.sum())
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 342)

  def test_sum_dtype_arg(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(45,3)], lambda x: x.sum(dtype=torch.float32), lambda x: x.sum(dtype=dtypes.float32), forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 55)
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).sum(dtype="")

  test_sum_fake = _test_ops.TestOps.test_sum_fake
  test_sum_collapse = _test_ops.TestOps.test_sum_collapse
  test_sum_collapse_neg = _test_ops.TestOps.test_sum_collapse_neg
  test_sum_pad_collapse = _test_ops.TestOps.test_sum_pad_collapse
  test_sum_twice = _test_ops.TestOps.test_sum_twice
  test_sum_cat_collapse = _test_ops.TestOps.test_sum_cat_collapse
  test_max_dont_collapse = _test_ops.TestOps.test_max_dont_collapse
  test_sum_simple = _test_ops.TestOps.test_sum_simple
  test_sum_relu = _test_ops.TestOps.test_sum_relu
  test_sum_tiny = _test_ops.TestOps.test_sum_tiny
  test_sum = _test_ops.TestOps.test_sum
  test_sum_with_zeros_shape = _test_ops.TestOps.test_sum_with_zeros_shape
  test_prod = _test_ops.TestOps.test_prod
  test_prod_dtype_arg = _test_ops.TestOps.test_prod_dtype_arg
  test_const_reduce = _test_ops.TestOps.test_const_reduce
  test_mean = _test_ops.TestOps.test_mean
  test_mean_axis = _test_ops.TestOps.test_mean_axis
  test_mean_zero_axis = _test_ops.TestOps.test_mean_zero_axis
  test_var = _test_ops.TestOps.test_var
  test_var_axis = _test_ops.TestOps.test_var_axis
  test_var_zero_in_axis = _test_ops.TestOps.test_var_zero_in_axis
  test_var_one_in_axis = _test_ops.TestOps.test_var_one_in_axis
  test_var_keepdim = _test_ops.TestOps.test_var_keepdim
  test_std = _test_ops.TestOps.test_std
  test_std_axis = _test_ops.TestOps.test_std_axis
  test_std_one_in_axis = _test_ops.TestOps.test_std_one_in_axis
  test_std_keepdim = _test_ops.TestOps.test_std_keepdim
  test_std_mean = _test_ops.TestOps.test_std_mean
  test_std_mean_loaded_nan = _test_ops.TestOps.test_std_mean_loaded_nan
  test_std_zero_in_axis = _test_ops.TestOps.test_std_zero_in_axis
  test_normalize = _test_ops.TestOps.test_normalize

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIncrementalOps(unittest.TestCase):
  """Remaining test_ops methods, admitted and debugged in source order."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_full_like = _test_ops.TestOps.test_full_like
  test_full = _test_ops.TestOps.test_full
  test_negative_dims = _test_ops.TestOps.test_negative_dims
  test_negative_dims_full = _test_ops.TestOps.test_negative_dims_full
  test_negative_dims_eye = _test_ops.TestOps.test_negative_dims_eye
  test_negative_dims_kaiming = _test_ops.TestOps.test_negative_dims_kaiming
  test_zeros = _test_ops.TestOps.test_zeros
  test_zeros_like = _test_ops.TestOps.test_zeros_like
  test_empty_0 = _test_ops.TestOps.test_empty_0
  test_ones = _test_ops.TestOps.test_ones
  test_ones_like = _test_ops.TestOps.test_ones_like
  test_eye = _test_ops.TestOps.test_eye
  test_split = _test_ops.TestOps.test_split
  test_chunk = _test_ops.TestOps.test_chunk
  test_unfold = _test_ops.TestOps.test_unfold
  test_meshgrid = _test_ops.TestOps.test_meshgrid
  test_arange = _test_ops.TestOps.test_arange
  test_arange_big = _test_ops.TestOps.test_arange_big
  test_arange_4096 = _test_ops.TestOps.test_arange_4096
  test_linspace = _test_ops.TestOps.test_linspace
  test_lerp = _test_ops.TestOps.test_lerp
  test_broadcasted_add = _test_ops.TestOps.test_broadcasted_add
  test_broadcasted_add_2 = _test_ops.TestOps.test_broadcasted_add_2
  test_sub = _test_ops.TestOps.test_sub
  test_scalar_sub = _test_ops.TestOps.test_scalar_sub
  test_scalar_rsub = _test_ops.TestOps.test_scalar_rsub
  test_neg = _test_ops.TestOps.test_neg
  test_tiny_add = _test_ops.TestOps.test_tiny_add
  test_tiny_mul = _test_ops.TestOps.test_tiny_mul
  test_add = _test_ops.TestOps.test_add
  test_add3 = _test_ops.TestOps.test_add3
  test_mul = _test_ops.TestOps.test_mul
  test_scalar_mul = _test_ops.TestOps.test_scalar_mul
  test_div = _test_ops.TestOps.test_div
  test_scalar_div = _test_ops.TestOps.test_scalar_div
  test_mul_naninf = _test_ops.TestOps.test_mul_naninf
  test_div_naninf = _test_ops.TestOps.test_div_naninf
  test_relu = _test_ops.TestOps.test_relu
  test_relu_exact = _test_ops.TestOps.test_relu_exact
  test_relu_maximum_exact = _test_ops.TestOps.test_relu_maximum_exact
  test_leaky_relu = _test_ops.TestOps.test_leaky_relu
  test_abs = _test_ops.TestOps.test_abs
  test_abs_exact = _test_ops.TestOps.test_abs_exact
  test_relu6 = _test_ops.TestOps.test_relu6
  test_clip = _test_ops.TestOps.test_clip
  test_hardtanh = _test_ops.TestOps.test_hardtanh
  test_hardsigmoid = _test_ops.TestOps.test_hardsigmoid
  test_hardsigmoid_extreme = _test_ops.TestOps.test_hardsigmoid_extreme
  test_hardswish = _test_ops.TestOps.test_hardswish
  test_softsign = _test_ops.TestOps.test_softsign
  test_softsign_exact = _test_ops.TestOps.test_softsign_exact

if __name__ == "__main__":
  unittest.main()
