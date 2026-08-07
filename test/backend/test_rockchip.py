"""Rockchip NPU census: ops known to pass, with DRM_IOCTL_RKNPU_SUBMIT counts.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip.py -q -n12

The backend accepts FP16 inputs and emits contiguous FP16 DPU EW output. It does not
perform host arithmetic or FP32→FP16 conversion. GEMM gather layout is prepared on
the host, then every MUL/ADD is submitted to DPU EW.

Reduction via ROCKCHIP_EW_REDUCE=sequential|kahan|twoproduct (default sequential).
TwoProduct uses a conservative 256-task mixed MUL/ADD chain cap.
"""
from __future__ import annotations
import math, os, unittest
import numpy as np
import torch
from tinygrad import Tensor, Device
from test.backend import test_ops as _test_ops
from test.backend.test_ops import helper_test_op, slow_test

# fp16 tol matches test_ops.test_gemm_fp16
_FP16 = dict(atol=5e-3, rtol=5e-3)
_FP16_WITH_GRAD = dict(atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
_EW_CHAIN_FP16, _EW_CHAIN_TWOPRODUCT = 512, 256
_EW_REDUCE = os.getenv("ROCKCHIP_EW_REDUCE", "sequential").strip().lower()
_TEST_OPS_HELPER = _test_ops.helper_test_op

def _fp16_test_op(*args, **kwargs):
  """Run a test_ops case at the same FP16 tolerance as test_gemm_fp16."""
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(*args, **kwargs)

def _fp16_fp32_golden_test_op(shps, torch_fxn, tinygrad_fxn, **kwargs):
  """Use an FP32-accumulated golden when CPU FP16 reduction is less accurate than DPU EW."""
  def fp32_golden(*tensors):
    out = torch_fxn(*(x.float() if x.is_floating_point() else x for x in tensors))
    return out.to(tensors[0].dtype) if tensors and tensors[0].is_floating_point() else out
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(shps, fp32_golden, tinygrad_fxn, **kwargs)

def _ew_submits(n:int) -> int:
  """EW ioctl count for one logical op over n half elements."""
  from tinygrad.renderer.rockchip import _MAX_EW_ELEMS_FP16
  tiles = (n + _MAX_EW_ELEMS_FP16 - 1) // _MAX_EW_ELEMS_FP16
  chain = _EW_CHAIN_TWOPRODUCT if _EW_REDUCE == "twoproduct" else _EW_CHAIN_FP16
  return (tiles + chain - 1) // chain

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["ROCKCHIP"]

  def _half(self, shape, seed:int=0) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-2, 2, size=shape).astype(np.float16))

  def _check(self, expected_submits:int, out:Tensor, ref:np.ndarray, atol=5e-3, rtol=5e-3):
    """Realize `out`, compare to `ref`, assert ioctl submit delta.

    Default tol matches test_ops half gemm (DEFAULT_FLOAT=HALF / test_gemm_fp16).
    """
    before = self.dev.submit_count
    got = out.realize().numpy()
    submits = self.dev.submit_count - before
    print(f"  {self._testMethodName}: submits={submits} (expected {expected_submits})")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    self.assertEqual(submits, expected_submits, f"{self._testMethodName}: submits={submits} expected={expected_submits}")

  def _check_conv2d(self, expected_submits:int, x_shape:tuple[int, ...], w_shape:tuple[int, ...], seed:int, **kwargs):
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

  def test_add_scalar_constfold(self):
    # Tensor(1)+0.5 folds on device=None — no NPU submit
    self._check(0, Tensor(1) + 0.5, np.array(1.5, dtype=np.float16))

  def test_add_empty(self):
    # rank-0 buffers materialize without NPU submit
    a, b = self._half((), 5), self._half((), 6)
    self._check(0, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

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

  def test_scalar_mul_empty(self):
    # rank-0 scalar mul — no NPU submit
    a = self._half((), 15)
    self._check(0, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(0, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_mul_naninf(self):
    self.skipTest("RK3588 FP16 DPU EW MUL returns NaN for infinity operands")
    a = self._half((45, 65), 16)
    n = _ew_submits(45*65)
    self._check(n, a * math.inf, (a.numpy().astype(np.float32) * np.float32(np.inf)).astype(np.float16))
    self._check(n, a * -math.inf, (a.numpy().astype(np.float32) * np.float32(-np.inf)).astype(np.float16))
    self._check(n, a * math.nan, (a.numpy().astype(np.float32) * np.float32(np.nan)).astype(np.float16))

  # ---- MAX ----
  def test_maximum_fp16(self):
    a, b = self._half((45, 65), 17), self._half((45, 65), 18)
    self._check(_ew_submits(45*65), a.maximum(b), np.maximum(a.numpy(), b.numpy()))

  def test_max_pool2d_simple_submit(self):
    xn = np.array([[[[-1, 2, 0], [3, -4, 1]]]], dtype=np.float16)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(xn), kernel_size=(2, 2)).numpy()
    self._check(1, Tensor(xn).max_pool2d(kernel_size=(2, 2)), ref)

  def test_avg_pool2d_valid_count_submit(self):
    xn = np.arange(16, dtype=np.float16).reshape(1, 1, 4, 4)
    args = dict(kernel_size=(3, 3), padding=1, count_include_pad=False)
    ref = torch.nn.functional.avg_pool2d(torch.from_numpy(xn), **args).numpy()
    self._check(1, Tensor(xn).avg_pool2d(**args), ref)

  def test_interpolate_nearest_submit(self):
    x = self._half((2,3,13), 502)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(9,), mode="nearest").numpy()
    self._check(1, x.interpolate((9,), mode="nearest"), ref, **_FP16)

  def test_interpolate_trilinear_submit(self):
    x = self._half((1,1,3,2,4), 503)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(2,4,3), mode="trilinear").numpy()
    self._check(3, x.interpolate((2,4,3), mode="linear"), ref, **_FP16)

  # ---- GEMM / MATMUL (from test_ops, fp16 tol) ----
  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  @slow_test
  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_matmul_batched(self):
    helper_test_op([(3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_matmul_batched_vector(self):
    helper_test_op([(4,3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_small_gemm(self):
    helper_test_op([(8,8), (8,8)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_9_gemm(self):
    helper_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_small_gemm_padded(self):
    helper_test_op([(9,9), (9,9)],
                   lambda x,y: torch.nn.functional.pad(x, (0,7,0,7)).matmul(torch.nn.functional.pad(y, (0,7,0,7))),
                   lambda x,y: x.pad(((0,7),(0,7)))@y.pad(((0,7),(0,7))), **_FP16)
  def test_small_gemm_range(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.arange(0,64,dtype=np.float16).reshape(8,8),
                         np.arange(64,128,dtype=np.float16).reshape(8,8)], **_FP16)
  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.eye(8).astype(np.float16), np.eye(8).astype(np.float16)], **_FP16)
  @slow_test
  def test_gemm_fp16(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()), **_FP16)
  @slow_test
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y), **_FP16)
  @slow_test
  def test_big_gemm(self):
    helper_test_op([(256,256), (256,256)], lambda x,y: x.matmul(y), **_FP16)
  def test_gemm_with_zeros_shape(self):
    helper_test_op([(8,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,8), (8,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(8,0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,0), (0,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0), (0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)

  # ---- CONV2D (from test_ops, fp16 tol) ----
  def test_simple_conv2d_1x1(self):
    self._check_conv2d(1, (1,4,9,9), (4,4,1,1), 100)

  def test_simple_conv2d(self):
    self._check_conv2d(9, (1,4,9,9), (4,4,3,3), 101)

  def test_simple_conv2d_batched(self):
    self._check_conv2d(9, (2,4,9,9), (4,4,3,3), 102)

  def test_padded_conv2d(self):
    self._check_conv2d(9, (1,4,9,9), (4,4,3,3), 103, padding=1)

  def test_strided_conv2d(self):
    self._check_conv2d(9, (1,4,9,9), (4,4,3,3), 104, stride=2)

  def test_depthwise_conv2d(self):
    self._check_conv2d(3, (1,4,9,9), (4,1,3,3), 105, groups=4)

  def test_simple_conv2d_reduce63(self):
    self._check_conv2d(16, (1,7,9,9), (4,7,3,3), 207)

  def test_simple_conv2d_reduce72(self):
    self._check_conv2d(18, (1,8,9,9), (4,8,3,3), 208)

  def test_simple_conv2d_m4(self):
    self._check_conv2d(35, (1,16,9,9), (16,16,3,3), 300)

  def test_simple_conv2d_1x1_m4(self):
    self._check_conv2d(4, (1,16,32,32), (16,16,1,1), 301)

  def test_grouped_conv2d(self):
    self._check_conv2d(9, (1,8,9,9), (8,4,3,3), 302, groups=2)

  def test_dilated_conv2d(self):
    self._check_conv2d(9, (1,4,9,9), (4,4,3,3), 303, dilation=2)

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
    self._check(9, Tensor(xn).conv_transpose2d(Tensor(wn), Tensor(bn), **args), ref, **_FP16)

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

  @slow_test
  def test_simple_conv_transpose3d(self):
    # The DPU expansion is also closer to FP32 accumulation than PyTorch CPU's FP16 ConvTranspose3D reduction.
    _fp16_fp32_golden_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w), lambda x,w: Tensor.conv_transpose2d(x,w))

# Keep the Rockchip convolution census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test") and "conv" in _name and _name not in ("test_bias_conv_transpose2d", "test_simple_conv_transpose3d"):
    setattr(TestRockchipConvOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxPoolOps(unittest.TestCase):
  """Every numeric FP16 MaxPool2D case from test_ops at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  @unittest.skip("Rockchip accepts FP16 inputs only")
  def test_max_pool2d_padding_int(self): pass

  @unittest.skip("Rockchip DPU EW has no integer index output")
  def test_max_pool2d_return_indices(self): pass

# Keep the numeric MaxPool2D census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test_max_pool2d") and _name not in ("test_max_pool2d_padding_int", "test_max_pool2d_return_indices"):
    setattr(TestRockchipMaxPoolOps, _name, _test)

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
class TestRockchipInterpolateOps(unittest.TestCase):
  """Every FP16 interpolation case from test_ops at the test_gemm_fp16 tolerance."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

# Keep the FP16 interpolation census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test_interpolate"):
    setattr(TestRockchipInterpolateOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipActivationOps(unittest.TestCase):
  """FP16 ADD/MUL/MAX activation and clamp census at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_maximum(self):
    _fp16_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    _fp16_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], 3.])

  def test_minimum(self):
    _fp16_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    _fp16_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], 3.])

for _name in ("test_relu", "test_relu_exact", "test_relu_maximum_exact", "test_leaky_relu", "test_relu6", "test_abs", "test_abs_exact", "test_clip"):
  setattr(TestRockchipActivationOps, _name, getattr(_test_ops.TestOps, _name))

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBroadcastOps(unittest.TestCase):
  """FP16 broadcasted ADD/MUL/SUB across scalar, row, channel, and batch dimensions."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def _test_broadcast(self, torch_fxn, tinygrad_fxn):
    for rhs in ((), (5,), (4,1), (1,3,1,1), (2,1,1,1), (1,1,4,5)):
      _fp16_test_op([(2,3,4,5), rhs], torch_fxn, tinygrad_fxn)

  def test_broadcast_add_fp16(self): self._test_broadcast(lambda x,y: x+y, Tensor.add)
  def test_broadcast_mul_fp16(self): self._test_broadcast(lambda x,y: x*y, Tensor.mul)
  def test_broadcast_sub_fp16(self): self._test_broadcast(lambda x,y: x-y, Tensor.sub)

for _name in ("test_broadcasted_add", "test_broadcasted_add_2", "test_sub", "test_scalar_sub", "test_scalar_rsub", "test_neg"):
  setattr(TestRockchipBroadcastOps, _name, getattr(_test_ops.TestOps, _name))

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipWhereOps(unittest.TestCase):
  """FP16 WHERE and masked arithmetic lowered to DPU comparison masks and EW selection."""

  test_inf_where = _test_ops.TestOps.test_inf_where

  def test_where_tensor(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x > .1, a, b), lambda x,a,b: (x > .1).where(a, b),
                  forward_only=True)

  def test_where_scalar(self):
    _fp16_test_op([(2,3,4,5)], lambda x: torch.where(x < -.2, 3.0, -2.0), lambda x: (x < -.2).where(3.0, -2.0),
                  forward_only=True)

  def test_where_broadcast(self):
    _fp16_test_op([(2,3,4,5), (5,), (1,3,1,1)], lambda x,a,b: torch.where(x > 0, a, b),
                  lambda x,a,b: (x > 0).where(a, b), forward_only=True)

  def test_where_cmpne_exact(self):
    values = [[-2., -1., 0., 1., 2.], [-2., 0., 0., 0., 2.], [4., 3., 2., 1., 0.], [-4., -3., -2., -1., 0.]]
    _fp16_test_op(None, lambda x,y,a,b: torch.where(x != y, a, b), lambda x,y,a,b: (x != y).where(a, b), vals=values,
                  forward_only=True)

  def test_where_boolean_composition(self):
    _fp16_test_op([(100,)], lambda x: torch.where((x > -.5) & (x < .5), x*2, x-1),
                  lambda x: ((x > -.5) & (x < .5)).where(x*2, x-1), forward_only=True)
    _fp16_test_op([(100,)], lambda x: torch.where((x < -.5) | (x > .5), x+2, x-2),
                  lambda x: ((x < -.5) | (x > .5)).where(x+2, x-2), forward_only=True)

  def test_where_nested(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x < -.5, a, torch.where(x > .5, b, x)),
                  lambda x,a,b: (x < -.5).where(a, (x > .5).where(b, x)), forward_only=True)

  def test_masked_fill_finite(self):
    _fp16_test_op([(32,10)], lambda x: x.masked_fill(x > .1, -3.25), lambda x: x.masked_fill(x > .1, -3.25), forward_only=True)

  def test_masked_fill_inf(self):
    _fp16_test_op([(32,10)], lambda x: x.masked_fill(x > .1, -math.inf), lambda x: x.masked_fill(x > .1, -math.inf),
                  forward_only=True)
    _fp16_test_op([(32,10)], lambda x: x.masked_fill(x < .1, -math.inf), lambda x: x.masked_fill(x < .1, -math.inf),
                  forward_only=True)

if __name__ == "__main__":
  unittest.main()
