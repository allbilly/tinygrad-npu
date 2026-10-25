"""Routing and exact-bit contracts for the qualified tiled Mali GEMV rows path."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels import mali_linear


class TestMaliTiledGemvPolicy(unittest.TestCase):
  @staticmethod
  def route(rows, columns, depth, weight_dtype, supported):
    x = Tensor.empty(1, rows, depth, device='CL', dtype=dtypes.float)
    weight = Tensor.empty(columns, depth, device='CL', dtype=weight_dtype)
    selected = []

    def fake_custom_kernel(output, *inputs, fxn, **kwargs):
      fxn(output.uop, *(value.uop for value in inputs))
      return [output]

    def use_tiled(*args):
      selected.append('tiled')
      return args[0]

    def use_legacy(*args):
      selected.append('legacy')
      return args[0]

    with patch.object(mali_linear, '_small_gemm_supported', return_value=supported) as capability, \
         patch.object(mali_linear, '_gemv_rows_tiled', side_effect=use_tiled) as tiled, \
         patch.object(mali_linear, '_gemv_rows', side_effect=use_legacy) as legacy, \
         patch.object(Tensor, 'custom_kernel', fake_custom_kernel):
      result = mali_linear.gemv_rows(x, weight)
    return result, selected, capability, tiled, legacy

  def assert_route(self, selected, capability, tiled, legacy, expected, expect_capability):
    self.assertEqual(selected, [expected])
    self.assertEqual(capability.call_count, int(expect_capability))
    self.assertEqual(tiled.call_count, int(expected == 'tiled'))
    self.assertEqual(legacy.call_count, int(expected == 'legacy'))

  def test_only_qualified_fp16_full_head_and_down_shapes_route_to_tile2(self):
    cases = (
      (8, 248320, 2048, dtypes.half, True, 'tiled', True),
      (3, 248320, 2048, dtypes.half, True, 'legacy', False),
      (8, 2048, 6144, dtypes.half, True, 'tiled', True),
      (3, 2048, 6144, dtypes.half, True, 'tiled', True),
      (1, 2048, 6144, dtypes.half, True, 'legacy', False),
      (8, 2048, 6144, dtypes.float, True, 'legacy', False),
      (8, 2048, 6144, dtypes.half, False, 'legacy', True),
      (8, 2048, 2048, dtypes.half, True, 'legacy', False),
      (3, 31, 256, dtypes.half, True, 'legacy', False),
    )
    for rows, columns, depth, dtype, supported, expected, expect_capability in cases:
      with self.subTest(rows=rows, columns=columns, depth=depth, dtype=dtype, supported=supported):
        result, selected, capability, tiled, legacy = self.route(rows, columns, depth, dtype, supported)
        self.assertEqual(result.shape, (1, rows, columns))
        self.assert_route(selected, capability, tiled, legacy, expected, expect_capability)
        if expected == 'tiled':
          self.assertEqual(tiled.call_args.args[3:6], (rows, columns, depth))
          self.assertEqual(len(tiled.call_args.args), 6)
          self.assertEqual(mali_linear._gemv_rows_tiled.__wrapped__.__defaults__, (16, 2))


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware test')
class TestMaliTiledGemv(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not mali_linear._small_gemm_supported('CL'):
      raise unittest.SkipTest('tiled GEMV was measured only on the qualified Mali device')

  @staticmethod
  def make_values(columns, depth, rows, dtype, seed):
    rng = np.random.default_rng(seed + columns + depth + rows)
    weight_values = (rng.standard_normal((columns, depth), dtype=np.float32) / np.sqrt(depth)).astype(dtype)
    x_values = rng.normal(0, .2, (1, rows, depth)).astype(np.float32)
    if depth >= 8:
      smallest = np.nextafter(np.float32(0), np.float32(1))
      x_values[0, 0, :8] = (smallest, -smallest, .001, -.001, 0., -0., 100., -100.)
      weight_values[:, :8] = np.asarray((1., 1., .0001, .0001, .5, .5, .001, .001), dtype=dtype)
    assert np.isfinite(weight_values).all() and np.isfinite(x_values).all()
    return x_values, weight_values

  @staticmethod
  def tiled(x, weight):
    rows = int(np.prod(x.max_shape[:-1]))
    columns, depth = weight.shape
    output = Tensor.empty(*x.max_shape[:-1], columns, device=x.device, dtype=dtypes.float)
    result = output.custom_kernel(x.pad_to(x.max_shape).contiguous(), weight.contiguous(),
      fxn=lambda out, values, weights: mali_linear._gemv_rows_tiled(out, values, weights,
        rows, columns, depth, 16, 2))[0]
    return result.shrink(tuple((0, size) for size in (*x.shape[:-1], columns)))

  @staticmethod
  def legacy(x, weight):
    with patch.object(mali_linear, '_small_gemm_supported', return_value=False):
      return mali_linear.gemv_rows(x, weight)

  def assert_bits_equal(self, actual, expected):
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

  def test_small_partial_output_columns_preserve_legacy_bits(self):
    for columns, depth, seed in ((17, 256, 8311), (31, 2048, 8312), (17, 6144, 8313)):
      for dtype in (np.float16, np.float32):
        weight_values = None
        weight = None
        for rows in (3, 8):
          x_values, current_weight = self.make_values(columns, depth, rows, dtype, seed)
          if weight is None:
            weight_values = current_weight
            weight = Tensor(weight_values, device='CL', dtype=dtypes.half if dtype is np.float16 else dtypes.float).realize()
          x = Tensor(x_values, device='CL', dtype=dtypes.float).realize()
          expected = self.legacy(x, weight).numpy()
          actual = self.tiled(x, weight).numpy()
          with self.subTest(columns=columns, depth=depth, dtype=dtype.__name__, rows=rows):
            self.assert_bits_equal(actual, expected)
            self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(weight.numpy().view(np.uint8), weight_values.view(np.uint8))

  def test_dynamic_capacity_eight_rows_match_legacy_with_nan_tails(self):
    columns, depth = 31, 256
    x_values, weight_values = self.make_values(columns, depth, 8, np.float16, 8321)
    weight = Tensor(weight_values, device='CL', dtype=dtypes.half).realize()
    run = TinyJit(lambda x, rows: self.tiled(x[:, :rows], weight).realize())
    weight_before = weight.numpy().copy()
    for rows in (1, 3, 8, 3, 1):
      values = x_values.copy()
      values[:, rows:] = np.nan
      x = Tensor(values, device='CL', dtype=dtypes.float).realize()
      bound = UOp.variable('tiled_gemv_rows', 1, 8).bind(rows)
      actual = run(x, bound).pad_to((1,8,columns)).numpy()
      expected = self.legacy(Tensor(values[:, :rows].copy(), device='CL', dtype=dtypes.float).realize(), weight).numpy()
      with self.subTest(rows=rows):
        self.assert_bits_equal(actual[:, :rows], expected)
        self.assertTrue(np.all(actual[:, rows:]==0))
        self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_array_equal(weight.numpy().view(np.uint8), weight_before.view(np.uint8))
