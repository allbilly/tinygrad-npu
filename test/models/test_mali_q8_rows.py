"""Contracts and explicit Mali qualification for row-batched Q8 GEMV."""
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.mali_linear import gemv_rows
from tinygrad.llm.kernels.mali_q8 import gemv_q8_rows, _gemv_q8_rows
from tinygrad.llm.kernels import mali_q8
from test.models.test_mali_q8 import q8_fixture


class TestMaliQ8RowsContract(unittest.TestCase):
  """Build only lazy graphs; contract checks must not initialize OpenCL."""
  def test_tuned_geometry_is_limited_to_the_qualified_head_and_device(self):
    for rows, columns, depth, supported, expected in ((8, 248320, 2048, True, True),
        (8, 248320, 2048, False, False), (3, 248320, 2048, True, False),
        (8, 1024, 2048, True, False), (8, 248320, 1024, True, False)):
      with self.subTest(rows=rows, columns=columns, depth=depth, supported=supported):
        x = Tensor.empty(1, rows, depth, device='CL', dtype=dtypes.float)
        raw = Tensor.empty(columns * depth // 32 * 34, device='CL', dtype=dtypes.uint8)
        def fake_kernel(output, *inputs, fxn, **kwargs):
          fxn(output.uop, *(value.uop for value in inputs))
          return [output]
        with patch.object(mali_q8, '_small_gemm_supported', return_value=supported) as capability, \
             patch.object(mali_q8, '_gemv_q8_rows', side_effect=lambda *args: args[0]) as kernel, \
             patch.object(Tensor, 'custom_kernel', fake_kernel):
          result = gemv_q8_rows(x, raw, columns, depth)
        self.assertEqual(result.shape, (1, rows, columns))
        self.assertEqual(kernel.call_args.args[6:], (2, 8) if expected else ())
        self.assertEqual(capability.call_count, int(rows == 8 and (columns, depth) == (248320, 2048)))

  def test_symbolic_capacity_and_batched_shapes_keep_valid_result_shape(self):
    rows = UOp.variable('q8_rows_capacity', 1, 4).bind(3)
    raw = Tensor.empty(17 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8)
    x = Tensor.empty(2, 4, 96, device='CL', dtype=dtypes.float)[:, :rows]
    result = gemv_q8_rows(x, raw, 17, 96)
    self.assertEqual(result.shape, (2, rows, 17))
    self.assertEqual(result.max_shape, (2, 4, 17))
    self.assertIs(result.dtype, dtypes.float)
    for shape in ((96,), (1, 3, 96), (2, 3, 96)):
      with self.subTest(shape=shape):
        result = gemv_q8_rows(Tensor.empty(*shape, device='CL', dtype=dtypes.float), raw, 17, 96)
        self.assertEqual(result.shape, (*shape[:-1], 17))
        self.assertIs(result.dtype, dtypes.float)

  def test_rejects_unsupported_dtype_storage_geometry_and_row_capacity(self):
    raw = Tensor.empty(31 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8)
    with self.assertRaises(ValueError):
      gemv_q8_rows(Tensor.empty(1, 3, 96, device='CL', dtype=dtypes.half), raw, 31, 96)
    for storage in (Tensor.empty(raw.numel(), device='CL', dtype=dtypes.int8),
                    Tensor.empty(raw.numel(), device='CPU', dtype=dtypes.uint8),
                    Tensor.empty(raw.numel() - 1, device='CL', dtype=dtypes.uint8)):
      with self.subTest(storage=storage), self.assertRaises(ValueError):
        gemv_q8_rows(Tensor.empty(1, 3, 96, device='CL', dtype=dtypes.float), storage, 31, 96)
    for columns, depth in ((0, 96), (31, 0), (31, 31), (True, 96), (31, True)):
      with self.subTest(columns=columns, depth=depth), self.assertRaises(ValueError):
        gemv_q8_rows(Tensor.empty(1, 3, 96, device='CL', dtype=dtypes.float), raw, columns, depth)
    for capacity in (0, 9):
      with self.subTest(capacity=capacity), self.assertRaises(ValueError):
        bad_raw = Tensor.empty(17 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8)
        gemv_q8_rows(Tensor.empty(1, capacity, 96, device='CL', dtype=dtypes.float), bad_raw, 17, 96)


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware qualification')
class TestMaliQ8RowsHardware(unittest.TestCase):
  def test_experimental_row_tiles_preserve_dense_bits(self):
    raw_values, weights = q8_fixture(31, 256, seed=419280)
    rng = np.random.default_rng(419280)
    values = rng.normal(0, .3, (1, 8, 256)).astype(np.float32)
    raw = Tensor(raw_values, device='CL').realize()
    weight = Tensor(weights, device='CL').realize()
    x = Tensor(values, device='CL').realize()
    expected = gemv_rows(x, weight).numpy()
    for group, tile in ((16, 4), (8, 2), (8, 4), (8, 8), (16, 8), (4, 4), (4, 8), (2, 8)):
      with self.subTest(group=group, tile=tile):
        output = Tensor.empty(1, 8, 31, device='CL', dtype=dtypes.float)
        actual = output.custom_kernel(x, raw, fxn=lambda o,a,w:_gemv_q8_rows(o,a,w,8,31,256,group,tile))[0].numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(raw.numpy(), raw_values)
    np.testing.assert_array_equal(weight.numpy().view(np.uint16), weights.view(np.uint16))

  def test_row_batches_match_dense_fp16_gemv_bits_and_preserve_inputs(self):
    rng = np.random.default_rng(419278)
    cases = ((17, 32, (1, 32)), (31, 32, (1, 2, 32)), (31, 96, (1, 3, 96)),
             (65, 256, (1, 8, 256)), (17, 96, (1, 7, 96)), (31, 96, (2, 3, 96)))
    for columns, depth, shape in cases:
      with self.subTest(columns=columns, depth=depth, shape=shape):
        raw_values, weights = q8_fixture(columns, depth, seed=depth + columns)
        raw_values_before, weights_before = raw_values.copy(), weights.copy()
        raw = Tensor(raw_values, device='CL', dtype=dtypes.uint8).realize()
        weight = Tensor(weights, device='CL', dtype=dtypes.half).realize()
        values = rng.normal(0, .3, shape).astype(np.float32)
        x = Tensor(values, device='CL').realize()
        expected = gemv_rows(x, weight).numpy()
        actual = gemv_q8_rows(x, raw, columns, depth).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(raw.numpy(), raw_values_before)
        np.testing.assert_array_equal(weight.numpy().view(np.uint16), weights_before.view(np.uint16))
        np.testing.assert_array_equal(x.numpy().view(np.uint32), values.view(np.uint32))

  def test_dynamic_rows_ignore_nan_padding_and_keep_output_tail_zero(self):
    rng = np.random.default_rng(419279)
    columns, depth = 31, 96
    raw_values, weights = q8_fixture(columns, depth, seed=419279)
    raw_before, weights_before = raw_values.copy(), weights.copy()
    raw = Tensor(raw_values, device='CL', dtype=dtypes.uint8).realize()
    weight = Tensor(weights, device='CL', dtype=dtypes.half).realize()
    rows = UOp.variable('q8_dynamic_rows', 1, 8)
    run = TinyJit(lambda x, count: gemv_q8_rows(x[:, :count], raw, columns, depth).realize())
    for count in (1, 3, 8, 3):
      values = rng.normal(0, .3, (1, 8, depth)).astype(np.float32)
      values[:, count:] = np.nan
      actual = run(Tensor(values, device='CL').realize(), rows.bind(count)).pad_to((1, 8, columns)).numpy()
      expected = gemv_rows(Tensor(values[:, :count].copy(), device='CL').realize(), weight).numpy()
      np.testing.assert_array_equal(actual[:, :count].view(np.uint32), expected.view(np.uint32))
      np.testing.assert_array_equal(actual[:, count:].view(np.uint32), np.zeros_like(actual[:, count:]).view(np.uint32))
    np.testing.assert_array_equal(raw.numpy(), raw_before)
    np.testing.assert_array_equal(weight.numpy().view(np.uint16), weights_before.view(np.uint16))


if __name__ == '__main__':
  unittest.main()
