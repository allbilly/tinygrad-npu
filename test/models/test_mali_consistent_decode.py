"""Short verification blocks preserve the scalar Mali GEMV reduction order."""
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.kernels import mali_linear
from tinygrad.llm.kernels.mali import MaliLinear


def bits_equal(test: unittest.TestCase, actual: np.ndarray, expected: np.ndarray) -> None:
  test.assertEqual(actual.shape, expected.shape)
  np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware test')
class TestMaliConsistentDecode(unittest.TestCase):
  def make_layer(self, rows: int, columns: int, depth: int, weight_dtype, *, consistent_decode: bool = True):
    rng = np.random.default_rng(7401 + rows + columns + depth)
    values = (rng.standard_normal((columns, depth), dtype=np.float32) / np.sqrt(depth)).astype(weight_dtype)
    source = nn.Linear(depth, columns, bias=False)
    source.weight = Tensor(values, device='CL').realize()
    return MaliLinear(source, prepare_gemm=False, consistent_decode=consistent_decode), source.weight

  def scalar_rows(self, values: np.ndarray, weight: Tensor) -> np.ndarray:
    outputs = []
    for row in range(values.shape[1]):
      x = Tensor(values[:, row:row+1], device='CL').realize()
      outputs.append(mali_linear.gemv(x, weight).numpy())
    return np.concatenate(outputs, axis=1)

  def test_rows3_and8_match_individual_scalar_gemv_for_storage_dtypes_and_tails(self):
    rng = np.random.default_rng(7402)
    # Both output widths leave a tail in the scalar kernel's 16-row tile;
    # depths are divisible by four, as required by its vector loads.
    for weight_dtype in (np.float16, np.float32):
      for columns, depth in ((19, 64), (37, 68)):
        with self.subTest(weight_dtype=weight_dtype.__name__, columns=columns, depth=depth):
          layer, weight = self.make_layer(8, columns, depth, weight_dtype)
          for rows in (3, 8):
            values = rng.normal(0, .3, (1, rows, depth)).astype(np.float32)
            x = Tensor(values, device='CL').realize()
            actual = layer(x).numpy()
            expected = self.scalar_rows(values, weight)
            bits_equal(self, actual, expected)

  def test_tinyjit_rebinds_capacity8_to_rows3_and1_with_nan_padding(self):
    rng = np.random.default_rng(7403)
    columns, depth = 37, 68
    layer, weight = self.make_layer(8, columns, depth, np.float16)
    run = TinyJit(lambda x, rows: layer(x[:, :rows]).realize())
    for rows in (8, 3, 1, 8):
      values = rng.normal(0, .3, (1, 8, depth)).astype(np.float32)
      values[:, rows:] = np.nan
      actual = run(Tensor(values, device='CL').realize(), UOp.variable('decode_rows', 1, 8).bind(rows))
      self.assertEqual(actual.shape[1].unbind()[1], rows)
      actual_np = actual.pad_to((1, 8, columns)).numpy()[:, :rows]
      expected = self.scalar_rows(values[:, :rows], weight)
      bits_equal(self, actual_np, expected)

  def test_gemv_rows_rejects_unsupported_capacity_and_geometry(self):
    valid_weight = Tensor.zeros(19, 64, device='CL', dtype=dtypes.half).realize()
    invalid_cases = (
      (Tensor.empty(1, 0, 64, device='CL', dtype=dtypes.float), valid_weight),
      (Tensor.empty(1, 9, 64, device='CL', dtype=dtypes.float), valid_weight),
      (Tensor.empty(1, 3, 64, device='CL', dtype=dtypes.float), Tensor.zeros(19, 60, device='CL', dtype=dtypes.half).realize()),
      (Tensor.empty(1, 3, 66, device='CL', dtype=dtypes.float), Tensor.zeros(19, 66, device='CL', dtype=dtypes.half).realize()),
    )
    for x, weight in invalid_cases:
      with self.subTest(input_shape=x.shape, weight_shape=weight.shape):
        with self.assertRaises((AssertionError, ValueError)):
          mali_linear.gemv_rows(x, weight)

  def test_mode_off_preserves_legacy_single_row_gemv_and_multirow_gemm_dispatch(self):
    layer, _ = self.make_layer(3, 19, 64, np.float16, consistent_decode=False)
    self.assertFalse(layer.consistent_decode)
    with patch.object(mali_linear, 'gemv', wraps=mali_linear.gemv) as scalar, \
         patch.object(mali_linear, 'gemm', wraps=mali_linear.gemm) as matrix, \
         patch.object(mali_linear, 'gemv_rows', wraps=mali_linear.gemv_rows) as batched:
      layer(Tensor.randn(1, 1, 64, device='CL').realize()).realize()
      self.assertEqual(scalar.call_count, 1)
      self.assertEqual(matrix.call_count, 0)
      self.assertEqual(batched.call_count, 0)
      layer(Tensor.randn(1, 3, 64, device='CL').realize()).realize()
      self.assertEqual(scalar.call_count, 1)
      self.assertEqual(matrix.call_count, 1)
      self.assertEqual(batched.call_count, 0)
