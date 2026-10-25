"""FP16 verification rows against the original scalar projection, including rounding."""
import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.kernels.mali import MaliLinear


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware test')
class TestMaliHalfRows(unittest.TestCase):
  def make_projection(self, columns: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    weight_fp32 = rng.normal(0, 1 / np.sqrt(depth), (columns, depth)).astype(np.float32)
    input_fp32 = rng.normal(0, .3, (1, 8, depth)).astype(np.float32)
    weight_fp16, input_fp16 = weight_fp32.astype(np.float16), input_fp32.astype(np.float16)
    # Make sure these fixtures exercise an actual conversion from FP32 values.
    self.assertTrue(np.any(weight_fp32 != weight_fp16.astype(np.float32)))
    self.assertTrue(np.any(input_fp32 != input_fp16.astype(np.float32)))
    # Include nontrivial representable neighbors and small magnitudes after rounding.
    if depth >= 8:
      input_fp16[0, 0, :8] = np.array([0.1, -0.3, 1.0003, -2.0008, 0.0007, -0.0011, 3.14159, -7.25],
                                      dtype=np.float32).astype(np.float16)
      weight_fp16[:min(columns, 4), :8] = np.array(
        [[0.2, -0.6, 0.3333, -1.0004, 0.0023, -0.0047, 2.71828, -0.125]] * min(columns, 4),
        dtype=np.float32).astype(np.float16)
    source = nn.Linear(depth, columns, bias=False)
    source.weight = Tensor(weight_fp16, device='CL', dtype=dtypes.half).realize()
    layer = MaliLinear(source, prepare_gemm=False, consistent_decode=True)
    x = Tensor(input_fp16, device='CL', dtype=dtypes.half).realize()
    return source, layer, x

  def assert_source_rows_equal(self, actual: np.ndarray, source, x: Tensor, rows: int) -> None:
    expected = np.concatenate([source(x[:, row:row+1]).numpy() for row in range(rows)], axis=1)
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, np.float16)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

  def test_static_rows_match_original_single_row_linear_for_rounding_fixtures(self):
    for columns, depth, seed in ((19, 68, 7601), (37, 2048, 7602), (128, 2048, 7603), (2048, 2048, 7605), (6144, 2048, 7606)):
      source, layer, x8 = self.make_projection(columns, depth, seed)
      for rows in (1, 3, 8):
        with self.subTest(columns=columns, depth=depth, rows=rows):
          actual = layer(x8[:, :rows]).numpy()
          self.assert_source_rows_equal(actual, source, x8, rows)

  def test_tinyjit_rebinds_both_capacities_with_nan_padding(self):
    for columns, depth, seed in ((37, 2048, 7604), (2048, 2048, 7607), (6144, 2048, 7608)):
      source, layer, fixture = self.make_projection(columns, depth, seed)
      for capacity, counts in ((8, (8, 3, 1, 8)), (3, (3, 1, 3))):
        run = TinyJit(lambda x, rows: layer(x[:, :rows]).realize())
        for rows in counts:
          with self.subTest(columns=columns, capacity=capacity, rows=rows):
            values = fixture.numpy()[:, :capacity].copy()
            values[:, rows:] = np.nan
            x = Tensor(values, device='CL', dtype=dtypes.half).realize()
            actual = run(x, UOp.variable('half_decode_rows', 1, capacity).bind(rows))
            self.assertEqual(actual.shape[1].unbind()[1], rows)
            actual_np = actual.pad_to((1, capacity, columns)).numpy()[:, :rows]
            self.assert_source_rows_equal(actual_np, source, x, rows)
