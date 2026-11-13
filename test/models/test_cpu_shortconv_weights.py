"""CPU short-convolution weight preparation preserves model math and validates atomically."""
import unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import ShortConvBlock, ShortConvolution
from tinygrad.llm.kernels.rockchip import prepare_cpu_shortconv_weights


class _RepeatProjection:
  """Identity on each of the three short-convolution projection lanes."""
  def __call__(self, x): return x.cat(x, dim=-1).cat(x, dim=-1)


class _IdentityProjection:
  def __call__(self, x): return x


def make_block(weight):
  block = ShortConvBlock.__new__(ShortConvBlock)
  block.shortconv = ShortConvolution.__new__(ShortConvolution)
  block.shortconv.config = SimpleNamespace(dim=weight.shape[0], conv_kernel=weight.shape[1], max_context=32)
  block.shortconv.conv = {"weight": weight}
  block.shortconv.in_proj, block.shortconv.out_proj = _RepeatProjection(), _IdentityProjection()
  return block


def make_model(*weights):
  return SimpleNamespace(blk=[make_block(weight) for weight in weights])


class TestPrepareCPUShortConvWeights(unittest.TestCase):
  def test_materialized_half_weights_preserve_chunked_outputs_and_cache(self):
    values = np.array([[0.1, -0.3, 0.7], [1.1, -1.3, 0.2], [0.25, -0.75, 0.5],
                       [1.25, -1.5, 0.125]], dtype=np.float32)
    lazy = Tensor(values.copy().view(np.uint8).reshape(-1), device="CPU").realize().bitcast(dtypes.float).reshape(4, 3).half()
    self.assertFalse(lazy.uop.is_realized)
    block = make_block(lazy)
    model = SimpleNamespace(blk=[block])
    # Exercise an initial multi-row chunk, a continuation, and a one-row tail.
    chunks = [(0, [[0.2, 0.4, 0.1, 0.3], [-0.5, 0.1, 0.6, -0.2], [0.7, -0.8, 0.2, 0.5]]),
              (3, [[0.3, -0.6, 0.8, 0.1], [0.9, 0.2, -0.3, 0.7]]),
              (5, [[-0.4, 0.8, 0.5, -0.1]])]
    def trace():
      block.shortconv.cache_conv = Tensor.zeros(1, 2, 4, dtype=dtypes.half)
      outputs, caches = [], []
      for start, rows in chunks:
        x = Tensor(np.asarray(rows, dtype=np.float16).reshape(1, len(rows), 4), dtype=dtypes.half)
        outputs.append(block.shortconv(x, start).numpy().copy())
        caches.append(block.shortconv.cache_conv.numpy().copy())
      return outputs, caches
    expected_outputs, expected_caches = trace()
    self.assertFalse(lazy.uop.is_realized)
    self.assertEqual(prepare_cpu_shortconv_weights(model), 1)
    optimized_weight = block.shortconv.conv["weight"]
    self.assertEqual(optimized_weight.dtype, dtypes.half)
    self.assertEqual(optimized_weight.shape, (4, 3))
    np.testing.assert_array_equal(optimized_weight.numpy(), values.astype(np.float16))
    actual_outputs, actual_caches = trace()
    for actual, expected in zip(actual_outputs, expected_outputs): np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_caches, expected_caches): np.testing.assert_array_equal(actual, expected)

  def test_shape_dtype_values_and_idempotence(self):
    original_values = np.array([[0.3333, -0.78125], [2.125, -0.125], [0.0, 1.01], [0.2, 0.9]], dtype=np.float32)
    weight = Tensor(original_values, dtype=dtypes.half)
    model = make_model(weight)
    self.assertEqual(prepare_cpu_shortconv_weights(model), 1)
    prepared = model.blk[0].shortconv.conv["weight"]
    expected = prepared.numpy().copy()
    self.assertEqual(prepared.shape, (4, 2))
    self.assertIs(prepared.dtype, dtypes.half)
    np.testing.assert_array_equal(prepared.numpy(), original_values.astype(np.float16))
    self.assertEqual(prepare_cpu_shortconv_weights(model), 1)
    again = model.blk[0].shortconv.conv["weight"]
    self.assertEqual(again.shape, prepared.shape)
    self.assertIs(again.dtype, prepared.dtype)
    self.assertEqual(again.numpy().tobytes(), expected.tobytes())

  def test_accepts_float_weights_and_preserves_values(self):
    values = np.array([[0.25, -0.5, 1.5], [2.0, -3.25, 0.125], [0.1, 0.4, 0.8], [-1.2, 0.6, 2.2]], dtype=np.float32)
    model = make_model(Tensor(values, dtype=dtypes.float))
    self.assertEqual(prepare_cpu_shortconv_weights(model), 1)
    weight = model.blk[0].shortconv.conv["weight"]
    self.assertIs(weight.dtype, dtypes.float)
    self.assertEqual(weight.shape, (4, 3))
    np.testing.assert_array_equal(weight.numpy(), values)

  def test_rejects_invalid_cpu_dtype_and_geometry(self):
    cases = [
      (SimpleNamespace(shape=(4, 3), dtype=dtypes.half, device="ROCKCHIP"), "CPU"),
      (Tensor(np.ones((4, 3), dtype=np.int32)), "dtype"),
      (Tensor(np.ones((3, 3), dtype=np.float32)), "geometry"),
      (Tensor(np.ones((4, 9), dtype=np.float32)), "geometry"),
    ]
    for weight, label in cases:
      with self.subTest(label=label, shape=weight.shape, dtype=weight.dtype):
        with self.assertRaises(ValueError):
          prepare_cpu_shortconv_weights(make_model(weight))

  def test_validates_all_blocks_before_mutating_any_weight(self):
    first_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = Tensor(first_values.view(np.uint8).reshape(-1), device="CPU").realize().bitcast(dtypes.float).reshape(4, 3).half()
    self.assertFalse(first.uop.is_realized)
    model = make_model(first, Tensor(np.ones((3, 3), dtype=np.float32)))
    with self.assertRaises(ValueError):
      prepare_cpu_shortconv_weights(model)
    self.assertIs(model.blk[0].shortconv.conv["weight"], first)
    self.assertFalse(first.uop.is_realized)

  def test_rejects_models_without_eligible_shortconv_blocks(self):
    with self.assertRaises(ValueError):
      prepare_cpu_shortconv_weights(SimpleNamespace(blk=[]))
    with self.assertRaises(ValueError):
      prepare_cpu_shortconv_weights(SimpleNamespace(blk=[SimpleNamespace(shortconv=SimpleNamespace())]))


if __name__ == "__main__": unittest.main()
