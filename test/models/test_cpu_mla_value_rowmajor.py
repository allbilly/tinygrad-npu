"""CPU coverage for ONPU_MLA_VALUE_ROWMAJOR: the MLA value projection writes the
attention-output linear's row-major layout directly, so the model does not need
the transpose view (and its contiguous materialisation). Layout-only change:
every output keeps its exact value, so the check is uint32 equality."""

import numpy as np
import pytest

from tinygrad import Tensor, Context
from tinygrad.llm.kernels.rockchip_mla_cpu import CPUMLAProjections


def _tables(heads=40, key_columns=256, key_depth=64, value_columns=64, value_depth=256, seed=7):
  rng = np.random.default_rng(seed)
  key = Tensor(rng.standard_normal((heads, key_columns, key_depth)).astype(np.float32), device='CPU').half().realize()
  value = Tensor(rng.standard_normal((heads, value_columns, value_depth)).astype(np.float32), device='CPU').half().realize()
  return key, value


def _hook():
  key, value = _tables()
  hook = CPUMLAProjections(key, value)
  hook.prepare()
  return hook


def _value(hook, x, row_major):
  with Context(ONPU_MLA_VALUE_ROWMAJOR=1 if row_major else 0):
    return hook.value(x)


def test_value_rowmajor_prefill_matches_shipped_head_major_bits():
  hook = _hook()
  rng = np.random.default_rng(11)
  x = Tensor(rng.standard_normal((1, 40, 128, 256)).astype(np.float32), device='CPU').realize()
  head_major = _value(hook, x, False)
  row_major = _value(hook, x, True)
  assert head_major.shape == (1, 40, 128, 64) and row_major.shape == (1, 128, 2560)
  expected = head_major.transpose(1, 2).reshape(1, 128, 2560).numpy()
  np.testing.assert_array_equal(row_major.numpy().view(np.uint32), expected.view(np.uint32))


def test_value_rowmajor_partial_rows_matches_shipped_bits():
  hook = _hook()
  rng = np.random.default_rng(13)
  for rows in (1, 3, 7, 64, 127):
    x = Tensor(rng.standard_normal((1, 40, rows, 256)).astype(np.float32), device='CPU').realize()
    head_major = _value(hook, x, False)
    row_major = _value(hook, x, True)
    if rows == 1:
      # capacity==1 keeps the shipped layout in both modes (decode is unchanged)
      assert row_major.shape == head_major.shape == (1, 40, 1, 64)
      np.testing.assert_array_equal(row_major.numpy().view(np.uint32), head_major.numpy().view(np.uint32))
    else:
      assert row_major.shape == (1, rows, 2560)
      expected = head_major.transpose(1, 2).reshape(1, rows, 2560).numpy()
      np.testing.assert_array_equal(row_major.numpy().view(np.uint32), expected.view(np.uint32))


def test_value_rowmajor_blocked_projection_matches_unblocked_bits():
  hook = _hook()
  rng = np.random.default_rng(17)
  x = Tensor(rng.standard_normal((1, 40, 128, 256)).astype(np.float32), device='CPU').realize()
  row_major = _value(hook, x, True)
  with Context(ONPU_MLA_VALUE_ROWMAJOR=1, ONPU_MLA_PROJ_BLOCKED=1):
    blocked = hook.value(x)
  np.testing.assert_array_equal(blocked.numpy().view(np.uint32), row_major.numpy().view(np.uint32))


def test_value_rowmajor_query_is_untouched():
  hook = _hook()
  rng = np.random.default_rng(19)
  x = Tensor(rng.standard_normal((1, 40, 128, 64)).astype(np.float32), device='CPU').realize()
  with Context(ONPU_MLA_VALUE_ROWMAJOR=0):
    shipped = hook.query(x)
  with Context(ONPU_MLA_VALUE_ROWMAJOR=1):
    flagged = hook.query(x)
  assert shipped.shape == flagged.shape == (1, 40, 128, 256)
  np.testing.assert_array_equal(flagged.numpy().view(np.uint32), shipped.numpy().view(np.uint32))


def test_value_rowmajor_flag_default_is_off():
  hook = _hook()
  x = Tensor(np.random.default_rng(23).standard_normal((1, 40, 8, 256)).astype(np.float32), device='CPU').realize()
  default = hook.value(x)
  assert default.shape == (1, 40, 8, 64)
  assert hook.value_row_major is False


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
