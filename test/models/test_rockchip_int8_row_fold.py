"""CPU coverage for ONPU_INT8_ROW_FOLD: quantize a row window of the shared dense
activation buffer instead of materialising each row tile into its own temporary.

Bit-exactness: the window path reads the same rows in the same order and applies
the identical per-row max/scale/rounding, so payload and scale bytes must match
the shipped per-tile quantize for every layout."""

import numpy as np
import pytest

from tinygrad import Tensor, Context
from tinygrad.llm.kernels.rockchip_int8 import ONPU_INT8_ROW_FOLD, quantize


def _matrix(rows=128, depth=2560, seed=5):
  rng = np.random.default_rng(seed)
  return Tensor((rng.standard_normal((rows, depth)) * 3).astype(np.float32), device='CPU').realize()


def _bits_equal(a, b):
  np.testing.assert_array_equal(a.numpy().view(np.uint8), b.numpy().view(np.uint8))


@pytest.mark.parametrize('sdk_prefill', [False, True])
@pytest.mark.parametrize('offset', [0, 64])
def test_row_window_payload_and_scales_match_shipped_tiles(sdk_prefill, offset):
  x = _matrix()
  shipped_payload, shipped_scale = quantize(x[offset:offset + 64], 2560, sdk_prefill=sdk_prefill)
  window_payload, window_scale = quantize(x, 2560, sdk_prefill=sdk_prefill, rows=64, row_offset=offset)
  assert window_payload.shape == shipped_payload.shape and window_scale.shape == shipped_scale.shape
  _bits_equal(window_payload, shipped_payload)
  np.testing.assert_array_equal(window_scale.numpy().view(np.uint32), shipped_scale.numpy().view(np.uint32))


def test_row_window_full_matrix_is_the_shipped_call():
  x = _matrix()
  shipped_payload, shipped_scale = quantize(x, 2560)
  window_payload, window_scale = quantize(x, 2560, rows=128, row_offset=0)
  _bits_equal(window_payload, shipped_payload)
  np.testing.assert_array_equal(window_scale.numpy().view(np.uint32), shipped_scale.numpy().view(np.uint32))


def test_row_window_rejects_bad_bounds():
  x = _matrix(rows=64)
  for kwargs in ({'rows': 128}, {'rows': 64, 'row_offset': 64}, {'rows': 0}, {'rows': -1},
                 {'row_offset': -1}, {'rows': 1.5}):
    with pytest.raises(ValueError):
      quantize(x, 2560, **kwargs)


def test_row_fold_flag_is_off_by_default():
  assert not ONPU_INT8_ROW_FOLD
  with Context(ONPU_INT8_ROW_FOLD=1):
    assert ONPU_INT8_ROW_FOLD


def test_raw_projection_row_fold_matches_shipped_payloads(monkeypatch):
  """Drive the real raw_projection with a prepared INT8 linear and compare the
  payload/scale stream of the flag-off and flag-on paths."""
  from tinygrad import dtypes
  from tinygrad.llm.kernels import rockchip_int8 as K
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  from tinygrad.runtime.support.rockchip_int8 import padded_depth

  def fake_linear(columns=2560, depth=2560):
    lin = RockchipInt8Linear.__new__(RockchipInt8Linear)
    k, n = padded_depth(depth), -(-columns // 32) * 32
    lin.columns, lin.depth = columns, depth
    lin.packed = [Tensor.empty(n // 32, k // 32, 32, 32, dtype=dtypes.int8, device='ROCKCHIP')]
    lin.scales = [Tensor.empty(n, dtype=dtypes.float, device='CPU')]
    lin.k_parts, lin.input_scale, lin.bias, lin.shared_projection = [], None, None, None
    lin.phase, lin._grouped_chunks = None, []
    lin.tile_columns = lin.decode_tile_columns = 24384
    lin.decode_packed, lin.decode_scales = [], []
    return lin

  rng = np.random.default_rng(9)
  x = Tensor((rng.standard_normal((128, 2560)) * 2).astype(np.float32), device='CPU').realize()
  recorded = {}
  def recording_matmul(payload, weight, rows, columns, column_offset=0, **kwargs):
    recorded.setdefault('payloads', []).append(payload.numpy().view(np.uint8).copy())
    return Tensor.zeros(rows * round_up(columns, 32), dtype=dtypes.int32, device='CPU')
  from tinygrad.helpers import round_up
  monkeypatch.setattr(K, 'raw_int8_matmul', recording_matmul)
  hook = fake_linear()
  with Context(ONPU_INT8_ROW_FOLD=0):
    hook.raw_projection(x)
  off = list(recorded['payloads'])
  recorded.clear()
  with Context(ONPU_INT8_ROW_FOLD=1):
    hook.raw_projection(x)
  on = list(recorded['payloads'])
  assert len(off) == len(on) == 2
  for shipped, folded in zip(off, on):
    np.testing.assert_array_equal(shipped, folded)
