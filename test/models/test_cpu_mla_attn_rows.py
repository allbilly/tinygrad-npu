"""Bit-exactness of the row-paired fused MLA attention kernel (ONPU_MLA_ATTN_ROWS).

The `attn3` row-paired variant processes two rows per 2-head block sharing each 16-key K tile.
It must be uint32-identical to the shipped single-row kernel: same per-output ascending-c/key
FP accumulation order, same mask/scale/softmax arithmetic and the same signed-zero/non-finite
tail guard. This test drives the real fused prefill path (`query_key_softmax_value`) for the
MiniCPM3 geometry (40 heads, context 512/128, width 288, values 256) with the flag off/on and
compares raw output bits, including poisoned tail rows/keys.
"""
import math
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_mla_attention import MLAAttention


_RECORDED: list[int] = []


def _helper() -> MLAAttention:
  with Context(ONPU_MLA_FUSED_ATTN=1, ONPU_MLA_HEAD_BLOCK=4):
    return MLAAttention(40, 512, 288, 256, 1 / math.sqrt(96))


def _fused(helper, q, rope, cache, start, rows, attn_rows:int):
  # Production recipe is ONPU_MLA_HEAD_BLOCK=4; the C rows-paired path requires it
  # (rockchip_mla_attention.c: MLA_HEAD_BLOCK==4), otherwise the shipped ref kernel runs even
  # with ONPU_MLA_ATTN_ROWS set. Record the generated program's attn_rows argument so the test
  # proves the fast path was actually selected, not just that the output matches.
  import tinygrad.llm.kernels.rockchip_mla_attention as _attn
  original = _attn._fused_program
  def _recording(*args, **kwargs):
    _RECORDED.append(int(args[-1]))
    return original(*args, **kwargs)
  _attn._fused_program = _recording
  try:
    with Context(ONPU_MLA_ATTN_ROWS=attn_rows, ONPU_MLA_HEAD_BLOCK=4):
      return helper.query_key_softmax_value(q[:, :, :rows], cache, start, rope=rope[:, :, :rows]).numpy()
  finally:
    _attn._fused_program = original


def _operands(rng, start:int, rows:int, capacity:int, context:int, poison_tail:bool):
  heads = 40
  q = rng.normal(0, 2.0, (1, heads, capacity, 256)).astype(np.float32)
  rope = rng.normal(0, 1.0, (1, heads, capacity, 32)).astype(np.float32)
  cache = rng.normal(0, 1.0, (1, 1, context, 288)).astype(np.float32)
  if poison_tail:
    q[:, :, rows:] = np.nan
    rope[:, :, rows:] = np.nan
    cache[:, :, start + rows:] = np.nan
  return [Tensor(x, device='CPU').realize() for x in (q, rope, cache)]


@pytest.mark.parametrize('context,start,rows', [
  (512, 0, 2), (512, 0, 3), (512, 0, 4), (512, 0, 7), (512, 0, 8),
  (512, 0, 16), (512, 0, 32), (512, 0, 64), (512, 0, 128),
  (512, 64, 64), (512, 128, 128), (512, 384, 64),
])
def test_attn_rows_matches_single_row_bits(context, start, rows):
  helper = _helper()
  assert helper.fused_attn, 'fused attention must be enabled for the qualified geometry'
  rng = np.random.default_rng(20261110 + start * 131 + rows)
  capacity = 128
  for poison_tail in (False, True):
    q, rope, cache = _operands(rng, start, rows, capacity, context, poison_tail)
    _RECORDED.clear()
    base = _fused(helper, q, rope, cache, start, rows, 0)
    assert _RECORDED == [0], f'flag-off must generate the shipped kernel define, got {_RECORDED}'
    _RECORDED.clear()
    paired = _fused(helper, q, rope, cache, start, rows, 2)
    assert _RECORDED == [2], f'flag-on must select the rows-paired kernel define, got {_RECORDED}'
    assert base.shape == paired.shape == (1, 40, rows, 256) and np.isfinite(base).all()
    np.testing.assert_array_equal(base.view(np.uint32), paired.view(np.uint32),
                                  err_msg=f'rows={rows} context={context} start={start} poison={poison_tail}')


def test_attn_rows_real_model_path_bits():
  """Drive the real tinygrad path the model uses (capacity-padded q + zero-tail marking)."""
  helper = _helper()
  rng = np.random.default_rng(424242)
  q, rope, cache = _operands(rng, 0, 128, 128, 512, poison_tail=False)
  _RECORDED.clear()
  base = _fused(helper, q, rope, cache, 0, 128, 0)
  assert _RECORDED == [0]
  _RECORDED.clear()
  paired = _fused(helper, q, rope, cache, 0, 128, 2)
  assert _RECORDED == [2]
  assert np.isfinite(base).all()
  np.testing.assert_array_equal(base.view(np.uint32), paired.view(np.uint32))
