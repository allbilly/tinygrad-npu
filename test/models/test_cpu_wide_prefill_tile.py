import functools

import numpy as np
import pytest

from tinygrad import Tensor, UOp
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip_cpu
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention


@pytest.mark.parametrize("dim", [256, 512])
@pytest.mark.parametrize("active", [17, 127, 128])
def test_gemma4_wide_prefill_tile_preserves_exact_bits_with_changed_inputs_and_future_nan(dim, active):
  rng = np.random.default_rng(20261104 + dim + active)
  query_data = rng.normal(0, .3, (1, 8, 128, dim)).astype(np.float32)
  cache_data = rng.normal(0, .3, (2, 1, 1, 512, dim)).astype(np.float16)
  cache_data[:, :, :, 128:] = np.nan
  query, cache = Tensor(query_data, device="CPU").realize(), Tensor(cache_data, device="CPU").realize()
  rows = UOp.variable(f"wide_tile_rows_{dim}_{active}", 1, 128).bind(active)

  with Context(ONPU_CPU_PREFILL_WIDE_TILE=0):
    control = prefill_attention(query[:, :, :rows], cache, 128, score_scale=1.0).pad_to((1, 8, 128, dim)).realize().numpy()[:, :, :active]
  with Context(ONPU_CPU_PREFILL_WIDE_TILE=1):
    candidate = prefill_attention(query[:, :, :rows], cache, 128, score_scale=1.0).pad_to((1, 8, 128, dim)).realize().numpy()[:, :, :active]
  np.testing.assert_array_equal(candidate.view(np.uint32), control.view(np.uint32))
  assert np.isfinite(candidate).all()


def _attention_source_and_tile(monkeypatch, *, heads=8, kv_heads=1, dim=256, context=512, max_rows=128,
                               score_scale=1.0, window=0, softcap=0.0, decode=False, enabled=True):
  captured = []
  original = rockchip_cpu._attention

  @functools.wraps(original)
  def record(*args, **kwargs):
    program = original(*args, **kwargs)
    captured.append((kwargs.get("query_rows"), program.src[2].arg))
    return program

  monkeypatch.setattr(rockchip_cpu, "_attention", record)
  query_data = np.ones((1, heads, max_rows if not decode else 1, dim), dtype=np.float32)
  cache_data = np.ones((2, 1, kv_heads, context, dim), dtype=np.float16)
  query, cache = Tensor(query_data, device="CPU").realize(), Tensor(cache_data, device="CPU").realize()
  with Context(ONPU_CPU_PREFILL_WIDE_TILE=int(enabled)):
    if decode:
      decode_attention(query, cache, 1, score_scale=score_scale, window=window, softcap=softcap).realize()
    else:
      prefill_attention(query, cache, max_rows, score_scale=score_scale, window=window,
                        softcap=softcap).realize()
  assert len(captured) == 1
  return captured[0]


def test_gemma4_wide_tile_source_uses_eight_rows_only_when_enabled(monkeypatch):
  selected, enabled_source = _attention_source_and_tile(monkeypatch, enabled=True)
  assert selected == 8
  assert "for(;row+8<=active_rows;row+=8)" in enabled_source

  selected, disabled_source = _attention_source_and_tile(monkeypatch, enabled=False)
  assert selected is None
  assert "for(;row+4<=active_rows;row+=4)" in disabled_source
  assert "for(;row+8<=active_rows;row+=8)" not in disabled_source


@pytest.mark.parametrize("kwargs", [
  {"heads": 4},
  {"context": 256},
  {"max_rows": 64},
  {"score_scale": 0.5},
  {"window": 128},
  {"softcap": 1.0},
  {"decode": True},
])
def test_gemma4_wide_tile_source_guard_falls_back_to_four_or_decode(monkeypatch, kwargs):
  selected, source = _attention_source_and_tile(monkeypatch, enabled=True, **kwargs)
  assert selected is None
  assert "for(;row+8<=active_rows;row+=8)" not in source
  if not kwargs.get("decode", False) and not kwargs.get("window", 0):
    assert "for(;row+4<=active_rows;row+=4)" in source
