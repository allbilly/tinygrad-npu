"""Split MLA QK pointers must preserve contiguous 288-wide FP32 dot bits."""
import math
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels import rockchip_mla_attention as mla

HEADS, CONTEXT, WIDTH, VALUES = 4, 512, 288, 256
SCALE = 1 / math.sqrt(96)
ROPE = WIDTH - 256
NOPE = WIDTH - ROPE


def _inputs(kind, capacity, total, seed, context=CONTEXT):
  rng = np.random.default_rng(seed)
  q_nope = rng.normal(0, .2, (1, HEADS, capacity, NOPE)).astype(np.float32)
  q_rope = rng.normal(0, .2, (1, HEADS, capacity, ROPE)).astype(np.float32)
  cache = rng.normal(0, .2, (1, 1, context, WIDTH)).astype(np.float32)
  # Poison every query row and cache key the kernel must not read.
  q_nope[:, :, total:] = np.nan
  q_rope[:, :, total:] = np.nan
  cache[:, :, total:] = np.nan
  if kind == 'zero':
    q_nope[:, :, :total] = 0.0
    q_rope[:, :, :total] = -0.0
    cache[:, :, :total] = 0.0
  elif kind == 'subnormal':
    tiny = np.nextafter(np.float32(0), np.float32(1))
    q_nope[:, :, :total] = tiny
    q_rope[:, :, :total] = -tiny
    cache[:, :, :total] = np.nextafter(np.float32(0), np.float32(-1))
    q_rope[:, :, 0, :] = tiny
  elif kind == 'nonfinite':
    q_nope[0, 0, 0, 3] = np.nan
    q_rope[0, 0, 0, 1] = np.inf
    cache[0, 0, 0, 7] = -np.inf
    cache[0, 0, total - 1, 11] = np.nan
  return q_nope, q_rope, cache


@pytest.mark.parametrize('kind', ['finite', 'zero', 'subnormal', 'nonfinite'])
@pytest.mark.parametrize('capacity,start,rows', [(128, 0, 128), (128, 64, 64), (128, 127, 1), (1, 0, 1)])
@pytest.mark.parametrize('causal_work', [False, True])
def test_split_qk_matches_contiguous_bits(monkeypatch, kind, capacity, start, rows, causal_work):
  monkeypatch.setattr(mla, 'getenv', lambda name, default: int(causal_work))
  helper = mla.MLAAttention(HEADS, CONTEXT, WIDTH, VALUES, SCALE)
  assert helper.causal_work is causal_work
  q_nope, q_rope, cache = _inputs(kind, capacity, start + rows, seed=910 + capacity)
  contiguous = np.concatenate((q_nope, q_rope), axis=-1)
  tensors = [Tensor(x, device='CPU').realize() for x in (contiguous, q_nope, q_rope, cache)]
  startvar, rowvar = UOp.variable('mla_split_start', 0, CONTEXT - 1), UOp.variable('mla_split_rows', 1, capacity)
  shape = (1, HEADS, capacity, CONTEXT + capacity - 1)

  def compare(q, qn, qr, k, s, n):
    old = helper.query_key(q[:, :, :n], k, s).pad_to(shape).contiguous().realize()
    new = helper.query_key(qn[:, :, :n], k, s, rope=qr[:, :, :n]).pad_to(shape).contiguous().realize()
    return old, new

  old, new = TinyJit(compare)(*tensors, startvar.bind(start), rowvar.bind(rows))
  np.testing.assert_array_equal(old.numpy().view(np.uint32), new.numpy().view(np.uint32))


@pytest.mark.parametrize('start,rows', [(0, 17), (7, 1)])
def test_split_qk_static_positions_match_contiguous_bits(start, rows):
  helper = mla.MLAAttention(HEADS, 64, WIDTH, VALUES, SCALE)
  q_nope, q_rope, cache = _inputs('finite', 17, start + rows, seed=950, context=64)
  contiguous = np.concatenate((q_nope, q_rope), axis=-1)
  q, qn, qr, k = [Tensor(x, device='CPU').realize() for x in (contiguous, q_nope, q_rope, cache)]
  old = helper.query_key(q[:, :, :rows], k, start).numpy()
  new = helper.query_key(qn[:, :, :rows], k, start, rope=qr[:, :, :rows]).numpy()
  np.testing.assert_array_equal(old.view(np.uint32), new.view(np.uint32))


def test_split_qk_model_with_prepared_projections_prefill_and_decode_bits():
  from test.models.test_cpu_mla_projections import _model
  from tinygrad.llm.kernels.rockchip_mla_cpu import configure as configure_projections
  serial, prepared = _model(seed=77), _model(seed=77)
  assert configure_projections(prepared) == 1
  assert mla.configure(prepared) == 1
  startvar = UOp.variable('mla_split_model_start', 0, 63)
  prefill = [TinyJit(lambda ids, start, _m=m: _m.logits(ids, start).realize()) for m in (serial, prepared)]
  decode = [TinyJit(lambda ids, start, _m=m: _m.logits(ids, start).realize()) for m in (serial, prepared)]
  tokens = Tensor([list(range(1, 14))], device='CPU')
  old, new = [run(tokens, startvar.bind(0)).numpy() for run in prefill]
  np.testing.assert_array_equal(old.view(np.uint32), new.view(np.uint32))
  token = Tensor([[31]], device='CPU')
  old, new = [run(token, startvar.bind(13)).numpy() for run in decode]
  np.testing.assert_array_equal(old.view(np.uint32), new.view(np.uint32))


def test_split_qk_rejects_mismatched_rope_geometry():
  helper = mla.MLAAttention(HEADS, 64, WIDTH, VALUES, SCALE)
  q_nope, q_rope, cache = _inputs('finite', 17, 17, seed=960, context=64)
  qn, qr, k = [Tensor(x, device='CPU').realize() for x in (q_nope, q_rope, cache)]
  with pytest.raises(ValueError):
    helper.query_key(qn[:, :, :, :255], k, 0, rope=qr)
  with pytest.raises(ValueError):
    helper.query_key(qn, k, 0, rope=qr[:, :, :, :16])
  with pytest.raises(ValueError):
    helper.query_key(qn, k, 0, rope=qr[:, :, :1])
