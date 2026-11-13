"""MiniCPM3 prefill: fold the block-output / attention residual adds into the dequantizers.

`ONPU_PREFILL_RESIDUAL_FUSE=1` registers the CPU prefill part-dequantizers of the INT8
projections and, when the consumer is `h + value*residual_scale`, emits ONE custom kernel
that dequantizes every part in registers, sums them in the reference order, scales and adds
the residual.

Every assertion compares against the REAL reference chain (`dequantize_tiles` + the same 3-D
reshape the projection emits + Tensor adds + `h + value*scale`) bit-exactly via a uint32 view,
so +0.0/-0.0 and NaN payload bits must match. The fixture deliberately reproduces the part of
the real graph the earlier kernel-only tests missed: the dequantizer is registered on its 2-D
output but consumed as a 3-D `(1, rows, columns)` value, and SEVERAL blocks fold in the same
forward with identical geometry and different data.
"""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip_int8
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_int8 import (dequantize_tiles, _register_prefill_dequant,
                                                _register_prefill_grouped, _register_prefill_sum,
                                                _walk_registered, try_prefill_residual)
from tinygrad.llm.kernels.rockchip_grouped_int8 import dequantize as grouped_dequantize

COLUMNS = 2560            # minicpm3.embedding_length = ffn_down output width
PARTS = (4096, 2304)      # ffn_down K parts
SCALE = 1.4/np.sqrt(62)   # minicpm3 residual_scale
_FLAG = dict(ONPU_PREFILL_RESIDUAL_FUSE=1)
_OFF = dict(ONPU_PREFILL_RESIDUAL_FUSE=0)


def _bits(t:Tensor) -> np.ndarray:
  return t.realize().numpy().view(np.uint32)


def _part_inputs(counts, kind='typical', seed=7):
  """(raw_tiles, scale_tiles, weight_scales) for one prefill part at the given row tiles."""
  rng = np.random.default_rng(seed)
  raws, sas = [], []
  for count in counts:
    if kind == 'typical':
      raws.append(Tensor(rng.integers(-2**20, 2**20, (count, COLUMNS)).astype(np.int32), device='CPU').realize())
      sas.append(Tensor(rng.uniform(.005, .02, count).astype(np.float32), device='CPU').realize())
    elif kind == 'zeros':
      raws.append(Tensor(np.zeros((count, COLUMNS), np.int32), device='CPU').realize())
      sas.append(Tensor(np.ones(count, np.float32), device='CPU').realize())
    elif kind == 'nonfinite':
      raw = rng.integers(-2**31, 2**31-1, (count, COLUMNS)).astype(np.int32)
      raw[0, 3] = 0
      raws.append(Tensor(raw, device='CPU').realize())
      sa = np.full(count, 3e30, np.float32)
      sa[0] = np.nan
      sas.append(Tensor(sa, device='CPU').realize())
    else: raise ValueError(kind)
  sw = Tensor(rng.uniform(1e-3, 1e-2, COLUMNS).astype(np.float32), device='CPU').realize()
  return raws, sas, sw


def _projected(raws, sas, sw, counts, rows):
  """Exactly what `RockchipInt8Linear.npu` emits: dequantize 2-D, then reshape to 3-D."""
  dequant = dequantize_tiles(raws, sas, sw, list(counts), COLUMNS)
  return dequant, dequant.reshape(1, rows, COLUMNS)


def _reference(hidden, part_inputs, counts, scale):
  rows = sum(counts)
  dequants, parts = [], []
  for raws, sas, sw in part_inputs:
    dequant, part = _projected(raws, sas, sw, counts, rows)
    dequants.append(dequant)
    parts.append(part)
  feed = parts[0]
  for part in parts[1:]:feed = feed+part
  return (hidden+feed*scale).realize(), feed, parts, dequants


@pytest.mark.parametrize('kind', ['typical', 'zeros', 'nonfinite'])
@pytest.mark.parametrize('counts', [(64, 64), (64, 32)])
def test_fused_residual_add_matches_the_real_chain(counts, kind):
  rows = sum(counts)
  rng = np.random.default_rng(9)
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  part_inputs = [_part_inputs(counts, kind, seed=10+index) for index in range(len(PARTS))]
  with Context(**_OFF):
    reference, feed_off, _, _ = _reference(hidden, part_inputs, counts, SCALE)
    reference_bits = _bits(reference)
    assert try_prefill_residual(hidden, feed_off, SCALE) is None, 'flag off must never fold'
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    parts = []
    for raws, sas, sw in part_inputs:
      dequant, part = _projected(raws, sas, sw, counts, rows)
      # registered on the 2-D dequantizer, consumed as the 3-D projection output
      _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
      parts.append(part)
    feed = parts[0]
    for part in parts[1:]:feed = feed+part
    _register_prefill_sum(feed, [_walk_registered(part.uop) for part in parts], COLUMNS, rows)
    fused = try_prefill_residual(hidden, feed, SCALE)
  assert fused is not None, 'registered sum must fold'
  assert fused.shape == reference.shape
  fused_bits = _bits(fused)
  assert np.array_equal(fused_bits, reference_bits), f'{kind} un-equal'
  # NaN rows: compare with equal_nan too so failures report which rows differ
  assert np.array_equal(fused.numpy().reshape(-1), reference.numpy().reshape(-1), equal_nan=True)


def test_two_blocks_with_identical_geometry_fold_with_their_own_operands():
  """62 blocks run in one forward with the same shapes: a fold must never reuse block N-1.

  Registrations are keyed on UOp identity, so this must hold; the fixture feeds DIFFERENT
  raw/scale data per block and compares each block's fused output against its own reference.
  """
  counts, rows = (64, 64), 128
  rng = np.random.default_rng(101)
  hidden_a = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  hidden_b = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  inputs_a = [_part_inputs(counts, 'typical', seed=200+index) for index in range(len(PARTS))]
  inputs_b = [_part_inputs(counts, 'typical', seed=300+index) for index in range(len(PARTS))]
  with Context(**_OFF):
    reference_a = _reference(hidden_a, inputs_a, counts, SCALE)[0]
    reference_b = _reference(hidden_b, inputs_b, counts, SCALE)[0]
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    feeds = []
    for hidden, inputs in ((hidden_a, inputs_a), (hidden_b, inputs_b)):
      parts = []
      for raws, sas, sw in inputs:
        dequant, part = _projected(raws, sas, sw, counts, rows)
        _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
        parts.append(part)
      feed = parts[0]
      for part in parts[1:]:feed = feed+part
      _register_prefill_sum(feed, [_walk_registered(part.uop) for part in parts], COLUMNS, rows)
      feeds.append(feed)
    # both blocks are registered before either folds: interleaved like a captured graph
    fused_a = try_prefill_residual(hidden_a, feeds[0], SCALE)
    fused_b = try_prefill_residual(hidden_b, feeds[1], SCALE)
  assert fused_a is not None and fused_b is not None
  assert np.array_equal(_bits(fused_a), _bits(reference_a)), 'block A folded with foreign operands'
  assert np.array_equal(_bits(fused_b), _bits(reference_b)), 'block B folded with foreign operands'
  assert not np.array_equal(_bits(reference_a), _bits(reference_b)), 'fixture blocks must differ'


def test_single_part_attention_residual_is_folded():
  rows, counts = 128, (64, 64)
  rng = np.random.default_rng(13)
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  part_inputs = (_part_inputs(counts, 'typical', seed=17),)
  with Context(**_OFF):
    reference, attention, _, _ = _reference(hidden, part_inputs, counts, SCALE)
    reference_bits = _bits(reference)
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    raws, sas, sw = part_inputs[0]
    dequant, attention = _projected(raws, sas, sw, counts, rows)
    _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
    fused = try_prefill_residual(hidden, attention, SCALE)
  assert fused is not None
  assert np.array_equal(_bits(fused), reference_bits)


def test_fold_requires_a_registration_matching_the_prefill_geometry():
  rows, counts = 128, (64, 64)
  rng = np.random.default_rng(19)
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  raws, sas, sw = _part_inputs(counts, 'typical', seed=23)
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    assert try_prefill_residual(hidden, hidden*1.0, SCALE) is None, 'unregistered value must fall back'
    dequant, part = _projected(raws, sas, sw, counts, rows)
    _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
    assert try_prefill_residual(hidden[:, :64], part, SCALE) is None, 'wrong residual shape must fall back'
    one_row = Tensor(rng.normal(0, .3, (1, 1, COLUMNS)).astype(np.float32), device='CPU').realize()
    assert try_prefill_residual(one_row, part, SCALE) is None, 'decode rows are not folded'
    # a plain materialized tensor that was never registered must not fold
    assert try_prefill_residual(hidden, Tensor(part.numpy().copy(), device='CPU').realize(), SCALE) is None


def test_split_linear_prefill_path_registers_and_folds(monkeypatch):
  """End-to-end through the real classes: two-part ffn_down, CPU-emulated matmuls."""
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  from tinygrad.helpers import round_up
  rows = 128
  rng = np.random.default_rng(29)
  parts = []
  for depth, blocks in ((PARTS[0], 128), (PARTS[1], 72)):
    part = RockchipInt8Linear.__new__(RockchipInt8Linear)
    part.packed = [Tensor.empty(80, blocks, 32, 32, device='CPU', dtype=dtypes.int8)]
    part.scales = [Tensor.ones(COLUMNS, device='CPU')]
    part.depth, part.columns, part.bias, part.input_scale = depth, COLUMNS, None, None
    part.k_parts, part._grouped_chunks, part.decode_packed, part.decode_scales = [], [], [], []
    part.tile_columns, part.decode_tile_columns, part.phase, part.shared_projection = depth, depth, 'prefill', None
    parts.append(part)
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.parts, down.depth, down.columns, down.bias = parts, sum(PARTS), COLUMNS, None
  down.input_scale, down.phase, down.shared_projection = None, 'prefill', None
  down.grouped_weight, down.grouped_scale, down.grouped_count = None, None, 0
  x = Tensor(rng.normal(0, .3, (1, rows, sum(PARTS))).astype(np.float32), device='CPU').realize()
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  state = {'n': 0}
  def fake_raw(payload, weight, rows_, columns, column_offset=0, **kwargs):
    state['n'] += 1
    n = round_up(columns, 32)
    return Tensor(np.random.default_rng(state['n']).integers(-2**20, 2**20, rows_*n).astype(np.int32),
                  device='CPU', dtype=dtypes.int32)
  monkeypatch.setattr(rockchip_int8, 'raw_int8_matmul', fake_raw)
  reference = fused = None
  for flag in (0, 1):
    state['n'] = 0
    with Context(ONPU_PREFILL_RESIDUAL_FUSE=flag):
      rockchip_int8._pending_dequant.clear()
      feed = down.npu(x)
      if flag: fused = try_prefill_residual(hidden, feed, SCALE)
      else: reference = (hidden+feed*SCALE).realize()
  assert fused is not None, 'registered two-part prefill output must fold'
  assert np.array_equal(_bits(fused), _bits(reference)), 'fused residual add differs from the real chain'
  # with the flag off nothing may be registered: the graph must be unchanged
  for flag in (0,):
    state['n'] = 0
    with Context(ONPU_PREFILL_RESIDUAL_FUSE=flag):
      rockchip_int8._pending_dequant.clear()
      down.npu(x).realize()
    assert len(rockchip_int8._pending_dequant) == 0, 'flag off must not register anything'


def test_short_logical_rows_fall_back_byte_identically():
  """Hypothesis (d) guard: the fold is generated at the padded (max-shape) row count, so it only
  fires when the residual's max_shape product equals the registered row count. In a captured
  graph the row dim is symbolic (`toks`, vmax 128) and that product is 128; a CONCRETE short
  tensor has max_shape == shape, so it must fall back - and the fallback must be the exact
  previous expression, never a partial/padded compute."""
  counts, rows, logical = (64, 64), 128, 96
  rng = np.random.default_rng(41)
  full = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  hidden = full.shrink(((0,1),(0,logical),(0,COLUMNS)))
  part_inputs = [_part_inputs(counts, 'typical', seed=50+index) for index in range(len(PARTS))]
  with Context(**_OFF):
    _, feed_off, _, _ = _reference(full, part_inputs, counts, SCALE)
    short_value = feed_off.shrink(((0,1),(0,logical),(0,COLUMNS)))
    reference_bits = _bits((hidden+short_value*SCALE).realize())
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    parts = []
    for raws, sas, sw in part_inputs:
      dequant, part = _projected(raws, sas, sw, counts, rows)
      _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
      parts.append(part)
    feed = parts[0]
    for part in parts[1:]:feed = feed+part
    _register_prefill_sum(feed, [_walk_registered(part.uop) for part in parts], COLUMNS, rows)
    short_value = feed.shrink(((0,1),(0,logical),(0,COLUMNS)))
    assert try_prefill_residual(hidden, short_value, SCALE) is None, 'concrete short rows must fall back'
    # the fallback expression must stay byte-identical
    assert np.array_equal(_bits((hidden+short_value*SCALE).realize()), reference_bits)


def test_fused_kernel_joins_the_shared_prefill_pool():
  """The fold must run on the same 4-worker prefill pool as the dequantizers it replaces.

  The shipped `_dequantize_tiles` is `@shared_program`, so under a shared host pool its
  emitted `pthread_create` dispatch is rewritten into `shared_work_callback`. The fused
  kernel must offer the identical dispatch shape (a 4-entry `Work work[4]` initialiser,
  one `worker` name, and the exact join tail) so the SAME rewrite applies; otherwise it
  pays three thread creations per call while the kernels it replaces are parked workers.
  This asserts the source-side contract only (no pool is started here).
  """
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.kernels.rockchip_shared_pool import pooled_source
  counts, rows, n = (64, 64), 128, COLUMNS
  raws = [Tensor.empty(count, n, device='CPU', dtype=dtypes.int32).realize() for count in counts]
  sas = [Tensor.empty(count, device='CPU', dtype=dtypes.float).realize() for count in counts]
  sw0 = Tensor.empty(COLUMNS, device='CPU', dtype=dtypes.float).realize()
  sw1 = Tensor.empty(COLUMNS, device='CPU', dtype=dtypes.float).realize()
  hidden = Tensor.empty(rows, COLUMNS, device='CPU', dtype=dtypes.float).realize()
  with Context(ONPU_PREFILL_RESIDUAL_FUSE=1):
    out = Tensor.empty(rows, COLUMNS, device='CPU', dtype=dtypes.float)
    program = rockchip_int8._dequant_residual(
      out.uop, hidden.uop, tuple(r.uop for r in (*raws, *raws)),
      tuple(a.uop for a in (*sas, *sas)), (sw0.uop, sw1.uop), counts, COLUMNS, counts[0], 2, SCALE)
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  pooled, record = pooled_source(source, 0x1000)
  assert record['changed'] is True, 'the fused kernel must expose the pool-rewritable dispatch'
  assert record['thread_creates_replaced_per_call'] == 3
  assert 'shared_work_callback' in pooled and 'pthread_create(&threads[i]' not in pooled
  # the rewrite must be deterministic on repeat (the pool caches one program per shape)
  assert pooled_source(source, 0x1000)[0] == pooled
  # and the shipped dequantizer must still offer the same dispatch shape
  shipped = next(u.arg for u in rockchip_int8._dequantize_tiles(
    out.uop, tuple(r.uop for r in raws), tuple(a.uop for a in sas), sw0.uop,
    counts, COLUMNS, n, 0).src if u.op is Ops.SOURCE)
  assert pooled_source(shipped, 0x1000)[1]['changed'] is True


def test_pooled_fused_kernel_is_bit_identical_end_to_end():
  """With a live shared pool the rewritten fused kernel must stay uint32-equal to the shipped chain.

  Regression guard for the pool rewrite: `pooled_source` only swaps the thread-creation
  tail for parked-worker dispatch, so the arithmetic must be untouched. Runs both sites
  (attention: 1 part, ffn: 2 parts) through the real shared pool and compares against the
  unmixed `dequantize_tiles` + Tensor-add reference.
  """
  import os, platform
  if platform.machine().lower() not in ("aarch64", "arm64"):
    pytest.skip("shared Rockchip CPU workers require AArch64")
  if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
    pytest.skip("shared Rockchip CPU workers require CPU4-7")
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
  counts, rows = (64, 64), 128
  rng = np.random.default_rng(9)
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  part_inputs = [_part_inputs(counts, 'typical', seed=10+index) for index in range(len(PARTS))]
  with Context(**_OFF):
    _, feed_off, parts_off, _ = _reference(hidden, part_inputs, counts, SCALE)
    reference = _bits((hidden+feed_off*SCALE).realize())
    reference_attn = _bits((hidden+parts_off[0]*SCALE).realize())
  pool = AttentionPool()
  state = SharedHostPool(pool)
  try:
    with state.scope(), Context(**_FLAG):
      rockchip_int8._pending_dequant.clear()
      parts = []
      for raws, sas, sw in part_inputs:
        dequant, part = _projected(raws, sas, sw, counts, rows)
        _register_prefill_dequant(dequant, raws, sas, sw, list(counts), COLUMNS)
        parts.append(part)
      feed = parts[0]
      for part in parts[1:]:feed = feed+part
      _register_prefill_sum(feed, [_walk_registered(part.uop) for part in parts], COLUMNS, rows)
      fused = try_prefill_residual(hidden, feed, SCALE)
      attn = try_prefill_residual(hidden, parts[0], SCALE)
    assert fused is not None and attn is not None, 'fold did not fire under the pool scope'
    assert any(c['kernel']=='rockchip_dequantize_residual_add' for c in state.changes), \
      'the fused kernel was not pooled; only the dequantizers were rewritten'
    assert np.array_equal(_bits(fused), reference), 'pooled ffn-site fold differs from the shipped chain'
    assert np.array_equal(_bits(attn), reference_attn), 'pooled attn-site fold differs from the shipped chain'
  finally:
    state.restore()
    pool.close()


# ---------------------------------------------------------------------------
# Row-major grouped parts (`rockchip_grouped_int8`) folded together with blocked parts.
#
# The grouped ffn_down branch (`RockchipInt8SplitLinear.npu`) dequantizes a row-major INT32
# tensor (`out[row*padded+c] = ((float)raw[row*padded+c]*a[row])*w[c]`) instead of the
# NPU's blocked per-part layout. `_dequant_residual` therefore takes a per-part layout
# descriptor: 0 keeps the blocked source index, a positive value is the row-major stride.
# These fixtures reproduce the real grouped branch's registered descriptors and assert that
# the folded output is uint32-identical to `grouped_int8_matmul(..., materialize=True)`
# plus the tail part sums plus `hidden + value*scale`.
GROUP_PARTS = 3                       # regions the grouped branch stacks into one weight
PADDED = 4096                         # grouped raw row stride == the grouped K
TAIL_COUNTS = (64, 64)


def _grouped_inputs(kind='typical', seed=7, groups=GROUP_PARTS, padded=PADDED, rows=128):
  """One row-major grouped raw plus its per-row activation and per-group weight scales."""
  rng = np.random.default_rng(seed)
  if kind == 'typical':
    raw = rng.integers(-2**20, 2**20, (groups, rows, padded)).astype(np.int32)
    sa = rng.uniform(.005, .02, groups*rows).astype(np.float32)
    sw = rng.uniform(1e-3, 1e-2, (groups, padded)).astype(np.float32)
  elif kind == 'zeros':
    raw = np.zeros((groups, rows, padded), np.int32)
    sa = np.ones(groups*rows, np.float32)
    sw = np.ones((groups, padded), np.float32)
  elif kind == 'nonfinite':
    raw = rng.integers(-2**31, 2**31-1, (groups, rows, padded)).astype(np.int32)
    raw[0, 0, 3] = 0
    sa = np.full(groups*rows, 3e30, np.float32)
    sa[0] = np.nan
    sw = np.full((groups, padded), 3e30, np.float32)
  else: raise ValueError(kind)
  return (Tensor(raw, device='CPU').realize(), Tensor(sa, device='CPU').realize(),
          Tensor(sw, device='CPU').realize())


def _blocked_tail(counts, seed=90):
  """A standing blocked-layout part, as the non-grouped parts of the same sum emit."""
  rng = np.random.default_rng(seed)
  raws = [Tensor(rng.integers(-2**20, 2**20, (count, round_up(COLUMNS, 32))).astype(np.int32),
                 device='CPU').realize() for count in counts]
  sas = [Tensor(rng.uniform(.005, .02, count).astype(np.float32), device='CPU').realize() for count in counts]
  sw = Tensor(rng.uniform(1e-3, 1e-2, round_up(COLUMNS, 32)).astype(np.float32), device='CPU').realize()
  return raws, sas, sw


def _grouped_reference(hidden, raw, sa, sw, tail, counts, scale, groups=GROUP_PARTS, padded=PADDED):
  """`grouped_int8_matmul(..., materialize=True)` + tail sums + `hidden + value*scale`.

  `rockchip_grouped_int8.dequantize` IS the dequantizer `matmul(materialize=True)` calls, so
  this reproduces the unfused chain exactly, including the grouped branch's order: the
  grouped outputs are summed first (`((g0+g1)+g2)`), then the remaining parts are added.
  """
  dequant = grouped_dequantize(raw, sa, sw, groups, sum(counts), COLUMNS, padded)
  groups_out = [dequant[index].reshape(1, sum(counts), COLUMNS) for index in range(groups)]
  value = groups_out[0]
  for output in groups_out[1:]:value = value+output
  value = value+tail
  return (hidden+value*scale).realize(), value, groups_out


@pytest.mark.parametrize('kind', ['typical', 'zeros', 'nonfinite'])
def test_grouped_row_major_part_folds_bit_exactly(kind):
  """A row-major grouped part folds with a blocked tail part: uint32-identical to the chain."""
  raw, sa, sw = _grouped_inputs(kind, seed=7+hash(kind) % 97)
  rows = raw.shape[1]
  tail_raws, tail_sas, tail_sw = _blocked_tail(TAIL_COUNTS)
  tail = dequantize_tiles(tail_raws, tail_sas, tail_sw, list(TAIL_COUNTS), COLUMNS).reshape(1, rows, COLUMNS)
  rng = np.random.default_rng(5)
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  with Context(**_OFF):
    rockchip_int8._pending_dequant.clear()
    reference, value_off, _ = _grouped_reference(hidden, raw, sa, sw, tail, TAIL_COUNTS, SCALE)
    reference_bits = _bits(reference)
    assert try_prefill_residual(hidden, value_off, SCALE) is None, 'flag off must never fold'
    assert len(rockchip_int8._pending_dequant) == 0, 'flag off must not register anything'
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    # what the real grouped branch registers: per-group descriptors plus the blocked tail.
    # The descriptors are keyed on UOp identity, so ONE tensor per group must be registered
    # and that same tensor must be used for the sum (as the real branch does).
    groups_out = _grouped_outputs(raw, sa, sw)
    assert _register_prefill_grouped(groups_out, raw, sa, sw, GROUP_PARTS, rows, TAIL_COUNTS,
                                     COLUMNS, PADDED)
    _register_prefill_dequant(tail, tail_raws, tail_sas, tail_sw, list(TAIL_COUNTS), COLUMNS)
    value = groups_out[0]
    for output in groups_out[1:]:value = value+output
    value = value+tail
    _register_prefill_sum(value, [_walk_registered(o.uop) for o in groups_out]
                          + [_walk_registered(tail.uop)], COLUMNS, rows)
    fused = try_prefill_residual(hidden, value, SCALE)
  assert fused is not None, 'mixed-layout sum must fold'
  assert fused.shape == reference.shape
  assert np.array_equal(_bits(fused), reference_bits), f'{kind} row-major grouped fold un-equal'
  assert np.array_equal(fused.numpy().reshape(-1), reference.numpy().reshape(-1), equal_nan=True)


def _grouped_outputs(raw, sa, sw):
  return [grouped_dequantize(raw, sa, sw, GROUP_PARTS, raw.shape[1], COLUMNS, PADDED)[index]
          .reshape(1, raw.shape[1], COLUMNS) for index in range(GROUP_PARTS)]


def test_grouped_fold_requires_the_padded_stride():
  """A row-major grouped part cannot fold unless its row stride covers the output width."""
  raw, sa, sw = _grouped_inputs('typical', seed=13)
  outputs = _grouped_outputs(raw, sa, sw)
  with Context(**_FLAG):
    rockchip_int8._pending_dequant.clear()
    # a stride narrower than the output cannot address the row-major columns
    assert not _register_prefill_grouped(outputs, raw, sa, sw, GROUP_PARTS, raw.shape[1],
                                         (64, 64), COLUMNS, COLUMNS-32)
    assert not rockchip_int8._pending_dequant, 'a rejected grouped part must register nothing'
    # and the row-tile counts of every part of one sum must agree
    assert not _register_prefill_grouped(outputs, raw, sa, sw, GROUP_PARTS, raw.shape[1],
                                         (64, 32), COLUMNS, PADDED)
    assert not rockchip_int8._pending_dequant


# ---------------------------------------------------------------------------
# The real MiniCPM3 prefill FFN route registers and folds.
#
# MiniCPM3-4B ffn_down is `in_features=6400`, so `RockchipInt8SplitLinear` builds only TWO
# parts (4096+2304) and its early grouped branch is never taken (it needs >=4 parts). The
# prefill FFN is the fused SwiGLU weight-streaming route instead: `rockchip_int8_swiglu_pack`
# writes the down payload from the projected gate/up tile, and the down projection runs one
# `raw_int8_matmul` + `dequantize_tiles` per K group. That route is what produced the
# hardware-observed `SHRINK>RESHAPE>ADD>AFTER` chain, so its ordered sum is what must
# register. `_RockchipInt8GatedFeedForward.__call__` now registers each group dequantizer and
# the sum in the loop's own order.


def _mini_kernel():
  """The real MiniCPM3 fused SwiGLU prefill kernel, CPU-emulated."""
  from types import SimpleNamespace
  from tinygrad.llm.kernels import rockchip_gelu_ffn as ffn
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.packed = [Tensor.empty(400, 80, 32, 32, device='CPU', dtype=dtypes.int8)]   # 12800 cols, k=2560
  joint.scales = [Tensor.ones(2*6400, device='CPU')]
  joint.depth, joint.columns, joint.tile_columns, joint.bias = COLUMNS, 2*6400, 24384, None
  joint.input_scale, joint.shared_projection = None, None
  joint.prefill, joint.decode = 'on', 'on'
  parts = []
  for depth, blocks in ((4096, 128), (2304, 72)):
    part = RockchipInt8Linear.__new__(RockchipInt8Linear)
    part.packed = [Tensor.empty(80, blocks, 32, 32, device='CPU', dtype=dtypes.int8)]
    part.scales = [Tensor.ones(COLUMNS, device='CPU')]
    part.depth, part.columns, part.bias, part.input_scale = depth, COLUMNS, None, None
    part.k_parts, part._grouped_chunks, part.decode_packed, part.decode_scales = [], [], [], []
    part.tile_columns, part.decode_tile_columns, part.phase, part.shared_projection = depth, depth, 'prefill', None
    part.prefill, part.decode = 'on', 'on'
    parts.append(part)
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.parts, down.depth, down.columns, down.bias = parts, 6400, COLUMNS, None
  down.input_scale, down.phase, down.shared_projection = None, 'prefill', None
  down.prefill, down.decode = 'on', 'on'
  down.grouped_weight, down.grouped_scale, down.grouped_count = None, None, 0
  kernel = ffn.RockchipInt8SwiGLUFeedForward.__new__(ffn.RockchipInt8SwiGLUFeedForward)
  kernel.activation, kernel.decode_enabled, kernel.pool = 'silu', False, None
  kernel.projections = SimpleNamespace(joint=joint, layers=[SimpleNamespace(phase='prefill', prefill='on', decode='on')])
  kernel.down, kernel.combined_scales, kernel.split_prefill = down, None, None
  return ffn, kernel


def _mini_raws(ffn, seed=11):
  """Deterministic per-call raw INT32 outputs so flag on/off see identical operands."""
  state = {'n': seed}
  def fake(payload, weight, rows, columns, column_offset=0, **kwargs):
    state['n'] += 1
    return Tensor(np.random.default_rng(state['n']).integers(-2**20, 2**20,
                  rows*round_up(columns, 32)).astype(np.int32), device='CPU', dtype=dtypes.int32)
  original = ffn.raw_int8_matmul
  return fake, original, state


@pytest.mark.parametrize('rows', [128, 64])
def test_minicpm3_prefill_ffn_route_registers_and_folds(rows):
  """The real fused SwiGLU prefill FFN must register its sum and fold the residual."""
  ffn, kernel = _mini_kernel()
  rng = np.random.default_rng(29)
  x = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  hidden = Tensor(rng.normal(0, .3, (1, rows, COLUMNS)).astype(np.float32), device='CPU').realize()
  fake, original, state = _mini_raws(ffn)
  ffn.raw_int8_matmul = fake
  reference = fused = None
  try:
    for flag in (0, 1):
      state['n'] = 11
      with Context(ONPU_PREFILL_RESIDUAL_FUSE=flag):
        rockchip_int8._pending_dequant.clear()
        feed = kernel(x)
        if flag:
          assert len(rockchip_int8._pending_dequant) > 0, 'the FFN route registered nothing'
          fused = try_prefill_residual(hidden, feed, SCALE)
        else:
          assert len(rockchip_int8._pending_dequant) == 0, 'flag off must not register anything'
          reference = (hidden+feed*SCALE).realize()
  finally:
    ffn.raw_int8_matmul = original
  assert fused is not None, 'the MiniCPM3 prefill FFN residual must fold'
  assert fused.shape == reference.shape == (1, rows, COLUMNS)
  assert np.array_equal(_bits(fused), _bits(reference)), 'FFN fold differs from the real chain'


def test_minicpm3_prefill_ffn_fold_is_skipped_with_the_flag_off():
  """Flag off: byte-identical fallback, nothing registered, no fused kernel emitted."""
  ffn, kernel = _mini_kernel()
  rng = np.random.default_rng(31)
  x = Tensor(rng.normal(0, .3, (1, 128, COLUMNS)).astype(np.float32), device='CPU').realize()
  hidden = Tensor(rng.normal(0, .3, (1, 128, COLUMNS)).astype(np.float32), device='CPU').realize()
  fake, original, state = _mini_raws(ffn)
  ffn.raw_int8_matmul = fake
  try:
    with Context(**_OFF):
      rockchip_int8._pending_dequant.clear()
      value = kernel(x)
      assert len(rockchip_int8._pending_dequant) == 0
      assert try_prefill_residual(hidden, value, SCALE) is None
      fallback = (hidden+value*SCALE).realize()
    with Context(**_OFF):
      # the same graph twice must be byte-identical and must not have created a fused kernel
      state['n'] = 11
      rockchip_int8._pending_dequant.clear()
      value2 = kernel(x)
      assert np.array_equal(_bits((hidden+value2*SCALE).realize()), _bits(fallback))
      assert len(rockchip_int8._pending_dequant) == 0
  finally:
    ffn.raw_int8_matmul = original
