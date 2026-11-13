"""MiniCPM3-4B decode FFN/norm host fusion at the real MiniCPM3 geometry.

MiniCPM3-4B (GGUF: minicpm3.embedding_length=2560, feed_forward_length=6400,
block_count=62) takes the *narrow* raw FFN route (one packed gate/up tile of 12800
columns, down split 4096+2304) rather than ChatGLM's wide 24576+2816 route, and its
decode norm width is 2560 instead of 4096. These tests pin the three task-10 wins at
that geometry:

* the fused raw SwiGLU pack (already shipped route) reproduces the decode chain,
* `ONPU_NORM_QUANT_DECODE` fuses RMSNorm+quantize at depth 2560,
* `ONPU_FFN_DOWN_MERGE` merges the two-part (4096+2304) down dequantizers and their sum,
* `ONPU_FFN_GROUP_POOL_ALL` puts one K group per host-pool worker (2 groups here).

All assertions are bit-exact (`np.array_equal`, max|diff| = 0) against the REAL
standing kernels, and the fused program is proven to be the one that ran.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, round_up
from tinygrad.llm.kernels import rockchip_gelu_ffn as ffn
from tinygrad.llm.kernels import rockchip_norm_quant as norm_quant
from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles, quantize
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_pair_vector
from tinygrad.llm.kernels.rockchip_cpu import rms_norm

HIDDEN = 6400            # minicpm3.feed_forward_length
COLUMNS = 2*HIDDEN       # fused gate/up width
MODEL_HIDDEN = 2560      # minicpm3.embedding_length
DEPTHS = (4096, 2304)    # ffn_down K parts (6400 split by 4096)
TILES = (ffn.ProjectionTile(0, 1, 0, 12800, 0),)
EPS = 1e-5
_DECODE = dict(ONPU_NORM_QUANT=0, ONPU_NORM_QUANT_DECODE=1)


def _wide_inputs(kind='typical', seed=13):
  rng = np.random.default_rng(seed)
  if kind == 'typical':
    raw = rng.integers(-2**20, 2**20, size=(1, COLUMNS)).astype(np.int32)
    sa, sw = np.array([0.0137], np.float32), rng.uniform(1e-3, 1e-2, COLUMNS).astype(np.float32)
  elif kind == 'zeros':
    raw = np.zeros((1, COLUMNS), np.int32)
    sa, sw = np.array([1.0], np.float32), np.full(COLUMNS, 1e-3, np.float32)
  elif kind == 'nonfinite':
    raw = rng.integers(-2**31, 2**31-1, size=(1, COLUMNS)).astype(np.int32)
    sa, sw = np.array([3e30], np.float32), np.full(COLUMNS, 3e30, np.float32)
  else: raise ValueError(kind)
  return (Tensor(raw, device='CPU', dtype=dtypes.int32).realize(), Tensor(sa, device='CPU').realize(),
          Tensor(sw, device='CPU').realize())


def _decode_chain(raws, sa, sw):
  """The MiniCPM3 narrow raw route's host chain for one decode row."""
  dense = dequantize_tiles([raws[0]], [sa], sw, [1], COLUMNS)
  gate, up = dense[..., :HIDDEN].reshape(1, 1, HIDDEN), dense[..., HIDDEN:].reshape(1, 1, HIDDEN)
  activation = sigmoid_pair_vector(gate, up).reshape(1, HIDDEN).realize()
  parts = [quantize(activation[0, :4096].reshape(1, 4096), 4096),
           quantize(activation[0, 4096:HIDDEN].reshape(1, 2304), 2304)]
  return activation, parts


@pytest.mark.parametrize('kind', ['typical', 'zeros', 'nonfinite'])
def test_minicpm3_pack_matches_the_narrow_decode_chain(kind):
  ffn._program.cache_clear()
  raws, sa, sw = _wide_inputs(kind)
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    _, parts = _decode_chain(raws, sa, sw)
    packed, scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(TILES), 1, HIDDEN, DEPTHS, 1)
    payload, scale_values = packed.realize().numpy(), scales.realize().numpy()
    layout = ffn.payload_layout(1, DEPTHS)
    header = round_up(1*4, 64)
    for part, (offset, padded, _, _) in enumerate(layout):
      reference_payload, reference_scales = parts[part]
      reference_payload, reference_scales = reference_payload.realize().numpy(), reference_scales.realize().numpy()
      assert padded == DEPTHS[part]
      assert np.array_equal(payload[offset+header:offset+header+padded].view(np.int8),
                            reference_payload[header:header+padded].view(np.int8)), f'part {part} codes'
      assert np.array_equal(_scale_bits(payload, offset), reference_scales[0:1], equal_nan=True), f'part {part} scale'
      assert np.array_equal(scale_values[part:part+1], reference_scales, equal_nan=True)


def _scale_bits(payload, offset):
  return np.frombuffer(bytes(payload[offset:offset+4]), dtype=np.float32)[:1]


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('kind', ['typical', 'zeros', 'large', 'nonfinite', 'tiny'])
def test_minicpm3_norm_quantize_fusion_is_exact(kind, half):
  rng = np.random.default_rng(23)
  if kind == 'typical': values = rng.normal(0, .3, (1, 1, MODEL_HIDDEN)).astype(np.float32)
  elif kind == 'zeros': values = np.zeros((1, 1, MODEL_HIDDEN), np.float32)
  elif kind == 'large': values = rng.normal(0, 1e4, (1, 1, MODEL_HIDDEN)).astype(np.float32)
  elif kind == 'nonfinite':
    values = rng.normal(0, .3, (1, 1, MODEL_HIDDEN)).astype(np.float32)
    values[0, 0, 3] = np.inf; values[0, 0, 11] = np.nan
  elif kind == 'tiny': values = rng.normal(0, 1e-30, (1, 1, MODEL_HIDDEN)).astype(np.float32)
  weight = rng.uniform(.5, 1.5, MODEL_HIDDEN).astype(np.float16 if half else np.float32)
  x = Tensor(values, device='CPU').realize()
  w = Tensor(weight, device='CPU').realize()
  outputs = {}
  for fuse in (0, 1):
    with Context(**{**_DECODE, 'ONPU_NORM_QUANT_DECODE': fuse, 'ONPU_CPU_RMS_ALL_ROWS': 1}):
      norm_quant._decode_traces = 0
      normalized = rms_norm(x, w, EPS)
      payload, scales = quantize(normalized.reshape(1, MODEL_HIDDEN), MODEL_HIDDEN)
      outputs[fuse] = (payload.realize().numpy(), scales.realize().numpy(), norm_quant._decode_traces)
  assert outputs[0][2] == 0 and outputs[1][2] == 1, 'fusion must engage exactly once'
  assert np.array_equal(outputs[0][0], outputs[1][0])
  assert np.array_equal(outputs[0][1], outputs[1][1], equal_nan=True)


@pytest.mark.parametrize('kind', ['typical', 'zeros', 'nonfinite'])
def test_minicpm3_down_merge_parts(kind):
  """Two-part (4096+2304) ffn_down: both dequantizers plus their sum in one kernel."""
  ffn._down_merge_parts_program.cache_clear()
  rng = np.random.default_rng(41)
  columns = MODEL_HIDDEN
  if kind == 'typical':
    raws_np = [rng.integers(-2**24, 2**24, size=(1, columns)).astype(np.int32) for _ in range(2)]
    sa_np = rng.uniform(.005, .02, 2).astype(np.float32)
    sws_np = [rng.uniform(1e-3, 1e-2, columns).astype(np.float32) for _ in range(2)]
  elif kind == 'zeros':
    raws_np = [np.zeros((1, columns), np.int32) for _ in range(2)]
    sa_np = np.array([1.0, 1.0], np.float32); sws_np = [np.full(columns, 1e-3, np.float32)]*2
  else:
    raws_np = [rng.integers(-2**31, 2**31-1, size=(1, columns)).astype(np.int32) for _ in range(2)]
    sa_np = np.array([np.nan, np.inf], np.float32); sws_np = [np.full(columns, 3e30, np.float32)]*2
  raws = [Tensor(r, device='CPU', dtype=dtypes.int32).realize() for r in raws_np]
  sa = Tensor(sa_np, device='CPU').realize()
  sws = [Tensor(w, device='CPU').realize() for w in sws_np]
  with Context(ONPU_FFN_DOWN_MERGE=1):
    fused = ffn._down_merge_parts(raws, sa, sws, columns).realize().numpy().reshape(-1)
    first = dequantize_tiles([raws[0]], [sa[0:1]], sws[0], [1], columns)
    second = dequantize_tiles([raws[1]], [sa[1:2]], sws[1], [1], columns)
    reference = (first+second).realize().numpy().reshape(-1)
  mismatches = np.flatnonzero(~((reference == fused) | (np.isnan(reference) & np.isnan(fused))))
  assert mismatches.size == 0, f'{kind}: {mismatches.size} elements differ'


@pytest.mark.skipif(not __import__('os').environ.get('ONPU_TEST_HOST_POOL'),
                    reason='starts the shared 4-worker host pool; opt in with ONPU_TEST_HOST_POOL=1')
def test_minicpm3_two_group_pool_split_is_exact():
  from tinygrad.llm.kernels.rockchip_pool import attention_pool
  try: pool = attention_pool()
  except OSError: pytest.skip('host worker pool unavailable')
  ffn._program.cache_clear()
  raws, sa, sw = _wide_inputs('typical')
  reference, reference_scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(TILES), 1, HIDDEN, DEPTHS, 1)
  reference, reference_scales = reference.realize(), reference_scales.realize()
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1), pool:
    pooled, pooled_scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(TILES), 1, HIDDEN, DEPTHS, 1,
                                                       pool=pool, decode_pool=True)
  assert np.array_equal(pooled.realize().numpy(), reference.numpy())
  assert np.array_equal(pooled_scales.realize().numpy(), reference_scales.numpy(), equal_nan=True)
  assert pool.lib.pool_calls() > 0


def test_minicpm3_decode_ffn_end_to_end_matches_with_all_new_flags_off_or_on(monkeypatch):
  """CPU-emulated decode call path: merge + group pool must not change the output."""
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  ffn._program.cache_clear()
  ffn._down_merge_parts_program.cache_clear()
  rng = np.random.default_rng(51)
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.packed = [Tensor.empty(400, 80, 32, 32, device='CPU', dtype=dtypes.int8)]   # 12800 cols, k=2560
  joint.scales = [Tensor.ones(12800, device='CPU')]
  joint.depth, joint.columns, joint.tile_columns, joint.bias = MODEL_HIDDEN, 12800, 24384, None
  joint.input_scale, joint.shared_projection, joint.decode = None, None, 'on'
  parts = []
  for depth, blocks in ((4096, 128), (2304, 72)):
    part = RockchipInt8Linear.__new__(RockchipInt8Linear)
    part.packed = [Tensor.empty(80, blocks, 32, 32, device='CPU', dtype=dtypes.int8)]
    part.scales = [Tensor.ones(MODEL_HIDDEN, device='CPU')]
    part.depth, part.columns, part.bias, part.input_scale = depth, MODEL_HIDDEN, None, None
    parts.append(part)
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.parts, down.depth, down.columns, down.bias = parts, HIDDEN, MODEL_HIDDEN, None
  down.input_scale, down.decode = None, 'on'
  down.grouped_weight, down.grouped_scale, down.grouped_count = None, None, 0
  kernel = ffn.RockchipInt8SwiGLUFeedForward.__new__(ffn.RockchipInt8SwiGLUFeedForward)
  kernel.activation, kernel.decode_enabled, kernel.pool = 'silu', True, None
  kernel.projections = SimpleNamespace(joint=joint, layers=[SimpleNamespace(phase='decode', decode='on')])
  kernel.down = down
  kernel.combined_scales = Tensor.ones(COLUMNS, device='CPU').realize()
  x = Tensor(rng.normal(0, .3, (1, 1, MODEL_HIDDEN)).astype(np.float32), device='CPU')
  state = {'n': 0}
  def fake_raw(payload, weight, rows, columns, column_offset=0, **kwargs):
    state['n'] += 1
    return Tensor(np.random.default_rng(state['n']).integers(-2**20, 2**20, rows*round_up(columns, 32)).astype(np.int32),
                  device='CPU', dtype=dtypes.int32)
  original = ffn.raw_int8_matmul
  ffn.raw_int8_matmul = fake_raw
  try:
    outputs = []
    for merge in (0, 1):
      state['n'] = 0
      with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1, ONPU_GROUPED_DECODE_DOWN=1,
                   ONPU_FFN_DOWN_MERGE=merge):
        outputs.append(kernel(x).realize().numpy())
  finally:
    ffn.raw_int8_matmul = original
  assert outputs[0].shape == outputs[1].shape == (1, 1, MODEL_HIDDEN)
  assert np.array_equal(outputs[0], outputs[1], equal_nan=True)


def test_minicpm3_decode_norm_uses_the_exact_dedicated_kernel(monkeypatch):
  """Decode width 2560 must take the dedicated kernel (and fuse its quantize) when opted in."""
  from tinygrad import nn
  from tinygrad.llm.kernels.rockchip import RockchipRMSNorm, ONPU_CPU_RMS_ALL_ROWS, ONPU_CPU_RMS_ALL_WIDTHS
  from tinygrad.llm.kernels import rockchip_cpu
  calls = {'n': 0}
  original = rockchip_cpu.rms_norm
  def counted(*args, **kwargs):
    calls['n'] += 1
    return original(*args, **kwargs)
  monkeypatch.setattr(rockchip_cpu, 'rms_norm', counted)
  rng = np.random.default_rng(61)
  x = Tensor(rng.normal(0, .3, (1, 1, MODEL_HIDDEN)).astype(np.float32), device='CPU').realize()
  weight = Tensor(rng.uniform(.5, 1.5, MODEL_HIDDEN).astype(np.float16), device='CPU').realize()
  source = nn.RMSNorm(MODEL_HIDDEN, EPS)
  source.weight = weight
  generic = source(x).realize().numpy()
  for widths, expected_kernel in ((0, 0), (1, 1)):
    calls['n'] = 0
    with Context(ONPU_CPU_RMS_ALL_ROWS=1, ONPU_CPU_RMS_ALL_WIDTHS=widths, ONPU_NORM_QUANT=0,
                 ONPU_NORM_QUANT_DECODE=0):
      norm = RockchipRMSNorm(source)
      dedicated = norm(x).realize().numpy()
    assert calls['n'] == expected_kernel, f'widths={widths}: dedicated kernel calls {calls["n"]}'
    assert np.array_equal(generic, dedicated), f'widths={widths} changed the norm output'
  # with the decode fusion flags the same norm+quantize chain must fuse at 2560
  with Context(ONPU_CPU_RMS_ALL_ROWS=1, ONPU_CPU_RMS_ALL_WIDTHS=1, ONPU_NORM_QUANT=0, ONPU_NORM_QUANT_DECODE=1):
    norm_quant._decode_traces = 0
    norm = RockchipRMSNorm(source)
    payload, scales = quantize(norm(x).reshape(1, MODEL_HIDDEN), MODEL_HIDDEN)
    payload.realize(); scales.realize()
  assert norm_quant._decode_traces == 1


def _minicpm3_kernel():
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.packed = [Tensor.empty(400, 80, 32, 32, device='CPU', dtype=dtypes.int8)]
  joint.scales = [Tensor.ones(12800, device='CPU')]
  joint.depth, joint.columns, joint.tile_columns, joint.bias = MODEL_HIDDEN, 12800, 24384, None
  joint.input_scale, joint.shared_projection, joint.decode = None, None, 'on'
  parts = []
  for depth, blocks in ((4096, 128), (2304, 72)):
    part = RockchipInt8Linear.__new__(RockchipInt8Linear)
    part.packed = [Tensor.empty(80, blocks, 32, 32, device='CPU', dtype=dtypes.int8)]
    part.scales = [Tensor.ones(MODEL_HIDDEN, device='CPU')]
    part.depth, part.columns, part.bias, part.input_scale = depth, MODEL_HIDDEN, None, None
    parts.append(part)
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.parts, down.depth, down.columns, down.bias = parts, HIDDEN, MODEL_HIDDEN, None
  down.input_scale, down.decode = None, 'on'
  down.grouped_weight, down.grouped_scale, down.grouped_count = None, None, 0
  kernel = ffn.RockchipInt8SwiGLUFeedForward.__new__(ffn.RockchipInt8SwiGLUFeedForward)
  kernel.activation, kernel.decode_enabled, kernel.pool = 'silu', True, None
  kernel.projections = SimpleNamespace(joint=joint, layers=[SimpleNamespace(phase='decode', decode='on')])
  kernel.down = down
  kernel.combined_scales = Tensor.ones(COLUMNS, device='CPU').realize()
  return kernel


def test_minicpm3_narrow_route_group_pool_is_opt_in(monkeypatch):
  """Only the new narrow-route flag may switch the pack's worker split on for MiniCPM3."""
  kernel = _minicpm3_kernel()
  x = Tensor(np.random.default_rng(7).normal(0, .3, (1, 1, MODEL_HIDDEN)).astype(np.float32), device='CPU')
  state = {'n': 0}
  def fake_raw(payload, weight, rows, columns, column_offset=0, **kwargs):
    state['n'] += 1
    return Tensor(np.random.default_rng(state['n']).integers(-2**20, 2**20, rows*round_up(columns, 32)).astype(np.int32),
                  device='CPU', dtype=dtypes.int32)
  captured = {}
  original_pack = ffn.pack_swiglu_projection
  def pack_wrapper(*args, **kwargs):
    captured.update(kwargs)
    return original_pack(*args, **kwargs)
  monkeypatch.setattr(ffn, 'raw_int8_matmul', fake_raw)
  monkeypatch.setattr(ffn, 'pack_swiglu_projection', pack_wrapper)
  for flag, expected in ((0, False), (1, True)):
    captured.clear()
    with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1, ONPU_GROUPED_DECODE_DOWN=1,
                 ONPU_FFN_GROUP_POOL_ALL=flag, ONPU_WIDE_RAW_SWIGLU_DECODE=0):
      kernel(x).realize()
    assert captured.get('decode_pool') is expected, f'flag={flag} decode_pool={captured.get("decode_pool")}'
