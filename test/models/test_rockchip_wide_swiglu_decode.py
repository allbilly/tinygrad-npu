"""Bit-exactness of the wide (24576+2816) raw SwiGLU decode route.

The decode FFN currently materializes gate/up FP32 from two raw INT32 tiles,
copies each half out, fuses them with a standalone gated-activation kernel, then
re-quantizes the 13696-wide result in four independent K groups for ffn_down.
`ONPU_WIDE_RAW_SWIGLU_DECODE` replaces that chain with the existing prefill pack
kernel, which dequantizes and activates in registers and writes the down payload
directly.

Every assertion here is bit-exact (`np.array_equal`), including the activation
scale bits, because the activation scales come out of one FP32 max reduction.
The route must therefore reproduce:
  * `(float)raw * activation_scale * weight_scale` (that multiply order),
  * `(g * (1/(exp2(g*-log2e)+1))) * u` (reciprocal form, not a division),
  * the per-K-group FP32 range reduction,
  * `largest/127` with the `1e-8` floor and nearest-even conversion with clamp.
Both the fast-exp2 pragma and its fallback range are covered, as are non-finite
and clamp-boundary inputs.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, round_up
from tinygrad.llm.kernels import rockchip_gelu_ffn as ffn
from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles, quantize
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_pair_vector
from tinygrad.runtime.support.rockchip_int8 import padded_depth

HIDDEN = 13696          # ChatGLM3-6B ffn hidden width
COLUMNS = 2*HIDDEN      # joint gate/up projection width
DEPTHS = (4096, 4096, 4096, 1408)   # ffn_down K groups used by the split down projection
WIDE_TILES = (ffn.ProjectionTile(0, 1, 0, 24576, 0), ffn.ProjectionTile(0, 1, 24576, 2816, 0))


def _wide_inputs(kind, seed=1234):
  rng = np.random.default_rng(seed)
  if kind == 'typical':
    raw = [rng.integers(-2**20, 2**20, size=(1, 24576)).astype(np.int32),
           rng.integers(-2**20, 2**20, size=(1, 2816)).astype(np.int32)]
    sa, sw = np.array([0.0137], np.float32), rng.uniform(1e-3, 1e-2, COLUMNS).astype(np.float32)
  elif kind == 'huge':    # |g| > 200 so exp2 takes its fallback branch in both routes
    raw = [rng.integers(-2**30, 2**30, size=(1, 24576)).astype(np.int32),
           rng.integers(-2**30, 2**30, size=(1, 2816)).astype(np.int32)]
    sa, sw = np.array([0.5], np.float32), rng.uniform(.4, .6, COLUMNS).astype(np.float32)
  elif kind == 'zeros':   # zero range floor (scale clamps to 1e-8)
    raw = [np.zeros((1, 24576), np.int32), np.zeros((1, 2816), np.int32)]
    sa, sw = np.array([1.0], np.float32), np.full(COLUMNS, 1e-3, np.float32)
  elif kind == 'nonfinite':  # FP32 overflow in the dequantize -> invalid row -> NaN scale
    raw = [rng.integers(-2**31, 2**31-1, size=(1, 24576)).astype(np.int32),
           rng.integers(-2**31, 2**31-1, size=(1, 2816)).astype(np.int32)]
    sa, sw = np.array([3e30], np.float32), np.full(COLUMNS, 3e30, np.float32)
  elif kind == 'clamp':
    raw = [rng.integers(-127, 128, size=(1, 24576)).astype(np.int32),
           rng.integers(-127, 128, size=(1, 2816)).astype(np.int32)]
    sa, sw = np.array([1.0], np.float32), np.full(COLUMNS, np.float32(1/127.0)*np.float32(1.0000001), np.float32)
  else: raise ValueError(kind)
  return ([Tensor(r, device='CPU', dtype=dtypes.int32).realize() for r in raw],
          Tensor(sa, device='CPU', dtype=dtypes.float).realize(),
          Tensor(sw, device='CPU', dtype=dtypes.float).realize())


def _decode_chain(raws, sa, sw):
  """The exact kernel sequence `RockchipInt8FeedForward` + `RockchipInt8SplitLinear` run at decode."""
  parts = [dequantize_tiles([raws[0]], [sa], sw[:24576], [1], 24576),
           dequantize_tiles([raws[1]], [sa], sw[24576:], [1], 2816)]
  dense = Tensor.cat(*parts, dim=1)                                    # (1, COLUMNS) FP32
  gate, up = dense[..., :HIDDEN].reshape(1, 1, HIDDEN), dense[..., HIDDEN:].reshape(1, 1, HIDDEN)
  act = sigmoid_pair_vector(gate, up).reshape(1, HIDDEN).realize()     # standalone gated activation
  grouped = quantize(act[0, :12288].reshape(3, 4096), 4096)            # grouped down input
  tail = quantize(act[0, 12288:13696].reshape(1, 1408), padded_depth(1408))
  return act, grouped, tail


def _scale_bits(packed_bytes, offset):
  return np.array([np.frombuffer(bytes(packed_bytes[offset:offset+4]), dtype=np.float32)[0]], np.float32)


@pytest.mark.parametrize('fast', [1, 0])
@pytest.mark.parametrize('kind', ['typical', 'huge', 'zeros', 'nonfinite', 'clamp'])
def test_fused_wide_pack_matches_decode_chain_exactly(kind, fast):
  ffn._program.cache_clear()
  with Context(ONPU_EXP2_FAST_PATH=fast, ONPU_EXP2_SCALE_BITS=1):
    raws, sa, sw = _wide_inputs(kind)
    act, (grouped_payload, grouped_scales), (tail_payload, tail_scales) = _decode_chain(raws, sa, sw)
    packed, scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN, DEPTHS, 1)
    payload = packed.realize().numpy()
    scale_values = scales.realize().numpy()
    layout = ffn.payload_layout(1, DEPTHS)
    header = round_up(1*4, 64)
    grouped_header = round_up(3*4, 64)
    grouped_bytes = grouped_payload.realize().numpy()
    tail_bytes = tail_payload.realize().numpy()
    grouped_scale_values = grouped_scales.realize().numpy()
    tail_scale_values = tail_scales.realize().numpy()

    assert scale_values.shape == (len(DEPTHS),)
    for group, (offset, padded, _, size) in enumerate(layout):
      fused_codes = payload[offset+header:offset+header+padded].view(np.int8)
      assert np.array_equal(_scale_bits(payload, offset), scale_values[group:group+1], equal_nan=True)
      if group < 3:
        reference_codes = grouped_bytes[grouped_header+group*4096:grouped_header+(group+1)*4096].view(np.int8)
        assert np.array_equal(_scale_bits(payload, offset), grouped_scale_values[group:group+1], equal_nan=True)
      else:
        reference_codes = tail_bytes[header:header+padded].view(np.int8)
        assert np.array_equal(_scale_bits(payload, offset), tail_scale_values[0:1], equal_nan=True)
      assert np.array_equal(fused_codes, reference_codes), f'group {group} codes differ'
      # Padding bytes are zero in both routes.
      if padded > DEPTHS[group]:
        assert not payload[offset+header+DEPTHS[group]:offset+header+padded].any()


@pytest.mark.parametrize('fast', [1, 0])
def test_grouped_pack_layout_matches_the_assembled_grouped_payload(fast):
  """The grouped payload mode must equal what the legacy reassembly produced."""
  ffn._program.cache_clear()
  with Context(ONPU_EXP2_FAST_PATH=fast, ONPU_EXP2_SCALE_BITS=1):
    raws, sa, sw = _wide_inputs('typical')
    _, (grouped_payload, grouped_scales), (tail_payload, tail_scales) = _decode_chain(raws, sa, sw)
    grouped_payload, grouped_scales = grouped_payload.realize(), grouped_scales.realize()
    tail_payload, tail_scales = tail_payload.realize(), tail_scales.realize()
    packed, scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN, DEPTHS, 1, grouped=3)
    payload, scale_values = packed.realize().numpy(), scales.realize().numpy()
    layout = ffn.payload_layout(1, DEPTHS, 3)
    header = round_up(3*1*4, 64)
    end = layout[2][0]+4096
    assert end == header+3*4096
    assert np.array_equal(payload[header:end].view(np.int8), grouped_payload.numpy()[header:header+3*4096].view(np.int8))
    assert np.array_equal(scale_values[:3], grouped_scales.numpy(), equal_nan=True)
    # The tail keeps an ordinary per-row header after the grouped region.
    offset, padded, _, _ = layout[3]
    assert np.array_equal(payload[offset+header:offset+header+padded].view(np.int8),
                          tail_payload.numpy()[header:header+padded].view(np.int8))
    assert np.array_equal(_scale_bits(payload, offset), tail_scales.numpy(), equal_nan=True)
    # Direct slicing is the whole grouped payload: no reassembly kernels are needed.
    elements, element_scales = ffn.grouped_down_payload(packed.realize(), scales.realize(), 1, DEPTHS, 3,
                                                       packed_grouped=True)
    assert np.array_equal(elements.numpy(), grouped_payload.numpy())
    assert np.array_equal(element_scales.numpy(), grouped_scales.numpy())


def test_grouped_down_payload_reassembly_matches_direct_quantization():
  ffn._program.cache_clear()
  raws, sa, sw = _wide_inputs('typical')
  act, (grouped_payload, grouped_scales), _ = _decode_chain(raws, sa, sw)
  packed, scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN, DEPTHS, 1)
  assembled, assembled_scales = ffn.grouped_down_payload(packed.realize(), scales.realize(), 1, DEPTHS, 3)
  assert np.array_equal(assembled.realize().numpy(), grouped_payload.realize().numpy())
  assert np.array_equal(assembled_scales.realize().numpy(), grouped_scales.realize().numpy())


def _fake_wide_ffn(decode_enabled):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.packed = [Tensor.empty(768, 128, 32, 32, device='CPU', dtype=dtypes.int8),
                  Tensor.empty(88, 128, 32, 32, device='CPU', dtype=dtypes.int8)]
  joint.scales = [Tensor.ones(24576, device='CPU'), Tensor.ones(2816, device='CPU')]
  joint.bias, joint.columns = None, 27392
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8SplitLinear
  parts = [SimpleNamespace(depth=d, columns=4096, packed=[object()], scales=[object()], bias=None,
                           input_scale=None, decode='on') for d in DEPTHS]
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.depth, down.columns, down.bias, down.input_scale = HIDDEN, 4096, None, None
  down.parts, down.decode, down.phase = parts, 'on', None
  kernel = ffn.RockchipInt8SwiGLUFeedForward.__new__(ffn.RockchipInt8SwiGLUFeedForward)
  kernel.activation, kernel.decode_enabled, kernel.pool = 'silu', decode_enabled, None
  kernel.projections = SimpleNamespace(joint=joint, layers=[SimpleNamespace(phase='decode', decode='on')])
  kernel.down, kernel.combined_scales = down, Tensor.ones(COLUMNS, device='CPU')
  return kernel


def test_wide_decode_route_requires_the_opt_in(monkeypatch):
  monkeypatch.setattr(ffn, 'wide_swiglu_supported', lambda joint, down: True)
  x = Tensor.zeros(1, 1, 4096, device='CPU', dtype=dtypes.float)
  assert not _fake_wide_ffn(False).enabled(x)    # unchanged default: decode falls back to the FP32 chain
  assert _fake_wide_ffn(True).enabled(x)         # opt-in selects the raw pack route
  # Other activations must never take this route.
  kernel = _fake_wide_ffn(True)
  kernel.activation = 'gelu_tanh'
  assert not kernel.enabled(x)


@pytest.mark.skipif(not __import__('os').environ.get('ONPU_TEST_HOST_POOL'),
                    reason='starts the shared 4-worker host pool; opt in with ONPU_TEST_HOST_POOL=1')
def test_group_split_on_the_host_pool_is_bit_identical():
  """Four workers, one independent K group each, must reproduce the serial pack."""
  from tinygrad.llm.kernels.rockchip_pool import attention_pool
  try: pool = attention_pool()
  except OSError: pytest.skip('host worker pool unavailable')
  ffn._program.cache_clear()
  raws, sa, sw = _wide_inputs('typical')
  reference, reference_scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN,
                                                           DEPTHS, 1, grouped=3)
  reference, reference_scales = reference.realize(), reference_scales.realize()
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1), pool:
    assert pool.active
    pooled, pooled_scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN,
                                                       DEPTHS, 1, pool=pool, grouped=3, decode_pool=True)
  assert np.array_equal(pooled.realize().numpy(), reference.numpy())
  assert np.array_equal(pooled_scales.realize().numpy(), reference_scales.numpy(), equal_nan=True)
  assert pool.lib.pool_calls() > 0, 'the pack did not dispatch through the host pool'


def test_decode_call_path_consumes_the_packed_payload_without_reassembly():
  """`_RockchipInt8GatedFeedForward.__call__` at decode must hand the down matmuls
  exactly the pack slices (grouped header+codes, tail header+codes) and the matching
  activation scales. The int8 matmuls themselves are shared code and are faked out."""
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear, quantize
  from tinygrad.helpers import round_up
  ffn._program.cache_clear()
  captured_payloads, captured_grouped = [], []
  def fake_raw(payload, weight, rows, columns, column_offset=0, **kwargs):
    captured_payloads.append((payload.numpy().copy(), rows, columns, column_offset))
    return Tensor.ones(rows*round_up(columns, 32), device='CPU', dtype=dtypes.int32)
  def fake_grouped(payload, weight, rows, columns, cores=3):
    captured_grouped.append((payload.numpy().copy(), tuple(weight.shape), rows, columns))
    groups = weight.shape[0]
    return Tensor.ones(groups*rows*round_up(columns, 32), device='CPU', dtype=dtypes.int32).reshape(groups, rows, -1)
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.packed = [Tensor.empty(768, 128, 32, 32, device='CPU', dtype=dtypes.int8),
                  Tensor.empty(88, 128, 32, 32, device='CPU', dtype=dtypes.int8)]
  joint.scales = [Tensor.ones(24576, device='CPU'), Tensor.ones(2816, device='CPU')]
  joint.depth, joint.columns, joint.tile_columns, joint.bias = 4096, 27392, 24576, None
  joint.input_scale, joint.shared_projection, joint.decode = None, None, 'on'
  parts = []
  for depth, blocks in ((4096, 128), (4096, 128), (4096, 128), (1408, 48)):
    part = RockchipInt8Linear.__new__(RockchipInt8Linear)
    part.packed = [Tensor.empty(blocks, 128 if depth == 4096 else 48, 32, 32, device='CPU', dtype=dtypes.int8)]
    part.scales = [Tensor.ones(4096, device='CPU')]
    part.depth, part.columns, part.bias, part.input_scale = depth, 4096, None, None
    parts.append(part)
  down = RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
  down.parts, down.depth, down.columns, down.bias = parts, HIDDEN, 4096, None
  down.input_scale, down.decode = None, 'on'
  down.grouped_weight = Tensor.empty(3, 128, 128, 32, 32, device='CPU', dtype=dtypes.int8)
  down.grouped_scale = Tensor.ones(3, 4096, device='CPU')
  down.grouped_count = 3
  kernel = ffn.RockchipInt8SwiGLUFeedForward.__new__(ffn.RockchipInt8SwiGLUFeedForward)
  kernel.activation, kernel.decode_enabled, kernel.pool = 'silu', True, None
  kernel.projections = SimpleNamespace(joint=joint, layers=[SimpleNamespace(phase='decode', decode='on')])
  kernel.down = down
  kernel.combined_scales = Tensor.cat(joint.scales[0], joint.scales[1]).realize()
  x = Tensor(np.random.default_rng(11).normal(0, .3, (1, 1, 4096)).astype(np.float32), device='CPU')
  original_raw, original_grouped = ffn.raw_int8_matmul, ffn.grouped_raw_matmul
  ffn.raw_int8_matmul, ffn.grouped_raw_matmul = fake_raw, fake_grouped
  try:
    with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1, ONPU_GROUPED_DECODE_DOWN=1):
      out = kernel(x).realize()
  finally:
    ffn.raw_int8_matmul, ffn.grouped_raw_matmul = original_raw, original_grouped
  assert out.shape == (1, 1, 4096) and out.dtype is dtypes.float and np.isfinite(out.numpy()).all()
  # gate+up: one reused activation payload for both packed weight tiles.
  gate_payloads = [p for p, _, columns, _ in captured_payloads if columns in (24576, 2816)]
  assert len(gate_payloads) == 2 and np.array_equal(gate_payloads[0], gate_payloads[1])
  # rebuild what the pack must have produced from the same activation values
  dense = capacity_input(x).reshape(1, 4096)
  act_payload, act_scales = quantize(dense, 4096)
  assert np.array_equal(gate_payloads[0], act_payload.realize().numpy())
  tiles = [ffn.ProjectionTile(0, 1, 0, 24576, 0), ffn.ProjectionTile(0, 1, 24576, 2816, 0)]
  raws = [Tensor.ones(24576, device='CPU', dtype=dtypes.int32), Tensor.ones(2816, device='CPU', dtype=dtypes.int32)]
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    expected, expected_scales = ffn.pack_swiglu_projection(raws, [act_scales.realize()], kernel.combined_scales,
                                                          tiles, 1, HIDDEN, DEPTHS, 1, grouped=3)
  expected, expected_scales = expected.realize().numpy(), expected_scales.realize().numpy()
  layout = ffn.payload_layout(1, DEPTHS, 3)
  grouped_end = layout[2][0]+4096
  tail_offset = layout[3][0]
  tail_size = round_up(1*4, 64)+1536
  assert len(captured_grouped) == 1
  grouped_payload, grouped_shape, grouped_rows, grouped_columns = captured_grouped[0]
  assert grouped_shape == (3, 128, 128, 32, 32) and (grouped_rows, grouped_columns) == (1, 4096)
  assert np.array_equal(grouped_payload, expected[:grouped_end]), 'grouped payload is not the pack prefix'
  tail_payload = [p for p, _, columns, _ in captured_payloads if columns == 4096]
  assert len(tail_payload) == 1 and tail_payload[0].size == tail_size
  # The tail activation scale the pack wrote must be the slice the caller forwards.
  tail_scale_bits = np.frombuffer(bytes(expected[tail_offset:tail_offset+4]), dtype=np.float32)
  assert np.array_equal(tail_scale_bits, expected_scales[3:4], equal_nan=True)


def test_prefill_shape_pack_matches_the_per_group_quantization_chain():
  """The shared pack body must stay exact for the prefill row count too (n rows>1)."""
  ffn._program.cache_clear()
  rows = 16
  rng = np.random.default_rng(29)
  raw_np = [rng.integers(-2**20, 2**20, size=(rows, 24576)).astype(np.int32),
            rng.integers(-2**20, 2**20, size=(rows, 2816)).astype(np.int32)]
  raws = [Tensor(r, device='CPU', dtype=dtypes.int32).realize() for r in raw_np]
  sa = Tensor(np.full((rows,), 0.0137, np.float32), device='CPU').realize()
  sw = Tensor(rng.uniform(1e-3, 1e-2, COLUMNS).astype(np.float32), device='CPU').realize()
  tiles = [ffn.ProjectionTile(0, rows, 0, 24576, 0), ffn.ProjectionTile(0, rows, 24576, 2816, 0)]
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    parts = [dequantize_tiles([raws[0]], [sa], sw[:24576], [rows], 24576),
             dequantize_tiles([raws[1]], [sa], sw[24576:], [rows], 2816)]
    dense = Tensor.cat(*parts, dim=1)
    act = sigmoid_pair_vector(dense[..., :HIDDEN].reshape(1, rows, HIDDEN),
                              dense[..., HIDDEN:].reshape(1, rows, HIDDEN)).reshape(rows, HIDDEN).realize()
    reference = [quantize(act[:, :4096].contiguous(), 4096),
                 quantize(act[:, 4096:8192].contiguous(), 4096),
                 quantize(act[:, 8192:12288].contiguous(), 4096),
                 quantize(act[:, 12288:13696].contiguous(), padded_depth(1408))]
    packed, scales = ffn.pack_swiglu_projection(list(raws), [sa], sw, tiles, rows, HIDDEN, DEPTHS, rows)
    payload, scale_values = packed.realize().numpy(), scales.realize().numpy()
    layout = ffn.payload_layout(rows, DEPTHS)
    header = round_up(rows*4, 64)
    for group, (offset, padded, _, _) in enumerate(layout):
      payload_bytes, payload_scales = reference[group]
      payload_bytes, payload_scales = payload_bytes.realize().numpy(), payload_scales.realize().numpy()
      assert np.array_equal(payload[offset+header:offset+header+rows*padded].view(np.int8),
                            payload_bytes[header:header+rows*padded].view(np.int8)), f'group {group} codes'
      assert np.array_equal(scale_values[group*rows:(group+1)*rows],
                            payload_scales, equal_nan=True), f'group {group} scales'


def _capture_sources(monkeypatch, module, name, call):
  """Emit the C sources of every program built by `module.name` during `call`."""
  from tinygrad.uop.ops import Ops
  programs = []
  original = getattr(module, name)
  def wrapper(*args, **kwargs):
    program = original(*args, **kwargs)
    programs.append(program)
    return program
  monkeypatch.setattr(module, name, wrapper)
  call()
  return [next(u.arg for u in program.src if u.op is Ops.SOURCE) for program in programs]


def test_dequantize_and_activation_arithmetic_matches_the_standing_kernels(monkeypatch):
  """(a)/(b): the pack must spell the same operations in the same order as the
  kernels it replaces, not merely agree on rounded output."""
  import re
  from tinygrad.llm.kernels import rockchip_int8, rockchip_sigmoid
  raws, sa, sw = _wide_inputs('typical')
  ffn._program.cache_clear()
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    dequantize_src = _capture_sources(monkeypatch, rockchip_int8, '_dequantize_tiles',
                                      lambda: dequantize_tiles([raws[0]], [sa], sw[:24576], [1], 24576).realize())[0]
    packed_src = _capture_sources(monkeypatch, ffn, '_program',
                                  lambda: ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1,
                                                                    HIDDEN, DEPTHS, 1, grouped=3)[0].realize())[0]
    dense = Tensor.cat(dequantize_tiles([raws[0]], [sa], sw[:24576], [1], 24576),
                       dequantize_tiles([raws[1]], [sa], sw[24576:], [1], 2816), dim=1)
    gated_src = _capture_sources(monkeypatch, rockchip_sigmoid, '_program_pair_vector',
                                 lambda: sigmoid_pair_vector(dense[..., :HIDDEN].reshape(1, 1, HIDDEN),
                                                             dense[..., HIDDEN:].reshape(1, 1, HIDDEN)).realize())[0]
  flat = re.sub(r'\s+', ' ', packed_src)
  # (a) (float)raw * activation_scale, then * weight_scale, in that order.
  assert '(value*(float4)a)*(*(const float4*)(scales+c))' in dequantize_src
  # emitted both by project() and by the static per-fragment loop, so this one
  # substring pins the multiply order used on every loaded raw element
  assert re.search(r'\*\(float4\)sa\[\d\]\[row-\d+\]\)\*\(\*\(const float4\*\)\(sw\+', flat), \
      'pack dequant order differs'
  assert 'return (value*(float4)sa[0][row-0])*(*(const float4*)(sw+column));' in flat
  # (b) sigmoid = 1/(1+exp2(g*-log2e)), then (g*sigmoid)*up.
  assert 'float4 sig=rockchip_sigmoid_vector(gv);' in gated_src and 'gv*sig*uv' in gated_src
  assert '1.4426950216293335f' in flat and 'rockchip_exp2_vector' in flat
  assert 'float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(' in flat and '(g*sigmoid)*u' in flat
  assert '1.4426950216293335f' in gated_src and '1.0f/(rockchip_exp2_vector(x)+1.0f)' in gated_src


def test_pack_quantization_matches_the_standing_quantize_source(monkeypatch):
  """(c): per-element range reduction and rounding must be the same source text."""
  import re
  from tinygrad.llm.kernels import rockchip_int8
  raws, sa, sw = _wide_inputs('typical')
  ffn._program.cache_clear()
  with Context(ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    quantize_src = _capture_sources(monkeypatch, rockchip_int8, '_quantize',
                                    lambda: quantize(Tensor.ones(1, 4096, device='CPU', dtype=dtypes.float), 4096)[0]
                                    .realize())[0]
    pack_src = _capture_sources(monkeypatch, ffn, '_program',
                                lambda: ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1,
                                                                  HIDDEN, DEPTHS, 1, grouped=3)[0].realize())[0]
  shared = ['float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];',
            'int invalid=__builtin_reduce_or(bad)!=0;',
            'float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;',
            'invalid?__builtin_nanf(""):scale',
            'float inverse=1.0f/scale;',
            'rockchip_quantize_vector(value,inverse)',
            'bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000']
  for fragment in shared:
    assert fragment in quantize_src, fragment
    assert fragment in pack_src, fragment
  # same float4 accumulator and the same horizontal fold before the division
  assert 'maximum=a>maximum?a:maximum' in quantize_src and 'maximum=a>maximum?a:maximum' in pack_src


def test_pooled_group_dispatch_is_opt_in_only(monkeypatch):
  """Default-off contract: with the new route flag off, no pool address may be baked
  into the pack program and the dispatch stays the serial one."""
  import re
  ffn._program.cache_clear()
  raws, sa, sw = _wide_inputs('typical')
  with Context(ONPU_WIDE_RAW_SWIGLU_DECODE=0, ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    source = _capture_sources(monkeypatch, ffn, '_program',
                              lambda: ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1,
                                                                HIDDEN, DEPTHS, 1, grouped=3)[0].realize())[0]
  flat = re.sub(r'\s+', ' ', source)
  assert 'pool_group_part' not in flat and 'GroupWork' not in flat
  assert 'part(out,raw,sa,sw,scales,*active,0,1);' in flat
  # Even with an active pool object, the default call must not emit the dispatch.
  class FakePool:
    active = True
    address = 0x1234
    prefill_address = 0x5678
  with Context(ONPU_WIDE_RAW_SWIGLU_DECODE=0, ONPU_EXP2_FAST_PATH=1, ONPU_EXP2_SCALE_BITS=1):
    source = _capture_sources(monkeypatch, ffn, '_program',
                              lambda: ffn.pack_swiglu_projection(list(raws), [sa], sw, list(WIDE_TILES), 1, HIDDEN,
                                                                DEPTHS, 1, grouped=3, pool=FakePool())[0].realize())[0]
  assert 'pool_group_part' not in re.sub(r'\s+', ' ', source)
