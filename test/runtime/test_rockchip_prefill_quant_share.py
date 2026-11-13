"""MiniCPM3 prefill: share one quantize between the MLA q_a and kv_a projections.

`ONPU_PREFILL_QUANT_SHARE=1` makes the second INT8 projection fed the SAME activation tensor
in the same forward reuse the first one's quantized (payload, scales) instead of quantizing
identical bytes again (`MLATransformerBlock._attention` calls `attn_q_a(x)` and
`attn_kv_a_mqa(x)` on the same normalized hidden, both k=2560, so every 64-row tile was
quantized and uploaded twice - 2 redundant launches per block x 62 blocks).

Bit-exactness is structural: the second consumer receives the *same* payload tensor (same
buffer, no recompute), so its bytes and scales cannot differ. These tests prove that with a
byte-level comparison (numpy views, uint32 output equality) and prove the flag-off graph is
unchanged (same number of quantize launches, same outputs).
"""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip_int8
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, quantize_share_counts, clear_quantize_share

DEPTH = 2560                     # k of both MLA projections (embedding_length)
DQ, DKV = 1536, 576              # q_lora_rank, kv_lora_rank + rope_dim
ROWS = 128
_FLAG = dict(ONPU_PREFILL_QUANT_SHARE=1)
_OFF = dict(ONPU_PREFILL_QUANT_SHARE=0)


def _projection(columns:int, depth:int=DEPTH) -> RockchipInt8Linear:
  """A prepared INT8 linear with a CPU-packed (fake) weight tile, like the real prefill path."""
  layer = RockchipInt8Linear.__new__(RockchipInt8Linear)
  layer.packed = [Tensor.empty(columns, depth//32, 32, 32, device='CPU', dtype=dtypes.int8)]
  layer.scales = [Tensor.ones(columns, device='CPU').realize()]
  layer.depth, layer.columns, layer.bias, layer.input_scale = depth, columns, None, None
  layer.k_parts, layer.decode_packed, layer.decode_scales = [], [], []
  layer._grouped_chunks, layer._grouped_first = [], 0
  layer.tile_columns = layer.decode_tile_columns = columns
  layer.phase, layer.shared_projection, layer._prepared = 'prefill', None, True
  return layer


def _bytes(t:Tensor) -> np.uint8:
  return t.realize().numpy().view(np.uint8)


def _run(monkeypatch, flag:int, rows:int=ROWS):
  """Run q_a then kv_a on the same activation; report the quantize count and the outputs."""
  q_a, kv_a = _projection(DQ), _projection(DKV)
  x = Tensor(np.random.default_rng(3).normal(0, .3, (1, rows, DEPTH)).astype(np.float32), device='CPU').realize()
  calls, payloads = [], []
  real_quantize = rockchip_int8.quantize
  def counting_quantize(t, padded, **kwargs):
    calls.append(1)
    return real_quantize(t, padded, **kwargs)
  def fake_raw(payload, weight, rows_, columns, column_offset=0, **kwargs):
    # the raw output is a pure function of the payload bytes, so any payload difference shows up
    # in the final (real) dequantized output
    full = payload.realize().numpy().tobytes()
    data = np.frombuffer(full, dtype=np.uint8)[:rows_*columns]
    payloads.append((payload, np.frombuffer(full, dtype=np.uint8).copy(), rows_, columns))
    return Tensor(np.repeat(data.astype(np.int32), 1)[:rows_*columns].reshape(rows_, columns),
                  device='CPU', dtype=dtypes.int32).realize()
  def fake_packed(x, weight, scales, columns, column_offset=0):
    # the one-row (decode) path; it consumes the quantized payload the same way
    payloads.append((x, np.frombuffer(x.realize().numpy().tobytes(), dtype=np.uint8).copy(),
                     x.shape[0], columns))
    return Tensor(np.zeros((x.shape[0], columns), np.int32), device='CPU', dtype=dtypes.int32).realize()
  monkeypatch.setattr(rockchip_int8, 'quantize', counting_quantize)
  monkeypatch.setattr(rockchip_int8, 'raw_int8_matmul', fake_raw)
  monkeypatch.setattr(rockchip_int8, 'packed_int8_matmul', fake_packed)
  with Context(ONPU_PREFILL_QUANT_SHARE=flag):
    clear_quantize_share()
    q = q_a.npu(x)
    kv = kv_a.npu(x)
    out = (q.sum(-1) + kv.sum(-1)).realize()
    counts = quantize_share_counts()
  return dict(q=q.realize(), kv=kv.realize(), out=out, calls=len(calls), payloads=payloads, counts=counts)


def test_share_quantizes_once_and_hands_the_same_payload_to_both_projections(monkeypatch):
  off = _run(monkeypatch, 0)
  on = _run(monkeypatch, 1)
  # flag off: one quantize per 64-row tile per projection = 2 tiles x 2 projections
  assert off['calls'] == 4, off['calls']
  # flag on: the two projections share each tile's quantize
  assert on['calls'] == 2, on['calls']
  assert on['counts'].get('payload_hit') == 2, on['counts']
  # both projections consumed the SAME payload object (and therefore the same bytes/scales).
  # launch order is [q_a tile0, q_a tile1, kv_a tile0, kv_a tile1]: tiles pair up across the two.
  assert [rows_ for _, _, rows_, _ in on['payloads']] == [64, 64, 64, 64]
  for first, second in ((0,2), (1,3)):
    payload_a, bytes_a, _, columns_a = on['payloads'][first]
    payload_b, bytes_b, _, columns_b = on['payloads'][second]
    assert columns_a != columns_b, 'the two consumers must be the two different projections'
    assert payload_a.uop is payload_b.uop, 'payload must be the same tensor, not a recompute'
    assert np.array_equal(bytes_a, bytes_b), 'payload bytes differ'
  # and the projections still produce the previous values, bit for bit
  for key in ('q', 'kv', 'out'):
    assert np.array_equal(on[key].numpy().view(np.uint32), off[key].numpy().view(np.uint32)), key


def test_share_is_per_forward_and_never_leaks_across_forwards(monkeypatch):
  q_a = _projection(DKV)
  x = Tensor(np.random.default_rng(5).normal(0, .3, (1, ROWS, DEPTH)).astype(np.float32), device='CPU').realize()
  calls = []
  real_quantize = rockchip_int8.quantize
  def counting_quantize(t, padded, **kwargs):
    calls.append(1)
    return real_quantize(t, padded, **kwargs)
  def fake_raw(payload, weight, rows_, columns, column_offset=0, **kwargs):
    return Tensor(np.zeros(rows_*columns, np.int32), device='CPU', dtype=dtypes.int32)
  def fake_packed(x, weight, scales, columns, column_offset=0):
    # the one-row (decode) path; it consumes the quantized payload the same way
    payloads.append((x, np.frombuffer(x.realize().numpy().tobytes(), dtype=np.uint8).copy(),
                     x.shape[0], columns))
    return Tensor(np.zeros((x.shape[0], columns), np.int32), device='CPU', dtype=dtypes.int32).realize()
  monkeypatch.setattr(rockchip_int8, 'quantize', counting_quantize)
  monkeypatch.setattr(rockchip_int8, 'raw_int8_matmul', fake_raw)
  monkeypatch.setattr(rockchip_int8, 'packed_int8_matmul', fake_packed)
  with Context(ONPU_PREFILL_QUANT_SHARE=1):
    clear_quantize_share()
    q_a.npu(x)
    first = len(calls)
    clear_quantize_share()      # the per-forward hook in rockchip.py does exactly this
    q_a.npu(x)
    assert len(calls) == 2*first, 'a cleared cache must re-quantize, never reuse a stale payload'


def test_share_leaves_decode_and_flag_off_untouched(monkeypatch):
  # decode (one row) cannot share: the fold is prefill-only
  on = _run(monkeypatch, 1, rows=1)
  assert on['counts'].get('payload_hit') is None, on['counts']
  off = _run(monkeypatch, 0, rows=1)
  assert np.array_equal(on['out'].numpy().view(np.uint32), off['out'].numpy().view(np.uint32))
  # flag off never populates the cache at all
  assert off['counts'] == {}, off['counts']


def test_flag_off_is_byte_identical_for_prefill(monkeypatch):
  off = _run(monkeypatch, 0)
  on = _run(monkeypatch, 1)
  for key in ('q', 'kv', 'out'):
    assert np.array_equal(on[key].numpy().view(np.uint32), off[key].numpy().view(np.uint32)), key
  # the payload bytes handed to the matmul are identical in both modes
  off_bytes = [data for _, data, _, _ in off['payloads']]
  on_bytes = [data for _, data, _, _ in on['payloads']]
  assert len(off_bytes) == 4 and len(on_bytes) == 4
  for a, b in zip(off_bytes, on_bytes):
    assert np.array_equal(a, b), 'shared payload bytes must equal the independently quantized ones'
