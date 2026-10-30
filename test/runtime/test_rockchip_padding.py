"""Safety tests for explicitly registered zero-tailed CPU tensor views."""

from __future__ import annotations
import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_padding import ONPU_ZERO_PAD_VIEWS, capacity_input, capacity_view_counts, mark_zero_tail
from tinygrad.llm.kernels.rockchip_cpu import rms_norm
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
  return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def test_zero_tail_recovery_preserves_token_major_and_head_major_capacity_layouts():
  rng = np.random.default_rng(91)
  rows = 17
  data = np.zeros((1, 128, 16, 8), np.float32)
  data[:, :rows] = rng.normal(0, 0.4, (1, rows, 16, 8)).astype(np.float32)
  full = Tensor(data, device="CPU").realize()
  row = UOp.variable("capacity_rows", 1, 128).bind(rows)
  logical = full[:, :row, :, :]
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    assert ONPU_ZERO_PAD_VIEWS.value == 1
    marked = mark_zero_tail(logical, full, "token")
    before = capacity_view_counts().get("token", 0)
    recovered = capacity_input(marked)
    assert recovered.shape == full.shape and _bits_equal(recovered.numpy(), data)
    assert capacity_view_counts().get("token", 0) == before + 1

    # Flatten only trailing feature axes, preserving the symbolic row axis.
    flat = logical.reshape(1, row, 128)
    flat_full = capacity_input(flat)
    assert flat_full.shape == (1, 128, 128)
    assert _bits_equal(flat_full.numpy(), data.reshape(1, 128, 128))

    # A token-major to head-major permutation carries the same registered capacity.
    head = logical.permute(0, 2, 1, 3)
    head_full = capacity_input(head)
    assert head_full.shape == (1, 16, 128, 8)
    assert _bits_equal(head_full.numpy(), data.transpose(0, 2, 1, 3))
    assert not head_full.numpy()[:, :, rows:, :].any()

    # Flattening the head and row axes would compact the inter-head gaps; it must not reuse.
    before = capacity_view_counts().get("token", 0)
    compact = head.reshape(1, 16 * row, 8)
    compact_full = capacity_input(compact)
    assert compact_full.shape == compact.max_shape
    assert capacity_view_counts().get("token", 0) == before


def test_unregistered_shrink_cast_static_and_wrong_capacity_are_not_reused():
  data = np.zeros((1, 128, 16, 8), np.float32)
  full = Tensor(data, device="CPU").realize()
  rows = UOp.variable("reject_rows", 1, 128).bind(17)
  logical = full[:, :rows, :, :]
  before = capacity_view_counts().get("reject", 0)
  with Context(ONPU_ZERO_PAD_VIEWS=0):
    mark_zero_tail(logical, full, "reject")
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    # A registration created while the feature is off is absent.
    capacity_input(logical)
    assert capacity_view_counts().get("reject", 0) == before
    # The allowed head-major transform is PERMUTE; an arbitrary crop is SHRINK.
    registered = mark_zero_tail(logical, full, "reject")
    capacity_input(registered.permute(0, 2, 1, 3))
    after_valid = capacity_view_counts().get("reject", 0)
    assert after_valid == before + 1
    capacity_input(registered[:, :, :8, :])
    assert capacity_view_counts().get("reject", 0) == after_valid
    capacity_input(registered.cast(dtypes.half))
    assert capacity_view_counts().get("reject", 0) == after_valid
    other_full = Tensor(data.copy(), device="CPU").realize()
    unregistered = other_full[:, :rows, :, :]
    assert unregistered.uop is not registered.uop
    capacity_input(unregistered)
    assert capacity_view_counts().get("reject", 0) == after_valid
    wrong_full = Tensor.zeros(1, 129, 16, 8, device="CPU", dtype=dtypes.float).realize()
    other = Tensor.zeros_like(full)[:, :rows, :, :]
    assert other.max_shape != wrong_full.shape
    mark_zero_tail(other, wrong_full, "reject-wrong-max")
    before_wrong = capacity_view_counts().get("reject-wrong-max", 0)
    capacity_input(other)
    assert capacity_view_counts().get("reject-wrong-max", 0) == before_wrong
    static = full[:, :17, :, :]
    mark_zero_tail(static, full, "reject-static")
    before_static = capacity_view_counts().get("reject-static", 0)
    capacity_input(static)
    assert capacity_view_counts().get("reject-static", 0) == before_static


@pytest.mark.parametrize("rows", (1, 17, 65, 128))
@pytest.mark.parametrize("gamma_dtype", (np.float16, np.float32))
def test_rmsnorm_then_quantize_reuses_capacity_without_changing_active_payload(rows, gamma_dtype):
  rng = np.random.default_rng(440 + rows)
  data = rng.normal(0, 0.8, (1, 128, 1024)).astype(np.float32)
  gamma = rng.uniform(0.5, 1.5, 1024).astype(gamma_dtype)
  gamma[3] = np.nan
  gamma[7] = np.inf
  x = Tensor(data, device="CPU").realize()
  w = Tensor(gamma, device="CPU").realize()
  row0 = UOp.variable(f"rows_off_{rows}", 1, 128).bind(rows)
  with Context(ONPU_ZERO_PAD_VIEWS=0):
    y0 = rms_norm(x[:, :row0, :], w, 1e-6)
    active0_view = y0[:, :rows, :]
    raw0, scale0 = quantize(active0_view.reshape(rows, 1024), 1024)
    Tensor.realize(active0_view, raw0, scale0)
    active0 = active0_view.numpy().copy()
    bytes0 = raw0.numpy()
    scales0 = scale0.numpy().view(np.uint32).copy()
  before = capacity_view_counts().get("rms_norm", 0)
  row1 = UOp.variable(f"rows_on_{rows}", 1, 128).bind(rows)
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    y1 = rms_norm(x[:, :row1, :], w, 1e-6)
    full = capacity_input(y1)
    active1_view = full[:, :rows, :]
    raw1, scale1 = quantize(full.reshape(128, 1024), 1024)
    Tensor.realize(full, active1_view, raw1, scale1)
    active1 = active1_view.numpy().copy()
    full_values = full.numpy()
    assert capacity_view_counts().get("rms_norm", 0) == before + 1
    assert full.shape == (1, 128, 1024) and not full_values[:, rows:, :].any()
    bytes1 = raw1.numpy()
    scales1 = scale1.numpy().view(np.uint32).copy()
  assert _bits_equal(active0, active1)
  h0 = (rows * 4 + 63) // 64 * 64
  h1 = (128 * 4 + 63) // 64 * 64
  codes0 = bytes0[h0:].view(np.int8).reshape(rows, 1024)
  codes1 = bytes1[h1:].view(np.int8).reshape(128, 1024)
  np.testing.assert_array_equal(codes0, codes1[:rows])
  np.testing.assert_array_equal(scales0, scales1[:rows])
  assert not codes1[rows:].any()


@pytest.mark.parametrize("rows", (1, 17, 65, 128))
@pytest.mark.parametrize("seed", (53, 79))
def test_qkv_cpu_padded_query_view_is_exact_and_preserves_cache(rows, seed):
  rng = np.random.default_rng(seed)
  heads, kv, dim, context, start = 16, 8, 128, 512, 7
  columns = (heads + 2 * kv) * dim
  projected_data = rng.normal(0, 0.8, (1, 128, columns)).astype(np.float32)
  qn = rng.uniform(0.5, 1.5, dim).astype(np.float16)
  kn = rng.uniform(0.5, 1.5, dim).astype(np.float32)
  freqs = rng.normal(0, 0.5, (context, dim)).astype(np.float32)
  cache_data = rng.normal(0, 0.2, (2, 1, kv, context, dim)).astype(np.float16)
  projected = Tensor(projected_data, device="CPU").realize()
  qwt = Tensor(qn, device="CPU").realize()
  kwt = Tensor(kn, device="CPU").realize()
  freq = Tensor(freqs, device="CPU").realize()
  row0 = UOp.variable(f"qkv_rows_off_{rows}", 1, 128).bind(rows)
  cache0 = Tensor(cache_data.copy(), device="CPU").realize()
  with Context(ONPU_ZERO_PAD_VIEWS=0):
    q0, k0 = prepare_qkv(projected[:, :row0, :], cache0, qwt, kwt, freq, start, heads, 1e-6)
    active0_view = q0[:, :, :rows, :]
    Tensor.realize(active0_view, k0)
    active0 = active0_view.numpy().copy()
    cached0 = k0.numpy().copy()
    before = capacity_view_counts().get("qkv", 0)
    reused0 = capacity_input(q0)
    assert capacity_view_counts().get("qkv", 0) == before
    assert reused0.max_shape == q0.max_shape
  row1 = UOp.variable(f"qkv_rows_on_{rows}", 1, 128).bind(rows)
  cache1 = Tensor(cache_data.copy(), device="CPU").realize()
  before = capacity_view_counts().get("qkv", 0)
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    q1, k1 = prepare_qkv(projected[:, :row1, :], cache1, qwt, kwt, freq, start, heads, 1e-6)
    full = capacity_input(q1)
    active1_view = full[:, :, :rows, :]
    Tensor.realize(full, active1_view, k1)
    active1 = active1_view.numpy().copy()
    cached1 = k1.numpy().copy()
    full_values = full.numpy()
    assert capacity_view_counts().get("qkv", 0) == before + 1
    assert full.shape == (1, heads, 128, dim) and not full_values[:, :, rows:, :].any()
  assert _bits_equal(active0, active1)
  assert _bits_equal(cached0, cached1)
  # Only the KV rows written for this call may differ from the original cache.
  original = cache_data.view(np.uint16)
  actual = cached1.view(np.uint16)
  written = np.zeros(cache_data.shape[:4], dtype=bool)
  written[:, :, :, start : start + rows] = True
  np.testing.assert_array_equal(actual[~written].reshape(-1), original[~written].reshape(-1))


def test_packed_int8_ffn_call_reuses_registered_rmsnorm_capacity_before_npu_matmul(monkeypatch):
  from types import SimpleNamespace
  from tinygrad.llm.kernels import rockchip_int8
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8FeedForward, RockchipInt8Linear

  rows, capacity, input_width, hidden, output_width = 17, 128, 1024, 3072, 1024
  rng = np.random.default_rng(808)
  data = rng.normal(0, 0.8, (1, capacity, input_width)).astype(np.float32)
  gamma = np.ones(input_width, np.float32)
  x = Tensor(data, device="CPU").realize()
  weight = Tensor(gamma, device="CPU").realize()
  # Independent static-shape RMSNorm oracle for the active prefix, padded with zeros.
  with Context(ONPU_ZERO_PAD_VIEWS=0):
    active = rms_norm(Tensor(data[:, :rows].copy(), device="CPU"), weight, 1e-6).realize().numpy()
  expected = np.zeros((1, capacity, input_width), dtype=np.float32)
  expected[:, :rows] = active

  def fake_int8_linear(input_width, columns):
    source = SimpleNamespace(weight=Tensor.zeros(columns, input_width, device="CPU", dtype=dtypes.float), bias=None)
    layer = RockchipInt8Linear(source, "on", "on")
    # Shapes are sufficient for the packed-path predicate and the first matmul's K query.
    packed_k = input_width // 32
    layer.packed = [Tensor.empty(1, packed_k, 32, 32, device="CPU", dtype=dtypes.int8)]
    layer.scales = [Tensor.ones(columns, device="CPU", dtype=dtypes.float).realize()]
    return layer

  joint = fake_int8_linear(input_width, hidden * 2)
  down = fake_int8_linear(hidden, output_width)
  projections = SimpleNamespace(joint=joint, layers=(joint,))
  ffn = RockchipInt8FeedForward(projections, down)
  assert ffn._packed_enabled(Tensor(data[:, :rows].copy(), device="CPU"))

  observed = {}
  old_capacity_input = rockchip_int8.capacity_input
  old_quantize = rockchip_int8.quantize

  def capture_capacity(value):
    result = old_capacity_input(value)
    observed["capacity_tensor"] = result
    observed["capacity_shape"] = result.shape
    return result

  def capture_quantize(value, padded, **kwargs):
    observed["quant_input_shape"] = value.shape
    observed["quant_input"] = value.numpy().copy()
    payload, scales = old_quantize(value, padded, **kwargs)
    Tensor.realize(payload, scales)
    observed["quant_payload"] = payload.numpy().copy()
    observed["quant_scales"] = scales.numpy().copy()
    return payload, scales

  def stop_before_matmul(payload, packed_weight, active_rows, columns, *args, **kwargs):
    observed["matmul_rows"] = active_rows
    observed["matmul_columns"] = columns
    observed["matmul_payload"] = payload.numpy().copy()
    raise RuntimeError("stop before any matmul device operation")

  monkeypatch.setattr(rockchip_int8, "capacity_input", capture_capacity)
  monkeypatch.setattr(rockchip_int8, "quantize", capture_quantize)
  monkeypatch.setattr(rockchip_int8, "raw_int8_matmul", stop_before_matmul)
  before = capacity_view_counts().get("rms_norm", 0)
  row = UOp.variable("packed_ffn_active_rows", 1, capacity).bind(rows)
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    normalized = rms_norm(x[:, :row, :], weight, 1e-6)
    with pytest.raises(RuntimeError, match="stop before any matmul"):
      ffn(normalized)
    assert capacity_view_counts().get("rms_norm", 0) == before + 1

  full = observed["capacity_tensor"].numpy()
  assert observed["capacity_shape"] == (1, capacity, input_width)
  np.testing.assert_array_equal(full.view(np.uint32), expected.view(np.uint32))
  assert observed["quant_input_shape"] == (capacity, input_width)
  np.testing.assert_array_equal(observed["quant_input"].view(np.uint32), expected.reshape(capacity, input_width).view(np.uint32))
  assert observed["matmul_rows"] == capacity and observed["matmul_columns"] == hidden * 2
  np.testing.assert_array_equal(observed["matmul_payload"], observed["quant_payload"])
  reference_payload, reference_scales = old_quantize(Tensor(expected.reshape(capacity, input_width), device="CPU"), input_width)
  Tensor.realize(reference_payload, reference_scales)
  np.testing.assert_array_equal(observed["quant_payload"], reference_payload.numpy())
  np.testing.assert_array_equal(observed["quant_scales"].view(np.uint32), reference_scales.numpy().view(np.uint32))
