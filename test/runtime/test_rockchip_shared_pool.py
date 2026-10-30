"""Standalone correctness and lifecycle tests for shared CPU workers in Rockchip kernel factories."""

from __future__ import annotations
import fcntl, os, platform
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, ceildiv, round_up
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels import rockchip_cpu, rockchip_int8, rockchip_qkv
from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_SHARED_HOST_POOL, SharedHostPool


def _needs_pool():
  if platform.machine().lower() not in ("aarch64", "arm64"):
    pytest.skip("shared Rockchip CPU workers require AArch64")
  if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
    pytest.skip("shared Rockchip CPU workers require CPU4-7")


@contextmanager
def _pool_test_lock():
  """pytest -n12 still runs only one CPU5-7 worker-pool test group at a time."""
  path = Path("/tmp") / f"tinygrad-shared-host-pool-{os.getuid()}.lock"
  with path.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


@pytest.fixture(scope="module")
def shared_pool():
  _needs_pool()
  with _pool_test_lock():
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool

    pool = AttentionPool()
    parked = pool.prefill_address
    state = SharedHostPool(pool)
    try:
      yield pool, state, parked
    finally:
      state.restore()
      pool.close()
      assert "ROCKCHIP" not in Device._opened_devices and "CL" not in Device._opened_devices


def _logical_codes(values):
  rows, columns = values.shape
  n = round_up(columns, 32)
  width = round_up(ceildiv(n, 3), 32)
  parts = [(start, min(width, n - start)) for start in range(0, n, width)]
  return np.concatenate([values[:, start : start + count].ravel() for start, count in parts]).astype(np.int32)


def _make_case(family: str, rows: int, seed: int, *, depth: int | None = None, nonfinite: bool = False):
  rng = np.random.default_rng(seed)
  if family == "rmsnorm":
    width = 1024
    x = rng.normal(0, 0.8, (1, rows, width)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, width).astype(np.float16)
    if nonfinite:
      x[0, 0, 3] = np.inf
    xt = Tensor(x, device="CPU").realize()
    gt = Tensor(gamma, device="CPU").realize()
    return lambda: (rockchip_cpu.rms_norm(xt, gt, 1e-6),)
  if family == "quantize":
    assert depth is not None
    x = rng.normal(0, 0.1, (rows, depth)).astype(np.float32)
    x[0] = 0
    if rows > 1:
      x[1, :9] = [0, 127, -127, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
    if nonfinite:
      x[0, 0] = np.nan
      x[-1, -1] = np.inf
    xt = Tensor(x, device="CPU").realize()
    return lambda: rockchip_int8.quantize(xt, depth)
  if family == "dequantize":
    assert depth is not None
    columns = depth
    n = round_up(columns, 32)
    counts = [rows] if rows < 64 else [64, 64]
    raw_parts = []
    scale_parts = []
    for count in counts:
      values = rng.integers(-100000, 100000, (count, n), dtype=np.int32)
      scales = rng.uniform(0.001, 0.1, count).astype(np.float32)
      raw_parts.append(Tensor(_logical_codes(values), device="CPU").realize())
      scale_parts.append(scales)
    if nonfinite:
      scale_parts[0][0] = np.nan
    scale_t = [Tensor(x, device="CPU").realize() for x in scale_parts]
    sw_array = rng.uniform(0.001, 0.1, columns).astype(np.float32)
    if nonfinite:
      sw_array[-1] = np.inf
    sw = Tensor(sw_array, device="CPU").realize()
    return lambda: (rockchip_int8.dequantize_tiles(raw_parts, scale_t, sw, counts, columns),)
  if family == "qkv":
    heads, kv, dim, context = 16, 8, 128, 512
    columns = (heads + 2 * kv) * dim
    values = rng.integers(-100000, 100000, (rows, columns), dtype=np.int32)
    packed = _logical_codes(values)
    sa = rng.uniform(0.001, 0.003, rows).astype(np.float32)
    sw_array = rng.uniform(0.001, 0.003, columns).astype(np.float32)
    qn = rng.normal(1, 0.2, dim).astype(np.float16)
    kn = rng.normal(1, 0.2, dim).astype(np.float32)
    freq = rng.normal(0, 0.5, (context, dim)).astype(np.float32)
    cache = rng.normal(0, 0.2, (2, 1, kv, context, dim)).astype(np.float16)
    if nonfinite:
      sa[0] = np.nan
    raw_t = Tensor(packed, device="CPU", dtype=dtypes.int32).realize()
    sa_t = Tensor(sa, device="CPU").realize()
    sw_t = Tensor(sw_array, device="CPU").realize()
    qn_t = Tensor(qn, device="CPU").realize()
    kn_t = Tensor(kn, device="CPU").realize()
    freq_t = Tensor(freq, device="CPU").realize()

    def invoke():
      cache_t = Tensor(cache.copy(), device="CPU", dtype=dtypes.half).realize()
      return rockchip_qkv.prepare_qkv_int8([raw_t], [sa_t], sw_t, [rows], cache_t, qn_t, kn_t, freq_t, 0, heads, 1e-6, rows)

    return invoke
  if family == "ffn_pack":
    assert depth is not None
    values = rng.integers(-100000, 100000, (rows, depth * 2), dtype=np.int32)
    packed = _logical_codes(values)
    sa = rng.uniform(0.001, 0.003, rows).astype(np.float32)
    sw_array = rng.uniform(0.001, 0.003, depth * 2).astype(np.float32)
    if nonfinite:
      sa[0] = np.nan
    raw_t = Tensor(packed, device="CPU", dtype=dtypes.int32).realize()
    sa_t = Tensor(sa, device="CPU").realize()
    sw_t = Tensor(sw_array, device="CPU").realize()

    def invoke():
      old = rockchip_int8.getenv
      rockchip_int8.getenv = lambda key, default=0: 1 if key == "ONPU_SWIGLU_PACK_STATIC_LAYOUT" else old(key, default)
      try:
        return rockchip_int8.ffn_pack(raw_t, sa_t, sw_t, rows, depth, tile_rows=64)
      finally:
        rockchip_int8.getenv = old

    return invoke
  raise AssertionError(family)


def _tensors(result):
  return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def _run(invoke):
  outputs = _tensors(invoke())
  Tensor.realize(*outputs)
  return tuple(x.numpy().copy() for x in outputs)


def _assert_exact(expected, actual):
  assert len(expected) == len(actual)
  for a, b in zip(expected, actual):
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def _source(invoke, family):
  outputs = _tensors(invoke())
  linear, _ = Tensor.linear_with_vars(*outputs)
  name = {
    "rmsnorm": "rockchip_cpu_rmsnorm",
    "quantize": "rockchip_quantize",
    "dequantize": "rockchip_dequantize",
    "qkv": "rockchip_prepare_qkv",
    "ffn_pack": "rockchip_int8_ffn_pack",
  }[family]
  progs = [u for u in linear.toposort(enter_calls=True) if u.op is Ops.PROGRAM and u.src[0].arg.function_name == name]
  assert len(progs) == 1
  return next(u.arg for u in progs[0].src if u.op is Ops.SOURCE)


def _parallel(family, rows, depth=None):
  if family == "rmsnorm":
    return rows * 1024 >= 131072
  if family == "quantize":
    return rows * int(depth) >= 32768
  if family == "dequantize":
    return rows * int(depth) >= 131072
  if family == "qkv":
    return rows >= 16
  if family == "ffn_pack":
    return rows * int(depth) >= 32768
  raise AssertionError(family)


@pytest.mark.parametrize("family", ("rmsnorm", "quantize", "dequantize", "qkv", "ffn_pack"))
def test_shared_pool_serial_and_parallel_outputs_match_for_changed_inputs(shared_pool, family):
  pool, state, _ = shared_pool
  shapes = {
    "rmsnorm": [(8, None), (128, None)],
    "quantize": [(16, 512), (128, 1024), (128, 2048)],
    "dequantize": [(16, 512), (128, 1024)],
    "qkv": [(8, None), (128, None)],
    "ffn_pack": [(16, 512), (128, 3072)],
  }[family]
  for rows, depth in shapes:
    for seed in (51, 92):
      baseline = _run(_make_case(family, rows, seed, depth=depth))
      before = pool.lib.pool_calls()
      with state.scope():
        actual = _run(_make_case(family, rows, seed, depth=depth))
      delta = pool.lib.pool_calls() - before
      assert delta == int(_parallel(family, rows, depth)), (family, rows, depth, delta)
      _assert_exact(baseline, actual)
      assert pool.lib.pool_parked() and state.active_flag.value == 0
      assert "ROCKCHIP" not in Device._opened_devices and "CL" not in Device._opened_devices


@pytest.mark.parametrize("family", ("rmsnorm", "quantize", "dequantize", "qkv", "ffn_pack"))
def test_shared_pool_nonfinite_rows_match_exactly(shared_pool, family):
  pool, state, _ = shared_pool
  depth = {"rmsnorm": None, "quantize": 1024, "dequantize": 1024, "qkv": None, "ffn_pack": 3072}[family]
  baseline = _run(_make_case(family, 128, 177, depth=depth, nonfinite=True))
  before = pool.lib.pool_calls()
  with state.scope():
    actual = _run(_make_case(family, 128, 177, depth=depth, nonfinite=True))
  assert pool.lib.pool_calls() - before == 1
  _assert_exact(baseline, actual)


@pytest.mark.parametrize("family", ("rmsnorm", "quantize", "dequantize", "qkv", "ffn_pack"))
def test_shared_program_is_identity_outside_scope_and_skips_serial_threshold(shared_pool, family):
  _, state, _ = shared_pool
  large = {"rmsnorm": (128, None), "quantize": (128, 1024), "dequantize": (128, 1024), "qkv": (128, None), "ffn_pack": (128, 3072)}[family]
  small = {"rmsnorm": (8, None), "quantize": (16, 512), "dequantize": (16, 512), "qkv": (8, None), "ffn_pack": (16, 512)}[family]
  # Earlier cases intentionally populate the module-scoped rewrite cache.
  # Start fresh here to test exactly one new rewrite, then verify reuse.
  state.rewrite.cache_clear()
  before = len(state.changes)
  source = _source(_make_case(family, large[0], seed=411, depth=large[1]), family)
  assert len(state.changes) == before and "shared_work_callback" not in source
  assert "pthread_create(&threads[i]" in source
  with state.scope():
    pooled = _source(_make_case(family, large[0], seed=411, depth=large[1]), family)
    assert len(state.changes) == before + 1
    assert "shared_work_callback" in pooled and "pthread_create(&threads[i]" not in pooled
    assert _source(_make_case(family, large[0], seed=411, depth=large[1]), family) == pooled
    assert len(state.changes) == before + 1
  changed = len(state.changes)
  with state.scope():
    serial = _source(_make_case(family, small[0], seed=412, depth=small[1]), family)
    assert len(state.changes) == changed
    assert "shared_work_callback" not in serial and "pthread_create(&threads[i]" not in serial
  assert state.pool is shared_pool[0]


def test_explicit_scope_ignores_model_admission_flag_and_restores_pool_address(shared_pool):
  pool, state, parked = shared_pool
  old = ONPU_SHARED_HOST_POOL.value
  affinity = os.sched_getaffinity(0)
  calls = state.scope_calls
  try:
    with Context(ONPU_SHARED_HOST_POOL=0):
      with state.scope():
        assert state.active_flag.value == 1 and pool.active
        assert os.sched_getaffinity(0) == {4}
        with state.scope():
          assert state.active_flag.value == 1 and state.scope_calls == calls + 1
        assert state.scope_calls == calls + 1
    assert state.active_flag.value == 0 and not pool.active and pool.lib.pool_parked()
    assert os.sched_getaffinity(0) == affinity
    assert pool.prefill_address == state.address
    state.restore()
    assert pool.started and pool.prefill_address == parked and state.active_flag.value == 0
    assert os.sched_getaffinity(0) == affinity
  finally:
    ONPU_SHARED_HOST_POOL.value = old


def test_exception_parks_workers_and_restores_main_affinity(shared_pool):
  pool, state, _ = shared_pool
  affinity = os.sched_getaffinity(0)
  with pytest.raises(RuntimeError, match="synthetic"):
    with state.scope():
      assert state.active_flag.value == 1 and os.sched_getaffinity(0) == {4}
      raise RuntimeError("synthetic shared-pool exception")
  assert state.active_flag.value == 0 and not pool.active and pool.lib.pool_parked()
  assert os.sched_getaffinity(0) == affinity


def test_stopped_pool_falls_back_for_program_cached_while_active(shared_pool):
  pool, state, _ = shared_pool
  try:
    invoke = _make_case("quantize", 128, 617, depth=1024)
    baseline = _run(invoke)
    with state.scope():
      outputs = _tensors(invoke())
      linear, _ = Tensor.linear_with_vars(*outputs)
      program = next(u for u in linear.toposort(enter_calls=True) if u.op is Ops.PROGRAM and u.src[0].arg.function_name == "rockchip_quantize")
      source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
      assert "shared_work_callback" in source
    state.restore()
    calls = pool.lib.pool_calls()
    pool.close()
    Tensor.realize(*outputs)
    actual = tuple(x.numpy().copy() for x in outputs)
    _assert_exact(baseline, actual)
    assert not pool.started and pool.lib.pool_calls() == calls
    assert "ROCKCHIP" not in Device._opened_devices and "CL" not in Device._opened_devices
  finally:
    state.restore()
    pool.close()
