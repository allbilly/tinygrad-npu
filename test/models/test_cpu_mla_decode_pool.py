"""CPU-only coverage for the opt-in active M1 MLA projection pool path."""
import numpy as np
import pytest

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_mla_cpu import CPUMLAProjections, configure


def _tables(seed=20261102):
  rng = np.random.default_rng(seed)
  key = Tensor(rng.normal(0, 0.03, (40, 256, 64)).astype(np.float32), device="CPU").half().realize()
  value = Tensor(rng.normal(0, 0.03, (40, 64, 256)).astype(np.float32), device="CPU").half().realize()
  return key, value


def _model():
  config = TransformerConfig(1, 2560, 512, 40, 40, 1e-5, 128, 96, 10000.0, 32, 64,
                            max_context=128, q_lora_rank=32, kv_lora_rank=256)
  return Transformer(config)


def _assert_bits(actual, expected):
  assert actual.dtype == expected.dtype == np.float32
  np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _source_text(jit):
  assert jit.captured is not None
  return "\n".join(str(source.arg) for call in jit.captured.linear.src if call.body.op.name == "PROGRAM"
                   for source in call.body.src if source.op.name == "SOURCE")


def _project_source_address(jit, depth, columns, capacity=1):
  import re
  match = re.search(rf"mla_project\(out,input,weight,40,{capacity},\*rows,{depth},{columns},(\d+)ULL\)", _source_text(jit))
  assert match is not None
  return int(match.group(1))


def _needs_pool_lock():
  from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock
  _needs_pool()
  return _pool_test_lock()


def _case_inputs(case, rng):
  key = rng.normal(0, 0.03, (40, 256, 64)).astype(np.float32).astype(np.float16)
  value = rng.normal(0, 0.03, (40, 64, 256)).astype(np.float32).astype(np.float16)
  query_x = rng.normal(0, 0.2, (1, 40, 1, 64)).astype(np.float32)
  value_x = rng.normal(0, 0.2, (1, 40, 1, 256)).astype(np.float32)
  if case == "zeros":
    key.fill(0)
    value.fill(0)
    query_x.fill(0)
    value_x.fill(0)
  elif case == "signedzero":
    key.fill(np.float16(-0.0))
    value.fill(np.float16(-0.0))
    query_x.fill(np.float32(-0.0))
    value_x.fill(np.float32(-0.0))
  elif case == "inputNaN":
    query_x.fill(np.float32(np.nan))
    value_x.fill(np.float32(np.nan))
  elif case == "inputInf":
    query_x.fill(np.float32(np.inf))
    value_x.fill(np.float32(np.inf))
  elif case == "weightNaN":
    key.fill(np.float16(np.nan))
    value.fill(np.float16(np.nan))
  elif case == "weightInf":
    key.fill(np.float16(np.inf))
    value.fill(np.float16(np.inf))
  elif case == "largeFinite":
    query_x.fill(np.float32(2.0**60))
    value_x.fill(np.float32(2.0**60))
    key.fill(np.float16(2.0**10))
    value.fill(np.float16(2.0**10))
  return (Tensor(key, device="CPU").realize(), Tensor(value, device="CPU").realize(),
          Tensor(query_x, device="CPU").realize(), Tensor(value_x, device="CPU").realize())


@pytest.mark.parametrize("case", ["ordinary", "changedFinite", "zeros", "signedzero", "inputNaN", "inputInf",
                                   "weightNaN", "weightInf", "largeFinite"])
@pytest.mark.parametrize("event_wait", [False, True])
def test_active_decode_pool_query_and_value_are_byte_exact_to_serial(case, event_wait, monkeypatch):
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

  monkeypatch.setenv("ONPU_HOST_POOL_WFE", "1" if event_wait else "0")
  with _needs_pool_lock():
    key, value, query_x, value_x = _case_inputs(case, np.random.default_rng(20261103 + len(case)))
    serial = CPUMLAProjections(key, value)
    pool = AttentionPool()
    assert pool.lib.pool_wait_mode() == int(event_wait)
    pooled = CPUMLAProjections(key, value, pool=pool, decode_pool=True)
    serial.prepare()
    pooled.prepare()
    before = pool.lib.pool_calls()
    try:
      with pool:
        expected_query = serial.query(query_x).numpy().copy()
        expected_value = serial.value(value_x).numpy().copy()
        _assert_bits(pooled.query(query_x).numpy(), expected_query)
        _assert_bits(pooled.value(value_x).numpy(), expected_value)
      assert pool.lib.pool_calls() - before == 2
    finally:
      pool.close()


@pytest.mark.parametrize("role,depth,columns", [("query", 64, 256), ("value", 256, 64)])
def test_decode_pool_tinyjit_uses_active_callback_address(role, depth, columns):
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

  with _needs_pool_lock():
    key, value = _tables(1401)
    serial = CPUMLAProjections(key, value)
    pool = AttentionPool()
    pooled = CPUMLAProjections(key, value, pool=pool, decode_pool=True)
    serial.prepare()
    pooled.prepare()
    method, baseline = getattr(pooled, role), getattr(serial, role)
    call_count = pool.lib.pool_calls()
    try:
      with pool:
        runner = TinyJit(lambda x: method(x).realize())
        for seed in (11, 12, 13, 14):
          rng = np.random.default_rng(seed)
          x = Tensor(rng.normal(0, 0.2, (1, 40, 1, depth)).astype(np.float32), device="CPU").realize()
          expected = baseline(x).numpy()
          _assert_bits(runner(x).numpy(), expected)
        assert _project_source_address(runner, depth, columns) == pool.address
        assert pool.lib.pool_calls() > call_count
    finally:
      pool.close()


def test_decode_pool_defaults_to_serial_and_requires_active_scope_for_dispatch():
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

  with _needs_pool_lock():
    key, value = _tables(1402)
    pool = AttentionPool()
    default_hook = CPUMLAProjections(key, value, pool=pool)
    opt_in_hook = CPUMLAProjections(key, value, pool=pool, decode_pool=True)
    default_hook.prepare()
    opt_in_hook.prepare()
    x = Tensor(np.ones((1, 40, 1, 64), dtype=np.float32), device="CPU").realize()
    calls = pool.lib.pool_calls()
    try:
      # Building and executing outside the active owner uses address zero.
      outside = TinyJit(lambda inp: opt_in_hook.query(inp).realize())
      for _ in range(4): outside(x)
      assert _project_source_address(outside, 64, 256) == 0
      assert pool.lib.pool_calls() == calls

      # The default stays serial even when an owning active scope exists.
      with pool:
        default = TinyJit(lambda inp: default_hook.query(inp).realize())
        for _ in range(4): default(x)
        assert _project_source_address(default, 64, 256) == 0
        assert pool.lib.pool_calls() == calls
    finally:
      pool.close()


def test_prefill_capacity_still_uses_parked_callback_and_exact_projection():
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

  with _needs_pool_lock():
    key, value = _tables(1403)
    serial = CPUMLAProjections(key, value)
    pool = AttentionPool()
    pooled = CPUMLAProjections(key, value, pool=pool, decode_pool=True)
    serial.prepare()
    pooled.prepare()
    inputs = Tensor(np.random.default_rng(1404).normal(size=(1, 40, 17, 64)).astype(np.float32),
                    device="CPU").realize()
    expected = serial.query(inputs).numpy()
    before = pool.lib.pool_calls()
    try:
      with pool:
        run = TinyJit(lambda x: pooled.query(x).realize())
        for _ in range(4): _assert_bits(run(inputs).numpy(), expected)
        assert _project_source_address(run, 64, 256, 17) == pool.prefill_address
        assert pool.lib.pool_calls() > before
    finally:
      pool.close()


def test_decode_pool_flags_and_unsupported_tables_are_rejected_cpu_only(monkeypatch):
  from tinygrad.llm.kernels import rockchip

  key, value = _tables(1405)
  with pytest.raises(ValueError, match="decode pool selection"):
    CPUMLAProjections(key, value, decode_pool=1)
  with pytest.raises(ValueError, match="requires a worker pool"):
    CPUMLAProjections(key, value, decode_pool=True)
  with pytest.raises(ValueError, match="requires projections and a worker pool"):
    configure(_model(), enabled=False, decode_pool=True)
  with pytest.raises(ValueError, match="requires projections and a worker pool"):
    configure(_model(), enabled=True, decode_pool=True)
  with pytest.raises(ValueError, match="decode pool selection"):
    configure(_model(), decode_pool=1)
  with pytest.raises(ValueError, match="requires CPU projections and workers"):
    rockchip.configure(_model(), cpu_mla_decode_pool=True)
  with pytest.raises(ValueError, match="requires CPU projections and workers"):
    rockchip.configure(_model(), cpu_mla_projections=True, cpu_mla_decode_pool=True, cpu_pool=False)
  with pytest.raises(ValueError, match="CPU MLA decode pool selection"):
    rockchip.configure(_model(), cpu_mla_decode_pool=1)

  bad = _model()
  bad.blk[0].attn_k_b["weight"] = bad.blk[0].attn_k_b["weight"].float()
  before = {name for name in Device._opened_devices if name.split(":")[0] == "ROCKCHIP"}
  placement = rockchip.configure(bad, prefill="off", decode="off", cpu_pool=False, cpu_mla_projections=True)
  after = {name for name in Device._opened_devices if name.split(":")[0] == "ROCKCHIP"}
  assert placement["cpu_mla_projection_blocks"] == 0
  assert not hasattr(bad.blk[0], "cpu_mla_projections")
  assert after == before

  before = set(Device._opened_devices)
  assert configure(bad, enabled=True, pool=object(), decode_pool=True) == 0
  assert not hasattr(bad.blk[0], "cpu_mla_projections")
  assert set(Device._opened_devices) == before
