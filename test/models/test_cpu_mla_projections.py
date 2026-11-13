"""CPU coverage for prepared FP16 MLA projections."""

import itertools, os, platform
import numpy as np
import pytest

from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.nn.state import get_state_dict
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_mla_cpu import CPUMLAProjections, configure


def _tables(heads=4, key_columns=64, key_depth=64, value_columns=64, value_depth=64, seed=17):
  rng = np.random.default_rng(seed)
  key = Tensor(rng.normal(0, 0.03, (heads, key_columns, key_depth)).astype(np.float32), device="CPU").half().realize()
  value = Tensor(rng.normal(0, 0.03, (heads, value_columns, value_depth)).astype(np.float32), device="CPU").half().realize()
  return key, value


def _model(seed=23, blocks=1):
  config = TransformerConfig(blocks, 256, 512, 4, 4, 1e-5, 96, 80, 10000.0, 16, 64, max_context=64, q_lora_rank=32, kv_lora_rank=64)
  model = Transformer(config)
  rng = np.random.default_rng(seed)
  for name, tensor in get_state_dict(model).items():
    values = rng.normal(0, 0.02, tensor.shape).astype(np.float16)
    if "norm" in name:
      values += 1
    tensor.replace(Tensor(values, device="CPU").realize())
  return model


def _reference(x, weight):
  return (x @ weight.transpose(-1, -2)).realize()


def _assert_float_bits(actual, expected):
  assert actual.dtype == expected.dtype == np.float32
  np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("heads,key_columns,key_depth,value_columns,value_depth", [(4, 64, 64, 64, 64), (40, 256, 64, 64, 256)])
def test_cpu_mla_projection_is_bitwise_reference_and_preparation_is_idempotent(heads, key_columns, key_depth, value_columns, value_depth):
  key, value = _tables(heads, key_columns, key_depth, value_columns, value_depth)
  key_identity, value_identity = id(key), id(value)
  hook = CPUMLAProjections(key, value)
  hook.prepare()
  packed_once = hook._packed()
  hook.prepare()
  assert hook._packed() is packed_once
  assert hook.packed_bytes == sum(w.nbytes() for w in packed_once)
  assert id(hook._sources()[0]) == key_identity and id(hook._sources()[1]) == value_identity
  rng = np.random.default_rng(29)
  rows = 17 if heads == 4 else 128
  x = Tensor(rng.normal(size=(1, heads, rows, key_depth)).astype(np.float32)).realize()
  np.testing.assert_array_equal(hook.query(x).numpy().view(np.uint32), _reference(x, key).numpy().view(np.uint32))
  xv = Tensor(rng.normal(size=(1, heads, rows, value_depth)).astype(np.float32)).realize()
  np.testing.assert_array_equal(hook.value(xv).numpy().view(np.uint32), _reference(xv, value).numpy().view(np.uint32))


def test_cpu_mla_projection_dynamic_rows_and_zero_padding_with_tinyjit():
  key, value = _tables()
  hook = CPUMLAProjections(key, value)
  hook.prepare()
  rows = Variable("mla_rows", 1, 128)

  def project(x, n):
    shaped = x[:, :, :n, :]
    q = hook.query(shaped).pad_to((1, 4, 128, 64))
    v = hook.value(shaped).pad_to((1, 4, 128, 64))
    q.realize(v)
    return q, v

  run = TinyJit(project)
  rng = np.random.default_rng(31)
  for count in (1, 17, 65, 128):
    data = rng.normal(size=(1, 4, 128, 64)).astype(np.float32)
    data[:, :, count:, :] = np.nan
    x = Tensor(data).realize()
    q, v = run(x, rows.bind(count))
    clean = Tensor(np.nan_to_num(data, nan=0.0)).realize()
    _assert_float_bits(q.numpy()[:, :, :count], _reference(clean[:, :, :count], key).numpy())
    _assert_float_bits(v.numpy()[:, :, :count], _reference(clean[:, :, :count], value).numpy())
    np.testing.assert_array_equal(q.numpy()[:, :, count:, :], np.zeros((1, 4, 128 - count, 64), dtype=np.float32))
    np.testing.assert_array_equal(v.numpy()[:, :, count:, :], np.zeros((1, 4, 128 - count, 64), dtype=np.float32))


def test_cpu_mla_projection_configure_is_atomic_and_disable_removes_hook():
  model = _model()
  block = model.blk[0]
  assert configure(model) == 1
  first = block.cpu_mla_projections
  assert first._sources() == (block.attn_k_b["weight"], block.attn_v_b["weight"])
  assert configure(model) == 1 and block.cpu_mla_projections is not first
  assert configure(model, enabled=False) == 0
  assert not hasattr(block, "cpu_mla_projections")

  model = _model()
  model.blk[0].attn_k_b["weight"] = Tensor.zeros(4, 64, 64)
  assert configure(model) == 0
  assert not hasattr(model.blk[0], "cpu_mla_projections")
  model = _model()
  model.blk[0].attn_v_b["weight"] = Tensor.zeros(4, 64, 63, dtype=dtypes.half)
  assert configure(model) == 0
  assert not hasattr(model.blk[0], "cpu_mla_projections")
  mixed = _model(blocks=2)
  mixed.blk[1].attn_v_b["weight"] = mixed.blk[1].attn_v_b["weight"].float()
  assert configure(mixed) == 0 and all(not hasattr(block, "cpu_mla_projections") for block in mixed.blk)
  with pytest.raises(ValueError):
    configure(_model(), enabled=1)


def test_configured_pool_dispatches_and_parks_when_graph_is_built_outside_scope():
  from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  _needs_pool()
  with _pool_test_lock():
    pool=AttentionPool()
    try:
      model=_model()
      assert configure(model,pool=pool)==1
      hook=model.blk[0].cpu_mla_projections
      assert hook._pool() is pool
      x=Tensor.randn(1,4,17,64,device='CPU').realize()
      output=hook.query(x)
      before=pool.lib.pool_calls()
      _assert_float_bits(output.numpy(),_reference(x,hook._sources()[0]).numpy())
      assert pool.lib.pool_calls()>before
      assert not pool.active and pool.lib.pool_parked()
    finally:pool.close()


def test_configured_prefill_pool_keeps_single_row_decode_serial():
  from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  _needs_pool()
  with _pool_test_lock():
    pool=AttentionPool()
    try:
      key,value=_tables(heads=40,key_columns=256,key_depth=64,value_columns=64,value_depth=256)
      hook=CPUMLAProjections(key,value,pool=pool)
      hook.prepare()
      for method,weight in ((hook.query,key),(hook.value,value)):
        x=Tensor.randn(1,40,1,int(weight.shape[-1]),device='CPU').realize()
        before=pool.lib.pool_calls()
        _assert_float_bits(method(x).numpy(),_reference(x,weight).numpy())
        assert pool.lib.pool_calls()==before
      assert not pool.active and pool.lib.pool_parked()
    finally:pool.close()


def test_cpu_mla_projection_full_model_matches_serial_prefill_decode_reset_and_generation():
  serial = _model(seed=41)
  prepared = _model(seed=41)
  serial_state = get_state_dict(serial)
  prepared_state = get_state_dict(prepared)
  assert serial_state.keys() == prepared_state.keys()
  for name in serial_state:
    np.testing.assert_array_equal(serial_state[name].numpy(), prepared_state[name].numpy(), err_msg=name)
  for attr in ("attn_k_b", "attn_v_b"):
    np.testing.assert_array_equal(getattr(serial.blk[0], attr)["weight"].numpy(), getattr(prepared.blk[0], attr)["weight"].numpy())
  identities = {id(t) for t in get_state_dict(prepared).values()}
  identities.update((id(prepared.blk[0].attn_k_b["weight"]), id(prepared.blk[0].attn_v_b["weight"])))
  assert configure(prepared) == 1
  assert identities == {id(t) for t in get_state_dict(prepared).values()} | {
    id(prepared.blk[0].attn_k_b["weight"]),
    id(prepared.blk[0].attn_v_b["weight"]),
  }

  prefill_tokens = Tensor([[1, 7, 3, 9, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]], dtype=dtypes.int)
  serial_prefill = serial.logits(prefill_tokens, 0).realize()
  prepared_prefill = prepared.logits(prefill_tokens, 0).realize()
  _assert_float_bits(prepared_prefill.numpy(), serial_prefill.numpy())
  serial_decode = serial.logits(Tensor([[5]], device="CPU", dtype=dtypes.int), 17).realize()
  prepared_decode = prepared.logits(Tensor([[5]], device="CPU", dtype=dtypes.int), 17).realize()
  _assert_float_bits(prepared_decode.numpy(), serial_decode.numpy())

  serial = _model(seed=41)
  prepared = _model(seed=41)
  assert configure(prepared) == 1
  prefix = Tensor([[2, 4, 6]], dtype=dtypes.int)
  _assert_float_bits(prepared.logits(prefix, 0).realize().numpy(), serial.logits(prefix, 0).realize().numpy())
  serial = _model(seed=41)
  prepared = _model(seed=41)
  assert configure(prepared) == 1
  assert list(itertools.islice(prepared.generate([3, 5, 7], chunk_size=3), 3)) == list(itertools.islice(serial.generate([3, 5, 7], chunk_size=3), 3))


def test_cpu_mla_projection_rejects_unsupported_tables_without_partial_installation():
  key, value = _tables()
  with pytest.raises(ValueError):
    CPUMLAProjections(key.float(), value)
  with pytest.raises(ValueError):
    CPUMLAProjections(key, Tensor.zeros(4, 64, 63, device="CPU", dtype=dtypes.half))
  model = _model()
  model.blk[0].attn_k_b["weight"] = key.float()
  assert configure(model) == 0 and not hasattr(model.blk[0], "cpu_mla_projections")


def test_rockchip_cpu_npu_mla_projection_conflict_and_bool_validation():
  from tinygrad.llm.kernels import rockchip

  model = _model()
  with pytest.raises(ValueError):
    rockchip.configure(model, cpu_mla_projections=1)
  with pytest.raises(ValueError):
    rockchip.configure(model, cpu_mla_projections=True, npu_mla_prefill=True)
  with pytest.raises(ValueError):
    rockchip.configure(model, cpu_mla_projections=True, npu_mla_decode=True)


def test_cpu_mla_projection_optional_pool_uses_parked_workers_and_matches_serial():
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

  if platform.machine().lower() not in ("aarch64", "arm64") or not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
    pytest.skip("CPU4-7 unavailable for optional attention pool")
  key, value = _tables()
  serial = CPUMLAProjections(key, value)
  serial.prepare()
  pool = AttentionPool()
  affinity = os.sched_getaffinity(0)
  before = pool.lib.pool_calls()
  try:
    pooled = CPUMLAProjections(key, value, pool=pool)
    pooled.prepare()
    x = Tensor(np.random.default_rng(73).normal(size=(1, 4, 17, 64)).astype(np.float32)).realize()
    expected = serial.query(x).numpy()
    expected_value = serial.value(x).numpy()
    with pool.prefill():
      _assert_float_bits(pooled.query(x).numpy(), expected)
    with pool.prefill():
      _assert_float_bits(pooled.value(x).numpy(), expected_value)
    assert pool.lib.pool_calls() - before == 2
    assert pool.lib.pool_parked()
    assert os.sched_getaffinity(0) == affinity
    assert pool.owner_thread is None
    assert np.isfinite(expected).all()
  finally:
    pool.close()


def test_cpu_projection_option_activates_without_opening_npu(tmp_path, monkeypatch):
  import json
  import sys
  from tinygrad import Device
  from extra.rockchip import benchmark_inference
  checkpoint = tmp_path / 'fixture.gguf'
  checkpoint.write_bytes(b'model_override provides the trained-shape fixture')
  inputs = tmp_path / 'input.ids'
  inputs.write_text('3 7 2\n')
  report = tmp_path / 'report.json'
  monkeypatch.delenv('ONPU_BENCHMARK_CHECKPOINT_SHA256', raising=False)
  monkeypatch.setattr(sys, 'argv', ['benchmark_inference', '--checkpoint', str(checkpoint), '--input', str(inputs),
    '--report', str(report), '--device', 'CPU', '--prefill', 'off', '--decode', 'off', '--precision', 'fp16',
    '--context', '64', '--tokens', '2', '--warmups', '1', '--runs', '1', '--cpu-mla-projections'])
  before = {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'}
  benchmark_inference.main(model_override=_model())
  data = json.loads(report.read_text())
  assert data['complete'] and data['cpu_mla_projections']
  assert data['placement']['cpu_mla_projection_blocks'] == 1
  assert data['placement']['cpu_mla_projection_packed_weight_bytes'] == 65536
  assert {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'} == before
