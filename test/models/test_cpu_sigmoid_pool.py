"""CPU-only exact contracts for the pooled gate sigmoid helper."""
import platform
import os
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8FeedForward, RockchipInt8Linear
from tinygrad.llm.model import gated_activation


@pytest.fixture(scope='module')
def attention_pool():
  if platform.machine().lower() not in ('aarch64', 'arm64'):
    pytest.skip('CPU gate sigmoid pool requires AArch64')
  if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
    pytest.skip('CPU gate sigmoid pool requires CPU4-7')
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  pool = AttentionPool()
  try:
    yield pool
  finally:
    pool.close()


def _logical_rows(joint, rows, capacity):
  return joint[:, :UOp.variable('sigmoid_rows', 1, capacity).bind(rows)]


def _storage_values(result):
  result.realize()
  return Tensor(result.uop.base, device='CPU').numpy()


def test_pool_matches_tensor_sigmoid_for_row_and_projection_profiles(attention_pool):
  rng = np.random.default_rng(20261025)
  for rows, capacity, depth in ((1, 8, 16), (3, 8, 512), (16, 32, 6144), (128, 128, 16)):
    values = rng.normal(0, 4, (1, capacity, depth * 2)).astype(np.float32)
    if rows < capacity:
      values[:, rows:, :] = np.nan
    joint = Tensor(values, device='CPU').realize()
    original = joint.numpy().copy()
    logical = _logical_rows(joint, rows, capacity)
    expected = joint[:, :, :depth].sigmoid().numpy()[:, :rows]

    before = attention_pool.lib.pool_calls()
    inactive = sigmoid_gate(logical, pool=attention_pool)
    inactive_values = _storage_values(inactive).reshape(1, capacity, depth)[:, :rows]
    assert attention_pool.lib.pool_calls() - before == 0
    assert inactive_values.shape == (1, rows, depth)
    np.testing.assert_array_equal(inactive_values.view(np.uint32), expected.view(np.uint32))

    with attention_pool.prefill():
      assert attention_pool.active
      before = attention_pool.lib.pool_calls()
      actual = sigmoid_gate(logical, pool=attention_pool)
      storage = _storage_values(actual).reshape(1, capacity, depth)
      observed = storage[:, :rows]
      calls = attention_pool.lib.pool_calls() - before
      assert calls == int(rows >= 4)
      assert observed.shape == (1, rows, depth)
      np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
      assert actual.max_shape == (1, capacity, depth)
      if rows < capacity:
        assert np.array_equal(storage[:, rows:].view(np.uint32), np.zeros_like(storage[:, rows:]).view(np.uint32))
      np.testing.assert_array_equal(joint.numpy().view(np.uint32), original.view(np.uint32))
    assert attention_pool.lib.pool_parked()


def test_special_values_match_tensor_sigmoid_bits_with_active_pool(attention_pool):
  values = np.array((np.nan, np.inf, -np.inf, 0.0, -0.0, 100.0, -100.0, 1e20, -1e20,
                     1.0, -1.0, 3.0, -3.0, 7.0, -7.0, 0.25), dtype=np.float32)
  joint_values = np.zeros((1, 8, 32), dtype=np.float32)
  for row in range(8):
    joint_values[0, row, :16] = np.roll(values, row)
    joint_values[0, row, 16:] = 1.0
  joint = Tensor(joint_values, device='CPU').realize()
  logical = _logical_rows(joint, 8, 8)
  expected = joint[:, :, :16].sigmoid().numpy()
  with attention_pool.prefill():
    before = attention_pool.lib.pool_calls()
    actual = sigmoid_gate(logical, pool=attention_pool)
    observed = _storage_values(actual).reshape(1, 8, 16)
    assert attention_pool.lib.pool_calls() - before == 1
  np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
  assert attention_pool.lib.pool_parked()


def test_dynamic_rows_one_through_sixteen_keep_prefix_output_stable(attention_pool):
  rng = np.random.default_rng(1643)
  capacity, depth = 16, 16
  values = rng.normal(0, 2, (1, capacity, depth * 2)).astype(np.float32)
  joint = Tensor(values, device='CPU').realize()
  original = joint.numpy().copy()
  reference = joint[:, :, :depth].sigmoid().numpy()
  run = TinyJit(lambda x, rows: sigmoid_gate(x[:, :rows], pool=attention_pool).realize())
  for rows in range(1, capacity + 1):
    bound = UOp.variable('jit_sigmoid_rows', 1, capacity).bind(rows)
    with attention_pool.prefill():
      actual = run(joint, bound)
      storage = _storage_values(actual).reshape(1, capacity, depth)
      observed = storage[:, :rows]
      np.testing.assert_array_equal(observed.view(np.uint32), reference[:, :rows].view(np.uint32))
      assert observed.shape == (1, rows, depth)
      assert actual.max_shape == (1, capacity, depth)
      if rows < capacity:
        assert np.array_equal(storage[:, rows:].view(np.uint32), np.zeros_like(storage[:, rows:]).view(np.uint32))
    assert attention_pool.lib.pool_parked()
  np.testing.assert_array_equal(joint.numpy().view(np.uint32), original.view(np.uint32))


def test_rejects_unsupported_dtype_geometry_and_capacity_without_gpu():
  with pytest.raises(ValueError, match='CPU sigmoid requires'):
    sigmoid_gate(Tensor.zeros(1, 8, 16, device='CPU', dtype=dtypes.float16))
  with pytest.raises(ValueError, match='CPU sigmoid requires'):
    sigmoid_gate(Tensor.zeros(2, 8, 16, device='CPU'))
  with pytest.raises(ValueError, match='CPU sigmoid requires'):
    sigmoid_gate(Tensor.zeros(8, 16, device='CPU'))
  with pytest.raises(ValueError, match='Unsupported CPU sigmoid projection shape'):
    sigmoid_gate(Tensor.zeros(1, 129, 8, device='CPU'))
  with pytest.raises(ValueError, match='Unsupported CPU sigmoid projection shape'):
    sigmoid_gate(Tensor.zeros(1, 8, 10, device='CPU'))


def test_int8_ffn_uses_pooled_float_gate_fallback_with_preserved_multiply_order(attention_pool):
  rng = np.random.default_rng(6144)
  rows, depth = 16, 6144
  projected_values = rng.normal(0, 1, (1, rows, depth * 2)).astype(np.float32)
  projected = Tensor(projected_values, device='CPU').realize()
  joint = RockchipInt8Linear.__new__(RockchipInt8Linear)
  joint.columns = depth * 2
  joint.depth = 32
  joint.packed = []
  joint.bias = None
  joint.prefill = 'on'
  joint.decode = 'on'
  joint.phase = 'prefill'
  projected_calls = []
  joint.npu = lambda value: (projected_calls.append(value) or projected)
  sibling = SimpleNamespace(prefill='on', decode='on', phase='prefill')
  projections = SimpleNamespace(joint=joint, layers=(joint, sibling))

  class FakeDown:
    def __init__(self):
      self.depth = depth
      self.phase = 'prefill'
      self.prefill = 'on'
      self.decode = 'on'
      self.input_scale = None
      self.packed = []
      self.bias = None
      self.received = []

    def __call__(self, value):
      self.received.append(value)
      return value

  down = FakeDown()
  fused = RockchipInt8FeedForward(projections, down, pool=attention_pool)
  x = Tensor(rng.normal(0, 1, (1, rows, 32)).astype(np.float32), device='CPU').realize()
  assert fused.enabled(x) is False
  assert RockchipInt8FeedForward(projections, down, pool=None).enabled(x) is False

  with attention_pool.prefill():
    assert fused.enabled(x) is True
    result = fused(x)
    assert len(projected_calls) == 1 and projected_calls[0] is x
    assert len(down.received) == 1 and result is down.received[0]
    gate = projected[:, :, :depth]
    up = projected[:, :, depth:]
    expected = gated_activation(gate, up)
    np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.numpy().view(np.uint32))
  assert attention_pool.lib.pool_parked()

  down.prefill = 'off'
  assert fused.enabled(x) is False
  down.prefill = 'on'
  down.phase = 'decode'
  decode_x = Tensor(rng.normal(0, 1, (1, 1, 32)).astype(np.float32), device='CPU').realize()
  assert fused.enabled(decode_x) is False
