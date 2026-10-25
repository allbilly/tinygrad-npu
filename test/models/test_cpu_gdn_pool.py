"""Native CPU pool binding checks; model/NPU performance is tested separately."""
import os
import platform
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from tinygrad.llm.kernels.rockchip_recurrent import gated_delta_scan

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64'),reason='Qualified ARM CPU worker topology')


@pytest.fixture(scope='module')
def pool():
  value = AttentionPool()
  try:yield value
  finally:value.close()


def operands(batch, heads, capacity, key_dim, value_dim, alpha_dim, rows, seed=341):
  rng = np.random.default_rng(seed)
  arrays = [rng.normal(0,.1,(batch,heads,capacity,d)).astype(np.float32) for d in (key_dim,key_dim,value_dim)]
  arrays += [rng.uniform(.1,.9,(batch,heads,capacity)).astype(np.float32),
             rng.uniform(.65,.98,(batch,heads,capacity,alpha_dim)).astype(np.float32)]
  for value in arrays:value[:,:,rows:] = np.nan
  return tuple(Tensor(value,device='CPU').realize() for value in arrays), \
    rng.normal(0,.04,(batch,heads,value_dim,key_dim)).astype(np.float32)


def exact(actual, expected):
  np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('geometry',[(1,4,8,5,3,1,3,True),(1,4,8,5,3,1,8,True),
                                    (2,3,8,8,5,8,3,False),(1,16,128,128,128,1,127,False)])
def test_pooled_tensor_output_state_history_and_reset_match_serial_bits(pool, geometry):
  batch,heads,capacity,key_dim,value_dim,alpha_dim,rows,history = geometry
  inputs,initial = operands(batch,heads,capacity,key_dim,value_dim,alpha_dim,rows)
  reference_state = Tensor(initial.copy(),device='CPU').realize()
  pooled_state = Tensor(initial.copy(),device='CPU').realize()
  reference_history = Tensor.full((8,batch,heads,value_dim,key_dim),float('nan'),device='CPU').realize() if history else None
  pooled_history = Tensor.full((8,batch,heads,value_dim,key_dim),float('nan'),device='CPU').realize() if history else None
  affinity = os.sched_getaffinity(0)
  for start in (0,17,0):
    expected = gated_delta_scan(*inputs,reference_state,start,rows,history=reference_history).numpy().copy()
    before = pool.lib.pool_calls()
    with pool.prefill():
      actual = gated_delta_scan(*inputs,pooled_state,start,rows,history=pooled_history,pool=pool).numpy().copy()
    assert pool.lib.pool_calls()==before+1
    assert pool.lib.pool_parked() and os.sched_getaffinity(0)==affinity
    exact(actual,expected)
    exact(pooled_state.numpy(),reference_state.numpy())
    assert np.isfinite(actual[:,:,:rows]).all() and not np.any(actual[:,:,rows:])
    if history:
      exact(pooled_history.numpy(),reference_history.numpy())
      assert np.isnan(pooled_history.numpy()[rows:]).all()


def test_inactive_pool_and_small_head_count_keep_serial_path(pool):
  for heads,active in ((4,False),(2,True)):
    inputs,initial = operands(1,heads,4,8,3,1,3)
    reference_state = Tensor(initial.copy(),device='CPU').realize()
    expected = gated_delta_scan(*inputs,reference_state,0,3).numpy().copy()
    state = Tensor(initial.copy(),device='CPU').realize()
    before = pool.lib.pool_calls()
    if active:
      with pool.prefill():actual = gated_delta_scan(*inputs,state,0,3,pool=pool).numpy().copy()
    else:actual = gated_delta_scan(*inputs,state,0,3,pool=pool).numpy().copy()
    assert pool.lib.pool_calls()==before
    exact(actual,expected)
    exact(state.numpy(),reference_state.numpy())


def test_tinyjit_pooled_dynamic_rows_positions_and_history_match_serial(pool):
  inputs,initial = operands(2,3,8,8,3,1,8,seed=342)
  serial_state = Tensor(initial.copy(),device='CPU').realize()
  pooled_state = Tensor(initial.copy(),device='CPU').realize()
  serial_history = Tensor.full((8,2,3,3,8),float('nan'),device='CPU').realize()
  pooled_history = Tensor.full((8,2,3,3,8),float('nan'),device='CPU').realize()
  serial = TinyJit(lambda *args:gated_delta_scan(*args[:5],serial_state,args[5],args[6],history=serial_history).realize())
  pooled = TinyJit(lambda *args:gated_delta_scan(*args[:5],pooled_state,args[5],args[6],history=pooled_history,pool=pool).realize())
  for index,(start,rows) in enumerate(((0,8),(11,5),(20,2),(0,3),(7,6))):
    position = UOp.variable('pooled_gdn_test_start',0,64).bind(start)
    count = UOp.variable('pooled_gdn_test_rows',1,8).bind(rows)
    expected = serial(*inputs,position,count).numpy().copy()
    before = pool.lib.pool_calls()
    with pool.prefill():actual = pooled(*inputs,position,count).numpy().copy()
    assert pool.lib.pool_calls()==before+1,(index,pooled.cnt)
    exact(actual,expected)
    exact(pooled_state.numpy(),serial_state.numpy())
    exact(pooled_history.numpy(),serial_history.numpy())
  assert serial.captured is not None and pooled.captured is not None
