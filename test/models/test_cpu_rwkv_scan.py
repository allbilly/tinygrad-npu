"""Independent FP64 recurrence, state mutation and bounded CPU RWKV7 replay checks."""
import os, platform
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.rockchip_rwkv import rwkv_scan


def fixture(batch,capacity,heads,width,seed=771):
  rng=np.random.default_rng(seed)
  shape=(batch,capacity,heads,width)
  r,k,v=[rng.normal(0,.15,shape).astype(np.float32) for _ in range(3)]
  decay=rng.uniform(.65,.99,shape).astype(np.float32)
  rate=rng.uniform(.05,.95,shape).astype(np.float32)
  unit=rng.normal(size=shape)
  unit=(unit/np.linalg.norm(unit,axis=-1,keepdims=True)).astype(np.float32)
  state=rng.normal(0,.04,(batch,heads,width,width)).astype(np.float32)
  return [r,k,v,decay,rate,unit],state


def reference(inputs,state,start,rows):
  r,k,v,decay,rate,unit=[x.astype(np.float64) for x in inputs]
  state=state.astype(np.float64).copy() if start else np.zeros_like(state,dtype=np.float64)
  out=np.zeros_like(r)
  for t in range(rows):
    # Matrix multiplication is deliberately independent of the C column-major scratch loop.
    correction=-unit[:,t,:,:,None]*(unit[:,t]*rate[:,t])[:,:,None,:]
    state=state*decay[:,t,:,None,:]+state@correction+v[:,t,:,:,None]*k[:,t,:,None,:]
    out[:,t]=(state@r[:,t,:,:,None])[...,0]
  return out,state


@pytest.mark.parametrize('batch,capacity,heads,width,rows,start',[(2,7,3,5,3,0),(1,128,8,64,128,0),
  (1,17,4,128,13,9),(1,2,1,256,1,4)])
def test_reference_state_reset_and_poisoned_padding(batch,capacity,heads,width,rows,start):
  inputs,initial=fixture(batch,capacity,heads,width)
  for x in inputs:x[:,rows:]=np.nan
  if not start:initial[:]=np.nan
  expected,final=reference(inputs,initial,start,rows)
  tensors=[Tensor(x,device='CPU').realize() for x in inputs]
  state=Tensor(initial.copy(),device='CPU').realize()
  actual=rwkv_scan(*tensors,state,start,rows).numpy()
  assert np.isfinite(actual).all()
  np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-4)
  np.testing.assert_allclose(state.numpy(),final,atol=3e-6,rtol=3e-4)
  assert not actual[:,rows:].any()


def test_dynamic_changed_inputs_and_reset_survive_jit_with_mutated_state():
  batch,capacity,heads,width=1,17,4,16
  state=Tensor.full((batch,heads,width,width),float('nan'),device='CPU').realize()
  expected_state=state.numpy().copy()
  run=TinyJit(lambda *args:rwkv_scan(*args).realize())
  for index,(start,rows) in enumerate(((0,17),(17,1),(18,13),(31,17),(0,3),(3,16),(0,17))):
    inputs,_=fixture(batch,capacity,heads,width,seed=772+index)
    for x in inputs:x[:,rows:]=np.nan
    expected,expected_state=reference(inputs,expected_state,start,rows)
    actual=run(*[Tensor(x,device='CPU').realize() for x in inputs],state,
      UOp.variable('rwkv_start',0,100).bind(start),UOp.variable('rwkv_rows',1,capacity).bind(rows)).numpy().copy()
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-4)
    np.testing.assert_allclose(state.numpy(),expected_state,atol=3e-6,rtol=3e-4)
    assert not actual[:,rows:].any()


def test_chunk_composition_preserves_every_output_and_final_state_bits():
  inputs,initial=fixture(1,17,4,64)
  full_state=Tensor(initial.copy(),device='CPU').realize()
  full=rwkv_scan(*[Tensor(x,device='CPU').realize() for x in inputs],full_state,3,17).numpy()
  chunk_state=Tensor(initial.copy(),device='CPU').realize()
  outputs=[]
  for first,last in ((0,5),(5,6),(6,17)):
    outputs.append(rwkv_scan(*[Tensor(x[:,first:last].copy(),device='CPU').realize() for x in inputs],
                            chunk_state,first+3,last-first).numpy())
  np.testing.assert_array_equal(np.concatenate(outputs,axis=1).view(np.uint32),full.view(np.uint32))
  np.testing.assert_array_equal(chunk_state.numpy().view(np.uint32),full_state.numpy().view(np.uint32))


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64'),reason='Qualified ARM CPU reduction order')
def test_decode_matches_existing_tensor_recurrence_output_and_state_bits():
  from tinygrad import Device
  from tinygrad.renderer.cstyle import ClangRenderer
  if not isinstance(Device['CPU'].renderer,ClangRenderer):pytest.skip('Qualified Clang CPU reduction order')
  def original(r,k,v,decay,rate,unit,storage,start):
    state=Tensor(start,device='CPU').eq(0).where(0,storage.clone())
    kk=unit[:,0]
    updated=state*decay[:,0].unsqueeze(-2)-(state*kk.unsqueeze(-2)).sum(-1,keepdim=True)*(kk*rate[:,0]).unsqueeze(-2)
    updated=updated+v[:,0].unsqueeze(-1)*k[:,0].unsqueeze(-2)
    store=storage.uop.store(updated.uop)
    return Tensor((updated*r[:,0].unsqueeze(-2)).sum(-1).unsqueeze(1).contiguous().uop.after(store)).realize()
  reference_run=TinyJit(original)
  for seed in (777,778,779):
    inputs,initial=fixture(1,1,32,64,seed=seed)
    tensors=[Tensor(x,device='CPU').realize() for x in inputs]
    for start in (0,5):
      control_state=Tensor(initial.copy(),device='CPU').realize()
      native_state=Tensor(initial.copy(),device='CPU').realize()
      pos=UOp.variable('rwkv_bitwise_start',0,100).bind(start)
      expected=reference_run(*tensors,control_state,pos).numpy().copy()
      actual=rwkv_scan(*tensors,native_state,pos,1).numpy().copy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      np.testing.assert_array_equal(native_state.numpy().view(np.uint32),control_state.numpy().view(np.uint32))


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64'),reason='Qualified ARM CPU reduction order')
def test_lazy_normalized_key_matches_existing_tensor_formation_bits():
  def original(r,k,v,decay,rate,raw_key,weight,storage,start):
    weight=weight.bitcast(dtypes.float).half().reshape(32,64)
    unit=raw_key*weight.reshape(1,1,32,64)
    unit=unit/unit.square().sum(-1,keepdim=True).sqrt().maximum(1e-12)
    state=Tensor(start,device='CPU').eq(0).where(0,storage.clone())
    updated=state*decay[:,0].unsqueeze(-2)-(state*unit[:,0].unsqueeze(-2)).sum(-1,keepdim=True)*(unit[:,0]*rate[:,0]).unsqueeze(-2)
    updated=updated+v[:,0].unsqueeze(-1)*k[:,0].unsqueeze(-2)
    store=storage.uop.store(updated.uop)
    return Tensor((updated*r[:,0].unsqueeze(-2)).sum(-1).unsqueeze(1).contiguous().uop.after(store)).realize()
  inputs,initial=fixture(1,1,32,64,seed=781)
  rng=np.random.default_rng(782)
  raw_key=Tensor(rng.normal(0,.3,(1,1,32,64)).astype(np.float32),device='CPU').realize()
  values=rng.uniform(.5,1.5,(32,64)).astype(np.float32)
  raw_weight=Tensor(np.frombuffer(values.tobytes(),np.uint8).copy(),device='CPU').realize()
  weight=raw_weight.bitcast(dtypes.float).half().reshape(32,64)
  operands=[Tensor(x,device='CPU').realize() for x in inputs[:5]]
  run=TinyJit(original)
  for start in (5,0,7,9):
    control_state=Tensor(initial.copy(),device='CPU').realize()
    native_state=Tensor(initial.copy(),device='CPU').realize()
    pos=UOp.variable('rwkv_formed_start',0,100).bind(start)
    expected=run(*operands,raw_key,raw_weight,control_state,pos).numpy().copy()
    unit=raw_key*weight.reshape(1,1,32,64)
    norm=unit.square().sum(-1,keepdim=True).sqrt().maximum(1e-12)
    unit=unit/norm
    actual=rwkv_scan(*operands,unit,native_state,pos,1,prediction_key=raw_key,prediction_weight=weight.float(),
                     prediction_norm=norm,correction_key=unit*operands[4]).numpy().copy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(native_state.numpy().view(np.uint32),control_state.numpy().view(np.uint32))


def test_invalid_bounds_shapes_precision_and_state_views_are_rejected():
  inputs,initial=fixture(1,7,2,5)
  tensors=[Tensor(x,device='CPU').realize() for x in inputs]
  state=Tensor(initial,device='CPU').realize()
  for start,rows in ((-1,1),(0,0),(0,8),(True,1),(0,True),(0.,1),(0,1.5)):
    with pytest.raises(ValueError):rwkv_scan(*tensors,state,start,rows)
  with pytest.raises(ValueError):rwkv_scan(tensors[0].half(),*tensors[1:],state,0,1)
  with pytest.raises(ValueError):rwkv_scan(*tensors,state.half(),0,1)
  big=Tensor.empty(2,*initial.shape[1:],device='CPU',dtype=dtypes.float).realize()
  with pytest.raises(ValueError):rwkv_scan(*tensors,big[:1],0,1)
  with pytest.raises(ValueError):rwkv_scan(*tensors,state,UOp.variable('bad_start',-1,4).bind(0),1)
  with pytest.raises(ValueError):rwkv_scan(*tensors,state,0,1,prediction_key=tensors[1])
  with pytest.raises(ValueError):rwkv_scan(*tensors,state,0,1,prediction_key=tensors[1],
    prediction_weight=Tensor.ones(2,5,device='CPU',dtype=dtypes.half),
    prediction_norm=Tensor.ones(1,7,2,1,device='CPU'),correction_key=tensors[1])


def test_ordered_normalization_dynamic_rows_reset_and_poisoned_padding():
  batch,capacity,heads,width=1,17,4,16
  rng=np.random.default_rng(783)
  factor=rng.uniform(.5,1.5,(heads,width)).astype(np.float16)
  weight=Tensor(factor,device='CPU').realize()
  state=Tensor.full((batch,heads,width,width),float('nan'),device='CPU').realize()
  expected_state=state.numpy().copy()
  def calculate(r,k,v,decay,rate,raw,start,count):
    weighted=raw*weight.reshape(1,1,heads,width)
    norm=weighted.square().sum(-1,keepdim=True).sqrt().maximum(1e-12)
    unit=weighted/norm
    return rwkv_scan(r,k,v,decay,rate,unit,state,start,count,prediction_key=raw,
      prediction_weight=weight.float(),prediction_norm=norm,correction_key=unit*rate).realize()
  run=TinyJit(calculate)
  for index,(start,rows) in enumerate(((0,17),(17,1),(18,13),(31,17),(0,3),(3,16),(0,17))):
    inputs,_=fixture(batch,capacity,heads,width,seed=784+index)
    raw=rng.normal(0,.2,(batch,capacity,heads,width)).astype(np.float32)
    for x in (*inputs,raw):x[:,rows:]=np.nan
    weighted=raw.astype(np.float64)*factor.astype(np.float64)
    normalized=weighted/np.maximum(np.linalg.norm(weighted,axis=-1,keepdims=True),1e-12)
    expected,expected_state=reference([*inputs[:5],normalized],expected_state,start,rows)
    actual=run(*[Tensor(x,device='CPU').realize() for x in (*inputs[:5],raw)],
      UOp.variable('ordered_rwkv_start',0,100).bind(start),UOp.variable('ordered_rwkv_rows',1,capacity).bind(rows)).numpy()
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-4)
    np.testing.assert_allclose(state.numpy(),expected_state,atol=3e-6,rtol=3e-4)
    assert not actual[:,rows:].any()


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64') or not {4,5,6,7}.issubset(os.sched_getaffinity(0)),
                    reason='Persistent CPU workers require CPU4-7')
def test_pooled_scan_preserves_bits_state_affinity_and_parks():
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  pool=AttentionPool()
  before=os.sched_getaffinity(0)
  try:
    for rows,start in ((17,0),(1,17),(13,18)):
      inputs,initial=fixture(2,17,3,17,seed=774+rows)
      tensors=[Tensor(x,device='CPU').realize() for x in inputs]
      serial_state=Tensor(initial.copy(),device='CPU').realize()
      pool_state=Tensor(initial.copy(),device='CPU').realize()
      options={'prediction_key':tensors[-1],'prediction_weight':Tensor.ones(3,17,device='CPU').realize(),
               'prediction_norm':Tensor.ones(2,17,3,1,device='CPU').realize(),
               'correction_key':(tensors[-1]*tensors[4]).realize()}
      expected=rwkv_scan(*tensors,serial_state,start,rows,**options).numpy()
      calls=pool.lib.pool_calls()
      with pool.prefill():actual=rwkv_scan(*tensors,pool_state,start,rows,pool=pool,**options).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      np.testing.assert_array_equal(pool_state.numpy().view(np.uint32),serial_state.numpy().view(np.uint32))
      assert pool.lib.pool_calls()==calls+1 and pool.lib.pool_parked()
      assert os.sched_getaffinity(0)==before
  finally:pool.close()


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64') or not {4,5,6,7}.issubset(os.sched_getaffinity(0)),
                    reason='Persistent CPU workers require CPU4-7')
@pytest.mark.parametrize('capacity,spin',[(1,0),(1,1),(17,0),(17,1)])
def test_decode_pool_policy_preserves_changed_input_bits_and_parks_on_exit(capacity,spin):
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  pool=AttentionPool()
  before=os.sched_getaffinity(0)
  _,initial=fixture(1,capacity,4,17)
  serial_state=Tensor(initial.copy(),device='CPU').realize()
  pool_state=Tensor(initial.copy(),device='CPU').realize()
  def scan(state,operands,start,workers=None):
    options={'prediction_key':operands[-1],'prediction_weight':Tensor.ones(4,17,device='CPU'),
             'prediction_norm':Tensor.ones(1,capacity,4,1,device='CPU'),
             'correction_key':operands[-1]*operands[4]}
    return rwkv_scan(*operands,state,start,capacity,pool=workers,**options).realize()
  run=TinyJit(lambda *operands:scan(pool_state,operands[:-1],operands[-1],pool))
  try:
    with patch.dict(os.environ,{'ONPU_RWKV_DECODE_SPIN':str(spin)}):
      getenv.cache_clear()
      for index in range(4):
        inputs,_=fixture(1,capacity,4,17,seed=890+index)
        operands=[Tensor(x,device='CPU').realize() for x in inputs]
        start=UOp.variable('rwkv_pool_policy_start',0,100).bind(index*capacity)
        expected=scan(serial_state,operands,start).numpy()
        calls=pool.lib.pool_calls()
        with pool:
          actual=run(*operands,start).numpy()
          assert bool(pool.lib.pool_parked()) is not (bool(spin) and capacity==1)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_array_equal(pool_state.numpy().view(np.uint32),serial_state.numpy().view(np.uint32))
        assert pool.lib.pool_calls()==calls+1 and pool.lib.pool_parked()
        assert os.sched_getaffinity(0)==before
  finally:
    getenv.cache_clear()
    pool.close()


@pytest.mark.parametrize('capacity,width,rows,start',[(128,64,128,0),(128,64,17,128),(1,64,1,128),
                                                    (1,64,1,0),(17,63,1,128),(7,16,5,0)])
def test_specialized_ordered_scan_preserves_full_output_state_and_padding(capacity,width,rows,start):
  heads=32 if capacity==128 and rows==128 else 4
  inputs,initial=fixture(1,capacity,heads,width,seed=20261101)
  rng=np.random.default_rng(20261102)
  raw=Tensor(rng.normal(0,.3,(1,capacity,heads,width)).astype(np.float32),device='CPU').realize()
  weight=Tensor(rng.uniform(.5,1.5,(heads,width)).astype(np.float16),device='CPU').float().realize()
  weighted=raw*weight.reshape(1,1,heads,width)
  norm=weighted.square().sum(-1,keepdim=True).sqrt().maximum(1e-12).realize()
  unit=(weighted/norm).realize()
  operands=[Tensor(x,device='CPU').realize() for x in inputs[:5]]
  results=[]
  for specialize in (False,True):
    state=Tensor(initial.copy(),device='CPU').realize()
    result=rwkv_scan(*operands,unit,state,start,rows,prediction_key=raw,prediction_weight=weight,
                     prediction_norm=norm,correction_key=unit*operands[4],specialize=specialize).numpy().copy()
    results.append((result.tobytes(),state.numpy().tobytes()))
    assert not result[:,rows:].any()
  assert results[0]==results[1]
