"""CPU-only byte, view and rebinding checks for exact RMSNorm/INT8 fusion."""
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_cpu import rms_norm
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_norm_quant import norm_quant_stats
from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock

@pytest.fixture(scope='module')
def state():
  _needs_pool()
  with _pool_test_lock():
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
    pool=AttentionPool()
    shared=SharedHostPool(pool)
    try:yield shared
    finally:
      shared.restore()
      pool.close()
      assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)


def arrays(seed=179):
  rng=np.random.default_rng(seed)
  return rng.normal(0,.3,(1,128,896)).astype(np.float32),rng.normal(1,.1,896).astype(np.float32)


def payload(x,w,padded,sdk,view=lambda t:t):
  return quantize(view(rms_norm(x,w,1e-6)).reshape(-1,896),padded,sdk_prefill=sdk)


@pytest.mark.parametrize('sdk,padded',[(False,896),(True,896),(False,1024),(True,1024)])
def test_exact_payload_scale_padding_and_nonfinite_contract(state,sdk,padded):
  for kind in ('ordinary','half','zero','overflow','nan_input','inf_input','nan_weight','inf_weight'):
    data,gamma=arrays()
    if kind=='half':gamma=gamma.astype(np.float16)
    if kind=='zero':data[:]=0
    if kind=='overflow':data[0,0]=1e30
    if kind=='nan_input':data[0,7,17]=np.nan
    if kind=='inf_input':data[0,17,3]=np.inf
    if kind=='nan_weight':gamma[17]=np.nan
    if kind=='inf_weight':gamma[3]=np.inf
    x,w=Tensor(data,device='CPU').realize(),Tensor(gamma,device='CPU').realize()
    with state.scope():
      with Context(ONPU_NORM_QUANT=0):expected=payload(x,w,padded,sdk)
      before=norm_quant_stats()['traces']
      with Context(ONPU_NORM_QUANT=1):actual=payload(x,w,padded,sdk)
      assert norm_quant_stats()['traces']==before+1
      for a,b in zip(actual,expected):assert a.numpy().tobytes()==b.numpy().tobytes(),kind


def test_partial_and_reordered_views_keep_the_ordinary_path(state):
  data,gamma=arrays()
  for rows,view in ((17,lambda t:t),(128,lambda t:t.flip(-1))):
    x,w=Tensor(data[:,:rows],device='CPU').realize(),Tensor(gamma,device='CPU').realize()
    with state.scope():
      with Context(ONPU_NORM_QUANT=0):expected=payload(x,w,896,True,view)
      before=norm_quant_stats()['traces']
      with Context(ONPU_NORM_QUANT=1):actual=payload(x,w,896,True,view)
      assert norm_quant_stats()['traces']==before
      for a,b in zip(actual,expected):assert a.numpy().tobytes()==b.numpy().tobytes()


def test_changed_inputs_and_gamma_rebind_through_jit(state):
  run=TinyJit(lambda x,w:payload(x,w,896,True))
  with state.scope():
    for seed in range(6):
      data,gamma=arrays(seed)
      x,w=Tensor(data,device='CPU').realize(),Tensor(gamma,device='CPU').realize()
      with Context(ONPU_NORM_QUANT=0):expected=payload(x,w,896,True)
      with Context(ONPU_NORM_QUANT=1):actual=run(x,w)
      for a,b in zip(actual,expected):assert a.numpy().tobytes()==b.numpy().tobytes()


def test_realized_normalization_keeps_its_snapshot_after_input_mutation(state):
  data,gamma=arrays()
  x,w=Tensor(data,device='CPU').realize(),Tensor(gamma,device='CPU').realize()
  with Context(ONPU_NORM_QUANT=1),state.scope():
    normalized=rms_norm(x,w,1e-6).realize()
    snapshot=normalized.numpy().copy()
    x.assign(Tensor.zeros_like(x)).realize()
    w.assign(Tensor.zeros_like(w)).realize()
    before=norm_quant_stats()['traces']
    actual=quantize(normalized.reshape(128,896),896,sdk_prefill=True)
    expected=quantize(Tensor(snapshot,device='CPU').reshape(128,896),896,sdk_prefill=True)
    assert norm_quant_stats()['traces']==before
    for a,b in zip(actual,expected):assert a.numpy().tobytes()==b.numpy().tobytes()


@pytest.mark.parametrize('sdk',[False,True])
def test_full_write_proof_requires_the_exact_packed_extent(sdk):
  from types import SimpleNamespace
  from tinygrad import UOp, dtypes
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.kernels.rockchip_norm_quant import _program
  from tinygrad.llm.kernels.rockchip_replay import _full_write
  size=512+128*896
  params=[UOp.param(i,dtype,(count,)) for i,(dtype,count) in enumerate(
    ((dtypes.uint8,size),(dtypes.float,128*896),(dtypes.float,896),(dtypes.float,128)))]
  program=_program(*params,128,896,896,sdk,1e-6,False,1234)
  call=SimpleNamespace(body=program,device='CPU')
  assert program.op is Ops.PROGRAM and _full_write(call,0,size)
  assert not _full_write(call,0,size+64) and not _full_write(call,3,512)
