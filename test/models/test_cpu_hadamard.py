"""CPU-only Hadamard quantization and shared-worker cache regressions."""
import numpy as np
import pytest
from tinygrad import Tensor,Device,dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip_hadamard as h
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.uop.ops import Ops,UOp


def transformed(x,weight=False):
  sign=np.where(np.random.default_rng(1281536+x.shape[1]).integers(0,2,x.shape[1]),np.float32(1),np.float32(-1))
  y=(x*sign).reshape(x.shape[0],-1,128).copy()
  for step in (1,2,4,8,16,32,64):
    z=y.reshape(x.shape[0],-1,2,step)
    a,b=z[:,:,0,:].copy(),z[:,:,1,:].copy()
    z[:,:,0,:],z[:,:,1,:]=a+b,a-b
  return y.reshape(x.shape)*np.float32(1/128 if weight else 1)


@pytest.mark.parametrize('depth',(2048,3072,4096))
@pytest.mark.parametrize('rows',(1,17,128))
@pytest.mark.parametrize('weight',(False,True))
def test_public_rotated_quantizer_matches_independent_float_transform(depth,rows,weight):
  rng=np.random.default_rng(depth+rows)
  x=rng.normal(0,.2,(rows,depth)).astype(np.float32)
  x[:,11::128]*=32
  x[0]=0
  x[0,1::2]=-0.
  with Context(ONPU_INT8_HADAMARD=0):
    expected,scale=quantize(Tensor(transformed(x,weight),device='CPU'),depth)
    expected_bytes,scale_bytes=expected.numpy().tobytes(),scale.numpy().tobytes()
  with Context(ONPU_INT8_HADAMARD=128,ONPU_NORM_QUANT=0):
    actual,actual_scale=quantize(Tensor(x,device='CPU'),depth,weight=weight)
    assert actual.numpy().tobytes()==expected_bytes
    assert actual_scale.numpy().tobytes()==scale_bytes
  assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)


@pytest.mark.parametrize('bad',(-1,64,256))
def test_unqualified_hadamard_block_rejected(bad):
  with Context(ONPU_INT8_HADAMARD=bad),pytest.raises(ValueError,match='Hadamard block'):
    quantize(Tensor.zeros(1,3072,device='CPU'),3072)


def test_hadamard_rejects_incompatible_layout_and_norm_fusion():
  x=Tensor.zeros(1,3072,device='CPU')
  with Context(ONPU_INT8_HADAMARD=128,ONPU_NORM_QUANT=0),pytest.raises(ValueError,match='ordinary layout'):
    quantize(x,3072,sdk_prefill=True)
  with Context(ONPU_INT8_HADAMARD=128,ONPU_NORM_QUANT=1),pytest.raises(ValueError,match='separate CPU normalization'):
    quantize(x,3072)


def test_fixed_signs_match_declared_seed_without_numpy_runtime_dependency():
  for depth in (2048,3072,4096):
    expected=tuple(1 if value else -1 for value in np.random.default_rng(1281536+depth).integers(0,2,depth))
    assert h.signs(depth)==expected


def test_cached_quantizer_applies_current_shared_worker_scope():
  from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
  _needs_pool()
  with _pool_test_lock():
    pool=AttentionPool()
    state=SharedHostPool(pool)
    try:
      args=(UOp.param(0,dtypes.uint8,(128*3072+512,),device='CPU'),
            UOp.param(1,dtypes.float,(128,3072),device='CPU'),UOp.param(2,dtypes.float,(128,),device='CPU'),128,3072,3072,False)
      plain=h._program(*args)
      with state.scope():pooled=h._program(*args)
      plain_source=next(u.arg for u in plain.src if u.op is Ops.SOURCE)
      pooled_source=next(u.arg for u in pooled.src if u.op is Ops.SOURCE)
      assert 'shared_work_callback' not in plain_source and 'shared_work_callback' in pooled_source
      assert h._program(*args)==plain
    finally:
      state.restore()
      pool.close()
