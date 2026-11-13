"""Exact reviewed RWKV row programs, including masked rows and overlap fallback."""
from pathlib import Path
import numpy as np
import pytest
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_prepare_pool import split_rwkv_source,_RWKV_POINTWISE_SOURCES
from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock


@pytest.fixture(scope='module')
def pool():
  _needs_pool()
  with _pool_test_lock():
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    instance=AttentionPool()
    try:yield instance
    finally:
      instance.close()
      assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)


@pytest.mark.parametrize('digest,vector_decay',[(digest,False) for digest in _RWKV_POINTWISE_SOURCES]+
                         [('d1defdd357786b4f9f6d99e94a876bbb5bdeac0f0bbb18957e78f9cb283b8528',True)])
def test_original_row_math_and_runtime_overlap_guard(pool,digest,vector_decay):
  source=(Path(__file__).parent/'fixtures/rwkv7_pointwise'/(digest+'.c')).read_text()
  name=_RWKV_POINTWISE_SOURCES[digest]
  updated,args=split_rwkv_source(source,name,pool.address)
  if vector_decay:
    from tinygrad.llm.kernels.rockchip_rwkv_activations import vector_decay_source
    updated=vector_decay_source(source,name,pool.address)
  renderer=_renderer()
  programs=[Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(text),name,renderer.target,())) for text in (source,updated)]
  rng=np.random.default_rng(20261101)
  arrays=[]
  for typ,var in args:
    if '*' not in typ:
      arrays.append(None)
      continue
    dtype=np.float16 if '__fp16' in typ else np.uint8 if 'char' in typ else np.int32 if 'int' in typ else np.float32
    count=int(var.rsplit('_',1)[1])
    arrays.append(rng.uniform(.02,.1,count).astype(dtype))
  def run(program,rows):program(*(a.ctypes.data if a is not None else rows for a in arrays))
  with pool:
    for rows in (1,17,128):
      for mode in ('finite','nonfinite','exponent_boundary','extreme_bias') if vector_decay else ('finite','nonfinite'):
        if mode=='nonfinite':
          for a in arrays[1:]:
            if a is not None and a.dtype==np.float32:
              a[0],a[-1]=np.nan,np.inf
        if mode=='exponent_boundary':
          arrays[1][:]=0
          arrays[2][:]=0
          magnitude=np.float32(125/1.4426950216293335)
          values=np.array([magnitude,-magnitude,np.nextafter(magnitude,np.float32(0)),
                           np.nextafter(-magnitude,np.float32(-np.inf)),1e-40,-1e-40],np.float32)
          arrays[1][:]=np.resize(values,arrays[1].size)
        if mode=='extreme_bias':
          arrays[2][:]=np.resize(np.array([np.nan,np.inf,-np.inf,65504,-65504,0],np.float16),arrays[2].size)
        arrays[0][:]=np.nan
        run(programs[0],rows)
        expected=arrays[0].copy()
        arrays[0][:]=np.nan
        run(programs[1],rows)
        assert arrays[0].tobytes()==expected.tobytes()
      for a in arrays[1:]:
        if a is not None and np.issubdtype(a.dtype,np.floating):a[:]=rng.uniform(.02,.1,a.size)
    # Aliases keep the original restrict-qualified serial contract. Verify that
    # they never dispatch concurrent writers; no stronger alias math is claimed.
    alias=next((a for a in arrays[1:] if a is not None and a.dtype==np.float32 and a.size>=arrays[0].size),None)
    if alias is not None:
      original=arrays[0]
      arrays[0]=alias[:original.size]
      calls=pool.lib.pool_calls()
      run(programs[1],1)
      assert pool.lib.pool_calls()==calls


def test_unreviewed_source_or_invalid_dispatcher_is_rejected():
  source='void E(float *out,const float *x){out[0]=x[0];}'
  for address in (0,-1,True,2**64,1234):
    with pytest.raises(ValueError):split_rwkv_source(source,'E',address)


def test_vector_decay_rejects_unreviewed_source_and_wrong_abi():
  from tinygrad.llm.kernels.rockchip_rwkv_activations import vector_decay_source
  source=(Path(__file__).parent/'fixtures/rwkv7_pointwise/d1defdd357786b4f9f6d99e94a876bbb5bdeac0f0bbb18957e78f9cb283b8528.c').read_text()
  for text,name,address in ((source+' ','E_32_512_4_4',1),(source,'wrong_name',1),(source,'E_32_512_4_4',True),
                             (source,'E_32_512_4_4',0)):
    with pytest.raises(ValueError):vector_decay_source(text,name,address)
