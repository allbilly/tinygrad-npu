"""CPU-only numerical, ABI and eligibility checks for parallel GDN preparation."""
import ctypes,hashlib
from dataclasses import replace
import numpy as np
import pytest
from tinygrad import Device,dtypes
from tinygrad.device import TinyELF
from tinygrad.llm.kernels import rockchip_prepare_pool as preparation
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import Ops,KernelInfo,ProgramInfo,UOp
from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock

SOURCE='''void prep_fixture(float* restrict data0_128,const float* restrict data1_1024,const int seed) {
 float scratch[4];
 for (int row = 0; row < 32; row++) {
  for(int j=0;j<4;j++)scratch[j]=0;
  for(int k=0;k<8;k++)for(int j=0;j<4;j++){
   float v=data1_1024[row*32+k*4+j];scratch[j]=scratch[j]+v*v+seed*.25f;
  }
  for(int j=0;j<4;j++)data0_128[row*4+j]=scratch[j];
 }
}
'''


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


def test_parallel_private_reduction_scratch_scalar_abi_and_parked_fallback_are_exact(state):
  renderer=_renderer()
  transformed,record=preparation.split_preparation_source(SOURCE,state.address)
  original=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(SOURCE),'prep_fixture',renderer.target,()))
  parallel=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(transformed),'prep_fixture',renderer.target,()))
  rng=np.random.default_rng(3516)
  for seed in (0,17,-2,0):
    data=rng.normal(0,.2,1024).astype(np.float32)
    expected=np.full(128,np.nan,dtype=np.float32)
    actual=np.full(128,np.nan,dtype=np.float32)
    original.fxn(ctypes.c_void_p(expected.ctypes.data),ctypes.c_void_p(data.ctypes.data),ctypes.c_int(seed))
    assert np.isfinite(expected).all()
    with state.scope():
      parallel.fxn(ctypes.c_void_p(actual.ctypes.data),ctypes.c_void_p(data.ctypes.data),ctypes.c_int(seed))
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    actual[:]=np.nan
    parallel.fxn(ctypes.c_void_p(actual.ctypes.data),ctypes.c_void_p(data.ctypes.data),ctypes.c_int(seed))
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert state.active_flag.value==0 and state.pool.lib.pool_parked()==1
  assert record['outer_axis']=='row' and record['outer_bound']==32 and record['workers']==4


def test_only_approved_cpu_sources_with_separate_outputs_are_rewritten(state,monkeypatch):
  source=('void approved(float* restrict data0_32,const float* restrict data1_32) {\n'
          ' for (int row = 0; row < 32; row++) {data0_32[row]=data1_32[row]*2;}\n}\n')
  out,inp=(UOp.param(i,dtypes.float,(32,),device='CPU') for i in range(2))
  renderer=_renderer()
  sink=UOp.sink(out,inp,arg=KernelInfo('approved'))
  # Generic CPU kernels retain the device's default renderer selector, unlike
  # our explicit CLANG helper factories. Exercise that real metadata here.
  info=replace(ProgramInfo.from_sink(sink,Device['CPU'].renderer.target),outs=(0,),ins=(1,))
  program=UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp)),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))
  linear=UOp(Ops.LINEAR,src=(program,))
  assert preparation.rewrite_preparation_programs(linear,state) is linear
  monkeypatch.setattr(preparation,'_QUALIFIED_SOURCES',frozenset((hashlib.sha256(source.encode()).hexdigest(),)))
  result=preparation.rewrite_preparation_programs(linear,state)
  assert result is not linear and result.src[0].arg==program.arg
  assert next(u.arg for u in result.src[0].src if u.op is Ops.SOURCE)!=source
  count=len(state.preparation_changes)
  assert preparation.rewrite_preparation_programs(linear,state)==result and len(state.preparation_changes)==count
  inplace=linear.replace(src=(program.replace(arg=replace(info,ins=(0,1))),))
  assert preparation.rewrite_preparation_programs(inplace,state) is inplace


@pytest.mark.parametrize('address',[0,-1,True,2**64])
def test_invalid_dispatchers_rejected(address):
  with pytest.raises(ValueError,match='dispatcher'):preparation.split_preparation_source(SOURCE,address)


def test_unreviewed_loop_geometry_rejected():
  with pytest.raises(ValueError,match='outer loop'):
    preparation.split_preparation_source(SOURCE.replace('row < 32','row < 8'),1234)


def test_generic_cpu_jit_metadata_and_execution_survive_program_rewrite(state,monkeypatch):
  from tinygrad import Tensor,TinyJit
  from tinygrad.engine.jit import CapturedJit
  run=TinyJit(lambda x:(x*x).sum(-1).realize())
  rng=np.random.default_rng(12816)
  inputs=[Tensor(rng.normal(0,.1,(32,128)).astype(np.float32),device='CPU').realize() for _ in range(5)]
  for x in inputs[:3]:run(x)
  original=run.captured
  programs=[p for p in original._linear.toposort(enter_calls=True) if p.op is Ops.PROGRAM]
  assert len(programs)==1
  source=next(u.arg for u in programs[0].src if u.op is Ops.SOURCE)
  monkeypatch.setattr(preparation,'_QUALIFIED_SOURCES',frozenset((hashlib.sha256(source.encode()).hexdigest(),)))
  rewritten=preparation.rewrite_preparation_programs(original._linear,state)
  assert rewritten is not original._linear
  parallel=CapturedJit(original.ret,rewritten,original.expected_names,original.expected_input_info)
  for x in inputs[3:]:
    run.captured=original
    expected=run(x).numpy().copy()
    run.captured=parallel
    with state.scope():actual=run(x).numpy().copy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('bound',(2,16,512,1024))
def test_decode_approval_is_separate_and_phase_records_do_not_alias(state,monkeypatch,bound):
  source=(f'void decode_fixture(float* restrict data0_{bound},const float* restrict data1_{bound}) {{\n'
          f' for (int row = 0; row < {bound}; row++) {{data0_{bound}[row]=data1_{bound}[row]*2;}}\n}}\n')
  digest=hashlib.sha256(source.encode()).hexdigest()
  out,inp=(UOp.param(i,dtypes.float,(bound,),device='CPU') for i in range(2))
  renderer=_renderer()
  sink=UOp.sink(out,inp,arg=KernelInfo('decode_fixture'))
  info=replace(ProgramInfo.from_sink(sink,Device['CPU'].renderer.target),outs=(0,),ins=(1,))
  program=UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp)),UOp(Ops.SOURCE,arg=source),
                                    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))
  linear=UOp(Ops.LINEAR,src=(program,))
  monkeypatch.setattr(preparation,'_QUALIFIED_DECODE_SOURCES',frozenset((digest,)))
  monkeypatch.setattr(preparation,'_QUALIFIED_SOURCES',frozenset())
  assert preparation.rewrite_preparation_programs(linear,state) is linear
  rewritten=preparation.rewrite_preparation_programs(linear,state,phase='decode')
  assert rewritten is not linear
  monkeypatch.setattr(preparation,'_QUALIFIED_SOURCES',frozenset((digest,)))
  preparation.rewrite_preparation_programs(linear,state,phase='prefill')
  assert {r['phase'] for r in state.preparation_changes if r['source_before_sha256']==digest}=={'prefill','decode'}
  original=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'decode_fixture',renderer.target,()))
  transformed=next(u.arg for u in rewritten.src[0].src if u.op is Ops.BINARY)
  parallel=Device['CPU'].runtime(TinyELF(transformed,'decode_fixture',renderer.target,()))
  values=np.linspace(-3,3,bound,dtype=np.float32)
  expected=np.full(bound,np.nan,dtype=np.float32)
  actual=expected.copy()
  original.fxn(ctypes.c_void_p(expected.ctypes.data),ctypes.c_void_p(values.ctypes.data))
  with state.scope():parallel.fxn(ctypes.c_void_p(actual.ctypes.data),ctypes.c_void_p(values.ctypes.data))
  np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_scalar_scope_releases_shared_workers_if_helper_entry_fails(state):
  import weakref
  from types import SimpleNamespace
  from tinygrad.llm.kernels.rockchip_decode import ScalarDecoder
  class FailingHelper:
    def __enter__(self):raise RuntimeError('helper startup failed')
    def __exit__(self,*args):pass
  model=SimpleNamespace(max_context=512,config=SimpleNamespace(vocab_size=128),_shared_host_pool=weakref.ref(state))
  decoder=ScalarDecoder(model,None,pool=state.pool,helper=FailingHelper())
  from tinygrad.helpers import Context
  with Context(ONPU_DECODE_HOST_WFE=1),pytest.raises(RuntimeError,match='helper startup failed'):
    with decoder.scope():pass
  assert state.pool.lib.pool_wait_mode()==int(state.pool.event_wait)
  assert state.active_flag.value==0 and not state.pool.active and state.pool.lib.pool_parked()==1
  with state.scope():assert state.active_flag.value==1

@pytest.mark.parametrize('digest',sorted(preparation._QUALIFIED_INNER_HEAD_SOURCES))
def test_reviewed_inner_head_partition_preserves_complete_output_bits(state,digest):
  from pathlib import Path
  source=(Path(__file__).parent/'fixtures'/'gdn_inner_heads'/(digest+'.c')).read_text()
  assert hashlib.sha256(source.encode()).hexdigest()==digest
  updated,record=preparation.split_preparation_source(source,state.address)
  assert record['outer_axis']=='Lidx0_1' and record['outer_bound']==16
  assert 'Lidx0_0 = 0; Lidx0_0 < 2;' in updated
  renderer=_renderer()
  programs=[Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(src),record['kernel'],renderer.target,()))
            for src in (source,updated)]
  rng=np.random.default_rng(350416)
  for case in range(6):
    x=rng.normal(0,.2,1073152).astype(np.float32)
    weight=rng.normal(0,.1,32768).astype(np.float32)
    norm=rng.uniform(.1,2,2048).astype(np.float32)
    if case==1:
      x[:]=0
      weight[::3]=-0.
    if case==2:
      x*=20
      weight*=20
    if case==3:
      weight[::3]=np.nextafter(np.float32(0),np.float32(1))
      weight[1::3]=-0.
    if case==4:
      weight[::131]=np.nan
      weight[::137]=np.inf
      weight[::139]=-np.inf
    if case==5:
      x[-8192:]=0
      norm[:]=1
    expected=np.full(524288,np.nan,np.float32)
    actual=expected.copy()
    tail=(ctypes.c_void_p(x.ctypes.data),ctypes.c_void_p(weight.ctypes.data),ctypes.c_void_p(norm.ctypes.data))
    programs[0].fxn(ctypes.c_void_p(expected.ctypes.data),*tail)
    for active in (True,False,True):
      actual[:]=np.nan
      if active:
        with state.scope():programs[1].fxn(ctypes.c_void_p(actual.ctypes.data),*tail)
      else:programs[1].fxn(ctypes.c_void_p(actual.ctypes.data),*tail)
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_unreviewed_nested_loop_retains_outer_axis():
  source=('void unknown(float* restrict data0_32,const float* restrict data1_32) {\n'
          ' for (int repeat = 0; repeat < 2; repeat++) {\n'
          '  for (int head = 0; head < 16; head++) {data0_32[repeat*16+head]=data1_32[head];}\n }\n}\n')
  _,record=preparation.split_preparation_source(source,1234)
  assert record['outer_axis']=='repeat' and record['outer_bound']==2
