"""Reviewed CPU preparation rewrites retain bits, layout, and alias fallback."""
import ctypes,functools,hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Device,UOp
from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_exact_prepare import _ROW_SOURCES,rewrite_exact_preparation
from tinygrad.llm.kernels.rockchip_prepare_pool import _split_independent_rows
from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock

FIXTURES=Path(__file__).parent/'fixtures/exact_preparation'


def program(source,name,*,ins=(1,),outs=(0,),target='CPU'):
  sink=UOp.sink(arg=KernelInfo(name))
  info=replace(ProgramInfo.from_sink(sink,Target(target)),outs=outs,ins=ins)
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=b'')))


@functools.cache
def compile_cpu(source,name):
  renderer=_renderer()
  return Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),name,renderer.target,()))


def invoke(executable,*arrays):executable.fxn(*(ctypes.c_void_p(a.ctypes.data) for a in arrays))


@pytest.mark.parametrize('digest',list(_ROW_SOURCES))
def test_independent_rows_preserve_original_bits_and_overlap_fallback(digest):
  _needs_pool()
  with _pool_test_lock():
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    pool=AttentionPool()
    try:
      source=(FIXTURES/'rows'/(digest+'.c')).read_text()
      name=_ROW_SOURCES[digest]
      assert hashlib.sha256(source.encode()).hexdigest()==digest
      _,args=_split_independent_rows(source,name,pool.prefill_address)
      original=program(source,name)
      linear=UOp(Ops.LINEAR,src=(original,))
      records=[]
      updated=rewrite_exact_preparation(linear,pool=pool,rows=True,records=records)
      assert updated is not linear and len(records)==1 and records[0]['kind']=='independent_rows'
      text=next(u.arg for u in updated.src[0].src if u.op is Ops.SOURCE)
      functions=[compile_cpu(s,name) for s in (source,text)]
      rng=np.random.default_rng(391)
      values=[rng.uniform(.5,1.5,int(var.rsplit('_',1)[1])).astype(np.float16 if '__fp16' in typ else np.float32) for typ,var in args]
      for case in ('finite','nonfinite'):
        if case=='nonfinite':
          values[1][::257]=np.nan
          values[1][1::257]=np.inf
        outputs=[]
        before=int(pool.lib.pool_calls())
        for function in functions:
          out=np.full_like(values[0],np.nan)
          invoke(function,out,*values[1:])
          outputs.append(out)
        assert outputs[0].tobytes()==outputs[1].tobytes()
        assert int(pool.lib.pool_calls())==before+1
      # The output aliases the first input: only the original serial function may execute.
      values=[np.ones_like(a) for a in values]
      outputs=[]
      for index,function in enumerate(functions):
        inputs=[a.copy() for a in values[1:]]
        out=inputs[0][:values[0].size]
        before=int(pool.lib.pool_calls())
        invoke(function,out,*inputs)
        outputs.append(out.copy())
        if index:assert int(pool.lib.pool_calls())==before
      assert outputs[0].tobytes()==outputs[1].tobytes()
    finally:pool.close()
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)


def test_sparse_lookup_full_intermediate_and_independent_layout_oracle():
  source=(FIXTURES/'sparse_lookup.c').read_text()
  name='E_64_32_64_4'
  linear=UOp(Ops.LINEAR,src=(program(source,name,ins=(1,2)),))
  records=[]
  updated=rewrite_exact_preparation(linear,sparse_lookup=True,phase='decode',records=records)
  assert updated is not linear and records[0]['kind']=='sparse_lookup'
  text=next(u.arg for u in updated.src[0].src if u.op is Ops.SOURCE)
  functions=[compile_cpu(s,name) for s in (source,text)]
  raw=np.zeros(570425344,np.uint8)
  token=np.zeros(1,np.int32)
  rng=np.random.default_rng(192)
  tokens=[-1,0,1023,1024,2047,2048,3071,3072,4095,4096,65535,131072,262143,262144]
  for value in tokens:
    token[0]=value
    expected=np.zeros(524288,np.float16)
    if 0<=value<262144:
      packed=raw[value*2176:(value+1)*2176].reshape(64,34)
      scales=rng.uniform(-.2,.2,64).astype(np.float16)
      packed[:,:2]=scales.view(np.uint8).reshape(64,2)
      packed[:,2:]=rng.integers(0,256,(64,32),dtype=np.uint8)
      active=(scales[:,None].astype(np.float32)*packed[:,2:].view(np.int8).astype(np.float32)).astype(np.float16)
      indices=(np.arange(64)[:,None]<<13)+(np.arange(32)[None,:]<<8)+((value>>12)<<2)+((value>>10)&3)
      expected[indices]=active
    for function in functions:
      out=np.full(524288,np.nan,np.float16)
      invoke(function,out,token,raw)
      assert out.tobytes()==expected.tobytes(),value
  # Extreme signed quants and nonfinite/subnormal/signed-zero scales retain original bits.
  token[0]=4096
  packed=raw[4096*2176:4097*2176].reshape(64,34)
  packed[:,:2]=np.resize(np.array([0,0x8000,1,0x3c00,0x7bff,0x7c00,0xfc00,0x7e01],np.uint16),64).view(np.uint8).reshape(64,2)
  packed[:,2:]=np.resize(np.array([-128,-1,0,1,127],np.int8),64*32).view(np.uint8).reshape(64,32)
  outputs=[]
  for function in functions:
    out=np.empty(524288,np.float16)
    invoke(function,out,token,raw)
    outputs.append(out)
  assert outputs[0].tobytes()==outputs[1].tobytes()
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)


def test_rewrite_rejects_unreviewed_and_inplace_programs_and_is_default_off():
  source=(FIXTURES/'sparse_lookup.c').read_text()
  name='E_64_32_64_4'
  for text,ins,outs,target,fn in [(source+' ',(1,2),(0,),'CPU',name),(source,(0,1,2),(0,),'CPU',name),
                                  (source,(1,2),(0,1),'CPU',name),(source,(1,2),(0,),'ROCKCHIP',name),
                                  (source,(1,2),(0,),'CPU','wrong_name')]:
    linear=UOp(Ops.LINEAR,src=(program(text,fn,ins=ins,outs=outs,target=target),))
    assert rewrite_exact_preparation(linear,sparse_lookup=True) is linear
  linear=UOp(Ops.LINEAR,src=(program(source,name,ins=(1,2)),))
  assert rewrite_exact_preparation(linear) is linear
  with pytest.raises(ValueError,match='workers'):rewrite_exact_preparation(linear,rows=True)
  for key in ('rows','sparse_lookup'):
    with pytest.raises(ValueError,match='bool'):rewrite_exact_preparation(linear,**{key:1})


def test_row_rewrite_has_prefill_scope():
  digest=next(iter(_ROW_SOURCES))
  source=(FIXTURES/'rows'/(digest+'.c')).read_text()
  linear=UOp(Ops.LINEAR,src=(program(source,_ROW_SOURCES[digest]),))
  assert rewrite_exact_preparation(linear,pool=SimpleNamespace(prefill_address=1),rows=True,phase='decode') is linear
