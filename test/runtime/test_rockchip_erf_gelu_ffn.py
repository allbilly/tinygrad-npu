"""Raw erf-GELU FFN packing matches independent Tensor activation quantization."""
import os
import unittest
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_gelu_erf import gated_erf_gelu
from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_erf_projection, payload_layout, RockchipInt8ErfGeluFeedForward
from tinygrad.llm.kernels.rockchip_int8 import (dequantize_tiles, quantize, RockchipInt8Linear,
                                                RockchipInt8SplitLinear)
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.uop.ops import Ops
from test.runtime.test_rockchip_gelu_ffn import fixture


def _reference(projected,active,hidden,depths):
  joint=projected[:active].reshape(1,active,hidden*2).contiguous().realize()
  activation=(joint[:,:,:hidden].gelu(approximate='none')*joint[:,:,hidden:]).reshape(active,hidden)
  activation=activation.pad(((0,projected.shape[0]-active),(0,0))).contiguous().realize()
  payloads,scales=[],[]
  column=0
  for depth,(_,padded,row_tile,_) in zip(depths,payload_layout(projected.shape[0],depths)):
    for first in range(0,projected.shape[0],row_tile):
      packet,scale=quantize(activation[first:first+row_tile,column:column+depth].contiguous(),padded)
      payloads.append(packet)
      scales.append(scale)
    column+=depth
  payload,headers=Tensor.cat(*payloads),Tensor.cat(*scales)
  payload.realize(headers)
  return payload.numpy().copy(),headers.numpy().view(np.uint32).copy()


def _poison_inactive(scales,input_tile,active):
  for index,scale in enumerate(scales):
    first=index*input_tile
    local=max(0,active-first)
    if local<scale.shape[0]:
      values=scale.numpy().copy()
      values[local:]=np.nan
      scales[index]=Tensor(values,device='CPU').realize()


@pytest.mark.parametrize('rows,active,bad_weight',[(1,1,False),(128,1,False),(128,65,False),(128,128,False),(128,65,True)])
def test_erf_pack_fully_overwrites_poison_before_opt_in_copy_alias(rows,active,bad_weight):
  import ctypes
  from unittest.mock import patch
  from tinygrad import Device,dtypes
  from tinygrad.device import TinyELF
  from tinygrad.helpers import Context,getenv
  from tinygrad.runtime.ops_cpu import CPUProgram
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  from tinygrad.llm.kernels.rockchip_gelu_ffn import _program
  from tinygrad.llm.kernels.rockchip_replay import _full_write

  hidden,depths=6656,(4096,2560)
  raw,scales,sw,tiles,projected=fixture(rows,hidden,rows,seed=75500+active+rows)
  if bad_weight:
    weights=sw.numpy().copy()
    weights[0]=np.inf
    sw=Tensor(weights,device='CPU').realize()
    projected=_dense_dequant(raw,scales,sw,tiles,rows,hidden)
  _poison_inactive(scales,rows,active)
  expected=_reference(projected,active,hidden,depths)
  values=[x.numpy().copy() for x in (*raw,*scales,sw)]
  size=len(expected[0])
  definitions=[(dtypes.uint8,size),*((dtypes.int,x.size) for x in values[:len(raw)]),
               *((dtypes.float,x.size) for x in values[len(raw):]),(dtypes.int,1),(dtypes.float,rows*len(depths))]
  params=[UOp.param(i,dt,(n,),device='CPU') for i,(dt,n) in enumerate(definitions)]
  program=_program(params[0],tuple(params[1:1+len(raw)]),tuple(params[1+len(raw):1+len(raw)+len(scales)]),
                   params[-3],params[-2],params[-1],rows,hidden,depths,tuple(tiles),True,'gelu')
  binary=next(x.arg for x in program.src if x.op is Ops.BINARY)
  runtime=CPUProgram(Device['CPU'],TinyELF(binary,program.src[0].arg.function_name,_renderer().target,()))
  count=ctypes.c_int(active)
  for poison in (0xa5,0x5a):
    packet=np.full(size,poison,np.uint8)
    headers=np.full(rows*len(depths),poison*0x01010101,np.uint32)
    runtime(packet.ctypes.data,*[x.ctypes.data for x in values],ctypes.addressof(count),headers.ctypes.data)
    np.testing.assert_array_equal(packet,expected[0])
    np.testing.assert_array_equal(headers,expected[1])
  call=program.call(*(UOp.new_buffer('CPU',n,dt) for dt,n in definitions))
  try:
    with patch.dict(os.environ,{'ONPU_FFN_PACKED_UPLOAD':'1'}):
      getenv.cache_clear()
      with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=0):assert not _full_write(call,0,size)
      with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=1):
        assert _full_write(call,0,size)
        assert not _full_write(call,0,size+64)
        assert not _full_write(call,1,values[0].nbytes)
  finally:getenv.cache_clear()


@pytest.mark.parametrize('rows,hidden,input_tile,actives',[
  (1,6656,1,(1,)),(17,6656,17,(1,17)),(65,6656,64,(1,17,65)),(128,6656,64,(1,17,65,128)),
  (1,10240,1,(1,)),(17,10240,17,(1,17)),(65,10240,64,(1,17,65)),(128,10240,64,(1,17,65,128)),
])
def test_erf_projection_packet_and_scale_bits_match_tensor_reference(rows,hidden,input_tile,actives):
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  for seed,active in enumerate(actives):
    raw,scales,sw,tiles,projected=fixture(rows,hidden,input_tile,seed=8800+hidden+seed)
    _poison_inactive(scales,input_tile,active)
    expected=_reference(projected,active,hidden,depths)
    packet,headers=pack_erf_projection(raw,scales,sw,tiles,rows,hidden,depths,active)
    packet.realize(headers)
    np.testing.assert_array_equal(packet.numpy(),expected[0])
    np.testing.assert_array_equal(headers.numpy().view(np.uint32),expected[1])


@pytest.mark.parametrize('hidden',[6656,10240])
def test_erf_projection_dynamic_active_rows_rebind_poisoned_tails(hidden):
  rows,input_tile=128,64
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  raw,scales,sw,tiles,projected=fixture(rows,hidden,input_tile,seed=9900+hidden)
  active=UOp.variable(f'erf_pack_rows_{hidden}',1,rows)
  def run(*args):
    nraw,nscales=len(raw),len(scales)
    result=pack_erf_projection(list(args[:nraw]),list(args[nraw:nraw+nscales]),args[nraw+nscales],tiles,
                               rows,hidden,depths,args[-1])
    result[0].realize(result[1])
    return result
  jit=TinyJit(run)
  for step,count in enumerate((17,128,1,65,17)):
    if step:
      next_raw,next_scales,next_sw,next_tiles,projected=fixture(rows,hidden,input_tile,seed=9910+hidden+step)
      assert next_tiles==tiles
    else: next_raw,next_scales,next_sw=raw,scales,sw
    expected=_reference(projected,count,hidden,depths)
    _poison_inactive(next_scales,input_tile,count)
    actual=jit(*next_raw,*next_scales,next_sw,active.bind(count))
    np.testing.assert_array_equal(actual[0].numpy(),expected[0])
    np.testing.assert_array_equal(actual[1].numpy().view(np.uint32),expected[1])
  assert jit.captured is not None


def _dense_dequant(raw,scales,weights,tiles,rows,hidden):
  rows_out=[]
  for first in sorted({tile.first_row for tile in tiles}):
    fragments=[]
    for index,tile in enumerate(tiles):
      if tile.first_row!=first: continue
      fragments.append(dequantize_tiles([raw[index]],[scales[tile.scale_index]],
        weights[tile.first_column:tile.first_column+tile.columns],[tile.rows],tile.columns))
    rows_out.append(Tensor.cat(*fragments,dim=1))
  return Tensor.cat(*rows_out,dim=0).reshape(rows,hidden*2).realize()


def test_erf_projection_nonfinite_rows_have_nan_scales_and_zero_codes():
  rows,hidden,input_tile=17,6656,17
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  raw,scales,sw,tiles,_=fixture(rows,hidden,input_tile,seed=733)
  weights=sw.numpy().copy()
  weights[:]=np.inf
  bad_weights=Tensor(weights,device='CPU').realize()
  projected=_dense_dequant(raw,scales,bad_weights,tiles,rows,hidden)
  expected=_reference(projected,rows,hidden,depths)
  packet,headers=pack_erf_projection(raw,scales,bad_weights,tiles,rows,hidden,depths,rows)
  packet.realize(headers)
  np.testing.assert_array_equal(packet.numpy(),expected[0])
  np.testing.assert_array_equal(headers.numpy().view(np.uint32),expected[1])
  assert np.isnan(headers.numpy()).any()
  assert np.count_nonzero(packet.numpy()[round_up(rows*4,64):round_up(rows*4,64)+depths[0]])==0


def _phase(hook,phase):
  for layer in (*hook.projections.layers,hook.down): layer.phase=phase
  for part in getattr(hook.down,'parts',[]): part.phase=phase


@pytest.mark.skipif(os.environ.get('ROCKCHIP_TEST')!='1',reason='explicit exclusive NPU FFN test')
def test_npu_erf_ffn_matches_multistage_and_native_replay():
  from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
  rng=np.random.default_rng(20261028)
  for input_width,hidden in ((2048,6656),(2560,10240)):
    sources=[]
    for inputs,outputs in ((input_width,hidden),(input_width,hidden),(hidden,input_width)):
      layer=nn.Linear(inputs,outputs,bias=False)
      layer.weight=Tensor(rng.normal(0,.025,(outputs,inputs)).astype(np.float16),device='CPU').realize()
      sources.append(layer)
    members=tuple(RockchipInt8Linear(layer,'on','on',24576) for layer in sources[:2])
    projections=RockchipProjections(members,24576,share_packed=True)
    down=RockchipInt8SplitLinear(sources[2],'on','on',24576)
    down.prepare()
    fused=RockchipInt8ErfGeluFeedForward(projections,down,decode=True)

    def baseline(x):
      gate,up=projections(x)
      return down(gated_erf_gelu(gate,up)).realize()

    for rows in (1,17,128):
      phase='decode' if rows==1 else 'prefill'
      _phase(fused,phase)
      control=TinyJit(baseline)
      candidate=TinyJit(lambda x:fused(x).realize())
      for step in range(7):
        if step==1: values=np.zeros((1,rows,input_width),dtype=np.float32)
        elif step==2: values=np.where(np.arange(input_width)%2,.125,-.125).astype(np.float32).reshape(1,1,input_width).repeat(rows,axis=1)
        else: values=rng.normal(0,.2,(1,rows,input_width)).astype(np.float32)
        x=Tensor(values,device='CPU').realize()
        assert fused.enabled(x)
        expected=control(x).realize().numpy().copy()
        actual=candidate(x).realize().numpy().copy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        if step==2:
          control.captured=RockchipCapturedJit(control.captured)
          candidate.captured=RockchipCapturedJit(candidate.captured)
      assert candidate.captured.reason is None and candidate.captured.native_runs>=2
      pack_calls=[call for call in candidate.captured.native.calls if call.body.op is Ops.PROGRAM
        and call.body.src[0].arg.function_name=='rockchip_int8_erf_gelu_pack']
      assert len(pack_calls)==1


if __name__=='__main__': unittest.main()


@pytest.mark.parametrize('sdk',[False,True])
@pytest.mark.parametrize('active',[1,17,65,128])
def test_erf_row_batch_preserves_packet_bits_and_requires_sdk_layout(monkeypatch,sdk,active):
  from tinygrad.llm.kernels import rockchip_gelu_ffn as module
  from tinygrad.helpers import ceildiv
  raw,scales,sw,tiles,projected=fixture(128,6656,128,seed=9200+active)
  expected=_reference(projected,active,6656,(4096,2560))
  _poison_inactive(scales,128,active)
  if sdk:
    converted=[]
    for value,tile in zip(raw,tiles):
      values=value.numpy()
      width=round_up(ceildiv(tile.columns,3),32)
      parts=[(i,min(width,tile.columns-i)) for i in range(0,tile.columns,width)]
      if len(parts)!=3:parts=[(0,tile.columns)]
      converted.append(Tensor(np.concatenate([values[i*128:(i+n)*128].reshape(128,n//4,4).transpose(1,0,2).copy().ravel()
                                              for i,n in parts]),device='CPU').realize())
    raw,tiles=converted,[tile._replace(sdk=True) for tile in tiles]
  getenv=module.getenv
  monkeypatch.setattr(module,'getenv',lambda key,default=0:4 if key=='ONPU_SWIGLU_PREFILL_ROWS' else getenv(key,default))
  program=module._program
  recorded=[]
  def record(*args):
    result=program(*args)
    source=next(u.arg for u in result.src if u.op is Ops.SOURCE)
    assert ('float batch_values[4][' in source)==sdk
    import inspect
    recorded.append(inspect.signature(program).bind(*args).arguments['row_batch'])
    return result
  monkeypatch.setattr(module,'_program',record)
  packet,headers=pack_erf_projection(raw,scales,sw,tiles,128,6656,(4096,2560),active)
  packet.realize(headers)
  assert recorded==[4 if sdk else 1]
  np.testing.assert_array_equal(packet.numpy(),expected[0])
  np.testing.assert_array_equal(headers.numpy().view(np.uint32),expected[1])
