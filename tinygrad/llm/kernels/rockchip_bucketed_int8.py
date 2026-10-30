"""Experimental grouped INT8 matrices with dynamic active rows per expert."""
from __future__ import annotations
import functools,struct
from dataclasses import replace
from tinygrad import Tensor,dtypes
from tinygrad.helpers import Target,round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_grouped_int8 import (_remainder,_dequantize_activation_residual,
                                                    _dequantize_weight_residual,_check_weight_residual)
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.runtime.support.rockchip_int8 import supported_depth
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo,UOp


@functools.cache
def _program(out:UOp,lhs:UOp,rhs:UOp,groups:int,capacity:int,columns:int,depth:int,cores:int,rows:int|None) -> UOp:
  sink=UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_bucketed_int8_{groups}_{capacity}_{columns}_{depth}_{cores}_{rows}'))
  info=replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
    UOp(Ops.SOURCE,arg=f'Bucketed INT8 {groups} experts, row capacity {capacity}, K={depth}, N={columns}'),
    UOp(Ops.BINARY,arg=struct.pack('<8I' if rows is None else '<9I',capacity,columns,depth,cores,groups,
      min(capacity,128 if depth<=2048 else 64),0x47524d4d,
      4 if rows is None else 5 if capacity<=(128 if depth<=2048 else 64) else 6,*(() if rows is None else (rows,))))))


def raw_matmul(payload:Tensor,weight:Tensor,capacity:int,columns:int,cores:int=3,*,rows:int|None=None) -> Tensor:
  """Payload: G*M FP32 scales, G int32 counts, 64-byte padding, G*M INT8 rows.

  With rows=R, payload and output use R consecutive rows grouped by expert.
  Counts sum to at most R. Otherwise each expert reserves M row slots. Counts
  are in [0,M], nonempty; inactive rows and unused compact tails are unspecified.
  """
  if (weight.ndim!=5 or weight.shape[-2:]!=(32,32) or weight.dtype is not dtypes.int8
      or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP'
      or payload.device!='CPU' or payload.dtype is not dtypes.uint8 or payload.ndim!=1):
    raise ValueError('Invalid bucketed INT8 packed operands')
  groups,n,k=weight.shape[0],weight.shape[1]*32,weight.shape[2]*32
  assert isinstance(groups,int) and isinstance(n,int) and isinstance(k,int)
  if type(capacity) is not int:raise ValueError('Unsupported bucketed INT8 row capacity')
  if rows is not None and (type(rows) is not int or not 1<=rows<=groups*capacity):
    raise ValueError('Unsupported compact expert row storage')
  storage=groups*capacity if rows is None else rows
  if (not 1<=capacity<=(128 if k<=2048 else 64)*(2 if rows is not None else 1) or not 1<=groups<=128
      or not 0<columns<=n<=4096 or round_up(columns,32)!=n or not supported_depth(k) or cores not in (1,3)
      or payload.numel()!=round_up((storage+groups)*4,64)+storage*k):
    raise ValueError('Unsupported bucketed INT8 matrix geometry')
  result=Tensor.empty(storage*n,device=weight.device,dtype=dtypes.int32).custom_kernel(payload.to(weight.device),weight,
    fxn=lambda o,a,w:_program(o,a,w,groups,capacity,n,k,cores,rows))[0]
  return result.to('CPU').reshape((groups,capacity,n) if rows is None else (rows,n))


@functools.cache
def _expert_ids(out:UOp,counts:UOp,groups:int,rows:int,capacity:int) -> UOp:
  source=f'''void rockchip_compact_expert_ids(int *out,const int *counts) {{
    int used=0;
    for(int g=0;g<{groups};g++){{if((unsigned)counts[g]>{capacity}U){{used=-1;break;}}
      used+=counts[g];if(used>{rows}){{used=-1;break;}}}}
    for(int i=0;i<{rows};i++)out[i]=0;
    if(used<0)return;
    int slot=0;for(int g=0;g<{groups};g++)for(int j=0;j<counts[g];j++)out[slot++]=g;
  }}'''
  sink=UOp.sink(out,counts,arg=KernelInfo('rockchip_compact_expert_ids'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,counts)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def compact_matmul(x:Tensor,counts:Tensor,weight:Tensor,scales:Tensor,columns:int,capacity:int,cores:int=3,
                   *,activation_residual:bool=False,weight_residual:tuple[Tensor,Tensor]|None=None) -> Tensor:
  """Consecutive expert-grouped input rows; counts define weight/scaling selection."""
  if (x.device!='CPU' or x.ndim!=2 or not all(isinstance(v,int) and v>0 for v in x.shape)
      or weight.ndim!=5 or counts.device!='CPU' or counts.dtype is not dtypes.int or counts.shape!=(weight.shape[0],)
      or scales.device!='CPU' or scales.dtype is not dtypes.float):raise ValueError('Invalid compact expert projection operands')
  rows,depth=x.shape
  groups,k,n=weight.shape[0],weight.shape[2]*32,weight.shape[1]*32
  assert isinstance(rows,int) and isinstance(depth,int) and isinstance(groups,int) and isinstance(k,int) and isinstance(n,int)
  if depth>k or scales.shape!=(groups,n):raise ValueError('Compact expert dimensions differ')
  if type(activation_residual) is not bool:raise ValueError('Activation residual selection must be a bool')
  _check_weight_residual(weight_residual,weight,scales,activation_residual)
  payload,activation_scales=quantize(x,k)
  if activation_residual:
    codes=payload[round_up(rows*4,64):].bitcast(dtypes.int8).reshape(rows,k)
    remainder=_remainder(x,codes,activation_scales)
    extra,extra_scales=quantize(remainder,k)
    extra_codes=extra[round_up(rows*4,64):].bitcast(dtypes.int8).reshape(rows,k)
    values=codes.stack(extra_codes,dim=1).flatten().bitcast(dtypes.uint8)
    a_scales=activation_scales.stack(extra_scales,dim=1).reshape(rows*2)
    doubled=(counts*2).contiguous()
    header=a_scales.bitcast(dtypes.uint8).cat(doubled.bitcast(dtypes.uint8)).pad_to((round_up((rows*2+groups)*4,64),))
    raw=raw_matmul(header.cat(values),weight,capacity*2,columns,cores,rows=rows*2)
    ids=Tensor.empty(rows,device='CPU',dtype=dtypes.int).custom_kernel(counts.contiguous(),
      fxn=lambda o,c:_expert_ids(o,c,groups,rows,capacity))[0]
    if weight_residual is not None:
      extra_header=activation_scales.bitcast(dtypes.uint8).cat(counts.contiguous().bitcast(dtypes.uint8))
      extra_header=extra_header.pad_to((round_up((rows+groups)*4,64),))
      extra_raw=raw_matmul(extra_header.cat(payload[round_up(rows*4,64):]),weight_residual[0],capacity,columns,cores,rows=rows)
      return _dequantize_weight_residual(raw.reshape(rows,2,n),extra_raw,a_scales.reshape(rows,2),
                                         scales[ids],weight_residual[1][ids])[:,:columns]
    return _dequantize_activation_residual(raw.reshape(rows,2,n),a_scales.reshape(rows,2),scales[ids])[:,:columns]
  header=activation_scales.bitcast(dtypes.uint8).cat(counts.contiguous().bitcast(dtypes.uint8))
  header=header.pad_to((round_up((rows+groups)*4,64),))
  result=raw_matmul(header.cat(payload[round_up(rows*4,64):]),weight,capacity,columns,cores,rows=rows)
  ids=Tensor.empty(rows,device='CPU',dtype=dtypes.int).custom_kernel(counts.contiguous(),
    fxn=lambda o,c:_expert_ids(o,c,groups,rows,capacity))[0]
  return ((result.float()*activation_scales.reshape(rows,1))*scales[ids])[:,:columns]


def matmul(x:Tensor,counts:Tensor,weight:Tensor,scales:Tensor,columns:int,cores:int=3) -> Tensor:
  """Quantize fixed-capacity groups, execute only active rows; inactive results are unspecified."""
  if (x.device!='CPU' or x.ndim!=3 or not all(isinstance(v,int) and v>0 for v in x.shape)
      or weight.ndim!=5 or counts.device!='CPU' or counts.dtype is not dtypes.int or counts.shape!=(x.shape[0],)
      or scales.device!='CPU' or scales.dtype is not dtypes.float):raise ValueError('Invalid bucketed INT8 projection operands')
  groups,capacity,depth=x.shape
  k,n=weight.shape[2]*32,weight.shape[1]*32
  assert isinstance(groups,int) and isinstance(capacity,int) and isinstance(depth,int) and isinstance(k,int) and isinstance(n,int)
  if weight.shape[0]!=groups or depth>k or scales.shape!=(groups,n):raise ValueError('Bucketed INT8 dimensions differ')
  payload,activation_scales=quantize(x.reshape(groups*capacity,depth),k)
  header=activation_scales.bitcast(dtypes.uint8).cat(counts.contiguous().bitcast(dtypes.uint8))
  header=header.pad_to((round_up((groups*capacity+groups)*4,64),))
  result=raw_matmul(header.cat(payload[round_up(groups*capacity*4,64):]),weight,capacity,columns,cores)
  return ((result.float()*activation_scales.reshape(groups,capacity,1))*scales.reshape(groups,1,n))[:,:,:columns]
