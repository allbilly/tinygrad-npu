"""Experimental single-token routed INT8 projections with a resident expert atlas."""
from __future__ import annotations
import functools,struct
from dataclasses import replace
from tinygrad import Tensor,dtypes
from tinygrad.helpers import Target,round_up
from tinygrad.llm.kernels.rockchip_grouped_int8 import (_weight_parts,_remainder,_dequantize_activation_residual,
                                                    _dequantize_weight_residual,_check_weight_residual)
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.runtime.support.rockchip_int8 import supported_depth
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo,UOp


def _pack_weight(weight:Tensor,residual:bool) -> tuple[Tensor,Tensor,Tensor|None,Tensor|None]:
  """Prepare all immutable expert matrices once; quantization matches grouped INT8."""
  if (weight.device!='CPU' or weight.ndim!=3 or not dtypes.is_float(weight.dtype)
      or not all(isinstance(n,int) and n>0 for n in weight.shape)):
    raise ValueError('Routed weights require a static CPU floating-point expert table')
  experts,columns,depth=weight.shape
  assert isinstance(experts,int) and isinstance(columns,int) and isinstance(depth,int)
  if not 1<=experts<=128 or columns>4096 or depth>4096:raise ValueError('Unsupported routed expert geometry')
  # The existing quantizer accepts at most 24,576 output rows per preparation.
  parts:list[tuple[Tensor,Tensor]]=[]
  extra:list[tuple[Tensor,Tensor]]=[]
  for first in range(0,experts,max(1,24576//columns)):
    chunk=weight[first:first+max(1,24576//columns)]
    packed,scale,codes=_weight_parts(chunk)
    parts.append((packed,scale))
    if residual:extra.append(_weight_parts(_remainder(chunk,codes[:,:columns],scale[:,:columns]))[:2])
  def upload(items:list[tuple[Tensor,Tensor]]) -> tuple[Tensor,Tensor]:
    from tinygrad.runtime.ops_rockchip import int8_weight_device
    packed=items[0][0].cat(*(p[0] for p in items[1:]),dim=0).contiguous()
    scale=items[0][1].cat(*(p[1] for p in items[1:]),dim=0).contiguous().realize()
    return packed.to(int8_weight_device(packed.nbytes())).realize(),scale
  packed,scale=upload(parts)
  residual_weight,residual_scale=upload(extra) if residual else (None,None)
  return packed,scale,residual_weight,residual_scale


def pack_weight(weight:Tensor) -> tuple[Tensor,Tensor]:
  packed,scale,_,_=_pack_weight(weight,False)
  return packed,scale


def pack_weight_residual(weight:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
  """Separate W0/W1 atlases; W0 is shared by its two activation products."""
  packed,scale,extra,extra_scale=_pack_weight(weight,True)
  assert extra is not None and extra_scale is not None
  return packed,scale,extra,extra_scale


@functools.cache
def _program(out:UOp,lhs:UOp,rhs:UOp,groups:int,rows:int,columns:int,depth:int,experts:int,cores:int) -> UOp:
  sink=UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_routed_int8_{groups}_{rows}_{columns}_{depth}_{experts}_{cores}'))
  info=replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
    UOp(Ops.SOURCE,arg=f'Routed INT8 {groups} selected experts from {experts}, K={depth}, N={columns}'),
    UOp(Ops.BINARY,arg=struct.pack('<9I',rows,columns,depth,cores,groups,rows,0x47524d4d,3,experts))))


def raw_matmul(payload:Tensor,weight:Tensor,groups:int,columns:int,cores:int=3,*,rows:int=1) -> Tensor:
  """Header contains G FP32 activation scales, G int32 expert IDs, then INT8 rows."""
  if (weight.ndim!=5 or weight.shape[-2:]!=(32,32) or weight.dtype is not dtypes.int8
      or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP'
      or payload.device!='CPU' or payload.dtype is not dtypes.uint8 or payload.ndim!=1):
    raise ValueError('Invalid routed INT8 packed operands')
  experts,n,k=weight.shape[0],weight.shape[1]*32,weight.shape[2]*32
  assert isinstance(experts,int) and isinstance(n,int) and isinstance(k,int)
  if (not all(isinstance(v,int) for v in (experts,n,k)) or type(groups) is not int or not 1<=groups<=8
      or not 1<=experts<=128 or not 0<columns<=n<=4096 or round_up(columns,32)!=n
      or type(rows) is not int or not 1<=rows<=(128 if k<=2048 else 64)
      or not supported_depth(k) or cores not in (1,3) or payload.numel()!=round_up((groups*rows+groups)*4,64)+groups*rows*k):
    raise ValueError('Unsupported routed INT8 matrix geometry')
  result=Tensor.empty(groups*rows*n,device=weight.device,dtype=dtypes.int32).custom_kernel(payload.to(weight.device),weight,
    fxn=lambda o,a,w:_program(o,a,w,groups,rows,n,k,experts,cores))[0]
  return result.to('CPU').reshape((groups,n) if rows==1 else (groups,rows,n))


def matmul(x:Tensor,selected:Tensor,weight:Tensor,scales:Tensor,columns:int,cores:int=3,*,activation_residual:bool=False,
           weight_residual:tuple[Tensor,Tensor]|None=None) -> Tensor:
  """Each input row selects one expert; quantization changes require model quality checks."""
  if (x.device!='CPU' or x.ndim!=2 or not all(isinstance(n,int) and n>0 for n in x.shape)
      or selected.device!='CPU' or selected.dtype is not dtypes.int or selected.shape!=(x.shape[0],)
      or weight.ndim!=5 or scales.device!='CPU' or scales.dtype is not dtypes.float):
    raise ValueError('Invalid routed INT8 projection operands')
  groups,depth=x.shape
  k,n=weight.shape[2]*32,weight.shape[1]*32
  assert isinstance(groups,int) and isinstance(depth,int) and isinstance(k,int) and isinstance(n,int)
  if not 1<=groups<=8 or depth>k or scales.shape!=(weight.shape[0],n):raise ValueError('Routed INT8 dimensions differ')
  if type(activation_residual) is not bool:raise ValueError('Activation residual selection must be a bool')
  _check_weight_residual(weight_residual,weight,scales,activation_residual)
  payload,activation_scales=quantize(x,k)
  if activation_residual:
    codes=payload[round_up(groups*4,64):].bitcast(dtypes.int8).reshape(groups,k)
    remainder=_remainder(x,codes,activation_scales)
    extra,extra_scales=quantize(remainder,k)
    extra_codes=extra[round_up(groups*4,64):].bitcast(dtypes.int8).reshape(groups,k)
    values=codes.stack(extra_codes,dim=1).flatten().bitcast(dtypes.uint8)
    a_scales=activation_scales.stack(extra_scales,dim=1).reshape(groups*2)
    header=a_scales.bitcast(dtypes.uint8).cat(selected.contiguous().bitcast(dtypes.uint8)).pad_to((round_up(groups*12,64),))
    raw=raw_matmul(header.cat(values),weight,groups,columns,cores,rows=2)
    if weight_residual is not None:
      extra_header=activation_scales.bitcast(dtypes.uint8).cat(selected.contiguous().bitcast(dtypes.uint8)).pad_to((round_up(groups*8,64),))
      extra_raw=raw_matmul(extra_header.cat(payload[round_up(groups*4,64):]),weight_residual[0],groups,columns,cores)
      return _dequantize_weight_residual(raw,extra_raw,a_scales.reshape(groups,2),scales[selected],weight_residual[1][selected])[:,:columns]
    return _dequantize_activation_residual(raw,a_scales.reshape(groups,2),scales[selected])[:,:columns]
  header=activation_scales.bitcast(dtypes.uint8).cat(selected.contiguous().bitcast(dtypes.uint8)).pad_to((round_up(groups*8,64),))
  routed=header.cat(payload[round_up(groups*4,64):])
  result=raw_matmul(routed,weight,groups,columns,cores)
  return ((result.float()*activation_scales.reshape(groups,1))*scales[selected])[:,:columns]
