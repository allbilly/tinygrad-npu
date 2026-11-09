"""Explicit grouped INT8 projections with per-row and per-output-channel scales."""
from __future__ import annotations
import functools,struct
from dataclasses import replace
from tinygrad import Tensor,dtypes
from tinygrad.helpers import Target,round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.runtime.support.rockchip_int8 import SMALL_DEPTHS,padded_depth,supported_depth
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo,UOp


@functools.cache
def _remainder_program(out:UOp,inp:UOp,codes:UOp,scales:UOp,rows:int,depth:int,padded:int) -> UOp:
  source=f'''void rockchip_int8_remainder(float *out,const float *input,const signed char *codes,const float *scales) {{
    for(int row=0;row<{rows};row++)for(int i=0;i<{depth};i++)
      out[row*{depth}+i]=__builtin_fmaf(-(float)codes[row*{padded}+i],scales[row],input[row*{depth}+i]);
  }}'''
  sink=UOp.sink(out,inp,codes,scales,arg=KernelInfo('rockchip_int8_remainder'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp,codes,scales)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _remainder(x:Tensor,codes:Tensor,scales:Tensor) -> Tensor:
  """One correctly rounded FMA per residual, independent of graph fusion choices."""
  rows,depth,padded=x.numel()//x.shape[-1],x.shape[-1],codes.shape[-1]
  assert isinstance(rows,int) and isinstance(depth,int) and isinstance(padded,int)
  return Tensor.empty(x.numel(),device='CPU',dtype=dtypes.float).custom_kernel(
    x.float().contiguous(),codes.contiguous(),scales.contiguous(),
    fxn=lambda o,a,c,s:_remainder_program(o,a,c,s,rows,depth,padded))[0].reshape(x.shape)


@functools.cache
def _residual_dequantize_program(out:UOp,raw:UOp,scales:UOp,weights:UOp,rows:int,columns:int) -> UOp:
  source=f'''void rockchip_dequantize_activation_residual(float *out,const int *raw,const float *scales,const float *weights) {{
    for(int row=0;row<{rows};row++)for(int col=0;col<{columns};col++) {{
      float first=(float)raw[(row*2)*{columns}+col]*scales[row*2];
      float second=(float)raw[(row*2+1)*{columns}+col]*scales[row*2+1];
      float weight=weights[row*{columns}+col];
      out[row*{columns}+col]=__builtin_fmaf(first,weight,second*weight);
    }}
  }}'''
  sink=UOp.sink(out,raw,scales,weights,arg=KernelInfo('rockchip_dequantize_activation_residual'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,raw,scales,weights)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _residual_shape(raw:Tensor,scales:Tensor,weight_scales:Tensor) -> tuple[int,int]:
  if (raw.device!='CPU' or raw.dtype is not dtypes.int or raw.ndim!=3 or raw.shape[1]!=2
      or not all(isinstance(n,int) and n>0 for n in raw.shape)
      or scales.device!='CPU' or scales.dtype is not dtypes.float or scales.shape!=(raw.shape[0],2)
      or weight_scales.device!='CPU' or weight_scales.dtype is not dtypes.float
      or weight_scales.shape!=(raw.shape[0],raw.shape[2])):raise ValueError('Invalid activation residual dequantization operands')
  rows,_,columns=raw.shape
  assert isinstance(rows,int) and isinstance(columns,int)
  return rows,columns


def _dequantize_activation_residual(raw:Tensor,scales:Tensor,weight_scales:Tensor) -> Tensor:
  """Match the CPU recipe: fma(float(dot0)*a0, w, (float(dot1)*a1)*w)."""
  rows,columns=_residual_shape(raw,scales,weight_scales)
  return Tensor.empty(rows,columns,device='CPU',dtype=dtypes.float).custom_kernel(
    raw.contiguous(),scales.contiguous(),weight_scales.contiguous(),
    fxn=lambda o,r,s,w:_residual_dequantize_program(o,r,s,w,rows,columns))[0]


@functools.cache
def _weight_residual_dequantize_program(out:UOp,raw:UOp,extra:UOp,scales:UOp,weights:UOp,residual:UOp,rows:int,columns:int) -> UOp:
  source=f'''void rockchip_dequantize_weight_residual(float *out,const int *raw,const int *extra,
                                                       const float *scales,const float *weights,const float *residual) {{
    for(int row=0;row<{rows};row++)for(int col=0;col<{columns};col++) {{
      int index=row*{columns}+col;
      float first=(float)raw[(row*2)*{columns}+col]*scales[row*2];
      float second=(float)raw[(row*2+1)*{columns}+col]*scales[row*2+1];
      float third=(float)extra[index]*scales[row*2];
      float combined=__builtin_fmaf(first,weights[index],second*weights[index]);
      out[index]=__builtin_fmaf(third,residual[index],combined);
    }}
  }}'''
  sink=UOp.sink(out,raw,extra,scales,weights,residual,arg=KernelInfo('rockchip_dequantize_weight_residual'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4,5))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,raw,extra,scales,weights,residual)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _dequantize_weight_residual(raw:Tensor,extra:Tensor,scales:Tensor,weights:Tensor,residual:Tensor) -> Tensor:
  """Ordered three products: the activation pair against W0, then A0 against W1."""
  rows,columns=_residual_shape(raw,scales,weights)
  if (extra.device!='CPU' or extra.dtype is not dtypes.int or extra.shape!=(rows,columns)
      or residual.device!='CPU' or residual.dtype is not dtypes.float or residual.shape!=(rows,columns)):
    raise ValueError('Invalid weight residual dequantization operands')
  return Tensor.empty(rows,columns,device='CPU',dtype=dtypes.float).custom_kernel(
    raw.contiguous(),extra.contiguous(),scales.contiguous(),weights.contiguous(),residual.contiguous(),
    fxn=lambda o,r,e,s,w,v:_weight_residual_dequantize_program(o,r,e,s,w,v,rows,columns))[0]


def _check_weight_residual(pair:tuple[Tensor,Tensor]|None,weight:Tensor,scales:Tensor,enabled:bool) -> None:
  if pair is None:return
  if (not enabled or type(pair) is not tuple or len(pair)!=2 or not all(isinstance(v,Tensor) for v in pair)
      or pair[0].shape!=weight.shape or pair[0].dtype is not dtypes.int8
      or not isinstance(pair[0].device,str) or pair[0].device.split(':')[0]!='ROCKCHIP'
      or pair[1].device!='CPU' or pair[1].dtype is not dtypes.float or pair[1].shape!=scales.shape):
    raise ValueError('Weight residuals require a matching packed atlas and activation residuals')


def _weight_parts(weight:Tensor) -> tuple[Tensor,Tensor,Tensor]:
  groups,columns,depth=weight.shape
  assert isinstance(groups,int) and isinstance(columns,int) and isinstance(depth,int)
  k=next(d for d in sorted(SMALL_DEPTHS) if d>=depth) if depth<=256 else padded_depth(depth)
  n=round_up(columns,32)
  payload,scale=quantize(weight.reshape(groups*columns,depth),k)
  scale=scale.reshape(groups,columns).pad(((0,0),(0,n-columns)),value=1).contiguous().realize()
  values=scale.tolist()
  assert isinstance(values,list)
  if any(not 0<s<float('inf') for row in values for s in row):raise ValueError('Grouped INT8 weight scales must be finite and positive')
  codes=payload[round_up(groups*columns*4,64):].bitcast(dtypes.int8).reshape(groups,columns,k).pad(((0,0),(0,n-columns),(0,0)))
  packed=codes.reshape(groups,n//32,32,k//32,32).permute(0,1,3,2,4).contiguous()
  return packed,scale,codes


def pack_weight(weight:Tensor,*,residual:bool=False) -> tuple[Tensor,Tensor]:
  """Pack immutable CPU weights; optional three-product residual group order is W0,W0,W1."""
  if (weight.device!='CPU' or weight.ndim!=3 or not all(isinstance(n,int) and n>0 for n in weight.shape)
      or not dtypes.is_float(weight.dtype)):raise ValueError('Grouped INT8 weights require a static CPU float tensor')
  groups,columns,depth=weight.shape
  assert isinstance(groups,int) and isinstance(columns,int) and isinstance(depth,int)
  if type(residual) is not bool:raise ValueError('Grouped INT8 residual selection must be a bool')
  if groups>(42 if residual else 128) or columns>4096 or depth>4096 or groups*columns>24576:
    raise ValueError('Grouped INT8 weight geometry exceeds preparation bounds')
  packed,scale,codes=_weight_parts(weight)
  if residual:
    remainder=_remainder(weight,codes[:,:columns],scale[:,:columns])
    extra,extra_scales,_=_weight_parts(remainder)
    # General tensor operations run on CPU; upload the completed immutable layout once.
    packed=packed.cat(packed,extra,dim=0).contiguous()
    scale=scale.cat(scale,extra_scales,dim=0).contiguous()
  return packed.to('ROCKCHIP'),scale


@functools.cache
def _dequantize_program(out:UOp,raw:UOp,sa:UOp,sw:UOp,groups:int,rows:int,columns:int,padded:int) -> UOp:
  source=f'''void rockchip_grouped_dequantize(float *out,const int *raw,const float *sa,const float *sw) {{
    for(int row=0;row<{groups*rows};row++) {{
      const float a=sa[row];const float *w=sw+(row/{rows})*{padded};
      for(int c=0;c<{columns};c++) out[row*{columns}+c]=((float)raw[row*{padded}+c]*a)*w[c];
    }}
  }}'''
  sink=UOp.sink(out,raw,sa,sw,arg=KernelInfo('rockchip_grouped_dequantize'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,raw,sa,sw)),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def dequantize(raw:Tensor,activation_scales:Tensor,scales:Tensor,groups:int,rows:int,columns:int,padded:int) -> Tensor:
  """Row-major grouped INT32 to FP32 with the reference (value*a)*w rounding."""
  dense=raw.reshape(groups*rows,padded).contiguous()
  out=Tensor.empty(groups*rows*columns,device='CPU',dtype=dtypes.float).custom_kernel(
    dense,activation_scales.contiguous(),scales.contiguous(),
    fxn=lambda o,r,a,w:_dequantize_program(o,r,a,w,groups,rows,columns,padded))[0]
  return out.reshape(groups,rows,columns)


def shared_matmul(x:Tensor,weight:Tensor,scales:Tensor,columns:int,cores:int=3) -> Tensor:
  """Grouped INT8 projection where every group consumes the same activation rows.

  Quantize the activation once, then replicate its header/codes per group. This is
  the gate/up shape: three 4096-column weight groups share one 4096-depth input.
  """
  if (x.device!='CPU' or x.ndim!=2 or weight.ndim!=5 or scales.device!='CPU'
      or scales.dtype is not dtypes.float):raise ValueError('Invalid shared grouped INT8 operands')
  groups=weight.shape[0]
  rows,depth=x.shape
  k=weight.shape[2]*32
  n=weight.shape[1]*32
  assert isinstance(groups,int) and isinstance(rows,int) and isinstance(depth,int) and isinstance(k,int) and isinstance(n,int)
  if (not 1<=groups<=128 or not 1<=rows<=128 or not 0<depth<=k or scales.shape!=(groups,n)
      or not 0<columns<=n or round_up(columns,32)!=n):raise ValueError('Unsupported shared grouped INT8 geometry')
  payload,activation_scales=quantize(x,k)
  codes=payload[round_up(rows*4,64):]
  repeated=activation_scales.repeat(groups)
  header=repeated.bitcast(dtypes.uint8).pad_to((round_up(groups*rows*4,64),))
  raw=raw_matmul(header.cat(codes.repeat(groups)),weight,rows,columns,cores)
  return dequantize(raw,repeated,scales,groups,rows,columns,k)


@functools.cache
def _program(out:UOp,lhs:UOp,rhs:UOp,groups:int,rows:int,columns:int,depth:int,tile:int,cores:int) -> UOp:
  sink=UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_grouped_int8_{groups}_{rows}_{columns}_{depth}_{cores}'))
  info=replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
    UOp(Ops.SOURCE,arg=f'Grouped INT8 {groups}x{rows}x{depth}x{columns}'),
    UOp(Ops.BINARY,arg=struct.pack('<8I',rows,columns,depth,cores,groups,tile,0x47524d4d,2))))


def raw_matmul(payload:Tensor,weight:Tensor,rows:int,columns:int,cores:int=3) -> Tensor:
  """Consume one shared scale header and group-major row-major INT8 payload."""
  if (weight.ndim!=5 or weight.shape[-2:]!=(32,32) or weight.dtype is not dtypes.int8
      or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP'
      or payload.device!='CPU' or payload.dtype is not dtypes.uint8 or payload.ndim!=1):
    raise ValueError('Invalid grouped INT8 packed operands')
  groups,n,k=weight.shape[0],weight.shape[1]*32,weight.shape[2]*32
  assert isinstance(groups,int) and isinstance(n,int) and isinstance(k,int)
  if (type(rows) is not int or not 1<=rows<=128 or not 1<=groups<=128 or cores not in (1,3)
      or not 0<columns<=n<=4096 or round_up(columns,32)!=n or not supported_depth(k)
      or payload.numel()!=round_up(groups*rows*4,64)+groups*rows*k):raise ValueError('Unsupported grouped INT8 matrix geometry')
  tile=128 if k<=2048 else 64
  result=Tensor.empty(groups*rows*n,device=weight.device,dtype=dtypes.int32).custom_kernel(payload.to(weight.device),weight,
    fxn=lambda o,a,w:_program(o,a,w,groups,rows,n,k,tile,cores))[0]
  return result.to('CPU').reshape(groups,rows,n)


def matmul(x:Tensor,weight:Tensor,scales:Tensor,columns:int,cores:int=3,*,residual:bool=False,materialize:bool=False) -> Tensor:
  """Scaled INT32 dots; residual mode adds A0W0 + A1W0 + A0W1 in that order."""
  if (x.device!='CPU' or x.ndim!=3 or not all(isinstance(n,int) and n>0 for n in x.shape)
      or weight.ndim!=5 or scales.device!='CPU' or scales.dtype is not dtypes.float):
    raise ValueError('Invalid grouped INT8 projection operands')
  groups,rows,depth=x.shape
  k,n=weight.shape[2]*32,weight.shape[1]*32
  assert isinstance(groups,int) and isinstance(rows,int) and isinstance(depth,int) and isinstance(k,int) and isinstance(n,int)
  if type(residual) is not bool or type(materialize) is not bool:raise ValueError('Grouped INT8 residual/materialize selection must be a bool')
  packed_groups=groups*(3 if residual else 1)
  if packed_groups!=weight.shape[0] or depth>k or scales.shape!=(packed_groups,n):
    raise ValueError('Grouped INT8 scale or feature shape differs')
  payload,activation_scales=quantize(x.reshape(groups*rows,depth),k)
  if residual:
    codes=payload[round_up(groups*rows*4,64):].bitcast(dtypes.int8).reshape(groups,rows,k)
    remainder=_remainder(x,codes,activation_scales)
    extra,extra_scales=quantize(remainder.reshape(groups*rows,depth),k)
    extra_codes=extra[round_up(groups*rows*4,64):].bitcast(dtypes.int8).reshape(groups,rows,k)
    activation_scales=activation_scales.cat(extra_scales,activation_scales)
    header=activation_scales.bitcast(dtypes.uint8).pad_to((round_up(packed_groups*rows*4,64),))
    payload=header.cat(codes.cat(extra_codes,codes,dim=0).flatten().bitcast(dtypes.uint8))
  raw=raw_matmul(payload,weight,rows,columns,cores)
  if not residual and materialize:return dequantize(raw,activation_scales,scales,groups,rows,columns,n)
  result=(raw.float()*activation_scales.reshape(packed_groups,rows,1))*scales.reshape(packed_groups,1,n)
  if residual:
    parts=result.reshape(3,groups,rows,n)
    result=(parts[0]+parts[1])+parts[2]
  return result[:,:,:columns]
