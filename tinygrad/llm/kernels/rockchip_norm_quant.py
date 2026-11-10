"""Exact RMSNorm/INT8 quantization for prepared 128-row, 896-wide CPU inputs.

Only complete static normalization results are recorded. Views that change row
order, partial chunks, scaled inputs and calls outside the shared pool keep the
ordinary path. Packed outputs fill every scale, header and padding byte.
"""
from __future__ import annotations
import functools, math, weakref
from collections import Counter
from dataclasses import replace
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_cpu import ONPU_NORM_QUANT, ONPU_NORM_QUANT_DECODE, _renderer
from tinygrad.llm.kernels.rockchip_shared_pool import _current
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

_normalizations:weakref.WeakKeyDictionary[UOp,tuple[UOp,UOp,float]] = weakref.WeakKeyDictionary()
_traces:Counter[bool] = Counter()
_decode_traces:int = 0


def register_norm(result:Tensor, source:Tensor, weight:Tensor, eps:float) -> Tensor:
  # Accept any CPU float norm whose last width is 4-aligned and whose row count is
  # a positive int. The kernel already takes rows/depth/padded as parameters, so
  # only the gate and the two entry points needed generalising from 128x896.
  if ((ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE) and source.device==weight.device==result.device=='CPU' and source.ndim>=2
      and source.dtype is dtypes.float and result.dtype is dtypes.float
      and isinstance((depth:=source.shape[-1]),int) and depth%4==0 and 4<=depth<=8192
      and all(isinstance(n,int) for n in source.shape) and (rows:=math.prod(source.shape[:-1]))>0
      and rows<=128
      and weight.shape==(depth,) and weight.dtype in (dtypes.float,dtypes.half) and math.isfinite(eps) and eps>0):
    _normalizations[result.uop] = source.uop,weight.uop,eps
  return result


def _recover(u:UOp) -> tuple[UOp,UOp,float]|None:
  if (found:=_normalizations.get(u)) is not None:return found
  if u.op is Ops.RESHAPE and u.numel()==u.shape[-1]*max(1,u.numel()//u.shape[-1]):return _recover(u.src[0])
  if (u.op is Ops.SHRINK and len(u.arg)==len(u.src[0].shape)
      and all(start==0 and end==size for (start,end),size in zip(u.arg,u.src[0].shape))):return _recover(u.src[0])
  return None


@functools.cache
def _program(out:UOp, source:UOp, gamma:UOp, scales:UOp, rows:int, depth:int, padded:int, sdk:bool,
             eps:float, half:bool, address:int) -> UOp:
 from tinygrad.llm.kernels.rockchip_int8 import _quantize_vector_source
 header=round_up(rows*4,64)
 dtype='__fp16' if half else 'float'
 load='__builtin_convertvector(*(const half4*)(weight+c),float4)' if half else '*(const float4*)(weight+c)'
 index=f'((c+j)/16*{rows}+first+r)*16+(c+j)%16' if sdk else f'(first+r)*{padded}+c+j'
 text=f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
 typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
 typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
 {_quantize_vector_source()}
 typedef struct {{unsigned char *raw;const float *input;const {dtype} *weight;float *scales;}} Work;
 static void part(void **args,int unused,int worker) {{
  Work *w=args[0];unsigned char *raw=w->raw;const float *input=w->input;const {dtype} *weight=w->weight;
  float *scales=(float*)raw;signed char *output=(signed char*)(raw+{header});
  int begin={rows}*worker/4,end={rows}*(worker+1)/4;
  for(int first=begin;first<end;first+=4) {{
   int count=end-first<4?end-first:4;float values[4][{depth}] __attribute__((aligned(16))),inverse[4];int invalid[4];
   for(int r=0;r<count;r++) {{
    const float *x=input+(first+r)*{depth};float4 acc=(float4)0;
    for(int c=0;c<{depth};c+=4) {{float4 v=*(const float4*)(x+c);acc=acc+v*v;}}
    float sum=((acc[0]+acc[1])+acc[2])+acc[3];
    float inv=1.0f/__builtin_sqrtf(sum*{1/depth:.17g}f+{eps:.17g}f);
    float4 maximum={{0}};uint4 bad={{0}};
    for(int c=0;c<{depth};c+=4) {{float4 v=*(const float4*)(x+c),g={load};v=(v*(float4)inv)*g;
     *(float4*)(values[r]+c)=v;uint4 bits=__builtin_bit_cast(uint4,v);
     bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;float4 a=v>=(float4)0?v:-v;maximum=a>maximum?a:maximum;
    }}
    float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
    invalid[r]=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
    scales[first+r]=w->scales[first+r]=invalid[r]?__builtin_nanf(""):scale;inverse[r]=1.0f/scale;
   }}
   for(int c=0;c<{padded};c+=16)for(int r=0;r<count;r++)for(int j=0;j<16;j+=4) {{
    float4 v=(c+j<{depth}&&!invalid[r])?*(const float4*)(values[r]+c+j):(float4)0;
    *(char4*)(output+{index})=rockchip_quantize_vector(v,inverse[r]);
   }}
  }}
 }}
 void rockchip_norm_quantize{('_sdk' if sdk else '')}(unsigned char *raw,const float *input,const {dtype} *weight,float *scale_out) {{
  for(int i={rows*4};i<{header};i++)raw[i]=0;Work work={{raw,input,weight,scale_out}};void *args[4]={{&work,0,0,0}};
  if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(part,args,0))for(int i=0;i<4;i++)part(args,0,i);
 }}'''
 inputs=(out,source,gamma,scales)
 sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_norm_quantize'+('_sdk' if sdk else '')))
 renderer=_renderer()
 info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,3),ins=(1,2))
 return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=text),
                                   UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(text))))


@functools.cache
def _decode_program(out:UOp, source:UOp, gamma:UOp, scales:UOp, depth:int, padded:int, eps:float, half:bool) -> UOp:
  """One-token RMSNorm + INT8 quantization: the exact source text of both kernels.

  Decode rows are 1, so the standing fused kernel's four-worker row partition cannot
  help and it requires the shared host pool (off in the shipped stack). This variant is
  a single serial row: the same FMA variance chain and division-form normalization as
  `_rms_norm(exact=True)`, then the exact `_quantize` range reduction and rounding.
  """
  from tinygrad.llm.kernels.rockchip_int8 import _quantize_vector_source
  header=round_up(1*4,64)
  dtype='__fp16' if half else 'float'
  load='__builtin_convertvector(*(const half4*)(weight+c),float4)' if half else '*(const float4*)(weight+c)'
  inv_w,eps_lit=f'{1/depth:.17g}f',f'{eps:.17g}f'
  text=f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {_quantize_vector_source()}
  void rockchip_norm_quantize_decode(unsigned char *raw,const float *input,const {dtype} *weight,float *scale_out) {{
    const float *x=input;float *scales=(float*)raw;signed char *output=(signed char*)(raw+{header});
    for(int i=4;i<{header};i++)raw[i]=0;
    float acc=0.0f;
    for(int c=0;c<{depth};c+=4) {{float4 v=*(const float4*)(x+c);
      acc=__builtin_fmaf(v[0],v[0],acc); acc=__builtin_fmaf(v[1],v[1],acc);
      acc=__builtin_fmaf(v[2],v[2],acc); acc=__builtin_fmaf(v[3],v[3],acc);}}
    float denominator=__builtin_sqrtf(acc*{inv_w}+{eps_lit});
    float values[{depth}] __attribute__((aligned(16)));float4 maximum={{0}};uint4 bad={{0}};
    for(int c=0;c<{depth};c+=4) {{float4 value=*(const float4*)(x+c),gamma={load};
      value=((value/(float4)denominator))*gamma;
      *(float4*)(values+c)=value;uint4 bits=__builtin_bit_cast(uint4,value);
      bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;}}
    float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
    int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
    scales[0]=scale_out[0]=invalid?__builtin_nanf(""):scale;float inverse=1.0f/scale;int i=0;
    for(;i+3<{depth};i+=4){{float4 value=invalid?(float4)0:*(const float4*)(values+i);
      *(char4*)(output+i)=rockchip_quantize_vector(value,inverse);}}
    for(;i<{depth};i++){{float q=invalid?0:__builtin_nearbyintf(values[i]*inverse);
      if(q>127)q=127;if(q< -127)q=-127;output[i]=(signed char)q;}}
    for(;i<{padded};i++)output[i]=0;
  }}'''
  inputs=(out,source,gamma,scales)
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_norm_quantize_decode'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,3),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=text),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(text))))


def try_quantize(x:Tensor, padded:int, *, sdk_prefill:bool=False) -> tuple[Tensor,Tensor]|None:
  global _decode_traces
  if not (ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE) or padded%16 or padded<x.shape[-1]:return None
  # Decode fast path: one row, no shared host pool, exact arithmetic from both kernels.
  if ONPU_NORM_QUANT_DECODE and not ONPU_NORM_QUANT:
    depth = x.shape[-1]
    rows = math.prod(x.shape[:-1])
    if not (isinstance(rows,int) and isinstance(depth,int) and rows==1 and not sdk_prefill
            and 4<=depth<=8192 and depth%4==0 and padded>=depth):return None
    found = _recover(x.uop)
    if found is None:return None
    source,gamma,eps = found
    a = Tensor(source,device='CPU').reshape(depth).contiguous()
    weight = Tensor(gamma,device='CPU').contiguous()
    result = Tensor.empty(round_up(1*4,64)+padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
      a,weight,Tensor.empty(1,device='CPU',dtype=dtypes.float),
      fxn=lambda o,i,g,s:_decode_program(o,i,g,s,depth,padded,eps,g.dtype is dtypes.half))
    _decode_traces += 1
    return result[0],result[3]
  # Decode norms arrive as (1,1,depth); prefill as (rows,depth). The fused kernel
  # only needs the flattened row count and the width.
  depth = x.shape[-1]
  rows = math.prod(x.shape[:-1])
  if not isinstance(rows,int) or not isinstance(depth,int) or not 1<=rows<=128 or depth%4:return None
  found,state = _recover(x.uop),_current.get()
  # The parallel dispatcher is MANDATORY, not optional: with address==0 the
  # generated body runs all four partitions serially but still partitions by
  # `rows`, which segfaults when rows==1 (three partitions get an empty range and
  # the padding loop writes past the row). Measured: ONPU_CPU_RMS_ALL_ROWS=1 with
  # ONPU_NORM_QUANT=1 segfaults. Require the pool as before.
  if found is None or state is None or not state.pool.active:return None
  address = state.address
  source,gamma,eps = found
  a = Tensor(source,device='CPU').reshape(rows,depth).contiguous()
  weight = Tensor(gamma,device='CPU').contiguous()
  result = Tensor.empty(round_up(rows*4,64)+rows*padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
    a,weight,Tensor.empty(rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,a,w,s:_program(o,a,w,s,rows,depth,padded,sdk_prefill,eps,w.dtype is dtypes.half,address))
  _traces[sdk_prefill] += 1
  return result[0],result[3]


def norm_quant_stats() -> dict[str,Any]:
  return {'traces':sum(_traces.values()),'ordinary':_traces[False],'sdk':_traces[True],
          'decode':_decode_traces}
