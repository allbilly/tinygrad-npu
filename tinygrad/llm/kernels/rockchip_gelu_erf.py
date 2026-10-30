"""Exact-order vector erf-GELU gating for CPU prefill."""
import functools, math
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def _vector_erf_source() -> str:
  # Preserve the legalized tensor program's division and multiplication order.
  return _vector_exp2_source()+r"""
  static __attribute__((always_inline)) float4 erf_gate(float4 g,float4 u) {
    float4 exponent=rockchip_exp2_vector((g*g)*(float4)-0.7213475108146667f);
    float4 sign=(g*(float4)0.7071067690849304f)<(float4)0?(float4)-1:(float4)1;
    float4 denominator=(float4)1+(g*sign)*(float4)0.23164188861846924f;
    float4 polynomial=(float4)1.0614054203033447f/denominator+(float4)-1.453152060508728f;
    polynomial=polynomial/denominator+(float4)1.421413779258728f;
    polynomial=polynomial/denominator+(float4)-0.2844967246055603f;
    polynomial=polynomial/denominator+(float4)0.2548295855522156f;
    polynomial=polynomial/denominator;
    return ((g*((float4)1+sign*((float4)1-polynomial*exponent)))*u)*(float4)0.5f;
  }
  """


@functools.cache
def _program(out:UOp, gate:UOp, up:UOp, count:UOp, capacity:int, depth:int) -> UOp:
  source = _vector_erf_source()+f"""
  void rockchip_cpu_erf_gelu_gate(float *out,const float *gate,const float *up,const int *rows) {{
    if(*rows<1 || *rows>{capacity})__builtin_trap();
    for(int i=0;i<*rows*{depth//4};i++)
      ((float4*)out)[i]=erf_gate(((const float4*)gate)[i],((const float4*)up)[i]);
    for(int i=*rows*{depth};i<{capacity*depth};i++)out[i]=0;
  }}"""
  inputs = (out,gate,up,count)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_erf_gelu_gate'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def gated_erf_gelu(gate:Tensor, up:Tensor) -> Tensor:
  """Keep the tensor erf formula and multiplication order; one-row decode stays generic."""
  if gate.ndim<2:return gate.gelu(approximate='none')*up
  rows = math.prod(gate.shape[:-1])
  capacity,depth = math.prod(gate.max_shape[:-1]),gate.max_shape[-1]
  if (gate.device!='CPU' or up.device!='CPU' or gate.dtype is not dtypes.float or up.dtype is not dtypes.float
      or gate.shape!=up.shape or not 16<=capacity<=128 or not 512<=depth<=32768 or depth%4
      or isinstance(rows,int) and rows<1 or isinstance(rows,UOp) and rows.vmin<1):
    return gate.gelu(approximate='none')*up
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  result = Tensor.empty(*gate.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
    gate.pad_to(gate.max_shape).contiguous(),up.pad_to(up.max_shape).contiguous(),count,
    fxn=lambda o,g,u,n:_program(o,g,u,n,capacity,depth))[0]
  return result.shrink(tuple((0,n) for n in gate.shape))
