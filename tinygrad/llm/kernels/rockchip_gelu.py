"""Vector CPU GELU-tanh gating with parallel rows for prefill."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

ONPU_CPU_SMALL_GELU_PREFILL = ContextVar('ONPU_CPU_SMALL_GELU_PREFILL',0)


@functools.cache
def _program(out:UOp, gate:UOp, up:UOp, count:UOp, capacity:int, depth:int) -> UOp:
  parallel = capacity*depth>=131072
  workers = '''typedef unsigned long Thread;
  extern int pthread_create(Thread*,const void*,void*(*)(void*),void*);
  extern int pthread_join(Thread,void**);
  typedef struct {float *out;const float *gate,*up;int first,last;} Work;
  static void *worker(void *p){Work *w=p;part(w->out,w->gate,w->up,w->first,w->last);return (void*)0;}''' if parallel else ''
  dispatch = '''Work work[4];Thread ids[3];int created[3];
    int blocks=*rows*DEPTH/4;
    for(int i=0;i<4;i++)work[i]=(Work){out,gate,up,blocks*i/4,blocks*(i+1)/4};
    for(int i=0;i<3;i++){created[i]=pthread_create(ids+i,(void*)0,worker,work+i)==0;if(!created[i])worker(work+i);}
    worker(work+3);for(int i=0;i<3;i++)if(created[i])pthread_join(ids[i],(void**)0);''' if parallel else 'part(out,gate,up,0,*rows*DEPTH/4);'
  source = _vector_exp2_source()+f'''
  #define DEPTH {depth}
  static void part(float *out,const float *gate,const float *up,int first,int last) {{
    for(int index=first;index<last;index++) {{
      float4 g=*((const float4*)gate+index),u=*((const float4*)up+index);
      float4 exponent=(g+0.044714998453855515f*g*g*g)*-2.302208185195923f;
      *((float4*)out+index)=(g/(1.0f+rockchip_exp2_vector(exponent)))*u;
    }}
  }}
  {workers}
  void rockchip_cpu_gelu_gate(float *out,const float *gate,const float *up,const int *rows) {{
    if(*rows<1 || *rows>{capacity})__builtin_trap();
    {'if(*rows*DEPTH<131072)part(out,gate,up,0,*rows*DEPTH/4);else {'+dispatch+'}' if parallel else dispatch}
    for(int i=*rows*DEPTH;i<{capacity*depth};i++)out[i]=0;
  }}'''
  inputs = (out,gate,up,count)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_gelu_gate'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def gated_gelu(gate:Tensor, up:Tensor) -> Tensor:
  if gate.ndim<2:return gate.gelu(approximate='tanh')*up
  rows = math.prod(gate.shape[:-1])
  capacity,depth = math.prod(gate.max_shape[:-1]),gate.max_shape[-1]
  small_prefill = bool(ONPU_CPU_SMALL_GELU_PREFILL) and gate.max_shape==(1,128,256)
  if (gate.device!='CPU' or up.device!='CPU' or gate.dtype is not dtypes.float or up.dtype is not dtypes.float
      or gate.shape!=up.shape or not 1<=capacity<=256 or not (512<=depth<=32768 or small_prefill) or depth%4
      or isinstance(rows,int) and rows<1 or isinstance(rows,UOp) and rows.vmin<1):
    return gate.gelu(approximate='tanh')*up
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  out = Tensor.empty(*gate.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
    gate.pad_to(gate.max_shape).contiguous(),up.pad_to(up.max_shape).contiguous(),count,
    fxn=lambda o,g,u,n:_program(o,g,u,n,capacity,depth))[0]
  return out.shrink(tuple((0,n) for n in gate.shape))
