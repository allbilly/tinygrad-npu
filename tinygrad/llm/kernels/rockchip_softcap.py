"""Vector FP32 final-logit soft cap, preserving the ordinary tanh reduction input."""
from __future__ import annotations
import functools, math, os, struct
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_shared_pool import _current, shared_program, pooled_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

SHARED_POOL_SUPPORTED = True

def _f32(value:float) -> float:return struct.unpack('f',struct.pack('f',value))[0]


@shared_program
@functools.cache
def _program(out:UOp, values:UOp, count:int, cap:float, parallel:bool, pool:int|None=None) -> UOp:
  coefficient = _f32((-2/math.log(2))*(1/cap))
  workers = '''typedef unsigned long Thread;
  extern int pthread_create(Thread*,const void*,void*(*)(void*),void*);
  extern int pthread_join(Thread,void**);
  typedef struct {float *out;const float *input;int first,last;} Work;
  static void *worker(void *p){Work *w=p;part(w->out,w->input,w->first,w->last);return (void*)0;}''' if parallel else ''
  dispatch = f'''Work work[4];Thread threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,input,{count//4}*i/4,{count//4}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;if(!created[i])worker(work+i);}}
    worker(work+3);for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(out,input,0,{count//4});'
  tail = f'''float4 tail={{0}};
    for(int i=0;i<{count%4};i++)tail[i]=input[{count-count%4}+i];
    tail=transform(tail);
    for(int i=0;i<{count%4};i++)out[{count-count%4}+i]=tail[i];''' if count%4 else ''
  source = _vector_exp2_source()+f'''
  static __attribute__((always_inline)) float4 transform(float4 x) {{
    float4 y=rockchip_exp2_vector(x*{coefficient.hex()}f);
    return (2.0f/(1.0f+y)-1.0f)*{_f32(cap).hex()}f;
  }}
  static void part(float *out,const float *input,int first,int last) {{
    for(int i=first;i<last;i++)*((float4*)out+i)=transform(*((const float4*)input+i));
  }}
  {workers}
  void rockchip_cpu_softcap(float *out,const float *input) {{
    {dispatch}
    {tail}
  }}'''
  if pool is not None and parallel:
    source,record=pooled_source(source,pool)
    if not record['changed']:raise ValueError('Unexpected softcap worker source')
  inputs = (out,values)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_softcap'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def softcap(logits:Tensor, cap:float, *, pool:AttentionPool|None=None) -> Tensor:
  if not math.isfinite(cap) or cap<=0:raise ValueError('Logit soft cap must be finite and positive')
  if (logits.device!='CPU' or logits.dtype is not dtypes.float or logits.shape!=logits.max_shape
      or not 1024<=logits.numel()<=4194304 or not .125<=cap<=128):
    return (logits/cap).tanh()*cap
  count = logits.numel()
  assert isinstance(count,int)
  cpus = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else 1
  state = _current.get()
  address=None
  if state is None and pool is not None and pool.started:
    address=pool.address if pool.active and cpus==1 else pool.prefill_address
  parallel = count>=131072 and (cpus>=4 or state is not None and state.pool.active or address is not None)
  return Tensor.empty(*logits.shape,device='CPU',dtype=dtypes.float).custom_kernel(logits.contiguous(),
    fxn=lambda o,x:_program(o,x,count,cap,parallel,address))[0]
