"""Bounded FP32 bidirectional vision attention for CPU/NPU inference.

Unlike decode attention, K and V stay FP32 and every query attends to every key.
Eight queries share each K/V load; probabilities stay in a per-thread tile.
"""
import functools
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _attention_rows, _fast_softmax_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

@functools.cache
def _vision_attention(out:UOp, q:UOp, k:UOp, v:UOp, heads:int, rows:int, dim:int) -> UOp:
  source = f'''{_fast_softmax_source()}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {{float *out;const float *q,*k,*v;int begin,end;}} Work;
  static void part(float *out,const float *query,const float *key,const float *value,int begin,int end) {{
    const int total={rows},active_rows={rows};
    for(int head=begin;head<end;head++) {{
      const float *keys=key+head*{rows*dim},*values=value+head*{rows*dim};int row=0;
      {_attention_rows(dim,rows,rows,block=8,channel_tile=2,fast_softmax=True,causal=False,kv_float=True)}
    }}
  }}
  static void *worker(void *opaque){{Work *w=(Work*)opaque;part(w->out,w->q,w->k,w->v,w->begin,w->end);return (void*)0;}}
  void rockchip_vision_attention(float *out,const float *q,const float *k,const float *v) {{
    Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,q,k,v,{heads}*i/4,{heads}*(i+1)/4}};
    for(int i=0;i<3;i++){{
      created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(out,q,k,v,work[i].begin,work[i].end);
    }}
    part(out,q,k,v,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
  }}'''
  renderer = _renderer()
  inputs = (out,q,k,v)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_vision_attention'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),
             UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))

def vision_attention(q:Tensor, k:Tensor, v:Tensor) -> Tensor:
  if any(t.device!='CPU' or t.dtype is not dtypes.float or t.shape!=q.shape for t in (q,k,v)):
    raise ValueError('Vision attention requires matching CPU FP32 Q/K/V tensors')
  if q.ndim!=4 or q.shape[0]!=1: raise ValueError('Vision attention requires one image with (batch, heads, patches, channels)')
  _,heads,rows,dim = q.shape
  if not all(isinstance(n,int) for n in (heads,rows,dim)) or not 1<=heads<=32 or not 8<=rows<=1024 or rows%8 or not 8<=dim<=256 or dim%8:
    raise ValueError('Vision attention exceeds its tested tile bounds')
  return Tensor.empty(*q.shape,device='CPU',dtype=dtypes.float).custom_kernel(q.contiguous(),k.contiguous(),v.contiguous(),
    fxn=lambda out,query,key,value:_vision_attention(out,query,key,value,heads,rows,dim))[0]
