"""Opt-in prepared NPU causal prefill, retaining the bounded CPU fallback."""
import functools, struct
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, prefill_attention as cpu_prefill_attention
from tinygrad.llm.kernels.rockchip_padding import capacity_input, mark_zero_tail
from tinygrad.runtime.support.rockchip_attention import ATTENTION_MAGIC
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _program(out:UOp,query:UOp,cache:UOp,length:UOp,rows:UOp,context:int,capacity:int,pooled:bool,fast_softmax:bool,
             heads:int,kv_heads:int,dim:int) -> UOp:
  inputs=(out,query,cache,length,rows)
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_prefill_attention'))
  info=replace(ProgramInfo.from_sink(sink,_renderer().target),outs=(0,),ins=(1,2,3,4))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),
    UOp(Ops.SOURCE,arg='Prepared causal FP16 NPU QK/PV with dynamic CPU K/V cache and FP32 CPU softmax'),
    UOp(Ops.BINARY,arg=struct.pack('<8I',ATTENTION_MAGIC,heads,kv_heads,dim,context,capacity,12|int(pooled)|(int(fast_softmax)<<1),1))))


def _integer_buffer(value:int|UOp) -> Tensor:
  return Tensor([value] if isinstance(value,int) else value,device='CPU',dtype=dtypes.int).reshape(1)


def prefill_attention(query:Tensor,cache:Tensor,length:int|UOp,fast_softmax:bool=False,pool=None,*,
                      score_scale:float|None=None,window:int=0,softcap:float=0.0) -> Tensor:
  eligible=(query.device==cache.device=='CPU' and query.dtype is dtypes.float and cache.dtype is dtypes.half
    and query.ndim==4 and cache.ndim==5 and query.max_shape[0]==1
    and (query.max_shape[1],cache.shape[2],query.max_shape[-1]) in ((16,8,128),(32,4,64))
    and 2<=query.max_shape[2]<=128 and cache.shape[:2]==(2,1) and cache.shape[-1]==query.max_shape[-1]
    and cache.shape[3] in (64,128,256,512) and not window and not softcap and score_scale is None)
  if not eligible:
    return cpu_prefill_attention(query,cache,length,fast_softmax,pool,score_scale=score_scale,window=window,softcap=softcap)
  context,capacity=int(cache.shape[3]),query.max_shape[2]
  heads,kv_heads,dim=int(query.max_shape[1]),int(cache.shape[2]),int(query.max_shape[-1])
  total=_integer_buffer(length)
  rows=_integer_buffer(query.shape[2])
  # Token-major physical storage makes the model's following transpose a view.
  out=Tensor.empty(1,capacity,heads,dim,device='CPU',dtype=dtypes.float)
  result=out.custom_kernel(capacity_input(query).contiguous(),cache,total,rows,
    fxn=lambda o,q,k,l,r:_program(o,q,k,l,r,context,capacity,pool is not None and pool.started,fast_softmax,heads,kv_heads,dim))[0]
  full=result.transpose(1,2)
  return mark_zero_tail(full.shrink(tuple((0,s) for s in query.shape)),full,'attention')
