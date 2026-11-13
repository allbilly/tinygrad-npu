"""Opt-in exact softmax for the qualified dynamic MiniCPM3 CPU score layout."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_padding import capacity_input, mark_zero_tail
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _program(out:UOp, scores:UOp, start:UOp, rows:UOp, capacity:int, address:int|None) -> UOp:
  # Keep the tested per-head sequential max/sum and exact exp2 polynomial.
  # The output includes zeroed inactive rows/columns, permitting view reuse.
  source=_vector_exp2_source()+Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_mla_softmax(float *out,const float *scores,const int *start,const int *rows) {{
  if(*start<0||*rows<1||*rows>{capacity}||*start+*rows>512)__builtin_trap();
  SoftmaxWork work={{out,scores,*rows,*start,{capacity},512,{512+capacity-1}}};
  if({address or 0}ULL) {{
    void *args[4]={{&work,0,0,0}};
    if(!((int(*)(void(*)(void**,int,int),void**,int)){address or 0}ULL)(softmax_pool_callback,args,0))return;
  }}
  softmax_head_range(out,scores,*rows,*start,{capacity},512,{512+capacity-1},0,40);
}}
'''
  inputs=(out,scores,start,rows)
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_mla_softmax'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def softmax(scores:Tensor, start:int|UOp, *, pool:AttentionPool|None=None) -> Tensor:
  from tinygrad.llm.kernels.rockchip_mla_attention import _bounds
  if (scores.device!='CPU' or scores.dtype is not dtypes.float or scores.ndim!=4 or scores.shape[:2]!=(1,40)
      or scores.max_shape[2] not in (1,128)):
    return scores.softmax(-1)
  capacity=scores.max_shape[2]
  assert isinstance(capacity,int)
  rows=scores.shape[2]
  # Static reductions may have another contraction/reduction tree. Preserve it.
  if (isinstance(start,int) and isinstance(rows,int) or scores.max_shape[-1]!=512+capacity-1
      or scores.shape[-1]!=start+rows):return scores.softmax(-1)
  _bounds(start,rows,capacity,512)
  dense=capacity_input(scores).contiguous()
  position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
  count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  address=None
  if pool is not None:
    if capacity>=16:address=pool.prefill_address
    elif pool.active:address=pool.address
  full=Tensor.empty(1,40,capacity,512+capacity-1,device='CPU',dtype=dtypes.float).custom_kernel(
    dense,position,count,fxn=lambda o,s,p,n:_program(o,s,p,n,capacity,address))[0]
  logical=full[:,:,:rows,:start+rows]
  return mark_zero_tail(logical,full,'mla_softmax')
