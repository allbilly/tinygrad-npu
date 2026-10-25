"""CPU gate-half sigmoid with unchanged arithmetic and optional parked row workers."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _program(out:UOp, joint:UOp, count:UOp, capacity:int, depth:int, pool:int|None) -> UOp:
  source = Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_sigmoid_gate(float *out,const float *input,const int *rows) {{
  if(sigmoid_gate(out,input,*rows,{capacity},{depth},(sigmoid_parallel_fn){pool or 0}ULL))__builtin_trap();
}}
'''
  inputs = (out,joint,count)
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_sigmoid_gate'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def sigmoid_gate(joint:Tensor, *, pool:AttentionPool|None=None) -> Tensor:
  """Apply sigmoid to the first half of a joint gate/up projection."""
  if joint.device!='CPU' or joint.dtype is not dtypes.float or joint.ndim!=3 or joint.shape[0]!=1:
    raise ValueError('CPU sigmoid requires batch-one float32 gate/up projections')
  capacity,columns = joint.max_shape[1:]
  depth = columns//2
  if not 1<=capacity<=128 or columns%8 or not 4<=depth<=32768:
    raise ValueError('Unsupported CPU sigmoid projection shape')
  rows = joint.shape[1]
  if isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity):raise ValueError('Invalid CPU sigmoid row bounds')
  dense = joint.pad_to(joint.max_shape).contiguous()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  address = pool.prefill_address if pool is not None and pool.active else None
  result = Tensor.empty(1,capacity,depth,device='CPU',dtype=dtypes.float).custom_kernel(dense,count,
    fxn=lambda o,x,n:_program(o,x,n,capacity,depth,address))[0]
  return result.shrink(((0,1),(0,rows),(0,depth)))
