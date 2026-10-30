"""Prepared CPU linear projections with FP16 weights and FP32 accumulation."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from pathlib import Path
from typing import Callable
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _program(out:UOp, inp:UOp, packed:UOp, rows:UOp, capacity:int, depth:int, columns:int) -> UOp:
  source = Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_static_gemm(float *out,const float *input,const __fp16 *weight,const int *rows) {{
  if(*rows<1 || *rows>{capacity})__builtin_trap();
  static_gemm(out,input,weight,*rows,{depth},{columns},4);
  for(unsigned long i=(unsigned long)*rows*{columns};i<{capacity*columns}UL;i++)out[i]=0;
}}
'''
  inputs = (out,inp,packed,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_static_gemm'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _apply(x:Tensor, *, source:Tensor, packed:Tensor, columns:int, depth:int) -> Tensor:
  rows = math.prod(x.shape[:-1])
  capacity = math.prod(x.max_shape[:-1])
  if (x.device!='CPU' or x.dtype is not dtypes.float or x.ndim<2 or x.shape[-1]!=depth
      or not 1<=capacity<=4096 or isinstance(rows,int) and rows<1 or isinstance(rows,UOp) and rows.vmin<1):
    return x@source.T
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  out = Tensor.empty(*x.max_shape[:-1],columns,device='CPU',dtype=dtypes.float).custom_kernel(
    x.pad_to(x.max_shape).contiguous(),packed,count,
    fxn=lambda o,a,w,r:_program(o,a,w,r,capacity,depth,columns))[0]
  return out.shrink(tuple((0,n) for n in (*x.shape[:-1],columns)))


def prepare(weight:Tensor) -> Callable[[Tensor],Tensor]:
  """Snapshot immutable weights before JIT capture; recreate captures after changing them."""
  if (weight.device!='CPU' or weight.dtype is not dtypes.half or weight.ndim!=2
      or not all(isinstance(n,int) and 1<=n<=4096 for n in weight.shape)):
    raise ValueError('Static CPU projections require a bounded FP16 CPU matrix')
  columns,depth = weight.shape
  assert isinstance(columns,int) and isinstance(depth,int)
  packed = weight.pad(((0,(-columns)%8),(0,0))).reshape(-1,8,depth).permute(0,2,1).contiguous().realize()
  # A partial retains the cache without adding derived tensors to state_dict.
  return functools.partial(_apply,source=weight,packed=packed,columns=columns,depth=depth)


def configure(model, enabled:bool=True) -> int:
  """Prepare AltUp's static projections before tracing the model's JITs."""
  if type(enabled) is not bool:raise ValueError('Static CPU projection selection must be a bool')
  names = (('altup_proj','altup_input_projections'),('altup_unembd_proj','altup_output_projections'))
  if not enabled:
    for _,hook in names:
      if hasattr(model,hook):delattr(model,hook)
    return 0
  tables = [getattr(model,name,None) for name,_ in names]
  weights = [table.get('weight') if isinstance(table,dict) else None for table in tables]
  if any(not isinstance(w,Tensor) or w.device!='CPU' or w.dtype is not dtypes.half or w.ndim!=3
         or not all(isinstance(n,int) and 1<=n<=4096 for n in w.shape) for w in weights):return configure(model,enabled=False)
  typed_weights = [w for w in weights if isinstance(w,Tensor)]
  if sum(w.nbytes() for w in typed_weights)>256*1024**2:return configure(model,enabled=False)
  count = 0
  for (_,hook),weight in zip(names,typed_weights):
    branches = weight.shape[0]
    assert isinstance(branches,int)
    setattr(model,hook,[prepare(weight[index]) for index in range(branches)])
    count += branches
  return count
