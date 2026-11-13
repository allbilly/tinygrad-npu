"""Prepared CPU linear projections with FP16 weights and FP32 accumulation."""
from __future__ import annotations
import functools, math, os
from dataclasses import replace
from pathlib import Path
from typing import Callable, TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


def _pooled_source(source:str,address:int) -> str:
  if type(address) is not int or not 0<address<2**64:raise ValueError('Invalid static GEMM dispatcher')
  start=source.index('  Work work[4];unsigned long ids[3];int created[3];')
  end=source.index('\n}',start)
  replacement=f'''  Work work[4];
   for(int i=0;i<4;i++)work[i]=(Work){{out,x,w,rows,k,n,padded,blocks*i/4*8,blocks*(i+1)/4*8}};
   void *args[4]={{work,0,0,0}};
   if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(static_pool_callback,args,0))__builtin_trap();'''
  source=source[:start]+replacement+source[end:]
  needle='static void static_gemm('
  assert source.count(needle)==1
  callback='static void static_pool_callback(void **args,int unused,int group) {\n  (void)unused;worker(((Work*)args[0])+group);\n}\n'
  source=source.replace(needle,callback+needle)
  return source


def _eight_row_source(source:str) -> str:
  statements=['    for(;row+8<=rows;row+=8) {']
  statements += ['      F4 '+','.join(f'l{i}={{0}},h{i}={{0}}' for i in range(8))+';']
  statements += ['      const float '+','.join(f'*x{i}=x+(size_t)(row+{i})*k' for i in range(8))+';']
  statements += ['      for(int depth=0;depth<k;depth++) {',
                 '        const __fp16 *wptr=w+(size_t)(column/8)*k*8+depth*8;',
                 '        F4 lo=__builtin_convertvector(*(const H4*)wptr,F4),hi=__builtin_convertvector(*(const H4*)(wptr+4),F4);']
  statements += [f'        l{i}=l{i}+x{i}[depth]*lo;h{i}=h{i}+x{i}[depth]*hi;' for i in range(8)]
  statements += ['      }','      float *p=out+(size_t)row*n+column;']
  statements += [f'      put(p+{i}*n,l{i},left<4?left:4);if(left>4)put(p+{i}*n+4,h{i},left-4);' for i in range(8)]
  statements += ['    }']
  needle='    for(;row+4<=rows;row+=4) {'
  assert source.count(needle)==1
  return source.replace(needle,'\n'.join(statements)+'\n'+needle)


@functools.cache
def _program(out:UOp, inp:UOp, packed:UOp, rows:UOp, capacity:int, depth:int, columns:int, pool:int|None=None, row_block:int=4) -> UOp:
  if type(row_block) is not int or row_block not in (4,8):raise ValueError('Invalid static GEMM row block')
  source = Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_static_gemm(float *out,const float *input,const __fp16 *weight,const int *rows) {{
  if(*rows<1 || *rows>{capacity})__builtin_trap();
  static_gemm(out,input,weight,*rows,{depth},{columns},4);
  for(unsigned long i=(unsigned long)*rows*{columns};i<{capacity*columns}UL;i++)out[i]=0;
}}
'''
  if row_block==8:source=_eight_row_source(source)
  if pool is not None:source=_pooled_source(source,pool)
  inputs = (out,inp,packed,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_static_gemm'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _apply(x:Tensor, *, source:Tensor, packed:Tensor, columns:int, depth:int, pool:AttentionPool|None=None) -> Tensor:
  rows = math.prod(x.shape[:-1])
  capacity = math.prod(x.max_shape[:-1])
  if (x.device!='CPU' or x.dtype is not dtypes.float or x.ndim<2 or x.shape[-1]!=depth
      or not 1<=capacity<=4096 or isinstance(rows,int) and rows<1 or isinstance(rows,UOp) and rows.vmin<1):
    return x@source.T
  address=None
  if pool is not None and pool.started:
    pinned=hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))==1
    address=pool.address if pool.active and pinned else pool.prefill_address
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  out = Tensor.empty(*x.max_shape[:-1],columns,device='CPU',dtype=dtypes.float).custom_kernel(
    x.pad_to(x.max_shape).contiguous(),packed,count,
    fxn=lambda o,a,w,r:_program(o,a,w,r,capacity,depth,columns,address,getenv('ONPU_STATIC_GEMM_ROWS',4)))[0]
  return out.shrink(tuple((0,n) for n in (*x.shape[:-1],columns)))


def prepare(weight:Tensor, *, pool:AttentionPool|None=None) -> Callable[[Tensor],Tensor]:
  """Snapshot immutable weights before JIT capture; recreate captures after changing them."""
  if (weight.device!='CPU' or weight.dtype is not dtypes.half or weight.ndim!=2
      or not all(isinstance(n,int) and 1<=n<=4096 for n in weight.shape)):
    raise ValueError('Static CPU projections require a bounded FP16 CPU matrix')
  columns,depth = weight.shape
  assert isinstance(columns,int) and isinstance(depth,int)
  packed = weight.pad(((0,(-columns)%8),(0,0))).reshape(-1,8,depth).permute(0,2,1).contiguous().realize()
  # A partial retains the cache without adding derived tensors to state_dict.
  return functools.partial(_apply,source=weight,packed=packed,columns=columns,depth=depth,pool=pool)


def configure(model, enabled:bool=True, *, pool:AttentionPool|None=None) -> int:
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
    setattr(model,hook,[prepare(weight[index],pool=pool) for index in range(branches)])
    count += branches
  return count
