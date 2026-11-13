"""Optional bounded FP32 MLA projections with prepared FP16 CPU tables."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
from tinygrad.helpers import ContextVar

ONPU_MLA_CPU_FP16 = ContextVar('ONPU_MLA_CPU_FP16',0)
# Opt-in row/column-blocked projection tiling: bit-identical arithmetic, but one
# weight vector feeds 8 rows x 2 column vectors so the FP16 table is re-read half
# as often and each row group covers twice the output columns.
ONPU_MLA_PROJ_BLOCKED = ContextVar('ONPU_MLA_PROJ_BLOCKED',0)
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _program(out:UOp, inp:UOp, weight:UOp, rows:UOp, heads:int, capacity:int, depth:int, columns:int, pool:int|None,
             fp16:bool=False, blocked:bool=False) -> UOp:
  source=f'#define MLA_FP16 {int(fp16)}\n'+Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_mla_projection(float *out,const float *input,const __fp16 *weight,const int *rows) {{
  if(mla_project(out,input,weight,{heads},{capacity},*rows,{depth},{columns},{pool or 0}ULL,{int(blocked)}))__builtin_trap();
}}
'''
  inputs=(out,inp,weight,rows)
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_mla_projection'))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _supported(key:Tensor, value:Tensor) -> bool:
  return bool(all(w.device=='CPU' and w.dtype is dtypes.half and w.ndim==3
              and all(type(n) is int for n in w.shape) for w in (key,value))
    and key.shape[0]==value.shape[0] and 1<=key.shape[0]<=128 and key.shape[1]==value.shape[2]
    and all(64<=n<=4096 and n%4==0 for w in (key,value) for n in w.shape[1:]))


class CPUMLAProjections:
  def __init__(self,key:Tensor,value:Tensor,*,pool:AttentionPool|None=None,decode_pool:bool=False):
    if not _supported(key,value):raise ValueError('Unsupported CPU MLA projection tables')
    if type(decode_pool) is not bool:raise ValueError('CPU MLA decode pool selection must be a bool')
    if decode_pool and pool is None:raise ValueError('CPU MLA decode pooling requires a worker pool')
    self.heads=int(key.shape[0])
    self.decode_pool=decode_pool
    # Derived tables and worker handles must stay outside recursive state walks.
    self._sources=lambda:(key,value)
    self._packed:Callable[[],tuple[Tensor,...]|None]=lambda:None
    self._pool=lambda:pool
    self.packed_bytes=0

  def prepare(self):
    if self._packed() is not None:return
    packed=tuple(w.reshape(self.heads,int(w.shape[1])//4,4,int(w.shape[2])).permute(0,1,3,2).contiguous().realize()
                 for w in self._sources())
    self._packed=lambda:packed
    self.packed_bytes=sum(w.nbytes() for w in packed)

  def _run(self,x:Tensor,key:bool) -> Tensor:
    weight=self._sources()[0 if key else 1]
    if (x.device!='CPU' or x.dtype is not dtypes.float or x.ndim!=4 or x.shape[:2]!=(1,self.heads)
        or x.shape[-1]!=weight.shape[-1] or not 1<=x.max_shape[2]<=128):return x@weight.transpose(-1,-2)
    rows,capacity=x.shape[2],x.max_shape[2]
    if isinstance(rows,int) and rows<1 or isinstance(rows,UOp) and not 1<=rows.vmin<=rows.vmax<=capacity:
      return x@weight.transpose(-1,-2)
    assert isinstance(capacity,int)
    depth,columns=int(weight.shape[2]),int(weight.shape[1])
    packed=self._packed()
    if packed is None:raise RuntimeError('Prepare CPU MLA projection tables before capture')
    dense=capacity_input(x).reshape(self.heads,capacity,depth).contiguous()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    pool=self._pool()
    # The parked endpoint is safe both inside and outside the runtime scope;
    # graph construction need not coincide with an active inference pool.
    address=pool.prefill_address if pool is not None and self.heads>=4 and capacity>=16 else None
    if capacity==1 and self.decode_pool and pool is not None and pool.active:address=pool.address
    out=Tensor.empty(self.heads,capacity,columns,device='CPU',dtype=dtypes.float).custom_kernel(
      dense,packed[0 if key else 1],count,
      fxn=lambda o,x,w,n:_program(o,x,w,n,self.heads,capacity,depth,columns,address,
                                  bool(ONPU_MLA_CPU_FP16) and capacity==1,bool(ONPU_MLA_PROJ_BLOCKED)))[0]
    return out.reshape(1,self.heads,capacity,columns)[:,:,:rows]

  def query(self,x:Tensor) -> Tensor:return self._run(x,True)
  def value(self,x:Tensor) -> Tensor:return self._run(x,False)


def configure(model,enabled:bool=True,*,pool:AttentionPool|None=None,decode_pool:bool=False) -> int:
  """Prepare supported CPU projections before capture; serial dispatch is the default."""
  from tinygrad.llm.model import MLATransformerBlock
  if type(enabled) is not bool:raise ValueError('CPU MLA projection selection must be a bool')
  if type(decode_pool) is not bool:raise ValueError('CPU MLA decode pool selection must be a bool')
  if decode_pool and (not enabled or pool is None):raise ValueError('CPU MLA decode pooling requires projections and a worker pool')
  blocks=[b for b in getattr(model,'blk',[]) if isinstance(b,MLATransformerBlock)]
  for block in blocks:
    if isinstance(getattr(block,'cpu_mla_projections',None),CPUMLAProjections):delattr(block,'cpu_mla_projections')
  if not enabled or not blocks:return 0
  pairs=[(block.attn_k_b['weight'],block.attn_v_b['weight']) for block in blocks]
  if any(not _supported(k,v) for k,v in pairs) or sum(w.nbytes() for pair in pairs for w in pair)>256*1024**2:return 0
  for block,(key,value) in zip(blocks,pairs):
    helper=CPUMLAProjections(key,value,pool=pool,decode_pool=decode_pool)
    helper.prepare()
    setattr(block,'cpu_mla_projections',helper)
  return len(blocks)
