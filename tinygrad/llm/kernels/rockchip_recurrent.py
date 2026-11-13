"""Experimental FP32 CPU recurrent scans with explicit opt-in placement."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_GDN_DECODE_SPIN = ContextVar('ONPU_GDN_DECODE_SPIN',0)
ONPU_GDN_DIRECT_DECODE = ContextVar('ONPU_GDN_DIRECT_DECODE',0)


@functools.cache
def _scan(out:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, start:UOp, rows:UOp,
          batch:int, heads:int, capacity:int, key_dim:int, value_dim:int, alpha_dim:int, history:UOp|None=None,
          pool:int|None=None, direct_decode:bool=False) -> UOp:
  body = Path(__file__).with_suffix('.c').read_text().replace('int cpu_gated_delta_scan(', 'static int cpu_gated_delta_scan(').replace(
    'int cpu_gated_delta_scan_pooled(', 'static int cpu_gated_delta_scan_pooled(')
  arguments = f'out,q,k,v,beta,alpha,state,*start,*rows,{batch},{heads},{capacity},{key_dim},{value_dim},{alpha_dim},' + \
    ('history' if history is not None else 'NULL')
  # Dispatch errors must never trigger another stateful scan. Fail the worker
  # rather than consume a partially computed result after an internal pool fault.
  call = f'if(cpu_gated_delta_scan_pooled({arguments},(scan_parallel_fn){pool}ULL,NULL))__builtin_trap();' if pool is not None else \
    f'cpu_gated_delta_scan({arguments});'
  source = f'#define ONPU_GDN_DIRECT_DECODE {int(direct_decode)}\n' + body + f'''
void rockchip_cpu_gated_delta_scan(float *out, const float *q, const float *k, const float *v,
    const float *beta, const float *alpha, float *state, const int *start, const int *rows
    {', float *history' if history is not None else ''}) {{
  {call}
}}
'''
  inputs = (out,q,k,v,beta,alpha,state,start,rows)+((history,) if history is not None else ())
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_gated_delta_scan'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,6,9) if history is not None else (0,6),ins=tuple(range(1,9)))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),
    UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _whole_storage(x:Tensor) -> bool:
  storage = x.uop
  while storage.op in (Ops.RESHAPE,Ops.AFTER):storage=storage.src[0]
  return storage.op in (Ops.BUFFER,Ops.PARAM) and storage.numel()==x.numel()


def configure(model, enabled:bool=True, *, pool:AttentionPool|None=None) -> int:
  """Opt into the CPU hook before prefill chunk selection and JIT warmup."""
  from tinygrad.llm.model import GatedDeltaNetBlock
  if type(enabled) is not bool:raise ValueError('CPU recurrent scan selection must be a bool')
  blocks=[block for block in getattr(model,'blk',[]) if isinstance(block,GatedDeltaNetBlock)]
  if not blocks:
    from tinygrad.llm.rwkv import RWKV7Block
    if any(isinstance(b,RWKV7Block) for b in getattr(model,'blk',[])):
      from tinygrad.llm.kernels.rockchip_rwkv import configure as configure_rwkv
      return configure_rwkv(model,enabled=enabled,pool=pool)
  if not enabled:
    for block in blocks:
      hook = getattr(block,'gated_delta_scan',None)
      if hook is gated_delta_scan or isinstance(hook,functools.partial) and hook.func is gated_delta_scan:delattr(block,'gated_delta_scan')
    return 0
  if not blocks:raise ValueError('CPU recurrent scan requires GatedDeltaNet blocks')
  for block in blocks:
    weight=getattr(getattr(block,'attn_qkv',None),'weight',None)
    if (not isinstance(weight,Tensor) or weight.device!='CPU' or not 1<=block.head_k_dim<=256
        or not 1<=block.head_v_dim<=256 or block.num_v_heads<1):
      raise ValueError('Unsupported CPU recurrent block geometry or device')
  for block in blocks:
    # Installation is idempotent: the benchmark selects its chunk before the
    # general matrix-placement setup, which can call this helper again.
    hook = getattr(block,'gated_delta_scan',None)
    if pool is None:setattr(block,'gated_delta_scan',gated_delta_scan)
    elif not (isinstance(hook,functools.partial) and hook.func is gated_delta_scan and hook.keywords.get('pool') is pool):
      setattr(block,'gated_delta_scan',functools.partial(gated_delta_scan,pool=pool))
    if hasattr(block,'recurrent_state'):Tensor.realize(block.conv_state,block.recurrent_state)
  return len(blocks)


def gated_delta_scan(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                     start_pos:int|UOp, rows:int|UOp, *, history:Tensor|None=None, pool:AttentionPool|None=None) -> Tensor:
  """Process valid rows; reset at position zero and leave padding out of the state."""
  if q.device!='CPU' or any(x.device!=q.device or x.dtype is not dtypes.float for x in (q,k,v,beta,alpha,state)):
    raise ValueError('CPU recurrent scan requires FP32 CPU operands')
  if q.ndim!=4 or v.ndim!=4 or alpha.ndim not in (3,4):raise ValueError('Expected (batch, heads, capacity, channels) operands')
  batch,heads,capacity,key_dim = q.max_shape
  value_dim,alpha_dim = v.max_shape[-1],1 if alpha.ndim==3 else alpha.max_shape[-1]
  if (q.shape!=q.max_shape or k.shape!=q.shape or v.shape!=(batch,heads,capacity,value_dim)
      or beta.shape!=(batch,heads,capacity) or alpha.shape not in ((batch,heads,capacity),(batch,heads,capacity,alpha_dim))
      or state.shape!=(batch,heads,value_dim,key_dim) or alpha_dim not in (1,key_dim)
      or batch<1 or heads<1 or not 1<=key_dim<=256 or not 1<=value_dim<=256 or not 1<=capacity<=256
      or not _whole_storage(state)):raise ValueError('Unsupported CPU recurrent scan geometry')
  if (isinstance(rows,int) and not 1<=rows<=capacity or isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity)
      or isinstance(start_pos,int) and not 0<=start_pos<=2147483647
      or isinstance(start_pos,UOp) and (start_pos.vmin<0 or start_pos.vmax>2147483647)):
    raise ValueError('Invalid CPU recurrent scan row count or position')
  if history is not None:
    history_capacity = history.max_shape[0] if history.ndim==5 else 0
    if (alpha_dim!=1 or not capacity<=history_capacity<=8 or history.shape!=(history_capacity,batch,heads,value_dim,key_dim)
        or history.device!=q.device or history.dtype is not dtypes.float or not _whole_storage(history)):
      raise ValueError('Invalid bounded FP32 CPU prefix state history')
  start = Tensor(start_pos,device='CPU',dtype=dtypes.int).reshape(1).clone()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  out = Tensor.empty(batch,heads,capacity,value_dim,device='CPU',dtype=dtypes.float)
  address = ((pool.address if capacity==1 and ONPU_GDN_DECODE_SPIN else pool.prefill_address)
             if pool is not None and pool.active and batch*heads>=4 else None)
  direct=bool(ONPU_GDN_DIRECT_DECODE)
  if history is not None:
    return out.custom_kernel(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),alpha.contiguous(),state,start,count,history,
      fxn=lambda o,a,b,c,d,e,s,p,r,h:_scan(o,a,b,c,d,e,s,p,r,batch,heads,capacity,key_dim,value_dim,alpha_dim,h,address,direct_decode=direct))[0]
  return out.custom_kernel(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),alpha.contiguous(),state,start,count,
    fxn=lambda o,a,b,c,d,e,s,p,r:_scan(o,a,b,c,d,e,s,p,r,batch,heads,capacity,key_dim,value_dim,alpha_dim,pool=address,direct_decode=direct))[0]
