"""Optional bounded FP32 RWKV7 scan for batched CPU/NPU prefill."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_recurrent import _whole_storage
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _scan(out:UOp, r:UOp, k:UOp, v:UOp, decay:UOp, rate:UOp, unit:UOp, state:UOp, start:UOp, rows:UOp,
          batch:int, heads:int, capacity:int, width:int, pool:int|None, weight:UOp|None=None, norm:UOp|None=None) -> UOp:
  body=f'#define RWKV_ORDERED_NORMALIZATION {int(norm is not None)}\n'+Path(__file__).with_suffix('.c').read_text()
  parallel=f'(rwkv_parallel_fn){pool}ULL' if pool is not None else '(rwkv_parallel_fn)0'
  source=body+f'''
void rockchip_cpu_rwkv7_scan(float *out,const float *r,const float *k,const float *v,
    const float *decay,const float *rate,const float *unit,float *state,const int *start,const int *rows{
      ',const float *weight,const float *norm' if norm is not None else ''}) {{
  rwkv_scan(out,r,k,v,decay,rate,unit,state,*start,*rows,{batch},{heads},{capacity},{width},{parallel},{
    'weight,norm' if norm is not None else '(const float*)0,(const float*)0'});
}}
'''
  inputs=(out,r,k,v,decay,rate,unit,state,start,rows)+((weight,norm) if weight is not None and norm is not None else ())
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_rwkv7_scan'))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,7),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),
    UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def rwkv_scan(r:Tensor, k:Tensor, v:Tensor, decay:Tensor, rate:Tensor, unit:Tensor, state:Tensor,
              start_pos:int|UOp, rows:int|UOp, *, pool:AttentionPool|None=None, prediction_key:Tensor|None=None,
              prediction_weight:Tensor|None=None, prediction_norm:Tensor|None=None, correction_key:Tensor|None=None) -> Tensor:
  """Consume valid B/T/H/C rows and mutate only the FP32 matrix state; reset at zero."""
  operands=(r,k,v,decay,rate,unit)
  if any(x.device!='CPU' or x.dtype is not dtypes.float for x in (*operands,state)):
    raise ValueError('CPU RWKV7 scan requires FP32 CPU operands')
  if r.ndim!=4 or any(x.shape!=r.shape or x.shape!=x.max_shape for x in operands):
    raise ValueError('CPU RWKV7 scan requires static (batch, capacity, heads, channels) operands')
  batch,capacity,heads,width=r.shape
  if (not all(isinstance(n,int) and n>=1 for n in r.shape) or not 1<=capacity<=256 or not 1<=width<=256
      or state.shape!=(batch,heads,width,width) or not _whole_storage(state)):
    raise ValueError('Unsupported CPU RWKV7 scan geometry or state storage')
  if (not isinstance(rows,(int,UOp)) or not isinstance(start_pos,(int,UOp)) or isinstance(rows,bool) or isinstance(start_pos,bool)
      or isinstance(rows,int) and not 1<=rows<=capacity or isinstance(rows,UOp) and not 1<=rows.vmin<=rows.vmax<=capacity
      or isinstance(start_pos,int) and not 0<=start_pos<=2147483647
      or isinstance(start_pos,UOp) and not 0<=start_pos.vmin<=start_pos.vmax<=2147483647):
    raise ValueError('Invalid CPU RWKV7 row count or position')
  assert isinstance(batch,int) and isinstance(capacity,int) and isinstance(heads,int) and isinstance(width,int)
  ordered=any(x is not None for x in (prediction_key,prediction_weight,prediction_norm,correction_key))
  if ordered:
    if any(x is None or x.device!='CPU' or x.dtype is not dtypes.float
           for x in (prediction_key,prediction_weight,prediction_norm,correction_key)):
      raise ValueError('Ordered RWKV7 normalization requires four FP32 CPU operands')
    assert prediction_key is not None and prediction_weight is not None and prediction_norm is not None and correction_key is not None
    if (prediction_key.shape!=r.shape or correction_key.shape!=r.shape or prediction_weight.shape!=(heads,width)
        or prediction_norm.shape!=(batch,capacity,heads,1)):
      raise ValueError('Invalid ordered RWKV7 normalization geometry')
  start=Tensor(start_pos,device='CPU',dtype=dtypes.int).reshape(1).clone()
  count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  out=Tensor.empty(batch,capacity,heads,width,device='CPU',dtype=dtypes.float)
  address=None
  if pool is not None and pool.active and batch*heads>=4:
    address=pool.address if capacity==1 and getenv('ONPU_RWKV_DECODE_SPIN',0) else pool.prefill_address
  if ordered:
    assert prediction_key is not None and prediction_weight is not None and prediction_norm is not None and correction_key is not None
    return out.custom_kernel(*(x.contiguous() for x in (r,k,v,decay,correction_key,prediction_key)),state,start,count,
      prediction_weight.contiguous(),prediction_norm.contiguous(),
      fxn=lambda o,a,b,c,d,e,f,s,p,n,w,z:_scan(o,a,b,c,d,e,f,s,p,n,batch,heads,capacity,width,address,w,z))[0]
  return out.custom_kernel(*(x.contiguous() for x in operands),state,start,count,
    fxn=lambda o,a,b,c,d,e,f,s,p,n:_scan(o,a,b,c,d,e,f,s,p,n,batch,heads,capacity,width,address))[0]


def configure(model, enabled:bool=True, *, pool:AttentionPool|None=None) -> int:
  from tinygrad.llm.rwkv import RWKV7Block
  if type(enabled) is not bool:raise ValueError('CPU RWKV7 scan selection must be a bool')
  blocks=[b for b in getattr(model,'blk',[]) if isinstance(b,RWKV7Block)]
  if not enabled:
    for b in blocks:
      hook=getattr(b,'rwkv_scan',None)
      if hook is rwkv_scan or isinstance(hook,functools.partial) and hook.func is rwkv_scan:delattr(b,'rwkv_scan')
    return 0
  if not blocks:raise ValueError('CPU RWKV7 scan requires RWKV7 blocks')
  for b in blocks:
    weight=getattr(getattr(b,'time_mix_receptance',None),'weight',None)
    if (not isinstance(weight,Tensor) or weight.device!='CPU' or not 1<=b.config.head_dim<=256
        or b.config.n_heads<1 or b.config.n_heads*b.config.head_dim!=b.config.dim):
      raise ValueError('Unsupported CPU RWKV7 block geometry or device')
  for b in blocks:
    hook=getattr(b,'rwkv_scan',None)
    if pool is None:setattr(b,'rwkv_scan',rwkv_scan)
    elif not (isinstance(hook,functools.partial) and hook.func is rwkv_scan and hook.keywords.get('pool') is pool):
      setattr(b,'rwkv_scan',functools.partial(rwkv_scan,pool=pool))
    if hasattr(b,'recurrent_state'):Tensor.realize(b.recurrent_state,b.cache_attn_x,b.cache_ffn_x)
  return len(blocks)
