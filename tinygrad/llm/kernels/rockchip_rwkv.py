"""Optional bounded FP32 RWKV7 scan for batched CPU/NPU prefill."""
from __future__ import annotations
import functools
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_recurrent import _whole_storage
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _scan(out:UOp, r:UOp, k:UOp, v:UOp, decay:UOp, rate:UOp, unit:UOp, state:UOp, start:UOp, rows:UOp,
          batch:int, heads:int, capacity:int, width:int, pool:int|None, weight:UOp|None=None, norm:UOp|None=None, specialize:bool=False) -> UOp:
  body=f'#define RWKV_SCAN_WIDTH {64 if specialize and norm is not None and width==64 else 0}\n'+\
    f'#define RWKV_ORDERED_NORMALIZATION {int(norm is not None)}\n'+Path(__file__).with_suffix('.c').read_text()
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
              prediction_weight:Tensor|None=None, prediction_norm:Tensor|None=None, correction_key:Tensor|None=None, specialize:bool=False) -> Tensor:
  """Consume valid B/T/H/C rows and mutate only the FP32 matrix state; reset at zero."""
  if type(specialize) is not bool:raise ValueError('RWKV7 specialization must be a bool')
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
      fxn=lambda o,a,b,c,d,e,f,s,p,n,w,z:_scan(o,a,b,c,d,e,f,s,p,n,batch,heads,capacity,width,address,w,z,specialize))[0]
  return out.custom_kernel(*(x.contiguous() for x in operands),state,start,count,
    fxn=lambda o,a,b,c,d,e,f,s,p,n:_scan(o,a,b,c,d,e,f,s,p,n,batch,heads,capacity,width,address))[0]


_RWKV7_PREPARED_AUX = ('time_mix_lerp_fused','channel_mix_lerp_k','time_mix_w0','time_mix_a0',
                       'time_mix_k_k','time_mix_k_a','time_mix_r_k','time_mix_v0')


def prepare_weights(model:Any) -> int:
  """Materialize only RWKV7 learned norms and auxiliary vectors as bounded CPU tensors."""
  from tinygrad.llm.rwkv import RWKV7Block
  blocks=getattr(model,'blk',None)
  dim=getattr(getattr(model,'config',None),'dim',None)
  if (not isinstance(blocks,(list,tuple)) or not 1<=len(blocks)<=256 or type(dim) is not int or not 1<=dim<=8192
      or any(not isinstance(block,RWKV7Block) or type(getattr(block,'layer_id',None)) is not int
             or block.layer_id!=index or getattr(getattr(block,'config',None),'dim',None)!=dim for index,block in enumerate(blocks))):
    raise ValueError('CPU RWKV7 weight preparation requires a static RWKV7 model')

  entries:list[tuple[Any,str,Tensor]]=[]
  def add(owner:Any,key:str,name:str,shape:tuple[int,...]):
    value=owner.get(key) if isinstance(owner,dict) else getattr(owner,key,None)
    if (not isinstance(value,Tensor) or value.device!='CPU' or value.dtype not in (dtypes.half,dtypes.float)
        or any(type(size) is not int or size<1 for size in value.shape) or value.shape!=shape
        or value.numel()>6*dim):
      raise ValueError(f'Unsupported CPU RWKV7 prepared parameter {name}')
    entries.append((owner,key,value))

  # Validate every target before realizing or replacing any model field.
  for name in ('token_embd_norm','output_norm'):
    norm=getattr(model,name,None)
    if norm is None:raise ValueError(f'Missing RWKV7 learned norm {name}')
    for key in ('weight','bias'):add(norm,key,name+'.'+key,(dim,))
  for index,block in enumerate(blocks):
    for name in ('attn_norm','attn_norm_2','time_mix_ln'):
      norm=getattr(block,name,None)
      if norm is None:raise ValueError(f'Missing RWKV7 learned norm blk.{index}.{name}')
      for key in ('weight','bias'):add(norm,key,f'blk.{index}.{name}.{key}',(dim,))
    for name in _RWKV7_PREPARED_AUX:
      present=hasattr(block,name)
      if name=='time_mix_v0':
        if present != (index>0):raise ValueError(f'Invalid RWKV7 auxiliary parameter blk.{index}.{name}')
        if not present:continue
      if not present:raise ValueError(f'Missing RWKV7 auxiliary parameter blk.{index}.{name}')
      shape=(6,1,1,dim) if name=='time_mix_lerp_fused' else (dim,)
      add(getattr(block,name),'weight',f'blk.{index}.{name}.weight',shape)

  def is_materialized_contiguous(value:Tensor) -> bool:
    view=value.uop.contiguous_view()
    return value.uop.is_realized and view is not None and view[1]==0 and view[0].numel()==value.numel()

  for owner,key,value in entries:
    if is_materialized_contiguous(value):continue
    prepared=value.contiguous().clone().realize()
    if prepared.device!='CPU' or prepared.dtype is not value.dtype or prepared.shape!=value.shape:
      raise RuntimeError('RWKV7 prepared parameter changed device, dtype or shape')
    if isinstance(owner,dict):owner[key]=prepared
    else:setattr(owner,key,prepared)
  return len(entries)


def configure(model, enabled:bool=True, *, pool:AttentionPool|None=None, specialize:bool=False) -> int:
  from tinygrad.llm.rwkv import RWKV7Block
  if type(enabled) is not bool:raise ValueError('CPU RWKV7 scan selection must be a bool')
  if type(specialize) is not bool:raise ValueError('RWKV7 specialization must be a bool')
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
    if pool is None and not specialize:setattr(b,'rwkv_scan',rwkv_scan)
    elif not (isinstance(hook,functools.partial) and hook.func is rwkv_scan and hook.keywords.get('pool') is pool
              and hook.keywords.get('specialize',False)==specialize):
      setattr(b,'rwkv_scan',functools.partial(rwkv_scan,pool=pool,specialize=specialize))
    if hasattr(b,'recurrent_state'):Tensor.realize(b.recurrent_state,b.cache_attn_x,b.cache_ffn_x)
  return len(blocks)
