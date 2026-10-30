"""Bounded FP32 CPU MLA dots with the ordinary Tensor softmax retained."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_recurrent import _whole_storage
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp, resolve
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _program(out:UOp, inp:UOp, cache:UOp, start:UOp, rows:UOp, heads:int, capacity:int, context:int,
             width:int, values:int, score_stride:int, scale:float, pool:int|None, qk:bool, causal_work:bool=True) -> UOp:
  name='rockchip_cpu_mla_qk' if qk else 'rockchip_cpu_mla_pv'
  scratch=f'''float packed[{width*(((context+15)//16*16+4))}] __attribute__((aligned(16)));
  for(int c=0;c<{width};c++) {{
    for(int key=0;key<*start+*rows;key++)packed[c*{((context+15)//16*16+4)}+key]=cache[key*{width}+c];
    for(int key=*start+*rows;key<((*start+*rows+15)&~15);key++)packed[c*{((context+15)//16*16+4)}+key]=0;
  }}''' if qk else ''
  elements=heads*capacity*(score_stride if qk else values)
  source=f'#define MLA_CAUSAL_WORK {int(causal_work)}\n'+Path(__file__).with_suffix('.c').read_text()+f'''
void {name}(float *out,const float *input,const float *cache,const int *start,const int *rows) {{
  if(*start<0||*rows<1||*rows>{capacity}||*start+*rows>{context})__builtin_trap();
  for(unsigned long i=0;i<{elements}UL;i++)out[i]=0;
  {scratch}
  MLAWork work={{out,input,cache,{'packed' if qk else '(const float*)0'},{heads},{capacity},*rows,*start,
    {context},{width},{values},{score_stride},{float(scale).hex()}f,{{0,0,0,0}}}};
  mla_execute(&work,(mla_parallel_fn){pool or 0}ULL,{int(qk)});
}}
'''
  inputs=(out,inp,cache,start,rows)
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo(name))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _bounds(start:int|UOp, rows:int|UOp, capacity:int, context:int):
  if (isinstance(start,bool) or not isinstance(start,(int,UOp)) or isinstance(rows,bool) or not isinstance(rows,(int,UOp))
      or isinstance(start,int) and not 0<=start<context or isinstance(start,UOp) and not 0<=start.vmin<=start.vmax<context
      or isinstance(rows,int) and not 1<=rows<=capacity or isinstance(rows,UOp) and not 1<=rows.vmin<=rows.vmax<=capacity
      or isinstance(start,int) and isinstance(rows,int) and start+rows>context):
    raise ValueError('Invalid CPU MLA position or row bounds')


class MLAAttention:
  def __init__(self, heads:int, context:int, width:int, values:int, scale:float, pool:AttentionPool|None=None):
    if (not all(type(n) is int for n in (heads,context,width,values)) or not 1<=heads<=128 or not 1<=context<=1024
        or not 4<=values<=width<=512 or width%4 or values%4 or width*(((context+15)//16*16+4))*4>1024**2
        or not math.isfinite(scale) or scale<=0):raise ValueError('Unsupported CPU MLA geometry')
    self.heads,self.context,self.width,self.values,self.scale=heads,context,width,values,scale
    self.causal_work=bool(getenv('ONPU_MLA_CAUSAL_WORK',1))
    self._pool=lambda:pool  # Keep worker-library internals out of state-dict walks.

  def _validate(self, x:Tensor, cache:Tensor, start:int|UOp):
    if (x.device!='CPU' or cache.device!='CPU' or x.dtype is not dtypes.float or cache.dtype is not dtypes.float
        or x.ndim!=4 or x.shape[:2]!=(1,self.heads) or not 1<=x.max_shape[2]<=128
        or cache.shape!=(1,1,self.context,self.width) or not _whole_storage(cache)):
      raise ValueError('CPU MLA requires bounded FP32 CPU queries/probabilities and a whole latent cache')
    capacity=x.max_shape[2]
    assert isinstance(capacity,int)
    rows=x.shape[2]
    _bounds(start,rows,capacity,self.context)
    return rows,capacity

  def _run(self, x:Tensor, cache:Tensor, start:int|UOp, qk:bool) -> Tensor:
    rows,capacity=self._validate(x,cache,start)
    if qk and x.shape[-1]!=self.width or not qk and x.shape[-1]!=start+rows:
      raise ValueError('CPU MLA input feature or probability shape differs')
    # Small/static reductions may be fully unrolled with a different first
    # multiply/add contraction. Keep that compiler path exactly as written.
    if self.width<64 or self.values<32 or isinstance(start,int) and isinstance(rows,int):
      prefix=cache[:,:,:start+rows]
      if not qk:return x@prefix[...,:self.values]
      scores=x@prefix.transpose(-1,-2)*self.scale
      if resolve(rows!=1):
        scores=scores+Tensor.full((1,1,rows,start+rows),-math.inf,device='CPU',dtype=dtypes.float).triu(start+1)
      return scores
    stride=self.context+capacity-1
    shape=(1,self.heads,capacity,stride if qk else self.values)
    dense=x.pad_to((1,self.heads,capacity,self.width if qk else stride)).contiguous()
    position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    address=None
    pool=self._pool()
    if pool is not None and pool.active and self.heads>=4:
      address=pool.address if capacity==1 else pool.prefill_address
    out=Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(dense,cache,position,count,
      fxn=lambda o,a,k,s,n:_program(o,a,k,s,n,self.heads,capacity,self.context,self.width,self.values,stride,self.scale,address,qk,
                                  self.causal_work))[0]
    return out.shrink(((0,1),(0,self.heads),(0,rows),(0,start+rows if qk else self.values)))

  def query_key(self, query:Tensor, cache:Tensor, start:int|UOp) -> Tensor:return self._run(query,cache,start,True)
  def weighted_value(self, probabilities:Tensor, cache:Tensor, start:int|UOp) -> Tensor:return self._run(probabilities,cache,start,False)


def _hooks(model, pool:AttentionPool|None=None):
  from tinygrad.llm.model import MLATransformerBlock
  blocks=[b for b in getattr(model,'blk',[]) if isinstance(b,MLATransformerBlock)]
  hooks=[]
  for block in blocks:
    c=block.config
    if block.indexer is not None or block.attn_k_b['weight'].device!='CPU':return []
    try:hooks.append(MLAAttention(c.n_heads,c.max_context,c.kv_lora_rank+c.rope_dim,c.kv_lora_rank,1/c.head_dim**.5,pool))
    except ValueError:return []
  return list(zip(blocks,hooks))


def supported(model) -> bool:return bool(_hooks(model))


def configure(model, enabled:bool=True, *, pool:AttentionPool|None=None) -> int:
  from tinygrad.llm.model import MLATransformerBlock
  if type(enabled) is not bool:raise ValueError('CPU MLA attention selection must be a bool')
  for block in getattr(model,'blk',[]):
    if isinstance(block,MLATransformerBlock) and isinstance(getattr(block,'cpu_mla_attention',None),MLAAttention):
      delattr(block,'cpu_mla_attention')
  hooks=_hooks(model,pool) if enabled else []
  for block,hook in hooks:
    block.cpu_mla_attention=hook
    if hasattr(block,'cache_k'):block.cache_k.realize()
  return len(hooks)
