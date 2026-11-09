"""Bounded FP32 CPU MLA dots with the ordinary Tensor softmax retained."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv, ContextVar
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_recurrent import _whole_storage
from tinygrad.llm.kernels.rockchip_padding import capacity_input, mark_zero_tail
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp, resolve
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_MLA_FUSED_SOFTMAX = ContextVar('ONPU_MLA_FUSED_SOFTMAX',0)
ONPU_MLA_ATTN_FP16 = ContextVar('ONPU_MLA_ATTN_FP16',0)


@functools.cache
def _program(out:UOp, inp:UOp, rope_inp:UOp, cache:UOp, start:UOp, rows:UOp, heads:int, capacity:int, context:int,
             width:int, rope_width:int, values:int, score_stride:int, scale:float, pool:int|None, qk:bool,
             causal_work:bool=True, fp16:bool=False) -> UOp:
  name='rockchip_cpu_mla_qk' if qk else 'rockchip_cpu_mla_pv'
  scratch=f'''float packed[{width*(((context+15)//16*16+4))}] __attribute__((aligned(16)));
  for(int c=0;c<{width};c++) {{
    for(int key=0;key<*start+*rows;key++)packed[c*{((context+15)//16*16+4)}+key]=cache[key*{width}+c];
    for(int key=*start+*rows;key<((*start+*rows+15)&~15);key++)packed[c*{((context+15)//16*16+4)}+key]=0;
  }}''' if qk else ''
  elements=heads*capacity*(score_stride if qk else values)
  split=rope_width>0
  args=('float *out,const float *input,const float *input_rope,const float *cache,const int *start,const int *rows' if split
        else 'float *out,const float *input,const float *cache,const int *start,const int *rows')
  work=f'out,input,{("input_rope" if split else "(const float*)0")},cache,{("packed" if qk else "(const float*)0")}'
  source=f'#define MLA_CAUSAL_WORK {int(causal_work)}\n#define MLA_FP16 {int(fp16)}\n'+Path(__file__).with_suffix('.c').read_text()+f'''
void {name}({args}) {{
  if(*start<0||*rows<1||*rows>{capacity}||*start+*rows>{context})__builtin_trap();
  for(unsigned long i=0;i<{elements}UL;i++)out[i]=0;
  {scratch}
  MLAWork work={{{work},{heads},{capacity},*rows,*start,
    {context},{width},{width-rope_width},{rope_width},{values},{score_stride},{float(scale).hex()}f,{{0,0,0,0}}}};
  mla_execute(&work,(mla_parallel_fn){pool or 0}ULL,{int(qk)});
}}
'''
  inputs=(out,inp,rope_inp,cache,start,rows) if split else (out,inp,cache,start,rows)
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo(name))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
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
    self.fused_softmax=bool(ONPU_MLA_FUSED_SOFTMAX)
    self._pool=lambda:pool  # Keep worker-library internals out of state-dict walks.

  def _validate(self, x:Tensor, cache:Tensor, start:int|UOp, rope:Tensor|None=None):
    if (x.device!='CPU' or cache.device!='CPU' or x.dtype is not dtypes.float or cache.dtype is not dtypes.float
        or x.ndim!=4 or x.shape[:2]!=(1,self.heads) or not 1<=x.max_shape[2]<=128
        or cache.shape!=(1,1,self.context,self.width) or not _whole_storage(cache)):
      raise ValueError('CPU MLA requires bounded FP32 CPU queries/probabilities and a whole latent cache')
    capacity=x.max_shape[2]
    assert isinstance(capacity,int)
    rows=x.shape[2]
    if rope is not None and (rope.device!='CPU' or rope.dtype is not dtypes.float or rope.ndim!=4
        or rope.shape[:2]!=(1,self.heads) or not 1<=rope.max_shape[2]<=capacity
        or not isinstance(rope.shape[-1],int) or not 0<rope.shape[-1]<self.width
        or not isinstance(x.shape[-1],int) or x.shape[-1]+rope.shape[-1]!=self.width
        or not resolve(rope.shape[2]==x.shape[2])):
      raise ValueError('CPU MLA split queries require matching FP32 CPU rope rows')
    _bounds(start,rows,capacity,self.context)
    return rows,capacity

  def _run(self, x:Tensor, cache:Tensor, start:int|UOp, qk:bool, rope:Tensor|None=None) -> Tensor:
    rows,capacity=self._validate(x,cache,start,rope)
    rope_width=0 if rope is None else rope.shape[-1]
    assert isinstance(rope_width,int)
    if qk and x.shape[-1]!=self.width-rope_width or not qk and x.shape[-1]!=start+rows:
      raise ValueError('CPU MLA input feature or probability shape differs')
    # Small/static reductions may be fully unrolled with a different first
    # multiply/add contraction. Keep that compiler path exactly as written.
    if self.width<64 or self.values<32 or isinstance(start,int) and isinstance(rows,int):
      prefix=cache[:,:,:start+rows]
      if not qk:return x@prefix[...,:self.values]
      scores=(x.cat(rope,dim=-1) if rope is not None else x)@prefix.transpose(-1,-2)*self.scale
      if resolve(rows!=1):
        scores=scores+Tensor.full((1,1,rows,start+rows),-math.inf,device='CPU',dtype=dtypes.float).triu(start+1)
      return scores
    stride=self.context+capacity-1
    shape=(1,self.heads,capacity,stride if qk else self.values)
    dense=(capacity_input(x) if self.fused_softmax else x.pad_to(
      (1,self.heads,capacity,self.width-rope_width if qk else stride))).contiguous()
    dense_rope=None if rope is None else (capacity_input(rope) if self.fused_softmax else rope.pad_to(
      (1,self.heads,capacity,rope_width))).contiguous()
    position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    address=None
    pool=self._pool()
    if pool is not None and pool.active and self.heads>=4:
      address=pool.address if capacity==1 else pool.prefill_address
    fp16=bool(ONPU_MLA_ATTN_FP16) and capacity==1 and not qk
    if dense_rope is None:
      out=Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(dense,cache,position,count,
        fxn=lambda o,a,k,s,n:_program(o,a,a,k,s,n,self.heads,capacity,self.context,self.width,0,self.values,stride,self.scale,
                                      address,qk,self.causal_work,fp16))[0]
    else:
      out=Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(dense,dense_rope,cache,position,count,
        fxn=lambda o,a,r,k,s,n:_program(o,a,r,k,s,n,self.heads,capacity,self.context,self.width,rope_width,self.values,stride,
                                      self.scale,address,qk,self.causal_work,fp16))[0]
    logical=out.shrink(((0,1),(0,self.heads),(0,rows),(0,start+rows if qk else self.values)))
    return mark_zero_tail(logical,out,'mla_scores' if qk else 'mla_context') if self.fused_softmax else logical

  def softmax(self, scores:Tensor, start:int|UOp) -> Tensor:
    if not self.fused_softmax or self.heads!=40 or self.context!=512 or self.width!=288 or self.values!=256:
      return scores.softmax(-1)
    from tinygrad.llm.kernels.rockchip_mla_softmax import softmax
    return softmax(scores,start,pool=self._pool())

  def query_key(self, query:Tensor, cache:Tensor, start:int|UOp, *, rope:Tensor|None=None) -> Tensor:
    return self._run(query,cache,start,True,rope)
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
