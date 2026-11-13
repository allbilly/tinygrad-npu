"""Bounded FP32 CPU MLA dots with the ordinary Tensor softmax retained."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv, ContextVar
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_recurrent import _whole_storage
from tinygrad.llm.kernels.rockchip_padding import capacity_input, mark_zero_tail
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp, resolve
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_MLA_FUSED_SOFTMAX = ContextVar('ONPU_MLA_FUSED_SOFTMAX',0)
ONPU_MLA_ATTN_FP16 = ContextVar('ONPU_MLA_ATTN_FP16',0)
# Opt-in (default 0): process MLA_HEAD_BLOCK heads per K/cache pass in the
# single-row decode kernels so the shared K block / latent cache row is loaded
# once for several heads instead of once per head. Each output keeps its
# ascending-c/key accumulation, so results are bit-identical.
ONPU_MLA_HEAD_BLOCK = ContextVar('ONPU_MLA_HEAD_BLOCK',1)
# Opt-in (default 0): row-paired prefill attention fast path. Two rows share every
# K tile (8-key tiles, 2*4*2 = 16 M4 accumulators) so the K traffic halves and the
# 10 head-blocks become 20 (block,row-half) work units over the 4 pool workers.
# Per-output FP order is unchanged, so outputs are bit-identical; only used when the
# fused attention/qk-softmax path is enabled and the head count is a multiple of 4.
ONPU_MLA_ATTN_ROWS = ContextVar('ONPU_MLA_ATTN_ROWS',1)
# Opt-in (default 0): fuse qk + the exact softmax into one kernel on the prefill
# path. The padded FP32 score buffer is never materialised: each row's scores stay
# in a small local buffer, removing the 13 MB score write + read + memset per
# layer while producing bit-identical probabilities.
ONPU_MLA_QK_SOFTMAX = ContextVar('ONPU_MLA_QK_SOFTMAX',1)
# Opt-in: the fused prefill kernel consumes the query/rope projections' buffers directly instead of
# materialising `capacity_input(x).contiguous()` first. The projection writes
# `(head*capacity+row)*columns+column` (mla_out_offset, transposed=0) and the kernel reads
# `input[(head*capacity+row)*nope+c]` -- the same head-major layout -- so the copy is pure overhead.
# Valid only when rows==capacity and max_shape already equals the padded shape; else the old path.
# Bit-exact: the kernel reads the same bytes it would have read from the copy.
ONPU_MLA_ATTN_DIRECT_VALUE = ContextVar('ONPU_MLA_ATTN_DIRECT_VALUE',0)

# Opt-in (default 0): additionally fuse the weighted value into the same prefill
# kernel, so the probability matrix never reaches memory either (13 MB write +
# read per layer removed). Same exact pv accumulation/shortcut/tail guard.
ONPU_MLA_FUSED_ATTN = ContextVar('ONPU_MLA_FUSED_ATTN',1)
# Opt-in (default 0): one fused kernel splits the q_b projection output into the
# padded head-major q_nope and the rotated q_rope, replacing the transpose copy,
# the q_nope pad and the separate rope kernels (pure layout + exact elementwise).
ONPU_MLA_FUSED_QUERY = ContextVar('ONPU_MLA_FUSED_QUERY',1)


@functools.cache
def _program(out:UOp, inp:UOp, rope_inp:UOp, cache:UOp, start:UOp, rows:UOp, heads:int, capacity:int, context:int,
             width:int, rope_width:int, values:int, score_stride:int, scale:float, pool:int|None, qk:bool,
             causal_work:bool=True, fp16:bool=False, head_block:int=0) -> UOp:
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
  source=f'#define MLA_CAUSAL_WORK {int(causal_work)}\n#define MLA_FP16 {int(fp16)}\n#define MLA_HEAD_BLOCK {int(head_block)}\n'+Path(__file__).with_suffix('.c').read_text()+f'''
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


@functools.cache
def _fused_program(out:UOp, inp:UOp, rope_inp:UOp, cache:UOp, start:UOp, rows:UOp, heads:int, capacity:int,
                   context:int, width:int, rope_width:int, values:int, stride:int, scale:float, pool:int|None,
                   causal_work:bool=True, head_block:int=0, pv:bool=False, attn_rows:int=0) -> UOp:
  name='rockchip_cpu_mla_fused_attn' if pv else 'rockchip_cpu_mla_qk_softmax'
  pstride=((context+15)//16*16+4)
  scratch=('float packed['+str(width*pstride)+'] __attribute__((aligned(16)));\n'
           '  for(int c=0;c<'+str(width)+';c++) {\n'
           '    for(int key=0;key<*start+*rows;key++)packed[c*'+str(pstride)+'+key]=cache[key*'+str(width)+'+c];\n'
           '    for(int key=*start+*rows;key<((*start+*rows+15)&~15);key++)packed[c*'+str(pstride)+'+key]=0;\n'
           '  }')
  split=rope_width>0
  args=('float *out,const float *input,const float *input_rope,const float *cache,const int *start,const int *rows' if split
        else 'float *out,const float *input,const float *cache,const int *start,const int *rows')
  work='out,input,'+('input_rope' if split else '(const float*)0')+',cache,packed'
  work_init=('MLAWork work={'+work+','+str(heads)+','+str(capacity)+',*rows,*start,\n'
             '    '+str(context)+','+str(width)+','+str(width-rope_width)+','+str(rope_width)+','+str(values)+','+str(stride)+','
             +float(scale).hex()+'f,{0,0,0,0}};')
  # ONPU_MLA_ATTN_ZERO_CAP: the pv path stores every (head, row<*rows, c) element, so the
  # full heads*capacity*values zeroing was 5.24 MB of main-thread work per call whose every
  # byte was overwritten (measured 0.143-0.154 ms/call => 8.9-9.5 ms/prefill). Zero only the
  # unused capacity tail [*rows, capacity) instead: consumers read rows < *rows, so this is
  # bit-exact. ONPU_MLA_ATTN_ZERO_KEEP=1 restores the old full-array zero for A/B.
  _zero = ('  {int _r=*rows;if(_r<'+str(capacity)+')for(size_t _h=0;_h<'+str(heads)+'UL;_h++)'
           'for(int _rr=_r;_rr<'+str(capacity)+';_rr++)for(size_t _c=0;_c<'+str(values)+'UL;_c++)'
           'out[(_h*'+str(capacity)+'UL+_rr)*'+str(values)+'UL+_c]=0.0f;}\n'
           if not getenv('ONPU_MLA_ATTN_ZERO_KEEP',0)
           else '  for(unsigned long i=0;i<'+str(heads*capacity*values)+'UL;i++)out[i]=0;\n')
  body=('void '+name+'('+args+') {\n'
        '  if(*start<0||*rows<1||*rows>'+str(capacity)+'||*start+*rows>'+str(context)+')__builtin_trap();\n'
        + (_zero if pv else '')
        + '  '+scratch+'\n'
        '  '+work_init+'\n'
        '  mla_execute_qk_softmax(&work,(mla_parallel_fn)'+(str(pool or 0))+'ULL,'+('1' if pv else '0')+');\n'
        '}\n')
  source=(f'#define MLA_CAUSAL_WORK {int(causal_work)}\n#define MLA_FP16 0\n#define MLA_HEAD_BLOCK {int(head_block)}\n'
          f'#define MLA_ATTN_ROWS {int(attn_rows)}\n#define MLA_FUSED_QK_SOFTMAX 1\n'
          +_vector_exp2_source()+Path(__file__).with_suffix('.c').read_text()+body)
  inputs=(out,inp,rope_inp,cache,start,rows) if split else (out,inp,cache,start,rows)
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo(name))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


_QUERY_HEAD_ENTRY = '''
void rockchip_mla_query_head_entry(float *nope,float *rope,const float *src,const float *table,
                                   const int *start,const int *rows) {
  rockchip_mla_query_head(nope,rope,src,table,start,rows);
}
'''


@functools.cache
def _query_head_program(nope:UOp, rope:UOp, src:UOp, table:UOp, start:UOp, rows:UOp) -> UOp:
  source=Path(__file__).with_name('rockchip_mla_rope.c').read_text()+_QUERY_HEAD_ENTRY
  inputs=(nope,rope,src,table,start,rows)
  renderer=_renderer()
  sink=UOp.sink(*inputs,arg=KernelInfo('rockchip_mla_query_head_entry'))
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,1),ins=(2,3,4,5))
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
    self.qk_softmax=bool(ONPU_MLA_QK_SOFTMAX) and (heads,context,width,values)==(40,512,288,256)
    self.fused_attn=bool(ONPU_MLA_FUSED_ATTN) and (heads,context,width,values)==(40,512,288,256)
    self.fused_query=bool(ONPU_MLA_FUSED_QUERY) and (heads,context,width,values)==(40,512,288,256)
    self._pool=lambda:pool  # Keep worker-library internals out of state-dict walks.

  def _attn_rows(self) -> int:
    # 2 when the opt-in row-paired kernel should be compiled in, else 0 (shipped path).
    return 2 if bool(ONPU_MLA_ATTN_ROWS) and self.heads%4==0 else 0

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
    head_block=int(ONPU_MLA_HEAD_BLOCK.value)
    head_block=4 if head_block else 0  # the C fast path is written for 4 heads per pass
    if dense_rope is None:
      out=Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(dense,cache,position,count,
        fxn=lambda o,a,k,s,n:_program(o,a,a,k,s,n,self.heads,capacity,self.context,self.width,0,self.values,stride,self.scale,
                                      address,qk,self.causal_work,fp16,head_block))[0]
    else:
      out=Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(dense,dense_rope,cache,position,count,
        fxn=lambda o,a,r,k,s,n:_program(o,a,r,k,s,n,self.heads,capacity,self.context,self.width,rope_width,self.values,stride,
                                      self.scale,address,qk,self.causal_work,fp16,head_block))[0]
    logical=out.shrink(((0,1),(0,self.heads),(0,rows),(0,start+rows if qk else self.values)))
    return mark_zero_tail(logical,out,'mla_scores' if qk else 'mla_context') if self.fused_softmax else logical

  def query_key_softmax_value(self, query:Tensor, cache:Tensor, start:int|UOp, *, rope:Tensor|None=None) -> Tensor:
    """Fused prefill qk + exact softmax + weighted value: returns the context the
    three-kernel path would produce, with no score or probability buffer traffic."""
    if not self.fused_attn:raise ValueError('Fused MLA attention is not enabled for this geometry')
    rows,capacity=self._validate(query,cache,start,rope)
    if capacity<2:raise ValueError('Fused MLA attention is a prefill path')
    rope_width=0 if rope is None else rope.shape[-1]
    assert isinstance(rope_width,int)
    stride=self.context+capacity-1
    if ONPU_MLA_ATTN_DIRECT_VALUE and rows==capacity and query.max_shape==query.shape:
      # The producer already wrote this exact head-major layout; the kernel reads it in place.
      # The `AFTER` dependency on the projection's output orders the read correctly.
      dense=query
      dense_rope=None if rope is None else (rope if rope.max_shape==rope.shape else capacity_input(rope).contiguous())
    else:
      dense=capacity_input(query).contiguous()
      dense_rope=None if rope is None else capacity_input(rope).contiguous()
    position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    pool=self._pool()
    address=pool.prefill_address if pool is not None and pool.active and self.heads>=4 else None
    head_block=int(ONPU_MLA_HEAD_BLOCK.value)
    head_block=4 if head_block else 0
    out=Tensor.empty(1,self.heads,capacity,self.values,device='CPU',dtype=dtypes.float)
    if dense_rope is None:
      full=out.custom_kernel(dense,cache,position,count,fxn=lambda o,a,k,s,n:_fused_program(
        o,a,a,k,s,n,self.heads,capacity,self.context,self.width,0,self.values,stride,self.scale,address,
        self.causal_work,head_block,True,self._attn_rows()))[0]
    else:
      full=out.custom_kernel(dense,dense_rope,cache,position,count,fxn=lambda o,a,r,k,s,n:_fused_program(
        o,a,r,k,s,n,self.heads,capacity,self.context,self.width,rope_width,self.values,stride,self.scale,address,
        self.causal_work,head_block,True,self._attn_rows()))[0]
    return mark_zero_tail(full[:,:,:rows,:],full,'mla_fused_attn')

  def softmax(self, scores:Tensor, start:int|UOp) -> Tensor:
    if not self.fused_softmax or self.heads!=40 or self.context!=512 or self.width!=288 or self.values!=256:
      return scores.softmax(-1)
    from tinygrad.llm.kernels.rockchip_mla_softmax import softmax
    return softmax(scores,start,pool=self._pool())

  def query_key(self, query:Tensor, cache:Tensor, start:int|UOp, *, rope:Tensor|None=None) -> Tensor:
    return self._run(query,cache,start,True,rope)
  def split_query(self, q_proj:Tensor, table:Tensor, start:int|UOp) -> tuple[Tensor,Tensor]:
    """Fused q_b-output split: returns (q_nope, q_rope) in the kernel layout."""
    if not self.fused_query:raise ValueError('Fused MLA query split is not enabled for this geometry')
    rows=q_proj.shape[1]
    capacity=128
    _bounds(start,rows,capacity,self.context)
    src=capacity_input(q_proj).reshape(capacity,self.heads*96).contiguous()
    table_view=table.reshape(-1,32).contiguous()
    position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    nope=Tensor.empty(1,self.heads,capacity,64,device='CPU',dtype=dtypes.float)
    rope=Tensor.empty(1,self.heads,capacity,32,device='CPU',dtype=dtypes.float)
    result=nope.custom_kernel(rope,src,table_view,position,count,
      fxn=lambda n,r,s,t,p,c:_query_head_program(n,r,s,t,p,c))
    nope_out,rope_out=result[0],result[1]
    return (mark_zero_tail(nope_out[:,:,:rows,:],nope_out,'mla_q_nope'),
            mark_zero_tail(rope_out[:,:,:rows,:],rope_out,'mla_q_rope'))

  def query_key_softmax(self, query:Tensor, cache:Tensor, start:int|UOp, *, rope:Tensor|None=None) -> Tensor:
    """Fused prefill qk + exact softmax: returns the same probability tensor the
    qk + rockchip_mla_softmax pair would produce, without the padded score buffer."""
    if not self.qk_softmax:raise ValueError('Fused MLA qk+softmax is not enabled for this geometry')
    rows,capacity=self._validate(query,cache,start,rope)
    if capacity<2:raise ValueError('Fused MLA qk+softmax is a prefill path')
    rope_width=0 if rope is None else rope.shape[-1]
    assert isinstance(rope_width,int)
    stride=self.context+capacity-1
    dense=capacity_input(query).contiguous()
    dense_rope=None if rope is None else capacity_input(rope).contiguous()
    position=Tensor(start,device='CPU',dtype=dtypes.int).reshape(1).clone()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    pool=self._pool()
    address=pool.prefill_address if pool is not None and pool.active and self.heads>=4 else None
    head_block=int(ONPU_MLA_HEAD_BLOCK.value)
    head_block=4 if head_block else 0
    out=Tensor.empty(1,self.heads,capacity,stride,device='CPU',dtype=dtypes.float)
    if dense_rope is None:
      full=out.custom_kernel(dense,cache,position,count,fxn=lambda o,a,k,s,n:_fused_program(
        o,a,a,k,s,n,self.heads,capacity,self.context,self.width,0,self.values,stride,self.scale,address,
        self.causal_work,head_block))[0]
    else:
      full=out.custom_kernel(dense,dense_rope,cache,position,count,fxn=lambda o,a,r,k,s,n:_fused_program(
        o,a,r,k,s,n,self.heads,capacity,self.context,self.width,rope_width,self.values,stride,self.scale,address,
        self.causal_work,head_block))[0]
    return mark_zero_tail(full[:,:,:rows,:start+rows],full,'mla_qk_softmax')
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
