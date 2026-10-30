"""Opt-in exact FP32 expert dots with packed Half weights and stable expert grouping."""
from __future__ import annotations
import functools
from dataclasses import replace
from typing import Callable,TYPE_CHECKING
from tinygrad import Tensor,dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo,Ops,ProgramInfo,UOp
if TYPE_CHECKING:
  from tinygrad.llm.model import ExpertWeights
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


def _source(experts:int,columns:int,depth:int,topk:int,broadcast:bool,capacity:int,pool:int|None,prefill_rows:int=4) -> str:
 tile=prefill_rows
 accum='\n'.join(f'M4 a{r}{v}={{0}};' for r in range(tile) for v in range(4))
 inputs='\n'.join(f'const float *x{r}=input+((size_t)indices[first+{r}]{"/"+str(topk) if broadcast else ""})*{depth};' if r==0 else
   f'const float *x{r}=count>{r}?input+((size_t)indices[first+{r}]{"/"+str(topk) if broadcast else ""})*{depth}:(const float*)0;'
   for r in range(tile))
 weights='\n'.join(f'M4 w{v}=__builtin_convertvector(*(const H4*)(base+(size_t)k*16+{v*4}),M4);' for v in range(4))
 updates='\n'.join((f'if(count>{r}){{' if r else '{')+f'float p=x{r}[k];'+
   ''.join(f'a{r}{v}=a{r}{v}+(M4){{p,p,p,p}}*w{v};' for v in range(4))+'}' for r in range(tile))
 full_updates='\n'.join('{'+f'float p=x{r}[k];'+''.join(f'a{r}{v}=a{r}{v}+(M4){{p,p,p,p}}*w{v};' for v in range(4))+'}'
                        for r in range(tile))
 feature_loop=f'for(int k=0;k<{depth};k++){{{weights}{updates}}}'
 if tile!=4:feature_loop=f'if(count=={tile}){{for(int k=0;k<{depth};k++){{{weights}{full_updates}}}}}else{{{feature_loop}}}'
 stores='\n'.join((f'if(count>{r}){{' if r else '{')+
   f'float *dest=out+(size_t)indices[first+{r}]*{columns}+block*16;'+
   ''.join(f'*(M4*)(dest+{v*4})=a{r}{v};' for v in range(4))+'}' for r in range(tile))
 return f'''#include <stddef.h>
 typedef float M4 __attribute__((vector_size(16),aligned(1)));
 typedef __fp16 H4 __attribute__((vector_size(8),aligned(1)));
 typedef int (*parallel_fn)(void(*)(void**,int,int),void**,int);
 typedef struct {{float *out;const float *input;const __fp16 *weight;const int *offsets,*indices;int done[4];}} ExpertWork;
 static void part(ExpertWork *w,int group,int groups) {{
  float *out=w->out;const float *input=w->input;const __fp16 *weight=w->weight;const int *indices=w->indices;
  for(int expert=0;expert<{experts};expert++) {{
   if(w->offsets[expert]==w->offsets[expert+1])continue;
   for(int block=group;block<{columns//16};block+=groups) {{
    const __fp16 *base=weight+((size_t)expert*{columns//16}+block)*{depth*16};
    for(int first=w->offsets[expert];first<w->offsets[expert+1];first+={tile}) {{
     int count=w->offsets[expert+1]-first;if(count>{tile})count={tile};
     {inputs}
     {accum}
     {feature_loop}
     {stores}
    }}
   }}
  }}
 }}
 static void worker(void **args,int unused,int group) {{
  (void)unused;if(group<0||group>=4)return;ExpertWork *w=args[0];part(w,group,4);w->done[group]=1;
 }}
 void rockchip_cpu_expert(float *out,const float *input,const int *selected,const __fp16 *weight,const int *row_count) {{
  const int rows=*row_count;const unsigned long callback={pool or 0}ULL;
  if(rows<1||rows>{capacity})__builtin_trap();
  for(unsigned long j=0;j<{capacity*topk*columns}UL;j++)out[j]=0;
  int counts[{experts}]={{0}},offsets[{experts+1}],cursor[{experts}],indices[{capacity*topk}];
  for(int i=0;i<rows*{topk};i++) {{int e=selected[i];
   if((unsigned)e>={experts}U){{for(int j=0;j<{capacity*topk*columns};j++)out[j]=__builtin_nanf("");return;}}counts[e]++;}}
  offsets[0]=0;for(int e=0;e<{experts};e++){{offsets[e+1]=offsets[e]+counts[e];cursor[e]=offsets[e];}}
  for(int i=0;i<rows*{topk};i++)indices[cursor[selected[i]]++]=i;
  ExpertWork work={{out,input,weight,offsets,indices,{{0,0,0,0}}}};
  if(!callback){{part(&work,0,1);return;}}
  void *args[4]={{&work,0,0,0}};
  if(((parallel_fn)callback)(worker,args,0))__builtin_trap();
  for(int i=0;i<4;i++)if(!work.done[i])__builtin_trap();
 }}'''


@functools.cache
def _program(out:UOp,inp:UOp,selected:UOp,weight:UOp,rows:UOp,experts:int,columns:int,depth:int,topk:int,
             capacity:int,broadcast:bool,pool:int|None,prefill_rows:int=4) -> UOp:
  text=_source(experts,columns,depth,topk,broadcast,capacity,pool,prefill_rows if capacity>1 else 4)
  sink=UOp.sink(out,inp,selected,weight,rows,arg=KernelInfo('rockchip_cpu_expert'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp,selected,weight,rows)),UOp(Ops.SOURCE,arg=text),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(text))))


class CPUExpertWeights:
  def __init__(self,original:ExpertWeights,pool:AttentionPool|None=None,boundary:Callable[[],bool]|None=None,
               prefill_rows:int=4):
    if type(prefill_rows) is not int or prefill_rows not in (4,6):
      raise ValueError('Unsupported CPU expert phase tiles')
    self.prefill_rows=prefill_rows
    self.weight=original.weight
    if hasattr(original,'bias'):self.bias=original.bias
    experts,columns,depth=self.weight.shape
    assert isinstance(experts,int) and isinstance(columns,int) and isinstance(depth,int)
    self.experts,self.columns,self.depth=experts,columns,depth
    self._original=lambda:original
    self._pool=lambda:pool
    self._boundary=boundary or (lambda:False)
    self._packed:Callable[[],Tensor|None]=lambda:None
    self.packed_bytes=0

  def prepare(self,prepared:dict[int,Tensor]):
    key=id(self.weight)
    if key not in prepared:
      packed=self.weight.reshape(self.experts,self.columns//16,16,self.depth).permute(0,1,3,2).contiguous().realize()
      # Keep the original expression/layout for paths that retain compiler
      # fusion. Packed runtime storage stays outside recursive state walks.
      prepared[key]=packed
    packed=prepared[key]
    self._packed=lambda:packed
    self.packed_bytes=packed.nbytes()

  def __call__(self,selected:Tensor,x:Tensor) -> Tensor:
    if (x.device!='CPU' or selected.device!='CPU' or x.dtype is not dtypes.float or selected.dtype is not dtypes.int
        or x.ndim!=4 or selected.ndim!=3 or x.shape[0]!=1 or selected.shape[0]!=1
        or x.shape[1]!=selected.shape[1] or x.shape[-1]!=self.depth
        or not isinstance(topk:=selected.shape[2],int) or not 1<=topk<=8 or x.shape[2] not in (1,topk)
        or not 1<=x.max_shape[1]<=128):return self._original()(selected,x)
    storage=x.uop.base
    if storage.op is Ops.AFTER:storage=storage.src[0].base
    if storage.op not in (Ops.BUFFER,Ops.ALLOC,Ops.PARAM,Ops.STAGE) and not self._boundary():return self._original()(selected,x)
    packed=self._packed()
    if packed is None:raise RuntimeError('Prepare CPU expert weights before capture')
    rows,capacity=x.shape[1],x.max_shape[1]
    assert isinstance(capacity,int)
    if isinstance(rows,UOp) and rows.vmin<1:return self._original()(selected,x)
    broadcast=x.shape[2]==1
    dense=x.pad_to((1,capacity,x.shape[2],self.depth)).contiguous()
    indices=selected.pad_to((1,capacity,topk)).contiguous()
    count=Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
    address=None
    pool=self._pool()
    if pool is not None and pool.active:address=pool.address if capacity==1 else pool.prefill_address
    out=Tensor.empty(1,capacity,topk,self.columns,device='CPU',dtype=dtypes.float).custom_kernel(dense,indices,packed,count,
      fxn=lambda o,a,s,w,n:_program(o,a,s,w,n,self.experts,self.columns,self.depth,topk,capacity,broadcast,address,
                                   self.prefill_rows))[0]
    result=out[:,:rows]
    return result+self.bias[selected] if hasattr(self,'bias') else result


def configure(model,enabled:bool=False,*,pool:AttentionPool|None=None,prefill_rows:int=4) -> int:
  """Prepare immutable expert weights before captures; routing IDs must be in range."""
  from tinygrad.llm.model import ExpertWeights
  from tinygrad.llm.kernels.rockchip_npu_experts import NPUExperts
  if type(enabled) is not bool:raise ValueError('CPU expert selection must be a bool')
  if type(prefill_rows) is not int or prefill_rows not in (4,6):
    raise ValueError('Unsupported CPU expert phase tiles')
  candidates=[]
  for block in getattr(model,'blk',[]):
    for name in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps'):
      layer=getattr(block,name,None)
      if isinstance(layer,CPUExpertWeights):
        layer=layer._original()
        setattr(block,name,layer)
      npu=getattr(block,'npu_experts',None)
      if isinstance(layer,ExpertWeights) and not (isinstance(npu,NPUExperts) and npu.owns(name)):
        candidates.append((block,name,layer))
  if not enabled or not candidates:return 0
  for _,_,layer in candidates:
    w=layer.weight
    if (w.device!='CPU' or w.dtype is not dtypes.half or w.ndim!=3 or not all(isinstance(n,int) for n in w.shape)
        or not 1<=w.shape[0]<=128 or not 16<=w.shape[1]<=4096 or w.shape[1]%16
        or not 4<=w.shape[2]<=4096 or w.shape[2]%4):return 0
  unique={id(layer.weight):layer.weight for _,_,layer in candidates}
  if sum(w.nbytes() for w in unique.values())>5*1024**3:return 0
  prepared:dict[int,Tensor]={}
  wrappers=[]
  for block,name,layer in candidates:
    def boundary(b=block):
      # An active INT8 router consumes the same unscaled FP32 normalization
      # buffer before expert dots. FP16 packing or channel scaling is a
      # different boundary and must retain the generic Tensor expression.
      from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
      router=getattr(b,'ffn_gate_inp',None)
      if not isinstance(router,RockchipInt8Linear) or router.input_scale is not None:return False
      if router.phase in ('prefill','decode'):return getattr(router,router.phase)=='on'
      return router.prefill==router.decode=='on'
    wrapper=CPUExpertWeights(layer,pool,boundary,prefill_rows)
    wrapper.prepare(prepared)
    wrappers.append((block,name,wrapper))
  for block,name,wrapper in wrappers:setattr(block,name,wrapper)
  return len(wrappers)
