"""Direct packed Q8_0 CPU embedding rows, without a vocabulary-sized temporary."""
import functools,math,weakref
from pathlib import Path
from typing import Any
from dataclasses import replace
from tinygrad import Tensor,dtypes
from tinygrad.helpers import TRAINING
from tinygrad.llm.gguf import gguf_parse,_mapped_cpu_bytes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_padding import capacity_input,mark_zero_tail
from tinygrad.uop.ops import UOp,Ops,KernelInfo,ProgramInfo
@functools.cache
def program(out:UOp,table:UOp,ids:UOp,count:UOp,capacity:int,vocab:int,width:int) -> UOp:
 source=f"""typedef float F4 __attribute__((ext_vector_type(4),aligned(1)));
 typedef signed char C4 __attribute__((ext_vector_type(4),aligned(1)));
 typedef __fp16 H4 __attribute__((ext_vector_type(4),aligned(1)));
 void cpu_packed_q8_embedding(__fp16 *out,const unsigned char *table,const int *ids,const int *count) {{
  int active=*count;
  if(active<1 || active>{capacity}){{for(int j=0;j<{capacity*width};j++)out[j]=0;return;}}
  for(int row=0;row<{capacity};row++){{
   int token=row<active?ids[row]:-1;
   if(token<0 || token>={vocab}){{for(int j=0;j<{width};j++)out[row*{width}+j]=0;continue;}}
   const unsigned char *p=table+(unsigned long)token*{width//32*34};
   for(int block=0;block<{width//32};block++){{
    float scale=(float)*(const __fp16*)(p+block*34);
    for(int j=0;j<32;j+=4){{
     F4 value=__builtin_convertvector(*(const C4*)(p+block*34+2+j),F4)*(F4){{scale,scale,scale,scale}};
     *(H4*)(out+row*{width}+block*32+j)=__builtin_convertvector(value,H4);
    }}
   }}
  }}
 }}"""
 renderer=_renderer()
 sink=UOp.sink(out,table,ids,count,arg=KernelInfo('cpu_packed_q8_embedding'))
 info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
 return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,table,ids,count)),UOp(Ops.SOURCE,arg=source),
                                   UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))
class RockchipQ8Embedding:
  """Inference-only Q8_0 lookup retaining the original logical FP16 weight.

  Changed or realized weights, gradients and unqualified index layouts use the
  original embedding. The checkpoint bytes must remain immutable, as with the
  other prepared Rockchip weights.
  """
  def __init__(self, source:Any, raw:Tensor, vocab:int, width:int, checkpoint:str=''):
    if (type(vocab) is not int or type(width) is not int or not 1<=vocab<2**31 or not 32<=width<=32768 or width%32
        or source.weight.device!='CPU' or source.weight.dtype is not dtypes.half or source.weight.shape!=(vocab,width)
        or raw.device!='CPU' or raw.dtype is not dtypes.uint8 or raw.shape!=(vocab*(width//32)*34,)):
      raise ValueError('Unsupported packed CPU embedding geometry')
    self.source,self.weight,self.raw=source,source.weight,raw
    self.vocab,self.width,self.checkpoint=vocab,width,checkpoint
    self._origin=weakref.ref(self.weight.uop)
    self.fast_traces,self.fallback_traces=0,0

  def __call__(self, indices:Tensor) -> Tensor:
    capacity=math.prod(indices.max_shape)
    if (indices.device!='CPU' or indices.dtype is not dtypes.int32 or indices.ndim not in (1,2)
        or indices.ndim==2 and indices.max_shape[0]!=1 or not 1<=capacity<=512
        or self.source.weight is not self.weight or self.weight.uop is not self._origin()
        or self.weight.uop.is_realized or TRAINING):
      self.fallback_traces+=1
      return self.source(indices)
    self.fast_traces+=1
    active=math.prod(indices.shape)
    count=Tensor(active,device='CPU',dtype=dtypes.int32).reshape(1).clone()
    result=Tensor.empty(capacity,self.width,device='CPU',dtype=dtypes.half).custom_kernel(
      self.raw,capacity_input(indices).contiguous(),count,
      fxn=lambda o,t,i,c:program(o,t,i,c,capacity,self.vocab,self.width))[0]
    full=result.reshape(*indices.max_shape,self.width)
    logical=result[:active].reshape(*indices.shape,self.width)
    return mark_zero_tail(logical,full,'embedding')

  def metadata(self) -> dict:
    return {'enabled':True,'format':'Q8_0','vocabulary':self.vocab,'width':self.width,'raw_bytes':self.raw.nbytes(),
            'fast_traces':self.fast_traces,'fallback_traces':self.fallback_traces,'checkpoint':self.checkpoint}


def configure(model:Any, checkpoint:Path|str, enabled:bool=True) -> dict:
  from tinygrad import nn
  if type(enabled) is not bool:raise ValueError('CPU packed embedding selection must be a bool')
  source=getattr(model,'token_embd',None)
  if not enabled:
    if isinstance(source,RockchipQ8Embedding):model.token_embd=source.source
    return {}
  path=str(Path(checkpoint).resolve())
  if isinstance(source,RockchipQ8Embedding):
    if source.checkpoint!=path:raise ValueError('Embedding checkpoint changed after preparation')
    return source.metadata()
  if not isinstance(source,nn.Embedding) or source.weight.device!='CPU' or source.weight.dtype is not dtypes.half:return {}
  _,entries=gguf_parse(path)
  if 'token_embd.weight' not in entries:return {}
  data,shape,kind=entries['token_embd.weight']
  if kind!=8 or shape!=source.weight.shape or len(shape)!=2:return {}
  model.token_embd=RockchipQ8Embedding(source,_mapped_cpu_bytes(data,{}),shape[0],shape[1],checkpoint=path)
  return model.token_embd.metadata()


class RockchipSegmentedQ8Embedding:
  """Inference-only whole Q8 lookup retaining the original segmented fallback.

  The checked Gemma4 table exceeds signed 32-bit indexing. The existing CPU
  decoder uses unsigned-long row offsets and keeps the mapped file alive.
  Changed weights/geometry, training and unsupported indices use the source.
  """
  def __init__(self, source:Any, raw:Tensor, checkpoint:str=''):
    from tinygrad.llm.gemma4 import SegmentedEmbedding
    if (type(source) is not SegmentedEmbedding or (source.vocab,source.width,source.rows)!=(262144,8960,16384)
        or len(source.chunks)!=16 or raw.device!='CPU' or raw.dtype is not dtypes.uint8 or raw.shape!=(2495610880,)
        or any(c.weight.device!='CPU' or c.weight.dtype is not dtypes.half or c.weight.shape!=(16384,8960)
               for c in source.chunks)):
      raise ValueError('Unsupported packed segmented CPU embedding geometry')
    self.source,self.raw,self.checkpoint=source,raw,checkpoint
    self.weights=tuple(c.weight for c in source.chunks)
    self._origins=tuple(weakref.ref(w.uop) for w in self.weights)
    self.fast_traces,self.fallback_traces=0,0

  def __call__(self, indices:Tensor) -> Tensor:
    capacity=math.prod(indices.max_shape)
    if (indices.device!='CPU' or indices.dtype is not dtypes.int32 or indices.ndim not in (1,2)
        or indices.ndim==2 and indices.max_shape[0]!=1 or not 1<=capacity<=512 or TRAINING
        or (self.source.vocab,self.source.width,self.source.rows)!=(262144,8960,16384)
        or len(self.source.chunks)!=16
        or any(c.weight is not w or w.uop is not origin() or w.uop.is_realized
               for c,w,origin in zip(self.source.chunks,self.weights,self._origins))):
      self.fallback_traces+=1
      return self.source(indices)
    self.fast_traces+=1
    active=math.prod(indices.shape)
    count=Tensor(active,device='CPU',dtype=dtypes.int32).reshape(1).clone()
    result=Tensor.empty(capacity,8960,device='CPU',dtype=dtypes.half).custom_kernel(
      self.raw,capacity_input(indices).contiguous(),count,
      fxn=lambda o,t,i,c:program(o,t,i,c,capacity,262144,8960))[0]
    full=result.reshape(*indices.max_shape,8960)
    logical=result[:active].reshape(*indices.shape,8960)
    values=mark_zero_tail(logical,full,'embedding').float()
    # The source's masked chunk sum turns every signed zero into positive zero.
    return values.eq(0).where(0.0,values)

  def metadata(self) -> dict:
    return {'enabled':True,'format':'Q8_0','vocabulary':262144,'width':8960,'chunks':16,
            'raw_bytes':self.raw.nbytes(),'fast_traces':self.fast_traces,'fallback_traces':self.fallback_traces,
            'checkpoint':self.checkpoint}


def configure_segmented(model:Any, checkpoint:Path|str, enabled:bool=True) -> dict:
  """Opt into the checked per-layer table without changing other embeddings."""
  from tinygrad.llm.gemma4 import SegmentedEmbedding
  if type(enabled) is not bool:raise ValueError('CPU packed side embedding selection must be a bool')
  source=getattr(model,'per_layer_token_embd',None)
  if not enabled:
    if isinstance(source,RockchipSegmentedQ8Embedding):model.per_layer_token_embd=source.source
    return {}
  path=str(Path(checkpoint).resolve())
  if isinstance(source,RockchipSegmentedQ8Embedding):
    if source.checkpoint!=path:raise ValueError('Side embedding checkpoint changed after preparation')
    return source.metadata()
  if type(source) is not SegmentedEmbedding:return {}
  _,entries=gguf_parse(path)
  if 'per_layer_token_embd.weight' not in entries:return {}
  data,shape,kind=entries['per_layer_token_embd.weight']
  if kind!=8 or shape!=(262144,8960):return {}
  model.per_layer_token_embd=RockchipSegmentedQ8Embedding(source,_mapped_cpu_bytes(data,{}),checkpoint=path)
  return model.per_layer_token_embd.metadata()
