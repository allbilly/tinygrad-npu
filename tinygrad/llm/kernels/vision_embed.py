"""CPU text embedding lookup and contiguous image-token substitution."""
import array, ctypes, functools, math
from dataclasses import replace
from typing import Any, cast
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer, TinyELF
from tinygrad.dtype import DType
from tinygrad.helpers import TRAINING
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

@functools.cache
def _embed(out:UOp, ids:UOp, words:UOp, image:UOp, tokens:int, dim:int, start:int, count:int) -> UOp:
  half=words.dtype is dtypes.half
  typ='__fp16' if half else 'float'
  load='__builtin_convertvector(*(const half4*)(source+c),float4)' if half else '*(const float4*)(source+c)'
  source=f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  void vision_embed_prompt(float *out,const int *ids,const {typ} *words,const float *image){{
    for(int row=0;row<{tokens};row++){{
      if(row>={start} && row<{start+count}){{
        __builtin_memcpy(out+row*{dim},image+(row-{start})*{dim},{dim*4}UL);
      }}else{{const {typ} *source=words+((unsigned long)ids[row])*{dim};
        for(int c=0;c<{dim};c+=4)*(float4*)(out+row*{dim}+c)={load};
      }}
    }}
  }}'''
  inputs=(out,ids,words,image)
  sink=UOp.sink(*inputs,arg=KernelInfo('vision_embed_prompt'))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
    UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))

@functools.cache
def _embedding_program(half:bool):
  typ='__fp16' if half else 'float'
  load='__builtin_convertvector(*(const half4*)(source+c),float4)' if half else '*(const float4*)(source+c)'
  source=f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  void prompt_embedding(float *out,const int *ids,const {typ} *words,const float *image,int rows,int dim,int start,int count){{
    for(int row=0;row<rows;row++){{
      if(row>=start && row<start+count)__builtin_memcpy(out+row*dim,image+(row-start)*dim,(unsigned long)dim*4);
      else{{const {typ} *source=words+(unsigned long)ids[row]*dim;
        for(int c=0;c<dim;c+=4)*(float4*)(out+row*dim+c)={load};}}
    }}
  }}'''
  renderer=_renderer()
  program=cast(Any,Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'prompt_embedding',renderer.target,())))
  program.fxn.restype=None
  program.fxn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4
  return program


class PromptEmbedding:
  # Eager inference helper; returned CPU tensors own fresh allocations.
  __slots__ = ()

  @staticmethod
  def owned(shape:tuple[int,...], dtype:DType, values:list[int]|None=None) -> Tensor:
    uop = UOp.new_buffer('CPU',math.prod(shape),dtype)
    buf = cast(Buffer,uop.buffer).ensure_allocated()
    if values is not None:buf.allocator._copyin(buf._buf,memoryview(array.array('i',values)).cast('B'))
    return Tensor(uop.reshape(shape))

  def __call__(self, tokens:list[int], words:Tensor, image:Tensor, start:int, count:int) -> Tensor:
    from tinygrad.engine.realize import capturing
    if capturing or TRAINING:return embed_prompt(tokens,words,image,start,count)
    rows,dim = len(tokens),words.shape[1]
    assert isinstance(dim,int)
    ids = self.owned((rows,),dtypes.int,tokens)
    out = self.owned((1,rows,dim),dtypes.float)
    buffers=[tensor._buffer() for tensor in (out,ids,words,image)]
    _embedding_program(words.dtype is dtypes.half).fxn(*(ctypes.c_void_p(buffer._buf) for buffer in buffers),rows,dim,start,count)
    return out


def stage_embeddings(values:Tensor, first:int, count:int, capacity:int) -> Tensor:
  """Copy one CPU prefill chunk into fresh zero-padded storage without graph setup."""
  if (values.device!='CPU' or values.dtype is not dtypes.float or values.ndim!=3 or values.shape[0]!=1
      or not isinstance(values.shape[1],int) or not isinstance(values.shape[2],int)
      or first<0 or not 1<=count<=capacity or first+count>values.shape[1]):
    raise ValueError('Invalid CPU embedding chunk')
  dim=values.shape[2]
  out=PromptEmbedding.owned((1,capacity,dim),dtypes.float)
  source,target=values._buffer(),out._buffer()
  ctypes.memmove(target._buf,source._buf+first*dim*4,count*dim*4)
  if count<capacity:ctypes.memset(target._buf+count*dim*4,0,(capacity-count)*dim*4)
  return out


def embed_prompt(tokens:list[int], words:Tensor, image:Tensor, start:int, count:int, runner:PromptEmbedding|None=None) -> Tensor:
  if (words.device!='CPU' or words.ndim!=2 or words.dtype not in (dtypes.half,dtypes.float) or words.shape[1]%4
      or image.device!='CPU' or image.dtype is not dtypes.float or image.shape!=(1,count,words.shape[1])
      or start<0 or count<1 or start+count>len(tokens) or any(not isinstance(t,int) or not 0<=t<words.shape[0] for t in tokens)):
    raise ValueError('Invalid CPU vision embedding input')
  if runner is not None:return runner(tokens,words,image,start,count)
  dim=words.shape[1]
  assert isinstance(dim,int)
  return Tensor.empty(1,len(tokens),dim,device='CPU',dtype=dtypes.float).custom_kernel(
    Tensor(tokens,device='CPU',dtype=dtypes.int).contiguous(),words,image.contiguous(),
    fxn=lambda out,ids,w,p:_embed(out,ids,w,p,len(tokens),dim,start,count))[0]
