"""Check mutable-output JIT embedding with fresh owned CPU buffers."""
import array
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.vision_embed import _embed, embed_prompt
from tinygrad.uop.ops import UOp


def owned(shape,dtype,values=None):
  u=UOp.new_buffer('CPU',int(np.prod(shape)),dtype)
  u.buffer.ensure_allocated()
  if values is not None:u.buffer.allocator._copyin(u.buffer._buf,memoryview(array.array('i',values)).cast('B'))
  return Tensor(u.reshape(shape))


def main():
  words=Tensor.randn(256,32,device='CPU').half().realize()
  image=Tensor.randn(1,4,32,device='CPU').realize()
  rows,start,count,dim=13,2,4,32
  def run(out,ids,word,image):
    out.custom_kernel(ids,word,image,fxn=lambda o,i,w,p:_embed(o,i,w,p,rows,dim,start,count))[0].realize()
  jit=TinyJit(run)
  previous=[]
  for index in range(8):
    tokens=[(i+index*7)%256 for i in range(rows)]
    ids=owned((rows,),dtypes.int,tokens)
    out=owned((1,rows,dim),dtypes.float)
    jit(out,ids,words,image)
    expected=embed_prompt(tokens,words,image,start,count).numpy()
    np.testing.assert_array_equal(out.numpy(),expected)
    previous.append((out,expected.copy()))
  for out,expected in previous:np.testing.assert_array_equal(out.numpy(),expected)
  print('changed IDs and fresh output ownership pass')


if __name__=='__main__':main()
