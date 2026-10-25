"""First-maximum CPU reduction for finite Rockchip output scores."""
import functools
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _argmax(out:UOp, values:UOp, rows:int, columns:int) -> UOp:
  source = f'''typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
  void gpt2_argmax(int *out,const float *input) {{
    for(int row=0;row<{rows};row++) {{
      const float *data=input+row*{columns};float4 best[4];int4 index[4];
      for(int b=0;b<4;b++){{best[b]=(float4)(-__builtin_inff());index[b]=(int4){{b*4,b*4+1,b*4+2,b*4+3}};}}
      int i=0;
      for(;i+15<{columns};i+=16) {{
        #pragma clang loop unroll(full)
        for(int b=0;b<4;b++){{float4 value=*(const float4*)(data+i+b*4);int4 found=value>best[b];
          best[b]=found?value:best[b];index[b]=found?((int4){{i+b*4,i+b*4+1,i+b*4+2,i+b*4+3}}):index[b];}}
      }}
      float maximum=-__builtin_inff();int selected=0;
      for(int b=0;b<4;b++)for(int lane=0;lane<4;lane++){{
        if(best[b][lane]>maximum || (best[b][lane]==maximum && index[b][lane]<selected)){{
          maximum=best[b][lane];selected=index[b][lane];}}
      }}
      for(;i<{columns};i++)if(data[i]>maximum){{maximum=data[i];selected=i;}}
      out[row]=selected;
    }}
  }}'''
  sink = UOp.sink(out,values,arg=KernelInfo('gpt2_argmax'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,values)),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def greedy_argmax(logits:Tensor) -> Tensor:
  """First-maximum SIMD reduction for the finite FP32 output of NPU projections."""
  if logits.device!='CPU' or logits.dtype is not dtypes.float or logits.ndim!=2: return logits.argmax(-1)
  rows,columns = logits.shape
  if not isinstance(rows,int) or not isinstance(columns,int) or not 1<=rows<=64 or not 1<=columns<=1048576: return logits.argmax(-1)
  return Tensor.empty(rows,device='CPU',dtype=dtypes.int).custom_kernel(logits.contiguous(),
    fxn=lambda out,x:_argmax(out,x,rows,columns))[0]
