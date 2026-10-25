"""Test bounded CPU exponential approximations for FP16 vision attention."""
import argparse, ctypes, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from tinygrad.llm.kernels.rockchip_cpu import _vector_exp2_source

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--math-only',action='store_true')
  args=parser.parse_args()
  table=','.join(f'{np.exp(-i/16):.9e}f' for i in range(321))
  polynomial='((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f)'
  source=_vector_exp2_source()+f'''
  static const float lut[321]={{{table}}};
  void generic(float *out,const float *in,int count){{
    for(int i=0;i<count;i+=4){{float4 x=*(const float4*)(in+i)*1.4426950216293335f;
      *(float4*)(out+i)=rockchip_exp2_vector(x);}}
  }}
  void bounded(float *out,const float *in,int count){{
    for(int i=0;i<count;i+=4){{float4 x=*(const float4*)(in+i)*1.4426950216293335f;
      int4 q=__builtin_convertvector(x-.5f,int4);float4 s=x-__builtin_convertvector(q,float4);
      *(float4*)(out+i)={polynomial}*__builtin_bit_cast(float4,(q+127)<<23);}}
  }}
  void lookup(float *out,const float *in,int count){{
    for(int i=0;i<count;i+=4){{float4 x=-*(const float4*)(in+i);int4 q=__builtin_convertvector(x*16,int4);
      float4 r=__builtin_convertvector(q,float4)*.0625f-x;
      float4 base=(float4){{lut[q[0]],lut[q[1]],lut[q[2]],lut[q[3]]}};
      *(float4*)(out+i)=base*(((r*(1.0f/6.0f)+.5f)*r+1.0f)*r+1.0f);}}
  }}'''
  values=np.linspace(-20,0,1048576,dtype=np.float32)
  output=np.empty_like(values)
  reference=np.exp(values.astype(np.float64))
  with tempfile.TemporaryDirectory(prefix='onpu-softmax-') as folder:
    code=Path(folder)/'exp.c';binary=Path(folder)/'exp.so';code.write_text(source)
    subprocess.run(['clang','-O2','-fPIC','-shared','-mcpu=native',str(code),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary))
    generic_values=None
    for name in ('generic','bounded','lookup'):
      fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
      fn(output.ctypes.data,values.ctypes.data,len(values))
      if name=='generic': generic_values=output.copy()
      if name=='bounded': np.testing.assert_array_equal(output,generic_values)
      error=float(np.max(abs(output-reference)/reference))
      np.testing.assert_allclose(output,reference,rtol=1e-6,atol=1e-10)
      samples=[]
      if not args.math_only:
        for _ in range(20):
          start=time.perf_counter();fn(output.ctypes.data,values.ctypes.data,len(values));samples.append((time.perf_counter()-start)*1000)
      print(json.dumps({'mode':name,'max_relative_error':error,'median_ms':float(np.median(samples)) if samples else None}),flush=True)

if __name__=='__main__':main()
