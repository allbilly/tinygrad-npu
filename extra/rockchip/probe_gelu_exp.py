"""Check a bounded vision GELU exponential against the existing FP32 arithmetic."""
import argparse, ctypes, json, statistics, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from tinygrad.llm.kernels.rockchip_cpu import _vector_exp2_source


BOUNDED = '''static __attribute__((always_inline)) float4 bounded(float4 x) {
  int4 nan=x!=x;
  float4 safe=nan?(float4)0:x;
  safe=safe<(float4)-24.f?(float4)-24.f:safe;
  safe=safe>(float4)27.f?(float4)27.f:safe;
  int4 q=__builtin_convertvector(safe+(safe<0?(float4)-.5f:(float4).5f),int4);
  float4 s=safe-__builtin_convertvector(q,float4);
  float4 p=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
    0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.f);
  return nan?(float4)__builtin_nanf(""):p*__builtin_bit_cast(float4,(q+127)<<23);
}'''


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--timing',action='store_true')
  args = parser.parse_args()
  functions = []
  for name,exponent in [('generic','rockchip_exp2_vector'),('bounded_gelu','bounded')]:
    functions.append(f'''void {name}(float *out,const float *src,int count){{
      for(int i=0;i<count;i+=4){{
        float4 value=*(const float4*)(src+i),cube=(value*value)*value;
        float4 inner=(value+cube*.044715f)*.7978845608028654f;
        float4 sigmoid=1.f/(1.f+{exponent}((inner*2.f)*-1.4426950216293335f));
        *(float4*)(out+i)=(value*.5f)*(1.f+(2.f*sigmoid-1.f));
      }}
    }}''')
  rng = np.random.default_rng(73)
  half = np.arange(65536,dtype=np.uint16).view(np.float16)
  source = np.concatenate([rng.uniform(-12,12,4*1024*1024).astype(np.float32),half.astype(np.float32),
                           np.linspace(-6,6,1024*1024,dtype=np.float32)])
  with tempfile.TemporaryDirectory(prefix='onpu-gelu-math-') as temporary:
    path = Path(temporary)
    (path/'kernel.c').write_text(_vector_exp2_source()+BOUNDED+'\n'.join(functions))
    subprocess.run(['clang','-O2','-mcpu=native','-shared','-fPIC',str(path/'kernel.c'),'-o',str(path/'kernel.so')],check=True)
    library = ctypes.CDLL(str(path/'kernel.so'))
    outputs = []
    for name in ('generic','bounded_gelu'):
      output = np.empty_like(source)
      getattr(library,name)(ctypes.c_void_p(output.ctypes.data),ctypes.c_void_p(source.ctypes.data),ctypes.c_int(len(source)))
      outputs.append(output)
    finite = np.isfinite(outputs[0])
    np.testing.assert_array_equal(np.isnan(outputs[0]),np.isnan(outputs[1]))
    np.testing.assert_array_equal(outputs[0][finite].view(np.uint32),outputs[1][finite].view(np.uint32))
    print(json.dumps({'passed':True,'values':len(source),'finite_bitwise_identical':True}),flush=True)
    if args.timing:
      data = rng.normal(0,3,1024*3072).astype(np.float32)
      output = np.empty_like(data)
      for name in ('generic','bounded_gelu'):
        samples = []
        for i in range(12):
          start = time.perf_counter()
          getattr(library,name)(ctypes.c_void_p(output.ctypes.data),ctypes.c_void_p(data.ctypes.data),ctypes.c_int(len(data)))
          if i>=4:samples.append((time.perf_counter()-start)*1000)
        print(json.dumps({'method':name,'median_ms':statistics.median(samples),'all_ms':samples}),flush=True)


if __name__=='__main__': main()
