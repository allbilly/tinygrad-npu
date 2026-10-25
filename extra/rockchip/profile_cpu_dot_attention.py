"""Isolated INT8 QK experiment; serving remains on qualified FP32 attention.

ARM SDOT semantics: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
Quantize each query/key row independently, retaining FP32 softmax and PV.
The fixed primitive gate is normalized RMSE <=0.02, not bitwise equivalence.
Full-model quality and cached-output gates are still required before serving.
"""
import argparse, functools, hashlib, json, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_int8 import _quantize_vector_source
from tinygrad.llm.kernels.rockchip_packed_attention import PackedCPUAttention
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
from extra.rockchip.profile_cpu_grouped_attention import source


def dot_source(block:int, context:int=8192, capacity:int=128) -> str:
  queries,width = 3*block,(3*block+3)//4*4
  code = source(block,context,capacity)
  position = code.index('  typedef unsigned long thread_t;')
  helpers = _quantize_vector_source()+'''
  typedef signed char char16 __attribute__((aligned(16),ext_vector_type(16)));
  typedef int dot4 __attribute__((aligned(16),ext_vector_type(4)));
  static inline dot4 dot(dot4 a,char16 b,char16 c){
    __asm__("sdot %0.4s, %1.16b, %2.16b":"+w"(a):"w"(b),"w"(c));return a;}
  '''
  code = code[:position]+helpers+code[position:]
  first,last = code.index('      float4 q['),code.index('      float p[')
  prepare = f'''char16 q[{queries}][4];float qs[{queries}];int used[{queries}],tail[{queries}];
    for(int i=0;i<{queries};i++){{int r=i%{block};used[i]=total-rows+row+(r<count?r:count-1)+1;tail[i]=used[i]/4*4;
      const float4 *input=(const float4*)(query+((kv*3+i/{block})*{capacity}+row+r)*64);float4 maximum={{0}};
      for(int c=0;c<16;c++){{float4 x=r<count?input[c]:(float4)0;x=x>=(float4)0?x:-x;maximum=x>maximum?x:maximum;}}
      float largest=maximum[0];for(int c=1;c<4;c++)if(maximum[c]>largest)largest=maximum[c];
      float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;qs[i]=scale;
      for(int c=0;c<16;c++)((char4*)q[i])[c]=rockchip_quantize_vector(r<count?input[c]:(float4)0,1.0f/scale);
    }}
  '''
  code = code[:first]+prepare+code[last:]
  first = code.index('      for(int t=0;t<used[')
  last = code.index(f'for(int i={queries};i<{width};i++)p[t*{width}+i]=0;',first)
  accum = ','.join(f'a{i}={{0}}' for i in range(queries))
  dots = ''.join(f'a{i}=dot(a{i},q[{i}][c],key);' for i in range(queries))
  scores = ''.join(f'''p[t*{width}+{i}]=t<used[{i}]?
    ((float)(((a{i}[0]+a{i}[1])+a{i}[2])+a{i}[3])*qs[{i}])*scale*0.125f:-__builtin_inff();''' for i in range(queries))
  body = f'''for(int t=0;t<used[{queries-1}];t++){{dot4 {accum};
    const signed char *keys=(const signed char*)cache+(kv*{context}+t)*64;
    float scale=((const float*)((const signed char*)cache+{3*context*64}))[kv*{context}+t];
    for(int c=0;c<4;c++){{char16 key=((const char16*)keys)[c];{dots}}}
    {scores}'''
  return code[:first]+body+code[last:]


@functools.cache
def program(out,q,cache,values,end,rows,block):
  inputs = (out,q,cache,values,end,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('grouped_attention_probe'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4,5))
  code = dot_source(block)
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=code),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(code))))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(111)
  raw_query = rng.normal(0,.3,(1,9,128,64)).astype(np.float32)
  raw_cache = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
  raw_cache[:,:,:,8191:] = np.nan
  keys = np.nan_to_num(raw_cache[0,0].astype(np.float32))
  scales = np.maximum(np.max(abs(keys),axis=-1)/np.float32(127),np.float32(1e-8))
  quantized = np.clip(np.rint(keys/scales[:,:,None]),-127,127).astype(np.int8)
  packed = np.concatenate((quantized.ravel(),scales.ravel().view(np.int8)))
  values = np.pad(raw_cache[1,0].astype(np.float32).reshape(3,8192,16,4).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  q,cache,k,v = (Tensor(a,device='CPU').realize() for a in (raw_query,raw_cache,packed,values))
  end = Tensor([8191],device='CPU',dtype=dtypes.int).realize()
  state = PackedCPUAttention(8192)
  accepted = TinyJit(lambda q,k:state(q,k,8191).realize())
  variants = {}
  for block in (1,2,4,8):
    def execute(q,k,v,e,r,block=block):
      return Tensor.empty(*raw_query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,k,v,e,r,
        fxn=lambda o,q,k,v,e,r:program(o,q,k,v,e,r,block))[0].realize()
    variants[f'int8_qk_group{block}'] = TinyJit(execute)
  report = {'complete':False,'scope':'synthetic primitive; INT8 key/value preparation excluded; not serving qualification',
            'max_normalized_rmse':.02,'checks':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    for count in (128,65,7,1,127):
      expected = state(q[:,:,:count],cache,8191).numpy()
      rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
      for label,run in variants.items():
        actual = run(q,k,v,end,rows).numpy()[:,:,:count]
        assert np.isfinite(actual).all()
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        report['checks'].append({'variant':label,'rows':count,'normalized_rmse':error})
        assert error<=report['max_normalized_rmse'],report['checks'][-1]
    rows = Tensor([128],device='CPU',dtype=dtypes.int).realize()
    samples = {name:[] for name in ['accepted',*variants]}
    for index in range(9):
      for label in samples if index%2==0 else reversed(samples):
        begin = time.perf_counter()
        accepted(q,cache) if label=='accepted' else variants[label](q,k,v,end,rows)
        if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
    report.update(complete=True,variants={name:{'median_ms':statistics.median(v),'samples_ms':v} for name,v in samples.items()})
    print(json.dumps(report),flush=True)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:save()


if __name__=='__main__': main()
