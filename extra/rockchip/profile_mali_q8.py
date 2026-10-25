"""Diagnostic Q8_0 GEMV: retain exact FP16 operands while reading packed weights.

This is an isolated kernel experiment, not an end-to-end RKLLM speed claim.
"""
import argparse, functools, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import ceildiv, prod
from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.llm.kernels.mali_linear import gemv


@functools.cache
def kernel(out,x,raw,columns,depth):
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_q8_gemv(__global float *out,__global const float *x,__global const uchar *raw) {{
    int lane=get_local_id(0)%16,row=get_group_id(0)*16+get_local_id(0)/16;
    float4 sum=(float4)0;
    if(row<{columns})for(int k=lane*4;k<{depth};k+=64) {{
      ulong block=(((ulong)row)*{depth//32}+k/32)*34;
      float scale=(float)(*(__global const half *)(raw+block));
      char4 q=vload4(0,(__global const char *)(raw+block+2+k%32));
      float4 w=convert_float4(convert_half4_rte(convert_float4(q)*(float4)scale));
      sum=fma(vload4(0,x+k),w,sum);
    }}
    __local float partial[256];partial[get_local_id(0)]=(sum.x+sum.y)+(sum.z+sum.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && row<{columns}) {{float result=0;
      for(int k=0;k<16;k++)result+=partial[get_local_id(0)+k];out[row]=result;}}
  }}'''
  return _program('mali_q8_gemv',source,(out,x,raw),(ceildiv(columns,16),1,1),(256,1,1))


def q8_gemv(x,raw,columns,depth):
  out = Tensor.empty(1,1,columns,device='CL',dtype=dtypes.float)
  return out.custom_kernel(x.contiguous(),raw,fxn=lambda o,a,w:kernel(o,a,w,columns,depth))[0]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--iterations',type=int,default=15)
  args = parser.parse_args()
  _,entries = gguf_parse(args.checkpoint)
  names = ['blk.0.attn_q.weight','blk.0.ffn_down.weight','output.weight']
  if 'output.weight' not in entries:names[-1] = 'token_embd.weight'
  report = {'complete':False,'scope':'Primitive diagnostic; may overlap CPU reference conversion','cases':[]}
  rng = np.random.default_rng(19)
  for name in names:
    data,shape,typ = entries[name]
    if typ!=8:raise ValueError(f'{name} is not Q8_0')
    columns,depth = shape
    assert depth%32==0
    raw = data.to('CL').contiguous().realize()
    dense = ggml_data_to_tensor(raw,prod(shape),8).reshape(columns,depth).half().realize()
    old = TinyJit(lambda x:gemv(x,dense).realize())
    new = TinyJit(lambda x:q8_gemv(x,raw,columns,depth).realize())
    x = Tensor(rng.normal(0,.3,(1,1,depth)).astype(np.float32),device='CL').realize()
    for _ in range(5):old(x);new(x)
    a,b = old(x).numpy(),new(x).numpy()
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    timings = {'dense':[],'q8':[]}
    for index in range(args.iterations):
      for label,call in ((('dense',old),('q8',new)) if index%2==0 else (('q8',new),('dense',old))):
        Device['CL'].synchronize();start=time.perf_counter();call(x);Device['CL'].synchronize()
        timings[label].append((time.perf_counter()-start)*1000)
    row = {'name':name,'columns':columns,'depth':depth,'bitwise_equal':True,'times_ms':timings,
           'median_ms':{k:statistics.median(v) for k,v in timings.items()}}
    report['cases'].append(row);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
  report['complete']=True;args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
