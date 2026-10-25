"""Isolated Mali INT8 score-dot experiment; serving kernels remain unchanged.

arm_dot_acc(char4,char4,int) is declared by the installed OpenCL C header.
Quantization and copies are excluded from primitive timings. Full-model and
cached-output qualification are required before considering serving changes.
"""
import argparse, functools, hashlib, json, math, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _partial, _merge, _program
from tinygrad.uop.ops import Ops, UOp


@functools.cache
def dot_partial(out,q,cache,end,count,qs,keys,ks,heads,kv_heads,dim,context,capacity,query_rows,subgroups=False,onfly=False):
  base = _partial(out,q,cache,end,count,heads,kv_heads,dim,context,capacity,512,query_rows,subgroups)
  source = next(u.arg for u in base.src if u.op is Ops.SOURCE)
  group,gqa = heads//kv_heads*query_rows,heads//kv_heads
  pragmas = ('#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable\n'
             '#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable\n')
  language = '// tinygrad-cl-std=CL2.0\n'
  source = language+pragmas+source[len(language):] if source.startswith(language) else pragmas+source
  source = source.replace('__global const float *query','__global const char *query')
  source = source.replace('__local float q[G*D]','__local char q[G*D]')
  signature = '__global const int *rows) {'
  assert source.count(signature)==1
  source = source.replace(signature,'__global const int *rows,__global const float *qs,__global const char *keys,__global const float *ks) {')
  if subgroups:
    assert dim==128
    source = source.replace('float8 q','char8 q').replace('(float8)0','(char8)0')
    first = source.index('for(int key=subgroup;key<N;key+=get_num_sub_groups())')
    last = source.index('\n    barrier(CLK_LOCAL_MEM_FENCE);',first)
    init = ''.join(f'int dot{h}=0;' for h in range(group))
    dots = ''.join(f'dot{h}=arm_dot_acc(q{h}.lo,k.lo,dot{h});dot{h}=arm_dot_acc(q{h}.hi,k.hi,dot{h});' for h in range(group))
    stores = ''.join(f'''{{int result=sub_group_reduce_add(dot{h});if(sublane==0)prob[{h}*N+key]=
      row+{h//gqa}<count&&first+key<min(total,total-count+row+{h//gqa}+1)?
      ((float)result*qs[(kv*{gqa}+{h%gqa})*C+row+{h//gqa}])*ks[kv*{context}+first+key]*{1/math.sqrt(dim):.17g}f:-INFINITY;}}'''
      for h in range(group))
    load = (f'''float8 raw=first+key<last?convert_float8(vload8(0,cache+(kv*{context}+first+key)*D+sublane*8)):(float8)0;
      float inverse=first+key<last?1.0f/ks[kv*{context}+first+key]:1;
      char8 k=convert_char8_sat_rte(clamp(raw*(float8)inverse,(float8)-127,(float8)127));''' if onfly else
      f'char8 k=first+key<last?vload8(0,keys+(kv*{context}+first+key)*D+sublane*8):(char8)0;')
    source = source[:first]+f'for(int key=subgroup;key<N;key+=get_num_sub_groups()){{{load}{init}{dots}{stores}}}'+source[last:]
    args = (out,q,cache,end,count,qs,keys,ks)
    return _program('mali_attention_partial',source,args,base.arg.global_size,base.arg.local_size)
  for h in range(group):
    source = source.replace(f'float4 dot{h}=(float4)0;',f'int dot{h}=0;')
    source = source.replace(f'dot{h}=fma(vload4(0,q+{h}*D+c),k,dot{h});',f'dot{h}=arm_dot_acc(vload4(0,q+{h}*D+c),k,dot{h});')
    old = f'((dot{h}.x+dot{h}.y)+(dot{h}.z+dot{h}.w))*{1/math.sqrt(dim):.17g}f'
    new = f'((float)dot{h}*qs[(kv*{gqa}+{h%gqa})*C+row+{h//gqa}])*ks[kv*{context}+first+key]*{1/math.sqrt(dim):.17g}f'
    assert source.count(old)==1
    source = source.replace(old,new)
  old = f'float4 k=convert_float4(vload4(0,cache+(kv*{context}+first+key)*D+c));'
  assert source.count(old)==1
  source = source.replace(old,f'char4 k=vload4(0,keys+(kv*{context}+first+key)*D+c);')
  args = (out,q,cache,end,count,qs,keys,ks)
  return _program('mali_attention_partial',source,args,base.arg.global_size,base.arg.local_size)


def quantize(x):
  scale = np.maximum(np.max(abs(x),axis=-1)/np.float32(127),np.float32(1e-8))
  return np.clip(np.rint(x*(np.float32(1)/scale)[...,None]),-127,127).astype(np.int8),scale


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--compile-only',action='store_true')
  parser.add_argument('--subgroups',action='store_true')
  parser.add_argument('--onfly-keys',action='store_true')
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  heads,kv_heads,dim,capacity,context = 16,8,128,args.rows,args.context
  assert 1<=capacity<=64 and context>=capacity
  parts = ceildiv(context,512)
  report = {'complete':False,'scope':'isolated GPU attention, offline quantization/copies excluded, not serving qualification',
            'context':context,'rows':capacity,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    required = ['cl_arm_integer_dot_product_int8','cl_arm_integer_dot_product_accumulate_int8']
    assert all(x in Device['CL'].device_exts for x in required)
    if args.compile_only:
      types = [dtypes.float,dtypes.int8,dtypes.half,dtypes.int,dtypes.int,dtypes.float,dtypes.int8,dtypes.float]
      shapes = [(heads,capacity,parts,dim+4),(1,heads,capacity,dim),(2,1,kv_heads,context,dim),(1,),(1,),
                (1,heads,capacity),(kv_heads,context,dim),(kv_heads,context)]
      params = [UOp.param(i,dtype,shape) for i,(dtype,shape) in enumerate(zip(types,shapes))]
      node = dot_partial(*params,heads,kv_heads,dim,context,capacity,min(capacity,4),args.subgroups,args.onfly_keys)
      source = next(u.arg for u in node.src if u.op is Ops.SOURCE)
      binary = Device['CL'].cl_compiler.compile_cached(source)
      report.update(complete=True,compiled_binary_bytes=len(binary),kernel_source_sha256=hashlib.sha256(source.encode()).hexdigest())
      save()
      print(json.dumps(report),flush=True)
      return
    rng = np.random.default_rng(120)
    raw_q = rng.normal(0,.3,(1,heads,capacity,dim)).astype(np.float32)
    raw_cache = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
    end_value = context-1
    raw_cache[:,:,:,end_value:] = np.nan
    q8,qs = quantize(raw_q)
    k8,ks = quantize(np.nan_to_num(raw_cache[0,0].astype(np.float32)))
    q,cache,qi,qscale,keys,kscale = (Tensor(x,device='CL').realize() for x in (raw_q,raw_cache,q8,qs,k8,ks))
    end = Tensor([end_value],device='CL',dtype=dtypes.int).realize()
    def build(dot,query_rows):
      workspace = Tensor.empty(heads,capacity,parts,dim+4,device='CL',dtype=dtypes.float).realize()
      def run(q,cache,end,count,qs,keys,ks):
        if dot:
          partial = workspace.custom_kernel(q,cache,end,count,qs,keys,ks,
            fxn=lambda o,q,k,e,c,s,ki,si:dot_partial(o,q,k,e,c,s,ki,si,heads,kv_heads,dim,context,capacity,query_rows,
                                                  args.subgroups,args.onfly_keys))[0]
        else:
          partial = workspace.custom_kernel(q,cache,end,count,
            fxn=lambda o,q,k,e,c:_partial(o,q,k,e,c,heads,kv_heads,dim,context,capacity,512,query_rows,True))[0]
        out = Tensor.empty(1,heads,capacity,dim,device='CL',dtype=dtypes.float)
        return out.custom_kernel(partial,count,fxn=lambda o,p,c:_merge(o,p,c,heads,dim,capacity,parts))[0].realize()
      return TinyJit(run)
    variants = {'accepted':(build(False,min(capacity,2)),q),**{f'dot{n}':(build(True,min(capacity,n)),qi) for n in (1,2,3,4)}}
    checks = []
    for active in dict.fromkeys([capacity,max(1,capacity-1),min(7,capacity),1]):
      count = Tensor([active],device='CL',dtype=dtypes.int).realize()
      expected = variants['accepted'][0](q,cache,end,count,qscale,keys,kscale).numpy()[:,:,:active]
      for label,(run,inputs) in variants.items():
        actual = run(inputs,cache,end,count,qscale,keys,kscale).numpy()[:,:,:active]
        assert np.isfinite(actual).all()
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        assert error<=.02,(label,active,error)
        checks.append({'variant':label,'rows':active,'normalized_rmse':error})
    count = Tensor([capacity],device='CL',dtype=dtypes.int).realize()
    samples = {label:[] for label in variants}
    for index in range(9):
      for label in variants if index%2==0 else reversed(variants):
        run,inputs = variants[label]
        Device['CL'].synchronize()
        start=time.perf_counter()
        run(inputs,cache,end,count,qscale,keys,kscale)
        Device['CL'].synchronize()
        if index>=2:samples[label].append((time.perf_counter()-start)*1000)
    report.update(complete=True,checks=checks,variants={k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()})
    save()
    print(json.dumps(report),flush=True)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise


if __name__=='__main__':main()
