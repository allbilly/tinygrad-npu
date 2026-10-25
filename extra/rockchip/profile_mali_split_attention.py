"""Separate score/softmax and value accumulation, retaining FP32 operations.

This diagnostic measures the register/scheduling cost of the combined shader.
Global FP32 probabilities add scratch storage; serving is unchanged.
"""
import argparse, collections, functools, json, re, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _partial, _merge, _program
from tinygrad.runtime.ops_cl import CLProgram
from tinygrad.uop.ops import Ops, UOp


def sources(heads,kv_heads,dim,context,capacity,query_rows):
  params = [UOp.param(i,dt,shape) for i,(dt,shape) in enumerate([
    (dtypes.float,(heads,capacity,ceildiv(context,512),dim+4)),(dtypes.float,(1,heads,capacity,dim)),
    (dtypes.half,(2,1,kv_heads,context,dim)),(dtypes.int,(1,)),(dtypes.int,(1,))])]
  node = _partial(*params,heads,kv_heads,dim,context,capacity,512,query_rows,True)
  source = next(u.arg for u in node.src if u.op is Ops.SOURCE)
  coordinates = re.search(r'part=get_group_id\(\d\),kv=get_group_id\(\d\),row=get_group_id\(\d\)',source)[0]
  threads,gqa,group,parts = 128 if query_rows>1 else 64,heads//kv_heads,heads//kv_heads*query_rows,ceildiv(context,512)
  first = source.index('    for(int item=lane;item<G*D/4;item+=')
  last = source.index('\n    if(lane==0',first)
  pv = source[first:last].replace('prob[h*N+','prob[((head*C+t)*P+part)*N+')
  copy = f'''for(int i=lane;i<G*N;i+={threads}){{int h=i/N,key=i%N,t=row+h/{gqa},head=kv*{gqa}+h%{gqa};
    if(t<C)saved_prob[((head*C+t)*P+part)*N+key]=prob[i];}}
  '''
  qk = source[:first]+copy+source[last:]
  qk = qk.replace('void mali_attention_partial(','void mali_attention_scores(')
  signature = '__global const int *rows) {'
  assert qk.count(signature)==1
  qk = qk.replace(signature,'__global const int *rows,__global float *saved_prob) {')
  value = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #define G {group}
  #define D {dim}
  #define N 512
  #define C {capacity}
  #define P {parts}
  #define S {dim+4}
  __kernel void mali_attention_values(__global float *out,__global const half *cache,
      __global const int *length,__global const int *rows,__global const float *prob){{
    int lane=get_local_id(0),{coordinates}*{query_rows};
    int total=*length,count=*rows,first=part*N,last=min(total,total-count+row+{query_rows});
    if(count<1||count>C||total<count||total>{context}||row>=count||first>=last){{
      int invalid=count<1||count>C||total<count||total>{context};
      for(int i=lane;i<G*D;i+={threads}){{int h=i/D,c=i%D,t=row+h/{gqa},head=kv*{gqa}+h%{gqa};
        if(t<C)out[((head*C+t)*P+part)*S+c]=invalid?NAN:0;}}
      return;
    }}
    {pv}
  }}'''
  return qk,value,node.arg.global_size,node.arg.local_size


@functools.cache
def scores(out,q,cache,end,count,prob,heads,kv_heads,dim,context,capacity,query_rows):
  source,_,groups,local = sources(heads,kv_heads,dim,context,capacity,query_rows)
  args = (out,q,cache,end,count,prob)
  node = _program('mali_attention_scores',source,args,groups,local)
  return node.replace(arg=replace(node.arg,outs=(0,5),ins=(1,2,3,4)))


@functools.cache
def values(out,cache,end,count,prob,heads,kv_heads,dim,context,capacity,query_rows):
  _,source,groups,local = sources(heads,kv_heads,dim,context,capacity,query_rows)
  node = _program('mali_attention_values',source,(out,cache,end,count,prob),groups,local)
  return node.replace(arg=replace(node.arg,ins=(0,1,2,3,4)))


def runner(heads,kv_heads,dim,context,capacity,split):
  parts = ceildiv(context,512)
  gqa = heads//kv_heads
  query_rows = min(capacity,4,max(1,((48-dim//16)//(dim//16+1))//gqa),max(1,12//gqa))
  workspace = Tensor.empty(heads,capacity,parts,dim+4,device='CL',dtype=dtypes.float).realize()
  probabilities = Tensor.empty(heads,capacity,parts,512,device='CL',dtype=dtypes.float).realize() if split else None
  def run(q,cache,end,count):
    if probabilities is None:
      p = workspace.custom_kernel(q,cache,end,count,
        fxn=lambda o,q,k,e,c:_partial(o,q,k,e,c,heads,kv_heads,dim,context,capacity,512,query_rows,True))[0]
    else:
      result = workspace.custom_kernel(q,cache,end,count,probabilities,
        fxn=lambda o,q,k,e,c,pr:scores(o,q,k,e,c,pr,heads,kv_heads,dim,context,capacity,query_rows))
      p = result[0].custom_kernel(cache,end,count,result[5],
        fxn=lambda o,k,e,c,pr:values(o,k,e,c,pr,heads,kv_heads,dim,context,capacity,query_rows))[0]
    out = Tensor.empty(1,heads,capacity,dim,device='CL',dtype=dtypes.float)
    return out.custom_kernel(p,count,fxn=lambda o,p,c:_merge(o,p,c,heads,dim,capacity,parts))[0].realize()
  return TinyJit(run)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  heads,kv_heads,dim = 16,8,128
  rng = np.random.default_rng(123)
  qdata = rng.normal(0,.3,(1,heads,args.rows,dim)).astype(np.float32)
  cachedata = rng.normal(0,.3,(2,1,kv_heads,args.context,dim)).astype(np.float16)
  cachedata[:,:,:,args.context-1:] = np.nan
  q,cache = Tensor(qdata,device='CL').realize(),Tensor(cachedata,device='CL').realize()
  end = Tensor([args.context-1],device='CL',dtype=dtypes.int).realize()
  stages = collections.defaultdict(lambda:collections.defaultdict(list))
  current = None
  original_init,original_call = CLProgram.__init__,CLProgram.__call__
  def initialize(self,device,obj):
    original_init(self,device,obj)
    self.diagnostic_name = obj.name
  def timed(self,*a,**kw):
    kw['wait'] = True
    result = original_call(self,*a,**kw)
    if current is not None:stages[current][self.diagnostic_name].append((result or 0)*1000)
    return result
  CLProgram.__init__,CLProgram.__call__ = initialize,timed
  try:
    runs = {name:runner(heads,kv_heads,dim,args.context,args.rows,split) for name,split in [('accepted',False),('split',True)]}
    checks = []
    for active in dict.fromkeys([args.rows,max(1,args.rows-1),min(7,args.rows),1]):
      count = Tensor([active],device='CL',dtype=dtypes.int).realize()
      expected = runs['accepted'](q,cache,end,count).numpy()[:,:,:active]
      actual = runs['split'](q,cache,end,count).numpy()[:,:,:active]
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=4e-4)
      checks.append({'rows':active,'bitwise_equal':bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32))),
                     'max_abs_error':float(abs(actual-expected).max())})
    count = Tensor([args.rows],device='CL',dtype=dtypes.int).realize()
    samples = {name:[] for name in runs}
    for index in range(9):
      for name in runs if index%2==0 else reversed(runs):
        current = name if index>=2 else None
        begin = time.perf_counter()
        runs[name](q,cache,end,count)
        Device['CL'].synchronize()
        if index>=2:samples[name].append((time.perf_counter()-begin)*1000)
    report = {'complete':True,'scope':'diagnostic FP32 kernel split, per-kernel synchronization','checks':checks,
              'context':args.context,'rows':args.rows,'variants':{name:{'median_ms':statistics.median(v),'samples_ms':v,
                'stages':{k:statistics.median(times) for k,times in stages[name].items()}} for name,v in samples.items()}}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
  finally:CLProgram.__init__,CLProgram.__call__ = original_init,original_call


if __name__=='__main__':main()
