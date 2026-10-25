"""Test FP16 eight-feature partial dots with FP32 reduction/softmax/PV.

No serving changes. Quantization/rounding is uniform across query counts. The
fixed primitive normalized-RMSE gate is0.02; real model quality and cache-output
gates are required independently before this can be considered for serving.
"""
import argparse, functools, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _partial, _merge
from tinygrad.uop.ops import Ops, UOp


@functools.cache
def partial(out,q,cache,end,count,heads,kv_heads,dim,context,capacity,query_rows):
  assert dim==128
  node = _partial(out,q,cache,end,count,heads,kv_heads,dim,context,capacity,512,query_rows,True)
  source = next(u.arg for u in node.src if u.op is Ops.SOURCE)
  group,gqa = heads//kv_heads*query_rows,heads//kv_heads
  assert group%2==0
  first = source.index('int subgroup=get_sub_group_id(),sublane=get_sub_group_local_id();')
  last = source.index('\n    barrier(CLK_LOCAL_MEM_FENCE);',first)
  query = 'int subgroup=get_sub_group_id(),sublane=get_sub_group_local_id();'
  for h in range(group):
    pointer = f'query+((kv*{gqa}+{h%gqa})*C+row+{h//gqa})*D+sublane*8'
    query += f'half8 q{h}=row+{h//gqa}<count?convert_half8_rte(vload8(0,{pointer})):(half8)0;'
  dots = ''.join(f'half2 dot{i}=(half2)0;' for i in range(group//2))
  for c in range(8):
    dots += ''.join(f'dot{i}=fma((half2)(q{2*i}.s{c:x},q{2*i+1}.s{c:x}),(half2)k.s{c:x},dot{i});' for i in range(group//2))
  for h in range(group):
    dots += f'''{{float result=sub_group_reduce_add(convert_float(dot{h//2}.s{h%2}))*0.088388347648318447f;
      if(sublane==0)prob[{h}*N+key]=row+{h//gqa}<count&&first+key<min(total,total-count+row+{h//gqa}+1)?result:-INFINITY;}}'''
  body = f'''for(int key=subgroup;key<N;key+=get_num_sub_groups()){{
    half8 k=first+key<last?vload8(0,cache+(kv*{context}+first+key)*D+sublane*8):(half8)0;{dots}}}'''
  source = source[:first]+query+body+source[last:]
  return node.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
    UOp(Ops.BINARY,arg=source.encode()) if u.op is Ops.BINARY else u for u in node.src))


def runner(context,capacity,query_rows,half):
  parts = ceildiv(context,512)
  workspace = Tensor.empty(16,capacity,parts,132,device='CL',dtype=dtypes.float).realize()
  def run(q,cache,end,count):
    p = workspace.custom_kernel(q,cache,end,count,
      fxn=lambda o,q,k,e,c:partial(o,q,k,e,c,16,8,128,context,capacity,query_rows) if half else
        _partial(o,q,k,e,c,16,8,128,context,capacity,512,query_rows,True))[0]
    out = Tensor.empty(1,16,capacity,128,device='CL',dtype=dtypes.float)
    return out.custom_kernel(p,count,fxn=lambda o,p,c:_merge(o,p,c,16,128,capacity,parts))[0].realize()
  return TinyJit(run)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(124)
  raw_q = rng.normal(0,1,(1,16,args.rows,128)).astype(np.float32)
  raw_kv = rng.normal(0,.3,(2,1,8,args.context,128)).astype(np.float16)
  raw_kv[0] *= np.float16(1/.3)
  raw_kv[:,:,:,args.context-1:] = np.nan
  q,kv = Tensor(raw_q,device='CL').realize(),Tensor(raw_kv,device='CL').realize()
  end = Tensor([args.context-1],device='CL',dtype=dtypes.int).realize()
  runs = {'accepted':runner(args.context,args.rows,min(args.rows,2),False),
          **{f'half{n}':runner(args.context,args.rows,min(args.rows,n),True) for n in (1,2,3,4)}}
  checks = []
  for active in dict.fromkeys([args.rows,max(1,args.rows-1),min(args.rows,7),1]):
    count = Tensor([active],device='CL',dtype=dtypes.int).realize()
    expected = runs['accepted'](q,kv,end,count).numpy()[:,:,:active]
    for name,run in runs.items():
      actual = run(q,kv,end,count).numpy()[:,:,:active]
      assert np.isfinite(actual).all()
      error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
      assert error<=.02,(name,active,error)
      checks.append({'variant':name,'rows':active,'normalized_rmse':error})
  count = Tensor([args.rows],device='CL',dtype=dtypes.int).realize()
  samples = {name:[] for name in runs}
  for index in range(9):
    for name in runs if index%2==0 else reversed(runs):
      Device['CL'].synchronize()
      begin = time.perf_counter()
      runs[name](q,kv,end,count)
      Device['CL'].synchronize()
      if index>=2:samples[name].append((time.perf_counter()-begin)*1000)
  report = {'complete':True,'scope':'isolated half partial-score experiment; not serving qualification',
            'context':args.context,'rows':args.rows,'checks':checks,
            'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
