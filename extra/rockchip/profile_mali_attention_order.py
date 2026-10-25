"""Compare Mali attention workgroup order with unchanged FP32 arithmetic."""
import argparse, json, re, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _partial, _merge
from tinygrad.uop.ops import Ops, UOp


def runner(heads,kv_heads,dim,context,capacity,order):
  parts = ceildiv(context,512)
  gqa = heads//kv_heads
  query_rows = min(capacity,4,max(1,((48-dim//16)//(dim//16+1))//gqa),max(1,12//gqa))
  workspace = Tensor.empty(heads,capacity,parts,dim+4,device='CL',dtype=dtypes.float).realize()
  def partial(out,q,cache,end,count):
    node = _partial(out,q,cache,end,count,heads,kv_heads,dim,context,capacity,512,query_rows,True)
    source = next(u.arg for u in node.src if u.op is Ops.SOURCE)
    for label in ['part','kv','row']:
      source,n = re.subn(rf'{label}=get_group_id\(\d\)',f'{label}=get_group_id({order.index(label)})',source)
      assert n==1
    sizes = {'part':parts,'kv':kv_heads,'row':ceildiv(capacity,query_rows)}
    info = replace(node.arg,global_size=tuple(sizes[k] for k in order))
    return node.replace(arg=info,src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
      UOp(Ops.BINARY,arg=source.encode()) if u.op is Ops.BINARY else u for u in node.src))
  def run(q,cache,end,count):
    p = workspace.custom_kernel(q,cache,end,count,fxn=partial)[0]
    out = Tensor.empty(1,heads,capacity,dim,device='CL',dtype=dtypes.float)
    return out.custom_kernel(p,count,fxn=lambda o,p,c:_merge(o,p,c,heads,dim,capacity,parts))[0].realize()
  return TinyJit(run)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--heads',type=int,default=16)
  parser.add_argument('--kv-heads',type=int,default=8)
  parser.add_argument('--dim',type=int,default=128)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(122)
  raw_q = rng.normal(0,.3,(1,args.heads,args.rows,args.dim)).astype(np.float32)
  raw_kv = rng.normal(0,.3,(2,1,args.kv_heads,args.context,args.dim)).astype(np.float16)
  raw_kv[:,:,:,args.context-1:] = np.nan
  q,kv = Tensor(raw_q,device='CL').realize(),Tensor(raw_kv,device='CL').realize()
  end = Tensor([args.context-1],device='CL',dtype=dtypes.int).realize()
  orders = [('part','kv','row'),('row','part','kv'),('row','kv','part'),('kv','row','part')]
  runs = {'_'.join(order):runner(args.heads,args.kv_heads,args.dim,args.context,args.rows,order) for order in orders}
  checks = []
  for active in dict.fromkeys([args.rows,max(1,args.rows-1),min(7,args.rows),1]):
    count = Tensor([active],device='CL',dtype=dtypes.int).realize()
    expected = runs['part_kv_row'](q,kv,end,count).numpy()[:,:,:active]
    for label,run in runs.items():
      actual = run(q,kv,end,count).numpy()[:,:,:active]
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      checks.append({'order':label,'rows':active,'bitwise_equal':True})
  count = Tensor([args.rows],device='CL',dtype=dtypes.int).realize()
  samples = {label:[] for label in runs}
  for index in range(9):
    for label in runs if index%2==0 else reversed(runs):
      Device['CL'].synchronize()
      begin = time.perf_counter()
      runs[label](q,kv,end,count)
      Device['CL'].synchronize()
      if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
  report = {'complete':True,'context':args.context,'rows':args.rows,'heads':args.heads,'kv_heads':args.kv_heads,'dim':args.dim,
            'scope':'same arithmetic/allocations; workgroup coordinate order only','checks':checks,
            'variants':{label:{'median_ms':statistics.median(v),'samples_ms':v} for label,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
