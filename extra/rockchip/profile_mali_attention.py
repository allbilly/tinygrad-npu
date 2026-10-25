"""Compare bounded FP32 Mali attention tiles against the same causal reference."""
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.llm.kernels.mali_attention import _partial, _merge
from tinygrad.uop.ops import Ops, UOp


def runner(heads,kv_heads,dim,context,rows,chunk,query_rows,padding):
  parts = (context+chunk-1)//chunk
  workspace = Tensor.empty(heads,rows,parts,dim+4,device='CL',dtype=dtypes.float).realize()
  def partial(out,q,cache,end,count):
    base = _partial(out,q,cache,end,count,heads,kv_heads,dim,context,rows,chunk,query_rows,True)
    if not padding:return base
    source = next(u.arg for u in base.src if u.op is Ops.SOURCE)
    source = source.replace('prob[G*N]','prob[G*(N+16)]').replace('*N+','*(N+16)+')
    return base.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
                                  UOp(Ops.BINARY,arg=source.encode()) if u.op is Ops.BINARY else u for u in base.src))
  def execute(q,cache,end,count):
    p = workspace.custom_kernel(q,cache,end,count,fxn=partial)[0]
    out = Tensor.empty(1,heads,rows,dim,device='CL',dtype=dtypes.float)
    return out.custom_kernel(p,count,fxn=lambda o,p,c:_merge(o,p,c,heads,dim,rows,parts))[0].realize()
  return TinyJit(execute)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--runs',type=int,default=5)
  parser.add_argument('--heads',type=int,default=16)
  parser.add_argument('--kv-heads',type=int,default=8)
  parser.add_argument('--dim',type=int,default=128)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if not 1<=args.rows<=128 or args.rows>args.context or args.runs<1:parser.error('Unsupported query geometry')
  heads,kv_heads,dim = args.heads,args.kv_heads,args.dim
  if heads<1 or kv_heads<1 or heads%kv_heads or dim not in (32,64,128):parser.error('Unsupported head geometry')
  rng = np.random.default_rng(54)
  query = rng.normal(0,.3,(1,heads,args.rows,dim)).astype(np.float32)
  values = rng.normal(0,.3,(2,1,kv_heads,args.context,dim)).astype(np.float16)
  expected = np.empty_like(query)
  mask = np.arange(args.context)[None]<=args.context-args.rows+np.arange(args.rows)[:,None]
  for head in range(heads):
    scores = query[0,head]@values[0,0,head//(heads//kv_heads)].astype(np.float32).T/np.float32(np.sqrt(dim))
    scores = np.where(mask,scores,-np.inf)
    p = np.exp(scores-scores.max(-1,keepdims=True))
    p /= p.sum(-1,keepdims=True)
    expected[0,head] = p@values[1,0,head//(heads//kv_heads)].astype(np.float32)
  inputs = (Tensor(query,device='CL').realize(),Tensor(values,device='CL').realize(),
            Tensor([args.context],device='CL',dtype=dtypes.int).realize(),Tensor([args.rows],device='CL',dtype=dtypes.int).realize())
  variants = [(q,c,p,runner(heads,kv_heads,dim,args.context,args.rows,c,q,p))
              for q,c,p in ((2,512,0),(1,512,0),(2,256,0),(3,512,0),(3,256,0),(3,256,16),(4,256,16))]
  samples = {(q,c,p):[] for q,c,p,_ in variants}
  errors = {}
  for q,c,p,run in variants:
    for _ in range(3):run(*inputs)
    actual = run(*inputs).numpy()
    np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=4e-4)
    errors[q,c,p] = float(np.max(np.abs(actual-expected)))
  for index in range(args.runs+2):
    for q,c,p,run in variants if index%2==0 else reversed(variants):
      Device['CL'].synchronize()
      start = time.perf_counter()
      run(*inputs)
      Device['CL'].synchronize()
      if index>=2:samples[q,c,p].append((time.perf_counter()-start)*1000)
  result = {'context':args.context,'rows':args.rows,'heads':heads,'kv_heads':kv_heads,'dim':dim,
            'arithmetic':'FP32 scores/softmax/accumulation; FP16 KV unchanged',
            'variants':[{'query_rows':q,'chunk':c,'score_padding':p,'median_ms':statistics.median(v),
                         'samples_ms':v,'max_abs_error':errors[q,c,p]} for (q,c,p),v in samples.items()]}
  args.report.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)


if __name__=='__main__':main()
