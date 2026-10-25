"""Compare CPU attention query grouping and score-row padding without changing serving defaults."""
import argparse, json, re, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _attention, _renderer
from tinygrad.uop.ops import Ops, UOp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=8192)
  parser.add_argument('--rows',type=int,default=128)
  parser.add_argument('--heads',type=int,default=9)
  parser.add_argument('--kv-heads',type=int,default=3)
  parser.add_argument('--dim',type=int,default=64)
  parser.add_argument('--runs',type=int,default=5)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if not 1<=args.rows<=args.context<=8192 or args.heads%args.kv_heads or args.dim%8 or args.runs<1:
    parser.error('Unsupported attention geometry')
  rng = np.random.default_rng(73)
  query = rng.normal(0,.3,(1,args.heads,args.rows,args.dim)).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,args.kv_heads,args.context,args.dim)).astype(np.float16)
  expected = np.empty_like(query)
  for head in range(args.heads):
    kv = head//(args.heads//args.kv_heads)
    scores = query[0,head]@cache[0,0,kv].astype(np.float32).T/np.float32(np.sqrt(args.dim))
    scores[np.arange(args.context)[None]>args.context-args.rows+np.arange(args.rows)[:,None]] = -np.inf
    probability = np.exp(scores-scores.max(-1,keepdims=True))
    probability /= probability.sum(-1,keepdims=True)
    expected[0,head] = probability@cache[1,0,kv].astype(np.float32)
  q,kv = Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize()
  result = {'context':args.context,'rows':args.rows,'heads':args.heads,'kv_heads':args.kv_heads,'dim':args.dim,
            'scope':'isolated CPU attention; experimental source variants; serving defaults unchanged','variants':[]}
  for block,pad in ((4,0),(4,16),(8,0),(8,16)):
    def program(out,query,cache):
      base = _attention(out,query,cache,None,args.heads,args.kv_heads,args.dim,args.context,
                        max_rows=args.rows,static_length=args.context,query_rows=block,fast_softmax=True)
      original = next(u.arg for u in base.src if u.op is Ops.SOURCE)
      source,count = re.subn(r'float p\['+str(block)+r'\]\[\d+\]',f'float p[{block}][{args.context+pad}]',original)
      assert count==1
      if source==original:return base
      binary = _renderer().compiler.compile_cached(source)
      return base.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
                                    UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in base.src))
    run = TinyJit(lambda q,kv:Tensor.empty(*query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,kv,fxn=program)[0].realize())
    samples = []
    for index in range(args.runs+3):
      started = time.perf_counter()
      output = run(q,kv)
      elapsed = (time.perf_counter()-started)*1000
      if index>=3:samples.append(elapsed)
    actual = output.numpy()
    np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
    row = {'query_rows':block,'score_padding':pad,'median_ms':statistics.median(samples),'samples_ms':samples,
           'max_abs_error':float(np.max(np.abs(actual-expected)))}
    result['variants'].append(row)
    print(json.dumps(row),flush=True)
  args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
