"""Compare balanced query-row jobs and register tiles without changing serving code."""
import argparse, hashlib, json, re, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _attention, _renderer
from tinygrad.uop.ops import Ops, UOp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  for name,default in (('heads',9),('kv-heads',3),('dim',64),('context',8192),('capacity',128),('total',8191)):
    parser.add_argument('--'+name,type=int,default=default)
  args = parser.parse_args()
  heads,kv_heads,dim,context,capacity,total = args.heads,args.kv_heads,args.dim,args.context,args.capacity,args.total
  if not 1<=kv_heads<=heads or heads%kv_heads or not 128<=capacity<=total<=context:parser.error('Invalid attention geometry')
  rng = np.random.default_rng(84)
  query = rng.normal(0,.3,(1,heads,capacity,dim)).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
  cache[:,:,:,total:] = np.nan
  q,kv = Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize()
  end = Tensor([total],device='CPU',dtype=dtypes.int).realize()
  variants = {}
  for label,block,tile,jobs in (('existing',8,2,False),('jobs8',8,2,True),('jobs4x4',4,4,True),
                               ('jobs8x1',8,1,True),('jobs16x1',16,1,True),('jobs16x2',16,2,True)):
    def program(out,q,kv,end,rows,block=block,tile=tile,jobs=jobs):
      base = _attention(out,q,kv,end,heads,kv_heads,dim,context,rows=rows,max_rows=capacity,
                        query_rows=block,channel_tile=tile,fast_softmax=True)
      source = next(u.arg for u in base.src if u.op is Ops.SOURCE)
      source,count = re.subn(r'float p\['+str(block)+r'\]\[\d+\]',f'float p[{block}][{context+16}]',source)
      assert count==1
      if jobs:
        replacements = {
          'if(active_rows<16) attention_part(':f'int jobs={heads}*((active_rows+{block-1})/{block});if(active_rows<16) attention_part(',
          f',active_rows,0,{heads});':',active_rows,0,jobs);',
          f'active_rows,{heads}*i/4,{heads}*(i+1)/4': 'active_rows,jobs*i/4,jobs*(i+1)/4',
          'for(int head=begin;head<end;head++) {':f'''for(int job=begin;job<end;job++) {{
            int tiles=(active_rows+{block-1})/{block},head=job/tiles,first_row=(job%tiles)*{block};
            int stop_row=first_row+{block}<active_rows?first_row+{block}:active_rows;''',
          'int row=0;':'int row=first_row;',
          f'row+{block}<=active_rows':f'row+{block}<=stop_row',
          'row<active_rows;row++':'row<stop_row;row++',
        }
        for old,new in replacements.items():
          assert source.count(old)==1,(old,source.count(old))
          source = source.replace(old,new)
      binary = _renderer().compiler.compile_cached(source)
      return base.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
                                    UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in base.src))
    def execute(q,kv,end,rows,program=program):
      return Tensor.empty(*query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,kv,end,rows,fxn=program)[0].realize()
    variants[label] = TinyJit(execute)
  checks = []
  for count in (128,65,7,127):
    rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
    expected = np.empty((1,heads,count,dim),dtype=np.float32)
    for head in range(heads):
      key,value = cache[:,0,head//(heads//kv_heads),:total].astype(np.float32)
      scores = query[0,head,:count]@key.T/np.float32(np.sqrt(dim))
      scores[np.arange(total)[None]>total-count+np.arange(count)[:,None]] = -np.inf
      p = np.exp(scores-scores.max(-1,keepdims=True))
      p /= p.sum(-1,keepdims=True)
      expected[0,head] = p@value
    reference = None
    for label,run in variants.items():
      actual = run(q,kv,end,rows).numpy()[:,:,:count]
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
      if reference is None:reference = actual
      else:np.testing.assert_array_equal(actual,reference)
      checks.append({'variant':label,'rows':count,'max_abs_error':float(abs(actual-expected).max())})
  samples = {label:[] for label in variants}
  rows = Tensor([capacity],device='CPU',dtype=dtypes.int).realize()
  for index in range(9):
    for label in variants if index%2==0 else reversed(variants):
      start = time.perf_counter()
      variants[label](q,kv,end,rows)
      if index>=2:samples[label].append((time.perf_counter()-start)*1000)
  report = {'context':context,'heads':heads,'kv_heads':kv_heads,'dim':dim,'checks':checks,
            'total':total,'capacity':capacity,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (Path(__file__),Path('tinygrad/llm/kernels/rockchip_cpu.py'))},
            'scope':'CPU primitive experiment; all variants bitwise equal to original; no serving source changed',
            'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
