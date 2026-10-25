"""Check prepared Mali GEMM input strides, including the per-call padding cost."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.mali_linear import _gemm, pack_weight
from tinygrad.uop.ops import Ops, UOp


def runner(rows,columns,depth,padding):
  def program(out,x,weight):
    base = _gemm(out,x,weight,rows,columns,depth)
    if not padding:return base
    source = next(u.arg for u in base.src if u.op is Ops.SOURCE)
    old,new = f'*{depth}+k]',f'*{depth+padding}+k]'
    assert source.count(old)==4
    source = source.replace(old,new)
    return base.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
                                  UOp(Ops.BINARY,arg=source.encode()) if u.op is Ops.BINARY else u for u in base.src))
  def execute(x,weight):
    if padding:x = x.pad(((0,0),(0,0),(0,padding))).contiguous()
    return Tensor.empty(1,rows,columns,device='CL',dtype=dtypes.float).custom_kernel(x,weight,fxn=program)[0].realize()
  return TinyJit(execute)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--runs',type=int,default=7)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if not 1<=args.rows<=256 or args.runs<1:parser.error('Invalid query rows or repetitions')
  rng = np.random.default_rng(23)
  results = {'rows':args.rows,'scope':'isolated prepared GPU GEMM; unchanged FP32 arithmetic; includes input padding',
             'source_sha256':hashlib.sha256(Path('tinygrad/llm/kernels/mali_linear.py').read_bytes()).hexdigest(),'shapes':[]}
  for columns,depth in ((3072,1024),(1024,3072),(1024,2048)):
    values = rng.normal(0,.3,(1,args.rows,depth)).astype(np.float32)
    weights = (rng.standard_normal((columns,depth),dtype=np.float32)/np.float32(np.sqrt(depth))).astype(np.float16)
    expected = values@weights.astype(np.float32).T
    x = Tensor(values,device='CL').realize()
    weight = pack_weight(Tensor(weights,device='CL').realize()).realize()
    variants = [(padding,runner(args.rows,columns,depth,padding)) for padding in (0,8,16,32)]
    reference = None
    samples = {padding:[] for padding,_ in variants}
    for padding,run in variants:
      for _ in range(3):run(x,weight)
      Device['CL'].synchronize()
      actual = run(x,weight).numpy()
      np.testing.assert_allclose(actual,expected,atol=8e-6,rtol=4e-4)
      if reference is None:reference = actual
      else:np.testing.assert_array_equal(actual,reference)
    # Alternate variants after capture so driver warmup does not favor padding.
    for iteration in range(args.runs+3):
      for padding,run in variants if iteration%2==0 else reversed(variants):
        Device['CL'].synchronize()
        start = time.perf_counter()
        run(x,weight)
        Device['CL'].synchronize()
        elapsed = (time.perf_counter()-start)*1000
        if iteration>=3:samples[padding].append(elapsed)
    row = {'columns':columns,'depth':depth,'variants':[{'input_padding':p,'median_ms':statistics.median(v),'samples_ms':v}
                                                    for p,v in samples.items()],'bitwise_equal':True}
    results['shapes'].append(row)
    print(json.dumps(row),flush=True)
  args.report.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
