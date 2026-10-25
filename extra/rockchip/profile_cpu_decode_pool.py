"""Measure the existing CPU attention pool with three KV heads; serving is unchanged."""
import argparse, inspect, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels import rockchip_cpu as cpu
from tinygrad.llm.kernels.rockchip_pool import attention_pool


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  context,heads,kv_heads,dim,length = 8192,9,3,64,8189
  rng = np.random.default_rng(82)
  query = rng.normal(0,.3,(1,heads,1,dim)).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
  cache[:,:,:,length:] = np.nan
  expected = np.empty_like(query)
  for head in range(heads):
    key,value = cache[:,0,head//3,:length].astype(np.float32)
    scores = query[0,head]@key.T/np.float32(8)
    probability = np.exp(scores-scores.max(-1,keepdims=True))
    probability /= probability.sum(-1,keepdims=True)
    expected[0,head] = probability@value
  source = inspect.getsource(cpu._attention.__wrapped__)
  assert source.count('kv_heads>=4')==1
  namespace = dict(vars(cpu))
  exec(source.replace('kv_heads>=4','kv_heads>=2'),namespace)
  modified = namespace['_attention']
  pool = attention_pool()
  q,kv = Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize()
  variants = {}
  for name,build in [('existing',cpu._attention),('three_kv_pool',modified)]:
    def execute(q,kv,build=build):
      return Tensor.empty(*query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,kv,
        fxn=lambda out,q,kv:build(out,q,kv,None,heads,kv_heads,dim,context,static_length=length,
                                 pool=pool.address,fast_softmax=True))[0].realize()
    variants[name] = TinyJit(execute)
  samples,errors = {name:[] for name in variants},{}
  with pool:
    reference = None
    for name,run in variants.items():
      for _ in range(3):run(q,kv)
      actual = run(q,kv).numpy()
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
      if reference is None: reference = actual
      else: np.testing.assert_array_equal(actual,reference)
      errors[name] = float(abs(actual-expected).max())
    for index in range(12):
      for name in variants if index%2==0 else reversed(variants):
        begin = time.perf_counter()
        variants[name](q,kv)
        if index>=2:samples[name].append((time.perf_counter()-begin)*1000)
  report = {'context':context,'valid_positions':length,'heads':heads,'kv_heads':kv_heads,'dim':dim,
            'scope':'CPU primitive experiment; existing FP32 arithmetic and FP16 KV; serving source unchanged',
            'variants':{name:{'median_ms':statistics.median(values),'samples_ms':values,'max_abs_error':errors[name]}
                        for name,values in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
