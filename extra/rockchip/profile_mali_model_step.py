"""Attribute a near-limit Qwen decoder step to actual GPU kernels.

Initialize the past KV with deterministic random FP16 data, then execute real
model forwards. This is diagnostic attribution, not generation/quality timing.
Per-kernel waits affect scheduling, so compare full-request benchmarks separately.
"""
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, Device
from tinygrad.llm.model import Transformer
from tinygrad.llm.kernels.mali import configure
from tinygrad.runtime.ops_cl import CLProgram


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--context',type=int,default=16384)
  parser.add_argument('--rows',type=int,default=64)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  report = {'complete':False,'scope':'one real GPU model step with random prior KV; per-kernel synchronization; not a quality/performance gate',
            'context':args.context,'rows':args.rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  counts,timings = collections.Counter(),collections.Counter()
  original_init,original_call = CLProgram.__init__,CLProgram.__call__
  def initialize(self,device,obj):
    original_init(self,device,obj)
    self.diagnostic_name = obj.name
  def call(self,*a,**kw):
    kw['wait'] = True
    seconds = original_call(self,*a,**kw)
    counts[self.diagnostic_name] += 1
    timings[self.diagnostic_name] += seconds or 0
    return seconds
  CLProgram.__init__,CLProgram.__call__ = initialize,call
  try:
    model,_ = Transformer.from_gguf(args.checkpoint,max_context=args.context,realize=True)
    report['gpu_kernels'] = configure(model)
    dummy = Tensor.zeros(1,1,model.config.dim,device='CL')
    rng = np.random.default_rng(121)
    for index,block in enumerate(model.blk):
      block._init_state(dummy)
      raw = (rng.standard_normal(block.cache_kv.shape,dtype=np.float32)*np.float32(.3)).astype(np.float16)
      block.cache_kv.realize().assign(Tensor(raw,device='CL').realize()).realize()
      if index%7==0:print(json.dumps({'event':'cache_initialized','layer':index}),flush=True)
    inputs = Tensor([[42]*args.rows],device='CL',dtype='int32').realize()
    start = UOp.variable('start_pos',0,args.context-1).bind(args.context-args.rows-64)
    run = TinyJit(lambda ids,pos:model.logits(ids,pos).realize())
    for _ in range(3):run(inputs,start).numpy()
    counts.clear()
    timings.clear()
    begin = time.perf_counter()
    values = run(inputs,start).numpy()
    Device['CL'].synchronize()
    assert np.isfinite(values).all()
    report.update(complete=True,wall_seconds=time.perf_counter()-begin,kernels=[{'name':name,'calls':counts[name],'gpu_ms':seconds*1000}
      for name,seconds in timings.most_common()])
    save()
    print(json.dumps({k:report[k] for k in ['complete','wall_seconds','kernels']}),flush=True)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise
  finally:CLProgram.__init__,CLProgram.__call__ = original_init,original_call


if __name__=='__main__':main()
