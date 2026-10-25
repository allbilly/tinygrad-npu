"""Compare full-model projection arithmetic at distant KV positions.

Uses a synthetic zero past, not a semantic long-context quality benchmark.
One model is retained; checkpoint weights and earlier cache positions are shared
between the dense and fused/Q8 passes. No SDK performance claim is made here.
"""
import argparse, gc, json, os
from pathlib import Path
import numpy as np
from extra.rockchip.quality_inference import digest, sources, check_sources


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--context',type=int,default=40960)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  os.environ['DEV'] = 'CL'
  from tinygrad import Device, Tensor, TinyJit, UOp
  from tinygrad.helpers import GlobalCounters
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.kernels.mali import configure
  hashes = sources()
  report = {'complete':False,'scope':'synthetic past; arithmetic and cache-boundary equivalence, not semantic quality',
            'source_sha256':hashes,'checkpoint_sha256':digest(args.checkpoint),'context':args.context,'cases':[]}
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    model,_ = Transformer.from_gguf(args.checkpoint,max_context=args.context,realize=True)
    configure(model,capacities=(32,))
    cases = [(32736,32),(32767,1),(args.context-32,32),(args.context-1,1)]
    if any(start<0 or start+rows>model.max_context for start,rows in cases):raise ValueError('Context must include the documented32K boundary')
    expected = []
    def runner():
      return TinyJit(lambda tokens,position:model.logits(tokens,position).realize())
    run = runner()
    for start,rows in cases:
      ids = Tensor([[42]*rows],device='CL',dtype='int32').realize()
      position = UOp.variable('start_pos',0,model.max_context-1).bind(start)
      # Separate shape captures prevent mixing one-row and32-row JIT signatures.
      execute = run if rows==1 else runner()
      for _ in range(3):result = execute(ids,position).numpy()
      expected.append(result.copy())
    report['baseline_allocated_bytes'] = GlobalCounters.mem_used
    del run,execute
    gc.collect()
    report['gpu_kernels'] = configure(model,capacities=(32,),fuse_projections=True,q8_head_checkpoint=args.checkpoint)
    for (start,rows),reference in zip(cases,expected):
      ids = Tensor([[42]*rows],device='CL',dtype='int32').realize()
      position = UOp.variable('start_pos',0,model.max_context-1).bind(start)
      execute = runner()
      for _ in range(3):result = execute(ids,position).numpy()
      if not np.isfinite(result).all():raise AssertionError('Nonfinite candidate logits')
      np.testing.assert_array_equal(result.view(np.uint32),reference.view(np.uint32))
      report['cases'].append({'start':start,'rows':rows,'bitwise_equal':True})
      check_sources(hashes)
      save()
      print(json.dumps(report['cases'][-1]),flush=True)
    Device['CL'].synchronize()
    report.update(complete=True,optimized_allocated_bytes=GlobalCounters.mem_used)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:save()


if __name__=='__main__':main()
