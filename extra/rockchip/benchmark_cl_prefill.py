"""Measure actual-checkpoint OpenCL prefill with optional prepared Mali kernels."""
import argparse, json, os, statistics, time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--context',type=int,default=8192)
  parser.add_argument('--tokens',type=int,default=128)
  parser.add_argument('--chunk',type=int,default=128)
  parser.add_argument('--runs',type=int,default=2)
  parser.add_argument('--warmup-tokens',type=int)
  parser.add_argument('--prepared',action='store_true')
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CL',HALF='1')
  from tinygrad import Device
  from tinygrad.helpers import GlobalCounters
  from tinygrad.llm.model import Transformer
  start = time.perf_counter()
  model,_ = Transformer.from_gguf(args.checkpoint,args.context,realize=True)
  if args.prepared:
    from tinygrad.llm.kernels.mali import configure
    print(configure(model,capacities=(args.chunk,)),flush=True)
  report = {'context':args.context,'tokens':args.tokens,'chunk':args.chunk,'prepared':args.prepared,
            'load_seconds':time.perf_counter()-start,'config':str(model.config),'trials':[]}
  print('loaded',report['load_seconds'],flush=True)
  for index in range(args.runs+3):
    model._cached_tokens = []
    start = time.perf_counter()
    tokens = args.warmup_tokens if index<3 and args.warmup_tokens else args.tokens
    token = next(iter(model.generate([5]*tokens,chunk_size=args.chunk,temperature=0)))
    Device['CL'].synchronize()
    elapsed = time.perf_counter()-start
    row = {'index':index,'warmup':index<3,'tokens':tokens,'seconds':elapsed,'tok_s':tokens/elapsed,
           'token':token,'allocated_bytes':GlobalCounters.mem_used}
    report['trials'].append(row)
    print(json.dumps(row),flush=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
  report.update(complete=True,median_seconds=statistics.median(row['seconds'] for row in report['trials'] if not row['warmup']))
  args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
