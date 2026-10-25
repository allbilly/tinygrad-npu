"""Isolate one Qwen-shaped OpenCL decoder block before changing GPU kernels."""
import argparse, json, os, statistics, time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--context',type=int,default=8192)
  parser.add_argument('--rows',type=int,default=32)
  parser.add_argument('--position',type=int,default=0)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--symbolic',action='store_true')
  parser.add_argument('--prepared',action='store_true')
  args = parser.parse_args()
  if args.rows<1 or args.position<0 or args.position+args.rows>args.context:
    parser.error('Query rows and position must fit inside the context')
  os.environ['DEV']='CL'
  from tinygrad import Tensor, TinyJit, UOp, nn, Device
  from tinygrad.llm.model import TransformerConfig, TransformerBlock
  Tensor.manual_seed(14)
  config = TransformerConfig(1,1024,3072,16,8,1e-6,151936,128,1000000.,128,128,max_context=args.context,qk_norm=128)
  block = TransformerBlock(config)
  for parameter in nn.state.get_parameters(block):parameter.replace(parameter.half().contiguous().realize())
  if args.prepared:
    from tinygrad.llm.kernels.mali import configure
    print(configure(block,capacities=(args.rows,)),flush=True)
  values = Tensor.randn(1,args.rows,1024).realize()
  position = UOp.variable('start_pos',0,args.context-1)
  run = TinyJit(lambda x,start,rows:block(x[:,:rows],start).realize()) if args.symbolic else TinyJit(lambda x,start:block(x,start).realize())
  extra = (UOp.variable('toks',1,args.rows).bind(args.rows),) if args.symbolic else ()
  samples = []
  for index in range(args.runs+3):
    start = time.perf_counter()
    output = run(values,position.bind(args.position),*extra)
    Device['CL'].synchronize()
    elapsed = (time.perf_counter()-start)*1000
    if index>=3:samples.append(elapsed)
    print(json.dumps({'iteration':index,'ms':elapsed}),flush=True)
  report = {'context':args.context,'rows':args.rows,'position':args.position,'symbolic':args.symbolic,
            'median_ms':statistics.median(samples),'samples_ms':samples,
            'finite':bool(output.isfinite().all().item())}
  assert report['finite'],'Nonfinite block output'
  args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
