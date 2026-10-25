"""Warm GPT-2 124M benchmark. Prefill includes the first output token; decode excludes it."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path


def interrupts():
  rows = [line.split(':',1)[1].split() for line in Path('/proc/interrupts').read_text().splitlines()
          if 'fdab0000.npu' in line]
  return [sum(int(value) for value in row if value.isdecimal()) for row in rows]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--lengths',type=int,nargs='+',default=[8,32,128])
  parser.add_argument('--tokens',type=int,default=32)
  parser.add_argument('--context',type=int,default=1024)
  parser.add_argument('--warmups',type=int,default=4)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--npu',action='store_true')
  parser.add_argument('--optimized',action='store_true',help='Use the standalone GPT-2 adapter')
  parser.add_argument('--standard-cache',action='store_true',help='Disable head-major KV for comparison')
  parser.add_argument('--stock-attention',action='store_true',help='Disable custom CPU QK/PV for comparison')
  parser.add_argument('--no-wake-helper',action='store_true')
  parser.add_argument('--no-cache-cleaner',action='store_true')
  parser.add_argument('--no-attention-pool',action='store_true')
  parser.add_argument('--native',action='store_true')
  parser.add_argument('--bulk',action='store_true',help='Use the bounded native greedy generation loop')
  parser.add_argument('--prefill-jit',action='store_true')
  parser.add_argument('--reference',type=Path)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if min(args.lengths)<1 or max(args.lengths)+args.tokens>args.context or min(args.runs,args.warmups,args.tokens)<1:
    parser.error('Positive prompt lengths, warmups, runs, and outputs must fit context')
  if args.bulk and not (args.optimized and args.native): parser.error('--bulk requires --optimized --native')
  os.environ.update(DEV='CPU',HALF='1',MAX_CONTEXT=str(args.context),JIT_BATCH_SIZE='0')
  from tinygrad import Device, Tensor, TinyJit, Variable
  from examples.gpt2 import GPT2, VOCAB_SIZE
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit

  begin = time.perf_counter()
  model = GPT2.build('gpt2')
  transformer = model.model
  report = {'complete':False,'model':'GPT-2 124M','fp16':True,'context':args.context,'npu':args.npu,'native':args.native,
            'prefill_jit':args.prefill_jit,'bulk':args.bulk,'tokens':args.tokens,'warmups':args.warmups,'runs':args.runs,
            'affinity':sorted(os.sched_getaffinity(0)),'load_seconds':time.perf_counter()-begin,'cases':[]}
  root = Path(__file__).resolve().parents[2]
  sources = [Path(__file__),root/'examples/gpt2.py',*(root/'tinygrad').rglob('*rockchip*.py'),*(root/'tinygrad/llm/kernels').glob('rockchip*.c')]
  report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
  if args.optimized:
    from tinygrad.llm.kernels.rockchip_gpt2 import configure_gpt2
    report['placement'] = configure_gpt2(transformer,native_replay=args.native,packed_cache=not args.standard_cache,
                                       fast_attention=not args.stock_attention,wake_helper=not args.no_wake_helper,
                                       cache_cleaning=not args.no_cache_cleaner,cpu_pool=not args.no_attention_pool)
  else: report['placement'] = configure(transformer,prefill='on',decode='on',tile_columns=24576) if args.npu else None
  if args.optimized and not args.npu: parser.error('--optimized requires --npu')
  npu = Device['ROCKCHIP'] if args.npu else None
  reference = json.loads(args.reference.read_text()) if args.reference else None
  if reference and (reference['context']!=args.context or reference['tokens']!=args.tokens): raise ValueError('Reference workload differs')
  def save():
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
  def counts(): return (npu.submit_count,npu.task_count) if npu else (0,0)
  def run(jit,*inputs):
    result = jit(*inputs)
    if args.native and jit.captured is not None and not isinstance(jit.captured,RockchipCapturedJit):
      jit.captured = RockchipCapturedJit(jit.captured)
    return result
  save()
  try:
    for length in args.lengths:
      prompt = model.tokenizer.encode('The quick brown fox jumps over the lazy dog. The capital of France is Paris. '
                                      'A computer processes information using arithmetic. '*(length+1))[:length]
      expected = next(c['generated_ids'] for c in reference['cases'] if c['input_tokens']==length) if reference else None
      prefill = TinyJit(transformer.forward) if args.prefill_jit and not args.optimized else None
      case = {'input_tokens':length,'input_ids':prompt,'trials':[]}
      report['cases'].append(case)
      for index in range(args.warmups+args.runs):
        for block in transformer.h:
          if hasattr(block.attn,'cache_kv'): block.attn.cache_kv.assign(Tensor.zeros_like(block.attn.cache_kv)).realize()
        inp = Tensor([prompt],device='CPU').realize()
        before,irq_before = counts(),interrupts()
        start = time.perf_counter()
        first_pos = Variable('start_pos',0,args.context-1).bind(0)
        generated = (run(prefill,inp,first_pos,0.0) if prefill is not None else transformer(inp,first_pos,0.0)).tolist()
        first = time.perf_counter()
        first_counts = counts()
        if args.bulk: generated.extend(transformer.rockchip_decode_many(generated[-1],length,args.tokens-1))
        else:
          for step in range(args.tokens-1):
            token = Variable('tokens',0,VOCAB_SIZE-1).bind(generated[-1])
            position = Variable('start_pos',1,args.context-1).bind(length+step)
            generated.extend(run(transformer.forward_jit,token,position,0.0).tolist())
        end = time.perf_counter()
        after,irq_after = counts(),interrupts()
        row = {'index':index,'warmup':index<args.warmups,'generated_ids':generated,'start':start,'first':first,'end':end,
               'prefill_tps':length/(first-start),'decode_tps':(args.tokens-1)/(end-first),
               'prefill_submits':first_counts[0]-before[0],'decode_submits':after[0]-first_counts[0],
               'prefill_tasks':first_counts[1]-before[1],'decode_tasks':after[1]-first_counts[1],
               'irq_delta':[b-a for a,b in zip(irq_before,irq_after)]}
        case['trials'].append(row)
        print(json.dumps({'event':'run','length':length,**row}),flush=True)
        if expected is None: expected = generated
        if generated!=expected: raise AssertionError(f'Output differs at prompt length {length}: {generated} != {expected}')
        case['generated_ids'] = generated
        case['native'] = {name:{'runs':getattr(jit.captured,'native_runs',0),'reason':getattr(jit.captured,'reason',None)}
                          for name,jit in [('prefill',prefill),('decode',transformer.forward_jit)] if jit is not None}
        if args.optimized:
          case['bulk_native_tokens'] = getattr(transformer.rockchip_gpt2.get('loop'),'tokens',0)
          native = getattr(transformer.forward_jit.captured,'native',None)
          case['cache_sites'] = {'direct_to':native.clean_sites,'driver_from':native.driver_from_sites} if native is not None else None
          pool = transformer.rockchip_gpt2['decode'].pool
          case['pool_jobs'] = pool.lib.pool_calls() if pool is not None else 0
          case['native']['prefill_variants'] = [{'shape':str(key),'runs':getattr(jit.captured,'native_runs',0),
            'reason':getattr(jit.captured,'reason',None)} for key,jit in transformer.rockchip_gpt2['prefill'].variants.items()]
          jits = [('decode',transformer.forward_jit),*((str(key),jit) for key,jit in transformer.rockchip_gpt2['prefill'].variants.items())]
          case['profile'] = {name:[{'name':label,'ms_per_call':value/1e6/jit.captured.native.profile_counts[label],
                                   'calls':jit.captured.native.profile_counts[label]}
                                  for label,value in jit.captured.native.profile_ns.most_common()]
                             for name,jit in jits if getattr(jit.captured,'native',None) is not None and jit.captured.native.profiling}
        save()
      rows = [r for r in case['trials'] if not r['warmup']]
      case.update(prefill_tps=statistics.median(r['prefill_tps'] for r in rows),decode_tps=statistics.median(r['decode_tps'] for r in rows),
                  matches_reference=reference is not None)
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()


if __name__=='__main__': main()
