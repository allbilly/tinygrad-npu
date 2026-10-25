"""Full-cost ordinary versus trained DFlash2 greedy decoding on identical cold requests.

Paired mode runs separate processes in ABBA order, so ordinary memory excludes
draft residency. Loading, graph warmup and explicit state clearing are excluded;
all request work, target verification, rejection replay and iterator cleanup are
included. This is an ordinary oNPU comparison, never RKLLM quality certification.
Production workers must be unloaded before running either hardware engine.
"""
from __future__ import annotations
import argparse, dataclasses, datetime, hashlib, itertools, json, math, os, platform, resource, signal, statistics, subprocess, sys, time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

SPECULATIVE_FLAGS = ('prepared-target','prepared-dynamic-rows','prepared-prefill-dynamic','prepared-exact-replay',
                     'prepared-draft','prepared-observation','draft-mali-linear','early-anchor')


def _speculative_options(args) -> dict[str,Any]:
  return {**{name.replace('-','_'):getattr(args,name.replace('-','_')) for name in SPECULATIVE_FLAGS},
          'verification_budget':args.verification_budget}


def _prepare_speculation(args, target, report:dict):
  preparation:dict[str,Any] = {'engine':args.engine,'draft_loaded':False,'draft_mali':None,'draft_proposals':None,
                               'draft_observation':None,'decoder':None}
  report['preparation'] = preparation
  if args.engine!='speculative':return None,None
  from tinygrad import dtypes
  from tinygrad.llm.dflash2_config import load_dflash2
  from tinygrad.llm.dflash2 import DFlash2
  from tinygrad.llm.dflash2_selector import CandidateSelector
  from tinygrad.llm.speculative import GreedySpeculativeDecoder
  config,weights = load_dflash2(args.draft,target.config,target_checkpoint=args.checkpoint,device=args.device,
                               dtype=dtypes.float16 if args.device=='CL' else dtypes.float32)
  draft = DFlash2(config,weights,target.token_embd,target.output,target.output_norm,CandidateSelector(config,weights))
  preparation['draft_loaded'] = True
  if args.draft_mali_linear:preparation['draft_mali'] = draft.prepare_mali(pack_weights=False,prepare_shared_head=True)
  if args.prepared_draft:draft.prepare_proposals(max_context=args.context)
  if args.prepared_observation:draft.prepare_observation(max_context=args.context)
  preparation['draft_proposals'] = draft.proposal_preparation_metadata()
  preparation['draft_observation'] = draft.observation_preparation_metadata()
  decoder = GreedySpeculativeDecoder(target,draft,prepared_target=args.prepared_target,prepared_dynamic_rows=args.prepared_dynamic_rows,
    prepared_prefill_dynamic=args.prepared_prefill_dynamic,prepared_exact_replay=args.prepared_exact_replay,early_anchor=args.early_anchor)
  preparation['decoder'] = {key:value for key,value in dataclasses.asdict(decoder.stats).items()
                            if key=='early_anchor' or key.startswith('prepared_')}
  report['draft_config'] = dataclasses.asdict(config)
  return draft,decoder


def _generation(args, target, decoder, prompt:list[int]) -> Iterator[int]:
  if decoder is None:return _ordinary(target,prompt,args.tokens,args.chunk_size)
  return decoder.generate(prompt,max_new_tokens=args.tokens,prefill_chunk_size=args.chunk_size,
                          verification_budget=args.verification_budget)


def _child_command(args, engine:str, result:Path) -> list[str]:
  command = [sys.executable,'-m','extra.rockchip.benchmark_speculative','--engine',engine]
  for name in ('checkpoint','draft','input','device','context','tokens','warmups','runs','chunk-size','server','timeout'):
    command += ['--'+name,str(getattr(args,name.replace('-','_')))]
  command += ['--report',str(result)]
  for name in ('gpu-recurrent-scan','gpu-no-weight-packing','gpu-q8-projections',*SPECULATIVE_FLAGS):
    if getattr(args,name.replace('-','_')):command.append('--'+name)
  if args.verification_budget is not None:command += ['--verification-budget',str(args.verification_budget)]
  return command


def digest(path:Path) -> str:
  value = hashlib.sha256()
  with path.open('rb') as source:
    for chunk in iter(lambda:source.read(8<<20),b''):value.update(chunk)
  return value.hexdigest()


def source_snapshot(root:Path) -> dict[str,str]:
  files = sorted(p for p in (root/'tinygrad').rglob('*') if p.is_file() and p.suffix in ('.py','.c','.h'))
  return {str(p.relative_to(root)):digest(p) for p in [*files,Path(__file__).resolve()]}


def clocks() -> dict[str,str]:
  paths = [*Path('/sys/class/devfreq').glob('*/cur_freq'),
           *Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'),
           *Path('/sys/class/thermal').glob('thermal_zone*/temp')]
  result = {}
  for path in paths:
    try:result[str(path)] = path.read_text().strip()
    except OSError:pass
  return result


def consume_generation(factory:Callable[[],Iterator[int]], output_tokens:int, synchronize:Callable[[],None],
                       timer:Callable[[],float]=time.perf_counter) -> dict[str,Any]:
  """Time the actual iterator through exhaustion and cleanup, including final sync."""
  synchronize()
  ids, stamps = [],[]
  begin = timer()
  iterator = factory()
  try:
    for token in iterator:
      if type(token) is not int or token<0:raise ValueError('Generation returned an invalid token ID')
      ids.append(token)
      stamps.append(timer())
      if len(ids)>output_tokens:raise ValueError('Generation exceeded the fixed output count')
  finally:
    try:
      if (close:=getattr(iterator,'close',None)) is not None:close()
    finally:
      synchronize()
      end = timer()
  if len(ids)!=output_tokens or output_tokens<2:raise ValueError('Generation did not complete the fixed output count')
  first, delivered, total = stamps[0]-begin,stamps[-1]-stamps[0],end-begin
  if any(not math.isfinite(x) or x<=0 for x in (first,delivered,total,end-stamps[0])):
    raise ValueError('Invalid generation timings')
  return {'generated_ids':ids,'first_token_ms':1000*first,'total_seconds':total,'output_tps':output_tokens/total,
          'decode_tps':(output_tokens-1)/delivered,'decode_full_cost_tps':(output_tokens-1)/(end-stamps[0]),
          'cleanup_seconds':end-stamps[-1]}


def summarize(rows:Sequence[dict], *, prompt_tokens:int, output_tokens:int, context:int) -> dict[str,Any]:
  if prompt_tokens<1 or output_tokens<2 or prompt_tokens+output_tokens>context:raise ValueError('Invalid fixed workload')
  results, reference_ids = {},None
  for engine in ('ordinary','speculative'):
    group = [row for row in rows if row.get('engine')==engine]
    measured = [row for row in group if row.get('warmup') is False]
    if len(measured)<3 or sum(row.get('warmup') is True for row in group)<3:
      raise ValueError('Need at least three full warmups and three measured runs per engine')
    for row in group:
      if (row.get('context')!=context or row.get('prompt_tokens')!=prompt_tokens
          or len(row.get('generated_ids',[]))!=output_tokens or row.get('state_reset') is not True):
        raise ValueError('Different workload or missing cold-request reset')
      if reference_ids is None:reference_ids = row['generated_ids']
      if row['generated_ids']!=reference_ids:raise ValueError('Ordinary and speculative greedy token IDs differ')
      for key in ('first_token_ms','total_seconds','output_tps','decode_full_cost_tps'):
        if not math.isfinite(row[key]) or row[key]<=0:raise ValueError('Invalid measured timings')
      if engine=='speculative':
        stats = row.get('speculative_stats',{})
        proposed,accepted = stats.get('proposed_tokens',-1),stats.get('accepted_tokens',-1)
        if not 0<=accepted<=proposed or stats.get('output_tokens')!=output_tokens:
          raise ValueError('Invalid speculative acceptance counters')
    fields = ('first_token_ms','total_seconds','output_tps','decode_full_cost_tps')
    results[engine] = {'runs':len(measured),**{'median_'+key:statistics.median(row[key] for row in measured) for key in fields},
                       'output_tps_range':[min(row['output_tps'] for row in measured),max(row['output_tps'] for row in measured)]}
    if engine=='speculative':
      proposed = sum(row['speculative_stats']['proposed_tokens'] for row in measured)
      accepted = sum(row['speculative_stats']['accepted_tokens'] for row in measured)
      results[engine].update(acceptance_rate=accepted/proposed if proposed else None,
                            accepted_draft_tps=accepted/sum(row['total_seconds'] for row in measured))
  ordinary, speculative = results['ordinary'],results['speculative']
  return {'engines':results,'greedy_ids_equal':True,
          'full_request_speedup':speculative['median_output_tps']/ordinary['median_output_tps'],
          'full_cost_decode_speedup':speculative['median_decode_full_cost_tps']/ordinary['median_decode_full_cost_tps'],
          'ttft_speedup':ordinary['median_first_token_ms']/speculative['median_first_token_ms'],
          'output_tps_ranges_separated':speculative['output_tps_range'][0]>ordinary['output_tps_range'][1],
          'qualified_rkllm_win':False,'quality_scope':'Exact greedy IDs on this request; no SDK output-quality certification'}


def reset_target_state(target, draft=None) -> list[str]:
  """Clear every supported target cache before a request, retaining warmed graphs."""
  from tinygrad import Device, Tensor
  target._cached_tokens = []
  target._greedy_generation = False
  states,seen = [],set()
  for index,block in enumerate(target.blk):
    for owner,prefix in ((block,f'blk.{index}'),(getattr(block,'shortconv',None),f'blk.{index}.shortconv')):
      for name in ('cache_kv','cache_k','cache_kg','conv_state','recurrent_state','cache_conv'):
        if isinstance(value:=getattr(owner,name,None),Tensor) and id(value) not in seen:
          seen.add(id(value))
          states.append((prefix+'.'+name,value))
  if states:Tensor.realize(*(state.assign(Tensor.zeros_like(state)) for _,state in states))
  for device in {d for _,state in states for d in ((state.device,) if isinstance(state.device,str) else state.device)}:
    Device[device].synchronize()
  if draft is not None:draft.reset()
  return [name for name,_ in states]


def _ordinary(target, prompt:list[int], count:int, chunk_size:int) -> Iterator[int]:
  iterator = target.generate(prompt.copy(),chunk_size=chunk_size,temperature=0)
  try:yield from itertools.islice(iterator,count)
  finally:iterator.close()


def _idle_server(url:str) -> None:
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(url.rstrip('/')+'/api/status',timeout=10) as response:status = json.load(response)
  if status['runtime']['sessions']:raise RuntimeError('Unload production workers before exclusive hardware measurements')


def _run_child(command:list[str], log:Path, root:Path, timeout:float) -> None:
  with log.open('x') as output:
    process = subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    try:code = process.wait(timeout=timeout)
    except BaseException:
      try:os.killpg(process.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      process.wait()
      raise
  if code:raise RuntimeError(f'Benchmark exited {code}; see {log}')


def _engine(args, report:dict, save:Callable[[],None]) -> None:
  os.environ['DEV'] = args.device
  from tinygrad import Device, Tensor
  from tinygrad.helpers import GlobalCounters
  from tinygrad.llm.model import Transformer
  target,_ = Transformer.from_gguf(args.checkpoint,max_context=args.context,realize=True)
  if any(token>=target.config.vocab_size for token in report['input_ids']):raise ValueError('Prompt token exceeds target vocabulary')
  kernels = None
  if args.device=='CL':
    from tinygrad.llm.kernels.mali import configure
    kernels = configure(target,recurrent_scan=args.gpu_recurrent_scan,pack_weights=not args.gpu_no_weight_packing,
                        q8_projections_checkpoint=args.checkpoint if args.gpu_q8_projections else None)
    if args.gpu_q8_projections and not kernels.get('q8_projection_layers'):
      raise ValueError('Requested native Q8 body projections did not activate')
  draft,decoder = _prepare_speculation(args,target,report)
  Device[args.device].allocator.free_cache()
  report.update(gpu_kernels=kernels,load_seconds=time.perf_counter()-report.pop('_load_started'))
  save()
  for index in range(args.warmups+args.runs):
    reset_start = time.perf_counter()
    cleared = reset_target_state(target,draft)
    reset_seconds = time.perf_counter()-reset_start
    Tensor.manual_seed(0)
    counters = {key:getattr(GlobalCounters,key) for key in ('kernel_count','global_ops','global_mem','time_sum_s')}
    memory = GlobalCounters.mem_used
    before = clocks()
    def factory():return _generation(args,target,decoder,report['input_ids'])
    row = consume_generation(factory,args.tokens,Device[args.device].synchronize)
    row.update(engine=args.engine,index=index,warmup=index<args.warmups,context=args.context,prompt_tokens=len(report['input_ids']),
               state_reset=True,cleared_state=cleared,state_reset_seconds=reset_seconds,prefill_proxy_tps=1000*len(report['input_ids'])/row['first_token_ms'],
               clocks_before=before,clocks_after=clocks(),allocated_bytes_before=memory,allocated_bytes_after=GlobalCounters.mem_used,
               lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               hardware_counters={key:getattr(GlobalCounters,key)-value for key,value in counters.items()})
    if decoder is not None:
      row.update(speculative_stats=dataclasses.asdict(decoder.stats),finish_reason=decoder.finish_reason)
      row['accepted_draft_tps'] = decoder.stats.accepted_tokens/row['total_seconds']
      row['draft_preparation'] = draft.proposal_preparation_metadata()
      report['preparation']['draft_proposals'] = row['draft_preparation']
      row['draft_observation'] = draft.observation_preparation_metadata()
      report['preparation']['draft_observation_final'] = row['draft_observation']
    report['trials'].append(row)
    save()
    print(json.dumps({'event':'run',**row}),flush=True)


def _argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('checkpoint','draft','input','report'):parser.add_argument('--'+name,type=Path,required=True)
  parser.add_argument('--engine',choices=('paired','ordinary','speculative'),default='paired')
  parser.add_argument('--device',choices=('CPU','CL'),default='CL')
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--tokens',type=int,default=64)
  parser.add_argument('--warmups',type=int,default=3)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--chunk-size',type=int,default=128)
  parser.add_argument('--gpu-recurrent-scan',action='store_true')
  parser.add_argument('--gpu-no-weight-packing',action='store_true')
  parser.add_argument('--gpu-q8-projections',action='store_true')
  for name in SPECULATIVE_FLAGS:parser.add_argument('--'+name,action='store_true')
  parser.add_argument('--verification-budget',type=int)
  parser.add_argument('--server',default='http://127.0.0.1:8091')
  parser.add_argument('--timeout',type=float,default=3600)
  return parser


def _parse_args(argv:Sequence[str]|None=None):
  parser = _argument_parser()
  args = parser.parse_args(argv)
  if any((args.prepared_dynamic_rows,args.prepared_prefill_dynamic,args.prepared_exact_replay)) and not args.prepared_target:
    parser.error('Prepared row/prefill/replay options require --prepared-target')
  if args.prepared_exact_replay and not args.prepared_dynamic_rows:
    parser.error('--prepared-exact-replay requires --prepared-dynamic-rows')
  if args.prepared_draft and not 1<=args.context<=512:parser.error('--prepared-draft requires context1..512')
  if args.prepared_observation and not 1<=args.context<=512:parser.error('--prepared-observation requires context1..512')
  if args.draft_mali_linear and args.device!='CL':parser.error('--draft-mali-linear requires CL')
  if args.verification_budget is not None and not 1<=args.verification_budget<=7:parser.error('--verification-budget must be in 1..7')
  if args.warmups<3 or args.runs<3:parser.error('Need at least three full warmups and three measured runs')
  if not math.isfinite(args.timeout) or args.timeout<=0:parser.error('Timeout must be finite and positive')
  if args.device!='CL' and any((args.gpu_recurrent_scan,args.gpu_no_weight_packing,args.gpu_q8_projections)):
    parser.error('GPU kernel options require CL')
  return args


def main() -> None:
  args = _parse_args()
  for name in ('checkpoint','draft','input','report'):setattr(args,name,getattr(args,name).resolve())
  ids = [int(x) for x in args.input.read_text().split()]
  if not ids or min(ids)<0 or args.tokens<2 or len(ids)+args.tokens>args.context or not 1<=args.chunk_size<=256:
    _argument_parser().error('Need nonempty prompt, >=2 outputs, chunk1..256 and sufficient context')
  _idle_server(args.server)
  root = Path(__file__).resolve().parents[2]
  snapshot = source_snapshot(root)
  report = {'schema_version':1,'complete':False,'engine':args.engine,'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),
            'host':platform.node(),'device':args.device,'context':args.context,'input_ids':ids,'input_sha256':digest(args.input),
            'output_tokens':args.tokens,'warmups':args.warmups,'runs':args.runs,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint),
            'draft':str(args.draft.resolve()),'draft_config_sha256':digest(args.draft/'config.json'),
            'draft_weights_sha256':digest(args.draft/'model.safetensors'),'source_sha256':snapshot,'trials':[],
            'configuration':{'chunk_size':args.chunk_size,'gpu_recurrent_scan':args.gpu_recurrent_scan,
                             'gpu_pack_weights':not args.gpu_no_weight_packing,'gpu_q8_projections':args.gpu_q8_projections,
                             'draft_compute_dtype':'float16' if args.device=='CL' else 'float32','npu_assistance':'off',
                             **_speculative_options(args)},
            'protocol':{'cold_request':True,'eos':'ignored; fixed output count','sampling':'greedy; temperature0',
                        'timing':'Full iterator consumption, cleanup and final device synchronization; loading/warmup/state clearing excluded',
                        'memory':'Per-engine process lifetime peak RSS includes loading and compilation; device allocation is before/after, not peak',
                        'accepted_draft_tps':'Accepted proposed draft tokens divided by full request wall time',
                        'prefill_proxy':('Prompt tokens / time to first yielded target token; early-anchor mode defers draft observation '
                                         'and verification until after that yield. Full request timing includes all work.'),
                        'cache_semantics':('Speculation feeds the final delivered token; ordinary generation leaves it unconsumed. '
                                           'Both actual costs are included.'),
                        'clock_policy':'unchanged host policy',
                        'configuration_scope':'Requested common configuration; draft/decoder options apply only to speculative engines. '
                                              'Actual preparation is recorded per engine/cohort.'},'qualified_rkllm_win':False}
  with args.report.open('x') as output:output.write(json.dumps(report,indent=2)+'\n')
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  try:
    from tinygrad.llm.dflash2_config import DFLASH2_CONFIG_SHA256, DFLASH2_WEIGHTS_SHA256, DFLASH2_TARGET_SHA256, DFLASH2_REVISION
    if (report['checkpoint_sha256'],report['draft_config_sha256'],report['draft_weights_sha256']) != \
       (DFLASH2_TARGET_SHA256,DFLASH2_CONFIG_SHA256,DFLASH2_WEIGHTS_SHA256):
      raise ValueError('Expected the pinned trained DFlash2 draft and target checkpoint pair')
    report['draft_revision'] = DFLASH2_REVISION
    if args.engine=='paired':
      report['cohorts'] = []
      for index,engine in enumerate(('ordinary','speculative','speculative','ordinary')):
        result = args.report.with_name(args.report.stem+f'.{index}-{engine}.json')
        log = result.with_suffix('.log')
        command = _child_command(args,engine,result)
        _run_child(command,log,root,args.timeout)
        data = json.loads(result.read_text())
        for key in ('source_sha256','checkpoint_sha256','draft_config_sha256','draft_weights_sha256','input_ids','context','device',
                    'configuration','cpu_affinity'):
          if data[key]!=report[key]:raise ValueError('Different paired provenance: '+key)
        if not data['complete']:raise ValueError('Incomplete engine report')
        report['cohorts'].append({'engine':engine,'report':str(result),'report_sha256':digest(result),'command':command,
                                  'preparation':data['preparation']})
        report['trials'] += data['trials']
        if source_snapshot(root)!=snapshot:raise RuntimeError('Sources changed during benchmark')
        save()
      report['summary'] = summarize(report['trials'],prompt_tokens=len(ids),output_tokens=args.tokens,context=args.context)
    else:
      report['_load_started'] = time.perf_counter()
      _engine(args,report,save)
    if source_snapshot(root)!=snapshot:raise RuntimeError('Sources changed during benchmark')
    report['complete'] = True
  except BaseException as error:
    report.pop('_load_started',None)
    report['error'] = f'{type(error).__name__}: {error}'
    raise
  finally:save()


if __name__=='__main__':main()
