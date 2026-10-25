"""Controlled target-prefill ablations with the trained DFlash2 draft resident.

Run only under the operator's exclusive GPU lease. The primary oNPU GPU worker
must be unloaded; ready, inactive CPU/NPU workers may remain resident. No server
or model settings are changed by this tool. Loading, reset, numerical validation
and warmup are separate from the synchronized first-token measurements.
"""
from __future__ import annotations
import argparse, dataclasses, datetime, hashlib, json, math, os, platform, resource, statistics, time, urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from extra.rockchip.benchmark_speculative import clocks, digest, reset_target_state, source_snapshot


MODES = ('ordinary','dynamic-last','dynamic-none','dynamic-taps','eager-none','eager-taps','prepared-taps','selective-taps')
COUNTERS = ('kernel_count','global_ops','global_mem','time_sum_s')


def read_prompt(path:Path) -> list[int]:
  """Accept a JSON token list, or a report object with input_ids/token_ids."""
  data = json.loads(path.read_text())
  if isinstance(data,dict):data = data.get('input_ids',data.get('token_ids'))
  if not isinstance(data,list) or not data or any(type(token) is not int or token<0 for token in data):
    raise ValueError('Prompt JSON must contain a nonempty integer token list, input_ids, or token_ids')
  return data


def idle_gpu(server:str) -> list[dict]:
  """Reject GPU owners and active/preparing workers; permit idle NPU residency."""
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(server.rstrip('/')+'/api/status',timeout=10) as response:data = json.load(response)
  runtime = data.get('runtime')
  if not isinstance(runtime,dict) or not isinstance(runtime.get('sessions'),list):raise ValueError('Unrecognized oNPU runtime status')
  sessions = runtime['sessions']
  for session in sessions:
    if not isinstance(session,dict):raise ValueError('Unrecognized oNPU session')
    profile = str(session.get('selected_profile','')).upper()
    devices = ' '.join(str(session.get(key,'')) for key in ('actual_device','base_device')).upper()
    vision = (session.get('placement') or {}).get('vision') or {}
    if profile=='CL' or 'CL' in devices or vision.get('gpu_vision'):
      raise RuntimeError('Unload the primary GPU worker before profiling; this tool never unloads workers')
    if session.get('active_request') or session.get('state') in ('loading','preparing','busy','generating'):
      raise RuntimeError('Other resident workers must be ready and inactive during controlled profiling')
    if session.get('state')!='ready':raise RuntimeError('Other resident workers must be ready and inactive during controlled profiling')
  return sessions


def session_identity(sessions:Sequence[dict]) -> list[tuple]:
  return sorted((str(row.get('model_id')),str(row.get('selected_profile')),str(row.get('actual_device'))) for row in sessions)


def snapshot(root:Path) -> dict[str,str]:
  return {**source_snapshot(root),str(Path(__file__).resolve().relative_to(root)):digest(Path(__file__).resolve())}


def _rss() -> dict[str,int]:
  current = next((int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')),0)
  return {'rss_bytes':current,'lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def _array_record(value) -> dict:
  import numpy as np
  array = np.ascontiguousarray(value)
  return {'shape':list(array.shape),'dtype':str(array.dtype),'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
          'finite':bool(np.isfinite(array).all()),'max_abs':float(np.max(np.abs(array))) if array.size else 0.0}


def feature_array(value,valid_rows:int):
  """Read an already-materialized feature through a fixed-shape buffer.

  A bound symbolic row count is still symbolic to Tensor.numpy(). Padding makes
  its exported shape static; slice on the host so padding is never validated as
  real model output. This diagnostic readback remains outside timed intervals.
  """
  if value.ndim!=3 or type(valid_rows) is not int or not 0<valid_rows<=value.max_shape[1]:
    raise ValueError('Feature readback needs a valid nonempty row count')
  return value.float().pad_to(value.max_shape).contiguous().realize().numpy()[:,:valid_rows].copy()


def _state_arrays(target,prompt_tokens:int) -> dict:
  """Read complete recurrent state and only valid positional KV rows."""
  from tinygrad import Tensor
  result,seen = {},set()
  for index,block in enumerate(target.blk):
    for owner,prefix in ((block,f'blk.{index}'),(getattr(block,'shortconv',None),f'blk.{index}.shortconv')):
      for name in ('cache_kv','cache_k','cache_kg','conv_state','recurrent_state','cache_conv'):
        if not isinstance(value:=getattr(owner,name,None),Tensor) or id(value) in seen:continue
        seen.add(id(value))
        if name=='cache_kv':
          if value.ndim!=5:raise ValueError('Unknown cache_kv layout; refusing to compare invalid padding')
          value=value[:,:,:,:prompt_tokens]
        elif name in ('cache_k','cache_kg'):
          raise ValueError('The pinned Qwen3.5 target must use the known cache_kv layout')
        result[prefix+'.'+name]=value.float().numpy().copy()
  if not result:raise ValueError('No numerical target state found after prefill')
  return result


def _compare_arrays(actual:dict, reference:dict, *, atol:float, rtol:float) -> dict:
  import numpy as np
  if actual.keys()!=reference.keys():raise ValueError('Different numerical state/feature names')
  records = {}
  for name,value in actual.items():
    wanted=reference[name]
    if value.shape!=wanted.shape or not np.isfinite(value).all() or not np.isfinite(wanted).all():
      raise ValueError('Invalid numerical state/feature: '+name)
    error=np.abs(value-wanted)
    record={**_array_record(value),'bitwise_equal':bool(np.array_equal(value.view(np.uint8),wanted.view(np.uint8))),
            'max_abs_error':float(error.max()) if error.size else 0.0,
            'max_relative_error':float((error/np.maximum(np.abs(wanted),1e-12)).max()) if error.size else 0.0,
            'allclose':bool(np.all(error<=atol+rtol*np.abs(wanted)))}
    records[name]=record
  return {'passed':all(row['allclose'] for row in records.values()),'atol':atol,'rtol':rtol,'arrays':records}


def _geometry(mode:str,prompt_tokens:int,context:int,chunk_size:int,capture_layers:tuple[int,...]) -> dict:
  dynamic=mode=='ordinary' or mode.startswith('dynamic-')
  return {'geometry_group':'production-dynamic' if dynamic else 'exact-static',
          'input_ids_device':'CL' if dynamic else 'CPU','actual_rows':prompt_tokens,
          'input_capacity':context if dynamic else prompt_tokens,
          'row_bounds':[1,chunk_size] if dynamic else [prompt_tokens,prompt_tokens],
          'position':0,'position_bounds':[0,context-1 if dynamic else context-prompt_tokens],
          'position_variable':('start_pos' if dynamic else f'spec_target_pos_prefill_{prompt_tokens}_1'
                               if mode=='prepared-taps' else 'profile_exact_prefill_position'),
          'row_variable':'toks' if dynamic else f'spec_target_rows_prefill_{prompt_tokens}_1' if mode=='prepared-taps' else None,
          'capture_layers':list(capture_layers) if mode in ('dynamic-taps','eager-taps','prepared-taps','selective-taps') else [],
          'last_only':True,'full_final_hidden_materialized':mode in ('dynamic-none','dynamic-taps','eager-none','eager-taps','prepared-taps'),
          'path':('production Transformer.generate' if mode=='ordinary' else 'private selective materialization diagnostic'
                  if mode in ('dynamic-last','selective-taps') else 'exact-row _PreparedTargetForward'
                  if mode=='prepared-taps' else 'eager target_forward'),
          'geometry_note':('Production-matched controls create a fresh full CL buffer, slice the same bound position/row variables '
                           'and make the view contiguous; exact controls use CPU inputs with fixed rows and matched position bounds.')}


def _summarize(rows:Sequence[dict]) -> dict:
  result = {}
  for mode in dict.fromkeys(row['mode'] for row in rows):
    selected=[row for row in rows if row['mode']==mode and not row['warmup']]
    if not selected:continue
    result[mode]={'runs':len(selected),
                  'median_first_token_seconds':statistics.median(row['wall_seconds'] for row in selected),
                  'median_target_materialization_seconds':statistics.median(row['target_materialization_seconds'] for row in selected),
                  'median_first_token_tps':statistics.median(row['prefill_proxy_tps'] for row in selected),
                  'median_counter_delta':{name:statistics.median(row['hardware_counters'][name] for row in selected) for name in COUNTERS},
                  'first_token_seconds_range':[min(row['wall_seconds'] for row in selected),max(row['wall_seconds'] for row in selected)],
                  'geometry_group':selected[0]['geometry_group'],
                  'matched_reference':selected[0]['matched_reference'],
                  'all_matched_state_checks_pass':all(row['matched_state_comparison']['passed'] for row in selected),
                  'all_matched_ids_equal':all(row['matched_ids_equal'] for row in selected),
                  'all_ordinary_state_checks_pass':all(row['ordinary_state_comparison']['passed'] for row in selected),
                  'all_ordinary_ids_equal':all(row['ordinary_ids_equal'] for row in selected)}
  return result


def _profile(args,report:dict,save:Callable[[],None],check:Callable[[],list[dict]]) -> None:
  os.environ['DEV']='CL'
  from tinygrad import Device, Tensor, UOp, dtypes
  from tinygrad.helpers import Context, GlobalCounters
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.dflash2_config import load_dflash2
  from tinygrad.llm.dflash2 import DFlash2
  from tinygrad.llm.dflash2_selector import CandidateSelector
  from tinygrad.llm.kernels.mali import configure
  from tinygrad.llm.speculative import _PreparedTargetForward
  start=time.perf_counter()
  target,_=Transformer.from_gguf(args.checkpoint,max_context=args.context,realize=True)
  kernels=configure(target,recurrent_scan=True,pack_weights=not args.gpu_no_weight_packing,
                    q8_projections_checkpoint=args.checkpoint if args.gpu_q8_projections else None)
  if not kernels.get('fused_gated_delta_blocks'):raise ValueError('Pinned bulk-prefill recurrent kernels did not activate')
  if args.gpu_q8_projections and not kernels.get('q8_projection_layers'):raise ValueError('Requested Q8 body projections did not activate')
  report['target_load_seconds']=time.perf_counter()-start
  start=time.perf_counter()
  config,weights=load_dflash2(args.draft,target.config,target_checkpoint=args.checkpoint,device='CL',dtype=dtypes.float16)
  draft=DFlash2(config,weights,target.token_embd,target.output,target.output_norm,CandidateSelector(config,weights))
  draft_preparation=draft.prepare_mali(pack_weights=False) if args.prepare_draft else None
  Device['CL'].synchronize()
  report.update(draft_load_seconds=time.perf_counter()-start,draft_config=dataclasses.asdict(config),gpu_kernels=kernels,
                draft_preparation=draft_preparation,resident_after_loading={**_rss(),'allocated_bytes':GlobalCounters.mem_used},
                capture_layers=list(config.capture_layers))
  ids=report['input_ids']
  if any(token>=target.config.vocab_size for token in ids):raise ValueError('Prompt token exceeds target vocabulary')
  if target.prefill_chunk_size(args.chunk_size)!=args.chunk_size:raise ValueError('Requested production prefill chunk is unavailable')
  prompt_tokens=len(ids)
  tokens=Tensor([ids],device='CPU',dtype=dtypes.int32).realize()
  position=UOp.variable('profile_exact_prefill_position',0,args.context-prompt_tokens).bind(0)
  prepared=_PreparedTargetForward(target,config.capture_layers,dynamic_rows=False,max_graphs=2,max_prefill_graphs=1)
  reference_states,reference_ids,reference_features={},{},{}
  def ordinary():
    iterator=target.generate(ids.copy(),chunk_size=args.chunk_size,temperature=0)
    try:return {'token':next(iterator),'logits':None,'features':()}
    finally:iterator.close()
  def dynamic_inputs():
    # Match Transformer.generate exactly: the default-device full context buffer
    # is CL under DEV=CL, and __call__ makes this symbolic view contiguous.
    full=Tensor(ids+[0]*(args.context-prompt_tokens),device='CL',dtype=dtypes.int32).reshape(1,args.context)
    start=UOp.variable('start_pos',0,args.context-1).bind(0)
    rows=UOp.variable('toks',1,args.chunk_size).bind(prompt_tokens)
    return full[:,start:start+rows].contiguous(),start
  def eager(captures,dynamic=False):
    input_tokens,input_position=dynamic_inputs() if dynamic else (tokens,position)
    output=target.target_forward(input_tokens,input_position,capture_layers=captures,last_only=True)
    return {'token':None,'logits':output.logits,'features':output.features}
  def prepared_forward():
    output=prepared(ids,0,'prefill',prompt_tokens,True)
    return {'token':None,'logits':output.logits,'features':output.features}
  def selective(dynamic=False):
    # Diagnostic only. Keep the production TargetOutput realization contract
    # unchanged while testing whether full final-hidden materialization matters.
    input_tokens,input_position=dynamic_inputs() if dynamic else (tokens,position)
    embedded=target.token_embd(input_tokens.to(target.token_embd.weight.device)).float()*target.config.embedding_scale
    hidden,features=target._hidden_from_embeddings(embedded,input_position,() if dynamic else config.capture_layers)
    logits=target._project_logits(hidden[:,-1:])
    unique={id(value):value for value in (logits,*features)}
    Tensor.realize(*unique.values())
    return {'token':None,'logits':logits,'features':features}
  runners={'ordinary':ordinary,'dynamic-last':lambda:selective(True),'dynamic-none':lambda:eager((),True),
           'dynamic-taps':lambda:eager(config.capture_layers,True),'eager-none':lambda:eager(()),'eager-taps':lambda:eager(config.capture_layers),
           'prepared-taps':prepared_forward,'selective-taps':selective}
  modes=list(MODES if args.include_selective_control else MODES[:-1])
  report['modes']={mode:_geometry(mode,prompt_tokens,args.context,args.chunk_size,config.capture_layers) for mode in modes}
  report['rotation']=[modes[index%len(modes):]+modes[:index%len(modes)] for index in range(args.warmups+args.runs)]
  save()
  for round_index,order in enumerate(report['rotation']):
    for order_index,mode in enumerate(order):
      sessions=check()
      reset_started=time.perf_counter()
      cleared=reset_target_state(target,draft)
      Device['CL'].synchronize()
      reset_seconds=time.perf_counter()-reset_started
      Tensor.manual_seed(0)
      before={name:getattr(GlobalCounters,name) for name in COUNTERS}
      allocated=GlobalCounters.mem_used
      clock_before=clocks()
      with Context(DEBUG=2 if args.kernel_timing else 0):
        begin=time.perf_counter()
        output=runners[mode]()
        Device['CL'].synchronize()
        materialized=time.perf_counter()
        forward_counters={name:getattr(GlobalCounters,name)-before[name] for name in COUNTERS}
        if output['logits'] is not None:
          logits=output['logits']
          if logits.shape!=(1,1,target.config.vocab_size):raise ValueError('Expected exactly one projected output row')
          prediction=logits.isfinite().all(-1).where(logits.argmax(-1),target.config.vocab_size)
          output['token']=int(prediction.item())
        Device['CL'].synchronize()
        end=time.perf_counter()
      total_counters={name:getattr(GlobalCounters,name)-before[name] for name in COUNTERS}
      token=output['token']
      if type(token) is not int or not 0<=token<target.config.vocab_size:raise ValueError('Invalid/nonfinite greedy output')
      elapsed=end-begin
      row={'mode':mode,'round':round_index,'order_index':order_index,'warmup':round_index<args.warmups,
           'wall_seconds':elapsed,'target_materialization_seconds':materialized-begin,'greedy_extraction_seconds':end-materialized,
           'first_token_id':token,'prefill_proxy_tps':prompt_tokens/elapsed,'hardware_counters':total_counters,
           'target_materialization_counters':forward_counters,'allocated_bytes_before':allocated,
           'allocated_bytes_after':GlobalCounters.mem_used,'allocated_bytes_per_device':dict(GlobalCounters.mem_used_per_device),
           'resident_after_timing':_rss(),'clocks_before':clock_before,'clocks_after':clocks(),
           'state_reset_seconds':reset_seconds,'cleared_state':cleared,'state_reset':True,'server_sessions':sessions,
           'cached_token_count_after_prefill':len(target._cached_tokens),'prepared_graphs':len(prepared.calls),
           'ordinary_prefill_jit_calls':target.greedy_prefill_jit.cnt,
           'prepared_graph_calls':[call.jit.cnt for call in prepared.calls.values()]}
      # Readbacks and continuation checks are outside every timing/counter
      # interval. Report cross-geometry differences separately, without treating
      # them as a matched reference or aborting the remaining ablations.
      validate_started=time.perf_counter()
      state=_state_arrays(target,prompt_tokens)
      group=report['modes'][mode]['geometry_group']
      reference_mode='ordinary' if group=='production-dynamic' else 'eager-none'
      if group not in reference_states:
        if mode!=reference_mode:raise ValueError('Missing matched-geometry reference: '+reference_mode)
        reference_states[group]=state
      row.update(geometry_group=group,matched_reference=reference_mode)
      row['matched_state_comparison']=_compare_arrays(state,reference_states[group],atol=args.state_atol,rtol=args.state_rtol)
      row['ordinary_state_comparison']=_compare_arrays(state,reference_states['production-dynamic'],atol=args.state_atol,rtol=args.state_rtol)
      features={f'layer.{layer}':feature_array(value,prompt_tokens) for layer,value in zip(config.capture_layers,output['features'])}
      if mode in ('dynamic-taps','eager-taps') and group not in reference_features:reference_features[group]=features
      if features and group in reference_features:
        row['matched_feature_comparison']=_compare_arrays(features,reference_features[group],atol=args.state_atol,rtol=args.state_rtol)
      generated=[token]
      for index in range(1,args.verify_tokens):
        continuation_position=UOp.variable('profile_validation_position',0,args.context-1).bind(prompt_tokens+index-1)
        result=target.greedy_forward(Tensor([[generated[-1]]],device='CPU',dtype=dtypes.int32),continuation_position).realize()
        generated.append(int(result.item()))
      Device['CL'].synchronize()
      if group not in reference_ids:
        if mode!=reference_mode:raise ValueError('Missing matched-geometry ID reference: '+reference_mode)
        reference_ids[group]=generated
      row.update(validation_generated_ids=generated,matched_ids_equal=generated==reference_ids[group],
                 ordinary_ids_equal=generated==reference_ids['production-dynamic'],
                 validation_seconds=time.perf_counter()-validate_started)
      row['ordinary_equivalent']=(row['ordinary_state_comparison']['passed'] and row['ordinary_ids_equal'])
      row['matched_checks_passed']=(row['matched_state_comparison']['passed'] and row['matched_ids_equal']
                                    and row.get('matched_feature_comparison',{'passed':True})['passed'])
      row['validation_failures']=[name for name,passed in (
        ('matched_state',row['matched_state_comparison']['passed']),('matched_ids',row['matched_ids_equal']),
        ('matched_features',row.get('matched_feature_comparison',{'passed':True})['passed']),
        ('ordinary_state',row['ordinary_state_comparison']['passed']),('ordinary_ids',row['ordinary_ids_equal'])) if not passed]
      report['trials'].append(row)
      save()
      print(json.dumps({'event':'prefill_trial','mode':mode,'round':round_index,'warmup':row['warmup'],
                        'wall_seconds':elapsed,'kernel_count':total_counters['kernel_count'],
                        'ordinary_equivalent':row['ordinary_equivalent'],'validation_failures':row['validation_failures']}),flush=True)
      check()
      del output,state,features
  report['summary']=_summarize(report['trials'])
  report['diagnostic_equivalence']={
    'all_matched_checks_passed':all(row['matched_checks_passed'] for row in report['trials']),
    'all_ordinary_states_match':all(row['ordinary_state_comparison']['passed'] for row in report['trials']),
    'all_ordinary_continuation_ids_equal':all(row['ordinary_ids_equal'] for row in report['trials']),
    'ordinary_equivalent':all(row['ordinary_equivalent'] for row in report['trials']),
    'scope':f'{args.verify_tokens} greedy tokens and numerical state/features on this one prompt; not full output-quality qualification',
    'failures':[{'round':row['round'],'mode':row['mode'],'checks':row['validation_failures']}
                for row in report['trials'] if row['validation_failures']]}
  report['numerically_qualified_for_ordinary_comparison']=(report['diagnostic_equivalence']['all_matched_checks_passed']
                                                          and report['diagnostic_equivalence']['ordinary_equivalent'])
  report['collection_complete']=True
  report['complete']=True
  save()


def main() -> None:
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('checkpoint','draft','prompt-json','report'):parser.add_argument('--'+name,type=Path,required=True)
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--chunk-size',type=int,default=128)
  parser.add_argument('--warmups',type=int,default=3)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--verify-tokens',type=int,default=4)
  parser.add_argument('--state-atol',type=float,default=2e-3)
  parser.add_argument('--state-rtol',type=float,default=3e-3)
  parser.add_argument('--server',default='http://127.0.0.1:8091')
  parser.add_argument('--include-selective-control',action='store_true')
  parser.add_argument('--kernel-timing',action='store_true',
                      help='DEBUG=2 timings synchronize kernels and change enqueue overlap; compare like-for-like only')
  parser.add_argument('--prepare-draft',action='store_true',help='Keep optional prepared draft allocations resident during every mode')
  parser.add_argument('--gpu-no-weight-packing',action='store_true')
  parser.add_argument('--gpu-q8-projections',action='store_true')
  parser.add_argument('--source-freeze',type=Path,help='Optional expected source_sha256 map, or report containing that map')
  args=parser.parse_args()
  os.environ['DEV']='CL'
  for name in ('checkpoint','draft','prompt_json','report'):setattr(args,name,getattr(args,name).resolve())
  ids=read_prompt(args.prompt_json)
  if (args.context!=512 or len(ids)!=128 or args.chunk_size!=128 or args.warmups<3 or args.runs<1
      or args.verify_tokens<1 or len(ids)+args.verify_tokens>args.context):
    parser.error('Need context512/prompt128/chunk128, >=3 warmup rounds, >=1 measured round and a valid continuation budget')
  if any(not math.isfinite(value) or value<0 for value in (args.state_atol,args.state_rtol)):
    parser.error('State tolerances must be finite and nonnegative')
  root=Path(__file__).resolve().parents[2]
  initial_sources=snapshot(root)
  if args.source_freeze is not None:
    frozen=json.loads(args.source_freeze.read_text())
    expected=frozen.get('source_sha256',frozen)
    if not isinstance(expected,dict) or any(initial_sources.get(name)!=value for name,value in expected.items()):
      raise ValueError('Source-freeze manifest differs from the current profiler/backend sources')
  initial_sessions=idle_gpu(args.server)
  identity=session_identity(initial_sessions)
  report={'schema_version':2,'complete':False,'collection_complete':False,'numerically_qualified_for_ordinary_comparison':False,
          'kind':'controlled DFlash2-resident target feature-prefill ablation',
          'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),'host':platform.node(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
          'device':'CL','context':args.context,'input_ids':ids,'prompt_sha256':digest(args.prompt_json),
          'checkpoint':str(args.checkpoint),'draft':str(args.draft),'source_sha256':initial_sources,'initial_server_sessions':initial_sessions,
          'configuration':{'warmups':args.warmups,'runs':args.runs,'chunk_size':args.chunk_size,'verify_tokens':args.verify_tokens,
                           'kernel_timing':args.kernel_timing,'prepare_draft':args.prepare_draft,
                           'gpu_pack_weights':not args.gpu_no_weight_packing,'gpu_q8_projections':args.gpu_q8_projections},
          'protocol':{'timing':('Target/first-token work and final GPU synchronization; '
                                'loading, warmup, state reset, validation readbacks and continuation excluded'),
                      'residency':'Same loaded target and trained draft across all modes; warmed graphs and allocations remain resident',
                      'ordinary_geometry':('Unmodified production generate and matched dynamic controls use full CL buffers '
                                           'and identical row/position bounds'),
                      'references':('Dynamic modes compare against ordinary; fixed-row modes compare against eager-none with the same geometry. '
                                    'Every mode also reports its ordinary comparison; no cross-geometry failure is hidden or waived.'),
                      'completion':'complete/collection_complete mean all measurements finished; numerical equivalence is reported separately',
                      'rotation':'Mode order rotates by one each round; first round establishes ordinary state/ID reference',
                      'global_ops_mem':'Compiler estimates, which may omit custom-kernel arithmetic/traffic; not hardware bandwidth counters',
                      'time_sum_s':'May be zero/unavailable without --kernel-timing; that flag synchronizes kernels and changes queue overlap',
                      'gpu_ownership':'Server GPU owners rejected before loading and between trials; outer operator lease excludes external GPU jobs',
                      'selective_control':'Private diagnostic only; no production TargetOutput realization contract is changed',
                      'quality':'Greedy continuation IDs and numerical state/features on one prompt; no RKLLM quality or speed qualification'},
          'trials':[],'qualified_rkllm_win':False}
  args.report.parent.mkdir(parents=True,exist_ok=True)
  with args.report.open('x') as output:output.write(json.dumps(report,indent=2)+'\n')
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  def check():
    if snapshot(root)!=initial_sources:raise RuntimeError('Backend/profiler sources changed during profiling')
    sessions=idle_gpu(args.server)
    if session_identity(sessions)!=identity:raise RuntimeError('Background resident model set changed during profiling')
    return sessions
  try:
    provenance_start=time.perf_counter()
    from tinygrad.llm.dflash2_config import DFLASH2_CONFIG_SHA256, DFLASH2_WEIGHTS_SHA256, DFLASH2_TARGET_SHA256, DFLASH2_REVISION
    hashes=(digest(args.checkpoint),digest(args.draft/'config.json'),digest(args.draft/'model.safetensors'))
    if hashes!=(DFLASH2_TARGET_SHA256,DFLASH2_CONFIG_SHA256,DFLASH2_WEIGHTS_SHA256):
      raise ValueError('Expected the pinned trained 2B target/draft pair')
    report.update(checkpoint_sha256=hashes[0],draft_config_sha256=hashes[1],draft_weights_sha256=hashes[2],draft_revision=DFLASH2_REVISION,
                  provenance_seconds=time.perf_counter()-provenance_start)
    save()
    check()
    _profile(args,report,save,check)
    check()
  except BaseException as error:
    report.update(complete=False,numerically_qualified_for_ordinary_comparison=False,
                  error={'type':type(error).__name__,'message':str(error)})
    save()
    raise
  finally:
    report['final_source_sha256']=snapshot(root)
    report['finished_utc']=datetime.datetime.now(datetime.UTC).isoformat()
    save()


if __name__=='__main__':main()
