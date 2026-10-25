import builtins, copy, dataclasses, hashlib, json, sys
from types import SimpleNamespace
import pytest
from extra.rockchip.benchmark_speculative import consume_generation, reset_target_state, summarize, _ordinary
from extra.rockchip import benchmark_speculative as benchmark


def benchmark_args(*flags):
  return benchmark._parse_args(['--checkpoint','target.gguf','--draft','draft','--input','ids.txt','--report','result.json',*flags])


def test_optimized_options_are_explicit_and_default_off():
  args = benchmark_args()
  assert all(value is False for key,value in benchmark._speculative_options(args).items() if key!='verification_budget')
  assert args.verification_budget is None
  assert (args.engine,args.device,args.context,args.warmups,args.runs)==('paired','CL',512,3,3)


@pytest.mark.parametrize('flags',[
  ['--prepared-dynamic-rows'], ['--prepared-prefill-dynamic'], ['--prepared-exact-replay'],
  ['--prepared-target','--prepared-exact-replay'], ['--prepared-draft','--context','513'],
  ['--prepared-draft','--context','0'], ['--draft-mali-linear','--device','CPU'],
  ['--prepared-observation','--context','0'], ['--prepared-observation','--context','513'],
  ['--verification-budget','0'], ['--verification-budget','8'], ['--verification-budget','-1'],
  ['--warmups','2'], ['--runs','2'],
])
def test_incompatible_options_fail_before_files_server_or_hardware(monkeypatch,flags):
  monkeypatch.setattr(sys,'argv',['benchmark', '--checkpoint','missing.gguf','--draft','missing-draft',
                                 '--input','missing-input','--report','unused.json',*flags])
  def forbidden(*args,**kwargs):raise AssertionError('Invalid options must not reach hardware admission')
  monkeypatch.setattr(benchmark,'_idle_server',forbidden)
  monkeypatch.setattr(benchmark,'_engine',forbidden)
  with pytest.raises(SystemExit) as error:benchmark.main()
  assert error.value.code==2


def test_all_optimized_options_roundtrip_to_each_paired_child():
  flags = ['--'+name for name in benchmark.SPECULATIVE_FLAGS]
  args = benchmark_args(*flags,'--verification-budget','7','--timeout','55','--context','256',
                        '--gpu-recurrent-scan','--gpu-no-weight-packing','--gpu-q8-projections')
  for engine in ('ordinary','speculative','speculative','ordinary'):
    command = benchmark._child_command(args,engine,args.report)
    child = benchmark._parse_args(command[3:])
    assert child.engine==engine and child.timeout==55 and child.context==256
    assert benchmark._speculative_options(child)==benchmark._speculative_options(args)
    assert child.gpu_recurrent_scan and child.gpu_no_weight_packing and child.gpu_q8_projections


@pytest.mark.parametrize('device,all_options',[('CL',True),('CL',False),('CPU',False)])
def test_speculative_preparation_and_generation_receive_explicit_options(monkeypatch,device,all_options):
  flags = ['--'+name for name in benchmark.SPECULATIVE_FLAGS] if all_options else ['--prepared-observation']
  args = benchmark_args(*flags,'--engine','speculative','--device',device,'--verification-budget','3')
  config = dataclasses.make_dataclass('Config',[('block_size',int)])(8)
  Stats = dataclasses.make_dataclass('Stats',[(name,bool) for name in
    ('prepared_target','prepared_dynamic_rows','prepared_prefill_dynamic','prepared_exact_replay','early_anchor')])
  calls = []
  class Draft:
    def __init__(self,*values):
      calls.append(('draft',values))
      self.enabled = False
      self.observation = False
    def prepare_mali(self,**options):
      calls.append(('mali',options))
      return {'linear_layers':37,'packed_weights':False,'prepared_shared_head':True}
    def prepare_proposals(self,**options):
      calls.append(('proposals',options))
      self.enabled = True
    def proposal_preparation_metadata(self):return {'enabled':self.enabled,'graphs':int(self.enabled),'captured_graphs':0}
    def prepare_observation(self,**options):
      calls.append(('observation',options))
      self.observation = True
    def observation_preparation_metadata(self):return {'enabled':self.observation,'graphs':int(self.observation)}
  class Decoder:
    def __init__(self,target,draft,**options):
      calls.append(('decoder',options))
      self.stats = Stats(**options)
    def generate(self,prompt,**options):
      calls.append(('generate',options))
      return iter([7,8])
  def load(*values,**options):
    calls.append(('load',options))
    return config,{}
  modules = {'tinygrad':SimpleNamespace(dtypes=SimpleNamespace(float16='fp16',float32='fp32')),
    'tinygrad.llm.dflash2_config':SimpleNamespace(load_dflash2=load),
    'tinygrad.llm.dflash2':SimpleNamespace(DFlash2=Draft),
    'tinygrad.llm.dflash2_selector':SimpleNamespace(CandidateSelector=lambda *args:None),
    'tinygrad.llm.speculative':SimpleNamespace(GreedySpeculativeDecoder=Decoder)}
  original = builtins.__import__
  def fake_import(name,*values,**options):
    if name=='tinygrad' or name.startswith('tinygrad.'):return modules[name]
    return original(name,*values,**options)
  monkeypatch.setattr(builtins,'__import__',fake_import)
  target = SimpleNamespace(config=object(),token_embd=object(),output=object(),output_norm=object())
  report = {}
  _,decoder = benchmark._prepare_speculation(args,target,report)
  assert list(benchmark._generation(args,target,decoder,[1,2]))==[7,8]
  seen = dict((name,value) for name,value in calls)
  assert seen['load']=={'target_checkpoint':args.checkpoint,'device':device,'dtype':'fp16' if device=='CL' else 'fp32'}
  assert seen['observation']=={'max_context':512}
  if all_options:
    assert seen['mali']=={'pack_weights':False,'prepare_shared_head':True}
    assert seen['proposals']=={'max_context':512}
    assert report['preparation']['draft_mali']['linear_layers']==37
  else:
    assert 'mali' not in seen and 'proposals' not in seen
    assert report['preparation']['draft_mali'] is None
  assert seen['decoder']=={name:all_options for name in Stats.__dataclass_fields__}
  assert seen['generate']=={'max_new_tokens':64,'prefill_chunk_size':128,'verification_budget':3}
  assert report['draft_config']['block_size']==8
  assert report['preparation']['draft_proposals']=={'enabled':all_options,'graphs':int(all_options),'captured_graphs':0}
  assert report['preparation']['draft_observation']=={'enabled':True,'graphs':1}
  assert report['preparation']['decoder']==seen['decoder']


def test_ordinary_ignores_requested_draft_options_and_never_imports_speculation(monkeypatch):
  args = benchmark_args(*['--'+name for name in benchmark.SPECULATIVE_FLAGS],'--engine','ordinary',
                        '--verification-budget','7','--tokens','2')
  original = builtins.__import__
  def forbidden(name,*values,**options):
    if name=='tinygrad' or name.startswith('tinygrad.'):raise AssertionError('Ordinary must not load or prepare a draft')
    return original(name,*values,**options)
  monkeypatch.setattr(builtins,'__import__',forbidden)
  calls = []
  class Target:
    def generate(self,prompt,**options):
      calls.append((prompt,options))
      try:yield from [7,8,9]
      finally:calls.append('ordinary_closed')
  target,report = Target(),{}
  assert benchmark._prepare_speculation(args,target,report)==(None,None)
  assert list(benchmark._generation(args,target,None,[1,2]))==[7,8]
  assert calls==[([1,2],{'chunk_size':128,'temperature':0}),'ordinary_closed']
  assert report['preparation']=={'engine':'ordinary','draft_loaded':False,'draft_mali':None,'draft_proposals':None,
                                'draft_observation':None,'decoder':None}


def trials():
  rows = []
  for engine in ('ordinary','speculative'):
    for index in range(6):
      row = {'engine':engine,'warmup':index<3,'context':32,'prompt_tokens':8,'generated_ids':[1,2,3,4],
             'state_reset':True,'first_token_ms':100 if engine=='ordinary' else 150,
             'total_seconds':2 if engine=='ordinary' else 1,'output_tps':2 if engine=='ordinary' else 4,
             'decode_full_cost_tps':2 if engine=='ordinary' else 3}
      if engine=='speculative':row['speculative_stats'] = {'output_tokens':4,'proposed_tokens':3,'accepted_tokens':2}
      rows.append(row)
  return rows


def summarize_trials(rows):return summarize(rows,prompt_tokens=8,output_tokens=4,context=32)


def test_relative_speed_does_not_claim_an_rkllm_win():
  result = summarize_trials(trials())
  assert result['full_request_speedup']==2
  assert result['full_cost_decode_speedup']==1.5
  assert result['ttft_speedup']==pytest.approx(2/3)
  assert result['engines']['speculative']['acceptance_rate']==pytest.approx(2/3)
  assert result['engines']['speculative']['accepted_draft_tps']==2
  assert result['greedy_ids_equal'] and result['output_tps_ranges_separated']
  assert result['qualified_rkllm_win'] is False


@pytest.mark.parametrize('index',[0,3,5,6,9,11])
def test_changed_ids_in_either_engine_or_warmup_are_rejected(index):
  rows = trials()
  rows[index]['generated_ids'][-1] = 99
  with pytest.raises(ValueError,match='greedy token IDs differ'):summarize_trials(rows)


def test_stable_but_different_abba_cohorts_are_rejected():
  rows = trials()+trials()
  for row in rows[12:]:row['generated_ids'] = [9,9,9,9]
  with pytest.raises(ValueError,match='greedy token IDs differ'):summarize_trials(rows)


@pytest.mark.parametrize('field,value',[('context',64),('prompt_tokens',7),('state_reset',False),('generated_ids',[1,2,3])])
def test_partial_or_cached_requests_cannot_be_promoted(field,value):
  rows = trials()
  rows[-1][field] = value
  with pytest.raises(ValueError,match='workload|reset'):summarize_trials(rows)


@pytest.mark.parametrize('field,value',[('accepted_tokens',4),('proposed_tokens',-1),('output_tokens',3)])
def test_invalid_acceptance_counters_are_rejected(field,value):
  rows = trials()
  rows[-1]['speculative_stats'][field] = value
  with pytest.raises(ValueError,match='acceptance counters'):summarize_trials(rows)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),0,-1])
def test_invalid_times_are_rejected_in_warmups_too(value):
  rows = trials()
  rows[0]['total_seconds'] = value
  with pytest.raises(ValueError,match='timings'):summarize_trials(rows)


def test_three_full_warmups_and_measurements_are_required():
  rows = trials()
  with pytest.raises(ValueError,match='three full'):summarize_trials(rows[:5]+rows[6:])
  with pytest.raises(ValueError,match='three full'):summarize_trials(rows[1:])


def test_cleanup_and_final_synchronization_are_in_primary_cost():
  now,events = [0.0],[]
  def generate():
    try:
      now[0] += .1
      yield 1
      now[0] += .2
      yield 2
    finally:
      now[0] += .7
      events.append('close')
  def synchronize():
    if events:now[0] += .4
    events.append('sync')
  result = consume_generation(generate,2,synchronize,lambda:now[0])
  assert result['first_token_ms']==pytest.approx(100)
  assert result['total_seconds']==pytest.approx(1.4)
  assert result['decode_tps']==pytest.approx(5)
  assert result['decode_full_cost_tps']==pytest.approx(1/1.3)
  assert result['cleanup_seconds']==pytest.approx(1.1)
  assert events==['sync','close','sync']


@pytest.mark.parametrize('tokens',[[1],[1,2,3],[1,-1]])
def test_incomplete_extra_and_invalid_outputs_fail_and_close(tokens):
  events,now = [],[0.0]
  def generate():
    try:
      for token in tokens:
        now[0] += 1
        yield token
    finally:events.append('close')
  with pytest.raises(ValueError):consume_generation(generate,2,lambda:events.append('sync'),lambda:now[0])
  assert 'close' in events and events[-1]=='sync'


def test_ordinary_uses_production_generator_and_closes_fixed_length_request():
  events,prompt = [],[1,2,3]
  class Target:
    def generate(self,tokens,chunk_size,temperature):
      assert chunk_size==128 and temperature==0
      tokens.append(99)
      try:yield from [4,5,6]
      finally:events.append('close')
  assert list(_ordinary(Target(),prompt,2,128))==[4,5]
  assert prompt==[1,2,3] and events==['close']


def test_all_supported_state_buffers_are_cleared_and_shared_aliases_deduplicated():
  from tinygrad import Tensor
  states = {name:Tensor.ones(2,device='CPU').realize() for name in
            ('cache_kv','cache_k','cache_kg','conv_state','recurrent_state','cache_conv')}
  block = SimpleNamespace(**states,shortconv=SimpleNamespace(cache_conv=states['cache_conv']))
  target = SimpleNamespace(blk=[block],_cached_tokens=[1,2,3],_greedy_generation=True)
  reset = []
  cleared = reset_target_state(target,SimpleNamespace(reset=lambda:reset.append(True)))
  assert len(cleared)==6 and len(set(cleared))==6
  assert target._cached_tokens==[] and target._greedy_generation is False and reset==[True]
  assert all(value.tolist()==[0,0] for value in states.values())


def test_summary_does_not_mutate_measured_evidence():
  rows = trials()
  original = copy.deepcopy(rows)
  summarize_trials(rows)
  assert rows==original


@pytest.mark.parametrize('mutation',[None,'different_flag','inactive_body','fused_body','assisted_body','unexpected_body','undeclared_body'])
def test_quality_requires_actual_identical_q8_body_configuration(tmp_path,mutation):
  from extra.rockchip.benchmark_rkllm_pair import check_quality
  from extra.rockchip.rkllm_reference import file_hash
  manifest = tmp_path/'manifest.json'
  manifest.write_text(json.dumps({'context':512}))
  candidate = {'checkpoint_sha256':'candidate','device':'CL','gpu_q8_projections':True,
               'gpu_kernels':{'q8_projection_layers':196},'source_sha256':{'tinygrad/example.py':'source'}}
  report = {'complete':True,'passed':True,'positions':256,'max_perplexity_ratio':1.02,'perplexity_ratio_vs_rkllm':1.001,
            'manifest_sha256':file_hash(manifest),'provenance':[{'model_sha256':'reference'},candidate]}
  args = SimpleNamespace(device='CL',gpu_fuse_projections=False,gpu_q8_head=False,gpu_q8_projections=True,prefill='off',decode='off')
  if mutation=='different_flag':candidate['gpu_q8_projections'] = False
  elif mutation=='inactive_body':candidate['gpu_kernels']['q8_projection_layers'] = 0
  elif mutation=='fused_body':candidate['gpu_kernels']['fused_qkv'] = 28
  elif mutation=='assisted_body':candidate['gpu_npu'] = True
  elif mutation=='unexpected_body':args.gpu_q8_projections = False
  elif mutation=='undeclared_body':
    args.gpu_q8_projections = False
    candidate['gpu_q8_projections'] = False
  path = tmp_path/'quality.json'
  path.write_text(json.dumps(report))
  def check():return check_quality(path,manifest,'candidate','reference',512,args,{'tinygrad/example.py':'source'})
  if mutation is None:assert check()['passed']
  else:
    with pytest.raises(ValueError,match='configuration|activate'):check()


@pytest.mark.parametrize('changed_source',['mali_q8.py','mali_q8_checkpoint.py'])
def test_standalone_provenance_changes_with_q8_sources_without_initializing_tinygrad(tmp_path,monkeypatch,changed_source):
  from extra.rockchip.benchmark_inference import source_snapshot
  kernels = tmp_path/'tinygrad/llm/kernels'
  kernels.mkdir(parents=True)
  benchmark = tmp_path/'extra/rockchip/benchmark_inference.py'
  benchmark.parent.mkdir(parents=True)
  benchmark.write_text('raise AssertionError("Do not run the benchmark")\n')
  (tmp_path/'tinygrad/__init__.py').write_text('raise AssertionError("Do not initialize tinygrad")\n')
  for name in ('mali_q8.py','mali_q8_checkpoint.py'):
    (kernels/name).write_text('raise AssertionError("Do not initialize a device")\n')
  original_import = builtins.__import__
  def forbid_tinygrad(name,*args,**kwargs):
    if name=='tinygrad' or name.startswith('tinygrad.'):raise AssertionError('Provenance must not import numerical code')
    return original_import(name,*args,**kwargs)
  monkeypatch.setattr(builtins,'__import__',forbid_tinygrad)
  before = source_snapshot(tmp_path)
  changed = kernels/changed_source
  changed.write_text('raise AssertionError("Changed native Q8 implementation")\n')
  after = source_snapshot(tmp_path)
  relative = str(changed.relative_to(tmp_path))
  assert {name for name in before if before[name]!=after[name]}=={relative}
  assert after[relative]==hashlib.sha256(changed.read_bytes()).hexdigest()
  # Automatically cover a future implementation file rather than maintaining
  # another test-side copy of a production filename allowlist.
  future = kernels/'future_q8_decode.c'
  future.write_text('void decode(void) {}\n')
  extended = source_snapshot(tmp_path)
  assert extended.keys()-after.keys()=={str(future.relative_to(tmp_path))}
