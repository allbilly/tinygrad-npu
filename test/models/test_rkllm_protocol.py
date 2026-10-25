import copy, json
from types import SimpleNamespace
import pytest
from extra.rockchip.benchmark_rkllm_pair import check_quality, summarize
from extra.rockchip.rkllm_reference import file_hash


def trials(*,ttft=100,decode=40,context=512):
  return [{'warmup':index<3,'context':context,'prompt_tokens':128,'generated_ids':list(range(64)),
           'sdk_prefill_tokens':128,'kv_tokens':191,'first_token_ms':ttft,'decode_tps':decode} for index in range(6)]


def test_speed_without_quality_cannot_be_a_qualified_win():
  result = summarize(trials(),trials(ttft=120,decode=30),context=512)
  assert result['observed_medians_faster_both'] and not result['qualified_win'] and result['outcome']=='quality_pending'
  assert summarize(trials(),trials(ttft=120,decode=30),context=512,quality_passed=True)['qualified_win']


def test_one_phase_loss_and_overlapping_ranges_are_not_verified_wins():
  assert summarize(trials(),trials(ttft=120,decode=50),context=512,quality_passed=True)['outcome']=='measured_loss'
  candidate = trials()
  candidate[-1]['first_token_ms']=130
  assert summarize(candidate,trials(ttft=120,decode=30),context=512,quality_passed=True)['outcome']=='repeat_needed'


@pytest.mark.parametrize('field,value',[('context',1024),('prompt_tokens',256),('generated_ids',[1,2]),('kv_tokens',190)])
def test_mismatched_sdk_work_is_rejected(field,value):
  reference = trials()
  reference[-1][field]=value
  with pytest.raises(ValueError):summarize(trials(),reference,context=512,quality_passed=True)


def test_pilot_is_not_a_repeated_qualification():
  with pytest.raises(ValueError,match='three measured'):summarize(trials()[:4],trials(),context=512)


@pytest.mark.parametrize('engine',['onpu','rkllm'])
@pytest.mark.parametrize('index',[0,3,5])
def test_changing_greedy_ids_within_an_engine_is_rejected(engine,index):
  rows = {'onpu':trials(),'rkllm':trials(ttft=120,decode=30)}
  rows[engine][index]['generated_ids'][-1] = 999
  with pytest.raises(ValueError,match=engine+': greedy output changed'):
    summarize(rows['onpu'],rows['rkllm'],context=512,quality_passed=True)


@pytest.mark.parametrize('engine',['onpu','rkllm'])
def test_changed_greedy_ids_between_stable_abba_cohorts_are_rejected(engine):
  rows = {'onpu':trials()+trials(),'rkllm':trials(ttft=120,decode=30)+trials(ttft=120,decode=30)}
  for row in rows[engine][6:]:row['generated_ids'] = [999]*64
  with pytest.raises(ValueError,match=engine+': greedy output changed'):
    summarize(rows['onpu'],rows['rkllm'],context=512,quality_passed=True)


def test_cross_engine_greedy_differences_are_allowed_with_independent_quality_evidence():
  candidate = trials()+trials()
  reference = trials(ttft=120,decode=30)+trials(ttft=120,decode=30)
  for row in reference:row['generated_ids'] = [999]*64
  assert summarize(candidate,reference,context=512,quality_passed=True)['qualified_win']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),0,-1])
def test_invalid_timings_are_rejected(value):
  candidate = trials()
  candidate[-1]['decode_tps'] = value
  with pytest.raises(ValueError,match='Invalid timing'):summarize(candidate,trials(),context=512)


def test_quality_checks_weights_source_configuration_and_actual_ratio(tmp_path):
  manifest = tmp_path/'manifest.json'
  manifest.write_text(json.dumps({'context':512}))
  report = {'complete':True,'passed':True,'positions':256,'max_perplexity_ratio':1.02,'perplexity_ratio_vs_rkllm':1.001,
            'manifest_sha256':file_hash(manifest),'provenance':[{'model_sha256':'reference'},
            {'checkpoint_sha256':'candidate','device':'CL','source_sha256':{'tinygrad/example.py':'source'}}]}
  path = tmp_path/'quality.json'
  args = SimpleNamespace(device='CL',gpu_fuse_projections=False,gpu_q8_head=False,prefill='off',decode='off')
  snapshot = {'tinygrad/example.py':'source'}
  def check(data):
    path.write_text(json.dumps(data))
    return check_quality(path,manifest,'candidate','reference',512,args,snapshot)
  assert check(report)['passed']
  bad = copy.deepcopy(report)
  bad['perplexity_ratio_vs_rkllm']=1.1
  with pytest.raises(ValueError,match='contradicts'):check(bad)
  bad = copy.deepcopy(report)
  bad['provenance'][1]['source_sha256']['tinygrad/example.py']='other'
  with pytest.raises(ValueError,match='source differs'):check(bad)
  bad = copy.deepcopy(report)
  bad['provenance'][0]['model_sha256']='other'
  with pytest.raises(ValueError,match='weights differ'):check(bad)
  bad = copy.deepcopy(report)
  bad['provenance'][1]['input_scales_sha256']='unmatched calibration'
  with pytest.raises(ValueError,match='input scaling differs'):check(bad)


def test_cpu_npu_quality_requires_the_actual_prepared_int8_configuration(tmp_path):
  manifest = tmp_path/'manifest.json'
  manifest.write_text(json.dumps({'context':512}))
  placement = {key:True for key in ('native_replay','cpu_pool','cpu_fast_softmax','cache_cleaning','wake_helper')}
  placement.update(precision='int8',layers=197,prefill_layers=197,decode_layers=197,
                   cpu_attention_blocks=28,fused_ffn_blocks=28,cpu_norms=57)
  report = {'complete':True,'passed':True,'positions':256,'max_perplexity_ratio':1.02,'perplexity_ratio_vs_rkllm':1.001,
            'manifest_sha256':file_hash(manifest),'provenance':[{'model_sha256':'reference'},
            {'checkpoint_sha256':'candidate','device':'CPU','precision':'int8','placement':placement,
             'source_sha256':{'tinygrad/example.py':'source'}}]}
  path = tmp_path/'quality.json'
  args = SimpleNamespace(device='CPU',gpu_fuse_projections=False,gpu_q8_head=False,prefill='on',decode='on')
  def check():
    path.write_text(json.dumps(report))
    return check_quality(path,manifest,'candidate','reference',512,args,{'tinygrad/example.py':'source'})
  assert check()['passed']
  placement['decode_layers'] = 196
  with pytest.raises(ValueError,match='configuration differs'):check()
