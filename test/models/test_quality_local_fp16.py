"""Local FP16 comparison provenance and unchanged held-out quality gate."""
import hashlib, json
from types import SimpleNamespace
import numpy as np
import pytest
from extra.rockchip.quality_long_context import compare


def _write_json(path,data):path.write_text(json.dumps(data)+'\n')


def _fixture(tmp_path,degrade=False):
  manifest=tmp_path/'manifest.json'
  cases=[{'name':'seed','start':0,'gold':-1,'input_ids':[1]}]
  cases += [{'name':f't{i:04d}','start':i+1,'gold':2,'input_ids':[1]} for i in range(256)]
  data={'context':512,'vocab':8,'eos_id':7,'checkpoint_sha256':'c'*64,'scored_tokens':256,
    'max_perplexity_ratio_vs_rkllm':1.02,'max_perplexity_ratio_vs_fp16':1.02,'cases':cases}
  _write_json(manifest,data)
  digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
  reference,candidate=tmp_path/'reference',tmp_path/'candidate'
  for directory,device,precision in ((reference,'CL','fp16'),(candidate,'CPU','int8')):
    directory.mkdir()
    values=np.zeros(8,dtype=np.float32)
    values[2]=0 if degrade and device=='CPU' else 1
    rows=[]
    for case in cases[1:]:
      path=directory/(case['name']+'.f32')
      values.tofile(path)
      rows.append({'name':case['name'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    meta={'complete':True,'manifest_sha256':digest,'checkpoint_sha256':'c'*64,
      'source_sha256':{'tinygrad/llm/model.py':'a'*64},'device':device,'precision':precision,
      'gpu_npu':False,'gpu_q8_projections':False,'cases':rows}
    if device=='CPU':meta['placement']={'precision':'int8','layers':5,'prefill_layers':5,'decode_layers':5}
    _write_json(directory/'report.json',meta)
  return SimpleNamespace(manifest=manifest,stock=reference,candidate=candidate,directory=tmp_path/'result.json',reference_engine='fp16')


def test_equal_local_fp16_logits_pass_and_are_labeled_as_local(tmp_path):
  args=_fixture(tmp_path)
  compare(args)
  result=json.loads(args.directory.read_text())
  assert result['passed'] is True and result['positions']==256
  assert result['perplexity_ratio_vs_fp16']==1 and result['max_perplexity_ratio']==1.02
  assert result['reference_engine']=='fp16' and 'perplexity_ratio_vs_rkllm' not in result


def test_degraded_candidate_fails_existing_two_percent_limit(tmp_path):
  args=_fixture(tmp_path,degrade=True)
  with pytest.raises(AssertionError,match='Comparable-quality gate failed'):compare(args)
  result=json.loads(args.directory.read_text())
  assert result['passed'] is False and result['perplexity_ratio_vs_fp16']>1.02


@pytest.mark.parametrize('field,value',[('checkpoint_sha256','d'*64),('source_sha256',{}),('gpu_npu',True),('precision','int8')])
def test_wrong_local_reference_provenance_is_rejected(tmp_path,field,value):
  args=_fixture(tmp_path)
  path=args.stock/'report.json'
  meta=json.loads(path.read_text())
  meta[field]=value
  _write_json(path,meta)
  with pytest.raises(AssertionError):compare(args)
  assert not args.directory.exists()


def test_quality_limit_cannot_be_relaxed(tmp_path):
  args=_fixture(tmp_path)
  data=json.loads(args.manifest.read_text())
  data['max_perplexity_ratio_vs_fp16']=1.03
  _write_json(args.manifest,data)
  for directory in (args.stock,args.candidate):
    path=directory/'report.json'
    meta=json.loads(path.read_text())
    meta['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    _write_json(path,meta)
  with pytest.raises(AssertionError,match='must not exceed 1.02'):compare(args)


def test_default_rkllm_result_keys_remain_compatible(tmp_path):
  args=_fixture(tmp_path)
  args.reference_engine='rkllm'
  path=args.stock/'report.json'
  meta=json.loads(path.read_text())
  meta.pop('device')  # Legacy SDK reports do not carry a tinygrad device profile.
  meta['model_sha256']='b'*64
  meta['binary_sha256']='d'*64
  _write_json(path,meta)
  compare(args)
  result=json.loads(args.directory.read_text())
  assert result['passed'] is True and result['reference_engine']=='rkllm'
  assert result['perplexity_ratio_vs_rkllm']==1 and 'perplexity_ratio_vs_fp16' not in result


def test_local_fp16_report_cannot_be_silently_labeled_rkllm(tmp_path):
  args=_fixture(tmp_path)
  args.reference_engine='rkllm'
  with pytest.raises(AssertionError,match='SDK model and binary hashes'):compare(args)


def test_candidate_without_npu_placement_is_rejected(tmp_path):
  args=_fixture(tmp_path)
  path=args.candidate/'report.json'
  meta=json.loads(path.read_text())
  meta['placement']['decode_layers']=0
  _write_json(path,meta)
  with pytest.raises(AssertionError,match='positive INT8 NPU placement'):compare(args)


def test_local_fp16_requires_256_scored_positions(tmp_path):
  args=_fixture(tmp_path)
  data=json.loads(args.manifest.read_text())
  data['scored_tokens']=255
  data['cases']=data['cases'][:-1]
  _write_json(args.manifest,data)
  for directory in (args.stock,args.candidate):
    path=directory/'report.json'
    meta=json.loads(path.read_text())
    meta['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    _write_json(path,meta)
  with pytest.raises(AssertionError,match='256 declared held-out positions'):compare(args)
