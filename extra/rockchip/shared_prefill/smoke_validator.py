"""In-memory stdlib smoke checks for validate_report; creates no run artifacts."""
import copy
import json
from extra.rockchip.shared_prefill.validate_report import MODES, validate


EXPECTED = {
  'case':'smoke-case',
  'geometry':{'rows':64, 'columns':2048, 'depth':2048},
  'run_token':'smoke-run-token',
  'runner_sha256':'1'*64,
  'helper_sha256':'2'*64,
  'source_sha256':{'tinygrad/example.py':'3'*64},
}


def row(mode, phase, index, round_number):
  return {
    'mode':mode, 'phase':phase, 'input_index':index, 'round':round_number,
    'seconds':1.0+round_number/100, 'npu_submissions':0 if mode=='gpu_fp32' else 2,
    'npu_allocations_during_call':0, 'finite':True,
    'reference_kind':'FP32_input_FP16_weight' if mode=='gpu_fp32' else 'FP16_rounded_input_FP16_weight',
    'reference_allclose':True, 'max_abs_error':0.0, 'max_tolerance_ratio':0.0,
    'max_abs_error_against_FP32_input':0.0, 'relative_RMS_against_FP32_input':0.0,
    'bit_equal_to_copy':None if mode=='gpu_fp32' or (mode=='npu_copy_fp16' and phase=='warmup') else True,
    'output_sha256':('a' if mode=='gpu_fp32' else 'b')*64,
  }


def fixture():
  geometry=EXPECTED['geometry'];tile=min(geometry['rows'],64,65536//geometry['depth'])
  warmups=[row(mode,'warmup',index,index) for mode in MODES for index in range(3)]
  trials=[row(mode,'measured',round_number%3,round_number)
          for round_number in range(9) for mode in MODES[round_number%3:]+MODES[:round_number%3]]
  medians={mode:sorted(r['seconds'] for r in trials if r['mode']==mode)[4] for mode in MODES}
  return {
    'schema':'gpu-npu-prefill-sharing-operator-v1','complete':True,'case':EXPECTED['case'],
    'geometry':geometry,'source_sha256':dict(EXPECTED['source_sha256']),
    'source_sha256_after':dict(EXPECTED['source_sha256']),'run_token':EXPECTED['run_token'],
    'runner_sha256':EXPECTED['runner_sha256'],'helper_sha256':EXPECTED['helper_sha256'],
    'scope':'Synthetic helper smoke only. No model-quality or serving/speed qualification.',
    'model_qualified':False,'API_qualified':False,'quality_qualified':False,'speed_qualified':False,
    'device':{'extensions':['cl_arm_import_memory_dma_buf']},
    'data_sha256':{'weight':'4'*64,'inputs':['5'*64,'6'*64,'7'*64]},
    'npu_tile_rows':tile,'npu_tiles_per_call':geometry['rows']//tile,
    'warmups':warmups,'trials':trials,
    'guards':[
      {'kind':'invalid_activation','value':'nan','rejected':True,'npu_submissions':0},
      {'kind':'invalid_activation','value':'inf','rejected':True,'npu_submissions':0},
      {'kind':'invalid_activation','value':'10000000000.0','rejected':True,'npu_submissions':0},
      {'kind':'offset_input_view','rejected':True,'npu_submissions':0},
      {'kind':'offset_weight_view','rejected':True,'npu_submissions':0,'npu_allocations_in_constructor':0},
    ],
    'recovery_bit_equal_to_copy':True,
    'summaries':{mode:{'median_seconds':medians[mode],'measured_runs':9} for mode in MODES},
  }


def main():
  base=fixture()
  result=validate(base,EXPECTED)
  checks=[{'name':'positive fixture','accepted':result['valid'],'expected_accepted':True}]
  if not result['valid']: checks[-1]['errors']=result['errors']

  mutations=[]
  def add(name, mutate):
    candidate=copy.deepcopy(base);mutate(candidate);mutations.append((name,candidate))
  add('missing mode',lambda r:r['trials'].__delitem__(next(i for i,x in enumerate(r['trials']) if x['mode']=='npu_shared_fp16')))
  add('wrong input rotation',lambda r:r['trials'][0].update(input_index=2))
  add('false finite flag',lambda r:r['warmups'][0].update(finite=False))
  add('tolerance ratio over one',lambda r:r['trials'][0].update(max_tolerance_ratio=1.01))
  add('wrong NPU submission count',lambda r:r['trials'][1].update(npu_submissions=1))
  add('shared allocation after warmup',lambda r:r['trials'][2].update(npu_allocations_during_call=1))
  add('shared output differs from copy',lambda r:r['trials'][2].update(bit_equal_to_copy=False))
  add('offset view accepted',lambda r:r['guards'][3].update(rejected=False))
  add('guard submitted to NPU',lambda r:r['guards'][0].update(npu_submissions=1))
  add('wrong summary median',lambda r:r['summaries']['gpu_fp32'].update(median_seconds=999.0))
  add('source changed after run',lambda r:r['source_sha256_after'].update({'tinygrad/example.py':'8'*64}))
  add('qualification asserted',lambda r:r.update(speed_qualified=True))
  add('duplicate input hash',lambda r:r['data_sha256'].update(inputs=['5'*64,'5'*64,'7'*64]))
  add('offset weight constructor allocation',lambda r:r['guards'][4].update(npu_allocations_in_constructor=1))
  for name,candidate in mutations:
    verdict=validate(candidate,EXPECTED)
    checks.append({'name':name,'accepted':verdict['valid'],'expected_accepted':False})
    if verdict['valid']: checks[-1]['errors']=['mutated fixture unexpectedly validated']

  passed=all(item['accepted'] is item['expected_accepted'] for item in checks)
  print(json.dumps({'passed':passed,'checks':checks},indent=2))
  return 0 if passed else 1


if __name__ == '__main__': raise SystemExit(main())
