import json
import gzip
import numpy as np
import pytest
from extra.rockchip.calibrate_text_int8 import load_scales


def test_text_scales_bind_checkpoint_and_share_fused_group_identity(tmp_path):
  path = tmp_path/'calibration.json'
  data = {'schema_version':1,'checkpoint_sha256':'weights','groups':{
    'qkv':{'members':['blk.0.attn_q','blk.0.attn_k','blk.0.attn_v'],'scales':[.25,2.,4.]}}}
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError,match='checkpoint'):load_scales(path,'other')
  scales = load_scales(path,'weights')
  assert scales['blk.0.attn_q'] is scales['blk.0.attn_v']
  np.testing.assert_array_equal(scales['blk.0.attn_k'].numpy(),[.25,2.,4.])
  data['groups']['qkv']['scales'] = [1.,0.,float('nan')]
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError,match='finite positive'):load_scales(path,'weights')


def test_compressed_core_scales_preserve_float32_bits_and_shared_members(tmp_path):
  from tinygrad.llm.kernels.rockchip_calibration import load_input_scales
  values=[.1,2**-24,65504.,1/3]
  data={'schema_version':1,'checkpoint_sha256':'weights','groups':{
    'qkv':{'members':['q','k','v'],'scales':values}}}
  path=tmp_path/'scales.json.gz'
  path.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
  actual=load_input_scales(path,'weights')
  assert actual['q'] is actual['k'] is actual['v']
  assert actual['q'].numpy().tobytes()==np.asarray(values,np.float32).tobytes()
  for bad in ([],[[1.,2.]],[1e40],[-1.],[float('nan')]):
    data['groups']['qkv']['scales']=bad
    path.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
    with pytest.raises(ValueError,match='finite positive'):load_input_scales(path,'weights')
  data['groups']['qkv']['scales']=[1.]
  data['groups']['other']={'members':['q'],'scales':[2.]}
  path.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
  with pytest.raises(ValueError,match='Duplicate'):load_input_scales(path,'weights')
