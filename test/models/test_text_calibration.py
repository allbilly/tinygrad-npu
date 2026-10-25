import json
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
