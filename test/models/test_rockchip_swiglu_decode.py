"""CPU contracts for opt-in one-row raw INT8 SwiGLU decode."""
from types import SimpleNamespace
import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, dequantize_tiles, quantize
from tinygrad.llm.kernels.rockchip_gelu_ffn import (RockchipInt8SwiGLUFeedForward,
                                                   pack_swiglu_projection, payload_layout)
from test.runtime.test_rockchip_gelu_ffn import fixture


def _model(width=4864):
  return Transformer(TransformerConfig(1,512,width,8,2,1e-6,32,64,10000.,64,64,max_context=64,
                                      ffn_activation='silu'))


def _fake_prepare(layer):
  layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
  layer.scales=[Tensor.ones(layer.columns,device='CPU',dtype=dtypes.float)]


def _phase_decode(hook):
  for layer in (*hook.projections.layers,hook.down): layer.phase='decode'
  for part in getattr(hook.down,'parts',[]): part.phase='decode'


def _outputs(tensors):
  tensors[0].realize(tensors[1])
  return tensors[0].numpy().copy(),tensors[1].numpy().view(np.uint32).copy()


def _dense_projection(raw,sa,sw,tiles,rows,columns):
  pieces=[]
  for tile,value in zip(tiles,raw):
    pieces.append((tile.first_column,dequantize_tiles([value],[sa[tile.scale_index]],
                   sw[tile.first_column:tile.first_column+tile.columns],[tile.rows],tile.columns)))
  pieces.sort(key=lambda item:item[0])
  return Tensor.cat(*(value for _,value in pieces),dim=1).reshape(rows,columns).realize()


def _independent_swiglu_quantize(projected,hidden,depths):
  joint=projected.reshape(1,1,hidden*2).contiguous().realize()
  sigmoid=joint[:,:,:hidden].sigmoid().contiguous().realize()
  activated=((joint[:,:,:hidden]*sigmoid)*joint[:,:,hidden:]).reshape(1,hidden).contiguous().realize()
  payloads,scales=[],[]
  column=0
  for depth,(_,padded,tile,_) in zip(depths,payload_layout(1,depths)):
    payload,scale=quantize(activated[:,column:column+depth].contiguous(),padded)
    payloads.append(payload)
    scales.append(scale)
    column+=depth
  return Tensor.cat(*payloads),Tensor.cat(*scales)


def test_decode_enable_is_opt_in_phase_and_capacity_gated(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  disabled=_model()
  assert configure(disabled,prefill='off',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True)['fused_swiglu_ffn_blocks']==1
  hook=disabled.blk[0].fused_swiglu_ffn
  assert hook.decode_enabled is False
  _phase_decode(hook)
  assert not hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))

  enabled=_model()
  assert configure(enabled,prefill='off',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True,fused_swiglu_decode=True)['fused_swiglu_ffn_blocks']==1
  hook=enabled.blk[0].fused_swiglu_ffn
  assert hook.decode_enabled is True
  _phase_decode(hook)
  assert hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  for layer in hook.projections.layers: layer.phase='prefill'
  assert not hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  _phase_decode(hook)
  assert not hook.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))


def test_decode_requires_all_joint_and_down_decode_placement(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  model=_model()
  configure(model,prefill='off',decode='on',precision='int8',fuse_projections=True,
            fused_swiglu_ffn=True,fused_swiglu_decode=True)
  hook=model.blk[0].fused_swiglu_ffn
  _phase_decode(hook)
  assert hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  hook.projections.layers[0].decode='off'
  assert not hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  hook.projections.layers[0].decode='on'
  hook.down.decode='off'
  assert not hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))


@pytest.mark.parametrize('name',['ffn_gate','ffn_up','ffn_down'])
def test_decode_count_excludes_mixed_projection_placement(monkeypatch,name):
  model=_model()
  source=getattr(model.blk[0],name)
  original_init=RockchipInt8Linear.__init__
  def initialize(layer,*args):
    original_init(layer,*args)
    if layer.source is source:layer.decode='off'
  monkeypatch.setattr(RockchipInt8Linear,'__init__',initialize)
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  report=configure(model,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True,fused_swiglu_decode=True)
  hook=model.blk[0].fused_swiglu_ffn
  assert report['fused_swiglu_ffn_blocks']==1 and report['fused_swiglu_decode_blocks']==0
  assert hook.decode_enabled is False
  _phase_decode(hook)
  assert not hook.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))


def test_decode_option_validation_precedes_weight_preparation(monkeypatch):
  calls=[]
  monkeypatch.setattr(RockchipInt8Linear,'prepare',lambda layer:calls.append(layer))
  with pytest.raises(ValueError):
    configure(SimpleNamespace(),fused_swiglu_decode=1)
  with pytest.raises(ValueError):
    configure(SimpleNamespace(),fused_swiglu_decode=True)
  with pytest.raises(ValueError):
    configure(_model(),fused_swiglu_ffn=True,fused_swiglu_decode=True,precision='fp16')
  assert calls==[]


def test_decode_reconfiguration_clears_decode_enable_and_disable_removes_hook(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  model=_model()
  configure(model,prefill='off',decode='on',precision='int8',fuse_projections=True,
            fused_swiglu_ffn=True,fused_swiglu_decode=True)
  assert model.blk[0].fused_swiglu_ffn.decode_enabled is True
  configure(model,prefill='off',decode='on',precision='int8',fuse_projections=True,
            fused_swiglu_ffn=True,fused_swiglu_decode=False)
  assert model.blk[0].fused_swiglu_ffn.decode_enabled is False
  configure(model,prefill='off',decode='off',precision='int8',fuse_projections=True,
            fused_swiglu_ffn=False,fused_swiglu_decode=False)
  assert not hasattr(model.blk[0],'fused_swiglu_ffn')


def test_swiglu_decode_constructor_rejects_non_bool():
  hook=SimpleNamespace(projections=SimpleNamespace(),down=SimpleNamespace())
  with pytest.raises(ValueError):RockchipInt8SwiGLUFeedForward(hook.projections,hook.down,decode=1)


@pytest.mark.parametrize('hidden', [4864,4608])
def test_one_row_swiglu_packing_matches_independent_int32_dequant_activation_and_quantize(hidden):
  rows=1
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  raw,sa,sw,tiles,_=fixture(rows,hidden,1,seed=4200+hidden)
  # Exercise zero codes and zero scales as well as ordinary signed integer payloads.
  raw_values=raw[0].numpy().copy()
  raw_values[:4]=np.array([0,-1,127,-128],dtype=np.int32)
  raw[0]=Tensor(raw_values,device='CPU').realize()
  sw_values=sw.numpy().copy()
  sw_values[0]=0.0
  sw=Tensor(sw_values,device='CPU').realize()
  projected=_dense_projection(raw,sa,sw,tiles,rows,hidden*2)
  expected=_outputs(_independent_swiglu_quantize(projected,hidden,depths))
  actual=_outputs(pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,rows))
  np.testing.assert_array_equal(actual[0],expected[0])
  np.testing.assert_array_equal(actual[1],expected[1])


def test_one_row_swiglu_packing_nonfinite_projection_matches_reference_nan_and_zero_codes():
  rows,hidden=1,4864
  depths=(4096,768)
  raw,sa,sw,tiles,_=fixture(rows,hidden,1,seed=731)
  values=sw.numpy().copy()
  values[:]=np.inf
  bad_sw=Tensor(values,device='CPU').realize()
  projected=_dense_projection(raw,sa,bad_sw,tiles,rows,hidden*2)
  expected=_outputs(_independent_swiglu_quantize(projected,hidden,depths))
  actual=_outputs(pack_swiglu_projection(raw,sa,bad_sw,tiles,rows,hidden,depths,rows))
  np.testing.assert_array_equal(actual[0],expected[0])
  np.testing.assert_array_equal(actual[1],expected[1])
  assert np.isnan(actual[1].view(np.float32)).any()
  for offset,padded,_,_ in payload_layout(rows,depths):
    assert np.count_nonzero(actual[0][offset+64:offset+64+padded])==0


def test_one_row_swiglu_pack_with_inactive_pool_stays_serial_and_bitwise():
  import os
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  if not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
    pytest.skip('CPU4-7 unavailable for optional pool check')
  rows,hidden=1,4864
  depths=(4096,768)
  raw,sa,sw,tiles,_=fixture(rows,hidden,1,seed=991)
  expected=_outputs(pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,rows))
  pool=AttentionPool()
  try:
    before=pool.lib.pool_calls()
    actual=_outputs(pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,rows,pool=pool))
    assert pool.lib.pool_calls()==before
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])
  finally:
    pool.close()


@pytest.mark.parametrize('device,precision,base,decode',[
  ('CL','int8',True,'on'),('CPU','fp16',True,'on'),
  ('CPU','int8',False,'on'),('CPU','int8',True,'off'),
])
def test_decode_cli_rejects_invalid_options_before_reading_checkpoints(monkeypatch,capsys,tmp_path,device,precision,base,decode):
  from extra.rockchip import benchmark_inference as bench
  argv=['benchmark_inference','--checkpoint',str(tmp_path/'absent.gguf'),
        '--input',str(tmp_path/'absent.tokens'),'--report',str(tmp_path/'report.json'),
        '--device',device,'--precision',precision,'--decode',decode,'--fused-swiglu-decode']
  if base:argv.append('--fused-swiglu-ffn')
  monkeypatch.setattr('sys.argv',argv)
  with pytest.raises(SystemExit) as error:bench.main()
  assert error.value.code==2
  assert 'Fused SwiGLU decode requires CPU INT8' in capsys.readouterr().err
  assert not (tmp_path/'report.json').exists()
