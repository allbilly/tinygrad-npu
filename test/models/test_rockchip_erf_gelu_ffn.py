"""CPU placement tests for INT8 erf-GELU FFN prefill and decode hooks."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import RockchipProjections, configure
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8ErfGeluFeedForward
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear


def _model(activation='gelu',width=6656,down_bias=False):
  return Transformer(TransformerConfig(1,512,width,8,2,1e-6,32,64,10000.,64,64,max_context=64,
                                      ffn_activation=activation,ffn_down_bias=down_bias))


def _fake_prepare(layer):
  layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
  layer.scales=[Tensor.ones(layer.columns,device='CPU',dtype=dtypes.float)]


def _phase(kernel,phase):
  for layer in (*kernel.projections.layers,kernel.down): layer.phase=phase
  for part in getattr(kernel.down,'parts',[]): part.phase=phase


def _configure(model,**kwargs):
  return configure(model,prefill='on',decode='off',precision='int8',fuse_projections=True,**kwargs)


def test_prefill_installation_disable_and_non_float_inputs(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  opened={name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}
  model=_model()
  result=_configure(model,fused_erf_gelu_ffn=True)
  assert result['fused_erf_gelu_ffn_blocks']==1 and result['fused_erf_gelu_decode_blocks']==0
  kernel=model.blk[0].fused_erf_gelu_ffn
  assert isinstance(kernel,RockchipInt8ErfGeluFeedForward)
  _phase(kernel,'prefill')
  assert kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  assert not kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.half))

  disabled=configure(model,prefill='off',decode='off',precision='int8',fused_erf_gelu_ffn=False)
  assert disabled['fused_erf_gelu_ffn_blocks']==0 and not hasattr(model.blk[0],'fused_erf_gelu_ffn')
  assert {name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}==opened


def test_decode_requires_explicit_decode_flag_and_all_phase_modes_on(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  model=_model()
  result=configure(model,prefill='off',decode='on',precision='int8',fuse_projections=True,
                   fused_erf_gelu_ffn=True,fused_erf_gelu_decode=True)
  assert result['fused_erf_gelu_ffn_blocks']==1 and result['fused_erf_gelu_decode_blocks']==1
  kernel=model.blk[0].fused_erf_gelu_ffn
  _phase(kernel,'decode')
  assert kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  assert not kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].decode='auto'
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))

  model=_model()
  result=configure(model,prefill='off',decode='on',precision='int8',fuse_projections=True,
                   fused_erf_gelu_ffn=True,fused_erf_gelu_decode=False)
  kernel=model.blk[0].fused_erf_gelu_ffn
  _phase(kernel,'decode')
  assert result['fused_erf_gelu_decode_blocks']==0
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))


@pytest.mark.parametrize('activation,field,value',[
  ('silu',None,None),('gelu_tanh',None,None),('gelu','swiglu_limit',1.0),('gelu','swiglu_up_bias',.25),
])
def test_unsupported_activation_settings_do_not_install(monkeypatch,activation,field,value):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  model=_model(activation=activation)
  if field is not None: model.blk[0].config=replace(model.blk[0].config,**{field:value})
  result=_configure(model,fused_erf_gelu_ffn=True)
  assert result['fused_erf_gelu_ffn_blocks']==0
  assert not hasattr(model.blk[0],'fused_erf_gelu_ffn')


@pytest.mark.parametrize('kind',['gate_up_bias','down_bias','input_scale'])
def test_unsupported_bias_or_calibration_does_not_install(monkeypatch,kind):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  model=_model(down_bias=kind=='down_bias')
  block=model.blk[0]
  input_scales=None
  if kind=='gate_up_bias':
    block.ffn_gate.bias=Tensor.zeros(block.config.hidden_dim)
    block.ffn_up.bias=Tensor.zeros(block.config.hidden_dim)
  if kind=='input_scale':
    scale=Tensor.ones(block.config.hidden_dim,device='CPU',dtype=dtypes.float).realize()
    input_scales={'blk.0.ffn_down':scale}
  result=_configure(model,fused_erf_gelu_ffn=True,**({'input_scales':input_scales} if input_scales else {}))
  assert result['fused_erf_gelu_ffn_blocks']==0
  assert not hasattr(block,'fused_erf_gelu_ffn')


def test_decode_auto_member_prevents_decode_selection_but_keeps_prefill(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',_fake_prepare)
  original=RockchipProjections.__init__
  def mixed_decode(group,layers,*args,**kwargs):
    original(group,layers,*args,**kwargs)
    if len(layers)==2: layers[0].decode='auto'
  monkeypatch.setattr(RockchipProjections,'__init__',mixed_decode)
  model=_model()
  result=configure(model,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_erf_gelu_ffn=True,fused_erf_gelu_decode=True)
  kernel=model.blk[0].fused_erf_gelu_ffn
  assert result['fused_erf_gelu_ffn_blocks']==1
  assert result['fused_erf_gelu_decode_blocks']==0
  _phase(kernel,'decode')
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))


def test_decode_option_validation_is_strict_and_precedes_preparation(monkeypatch):
  calls=[]
  monkeypatch.setattr(RockchipInt8Linear,'prepare',lambda layer:calls.append(layer))
  with pytest.raises(ValueError): configure(SimpleNamespace(),fused_erf_gelu_ffn=1)
  with pytest.raises(ValueError): configure(SimpleNamespace(),fused_erf_gelu_decode=1)
  with pytest.raises(ValueError): configure(SimpleNamespace(),fused_erf_gelu_decode=True)
  with pytest.raises(ValueError): configure(_model(),fused_erf_gelu_ffn=True,precision='fp16')
  with pytest.raises(ValueError): configure(_model(),fused_erf_gelu_ffn=True,fused_erf_gelu_decode=True,decode='off')
  with pytest.raises(ValueError): RockchipInt8ErfGeluFeedForward(SimpleNamespace(),SimpleNamespace(),decode=1)
  assert calls==[]


@pytest.mark.parametrize('flags',[
  ['--device','CL','--precision','int8','--fused-erf-gelu-ffn'],
  ['--device','CPU','--precision','int8','--prefill','off','--decode','off','--fused-erf-gelu-ffn'],
  ['--device','CPU','--precision','int8','--fused-erf-gelu-decode'],
])
def test_cli_erf_ffn_flags_reject_unsupported_phases_before_checkpoint_io(monkeypatch,tmp_path,flags):
  from extra.rockchip import benchmark_inference
  argv=['benchmark_inference','--checkpoint',str(tmp_path/'missing.gguf'),'--input',str(tmp_path/'missing.ids'),
        '--report',str(tmp_path/'report.json'),*flags]
  monkeypatch.setattr(sys,'argv',argv)
  with pytest.raises(ValueError,match='erf GELU'): benchmark_inference.main()
