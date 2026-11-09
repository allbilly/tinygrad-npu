"""Placement must select only supported GELU-tanh FFNs before JIT capture."""
from types import SimpleNamespace
import pytest
from tinygrad import Tensor,dtypes
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8GeluFeedForward


def fake_prepare(layer):
  layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
  layer.scales=[Tensor.ones(layer.columns,device='CPU',dtype=dtypes.float)]


def model(activation='gelu_tanh'):
  return Transformer(TransformerConfig(1,512,9216,8,2,1e-6,32,64,10000.,64,64,max_context=64,ffn_activation=activation))


def test_installation_count_prefill_selection_decode_fallback_and_disable(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model()
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_gelu_ffn=True)
  assert result['fused_gelu_ffn_blocks']==1
  kernel=m.blk[0].fused_gelu_ffn
  assert isinstance(kernel,RockchipInt8GeluFeedForward)
  assert kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].phase='decode'
  assert not kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].phase=None
  disabled=configure(m,prefill='off',decode='off',precision='int8',fused_gelu_ffn=False)
  assert disabled['fused_gelu_ffn_blocks']==0 and not hasattr(m.blk[0],'fused_gelu_ffn')


@pytest.mark.parametrize('activation',['silu','gelu'])
def test_other_activations_do_not_install_tanh_gelu_kernel(monkeypatch,activation):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model(activation)
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_gelu_ffn=True)
  assert result['fused_gelu_ffn_blocks']==0


def test_invalid_option_and_precision_fail_before_preparation():
  with pytest.raises(ValueError):configure(SimpleNamespace(),fused_gelu_ffn=1)
  with pytest.raises(ValueError):configure(SimpleNamespace(),fused_gelu_ffn=True,precision='fp16')


@pytest.mark.parametrize('enabled',[False,True])
def test_gemma4_sdk_prefill_is_explicit_and_decode_falls_back(monkeypatch,enabled):
  from tinygrad.llm.gemma4 import Gemma4Transformer
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  config=TransformerConfig(1,1536,6144,8,1,1e-6,32,512,10000.,512,512,max_context=512)
  m=Gemma4Transformer(config,(6144,),(False,),256,0,0,0,(),128)
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,split_down=True,
                   fused_gelu_ffn=True,fused_gelu_sdk_prefill=enabled)
  assert result['fused_gelu_sdk_prefill_blocks']==int(enabled)
  if enabled:
    f=m.blk[0].fused_gelu_ffn
    assert f.sdk_prefill and f.enabled(Tensor.zeros(1,128,1536,device='CPU'))
    f.projections.layers[0].phase='decode'
    assert not f.enabled(Tensor.zeros(1,1,1536,device='CPU'))
  else:assert not hasattr(m.blk[0],'fused_gelu_ffn')


def test_sdk_tanh_gelu_rejects_invalid_phase_selection_before_preparation():
  for kwargs in ({'fused_gelu_sdk_prefill':1},{'fused_gelu_sdk_prefill':True},
                 {'fused_gelu_sdk_prefill':True,'fused_gelu_ffn':True,'precision':'fp16'},
                 {'fused_gelu_sdk_prefill':True,'fused_gelu_ffn':True,'precision':'int8','prefill':'off'},
                 {'fused_gelu_sdk_prefill':True,'fused_gelu_ffn':True,'fused_gelu_decode':True,'precision':'int8','prefill':'on'}):
    with pytest.raises(ValueError):configure(SimpleNamespace(),**kwargs)
