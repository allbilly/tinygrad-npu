"""Placement selects only supported SiLU SwiGLU FFNs before JIT capture."""
from types import SimpleNamespace
import pytest
from tinygrad import Tensor,dtypes
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8SwiGLUFeedForward


def fake_prepare(layer):
  layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
  layer.scales=[Tensor.ones(layer.columns,device='CPU',dtype=dtypes.float)]


def model(activation='silu',width=4864):
  return Transformer(TransformerConfig(1,512,width,8,2,1e-6,32,64,10000.,64,64,max_context=64,ffn_activation=activation))


def test_installation_count_prefill_selection_decode_fallback_and_disable(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model()
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True)
  assert result['fused_swiglu_ffn_blocks']==1
  kernel=m.blk[0].fused_swiglu_ffn
  assert isinstance(kernel,RockchipInt8SwiGLUFeedForward)
  assert kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].phase='decode'
  assert not kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].phase=None
  disabled=configure(m,prefill='off',decode='off',precision='int8',fused_swiglu_ffn=False)
  assert disabled['fused_swiglu_ffn_blocks']==0 and not hasattr(m.blk[0],'fused_swiglu_ffn')


@pytest.mark.parametrize('activation',['gelu_tanh','gelu'])
def test_other_activations_do_not_install_swiglu_kernel(monkeypatch,activation):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model(activation)
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True)
  assert result['fused_swiglu_ffn_blocks']==0


@pytest.mark.parametrize('legacy',[False,True])
def test_small_hidden_width_retains_the_selected_rounding_contract(monkeypatch,legacy):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model(width=3072)
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True,fused_ffn=legacy)
  assert result['fused_swiglu_ffn_blocks']==1
  assert m.blk[0].fused_swiglu_ffn.activation==('silu_div' if legacy else 'silu')


def test_invalid_option_and_precision_fail_before_preparation():
  with pytest.raises(ValueError):configure(SimpleNamespace(),fused_swiglu_ffn=1)
  with pytest.raises(ValueError):configure(SimpleNamespace(),fused_swiglu_ffn=True,precision='fp16')


@pytest.mark.parametrize('activation,option,attribute',[
  ('silu','fused_swiglu_ffn','fused_swiglu_ffn'),
  ('gelu_tanh','fused_gelu_ffn','fused_gelu_ffn'),
])
def test_raw_ffn_receives_only_explicit_prefill_pool(monkeypatch,activation,option,attribute):
  from tinygrad.llm.kernels import rockchip_pool
  pool=SimpleNamespace(active=False,prefill_address=None)
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:pool)
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  for enabled in (True,False):
    m=model(activation)
    result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                     cpu_pool=True,cpu_attention=True,cpu_prefill_pool=enabled,**{option:True})
    assert result[attribute+'_blocks']==1
    assert getattr(m.blk[0],attribute).pool is (pool if enabled else None)


@pytest.mark.parametrize('width',[3072,6144])
@pytest.mark.parametrize('dtype',[dtypes.half,dtypes.float])
def test_down_bias_admits_packed_prefill_decode_but_never_biased_k_parts(monkeypatch,width,dtype):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  m=model(width=width)
  m.blk[0].ffn_down.bias=Tensor.ones(512,device='CPU',dtype=dtype)
  result=configure(m,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_swiglu_ffn=True,fused_swiglu_decode=True)
  assert result['fused_swiglu_ffn_blocks']==result['fused_swiglu_decode_blocks']==1
  kernel=m.blk[0].fused_swiglu_ffn
  assert kernel.enabled(Tensor.zeros(1,17,512,device='CPU',dtype=dtypes.float))
  kernel.projections.layers[0].phase='decode'
  assert kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  kernel.down.bias=Tensor.ones(511,device='CPU',dtype=dtype)
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  kernel.down.bias=Tensor.ones(512,device='CPU',dtype=dtypes.int)
  assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
  kernel.down.bias=Tensor.ones(512,device='CPU',dtype=dtype)
  if hasattr(kernel.down,'parts'):
    kernel.down.parts[0].bias=Tensor.ones(512,device='CPU',dtype=dtype)
    assert not kernel.enabled(Tensor.zeros(1,1,512,device='CPU',dtype=dtypes.float))
