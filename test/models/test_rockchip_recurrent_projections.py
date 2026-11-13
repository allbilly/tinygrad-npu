"""CPU-only projection ordering, state continuation and configuration coverage."""
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.kernels.rockchip import configure,RockchipProjections
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
from test.models.test_rockchip_placement import _cpu_gdn_model


def test_joint_projection_hook_preserves_convolution_and_recurrent_continuation():
  Tensor.manual_seed(3504)
  model=_cpu_gdn_model(context=64)
  block=model.blk[0]
  inputs=[Tensor.randn(1,rows,16,device='CPU').realize() for rows in (17,1,7)]
  block._init_state(inputs[0])
  saved=[]
  position=0
  for x in inputs:
    output=block._attention(x,position).numpy().copy()
    saved.append((output,block.conv_state.numpy().copy(),block.recurrent_state.numpy().copy()))
    position+=x.shape[1]
  block.conv_state.assign(Tensor.zeros_like(block.conv_state)).realize()
  block.recurrent_state.assign(Tensor.zeros_like(block.recurrent_state)).realize()
  calls=[]
  def project(x):
    calls.append(x.shape)
    return block.attn_qkv(x),block.attn_gate(x),block.ssm_beta(x),block.ssm_alpha(x)
  block.recurrent_projections=project
  position=0
  for x,(expected,conv,state) in zip(inputs,saved):
    actual=block._attention(x,position).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(block.conv_state.numpy().view(np.uint32),conv.view(np.uint32))
    np.testing.assert_array_equal(block.recurrent_state.numpy().view(np.uint32),state.view(np.uint32))
    position+=x.shape[1]
  assert calls==[x.shape for x in inputs]


def test_recurrent_projection_selection_is_explicit_and_ordered(monkeypatch):
  def prepare(layer):
    layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype='int8')]
    layer.scales=[Tensor.ones(layer.columns,device='CPU')]
  monkeypatch.setattr(RockchipInt8Linear,'prepare',prepare)
  model=_cpu_gdn_model()
  result=configure(model,prefill='on',decode='on',precision='int8',fuse_projections=True,fused_recurrent_projections=True)
  block=model.blk[0]
  group=block.recurrent_projections
  assert isinstance(group,RockchipProjections)
  assert group.layers==(block.attn_qkv,block.attn_gate,block.ssm_beta,block.ssm_alpha)
  assert result['fused_recurrent_projection_blocks']==1


@pytest.mark.parametrize('options',[
  {'fused_recurrent_projections':1},
  {'fused_recurrent_projections':True,'fuse_projections':False,'precision':'int8'},
  {'fused_recurrent_projections':True,'fuse_projections':True,'precision':'fp16'},
])
def test_invalid_selection_fails_before_allocations(options):
  with pytest.raises(ValueError):configure(SimpleNamespace(),**options)
