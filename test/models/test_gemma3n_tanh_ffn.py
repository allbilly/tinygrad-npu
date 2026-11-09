"""Gemma3n fusion respects finite sparsity and explicit decode selection."""
import math
from types import SimpleNamespace
import pytest
from tinygrad import Device,Tensor,dtypes
from tinygrad.llm.gemma3n import Gemma3nTransformer,Gemma3nBlock
from tinygrad.llm.model import TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear


def fake_prepare(layer):
  layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
  layer.scales=[Tensor.ones(layer.columns,device='CPU',dtype=dtypes.float)]


def test_only_nonsparse_blocks_fuse_decode_and_uninstall(monkeypatch):
  monkeypatch.setattr(RockchipInt8Linear,'prepare',fake_prepare)
  config=TransformerConfig(2,512,512,8,2,1e-6,32,64,10000.,64,64,max_context=64)
  model=Gemma3nTransformer(config,(512,512),(False,False),0,4,32,4,0,4,(.7,-math.inf),7)
  result=configure(model,prefill='on',decode='on',precision='int8',fuse_projections=True,
                   fused_gelu_ffn=True,fused_gelu_decode=True)
  assert result['fused_gelu_ffn_blocks']==result['fused_gelu_decode_blocks']==1
  assert not hasattr(model.blk[0],'fused_gelu_ffn')
  kernel=model.blk[1].fused_gelu_ffn
  kernel.projections.layers[0].phase='prefill'
  assert kernel.enabled(Tensor.empty(1,17,512,device='CPU'))
  kernel.projections.layers[0].phase='decode'
  assert kernel.enabled(Tensor.empty(1,1,512,device='CPU'))
  assert not kernel.enabled(Tensor.empty(1,17,512,device='CPU'))
  disabled=configure(model,prefill='off',decode='off',precision='int8')
  assert disabled['fused_gelu_ffn_blocks']==disabled['fused_gelu_decode_blocks']==0
  assert all(not hasattr(block,'fused_gelu_ffn') for block in model.blk)
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)


def test_finite_sparse_block_cannot_bypass_threshold_with_a_fused_hook():
  config=TransformerConfig(1,32,64,4,1,1e-6,40,8,10000.,8,8,max_context=16)
  block=Gemma3nBlock(config,4,False,4,0,4,.7)
  class Trap:
    def enabled(self,x):raise AssertionError('Finite sparse block must preserve the threshold')
  block.fused_gelu_ffn=Trap()
  block.ffn_gate=block.ffn_up=block.ffn_down=lambda x:x
  out=block._sparse_ffn(Tensor.ones(1,1,32,device='CPU')).numpy()
  assert out.shape==(1,1,32)


@pytest.mark.parametrize('options',[
  {'fused_gelu_decode':1},{'fused_gelu_decode':True},
  {'fused_gelu_ffn':True,'fused_gelu_decode':True,'precision':'int8','decode':'off'},
  {'cpu_exact_rows':1},{'cpu_sparse_lookup':1},{'cpu_exact_rows':True},
  {'cpu_sparse_lookup':True},{'cpu_exact_rows':True,'native_replay':True},
])
def test_invalid_options_fail_before_device_preparation(options):
  with pytest.raises(ValueError):configure(SimpleNamespace(),**options)
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)
