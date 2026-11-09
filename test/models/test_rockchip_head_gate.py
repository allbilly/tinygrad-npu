"""Head-gate optimization is explicit and removable without NPU preparation."""
from tinygrad import Tensor,Device
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_cpu import head_gate
import pytest


def model(gated):
  return Transformer(TransformerConfig(1,64,128,4,2,1e-6,32,16,10000.,16,16,max_context=32,attn_head_gate=gated))


@pytest.mark.parametrize('gated',[False,True])
def test_enable_disable_only_gated_blocks(gated):
  before=set(Device._opened_devices)
  m=model(gated)
  placement=configure(m,prefill='off',decode='off',cpu_head_gate=True)
  assert placement['cpu_head_gate_blocks']==int(gated)
  if gated:
    assert m.blk[0].cpu_head_gate is head_gate
    x=Tensor.randn(1,17,64,device='CPU').realize()
    attention=Tensor.randn(1,4,17,16,device='CPU').realize()
    expected=(attention*m.blk[0].attn_gate(x).sigmoid().transpose(1,2).unsqueeze(-1)).numpy()
    assert m.blk[0]._apply_head_gate(attention,x).numpy().tobytes()==expected.tobytes()
  placement=configure(m,prefill='off',decode='off',cpu_head_gate=False)
  assert placement['cpu_head_gate_blocks']==0 and not hasattr(m.blk[0],'cpu_head_gate')
  assert not (set(Device._opened_devices)-before).intersection({'CL','ROCKCHIP'})


@pytest.mark.parametrize('value',[None,0,1,'on'])
def test_nonboolean_option_fails_before_preparation(value):
  with pytest.raises(ValueError,match='CPU head gate selection must be a bool'):
    configure(None,cpu_head_gate=value)
