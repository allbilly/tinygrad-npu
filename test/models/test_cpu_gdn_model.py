"""Run the CPU scan through existing independent full-block reference checks."""
from test.models import test_gated_delta_layout as reference_layout
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_recurrent import configure, gated_delta_scan


class TestCPUGatedDeltaModel(reference_layout.TestGatedDeltaLayout):
  def fixture(self, value_heads, convert=True):
    block,weights,ssm=super().fixture(value_heads,convert)
    block.gated_delta_scan=gated_delta_scan
    return block,weights,ssm

  def test_hybrid_bulk_logits_match_serial_prefixes_continuation_and_reset(self):
    ssm=SSMConfig(3,8,2,4,32)
    config=TransformerConfig(2,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=24,ssm=ssm,ssm_layers=(True,False))
    ordinary,bulk=Transformer(config),Transformer(config)
    rng=np.random.default_rng(3513)
    for name,p in nn.state.get_state_dict(ordinary).items():
      values=rng.normal(0,.12,p.shape).astype(np.float32)
      if 'norm.weight' in name:values+=1
      if name.endswith('ssm_a'):values=-rng.uniform(.5,2,p.shape).astype(np.float32)
      p.replace(Tensor(values.astype(np.float16),device='CPU').realize())
    nn.state.load_state_dict(bulk,{name:value.clone().realize() for name,value in nn.state.get_state_dict(ordinary).items()},verbose=False)
    self.assertEqual(configure(bulk),1)
    tokens=[2,5,11,17,7,31,29,19,3]
    start=0
    for end in (1,6,7,9):
      for index in range(start,end):expected=ordinary.logits(Tensor([[tokens[index]]],device='CPU'),index).numpy()
      actual=bulk.logits(Tensor([tokens[start:end]],device='CPU'),start).numpy()
      np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-3)
      np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
      start=end
    for index in range(3):expected=ordinary.logits(Tensor([[tokens[index]]],device='CPU'),index).numpy()
    actual=bulk.logits(Tensor([tokens[:3]],device='CPU'),0).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-3)
