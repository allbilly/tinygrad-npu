from types import SimpleNamespace
import unittest
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, nn
from tinygrad.llm.model import ExpertWeights
from tinygrad.llm.kernels.rockchip_experts import CPUExpertWeights, configure


def make_experts(bias=False):
  layer = ExpertWeights(4, 64, 32, bias=bias)
  layer.weight = (Tensor.arange(4*32*64).reshape(4,32,64) % 31 - 15).cast(dtypes.half).to('CPU',force=True).realize()
  if bias: layer.bias = (Tensor.arange(4*32).reshape(4,32) % 9 - 4).cast(dtypes.half).to('CPU',force=True).realize()
  return layer


class TestCPUExpertWeights(unittest.TestCase):
  def test_prepare_preserves_logical_weight_and_packs(self):
    original = make_experts()
    logical, values = original.weight, original.weight.numpy().copy()
    wrapper = CPUExpertWeights(original)
    prepared = {}
    wrapper.prepare(prepared)
    self.assertIs(original.weight, logical)
    self.assertIs(wrapper.weight, logical)
    self.assertEqual(original.weight.shape, (4,32,64))
    self.assertTrue((original.weight.numpy() == values).all())
    packed = wrapper._packed()
    self.assertEqual(packed.shape, (4,2,64,16))
    self.assertEqual(set(nn.state.get_state_dict(wrapper)), {'weight'})
    self.assertEqual(set(nn.state.get_state_dict(original)), {'weight'})
    # Fallback retains the original layout; runtime packing is a separate copy.
    self.assertIsNot(original.weight.uop.base, packed.uop.base)

  def test_forward_matches_reference_and_fallback(self):
    original = make_experts(bias=True)
    wrapper = CPUExpertWeights(original)
    wrapper.prepare({})
    selected = Tensor([[ [0,2], [2,1], [3,0] ]], dtype=dtypes.int)
    x = (Tensor.arange(3*64).reshape(1,3,1,64) % 17 - 8).cast(dtypes.float).to('CPU').realize()
    expected = original(selected, x).numpy()
    actual = wrapper(selected, x).numpy()
    self.assertEqual(actual.shape, (1,3,2,32))
    self.assertTrue((abs(actual-expected) < 2e-3).all())
    half_x = x.cast(dtypes.half).expand(1,3,2,64).contiguous()
    fallback = wrapper(selected, half_x).numpy()
    reference = original(selected, half_x).numpy()
    self.assertTrue((fallback == reference).all())

  def test_configure_enable_disable_and_invalid(self):
    original = make_experts()
    block = SimpleNamespace(ffn_gate_exps=original, ffn_up_exps=make_experts(), ffn_down_exps=None)
    model = SimpleNamespace(blk=[block])
    self.assertEqual(configure(model, enabled=True), 2)
    self.assertIsInstance(block.ffn_gate_exps, CPUExpertWeights)
    self.assertIsInstance(block.ffn_up_exps, CPUExpertWeights)
    self.assertEqual(set(nn.state.get_state_dict(block)), {'ffn_gate_exps.weight','ffn_up_exps.weight'})
    wrapped_gate = block.ffn_gate_exps
    self.assertEqual(configure(model, enabled=False), 0)
    self.assertIs(block.ffn_gate_exps, original)
    self.assertIs(wrapped_gate._original(), original)

    invalid = ExpertWeights(4, 63, 32)
    invalid.weight = invalid.weight.cast(dtypes.half)
    untouched = make_experts()
    model = SimpleNamespace(blk=[SimpleNamespace(ffn_gate_exps=invalid, ffn_up_exps=untouched)])
    self.assertEqual(configure(model, enabled=True), 0)
    self.assertIs(model.blk[0].ffn_gate_exps, invalid)
    self.assertIs(model.blk[0].ffn_up_exps, untouched)
    with self.assertRaises(ValueError): configure(model, enabled=1)

  def test_dynamic_rows_jit_and_no_rockchip_open(self):
    original = make_experts()
    wrapper = CPUExpertWeights(original)
    wrapper.prepare({})
    rowvar = Variable('rows', 1, 17)
    selected = Tensor([[ [0,1] ]], dtype=dtypes.int).expand(1,17,2).contiguous()
    x = Tensor.ones(1,17,1,64, dtype=dtypes.float,device='CPU').contiguous().realize()
    run = TinyJit(lambda s,a,n: wrapper(s[:,:n], a[:,:n]).pad_to((1,17,2,32)).contiguous().realize())
    before = set(Device._opened_devices)
    for n, ids in ((1, (0,1)), (7, (2,2)), (17, (3,0)), (3, (1,1))):
      selected = Tensor(ids, dtype=dtypes.int).reshape(1,1,2).expand(1,17,2).contiguous().realize()
      rows = rowvar.bind(n)
      actual = run(selected, x, rows).numpy()
      reference = original(selected[:,:rows], x[:,:rows]).pad_to((1,17,2,32)).contiguous().numpy()
      self.assertTrue((abs(actual-reference) < 2e-3).all())
    self.assertFalse(any(device.startswith('ROCKCHIP') for device in Device._opened_devices-before))


if __name__ == '__main__': unittest.main()
