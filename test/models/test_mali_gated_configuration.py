"""Allocation-free ownership checks for the explicit gated-attention profile."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, nn, dtypes
from tinygrad.llm.kernels import mali
from tinygrad.llm.kernels.mali_attention import SharedAttention
from tinygrad.llm.model import TransformerBlock
from test.models.test_mali_output_packing import fixture


def model_fixture():
  model=fixture(depth=32)
  block=TransformerBlock.__new__(TransformerBlock)
  block.config=SimpleNamespace(head_dim=16,n_heads=2,n_kv_heads=1,max_context=16,
    sliding_window=0,attn_sinks=False,attn_output_gate=True,attn_head_gate=False,attn_logit_softcap=0)
  block.prefill_attention=block.decode_attention=None
  for name,rows in (('attn_q',64),('attn_k',16),('attn_v',16),('attn_output',32)):
    linear=nn.Linear.__new__(nn.Linear)
    linear.weight,linear.bias=Tensor.empty(rows,32,device='CL',dtype=dtypes.half),None
    setattr(block,name,linear)
  model.blk=[block]
  return model,block


class TestGatedAttentionConfiguration(unittest.TestCase):
  def setUp(self):
    fake=SimpleNamespace(device_name='Mali-LODX r0p0',driver_version='2.1',device_exts=())
    device=patch.object(mali,'Device',{'CL':fake})
    device.start()
    self.addCleanup(device.stop)
    prepare=patch.object(SharedAttention,'prepare',autospec=True)
    self.prepare=prepare.start()
    self.addCleanup(prepare.stop)

  def test_explicit_profile_shares_attention_and_releases_only_its_hooks(self):
    model,block=model_fixture()
    baseline=mali.configure(model,pack_weights=False)
    self.assertEqual(baseline['attention_blocks'],0)
    self.assertEqual(baseline['gated_attention_blocks'],0)
    self.assertIsNone(block.prefill_attention)
    requested=mali.configure(model,pack_weights=False,gated_attention=True,consistent_decode=True)
    self.assertEqual(requested['attention_blocks'],1)
    self.assertEqual(requested['gated_attention_blocks'],1)
    attention=block.prefill_attention
    self.assertIsInstance(attention,SharedAttention)
    self.assertIs(block.decode_attention,attention)
    self.assertTrue(attention.consistent_decode)
    self.assertEqual({call.args[4] for call in self.prepare.call_args_list},{1,8,16})
    disabled=mali.configure(model,pack_weights=False)
    self.assertEqual(disabled['gated_attention_blocks'],0)
    self.assertIsNone(block.prefill_attention)
    self.assertIsNone(block.decode_attention)

  def test_reconfiguration_preserves_a_replacement_owned_by_the_caller(self):
    model,block=model_fixture()
    mali.configure(model,pack_weights=False,gated_attention=True)
    def replacement(*args):return None
    block.prefill_attention=replacement
    mali.configure(model,pack_weights=False)
    self.assertIs(block.prefill_attention,replacement)
    self.assertIsNone(block.decode_attention)

  def test_request_rejects_wrong_types_disabled_fusion_and_unsupported_device(self):
    model,_=model_fixture()
    with self.assertRaisesRegex(TypeError,'must be a bool'):
      mali.configure(model,gated_attention=1)
    with self.assertRaisesRegex(ValueError,'requires fused_attention'):
      mali.configure(model,gated_attention=True,fused_attention=False)
    cpu=fixture(device='CPU',output_device='CPU')
    with self.assertRaisesRegex(ValueError,'Mali OpenCL'):
      mali.configure(cpu,gated_attention=True)
    self.prepare.assert_not_called()


if __name__=='__main__':unittest.main()
