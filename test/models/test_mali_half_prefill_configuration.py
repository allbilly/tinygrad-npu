"""Configuration ownership and conservative fallbacks for bounded half prefill."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kernels import mali, mali_half_prefill
from test.models.test_mali_gdn_gate_up import fixture
from test.models.test_mali_output_packing import fixture as cpu_fixture


class TestMaliHalfPrefillConfiguration(unittest.TestCase):
  def setUp(self):
    fake=SimpleNamespace(device_name='Mali-LODX r0p0',driver_version='2.1',device_exts=())
    device=patch.object(mali,'Device',{'CL':fake})
    device.start()
    self.addCleanup(device.stop)
    scan=patch('tinygrad.llm.kernels.mali_recurrent.supported',return_value=True)
    scan.start()
    self.addCleanup(scan.stop)

  def test_default_enable_disable_and_recurrent_disable_preserve_weights(self):
    model=fixture(1)
    weight=model.blk[0].attn_qkv.weight
    with patch.object(mali_half_prefill,'prepare',return_value={'prepared_kernels':2}) as prepare:
      baseline=mali.configure(model,pack_weights=False,recurrent_scan=True)
      prepare.assert_not_called()
      self.assertFalse(baseline['bounded_half_prefill'])
      self.assertFalse(model.blk[0].attn_qkv.bounded_half_prefill)
      enabled=mali.configure(model,pack_weights=False,recurrent_scan=True,bounded_half_prefill=True)
      self.assertEqual(enabled['half_prefill_prepared_kernels'],2)
      self.assertTrue(model.blk[0].attn_qkv.bounded_half_prefill)
      self.assertIs(model.blk[0].attn_qkv.weight,weight)
      mali.configure(model,pack_weights=False,recurrent_scan=True)
      self.assertFalse(model.blk[0].attn_qkv.bounded_half_prefill)
      mali.configure(model,pack_weights=False,recurrent_scan=True,bounded_half_prefill=True)
      mali.configure(model,pack_weights=False,recurrent_scan=False)
      self.assertFalse(model.blk[0].attn_qkv.bounded_half_prefill)

  def test_unqualified_generated_source_keeps_ordinary_path(self):
    model=fixture(1)
    with patch.object(mali_half_prefill,'prepare',return_value={'prepared_kernels':0}):
      result=mali.configure(model,pack_weights=False,recurrent_scan=True,bounded_half_prefill=True)
    self.assertEqual(result['half_prefill_prepared_kernels'],0)
    self.assertFalse(model.blk[0].attn_qkv.bounded_half_prefill)

  def test_wrong_type_and_cpu_request_reject_before_source_preparation(self):
    with patch.object(mali_half_prefill,'prepare') as prepare:
      with self.assertRaisesRegex(TypeError,'must be a bool'):
        mali.configure(fixture(1),bounded_half_prefill=1)
      with self.assertRaisesRegex(ValueError,'Mali OpenCL'):
        mali.configure(cpu_fixture(device='CPU',output_device='CPU'),bounded_half_prefill=True)
      prepare.assert_not_called()

  def test_direct_unprepared_or_wrong_shape_call_never_opens_a_device(self):
    with patch.object(mali_half_prefill,'Device',{}) as _device:
      weight=Tensor.empty(2048,2048,device='CL',dtype=dtypes.half)
      x=Tensor.empty(1,128,2048,device='CL',dtype=dtypes.half)
      self.assertIsNone(mali_half_prefill.bounded_linear(x,weight))
      self.assertIsNone(mali_half_prefill.bounded_linear(x.float(),weight))
      self.assertIsNone(mali_half_prefill.bounded_linear(x,Tensor.empty(2048,2048,device='CL')))
      self.assertIsNone(mali_half_prefill.bounded_linear(Tensor.empty(1,64,2048,device='CL',dtype=dtypes.half),weight))

  def test_bias_and_float_inputs_keep_their_original_dispatch(self):
    source=nn.Linear.__new__(nn.Linear)
    source.weight=Tensor.empty(2048,2048,device='CL',dtype=dtypes.half)
    source.bias=Tensor.empty(2048,device='CL',dtype=dtypes.half)
    layer=mali.MaliLinear(source,prepare_gemm=False)
    layer.bounded_half_prefill=True
    half=Tensor.empty(1,128,2048,device='CL',dtype=dtypes.half)
    with patch.object(mali_half_prefill,'bounded_linear',side_effect=AssertionError('fast path must be bypassed')):
      self.assertEqual(layer(half).shape,(1,128,2048))
      layer.bias=source.bias=None
      result=Tensor.empty(1,128,2048,device='CL')
      with patch('tinygrad.llm.kernels.mali_linear.gemm',return_value=result) as gemm:
        self.assertIs(layer(half.float()),result)
        gemm.assert_called_once()


if __name__=='__main__':unittest.main()
