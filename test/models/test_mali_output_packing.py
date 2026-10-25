"""Allocation-free configuration/ownership gates for optional output packing."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kernels.mali import MaliLinear, configure


class ScaledLinear:
  def __init__(self,source,scale=2.5):self.source,self.scale=source,scale
  def __call__(self,x):return self.source(x)*self.scale


class Packed:
  def __init__(self,weight):
    self.shape=((weight.shape[0]+31)//32,weight.shape[1],32)
    self.dtype,self.device=weight.dtype,weight.device
  def realize(self):return self
  def nbytes(self):return self.shape[0]*self.shape[1]*self.shape[2]*self.dtype.itemsize


def fixture(*,wrapped=False,device='CL',output_device='CL',depth=8,dtype=dtypes.half):
  embedding=nn.Embedding.__new__(nn.Embedding)
  embedding.weight=Tensor.empty(16,depth,device=device,dtype=dtypes.half)
  head=nn.Linear.__new__(nn.Linear)
  head.weight,head.bias=Tensor.empty(16,depth,device=output_device,dtype=dtype),None
  return SimpleNamespace(token_embd=embedding,output=ScaledLinear(head) if wrapped else head,
                         config=SimpleNamespace(max_context=16),max_context=16,blk=[])


class TestMaliOutputPacking(unittest.TestCase):
  def setUp(self):
    self.device=patch('tinygrad.llm.kernels.mali.Device',{'CL':SimpleNamespace(device_name='Mali-LODX r0p0',driver_version='2.1',device_exts=())})
    self.device.start()
    self.addCleanup(self.device.stop)
    self.packing=patch('tinygrad.llm.kernels.mali_linear.pack_weight',side_effect=Packed)
    self.pack=self.packing.start()
    self.addCleanup(self.packing.stop)

  def test_default_then_repeat_enable_disable_keeps_one_shared_head_and_packed_copy(self):
    model=fixture()
    weight=model.output.weight
    plain=configure(model,pack_weights=False)
    borrowed=model.output
    self.assertIsInstance(borrowed,MaliLinear)
    self.assertEqual((plain['pack_output'],plain['packed_output_head'],plain['packed_output_bytes']),(False,False,0))
    self.pack.assert_not_called()
    enabled=configure(model,pack_weights=False,pack_output=True)
    packed=borrowed.packed_weight
    self.assertEqual((enabled['packed_output_head'],enabled['packed_output_bytes']),(True,512))
    self.assertIs(model.output,borrowed)
    self.assertIs(borrowed.weight,weight)
    self.assertIs(borrowed.source.weight,weight)
    configure(model,pack_weights=False,pack_output=True)
    self.assertIs(borrowed.packed_weight,packed)
    self.assertEqual(self.pack.call_count,1)
    disabled=configure(model,pack_weights=True,pack_output=False)
    self.assertEqual((disabled['packed_output_head'],disabled['packed_output_bytes']),(False,0))
    self.assertIsNone(borrowed.packed_weight)
    self.assertIs(model.output,borrowed)
    self.assertIs(borrowed.weight,weight)

  def test_scaled_wrapper_preserves_outer_and_inner_identity_and_reuses_pack(self):
    model=fixture(wrapped=True)
    wrapper,weight=model.output,model.output.source.weight
    configure(model,pack_weights=False,pack_output=True)
    inner,packed=wrapper.source,wrapper.source.packed_weight
    self.assertIsInstance(inner,MaliLinear)
    configure(model,pack_weights=False,pack_output=True)
    self.assertIs(model.output,wrapper)
    self.assertIs(wrapper.source,inner)
    self.assertIs(inner.packed_weight,packed)
    self.assertIs(inner.weight,weight)
    self.assertEqual(wrapper.scale,2.5)
    self.assertEqual(self.pack.call_count,1)
    configure(model,pack_weights=False)
    self.assertIsNone(inner.packed_weight)
    self.assertIs(model.output,wrapper)
    self.assertEqual(wrapper.scale,2.5)

  def test_stale_weight_bypasses_pack_then_reconfiguration_rebuilds_once(self):
    model=fixture()
    configure(model,pack_weights=False,pack_output=True)
    head,weight=model.output,model.output.weight
    previous=head.packed_weight
    weight.replace(Tensor.empty(*weight.shape,device='CL',dtype=dtypes.half))
    out=Tensor.empty(1,8,16,device='CPU',dtype=dtypes.float)
    with patch('tinygrad.llm.kernels.mali_linear.gemm',return_value=out) as gemm:
      head(Tensor.empty(1,8,8,device='CL',dtype=dtypes.float))
      self.assertIsNone(gemm.call_args.args[2])
    def replacement(value):
      self.assertIsNone(head.packed_weight)  # release old ownership before rebuilding
      return Packed(value)
    self.pack.side_effect=replacement
    configure(model,pack_weights=False,pack_output=True)
    self.assertIsNot(head.packed_weight,previous)
    self.assertIs(head.packed_source,weight.uop)
    self.assertEqual(self.pack.call_count,2)
    configure(model,pack_weights=False,pack_output=True)
    self.assertEqual(self.pack.call_count,2)

  def test_single_row_retains_gemv_with_packing_enabled(self):
    model=fixture()
    configure(model,pack_weights=False,pack_output=True)
    result=Tensor.empty(1,1,16,device='CPU',dtype=dtypes.float)
    with patch('tinygrad.llm.kernels.mali_linear.gemv',return_value=result) as gemv,\
         patch('tinygrad.llm.kernels.mali_linear.gemm') as gemm:
      self.assertIs(model.output(Tensor.empty(1,1,8,device='CL',dtype=dtypes.float)),result)
      self.assertIs(gemv.call_args.args[1],model.output.weight)
      gemm.assert_not_called()

  def test_requested_packing_rejects_unsupported_devices_head_or_alignment(self):
    cpu=fixture(device='CPU',output_device='CPU')
    with patch('tinygrad.llm.kernels.mali.Device',{}):
      self.assertEqual(configure(cpu),{'linear_layers':0})
      with self.assertRaisesRegex(ValueError,'Mali OpenCL model'):configure(cpu,pack_output=True)
    for kwargs in ({'output_device':'CPU'},{'depth':6},{'dtype':dtypes.int16}):
      with self.subTest(kwargs=kwargs):
        with self.assertRaisesRegex(ValueError,'aligned FP16/FP32 Mali OpenCL'):configure(fixture(**kwargs),pack_weights=False,pack_output=True)
    model=fixture()
    model.output=lambda x:x
    with self.assertRaisesRegex(ValueError,'supported linear output'):configure(model,pack_weights=False,pack_output=True)
    self.pack.assert_not_called()

  def test_consistent_decode_reconfiguration_preserves_borrowed_head_and_wrapper(self):
    for wrapped in (False,True):
      model=fixture(wrapped=wrapped)
      plain=configure(model,pack_weights=False)
      head=model.output.source if wrapped else model.output
      self.assertFalse(plain['consistent_decode'])
      self.assertFalse(head.consistent_decode)
      enabled=configure(model,pack_weights=False,consistent_decode=True)
      self.assertTrue(enabled['consistent_decode'])
      self.assertTrue(head.consistent_decode)
      self.assertIs(model.output.source if wrapped else model.output,head)
      configure(model,pack_weights=False)
      self.assertFalse(head.consistent_decode)

  def test_native_snapshot_option_is_explicit_and_reset_on_reconfiguration(self):
    model=fixture()
    enabled=configure(model,pack_weights=False,native_state_snapshots=True)
    self.assertTrue(enabled['native_state_snapshots'])
    self.assertTrue(model._mali_native_state_snapshots)
    disabled=configure(model,pack_weights=False)
    self.assertFalse(disabled['native_state_snapshots'])
    self.assertFalse(model._mali_native_state_snapshots)
    with self.assertRaisesRegex(TypeError,'must be a bool'):configure(model,native_state_snapshots=1)
    cpu=fixture(device='CPU',output_device='CPU')
    with self.assertRaisesRegex(ValueError,'Mali OpenCL'):configure(cpu,native_state_snapshots=True)

  def test_consistent_decode_rejects_quantized_projection_requests_before_mutation(self):
    for option in ('q8_head_checkpoint','q8_projections_checkpoint'):
      model=fixture()
      original=model.output
      with self.assertRaisesRegex(ValueError,'dense projections'):
        configure(model,consistent_decode=True,**{option:'unused.gguf'})
      self.assertIs(model.output,original)
    with self.assertRaisesRegex(TypeError,'must be a bool'):configure(fixture(),consistent_decode=1)


if __name__=='__main__':unittest.main()
