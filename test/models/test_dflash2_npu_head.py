"""Draft-only optional NPU head ownership and unchanged GPU fallback.

CPU fixtures substitute only the device executor. The optional two-device test
uses a small internal matrix; the public API still permits only the measured
Qwen3.5-2B FP16/R8 head, and trained token/performance qualification is separate.
"""
import os, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes, nn
from tinygrad.llm.dflash2 import DFlash2, _NPUDraftHead
from tinygrad.llm.kernels.mali import MaliLinear
from test.models.test_dflash2 import fixture


def shared_head(seed=771):
  rng=np.random.default_rng(seed)
  weights=rng.normal(0,.2,(16,8)).astype(np.float16)
  source=nn.Linear.__new__(nn.Linear)
  source.weight,source.bias=Tensor(weights,device='CPU').realize(),None
  return MaliLinear(source,prepare_gemm=False),weights,rng


class FakeNative:
  """CPU FP32 dot substitutes for only the external device execution."""
  def __init__(self,source,prefill,decode,tile_columns):
    self.source,self.prefill,self.decode,self.tile_columns=source,prefill,decode,tile_columns
    self.packed=[]
    self.received=[]
    self.released=False

  def prepare(self):self.packed=[self.source.weight]

  def npu(self,x):
    self.received.append(x.dtype)
    return x.float()@self.source.weight.float().T

  def release_packed(self):
    self.packed.clear()
    self.released=True


def devices():
  return {'CPU':SimpleNamespace(synchronize=Mock()),'ROCKCHIP':SimpleNamespace(synchronize=Mock())}


class TestDFlash2NPUHead(unittest.TestCase):
  def test_default_and_rejected_eligibility_do_not_initialize_npu_or_discard_graph(self):
    draft,_config,_weights,_embedding,_head,_rng=fixture(772)
    draft.prepare_proposals(max_context=16)
    graph=draft._proposal_forward
    with patch('tinygrad.llm.dflash2.Device',{}):
      self.assertEqual(draft.npu_head_preparation_metadata(),{'enabled':False,'additional_npu_packed_bytes':0})
      with self.assertRaisesRegex(ValueError,'exclusive NPU/RAM lease'):draft.prepare_npu_head()
      with self.assertRaisesRegex(ValueError,'FP16 Mali Qwen3.5-2B'):draft.prepare_npu_head(exclusive_lease=True)
    self.assertIs(draft._proposal_forward,graph)
    self.assertIsNone(draft._npu_head)
    draft.close_prepared_proposals()

  def test_public_geometry_guards_reject_before_npu_ownership_or_graph_mutation(self):
    def lazy_qualified():
      # Lazy metadata only: no allocation or device initialization for this
      # actual-sized head. Device access is restricted to mocked Mali metadata.
      source=nn.Linear.__new__(nn.Linear)
      source.weight,source.bias=Tensor.empty(248320,2048,device='CL',dtype=dtypes.float16),None
      draft=DFlash2.__new__(DFlash2)
      draft.device,draft.dtype,draft.block_size='CL',dtypes.float16,8
      draft.config=SimpleNamespace(hidden_size=2048,vocab_size=248320)
      draft.lm_head=MaliLinear(source,prepare_gemm=False)
      draft._proposal_forward=SimpleNamespace(close=Mock())
      draft._npu_head=None
      return draft
    guards=(
      ('draft dtype',lambda draft:setattr(draft,'dtype',dtypes.float32)),
      ('trained slots',lambda draft:setattr(draft,'block_size',4)),
      ('hidden size',lambda draft:setattr(draft.config,'hidden_size',1024)),
      ('vocabulary',lambda draft:setattr(draft.config,'vocab_size',248288)),
      ('head kind',lambda draft:setattr(draft,'lm_head',draft.lm_head.source)),
      ('head shape',lambda draft:setattr(draft.lm_head,'weight',Tensor.empty(248288,2048,device='CL',dtype=dtypes.float16))),
      ('head device',lambda draft:setattr(draft.lm_head,'weight',Tensor.empty(248320,2048,device='CPU',dtype=dtypes.float16))),
      ('head dtype',lambda draft:setattr(draft.lm_head,'weight',Tensor.empty(248320,2048,device='CL',dtype=dtypes.float32))),
      ('head bias',lambda draft:setattr(draft.lm_head,'bias',Tensor.empty(248320,device='CL',dtype=dtypes.float16))),
    )
    with patch('tinygrad.llm.dflash2.Device',{'CL':SimpleNamespace(device_name='Mali-G610')}):
      lazy_qualified()._validate_npu_head()
      for label,change in guards:
        with self.subTest(guard=label):
          draft=lazy_qualified()
          graph=draft._proposal_forward
          change(draft)
          with self.assertRaisesRegex(ValueError,'FP16 Mali Qwen3.5-2B'):draft.prepare_npu_head(exclusive_lease=True)
          self.assertIs(draft._proposal_forward,graph)
          self.assertIsNone(draft._npu_head)
          graph.close.assert_not_called()
    draft=lazy_qualified()
    with patch('tinygrad.llm.dflash2.Device',{'CL':SimpleNamespace(device_name='Other OpenCL')}):
      with self.assertRaisesRegex(ValueError,'requires Mali OpenCL'):draft.prepare_npu_head(exclusive_lease=True)
    draft._proposal_forward.close.assert_not_called()

  def test_adapter_uses_shared_weight_fp32_accumulation_and_preserves_half_outputs(self):
    head,weights,rng=shared_head()
    original,original_uop=head.weight,head.weight.uop
    with patch('tinygrad.llm.dflash2.Device',devices()),patch('tinygrad.llm.kernels.rockchip.RockchipLinear',FakeNative):
      adapter=_NPUDraftHead(head,rows=8,tile_columns=12288)
      native=adapter.native
      values=rng.normal(0,.4,(1,8,8)).astype(np.float16)
      expected=(values.astype(np.float32)@weights.astype(np.float32).T).astype(np.float16)
      actual=adapter(Tensor(values,device='CPU').realize(),head)
      self.assertEqual(actual.dtype,dtypes.float16)
      np.testing.assert_allclose(actual.numpy(),expected,atol=2e-3,rtol=3e-3)
      np.testing.assert_array_equal(actual.numpy().argmax(-1),expected.argmax(-1))
      self.assertEqual(native.received,[dtypes.float32])
      self.assertIs(head.weight,original)
      self.assertIs(head.weight.uop,original_uop)
      self.assertIs(head.source.weight,original)
      self.assertIsNone(head.packed_weight)
      self.assertEqual(adapter.metadata()['gpu_weight_copies'],0)
      for invalid in (Tensor(values[:,:1],device='CPU'),Tensor(values.astype(np.float32),device='CPU')):
        with self.assertRaisesRegex(ValueError,'complete resident FP16'):adapter(invalid,head)
      with self.assertRaisesRegex(ValueError,'Borrowed NPU draft head changed'):
        adapter(Tensor(values,device='CPU'),lambda hidden:hidden)
      adapter.close()
      self.assertTrue(native.released)
      self.assertIsNone(adapter.native)
      self.assertIsNone(adapter.head)
      self.assertTrue(adapter.closed)
      adapter.close()
      with self.assertRaisesRegex(RuntimeError,'closed'):adapter(Tensor(values,device='CPU'),head)

  def test_enable_close_invalidates_graphs_and_preserves_gpu_head_on_cleanup_failures(self):
    draft,_config,_weights,_embedding,_head,_rng=fixture(773)
    head,weights,rng=shared_head(774)
    draft.lm_head=head
    original=head.weight
    fake_devices=devices()
    # The public eligibility guard is separately tested above. Bypass only that
    # guard to exercise lifecycle without allocating the actual 970 MiB head.
    with patch.object(draft,'_validate_npu_head'),patch('tinygrad.llm.dflash2.Device',fake_devices),\
         patch('tinygrad.llm.kernels.rockchip.RockchipLinear',FakeNative):
      previous=SimpleNamespace(close=Mock())
      draft._proposal_forward=previous
      metadata=draft.prepare_npu_head(exclusive_lease=True)
      previous.close.assert_called_once_with()
      self.assertIsNone(draft._proposal_forward)
      self.assertTrue(metadata['enabled'])
      adapter=draft._npu_head
      native=adapter.native
      draft.reset()
      self.assertIs(draft._npu_head,adapter)
      values=rng.normal(0,.4,(1,8,8)).astype(np.float16)
      mixed=draft._project_head(Tensor(values,device='CPU')).numpy()
      expected=(values.astype(np.float32)@weights.astype(np.float32).T).astype(np.float16)
      np.testing.assert_allclose(mixed,expected,atol=2e-3,rtol=3e-3)
      draft._proposal_forward=SimpleNamespace(close=Mock(side_effect=RuntimeError('graph close')))
      fake_devices['CPU'].synchronize.side_effect=RuntimeError('device close')
      with self.assertRaises(BaseExceptionGroup) as caught:draft.close_npu_head()
      self.assertEqual([str(error) for error in caught.exception.exceptions],['graph close','device close'])
      self.assertIsNone(draft._proposal_forward)
      self.assertIsNone(draft._npu_head)
      self.assertTrue(adapter.closed)
      self.assertTrue(native.released)
      self.assertIsNone(adapter.head)
      self.assertIs(draft.lm_head,head)
      self.assertIs(head.weight,original)
      draft.close_npu_head()
    # A failed optional-path cleanup must still leave the unchanged ordinary
    # head callable, without silently retaining a closed NPU adapter.
    np.testing.assert_array_equal(draft._project_head(Tensor(values,device='CPU')).numpy(),head(Tensor(values,device='CPU')).numpy())

  def test_failed_preparation_releases_owned_packed_references_and_keeps_default_path(self):
    draft,_config,_weights,_embedding,_head,_rng=fixture(775)
    head,_weights,_rng=shared_head(776)
    draft.lm_head=head
    native=SimpleNamespace(prepare=Mock(side_effect=RuntimeError('packing failed')),release_packed=Mock())
    with patch.object(draft,'_validate_npu_head'),patch('tinygrad.llm.dflash2.Device',devices()),\
         patch('tinygrad.llm.kernels.rockchip.RockchipLinear',return_value=native):
      with self.assertRaisesRegex(RuntimeError,'packing failed'):draft.prepare_npu_head(exclusive_lease=True)
    native.release_packed.assert_called_once_with()
    self.assertIsNone(draft._npu_head)
    self.assertIs(draft.lm_head,head)


@unittest.skipUnless(os.environ.get('DFLASH2_NPU_HEAD_TEST')=='1' and os.environ.get('ONPU_HEAD_LEASE')=='1',
                     'explicit two-device/RAM lease; fresh child process required')
class TestDFlash2NPUHeadDevices(unittest.TestCase):
  def test_small_fp16_eight_row_cross_device_replay_matches_independent_dot(self):
    rng=np.random.default_rng(777)
    weights=rng.normal(0,.1,(96,64)).astype(np.float16)
    source=nn.Linear.__new__(nn.Linear)
    source.weight,source.bias=Tensor(weights,device='CL').realize(),None
    head=MaliLinear(source,prepare_gemm=False)
    adapter=_NPUDraftHead(head,rows=8,tile_columns=12288)
    gpu=TinyJit(lambda hidden:head(hidden.float()).half().realize())
    mixed=TinyJit(lambda hidden:adapter(hidden,head).realize())
    try:
      for _index in range(4):
        values=rng.normal(0,.4,(1,8,64)).astype(np.float16)
        hidden=Tensor(values,device='CL').realize()
        expected=(values.astype(np.float32)@weights.astype(np.float32).T).astype(np.float16)
        ordinary=gpu(hidden).numpy()
        actual=mixed(hidden).numpy()
        np.testing.assert_allclose(ordinary,expected,atol=2e-3,rtol=3e-3)
        np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-3)
        np.testing.assert_array_equal(actual.argmax(-1),ordinary.argmax(-1))
      self.assertIs(head.weight,source.weight)
      self.assertIsNone(head.packed_weight)
    finally:
      try:
        for run in (gpu,mixed):
          if run.captured is not None:run.captured.free_intermediates()
          run.fxn=None
          run.captured=None
      finally:
        try:adapter.close()
        finally:Device['ROCKCHIP'].finalize()


if __name__=='__main__':unittest.main()
