"""CPU-only contract tests for selective DFlash2 Mali packing.

All packing and device behavior is mocked so this file never initializes OpenCL.
"""
import os, unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2 import DFlash2


class FakeWeight:
  def __init__(self, shape):
    self.shape, self.ndim = shape, len(shape)
    self.device, self.dtype = 'CL', dtypes.float16


class FakePacked:
  def __init__(self, size): self._size = size
  def nbytes(self): return self._size


class FakeMaliLinear:
  made = []
  def __init__(self, source, prepare_gemm=True):
    self.source, self.weight, self.bias = source, source.weight, source.bias
    self.packed_weight = FakePacked(self.weight.packed_bytes) if prepare_gemm else None
    type(self).made.append(self)


class Owner:
  def __init__(self): self.closed = False
  def close(self): self.closed = True


def draft():
  # Minimal instance: exercising prepare_mali's policy and lifecycle only.
  model = DFlash2.__new__(DFlash2)
  model.device, model.dtype = 'CL', dtypes.float16
  model.config = SimpleNamespace(num_layers=2, vocab_size=16, hidden_size=8)
  model.weights = {
    'fc.weight': FakeWeight((8, 16)),
    'layers.0.proj.weight': FakeWeight((8, 8)),
    'layers.1.proj.weight': FakeWeight((8, 8)),
  }
  # The two packed sizes intentionally differ from any shape-derived estimate.
  model.weights['fc.weight'].packed_bytes = 37
  model.weights['layers.0.proj.weight'].packed_bytes = 53
  model.weights['layers.1.proj.weight'].packed_bytes = 71
  model.lm_head = object()  # borrowed target callable remains untouched
  model._prepared_linear = {}
  model._prepared_shared_head = False
  model._npu_head = None
  model._retain_target_features, model._observed_rows, model._cache_revision = True, 0, 0
  model._proposal_forward, model._prepared_observation = Owner(), Owner()
  model._target_hidden, model._prefix_kv = object(), (object(), object())
  return model


class TestDFlash2SelectivePacking(unittest.TestCase):
  def setUp(self):
    FakeMaliLinear.made = []
    self.device_patch = patch('tinygrad.llm.dflash2.Device', {'CL': SimpleNamespace(device_name='Mali mock')})
    self.device_patch.start()
    self.mali_patch = patch('tinygrad.llm.kernels.mali.MaliLinear', FakeMaliLinear)
    self.mali_patch.start()

  def tearDown(self):
    self.mali_patch.stop()
    self.device_patch.stop()

  def packed_names(self, model):
    return {name for name, layer in model._prepared_linear.items() if layer.packed_weight is not None}

  def test_fc_only_packing_reports_actual_bytes(self):
    model = draft()
    result = model.prepare_mali(pack_fc=True)
    self.assertEqual(self.packed_names(model), {'fc'})
    self.assertEqual(result, {'linear_layers': 3, 'packed_weights': False, 'prepared_shared_head': False,
                              'packed_fc': True, 'packed_bytes': 37})

  def test_all_weight_packing_reports_sum_of_actual_buffers(self):
    model = draft()
    result = model.prepare_mali(pack_weights=True)
    self.assertEqual(self.packed_names(model), {'fc', 'layers.0.proj', 'layers.1.proj'})
    self.assertEqual(result['packed_fc'], True)
    self.assertEqual(result['packed_bytes'], 37 + 53 + 71)

  def test_no_packing_reports_zero_and_default_keeps_fc_unpacked(self):
    model = draft()
    result = model.prepare_mali()
    self.assertEqual(self.packed_names(model), set())
    self.assertFalse(result['packed_fc'])
    self.assertEqual(result['packed_bytes'], 0)
    self.assertFalse(result['packed_weights'])

  def test_reconfiguration_closes_both_owners_and_discards_cached_observation(self):
    model = draft()
    old_proposal, old_observation = model._proposal_forward, model._prepared_observation
    result = model.prepare_mali(pack_fc=True)
    self.assertTrue(old_proposal.closed)
    self.assertTrue(old_observation.closed)
    self.assertIsNone(model._proposal_forward)
    self.assertIsNone(model._prepared_observation)
    self.assertIsNone(model._target_hidden)
    self.assertEqual(model._prefix_kv, (None, None))
    self.assertTrue(result['packed_fc'])

  def test_borrowed_target_head_is_not_replaced_or_mutated(self):
    model = draft()
    target_head = model.lm_head
    metadata = model.prepare_mali(pack_fc=True, prepare_shared_head=True)
    self.assertIs(model.lm_head, target_head)
    self.assertFalse(metadata['prepared_shared_head'])

  def test_raw_shared_head_uses_prepared_fp32_adapter_without_materializing_weight(self):
    from tinygrad.llm.kernels.mali_q8_shared import MaliQ8Table, MaliQ8Head
    model = draft()
    model.config.hidden_size, model.config.vocab_size = 32, 7
    for key in model.weights:
      model.weights[key] = FakeWeight((32, 64) if key == 'fc.weight' else (32, 32))
    table = MaliQ8Table(Tensor.empty(7 * 34, device='CL', dtype=dtypes.uint8), 7, 32)
    model.lm_head = MaliQ8Head(table)
    head = model.lm_head
    metadata = model.prepare_mali(prepare_shared_head=True)
    self.assertTrue(metadata['prepared_shared_head'])
    self.assertIs(model.lm_head, head)
    result = model._project_head(Tensor.empty(1, 8, 32, device='CL', dtype=dtypes.half))
    self.assertEqual(result.shape, (1, 8, 7))
    self.assertIs(result.dtype, dtypes.half)
    self.assertFalse(table.weight.uop.is_realized)

  def test_invalid_flag_type_fails_before_closing_prepared_owners(self):
    model = draft()
    proposal, observation = model._proposal_forward, model._prepared_observation
    with self.assertRaises(TypeError): model.prepare_mali(pack_fc=1)
    self.assertFalse(proposal.closed)
    self.assertFalse(observation.closed)
    self.assertIs(model._proposal_forward, proposal)
    self.assertIs(model._prepared_observation, observation)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestDFlash2SelectivePackingMali(unittest.TestCase):
  def test_fc_packing_preserves_prepared_observation_append_rollback_and_logits(self):
    from test.models.test_dflash2 import fixture
    from test.models.test_dflash2_cache import output
    _base,config,weights,embedding,head,rng = fixture(20261024)
    tensors = {name:Tensor(value.astype(np.float16),device='CL').realize() for name,value in weights.items()}
    emb = Tensor(embedding.astype(np.float16),device='CL').realize()
    target_head = Tensor(head.astype(np.float16),device='CL').realize()
    def embedding_fn(ids):return emb[ids]
    def head_fn(hidden):return hidden@target_head.T
    models = [DFlash2(config,tensors,embedding_fn,head_fn,lambda hidden:hidden) for _ in range(2)]
    metadata = [model.prepare_mali(pack_fc=packed) for model,packed in zip(models,(False,True))]
    self.assertEqual(metadata[0]['packed_bytes'],0)
    self.assertTrue(metadata[1]['packed_fc'])
    self.assertEqual(metadata[1]['packed_bytes'],models[1]._prepared_linear['fc'].packed_weight.nbytes())
    for name,layer in models[1]._prepared_linear.items():
      self.assertIs(layer.weight,tensors[name+'.weight'])
      if name!='fc':self.assertIsNone(layer.packed_weight)
    for model in models:
      self.assertIs(model.lm_head,head_fn)
      model.prepare_observation(max_context=64,pad_decode_rows=True,native_cache_copy=True,bound_prefix_memory=True)
    features = rng.normal(0,.4,(1,19,16)).astype(np.float16)
    replacement = rng.normal(0,.4,(1,1,16)).astype(np.float16)
    noise = Tensor([[3,15,15,15]],device='CL',dtype=dtypes.int32).realize()
    try:
      for chunk,start in ((features[:,:17],0),(features[:,17:],17),(replacement,5)):
        for model in models:model.observe(output(chunk),start)
        np.testing.assert_array_equal(models[0]._target_hidden.numpy(),models[1]._target_hidden.numpy())
        for left,right in zip(models[0]._prefix_kv,models[1]._prefix_kv):
          np.testing.assert_array_equal(left.key.numpy(),right.key.numpy())
          np.testing.assert_array_equal(left.value.numpy(),right.value.numpy())
        replies = [model._forward_projected(noise,model._target_hidden,position=start+chunk.shape[1],
                                            prefix_kv=model._prefix_kv).realize() for model in models]
        np.testing.assert_array_equal(replies[0].hidden.numpy(),replies[1].hidden.numpy())
        np.testing.assert_array_equal(replies[0].logits.numpy(),replies[1].logits.numpy())
      np.testing.assert_array_equal(tensors['fc.weight'].numpy(),weights['fc.weight'].astype(np.float16))
      np.testing.assert_array_equal(target_head.numpy(),head.astype(np.float16))
    finally:
      for model in models:model.close_prepared_observation()


if __name__ == '__main__': unittest.main()
