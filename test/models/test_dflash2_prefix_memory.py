"""Growing-prefix allocation ownership without changing draft mathematics."""
import unittest
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from test.models.test_dflash2_prepared import compare, observe, setup
from test.models.test_dflash2_model_context import compare_prepared_to_ordinary


def owning_buffer(value):
  buffer=value.uop.base.buffer
  assert isinstance(buffer,Buffer)
  return buffer.base


class TestDFlash2PrefixMemory(unittest.TestCase):
  def test_growing_buffers_release_without_changing_default_allocator_policy(self):
    raw,draft,rng=setup(9571)
    raw.prepare_observation(max_context=64,bound_prefix_memory=False)
    draft.prepare_observation(max_context=64,bound_prefix_memory=True)
    features=rng.normal(0,.3,(1,5,16)).astype(np.float32)
    for model in (raw,draft):observe(model,features,0)
    for model,expected in ((raw,False),(draft,True)):
      self.assertEqual(model.observation_preparation_metadata()['bound_prefix_memory'],expected)
      self.assertEqual(owning_buffer(model._target_hidden).options.nolru,expected)
      for kind,cache in zip(model.config.layer_types,model._prefix_kv):
        if kind=='full_attention':
          self.assertEqual(owning_buffer(cache.key).options.nolru,expected)
          self.assertEqual(owning_buffer(cache.value).options.nolru,expected)
    self.assertFalse(Device['CPU'].allocator.default_buffer_spec.nolru)
    np.testing.assert_array_equal(draft._target_hidden.numpy(),raw._target_hidden.numpy())
    for actual,reference in zip(draft._prefix_kv,raw._prefix_kv):
      np.testing.assert_array_equal(actual.key.numpy(),reference.key.numpy())
      np.testing.assert_array_equal(actual.value.numpy(),reference.value.numpy())
    raw.close_prepared_observation()
    draft.close_prepared_observation()

  def test_append_rollback_and_empty_truncation_preserve_reference_outputs(self):
    raw,draft,rng=setup(9572)
    draft.prepare_proposals(max_context=64)
    draft.prepare_observation(max_context=64,pad_decode_rows=True,bound_prefix_memory=True)
    for position,rows in ((0,17),(17,3),(20,8),(2,2),(4,0),(0,0)):
      features=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      for model in (raw,draft):observe(model,features,position)
      if position+rows:
        compare_prepared_to_ordinary(self,raw,draft,position+rows)
        self.assertTrue(owning_buffer(draft._target_hidden).options.nolru)
      else:
        self.assertIsNone(raw._target_hidden)
        self.assertIsNone(draft._target_hidden)
        compare(draft,0)
    self.assertIsNotNone(draft._proposal_forward.jit.captured)
    draft.close_prepared_observation()
    draft.close_prepared_proposals()

  def test_invalid_memory_flag_does_not_close_prepared_owner(self):
    _raw,draft,_rng=setup(9573)
    draft.prepare_observation(max_context=64,bound_prefix_memory=True)
    owner=draft._prepared_observation
    for value in (None,0,1,'yes'):
      with self.subTest(value=value),self.assertRaises(ValueError):
        draft.prepare_observation(max_context=64,bound_prefix_memory=value)
      self.assertIs(draft._prepared_observation,owner)
      self.assertFalse(owner.closed)
    draft.close_prepared_observation()


if __name__=='__main__':unittest.main()
