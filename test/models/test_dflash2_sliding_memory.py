"""Regression checks for pre-window sliding-cache allocation policy."""
import unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Device
from tinygrad.device import Allocator,BufferStorage
from tinygrad.helpers import Context
from test.models.test_dflash2_prepared import observe,setup
from test.models.test_dflash2_prefix_memory import owning_buffer


class ByteAllocator(Allocator):
  """Use the real cache/free policy with small host bytearrays, not GPU memory."""
  def __init__(self):
    super().__init__(SimpleNamespace(device='CPU'))
    self.freed_bytes=0
  def _alloc(self,size,options):return BufferStorage(bytearray(size))
  def _free(self,storage,options):self.freed_bytes+=len(storage.buf)
  def retained_bytes(self):return sum(size*len(values) for (size,_options),values in self.cache.items())


class TestSlidingPrefixMemory(unittest.TestCase):
  def models(self,seed):
    raw,bounded,rng=setup(seed)
    raw.config.sliding_window=16
    bounded.config.sliding_window=16
    raw.prepare_observation(max_context=64,bound_prefix_memory=False)
    bounded.prepare_observation(max_context=64,bound_prefix_memory=True)
    return raw,bounded,rng

  def equal_cache_values(self,raw,bounded):
    np.testing.assert_array_equal(raw._target_hidden.numpy(),bounded._target_hidden.numpy())
    for expected,actual in zip(raw._prefix_kv,bounded._prefix_kv):
      self.assertEqual((expected.start_pos,expected.end_pos),(actual.start_pos,actual.end_pos))
      np.testing.assert_array_equal(expected.key.numpy(),actual.key.numpy())
      np.testing.assert_array_equal(expected.value.numpy(),actual.value.numpy())

  def test_pre_window_growth_then_full_window_preserves_values_and_local_policy(self):
    raw,bounded,rng=self.models(20261025)
    try:
      for position in range(18):
        values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
        for model in (raw,bounded):observe(model,values,position)
        self.equal_cache_values(raw,bounded)
        sliding=bounded._prefix_kv[0]
        self.assertEqual(sliding.end_pos-sliding.start_pos,min(position+1,16))
        for value in (sliding.key,sliding.value):
          self.assertEqual(owning_buffer(value).options.nolru,position+1<16)
        for value in (raw._prefix_kv[0].key,raw._prefix_kv[0].value):
          self.assertFalse(owning_buffer(value).options.nolru)
      self.assertFalse(Device['CPU'].allocator.default_buffer_spec.nolru)
    finally:
      for model in (raw,bounded):model.close_prepared_observation()

  def test_early_sliding_options_do_not_retain_size_ladder_in_allocator_model(self):
    raw,bounded,rng=self.models(20261026)
    allocators=[ByteAllocator(),ByteAllocator()]
    try:
      with Context(LRU=1):
        for position in range(15):
          values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
          for model,allocator in zip((raw,bounded),allocators):
            observe(model,values,position)
            cache=model._prefix_kv[0]
            for value in (cache.key,cache.value):
              owner=owning_buffer(value)
              # Model the eventual free using the actual owning buffer's policy.
              allocator.free(BufferStorage(bytearray(owner.nbytes)),owner.nbytes,owner.options)
          self.equal_cache_values(raw,bounded)
        self.assertGreater(allocators[0].retained_bytes(),0)
        self.assertEqual(allocators[1].retained_bytes(),0)
        self.assertGreater(allocators[1].freed_bytes,0)
        for allocator in allocators:allocator.free_cache()
    finally:
      for model in (raw,bounded):model.close_prepared_observation()

  def test_rollback_under_window_rebuilds_without_changing_values(self):
    raw,bounded,rng=self.models(20261027)
    try:
      for position,rows in ((0,17),(4,2),(6,1)):
        values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
        for model in (raw,bounded):observe(model,values,position)
        self.equal_cache_values(raw,bounded)
        for kind,cache in zip(bounded.config.layer_types,bounded._prefix_kv):
          expected=kind=='full_attention' or cache.end_pos-cache.start_pos<bounded.config.sliding_window
          for value in (cache.key,cache.value):
            self.assertEqual(owning_buffer(value).options.nolru,expected)
    finally:
      for model in (raw,bounded):model.close_prepared_observation()


if __name__=='__main__':unittest.main()
