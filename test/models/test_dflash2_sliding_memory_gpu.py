"""OpenCL check for sliding-prefix allocation ownership and cache-copy parity."""
import os, unittest
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from test.models.test_dflash2_prepared import observe, setup


def owning_buffer(value):
  buffer=value.uop.base.buffer
  assert isinstance(buffer,Buffer)
  return buffer.base


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestDFlash2SlidingMemoryMali(unittest.TestCase):
  def test_sliding_prefix_memory_matches_unbounded_observation(self):
    raw,bound,rng=setup(9581,'CL',True)
    for model in (raw,bound):model.config.sliding_window=16
    try:
      for model in (raw,bound):model.prepare_mali(pack_weights=False)
      raw.prepare_observation(max_context=64,pad_decode_rows=True,native_cache_copy=True,bound_prefix_memory=False)
      bound.prepare_observation(max_context=64,pad_decode_rows=True,native_cache_copy=True,bound_prefix_memory=True)

      for position,rows in ((0,1),(1,1),(2,3),(5,8),(13,3),(16,1),(2,2)):
        features=rng.normal(0,.3,(1,rows,16)).astype(np.float16)
        for model in (raw,bound):observe(model,features,position)

        np.testing.assert_array_equal(bound._target_hidden.numpy(),raw._target_hidden.numpy())
        self.assertEqual(bound._target_hidden.shape,raw._target_hidden.shape)
        for actual,reference in zip(bound._prefix_kv,raw._prefix_kv):
          self.assertEqual((actual.start_pos,actual.end_pos),(reference.start_pos,reference.end_pos))
          for name in ('key','value'):
            actual_array=getattr(actual,name).numpy()
            reference_array=getattr(reference,name).numpy()
            self.assertEqual(actual_array.shape,reference_array.shape)
            self.assertEqual(actual_array.dtype,reference_array.dtype)
            self.assertEqual(actual_array.tobytes(),reference_array.tobytes())
            np.testing.assert_array_equal(actual_array[...,0,:],reference_array[...,0,:])
            np.testing.assert_array_equal(actual_array[...,-1,:],reference_array[...,-1,:])

        for kind,bound_cache,raw_cache in zip(bound.config.layer_types,bound._prefix_kv,raw._prefix_kv):
          expected=kind=='full_attention' or bound_cache.end_pos-bound_cache.start_pos<bound.config.sliding_window
          self.assertEqual(owning_buffer(bound_cache.key).options.nolru,expected)
          self.assertEqual(owning_buffer(bound_cache.value).options.nolru,expected)
          self.assertFalse(owning_buffer(raw_cache.key).options.nolru)
          self.assertFalse(owning_buffer(raw_cache.value).options.nolru)

      for model,expected in ((raw,False),(bound,True)):
        metadata=model.observation_preparation_metadata()
        self.assertEqual(metadata['bound_prefix_memory'],expected)
        self.assertGreater(metadata['native_copy_appends'],0)
      self.assertFalse(Device['CL'].allocator.default_buffer_spec.nolru)

      for model in (raw,bound):observe(model,np.empty((1,0,16),dtype=np.float16),0)
      self.assertIsNone(raw._target_hidden)
      self.assertIsNone(bound._target_hidden)
      self.assertTrue(all(cache is None for cache in raw._prefix_kv))
      self.assertTrue(all(cache is None for cache in bound._prefix_kv))
    finally:
      raw.close_prepared_observation()
      bound.close_prepared_observation()


if __name__=='__main__':unittest.main()
