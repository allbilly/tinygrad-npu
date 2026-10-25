"""Native Mali cache-copy tests; hardware cases require explicit MALI_TEST."""
import os, unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels import mali_copy
from tinygrad.llm.kernels.mali_copy import _copy_rect
from test.models.test_dflash2_observe import numpy_observation, observation, setup


class TestNativeCacheCopyContract(unittest.TestCase):
  def test_dense_buffer_accepts_whole_storage_without_generic_device_views(self):
    value=Tensor(np.arange(32,dtype=np.float32).reshape(1,4,8),device='CPU').realize()
    from tinygrad.uop.ops import UOp
    with patch.object(UOp,'contiguous_view',side_effect=AssertionError('generic CL views are unsupported')):
      self.assertIs(mali_copy._dense_buffer(value),value.uop.buffer)
      self.assertIs(mali_copy._dense_buffer(value.reshape(4,8)),value.uop.buffer)
    self.assertIsNone(mali_copy._dense_buffer(value[:,1:]))
    self.assertIsNone(mali_copy._dense_buffer(value.transpose(1,2)))

  def test_copy_rect_origins_extents_and_pitches(self):
    device=SimpleNamespace(queue=object())
    source,output=SimpleNamespace(_buf=object()),SimpleNamespace(_buf=object())
    with patch('tinygrad.runtime.autogen.opencl.clEnqueueCopyBufferRect',return_value=0) as enqueue, \
         patch('tinygrad.runtime.ops_cl.check') as check:
      _copy_rect(device,source,output,11,9,2,4,3,5,16)
    args=enqueue.call_args.args
    self.assertIs(args[0],device.queue)
    self.assertIs(args[1],source._buf)
    self.assertIs(args[2],output._buf)
    self.assertEqual(tuple(args[3]),(32,0,0))
    self.assertEqual(tuple(args[4]),(48,0,0))
    self.assertEqual(tuple(args[5]),(64,5,1))
    self.assertEqual(args[6:10],(176,0,144,0))
    self.assertEqual(args[10:],(0,None,None))
    check.assert_called_once_with(0)

  def test_copy_rect_propagates_driver_status_errors(self):
    device=SimpleNamespace(queue=object())
    source,output=SimpleNamespace(_buf=object()),SimpleNamespace(_buf=object())
    failure=RuntimeError('copy status rejected')
    with patch('tinygrad.runtime.autogen.opencl.clEnqueueCopyBufferRect',return_value=-30), \
         patch('tinygrad.runtime.ops_cl.check',side_effect=failure) as check:
      with self.assertRaises(RuntimeError) as caught:
        _copy_rect(device,source,output,4,5,0,2,3,2,8)
    self.assertIs(caught.exception,failure)
    check.assert_called_once_with(-30)

  def test_cpu_native_copy_option_falls_back_and_matches_numpy_history(self):
    _raw,model,c,w,rng=setup(651)
    model.prepare_observation(max_context=16,native_cache_copy=True)
    manager=model._prepared_observation
    history=np.empty((1,0,16),np.float32)
    retained=[]
    for position,rows in ((0,8),(8,1),(9,3),(2,1),(3,8),(4,0),(0,0)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      history=np.concatenate((history[:,:position],values),axis=1)
      model.observe(observation(model,values),position)
      for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
      if not history.shape[1]:
        self.assertIsNone(model._target_hidden)
        self.assertEqual(model._prefix_kv,(None,)*c.num_layers)
        continue
      expected=numpy_observation(history,0,c,w)
      np.testing.assert_allclose(model._target_hidden.numpy(),expected[0],atol=2e-6,rtol=3e-5)
      current=[model._target_hidden]
      for layer,cache in enumerate(model._prefix_kv):
        begin=max(0,history.shape[1]-c.sliding_window) if c.layer_types[layer]=='sliding_attention' else 0
        np.testing.assert_allclose(cache.key.numpy(),expected[1+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        np.testing.assert_allclose(cache.value.numpy(),expected[2+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        current.extend((cache.key,cache.value))
      retained.extend((value,value.numpy().copy()) for value in current)
    self.assertGreater(manager.native_copy_fallbacks,0)
    self.assertEqual(manager.native_copy_appends,0)
    model.close_prepared_observation()

  def test_invalid_native_copy_flags_preserve_existing_preparation(self):
    _raw,model,_c,_w,_rng=setup(652)
    model.prepare_observation(max_context=16)
    previous=model._prepared_observation
    for value in (0,1,None,'true',[],object()):
      with self.subTest(value=repr(value)):
        with self.assertRaises(ValueError):model.prepare_observation(max_context=16,native_cache_copy=value)
        self.assertIs(model._prepared_observation,previous)
    model.close_prepared_observation()

  def test_recording_capture_declines_native_helper_and_keeps_tensor_fallback(self):
    _raw,model,_c,_w,_rng=setup(655)
    model.prepare_observation(max_context=16,native_cache_copy=True)
    manager=model._prepared_observation
    previous=Tensor(np.arange(4*8,dtype=np.float32).reshape(1,4,8),device='CPU').realize()
    appended=Tensor(np.ones((1,2,8),np.float32),device='CPU').realize()
    with patch.object(mali_copy,'capturing',[object()]), patch.object(mali_copy,'_dense_buffer') as dense_buffer:
      self.assertIsNone(manager.append_rows(previous,appended,0,3))
    dense_buffer.assert_not_called()
    self.assertEqual(manager.native_copy_fallbacks,1)
    expected=previous[:,:3].cat(appended,dim=1).realize()
    np.testing.assert_array_equal(expected.numpy(),np.concatenate((previous.numpy()[:,:3],appended.numpy()),axis=1))
    model.close_prepared_observation()


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit Mali OpenCL hardware qualification')
class TestNativeCacheCopyMali(unittest.TestCase):
  def _exercise_append_shape(self,shape,dtype,seed):
    rng=np.random.default_rng(seed)
    shape=tuple(shape)
    axis=-2
    previous_np=rng.normal(0,.3,shape).astype(dtype)
    previous=Tensor(previous_np,device='CL').realize()
    retained=[(previous,previous.numpy().copy())]
    for index,rows in enumerate((1,3,8)):
      append_shape=(*shape[:-2],rows,shape[-1])
      appended_np=rng.normal(0,.3,append_shape).astype(dtype)
      appended=Tensor(appended_np,device='CL').realize()
      # The first operation keeps a suffix; the final one exercises an empty prefix.
      if index==0:begin,end=1,previous_np.shape[axis]
      elif index==2:begin=end=previous_np.shape[axis]
      else:begin,end=0,previous_np.shape[axis]
      wanted=np.concatenate((previous_np[...,begin:end,:],appended_np),axis=axis)
      result=mali_copy.append_rows(previous,appended,begin,end)
      self.assertIsNotNone(result)
      self.assertEqual(result.numpy().tobytes(),wanted.tobytes())
      for old,saved in retained:np.testing.assert_array_equal(old.numpy(),saved)
      np.testing.assert_array_equal(previous.numpy(),previous_np)
      np.testing.assert_array_equal(appended.numpy(),appended_np)
      retained.extend(((result,result.numpy().copy()),(appended,appended.numpy().copy())))
      previous,previous_np=result,wanted

  def test_append_rows_fp16_fp32_dense_cache_shapes_and_retained_inputs(self):
    for dtype in (np.float16,np.float32):
      for heads in (1,4):
        with self.subTest(dtype=dtype,heads=heads):
          self._exercise_append_shape((1,heads,4,8),dtype,660+heads+int(dtype is np.float32))
      with self.subTest(dtype=dtype,shape='hidden'):
        self._exercise_append_shape((1,4,8),dtype,670+int(dtype is np.float32))

  def test_append_rows_declines_noncontiguous_and_offset_inputs(self):
    base=Tensor(np.arange(1*2*5*8,dtype=np.float32).reshape(1,2,5,8),device='CL').realize()
    appended=Tensor(np.ones((1,2,1,4),np.float32),device='CL').realize()
    noncontiguous=base[:,:,:,::2]
    offset=base[:,:,:,1:5]
    self.assertIsNone(mali_copy.append_rows(noncontiguous,appended,0,noncontiguous.shape[-2]))
    self.assertIsNone(mali_copy.append_rows(offset,appended,0,offset.shape[-2]))

  def test_jit_capture_uses_tensor_fallback_and_replay_accepts_changing_inputs(self):
    def append(previous,appended):
      result=mali_copy.append_rows(previous,appended,1,4)
      return (previous[:,1:4].cat(appended,dim=1) if result is None else result).realize()
    jit=TinyJit(append)
    try:
      with patch.object(mali_copy,'_copy_rect',wraps=mali_copy._copy_rect) as copied:
        for index in range(4):
          old=(np.arange(40,dtype=np.float32).reshape(1,5,8)+index*100)
          new=np.full((1,2,8),index+.25,np.float32)
          result=jit(Tensor(old,device='CL').realize(),Tensor(new,device='CL').realize())
          self.assertEqual(result.numpy().tobytes(),np.concatenate((old[:,1:4],new),axis=1).tobytes())
        self.assertEqual(copied.call_count,2)
      self.assertEqual(jit.cnt,4)
      self.assertIsNotNone(jit.captured)
    finally:
      Device['CL'].synchronize()
      if jit.captured is not None:jit.captured.free_intermediates()
      jit.reset()

  def test_prepared_observe_native_copy_matches_eager_and_preserves_history(self):
    raw,model,_c,_w,rng=setup(653,'CL',True)
    raw.prepare_mali()
    model.prepare_mali()
    model.prepare_observation(max_context=32,pad_decode_rows=True,native_cache_copy=True)
    manager=model._prepared_observation
    retained=[]
    for position,rows in ((0,8),(8,1),(9,3),(2,1),(3,8)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float16)
      raw_input,prepared_input=observation(raw,values),observation(model,values)
      input_snapshots=[(value,value.numpy().copy()) for value in prepared_input.features]
      for draft,output in ((raw,raw_input),(model,prepared_input)):draft.observe(output,position)
      for old,saved in retained:np.testing.assert_array_equal(old.numpy(),saved)
      for old,saved in input_snapshots:np.testing.assert_array_equal(old.numpy(),saved)
      np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=3e-3,rtol=4e-3)
      current=[model._target_hidden]
      for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
        np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=3e-3,rtol=4e-3)
        np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=3e-3,rtol=4e-3)
        current.extend((actual.key,actual.value))
      retained.extend((value,value.numpy().copy()) for value in current)
    self.assertGreater(manager.native_copy_appends,0)
    self.assertGreater(manager.native_copy_commands,0)
    model.close_prepared_observation()

  def test_native_copy_failure_after_queued_command_is_transactional_and_retryable(self):
    raw,model,_c,_w,rng=setup(654,'CL',True)
    raw.prepare_mali()
    model.prepare_mali()
    model.prepare_observation(max_context=32,pad_decode_rows=True,native_cache_copy=True)
    initial=rng.normal(0,.3,(1,8,16)).astype(np.float16)
    for draft in (raw,model):draft.observe(observation(draft,initial),0)
    old_hidden,old_caches=model._target_hidden,model._prefix_kv
    retained=[(old_hidden,old_hidden.numpy().copy())]
    retained.extend((value,value.numpy().copy()) for cache in old_caches for value in (cache.key,cache.value))
    update=rng.normal(0,.3,(1,1,16)).astype(np.float16)
    original=mali_copy._copy_rect
    calls=0
    failure=RuntimeError('injected native rectangle failure')
    def fail_on_third(*args,**kwargs):
      nonlocal calls
      calls+=1
      original(*args,**kwargs)
      if calls==3:raise failure
    with patch.object(mali_copy,'_copy_rect',fail_on_third):
      with self.assertRaises(RuntimeError) as caught:model.observe(observation(model,update),8)
    self.assertIs(caught.exception,failure)
    self.assertEqual(calls,3)
    self.assertIs(model._target_hidden,old_hidden)
    self.assertIs(model._prefix_kv,old_caches)
    for old,saved in retained:np.testing.assert_array_equal(old.numpy(),saved)
    raw.observe(observation(raw,update),8)
    model.observe(observation(model,update),8)
    self.assertIsNot(model._target_hidden,old_hidden)
    np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=3e-3,rtol=4e-3)
    for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
      np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=3e-3,rtol=4e-3)
      np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=3e-3,rtol=4e-3)
    for old,saved in retained:np.testing.assert_array_equal(old.numpy(),saved)
    model.close_prepared_observation()


if __name__=='__main__':unittest.main()
