"""Coverage for the opt-in shared padded decode observation graph."""
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from test.models.test_dflash2_observe import fields, numpy_observation, observation, setup


class TestPreparedObservationRows(unittest.TestCase):
  def test_decode_rows_share_capacity_and_return_only_valid_rows(self):
    _raw,model,c,w,rng=setup(541)
    model.prepare_observation(max_context=32,pad_decode_rows=True)
    manager=model._prepared_observation
    retained=[]
    for rows,position in ((1,1),(2,6),(3,12),(5,20),(8,0)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      actual=manager(fields(model,values),position)
      expected=numpy_observation(values,position,c,w)
      self.assertEqual(len(actual),1+2*c.num_layers)
      self.assertEqual(actual[0].shape,(1,rows,c.hidden_size))
      self.assertEqual(actual[1].shape,(1,c.num_key_value_heads,rows,c.head_dim))
      for value,wanted in zip(actual,expected):
        np.testing.assert_allclose(value.numpy(),wanted,atol=2e-6,rtol=3e-5)
      for old,saved in retained:
        for value,wanted in zip(old,saved):np.testing.assert_array_equal(value.numpy(),wanted)
      retained.append((actual,tuple(value.numpy().copy() for value in actual)))
    self.assertEqual(set(manager.calls),{8})
    self.assertEqual(manager.calls[8].rows,8)
    self.assertEqual(manager.calls[8].calls,5)
    self.assertGreaterEqual(manager.calls[8].jit.cnt,3)
    self.assertIsNotNone(manager.calls[8].jit.captured)
    self.assertEqual(manager.metadata()['decode_capacity'],8)
    model.close_prepared_observation()

  def test_committed_history_rollback_sliding_reset_and_retained_outputs(self):
    _raw,model,c,w,rng=setup(542)
    model.prepare_observation(max_context=16,pad_decode_rows=True)
    manager=model._prepared_observation
    history=np.empty((1,0,16),np.float32)
    retained=[]
    for position,rows in ((0,8),(8,1),(2,3),(5,5),(4,1),(0,2),(0,0)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      history=np.concatenate((history[:,:position],values),axis=1)
      model.observe(observation(model,values),position)
      for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
      if history.shape[1]==0:
        self.assertIsNone(model._target_hidden)
        self.assertEqual(model._prefix_kv,(None,)*c.num_layers)
        continue
      expected=numpy_observation(history,0,c,w)
      np.testing.assert_allclose(model._target_hidden.numpy(),expected[0],atol=2e-6,rtol=3e-5)
      current=[model._target_hidden]
      for layer,cache in enumerate(model._prefix_kv):
        begin=max(0,history.shape[1]-c.sliding_window) if c.layer_types[layer]=='sliding_attention' else 0
        self.assertEqual((cache.start_pos,cache.end_pos),(begin,history.shape[1]))
        np.testing.assert_allclose(cache.key.numpy(),expected[1+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        np.testing.assert_allclose(cache.value.numpy(),expected[2+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        current.extend((cache.key,cache.value))
      retained.extend((value,value.numpy().copy()) for value in current)
    self.assertEqual(set(manager.calls),{8})
    model.close_prepared_observation()

  def test_near_end_uses_exact_rows_and_overflow_falls_back(self):
    _raw,model,_c,_w,rng=setup(543)
    model.prepare_observation(max_context=16,pad_decode_rows=True)
    manager=model._prepared_observation
    values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
    result=manager(fields(model,values),15)
    self.assertIsNotNone(result)
    self.assertEqual(set(manager.calls),{1})
    self.assertIsNone(manager(fields(model,values),16))
    self.assertEqual(manager.calls[1].calls,1)
    model.close_prepared_observation()

  def test_short_contexts_use_bounded_minimum_capacity(self):
    for context in range(1,8):
      with self.subTest(context=context):
        _raw,model,_c,_w,rng=setup(550+context)
        model.prepare_observation(max_context=context,pad_decode_rows=True)
        manager=model._prepared_observation
        values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
        actual=manager(fields(model,values),0)
        self.assertIsNotNone(actual)
        self.assertEqual(set(manager.calls),{context})
        self.assertEqual(manager.calls[context].rows,context)
        self.assertEqual(actual[0].shape,(1,1,8))
        model.close_prepared_observation()

  def test_invalid_pad_flag_does_not_replace_existing_manager(self):
    _raw,model,_c,_w,_rng=setup(544)
    model.prepare_observation(max_context=16)
    previous=model._prepared_observation
    for value in (0,1,None,'true',[],object()):
      with self.subTest(value=repr(value)):
        with self.assertRaises(ValueError):model.prepare_observation(max_context=16,pad_decode_rows=value)
        self.assertIs(model._prepared_observation,previous)
    model.close_prepared_observation()

  def test_partial_cache_materialization_failure_preserves_and_retries_shared_graph(self):
    _raw,model,c,_w,rng=setup(545)
    model.prepare_observation(max_context=16,pad_decode_rows=True)
    initial=rng.normal(0,.3,(1,5,16)).astype(np.float32)
    model.observe(observation(model,initial),0)
    old_hidden,old_caches=model._target_hidden,model._prefix_kv
    old_values=[old_hidden]
    old_values.extend(value for cache in old_caches for value in (cache.key,cache.value))
    retained=[(value,value.numpy().copy()) for value in old_values]
    update=rng.normal(0,.3,(1,1,16)).astype(np.float32)
    realize=Tensor.realize
    failure=RuntimeError('injected cache-copy materialization failure')
    triggered=False
    def fail_cache_materialization(*values,**kwargs):
      nonlocal triggered
      wanted=(1,c.num_key_value_heads,6,c.head_dim)
      if not triggered and any(value.shape==wanted and not value.uop.is_realized for value in values):
        triggered=True
        realize(values[0])
        raise failure
      return realize(*values,**kwargs)
    with patch.object(Tensor,'realize',fail_cache_materialization):
      with self.assertRaises(RuntimeError) as caught:model.observe(observation(model,update),5)
    self.assertTrue(triggered)
    self.assertIs(caught.exception,failure)
    self.assertIs(model._target_hidden,old_hidden)
    self.assertIs(model._prefix_kv,old_caches)
    for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
    manager=model._prepared_observation
    graph=manager.calls[8].jit
    model.observe(observation(model,update),5)
    self.assertIs(model._prepared_observation,manager)
    self.assertIs(manager.calls[8].jit,graph)
    self.assertEqual(manager.calls[8].calls,3)
    expected=numpy_observation(np.concatenate((initial,update),axis=1),0,c,_w)
    np.testing.assert_allclose(model._target_hidden.numpy(),expected[0],atol=2e-6,rtol=3e-5)
    for layer,cache in enumerate(model._prefix_kv):
      begin=max(0,6-c.sliding_window) if c.layer_types[layer]=='sliding_attention' else 0
      np.testing.assert_allclose(cache.key.numpy(),expected[1+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
      np.testing.assert_allclose(cache.value.numpy(),expected[2+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
    for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
    model.close_prepared_observation()


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestPreparedObservationRowsMali(unittest.TestCase):
  def test_fp16_padded_projection_keys_and_cache_match_eager(self):
    raw,model,_c,_w,rng=setup(546,'CL',True)
    raw.prepare_mali()
    model.prepare_mali()
    model.prepare_observation(max_context=32,pad_decode_rows=True)
    for position,rows in ((0,1),(1,3),(4,5),(9,8)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float16)
      for draft in (raw,model):draft.observe(observation(draft,values),position)
      np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=3e-3,rtol=4e-3)
      for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
        np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=3e-3,rtol=4e-3)
        np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=3e-3,rtol=4e-3)
    model.close_prepared_observation()


if __name__=='__main__':unittest.main()
