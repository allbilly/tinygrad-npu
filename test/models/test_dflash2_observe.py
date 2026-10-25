"""Prepared observations must own their caches and preserve absolute RoPE."""
import os, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.dflash2 import DFlash2
from tinygrad.llm.model import TargetOutput
from test.models.test_dflash2 import fixture


def setup(seed,device='CPU',half=False):
  base,c,weights,embedding,head,rng=fixture(seed)
  dtype=np.float16 if half else np.float32
  tensors={name:Tensor(value.astype(dtype),device=device).realize() for name,value in weights.items()}
  emb,out=Tensor(embedding.astype(dtype),device=device).realize(),Tensor(head.astype(dtype),device=device).realize()
  def embed(ids):return emb[ids]
  def project(hidden):return hidden@out.T
  ordinary=DFlash2(c,tensors,embed,project,base.verifier_norm)
  prepared=DFlash2(c,tensors,embed,project,base.verifier_norm)
  return ordinary,prepared,c,weights,rng


def fields(model,values):
  return tuple(Tensor(x,device=model.device,dtype=model.dtype).realize() for x in np.split(values,2,axis=-1))


def observation(model,values):
  features=fields(model,values)
  return TargetOutput(Tensor.zeros(1,values.shape[1],16,device=model.device),features[-1],features)


def numpy_observation(values,position,c,w):
  def norm(x,name):return x/np.sqrt((x*x).mean(-1,keepdims=True)+c.norm_eps)*w[name+'.weight']
  projected=norm(values@w['fc.weight'].T,'hidden_norm')
  result=[projected]
  angles=np.arange(position,position+values.shape[1],dtype=np.float32).reshape(1,1,-1,1)/(
    c.rope_theta**(np.arange(c.head_dim//2,dtype=np.float32)/np.float32(c.head_dim//2)))
  # Tensor.cos currently uses FP32 sin(pi/2-x), including the rounded argument
  # subtraction. Direct NumPy cos is more accurate at large phases and already
  # fails this gate for the unchanged eager implementation. Keep the NumPy
  # equations independent while matching that existing precision contract.
  cosine=np.sin(np.float32(np.pi/2)-angles)
  for layer in range(c.num_layers):
    name=f'layers.{layer}.self_attn'
    k=norm((projected@w[name+'.k_proj.weight'].T).reshape(1,values.shape[1],c.num_key_value_heads,c.head_dim),name+'.k_norm')
    k=k.transpose(0,2,1,3)
    a,b=np.split(k,2,axis=-1)
    k=np.concatenate((a*cosine-b*np.sin(angles),b*cosine+a*np.sin(angles)),axis=-1)
    v=(projected@w[name+'.v_proj.weight'].T).reshape(1,values.shape[1],c.num_key_value_heads,c.head_dim).transpose(0,2,1,3)
    result.extend((k,v))
  return result


class TestPreparedObservation(unittest.TestCase):
  def test_projected_features_and_absolute_keys_match_independent_equations(self):
    raw,model,c,w,rng=setup(531)
    model.prepare_observation(max_context=512)
    manager=model._prepared_observation
    retained=[]
    for rows,position in ((1,0),(1,17),(3,2),(3,31),(8,71),(128,0),(1,191)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      actual=manager(fields(model,values),position)
      expected=numpy_observation(values,position,c,w)
      for value,wanted in zip(actual,expected):np.testing.assert_allclose(value.numpy(),wanted,atol=2e-6,rtol=3e-5)
      projected=raw.project_features(fields(raw,values)).realize()
      unchanged=[projected]
      for layer in range(c.num_layers):
        kv=raw._make_prefix_kv(projected,position,layer)
        unchanged.extend((kv.key,kv.value))
      for value,wanted in zip(actual,unchanged):np.testing.assert_allclose(value.numpy(),wanted.numpy(),atol=2e-6,rtol=3e-5)
      for old,saved in retained:
        for value,wanted in zip(old,saved):np.testing.assert_array_equal(value.numpy(),wanted)
      retained.append((actual,tuple(value.numpy().copy() for value in actual)))
    self.assertEqual(set(manager.calls),{1,3,8,128})
    self.assertLessEqual(len(manager.calls),9)
    self.assertIsNone(manager(fields(model,np.zeros((1,9,16),np.float32)),0))
    model.close_prepared_observation()

  def test_observe_cache_ownership_rollback_window_reset_and_eager_fallback(self):
    raw,model,_c,_w,rng=setup(532)
    model.prepare_observation(max_context=512)
    manager=model._prepared_observation
    old_arrays=None
    for position,rows in ((0,128),(128,1),(129,3),(2,1),(3,8),(11,9),(4,0)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      for draft in (raw,model):draft.observe(observation(draft,values),position)
      np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=2e-6,rtol=3e-5)
      for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
        self.assertEqual((actual.start_pos,actual.end_pos),(expected.start_pos,expected.end_pos))
        np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=2e-6,rtol=3e-5)
        np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=2e-6,rtol=3e-5)
      if rows==1 and old_arrays is None:
        # Same-row JIT calls must not overwrite tensors already handed to caches.
        old_arrays=[(value,value.numpy().copy()) for value in (model._target_hidden,model._prefix_kv[1].key,model._prefix_kv[1].value)]
      if old_arrays:
        for value,wanted in old_arrays:np.testing.assert_array_equal(value.numpy(),wanted)
    model.reset()
    self.assertIs(model._prepared_observation,manager)
    self.assertIsNone(model._target_hidden)
    values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
    model.observe(observation(model,values),0)
    self.assertIs(model._prepared_observation,manager)
    model.close_prepared_observation()

  def test_copy_commits_match_independent_history_and_keep_all_retained_caches(self):
    _raw,model,c,w,rng=setup(535)
    model.prepare_observation(max_context=512)
    manager=model._prepared_observation
    history=np.empty((1,0,16),np.float32)
    retained=[]
    for position,rows in ((0,8),(8,1),(9,3),(2,1),(3,8),(4,0),(4,1),(0,0)):
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
        self.assertEqual((cache.start_pos,cache.end_pos),(begin,history.shape[1]))
        np.testing.assert_allclose(cache.key.numpy(),expected[1+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        np.testing.assert_allclose(cache.value.numpy(),expected[2+2*layer][:,:,begin:],atol=2e-6,rtol=3e-5)
        current.extend((cache.key,cache.value))
      retained.extend((value,value.numpy().copy()) for value in current)
      if rows:
        # Reusing the same observation graph without committing its new outputs
        # must not overwrite any current or previously retained cache allocation.
        probe=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
        manager(fields(model,probe),0)
        for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
    self.assertIs(model._prepared_observation,manager)
    model.close_prepared_observation()

  def test_partial_cache_materialization_failure_preserves_old_transaction_and_can_retry(self):
    raw,model,c,_w,rng=setup(536)
    model.prepare_observation(max_context=16)
    initial=rng.normal(0,.3,(1,5,16)).astype(np.float32)
    for draft in (raw,model):draft.observe(observation(draft,initial),0)
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
        # Model a failure after one new output has already materialized, rather
        # than failing before any work. Existing cache inputs must remain intact.
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
    graph=manager.calls[1].jit
    for draft in (raw,model):draft.observe(observation(draft,update),5)
    self.assertIs(model._prepared_observation,manager)
    self.assertIs(manager.calls[1].jit,graph)
    np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=2e-6,rtol=3e-5)
    for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
      self.assertEqual((actual.start_pos,actual.end_pos),(expected.start_pos,expected.end_pos))
      np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=2e-6,rtol=3e-5)
      np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=2e-6,rtol=3e-5)
    for old,wanted in retained:np.testing.assert_array_equal(old.numpy(),wanted)
    model.close_prepared_observation()

  def test_stale_signatures_capacity_guards_and_cleanup_failures(self):
    _raw,model,_c,_w,rng=setup(533)
    for limit in (0,513,True):
      with self.assertRaises(ValueError):model.prepare_observation(max_context=limit)
    model.cache_prefix_kv=False
    with self.assertRaises(ValueError):model.prepare_observation()
    model.cache_prefix_kv=True
    model.prepare_observation(max_context=16)
    manager=model._prepared_observation
    values=rng.normal(0,.3,(1,1,16)).astype(np.float32)
    manager(fields(model,values),0)
    call=manager.calls[1]
    model.weights['fc.weight'].replace(model.weights['fc.weight'].clone().realize())
    with self.assertRaisesRegex(ValueError,'weights or projections changed'):manager(fields(model,values),1)
    failure=RuntimeError('injected observation close')
    call.jit.captured=SimpleNamespace(free_intermediates=Mock())
    with patch('tinygrad.llm.dflash2.Device',{'CPU':SimpleNamespace(synchronize=Mock(side_effect=failure))}):
      with self.assertRaises(RuntimeError) as caught:model.close_prepared_observation()
    self.assertIs(caught.exception,failure)
    self.assertIsNone(model._prepared_observation)
    self.assertTrue(manager.closed)
    self.assertIsNone(manager.draft)
    self.assertFalse(manager.calls)
    self.assertTrue(call.closed)
    self.assertIsNone(call.draft)
    self.assertIsNone(call.jit.fxn)
    model.close_prepared_observation()


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestPreparedObservationMali(unittest.TestCase):
  def test_fp16_projection_keys_and_owned_cache_match_eager_across_positions(self):
    raw,model,_c,_w,rng=setup(534,'CL',True)
    raw.prepare_mali()
    model.prepare_mali()
    model.prepare_observation(max_context=512)
    head=model.lm_head
    for position,rows in ((0,128),(128,1),(129,3),(132,1),(2,1),(3,8)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float16)
      for draft in (raw,model):draft.observe(observation(draft,values),position)
      np.testing.assert_allclose(model._target_hidden.numpy(),raw._target_hidden.numpy(),atol=3e-3,rtol=4e-3)
      for actual,expected in zip(model._prefix_kv,raw._prefix_kv):
        np.testing.assert_allclose(actual.key.numpy(),expected.key.numpy(),atol=3e-3,rtol=4e-3)
        np.testing.assert_allclose(actual.value.numpy(),expected.value.numpy(),atol=3e-3,rtol=4e-3)
    self.assertIs(model.lm_head,head)
    model.close_prepared_observation()


if __name__=='__main__':unittest.main()
