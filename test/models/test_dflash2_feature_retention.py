"""CPU contracts for append-only DFlash2 observation without retained projections."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.dflash2 import DFlash2
from tinygrad.llm.dflash2_selector import CandidateSelector
from tinygrad.llm.model import SSMConfig, TargetOutput, Transformer, TransformerConfig
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from test.models.test_dflash2_prepared import setup


def observed_output(draft, values):
  fields = tuple(Tensor(value, device=draft.device, dtype=draft.dtype).realize()
                 for value in np.split(values, 2, axis=-1))
  return TargetOutput(Tensor.zeros(1, values.shape[1], draft.config.hidden_size, device=draft.device),
                      fields[-1], fields)


def observe(draft, values, position):
  draft.observe(observed_output(draft, values), position)


def bits(tensor):
  return tensor.numpy().view(np.uint8).copy()


class TestDFlash2FeatureRetention(unittest.TestCase):
  @staticmethod
  def prepare_pair(seed=901, max_context=256):
    control, append_only, _rng = setup(seed)
    for draft in (control, append_only):
      draft.prepare_proposals(max_context=max_context)
    control.prepare_observation(max_context=max_context, pad_decode_rows=True)
    append_only.prepare_observation(max_context=max_context, pad_decode_rows=True,
                                    retain_target_features=False)
    return control, append_only

  def assert_prefix_equal(self, control, append_only, expected_rows):
    self.assertEqual(control.cached_prefix_length, expected_rows)
    self.assertEqual(append_only.cached_prefix_length, expected_rows)
    self.assertIsNotNone(control._target_hidden)
    self.assertEqual(control._target_hidden.shape, (1, expected_rows, control.config.hidden_size))
    self.assertEqual(control._target_hidden.nbytes(), expected_rows * control.config.hidden_size * 4)
    self.assertIsNone(append_only._target_hidden)
    metadata = append_only.observation_preparation_metadata()
    self.assertFalse(metadata['retain_target_features'])
    self.assertEqual(metadata['cached_prefix_rows'], expected_rows)
    self.assertEqual(metadata['retained_projected_history_bytes'], 0)
    self.assertEqual(len(control._prefix_kv), len(append_only._prefix_kv))
    for left, right in zip(control._prefix_kv, append_only._prefix_kv):
      self.assertIsNotNone(left)
      self.assertIsNotNone(right)
      self.assertEqual((left.start_pos, left.end_pos), (right.start_pos, right.end_pos))
      np.testing.assert_array_equal(bits(left.key), bits(right.key))
      np.testing.assert_array_equal(bits(left.value), bits(right.value))
    left_output = control._proposal_output(3, expected_rows).realize()
    right_output = append_only._proposal_output(3, expected_rows).realize()
    np.testing.assert_array_equal(left_output.logits.numpy().view(np.uint32),
                                  right_output.logits.numpy().view(np.uint32))
    np.testing.assert_array_equal(left_output.hidden.numpy().view(np.uint32),
                                  right_output.hidden.numpy().view(np.uint32))
    self.assertEqual(control.propose(3, position=expected_rows, max_tokens=4),
                     append_only.propose(3, position=expected_rows, max_tokens=4))

  def test_append_only_retention_matches_feature_control_and_eager_row_fallback(self):
    control, append_only = self.prepare_pair()
    rng = np.random.default_rng(902)
    position = 0
    for rows in (128, 1, 13, 3):
      values = rng.normal(0, .4, (1, rows, 16)).astype(np.float32)
      observe(control, values, position)
      observe(append_only, values, position)
      position += rows
      self.assert_prefix_equal(control, append_only, position)
      if rows == 13:
        self.assertNotIn(13, control._prepared_observation.calls)
        self.assertNotIn(13, append_only._prepared_observation.calls)
    control.reset()
    append_only.reset()
    self.assertEqual(control.cached_prefix_length, 0)
    self.assertEqual(append_only.cached_prefix_length, 0)
    self.assertIsNone(control._target_hidden)
    self.assertIsNone(append_only._target_hidden)
    self.assertTrue(all(cache is None for cache in control._prefix_kv))
    self.assertTrue(all(cache is None for cache in append_only._prefix_kv))
    values = rng.normal(0, .4, (1, 5, 16)).astype(np.float32)
    observe(control, values, 0)
    observe(append_only, values, 0)
    self.assert_prefix_equal(control, append_only, 5)
    control.close_prepared_observation()
    append_only.close_prepared_observation()
    control.close_prepared_proposals()
    append_only.close_prepared_proposals()

  def test_retain_mode_change_resets_projected_prefix_and_cache_state(self):
    _control, append_only = self.prepare_pair(seed=903)
    observe(append_only, np.ones((1, 7, 16), dtype=np.float32), 0)
    self.assertEqual(append_only.cached_prefix_length, 7)
    revision = append_only.cache_revision
    append_only.prepare_observation(max_context=256, retain_target_features=True)
    self.assertGreater(append_only.cache_revision, revision)
    self.assertIsNone(append_only._target_hidden)
    self.assertEqual(append_only.cached_prefix_length, 0)
    self.assertTrue(all(cache is None for cache in append_only._prefix_kv))
    self.assertTrue(append_only.observation_preparation_metadata().get('retain_target_features', True))
    append_only.close_prepared_observation()
    append_only.close_prepared_proposals()

  def test_append_only_rejects_nonboolean_missing_or_short_proposal_contract_and_no_kv_cache(self):
    _control, draft, _rng = setup(904)
    with self.assertRaises(ValueError):
      draft.prepare_observation(max_context=16, retain_target_features=1)
    with self.assertRaisesRegex(ValueError, 'requires prepared proposals'):
      draft.prepare_observation(max_context=16, retain_target_features=False)
    draft.prepare_proposals(max_context=16)
    with self.assertRaisesRegex(ValueError, 'covering the observation context'):
      draft.prepare_observation(max_context=32, retain_target_features=False)
    draft.close_prepared_proposals()
    draft, _unused, _rng = setup(905)
    draft.cache_prefix_kv = False
    with self.assertRaisesRegex(ValueError, 'requires prefix KV caching'):
      draft.prepare_observation(max_context=16, retain_target_features=False)
    self.assertIsNone(draft._prepared_observation)

  def test_suffix_replacement_invalidates_revision_without_changing_cached_rows(self):
    _control, draft = self.prepare_pair(seed=906, max_context=128)
    values = np.random.default_rng(907).normal(0, .3, (1, 8, 16)).astype(np.float32)
    observe(draft, values, 0)
    revision = draft.cache_revision
    cached = [(cache.start_pos, cache.end_pos, bits(cache.key), bits(cache.value)) for cache in draft._prefix_kv]
    with self.assertRaisesRegex(ValueError, 'cannot replace a suffix'):
      observe(draft, np.zeros((1, 2, 16), dtype=np.float32), 4)
    self.assertGreater(draft.cache_revision, revision)
    self.assertEqual(draft.cached_prefix_length, 8)
    for cache, (start, end, key, value) in zip(draft._prefix_kv, cached):
      self.assertEqual((cache.start_pos, cache.end_pos), (start, end))
      np.testing.assert_array_equal(bits(cache.key), key)
      np.testing.assert_array_equal(bits(cache.value), value)
    draft.close_prepared_observation()
    draft.close_prepared_proposals()

  def test_append_capacity_failure_preserves_prefix_kv_and_row_counter(self):
    draft, _unused, rng = setup(908)
    draft.prepare_proposals(max_context=128)
    draft.prepare_observation(max_context=128, pad_decode_rows=True, retain_target_features=False)
    observe(draft, rng.normal(0, .3, (1, 128, 16)).astype(np.float32), 0)
    cached = [(cache.start_pos, cache.end_pos, bits(cache.key), bits(cache.value)) for cache in draft._prefix_kv]
    revision = draft.cache_revision
    with self.assertRaisesRegex(ValueError, 'exceeds prepared proposal capacity'):
      observe(draft, rng.normal(0, .3, (1, 1, 16)).astype(np.float32), 128)
    self.assertGreater(draft.cache_revision, revision)
    self.assertEqual(draft.cached_prefix_length, 128)
    self.assertIsNone(draft._target_hidden)
    for cache, (start, end, key, value) in zip(draft._prefix_kv, cached):
      self.assertEqual((cache.start_pos, cache.end_pos), (start, end))
      np.testing.assert_array_equal(bits(cache.key), key)
      np.testing.assert_array_equal(bits(cache.value), value)
    draft.close_prepared_observation()
    draft.close_prepared_proposals()

  def test_closing_append_only_proposals_resets_rows_and_requires_repreparation(self):
    _control, draft = self.prepare_pair(seed=909, max_context=32)
    observe(draft, np.random.default_rng(910).normal(0, .3, (1, 5, 16)).astype(np.float32), 0)
    self.assertEqual(draft.cached_prefix_length, 5)
    draft.close_prepared_proposals()
    self.assertIsNone(draft._target_hidden)
    self.assertEqual(draft.cached_prefix_length, 0)
    self.assertTrue(all(cache is None for cache in draft._prefix_kv))
    with self.assertRaisesRegex(ValueError, 'requires prepared proposals'):
      draft.propose(3, position=0, max_tokens=4)
    draft.close_prepared_observation()

  def test_smaller_observation_graph_uses_eager_append_beyond_its_bounds(self):
    control,draft = self.prepare_pair(seed=911)
    values = np.random.default_rng(912).normal(0,.3,(1,128,16)).astype(np.float32)
    for model in (control,draft):observe(model,values,0)
    for model,retain in ((control,True),(draft,False)):
      model.prepare_observation(max_context=16,retain_target_features=retain)
      self.assertEqual(model.cached_prefix_length,128)
      observe(model,np.ones((1,1,16),np.float32),128)
      self.assertEqual(model._prepared_observation.calls,{})
    self.assert_prefix_equal(control,draft,129)

  def test_empty_append_and_partial_materialization_failure_keep_existing_prefix(self):
    control,draft = self.prepare_pair(seed=913,max_context=16)
    rng = np.random.default_rng(914)
    values = rng.normal(0,.3,(1,5,16)).astype(np.float32)
    for model in (control,draft):
      observe(model,np.empty((1,0,16),np.float32),0)
      observe(model,values,0)
      observe(model,np.empty((1,0,16),np.float32),5)
    self.assert_prefix_equal(control,draft,5)
    previous = draft._prefix_kv
    old = [(bits(cache.key),bits(cache.value)) for cache in previous]
    realize = Tensor.realize
    triggered = False
    def fail_partial(*values,**kwargs):
      nonlocal triggered
      if not triggered and any(value.shape==(1,1,6,4) and not value.uop.is_realized for value in values):
        triggered = True
        realize(values[0])
        raise RuntimeError('partial cache materialization')
      return realize(*values,**kwargs)
    update = rng.normal(0,.3,(1,1,16)).astype(np.float32)
    with patch.object(Tensor,'realize',fail_partial):
      with self.assertRaisesRegex(RuntimeError,'partial cache materialization'):observe(draft,update,5)
    self.assertTrue(triggered)
    self.assertIs(draft._prefix_kv,previous)
    self.assertEqual(draft.cached_prefix_length,5)
    self.assertIsNone(draft._target_hidden)
    for cache,(key,value) in zip(previous,old):
      np.testing.assert_array_equal(bits(cache.key),key)
      np.testing.assert_array_equal(bits(cache.value),value)
    for model in (control,draft):observe(model,update,5)
    self.assert_prefix_equal(control,draft,6)

  def test_real_coordinator_reuse_fallback_and_close_match_retaining_control(self):
    base,_,rng = setup(915)
    cfg = TransformerConfig(3,8,16,2,1,1e-5,16,4,10000.,4,4,max_context=24)
    cfg = replace(cfg,ssm=SSMConfig(3,4,1,1,8),ssm_layers=(True,False,True))
    targets = [Transformer(cfg),Transformer(cfg)]
    for name,param in nn.state.get_state_dict(targets[0]).items():
      values = rng.normal(0,.2,param.shape).astype(np.float32)
      if 'norm.weight' in name:values += 1
      if name.endswith('ssm_a'):values = -rng.uniform(.5,2,param.shape).astype(np.float32)
      param.replace(Tensor(values,device='CPU').realize())
    source = nn.state.get_state_dict(targets[0])
    for name,param in nn.state.get_state_dict(targets[1]).items():param.replace(source[name].clone().realize())
    decoders = []
    for target,retain in zip(targets,(True,False)):
      draft = DFlash2(base.config,base.weights,target.token_embd,target.output,target.output_norm,
                     CandidateSelector(base.config,base.weights))
      draft.prepare_proposals(max_context=24)
      draft.prepare_observation(max_context=24,pad_decode_rows=True,retain_target_features=retain)
      decoders.append(GreedySpeculativeDecoder(target,draft,early_anchor=True,deferred_fallback=True,reuse_prefix=True))
    def generate(prompt,count,**kwargs):
      outputs = [list(model.generate(prompt,max_new_tokens=count,**kwargs)) for model in decoders]
      self.assertEqual(outputs[0],outputs[1])
      self.assertEqual(targets[0]._cached_tokens,targets[1]._cached_tokens)
      for left,right in zip(targets[0].blk,targets[1].blk):
        for name in ('cache_kv','conv_state','recurrent_state'):
          if hasattr(left,name):np.testing.assert_array_equal(bits(getattr(left,name)),bits(getattr(right,name)))
      return outputs[0]
    first = generate([2,5,3],4)
    prompt = [2,5,3,*first,6]
    generate(prompt,3)
    self.assertEqual(decoders[1].stats.cached_tokens,7)
    generate([2,5,3],8,ordinary_decode_seconds=1e-12,max_speculative_extra_seconds=1e-12)
    self.assertGreater(decoders[1].stats.fallback_calls,0)
    self.assertEqual(decoders[1].draft.cached_prefix_length,0)
    for model in decoders:
      stream = model.generate([2,5,3],max_new_tokens=4)
      next(stream)
      stream.close()
      self.assertEqual(model.draft.cached_prefix_length,0)
    generate([4,6],2)
