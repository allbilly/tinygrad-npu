"""Owned fallback features, bounded batching and causal cache lifetime contracts."""
from typing import cast
import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import TargetOutput, Transformer
from tinygrad.llm.speculative import GreedySpeculativeDecoder, SpeculativeStats, _FallbackFeatureBatch
from test.models.test_speculative import oracle
from test.models.test_speculative_prefix_reuse import ReusableCausalTarget, ReusableOracleDraft


class FeatureDraft(ReusableOracleDraft):
  def __init__(self):
    self.feature_calls = []
    self.fail_features = False
    super().__init__()

  def observe_features(self, features, start_pos):
    self.feature_calls.append((start_pos,features[0].shape[1]))
    self.cache_revision += 1
    if self.fail_features:raise RuntimeError('injected batch observation failure')
    if start_pos!=len(self.seen):raise ValueError('feature-only observation must append')
    self.seen += features[0].tolist()[0]


def make_decoder(*, deferred=True, early=False, batch=8, mutable=False):
  target, draft = ReusableCausalTarget(128), FeatureDraft()
  decoder = GreedySpeculativeDecoder(cast(Transformer,target),draft,reuse_prefix=True,retain_fallback_prefix=True,
    deferred_fallback=deferred,early_anchor=early,fallback_observation_batch_size=batch)
  if mutable:
    original = target.target_forward
    shared = Tensor.zeros(1,1,4,device='CPU').contiguous().realize()
    def forward(tokens, start_pos, **kwargs):
      output = original(tokens,start_pos,**kwargs)
      if decoder.stats.fallback_calls:
        shared.assign(output.features[0]).realize()
        return TargetOutput(output.logits,output.hidden,(shared,))
      return output
    target.target_forward = forward
  return target,draft,decoder


class TestFallbackFeatureBatch(unittest.TestCase):
  def test_copies_expiring_features_only_and_flushes_bounded_groups(self):
    stats, calls = SpeculativeStats(), []
    def observe(features, position):calls.append((position,[x.numpy().copy() for x in features]))
    batch = _FallbackFeatureBatch(8,observe,stats)
    left = Tensor.zeros(1,1,4,device='CPU').contiguous().realize()
    right = Tensor.zeros(1,1,2,device='CPU').contiguous().realize()
    cloned = []
    original = Tensor.clone
    def clone(value, *args, **kwargs):
      cloned.append(id(value))
      return original(value,*args,**kwargs)
    with patch.object(Tensor,'clone',clone):
      for i in range(11):
        left.assign(Tensor.full((1,1,4),float(i),device='CPU')).realize()
        right.assign(Tensor.full((1,1,2),float(100+i),device='CPU')).realize()
        batch.append((left,right),20+i)
        self.assertLess(len(batch.rows),8)
      # Overwrite the graph outputs once more before the final tail flush.
      left.assign(Tensor.full_like(left,-1)).realize()
      right.assign(Tensor.full_like(right,-1)).realize()
      batch.flush()
    self.assertEqual(cloned,[id(x) for _ in range(11) for x in (left,right)])
    self.assertEqual([(pos,features[0].shape[1]) for pos,features in calls],[(20,8),(28,3)])
    for pos,features in calls:
      for row in range(features[0].shape[1]):
        np.testing.assert_array_equal(features[0][0,row],np.full(4,pos-20+row))
        np.testing.assert_array_equal(features[1][0,row],np.full(2,100+pos-20+row))
    self.assertEqual(stats.fallback_feature_copy_bytes,11*6*4)
    self.assertEqual(stats.fallback_feature_peak_rows,8)
    self.assertEqual(stats.fallback_observation_calls,2)
    self.assertFalse(batch.rows)

  def test_causal_state_reuse_and_no_extra_target_work_with_reused_graph_buffer(self):
    prompt = [2,5,3]
    for deferred,early in ((False,False),(True,False),(True,True)):
      with self.subTest(deferred=deferred,early=early):
        target,draft,decoder = make_decoder(deferred=deferred,early=early,mutable=True)
        output = list(decoder.generate(prompt,max_new_tokens=20,ordinary_decode_seconds=1e-12))
        self.assertEqual(output,oracle(prompt,20)[0])
        processed = prompt+(output[:-1] if deferred else output)
        self.assertEqual(target._cached_tokens,processed)
        self.assertEqual(draft.cached_prefix_length,len(processed))
        self.assertEqual([rows for _,rows in draft.feature_calls],[8,3 if deferred else 4])
        self.assertEqual(decoder.stats.fallback_observation_calls,2)
        self.assertEqual(decoder.stats.fallback_feature_peak_rows,8)
        self.assertEqual(decoder.stats.fallback_feature_copy_bytes,decoder.stats.fallback_calls*4*4)
        self.assertEqual(decoder.get_start_pos(prompt+output+[12]),len(processed))
        seed_calls = decoder.stats.target_calls
        _,_,scalar = make_decoder(deferred=deferred,early=early,batch=1)
        self.assertEqual(list(scalar.generate(prompt,max_new_tokens=20,ordinary_decode_seconds=1e-12)),output)
        self.assertEqual(scalar.stats.target_calls,seed_calls)
        continuation = prompt+output+[12]
        actual = list(decoder.generate(continuation,max_new_tokens=4))
        self.assertEqual(decoder.stats.cached_tokens,len(processed))
        cold_target,_,cold = make_decoder(deferred=deferred,early=early)
        self.assertEqual(actual,list(cold.generate(continuation,max_new_tokens=4)))
        self.assertEqual(target.blk[0].recurrent_state.tolist(),cold_target.blk[0].recurrent_state.tolist())
        self.assertEqual(target.blk[0].conv_state.tolist(),cold_target.blk[0].conv_state.tolist())
        np.testing.assert_array_equal(target.blk[0].cache_kv.numpy(),cold_target.blk[0].cache_kv.numpy())

  def test_eos_flushes_processed_features_and_excludes_pending_eos(self):
    prompt = [2,5,3]
    expected = oracle(prompt,24)[0]
    eos_index = next(i for i in range(18,24) if expected[i] not in expected[:i])
    for deferred in (False,True):
      with self.subTest(deferred=deferred):
        target,draft,decoder = make_decoder(deferred=deferred)
        output = list(decoder.generate(prompt,max_new_tokens=24,eos_token_ids=[expected[eos_index]],ordinary_decode_seconds=1e-12))
        self.assertEqual(output,expected[:eos_index+1])
        self.assertEqual(decoder.finish_reason,'eos')
        self.assertEqual(target._cached_tokens,prompt+(output[:-1] if deferred else output))
        self.assertEqual(draft.cached_prefix_length,len(target._cached_tokens))
        self.assertEqual(sum(n for _,n in draft.feature_calls),decoder.stats.fallback_calls)
        self.assertEqual(decoder.get_start_pos(prompt+output+[12]),len(target._cached_tokens))

  def test_partial_queue_is_discarded_on_cancel_or_close(self):
    for deferred in (False,True):
      for mode in ('cancel','close'):
        with self.subTest(deferred=deferred,mode=mode):
          target,draft,decoder = make_decoder(deferred=deferred)
          stop = [False]
          stream = decoder.generate([2,5,3],max_new_tokens=24,ordinary_decode_seconds=1e-12,cancelled=lambda:stop[0])
          while decoder.stats.fallback_calls<2:next(stream)
          self.assertEqual(draft.feature_calls,[])
          if mode=='close':stream.close()
          else:
            stop[0] = True
            self.assertEqual(list(stream),[])
          self.assertEqual(draft.cached_prefix_length,0)
          self.assertEqual(decoder.get_start_pos(list(target._cached_tokens)+[7]),0)
          self.assertFalse(decoder._lock.locked())
          retry = list(decoder.generate([9,3],max_new_tokens=4))
          self.assertEqual(decoder.stats.cached_tokens,0)
          self.assertEqual(retry,oracle([9,3],4)[0])

  def test_full_and_tail_flush_errors_invalidate_reuse_and_preserve_delivered_counter(self):
    for deferred in (False,True):
      for budget in (12,20):
        with self.subTest(deferred=deferred,budget=budget):
          target,draft,decoder = make_decoder(deferred=deferred)
          draft.fail_features = True
          delivered = []
          with self.assertRaisesRegex(RuntimeError,'batch observation failure'):
            for token in decoder.generate([2,5,3],max_new_tokens=budget,ordinary_decode_seconds=1e-12):delivered.append(token)
          self.assertEqual(decoder.stats.output_tokens,len(delivered))
          self.assertEqual(decoder.finish_reason,'error')
          self.assertEqual(target._cached_tokens,[])
          self.assertEqual(draft.cached_prefix_length,0)
          self.assertFalse(decoder._lock.locked())
          draft.fail_features = False
          self.assertEqual(list(decoder.generate([9,3],max_new_tokens=4)),oracle([9,3],4)[0])
          self.assertEqual(decoder.stats.cached_tokens,0)

  def test_cancellation_during_full_or_final_flush_clears_draft_reuse(self):
    prompt = [2,5,3]
    for deferred in (False,True):
      for budget in (12,20):
        with self.subTest(deferred=deferred,budget=budget):
          target,draft,decoder = make_decoder(deferred=deferred)
          stop = [False]
          original = draft.observe_features
          def cancel_during_observe(features, position):
            original(features,position)
            stop[0] = True
          draft.observe_features = cancel_during_observe
          output = list(decoder.generate(prompt,max_new_tokens=budget,ordinary_decode_seconds=1e-12,cancelled=lambda:stop[0]))
          self.assertEqual(output,oracle(prompt,len(output))[0])
          self.assertEqual(decoder.stats.output_tokens,len(output))
          self.assertEqual(decoder.finish_reason,'cancelled')
          expected = prompt+(output[:-1] if decoder.stats.fallback_pending_last_token else output)
          self.assertEqual(target._cached_tokens,expected)
          self.assertEqual(draft.cached_prefix_length,0)
          self.assertEqual(decoder.get_start_pos(expected+[12]),0)
          self.assertFalse(decoder._lock.locked())

  def test_validation(self):
    for value in (0,9,True,1.5,None):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError,'fallback_observation_batch_size'):make_decoder(batch=value)
    target = ReusableCausalTarget()
    with self.assertRaisesRegex(ValueError,'Fallback batching'):
      GreedySpeculativeDecoder(cast(Transformer,target),FeatureDraft(),reuse_prefix=True,fallback_observation_batch_size=8)
    with self.assertRaisesRegex(ValueError,'feature-only'):
      GreedySpeculativeDecoder(cast(Transformer,target),ReusableOracleDraft(),reuse_prefix=True,retain_fallback_prefix=True,
                              fallback_observation_batch_size=8)
