"""CPU-only causal tests for opt-in speculative target/draft prefix reuse."""
from typing import cast
import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from test.models.test_speculative import CausalTarget, OracleDraft, oracle, ordinary_ids, real_pair


class ReusableOracleDraft(OracleDraft):
  def __init__(self, wrong_at=None):
    self.cache_revision = 0
    self.reset_calls = 0
    super().__init__(wrong_at)

  @property
  def cached_prefix_length(self):
    return len(self.seen)

  def reset(self):
    self.cache_revision += 1
    self.reset_calls += 1
    super().reset()

  def observe(self, output, start_pos):
    super().observe(output, start_pos)
    self.cache_revision += 1


class ReusableCausalTarget(CausalTarget):
  has_recurrent_block = True
  get_start_pos = Transformer.get_start_pos


class EmptyReusableDraft:
  block_size,capture_layers = 4,(0,)
  cache_revision,cached_prefix_length = 0,0
  def reset(self):
    self.cache_revision += 1
    self.cached_prefix_length = 0
  def observe(self,output,start_pos):
    if start_pos>self.cached_prefix_length:raise ValueError('Gapped prefix')
    self.cache_revision += 1
    self.cached_prefix_length = start_pos+output.features[0].shape[1]
  def propose(self,anchor,*,position,max_tokens):return []


class TestSpeculativePrefixReuse(unittest.TestCase):
  @staticmethod
  def decoder(*, context=64, wrong_at=None, reuse_prefix=True):
    target = ReusableCausalTarget(context)
    draft = ReusableOracleDraft(wrong_at)
    decoder = GreedySpeculativeDecoder(cast(Transformer, target), draft, reuse_prefix=reuse_prefix)
    return target, draft, decoder

  @staticmethod
  def spy_target(target):
    calls = []
    original = target.target_forward

    def traced(tokens, start_pos, **kwargs):
      calls.append((start_pos, list(tokens.tolist()[0])))
      return original(tokens, start_pos, **kwargs)

    target.target_forward = traced
    return calls

  @staticmethod
  def assert_causal_state(test, target, prompt, generated):
    expected, state, history = oracle(prompt, len(generated))
    test.assertEqual(generated, expected)
    test.assertEqual(target._cached_tokens, prompt + generated)
    test.assertEqual(int(target.blk[0].recurrent_state.item()), state)
    test.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]), history)
    test.assertEqual(target.blk[0].cache_kv.numpy()[:len(prompt + generated)].tolist(), prompt + generated)

  def seed_prefix(self, target, decoder, prompt=(2, 5, 3), output_tokens=4):
    tokens = list(prompt)
    generated = list(decoder.generate(tokens, max_new_tokens=output_tokens))
    self.assert_causal_state(self, target, tokens, generated)
    return list(target._cached_tokens)

  def test_strict_extension_reuses_exact_target_and_draft_prefix(self):
    target, draft, decoder = self.decoder()
    prefix = self.seed_prefix(target, decoder)
    continuation = prefix + [12, 6]
    self.assertEqual(decoder.get_start_pos(continuation), len(prefix))
    reset_count = draft.reset_calls
    calls = self.spy_target(target)

    actual = list(decoder.generate(continuation, max_new_tokens=6))
    expected = oracle(continuation, len(actual))[0]
    self.assertEqual(actual, expected)
    self.assertEqual(calls[0], (len(prefix), continuation[len(prefix):]))
    self.assertLess(len(calls[0][1]), len(continuation))
    self.assertEqual(draft.reset_calls, reset_count)
    self.assert_causal_state(self, target, continuation, actual)

    fresh_target, fresh_draft, fresh_decoder = self.decoder(reuse_prefix=False)
    fresh = list(fresh_decoder.generate(continuation, max_new_tokens=6))
    self.assertEqual(actual, fresh)
    self.assertEqual(int(target.blk[0].recurrent_state.item()),
                     int(fresh_target.blk[0].recurrent_state.item()))
    self.assertEqual(target.blk[0].conv_state.tolist(), fresh_target.blk[0].conv_state.tolist())
    self.assertEqual(target.blk[0].cache_kv.numpy().tolist(), fresh_target.blk[0].cache_kv.numpy().tolist())
    self.assertEqual(fresh_draft.seen, [])

  def test_divergent_equal_and_shorter_prompts_are_cold(self):
    target, _, decoder = self.decoder()
    prefix = self.seed_prefix(target, decoder)
    cases = (prefix[:-1] + [16], prefix, prefix[:-1])
    for prompt in cases:
      with self.subTest(prompt_length=len(prompt), exact=prompt == prefix):
        self.assertEqual(decoder.get_start_pos(prompt), 0)
    divergent = cases[0]
    calls = self.spy_target(target)
    generated = list(decoder.generate(divergent, max_new_tokens=4))
    self.assertEqual(calls[0], (0, divergent))
    self.assert_causal_state(self, target, divergent, generated)

  def test_equal_token_ids_in_a_replaced_target_marker_do_not_reuse(self):
    target, _, decoder = self.decoder()
    prefix = self.seed_prefix(target, decoder)
    extension = prefix + [9]
    old_marker = target._cached_tokens
    target._cached_tokens = list(old_marker)
    self.assertEqual(target._cached_tokens, prefix)
    self.assertIsNot(target._cached_tokens, old_marker)
    self.assertEqual(decoder.get_start_pos(extension), 0)
    calls = self.spy_target(target)
    generated = list(decoder.generate(extension, max_new_tokens=3))
    self.assertEqual(calls[0], (0, extension))
    self.assert_causal_state(self, target, extension, generated)

  def test_ordinary_interleaving_with_same_ids_invalidates_prior_draft_alignment(self):
    target, _, decoder = self.decoder()
    prefix = self.seed_prefix(target, decoder)
    extension = prefix + [4]
    # Model an ordinary cold target forward over the same IDs between two speculative calls.
    target._cached_tokens = []
    ordinary_output = target.target_forward(Tensor([prefix], device='CPU', dtype='int32'), 0)
    self.assertEqual(ordinary_output.logits.shape[1], len(prefix))
    target._cached_tokens = list(prefix)
    self.assertEqual(decoder.get_start_pos(extension), 0)
    calls = self.spy_target(target)
    generated = list(decoder.generate(extension, max_new_tokens=3))
    self.assertEqual(calls[0], (0, extension))
    self.assert_causal_state(self, target, extension, generated)

  def test_draft_reset_revision_invalidates_even_if_feature_length_is_restored(self):
    target, draft, decoder = self.decoder()
    prefix = self.seed_prefix(target, decoder)
    extension = prefix + [10]
    cached_features = list(draft.seen)
    revision = draft.cache_revision
    draft.reset()
    draft.seen = cached_features
    self.assertEqual(draft.cached_prefix_length, len(prefix))
    self.assertGreater(draft.cache_revision, revision)
    self.assertEqual(decoder.get_start_pos(extension), 0)

  def test_partial_fallback_lag_is_not_admitted_as_a_reusable_prefix(self):
    target, draft, decoder = self.decoder()
    prompt = [2, 5, 3]
    generated = list(decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                     max_speculative_extra_seconds=1e-12))
    self.assert_causal_state(self, target, prompt, generated)
    self.assertGreater(decoder.stats.fallback_tokens, 0)
    self.assertLess(draft.cached_prefix_length, len(target._cached_tokens))
    self.assertEqual(decoder.get_start_pos(list(target._cached_tokens) + [7]), 0)

  def test_cancel_close_and_error_leave_the_next_request_cold(self):
    for interruption in ('cancel', 'close', 'error'):
      with self.subTest(interruption=interruption):
        target, _, decoder = self.decoder()
        prefix = self.seed_prefix(target, decoder)
        continuation = prefix + [11]
        if interruption == 'cancel':
          self.assertEqual(list(decoder.generate(continuation, max_new_tokens=5, cancelled=lambda: True)), [])
        elif interruption == 'close':
          stream = decoder.generate(continuation, max_new_tokens=8)
          next(stream)
          stream.close()
        else:
          target.fail_once = True
          with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            list(decoder.generate(continuation, max_new_tokens=5))
        self.assertEqual(decoder.get_start_pos(continuation + [3]), 0)
        calls = self.spy_target(target)
        generated = list(decoder.generate(continuation, max_new_tokens=3))
        self.assertEqual(calls[0], (0, continuation))
        self.assert_causal_state(self, target, continuation, generated)

  def test_default_reuse_is_disabled_and_each_request_starts_cold(self):
    target = CausalTarget()
    draft = ReusableOracleDraft()
    decoder = GreedySpeculativeDecoder(cast(Transformer, target), draft)
    prompt = [2, 5, 3]
    first = list(decoder.generate(prompt, max_new_tokens=4))
    prefix = list(target._cached_tokens)
    extension = prefix + [8]
    self.assertEqual(decoder.get_start_pos(extension), 0)
    reset_count = draft.reset_calls
    calls = self.spy_target(target)
    second = list(decoder.generate(extension, max_new_tokens=3))
    self.assertEqual(calls[0], (0, extension))
    self.assertGreater(draft.reset_calls, reset_count)
    self.assert_causal_state(self, target, extension, second)
    self.assertEqual(first, oracle(prompt, len(first))[0])

  def test_invalid_flag_and_missing_draft_cache_contract_are_rejected(self):
    target = CausalTarget()
    with self.assertRaises((TypeError, ValueError)):
      GreedySpeculativeDecoder(cast(Transformer, target), ReusableOracleDraft(), reuse_prefix=1)
    with self.assertRaises((TypeError, ValueError)):
      GreedySpeculativeDecoder(cast(Transformer, target), OracleDraft(), reuse_prefix=True)
    self.assertEqual(target._cached_tokens, [])
    self.assertEqual(int(target.blk[0].recurrent_state.item()), 0)

  def test_early_final_anchor_retains_matching_draft_observation_for_next_request(self):
    target,draft = ReusableCausalTarget(),ReusableOracleDraft()
    decoder = GreedySpeculativeDecoder(cast(Transformer,target),draft,reuse_prefix=True,early_anchor=True)
    prompt = [2,5,3]
    actual = list(decoder.generate(prompt,max_new_tokens=1))
    self.assert_causal_state(self,target,prompt,actual)
    prefix = target._cached_tokens.copy()
    self.assertEqual(draft.cached_prefix_length,len(prefix))
    self.assertEqual(decoder.get_start_pos(prefix+[9]),len(prefix))
    self.assert_causal_state(self,target,prefix+[9],list(decoder.generate(prefix+[9],max_new_tokens=1)))
    self.assertEqual(decoder.stats.cached_tokens,len(prefix))

  def test_real_hybrid_direct_forward_invalidates_reuse_without_token_marker_publication(self):
    for operation in ('logits','target_forward','logits_from_embeddings'):
      with self.subTest(operation=operation):
        reference,target = real_pair(True)
        decoder = GreedySpeculativeDecoder(target,EmptyReusableDraft(),reuse_prefix=True,early_anchor=True)
        first = list(decoder.generate([2,5,3],max_new_tokens=1))
        self.assertEqual(first,ordinary_ids(reference,[2,5,3],1))
        prefix = target._cached_tokens.copy()
        extension = prefix+[6]
        self.assertEqual(decoder.get_start_pos(extension),len(prefix))
        tokens = Tensor([[7]],device='CPU',dtype='int32')
        if operation=='logits_from_embeddings':
          target.logits_from_embeddings(target.token_embd(tokens).float(),0).realize()
        elif operation=='logits':target.logits(tokens,0).realize()
        else:target.target_forward(tokens,0).realize()
        self.assertEqual(target._cached_tokens,[])
        self.assertEqual(decoder.get_start_pos(extension),0)
        actual = list(decoder.generate(extension,max_new_tokens=2))
        self.assertEqual(actual,ordinary_ids(reference,extension,2))
        self.assertEqual(decoder.stats.cached_tokens,0)

  def test_real_transformer_and_hybrid_cached_followups_match_fresh_output_and_state(self):
    for hybrid in (False,True):
      with self.subTest(hybrid=hybrid):
        reference,target = real_pair(hybrid)
        decoder = GreedySpeculativeDecoder(target,EmptyReusableDraft(),reuse_prefix=True,early_anchor=True)
        first = list(decoder.generate([2,5,3],max_new_tokens=1))
        extension = [2,5,3,*first,6,7]
        cached = len(target._cached_tokens)
        actual = list(decoder.generate(extension,max_new_tokens=2))
        self.assertEqual(actual,ordinary_ids(reference,extension,2))
        self.assertEqual(decoder.stats.cached_tokens,cached)
        for block,expected in zip(target.blk,reference.blk):
          for name in ('recurrent_state','conv_state'):
            if hasattr(block,name):
              np.testing.assert_allclose(getattr(block,name).numpy(),getattr(expected,name).numpy(),atol=1e-6,rtol=1e-5)
