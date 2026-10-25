"""CPU tests for opt-in target/draft prefix retention across fallback."""
from typing import cast
import unittest
from tinygrad.llm.model import Transformer
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from test.models.test_speculative import oracle
from test.models.test_speculative_prefix_reuse import ReusableCausalTarget, ReusableOracleDraft


class TestFallbackPrefixRetention(unittest.TestCase):
  def make(self, *, deferred=False, retain=True, reuse=True, early=False):
    target, draft = ReusableCausalTarget(), ReusableOracleDraft()
    decoder = GreedySpeculativeDecoder(cast(Transformer, target), draft, reuse_prefix=reuse,
      retain_fallback_prefix=retain, deferred_fallback=deferred, early_anchor=early)
    return target, draft, decoder

  def assert_processed_state(self, target, prompt, output, *, pending):
    processed = prompt + (output[:-1] if pending else output)
    _, state, history = oracle(prompt, len(processed) - len(prompt))
    self.assertEqual(target._cached_tokens, processed)
    self.assertEqual(int(target.blk[0].recurrent_state.item()), state)
    self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]), history)
    self.assertEqual(target.blk[0].cache_kv.numpy()[:len(processed)].tolist(), processed)

  @staticmethod
  def trace(target):
    calls = []
    original = target.target_forward
    def run(tokens, start_pos, **kwargs):
      calls.append((start_pos, list(tokens.tolist()[0])))
      return original(tokens, start_pos, **kwargs)
    target.target_forward = run
    return calls

  def test_fallback_retains_aligned_prefix_and_reuses_strict_continuation(self):
    prompt = [2, 5, 3]
    for deferred in (False, True):
      with self.subTest(deferred=deferred):
        target, draft, decoder = self.make(deferred=deferred)
        calls = self.trace(target)
        output = list(decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                       cancelled=lambda: False))
        self.assertEqual(output, oracle(prompt, len(output))[0])
        self.assertTrue(decoder.stats.retain_fallback_prefix)
        self.assertGreater(decoder.stats.fallback_tokens, 0)
        self.assertEqual(decoder.stats.fallback_pending_last_token, deferred)
        self.assert_processed_state(target, prompt, output, pending=deferred)
        processed = prompt + (output[:-1] if deferred else output)
        self.assertEqual(draft.cached_prefix_length, len(processed))
        self.assertEqual(len(draft.seen), len(processed))
        continuation = prompt + output + [12, 6]
        self.assertEqual(decoder.get_start_pos(continuation), len(processed))
        fallback_target_calls = decoder.stats.target_calls

        start = len(calls)
        actual = list(decoder.generate(continuation, max_new_tokens=6))
        self.assertEqual(calls[start][0], len(processed))
        self.assertEqual(actual, oracle(continuation, len(actual))[0])
        self.assert_processed_state(target, continuation, actual, pending=False)

        cold_target, _, cold = self.make(retain=False)
        expected = list(cold.generate(continuation, max_new_tokens=6))
        self.assertEqual(actual, expected)
        self.assertEqual(int(target.blk[0].recurrent_state.item()), int(cold_target.blk[0].recurrent_state.item()))
        self.assertEqual(target.blk[0].conv_state.tolist(), cold_target.blk[0].conv_state.tolist())
        self.assertEqual(target.blk[0].cache_kv.numpy().tolist(), cold_target.blk[0].cache_kv.numpy().tolist())

        # Observing existing fallback outputs must not add a target forward.
        _, _, cold_fallback = self.make(retain=False, deferred=deferred)
        cold_fallback_output = list(cold_fallback.generate(prompt, max_new_tokens=12,
          ordinary_decode_seconds=1e-12, cancelled=lambda: False))
        self.assertEqual(output, cold_fallback_output)
        self.assertEqual(fallback_target_calls, cold_fallback.stats.target_calls)

  def test_eos_retains_only_processed_fallback_inputs(self):
    prompt = [2, 5, 3]
    expected = oracle(prompt, 12)[0]
    eos_index = next(i for i in range(9, 12) if expected[i] not in expected[:i])
    for deferred in (False, True):
      with self.subTest(deferred=deferred):
        target, draft, decoder = self.make(deferred=deferred)
        output = list(decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                       eos_token_ids=[expected[eos_index]]))
        self.assertEqual(output, expected[:eos_index+1])
        self.assertEqual(decoder.finish_reason, 'eos')
        self.assertGreater(decoder.stats.fallback_calls, 0)
        self.assert_processed_state(target, prompt, output, pending=deferred)
        prefix = list(target._cached_tokens)
        self.assertEqual(draft.cached_prefix_length, len(prefix))
        self.assertEqual(decoder.get_start_pos(prefix+[12]), len(prefix))
        self.assertEqual(decoder.stats.fallback_calls, decoder.stats.fallback_tokens-int(deferred))

  def test_early_anchor_fallback_retains_the_processed_prefix(self):
    prompt = [2, 5, 3]
    target, draft, decoder = self.make(deferred=True,early=True)
    output = list(decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12))
    self.assertGreater(decoder.stats.fallback_calls,0)
    self.assertEqual(output,oracle(prompt,len(output))[0])
    self.assert_processed_state(target,prompt,output,pending=True)
    self.assertEqual(draft.cached_prefix_length,len(target._cached_tokens))
    self.assertEqual(decoder.get_start_pos(prompt+output+[12]),len(target._cached_tokens))

  def test_cancel_close_and_observation_error_leave_retry_cold(self):
    prompt = [2, 5, 3]
    for deferred in (False, True):
      for interruption in ('cancel', 'close', 'error'):
        with self.subTest(deferred=deferred, interruption=interruption):
          target, draft, decoder = self.make(deferred=deferred)
          calls = self.trace(target)
          stop = [False]
          original_observe = draft.observe
          if interruption == 'error':
            def fail_fallback(output, start_pos):
              if decoder.stats.fallback_calls:raise RuntimeError('injected fallback observation failure')
              original_observe(output, start_pos)
            draft.observe = fail_fallback
            delivered = []
            with self.assertRaisesRegex(RuntimeError, 'observation failure'):
              for token in decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12):delivered.append(token)
            self.assertEqual(decoder.stats.output_tokens,len(delivered))
          else:
            stream = decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                      cancelled=lambda: stop[0])
            delivered = []
            while not decoder.stats.fallback_tokens:delivered.append(next(stream))
            delivered.append(next(stream))
            if interruption == 'cancel':
              stop[0] = True
              self.assertEqual(list(stream), [])
            else:stream.close()
          self.assertGreater(decoder.stats.fallback_calls,0)
          self.assertEqual(decoder.get_start_pos(list(target._cached_tokens)+[7]),0)
          self.assertEqual(draft.cached_prefix_length,0)
          draft.observe = original_observe
          retry_start = len(calls)
          retry = list(decoder.generate([9, 3], max_new_tokens=4))
          self.assertEqual(calls[retry_start], (0, [9, 3]))
          self.assertEqual(retry, oracle([9, 3], len(retry))[0])
          self.assert_processed_state(target, [9, 3], retry, pending=False)

  def test_cancel_during_non_deferred_forward_rolls_back_before_observation(self):
    prompt = [2, 5, 3]
    target, draft, decoder = self.make()
    stop,observed = [False],[]
    original_forward,original_observe = target.target_forward,draft.observe
    def interrupt(tokens, start_pos, **kwargs):
      result = original_forward(tokens,start_pos,**kwargs)
      if decoder.stats.fallback_calls:stop[0] = True
      return result
    def observe(output, start_pos):
      observed.append(start_pos)
      original_observe(output,start_pos)
    target.target_forward,draft.observe = interrupt,observe
    output = list(decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12,cancelled=lambda:stop[0]))
    self.assertGreater(decoder.stats.fallback_calls,0)
    self.assertEqual(output,oracle(prompt,len(output))[0])
    self.assert_processed_state(target,prompt,output,pending=False)
    self.assertFalse(any(position>=len(prompt)+len(output) for position in observed))
    self.assertEqual(draft.cached_prefix_length,0)
    self.assertEqual(decoder.get_start_pos(list(target._cached_tokens)+[12]),0)
    self.assertFalse(decoder._lock.locked())

  def test_option_validation_and_default_remain_cold(self):
    target, draft = ReusableCausalTarget(), ReusableOracleDraft()
    with self.assertRaisesRegex(ValueError, 'retain_fallback_prefix'):
      GreedySpeculativeDecoder(cast(Transformer, target), draft, retain_fallback_prefix=True)
    for invalid in (1, None, 'yes'):
      with self.subTest(invalid=invalid):
        with self.assertRaisesRegex(ValueError, 'retain_fallback_prefix'):
          GreedySpeculativeDecoder(cast(Transformer, ReusableCausalTarget()), ReusableOracleDraft(),
            reuse_prefix=True, retain_fallback_prefix=invalid)
    target, draft = ReusableCausalTarget(), ReusableOracleDraft()
    decoder = GreedySpeculativeDecoder(cast(Transformer, target), draft, reuse_prefix=True)
    self.assertFalse(decoder.retain_fallback_prefix)
    prompt = [2, 5, 3]
    output = list(decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                   cancelled=lambda: False))
    self.assertEqual(output, oracle(prompt, len(output))[0])
    processed = prompt + output
    self.assertEqual(decoder.get_start_pos(processed + [12]), 0)
