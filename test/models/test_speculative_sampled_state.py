"""CPU-only sampled speculative state-machine regressions with independent oracles."""
from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import TargetOutput
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from tinygrad.llm.speculative_sampling import SampledProposal, SparseDistribution, categorical
from test.models.test_speculative import CausalTarget, OracleDraft, real_pair, transition


class NondegenerateCausalTarget(CausalTarget):
  """Keep the causal state transition; give every row a nondegenerate p(token)."""
  def target_forward(self, tokens, start_pos, **kwargs):
    output = super().target_forward(tokens, start_pos, **kwargs)
    rows = output.logits.shape[1]
    scores = np.full((1, rows, self.config.vocab_size), -64.0, dtype=np.float32)
    scores[:, :, 0:2] = 0.0
    logits = Tensor(scores, device='CPU').realize()
    return TargetOutput(logits, output.hidden, output.features)


class PointMassDraft:
  capture_layers = (0,)
  def __init__(self, block_size=4): self.block_size, self.seen = block_size, []
  def reset(self): self.seen = []
  def observe(self, output, start_pos):
    rows = output.features[0].tolist()[0]
    self.seen = self.seen[:start_pos] + rows
  def propose(self, anchor, *, position, max_tokens): return [0] * max(0, max_tokens - 1)
  def propose_sampled(self, anchor, *, position, max_tokens, temperature, random):
    count = max(0, max_tokens - 1)
    tokens = tuple(categorical([1.0], random) for _ in range(count))
    q = SparseDistribution((0,), (1.0,))
    return SampledProposal(tokens, (q,) * count)


class SequenceRandom:
  def __init__(self, values): self.values = list(values)
  def random(self): return self.values.pop(0) if self.values else 0.5


def assert_committed_state(test, target, prompt, output):
  state, history, _ = 0, (0, 0), 0
  for position, token in enumerate(prompt + output): state, history, _ = transition(state, history, token, position)
  test.assertEqual(target._cached_tokens, prompt + output)
  test.assertEqual(int(target.blk[0].recurrent_state.item()), state)
  test.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]), history)
  test.assertEqual(target.blk[0].cache_kv.numpy()[:len(prompt + output)].tolist(), prompt + output)


class TestSampledSpeculativeState(unittest.TestCase):
  def make(self, context=64, *, draft=None, verify_first_three=False):
    target = NondegenerateCausalTarget(context)
    draft = draft or PointMassDraft()
    return target, draft, GreedySpeculativeDecoder(target, draft, verify_first_three=verify_first_three)

  def test_rejection_replays_residual_token_not_unconditional_p_draw(self):
    target, draft, decoder = self.make()
    prompt = [2, 5, 3]
    # Anchor p draw -> 0; q point mass -> 0; acceptance draw -> reject p(0)/q(0)=1/2;
    # residual draw 0.1 selects token 1, while unconditional p draw 0.1 selects token 0.
    with patch.object(np.random, 'default_rng', side_effect=lambda seed: SequenceRandom([0.1, 0.2, 0.9, 0.1])):
      output = list(decoder.sample(prompt, temperature=1.0, seed=7, max_new_tokens=2))
    self.assertEqual(output, [0, 1])
    self.assert_committed_state(target, prompt, output)
    self.assertGreater(decoder.stats.rollback_count, 0)
    self.assertEqual(decoder.stats.output_tokens, 2)

  def test_first_block_extra_cost_fallback_keeps_sampled_residual_and_state(self):
    prompt = [2, 5, 3]
    for deferred in (False, True):
      with self.subTest(deferred=deferred):
        target = NondegenerateCausalTarget()
        decoder = GreedySpeculativeDecoder(target, PointMassDraft(), deferred_fallback=deferred)
        # Anchor 0, three q draws, reject q(0), residual 1, then p draws 1/0.
        random = SequenceRandom([0.1, 0.2, 0.2, 0.2, 0.9, 0.1, 0.9, 0.1])
        with patch.object(np.random, 'default_rng', return_value=random):
          output = list(decoder.sample(prompt, temperature=1.0, seed=7, max_new_tokens=4,
            ordinary_decode_seconds=1e-12, max_speculative_extra_seconds=1e-12))
        self.assertEqual(output, [0, 1, 1, 0])
        self.assertEqual(decoder.stats.blocks, 1)
        self.assertEqual(decoder.stats.fallback_reason, 'measured_extra_cost')
        self.assertEqual(decoder.stats.sampling_acceptance_tests, 1)
        self.assertEqual(decoder.stats.sampling_residual_draws, 1)
        self.assertEqual(decoder.stats.fallback_tokens, 3)
        self.assertEqual(decoder.stats.fallback_pending_last_token, deferred)
        assert_committed_state(self, target, prompt, output[:-1] if deferred else output)

  def assert_committed_state(self, target, prompt, output):
    assert_committed_state(self, target, prompt, output)

  def test_seed_reproducibility_eos_context_and_state(self):
    prompt = [2, 5, 3]
    results = []
    for _ in range(2):
      target, _, decoder = self.make()
      output = list(decoder.sample(prompt, temperature=1.0, seed=314, max_new_tokens=8))
      self.assert_committed_state(target, prompt, output)
      results.append(output)
    self.assertEqual(results[0], results[1])

    target, _, decoder = self.make()
    with patch.object(np.random, 'default_rng', side_effect=lambda seed: SequenceRandom([0.1, 0.2, 0.9, 0.1])):
      output = list(decoder.sample(prompt, temperature=1.0, seed=7, max_new_tokens=8, eos_token_ids=[1]))
    self.assertEqual(output, [0, 1])
    self.assertEqual(decoder.finish_reason, 'eos')
    self.assert_committed_state(target, prompt, output)

    target, _, decoder = self.make(context=5)
    output = list(decoder.sample(prompt, temperature=1.0, seed=314, max_new_tokens=8))
    self.assertEqual(len(output), 2)
    self.assertEqual(decoder.finish_reason, 'context')
    self.assert_committed_state(target, prompt, output)

  def test_close_and_cancel_keep_only_delivered_prefix_and_allow_reuse(self):
    prompt = [2, 5, 3]
    target, _, decoder = self.make()
    stream = decoder.sample(prompt, temperature=1.0, seed=81, max_new_tokens=8)
    delivered = [next(stream)]
    stream.close()
    self.assertEqual(decoder.finish_reason, 'closed')
    self.assert_committed_state(target, prompt, delivered)
    next_prompt = [7, 8]
    again = list(decoder.sample(next_prompt, temperature=1.0, seed=19, max_new_tokens=4))
    self.assert_committed_state(target, next_prompt, again)

    target, _, decoder = self.make()
    stop = {'value': False}
    stream = decoder.sample(prompt, temperature=1.0, seed=81, max_new_tokens=8,
                            cancelled=lambda: stop['value'])
    delivered = [next(stream)]
    stop['value'] = True
    self.assertEqual(list(stream), [])
    self.assertEqual(decoder.finish_reason, 'cancelled')
    self.assert_committed_state(target, prompt, delivered)
    again = list(decoder.sample(next_prompt, temperature=1.0, seed=19, max_new_tokens=4))
    self.assert_committed_state(target, next_prompt, again)

  def test_cancel_during_target_forward_restores_before_emission_without_second_prefill(self):
    target, _, decoder = self.make()
    prompt = [2, 5, 3]
    stop = {'value': False}
    calls = []
    original = target.target_forward
    def interrupting_forward(tokens, start_pos, **kwargs):
      ids = list(tokens.tolist()[0])
      calls.append((start_pos, ids))
      output = original(tokens, start_pos, **kwargs)
      if start_pos >= len(prompt) and len(ids) > 1: stop['value'] = True
      return output
    target.target_forward = interrupting_forward
    output = list(decoder.sample(prompt, temperature=1.0, seed=19, max_new_tokens=8,
                                 cancelled=lambda: stop['value']))
    self.assertEqual(sum(start == 0 for start, _ in calls), 1)
    self.assertEqual(calls[0], (0, prompt))
    self.assertEqual(decoder.finish_reason, 'cancelled')
    self.assert_committed_state(target, prompt, output)

  def test_unsupported_draft_and_verify_first_three_reject_before_target_write(self):
    prompt = [2, 5, 3]
    for draft, verify_first_three in ((OracleDraft(), False), (PointMassDraft(block_size=8), True)):
      with self.subTest(draft=type(draft).__name__, verify_first_three=verify_first_three):
        target, draft, decoder = self.make(draft=draft, verify_first_three=verify_first_three)
        calls = []
        original = target.target_forward
        def counted(*args, **kwargs):
          calls.append(args[1])
          return original(*args, **kwargs)
        target.target_forward = counted
        with self.assertRaises((ValueError, NotImplementedError)):
          list(decoder.sample(prompt, temperature=1.0, seed=3, max_new_tokens=4))
        self.assertEqual(calls, [])
        self.assertEqual(target._cached_tokens, [])
        self.assertEqual(int(target.blk[0].recurrent_state.item()), 0)
        self.assertEqual(target.blk[0].conv_state.tolist(), [[0, 0]])

  def test_hybrid_transformer_sample_matches_same_seed_and_commits(self):
    reference, target = real_pair(hybrid=True)
    prompt = [1, 2, 3]
    outputs = []
    for model in (reference, target):
      decoder = GreedySpeculativeDecoder(model, PointMassDraft())
      outputs.append(list(decoder.sample(prompt, temperature=0.8, seed=41, max_new_tokens=8)))
      self.assertEqual(decoder.stats.output_tokens, 8)
      self.assertTrue(decoder.stats.sampled)
    self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
  unittest.main()
