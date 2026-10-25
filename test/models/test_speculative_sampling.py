"""CPU contracts for exact sparse speculative sampling."""
from __future__ import annotations

import math
import unittest

import numpy as np
from tinygrad import Tensor
from tinygrad.llm.speculative_sampling import (
  SampledProposal, SamplingSelection, SparseDistribution, categorical, probabilities, residual, target_probabilities,
  temperature_value,
)


class SequenceRNG:
  def __init__(self, values): self.values, self.index = tuple(values), 0
  def __call__(self):
    value = self.values[self.index]
    self.index += 1
    return value


def logits_for(prob_rows):
  with np.errstate(divide="ignore"):
    return Tensor(np.log(np.asarray(prob_rows, dtype=np.float32)), device="CPU")


class SpeculativeSamplingTests(unittest.TestCase):
  def test_acceptance_rejection_mass_identity_with_sparse_proposal(self):
    p = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    proposal = SparseDistribution((0, 2), (0.8, 0.2))
    q = np.zeros_like(p)
    q[list(proposal.token_ids)] = proposal.probabilities
    accepted = q * np.where(q > 0, np.minimum(1.0, p / np.where(q > 0, q, 1.0)), 0.0)
    rejection_mass = 1.0 - accepted.sum()
    hand_residual = np.asarray([0.0, 0.5, 0.5], dtype=np.float64)
    self.assertAlmostEqual(rejection_mass, 0.6)
    np.testing.assert_allclose(p, accepted + rejection_mass * hand_residual, rtol=0, atol=1e-15)
    np.testing.assert_allclose(residual(p, proposal), hand_residual, rtol=0, atol=1e-15)

  def test_categorical_zero_tiny_mass_and_exact_cdf_boundaries_skip_zero_buckets(self):
    self.assertEqual(categorical([0.0, 1e-300, 1.0], lambda: 0.0), 1)
    self.assertEqual(categorical([0.2, 0.0, 0.8], lambda: 0.2), 2)
    self.assertEqual(categorical([0.0, 0.0, 1.0], lambda: 0.0), 2)

  def test_first_rejection_samples_hand_computed_residual_excluding_rejected_token(self):
    proposal = SparseDistribution((0, 2), (0.8, 0.2))
    selector = SamplingSelection(temperature=1.0, vocab_size=3, seed=7)
    selector.proposal = SampledProposal((0,), (proposal,))
    selector.random = SequenceRNG((0.9, 0.75))
    result = selector.verify((0,), logits_for([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]), set())
    self.assertEqual(result, (0, 2, True))
    self.assertNotEqual(result[1], 0)

  def test_second_rejection_uses_its_own_residual_and_accepts_first_token(self):
    proposal = SparseDistribution((0, 2), (0.8, 0.2))
    selector = SamplingSelection(temperature=1.0, vocab_size=3, seed=9)
    selector.proposal = SampledProposal((2, 0), (proposal, proposal))
    selector.random = SequenceRNG((0.1, 0.9, 0.75))
    result = selector.verify((2, 0), logits_for([[0.2, 0.3, 0.5]] * 3), set())
    self.assertEqual(result, (1, 2, True))
    self.assertNotEqual(result[1], 0)

  def test_all_accepted_emits_bonus_from_final_target_distribution(self):
    proposal = SparseDistribution((0, 2), (0.8, 0.2))
    selector = SamplingSelection(temperature=1.0, vocab_size=3, seed=10)
    selector.proposal = SampledProposal((2,), (proposal,))
    selector.random = SequenceRNG((0.1, 0.4))
    result = selector.verify((2,), logits_for([[0.2, 0.3, 0.5]] * 2), set())
    self.assertEqual(result, (1, 1, False))

  def test_accepted_eos_stops_without_marking_a_rejection(self):
    target = [0.1, 0.1, 0.3, 0.5]
    proposal = SparseDistribution((3,), (1.0,))
    selector = SamplingSelection(temperature=1.0, vocab_size=4, seed=11)
    selector.proposal = SampledProposal((3,), (proposal,))
    selector.random = SequenceRNG((0.1,))
    self.assertEqual(selector.verify((3,), logits_for([target, target]), {3}), (1, 0, False))

  def test_conditional_row_alignment_and_final_bonus_use_distinct_target_rows(self):
    rows=[[.8,.1,.1],[.1,.6,.3],[.1,.1,.8]]
    selector=SamplingSelection(1.,vocab_size=3,seed=14)
    selector.proposal=SampledProposal((1,0),(SparseDistribution((0,1),(.8,.2)),SparseDistribution((0,2),(.8,.2))))
    selector.random=SequenceRNG((.1,.5,.95))
    self.assertEqual(selector.verify((1,0),logits_for(rows),set()),(1,2,True))
    selector.proposal=SampledProposal((0,2),(SparseDistribution((0,1),(.8,.2)),SparseDistribution((0,2),(.8,.2))))
    selector.random=SequenceRNG((.1,.1,.5))
    self.assertEqual(selector.verify((0,2),logits_for(rows),set()),(2,2,False))

  def test_temperature_scaling_matches_target_and_draft_proposal_configuration(self):
    temperature = 0.5
    logits = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    selector = SamplingSelection(temperature=temperature, vocab_size=3, seed=12)
    target = target_probabilities(Tensor(logits, device="CPU"), temperature)
    expected = probabilities(logits, temperature_value(temperature))
    np.testing.assert_allclose(target, expected, rtol=1e-7, atol=1e-9)
    draft_scores = np.asarray([0.0, 1.0], dtype=np.float32)

    class Draft:
      def __init__(self): self.temperature = None
      def propose_sampled(self, anchor, *, position, max_tokens, temperature, random):
        self.temperature = temperature
        q = probabilities(draft_scores, temperature)
        token = categorical(q, random)
        return SampledProposal((token,), (SparseDistribution((0, 1), tuple(float(value) for value in q)),))

    draft = Draft()
    selector.propose(draft, 0, position=0, max_tokens=2)
    self.assertEqual(draft.temperature, selector.temperature)
    np.testing.assert_allclose(selector.proposal.distributions[0].probabilities,
                               probabilities(draft_scores, temperature_value(temperature)),
                               rtol=1e-7, atol=1e-9)

  def test_stable_extreme_logit_normalization(self):
    values = probabilities(np.asarray([10000.0, 10001.0, -10000.0], dtype=np.float64), temperature=1.0)
    self.assertTrue(np.isfinite(values).all())
    self.assertAlmostEqual(float(values.sum()), 1.0, places=15)
    self.assertGreater(values[1], values[0])
    self.assertEqual(values[2], 0.0)

  def test_invalid_distributions_scores_and_rng_draws_are_rejected(self):
    for token_ids, probs in (((0, 0), (0.5, 0.5)), ((-1,), (1.0,)), ((0, 1), (0.2, 0.2)),
                             ((0,), (math.nan,)), ((0,), (-0.1,))):
      with self.subTest(token_ids=token_ids, probs=probs), self.assertRaises(ValueError):
        SparseDistribution(token_ids, probs)
    with self.assertRaises(ValueError): SampledProposal((0,), (SparseDistribution((0, 1), (0.0, 1.0)),))
    for scores, temperature in (([], 1.0), ([math.nan, 0.0], 1.0), ([0.0, math.inf], 1.0), ([0.0], 0.0)):
      with self.subTest(scores=scores, temperature=temperature), self.assertRaises(ValueError):
        probabilities(np.asarray(scores, dtype=np.float64), temperature)
    for weights, draw in (([0.0, 0.0], 0.2), ([1.0, -0.1], 0.2), ([1.0, math.nan], 0.2),
                          ([1.0], -0.1), ([1.0], 1.0), ([1.0], math.inf)):
      with self.subTest(weights=weights, draw=draw), self.assertRaises(ValueError):
        categorical(weights, lambda draw=draw: draw)
    for target in (np.asarray([0.0, 0.0]), np.asarray([0.2, 0.3, 0.5, math.nan]), np.asarray([0.2, -0.1, 0.9])):
      with self.subTest(target=target), self.assertRaises(ValueError):
        residual(target, SparseDistribution((0,), (1.0,)))

  def test_invalid_selection_temperature_vocab_seed_and_bad_rows(self):
    for temperature in (0, -1.0, math.nan, math.inf, 1e100):
      with self.subTest(temperature=temperature), self.assertRaises(ValueError):
        SamplingSelection(temperature, vocab_size=3)
    for vocab in (0, -1, True, 3.0):
      with self.subTest(vocab=vocab), self.assertRaises(ValueError):
        SamplingSelection(1.0, vocab_size=vocab)
    with self.assertRaises(ValueError): SamplingSelection(1.0, vocab_size=3, seed=-1)
    selection = SamplingSelection(1.0, vocab_size=3)
    with self.assertRaises(ValueError): selection.verify((), logits_for([[0.0, 1.0]]), set())
    with self.assertRaises(ValueError): selection.verify((0,), logits_for([[0.0, 1.0, 2.0]]), set())
    class OutOfRangeDraft:
      def propose_sampled(self, anchor, *, position, max_tokens, temperature, random):
        return SampledProposal((2,), (SparseDistribution((2,), (1.0,)),))
    small_vocab = SamplingSelection(1.0, vocab_size=2)
    with self.assertRaises(ValueError): small_vocab.propose(OutOfRangeDraft(), 0, position=0, max_tokens=2)

  def test_stale_proposal_row_count_and_vocab_errors(self):
    dist = SparseDistribution((0, 2), (0.8, 0.2))
    selector = SamplingSelection(1.0, vocab_size=3, seed=13)
    selector.proposal = SampledProposal((0,), (dist,))
    with self.assertRaises(ValueError): selector.verify((2,), logits_for([[0.2, 0.3, 0.5]] * 2), set())
    with self.assertRaises(ValueError): selector.verify((0,), logits_for([[0.2, 0.3, 0.5]]), set())
    selector.vocab_size = 4
    with self.assertRaises(ValueError): selector.verify((0,), logits_for([[0.2, 0.3, 0.5]] * 2), set())


if __name__ == "__main__":
  unittest.main()
