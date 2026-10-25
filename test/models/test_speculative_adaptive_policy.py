"""Pure CPU policy tests for DecodeFallbackPolicy; staged for the root test batch."""
from __future__ import annotations

import math
import unittest

from tinygrad.llm.speculative_adaptive import DecodeFallbackPolicy


class DecodeFallbackPolicyTests(unittest.TestCase):
  def test_high_acceptance_with_low_compute_cost_keeps_speculation(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=1.1)
    self.assertFalse(policy.fallback)
    self.assertFalse(policy.record(tokens=8, seconds=0.04))
    self.assertFalse(policy.record(tokens=12, seconds=0.08))
    self.assertEqual((policy.blocks, policy.tokens, policy.seconds), (2, 20, 0.12))

  def test_low_acceptance_and_high_cost_falls_back_after_minimum_blocks(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=1.1)
    self.assertFalse(policy.record(tokens=1, seconds=0.02))
    self.assertFalse(policy.fallback)
    self.assertTrue(policy.record(tokens=1, seconds=0.02))
    self.assertTrue(policy.fallback)
    self.assertEqual((policy.blocks, policy.tokens, policy.seconds), (2, 2, 0.04))

  def test_rate_uses_cumulative_weighted_tokens_not_mean_of_round_rates(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.009, min_blocks=2, max_slowdown=1.1)
    self.assertFalse(policy.record(tokens=1000, seconds=1.0))
    self.assertFalse(policy.record(tokens=1, seconds=0.02))
    weighted_rate = policy.seconds / policy.tokens
    mean_of_round_rates = (1.0 / 1000 + 0.02 / 1) / 2
    self.assertLess(weighted_rate, 0.009 * 1.1)
    self.assertGreater(mean_of_round_rates, 0.009 * 1.1)
    self.assertFalse(policy.fallback)

  def test_exact_slowdown_threshold_does_not_fall_back(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=1.0, min_blocks=2, max_slowdown=2.0)
    self.assertFalse(policy.record(tokens=1, seconds=1.0))
    self.assertFalse(policy.record(tokens=1, seconds=3.0))
    self.assertEqual(policy.seconds / policy.tokens, 2.0)
    self.assertFalse(policy.fallback)

  def test_fallback_decision_is_permanent_after_later_fast_rounds(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=1.1)
    policy.record(tokens=1, seconds=0.02)
    self.assertTrue(policy.record(tokens=1, seconds=0.02))
    self.assertTrue(policy.record(tokens=100, seconds=0.01))
    self.assertTrue(policy.fallback)
    self.assertEqual((policy.blocks, policy.tokens), (3, 102))

  def test_extra_compute_budget_uses_strict_one_reference_token_boundary(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=1.0, min_blocks=2, max_slowdown=100.0,
                                  max_extra_seconds=1.0)
    self.assertFalse(policy.record(tokens=1, seconds=2.0))
    self.assertEqual(policy.blocks, 1)
    self.assertFalse(policy.fallback)

    early = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=100.0,
                                 max_extra_seconds=0.01)
    self.assertTrue(early.record(tokens=1, seconds=0.020001))
    self.assertEqual(early.blocks, 1)
    self.assertEqual(early.reason, 'measured_extra_cost')

  def test_extra_compute_budget_uses_cumulative_token_credit_and_sticks(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=1.0, min_blocks=2, max_slowdown=100.0,
                                  max_extra_seconds=2.0)
    self.assertFalse(policy.record(tokens=10, seconds=0.1))
    self.assertFalse(policy.record(tokens=1, seconds=11.9))
    self.assertEqual(policy.seconds-policy.tokens, 1.0)
    self.assertTrue(policy.record(tokens=1, seconds=2.0001))
    self.assertEqual(policy.reason, 'measured_extra_cost')
    self.assertTrue(policy.record(tokens=100, seconds=0.01))
    self.assertTrue(policy.fallback)

  def test_default_extra_budget_keeps_the_existing_two_block_slowdown_guard(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=1.1)
    self.assertIsNone(policy.max_extra_seconds)
    self.assertFalse(policy.record(tokens=1, seconds=0.05))
    self.assertEqual(policy.blocks, 1)
    self.assertTrue(policy.record(tokens=1, seconds=0.05))
    self.assertEqual(policy.reason, 'measured_round_cost')

  def test_budget_allows_marginal_first_block_but_keeps_two_block_slowdown_guard(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=2, max_slowdown=1.1,
                                  max_extra_seconds=0.02)
    self.assertFalse(policy.record(tokens=1, seconds=0.02))
    self.assertEqual(policy.blocks, 1)
    self.assertFalse(policy.fallback)
    self.assertTrue(policy.record(tokens=1, seconds=0.003))
    self.assertEqual(policy.blocks, 2)
    self.assertEqual(policy.reason, 'measured_round_cost')

  def test_extra_compute_budget_constructor_validation(self):
    for extra in (0.0, -0.01, math.nan, math.inf, -math.inf, True, '0.01'):
      with self.subTest(max_extra_seconds=extra), self.assertRaises(ValueError):
        DecodeFallbackPolicy(reference_seconds_per_token=0.01, max_extra_seconds=extra)

  def test_invalid_constructor_values_are_rejected(self):
    for reference in (0.0, -0.01, math.nan, math.inf, -math.inf):
      with self.subTest(reference=reference), self.assertRaises(ValueError):
        DecodeFallbackPolicy(reference_seconds_per_token=reference)
    for min_blocks in (0, -1, 1.5):
      with self.subTest(min_blocks=min_blocks), self.assertRaises(ValueError):
        DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=min_blocks)
    for slowdown in (0.99, 0.0, math.nan, math.inf, -math.inf):
      with self.subTest(max_slowdown=slowdown), self.assertRaises(ValueError):
        DecodeFallbackPolicy(reference_seconds_per_token=0.01, max_slowdown=slowdown)

  def test_invalid_records_are_rejected_without_state_mutation(self):
    policy = DecodeFallbackPolicy(reference_seconds_per_token=0.01, min_blocks=3, max_slowdown=1.1,
                                  max_extra_seconds=0.05)
    policy.record(tokens=2, seconds=0.01)
    before = (policy.blocks, policy.tokens, policy.seconds, policy.fallback)
    invalid = (
      (0, 0.01), (-1, 0.01), (1.5, 0.01), (True, 0.01),
      (1, 0.0), (1, -0.01), (1, math.nan), (1, math.inf), (1, -math.inf),
    )
    for tokens, seconds in invalid:
      with self.subTest(tokens=tokens, seconds=seconds), self.assertRaises(ValueError):
        policy.record(tokens=tokens, seconds=seconds)
      self.assertEqual((policy.blocks, policy.tokens, policy.seconds, policy.fallback), before)


if __name__ == "__main__":
  unittest.main()
