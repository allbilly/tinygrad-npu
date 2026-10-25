import unittest
from dataclasses import replace
from tinygrad.llm.speculative_policy import MeasuredSpeculativePolicy, PolicyKey


class TestMeasuredSpeculativePolicy(unittest.TestCase):
  def setUp(self):
    self.key = PolicyKey('target-sha', 'draft-sha', 'CL-driver-kernels-v1', 'fp16-fp32-state', 1, (0, 511))

  def add(self, policy, budget, emitted, seconds, count=3, **kwargs):
    for _ in range(count):policy.record(self.key, budget, emitted, seconds, qualified=True, **kwargs)

  def choose(self, policy, key=None, context=128, remaining=64):
    return policy.choose(key or self.key, context_tokens=context, remaining_tokens=remaining)

  def test_unqualified_warmup_and_missing_baseline_cannot_enable_speculation(self):
    policy = MeasuredSpeculativePolicy((7,))
    for _ in range(3):policy.record(self.key, 7, 8, .1)
    self.assertEqual(self.choose(policy).budget, 0)
    self.add(policy, 0, 1, .1, warmup=True)
    self.add(policy, 7, 8, .1)
    self.assertEqual(self.choose(policy).reason, 'insufficient_ordinary_samples')
    self.add(policy, 0, 1, .1)
    self.assertEqual(self.choose(policy).budget, 7)
    self.assertAlmostEqual(self.choose(policy).speedup, 8)

  def test_selects_measured_throughput_including_rejected_replay_cost(self):
    policy = MeasuredSpeculativePolicy((3, 7))
    self.add(policy, 0, 1, .1)
    self.add(policy, 3, 3, .2)
    # The longer draft accepts more, but replay/restore/observation make its complete round slower.
    self.add(policy, 7, 4, .4)
    decision = self.choose(policy)
    self.assertEqual(decision.budget, 3)
    self.assertAlmostEqual(decision.speedup, 1.5)
    self.assertEqual(self.choose(policy, remaining=3).budget, 0)
    self.assertEqual(self.choose(policy, remaining=4).budget, 3)

  def test_configuration_context_and_batch_never_share_measurements(self):
    policy = MeasuredSpeculativePolicy((7,))
    self.add(policy, 0, 1, .1)
    self.add(policy, 7, 8, .1)
    for key in (replace(self.key, target_revision='other'), replace(self.key, draft_revision='other'),
                replace(self.key, runtime='different-kernel'), replace(self.key, precision='int8'),
                replace(self.key, batch_size=2), replace(self.key, context_range=(512, 1023))):
      self.assertEqual(self.choose(policy, key).budget, 0)
    self.assertEqual(self.choose(policy, context=512).reason, 'outside_measured_context')
    self.assertEqual(self.choose(policy, context=511).budget, 7)
    self.assertEqual(self.choose(policy, remaining=1).budget, 0)

  def test_window_and_profile_count_are_bounded_and_failure_invalidates(self):
    policy = MeasuredSpeculativePolicy((7,), min_samples=2, window=2, max_profiles=1)
    self.add(policy, 0, 1, .1)
    self.add(policy, 7, 8, .1)
    self.add(policy, 7, 1, .2, count=2)
    self.assertEqual(self.choose(policy).budget, 0)
    self.add(policy, 7, 8, .1, count=2)
    self.assertEqual(self.choose(policy).samples, 2)
    policy.invalidate(self.key, 7)
    self.assertEqual(self.choose(policy).budget, 0)
    other = replace(self.key, runtime='new-kernels')
    policy.record(other, 0, 1, .1, qualified=True)
    self.assertEqual(self.choose(policy).reason, 'no_matching_measurements')

  def test_aggregate_round_time_and_gain_threshold(self):
    policy = MeasuredSpeculativePolicy((1,), min_samples=2)
    self.add(policy, 0, 1, 1, count=2)
    policy.record(self.key, 1, 2, .1, qualified=True)
    policy.record(self.key, 1, 1, 4, qualified=True)
    # Averaging round rates would wrongly claim a gain; aggregate committed tokens/time has none.
    self.assertEqual(self.choose(policy).budget, 0)
    policy.invalidate(self.key, 1)
    self.add(policy, 1, 2, 2 / 1.04, count=2)
    self.assertEqual(self.choose(policy).budget, 0)

  def test_invalid_measurements_and_keys(self):
    policy = MeasuredSpeculativePolicy((7,))
    for budget, emitted, seconds in ((2, 1, 1), (True, 1, 1), (7, 9, 1), (0, 2, 1), (7, 1, 0), (7, 1, float('nan'))):
      with self.assertRaises(ValueError):policy.record(self.key, budget, emitted, seconds, qualified=True)
    for kwargs in ({'batch_size':0}, {'batch_size':True}, {'context_range':(10, 5)}, {'runtime':''}):
      with self.assertRaises(ValueError):replace(self.key, **kwargs)
    for budgets, kwargs in (((0,), {}), ((1, 1), {}), ((1,), {'min_samples':4, 'window':3}), ((1,), {'min_speedup':float('inf')})):
      with self.assertRaises(ValueError):MeasuredSpeculativePolicy(budgets, **kwargs)


if __name__ == '__main__':unittest.main()
