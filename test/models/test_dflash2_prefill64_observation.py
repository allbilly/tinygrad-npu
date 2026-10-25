"""Behavioral coverage for the prepared 64-row observation graph."""
import unittest

import numpy as np

from test.models.test_dflash2_observe import fields, numpy_observation, observation, setup


class TestPreparedObservation64(unittest.TestCase):
  def test_64_row_calls_match_numpy_and_return_owned_outputs(self):
    _raw, model, c, w, rng = setup(641)
    model.prepare_observation(max_context=256)
    manager = model._prepared_observation
    values = rng.normal(0, .3, (1, 64, 16)).astype(np.float32)
    actuals = []

    for position in (0, 64, 96):
      actual = manager(fields(model, values), position)
      expected = numpy_observation(values, position, c, w)
      for value, wanted in zip(actual, expected):
        np.testing.assert_allclose(value.numpy(), wanted, atol=2e-6, rtol=3e-5)
      actuals.append((actual, tuple(value.numpy().copy() for value in actual)))

    for actual, saved in actuals:
      for value, wanted in zip(actual, saved):
        np.testing.assert_array_equal(value.numpy(), wanted)
    self.assertIn(64, manager.supported_rows)
    self.assertEqual(manager.metadata()["graph_calls"][64], 3)
    self.assertIs(type(next(iter(manager.calls))), int)
    model.close_prepared_observation()

  def test_64_row_observe_matches_independent_history_for_append_and_sliding_prefix(self):
    _raw, model, c, w, rng = setup(642)
    model.prepare_observation(max_context=256)
    first = rng.normal(0, .3, (1, 64, 16)).astype(np.float32)
    second = rng.normal(0, .3, (1, 64, 16)).astype(np.float32)
    history = np.concatenate((first, second), axis=1)

    model.observe(observation(model, first), 0)
    model.observe(observation(model, second), 64)

    expected = numpy_observation(history, 0, c, w)
    np.testing.assert_allclose(model._target_hidden.numpy(), expected[0], atol=2e-6, rtol=3e-5)
    for layer, cache in enumerate(model._prefix_kv):
      begin = max(0, history.shape[1] - c.sliding_window) if c.layer_types[layer] == "sliding_attention" else 0
      self.assertEqual((cache.start_pos, cache.end_pos), (begin, history.shape[1]))
      np.testing.assert_allclose(cache.key.numpy(), expected[1 + 2 * layer][:, :, begin:], atol=2e-6, rtol=3e-5)
      np.testing.assert_allclose(cache.value.numpy(), expected[2 + 2 * layer][:, :, begin:], atol=2e-6, rtol=3e-5)
    self.assertEqual(model._prepared_observation.metadata()["graph_calls"][64], 2)
    model.close_prepared_observation()

  def test_64_row_append_without_projected_history_matches_independent_prefix(self):
    _raw, model, c, w, rng = setup(643)
    model.prepare_proposals(max_context=256)
    model.prepare_observation(max_context=256, pad_decode_rows=True, retain_target_features=False)
    history = []
    saved_first = None
    try:
      for position in (0, 64):
        values = rng.normal(0, .3, (1, 64, 16)).astype(np.float32)
        history.append(values)
        model.observe(observation(model, values), position)
        self.assertIsNone(model._target_hidden)
        expected = numpy_observation(np.concatenate(history, axis=1), 0, c, w)
        end = position + 64
        self.assertEqual(model.cached_prefix_length, end)
        for layer, cache in enumerate(model._prefix_kv):
          begin = max(0, end - c.sliding_window) if c.layer_types[layer] == "sliding_attention" else 0
          self.assertEqual((cache.start_pos, cache.end_pos), (begin, end))
          np.testing.assert_allclose(cache.key.numpy(), expected[1 + 2 * layer][:, :, begin:], atol=2e-6, rtol=3e-5)
          np.testing.assert_allclose(cache.value.numpy(), expected[2 + 2 * layer][:, :, begin:], atol=2e-6, rtol=3e-5)
        if saved_first is None:
          saved_first = [(cache.key, cache.value, cache.key.numpy().copy(), cache.value.numpy().copy())
                         for cache in model._prefix_kv]
      for key, value, expected_key, expected_value in saved_first:
        np.testing.assert_array_equal(key.numpy(), expected_key)
        np.testing.assert_array_equal(value.numpy(), expected_value)
      metadata = model.observation_preparation_metadata()
      self.assertEqual(metadata["retained_projected_history_bytes"], 0)
      self.assertEqual(metadata["graph_calls"][64], 2)
    finally:
      model.close_prepared_observation()
      model.close_prepared_proposals()


if __name__ == '__main__':
  unittest.main()
