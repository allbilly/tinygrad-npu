"""CPU contracts for reusing owned fixed-capacity DFlash2 proposal KV inputs."""
import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes
from test.models.test_dflash2_prepared import observe, setup


def observe_pair(control, reusable, values, position):
  observe(control, values, position)
  observe(reusable, values, position)


def buffer_bytes(value):
  return value.numpy().view(np.uint8).copy()


def prefix_bytes(draft):
  return [(cache.start_pos, cache.end_pos, buffer_bytes(cache.key), buffer_bytes(cache.value))
          for cache in draft._prefix_kv]


def assert_prefix_bytes(test, left, right):
  test.assertEqual(len(left), len(right))
  for a, b in zip(left, right):
    test.assertEqual(a[:2], b[:2])
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])


class TestDFlash2ReusableProposalInputs(unittest.TestCase):
  @staticmethod
  def pair(seed):
    control, reusable, rng = setup(seed)
    control.prepare_proposals(max_context=32, reuse_prefix_inputs=False)
    reusable.prepare_proposals(max_context=32, reuse_prefix_inputs=True)
    return control, reusable, rng

  @staticmethod
  def noise():
    return Tensor([[3, 15, 15, 15]], device='CPU', dtype=dtypes.int32).realize()

  def assert_inputs_equal(self, control, reusable, position):
    noise = self.noise()
    expected_inputs = control._proposal_forward._inputs(noise, control._prefix_kv, position)
    actual_inputs = reusable._proposal_forward._inputs(noise, reusable._prefix_kv, position)
    self.assertEqual(len(expected_inputs), len(actual_inputs))
    for expected, actual in zip(expected_inputs, actual_inputs):
      np.testing.assert_array_equal(expected.numpy().view(np.uint8), actual.numpy().view(np.uint8))
      self.assertIsNot(expected.uop.base.buffer, actual.uop.base.buffer)
    for layer, capacity in enumerate(reusable._proposal_forward.prefix_capacities):
      valid = min(position, reusable.config.sliding_window) if reusable.config.layer_types[layer] == 'sliding_attention' else position
      for buffer in actual_inputs[2 * layer:2 * layer + 2]:
        np.testing.assert_array_equal(buffer.numpy()[:, :, :valid].view(np.uint8),
                                      expected_inputs[2 * layer].numpy()[:, :, :valid].view(np.uint8)
                                      if buffer is actual_inputs[2 * layer] else
                                      expected_inputs[2 * layer + 1].numpy()[:, :, :valid].view(np.uint8))
        if valid < capacity:
          tail = buffer.numpy()[:, :, valid:]
          np.testing.assert_array_equal(tail.view(np.uint32), np.zeros_like(tail).view(np.uint32))
    return expected_inputs, actual_inputs

  def assert_proposal_equal(self, control, reusable, position):
    left = control._proposal_forward(self.noise(), control._prefix_kv, position)
    right = reusable._proposal_forward(self.noise(), reusable._prefix_kv, position)
    np.testing.assert_array_equal(left.logits.numpy().view(np.uint32), right.logits.numpy().view(np.uint32))
    np.testing.assert_array_equal(left.hidden.numpy().view(np.uint32), right.hidden.numpy().view(np.uint32))
    self.assertEqual(control.propose(3, position=position, max_tokens=4),
                     reusable.propose(3, position=position, max_tokens=4))

  def test_initial_full_prefix_copy_append_bytes_and_sliding_rollover(self):
    control, reusable, rng = self.pair(921)
    first = rng.normal(0, .3, (1, 7, 16)).astype(np.float32)
    second = rng.normal(0, .3, (1, 3, 16)).astype(np.float32)
    observe_pair(control, reusable, first, 0)
    observe_pair(control, reusable, second, 7)
    source_before = prefix_bytes(reusable)
    graph = reusable._proposal_forward
    input_bytes_before = graph.input_copy_bytes
    zero_bytes_before = graph.input_zero_bytes
    reused_bytes_before = graph.input_reused_bytes
    expected_inputs, actual_inputs = self.assert_inputs_equal(control, reusable, 10)
    input_bytes_per_row = 2 * reusable.config.num_key_value_heads * reusable.config.head_dim * reusable.dtype.itemsize
    # Initial update copies a two-row sliding window and ten full-prefix rows.
    self.assertEqual(graph.input_copy_bytes - input_bytes_before, (2 + 10) * input_bytes_per_row)
    self.assertEqual(graph.input_zero_bytes - zero_bytes_before, graph.input_bytes)
    self.assertEqual(graph.input_reused_bytes - reused_bytes_before, 0)
    self.assertEqual(graph.input_valid_rows, (2, 10))
    self.assertEqual(reusable._proposal_forward.prefix_inputs_source, reusable._prefix_kv)
    assert_prefix_bytes(self, source_before, prefix_bytes(reusable))
    self.assert_proposal_equal(control, reusable, 10)

    third = rng.normal(0, .3, (1, 2, 16)).astype(np.float32)
    observe_pair(control, reusable, third, 10)
    source_before = prefix_bytes(reusable)
    input_bytes_before = graph.input_copy_bytes
    reused_bytes_before = graph.input_reused_bytes
    self.assert_inputs_equal(control, reusable, 12)
    # Sliding rows are refreshed; only the two new full-attention rows are copied.
    self.assertEqual(graph.input_copy_bytes - input_bytes_before, (2 + 2) * input_bytes_per_row)
    self.assertEqual(graph.input_reused_bytes - reused_bytes_before, 10 * input_bytes_per_row)
    self.assertEqual(graph.input_valid_rows, (2, 12))
    self.assertEqual(reusable._prefix_kv[0].start_pos, 10)
    self.assertEqual(reusable._prefix_kv[1].start_pos, 0)
    assert_prefix_bytes(self, source_before, prefix_bytes(reusable))
    self.assert_proposal_equal(control, reusable, 12)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_equal_length_suffix_replacement_and_reset_refresh_reusable_inputs(self):
    control, reusable, rng = self.pair(922)
    prefix = rng.normal(0, .3, (1, 10, 16)).astype(np.float32)
    observe_pair(control, reusable, prefix, 0)
    self.assert_inputs_equal(control, reusable, 10)
    self.assert_proposal_equal(control, reusable, 10)

    replacement = rng.normal(0, .3, (1, 4, 16)).astype(np.float32)
    observe_pair(control, reusable, replacement, 6)
    self.assertEqual(control.cached_prefix_length, 10)
    self.assertEqual(reusable.cached_prefix_length, 10)
    self.assert_inputs_equal(control, reusable, 10)
    self.assert_proposal_equal(control, reusable, 10)
    self.assertEqual(reusable._proposal_forward.input_valid_rows, (2, 10))

    control.reset()
    reusable.reset()
    self.assertEqual(reusable._proposal_forward.preserved_prefix_rows, 0)
    self.assertIsNone(reusable._proposal_forward.prefix_inputs_source)
    reseed = rng.normal(0, .3, (1, 5, 16)).astype(np.float32)
    observe_pair(control, reusable, reseed, 0)
    self.assert_inputs_equal(control, reusable, 5)
    self.assertEqual(reusable._proposal_forward.input_valid_rows, (2, 5))
    self.assert_proposal_equal(control, reusable, 5)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_reusable_copy_failure_invalidates_partial_input_and_retry_clears_nan_tail(self):
    control, reusable, rng = self.pair(923)
    initial = rng.normal(0, .3, (1, 10, 16)).astype(np.float32)
    observe_pair(control, reusable, initial, 0)
    self.assert_inputs_equal(control, reusable, 10)
    self.assert_proposal_equal(control, reusable, 10)
    append = rng.normal(0, .3, (1, 2, 16)).astype(np.float32)
    observe_pair(control, reusable, append, 10)
    graph = reusable._proposal_forward
    source_before = prefix_bytes(reusable)
    for buffer in graph.prefix_inputs[2:4]:
      buffer[:, :, 12:].assign(float('nan')).realize()
    self.assertFalse(np.isfinite(graph.prefix_inputs[2].numpy()[:, :, 12:]).all())
    original_assign = Tensor.assign
    calls = []

    def fail_after_first_copy(tensor, values, *args, **kwargs):
      result = original_assign(tensor, values, *args, **kwargs)
      calls.append(1)
      if len(calls) == 1:
        result.realize()
        raise RuntimeError('injected reusable prefix staging-copy failure')
      return result

    with patch.object(Tensor, 'assign', fail_after_first_copy):
      with self.assertRaisesRegex(RuntimeError, 'staging-copy failure'):
        graph._inputs(self.noise(), reusable._prefix_kv, 12)
    self.assertEqual(graph.preserved_prefix_rows, 0)
    self.assertIsNone(graph.prefix_inputs_source)
    self.assertEqual(graph.input_valid_rows, graph.prefix_capacities)
    assert_prefix_bytes(self, source_before, prefix_bytes(reusable))

    expected_inputs = control._proposal_forward._inputs(self.noise(), control._prefix_kv, 12)
    actual_inputs = graph._inputs(self.noise(), reusable._prefix_kv, 12)
    for expected, actual in zip(expected_inputs, actual_inputs):
      np.testing.assert_array_equal(expected.numpy().view(np.uint8), actual.numpy().view(np.uint8))
    self.assertTrue(np.isfinite(graph.prefix_inputs[2].numpy()).all())
    self.assertTrue(np.isfinite(graph.prefix_inputs[3].numpy()).all())
    for layer, capacity in enumerate(graph.prefix_capacities):
      valid = min(12, reusable.config.sliding_window) if reusable.config.layer_types[layer] == 'sliding_attention' else 12
      for buffer in graph.prefix_inputs[2 * layer:2 * layer + 2]:
        tail = buffer.numpy()[:, :, valid:capacity]
        if tail.size:
          np.testing.assert_array_equal(tail.view(np.uint32), np.zeros_like(tail).view(np.uint32))
    assert_prefix_bytes(self, source_before, prefix_bytes(reusable))
    self.assert_proposal_equal(control, reusable, 12)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_empty_and_reset_prefixes_produce_fully_zeroed_owned_inputs(self):
    control, reusable, rng = self.pair(925)
    control_inputs = control._proposal_forward._inputs(self.noise(), control._prefix_kv, 0)
    reusable_inputs = reusable._proposal_forward._inputs(self.noise(), reusable._prefix_kv, 0)
    self.assertEqual(reusable._proposal_forward.input_zero_bytes, reusable._proposal_forward.input_bytes)
    self.assertEqual(reusable._proposal_forward.input_copy_bytes, 0)
    for left, right in zip(control_inputs, reusable_inputs):
      np.testing.assert_array_equal(left.numpy().view(np.uint32), np.zeros_like(left.numpy()).view(np.uint32))
      np.testing.assert_array_equal(right.numpy().view(np.uint32), np.zeros_like(right.numpy()).view(np.uint32))
      self.assertIsNot(left.uop.base.buffer, right.uop.base.buffer)

    observe_pair(control, reusable, rng.normal(0, .3, (1, 9, 16)).astype(np.float32), 0)
    self.assert_inputs_equal(control, reusable, 9)
    control.reset()
    reusable.reset()
    control_inputs = control._proposal_forward._inputs(self.noise(), control._prefix_kv, 0)
    reusable_inputs = reusable._proposal_forward._inputs(self.noise(), reusable._prefix_kv, 0)
    for left, right in zip(control_inputs, reusable_inputs):
      np.testing.assert_array_equal(left.numpy().view(np.uint32), np.zeros_like(left.numpy()).view(np.uint32))
      np.testing.assert_array_equal(right.numpy().view(np.uint32), np.zeros_like(right.numpy()).view(np.uint32))
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_equal_position_replacement_tuple_forces_a_full_prefix_refresh(self):
    control, reusable, rng = self.pair(926)
    observe_pair(control, reusable, rng.normal(0, .3, (1, 11, 16)).astype(np.float32), 0)
    self.assert_inputs_equal(control, reusable, 11)
    graph = reusable._proposal_forward
    input_bytes_before = graph.input_copy_bytes
    reused_bytes_before = graph.input_reused_bytes
    replacement_tuple = tuple(list(reusable._prefix_kv))
    self.assertIsNot(replacement_tuple, reusable._prefix_kv)
    self.assertEqual([(x.start_pos, x.end_pos) for x in replacement_tuple],
                     [(x.start_pos, x.end_pos) for x in reusable._prefix_kv])
    control_inputs = control._proposal_forward._inputs(self.noise(), control._prefix_kv, 11)
    reusable_inputs = graph._inputs(self.noise(), replacement_tuple, 11)
    row_bytes = 2 * reusable.config.num_key_value_heads * reusable.config.head_dim * reusable.dtype.itemsize
    self.assertEqual(graph.input_copy_bytes - input_bytes_before, (2 + 11) * row_bytes)
    self.assertEqual(graph.input_reused_bytes - reused_bytes_before, 0)
    for left, right in zip(control_inputs, reusable_inputs):
      np.testing.assert_array_equal(left.numpy().view(np.uint8), right.numpy().view(np.uint8))
    self.assertIsNone(graph.prefix_inputs_source)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_failed_observation_invalidates_reuse_and_successful_retry_copies_full_prefix(self):
    control, reusable, rng = self.pair(927)
    initial = rng.normal(0, .3, (1, 8, 16)).astype(np.float32)
    observe_pair(control, reusable, initial, 0)
    self.assert_inputs_equal(control, reusable, 8)
    graph = reusable._proposal_forward
    old_kv = prefix_bytes(reusable)
    copy_bytes_before = graph.input_copy_bytes
    reused_bytes_before = graph.input_reused_bytes
    update = rng.normal(0, .3, (1, 2, 16)).astype(np.float32)
    revision_before = reusable.cache_revision
    with patch.object(reusable, 'project_features', side_effect=RuntimeError('injected observation projection failure')):
      with self.assertRaisesRegex(RuntimeError, 'observation projection failure'):
        observe(reusable, update, 8)
    self.assertGreater(reusable.cache_revision, revision_before)
    self.assertEqual(reusable.cached_prefix_length, 8)
    self.assertEqual(graph.preserved_prefix_rows, 0)
    self.assertIsNone(graph.prefix_inputs_source)
    assert_prefix_bytes(self, old_kv, prefix_bytes(reusable))

    observe(control, update, 8)
    observe(reusable, update, 8)
    expected_inputs = control._proposal_forward._inputs(self.noise(), control._prefix_kv, 10)
    actual_inputs = graph._inputs(self.noise(), reusable._prefix_kv, 10)
    row_bytes = 2 * reusable.config.num_key_value_heads * reusable.config.head_dim * reusable.dtype.itemsize
    self.assertEqual(graph.input_copy_bytes - copy_bytes_before, (2 + 10) * row_bytes)
    self.assertEqual(graph.input_reused_bytes - reused_bytes_before, 0)
    for left, right in zip(expected_inputs, actual_inputs):
      np.testing.assert_array_equal(left.numpy().view(np.uint8), right.numpy().view(np.uint8))
    assert_prefix_bytes(self, prefix_bytes(control), prefix_bytes(reusable))
    self.assert_proposal_equal(control, reusable, 10)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()

  def test_default_metadata_and_invalid_option_preserve_existing_owner_then_close_releases_buffers(self):
    control, reusable, rng = self.pair(924)
    control_metadata = control.proposal_preparation_metadata()
    self.assertNotIn('reuse_prefix_inputs', control_metadata)
    metadata = reusable.proposal_preparation_metadata()
    self.assertIs(metadata.get('reuse_prefix_inputs'), True)
    self.assertEqual(metadata['input_copy_bytes'], 0)
    graph = reusable._proposal_forward
    with self.assertRaises(ValueError):
      reusable.prepare_proposals(max_context=32, reuse_prefix_inputs=1)
    self.assertIs(reusable._proposal_forward, graph)
    self.assertFalse(graph.closed)

    observe_pair(control, reusable, rng.normal(0, .3, (1, 5, 16)).astype(np.float32), 0)
    self.assert_inputs_equal(control, reusable, 5)
    owned_inputs = graph.prefix_inputs
    self.assertIsNotNone(owned_inputs)
    self.assertEqual(len(owned_inputs), 2 * reusable.config.num_layers)
    control.close_prepared_proposals()
    reusable.close_prepared_proposals()
    self.assertIsNone(reusable._proposal_forward)
    self.assertTrue(graph.closed)
    self.assertIsNone(graph.prefix_inputs)
    self.assertIsNone(graph.prefix_inputs_source)
    self.assertIsNone(graph.draft)
