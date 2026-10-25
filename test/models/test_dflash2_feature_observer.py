"""CPU contracts for DFlash2's feature-only observation entry point."""
import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from test.models.test_dflash2_feature_retention import observed_output, bits
from test.models.test_dflash2_prepared import setup


def feature_fields(draft, values):
  return tuple(Tensor(value, device=draft.device, dtype=draft.dtype).realize()
               for value in np.split(values, 2, axis=-1))


def prepare_pair(seed, *, retain):
  direct, wrapped, _rng = setup(seed)
  for draft in (direct, wrapped):draft.prepare_proposals(max_context=64)
  direct.prepare_observation(max_context=64, pad_decode_rows=True, retain_target_features=retain)
  wrapped.prepare_observation(max_context=64, pad_decode_rows=True, retain_target_features=retain)
  return direct, wrapped


def assert_same_prefix(test, direct, wrapped, rows):
  test.assertEqual(direct.cached_prefix_length, rows)
  test.assertEqual(wrapped.cached_prefix_length, rows)
  if direct.observation_preparation_metadata().get('retain_target_features',True):
    test.assertIsNotNone(direct._target_hidden)
    test.assertIsNotNone(wrapped._target_hidden)
    np.testing.assert_array_equal(bits(direct._target_hidden), bits(wrapped._target_hidden))
  else:
    test.assertIsNone(direct._target_hidden)
    test.assertIsNone(wrapped._target_hidden)
    for draft in (direct, wrapped):
      metadata=draft.observation_preparation_metadata()
      test.assertEqual(metadata['retained_projected_history_bytes'], 0)
      test.assertEqual(metadata['cached_prefix_rows'], rows)
  for left, right in zip(direct._prefix_kv, wrapped._prefix_kv):
    test.assertEqual((left.start_pos, left.end_pos), (right.start_pos, right.end_pos))
    np.testing.assert_array_equal(bits(left.key), bits(right.key))
    np.testing.assert_array_equal(bits(left.value), bits(right.value))
  left=direct._proposal_output(3, rows).realize()
  right=wrapped._proposal_output(3, rows).realize()
  np.testing.assert_array_equal(left.logits.numpy().view(np.uint32), right.logits.numpy().view(np.uint32))
  np.testing.assert_array_equal(left.hidden.numpy().view(np.uint32), right.hidden.numpy().view(np.uint32))
  test.assertEqual(direct.propose(3, position=rows, max_tokens=4),
                   wrapped.propose(3, position=rows, max_tokens=4))


class TestDFlash2FeatureObserver(unittest.TestCase):
  def test_feature_only_and_target_output_observation_match_for_one_to_eight_rows(self):
    for retain in (True, False):
      with self.subTest(retain_target_features=retain):
        direct, wrapped=prepare_pair(1201+int(retain), retain=retain)
        try:
          rng=np.random.default_rng(1203+int(retain))
          position=0
          for rows in range(1,9):
            values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
            direct_fields=feature_fields(direct, values)
            direct_revision,wrapped_revision=direct.cache_revision,wrapped.cache_revision
            # The direct API receives only captured feature tensors; no TargetOutput,
            # logits, or hidden-output tensors are constructed for it.
            with patch.object(Tensor, 'zeros', side_effect=AssertionError('feature-only observation allocated zeros')):
              direct.observe_features(direct_fields, position)
            wrapped.observe(observed_output(wrapped, values), position)
            self.assertEqual(direct.cache_revision, direct_revision+1)
            self.assertEqual(wrapped.cache_revision, wrapped_revision+1)
            position+=rows
            assert_same_prefix(self, direct, wrapped, position)

          # Invalid starts still invalidate external reuse revisions before failing.
          for draft in (direct, wrapped):
            revision=draft.cache_revision
            with self.assertRaisesRegex(ValueError, 'gap'):
              (draft.observe_features(feature_fields(draft, np.ones((1,1,16),np.float32)), position+1)
               if draft is direct else draft.observe(observed_output(draft, np.ones((1,1,16),np.float32)), position+1))
            self.assertGreater(draft.cache_revision, revision)
          assert_same_prefix(self, direct, wrapped, position)
        finally:
          for draft in (direct, wrapped):
            draft.close_prepared_observation()
            draft.close_prepared_proposals()


if __name__=='__main__':unittest.main()
