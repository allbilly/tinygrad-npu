"""CPU regressions for checkpoint-configured contexts beyond 512 tokens."""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from test.models.test_dflash2_prepared import observe, setup


MODEL_CONTEXT=8192
INITIAL_ROWS=600
WINDOW=2
F32_ATOL,F32_RTOL=2e-6,3e-5
NOISE_IDS=(3,15,15,15)


def compare_prepared_to_ordinary(test, raw, draft, position):
  noise=Tensor([NOISE_IDS],device='CPU',dtype=dtypes.int32).realize()
  actual=draft._proposal_forward(noise,draft._prefix_kv,position)
  prefix=raw._target_hidden
  if prefix is None:prefix=Tensor.zeros(1,0,8,device='CPU',dtype=raw.dtype)
  expected=raw._forward_projected(noise,prefix,position=position,prefix_kv=raw._prefix_kv).realize()
  np.testing.assert_allclose(actual.logits.numpy(),expected.logits.numpy(),atol=F32_ATOL,rtol=F32_RTOL)
  np.testing.assert_allclose(actual.hidden.numpy(),expected.hidden.numpy(),atol=F32_ATOL,rtol=F32_RTOL)
  np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected.logits.numpy().argmax(-1))
  np.testing.assert_allclose(draft._target_hidden.numpy(),raw._target_hidden.numpy(),atol=F32_ATOL,rtol=F32_RTOL)


class TestPreparedDFlash2ModelContext(unittest.TestCase):
  def test_long_prefix_then_prepared_small_appends_and_logits_match_ordinary_path(self):
    raw,draft,rng=setup(9501)
    raw.config.max_position_embeddings=draft.config.max_position_embeddings=MODEL_CONTEXT
    self.assertEqual(raw.config.max_position_embeddings,MODEL_CONTEXT)
    raw.config.sliding_window=draft.config.sliding_window=WINDOW
    metadata=draft.prepare_proposals(max_context=MODEL_CONTEXT)
    observation_metadata=draft.prepare_observation(max_context=MODEL_CONTEXT,pad_decode_rows=True,native_cache_copy=False)
    graph=draft._proposal_forward
    self.assertEqual(metadata['max_context'],MODEL_CONTEXT)
    self.assertEqual(observation_metadata['max_context'],MODEL_CONTEXT)
    self.assertTrue(observation_metadata['pad_decode_rows'])
    self.assertEqual(observation_metadata['decode_capacity'],8)
    self.assertFalse(observation_metadata['native_cache_copy'])
    self.assertEqual(graph.prefix_capacities,(WINDOW,MODEL_CONTEXT))
    self.assertLessEqual(graph.prefix_capacities[0],draft.config.sliding_window)
    self.assertEqual(graph.prefix_capacities[1],MODEL_CONTEXT)

    initial=rng.normal(0,.3,(1,INITIAL_ROWS,16)).astype(np.float32)
    # 600 rows exceed the observation manager's prepared row buckets and take its generic path.
    self.assertGreater(INITIAL_ROWS,512)
    for model in (raw,draft):observe(model,initial,0)
    self.assertEqual(draft._prepared_observation.calls,{})
    compare_prepared_to_ordinary(self,raw,draft,INITIAL_ROWS)

    # Prepared observation shares one 8-row graph while absolute positions rebind.
    for position,rows in ((INITIAL_ROWS,1),(INITIAL_ROWS+1,3),(INITIAL_ROWS+4,8)):
      values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
      for model in (raw,draft):observe(model,values,position)
      end=position+rows
      compare_prepared_to_ordinary(self,raw,draft,end)
      sliding,full=draft._prefix_kv
      self.assertEqual((sliding.start_pos,sliding.end_pos),(max(0,end-WINDOW),end))
      self.assertEqual((full.start_pos,full.end_pos),(0,end))
      self.assertEqual(sliding.key.shape[2],min(end,WINDOW))
      self.assertEqual(full.key.shape[2],end)
    self.assertEqual(draft._prepared_observation.decode_capacity,8)
    self.assertGreaterEqual(graph.calls,3)
    self.assertGreaterEqual(graph.jit.cnt,3)
    self.assertIsNotNone(graph.jit.captured)
    self.assertGreaterEqual(draft._prepared_observation.calls[8].calls,3)
    draft.close_prepared_observation()
    draft.close_prepared_proposals()

  def test_context_limits_reject_before_closing_or_mutating_existing_prepared_owners(self):
    _raw,draft,_rng=setup(9502)
    draft.config.max_position_embeddings=MODEL_CONTEXT
    draft.prepare_proposals(max_context=MODEL_CONTEXT)
    draft.prepare_observation(max_context=MODEL_CONTEXT,pad_decode_rows=True,native_cache_copy=False)
    graph,manager=draft._proposal_forward,draft._prepared_observation
    before_hidden=draft._target_hidden
    for invalid in (MODEL_CONTEXT+1,0,True):
      with self.subTest(invalid=invalid):
        with self.assertRaises(ValueError):draft.prepare_proposals(max_context=invalid)
        self.assertIs(draft._proposal_forward,graph)
        self.assertFalse(graph.closed)
        with self.assertRaises(ValueError):
          draft.prepare_observation(max_context=invalid,pad_decode_rows=True,native_cache_copy=False)
        self.assertIs(draft._prepared_observation,manager)
        self.assertFalse(manager.closed)
        self.assertIs(draft._target_hidden,before_hidden)
    draft.close_prepared_observation()
    draft.close_prepared_proposals()


if __name__=='__main__':unittest.main()
