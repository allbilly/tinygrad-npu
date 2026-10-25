"""CPU regressions for bounded sliding-layer draft inputs and rollback."""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from test.models.test_dflash2_prepared import observe, setup


WINDOW=4
PREPARED_CONTEXT=16
NOISE_IDS=(3,15,15,15)
F32_ATOL,F32_RTOL=2e-6,3e-5
F16_ATOL,F16_RTOL=3e-3,4e-3


def prefix_capacities(context, window=WINDOW):
  return (min(context,window),context)


def expected_padding_bytes(capacities, *, half=False):
  # Fixture has one KV head, head_dim=4, and each layer owns K and V inputs.
  return 2*sum(capacities)*1*4*(2 if half else 4)


def ordinary_reference(raw, noise, position):
  prefix=raw._target_hidden
  if prefix is None:prefix=Tensor.zeros(1,0,8,device=raw.device,dtype=raw.dtype)
  return raw._forward_projected(noise,prefix,position=position,prefix_kv=raw._prefix_kv).realize()


def assert_forward_matches(test, prepared, raw, position, *, half=False):
  noise=Tensor([NOISE_IDS],device=prepared.device,dtype=dtypes.int32).realize()
  actual=prepared._proposal_forward(noise,prepared._prefix_kv,position)
  expected=ordinary_reference(raw,noise,position)
  atol,rtol=(F16_ATOL,F16_RTOL) if half else (F32_ATOL,F32_RTOL)
  np.testing.assert_allclose(actual.logits.numpy(),expected.logits.numpy(),atol=atol,rtol=rtol)
  np.testing.assert_allclose(actual.hidden.numpy(),expected.hidden.numpy(),atol=atol,rtol=rtol)
  np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected.logits.numpy().argmax(-1))
  return actual


def set_window(raw, prepared, value=WINDOW):
  raw.config.sliding_window=value
  prepared.config.sliding_window=value


class TestPreparedDFlash2SlidingCapacity(unittest.TestCase):
  def test_per_layer_capacity_and_exact_padding_bytes_for_f32_and_f16(self):
    for half in (False,True):
      for capacity in (2,WINDOW,8):
        with self.subTest(half=half,capacity=capacity):
          _raw,draft,_rng=setup(9401+capacity+int(half),half=half)
          set_window(_raw,draft)
          metadata=draft.prepare_proposals(max_context=capacity)
          graph=draft._proposal_forward
          expected_caps=prefix_capacities(capacity)
          self.assertEqual(graph.prefix_capacities,expected_caps)
          self.assertEqual(metadata['padding_input_bytes'],expected_padding_bytes(expected_caps,half=half))
          noise=Tensor([NOISE_IDS],device='CPU',dtype=dtypes.int32).realize()
          inputs=graph._inputs(noise,draft._prefix_kv,0)
          self.assertEqual(len(inputs),2*draft.config.num_layers)
          for layer,layer_capacity in enumerate(expected_caps):
            for state in inputs[2*layer:2*layer+2]:
              self.assertEqual(state.shape,(1,draft.config.num_key_value_heads,layer_capacity,draft.config.head_dim))
          draft.close_prepared_proposals()

  def test_capacity_crossing_window_append_replacement_deep_rollback_and_reset(self):
    raw,draft,rng=setup(9402)
    set_window(raw,draft)
    metadata=draft.prepare_proposals(max_context=PREPARED_CONTEXT)
    graph=draft._proposal_forward
    self.assertEqual(graph.prefix_capacities,(WINDOW,PREPARED_CONTEXT))
    self.assertEqual(metadata['padding_input_bytes'],expected_padding_bytes((WINDOW,PREPARED_CONTEXT)))
    # Prefix lengths 3, 4, and 6 lie before, at, and beyond the sliding window.
    for position,rows in ((0,3),(3,1),(4,2)):
      features=rng.normal(0,.4,(1,rows,16)).astype(np.float32)
      for model in (raw,draft):observe(model,features,position)
      prefix_end=position+rows
      assert_forward_matches(self,draft,raw,prefix_end)
      sliding,full=draft._prefix_kv
      self.assertEqual((sliding.start_pos,sliding.end_pos),(max(0,prefix_end-WINDOW),prefix_end))
      self.assertEqual((full.start_pos,full.end_pos),(0,prefix_end))
      self.assertEqual(sliding.key.shape[2],min(prefix_end,WINDOW))
      self.assertEqual(full.key.shape[2],prefix_end)
    # Replacing a suffix after the sliding layer evicted its earlier rows must rebuild it.
    replacement=rng.normal(0,.4,(1,2,16)).astype(np.float32)
    for model in (raw,draft):observe(model,replacement,2)
    assert_forward_matches(self,draft,raw,4)
    self.assertEqual((draft._prefix_kv[0].start_pos,draft._prefix_kv[0].end_pos),(0,4))
    self.assertEqual((draft._prefix_kv[1].start_pos,draft._prefix_kv[1].end_pos),(0,4))
    # Append after rollback, then rebind the same capacity-16 TinyJit across 3+ positions.
    suffix=rng.normal(0,.4,(1,2,16)).astype(np.float32)
    for model in (raw,draft):observe(model,suffix,4)
    assert_forward_matches(self,draft,raw,6)
    self.assertGreaterEqual(graph.calls,3)
    self.assertGreaterEqual(graph.jit.cnt,3)
    self.assertIsNotNone(graph.jit.captured)
    draft.reset()
    raw.reset()
    assert_forward_matches(self,draft,raw,0)
    draft.close_prepared_proposals()

  def test_full_and_sliding_layer_suffix_replacement_matches_unprepared_after_multiple_rebinds(self):
    raw,draft,rng=setup(9403)
    set_window(raw,draft)
    draft.prepare_proposals(max_context=PREPARED_CONTEXT)
    graph=draft._proposal_forward
    committed=np.empty((1,0,16),np.float32)
    for position,rows in ((0,7),(7,3),(2,2),(4,5),(6,0)):
      suffix=rng.normal(0,.4,(1,rows,16)).astype(np.float32)
      committed=np.concatenate((committed[:,:position],suffix),axis=1)
      for model in (raw,draft):observe(model,suffix,position)
      end=committed.shape[1]
      assert_forward_matches(self,draft,raw,end)
      slide,full=draft._prefix_kv
      self.assertEqual((slide.start_pos,slide.end_pos),(max(0,end-WINDOW),end))
      self.assertEqual((full.start_pos,full.end_pos),(0,end))
    self.assertIsNotNone(graph.jit.captured)
    draft.close_prepared_proposals()

  def test_borrowed_head_and_source_weight_identity_guards_remain_active(self):
    _raw,draft,_rng=setup(9404)
    draft.prepare_proposals(max_context=PREPARED_CONTEXT)
    graph=draft._proposal_forward
    noise=Tensor([NOISE_IDS],device='CPU',dtype=dtypes.int32).realize()
    head=draft.lm_head
    weight=draft.weights['fc.weight']
    draft.lm_head=lambda hidden:hidden
    with self.assertRaisesRegex(ValueError,'borrowed modules changed'):
      graph(noise,draft._prefix_kv,0)
    draft.lm_head=head
    draft.weights['fc.weight']=Tensor.zeros(*weight.shape,device='CPU',dtype=weight.dtype).realize()
    with self.assertRaisesRegex(ValueError,'borrowed modules changed'):
      graph(noise,draft._prefix_kv,0)
    draft.weights['fc.weight']=weight
    draft.close_prepared_proposals()


if __name__=='__main__':unittest.main()
