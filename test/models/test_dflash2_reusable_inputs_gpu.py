"""Explicit Mali qualification of owned proposal-input writes and replay."""
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes
from test.models.test_dflash2_prepared import observe, setup


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestDFlash2ReusableInputsMali(unittest.TestCase):
  def test_half_inputs_and_complete_graph_replay_survive_append_rollback_and_reset(self):
    control,candidate,rng=setup(961,'CL',half=True)
    control.prepare_proposals(max_context=32)
    candidate.prepare_proposals(max_context=32,reuse_prefix_inputs=True)
    noise=Tensor([[3,15,15,15]],device='CL',dtype=dtypes.int32).realize()
    identities=None
    try:
      for start,rows in ((0,1),(1,1),(2,3),(5,3),(2,1),(3,4),(0,2)):
        values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
        for draft in (control,candidate):observe(draft,values,start)
        position=start+rows
        source=[(cache.key.numpy().copy(),cache.value.numpy().copy()) for cache in candidate._prefix_kv]
        expected=control._proposal_forward._inputs(noise,control._prefix_kv,position)
        actual=candidate._proposal_forward._inputs(noise,candidate._prefix_kv,position)
        for left,right in zip(expected,actual):np.testing.assert_array_equal(left.numpy().view(np.uint8),right.numpy().view(np.uint8))
        current=tuple(id(value.uop.base.buffer) for value in actual)
        if identities is not None:self.assertEqual(current,identities)
        identities=current
        for draft in (control,candidate):
          result=draft._proposal_forward(noise,draft._prefix_kv,position)
          arrays=result.logits.numpy().copy(),result.hidden.numpy().copy()
          if draft is control:reference=arrays
          else:
            for left,right in zip(reference,arrays):np.testing.assert_array_equal(left.view(np.uint8),right.view(np.uint8))
        self.assertEqual(control.propose(3,position=position,max_tokens=4),candidate.propose(3,position=position,max_tokens=4))
        for cache,(key,value) in zip(candidate._prefix_kv,source):
          np.testing.assert_array_equal(cache.key.numpy(),key)
          np.testing.assert_array_equal(cache.value.numpy(),value)
      for draft in (control,candidate):draft.reset()
      expected=control._proposal_forward._inputs(noise,control._prefix_kv,0)
      actual=candidate._proposal_forward._inputs(noise,candidate._prefix_kv,0)
      for left,right in zip(expected,actual):np.testing.assert_array_equal(left.numpy().view(np.uint8),right.numpy().view(np.uint8))
      self.assertGreater(candidate.proposal_preparation_metadata()['input_reused_bytes'],0)
      self.assertGreater(candidate.proposal_preparation_metadata()['native_input_copy_commands'],0)
    finally:
      for draft in (control,candidate):draft.close_prepared_proposals()

  def test_native_enqueue_failure_invalidates_staging_and_retry_restores_exact_inputs(self):
    from tinygrad.llm.kernels import mali_copy
    control,candidate,rng=setup(963,'CL',half=True)
    control.prepare_proposals(max_context=32)
    candidate.prepare_proposals(max_context=32,reuse_prefix_inputs=True)
    noise=Tensor([[3,15,15,15]],device='CL',dtype=dtypes.int32).realize()
    try:
      for start,rows in ((0,7),(7,1)):
        values=rng.normal(0,.3,(1,rows,16)).astype(np.float32)
        for draft in (control,candidate):observe(draft,values,start)
      graph=candidate._proposal_forward
      graph._inputs(noise,candidate._prefix_kv,8)
      self.assertGreater(graph.native_input_copy_commands,0)
      source=[(cache.key.numpy().copy(),cache.value.numpy().copy()) for cache in candidate._prefix_kv]
      graph.prefix_inputs[2][:,:,8:].assign(float('nan')).realize()
      copy_rect=mali_copy._copy_rect
      def fail(*args):
        copy_rect(*args)
        raise RuntimeError('injected native copy enqueue failure')
      graph.invalidate_prefix_inputs()
      with patch.object(mali_copy,'_copy_rect',fail),self.assertRaisesRegex(RuntimeError,'native copy enqueue failure'):
        graph._inputs(noise,candidate._prefix_kv,8)
      self.assertEqual(graph.preserved_prefix_rows,0)
      expected=control._proposal_forward._inputs(noise,control._prefix_kv,8)
      actual=graph._inputs(noise,candidate._prefix_kv,8)
      for left,right in zip(expected,actual):np.testing.assert_array_equal(left.numpy().view(np.uint8),right.numpy().view(np.uint8))
      for cache,(key,value) in zip(candidate._prefix_kv,source):
        np.testing.assert_array_equal(cache.key.numpy(),key)
        np.testing.assert_array_equal(cache.value.numpy(),value)
    finally:
      for draft in (control,candidate):draft.close_prepared_proposals()

  def test_half_retry_after_partial_write_clears_nonfinite_padding(self):
    control,candidate,rng=setup(962,'CL',half=True)
    control.prepare_proposals(max_context=32)
    candidate.prepare_proposals(max_context=32,reuse_prefix_inputs=True)
    noise=Tensor([[3,15,15,15]],device='CL',dtype=dtypes.int32).realize()
    try:
      values=rng.normal(0,.3,(1,7,16)).astype(np.float32)
      for draft in (control,candidate):observe(draft,values,0)
      graph=candidate._proposal_forward
      graph._inputs(noise,candidate._prefix_kv,7)
      graph.prefix_inputs[2][:,:,7:].assign(float('nan')).realize()
      assign=Tensor.assign
      def fail(value,data):
        assign(value,data).realize()
        raise RuntimeError('injected partial Mali staging write')
      with patch('tinygrad.llm.kernels.mali_copy.copy_rows',return_value=False), \
           patch.object(Tensor,'assign',fail),self.assertRaisesRegex(RuntimeError,'partial Mali'):
        graph._inputs(noise,candidate._prefix_kv,7)
      expected=control._proposal_forward._inputs(noise,control._prefix_kv,7)
      actual=graph._inputs(noise,candidate._prefix_kv,7)
      for left,right in zip(expected,actual):
        np.testing.assert_array_equal(left.numpy().view(np.uint8),right.numpy().view(np.uint8))
        self.assertTrue(np.isfinite(right.numpy()).all())
    finally:
      for draft in (control,candidate):draft.close_prepared_proposals()


if __name__=='__main__':unittest.main()
