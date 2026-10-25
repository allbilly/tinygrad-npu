"""Prepared verifier output, numerical-state and cancellation regression checks."""
import os, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _PreparedTargetForward
from test.models.test_speculative import FixedDraft, ordinary_ids, real_pair
from test.models.test_speculative_target import config, pair


def assert_states(test, actual, expected, prefix_len):
  for a,b in zip(actual.blk,expected.blk):
    for name in ('recurrent_state','conv_state','cache_kv'):
      if hasattr(a,name):
        av,bv = getattr(a,name).numpy(),getattr(b,name).numpy()
        if name=='cache_kv':av,bv = av[:,:,:,:prefix_len],bv[:,:,:,:prefix_len]
        np.testing.assert_array_equal(av.view(np.uint32),bv.view(np.uint32),err_msg=name)
  test.assertEqual(actual.config,expected.config)


def symbolic_eager_reference(model):
  """Use the prepared graph's bounds, without JIT or its output/cache helpers.

  Production generate passes bound symbolic positions. Static integer positions
  can select shorter attention reductions and change rounding before JIT capture.
  Matching bounds isolates JIT/state correctness from that existing difference.
  """
  forward = model.target_forward
  def run(tokens, position, *, capture_layers=(), last_only=False):
    rows = tokens.shape[1]
    if type(rows) is not int or type(position) is not int:raise ValueError('Reference expects concrete token count and position')
    suffix = f'{rows}_{int(last_only)}'
    pos = UOp.variable('reference_pos_'+suffix,0,model.max_context-rows).bind(position)
    count = UOp.variable('reference_rows_'+suffix,rows,rows).bind(rows)
    return forward(tokens[:,:count],pos,capture_layers=capture_layers,last_only=last_only)
  return run


def production_greedy_ids(model, prompt, count):
  stream = model.generate(list(prompt),chunk_size=4,temperature=0.0)
  try:return [next(stream) for _ in range(count)]
  finally:stream.close()


class TestPreparedTargetForward(unittest.TestCase):
  def test_mixed_rows_positions_context_tail_and_freed_scratch_match_raw_bound_graph(self):
    for hybrid in (False,True):
      with self.subTest(hybrid=hybrid):
        raw,target = real_pair(hybrid)
        prepared = _PreparedTargetForward(target,(0,1))
        position = 0
        cases = [(n,None) for n in (4,1,3,2,4,1,3,4,2)]+[(2,0),(4,None),(1,None)]
        for index,(rows,reset) in enumerate(cases):
          if reset is not None:position = reset
          capacity = min(4,target.max_context-position)
          ids = [(index*3+j*2)%13 for j in range(rows)]
          tokens = Tensor([ids+[0]*(capacity-rows)],device='CPU',dtype='int32').realize()
          pos = UOp.variable('raw_spec_pos_'+str(capacity),0,target.max_context-capacity).bind(position)
          count = UOp.variable('raw_spec_rows_'+str(capacity),1,capacity).bind(rows)
          expected = raw.target_forward(tokens[:,:count],pos,capture_layers=(0,1))
          actual = prepared(ids,position,'verify',4,False)
          self.assertIs(actual.hidden,actual.features[-1])
          for av,bv in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
            reference = bv.pad_to(bv.max_shape).numpy()[:,:rows]
            np.testing.assert_array_equal(av.numpy().view(np.uint32),reference.view(np.uint32))
          np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected.logits.pad_to(expected.logits.max_shape).numpy()[:,:rows].argmax(-1))
          assert_states(self,target,raw,position+rows)
          # Re-reading results must never re-execute recurrent writes.
          before = [(getattr(block,name),getattr(block,name).numpy().copy()) for block in target.blk
                    for name in ('conv_state','recurrent_state') if hasattr(block,name)]
          actual.logits.numpy()
          actual.hidden.numpy()
          for feature in actual.features:feature.numpy()
          for state,saved in before:np.testing.assert_array_equal(state.numpy(),saved)
          if index==5:prepared.free_intermediates()
          position += rows
        self.assertEqual(set(capacity for _,capacity,_ in prepared.calls),{2,4})
        self.assertGreaterEqual(prepared.calls[('verify',4,False)].jit.cnt,3)
        self.assertIsNotNone(prepared.calls[('verify',4,False)].jit.captured)

  def test_prepared_coordinator_matches_raw_after_accept_reject_close_and_repeated_requests(self):
    for hybrid in (False,True):
      with self.subTest(hybrid=hybrid):
        raw,target = real_pair(hybrid)
        raw.target_forward = symbolic_eager_reference(raw)
        reference,_ = real_pair(hybrid)
        prompt = [2,5,3]
        expected = ordinary_ids(reference,prompt,7)
        raw_draft,jit_draft = FixedDraft(len(prompt),expected),FixedDraft(len(prompt),expected)
        ordinary = GreedySpeculativeDecoder(raw,raw_draft)
        prepared = GreedySpeculativeDecoder(target,jit_draft,prepared_target=True)
        for wrong_at in (None,None,None,1):
          raw_draft.wrong_at = jit_draft.wrong_at = wrong_at
          actual = list(prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
          baseline = list(ordinary.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
          self.assertEqual(actual,expected)
          self.assertEqual(actual,baseline)
          self.assertEqual(target._cached_tokens,raw._cached_tokens)
          assert_states(self,target,raw,len(prompt)+len(actual))
          self.assertEqual(prepared.stats.replay_calls,ordinary.stats.replay_calls)
          self.assertTrue(prepared.stats.prepared_target)
          self.assertFalse(prepared.stats.prepared_dynamic_rows)
          self.assertGreater(prepared.stats.prepared_graphs,0)
          self.assertFalse(ordinary.stats.prepared_target)
          self.assertEqual(ordinary.stats.prepared_graphs,0)
        raw_draft.wrong_at = jit_draft.wrong_at = None
        raw_stream = ordinary.generate(prompt,max_new_tokens=7,prefill_chunk_size=4)
        jit_stream = prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4)
        self.assertEqual([next(jit_stream),next(jit_stream)],[next(raw_stream),next(raw_stream)])
        jit_stream.close()
        raw_stream.close()
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        self.assertEqual(target._cached_tokens,prompt+expected[:2])
        assert_states(self,target,raw,len(prompt)+2)
        self.assertEqual(prepared.stats.replay_calls,1)

  def test_invalid_prepared_bounds_do_not_allocate_decoder_state(self):
    _,target = real_pair(False)
    prepared = _PreparedTargetForward(target,(0,))
    for ids,position,capacity in (([],0,4),([1],-1,4),([1],24,4),([1,2],23,4),([1,2],0,1)):
      with self.assertRaises(ValueError):prepared(ids,position,'verify',capacity,False)
    self.assertTrue(all(not hasattr(block,'cache_kv') for block in target.blk))


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestPreparedTargetForwardMali(unittest.TestCase):
  def test_fp16_cl_exact_row_graphs_state_replay_and_closed_stream_match_eager(self):
    from tinygrad.llm.kernels.mali import configure
    for hybrid in (False,True):
      with self.subTest(hybrid=hybrid):
        cfg = replace(config(hybrid),attn_output_gate=hybrid)
        raw,target = pair(cfg,'CL',half=True)
        configure(raw,recurrent_scan=hybrid,pack_weights=False)
        configure(target,recurrent_scan=hybrid,pack_weights=False)
        raw.target_forward = symbolic_eager_reference(raw)
        helper = _PreparedTargetForward(target,(0,2),dynamic_rows=False)
        cases = (([2,5,11,7],0),([3],4),([19,9,1],5),([8,3,7,4],0),([2,1,9,5],4),([6],8))
        for index,(ids,position) in enumerate(cases):
          expected = raw.target_forward(Tensor([ids],device='CPU',dtype='int32'),position,capture_layers=(0,2))
          actual = helper(ids,position,'verify',4,False)
          self.assertIs(actual.hidden,actual.features[-1])
          for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
            np.testing.assert_array_equal(a.numpy().view(np.uint32),b.numpy().view(np.uint32))
          np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected.logits.numpy().argmax(-1))
          assert_states(self,target,raw,position+len(ids))
          before = [(getattr(block,name),getattr(block,name).numpy().copy()) for block in target.blk
                    for name in ('conv_state','recurrent_state') if hasattr(block,name)]
          actual.logits.numpy()
          actual.hidden.numpy()
          for feature in actual.features:feature.numpy()
          for state,saved in before:np.testing.assert_array_equal(state.numpy(),saved)
          if index==3:helper.free_intermediates()
        self.assertIsNotNone(helper.calls[('verify',4,False)].jit.captured)

        prompt = [2,5,3]
        # Independent production path: ordinary generate uses its own greedy
        # JITs and never calls this fixture's target_forward adapter.
        expected_ids = production_greedy_ids(raw,prompt,7)
        raw_draft,jit_draft = FixedDraft(len(prompt),expected_ids),FixedDraft(len(prompt),expected_ids)
        eager = GreedySpeculativeDecoder(raw,raw_draft)
        prepared = GreedySpeculativeDecoder(target,jit_draft,prepared_target=True)
        for wrong_at in (None,None,1):
          raw_draft.wrong_at = jit_draft.wrong_at = wrong_at
          actual_ids = list(prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
          baseline_ids = list(eager.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
          self.assertEqual(actual_ids,expected_ids)
          self.assertEqual(actual_ids,baseline_ids)
          self.assertEqual(target._cached_tokens,raw._cached_tokens)
          self.assertEqual(prepared.stats.replay_calls,eager.stats.replay_calls)
          assert_states(self,target,raw,len(prompt)+len(actual_ids))
        raw_draft.wrong_at = jit_draft.wrong_at = None
        # Repeat closure to exercise capture and replay of last-only recovery,
        # rather than testing only its first eager TinyJit call.
        for _ in range(3):
          raw_stream = eager.generate(prompt,max_new_tokens=7,prefill_chunk_size=4)
          jit_stream = prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4)
          self.assertEqual([next(jit_stream),next(jit_stream)],[next(raw_stream),next(raw_stream)])
          jit_stream.close()
          raw_stream.close()
          self.assertEqual(target._cached_tokens,raw._cached_tokens)
          self.assertEqual(target._cached_tokens,prompt+expected_ids[:2])
          self.assertEqual(prepared.stats.replay_calls,1)
          assert_states(self,target,raw,len(prompt)+2)
        self.assertIsNotNone(prepared._prepared_target)
        if prepared._prepared_target is not None:
          self.assertIsNotNone(prepared._prepared_target.calls[('replay',2,True)].jit.captured)


if __name__=='__main__':unittest.main()
