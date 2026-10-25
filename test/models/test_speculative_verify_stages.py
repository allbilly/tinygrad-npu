"""Experimental first-three verifier: exact greedy/state protocol and real tiny backends."""
import os
import unittest
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import TargetOutput
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _join_outputs
from test.models.test_speculative import CausalTarget, OracleDraft, oracle
from test.models import test_speculative_prefix_trace as trace_helpers
from test.models.test_speculative_prefix_trace import EightDraft
from test.models.test_speculative_prepared import assert_states, production_greedy_ids


class EightOracle(OracleDraft):
  block_size = 8
  def __init__(self, wrong_at=None):
    super().__init__(wrong_at)
    self.proposals,self.observations = [],[]
  def propose(self, anchor, **kwargs):
    result = list(super().propose(anchor,**kwargs))
    self.proposals.append((anchor,kwargs['position'],result))
    return result
  def observe(self, output, start_pos):
    self.observations.append((start_pos,output.features[0].numpy().copy(),output.hidden is output.features[0]))
    super().observe(output,start_pos)


class TrackedEightDraft(EightDraft):
  def __init__(self, *args):
    super().__init__(*args)
    self.observations=[]
  def observe(self, output, start_pos):
    self.observations.append((start_pos,tuple(value.numpy().copy() for value in output.features)))
    super().observe(output,start_pos)


class RecordingTarget(CausalTarget):
  def __init__(self, context=64):
    super().__init__(context)
    self.calls,self.bad_position,self.bad_kind = [],None,None
  def target_forward(self, tokens, start_pos, *, capture_layers=(), last_only=False, state_trace=None):
    self.calls.append((start_pos,tokens.tolist()[0],last_only))
    output = super().target_forward(tokens,start_pos,capture_layers=capture_layers,last_only=last_only)
    if state_trace is not None:state_trace.record(output.hidden)
    if self.bad_position==start_pos:
      kind,self.bad_position = self.bad_kind,None
      if kind=='exception':raise RuntimeError('injected verifier-stage failure after state write')
      return TargetOutput(output.logits*float('nan'),output.hidden,output.features).realize()
    return output


class ReusedOutputTarget(RecordingTarget):
  """Every call overwrites the same owned logits/features buffers."""
  def __init__(self):
    super().__init__()
    self.logit_buffer=Tensor.empty(1,8,17,device='CPU',dtype='float32').realize()
    self.hidden_buffer=Tensor.empty(1,8,4,device='CPU',dtype='float32').realize()
    self.expected_features={}
  def target_forward(self, tokens, start_pos, **kwargs):
    output=super().target_forward(tokens,start_pos,**kwargs)
    self.expected_features[start_pos]=output.hidden.numpy().copy()
    logits=self.logit_buffer[:,:output.logits.shape[1]]
    hidden=self.hidden_buffer[:,:output.hidden.shape[1]]
    Tensor.realize(logits.assign(output.logits),hidden.assign(output.hidden))
    return TargetOutput(logits,hidden,(hidden,) if kwargs.get('capture_layers') else ()).realize()


class ProtocolTrace:
  """Deterministic row-state fixture for the coordinator, not a native-kernel proof."""
  nbytes,state_bytes,window_bytes,restored_bytes = 8*12,12,0,24
  def __init__(self, target):
    self.target,self.buffers,self.position,self.valid_rows = target,(),-1,0
    self.recorded,self.restores = [],[]
  def prepare(self):
    if not self.buffers:self.buffers=(Tensor([0],device='CPU').realize(),)
  def bind(self):return self
  def invalidate(self):self.position,self.valid_rows=-1,0
  def record(self, hidden):self.recorded=hidden.tolist()[0]
  def mark_valid(self, position, rows):self.position,self.valid_rows=position,rows
  def can_restore(self, position, rows):return self.position==position and 1<=rows<=self.valid_rows
  def restore(self, position, rows):
    if not self.can_restore(position,rows):raise RuntimeError('invalid fixture prefix history')
    state,h0,h1,_ = self.recorded[rows-1]
    block=self.target.blk[0]
    Tensor.realize(block.recurrent_state.assign(Tensor([int(state)],device='CPU',dtype='int32')),
                   block.conv_state.assign(Tensor([[int(h0),int(h1)]],device='CPU',dtype='int32')))
    self.restores.append((position,rows))
    return block.recurrent_state,block.conv_state


class TestFirstThreeProtocol(unittest.TestCase):
  prompt = [2,5,3]
  def make(self, *, staged=True, wrong_at=None, trace=False, early=False, context=64):
    target,draft = RecordingTarget(context),EightOracle(wrong_at)
    decoder=GreedySpeculativeDecoder(target,draft,verify_first_three=staged,early_anchor=early)
    if trace:decoder._prefix_trace=ProtocolTrace(target)
    return target,draft,decoder
  def assert_state(self, target, prompt, delivered):
    expected,state,history=oracle(prompt,len(delivered))
    self.assertEqual(delivered,expected)
    self.assertEqual(target._cached_tokens,prompt+expected)
    self.assertEqual(int(target.blk[0].recurrent_state.item()),state)
    self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]),history)
    np.testing.assert_array_equal(target.blk[0].cache_kv.numpy()[:len(prompt+expected)],prompt+expected)
  def assert_same(self, left, right):
    self.assertEqual(left._cached_tokens,right._cached_tokens)
    for name in ('recurrent_state','conv_state'):
      np.testing.assert_array_equal(getattr(left.blk[0],name).numpy(),getattr(right.blk[0],name).numpy())
  def test_all_accept_continues_at_f3_with_same_proposal_one_observe_and_full_alias_preserving_output(self):
    target,draft,decoder=self.make(trace=True)
    output=list(decoder.generate(self.prompt,max_new_tokens=8))
    self.assert_state(target,self.prompt,output)
    self.assertEqual(len(draft.proposals),1)
    self.assertEqual([len(ids) for _,ids,_ in target.calls],[3,3,5])
    self.assertEqual(target.calls[2][:2],(6,output[3:]))
    self.assertEqual([position for position,_,_ in draft.observations],[0,3])
    self.assertEqual([value.shape[1] for _,value,_ in draft.observations],[3,8])
    self.assertTrue(all(alias for _,_,alias in draft.observations))
    self.assertEqual(decoder.stats.blocks,1)
    self.assertEqual(decoder.stats.proposed_tokens,7)
    self.assertEqual(decoder.stats.accepted_tokens,7)
    self.assertEqual(decoder.stats.verify_calls,2)
    self.assertEqual((decoder.stats.verify_first_calls,decoder.stats.verify_rest_calls),(1,1))
    self.assertEqual(decoder.stats.target_input_tokens,11)
    self.assertEqual(decoder.stats.verify_skipped_input_tokens,0)
    self.assertGreater(decoder.stats.verify_join_seconds,0)
    self.assertGreaterEqual(decoder.stats.verify_seconds,decoder.stats.verify_join_seconds)
    self.assertEqual(decoder.stats.replay_calls,0)
  def test_rejection_at_every_candidate_matches_unstaged_oracle_ids_and_complete_state(self):
    for trace in (False,True):
      for wrong_at in range(7):
        with self.subTest(trace=trace,wrong_at=wrong_at):
          target,draft,decoder=self.make(wrong_at=wrong_at,trace=trace)
          raw,_,baseline=self.make(staged=False,wrong_at=wrong_at)
          actual=list(decoder.generate(self.prompt,max_new_tokens=17))
          expected=list(baseline.generate(self.prompt,max_new_tokens=17))
          self.assert_state(target,self.prompt,actual)
          self.assertEqual(actual,expected)
          self.assert_same(target,raw)
          self.assertEqual(decoder.stats.proposed_tokens,baseline.stats.proposed_tokens)
          self.assertEqual(decoder.stats.accepted_tokens,baseline.stats.accepted_tokens)
          self.assertEqual(decoder.stats.rejected_blocks,baseline.stats.rejected_blocks)
          if wrong_at<3:
            self.assertEqual(decoder.stats.verify_rest_calls,0)
            self.assertGreater(decoder.stats.verify_skipped_input_tokens,0)
            self.assertLess(decoder.stats.target_input_tokens,baseline.stats.target_input_tokens)
          else:self.assertGreater(decoder.stats.verify_rest_calls,0)
          if trace:self.assertEqual(decoder.stats.replay_calls,0)
  def test_close_at_each_stage_boundary_restores_latest_trace_or_original_snapshot_and_timed_replay(self):
    for early in (False,True):
      for delivered in (1,2,3,4,5,7):
        with self.subTest(early=early,delivered=delivered):
          target,_,decoder=self.make(trace=True,early=early)
          stream=decoder.generate(self.prompt,max_new_tokens=12)
          output=[next(stream) for _ in range(delivered)]
          stream.close()
          self.assert_state(target,self.prompt,output)
          self.assertEqual(decoder.finish_reason,'closed')
          if delivered<=3:
            self.assertEqual(decoder.stats.replay_calls,1)
            self.assertGreater(decoder.stats.replay_seconds,0)
          else:
            self.assertEqual(decoder.stats.replay_calls,0)
            self.assertEqual(decoder._prefix_trace.restores[-1],(6,delivered-3))
          self.assertEqual(list(decoder.generate([9,1],max_new_tokens=8)),oracle([9,1],8)[0])
  def test_eos_and_partial_output_context_verification_budgets_preserve_original_semantics(self):
    expected=oracle(self.prompt,18)[0]
    unique_eos=[token for i,token in enumerate(expected[:8]) if token not in expected[:i]]
    for eos in unique_eos:
      with self.subTest(eos=eos):
        target,_,decoder=self.make(trace=True)
        raw,_,baseline=self.make(staged=False)
        actual=list(decoder.generate(self.prompt,max_new_tokens=18,eos_token_ids=(eos,)))
        reference=list(baseline.generate(self.prompt,max_new_tokens=18,eos_token_ids=(eos,)))
        self.assertEqual(actual,reference)
        self.assert_state(target,self.prompt,actual)
        self.assert_same(target,raw)
        self.assertEqual(decoder.finish_reason,'eos')
    for context,count,budget in ((64,1,None),(64,4,None),(64,7,1),(64,9,2),(64,9,3),(64,9,5),(9,12,7)):
      with self.subTest(context=context,count=count,budget=budget):
        target,_,decoder=self.make(trace=True,context=context)
        raw,_,baseline=self.make(staged=False,context=context)
        actual=list(decoder.generate(self.prompt,max_new_tokens=count,verification_budget=budget))
        self.assertEqual(actual,list(baseline.generate(self.prompt,max_new_tokens=count,verification_budget=budget)))
        self.assert_state(target,self.prompt,actual)
        self.assert_same(target,raw)
  def test_cancel_after_first_or_second_physical_stage_restores_zero_delivered_original_snapshot_then_retry(self):
    for stage in ('verify_first','verify_rest'):
      with self.subTest(stage=stage):
        target,_,decoder=self.make(trace=True)
        flag={'stop':False}
        forward=decoder._forward
        def cancelling(ids,position,phase,*,last_only=False):
          result=forward(ids,position,phase,last_only=last_only)
          if phase==stage:flag['stop']=True
          return result
        with patch.object(decoder,'_forward',new=cancelling):
          self.assertEqual(list(decoder.generate(self.prompt,max_new_tokens=8,cancelled=lambda:flag['stop'])),[])
        self.assert_state(target,self.prompt,[])
        self.assertEqual(decoder.finish_reason,'cancelled')
        self.assertEqual(decoder.stats.trace_restore_calls,0)
        self.assertEqual(list(decoder.generate(self.prompt,max_new_tokens=8)),oracle(self.prompt,8)[0])
  def test_stage_exception_or_nonfinite_outputs_fail_closed_with_original_snapshot_and_clean_retry(self):
    for phase_position in (3,6):
      for kind in ('exception','nan'):
        for early in (False,True):
          with self.subTest(position=phase_position,kind=kind,early=early):
            target,_,decoder=self.make(trace=True,early=early)
            target.bad_position,target.bad_kind=phase_position,kind
            stream=decoder.generate(self.prompt,max_new_tokens=8)
            delivered=[next(stream)] if early else []
            with self.assertRaisesRegex(RuntimeError if kind=='exception' else ValueError,'injected|Nonfinite'):
              list(stream)
            self.assertEqual(decoder.finish_reason,'error')
            self.assertEqual(target._cached_tokens,[])
            self.assertEqual(decoder._prefix_trace.valid_rows,0)
            _,state,history=oracle(self.prompt,len(delivered))
            self.assertEqual(int(target.blk[0].recurrent_state.item()),state)
            self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]),history)
            self.assertEqual(decoder.stats.trace_restore_calls,0)
            self.assertEqual(list(decoder.generate(self.prompt,max_new_tokens=8)),oracle(self.prompt,8)[0])
  def test_join_preserves_feature_aliases_and_materialized_first_stage_survives_later_state_writes(self):
    target,_,_=self.make()
    first=target.target_forward(Tensor([self.prompt],device='CPU'),0,capture_layers=(0,))
    saved=first.hidden.numpy().copy()
    rest=target.target_forward(Tensor([[1,2]],device='CPU'),3,capture_layers=(0,))
    joined=_join_outputs(first,rest)
    self.assertIs(joined.hidden,joined.features[0])
    np.testing.assert_array_equal(joined.hidden.numpy()[:,:3],saved)
    np.testing.assert_array_equal(joined.logits.numpy(),np.concatenate((first.logits.numpy(),rest.logits.numpy()),axis=1))
  def test_first_stage_features_are_owned_before_reused_target_buffers_are_overwritten(self):
    target,draft=ReusedOutputTarget(),EightOracle()
    decoder=GreedySpeculativeDecoder(target,draft,verify_first_three=True)
    actual=list(decoder.generate(self.prompt,max_new_tokens=8))
    self.assert_state(target,self.prompt,actual)
    position,observed,alias=draft.observations[-1]
    self.assertEqual(position,3)
    self.assertTrue(alias)
    np.testing.assert_array_equal(observed[:,:3],target.expected_features[3])
    np.testing.assert_array_equal(observed[:,3:],target.expected_features[6])
    self.assertGreater(decoder.stats.verify_join_seconds,0)
  def test_invalid_option_and_default_do_not_change_existing_path(self):
    for invalid in (1,None,'3'):
      with self.assertRaisesRegex(ValueError,'must be a bool'):
        GreedySpeculativeDecoder(CausalTarget(),EightOracle(),verify_first_three=invalid)
    with self.assertRaisesRegex(ValueError,'eight-slot'):
      GreedySpeculativeDecoder(CausalTarget(),OracleDraft(),verify_first_three=True)
    target=RecordingTarget()
    decoder=GreedySpeculativeDecoder(target,EightOracle())
    list(decoder.generate(self.prompt,max_new_tokens=8))
    self.assertFalse(decoder.stats.verify_first_three)
    self.assertEqual([len(ids) for _,ids,_ in target.calls],[3,8])


class TestFirstThreeNativeGDN(unittest.TestCase):
  device='CPU'
  def models(self):
    fixture=trace_helpers.TestNativeGDNPrefixTrace()
    fixture.device=self.device
    return fixture.models()
  def test_real_hybrid_accept_reject_and_closed_prefix_bits_cache_and_independent_ordinary_ids(self):
    with self.models() as (raw,target):
      prompt=[2,5,3]
      expected=production_greedy_ids(raw,prompt,10)
      raw_draft,draft=TrackedEightDraft(len(prompt),expected),TrackedEightDraft(len(prompt),expected)
      baseline=GreedySpeculativeDecoder(raw,raw_draft,prepared_target=True,prepared_dynamic_rows=True,
          prepared_prefill_dynamic=True,prefix_trace=True)
      decoder=GreedySpeculativeDecoder(target,draft,prepared_target=True,prepared_dynamic_rows=True,
          prepared_prefill_dynamic=True,prefix_trace=True,verify_first_three=True)
      for wrong_at in (None,0,1,2,3,4,5,6,None):
        raw_draft.wrong_at=draft.wrong_at=wrong_at
        raw_draft.observations.clear()
        draft.observations.clear()
        actual=list(decoder.generate(prompt,max_new_tokens=10,prefill_chunk_size=4))
        reference=list(baseline.generate(prompt,max_new_tokens=10,prefill_chunk_size=4))
        self.assertEqual(actual,expected)
        self.assertEqual(actual,reference)
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        assert_states(self,target,raw,len(prompt)+len(actual))
        self.assertEqual(decoder.stats.accepted_tokens,baseline.stats.accepted_tokens)
        self.assertEqual([position for position,_ in draft.observations],[position for position,_ in raw_draft.observations])
        for (_,actual_features),(_,reference_features) in zip(draft.observations,raw_draft.observations):
          for value,reference_value in zip(actual_features,reference_features):
            np.testing.assert_array_equal(value.view(np.uint32),reference_value.view(np.uint32))
      raw_draft.wrong_at=draft.wrong_at=None
      for delivered in (1,2,3,4,7):
        actual=decoder.generate(prompt,max_new_tokens=10,prefill_chunk_size=4)
        reference=baseline.generate(prompt,max_new_tokens=10,prefill_chunk_size=4)
        self.assertEqual([next(actual) for _ in range(delivered)],expected[:delivered])
        self.assertEqual([next(reference) for _ in range(delivered)],expected[:delivered])
        actual.close()
        reference.close()
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        assert_states(self,target,raw,len(prompt)+delivered)
      # Repeated short/long and closed requests must not introduce a full
      # eight-row verifier merely to consume their final one-to-three inputs.
      self.assertNotIn(('verify',8,False),decoder._prepared_target.calls)
      first=decoder._prepared_target.calls[('verify_first',3,False)]
      rest=decoder._prepared_target.calls[('verify_rest',5,False)]
      self.assertEqual((first.capacity,rest.capacity),(3,5))
      self.assertIs(first.state_trace,decoder._prefix_trace)
      self.assertIs(rest.state_trace,decoder._prefix_trace)
      self.assertGreaterEqual(first.jit.cnt,3)
      self.assertGreaterEqual(rest.jit.cnt,3)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestFirstThreeNativeGDNMali(TestFirstThreeNativeGDN):
  device='CL'
