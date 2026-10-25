"""Staged early-anchor stream and final target-state contracts."""
import unittest
import numpy as np
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from tinygrad.llm.speculative import GreedySpeculativeDecoder as LateDecoder
from test.models.test_speculative import CausalTarget, OracleDraft, FixedDraft, oracle, ordinary_ids, real_pair


class TraceDraft(OracleDraft):
  def __init__(self, wrong_at=None):
    super().__init__(wrong_at)
    self.observations,self.proposals = [],[]
  def observe(self, output, start_pos):
    self.observations.append(start_pos)
    super().observe(output,start_pos)
  def propose(self, anchor, *, position, max_tokens):
    self.proposals.append((anchor,position,max_tokens))
    return super().propose(anchor,position=position,max_tokens=max_tokens)


def fixture(*, wrong_at=None, context=64, early=True):
  target,draft,events = CausalTarget(context),TraceDraft(wrong_at),[]
  forward = target.target_forward
  def traced(tokens, position, **kwargs):
    events.append((position,tokens.shape[1],kwargs.get('last_only',False)))
    return forward(tokens,position,**kwargs)
  target.target_forward = traced
  return target,draft,GreedySpeculativeDecoder(target,draft,early_anchor=early),events


def assert_state(test, target, prompt, output):
  expected,state,history = oracle(prompt,len(output))
  test.assertEqual(output,expected)
  test.assertEqual(target._cached_tokens,prompt+output)
  test.assertEqual(int(target.blk[0].recurrent_state.item()),state)
  test.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]),history)
  np.testing.assert_array_equal(target.blk[0].cache_kv.numpy()[:len(prompt+output)],prompt+output)


class TestEarlyAnchor(unittest.TestCase):
  def test_first_yield_precedes_draft_observation_proposal_snapshot_and_verification(self):
    target,draft,decoder,events = fixture()
    prompt = [2,5,3]
    stream = decoder.generate(prompt,max_new_tokens=7)
    first = next(stream)
    self.assertEqual(first,oracle(prompt,1)[0][0])
    self.assertEqual(events,[(0,3,True)])
    self.assertEqual((draft.observations,draft.proposals),([],[]))
    self.assertEqual((decoder.stats.verify_calls,decoder.stats.replay_calls,decoder.stats.snapshot_bytes),(0,0,0))
    self.assertEqual(target._cached_tokens,prompt)
    stream.close()
    self.assertEqual(events,[(0,3,True),(3,1,True)])
    self.assertEqual((decoder.stats.verify_calls,decoder.stats.replay_calls),(0,1))
    self.assertEqual((draft.observations,draft.proposals),([],[]))
    self.assertEqual(decoder.finish_reason,'closed')
    self.assertTrue(decoder.stats.early_anchor)
    self.assertGreater(decoder.stats.replay_seconds,0)
    self.assertGreaterEqual(decoder.stats.total_seconds,decoder.stats.replay_seconds)
    assert_state(self,target,prompt,[first])

  def test_complete_acceptance_and_every_rejection_slot_match_late_ids_and_total_target_work(self):
    for wrong_at in (None,0,1,2):
      with self.subTest(wrong_at=wrong_at):
        prompt = [9,1,7]
        target,_,decoder,_ = fixture(wrong_at=wrong_at)
        baseline,_,late,_ = fixture(wrong_at=wrong_at,early=False)
        actual = list(decoder.generate(prompt,max_new_tokens=11))
        expected = list(late.generate(prompt,max_new_tokens=11))
        self.assertEqual(actual,expected)
        assert_state(self,target,prompt,actual)
        assert_state(self,baseline,prompt,expected)
        self.assertEqual(decoder.stats.target_calls,late.stats.target_calls)
        self.assertEqual(decoder.stats.target_input_tokens,late.stats.target_input_tokens)
        self.assertEqual(decoder.stats.accepted_tokens,late.stats.accepted_tokens)

  def test_close_after_accepted_followup_restores_exact_delivered_prefix(self):
    target,_,decoder,_ = fixture()
    prompt = [2,5,3]
    stream = decoder.generate(prompt,max_new_tokens=10)
    output = [next(stream),next(stream)]
    stream.close()
    assert_state(self,target,prompt,output)
    self.assertEqual(decoder.stats.verify_calls,1)
    self.assertEqual(decoder.stats.replay_calls,1)
    self.assertEqual(decoder.stats.accepted_tokens,1)
    self.assertEqual(decoder.stats.output_tokens,2)

  def test_verification_budget_still_caps_followups_after_the_early_anchor(self):
    for budget in (1,2):
      with self.subTest(budget=budget):
        target,draft,decoder,events = fixture()
        prompt = [2,5,3]
        stream = decoder.generate(prompt,max_new_tokens=9,verification_budget=budget)
        first = next(stream)
        self.assertEqual(draft.proposals,[])
        output = [first,*list(stream)]
        assert_state(self,target,prompt,output)
        self.assertTrue(all(limit<=budget+1 for _,_,limit in draft.proposals))
        self.assertTrue(all(rows<=budget+1 for position,rows,_ in events if position>=len(prompt)))

  def test_cancel_after_anchor_consumes_it_and_cancelling_before_prefill_does_no_work(self):
    target,draft,decoder,_ = fixture()
    prompt,flag = [2,5,3],{'stop':False}
    stream = decoder.generate(prompt,max_new_tokens=9,cancelled=lambda:flag['stop'])
    first = next(stream)
    flag['stop'] = True
    self.assertEqual(list(stream),[])
    assert_state(self,target,prompt,[first])
    self.assertEqual(decoder.finish_reason,'cancelled')
    self.assertEqual((decoder.stats.verify_calls,decoder.stats.replay_calls),(0,1))
    self.assertEqual(draft.proposals,[])
    target,_,decoder,events = fixture()
    self.assertEqual(list(decoder.generate(prompt,max_new_tokens=9,cancelled=lambda:True)),[])
    self.assertEqual(events,[])
    self.assertEqual(decoder.stats.output_tokens,0)

  def test_known_eos_single_output_and_context_limit_include_last_target_call(self):
    for reason in ('eos','length','context'):
      with self.subTest(reason=reason):
        prompt = [2,5,3]
        target,draft,decoder,events = fixture(context=4 if reason=='context' else 64)
        first = oracle(prompt,1)[0][0]
        stream = decoder.generate(prompt,max_new_tokens=1 if reason=='length' else 9,
                                  eos_token_ids=[first] if reason=='eos' else [])
        self.assertEqual(next(stream),first)
        self.assertEqual(events,[(0,3,True)])
        self.assertEqual(list(stream),[])
        self.assertEqual(events,[(0,3,True),(3,1,True)])
        self.assertEqual(decoder.finish_reason,reason)
        self.assertEqual((decoder.stats.verify_calls,decoder.stats.replay_calls),(0,1))
        self.assertEqual((draft.observations,draft.proposals),([],[]))
        assert_state(self,target,prompt,[first])

  def test_eos_inside_verified_followups_emits_no_extra_anchor(self):
    prompt = [2,5,3]
    expected = oracle(prompt,10)[0]
    eos = next(token for i,token in enumerate(expected[1:4],1) if token not in expected[:i])
    target,_,decoder,_ = fixture()
    output = list(decoder.generate(prompt,max_new_tokens=10,eos_token_ids=[eos]))
    self.assertEqual(output,expected[:expected.index(eos)+1])
    assert_state(self,target,prompt,output)
    self.assertEqual(decoder.finish_reason,'eos')

  def test_chunked_prefill_observes_older_features_before_reuse_and_defers_last_chunk(self):
    target,draft,decoder,_ = fixture()
    prompt = [2,5,3,9,1,7,4]
    stream = decoder.generate(prompt,max_new_tokens=7,prefill_chunk_size=2)
    first = next(stream)
    self.assertEqual(draft.observations,[0,2,4])
    self.assertEqual(draft.proposals,[])
    output = [first,*list(stream)]
    self.assertEqual(draft.observations[:4],[0,2,4,6])
    assert_state(self,target,prompt,output)

  def test_cleanup_failure_invalidates_cache_releases_ownership_and_allows_retry(self):
    target,_,decoder,_ = fixture()
    target.fail_once = True
    stream = decoder.generate([2,5,3],max_new_tokens=7)
    next(stream)
    with self.assertRaisesRegex(RuntimeError,'injected failure'):stream.close()
    self.assertEqual(target._cached_tokens,[])
    self.assertEqual(decoder.finish_reason,'error')
    self.assertEqual(decoder.stats.replay_calls,1)
    output = list(decoder.generate([9,3],max_new_tokens=7))
    assert_state(self,target,[9,3],output)

  def test_invalid_proposal_after_valid_anchor_keeps_valid_output_and_recovers(self):
    target,draft,decoder,_ = fixture()
    propose = draft.propose
    draft.propose = lambda *args,**kwargs:[17]
    stream = decoder.generate([2,5,3],max_new_tokens=7)
    first = next(stream)
    self.assertEqual(first,oracle([2,5,3],1)[0][0])
    with self.assertRaisesRegex(ValueError,'draft token'):list(stream)
    self.assertEqual(target._cached_tokens,[])
    self.assertEqual(decoder.stats.output_tokens,1)
    self.assertEqual(decoder.stats.replay_calls,1)
    draft.propose = propose
    output = list(decoder.generate([9,3],max_new_tokens=7))
    assert_state(self,target,[9,3],output)

  def test_eos_validation_checks_bool_before_deduplicating_in_both_orders(self):
    for ids in ([1,True],[True,1]):
      with self.subTest(ids=ids):
        target,_,decoder,events = fixture()
        with self.assertRaisesRegex(ValueError,'EOS token IDs'):list(decoder.generate([2,5,3],max_new_tokens=1,eos_token_ids=ids))
        self.assertEqual(events,[])
        self.assertEqual(target._cached_tokens,[])


class TestEarlyAnchorTransformer(unittest.TestCase):
  def test_hybrid_full_request_and_first_anchor_close_match_late_state_bits(self):
    raw,target = real_pair(True)
    prompt = [2,5,3]
    expected = ordinary_ids(raw,prompt,7)
    eager = LateDecoder(raw,FixedDraft(len(prompt),expected))
    early = GreedySpeculativeDecoder(target,FixedDraft(len(prompt),expected),early_anchor=True)
    self.assertEqual(list(early.generate(prompt,max_new_tokens=7)),list(eager.generate(prompt,max_new_tokens=7)))
    self.assertEqual(target._cached_tokens,prompt+expected)
    for actual,baseline in zip(target.blk,raw.blk):
      for name in ('conv_state','recurrent_state','cache_kv'):
        if hasattr(actual,name):np.testing.assert_array_equal(getattr(actual,name).numpy(),getattr(baseline,name).numpy())
    early_stream = early.generate(prompt,max_new_tokens=7)
    late_stream = eager.generate(prompt,max_new_tokens=7)
    self.assertEqual(next(early_stream),next(late_stream))
    early_stream.close()
    late_stream.close()
    self.assertEqual(target._cached_tokens,raw._cached_tokens)
    for actual,baseline in zip(target.blk,raw.blk):
      for name in ('conv_state','recurrent_state','cache_kv'):
        if hasattr(actual,name):
          av,bv = getattr(actual,name).numpy(),getattr(baseline,name).numpy()
          if name=='cache_kv':av,bv = av[:,:,:,:4],bv[:,:,:,:4]
          np.testing.assert_array_equal(av,bv)


if __name__=='__main__':unittest.main()
