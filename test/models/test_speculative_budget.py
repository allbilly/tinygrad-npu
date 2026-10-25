"""Request routing must keep an unprofiled request on production ordinary generation."""
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import Mock
from tinygrad.llm.model import Transformer
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from tinygrad.llm.speculative_budget import MeasuredRequestBudgetPolicy, RequestBudgetRouter, RequestWorkload
from tinygrad.llm.speculative_policy import PolicyKey


class OrdinaryTarget:
  def __init__(self, context=64):
    self.config = SimpleNamespace(rwkv7=False,vocab_size=17)
    self.max_context,self._cached_tokens = context,[]
    self.generate_calls,self.forward_calls,self.closed = 0,0,0
    self.prefix_reuse,self.fail = 0,False
    self.blk = []

  def prefill_chunk_size(self, requested):return min(32,requested)
  def get_start_pos(self, prompt):return self.prefix_reuse
  def target_forward(self, *args, **kwargs):raise AssertionError('Ordinary fallback captured speculative target features')

  def generate(self, tokens, chunk_size=32, temperature=0.):
    self.generate_calls += 1
    self.last_options = chunk_size,temperature
    try:
      while len(tokens)<self.max_context:
        if self.fail:raise RuntimeError('injected ordinary backend failure')
        self.forward_calls += 1
        token = (sum(tokens)+len(tokens))%17
        self._cached_tokens = list(tokens)
        tokens.append(token)
        yield token
    finally:self.closed += 1


class ForbiddenDraft:
  block_size,capture_layers = 8,(0,)
  def reset(self):raise AssertionError('Ordinary fallback reset the draft')
  def observe(self, *args):raise AssertionError('Ordinary fallback observed target features')
  def propose(self, *args, **kwargs):raise AssertionError('Ordinary fallback executed the draft')


class TestRequestBudgetRouting(unittest.TestCase):
  def setUp(self):
    self.profile = PolicyKey('target-sha','draft-sha','CL-driver-kernels','fp16',1,(0,63))
    self.policy = MeasuredRequestBudgetPolicy((3,7))
    self.target = OrdinaryTarget()
    self.decoder = GreedySpeculativeDecoder(cast(Transformer,self.target),ForbiddenDraft())
    self.router = RequestBudgetRouter(self.decoder,self.policy,self.profile,workload='matched-heldout-chat')

  def request(self, output=16, prefix=0):
    return RequestWorkload(self.profile,'matched-heldout-chat',3,output,32,prefix)

  def add(self, request, budget, emitted, seconds, **kwargs):
    for _ in range(3):self.policy.record(request,budget,emitted,seconds,qualified=True,**kwargs)

  def test_unprofiled_calls_production_ordinary_without_extra_last_input(self):
    output = list(self.router.generate([2,5,3],max_new_tokens=5,prefill_chunk_size=128))
    self.assertEqual(len(output),5)
    self.assertEqual(self.target.generate_calls,1)
    self.assertEqual(self.target.forward_calls,5)
    self.assertEqual(self.target.last_options,(128,0.))
    self.assertEqual(self.target._cached_tokens,[2,5,3]+output[:-1])
    self.assertEqual(self.target.closed,1)
    self.assertEqual(self.router.stats.mode,'ordinary')
    self.assertEqual(self.router.stats.cache_contract,'ordinary_pending_last_token')
    self.assertEqual(self.decoder.stats.target_calls,0)

  def test_complete_request_cost_includes_feature_prefill_and_can_refuse_speculation(self):
    request = self.request()
    self.add(request,0,16,1.)
    # A fast verifier is insufficient when feature-prefill/observe/reset/replay makes the request slower.
    self.add(request,7,16,1.2)
    self.assertEqual(self.policy.choose(request).budget,0)
    self.policy.invalidate(request,7)
    self.add(request,7,16,.7)
    self.assertEqual(self.policy.choose(request).budget,7)
    self.assertEqual(self.policy.choose(replace(request,output_limit=15)).budget,0)
    self.assertEqual(self.policy.choose(replace(request,reusable_prefix_tokens=1)).budget,0)
    self.assertEqual(self.policy.choose(replace(request,workload='different-load')).budget,0)

  def test_qualified_request_passes_only_verification_budget_and_leaves_draft_shape_alone(self):
    request = self.request()
    self.add(request,0,16,1.)
    self.add(request,3,16,.7)
    def generated(*args, **kwargs):
      self.assertEqual(kwargs['verification_budget'],3)
      self.assertEqual(self.decoder.draft.block_size,8)
      yield 4
      yield 5
      self.decoder.finish_reason = 'eos'
    self.decoder.generate = Mock(side_effect=generated)
    output = list(self.router.generate([2,5,3],max_new_tokens=16))
    self.assertEqual(output,[4,5])
    self.assertEqual(self.target.generate_calls,0)
    self.assertEqual(self.router.stats.mode,'speculative')
    self.assertEqual(self.router.stats.finish_reason,'eos')

  def test_eos_identity_is_canonical_and_never_reuses_another_completion_cohort(self):
    request = self.request()
    self.add(request,0,16,1.)
    self.add(request,3,16,.7)
    self.assertEqual(self.policy.choose(request).budget,3)
    eos_request = replace(request,eos_token_ids=(2,5))
    self.assertEqual(self.policy.choose(eos_request).budget,0)
    # Same prompt and output limit with an EOS set cannot use a fixed-length, empty-EOS qualification.
    list(self.router.generate([2,5,3],max_new_tokens=16,eos_token_ids=[5,2,5]))
    self.assertEqual(self.router.stats.mode,'ordinary')
    self.assertEqual(self.router.stats.request.eos_token_ids,(2,5))
    self.assertEqual(self.target.generate_calls,1)
    self.add(eos_request,0,2,1.,finish_reason='eos')
    self.add(eos_request,3,2,.7,finish_reason='eos')
    def generated(*args, **kwargs):
      self.assertEqual(kwargs['eos_token_ids'],{2,5})
      yield 5
      self.decoder.finish_reason = 'eos'
    self.decoder.generate = Mock(side_effect=generated)
    self.assertEqual(list(self.router.generate([2,5,3],max_new_tokens=16,eos_token_ids=[2,5,2])),[5])
    self.assertEqual(self.router.stats.mode,'speculative')
    self.assertEqual(self.router.stats.request,eos_request)
    for values in ([1,True],[True,1]):
      with self.assertRaises(ValueError):list(self.router.generate([2,5,3],max_new_tokens=16,eos_token_ids=values))
    for values in ((5,2),(2,2),(True,)):
      with self.assertRaises(ValueError):replace(request,eos_token_ids=values)

  def test_close_cancel_eos_and_context_preserve_normal_generator_contract(self):
    stream = self.router.generate([2,5,3],max_new_tokens=10)
    first = next(stream)
    stream.close()
    self.assertEqual(self.target.forward_calls,1)
    self.assertEqual(self.target._cached_tokens,[2,5,3])
    self.assertEqual(self.router.stats.finish_reason,'closed')
    stream = self.router.generate([2,5,3],max_new_tokens=10,cancelled=lambda:self.target.forward_calls>=2)
    self.assertEqual(list(stream),[first])
    self.assertEqual(self.target.forward_calls,2)
    self.assertEqual(self.router.stats.finish_reason,'cancelled')
    self.assertEqual(list(self.router.generate([2,5,3],max_new_tokens=10,eos_token_ids=[first])),[first])
    self.assertEqual(self.router.stats.finish_reason,'eos')
    self.assertEqual(len(list(self.router.generate([2]*62,max_new_tokens=10))),2)
    self.assertEqual(self.router.stats.finish_reason,'context')

  def test_lock_excludes_shared_target_and_does_not_replace_active_metadata(self):
    stream = self.router.generate([2,5,3],max_new_tokens=10)
    next(stream)
    active_stats = self.router.stats
    with self.assertRaisesRegex(RuntimeError,'active iterator'):next(self.router.generate([2],max_new_tokens=1))
    self.assertIs(self.router.stats,active_stats)
    other = RequestBudgetRouter(self.decoder,self.policy,self.profile,workload='matched-heldout-chat')
    with self.assertRaisesRegex(RuntimeError,'already used'):next(other.generate([2],max_new_tokens=1))
    stream.close()
    self.assertEqual(len(list(other.generate([2],max_new_tokens=1))),1)

  def test_ordinary_failure_invalidates_its_baseline_and_releases_ownership(self):
    request = self.request(output=16)
    self.add(request,0,16,1.)
    self.target.fail = True
    with self.assertRaisesRegex(RuntimeError,'injected ordinary'):list(self.router.generate([2,5,3],max_new_tokens=16))
    self.assertEqual(self.policy.choose(request).reason,'unqualified_request')
    self.target.fail = False
    self.assertEqual(len(list(self.router.generate([2,5,3],max_new_tokens=16))),16)

  def test_ordinary_close_failure_invalidates_baseline_and_releases_both_locks(self):
    request = self.request()
    self.add(request,0,16,1.)
    ordinary_generate = self.target.generate
    def broken_close(*args, **kwargs):
      try:yield from ordinary_generate(*args,**kwargs)
      finally:raise KeyboardInterrupt('injected ordinary close failure')
    self.target.generate = broken_close
    stream = self.router.generate([2,5,3],max_new_tokens=16)
    next(stream)
    with self.assertRaisesRegex(KeyboardInterrupt,'ordinary close failure'):stream.close()
    self.assertEqual(self.router.stats.finish_reason,'error')
    self.assertGreater(self.router.stats.total_seconds,0.)
    self.assertEqual(self.policy.choose(request).reason,'unqualified_request')
    self.assertFalse(self.decoder._lock.locked())
    self.assertFalse(self.router._routing_lock.locked())
    self.assertEqual(self.target.closed,1)
    self.target.generate = ordinary_generate
    self.assertEqual(len(list(self.router.generate([2,5,3],max_new_tokens=16))),16)
    self.assertEqual(self.router.stats.mode,'ordinary')
    self.assertEqual(self.router.stats.finish_reason,'length')
    self.assertEqual(self.policy.choose(request).reason,'unqualified_request')

  def test_warmup_cancelled_unqualified_and_other_context_cannot_enable(self):
    request = self.request()
    self.add(request,0,16,1.)
    self.add(request,7,16,.1,warmup=True)
    self.add(request,7,16,.1,finish_reason='cancelled')
    self.assertEqual(self.policy.choose(request).budget,0)
    for _ in range(3):self.policy.record(request,7,16,.1)
    self.assertEqual(self.policy.choose(request).budget,0)
    self.assertEqual(self.policy.choose(replace(request,prompt_tokens=60)).reason,'outside_measured_request_context')

  def test_positive_coordinator_budget_caps_verified_rows_and_zero_is_refused(self):
    from test.models.test_speculative import CausalTarget, OracleDraft, oracle
    target,draft = CausalTarget(),OracleDraft()
    decoder = GreedySpeculativeDecoder(cast(Transformer,target),draft)
    expected,_,_ = oracle([2,5,3],9)
    output = list(decoder.generate([2,5,3],max_new_tokens=9,verification_budget=1))
    self.assertEqual(output,expected)
    self.assertEqual(decoder.stats.blocks,5)
    self.assertEqual(decoder.stats.proposed_tokens,4)
    self.assertEqual(target._cached_tokens,[2,5,3]+output)
    for budget in (0,-1,True,4):
      with self.assertRaises(ValueError):list(decoder.generate([2,5,3],max_new_tokens=9,verification_budget=budget))


class TestRealCPURequestBudgetRouting(unittest.TestCase):
  def make(self, target, draft, budgets=(1,3)):
    profile = PolicyKey('tiny-real-hybrid','independent-fixture-draft','CPU-prepared-exact','fp32',1,(0,target.max_context-1))
    policy = MeasuredRequestBudgetPolicy(budgets)
    decoder = GreedySpeculativeDecoder(target,draft,prepared_target=True)
    return decoder,policy,RequestBudgetRouter(decoder,policy,profile,workload='cpu-greedy-state-contract')

  def test_qualified_early_anchor_close_failure_invalidates_budget_and_retry_is_ordinary(self):
    from test.models.test_speculative import FixedDraft, ordinary_ids, real_pair
    reference,target = real_pair(True)
    prompt = [2,5,3]
    expected = ordinary_ids(reference,prompt,4)
    draft = FixedDraft(len(prompt),expected)
    decoder = GreedySpeculativeDecoder(target,draft,early_anchor=True)
    profile = PolicyKey('tiny-real-hybrid','fixture-draft','CPU-early-anchor-cleanup','fp32',1,(0,target.max_context-1))
    policy = MeasuredRequestBudgetPolicy((3,))
    router = RequestBudgetRouter(decoder,policy,profile,workload='close-state-failure')
    request = RequestWorkload(profile,router.workload,len(prompt),4,target.prefill_chunk_size(2),0)
    for _ in range(3):
      policy.record(request,0,4,2.,qualified=True)
      policy.record(request,3,4,1.,qualified=True)
    original_forward = target.target_forward
    forward_positions = []
    fail_once = {'value':True}
    def pending_commit_fails(tokens, start_pos, **kwargs):
      output = original_forward(tokens,start_pos,**kwargs)
      forward_positions.append(start_pos)
      if start_pos>=len(prompt) and fail_once['value']:
        fail_once['value'] = False
        raise RuntimeError('injected pending-anchor cleanup failure after state write')
      return output
    target.target_forward = pending_commit_fails
    stream = router.generate(prompt,max_new_tokens=4,prefill_chunk_size=2)
    self.assertEqual(next(stream),expected[0])
    with self.assertRaisesRegex(RuntimeError,'pending-anchor cleanup failure'):stream.close()
    self.assertEqual(router.stats.finish_reason,'error')
    self.assertEqual(decoder.finish_reason,'error')
    self.assertIs(router.stats.speculative,decoder.stats)
    self.assertGreater(router.stats.total_seconds,0.)
    self.assertGreater(decoder.stats.replay_seconds,0.)
    self.assertEqual(decoder.stats.replay_calls,1)
    self.assertEqual(target._cached_tokens,[])
    self.assertEqual(policy.choose(request).budget,0)
    self.assertFalse(decoder._lock.locked())
    self.assertFalse(router._routing_lock.locked())
    calls_before_retry = list(forward_positions)
    self.assertEqual(list(router.generate(prompt,max_new_tokens=4,prefill_chunk_size=2)),expected)
    self.assertEqual(router.stats.mode,'ordinary')
    self.assertEqual(target._cached_tokens,prompt+expected[:-1])
    self.assertEqual(forward_positions,calls_before_retry)
    self.assertEqual(policy.choose(request).budget,0)

  def test_real_hybrid_ordinary_close_cancel_eos_and_prefix_reuse_never_touches_draft(self):
    from test.models.test_speculative import ordinary_ids, real_pair
    reference,target = real_pair(True)
    decoder,_,router = self.make(target,ForbiddenDraft())
    # Exercise the production generator and its own JITs. A fallback through the feature API must fail.
    target.generate = Mock(wraps=target.generate)
    target.target_forward = Mock(side_effect=AssertionError('Ordinary fallback used the target feature API'))
    prompt = [2,5,3]
    expected = ordinary_ids(reference,prompt,6)
    stream = router.generate(prompt,max_new_tokens=6,prefill_chunk_size=2)
    actual = [next(stream),next(stream)]
    stream.close()
    self.assertEqual(actual,expected[:2])
    self.assertEqual(target._cached_tokens,prompt+actual[:-1])
    self.assertEqual(router.stats.finish_reason,'closed')

    prompt = [9,4]
    expected = ordinary_ids(reference,prompt,4)
    flag = {'stop':False}
    stream = router.generate(prompt,max_new_tokens=4,prefill_chunk_size=2,cancelled=lambda:flag['stop'])
    self.assertEqual(next(stream),expected[0])
    flag['stop'] = True
    self.assertEqual(list(stream),[])
    self.assertEqual(target._cached_tokens,prompt)
    self.assertEqual(router.stats.finish_reason,'cancelled')

    prompt = [6,1,8]
    eos = ordinary_ids(reference,prompt,4)[0]
    self.assertEqual(list(router.generate(prompt,max_new_tokens=4,prefill_chunk_size=2,eos_token_ids=[eos])),[eos])
    self.assertEqual(target._cached_tokens,prompt)
    self.assertEqual(router.stats.finish_reason,'eos')
    continuation = prompt+[eos,4]
    reused = target.get_start_pos(continuation)
    self.assertEqual(reused,len(prompt))
    expected = ordinary_ids(reference,continuation,4)
    actual = list(router.generate(continuation,max_new_tokens=4,prefill_chunk_size=2))
    self.assertEqual(actual,expected)
    self.assertEqual(router.stats.request.reusable_prefix_tokens,reused)
    self.assertEqual(target._cached_tokens,continuation+actual[:-1])
    self.assertEqual(router.stats.cache_contract,'ordinary_pending_last_token')

    cached = list(target._cached_tokens)
    self.assertEqual(list(router.generate([1,2],max_new_tokens=4,cancelled=lambda:True)),[])
    self.assertEqual(target._cached_tokens,cached)
    self.assertEqual(router.stats.finish_reason,'cancelled')
    self.assertEqual(target.generate.call_count,5)
    target.target_forward.assert_not_called()
    self.assertEqual(decoder.stats.target_calls,0)
    self.assertEqual(decoder.stats.prepared_graphs,0)
    self.assertIsNotNone(target.greedy_rollout_jit.captured)

  def test_real_hybrid_ordinary_prepared_speculative_then_ordinary_reuse_and_fresh_requests(self):
    from test.models.test_speculative import FixedDraft, ordinary_ids, real_pair
    reference,target = real_pair(True)
    draft = FixedDraft(3,[])
    calls = {name:0 for name in ('reset','observe','propose')}
    armed = {'value':False}
    def guard(name, original):
      def call(*args, **kwargs):
        if not armed['value']:raise AssertionError('Unprofiled ordinary request called draft.'+name)
        calls[name] += 1
        return original(*args,**kwargs)
      return call
    for name in calls:setattr(draft,name,guard(name,getattr(draft,name)))
    decoder,policy,router = self.make(target,draft)

    prompt = [2,5,3]
    expected = ordinary_ids(reference,prompt,6)
    stream = router.generate(prompt,max_new_tokens=6,prefill_chunk_size=2)
    self.assertEqual([next(stream),next(stream)],expected[:2])
    stream.close()
    self.assertEqual(calls,dict.fromkeys(calls,0))

    prompt = [9,1,7]
    expected = ordinary_ids(reference,prompt,6)
    draft.prompt_len,draft.predictions = len(prompt),expected
    request = RequestWorkload(router.profile,router.workload,len(prompt),6,target.prefill_chunk_size(2),target.get_start_pos(prompt))
    # Synthetic timings test routing only; real output/state checks below do not claim a measured speedup.
    for _ in range(3):
      policy.record(request,0,6,2.,qualified=True)
      policy.record(request,3,6,1.,qualified=True)
    armed['value'] = True
    actual = list(router.generate(prompt,max_new_tokens=6,prefill_chunk_size=2))
    self.assertEqual(actual,expected)
    self.assertEqual(router.stats.mode,'speculative')
    self.assertEqual(target._cached_tokens,prompt+actual)
    self.assertTrue(decoder.stats.prepared_target)
    self.assertGreater(decoder.stats.prepared_graphs,0)
    self.assertGreater(calls['propose'],0)
    armed['value'] = False
    previous = dict(calls)

    continuation = list(target._cached_tokens)+[2]
    reused = target.get_start_pos(continuation)
    self.assertEqual(reused,len(continuation)-1)
    expected = ordinary_ids(reference,continuation,4)
    actual = list(router.generate(continuation,max_new_tokens=4,prefill_chunk_size=2))
    self.assertEqual(actual,expected)
    self.assertEqual(router.stats.mode,'ordinary')
    self.assertEqual(router.stats.request.reusable_prefix_tokens,reused)
    self.assertEqual(target._cached_tokens,continuation+actual[:-1])
    self.assertEqual(calls,previous)

    prompt = [4,8,2,6]
    expected = ordinary_ids(reference,prompt,3)
    self.assertEqual(list(router.generate(prompt,max_new_tokens=3,prefill_chunk_size=2)),expected)
    self.assertEqual(target._cached_tokens,prompt+expected[:-1])
    self.assertEqual(calls,previous)

    # Reuse the prepared verifier graphs on a fresh, different prompt after ordinary prefix reuse.
    prompt = [1,3,8]
    expected = ordinary_ids(reference,prompt,6)
    draft.prompt_len,draft.predictions = len(prompt),expected
    armed['value'] = True
    self.assertEqual(list(router.generate(prompt,max_new_tokens=6,prefill_chunk_size=2)),expected)
    self.assertEqual(router.stats.mode,'speculative')
    self.assertEqual(target._cached_tokens,prompt+expected)
    armed['value'] = False
    previous = dict(calls)

    prompt = [3,6]
    expected = ordinary_ids(reference,prompt,2)
    self.assertEqual(list(router.generate(prompt,max_new_tokens=2,prefill_chunk_size=2)),expected)
    self.assertEqual(router.stats.mode,'ordinary')
    self.assertEqual(target._cached_tokens,prompt+expected[:-1])
    self.assertEqual(calls,previous)


if __name__=='__main__':unittest.main()
