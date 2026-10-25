"""Independent causal-state tests for opt-in pending-input fallback delivery."""
import os, unittest
import numpy as np
from tinygrad import UOp
from tinygrad.llm.speculative import GreedySpeculativeDecoder
from test.models.test_speculative import CausalTarget, FixedDraft, OracleDraft, oracle, real_pair, ordinary_ids, transition
from test.models.test_speculative_sampled_state import NondegenerateCausalTarget, PointMassDraft


class TestDeferredFallback(unittest.TestCase):
  def make(self, context=64, *, deferred=True):
    target = CausalTarget(context)
    return target,GreedySpeculativeDecoder(target,OracleDraft(),deferred_fallback=deferred)

  def processed_state(self, target, prompt, output, *, pending):
    processed = prompt+(output[:-1] if pending else output)
    state,history = 0,(0,0)
    for position,token in enumerate(processed):state,history,_ = transition(state,history,token,position)
    self.assertEqual(target._cached_tokens,processed)
    self.assertEqual(int(target.blk[0].recurrent_state.item()),state)
    self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]),history)
    np.testing.assert_array_equal(target.blk[0].cache_kv.numpy()[:len(processed)],processed)

  @staticmethod
  def traced(target):
    calls=[]
    original=target.target_forward
    def run(tokens,position,**kwargs):
      calls.append((position,list(tokens.tolist()[0])))
      return original(tokens,position,**kwargs)
    target.target_forward=run
    return calls

  def first_fallback(self, decoder, prompt, **kwargs):
    stream=decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12,**kwargs)
    output=[]
    while not decoder.stats.fallback_tokens:output.append(next(stream))
    return stream,output

  def test_opt_in_saves_final_forward_and_fallback_snapshots_without_changing_ids(self):
    prompt=[2,5,3]
    results=[]
    for deferred in (False,True):
      target,decoder=self.make(deferred=deferred)
      output=list(decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12,cancelled=lambda:False))
      self.assertEqual(output,oracle(prompt,12)[0])
      self.assertEqual(decoder.stats.deferred_fallback,deferred)
      self.assertEqual(decoder.stats.fallback_pending_last_token,deferred)
      self.processed_state(target,prompt,output,pending=deferred)
      self.assertGreater(decoder.stats.fallback_tokens,0)
      self.assertEqual(decoder.stats.fallback_calls,decoder.stats.fallback_tokens-int(deferred))
      results.append(decoder.stats)
    self.assertEqual(results[0].target_calls,results[1].target_calls+1)
    self.assertEqual(results[0].snapshot_calls-results[1].snapshot_calls,results[0].fallback_tokens)

  def test_pending_token_is_processed_once_only_after_resume(self):
    prompt=[2,5,3]
    target,decoder=self.make()
    calls=self.traced(target)
    stream,output=self.first_fallback(decoder,prompt)
    try:
      position=len(prompt)+len(output)-1
      token=output[-1]
      self.assertTrue(decoder.stats.fallback_pending_last_token)
      self.processed_state(target,prompt,output,pending=True)
      self.assertFalse(any(start==position for start,_ in calls))
      output.append(next(stream))
      self.assertEqual(sum(start==position and ids==[token] for start,ids in calls),1)
      self.assertTrue(decoder.stats.fallback_pending_last_token)
      self.processed_state(target,prompt,output,pending=True)
      before=list(calls)
    finally:stream.close()
    self.assertEqual(calls,before)
    self.assertEqual(decoder.finish_reason,'closed')
    self.processed_state(target,prompt,output,pending=True)
    self.assertFalse(decoder._lock.locked())

  def test_budget_eos_and_context_skip_the_unused_successor(self):
    prompt=[2,5,3]
    expected=oracle(prompt,12)[0]
    eos_index=next(i for i in range(8,12) if expected[i] not in expected[:i])
    for mode in ('budget','eos','context'):
      with self.subTest(mode=mode):
        target,decoder=self.make(13 if mode=='context' else 64)
        calls=self.traced(target)
        options={'max_new_tokens':9 if mode=='budget' else 20 if mode=='context' else 12}
        if mode=='eos':options['eos_token_ids']=[expected[eos_index]]
        output=list(decoder.generate(prompt,ordinary_decode_seconds=1e-12,**options))
        self.assertEqual(output,expected[:len(output)])
        if mode=='eos':self.assertEqual(len(output),eos_index+1)
        self.assertTrue(decoder.stats.fallback_pending_last_token)
        self.processed_state(target,prompt,output,pending=True)
        self.assertFalse(any(start==len(prompt)+len(output)-1 for start,_ in calls))
        self.assertEqual(decoder.stats.fallback_calls,decoder.stats.fallback_tokens-1)
        self.assertEqual(decoder.finish_reason,{'budget':'length','eos':'eos','context':'context'}[mode])

  def test_cancel_before_resume_preserves_pending_prefix_without_target_work(self):
    prompt=[2,5,3]
    target,decoder=self.make()
    calls=self.traced(target)
    stop=[False]
    stream,output=self.first_fallback(decoder,prompt,cancelled=lambda:stop[0])
    before=list(calls)
    stop[0]=True
    self.assertEqual(list(stream),[])
    self.assertEqual(calls,before)
    self.assertEqual(decoder.finish_reason,'cancelled')
    self.assertTrue(decoder.stats.fallback_pending_last_token)
    self.processed_state(target,prompt,output,pending=True)
    retry=list(decoder.generate([9,3],max_new_tokens=4))
    self.assertEqual(retry,oracle([9,3],4)[0])
    self.processed_state(target,[9,3],retry,pending=False)

  def test_cancel_during_forward_commits_only_the_already_delivered_input(self):
    prompt=[2,5,3]
    target,decoder=self.make()
    stop=[False]
    original=target.target_forward
    def interrupt(tokens,position,**kwargs):
      output=original(tokens,position,**kwargs)
      if position>=len(prompt) and tokens.shape[1]==1:stop[0]=True
      return output
    target.target_forward=interrupt
    output=list(decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12,cancelled=lambda:stop[0]))
    self.assertEqual(output,oracle(prompt,len(output))[0])
    self.assertEqual(len(output),9)
    self.assertEqual(decoder.stats.fallback_tokens,1)
    self.assertEqual(decoder.stats.fallback_calls,1)
    self.assertEqual(decoder.finish_reason,'cancelled')
    self.assertFalse(decoder.stats.fallback_pending_last_token)
    self.processed_state(target,prompt,output,pending=False)

  def test_partial_forward_failure_invalidates_cache_then_retry_starts_fresh(self):
    prompt=[2,5,3]
    target,decoder=self.make()
    original=target.target_forward
    def fail_after_write(tokens,position,**kwargs):
      output=original(tokens,position,**kwargs)
      if position>=len(prompt) and tokens.shape[1]==1:raise RuntimeError('fallback failure after state write')
      return output
    target.target_forward=fail_after_write
    output=[]
    with self.assertRaisesRegex(RuntimeError,'after state write'):
      for token in decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12):output.append(token)
    self.assertEqual(output,oracle(prompt,len(output))[0])
    self.assertEqual(target._cached_tokens,[])
    self.assertFalse(decoder.stats.fallback_pending_last_token)
    self.assertEqual(decoder.finish_reason,'error')
    self.assertFalse(decoder._lock.locked())
    target.target_forward=original
    again=list(decoder.generate([9,3],max_new_tokens=4))
    self.processed_state(target,[9,3],again,pending=False)

  def test_default_and_constructor_guard_preserve_existing_contract(self):
    target=CausalTarget()
    decoder=GreedySpeculativeDecoder(target,OracleDraft())
    output=list(decoder.generate([2,5,3],max_new_tokens=12,ordinary_decode_seconds=1e-12))
    self.assertFalse(decoder.stats.deferred_fallback)
    self.assertFalse(decoder.stats.fallback_pending_last_token)
    self.processed_state(target,[2,5,3],output,pending=False)
    for invalid in (0,1,None,'yes'):
      with self.subTest(invalid=invalid):
        target=CausalTarget()
        draft=OracleDraft()
        with self.assertRaisesRegex(ValueError,'deferred_fallback'):
          GreedySpeculativeDecoder(target,draft,deferred_fallback=invalid)
        self.assertEqual(target._cached_tokens,[])
        self.assertEqual(draft.seen,[])
        self.assertEqual(int(target.blk[0].recurrent_state.item()),0)
        self.assertFalse(hasattr(target,'_speculative_lock'))

  def test_early_anchor_close_before_fallback_keeps_full_commit_contract(self):
    target=CausalTarget()
    decoder=GreedySpeculativeDecoder(target,OracleDraft(),early_anchor=True,deferred_fallback=True)
    stream=decoder.generate([2,5,3],max_new_tokens=12,ordinary_decode_seconds=1e-12)
    output=[next(stream)]
    stream.close()
    self.assertEqual(decoder.stats.fallback_tokens,0)
    self.assertFalse(decoder.stats.fallback_pending_last_token)
    self.processed_state(target,[2,5,3],output,pending=False)

  def test_sampled_seed_matches_legacy_and_keeps_correct_processed_state(self):
    prompt=[2,5,3]
    outputs=[]
    stats=[]
    for deferred in (False,True):
      target=NondegenerateCausalTarget()
      decoder=GreedySpeculativeDecoder(target,PointMassDraft(),deferred_fallback=deferred)
      output=list(decoder.sample(prompt,temperature=1.,seed=91,max_new_tokens=24,
        ordinary_decode_seconds=1e-12,cancelled=lambda:False))
      self.assertGreater(decoder.stats.fallback_tokens,0)
      self.assertEqual(decoder.stats.fallback_pending_last_token,deferred)
      self.processed_state(target,prompt,output,pending=deferred)
      outputs.append(output)
      stats.append(decoder.stats)
    self.assertEqual(outputs[0],outputs[1])
    self.assertEqual(stats[0].target_calls,stats[1].target_calls+1)
    self.assertEqual(stats[0].sampling_acceptance_tests,stats[1].sampling_acceptance_tests)
    self.assertEqual(stats[0].sampling_residual_draws,stats[1].sampling_residual_draws)

  def test_real_hybrid_ordinary_prefix_reuse_consumes_pending_input_once(self):
    reference,target=real_pair(hybrid=True)
    prompt=[1,2,3]
    expected=ordinary_ids(reference,prompt,10)
    decoder=GreedySpeculativeDecoder(target,FixedDraft(len(prompt),expected),deferred_fallback=True)
    output=list(decoder.generate(prompt,max_new_tokens=10,ordinary_decode_seconds=1e-12))
    self.assertEqual(output,expected)
    self.assertTrue(decoder.stats.fallback_pending_last_token)
    processed=prompt+output[:-1]
    self.assertEqual(target._cached_tokens,processed)
    followup=prompt+output+[2,1]
    self.assertEqual(target.get_start_pos(followup),len(processed))
    calls=[]
    def instrument(original):
      def traced(tokens,position):
        start=position.unbind()[1] if isinstance(position,UOp) else position
        rows=tokens.shape[1].unbind()[1] if isinstance(tokens.shape[1],UOp) else tokens.shape[1]
        calls.append((start,rows))
        return original(tokens,position)
      return traced
    for name in ('greedy_prefill_jit','greedy_rollout_jit'):setattr(target,name,instrument(getattr(target,name)))
    def take(model):
      stream=model.generate(followup.copy(),temperature=0.)
      try:return [next(stream) for _ in range(3)]
      finally:stream.close()
    fresh,_=real_pair(hybrid=True)
    self.assertEqual(take(target),take(fresh))
    self.assertEqual(calls[0][0],len(processed))
    self.assertEqual(calls[0][1],1)
    self.assertEqual(sum(start==len(processed) for start,_ in calls),1)
    self.assertEqual(target._cached_tokens,fresh._cached_tokens)
    for block,other in zip(target.blk,fresh.blk):
      for name in ('recurrent_state','conv_state'):
        if (value:=getattr(block,name,None)) is not None:
          np.testing.assert_allclose(value.numpy(),getattr(other,name).numpy(),atol=1e-6,rtol=1e-5)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestDeferredFallbackMali(unittest.TestCase):
  def run_case(self, *, cancel_during_forward=False):
    from test.models.test_speculative_prefix_trace import TestNativeGDNPrefixTraceMali, EightDraft
    from test.models.test_speculative_prepared import assert_states, production_greedy_ids
    from tinygrad.llm.kernels.mali import configure
    helper=TestNativeGDNPrefixTraceMali()
    with helper.models() as (raw,target):
      for model in (raw,target):
        configure(model,recurrent_scan=True,pack_weights=False,consistent_decode=True,native_state_snapshots=True)
      prompt=[1,2,3]
      expected=production_greedy_ids(raw,prompt,19)
      options={'prepared_target':True,'prepared_dynamic_rows':True,'prepared_prefill_dynamic':True,
        'prepared_exact_replay':True,'prefix_trace':True,'early_anchor':True}
      legacy=GreedySpeculativeDecoder(raw,EightDraft(len(prompt),expected),**options)
      decoder=GreedySpeculativeDecoder(target,EightDraft(len(prompt),expected),deferred_fallback=True,**options)
      stop=[False]
      forward=decoder._forward
      def interrupt(ids,position,phase,**kwargs):
        output=forward(ids,position,phase,**kwargs)
        if cancel_during_forward and phase=='fallback':stop[0]=True
        return output
      decoder._forward=interrupt
      actual=list(decoder.generate(prompt,max_new_tokens=19,ordinary_decode_seconds=1e-12,cancelled=lambda:stop[0]))
      self.assertEqual(actual,expected[:len(actual)])
      self.assertGreater(decoder.stats.fallback_tokens,0)
      pending=decoder.stats.fallback_pending_last_token
      self.assertEqual(pending,not cancel_during_forward)
      processed=actual[:-1] if pending else actual
      reference=list(legacy.generate(prompt,max_new_tokens=len(processed),ordinary_decode_seconds=1e-12,cancelled=lambda:False))
      self.assertEqual(reference,processed)
      self.assertEqual(target._cached_tokens,prompt+processed)
      self.assertEqual(target._cached_tokens,raw._cached_tokens)
      assert_states(self,target,raw,len(prompt)+len(processed))
      self.assertGreater(decoder.stats.native_snapshot_calls,0)
      self.assertEqual(decoder.stats.native_snapshot_calls,decoder.stats.snapshot_calls)
      self.assertEqual(decoder.finish_reason,'cancelled' if cancel_during_forward else 'length')

  def test_prepared_native_state_matches_legacy_processed_prefix(self):self.run_case()

  def test_native_cancel_during_fallback_retains_delivered_input(self):self.run_case(cancel_during_forward=True)


if __name__=='__main__':unittest.main()
