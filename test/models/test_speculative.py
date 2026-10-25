"""Independent causal state-machine checks for greedy speculative coordination."""
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import SSMConfig, TargetOutput, Transformer, TransformerConfig
from tinygrad.llm.speculative import GreedySpeculativeDecoder


def transition(state, history, token, position):
  value = (state*3+token*5+history[0]*7+history[1]*11+position)%97
  history = (history[1],token)
  return value,history,(value+history[0]*3+history[1]+position*2)%17


def oracle(prompt, count):
  state,history,next_token = 0,(0,0),0
  for position,token in enumerate(prompt):state,history,next_token = transition(state,history,token,position)
  result = []
  for position in range(len(prompt),len(prompt)+count):
    result.append(next_token)
    state,history,next_token = transition(state,history,next_token,position)
  return result,state,history


class CausalTarget:
  def __init__(self, context=64):
    self.config = TransformerConfig(1,4,8,1,1,1e-5,17,4,10000.,4,4,max_context=context)
    self.max_context = context
    self._cached_tokens = []
    self.blk = [SimpleNamespace(recurrent_state=Tensor([0],device='CPU',dtype='int32').realize(),
                               conv_state=Tensor([[0,0]],device='CPU',dtype='int32').realize(),
                               cache_kv=Tensor.zeros(context,device='CPU',dtype='int32').contiguous().realize())]
    self.fail_once = False

  @staticmethod
  def prefill_chunk_size(requested):return requested

  def target_forward(self, tokens, start_pos, *, capture_layers=(), last_only=False):
    ids = tokens.tolist()[0]
    block = self.blk[0]
    state = 0 if start_pos==0 else int(block.recurrent_state.item())
    history = (0,0) if start_pos==0 else tuple(block.conv_state.tolist()[0])
    features,logits = [],[]
    cache = block.cache_kv.numpy().copy()
    for position,token in enumerate(ids,start_pos):
      state,history,prediction = transition(state,history,token,position)
      row = [-100.]*17
      row[prediction] = 100.
      logits.append(row)
      features.append([state,*history,position])
      cache[position] = token
    block.recurrent_state.assign(Tensor([state],device='CPU',dtype='int32')).realize()
    block.conv_state.assign(Tensor([list(history)],device='CPU',dtype='int32')).realize()
    block.cache_kv.assign(Tensor(cache,device='CPU')).realize()
    if self.fail_once and start_pos>=3:
      self.fail_once = False
      raise RuntimeError('injected failure after state write')
    hidden = Tensor([features],device='CPU',dtype='float32')
    return TargetOutput(Tensor([logits[-1:] if last_only else logits],device='CPU'),hidden,(hidden,) if capture_layers else ()).realize()


class OracleDraft:
  block_size,capture_layers = 4,(0,)
  def __init__(self, wrong_at=None):
    self.wrong_at = wrong_at
    self.seen = []

  def reset(self):self.seen = []

  def observe(self, output, start_pos):
    features = output.features[0].tolist()[0]
    if start_pos>len(self.seen):raise ValueError('gapped draft features')
    self.seen = self.seen[:start_pos]+features

  def propose(self, anchor, *, position, max_tokens):
    if len(self.seen)!=position:raise ValueError('stale draft feature prefix')
    state,h0,h1,_ = (int(v) for v in self.seen[-1])
    token,history = anchor,(h0,h1)
    result = []
    for slot in range(max_tokens-1):
      state,history,token = transition(state,history,token,position+slot)
      result.append(token)
    if self.wrong_at is not None and len(result)>self.wrong_at:result[self.wrong_at]=(result[self.wrong_at]+1)%17
    return result


class FixedDraft:
  """Proposals from an independently evaluated ordinary target, never the verifier."""
  block_size,capture_layers = 4,(0,)
  def __init__(self, prompt_len, predictions, wrong_at=None):
    self.prompt_len,self.predictions,self.wrong_at = prompt_len,predictions,wrong_at
    self.rows = 0
  def reset(self):self.rows = 0
  def observe(self, output, start_pos):
    if start_pos>self.rows:raise ValueError('gapped features')
    self.rows = start_pos+output.features[0].shape[1]
  def propose(self, anchor, *, position, max_tokens):
    if position!=self.rows:raise ValueError('stale features')
    index = position-self.prompt_len
    if anchor!=self.predictions[index]:raise ValueError('target diverged from independent greedy reference')
    result = list(self.predictions[index+1:index+max_tokens])
    if self.wrong_at is not None and self.wrong_at<len(result):result[self.wrong_at]=(result[self.wrong_at]+1)%13
    return result


def real_pair(hybrid):
  cfg = TransformerConfig(2,8,16,2,1,1e-5,13,4,10000.,4,4,max_context=24)
  if hybrid:cfg = replace(cfg,ssm=SSMConfig(3,4,1,1,8),ssm_layers=(True,False))
  reference,target = Transformer(cfg),Transformer(cfg)
  rng = np.random.default_rng(3491)
  for name,parameter in nn.state.get_state_dict(reference).items():
    values = rng.normal(0,.2,parameter.shape).astype(np.float32)
    if 'norm.weight' in name:values += 1
    if name.endswith('ssm_a'):values = -rng.uniform(.5,2,parameter.shape).astype(np.float32)
    parameter.replace(Tensor(values,device='CPU').realize())
  for parameter in nn.state.get_parameters(target):parameter.to_('CPU')
  nn.state.load_state_dict(target,{name:p.clone().realize() for name,p in nn.state.get_state_dict(reference).items()},verbose=False)
  return reference,target


def ordinary_ids(model, prompt, count):
  scores = model.logits(Tensor([prompt],device='CPU'),0).numpy()
  result = []
  for position in range(len(prompt),len(prompt)+count):
    token = int(scores.argmax(-1)[0])
    result.append(token)
    scores = model.logits(Tensor([[token]],device='CPU'),position).numpy()
  return result


class TestGreedySpeculative(unittest.TestCase):
  def make(self, *, wrong_at=None, context=64):
    target,draft = CausalTarget(context),OracleDraft(wrong_at)
    return target,draft,GreedySpeculativeDecoder(cast(Transformer,target),draft)

  def assert_state(self, target, prompt, output):
    expected,state,history = oracle(prompt,len(output))
    self.assertEqual(output,expected)
    self.assertEqual(target._cached_tokens,prompt+output)
    self.assertEqual(int(target.blk[0].recurrent_state.item()),state)
    self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]),history)
    np.testing.assert_array_equal(target.blk[0].cache_kv.numpy()[:len(prompt+output)],prompt+output)

  def test_all_accept_uses_one_verifier_per_block_and_copies_no_kv(self):
    target,draft,decoder = self.make()
    prompt = [2,5,3]
    output = list(decoder.generate(prompt,max_new_tokens=10,prefill_chunk_size=2))
    self.assert_state(target,prompt,output)
    self.assertEqual(decoder.stats.output_tokens,10)
    self.assertEqual(decoder.stats.blocks,3)
    self.assertEqual(decoder.stats.verify_calls,3)
    self.assertEqual(decoder.stats.replay_calls,0)
    self.assertEqual(decoder.stats.accepted_tokens,7)
    self.assertEqual(decoder.stats.snapshot_bytes,12)
    self.assertEqual(decoder.stats.target_calls,5)
    self.assertEqual(decoder.stats.target_input_tokens,13)
    self.assertEqual(draft.seen,[])

  def test_prompt_observation_timing_includes_deferred_prompt_but_excludes_decode(self):
    for early_anchor in (False,True):
      with self.subTest(early_anchor=early_anchor):
        target,draft = CausalTarget(),OracleDraft()
        decoder = GreedySpeculativeDecoder(cast(Transformer,target),draft,early_anchor=early_anchor)
        original = decoder._observe
        calls = []
        def observed(output, position, *, prompt=False):
          calls.append((position,prompt))
          with patch('tinygrad.llm.speculative.perf_counter',side_effect=(0.,3. if prompt else 7.)):
            original(output,position,prompt=prompt)
        decoder._observe = observed
        prompt = [2,5,3,1,4]
        output = list(decoder.generate(prompt,max_new_tokens=8,prefill_chunk_size=2))
        self.assertEqual(output,oracle(prompt,8)[0])
        self.assertEqual([position for position,is_prompt in calls if is_prompt],[0,2,4])
        self.assertEqual(decoder.stats.prompt_observe_seconds,9.)
        self.assertEqual(decoder.stats.observe_seconds,9.+7.*sum(not is_prompt for _,is_prompt in calls))
        self.assertTrue(any(not is_prompt for _,is_prompt in calls))

  def test_rejections_at_every_slot_match_independent_greedy(self):
    for wrong_at in range(3):
      with self.subTest(wrong_at=wrong_at):
        target,_,decoder = self.make(wrong_at=wrong_at)
        prompt = [9,1,7]
        output = list(decoder.generate(prompt,max_new_tokens=11))
        self.assert_state(target,prompt,output)
        self.assertGreater(decoder.stats.rejected_blocks,0)
        self.assertEqual(decoder.stats.replay_calls,decoder.stats.rejected_blocks)
        self.assertGreater(decoder.stats.target_input_tokens,len(prompt)+len(output))

  def test_close_midblock_replays_exactly_delivered_inputs_and_releases_lock(self):
    target,_,decoder = self.make()
    prompt = [2,5,3]
    stream = decoder.generate(prompt,max_new_tokens=10)
    output = [next(stream),next(stream)]
    stream.close()
    self.assert_state(target,prompt,output)
    self.assertEqual(decoder.finish_reason,'closed')
    self.assertEqual(decoder.stats.replay_calls,1)
    self.assertEqual(decoder.stats.output_tokens,2)
    again = list(decoder.generate([7,8],max_new_tokens=5))
    self.assert_state(target,[7,8],again)

  def test_cancel_midblock_and_before_prefill_leave_valid_state(self):
    target,_,decoder = self.make()
    prompt,flag = [2,5,3],{'stop':False}
    stream = decoder.generate(prompt,max_new_tokens=10,cancelled=lambda:flag['stop'])
    output = [next(stream)]
    flag['stop'] = True
    self.assertEqual(list(stream),[])
    self.assert_state(target,prompt,output)
    self.assertEqual(decoder.finish_reason,'cancelled')
    self.assertEqual(list(decoder.generate(prompt,max_new_tokens=10,cancelled=lambda:True)),[])
    self.assertEqual(target._cached_tokens,[])
    self.assertEqual(decoder.stats.target_calls,0)

  def test_eos_inside_accepted_block_and_context_limit(self):
    target,_,decoder = self.make()
    prompt = [2,5,3]
    expected,_,_ = oracle(prompt,10)
    eos = next(token for i,token in enumerate(expected[1:4],1) if token not in expected[:i])
    output = list(decoder.generate(prompt,max_new_tokens=10,eos_token_ids=[eos]))
    self.assertEqual(output,expected[:expected.index(eos)+1])
    self.assert_state(target,prompt,output)
    self.assertEqual(decoder.finish_reason,'eos')
    target,_,decoder = self.make(context=7)
    output = list(decoder.generate(prompt,max_new_tokens=10))
    self.assertEqual(len(output),4)
    self.assert_state(target,prompt,output)
    self.assertEqual(decoder.finish_reason,'context')

  def test_shared_target_iterators_exclude_each_other(self):
    target,_,decoder = self.make()
    other = GreedySpeculativeDecoder(cast(Transformer,target),OracleDraft())
    stream = decoder.generate([1,2,3],max_new_tokens=8)
    next(stream)
    with self.assertRaisesRegex(RuntimeError,'already used'):next(other.generate([4],max_new_tokens=2))
    stream.close()
    self.assertEqual(len(list(other.generate([4],max_new_tokens=2))),2)

  def test_invalid_inputs_and_draft_output_fail_without_reusing_partial_state(self):
    target,draft,decoder = self.make()
    for prompt,budget in (([],1),([True],1),([17],1),([1],-1),([1],True)):
      with self.assertRaises(ValueError):list(decoder.generate(prompt,max_new_tokens=budget))
    draft.propose = lambda *args,**kwargs:[17]
    with self.assertRaisesRegex(ValueError,'draft token'):list(decoder.generate([1,2],max_new_tokens=3))
    self.assertEqual(target._cached_tokens,[])

  def test_failure_after_mutable_state_write_invalidates_prefix_and_next_request_recovers(self):
    target,_,decoder = self.make()
    target.fail_once = True
    with self.assertRaisesRegex(RuntimeError,'injected failure'):list(decoder.generate([2,5,3],max_new_tokens=8))
    self.assertEqual(target._cached_tokens,[])
    self.assertEqual(decoder.finish_reason,'error')
    output = list(decoder.generate([9,3],max_new_tokens=8))
    self.assert_state(target,[9,3],output)

  def test_nonfinite_prefill_and_last_token_verifier_fail_before_observe_or_yield_and_retry(self):
    for mode in ('nan','inf','negative_inf','mixed_nan','mixed_inf'):
      for invalid_from in (0,3):
        with self.subTest(mode=mode,invalid_from=invalid_from):
          target,draft,decoder = self.make()
          original_forward,original_observe = target.target_forward,draft.observe
          observed = []
          def injected(tokens, start_pos, **kwargs):
            output = original_forward(tokens,start_pos,**kwargs)
            if start_pos<invalid_from:return output
            values = output.logits.numpy().copy()
            if mode in ('nan','inf','negative_inf'):values.fill({'nan':np.nan,'inf':np.inf,'negative_inf':-np.inf}[mode])
            else:values[0,0,0] = np.nan if mode=='mixed_nan' else np.inf
            return TargetOutput(Tensor(values,device='CPU').realize(),output.hidden,output.features)
          def observing(output, start_pos):
            observed.append(start_pos)
            original_observe(output,start_pos)
          target.target_forward,draft.observe = injected,observing
          with self.assertRaisesRegex(ValueError,'Nonfinite target logits'):
            list(decoder.generate([2,5,3],max_new_tokens=1))
          self.assertEqual(decoder.stats.output_tokens,0)
          self.assertEqual(observed,[] if invalid_from==0 else [0])
          self.assertEqual(target._cached_tokens,[])
          self.assertEqual(decoder.finish_reason,'error')
          target.target_forward = original_forward
          output = list(decoder.generate([9,3],max_new_tokens=3))
          self.assert_state(target,[9,3],output)

  def test_invalid_argmax_ids_dtype_and_shape_fail_closed(self):
    for value,dtype,shape in ((-1,'int32',(1,1)),(17,'int32',(1,1)),(1.25,'float32',(1,1)),
                              (True,'bool',(1,1)),(1,'int32',(1,2))):
      with self.subTest(value=value,dtype=dtype,shape=shape):
        target,_,decoder = self.make()
        with patch.object(Tensor,'argmax',lambda *args,**kwargs:Tensor.full(shape,value,device='CPU',dtype=dtype)):
          with self.assertRaisesRegex(ValueError,'target greedy IDs'):list(decoder.generate([2,5,3],max_new_tokens=1))
        self.assertEqual(decoder.stats.output_tokens,0)
        self.assertEqual(target._cached_tokens,[])
        output = list(decoder.generate([9,3],max_new_tokens=3))
        self.assert_state(target,[9,3],output)

  def test_zero_budget_preserves_existing_target_prefix(self):
    target,_,decoder = self.make()
    target._cached_tokens = [7,8]
    self.assertEqual(list(decoder.generate([1,2],max_new_tokens=0)),[])
    self.assertEqual(target._cached_tokens,[7,8])
    self.assertEqual(decoder.stats.target_calls,0)

  def test_unsupported_recurrent_family_rejected_before_any_work(self):
    target,draft,_ = self.make()
    target.config = replace(target.config,rwkv7=(1,))
    with self.assertRaises(NotImplementedError):GreedySpeculativeDecoder(cast(Transformer,target),draft)


class TestGreedySpeculativeTransformer(unittest.TestCase):
  def assert_model_state(self, target, reference, prefix_len):
    for actual,expected in zip(target.blk,reference.blk):
      for name in ('recurrent_state','conv_state','cache_kv'):
        if hasattr(actual,name):
          a,b = getattr(actual,name).numpy(),getattr(expected,name).numpy()
          if name=='cache_kv':a,b = a[:,:,:,:prefix_len],b[:,:,:,:prefix_len]
          np.testing.assert_allclose(a,b,atol=2e-5,rtol=3e-4)

  def test_attention_matches_ordinary_scalar_greedy_and_kv_after_rejection(self):
    prompt = [2,5,3]
    ordinary,target = real_pair(False)
    expected = ordinary_ids(ordinary,prompt,7)
    draft = FixedDraft(len(prompt),expected,wrong_at=1)
    decoder = GreedySpeculativeDecoder(target,draft)
    actual = list(decoder.generate(prompt,max_new_tokens=7))
    self.assertEqual(actual,expected)
    self.assertGreater(decoder.stats.rejected_blocks,0)
    self.assertEqual(target._cached_tokens,prompt+actual)
    self.assert_model_state(target,ordinary,len(prompt)+len(actual))

  def test_hybrid_all_accept_and_closed_stream_match_ordinary_state(self):
    prompt = [2,5,3]
    ordinary,target = real_pair(True)
    expected = ordinary_ids(ordinary,prompt,7)
    decoder = GreedySpeculativeDecoder(target,FixedDraft(len(prompt),expected))
    self.assertEqual(list(decoder.generate(prompt,max_new_tokens=7)),expected)
    self.assertEqual(decoder.stats.replay_calls,0)
    self.assert_model_state(target,ordinary,len(prompt)+len(expected))
    short_reference,_ = real_pair(True)
    short = ordinary_ids(short_reference,prompt,2)
    stream = decoder.generate(prompt,max_new_tokens=7)
    self.assertEqual([next(stream),next(stream)],short)
    stream.close()
    self.assertEqual(target._cached_tokens,prompt+short)
    self.assertEqual(decoder.stats.replay_calls,1)
    self.assert_model_state(target,short_reference,len(prompt)+len(short))


if __name__=='__main__':unittest.main()
