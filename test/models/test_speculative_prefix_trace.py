"""Explicit native-GDN prefix restoration; CPU injection tests protocol, not hardware speed."""
from contextlib import contextmanager
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, TargetOutput, Transformer, TransformerConfig
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _PreparedTargetForward, _map_outputs
from tinygrad.llm.speculative_trace import NativeGDNPrefixTrace
from test.models.test_speculative import FixedDraft, real_pair
from test.models.test_speculative_prepared import assert_states, production_greedy_ids
from test.models.test_speculative_target import pair


def cpu_scan(q,k,v,beta,alpha,state,start_pos,rows,*,history=None):
  """Independent functional CPU scan injected only under an explicit test patch."""
  current = Tensor(start_pos,device=q.device).eq(0).where(0,state.float())
  q,k,v,beta = q.unsqueeze(-2),k.unsqueeze(-2),v.unsqueeze(-1),beta.unsqueeze(-1).unsqueeze(-1)
  if alpha.ndim==3:alpha=alpha.unsqueeze(-1)
  alpha = alpha.unsqueeze(-2)
  outputs,states = [],[]
  for index in range(q.max_shape[2]):
    decayed = current*alpha[:,:,index]
    delta = (v[:,:,index]-(decayed*k[:,:,index]).sum(-1,keepdim=True))*beta[:,:,index]
    updated = decayed+delta*k[:,:,index]
    valid = Tensor(rows,device=q.device)>index
    current = valid.where(updated,current)
    outputs.append(valid.where((current*q[:,:,index]).sum(-1),0))
    states.append(current)
  stores = [state.uop.store(current.uop)]
  if history is not None:stores.append(history.uop.store(states[0].stack(*states[1:],dim=0).pad_to(history.shape).uop))
  return Tensor(outputs[0].stack(*outputs[1:],dim=2).contiguous().uop.after(*stores))


class EightDraft(FixedDraft):
  block_size = 8


def eager_block(model, ids, position, capacity=8, layers=(0,2)):
  capacity = min(capacity,model.max_context-position)
  tokens = Tensor([list(ids)+[0]*(capacity-len(ids))],device='CPU',dtype='int32').realize()
  pos = UOp.variable('trace_reference_pos',0,model.max_context-capacity).bind(position)
  rows = UOp.variable('trace_reference_rows',1,capacity).bind(len(ids))
  output = model.target_forward(tokens[:,:rows],pos,capture_layers=layers)
  return _map_outputs(output,lambda value:value.pad_to(value.max_shape)[:,:len(ids)]).realize()


class TestNativeGDNPrefixTrace(unittest.TestCase):
  device = 'CPU'

  @contextmanager
  def models(self):
    cfg = TransformerConfig(3,8,16,2,1,1e-5,13,4,10000.,4,4,max_context=24,
                            ssm=SSMConfig(4,4,1,1,8),ssm_layers=(True,False,True))
    raw,target = pair(cfg,self.device,half=self.device=='CL')
    if self.device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for model in (raw,target):configure(model,recurrent_scan=True,pack_weights=False)
      yield raw,target
    else:
      for model in (raw,target):
        for block in model.blk:
          if isinstance(block,GatedDeltaNetBlock):block.gated_delta_scan=cpu_scan
      with patch('tinygrad.llm.kernels.mali_recurrent.gated_delta_scan',new=cpu_scan), \
           patch('tinygrad.llm.kernels.mali_recurrent.supported',return_value=True):yield raw,target

  def test_every_prefix_conv_window_and_output_match_identically_bounded_eager_and_survive_scratch_release(self):
    with self.models() as (raw,target):
      trace = NativeGDNPrefixTrace(target,8)
      trace.prepare()
      identities = tuple(id(value) for value in trace.buffers)
      prepared = _PreparedTargetForward(target,(0,2),dynamic_rows=True,state_trace=trace)
      ids = [2,5,3,9,1,7,4,11]
      for rows in range(1,9):
        expected = eager_block(raw,ids[:rows],0)
        actual = prepared(ids,0,'verify',8,False)
        trace.mark_valid(0,8)
        if rows>=3:prepared.free_intermediates()
        _ = trace.restore(0,rows)
        for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
          np.testing.assert_array_equal(a.numpy()[:,:rows].view(np.uint32),b.numpy().view(np.uint32))
        assert_states(self,target,raw,rows)
        saved = [(getattr(block,name),getattr(block,name).numpy().copy()) for block in target.blk
                 for name in ('conv_state','recurrent_state') if hasattr(block,name)]
        actual.logits.numpy()
        actual.hidden.numpy()
        for value,snapshot in saved:np.testing.assert_array_equal(value.numpy().view(np.uint32),snapshot.view(np.uint32))
        layer_trace = trace.bind()[0]
        np.testing.assert_array_equal(layer_trace.state_history.numpy()[rows-1].view(np.uint32),raw.blk[0].recurrent_state.numpy().view(np.uint32))
        np.testing.assert_array_equal(layer_trace.conv_window.numpy()[:,rows:rows+3].view(np.uint32),raw.blk[0].conv_state.numpy().view(np.uint32))
        self.assertIs(actual.hidden,actual.features[-1])
        self.assertEqual(tuple(id(value) for value in trace.buffers),identities)
      self.assertEqual(len(prepared.calls),1)
      self.assertGreaterEqual(prepared.calls[('verify',8,False)].jit.cnt,3)
      self.assertIs(prepared.calls[('verify',8,False)].state_trace,trace)

  def test_mixed_rows_positions_repeated_reset_and_all_graphs_share_explicit_workspace(self):
    with self.models() as (raw,target):
      trace = NativeGDNPrefixTrace(target,8)
      trace.prepare()
      prepared = _PreparedTargetForward(target,(0,2),dynamic_rows=True,state_trace=trace)
      identities = tuple(id(value) for value in trace.buffers)
      for rows,position in ((8,0),(7,8),(5,15),(4,0),(1,4),(8,5),(3,0),(8,3),(5,11),(8,16),(5,0),(8,5),(6,13),(3,19),(2,22)):
        ids = [(position+i*3+1)%13 for i in range(rows)]
        expected = eager_block(raw,ids,position)
        actual = prepared(ids,position,'verify',8,False)
        for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
          np.testing.assert_array_equal(a.numpy().view(np.uint32),b.numpy().view(np.uint32))
        assert_states(self,target,raw,position+rows)
        trace.mark_valid(position,rows)
        self.assertTrue(trace.can_restore(position,rows))
        self.assertFalse(trace.can_restore(position,rows+1))
        self.assertFalse(trace.can_restore(position+1,rows))
        self.assertEqual(tuple(id(value) for value in trace.buffers),identities)
        prepared.free_intermediates()
      self.assertEqual(set(prepared.calls),{('verify',8,False),('verify',5,False),('verify',2,False)})
      self.assertTrue(all(call.state_trace is trace for call in prepared.calls.values()))
      captured = prepared.calls[('verify',8,False)].jit.captured
      # Tokens and both history buffers are explicit JIT parameters, never scratch.
      self.assertEqual(len(captured.expected_input_info),5)
      self.assertEqual(tuple(info[-1] for info in captured.expected_input_info),('CPU',self.device,self.device,self.device,self.device))

  def test_rejection_all_accept_eos_close_cancel_retry_match_independent_ordinary_ids_and_prefix_state(self):
    with self.models() as (raw,target):
      prompt = [2,5,3]
      expected = production_greedy_ids(raw,prompt,10)
      raw_draft,draft = EightDraft(len(prompt),expected),EightDraft(len(prompt),expected)
      baseline = GreedySpeculativeDecoder(raw,raw_draft,prepared_target=True,prepared_dynamic_rows=True,prepared_prefill_dynamic=True)
      decoder = GreedySpeculativeDecoder(target,draft,prepared_target=True,prepared_dynamic_rows=True,prepared_prefill_dynamic=True,prefix_trace=True)
      for wrong_at in (None,0,1,4,None):
        raw_draft.wrong_at=draft.wrong_at=wrong_at
        actual = list(decoder.generate(prompt,max_new_tokens=10,prefill_chunk_size=4))
        reference = list(baseline.generate(prompt,max_new_tokens=10,prefill_chunk_size=4))
        self.assertEqual(actual,expected)
        self.assertEqual(actual,reference)
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        assert_states(self,target,raw,len(prompt)+len(actual))
        self.assertEqual(decoder.stats.replay_calls,0)
        self.assertEqual(decoder.stats.trace_record_calls,decoder.stats.verify_calls)
        self.assertEqual(decoder.stats.trace_restore_calls,decoder.stats.rejected_blocks)
        self.assertGreater(decoder.stats.trace_bytes,0)
        self.assertGreater(decoder.stats.trace_recorded_bytes,0)
      raw_draft.wrong_at=draft.wrong_at=None
      for delivered,cancelled in ((1,False),(2,False),(3,False),(2,True)):
        flag = {'stop':False}
        actual = decoder.generate(prompt,max_new_tokens=10,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
        reference = baseline.generate(prompt,max_new_tokens=10,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
        self.assertEqual([next(actual) for _ in range(delivered)],expected[:delivered])
        self.assertEqual([next(reference) for _ in range(delivered)],expected[:delivered])
        if cancelled:
          flag['stop']=True
          self.assertEqual(list(actual),[])
          self.assertEqual(list(reference),[])
        else:
          actual.close()
          reference.close()
        self.assertEqual(target._cached_tokens,prompt+expected[:delivered])
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        assert_states(self,target,raw,len(prompt)+delivered)
        self.assertEqual(decoder.stats.replay_calls,0)
        self.assertGreater(decoder.stats.trace_restore_calls,0)
      eos = expected[2]
      self.assertEqual(list(decoder.generate(prompt,max_new_tokens=10,eos_token_ids=(eos,),prefill_chunk_size=4)),
                       list(baseline.generate(prompt,max_new_tokens=10,eos_token_ids=(eos,),prefill_chunk_size=4)))
      self.assertEqual(decoder.finish_reason,'eos')
      self.assertEqual(target._cached_tokens,raw._cached_tokens)
      assert_states(self,target,raw,len(target._cached_tokens))

  def test_zero_delivered_cancel_after_verify_restores_snapshot_without_reading_an_invalid_trace_row(self):
    with self.models() as (raw,target):
      prompt = [2,5,3]
      expected = production_greedy_ids(raw,prompt,9)
      decoder = GreedySpeculativeDecoder(target,EightDraft(len(prompt),expected),prepared_target=True,
                                        prepared_dynamic_rows=True,prepared_prefill_dynamic=True,prefix_trace=True)
      before = []
      flag = {'stop':False}
      forward = decoder._forward
      def cancelled_after_verify(ids,position,phase,*,last_only=False):
        if phase=='verify':
          for block in target.blk:
            for name in ('conv_state','recurrent_state'):
              if hasattr(block,name):
                state = getattr(block,name)
                before.append((state,state.numpy().copy()))
        result = forward(ids,position,phase,last_only=last_only)
        if phase=='verify':flag['stop']=True
        return result
      with patch.object(decoder,'_forward',new=cancelled_after_verify):
        self.assertEqual(list(decoder.generate(prompt,max_new_tokens=9,prefill_chunk_size=4,cancelled=lambda:flag['stop'])),[])
      self.assertEqual(decoder.finish_reason,'cancelled')
      self.assertEqual(target._cached_tokens,prompt)
      self.assertEqual(decoder.stats.trace_restore_calls,0)
      self.assertEqual(decoder.stats.rollback_count,1)
      self.assertEqual(decoder.stats.replay_calls,0)
      self.assertGreater(len(before),2)
      for state,saved in before:np.testing.assert_array_equal(state.numpy().view(np.uint32),saved.view(np.uint32))
      self.assertEqual(list(decoder.generate(prompt,max_new_tokens=9,prefill_chunk_size=4)),expected)

  def test_early_unverified_anchor_close_uses_replay_and_nonfinite_verify_uses_snapshot_then_retry(self):
    with self.models() as (raw,target):
      prompt = [2,5,3]
      expected = production_greedy_ids(raw,prompt,9)
      decoder = GreedySpeculativeDecoder(target,EightDraft(len(prompt),expected),prepared_target=True,
                                        prepared_dynamic_rows=True,prepared_prefill_dynamic=True,early_anchor=True,prefix_trace=True)
      stream = decoder.generate(prompt,max_new_tokens=9,prefill_chunk_size=4)
      self.assertEqual(next(stream),expected[0])
      stream.close()
      self.assertEqual(target._cached_tokens,prompt+expected[:1])
      self.assertEqual(decoder.stats.trace_record_calls,0)
      self.assertEqual(decoder.stats.replay_calls,1)
      self.assertEqual(decoder._prefix_trace.valid_rows,0)
      # A new traced graph that materializes all state and then emits bad logits
      # must not make that partially validated history restorable.
      failing = GreedySpeculativeDecoder(target,EightDraft(len(prompt),expected),early_anchor=True,prefix_trace=True)
      forward = target.target_forward
      inject = {'bad':None}
      def invalid_verify(tokens,position,*,capture_layers=(),last_only=False,state_trace=None):
        output = forward(tokens,position,capture_layers=capture_layers,last_only=last_only,state_trace=state_trace)
        if state_trace is not None and inject['bad'] is not None:
          bad,inject['bad'] = inject['bad'],None
          if bad=='exception':raise RuntimeError('injected failure after traced state write')
          return TargetOutput(output.logits*float('nan'),output.hidden,output.features).realize()
        return output
      with patch.object(target,'target_forward',new=invalid_verify):
        for bad in ('nan','exception'):
          with self.subTest(bad=bad):
            inject['bad'] = bad
            stream = failing.generate(prompt,max_new_tokens=9,prefill_chunk_size=4)
            self.assertEqual(next(stream),expected[0])
            with self.assertRaisesRegex(ValueError if bad=='nan' else RuntimeError,'Nonfinite' if bad=='nan' else 'injected failure'):
              next(stream)
            self.assertEqual(failing.finish_reason,'error')
            self.assertEqual(failing._prefix_trace.valid_rows,0)
            self.assertEqual(target._cached_tokens,[])
            self.assertEqual(failing.stats.trace_record_calls,1)
            self.assertEqual(failing.stats.trace_restore_calls,0)
            self.assertEqual(failing.stats.replay_calls,1)
            self.assertEqual(list(failing.generate(prompt,max_new_tokens=9,prefill_chunk_size=4)),expected)


class TestNativeGDNTraceEligibility(unittest.TestCase):
  def test_unsupported_device_and_pure_attention_fail_before_state_or_workspace_allocation(self):
    raw,target = real_pair(True)
    with patch('tinygrad.llm.kernels.mali_recurrent.supported',return_value=False):
      with self.assertRaisesRegex(ValueError,'qualified native Mali'):NativeGDNPrefixTrace(target,8)
    with patch('tinygrad.llm.kernels.mali_recurrent.supported',return_value=True):
      with self.assertRaisesRegex(ValueError,'native GatedDeltaNet'):NativeGDNPrefixTrace(target,8)
      _,pure = real_pair(False)
      with self.assertRaisesRegex(ValueError,'requires a native'):NativeGDNPrefixTrace(pure,8)
    self.assertTrue(all(not hasattr(block,'conv_state') and not hasattr(block,'cache_kv') for block in target.blk))
    self.assertIs(type(raw),Transformer)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestNativeGDNPrefixTraceMali(TestNativeGDNPrefixTrace):
  device = 'CL'


if __name__=='__main__':unittest.main()
