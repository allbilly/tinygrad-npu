"""Production-shaped prefill controls with exact verifier/replay comparisons."""
import os, unittest
import numpy as np
from tinygrad import Tensor, UOp, nn
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _PreparedTargetForward, _map_outputs
from tinygrad.llm.speculative import GreedySpeculativeDecoder as EagerDecoder
from test.models.test_speculative import FixedDraft, real_pair
from test.models.test_speculative_prepared import assert_states, production_greedy_ids, symbolic_eager_reference


def production_inputs(model, ids, position, capacity):
  values = [0]*position+list(ids)+[0]*(model.max_context-position-len(ids))
  full = Tensor(values,device=model.token_embd.weight.device,dtype='int32').reshape(1,model.max_context)
  start = UOp.variable('start_pos',0,model.max_context-1).bind(position)
  rows = UOp.variable('toks',1,capacity).bind(len(ids))
  return full[:,start:start+rows].contiguous(),start


class TestDynamicPreparedPrefill(unittest.TestCase):
  device = 'CPU'

  def models(self):
    raw,target = real_pair(True)
    if self.device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for model in (raw,target):
        for parameter in nn.state.get_parameters(model):parameter.replace(parameter.half().to('CL').realize())
        configure(model,recurrent_scan=True,pack_weights=False)
    return raw,target

  def test_production_bounds_visible_input_context_tail_and_shared_remainder_graph(self):
    raw,target = self.models()
    helper = _PreparedTargetForward(target,(0,1),dynamic_rows=False,prefill_dynamic=True)
    position = 0
    for index,rows in enumerate((4,1,3,4,2,4,4,2)):
      ids = [(position+j*3+1)%13 for j in range(rows)]
      tokens,start = production_inputs(raw,ids,position,4)
      expected = raw.target_forward(tokens,start,capture_layers=(0,1),last_only=True)
      actual = helper(ids,position,'prefill',4,True)
      self.assertIs(actual.hidden,actual.features[-1])
      for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
        valid = 1 if b is expected.logits else rows
        reference = b.pad_to(b.max_shape).numpy()[:,:valid]
        np.testing.assert_array_equal(a.numpy().view(np.uint32),reference.view(np.uint32))
      assert_states(self,target,raw,position+rows)
      if index==3:helper.free_intermediates()
      position += rows
    self.assertEqual(position,target.max_context)
    self.assertEqual(list(helper.calls),[('prefill',4,True)])
    call = helper.calls[('prefill',4,True)]
    self.assertEqual((call.position.vmin,call.position.vmax),(0,target.max_context-1))
    self.assertEqual((call.rows.vmin,call.rows.vmax),(1,4))
    self.assertIsNotNone(call.jit.captured)
    # Turning on production-shaped prefill must not change verifier geometry.
    helper([2,5],0,'verify',4,False)
    verify = helper.calls[('verify',2,False)]
    self.assertFalse(verify.production_prefill)
    self.assertEqual((verify.rows.vmin,verify.rows.vmax),(2,2))
    self.assertEqual(verify.position.vmax,target.max_context-2)

  def test_production_ids_exact_state_early_close_cancel_and_rejection(self):
    raw,target = self.models()
    prompt = [2,5,3,9,1,7,4]
    expected_ids = production_greedy_ids(raw,prompt,7)
    original,exact = raw.target_forward,symbolic_eager_reference(raw)
    capacity = raw.prefill_chunk_size(4)
    def reference(tokens, position, *, capture_layers=(), last_only=False):
      if position<len(prompt):
        ids = tokens.tolist()[0]
        visible,start = production_inputs(raw,ids,position,capacity)
        output = original(visible,start,capture_layers=capture_layers,last_only=last_only)
        # The eager reference uses symbolic inputs for matching arithmetic,
        # then presents the same concrete row views consumed by the draft.
        return _map_outputs(output,lambda value:value.pad_to(value.max_shape)
                            [:,:1 if last_only and value is output.logits else len(ids)]).realize()
      return exact(tokens,position,capture_layers=capture_layers,last_only=last_only)
    raw.target_forward = reference
    raw_draft,jit_draft = FixedDraft(len(prompt),expected_ids),FixedDraft(len(prompt),expected_ids)
    eager = EagerDecoder(raw,raw_draft,early_anchor=True)
    prepared = GreedySpeculativeDecoder(target,jit_draft,prepared_target=True,prepared_prefill_dynamic=True,early_anchor=True)
    for wrong_at in (None,None,1):
      raw_draft.wrong_at = jit_draft.wrong_at = wrong_at
      actual = list(prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
      baseline = list(eager.generate(prompt,max_new_tokens=7,prefill_chunk_size=4))
      self.assertEqual(actual,expected_ids)
      self.assertEqual(actual,baseline)
      self.assertEqual(target._cached_tokens,raw._cached_tokens)
      assert_states(self,target,raw,len(prompt)+len(actual))
      self.assertTrue(prepared.stats.prepared_prefill_dynamic)
      self.assertFalse(prepared.stats.prepared_dynamic_rows)
    raw_draft.wrong_at = jit_draft.wrong_at = None
    for cancelled in (False,True):
      flag = {'stop':False}
      a = prepared.generate(prompt,max_new_tokens=7,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
      b = eager.generate(prompt,max_new_tokens=7,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
      self.assertEqual(next(a),next(b))
      if cancelled:
        flag['stop'] = True
        self.assertEqual(list(a),[])
        self.assertEqual(list(b),[])
      else:
        a.close()
        b.close()
      self.assertEqual(target._cached_tokens,prompt+expected_ids[:1])
      self.assertEqual(target._cached_tokens,raw._cached_tokens)
      self.assertEqual(prepared.stats.replay_calls,1)
      assert_states(self,target,raw,len(prompt)+1)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestDynamicPreparedPrefillMali(TestDynamicPreparedPrefill):
  device = 'CL'


if __name__=='__main__':unittest.main()
