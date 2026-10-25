"""Dynamic verifier sharing with exact replay rows and ordinary greedy parity."""
from contextlib import nullcontext
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, nn
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _PreparedTargetForward, _map_outputs
from tinygrad.llm.speculative import GreedySpeculativeDecoder as EagerDecoder
from test.models.test_speculative import FixedDraft, real_pair
from test.models.test_speculative_dynamic_prefill import production_inputs
from test.models.test_speculative_prepared import assert_states, production_greedy_ids


class EightDraft(FixedDraft):
  block_size = 8


def eager_output(model, original, ids, position, phase, capacity, *, capture_layers, last_only):
  if phase=='prefill':tokens,start = production_inputs(model,ids,position,capacity)
  else:
    exact = phase=='replay'
    capacity = len(ids) if exact else min(capacity,model.max_context-position)
    tokens = Tensor([list(ids)+[0]*(capacity-len(ids))],device='CPU',dtype='int32').realize()
    suffix = f'{phase}_{capacity}'
    start = UOp.variable('reference_pos_'+suffix,0,model.max_context-capacity).bind(position)
    rows = UOp.variable('reference_rows_'+suffix,capacity if exact else 1,capacity).bind(len(ids))
    tokens = tokens[:,:rows]
  output = original(tokens,start,capture_layers=capture_layers,last_only=last_only)
  return _map_outputs(output,lambda value:value.pad_to(value.max_shape)
                      [:,:1 if last_only and value is output.logits else len(ids)]).realize()


class TestPreparedExactReplay(unittest.TestCase):
  device = 'CPU'

  def models(self):
    raw,target = real_pair(True)
    if self.device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for model in (raw,target):
        for parameter in nn.state.get_parameters(model):parameter.replace(parameter.half().to('CL').realize())
        configure(model,recurrent_scan=True,pack_weights=False)
    return raw,target

  def test_one_dynamic_verify_graph_and_exact_replay_one_two_rows_match_eager_state(self):
    raw,target = self.models()
    helper = _PreparedTargetForward(target,(0,1),dynamic_rows=True,exact_replay=True)
    original = raw.target_forward
    cases = ((8,0,'verify'),(7,8,'verify'),(5,15,'verify'),(1,20,'replay'),(2,21,'replay'))
    for rows,position,phase in cases:
      ids = [(position+j*3+1)%13 for j in range(rows)]
      expected = eager_output(raw,original,ids,position,phase,8,capture_layers=(0,1),last_only=False)
      actual = helper(ids,position,phase,8,False)
      for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
        np.testing.assert_array_equal(a.numpy().view(np.uint32),b.numpy().view(np.uint32))
      assert_states(self,target,raw,position+rows)
    self.assertEqual([key for key in helper.calls if key[0]=='verify'],[('verify',8,False)])
    self.assertIsNotNone(helper.calls[('verify',8,False)].jit.captured)
    for rows in (1,2):
      replay = helper.calls[('replay',rows,False)]
      self.assertEqual((replay.rows.vmin,replay.rows.vmax),(rows,rows))
      self.assertEqual(replay.capacity,rows)
    self.assertEqual(helper.calls[('replay',1,False)].capacity,1)  # Body projections are GEMV eligible.

  def test_independent_ordinary_ids_exact_state_rejections_close_and_gemv_replay(self):
    raw,target = self.models()
    prompt = [2,5,3,9,1,7,4]
    expected_ids = production_greedy_ids(raw,prompt,9)
    raw_draft,jit_draft = EightDraft(len(prompt),expected_ids),EightDraft(len(prompt),expected_ids)
    eager = EagerDecoder(raw,raw_draft,early_anchor=True)
    prepared = GreedySpeculativeDecoder(target,jit_draft,prepared_target=True,prepared_dynamic_rows=True,
                                        prepared_prefill_dynamic=True,prepared_exact_replay=True,early_anchor=True)
    phase = {'reference':None,'actual':None}
    original,forward = raw.target_forward,eager._forward
    def reference(tokens, position, *, capture_layers=(), last_only=False):
      current = phase['reference']
      capacity = raw.prefill_chunk_size(4) if current=='prefill' else 8
      return eager_output(raw,original,tokens.tolist()[0],position,current,capacity,capture_layers=capture_layers,last_only=last_only)
    def reference_forward(ids, position, current, *, last_only=False):
      phase['reference'] = current
      return forward(ids,position,current,last_only=last_only)
    raw.target_forward,eager._forward = reference,reference_forward
    actual_forward = prepared._forward
    def actual(ids, position, current, *, last_only=False):
      phase['actual'] = current
      try:return actual_forward(ids,position,current,last_only=last_only)
      finally:phase['actual'] = None
    prepared._forward = actual
    gemv_shapes = []
    instrumentation = nullcontext()
    if self.device=='CL':
      from tinygrad.llm.kernels.mali_linear import gemv
      def replay_gemv(x, weight):
        if phase['actual']=='replay' and weight.shape==(16,8):gemv_shapes.append(x.max_shape)
        return gemv(x,weight)
      instrumentation = patch('tinygrad.llm.kernels.mali_linear.gemv',new=replay_gemv)
    with instrumentation:
      for wrong_at in (None,0,1):
        raw_draft.wrong_at = jit_draft.wrong_at = wrong_at
        result = list(prepared.generate(prompt,max_new_tokens=9,prefill_chunk_size=4))
        baseline = list(eager.generate(prompt,max_new_tokens=9,prefill_chunk_size=4))
        self.assertEqual(result,expected_ids)
        self.assertEqual(result,baseline)
        self.assertEqual(target._cached_tokens,raw._cached_tokens)
        assert_states(self,target,raw,len(prompt)+len(result))
        self.assertTrue(prepared.stats.prepared_exact_replay)
      raw_draft.wrong_at = jit_draft.wrong_at = None
      for cancelled in (False,True):
        flag = {'stop':False}
        a = prepared.generate(prompt,max_new_tokens=9,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
        b = eager.generate(prompt,max_new_tokens=9,prefill_chunk_size=4,cancelled=lambda:flag['stop'])
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
        assert_states(self,target,raw,len(prompt)+1)
    calls = prepared._prepared_target.calls
    self.assertEqual([key for key in calls if key[0]=='verify'],[('verify',8,False)])
    for rows in (1,2):self.assertIn(('replay',rows,False),calls)
    if self.device=='CL':self.assertIn((1,1,8),gemv_shapes)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestPreparedExactReplayMali(TestPreparedExactReplay):
  device = 'CL'


if __name__=='__main__':unittest.main()
