"""Bounded complete-block draft replay; numerical geometry stays explicit."""
import os, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2 import DFlash2
from tinygrad.llm.dflash2_selector import CandidateSelector
from tinygrad.llm.model import TargetOutput
from test.models.test_dflash2 import fixture


def setup(seed,device='CPU',half=False):
  base,c,weights,embedding,head,rng=fixture(seed)
  c.selector_rank,c.selector_top_k=2,3
  weights.update({'candidate_selector.hidden_projection.weight':rng.normal(0,.1,(2,8)).astype(np.float32),
                  'candidate_selector.predecessor_codebook':rng.normal(0,.1,(16,2)).astype(np.float32),
                  'candidate_selector.successor_codebook':rng.normal(0,.1,(16,2)).astype(np.float32)})
  dtype=np.float16 if half else np.float32
  tensors={name:Tensor(value.astype(dtype),device=device).realize() for name,value in weights.items()}
  emb=Tensor(embedding.astype(dtype),device=device).realize()
  out=Tensor(head.astype(dtype),device=device).realize()
  def embed(ids):return emb[ids]
  def project(hidden):return hidden@out.T
  raw=DFlash2(c,tensors,embed,project,base.verifier_norm,CandidateSelector(c,tensors))
  replay=DFlash2(c,tensors,embed,project,base.verifier_norm,CandidateSelector(c,tensors))
  return raw,replay,rng


def observe(draft,features,start):
  fields=tuple(Tensor(x,device=draft.device,dtype=draft.dtype).realize() for x in np.split(features,2,axis=-1))
  draft.observe(TargetOutput(Tensor.zeros(1,features.shape[1],16,device=draft.device),fields[-1],fields),start)


def compare(draft,position,*,atol=2e-6,rtol=3e-5):
  graph=draft._proposal_forward
  noise=Tensor([[3,15,15,15]],device=draft.device,dtype=dtypes.int32).realize()
  expected=graph.eager(noise,draft._prefix_kv,position)
  expected_arrays=expected.logits.numpy().copy(),expected.hidden.numpy().copy()
  actual=graph(noise,draft._prefix_kv,position)
  np.testing.assert_array_equal(actual.logits.numpy(),expected_arrays[0])
  np.testing.assert_array_equal(actual.hidden.numpy(),expected_arrays[1])
  prefix=draft._target_hidden
  if prefix is None:prefix=Tensor.zeros(1,0,8,device=draft.device,dtype=draft.dtype)
  original=draft._forward_projected(noise,prefix,position=position,prefix_kv=draft._prefix_kv).realize()
  np.testing.assert_allclose(actual.logits.numpy(),original.logits.numpy(),atol=atol,rtol=rtol)
  np.testing.assert_allclose(actual.hidden.numpy(),original.hidden.numpy(),atol=atol,rtol=rtol)
  np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),original.logits.numpy().argmax(-1))


class TestPreparedDFlash2(unittest.TestCase):
  def test_matched_geometry_and_original_math_across_prefix_changes_and_reset(self):
    _raw,draft,rng=setup(491)
    head=draft.lm_head
    weights={name:value.uop for name,value in draft.weights.items()}
    metadata=draft.prepare_proposals(max_context=16)
    self.assertEqual(metadata['graphs'],1)
    # Two K/V tensors: sliding window 2 + full prefix 16, head dim 4, FP32.
    self.assertEqual(metadata['padding_input_bytes'],576)
    graph=draft._proposal_forward
    compare(draft,0)
    for position,rows in ((0,5),(5,3),(2,1),(3,4)):
      observe(draft,rng.normal(0,.4,(1,rows,16)).astype(np.float32),position)
      compare(draft,position+rows)
    draft.reset()
    self.assertIs(draft._proposal_forward,graph)
    compare(draft,0)
    observe(draft,rng.normal(0,.4,(1,15,16)).astype(np.float32),0)
    compare(draft,15)
    self.assertEqual(draft.proposal_preparation_metadata()['graphs'],1)
    self.assertEqual(draft.proposal_preparation_metadata()['captured_graphs'],1)
    self.assertIs(draft.lm_head,head)
    for name,value in draft.weights.items():self.assertIs(value.uop,weights[name])
    draft.close_prepared_proposals()
    self.assertIsNone(draft._proposal_forward)
    self.assertIsNone(graph.draft)
    self.assertIsNone(graph.jit.fxn)
    draft.close_prepared_proposals()
    with self.assertRaisesRegex(RuntimeError,'closed'):
      graph(Tensor([[3,15,15,15]],device='CPU',dtype=dtypes.int32),draft._prefix_kv,15)

  def test_actual_selector_stays_outside_graph_and_proposal_ids_match(self):
    raw,draft,rng=setup(492)
    calls=[]
    selector=draft.candidate_selector
    class ObserveSelector:
      def choose(self,logits,hidden,anchor,max_tokens):
        calls.append((logits.shape,hidden.shape,max_tokens))
        return selector.choose(logits,hidden,anchor,max_tokens)
    draft.candidate_selector=ObserveSelector()
    draft.prepare_proposals(max_context=16)
    for position,rows,budget in ((0,5,4),(5,2,4),(2,1,2),(3,5,4)):
      features=rng.normal(0,.4,(1,rows,16)).astype(np.float32)
      for model in (raw,draft):observe(model,features,position)
      before=len(calls)
      self.assertEqual(draft.propose(3,position=position+rows,max_tokens=budget),raw.propose(3,position=position+rows,max_tokens=budget))
      self.assertEqual(len(calls),before+1)
      self.assertEqual(calls[-1],((1,4,16),(1,4,8),budget-1))
    draft.close_prepared_proposals()

  def test_capacity_and_cache_guards_leave_generic_path_available(self):
    raw,draft,rng=setup(493)
    for invalid in (0,513,True):
      with self.assertRaises(ValueError):draft.prepare_proposals(max_context=invalid)
    draft.cache_prefix_kv=False
    with self.assertRaises(ValueError):draft.prepare_proposals(max_context=16)
    draft.cache_prefix_kv=True
    draft.prepare_proposals(max_context=8)
    features=rng.normal(0,.4,(1,8,16)).astype(np.float32)
    for model in (raw,draft):observe(model,features,0)
    with self.assertRaisesRegex(ValueError,'configured context'):draft.propose(3,position=8,max_tokens=4)
    draft.close_prepared_proposals()
    self.assertEqual(draft.propose(3,position=8,max_tokens=4),raw.propose(3,position=8,max_tokens=4))
    draft.prepare_proposals(max_context=16)
    draft.lm_head=lambda hidden:hidden
    with self.assertRaisesRegex(ValueError,'borrowed modules changed'):draft.propose(3,position=8,max_tokens=4)
    draft.close_prepared_proposals()

  def test_device_close_failures_release_all_graph_references_and_leave_eager_available(self):
    for failing_phase in ('synchronize','free_intermediates'):
      with self.subTest(phase=failing_phase):
        raw,draft,rng=setup(495)
        features=rng.normal(0,.4,(1,5,16)).astype(np.float32)
        for model in (raw,draft):observe(model,features,0)
        expected=raw.propose(3,position=5,max_tokens=4)
        draft.prepare_proposals(max_context=16)
        graph=draft._proposal_forward
        failure=RuntimeError('injected '+failing_phase)
        synchronize=Mock(side_effect=failure if failing_phase=='synchronize' else None)
        free=Mock(side_effect=failure if failing_phase=='free_intermediates' else None)
        graph.jit.captured=SimpleNamespace(free_intermediates=free)
        graph.jit._linears=[object()]
        with patch('tinygrad.llm.dflash2.Device',{'CPU':SimpleNamespace(synchronize=synchronize)}):
          with self.assertRaises(RuntimeError) as caught:draft.close_prepared_proposals()
        self.assertIs(caught.exception,failure)
        synchronize.assert_called_once_with()
        if failing_phase=='synchronize':free.assert_not_called()
        else:free.assert_called_once_with()
        self.assertIsNone(draft._proposal_forward)
        self.assertTrue(graph.closed)
        self.assertIsNone(graph.draft)
        self.assertIsNone(graph.jit.fxn)
        self.assertIsNone(graph.jit.captured)
        self.assertEqual(graph.jit.cnt,0)
        self.assertNotIn('_linears',graph.jit.__dict__)
        graph.close()
        draft.close_prepared_proposals()
        self.assertEqual(draft.propose(3,position=5,max_tokens=4),expected)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestPreparedDFlash2Mali(unittest.TestCase):
  def test_fp16_complete_block_replay_with_shared_prepared_head(self):
    from tinygrad import nn
    from tinygrad.llm.kernels.mali import MaliLinear
    raw,draft,rng=setup(494,'CL',True)
    source=nn.Linear(8,16,bias=False)
    source.weight.replace(Tensor(rng.normal(0,.2,(16,8)).astype(np.float16),device='CL').realize())
    shared=MaliLinear(source,prepare_gemm=False)
    for model in (raw,draft):
      model.lm_head=shared
      self.assertTrue(model.prepare_mali()['prepared_shared_head'])
    draft.prepare_proposals(max_context=32)
    for position,rows in ((0,7),(7,4),(2,1),(3,5)):
      features=rng.normal(0,.4,(1,rows,16)).astype(np.float16)
      for model in (raw,draft):observe(model,features,position)
      compare(draft,position+rows,atol=3e-3,rtol=4e-3)
      self.assertEqual(draft.propose(3,position=position+rows,max_tokens=4),raw.propose(3,position=position+rows,max_tokens=4))
    draft.reset()
    compare(draft,0,atol=3e-3,rtol=4e-3)
    self.assertIs(draft.lm_head,shared)
    self.assertIs(raw.lm_head,shared)
    draft.close_prepared_proposals()


if __name__=='__main__':unittest.main()
