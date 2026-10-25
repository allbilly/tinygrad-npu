"""Explicit OpenCL qualification of sampled probability extraction and sparse correction."""
import os, unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.speculative_sampling import SampledProposal, SamplingSelection, SparseDistribution, target_probabilities


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestSampledProbabilitiesGPU(unittest.TestCase):
  def test_full_vocabulary_fp32_scaling_matches_independent_cpu_probabilities(self):
    rng=np.random.default_rng(6011)
    scores=rng.normal(0,3,(1,8,248320)).astype(np.float32)
    stored=Tensor(scores,device='CL').realize()
    for temperature in (.6,1.,2.):
      actual=target_probabilities(stored,temperature)
      scaled=(scores/np.float32(temperature)).astype(np.float32).astype(np.float64)
      expected=np.exp(scaled-scaled.max(axis=-1,keepdims=True))
      expected/=expected.sum(axis=-1,keepdims=True)
      np.testing.assert_allclose(actual,expected,rtol=5e-6,atol=1e-10)
      np.testing.assert_allclose(actual.sum(axis=-1),1.,rtol=0,atol=1e-12)

  def test_rejection_uses_positive_residual_on_opencl_logits(self):
    logits=Tensor(np.log(np.array([[[.2,.3,.5],[.1,.6,.3]]],np.float32)),device='CL').realize()
    selection=SamplingSelection(1.,3,seed=9)
    selection.proposal=SampledProposal((0,),(SparseDistribution((0,2),(.8,.2)),))
    draws=iter((.9,.75))
    selection.random=lambda:next(draws)
    self.assertEqual(selection.verify((0,),logits,set()),(0,2,True))
