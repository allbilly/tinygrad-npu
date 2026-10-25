"""CPU tests for sampled sparse DFlash2 selector walks.

Reuses the small selector fixture. The probability oracle below is independent
Python math, not the production probabilities/categorical helpers.
"""
import math
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.speculative_sampling import SampledProposal
from test.models.test_dflash2_checkpoint import selector_fixture


def _softmax(scores,temperature):
  scaled=[float(score)/float(temperature) for score in scores]
  maximum=max(scaled)
  weights=[math.exp(value-maximum) for value in scaled]
  total=sum(weights)
  return tuple(value/total for value in weights)


def _draw_index(probabilities,uniform):
  threshold=uniform*sum(probabilities)
  cumulative=0.0
  for index,probability in enumerate(probabilities):
    cumulative+=probability
    if threshold<cumulative:return index
  return len(probabilities)-1


class _Draws:
  def __init__(self,values):
    self.values=iter(values)
    self.calls=0
  def __call__(self):
    self.calls+=1
    return next(self.values)


def _controlled_selector():
  config,arrays,selector=selector_fixture()
  predecessor=np.zeros((config.vocab_size,config.selector_rank),dtype=np.float32)
  successor=np.zeros_like(predecessor)
  projection=np.zeros((config.selector_rank,config.hidden_size),dtype=np.float32)
  predecessor[0]=[2.,0.,0.]
  predecessor[2]=[-2.,0.,0.]
  predecessor[5]=[0.,0.,0.]  # real anchor
  successor[0]=[1.,0.,0.]
  successor[1]=[-1.,0.,0.]
  successor[2]=[0.,0.,0.]
  projection[0,0]=1.
  selector.predecessor_codebook=Tensor(predecessor,device='CPU').realize()
  selector.successor_codebook=Tensor(successor,device='CPU').realize()
  selector.hidden_projection=Tensor(projection,device='CPU').realize()
  logits=np.full((1,config.block_size,config.vocab_size),-10.,dtype=np.float32)
  logits[0,0,10]=10000.  # slot zero is a trained mask/anchor slot and is skipped
  logits[0,1,:3]=[2.,1.,0.]
  logits[0,2,:3]=[2.,1.,0.]
  hidden=np.zeros((1,config.block_size,config.hidden_size),dtype=np.float32)
  hidden[0,1,0]=1.
  hidden[0,2,0]=1.
  return config,selector,logits,hidden,predecessor,successor,projection


def test_sampled_walk_uses_sparse_conditional_probabilities_and_sampled_predecessor():
  config,selector,logits,hidden,predecessor,successor,projection=_controlled_selector()
  draws=_Draws([0.99,0.5])
  sampled=selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,2,temperature=1.0,random=draws)
  assert isinstance(sampled,SampledProposal)

  # Independent edge oracle: step 0 conditions on anchor 5 (zero code), so its
  # sparse scores are [2,1,0]. Draw .99 selects token 2. Step 1 must then use
  # predecessor_codebook[token 2]=[-2,0,0], yielding [0,3,0], not the greedy
  # token-0 predecessor scores [4,-1,0]. Draw .5 therefore selects token 1.
  edge0=[2.,1.,0.]
  q0=_softmax(edge0,1.0)
  selected0=_draw_index(q0,0.99)
  predecessor_token=[0,1,2][selected0]
  projected0=hidden[0,2]@projection.T
  context0=predecessor[predecessor_token]*projected0
  edge1=[float(logits[0,2,token])+float(successor[token]@context0) for token in (0,1,2)]
  q1=_softmax(edge1,1.0)
  selected1=_draw_index(q1,0.5)
  assert selected1==1
  assert edge1==[0.,3.,0.]
  assert sampled.tokens==(2,1)
  assert tuple(sampled.distributions[0].token_ids)==(0,1,2)
  assert tuple(sampled.distributions[1].token_ids)==(0,1,2)
  np.testing.assert_allclose(sampled.distributions[0].probabilities,q0,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(sampled.distributions[1].probabilities,q1,rtol=1e-12,atol=1e-12)
  assert draws.calls==2
  # The old greedy path remains deterministic and retains the previous behavior.
  assert selector.choose(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,2)==[0,0]


def test_sparse_noncontiguous_ids_are_retained_in_each_distribution():
  config,selector,logits,hidden,_,_,_=_controlled_selector()
  predecessor=Tensor.zeros(config.vocab_size,config.selector_rank,device='CPU')
  successor=Tensor.zeros(config.vocab_size,config.selector_rank,device='CPU')
  projection=Tensor.zeros(config.selector_rank,config.hidden_size,device='CPU')
  selector.predecessor_codebook,selector.successor_codebook,selector.hidden_projection=predecessor,successor,projection
  logits[:]= -10.
  logits[0,1,[2,6,9]]=[3.,2.,1.]
  logits[0,2,[2,6,9]]=[3.,2.,1.]
  draws=_Draws([0.1,0.1])
  temperature=0.5
  sampled=selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,2,temperature=temperature,random=draws)
  expected=_softmax([3.,2.,1.],temperature)
  assert sampled.tokens==(2,2)
  assert tuple(sampled.distributions[0].token_ids)==(2,6,9)
  assert tuple(sampled.distributions[1].token_ids)==(2,6,9)
  np.testing.assert_allclose(sampled.distributions[0].probabilities,expected,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(sampled.distributions[1].probabilities,expected,rtol=1e-12,atol=1e-12)
  assert draws.calls==2


def test_sampled_candidates_keep_deterministic_topk_boundary_ties():
  config,selector,logits,hidden,_,_,_=_controlled_selector()
  selector.predecessor_codebook=Tensor.zeros(config.vocab_size,config.selector_rank,device='CPU')
  selector.successor_codebook=Tensor.zeros(config.vocab_size,config.selector_rank,device='CPU')
  selector.hidden_projection=Tensor.zeros(config.selector_rank,config.hidden_size,device='CPU')
  logits[:]=-10.
  logits[0,1,:]=[3.,2.,2.,2.,1.,0.,-1.,-2.,-3.,-4.,-5.]
  draws=_Draws([0.9])
  sampled=selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,1,temperature=1.0,random=draws)
  expected=_softmax([3.,2.,2.],1.0)
  assert tuple(sampled.distributions[0].token_ids)==(0,1,2)  # tied boundary keeps smaller token IDs
  np.testing.assert_allclose(sampled.distributions[0].probabilities,expected,rtol=1e-12,atol=1e-12)
  assert sampled.tokens==(2,) and draws.calls==1

def test_zero_budget_and_invalid_temperature_or_nonfinite_scores():
  _,selector,logits,hidden,_,_,_=_controlled_selector()
  draws=_Draws([])
  empty=selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,0,temperature=1.0,random=draws)
  assert empty.tokens==() and empty.distributions==() and draws.calls==0
  for temperature in (0.0,-1.0,float('nan'),float('inf')):
    with pytest.raises(ValueError,match='temperature'):
      selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,1,temperature=temperature,random=_Draws([0.5]))
  bad_logits=logits.copy()
  bad_logits[0,1,0]=float('nan')
  with pytest.raises(ValueError,match='finite'):
    selector.sample(Tensor(bad_logits,device='CPU'),Tensor(hidden,device='CPU'),5,1,temperature=1.0,random=_Draws([0.5]))
  _,bad_selector,logits,hidden,_,successor,projection=_controlled_selector()
  successor[0,0]=float('nan')
  bad_selector.successor_codebook=Tensor(successor,device='CPU').realize()
  with pytest.raises(ValueError,match='finite'):
    bad_selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,1,temperature=1.0,random=_Draws([0.5]))
  with pytest.raises(ValueError,match='draws'):
    selector.sample(Tensor(logits,device='CPU'),Tensor(hidden,device='CPU'),5,1,temperature=1.0,random=_Draws([1.0]))
