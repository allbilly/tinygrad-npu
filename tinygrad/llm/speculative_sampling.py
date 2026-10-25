"""Exact speculative acceptance/rejection for explicit sparse proposal distributions.

Uses Leviathan et al., ICML 2023, Algorithm 1: accept with min(1,p/q),
then sample the positive residual on the first rejection. Target probabilities
are defined by the same executed FP32 temperature scaling as ordinary sampling.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
import math
import numpy as np
from numpy.typing import NDArray
if TYPE_CHECKING:
  from tinygrad import Tensor
  from tinygrad.llm.speculative import DraftModel


@dataclass(frozen=True)
class SparseDistribution:
  token_ids:tuple[int,...]
  probabilities:tuple[float,...]

  def __post_init__(self):
    object.__setattr__(self,'token_ids',tuple(self.token_ids))
    object.__setattr__(self,'probabilities',tuple(self.probabilities))
    if (not self.token_ids or len(self.token_ids)!=len(self.probabilities) or len(set(self.token_ids))!=len(self.token_ids)
        or any(type(token) is not int or token<0 for token in self.token_ids)):
      raise ValueError('Sparse proposal IDs must be distinct nonnegative integers')
    if (any(not math.isfinite(value) or value<0 for value in self.probabilities)
        or not math.isclose(math.fsum(self.probabilities),1.0,rel_tol=1e-12,abs_tol=1e-12)):
      raise ValueError('Sparse proposal probabilities must be finite, nonnegative and normalized')


@dataclass(frozen=True)
class SampledProposal:
  tokens:tuple[int,...]
  distributions:tuple[SparseDistribution,...]

  def __post_init__(self):
    object.__setattr__(self,'tokens',tuple(self.tokens))
    object.__setattr__(self,'distributions',tuple(self.distributions))
    if len(self.tokens)!=len(self.distributions):raise ValueError('Every sampled proposal needs its conditional distribution')
    for token,distribution in zip(self.tokens,self.distributions):
      if type(token) is not int or token not in distribution.token_ids or distribution.probabilities[distribution.token_ids.index(token)]<=0:
        raise ValueError('A sampled token must have positive proposal probability')


def temperature_value(value:float) -> float:
  if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
    raise ValueError('Sampling temperature must be finite and positive')
  if value>np.finfo(np.float32).max:raise ValueError('Sampling temperature must fit FP32')
  return float(max(np.float32(value),np.float32(1e-12)))


def probabilities(scores:NDArray, temperature:float=1.0) -> NDArray[np.float64]:
  values=np.asarray(scores,dtype=np.float64)
  if values.ndim<1 or values.shape[-1]<1 or not np.isfinite(values).all():raise ValueError('Sampling scores must be finite nonempty rows')
  if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:
    raise ValueError('Probability temperature must be finite and positive')
  values=values/temperature
  if not np.isfinite(values).all():raise ValueError('Scaled sampling scores must be finite')
  values=np.exp(values-values.max(axis=-1,keepdims=True))
  return values/values.sum(axis=-1,keepdims=True)


def categorical(values:NDArray|Sequence[float], random:Callable[[],float]) -> int:
  weights=np.asarray(values,dtype=np.float64)
  if weights.ndim!=1 or not weights.size or not np.isfinite(weights).all() or np.any(weights<0):
    raise ValueError('Categorical weights must be finite nonnegative values')
  cumulative=np.cumsum(weights)
  if not math.isfinite(float(cumulative[-1])) or cumulative[-1]<=0:raise ValueError('Categorical weights must have positive mass')
  draw=random()
  if type(draw) not in (int,float) or not math.isfinite(draw) or not 0<=draw<1:
    raise ValueError('Sampling random draws must be in [0,1)')
  threshold=min(draw*cumulative[-1],np.nextafter(cumulative[-1],0.0))
  return int(np.searchsorted(cumulative,threshold,side='right'))


def residual(target:NDArray, proposal:SparseDistribution) -> NDArray[np.float64]:
  values=np.asarray(target,dtype=np.float64).copy()
  if values.ndim!=1 or not values.size or not np.isfinite(values).all() or np.any(values<0):
    raise ValueError('Target probabilities must be finite nonnegative values')
  if not math.isclose(float(values.sum()),1.0,rel_tol=1e-12,abs_tol=1e-12):
    raise ValueError('Target probabilities must be normalized')
  if max(proposal.token_ids)>=values.size:raise ValueError('Proposal token exceeds the target vocabulary')
  values[np.asarray(proposal.token_ids,dtype=np.intp)]-=np.asarray(proposal.probabilities,dtype=np.float64)
  np.maximum(values,0,out=values)
  total=values.sum()
  if not math.isfinite(float(total)) or total<=0:raise ValueError('Rejected proposal has no positive target residual')
  return values/total


def target_probabilities(logits:Tensor, temperature:float) -> NDArray[np.float64]:
  from tinygrad import Tensor, dtypes
  # Compute the division on the target device in FP32, as model.forward does.
  scale=Tensor([temperature],device=logits.device,dtype=dtypes.float).maximum(1e-12)
  return probabilities((logits.float()/scale).numpy())


class SamplingSelection:
  def __init__(self, temperature:float, vocab_size:int, seed:int|None=None):
    self.temperature=temperature_value(temperature)
    if type(vocab_size) is not int or vocab_size<1:raise ValueError('Sampling vocabulary must be positive')
    if seed is not None and (type(seed) is not int or seed<0):raise ValueError('Sampling seed must be a nonnegative integer')
    self.vocab_size=vocab_size
    self.rng=np.random.default_rng(seed)
    self.random:Callable[[],float]=lambda:float(self.rng.random())
    self.proposal:SampledProposal|None=None

  def next_token(self, logits:Tensor) -> int:
    values=target_probabilities(logits,self.temperature)
    if values.shape[-1]!=self.vocab_size:raise ValueError('Target sampling vocabulary differs')
    return categorical(values.reshape(-1,self.vocab_size)[-1],self.random)

  def propose(self, draft:DraftModel, anchor:int, *, position:int, max_tokens:int) -> Sequence[int]:
    sampler=getattr(draft,'propose_sampled',None)
    if not callable(sampler):raise ValueError('The trained draft does not expose sampled proposal probabilities')
    proposal=sampler(anchor,position=position,max_tokens=max_tokens,temperature=self.temperature,random=self.random)
    if not isinstance(proposal,SampledProposal):raise ValueError('Draft did not return a sampled proposal')
    if len(proposal.tokens)>max_tokens-1 or any(max(row.token_ids)>=self.vocab_size for row in proposal.distributions):
      raise ValueError('Sampled proposal exceeds its budget or vocabulary')
    self.proposal=proposal
    return proposal.tokens

  def verify(self, followups:Sequence[int], logits:Tensor, eos:set[int], *, need_bonus:bool=True) -> tuple[int,int,bool]:
    values=target_probabilities(logits,self.temperature)
    if values.shape[-1]!=self.vocab_size:raise ValueError('Target sampling vocabulary differs')
    values=values.reshape(-1,self.vocab_size)
    if len(values)!=len(followups)+1:raise ValueError('Target sampling rows do not match the proposed block')
    proposal=self.proposal
    if followups and (proposal is None or tuple(followups)!=proposal.tokens):raise ValueError('Sampled proposal probabilities are stale')
    if proposal is not None:
      for index,token in enumerate(followups):
        row=proposal.distributions[index]
        q=row.probabilities[row.token_ids.index(token)]
        draw=self.random()
        if type(draw) not in (int,float) or not 0<=draw<1 or not math.isfinite(draw):raise ValueError('Acceptance draw must be in [0,1)')
        if draw>=min(1.0,float(values[index,token])/q):
          return index,categorical(residual(values[index],row),self.random),True
        if token in eos:return index+1,0,False
    return len(followups),categorical(values[-1],self.random) if need_bonus else 0,False
