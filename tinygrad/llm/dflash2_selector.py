"""Sparse DFlash2 bigram reranking and the trained predecessor-conditioned greedy walk.

Math follows vllm-project/speculators 21192d6f CandidateSelector; walk follows vLLM e7bb7e2f
qwen3_dflash2.py:_score_edges and worker/gpu/spec_decode/dflash2/speculator.py:_selector_walk_kernel.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping
from collections.abc import Callable
if TYPE_CHECKING:
  from tinygrad.llm.speculative_sampling import SampledProposal
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2_config import DFlash2Config

class CandidateSelector:
  def __init__(self, config:DFlash2Config, weights:Mapping[str,Tensor]):
    self.config,self.top_k = config,config.selector_top_k
    names = ("predecessor_codebook","successor_codebook","hidden_projection.weight")
    shapes = ((config.vocab_size,config.selector_rank),(config.vocab_size,config.selector_rank),(config.selector_rank,config.hidden_size))
    values = []
    for name,shape in zip(names,shapes):
      tensor = weights[f"candidate_selector.{name}"]
      if tensor.shape != shape or tensor.dtype not in (dtypes.float16,dtypes.bfloat16,dtypes.float32):
        raise ValueError(f"Invalid DFlash2 selector tensor: {name}")
      values.append(tensor)
    self.predecessor_codebook,self.successor_codebook,self.hidden_projection = values
    if len({value.device for value in values}) != 1 or len({value.dtype for value in values}) != 1:
      raise ValueError("Selector weights must share a compute device and dtype")

  def _validate_inputs(self, unary_logits:Tensor, hidden:Tensor) -> None:
    if unary_logits.shape[:-1] != hidden.shape[:-1] or unary_logits.shape[-1] != self.config.vocab_size:
      raise ValueError("Selector logits and hidden states have incompatible shapes")
    if hidden.shape[-1] != self.config.hidden_size: raise ValueError("Selector hidden size differs from checkpoint")
    if hidden.device != self.hidden_projection.device or unary_logits.device != hidden.device:
      raise ValueError("Selector inputs and weights must share a device")

  @staticmethod
  def _rows(weight:Tensor, ids:Tensor) -> Tensor:
    # Generic advanced indexing builds a vocabulary-sized mask. Use the existing direct CL lookup for large resident codebooks.
    if isinstance(weight.device,str) and weight.device.split(":")[0] == "CL" and weight.dtype in (dtypes.float16,dtypes.float32):
      from tinygrad.llm.kernels.mali_linear import embedding
      return embedding(ids.cast(dtypes.int32),weight).cast(weight.dtype)
    return weight[ids]

  def score_candidates(self, unary_logits:Tensor, hidden:Tensor, predecessor_ids:Tensor, candidate_ids:Tensor) -> Tensor:
    self._validate_inputs(unary_logits,hidden)
    if predecessor_ids.shape != hidden.shape[:-1] or candidate_ids.shape[:-1] != hidden.shape[:-1]:
      raise ValueError("Selector predecessor or candidate IDs have incompatible shapes")
    if not dtypes.is_int(predecessor_ids.dtype) or not dtypes.is_int(candidate_ids.dtype): raise ValueError("Selector IDs must be integers")
    predecessor_ids,candidate_ids = predecessor_ids.to(hidden.device),candidate_ids.to(hidden.device)
    predecessor = self._rows(self.predecessor_codebook,predecessor_ids)
    projected = hidden.cast(self.hidden_projection.dtype).linear(self.hidden_projection.transpose())
    context = predecessor*projected.cast(predecessor.dtype)
    successors = self._rows(self.successor_codebook,candidate_ids)
    transitions = (context.unsqueeze(-2)*successors).sum(-1)
    return unary_logits.gather(-1,candidate_ids)+transitions.cast(unary_logits.dtype)

  def select(self, unary_logits:Tensor, hidden:Tensor, predecessor_ids:Tensor, top_k:int|None=None) -> tuple[Tensor,Tensor]:
    self._validate_inputs(unary_logits,hidden)
    ids = self._topk(unary_logits,self.top_k if top_k is None else top_k)
    return ids,self.score_candidates(unary_logits,hidden,predecessor_ids,ids)

  def _topk(self, logits:Tensor, k:int) -> Tensor:
    # A measured host path until there is a qualified device top-k; O(V) workspace, never the O(V^2) MoE helper.
    import numpy as np
    if type(k) is not int or not 1 <= k <= self.config.vocab_size: raise ValueError("Invalid selector top-k")
    values = logits.float().numpy()
    if not np.isfinite(values).all(): raise ValueError("Selector unary logits must be finite")
    # Resolve ties at the boundary by token ID, then sort only the selected k values.
    rows = values.reshape(-1,values.shape[-1])
    selected = np.empty((len(rows),k),dtype=np.int32)
    for i,row in enumerate(rows):
      threshold = np.partition(row,len(row)-k)[len(row)-k]
      better = np.flatnonzero(row>threshold)
      tied = np.flatnonzero(row==threshold)[:k-len(better)]
      ids = np.concatenate((better,tied))
      selected[i] = ids[np.lexsort((ids,-row[ids]))]
    return Tensor(selected.reshape(*values.shape[:-1],k),device=logits.device,dtype=dtypes.int32)

  def choose(self, unary_logits:Tensor, hidden:Tensor, anchor:int, max_tokens:int) -> list[int]:
    """Return up to max_tokens followups from slots 1..n, conditioning on each preceding selection, starting with the real anchor."""
    self._validate_inputs(unary_logits,hidden)
    if unary_logits.ndim != 3 or unary_logits.shape[:2] != (1,self.config.block_size):
      raise ValueError("Greedy DFlash2 selection requires one complete trained noise block")
    if type(anchor) is not int or not 0 <= anchor < self.config.vocab_size: raise ValueError("Invalid anchor token")
    if type(max_tokens) is not int or not 0 <= max_tokens < self.config.block_size: raise ValueError("Invalid proposal length")
    if max_tokens == 0: return []
    candidate_rows,edge_host = self._walk_scores(unary_logits,hidden,anchor,max_tokens)
    result = []
    previous = 0
    for step in range(max_tokens):
      previous = int(edge_host[0,step,previous].argmax())
      result.append(candidate_rows[step][previous])
    return result

  def _walk_scores(self, unary_logits:Tensor, hidden:Tensor, anchor:int, max_tokens:int) -> tuple[list[list[int]],Any]:
    logits,features = unary_logits[:,1:1+max_tokens],hidden[:,1:1+max_tokens]
    candidates = self._topk(logits,self.top_k)
    host = candidates.tolist()
    if not isinstance(host,list) or len(host)!=1 or not isinstance(host[0],list) or len(host[0])!=max_tokens:
      raise ValueError("Selector candidates must contain one batch of proposal rows")
    candidate_rows:list[list[int]] = []
    for row in host[0]:
      if not isinstance(row,list) or len(row)!=self.top_k: raise ValueError("Selector candidate row has the wrong width")
      tokens:list[int] = []
      for token in row:
        if not isinstance(token,int) or isinstance(token,bool) or not 0<=token<self.config.vocab_size:
          raise ValueError("Selector candidate IDs must be in-range integers")
        tokens.append(token)
      candidate_rows.append(tokens)
    anchor_ids = Tensor.full((1,1,self.top_k),anchor,device=hidden.device,dtype=dtypes.int32)
    previous_ids = anchor_ids.cat(candidates[:,:-1],dim=1)
    projected = features.cast(self.hidden_projection.dtype).linear(self.hidden_projection.transpose())
    contexts = self._rows(self.predecessor_codebook,previous_ids)*projected.unsqueeze(-2).cast(self.predecessor_codebook.dtype)
    successors = self._rows(self.successor_codebook,candidates)
    transitions = (contexts.unsqueeze(-2)*successors.unsqueeze(-3)).sum(-1)
    edges = logits.gather(-1,candidates).unsqueeze(-2)+transitions.cast(logits.dtype)
    edge_host = edges.float().numpy()
    import numpy as np
    if not np.isfinite(edge_host).all(): raise ValueError("DFlash2 selector scores must be finite")
    return candidate_rows,edge_host

  def sample(self, unary_logits:Tensor, hidden:Tensor, anchor:int, max_tokens:int, *,
             temperature:float, random:Callable[[],float]) -> SampledProposal:
    """Sample the trained predecessor-conditioned sparse walk and retain each q_i."""
    from tinygrad.llm.speculative_sampling import SampledProposal, SparseDistribution, categorical, probabilities, temperature_value
    self._validate_inputs(unary_logits,hidden)
    if unary_logits.ndim!=3 or unary_logits.shape[:2]!=(1,self.config.block_size):
      raise ValueError('Sampled DFlash2 selection requires one complete trained noise block')
    if type(anchor) is not int or not 0<=anchor<self.config.vocab_size:raise ValueError('Invalid anchor token')
    if type(max_tokens) is not int or not 0<=max_tokens<self.config.block_size:raise ValueError('Invalid proposal length')
    temperature=temperature_value(temperature)
    if max_tokens==0:return SampledProposal((),())
    candidates,edges=self._walk_scores(unary_logits,hidden,anchor,max_tokens)
    previous,tokens,distributions=0,[],[]
    for step in range(max_tokens):
      q=probabilities(edges[0,step,previous],temperature)
      previous=categorical(q,random)
      tokens.append(candidates[step][previous])
      distributions.append(SparseDistribution(tuple(candidates[step]),tuple(float(value) for value in q)))
    return SampledProposal(tuple(tokens),tuple(distributions))
