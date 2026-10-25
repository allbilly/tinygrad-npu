"""Optional request-level speculative routing, qualified by complete request timings.

Budget zero calls the production ordinary target.generate directly. It does not capture target features, execute
the draft, reset draft state, take speculative snapshots, or consume an extra last token for cache bookkeeping.
The speculative path requires the optional verification_budget coordinator change in coordinator.patch.
"""
from __future__ import annotations
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import math
from threading import Lock
from time import perf_counter
from tinygrad.llm.speculative import GreedySpeculativeDecoder, SpeculativeStats
from tinygrad.llm.speculative_policy import PolicyDecision, PolicyKey


@dataclass(frozen=True)
class RequestWorkload:
  profile: PolicyKey
  workload: str  # Representative prompt/domain/load identity, supplied by the calibration owner.
  prompt_tokens: int
  output_limit: int  # Effective limit after capping by target context room.
  prefill_chunk_size: int
  reusable_prefix_tokens: int  # Actual ordinary-cache reuse; speculation currently starts fresh.
  eos_token_ids: tuple[int,...] = ()  # Canonical sorted unique EOS IDs; part of the completion-workload identity.

  def __post_init__(self):
    if not isinstance(self.workload,str) or not self.workload:raise ValueError('A workload identity is required')
    if any(type(value) is not int for value in
           (self.prompt_tokens,self.output_limit,self.prefill_chunk_size,self.reusable_prefix_tokens)):
      raise ValueError('Request dimensions must be integers')
    if (self.profile.batch_size!=1 or self.prompt_tokens<1 or self.output_limit<0 or self.prefill_chunk_size<1
        or not 0<=self.reusable_prefix_tokens<=self.prompt_tokens):raise ValueError('Invalid single-request workload')
    if (type(self.eos_token_ids) is not tuple or any(type(token) is not int or token<0 for token in self.eos_token_ids)
        or self.eos_token_ids!=tuple(sorted(set(self.eos_token_ids)))):raise ValueError('EOS identity must be a canonical integer tuple')

  @property
  def covered(self) -> bool:
    low,high = self.profile.context_range
    return low<=self.prompt_tokens<=high and self.prompt_tokens+max(0,self.output_limit-1)<=high


class MeasuredRequestBudgetPolicy:
  """Complete-request qualification; no automatic exploration or optimistic round-cost extrapolation.

  All target/draft setup executed during a request, feature prefill, reset, observation, rejected replay, synchronization,
  and cleanup belong in total_seconds. Keep cold model loading and JIT warmup in separately identified runs. Timings of
  a fully drained iterator include its complete work; consumer pauses contaminate them and must not be calibrated.
  """
  def __init__(self, budgets:tuple[int,...], *, min_samples:int=3, min_speedup:float=1.05,
               window:int=32, max_profiles:int=16):
    if not budgets or len(set(budgets))!=len(budgets) or any(type(value) is not int or value<1 for value in budgets):
      raise ValueError('Distinct positive proposal budgets required')
    if any(type(value) is not int or value<1 for value in (min_samples,window,max_profiles)) or min_samples>window:
      raise ValueError('Invalid bounded calibration history')
    if not math.isfinite(min_speedup) or min_speedup<1:raise ValueError('Invalid minimum speedup')
    self.budgets,self.min_samples,self.min_speedup = tuple(sorted(budgets)),min_samples,min_speedup
    self.window,self.max_profiles = window,max_profiles
    self._profiles:OrderedDict[RequestWorkload,dict[int,deque[tuple[int,float]]]] = OrderedDict()

  def record(self, request:RequestWorkload, budget:int, emitted_tokens:int, total_seconds:float, *,
             qualified:bool=False, warmup:bool=False, finish_reason:str='length') -> None:
    if type(budget) is not int or budget not in (0,*self.budgets):raise ValueError('Unsupported proposal budget')
    if type(emitted_tokens) is not int or not 1<=emitted_tokens<=request.output_limit:raise ValueError('Invalid output token count')
    if not math.isfinite(total_seconds) or total_seconds<=0:raise ValueError('Invalid complete-request timing')
    if not qualified or warmup or finish_reason not in ('length','context','eos'):return
    if budget and budget+1>request.output_limit:raise ValueError('Draft block cannot fit in request output budget')
    if not request.covered:raise ValueError('Request extends beyond the measured context band')
    if request not in self._profiles:
      if len(self._profiles)>=self.max_profiles:self._profiles.popitem(last=False)
      self._profiles[request] = {}
    self._profiles.move_to_end(request)
    self._profiles[request].setdefault(budget,deque(maxlen=self.window)).append((emitted_tokens,total_seconds))

  def invalidate(self, request:RequestWorkload, budget:int) -> None:
    if request in self._profiles:self._profiles[request].pop(budget,None)

  def choose(self, request:RequestWorkload) -> PolicyDecision:
    if not request.covered:return PolicyDecision(0,1.,0,'outside_measured_request_context')
    samples = self._profiles.get(request,{})
    ordinary = samples.get(0,())
    if len(ordinary)<self.min_samples:return PolicyDecision(0,1.,len(ordinary),'unqualified_request')
    self._profiles.move_to_end(request)
    rate = sum(row[0] for row in ordinary)/sum(row[1] for row in ordinary)
    decision = PolicyDecision(0,1.,len(ordinary),'no_complete_request_gain')
    for budget in self.budgets:
      rows = samples.get(budget,())
      if len(rows)<self.min_samples or budget+1>request.output_limit:continue
      speedup = (sum(row[0] for row in rows)/sum(row[1] for row in rows))/rate
      if speedup>max(decision.speedup,self.min_speedup):decision = PolicyDecision(budget,speedup,len(rows),'measured_complete_request_gain')
    return decision


@dataclass
class RoutedStats:
  request: RequestWorkload
  decision: PolicyDecision
  mode: str
  cache_contract: str
  output_tokens: int = 0
  total_seconds: float = 0.
  finish_reason: str = 'running'
  speculative: SpeculativeStats|None = None


class RequestBudgetRouter:
  """Choose once before prefill. Ordinary and speculative cache semantics remain explicit.

  Target ownership is shared with GreedySpeculativeDecoder. External ordinary calls must honor that same lock.
  Profiles are supplied by a calibration owner; this router never labels its own output correct or records warm runs.
  """
  def __init__(self, decoder:GreedySpeculativeDecoder, policy:MeasuredRequestBudgetPolicy, profile:PolicyKey, *, workload:str):
    if profile.batch_size!=1:raise ValueError('Only one request is coordinated at a time')
    if any(budget>=decoder.draft.block_size for budget in policy.budgets):raise ValueError('Budget exceeds the trained draft block')
    self.decoder,self.policy,self.profile,self.workload = decoder,policy,profile,workload
    self.stats:RoutedStats|None = None
    self._routing_lock = Lock()

  def generate(self, prompt:Sequence[int], *, max_new_tokens:int, eos_token_ids:Iterable[int]=(),
               cancelled:Callable[[],bool]|None=None, prefill_chunk_size:int=128) -> Iterator[int]:
    if not self._routing_lock.acquire(blocking=False):raise RuntimeError('Router already has an active iterator')
    try:
      yield from self._generate(prompt,max_new_tokens=max_new_tokens,eos_token_ids=eos_token_ids,
                                cancelled=cancelled,prefill_chunk_size=prefill_chunk_size)
    finally:self._routing_lock.release()

  def _generate(self, prompt:Sequence[int], *, max_new_tokens:int, eos_token_ids:Iterable[int],
                cancelled:Callable[[],bool]|None, prefill_chunk_size:int) -> Iterator[int]:
    started = perf_counter()
    target = self.decoder.target
    tokens,eos_values = list(prompt),tuple(eos_token_ids)
    vocab = target.config.vocab_size
    if not tokens or any(type(token) is not int or not 0<=token<vocab for token in tokens):raise ValueError('Invalid prompt')
    if len(tokens)>target.max_context:raise ValueError('Prompt exceeds target context')
    if type(max_new_tokens) is not int or max_new_tokens<0:raise ValueError('Invalid output budget')
    if type(prefill_chunk_size) is not int or prefill_chunk_size<1:raise ValueError('Invalid prefill chunk size')
    if any(type(token) is not int or not 0<=token<vocab for token in eos_values):raise ValueError('Invalid EOS token IDs')
    eos = set(eos_values)
    effective = min(max_new_tokens,target.max_context-len(tokens))
    if not self.decoder._lock.acquire(blocking=False):raise RuntimeError('Target is already used by another iterator')
    try:
      request = RequestWorkload(self.profile,self.workload,len(tokens),effective,
                                target.prefill_chunk_size(prefill_chunk_size),target.get_start_pos(tokens),tuple(sorted(eos)))
      decision = self.policy.choose(request)
    except BaseException:
      self.decoder._lock.release()
      raise
    ordinary = decision.budget==0
    # Ordinary keeps ownership from selection through close. Speculation acquires that same lock inside its own
    # iterator before any model writes; its fresh-prompt path does not reuse the prefix observed at selection.
    if not ordinary:self.decoder._lock.release()
    self.stats = RoutedStats(request,decision,'ordinary' if ordinary else 'speculative',
                             'ordinary_pending_last_token' if ordinary else 'speculative_delivered_inputs_committed')
    stats,stop = self.stats,cancelled or (lambda:False)
    stream = None
    try:
      if effective==0:
        stats.finish_reason = 'length' if max_new_tokens==0 else 'context'
        return
      if ordinary:
        # Keep the production normal greedy implementation, its JITs and prefix reuse. In particular, do not
        # consume the final yielded token just to imitate the speculative coordinator's different cache contract.
        stream = iter(target.generate(tokens,chunk_size=prefill_chunk_size,temperature=0.))
        while stats.output_tokens<effective:
          if stop():
            stats.finish_reason = 'cancelled'
            return
          try:token = next(stream)
          except StopIteration:
            stats.finish_reason = 'context'
            return
          stats.output_tokens += 1
          yield token
          if token in eos:
            stats.finish_reason = 'eos'
            return
        stats.finish_reason = 'context' if effective<max_new_tokens else 'length'
      else:
        stream = self.decoder.generate(tokens,max_new_tokens=max_new_tokens,eos_token_ids=eos,cancelled=stop,
                                       prefill_chunk_size=prefill_chunk_size,verification_budget=decision.budget)
        for token in stream:
          stats.output_tokens += 1
          yield token
        stats.finish_reason = self.decoder.finish_reason
    except GeneratorExit:
      stats.finish_reason = 'closed'
      raise
    except BaseException:
      stats.finish_reason = 'error'
      self.policy.invalidate(request,decision.budget)
      raise
    finally:
      try:
        if stream is not None and (close:=getattr(stream,'close',None)) is not None:close()
      except BaseException:
        stats.finish_reason = 'error'
        self.policy.invalidate(request,decision.budget)
        raise
      finally:
        if not ordinary:stats.speculative = self.decoder.stats
        stats.total_seconds = perf_counter()-started
        if ordinary:self.decoder._lock.release()
