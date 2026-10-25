"""Greedy trained-draft verification, with transactional recurrent decoder state.

The caller must exclusively own the target while the iterator is live. Coordinators
sharing a target exclude each other; ordinary target calls must obey the same rule.
Batch/scalar numerical equivalence is a backend qualification requirement, as with
ordinary batched prefill. No draft token is emitted without target verification.
"""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import prod
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Protocol
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.model import TargetOutput, Transformer
from tinygrad.llm.speculative_trace import NativeGDNPrefixTrace
from tinygrad.llm.speculative_adaptive import DecodeFallbackPolicy
if TYPE_CHECKING:
  from tinygrad.llm.kernels.mali_state import PackedStateSnapshot
  from tinygrad.llm.speculative_sampling import SamplingSelection


class DraftModel(Protocol):
  block_size:int
  capture_layers:tuple[int,...]

  def reset(self) -> None: ...
  def observe(self, output:TargetOutput, start_pos:int) -> None:
    """Append committed target features, replacing any suffix at start_pos."""
    ...
  def propose(self, anchor:int, *, position:int, max_tokens:int) -> Sequence[int]:
    """Return followups after the known anchor, at most max_tokens-1 tokens."""
    ...


@dataclass
class SpeculativeStats:
  sampled:bool = False
  sampling_temperature:float = 0.0
  sampling_acceptance_tests:int = 0
  sampling_residual_draws:int = 0
  adaptive:bool = False
  ordinary_decode_seconds:float = 0.0
  max_speculative_extra_seconds:float = 0.0
  fallback_tokens:int = 0
  fallback_calls:int = 0
  fallback_seconds:float = 0.0
  fallback_reason:str = ''
  deferred_fallback:bool = False
  retain_fallback_prefix:bool = False
  fallback_observation_batch_size:int = 1
  fallback_observation_calls:int = 0
  fallback_feature_copy_bytes:int = 0
  fallback_feature_peak_rows:int = 0
  fallback_pending_last_token:bool = False
  early_anchor:bool = False
  prepared_target:bool = False
  prepared_dynamic_rows:bool = False
  prepared_prefill_dynamic:bool = False
  prepared_exact_replay:bool = False
  cached_tokens:int = 0
  prefix_trace:bool = False
  verify_first_three:bool = False
  verify_first_calls:int = 0
  verify_rest_calls:int = 0
  verify_skipped_input_tokens:int = 0
  verify_join_seconds:float = 0.0
  trace_bytes:int = 0
  trace_record_calls:int = 0
  trace_recorded_bytes:int = 0
  trace_restore_calls:int = 0
  trace_restore_bytes:int = 0
  trace_prepare_seconds:float = 0.0
  trace_restore_seconds:float = 0.0
  prepared_graphs:int = 0
  output_tokens:int = 0
  proposed_tokens:int = 0
  accepted_tokens:int = 0
  blocks:int = 0
  rejected_blocks:int = 0
  target_calls:int = 0
  target_input_tokens:int = 0
  prefill_calls:int = 0
  verify_calls:int = 0
  replay_calls:int = 0
  rollback_count:int = 0
  snapshot_bytes:int = 0
  snapshot_calls:int = 0
  native_snapshot_calls:int = 0
  native_snapshot_restores:int = 0
  prefill_seconds:float = 0.0
  draft_seconds:float = 0.0
  observe_seconds:float = 0.0
  prompt_observe_seconds:float = 0.0
  verify_seconds:float = 0.0
  snapshot_seconds:float = 0.0
  restore_seconds:float = 0.0
  replay_seconds:float = 0.0
  reset_seconds:float = 0.0
  total_seconds:float = 0.0


def _synchronize(tensors:Iterable[Tensor]) -> None:
  devices:set[str] = set()
  for tensor in tensors:
    if isinstance(device:=tensor.device,str):devices.add(device)
    elif device is not None:devices.update(device)
  for device in devices:Device[device].synchronize()


class _StateSnapshot:
  """Only non-positional state needs copying; later KV rows remain invisible."""
  def __init__(self, target:Transformer):
    states:list[Tensor] = []
    for block in target.blk:
      for owner,names in ((block,('conv_state','recurrent_state')), (getattr(block,'shortconv',None),('cache_conv',))):
        for name in names:
          if isinstance(value:=getattr(owner,name,None),Tensor) and not any(value is item for item in states):states.append(value)
    self.native:PackedStateSnapshot|None=None
    self.saved:tuple[tuple[Tensor,Tensor],...]=()
    if states and getattr(target,'_mali_native_state_snapshots',False):
      from tinygrad.llm.kernels.mali_state import PackedStateSnapshot, supported
      if supported(tuple(states)):
        self.native=PackedStateSnapshot(tuple(states))
        self.nbytes=self.native.nbytes
        return
    self.saved = tuple((state,state.clone()) for state in states)
    if self.saved:
      Tensor.realize(*(backup for _,backup in self.saved))
      _synchronize(backup for _,backup in self.saved)
    self.nbytes = sum(prod(backup.max_shape)*backup.dtype.itemsize for _,backup in self.saved)

  def restore(self) -> None:
    if self.native is not None:
      self.native.restore()
      return
    if self.saved:
      Tensor.realize(*(state.assign(backup) for state,backup in self.saved))
      _synchronize(state for state,_ in self.saved)


_lock_creation = Lock()


def _map_outputs(output:TargetOutput, fn:Callable[[Tensor],Tensor]) -> TargetOutput:
  converted:dict[int,Tensor] = {}
  def convert(value:Tensor) -> Tensor:
    if id(value) not in converted:converted[id(value)] = fn(value)
    return converted[id(value)]
  return TargetOutput(convert(output.logits),convert(output.hidden),tuple(convert(value) for value in output.features))


def _join_outputs(first:TargetOutput, rest:TargetOutput) -> TargetOutput:
  """Materialize one logical proposal's rows without rerunning target state writes."""
  if len(first.features)!=len(rest.features):raise ValueError('Verifier stages returned different feature counts')
  converted:dict[tuple[int,int],Tensor] = {}
  def join(left:Tensor, right:Tensor) -> Tensor:
    key = id(left),id(right)
    if key not in converted:converted[key] = left.cat(right,dim=1)
    return converted[key]
  return TargetOutput(join(first.logits,rest.logits),join(first.hidden,rest.hidden),
                      tuple(join(left,right) for left,right in zip(first.features,rest.features))).realize()


class _PreparedTargetCall:
  """One bounded graph; returned views expire on the next call to this graph.

  Bitwise eager references must use the same symbolic position/row bounds:
  static positions can select different floating-point reduction geometry.
  """
  def __init__(self, target:Transformer, capture_layers:tuple[int,...], phase:str, capacity:int, last_only:bool, dynamic_rows:bool,
               production_prefill:bool=False, state_trace:NativeGDNPrefixTrace|None=None):
    self.target,self.capture_layers,self.capacity,self.last_only = target,capture_layers,capacity,last_only
    self.production_prefill = production_prefill
    self.state_trace = state_trace
    suffix = f'{phase}_{capacity}_{int(last_only)}'
    self.position = UOp.variable('start_pos' if production_prefill else 'spec_target_pos_'+suffix,
                                 0,target.max_context-1 if production_prefill else target.max_context-capacity)
    self.rows = UOp.variable('toks' if production_prefill else 'spec_target_rows_'+suffix,
                             1 if production_prefill or dynamic_rows else capacity,capacity)
    self.jit = TinyJit(self._run_prefill if production_prefill else self._run)

  def _run(self, tokens:Tensor, position:UOp, rows:UOp, *trace_buffers:Tensor) -> TargetOutput:
    if self.state_trace is not None:
      output = self.target.target_forward(tokens[:,:rows],position,capture_layers=self.capture_layers,last_only=self.last_only,
                                          state_trace=self.state_trace.bind(trace_buffers))
    else:output = self.target.target_forward(tokens[:,:rows],position,capture_layers=self.capture_layers,last_only=self.last_only)
    # Fixed output extents keep symbolic row variables out of the consumer. This
    # padding work is captured and timed; all real decoder outputs were already
    # materialized together by target_forward, so no stateful block is rerun.
    return _map_outputs(output,lambda value:value.pad_to(value.max_shape)).realize()

  def _run_prefill(self, tokens:Tensor, position:UOp) -> TargetOutput:
    output = self.target.target_forward(tokens,position,capture_layers=self.capture_layers,last_only=self.last_only)
    return _map_outputs(output,lambda value:value.pad_to(value.max_shape)).realize()

  def __call__(self, ids:Sequence[int], position:int) -> TargetOutput:
    if self.jit.fxn is None:raise RuntimeError('Prepared target graph was evicted')
    if self.production_prefill:
      device = self.target.token_embd.weight.device
      if not isinstance(device,str):raise ValueError('Production-shaped prefill requires a single compute device')
      full = Tensor([0]*position+list(ids)+[0]*(self.target.max_context-position-len(ids)),device=device,
                     dtype='int32').reshape(1,self.target.max_context)
      start,rows = self.position.bind(position),self.rows.bind(len(ids))
      # Match generate's full device buffer and visible contiguous symbolic view
      # passed into TinyJit, including clipping near the context boundary.
      output = self.jit(full[:,start:start+rows].contiguous(),start)
    else:
      tokens = Tensor([list(ids)+[0]*(self.capacity-len(ids))],device='CPU',dtype='int32').realize()
      buffers = self.state_trace.buffers if self.state_trace is not None else ()
      output = self.jit(tokens,self.position.bind(position),self.rows.bind(len(ids)),*buffers)
    # Outputs remain views into this graph's owned result buffers. Consumers must
    # finish observation/score extraction before calling this graph again.
    return _map_outputs(output,lambda value:value[:,:1 if self.last_only and value is output.logits else len(ids)]).realize()

  def close(self) -> None:
    if self.jit.fxn is None:return
    weight = self.target.token_embd.weight
    captured = self.jit.captured
    try:
      _synchronize((weight,))
      if captured is not None:captured.free_intermediates()
    finally:
      self.jit.reset()
      # TinyJit's bound self._run otherwise retains this call and its target in
      # a cycle. Drop heavy graph/results immediately rather than waiting for GC.
      self.jit.fxn = None
      del captured  # An exception traceback must not retain the captured results.
    # Freed graph shapes have different allocation sizes. Flush their unused
    # allocator entries; live result views still retain their backing buffers.
    if isinstance(device:=weight.device,str):Device[device].allocator.free_cache()
    elif device is not None:
      for name in device:Device[name].allocator.free_cache()


class _PreparedTargetForward:
  def __init__(self, target:Transformer, capture_layers:tuple[int,...], *, dynamic_rows:bool=True, prefill_dynamic:bool=False,
               exact_replay:bool=False, state_trace:NativeGDNPrefixTrace|None=None,
               max_graphs:int=16, max_prefill_graphs:int=2):
    if (type(max_graphs) is not int or type(max_prefill_graphs) is not int or not 1<=max_prefill_graphs<=max_graphs):
      raise ValueError('Prepared graph limits must be positive integers with prefill no larger than total')
    self.target,self.capture_layers = target,capture_layers
    self.dynamic_rows = dynamic_rows
    self.prefill_dynamic = prefill_dynamic
    self.exact_replay = exact_replay
    self.state_trace = state_trace
    self.max_graphs,self.max_prefill_graphs = max_graphs,max_prefill_graphs
    self.calls:OrderedDict[tuple[str,int,bool],_PreparedTargetCall] = OrderedDict()
    self.evictions = 0

  def _evict(self, key:tuple[str,int,bool]) -> None:
    call = self.calls.pop(key)
    self.evictions += 1
    call.close()

  def _make_room(self, phase:str) -> None:
    if phase=='prefill':
      while sum(key[0]=='prefill' for key in self.calls)>=self.max_prefill_graphs:
        self._evict(next(key for key in self.calls if key[0]=='prefill'))
    while len(self.calls)>=self.max_graphs:self._evict(next(iter(self.calls)))

  def __call__(self, ids:Sequence[int], position:int, phase:str, capacity:int, last_only:bool) -> TargetOutput:
    production_prefill = self.prefill_dynamic and phase=='prefill'
    dynamic_rows = self.dynamic_rows and not (self.exact_replay and phase=='replay')
    if not 0<=position<self.target.max_context or not 1<=len(ids)<=min(capacity,self.target.max_context-position):
      raise ValueError('Prepared target input exceeds context')
    if not production_prefill:capacity = min(capacity,self.target.max_context-position)
    # Exact-length graphs preserve eager kernel geometry and rounding. Dynamic
    # row graphs are a separate opt-in requiring backend numerical qualification.
    if not production_prefill and not dynamic_rows:capacity = len(ids)
    key = (phase,capacity,last_only)
    if key not in self.calls:
      self._make_room(phase)
      self.calls[key] = _PreparedTargetCall(self.target,self.capture_layers,phase,capacity,last_only,dynamic_rows,production_prefill,
                                            self.state_trace if phase in ('verify','verify_first','verify_rest') else None)
    self.calls.move_to_end(key)
    return self.calls[key](ids,position)

  def free_intermediates(self) -> None:
    for call in self.calls.values():
      if call.jit.captured is not None:call.jit.captured.free_intermediates()


class _FallbackFeatureBatch:
  """Own expiring target feature views until a bounded observation is ready."""
  def __init__(self, capacity:int, observe:Callable[[tuple[Tensor,...],int],None], stats:SpeculativeStats):
    self.capacity,self.observe,self.stats = capacity,observe,stats
    self.rows:list[tuple[Tensor,...]] = []
    self.start_pos = 0

  def append(self, features:tuple[Tensor,...], position:int) -> None:
    start = perf_counter()
    try:
      if not features or any(value.ndim!=3 or value.shape[:2]!=(1,1) for value in features):
        raise ValueError('Fallback batching requires one row of target features')
      if self.rows:
        if position!=self.start_pos+len(self.rows):raise ValueError('Fallback feature rows must be contiguous')
        if len(features)!=len(self.rows[0]) or any((a.shape,a.dtype,a.device)!=(b.shape,b.dtype,b.device)
                                                 for a,b in zip(features,self.rows[0])):
          raise ValueError('Fallback feature layout changed')
      else:self.start_pos = position
      # Prepared target outputs expire on the next call to the same graph.
      owned = tuple(value.clone() for value in features)
      Tensor.realize(*owned)
      self.stats.fallback_feature_copy_bytes += sum(int(value.nbytes()) for value in owned)
      self.rows.append(owned)
      self.stats.fallback_feature_peak_rows = max(self.stats.fallback_feature_peak_rows,len(self.rows))
      if len(self.rows)==self.capacity:self._flush()
    finally:self.stats.observe_seconds += perf_counter()-start

  def _flush(self) -> None:
    if not self.rows:return
    rows,self.rows = self.rows,[]
    features = tuple(rows[0][i].cat(*(row[i] for row in rows[1:]),dim=1) for i in range(len(rows[0])))
    self.stats.fallback_observation_calls += 1
    self.observe(features,self.start_pos)
    _synchronize(features)

  def flush(self) -> None:
    if not self.rows:return
    start = perf_counter()
    try:self._flush()
    finally:self.stats.observe_seconds += perf_counter()-start

  def clear(self) -> None:self.rows.clear()


class GreedySpeculativeDecoder:
  """Greedy speculation, including stream-close/cancellation rollback.

  Draft implementations own their feature cache and must not mutate the target.
  Requests start at position zero unless opt-in prefix reuse validates a strict
  extension of this coordinator's completed target and draft cache. Ordinary
  target calls, draft mutations, failed or interrupted requests invalidate reuse.
  By default the final target cache contains
  the prompt and all delivered generated inputs. With deferred_fallback enabled,
  fallback delivery can leave its last token pending, as ordinary generation does;
  _cached_tokens still names exactly the processed, valid causal prefix.
  Opt-in retain_fallback_prefix observes existing fallback features so the draft
  stays aligned with that processed prefix; an unprocessed final token stays pending.
  Feature-only drafts can batch up to eight fallback observations with owned copies.
  Timing includes draft work, snapshots, transfers, rejection replay and cleanup;
  total_seconds also includes time spent by the consumer between iterator advances.
  With early_anchor enabled, a yielded anchor can be temporarily unconsumed while
  the iterator owns the target; resuming or closing consumes it with timed work.
  """
  def __init__(self, target:Transformer, draft:DraftModel, *, prepared_target:bool=False, prepared_dynamic_rows:bool=False,
               early_anchor:bool=False, prepared_prefill_dynamic:bool=False, prepared_exact_replay:bool=False,
               prefix_trace:bool=False, verify_first_three:bool=False, deferred_fallback:bool=False, reuse_prefix:bool=False,
               retain_fallback_prefix:bool=False, fallback_observation_batch_size:int=1):
    if target.config.rwkv7:raise NotImplementedError('RWKV speculative state rollback is not implemented')
    if type(draft.block_size) is not int or draft.block_size<1:raise ValueError('Draft block size must be a positive integer')
    if type(verify_first_three) is not bool:raise ValueError('verify_first_three must be a bool')
    if type(deferred_fallback) is not bool:raise ValueError('deferred_fallback must be a bool')
    if type(reuse_prefix) is not bool:raise ValueError('reuse_prefix must be a bool')
    if type(retain_fallback_prefix) is not bool:raise ValueError('retain_fallback_prefix must be a bool')
    if retain_fallback_prefix and not reuse_prefix:raise ValueError('retain_fallback_prefix requires prefix reuse')
    if type(fallback_observation_batch_size) is not int or not 1<=fallback_observation_batch_size<=8:
      raise ValueError('fallback_observation_batch_size must be an integer in 1..8')
    if fallback_observation_batch_size>1 and (not retain_fallback_prefix or not callable(getattr(draft,'observe_features',None))):
      raise ValueError('Fallback batching requires prefix retention and feature-only observation')
    if reuse_prefix and any(type(getattr(draft,name,None)) is not int for name in ('cache_revision','cached_prefix_length')):
      raise ValueError('Prefix reuse requires draft cache revision and prefix length metadata')
    if verify_first_three and draft.block_size!=8:raise ValueError('First-three verification requires the unchanged eight-slot draft block')
    self.target,self.draft = target,draft
    self.early_anchor = early_anchor
    self.verify_first_three = verify_first_three
    self.deferred_fallback = deferred_fallback
    self.reuse_prefix = reuse_prefix
    self.retain_fallback_prefix = retain_fallback_prefix
    self.fallback_observation_batch_size = fallback_observation_batch_size
    self._cached_target_marker:list[int]|None = None
    self._cached_prefix:tuple[int,...] = ()
    self._cached_draft_revision:int|None = None
    self._prefix_trace = NativeGDNPrefixTrace(target,draft.block_size) if prefix_trace else None
    if prepared_target and type(target) is not Transformer:raise ValueError('Prepared target requires the base Transformer')
    if prepared_dynamic_rows and not prepared_target:raise ValueError('Dynamic rows require prepared target execution')
    if prepared_prefill_dynamic and not prepared_target:raise ValueError('Dynamic prefill requires prepared target execution')
    if prepared_exact_replay and not (prepared_target and prepared_dynamic_rows):
      raise ValueError('Exact replay requires prepared target execution with dynamic verification rows')
    self._prepared_target = (_PreparedTargetForward(target,draft.capture_layers,dynamic_rows=prepared_dynamic_rows,
                                                    prefill_dynamic=prepared_prefill_dynamic,exact_replay=prepared_exact_replay,
                                                    state_trace=self._prefix_trace)
                             if prepared_target else None)
    self._prepared_prefill_capacity = 1
    self.stats = SpeculativeStats(early_anchor=early_anchor,prepared_target=prepared_target,prepared_dynamic_rows=prepared_dynamic_rows,
                                  prepared_prefill_dynamic=prepared_prefill_dynamic,prepared_exact_replay=prepared_exact_replay,
                                  prefix_trace=prefix_trace,verify_first_three=verify_first_three,deferred_fallback=deferred_fallback,
                                  retain_fallback_prefix=retain_fallback_prefix,fallback_observation_batch_size=fallback_observation_batch_size)
    self.finish_reason = 'idle'
    with _lock_creation:
      if (lock:=getattr(target,'_speculative_lock',None)) is None:
        lock = Lock()
        setattr(target,'_speculative_lock',lock)
      self._lock = lock

  def _reset_draft(self) -> None:
    self._clear_prefix()
    start = perf_counter()
    try:self.draft.reset()
    finally:self.stats.reset_seconds += perf_counter()-start

  def _clear_prefix(self) -> None:
    self._cached_target_marker,self._cached_prefix,self._cached_draft_revision = None,(),None

  def get_start_pos(self, prompt:Sequence[int]) -> int:
    """Query reuse without changing state; the target must be exclusively owned."""
    prefix = self._cached_prefix
    if (not self.reuse_prefix or not prefix or self._cached_target_marker is not self.target._cached_tokens
        or tuple(self.target._cached_tokens)!=prefix or len(prompt)<=len(prefix) or tuple(prompt[:len(prefix)])!=prefix
        or getattr(self.draft,'cache_revision',None)!=self._cached_draft_revision
        or getattr(self.draft,'cached_prefix_length',None)!=len(prefix)):
      return 0
    return len(prefix) if self.target.get_start_pos(list(prompt))==len(prefix) else 0

  def _retain_prefix(self) -> bool:
    marker = self.target._cached_tokens
    revision = getattr(self.draft,'cache_revision',None)
    if not marker or type(revision) is not int or getattr(self.draft,'cached_prefix_length',None)!=len(marker):return False
    self._cached_target_marker,self._cached_prefix,self._cached_draft_revision = marker,tuple(marker),revision
    return True

  def _observe(self, output:TargetOutput, position:int, *, prompt:bool=False) -> None:
    start = perf_counter()
    try:
      self.draft.observe(output,position)
      _synchronize(output.features)
    finally:
      elapsed = perf_counter()-start
      self.stats.observe_seconds += elapsed
      if prompt:self.stats.prompt_observe_seconds += elapsed

  def _observe_fallback(self, output:TargetOutput, position:int, batch:_FallbackFeatureBatch|None) -> None:
    if batch is not None:batch.append(output.features,position)
    else:
      self.stats.fallback_observation_calls += 1
      self._observe(output,position)

  def _forward(self, ids:Sequence[int], position:int, phase:str, *, last_only:bool=False) -> tuple[TargetOutput,list[int]]:
    start = perf_counter()
    verify = phase in ('verify','verify_first','verify_rest')
    counter_phase = 'verify' if verify else phase
    self.stats.target_calls += 1
    self.stats.target_input_tokens += len(ids)
    setattr(self.stats,counter_phase+'_calls',getattr(self.stats,counter_phase+'_calls')+1)
    if phase in ('verify_first','verify_rest'):setattr(self.stats,phase+'_calls',getattr(self.stats,phase+'_calls')+1)
    try:
      if verify and self._prefix_trace is not None:
        self._prefix_trace.invalidate()
        self.stats.trace_record_calls += 1
        # Charged before execution: an exceptional/partial call may write less.
        self.stats.trace_recorded_bytes += self._prefix_trace.state_bytes*len(ids)+self._prefix_trace.window_bytes
      if self._prepared_target is not None:
        capacity = self._prepared_prefill_capacity if phase=='prefill' else 1 if phase=='fallback' else self.draft.block_size
        if phase=='verify_first':capacity = 3
        elif phase=='verify_rest':capacity = 5
        output = self._prepared_target(ids,position,phase,capacity,last_only)
      else:
        tokens = Tensor([list(ids)],device='CPU',dtype='int32')
        if verify and self._prefix_trace is not None:
          output = self.target.target_forward(tokens,position,capture_layers=self.draft.capture_layers,last_only=last_only,
                                              state_trace=self._prefix_trace.bind())
        else:output = self.target.target_forward(tokens,position,capture_layers=self.draft.capture_layers,last_only=last_only)
      # NaN comparisons can make argmax return vocab_size. Reject any nonfinite
      # row, including a mixed finite/NaN row whose argmax happens to be in range.
      greedy = output.logits.argmax(-1)
      if greedy.shape!=output.logits.shape[:-1] or not dtypes.is_int(greedy.dtype):
        raise ValueError('Invalid target greedy IDs')
      prediction = output.logits.isfinite().all(-1).where(greedy,self.target.config.vocab_size)
      values = prediction.reshape(-1).tolist()
      if (not isinstance(values,list) or len(values)!=(1 if last_only else len(ids))
          or any(type(value) is not int or not 0<=value<self.target.config.vocab_size for value in values)):
        raise ValueError('Nonfinite target logits or invalid target greedy IDs')
      if verify and self._prefix_trace is not None:
        # No partial/nonfinite verification is eligible for history restoration.
        self._prefix_trace.mark_valid(position,len(ids))
      return output,values
    finally:
      setattr(self.stats,counter_phase+'_seconds',getattr(self.stats,counter_phase+'_seconds')+perf_counter()-start)
      if self._prepared_target is not None:self.stats.prepared_graphs = len(self._prepared_target.calls)

  def _verify(self, inputs:Sequence[int], position:int) -> tuple[TargetOutput,list[int]]:
    if not self.verify_first_three:return self._forward(inputs,position,'verify')
    # Final short fragments reuse the already-captured three-row graph.
    if len(inputs)<=3:return self._forward(inputs,position,'verify_first')
    first,greedy = self._forward(inputs[:3],position,'verify_first')
    # Logit row2 predicts input row3, which has not been consumed yet. All three
    # comparisons must pass before the same proposal continues from that row.
    if greedy!=list(inputs[1:4]):
      self.stats.verify_skipped_input_tokens += len(inputs)-3
      return first,greedy
    # Own the first rows before another target call. Generic eager adapters may
    # reuse result buffers, and a captured graph may be evicted before joining.
    # Preserve aliases and charge the actual clone/synchronization to verification.
    start = perf_counter()
    try:
      first = _map_outputs(first,lambda value:value.clone()).realize()
      _synchronize((first.logits,first.hidden,*first.features))
    finally:
      elapsed = perf_counter()-start
      self.stats.verify_join_seconds += elapsed
      self.stats.verify_seconds += elapsed
    rest,later = self._forward(inputs[3:],position+3,'verify_rest')
    start = perf_counter()
    try:
      joined = _join_outputs(first,rest)
      _synchronize((joined.logits,joined.hidden,*joined.features))
      return joined,greedy+later
    finally:
      elapsed = perf_counter()-start
      self.stats.verify_join_seconds += elapsed
      self.stats.verify_seconds += elapsed

  def _snapshot(self) -> _StateSnapshot:
    start=perf_counter()
    try:snapshot=_StateSnapshot(self.target)
    finally:self.stats.snapshot_seconds+=perf_counter()-start
    self.stats.snapshot_calls+=1
    self.stats.native_snapshot_calls+=int(snapshot.native is not None)
    self.stats.snapshot_bytes=max(self.stats.snapshot_bytes,snapshot.nbytes)
    return snapshot

  def _restore(self, snapshot:_StateSnapshot) -> None:
    start = perf_counter()
    self.stats.rollback_count += 1
    try:
      snapshot.restore()
      self.stats.native_snapshot_restores+=int(snapshot.native is not None)
    finally:self.stats.restore_seconds += perf_counter()-start

  def _restore_prefix(self, snapshot:_StateSnapshot, position:int, rows:int) -> bool:
    trace = self._prefix_trace
    trace_position,trace_rows = position,rows
    # The shared workspace contains only the most recent stage. Prefixes in
    # its tail restore directly; an earlier prefix uses the original snapshot
    # and the existing timed replay path, never stale first-stage history.
    if self.verify_first_three and trace is not None and trace.position==position+3 and rows>3:
      trace_position,trace_rows = position+3,rows-3
    if trace is None or not trace.can_restore(trace_position,trace_rows):
      self._restore(snapshot)
      return False
    start = perf_counter()
    self.stats.rollback_count += 1
    self.stats.trace_restore_calls += 1
    try:
      self.stats.trace_restore_bytes += trace.restored_bytes
      states = trace.restore(trace_position,trace_rows)
      _synchronize(states)
    except BaseException:
      trace.invalidate()
      raise
    finally:
      elapsed = perf_counter()-start
      self.stats.trace_restore_seconds += elapsed
      self.stats.restore_seconds += elapsed
    return True

  def _pick_token(self, output:TargetOutput, greedy:int, sampling:SamplingSelection|None, phase:str) -> int:
    if sampling is None:return greedy
    start=perf_counter()
    try:return sampling.next_token(output.logits)
    finally:setattr(self.stats,phase+'_seconds',getattr(self.stats,phase+'_seconds')+perf_counter()-start)

  def sample(self, prompt:Sequence[int], *, temperature:float, seed:int|None=None, **kwargs) -> Iterator[int]:
    """Sample from the target distribution using a trained conditional proposal walk."""
    from tinygrad.llm.speculative_sampling import SamplingSelection
    if self.verify_first_three:raise ValueError('Sampled verification does not support greedy first-three staging')
    if not callable(getattr(self.draft,'propose_sampled',None)):
      raise ValueError('The draft must expose conditional sampled proposal probabilities')
    selection=SamplingSelection(temperature,self.target.config.vocab_size,seed)
    return self.generate(prompt,_sampling=selection,**kwargs)

  def generate(self, prompt:Sequence[int], *, max_new_tokens:int, eos_token_ids:Iterable[int]=(),
               cancelled:Callable[[],bool]|None=None, prefill_chunk_size:int=128,
               verification_budget:int|None=None, ordinary_decode_seconds:float|None=None,
               max_speculative_extra_seconds:float|None=None,
               _sampling:SamplingSelection|None=None) -> Iterator[int]:
    """Yield verified greedy IDs. Close the iterator to promptly release the target.

    An optional extra compute budget can stop an expensive first block before
    the ordinary two-block rate check. It charges synchronized block work, not
    consumer pauses; the default retains the existing rate-only policy.
    """
    tokens = list(prompt)
    vocab = self.target.config.vocab_size
    eos_values = tuple(eos_token_ids)
    if not tokens or any(type(token) is not int or not 0<=token<vocab for token in tokens):raise ValueError('Invalid prompt token IDs')
    if len(tokens)>self.target.max_context:raise ValueError('Prompt exceeds target context')
    if type(max_new_tokens) is not int or max_new_tokens<0:raise ValueError('max_new_tokens must be a nonnegative integer')
    if type(prefill_chunk_size) is not int or prefill_chunk_size<1:raise ValueError('prefill_chunk_size must be positive')
    if any(type(token) is not int or not 0<=token<vocab for token in eos_values):raise ValueError('Invalid EOS token IDs')
    eos = set(eos_values)
    if verification_budget is not None and (type(verification_budget) is not int or not 0<verification_budget<self.draft.block_size):
      raise ValueError('verification_budget must be positive and smaller than the trained block size')
    verify_width = self.draft.block_size if verification_budget is None else verification_budget+1
    if max_speculative_extra_seconds is not None and ordinary_decode_seconds is None:
      raise ValueError('Extra compute budget requires an ordinary decode reference')
    policy = (DecodeFallbackPolicy(ordinary_decode_seconds,max_extra_seconds=max_speculative_extra_seconds)
              if ordinary_decode_seconds is not None else None)
    if _sampling is not None and self.verify_first_three:raise ValueError('Sampled verification requires complete proposal rows')
    if not self._lock.acquire(blocking=False):raise RuntimeError('Target is already used by another speculative iterator')
    started = perf_counter()
    self.stats = SpeculativeStats(sampled=_sampling is not None,sampling_temperature=_sampling.temperature if _sampling else 0.0,
                                  adaptive=policy is not None,ordinary_decode_seconds=policy.reference_seconds_per_token if policy else 0.0,
                                  max_speculative_extra_seconds=(policy.max_extra_seconds or 0.0) if policy else 0.0,
                                  early_anchor=self.early_anchor,prepared_target=self._prepared_target is not None,
                                  prepared_dynamic_rows=self._prepared_target.dynamic_rows if self._prepared_target is not None else False,
                                  prepared_prefill_dynamic=self._prepared_target.prefill_dynamic if self._prepared_target is not None else False,
                                  prepared_exact_replay=self._prepared_target.exact_replay if self._prepared_target is not None else False,
                                  prefix_trace=self._prefix_trace is not None,verify_first_three=self.verify_first_three,
                                  deferred_fallback=self.deferred_fallback,retain_fallback_prefix=self.retain_fallback_prefix,
                                  fallback_observation_batch_size=self.fallback_observation_batch_size)
    self.finish_reason = 'running'
    stop = cancelled or (lambda:False)
    failed = False
    fallback_batch:_FallbackFeatureBatch|None = None
    try:
      if self.retain_fallback_prefix and self.fallback_observation_batch_size>1:
        observer = getattr(self.draft,'observe_features',None)
        if not callable(observer):raise ValueError('Fallback batching requires feature-only observation')
        fallback_batch = _FallbackFeatureBatch(self.fallback_observation_batch_size,observer,self.stats)
      position = self.get_start_pos(tokens)
      if position==0:self._reset_draft()
      else:self._clear_prefix()
      self.stats.cached_tokens = position
      budget = min(max_new_tokens,self.target.max_context-len(tokens))
      if budget==0:
        self.finish_reason = 'length' if max_new_tokens==0 else 'context'
        return
      self.target._cached_tokens = tokens[:position]
      if self._prefix_trace is not None:
        self._prefix_trace.invalidate()
        start = perf_counter()
        try:
          self._prefix_trace.prepare()
          _synchronize(self._prefix_trace.buffers)
        finally:self.stats.trace_prepare_seconds += perf_counter()-start
        self.stats.trace_bytes = self._prefix_trace.nbytes
      chunk_size = self.target.prefill_chunk_size(prefill_chunk_size)
      self._prepared_prefill_capacity = chunk_size
      anchor = 0
      deferred_observation:tuple[TargetOutput,int]|None = None
      for offset in range(position,len(tokens),chunk_size):
        if stop():
          self.finish_reason = 'cancelled'
          return
        chunk = tokens[offset:offset+chunk_size]
        output,greedy = self._forward(chunk,offset,'prefill',last_only=True)
        position += len(chunk)
        self.target._cached_tokens = tokens[:position]
        if self.early_anchor and position==len(tokens):deferred_observation = (output,offset)
        else:self._observe(output,offset,prompt=True)
        anchor = self._pick_token(output,greedy[-1],_sampling if position==len(tokens) else None,'prefill')

      while self.stats.output_tokens<budget:
        if stop():
          self.finish_reason = 'cancelled'
          return
        if policy is not None and policy.fallback:
          if self.deferred_fallback:
            # This anchor was already selected from validated target logits (or
            # a sampling residual). Deliver it before computing its successor.
            # A closed/finished stream keeps the same pending-input contract as
            # ordinary generation, avoiding an unused forward and state copies.
            emitted = anchor
            self.stats.output_tokens += 1
            self.stats.fallback_tokens += 1
            self.stats.fallback_pending_last_token = True
            yield emitted
            if emitted in eos:
              self.finish_reason = 'eos'
              return
            if self.stats.output_tokens>=budget:
              self.finish_reason = 'context' if budget<max_new_tokens else 'length'
              return
            if stop():
              self.finish_reason = 'cancelled'
              return
            fallback_prefix = self.target._cached_tokens
            output,greedy = self._forward([emitted],position,'fallback',last_only=True)
            # This input was delivered before the forward. Even cancellation
            # during that call can retain its successfully completed state.
            self.target._cached_tokens = [*fallback_prefix,emitted]
            position += 1
            self.stats.fallback_pending_last_token = False
            if stop():
              self.finish_reason = 'cancelled'
              return
            if self.retain_fallback_prefix:self._observe_fallback(output,position-1,fallback_batch)
            if self.retain_fallback_prefix and stop():
              self.finish_reason = 'cancelled'
              return
            anchor = self._pick_token(output,greedy[-1],_sampling,'fallback')
            continue
          # The last delivered speculative input is already committed. Continue
          # from its validated next anchor under the same target lock; ordinary
          # generate would otherwise need another recurrent prefill.
          # Cancellation can arrive while this single target call is running.
          # Keep its previous recurrent state until the token is delivered.
          fallback_prefix = self.target._cached_tokens
          fallback_snapshot = self._snapshot() if cancelled is not None else None
          output,greedy = self._forward([anchor],position,'fallback',last_only=True)
          next_token=self._pick_token(output,greedy[-1],_sampling,'fallback')
          if stop():
            if fallback_snapshot is not None:self._restore(fallback_snapshot)
            self.target._cached_tokens = fallback_prefix
            self.finish_reason = 'cancelled'
            return
          self.target._cached_tokens = [*fallback_prefix,anchor]
          if self.retain_fallback_prefix:self._observe_fallback(output,position,fallback_batch)
          if self.retain_fallback_prefix and stop():
            if fallback_snapshot is not None:self._restore(fallback_snapshot)
            self.target._cached_tokens = fallback_prefix
            self.finish_reason = 'cancelled'
            return
          self.stats.output_tokens += 1
          self.stats.fallback_tokens += 1
          position += 1
          emitted,anchor = anchor,next_token
          yield emitted
          if emitted in eos:
            self.finish_reason = 'eos'
            return
          continue
        remaining = budget-self.stats.output_tokens
        prefix = list(self.target._cached_tokens)
        committed:list[int] = [anchor] if self.early_anchor else []
        delivered = 0
        credited = 0
        completed = False
        snapshot:_StateSnapshot|None = None
        cost_fields = ('draft_seconds','snapshot_seconds','verify_seconds','restore_seconds','replay_seconds','observe_seconds')
        cost_before = sum(getattr(self.stats,name) for name in cost_fields)
        try:
          if self.early_anchor:
            # The anchor already came from validated target logits. Keep this
            # pending input exclusive to the iterator until resume or cleanup.
            delivered = 1
            self.stats.output_tokens += 1
            yield anchor
            if anchor in eos:
              self.finish_reason = 'eos'
              return
            if self.stats.output_tokens>=budget:
              self.finish_reason = 'context' if budget<max_new_tokens else 'length'
              return
            if stop():
              self.finish_reason = 'cancelled'
              return
          if deferred_observation is not None:
            self._observe(*deferred_observation,prompt=True)
            deferred_observation = None
          followups:list[int] = []
          if remaining>1 and anchor not in eos:
            start = perf_counter()
            try:
              followups = list(self.draft.propose(anchor,position=position,max_tokens=min(remaining,verify_width)) if _sampling is None else
                _sampling.propose(self.draft,anchor,position=position,max_tokens=min(remaining,verify_width)))
            finally:self.stats.draft_seconds += perf_counter()-start
          if (len(followups)>min(remaining,verify_width)-1
              or any(type(token) is not int or not 0<=token<vocab for token in followups)):raise ValueError('Invalid draft token proposal')
          self.stats.proposed_tokens += len(followups)
          self.stats.blocks += 1
          inputs = [anchor,*followups]
          snapshot = self._snapshot()
          output,greedy = self._verify(inputs,position)
          if _sampling is None:
            accepted = 0
            for proposed,actual in zip(followups,greedy):
              if proposed!=actual:break
              accepted += 1
            rejected = accepted<len(followups)
          else:
            start=perf_counter()
            try:
              accepted,next_anchor,rejected=_sampling.verify(followups,output.logits,eos,
                need_bonus=remaining>len(inputs) and anchor not in eos)
              self.stats.sampling_acceptance_tests+=accepted+int(rejected)
              self.stats.sampling_residual_draws+=int(rejected)
            finally:self.stats.verify_seconds+=perf_counter()-start
          self.stats.rejected_blocks += int(rejected)
          committed = inputs[:1+accepted]
          if any(token in eos for token in committed):committed = committed[:next(i for i,token in enumerate(committed) if token in eos)+1]
          credited = len(committed)-1
          self.stats.accepted_tokens += credited
          if len(committed)!=len(inputs):
            if self._restore_prefix(snapshot,position,len(committed)):
              output = _map_outputs(output,lambda value:value[:,:len(committed)]).realize()
            else:output,greedy = self._forward(committed,position,'replay')
          self.target._cached_tokens = prefix+committed
          self._observe(output,position)
          # A sampled rejection's residual draw remains valid after replay. An
          # unconditional draw from replay logits would change its distribution.
          if _sampling is None:next_anchor = greedy[len(committed)-1]
          if policy is not None:
            cost = sum(getattr(self.stats,name) for name in cost_fields)-cost_before
            if policy.record(len(committed),cost):self.stats.fallback_reason = policy.reason
          completed = True
          for token in committed[delivered:]:
            if stop():
              self.finish_reason = 'cancelled'
              return
            delivered += 1
            self.stats.output_tokens += 1
            yield token
          position += len(committed)
          anchor = next_anchor
          if committed[-1] in eos:
            self.finish_reason = 'eos'
            return
        finally:
          # A closed stream can have an unconsumed early anchor or a verifier
          # that ran past its delivered prefix. Both require explicit, timed
          # target work. Speculative blocks keep their full-commit contract.
          if not completed or delivered<len(committed):
            self.stats.accepted_tokens -= max(0,credited-max(0,delivered-1))
            restored = self._restore_prefix(snapshot,position,delivered) if snapshot is not None else False
            if delivered and not restored:
              output,_ = self._forward(committed[:delivered],position,'replay',last_only=True)
              if self.reuse_prefix and self.finish_reason in ('length','eos','context'):
                if deferred_observation is not None:
                  self._observe(*deferred_observation,prompt=True)
                  deferred_observation = None
                if getattr(self.draft,'cached_prefix_length',None)==position:self._observe(output,position)
            self.target._cached_tokens = prefix+committed[:delivered]
      self.finish_reason = 'context' if budget<max_new_tokens else 'length'
    except GeneratorExit:
      self.finish_reason = 'closed'
      raise
    except BaseException:
      failed = True
      self.finish_reason = 'error'
      # A failed prefill may have updated only some recurrent layers. The next
      # request must start from zero rather than reuse its previous prefix claim.
      self.target._cached_tokens = []
      self.stats.fallback_pending_last_token = False
      raise
    finally:
      reset_attempted = False
      try:
        if fallback_batch is not None and not failed and self.finish_reason in ('length','eos','context'):
          if stop():self.finish_reason = 'cancelled'
          else:
            fallback_batch.flush()
            if stop():self.finish_reason = 'cancelled'
        if not (self.reuse_prefix and not failed and self.finish_reason in ('length','eos','context') and self._retain_prefix()):
          reset_attempted = True
          self._reset_draft()
      except BaseException:
        self.finish_reason = 'error'
        self._clear_prefix()
        if fallback_batch is not None and not reset_attempted:
          self.target._cached_tokens = []
          self.stats.fallback_pending_last_token = False
          try:self._reset_draft()
          except BaseException:pass  # Preserve the original cleanup failure.
        if not failed:raise  # A cleanup failure must not replace the primary error.
      finally:
        if fallback_batch is not None:fallback_batch.clear()
        self.stats.total_seconds = perf_counter()-started
        self._lock.release()
