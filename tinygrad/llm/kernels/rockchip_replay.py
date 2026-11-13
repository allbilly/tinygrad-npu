"""Native replay of prepared CPU/Rockchip inference calls, with guarded fallback."""
from __future__ import annotations
import ctypes, functools, itertools, json, os, subprocess, time, weakref
from collections import Counter
from dataclasses import replace
from typing import Any, Callable, cast
from tinygrad import Device
from tinygrad.device import Buffer, BufferStorage, TinyELF
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.jit import CapturedJit, _copy_input
from tinygrad.engine.realize import _resolve, estimate_uop, get_runtime, get_call_arg_uops, get_call_outs_ins
from tinygrad.helpers import DEBUG, PROFILE, ContextVar, GlobalCounters, ceildiv, getenv, unwrap
from tinygrad.renderer import Estimates
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.runtime.ops_rockchip import (RockchipBuffer, RockchipDevice, RockchipProgram, RockchipMatmul,
                                         RockchipGroupedMatmul, RockchipBatchedMatmul, RockchipRoutedMatmul,
                                         RockchipBucketedMatmul, _cmd, _FINITE_HALF_SOURCE)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import Ops, UOp, sym_infer


# ---------------------------------------------------------------------------
# Adjacent-submit merging (opt-in: ONPU_MERGE_SUBMITS=1 pairs, =2 pairs+triple).
#
# The decode graph issues two NPU submits back-to-back with no CPU call between
# them for gate+up (24576 main tile then the 2816 tail) and for ffn_down (the
# grouped 3-group k-chain then its 1536 k-tail); the lm_head issues three
# back-to-back shards. Every one of those submits owns exactly three one-per-
# core tasks and the same core mask/domain, so they can share ONE ioctl (and
# pay its fixed cost once) while keeping all their tasks.
#
# Mechanism (hardware-proven, scripts/probe_merged_submit.py in oNPU): the
# earlier plan's per-core terminating tail is patched in place into a PC jump
# to the later plan's body for the same core; a combined task array lists the
# descriptors core-major [A0,B0,A1,B1,A2,B2]; one ioctl submits all of them.
# The command bodies and all arithmetic are unchanged, so results are
# bit-identical; the group's bodies write disjoint output ranges, so a core may
# start body 2 while another core is still on body 1.
#
# The patch mutates the prepared plan, so every patched word is recorded and
# _restore_merged_tails() puts them back if the replay falls back to the
# ordinary executor. Thin plans/cross-domain groups are refused and fall back
# to one ioctl per plan.
# ---------------------------------------------------------------------------
ONPU_MERGE_BODY_LIMIT = 1024
# (id(plan), tail word offset) -> (plan, offset, FIRST recorded original tail words).
# A replay may be rebuilt for the same prepared plans; without recording the first
# original, a second build would record the jump words as "original" and a restore
# would leave the chain linked.
_MERGED_TAIL_ORIGINALS: dict[tuple[int,int],tuple[Any,int,list[int]]] = {}


def _restore_merged_tails() -> None:
  """Undo every in-place tail patch (executor fallback, or before a replay rebuild)."""
  for plan,offset,words in reversed(list(_MERGED_TAIL_ORIGINALS.values())):
    try:
      _write_words(plan.cmd,offset,words)
      plan.dev.sync(plan.cmd, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    except Exception:
      pass
  _MERGED_TAIL_ORIGINALS.clear()


def _write_words(cmd:RockchipBuffer, offset:int, words:list[int]) -> None:
  if not 0<=offset<=cmd.size//8-len(words): raise ValueError('word patch outside command buffer')
  for i,word in enumerate(words):
    ctypes.c_uint64.from_address(cmd.addr+(offset+i)*8).value = word


def _plan_merge_tails(plan:Any) -> list[tuple[int,int]]:
  """(body start word, body word count) of each of a plan's three one-per-core tasks."""
  if plan.tasks!=3 or plan.core_mask!=7: raise ValueError('merge requires three one-per-core tasks')
  descs=(rk.struct_rknpu_task*3).from_address(plan.task.addr)
  tails=[]
  for core in range(3):
    desc=descs[core]
    body=desc.regcfg_amount
    start=(desc.regcmd_addr-plan.cmd.dma)//8
    if not 1<=body<=ONPU_MERGE_BODY_LIMIT: raise ValueError('unexpected command body length')
    if desc.regcmd_addr%8 or start<0 or start+body+4>plan.cmd.size//8: raise ValueError('descriptor outside its command buffer')
    tails.append((start,body))
  return tails


def build_merged_submit(plans:list[Any]) -> tuple[dict,list[Any]]:
  """One combined PC submit for 2-3 plans, each owning three one-per-core tasks.

  Returns (submit descriptor for the statement, buffers that must stay alive).
  Patches the earlier plans' terminating command tails to jump to the next
  plan's body; on any failure the patches are restored before raising.
  """
  if not 2<=len(plans)<=3: raise ValueError('merge group must hold 2 or 3 plans')
  domain=plans[0].dev.domain
  if any(plan.dev.domain!=domain for plan in plans): raise ValueError('merge group crosses memory domains')
  tails=[_plan_merge_tails(plan) for plan in plans]
  group=len(plans)
  dev=plans[0].dev
  task=dev.allocate(max(4096,3*group*ctypes.sizeof(rk.struct_rknpu_task)),rk.RKNPU_MEM_KERNEL_MAPPING)
  patches: list[tuple[Any,int,list[int],bool]] = []  # (plan, offset, words before this build, newly tracked)
  try:
    # Combined task array: descriptors unchanged, interleaved core-major.
    for core in range(3):
      for position,plan in enumerate(plans):
        source=(rk.struct_rknpu_task*3).from_address(plan.task.addr)[core]
        ctypes.memmove(task.addr+(core*group+position)*ctypes.sizeof(rk.struct_rknpu_task),
                       ctypes.addressof(source),ctypes.sizeof(source))
    dev.sync(task, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    # Patch every plan's tail but the last one into a jump to the next plan's body.
    for position,plan in enumerate(plans[:-1]):
      following=plans[position+1]
      for core in range(3):
        start,body=tails[position][core]
        next_start,next_body=tails[position+1][core]
        offset=start+body
        key=(id(plan),offset)
        previous=[ctypes.c_uint64.from_address(plan.cmd.addr+(offset+i)*8).value for i in range(2)]
        tracked=key not in _MERGED_TAIL_ORIGINALS
        if tracked: _MERGED_TAIL_ORIGINALS[key]=(plan,offset,previous)
        try:
          _write_words(plan.cmd,offset,
                       [_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(following.cmd.dma+next_start*8)&0xfffffff0),
                        _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(next_body,2)+1)])
        except BaseException:
          _write_words(plan.cmd,offset,previous)
          if tracked: _MERGED_TAIL_ORIGINALS.pop(key,None)
          raise
        patches.append((plan,offset,previous,tracked))
      plan.dev.sync(plan.cmd, rk.RKNPU_MEM_SYNC_TO_DEVICE)
  except BaseException:
    for plan,offset,previous,tracked in reversed(patches):
      try:
        _write_words(plan.cmd,offset,previous)
        plan.dev.sync(plan.cmd, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      except Exception: pass
      if tracked: _MERGED_TAIL_ORIGINALS.pop((id(plan),offset),None)
    dev.free(task)
    raise
  return ({'tasks':3*group,'domain':domain,'obj_addr':int(task.meta.obj_addr),'core_mask':7,
           'subcores':((0,0),(0,0),(0,group),(group,group),(2*group,group))},[task])


def mergeable_plan(plan:Any) -> bool:
  """Only the verified decode geometry: single-row INT8, three one-per-core tasks, one domain."""
  return (isinstance(plan,(RockchipMatmul,RockchipGroupedMatmul))
          and not isinstance(plan,(RockchipBatchedMatmul,RockchipRoutedMatmul,RockchipBucketedMatmul))
          and plan.m==1 and bool(plan.int8) and not plan.dynamic_weights
          and not getattr(plan,'sdk_prefill',0)  # SDK prefill bodies use a different PC layout
          and plan.tasks==3 and plan.core_mask==7
          and tuple(plan.subcores)==((0,0),(0,0),(0,1),(1,1),(2,1)))


# ---------------------------------------------------------------------------
# Fast validity path (opt-in: ONPU_REPLAY_FAST_VALID=1; debug: ONPU_REPLAY_FAST_VALID_DEBUG=1).
#
# RockchipCapturedJit.__call__ runs NativeReplay.valid() for every decoded token.
# The authoritative check costs ~0.43 ms/token on ChatGLM3-6B decode: it walks
# every input size, every guarded intermediate storage, every weight version,
# every attention program and all ~171 plans. None of that state can change
# unless one of a handful of mutating operations runs, so the fast path caches
# the last success verdict against a global epoch and only re-runs the
# authoritative check when the epoch moves.
#
# Watched, not global: the epoch is bumped by
#   * prepare/close of the plan classes and RockchipAttentionProgram
#       -> plan.owned / plan.bufs identity / program liveness   (rare, global)
#   * Buffer.allocate/.deallocate of a *watched* guarded CPU buffer
#       -> the baked non-root CPU addresses used by the replay C  (guards check)
#   * RockchipAllocator._copyin into a *watched* NPU meta
#       -> weight meta.version / output cpu_clean_version checks
# plus same-file lifecycle points (new NativeReplay, free_intermediates,
# executor fallback). Unrelated per-token Buffer allocations do NOT bump the
# epoch, so steady-state decode keeps using the cached verdict.
#
# The per-input shape/device/size check is NEVER skipped: the replay C function
# reads through pointers passed on every call, so an input shape change must be
# caught even inside the epoch window. With the flag off, valid() is the
# original code unchanged and no hooks are installed. With _DEBUG=1 the
# authoritative check still runs on EVERY token and a disagreement with the fast
# verdict raises AssertionError (that is how an uncovered invalidation is found).
# ---------------------------------------------------------------------------
_FAST_VALID_EPOCH = 0
_FAST_VALID_HOOKS_INSTALLED = False
# Guarded CPU Buffer objects whose storage address is baked into the replay C.
_FAST_VALID_WATCHED_BUFFERS: "weakref.WeakSet[Any]" = weakref.WeakSet()
# id() of RockchipBuffer.meta objects whose version feeds a validity guard.
_FAST_VALID_WATCHED_META_IDS: set[int] = set()


def _bump_fast_valid_epoch() -> None:
  global _FAST_VALID_EPOCH
  _FAST_VALID_EPOCH += 1


def _register_fast_valid_watch(replay:Any) -> None:
  """Watch exactly the state this replay's validity depends on (called once per build)."""
  _FAST_VALID_WATCHED_BUFFERS.update(replay.guards.keys())
  _FAST_VALID_WATCHED_META_IDS.update(replay.weight_guards.keys())
  _FAST_VALID_WATCHED_META_IDS.update(id(meta) for meta in replay._output_guards)


def _install_fast_valid_hooks() -> bool:
  """Wrap every mutator that can invalidate a NativeReplay. Idempotent."""
  global _FAST_VALID_HOOKS_INSTALLED
  if _FAST_VALID_HOOKS_INSTALLED: return True
  import sys
  from tinygrad.runtime.ops_rockchip import (RockchipMatmul, RockchipBatchedMatmul, RockchipGroupedMatmul,
                                            RockchipRoutedMatmul, RockchipBucketedMatmul, RockchipAllocator)
  from tinygrad.runtime.support.rockchip_attention import RockchipAttentionProgram

  def hook_name(function:Any, name:str) -> Any:
    function._onpu_fast_valid_hook = name  # type: ignore[attr-defined]
    return function

  def invalidating(original:Any, name:str) -> Any:
    @functools.wraps(original)
    def wrapper(*args:Any, **kwargs:Any) -> Any:
      _bump_fast_valid_epoch()
      return original(*args, **kwargs)
    return hook_name(wrapper, name)

  plan_classes:list[Any] = [RockchipMatmul, RockchipBatchedMatmul, RockchipGroupedMatmul,
                            RockchipRoutedMatmul, RockchipBucketedMatmul, RockchipAttentionProgram]
  gpu_module = sys.modules.get('tinygrad.runtime.support.rockchip_gpu')
  if gpu_module is not None and (gpu_plan:=getattr(gpu_module,'RockchipGPUProgram',None)) is not None:
    plan_classes.append(gpu_plan)
  for cls in plan_classes:
    for method in ('prepare', 'close'):
      current = cls.__dict__.get(method)
      if current is not None and not getattr(current, '_onpu_fast_valid_hook', None):
        setattr(cls, method, invalidating(current, f'{cls.__name__}.{method}'))

  # Only watched mutations matter for Buffer storage identity and meta versions.
  def watched_buffer(original:Any, name:str) -> Any:
    @functools.wraps(original)
    def wrapper(self:Any, *args:Any, **kwargs:Any) -> Any:
      if self in _FAST_VALID_WATCHED_BUFFERS: _bump_fast_valid_epoch()
      return original(self, *args, **kwargs)
    return hook_name(wrapper, name)

  def watched_copyin(original:Any, name:str) -> Any:
    @functools.wraps(original)
    def wrapper(self:Any, dest:Any, src:Any) -> Any:
      if id(dest.meta) in _FAST_VALID_WATCHED_META_IDS: _bump_fast_valid_epoch()
      return original(self, dest, src)
    return hook_name(wrapper, name)

  for method in ('allocate', 'deallocate'):
    setattr(Buffer, method, watched_buffer(getattr(Buffer, method), f'Buffer.{method}'))
  RockchipAllocator._copyin = watched_copyin(RockchipAllocator._copyin, 'RockchipAllocator._copyin')
  _FAST_VALID_HOOKS_INSTALLED = True
  return True


class UnsupportedReplay(Exception): pass


# A shared pack arena can be far larger than the view an upload actually wrote.
# Aliasing the clean to the recorded view is safe only with an exact match, and
# the native-clean path can be disabled independently for diagnosis.
ONPU_REPLAY_EXTRA_PACK_ALIAS = ContextVar('ONPU_REPLAY_EXTRA_PACK_ALIAS',1)
ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN = ContextVar('ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN',1)


def _copy_scalar_input(u:UOp) -> UOp:
  """Snapshot a local CPU token without compiling a one-element copy graph."""
  if (getenv('ONPU_REPLAY_SCALAR_COPY',1) and u.device=='CPU' and u.dtype in (dtypes.int32,dtypes.int64)
      and u.op is Ops.BUFFER and u.max_numel()==1 and getattr(Device['CPU'],'remote',None) is None
      and isinstance(source:=u.buffer,Buffer)):
    new=UOp.new_buffer('CPU',1,u.dtype)
    cast(Buffer,new.buffer).ensure_allocated().as_memoryview(allow_zero_copy=True)[:] = source.as_memoryview(allow_zero_copy=True)
    return new
  return _copy_input(u)


def _attention_call(call:UOp) -> bool:
  return (call.device=='CPU' and call.body.op is Ops.PROGRAM and len(call.body.src)>=4
          and isinstance(binary:=call.body.src[3].arg,bytes) and len(binary)==32 and binary[:4]==b'RNPA')


def _full_write(call:UOp, slot:int, nbytes:int) -> bool:
  """Prove complete replacement, rather than inferring it from PROGRAM.outs."""
  if call.body.op is Ops.STORE: return slot==0
  if call.body.op is not Ops.PROGRAM or call.device!='CPU': return False
  program = call.body
  if slot not in program.arg.outs or slot in program.arg.ins: return False
  params = [p for p in program.kernel_params if p.arg.slot==slot and p.addrspace is not AddrSpace.ALU]
  if len(params)!=1 or params[0].max_numel()*params[0].dtype.itemsize!=nbytes: return False
  param = params[0]
  # These internal packers fill output 0 completely, including tile/header padding.
  # The exact extent check above also excludes a larger reused scratch allocation.
  # `rockchip_int8_swiglu_pack` can write a partial tile into a shared arena, so
  # trusting it to fill the whole view requires the pack alias to be enabled.
  # `rockchip_quantize_sdk` writes the SDK block layout, which only fills the
  # whole view under that same alias.
  if slot==0 and (program.src[0].arg.function_name in ('rockchip_swiglu_pack','rockchip_quantize','rockchip_int8_ffn_pack')
                 or (program.src[0].arg.function_name in ('rockchip_int8_swiglu_pack','rockchip_quantize_sdk',
                                                           'rockchip_int8_erf_gelu_pack','rockchip_int8_gelu_pack')
                     and ONPU_REPLAY_EXTRA_PACK_ALIAS
                     and (program.src[0].arg.function_name in ('rockchip_quantize_sdk','rockchip_int8_erf_gelu_pack',
                                                              'rockchip_int8_gelu_pack')
                          or getenv('ONPU_FFN_PACKED_UPLOAD',0)))):
    return True
  nodes = program.src[0].toposort()
  if any(u.op in (Ops.IF,Ops.BACKEDGE) for u in nodes): return False
  bounds = {}
  for u in nodes:
    if u.op is Ops.RANGE:
      bound = u.src[0].ssimplify()
      if not isinstance(bound,int) or bound<=0: return False
      bounds[u] = bound

  def affine(index:UOp) -> tuple[int,dict[UOp,int]]|None:
    def checked(offset:int, coefficients:dict[UOp,int]) -> tuple[int,dict[UOp,int]]|None:
      if index.dtype is not dtypes.weakint:
        low = offset+sum(c*(bounds[r]-1) for r,c in coefficients.items() if c<0)
        high = offset+sum(c*(bounds[r]-1) for r,c in coefficients.items() if c>0)
        if not dtypes.is_int(index.dtype) or low<index.dtype.min or high>index.dtype.max: return None
      return offset,coefficients
    if index.op is Ops.CONST and isinstance(index.val,int): return index.val,{}
    if index.op is Ops.RANGE and index in bounds: return checked(0,{index:1})
    if index.op is Ops.CAST and dtypes.is_int(index.dtype):
      value = affine(index.src[0])
      return None if value is None else checked(*value)
    if index.op not in (Ops.ADD,Ops.MUL,Ops.SHL): return None
    left,right = affine(index.src[0]),affine(index.src[1])
    if left is None or right is None: return None
    a,x = left
    b,y = right
    if index.op is Ops.ADD:
      return checked(a+b,{r:x.get(r,0)+y.get(r,0) for r in x.keys()|y.keys()})
    if index.op is Ops.SHL:
      if y or not 0<=b<32: return None
      b = 1<<b
    if y:
      if x: return None
      a,b,x = b,a,y
    return checked(a*b,{r:c*b for r,c in x.items()})

  intervals = []
  for store in nodes:
    if store.op is not Ops.STORE: continue
    dest = store.src[0]
    if dest.op not in (Ops.INDEX,Ops.SHRINK) or dest.src[0] is not param: continue
    if len(store.src)!=2 or len(dest.src)!=(2 if dest.op is Ops.INDEX else 3): return False
    width = 1 if dest.op is Ops.INDEX else dest.src[2].ssimplify()
    address = affine(dest.src[1])
    if not isinstance(width,int) or width<=0 or address is None: return False
    first,coefficients = address
    span = width
    # Each range must extend one contiguous interval without leaving holes.
    for stride,extent in sorted((abs(c),bounds[r]) for r,c in coefficients.items() if c):
      if stride>span: return False
      span += stride*(extent-1)
    first += sum(c*(bounds[r]-1) for r,c in coefficients.items() if c<0)
    if first<0 or first+span>param.max_numel(): return False
    intervals.append((first,first+span))
  end = 0
  for first,last in sorted(intervals):
    if first>end: return False
    end = max(end,last)
  return end==param.max_numel()


class NativeReplay:
  def __init__(self, linear:UOp, inputs:list[UOp], values:dict[str,int], cleaner:int|None=None):
    self.linear, self.inputs = linear, tuple(inputs)
    self.roots = [cast(Buffer,u.buffer).ensure_allocated() for u in inputs]
    if any(buf.device!='CPU' for buf in self.roots): raise UnsupportedReplay('non-CPU inputs')
    self.input_sizes = [buf.nbytes for buf in self.roots]
    self.variables = sorted(values)
    self.kept:list[Any] = []
    self.guards:dict[Buffer,BufferStorage] = {}
    self.writes:dict[int,Any] = {}
    self.write_counts:Counter[int] = Counter()
    self.weight_guards:dict[int,tuple[Any,int]] = {}
    self.plans:list[tuple[Any,tuple]] = []
    self.attention_programs:list[Any] = []
    self.mutated:set[int] = set()
    self.calls = [call.without_after for call in linear.src]
    if not any(call.op is Ops.CALL and (isinstance(call.device,str) and call.device.split(':')[0]=='ROCKCHIP' or _attention_call(call))
               for call in self.calls): raise UnsupportedReplay('no NPU work')
    self.dev = cast(RockchipDevice,Device['ROCKCHIP'])
    self.cleaner = cleaner
    self.clean_sites = 0
    self.driver_from_sites = 0
    self.invalidator:int|None = None
    self.invalidate_sites = 0
    self.npu_outputs:dict[int,tuple[int,int,int]] = {}
    self.pending_npu_outputs:set[tuple[int,int,int]] = set()
    self.borrow_batch = bool(getenv('ONPU_REPLAY_BORROW_BATCH',0))
    if getenv('ONPU_REPLAY_NATIVE_FROM',0):
      from tinygrad.llm.kernels.rockchip_cache import cache_invalidator
      try: library = cache_invalidator()
      except (OSError,subprocess.TimeoutExpired): pass
      else:
        self.kept.append(library)
        self.invalidator = ctypes.cast(library.clean_from,ctypes.c_void_p).value
    self.libc = ctypes.CDLL(None)
    ioctl = unwrap(ctypes.cast(self.libc.ioctl,ctypes.c_void_p).value)
    memcpy = unwrap(ctypes.cast(self.libc.memmove,ctypes.c_void_p).value)
    self.ioctl = f'((int(*)(int,unsigned long,...)){ioctl}ULL)'
    self.memcpy = f'((void*(*)(void*,const void*,unsigned long)){memcpy}ULL)'
    self.statements:list[str] = []
    self.borrow_outputs = bool(getenv('ONPU_REPLAY_BORROW_OUTPUTS',0))
    self.aliases,self.input_copies,self.output_copies = self.copy_aliases()
    self.alias_copies = self.input_copies | self.output_copies
    self.profiling = bool(getenv('ONPU_REPLAY_PROFILE',0))
    self.profile_ns:Counter[str] = Counter()
    self.profile_counts:Counter[str] = Counter()
    self.labels:list[str] = []
    self.estimates = Estimates()
    self.submits = 0
    self.gpu_calls = 0
    # ONPU_MERGE_SUBMITS: 0 off (default), 1 merge adjacent pairs, 2 pairs + the lm_head triple.
    self._merge_level = int(getenv('ONPU_MERGE_SUBMITS',0) or 0)
    self._merge_max_group = 2 if self._merge_level==1 else (3 if self._merge_level>=2 else 0)
    self._merge_slots: list[dict[str,Any]] = []
    self._merge_groups: list[list[int]] = []
    self._merge_run: list[int] = []
    self._last_merge_plan: Any = None
    # ONPU_REPLAY_FAST_VALID: 0 off (default), 1 epoch-gated authoritative check.
    # ONPU_REPLAY_FAST_VALID_DEBUG=1: run the authoritative check every token and assert agreement.
    self._fast_valid_debug = bool(getenv('ONPU_REPLAY_FAST_VALID_DEBUG',0))
    self._fast_valid = bool(getenv('ONPU_REPLAY_FAST_VALID',0)) or self._fast_valid_debug
    if self._fast_valid: _install_fast_valid_hooks()
    self._fast_valid_epoch: int|None = None
    self._fast_valid_full_runs = 0
    self._fast_valid_skips = 0
    self._fast_valid_mismatch = 0
    for index,call in enumerate(self.calls):
      if call.op is not Ops.CALL: raise UnsupportedReplay('non-call graph entry')
      if call.body.op is Ops.PROGRAM and call.device!='ROCKCHIP': self.pending_npu_outputs.clear()
      self.estimates += estimate_uop(call)
      start = len(self.statements)
      if call.body.op is Ops.STORE: self.copy(call,index)
      elif call.body.op is Ops.PROGRAM: self.kernel(call,index)
      else: raise UnsupportedReplay('unsupported call body')
      if (written:=self.npu_outputs.get(index)) is not None:
        meta,address,size = written
        self.pending_npu_outputs = {p for p in self.pending_npu_outputs
                                    if p[0]!=meta or p[1]+p[2]<=address or address+size<=p[1]}
        self.pending_npu_outputs.add(written)
      label = (f'{call.device}:{call.body.src[0].arg.function_name}' if call.body.op is Ops.PROGRAM else
               f'COPY:{call.src[2].device}->{call.src[1].device}')
      self.labels.append(label)
      if dump_path := os.getenv('ONPU_REPLAY_DUMP_DTYPES'):
        # Map opaque replay labels (E_*/r_*) back to their tensors: one JSONL row
        # per captured call with its input dtypes/shapes/numels. Opt-in only.
        info={'label':label,'ins':[]}
        for u in call.src[1:]:
          try:
            ins=info['ins']
            ins.append({'dtype':getattr(u.dtype,'name',str(u.dtype)),'shape':str(u.shape),
                        'numel':u.numel() if hasattr(u,'numel') else None})
          except Exception as exc: info['ins'].append({'err':str(exc)})
        with open(dump_path,'a') as f:
          f.write(json.dumps(info)+'\n')
      if self.profiling:
        self.statements[start:] = ['{U64 begin=now();']+self.statements[start:]+[f'stats[{index+2}]=now()-begin;}}']
    self._finalize_merged_submits()
    if not self.submits: raise UnsupportedReplay('no prepared NPU work')
    if self.mutated.intersection(self.weight_guards): raise UnsupportedReplay('mutable matrix weights')
    self.estimates = self.estimates.simplify()
    sym_infer(self.estimates.ops,values)
    sym_infer(self.estimates.mem,values)
    params = [*(f'void *input{i}' for i in range(len(inputs))),*(f'long long value{i}' for i in range(len(self.variables))),
              'unsigned long long *stats']
    clock = unwrap(ctypes.cast(self.libc.clock_gettime,ctypes.c_void_p).value)
    timing = (f'static inline U64 now(void){{long ts[2];((int(*)(int,void*)){clock}ULL)({time.CLOCK_MONOTONIC},ts);'
              'return (U64)ts[0]*1000000000ULL+ts[1];}') if self.profiling else ''
    statements = '\n'.join(self.statements)
    source = '''typedef unsigned int U32;typedef unsigned long long U64;
    struct Sync {U32 flags,reserved;U64 object,offset,size;} __attribute__((packed));
    struct Submit {U32 flags,timeout,start,number,counter;int priority;U64 object,regcfg,base,user;
                   U32 mask;int fence;U32 subcores[10];} __attribute__((packed));
    '''+_FINITE_HALF_SOURCE+timing+f'int replay({",".join(params)}) {{ stats[0]=stats[1]=0;\n'+statements+'\nreturn 0;}'
    cpu = Device['CPU']
    renderer = ClangRenderer(replace(cpu.renderer.target,renderer='CLANG'))
    self.program = cast(Any,cpu.runtime(TinyELF(renderer.compiler.compile_cached(source),'replay',renderer.target,())))
    self.program.fxn.restype = ctypes.c_int
    self.source = source
    if _FAST_VALID_HOOKS_INSTALLED:
      # Build the output-guard tuple here (one-off, not on the first token) and watch
      # exactly the state this replay's validity guards depend on.
      _register_fast_valid_watch(self)
      _bump_fast_valid_epoch()  # a new replay is new state for any other live replay

  def buffer(self, node:UOp) -> tuple[Buffer,Any,str]:
    # Symbolic views require their own pointer expression; leave them on the ordinary executor.
    if any(u.is_variable for u in node.toposort()): raise UnsupportedReplay('symbolic buffer view')
    buf = _resolve(node,self.inputs).buffer
    if not isinstance(buf,Buffer) or buf.device.split(':')[0] not in ('CPU','ROCKCHIP','PYTHON','NPY'):
      raise UnsupportedReplay('unsupported buffer device')
    buf.ensure_allocated()
    raw = buf.get_buf(buf.device)
    if isinstance(raw,RockchipBuffer): address = raw.addr
    elif isinstance(raw,int): address = raw
    else: raise UnsupportedReplay('unsupported raw buffer')
    self.kept.append(buf)
    for i,root in enumerate(self.roots):
      if buf.base is root.base: return buf,raw,f'((char*)input{i}+{buf.offset-root.offset})'
    self.guards[buf.base] = unwrap(buf.base._storage)
    return buf,raw,f'((void*){address}ULL)'

  def sync(self, raw:RockchipBuffer, flag:int, index:int) -> str:
    base_flags = rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
    offset = raw.dma-int(raw.meta.dma_addr)
    native_limit = 16777216 if getenv('ONPU_REPLAY_NATIVE_FROM_LARGE',0) else 4194304
    # A pack buffer may live in a large shared arena while the upload only wrote a
    # small view of it. Native-cleaning the whole view is only safe when an
    # exactly recorded upload proves that view was fully written, so the alias is
    # gated on an exact (meta, address, size) match rather than the arena size.
    uploaded = getattr(self,'extra_pack_uploads',{}).get(index)
    extra_pack = (bool(ONPU_REPLAY_EXTRA_PACK_ALIAS) and flag==rk.RKNPU_MEM_SYNC_TO_DEVICE
                  and self.cleaner is not None
                  and uploaded==(id(raw.meta),raw.addr,raw.size)
                  and raw.meta.flags==base_flags
                  and raw.meta.cpu_clean_version==raw.meta.version
                  and 0<raw.size<=1048576<=raw.meta.size<=16777216
                  and 0<=offset<=raw.meta.size-raw.size
                  and (raw.addr|raw.size)&63==0 and (raw.addr-offset)&4095==0)
    # No CPU kernel may intervene between the producer and this read. Batched
    # copies also forget any overlapping CPU upload. The output was clean at
    # warm replay, and NativeReplay.valid retains that invariant.
    if (self.invalidator is not None and flag==rk.RKNPU_MEM_SYNC_FROM_DEVICE and index>0
        and (self.npu_outputs.get(index-1)==(id(raw.meta),raw.addr,raw.size)
             or getattr(self,'borrow_batch',False) and (id(raw.meta),raw.addr,raw.size) in self.pending_npu_outputs)
        and raw.meta.flags==base_flags
        and raw.meta.cpu_clean_version==raw.meta.version
        and 0<raw.size<=raw.meta.size<=native_limit and 0<=offset<=raw.meta.size-raw.size
        and (raw.addr|raw.size)&63==0 and (raw.addr-offset)&4095==0):
      self.invalidate_sites += 1
      return (f'if(((int(*)(void*,unsigned long)){self.invalidator}ULL)((void*){raw.addr}ULL,{raw.size}UL))'
              f'return {2000000+index};')
    # The plain path keeps its 1 MB ceiling. The extra-pack alias handles larger
    # arenas up to 16 MB when an exact upload record proves the written view.
    if (self.cleaner is not None and flag==rk.RKNPU_MEM_SYNC_TO_DEVICE
        and raw.meta.flags in (base_flags,base_flags|rk.RKNPU_MEM_KERNEL_MAPPING)
        and 0<raw.size<=raw.meta.size<=1048576 and 0<=offset<=raw.meta.size-raw.size
        and (raw.addr|raw.size)&63==0 and (raw.addr-offset)&4095==0):
      self.clean_sites += 1
      return f'((void(*)(void*,unsigned long)){self.cleaner}ULL)((void*){raw.addr}ULL,{raw.size}UL);'

    if extra_pack and bool(ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN):
      # Proven pack view inside a larger arena: clean exactly the written span.
      # Reached only when the plain branch above declined (arena beyond 1 MB).
      self.clean_sites += 1
      return f'((void(*)(void*,unsigned long)){self.cleaner}ULL)((void*){raw.addr}ULL,{raw.size}UL);'
    if flag==rk.RKNPU_MEM_SYNC_FROM_DEVICE: self.driver_from_sites += 1
    request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_mem_sync)<<16)|(ord('d')<<8)|0x45
    return (f'{{struct Sync sync={{{flag},0,{raw.meta.obj_addr}ULL,{raw.dma-int(raw.meta.dma_addr)}ULL,{raw.size}ULL}};'
            f'if({self.ioctl}({self.dev.fd.fd},{request}UL,&sync)!=0)return {2000000+index};}}')

  def copy_aliases(self) -> tuple[dict,set[int],set[int]]:
    """Redirect private CPU pack outputs and adjacent read-only consumers into NPU mappings."""
    events = []
    copies = []
    for i,call in enumerate(self.calls):
      if call.op is not Ops.CALL or call.body.op not in (Ops.PROGRAM,Ops.STORE): continue
      outs,ins = get_call_outs_ins(call)
      slots = (range(2) if call.body.op is Ops.STORE else
               [p.arg.slot for p in call.body.kernel_params if p.addrspace is not AddrSpace.ALU])
      buffers = {}
      for slot in slots:
        node = call.src[1+slot]
        buf,raw,expr = self.buffer(node)
        key = (buf.base,buf.offset,buf.nbytes)
        event = (i,key,slot in outs,slot in ins,buf.device,node,raw,expr,slot)
        events.append(event)
        buffers[slot] = event
      if call.body.op is Ops.STORE and 0 in buffers and 1 in buffers: copies.append((buffers[0],buffers[1]))
    aliases,uploads,downloads = {},set(),set()
    redirected_ranges:dict[int,list[RockchipBuffer]] = {}
    def overlap(a,b): return a[0] is b[0] and a[1]<b[1]+b[2] and b[1]<a[1]+a[2]
    def scratch(event): return any(u.op is Ops.BUFFER and u.tag=='scratch' for u in event[5].toposort())
    def dead_after(key,index):
      future = [e for e in events if e[0]>index and overlap(e[1],key)]
      stale = [(key[1],key[1]+key[2])]
      for _,call_events in itertools.groupby(future,key=lambda e:e[0]):
        group = list(call_events)
        # A call can read and write overlapping views. Check every read before
        # letting proven writes kill the old contents of their exact ranges.
        if any(e[3] and any(start<e[1][1]+e[1][2] and e[1][1]<end for start,end in stale) for e in group): return False
        for e in group:
          if e[2] and _full_write(self.calls[e[0]],e[8],e[1][2]):
            first,last = e[1][1],e[1][1]+e[1][2]
            stale = [(a,b) for start,end in stale for a,b in ((start,min(end,first)),(max(start,last),end)) if a<b]
        if not stale: return True
      return True
    for dst,src in copies:
      i,key = src[:2]
      if (not isinstance(dst[6],RockchipBuffer) or src[4]!='CPU' or key[2]!=dst[1][2]
          or not scratch(src)): continue
      prior = [e for e in events if e[0]<i and overlap(e[1],key)]
      if not prior: continue
      writer = prior[-1]
      if (writer[1]!=key or not writer[2] or writer[3] or writer[4]!='CPU' or self.calls[writer[0]].device!='CPU'
          or writer[5].dtype not in (dtypes.half,dtypes.uint8)): continue
      if not _full_write(self.calls[writer[0]],writer[8],key[2]): continue
      if any(e[0]==writer[0] and e is not writer and overlap(e[1],key) for e in events): continue
      if not dead_after(key,i): continue
      if any(writer[0]<e[0]<i and isinstance(e[6],RockchipBuffer) and e[6].meta is dst[6].meta
             and e[6].addr<dst[6].addr+dst[1][2] and dst[6].addr<e[6].addr+e[1][2] for e in events): continue
      aliases[(writer[0],key)] = dst[7]
      redirected_ranges.setdefault(writer[0],[]).append(dst[6])
      uploads.add(i)
    if self.borrow_outputs:
      for dst,src in copies:
        i,key = dst[:2]
        raw = src[6]
        if (not isinstance(raw,RockchipBuffer) or dst[4]!='CPU' or key[2]!=src[1][2]
            or not scratch(dst) or not scratch(src) or any(key[0] is root.base for root in self.roots)
            or not raw.meta.flags & rk.RKNPU_MEM_CACHEABLE or raw.addr&63): continue
        consumer = i+1
        if getattr(self,'borrow_batch',False):
          while consumer<len(self.calls) and self.calls[consumer].body.op is Ops.STORE: consumer += 1
        if consumer>=len(self.calls) or self.calls[consumer].device!='CPU' or self.calls[consumer].body.op is not Ops.PROGRAM: continue
        # Redirected pack outputs may coexist with a borrowed input only when
        # their NPU cache-line ranges are disjoint. The planner can share an arena.
        # A batch may contain other copies before the synchronous CPU reader, but
        # neither its host destination nor its NPU source may change in between.
        if consumer in redirected_ranges and (not getenv('ONPU_FFN_PACKED_UPLOAD',0) or any(
            (buf.addr&~63)<((raw.addr+raw.size+63)&~63) and (raw.addr&~63)<((buf.addr+buf.size+63)&~63)
            for buf in redirected_ranges[consumer])):continue
        following = [e for e in events if e[0]>i and overlap(e[1],key)]
        if any(e[0]<consumer for e in following): continue
        if any(i<e[0]<consumer and e[2] and isinstance(e[6],RockchipBuffer) and e[6].meta is raw.meta
               and e[6].addr<raw.addr+raw.size and raw.addr<e[6].addr+e[1][2] for e in events): continue
        readers = [e for e in following if e[0]==consumer]
        if not readers or any(e[1]!=key or e[2] or not e[3] for e in readers): continue
        if not dead_after(key,consumer): continue
        later = [e for e in following if e[0]>consumer]
        # Keep output borrowing on the stricter COPY-only scratch-reuse policy.
        if later and self.calls[later[0][0]].body.op is not Ops.STORE: continue
        aliases[(consumer,key)] = src[7]
        downloads.add(i)
    return aliases,uploads,downloads

  def copy(self, call:UOp, index:int):
    self._last_merge_plan = getattr(self,'_last_merge_plan',None)
    nodes = get_call_arg_uops(call)
    if len(nodes)!=2: raise UnsupportedReplay('copy signature')
    dst,d,dp = self.buffer(nodes[0])
    src,s,sp = self.buffer(nodes[1])
    if dst.nbytes!=src.nbytes: raise UnsupportedReplay('copy sizes')
    if isinstance(s,RockchipBuffer): self.statements.append(self.sync(s,rk.RKNPU_MEM_SYNC_FROM_DEVICE,index))
    if index not in self.alias_copies: self.statements.append(f'{self.memcpy}({dp},{sp},{dst.nbytes}UL);')
    if isinstance(d,RockchipBuffer):
      self.pending_npu_outputs = {p for p in self.pending_npu_outputs
                                  if p[0]!=id(d.meta) or p[1]+p[2]<=d.addr or d.addr+d.size<=p[1]}
      self.statements.append(self.sync(d,rk.RKNPU_MEM_SYNC_TO_DEVICE,index))
      self.writes[id(d.meta)] = d.meta
      self.write_counts[id(d.meta)] += 1
      self.mutated.add(id(d.meta))

  def scalar(self, param:UOp, call:UOp) -> tuple[str,str]:
    if not (dtypes.is_int(param.dtype) or param.dtype is dtypes.bool): raise UnsupportedReplay('noninteger scalar')
    typ = ('unsigned ' if dtypes.is_unsigned(param.dtype) or param.dtype is dtypes.bool else '')+'long long'
    bound = {u.expr:u.arg.val for u in call.src[1:] if u.is_bound_var}
    if param.is_variable and param.expr not in bound:
      if param.expr not in self.variables: raise UnsupportedReplay('unknown scalar variable')
      return typ,f'value{self.variables.index(param.expr)}'
    value = bound[param.expr] if param.is_variable else _resolve(call.src[1+param.arg.slot],self.inputs).val
    return typ,f'(({typ}){int(value)})'

  def kernel(self, call:UOp, index:int):
    # getattr: tests call kernel() on NativeReplay.__new__ instances with no __init__.
    previous_merge_plan = getattr(self,'_last_merge_plan',None)
    self._last_merge_plan = None
    device = call.device
    if not isinstance(device,str) or device.split(':')[0] not in ('CPU','ROCKCHIP'): raise UnsupportedReplay('kernel device')
    runtime = get_runtime(device,call.body)
    self.kept.append(runtime)
    raw,expressions,types = [],[],[]
    for param in call.body.kernel_params:
      if param.addrspace is AddrSpace.ALU:
        typ,value = self.scalar(param,call)
        types.append(typ)
        expressions.append(value)
      else:
        buf,buffer,value = self.buffer(call.src[1+param.arg.slot])
        value = self.aliases.get((index,(buf.base,buf.offset,buf.nbytes)),value)
        raw.append(buffer)
        types.append('void*')
        expressions.append(value)
    from tinygrad.runtime.support.rockchip_attention import RockchipAttentionProgram
    if isinstance(runtime,RockchipAttentionProgram):
      if len(raw)!=5 or any(isinstance(buf,RockchipBuffer) for buf in raw):raise UnsupportedReplay('NPU attention CPU signature')
      if runtime.closed or not runtime.owned:raise UnsupportedReplay('NPU attention resources released')
      self.attention_programs.append(runtime)
      self.statements.append(runtime.emit(expressions,index))
      self.submits+=2
      return
    if isinstance(runtime,CPUProgram):
      if any(isinstance(buf,RockchipBuffer) for buf in raw): raise UnsupportedReplay('CPU kernel touches NPU mapping')
      if runtime.lvp or runtime.dev.remote is not None: raise UnsupportedReplay('CPU execution mode')
      address = unwrap(ctypes.cast(runtime.fxn,ctypes.c_void_p).value)
      self.statements.append(f'((void(*)({",".join(types)})){address}ULL)({",".join(expressions)});')
      return
    from tinygrad.runtime.support.rockchip_gpu import RockchipGPUProgram
    if isinstance(runtime,RockchipGPUProgram):
      self.pending_npu_outputs.clear()
      if len(raw)!=6 or not all(isinstance(buf,RockchipBuffer) for buf in raw):raise UnsupportedReplay('GPU FFN signature')
      gpu_key = runtime.key(tuple(raw))
      if (gpu_plan:=runtime.bindings.get(gpu_key)) is None or not gpu_plan.owned:raise UnsupportedReplay('GPU FFN binding not prepared')
      gpu_out,gpu_raw = cast(RockchipBuffer,raw[0]),cast(RockchipBuffer,raw[1])
      gpu_scales,gpu_second = cast(RockchipBuffer,raw[4]),cast(RockchipBuffer,raw[5])
      if any(buf.meta.cpu_clean_version!=buf.meta.version for buf in (gpu_out,gpu_scales,gpu_second)):
        raise UnsupportedReplay('dirty GPU FFN output mapping')
      self.kept.append(gpu_plan)
      self.plans.append((gpu_plan,gpu_plan.bufs))
      self.statements.append(self.sync(gpu_raw,rk.RKNPU_MEM_SYNC_FROM_DEVICE,index))
      self.statements.append(gpu_plan.emit(4000000+index))
      for output in (gpu_out,gpu_scales,gpu_second):
        self.statements.append(self.sync(output,rk.RKNPU_MEM_SYNC_FROM_DEVICE,index))
        self.writes[id(output.meta)] = output.meta
        self.write_counts[id(output.meta)] += 1
        self.mutated.add(id(output.meta))
      self.gpu_calls += 1
      return
    if not isinstance(runtime,RockchipProgram) or len(raw)!=3 or not all(isinstance(buf,RockchipBuffer) for buf in raw):
      raise UnsupportedReplay('NPU kernel signature')
    key = tuple((id(buf.meta),buf.addr,buf.size) for buf in raw)
    if (plan:=runtime.bindings.get(key)) is None: raise UnsupportedReplay('NPU binding not prepared')
    out,lhs,rhs = raw
    if out.meta.cpu_clean_version!=out.meta.version: raise UnsupportedReplay('dirty NPU output mapping')
    if plan.weight_version!=(rhs.meta.obj_addr,rhs.meta.version): raise UnsupportedReplay('unverified matrix weights')
    if plan.dynamic_weights:
      self.statements.append(f'if(!all_finite_half((const unsigned short*){rhs.addr+plan.rhs_offset}ULL,'
        f'{getattr(plan,"weight_groups",plan.groups)*plan.n*plan.k}UL))'
        f'return {5000000+index};')
    else:self.weight_guards[id(rhs.meta)] = rhs.meta,rhs.meta.version
    self.mutated.add(id(out.meta))
    self.kept.append(plan)
    self.plans.append((plan,plan.bufs))
    if plan.int8:
      scale_rows=plan.storage_rows if isinstance(plan,RockchipBucketedMatmul) else plan.groups*plan.m
      scale_address=lhs.addr+getattr(plan,'scale_offset',0)
      self.statements.append(f'for(int r=0;r<{scale_rows};r++){{float s=((const float*){scale_address}ULL)[r];'
        f'unsigned b=((const unsigned*){scale_address}ULL)[r];if(!(s>0.0f) || (b&0x7f800000)==0x7f800000)return {1000000+index};}}')
    else:
      self.statements.append(f'if(!all_finite_half((const unsigned short*){lhs.addr+plan.lhs_offset}ULL,{plan.groups*plan.m*plan.k}UL))'
        f'return {1000000+index};')
    if getattr(self,'_merge_max_group',0) and mergeable_plan(plan):
      # Defer this ioctl: a following adjacent NPU plan with the same geometry can
      # share it. The placeholder is replaced in _finalize_merged_submits().
      slot = len(self._merge_slots)
      token = f'/*MERGE-SLOT-{slot}-{index}*/'
      self._merge_slots.append({'plan':plan,'index':index,'token':token})
      self.statements.append(token)
      if (previous_merge_plan is not None and len(self._merge_run)<self._merge_max_group
          and mergeable_plan(previous_merge_plan) and previous_merge_plan.dev.domain==plan.dev.domain):
        self._merge_run.append(slot)
      else:
        self._merge_run=[slot]
        self._merge_groups.append(self._merge_run)
      self._last_merge_plan = plan
      self.npu_outputs[index] = (id(out.meta),out.addr,out.size)
      return
    if isinstance(plan,RockchipRoutedMatmul):
      # Validate the entire routing header before changing any command address.
      self.statements.append(f'{{const int *ids=(const int*){lhs.addr+plan.selector_offset}ULL;'
        f'for(int g=0;g<{plan.groups};g++)if((unsigned)ids[g]>={plan.experts}U)return {3000000+index};')
      for address,group in plan.weight_patches:
        self.statements.append(f'*(unsigned long long*){address}ULL=(0x201ULL<<48)|'
          f'(((unsigned long long)({rhs.dma}U+(unsigned)ids[{group}]*{plan.n*plan.k}U))<<16)|0x1110ULL;')
      self.statements.append('}')
      self.statements.append(self.sync(plan.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE,index))
    if isinstance(plan,RockchipBucketedMatmul):
      address=unwrap(ctypes.cast(plan.updater.fxn,ctypes.c_void_p).value)
      state=ctypes.addressof(plan.dynamic_state)
      self.statements.append(f'if(((int(*)(void*,unsigned long long,unsigned long long,unsigned long long,'
        f'unsigned long long,const int*,unsigned*)){address}ULL)((void*){plan.cmd.addr}ULL,{plan.cmd.dma}ULL,'
        f'{lhs.dma}ULL,{rhs.dma}ULL,{out.dma}ULL,(const int*){lhs.addr+plan.count_offset}ULL,'
        f'(unsigned*){state}ULL))return {3000000+index};')
      self.statements.append(self.sync(plan.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE,index))
    # The warm replay cleaned initial output pages. Subsequent CPU writes are
    # COPY destinations followed immediately by TO_DEVICE; NPU writes need FROM_DEVICE only.
    # Ordinary command descriptors are immutable; routed addresses were cleaned above.
    subcores:list[int|str] = [value for pair in (*plan.subcores,*((0,0),)*(5-len(plan.subcores))) for value in pair]
    fields:list[int|str] = [5,6000,0,plan.tasks,0,0,f'{plan.task.meta.obj_addr}ULL',plan.dev.domain,0,0,plan.core_mask,-1]
    tasks=str(plan.tasks)
    if isinstance(plan,RockchipBucketedMatmul):
      state=ctypes.addressof(plan.dynamic_state)
      tasks=f'((const unsigned*){state}ULL)[0]'
      fields[3],fields[10]=tasks,f'((const unsigned*){state}ULL)[1]'
      subcores=[f'((const unsigned*){state}ULL)[{i}]' for i in range(2,12)]
    request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_submit)<<16)|(ord('d')<<8)|0x41
    self.statements.append('{struct Submit submit={'+','.join(map(str,fields))+',{'+','.join(map(str,subcores))+'}};'
      f'if({self.ioctl}({self.dev.fd.fd},{request}UL,&submit)!=0)return {2000000+index};'
      f'stats[0]++;stats[1]+={tasks};}}')
    self.submits += 1
    self.npu_outputs[index] = (id(out.meta),out.addr,out.size)

  def _submit_code(self, tasks:Any, obj_addr:Any, domain:int, core_mask:Any, subcore_pairs:Any, index:int) -> str:
    subcores = [value for pair in subcore_pairs for value in pair]
    fields = [5,6000,0,tasks,0,0,f'{obj_addr}ULL',domain,0,0,core_mask,-1]
    request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_submit)<<16)|(ord('d')<<8)|0x41
    return ('{struct Submit submit={'+','.join(map(str,fields))+',{'+','.join(map(str,subcores))+'}};'
      f'if({self.ioctl}({self.dev.fd.fd},{request}UL,&submit)!=0)return {2000000+index};'
      f'stats[0]++;stats[1]+={tasks};}}')

  def _single_submit_code(self, plan:Any, index:int) -> str:
    pairs = (*plan.subcores,*((0,0),)*(5-len(plan.subcores)))
    return self._submit_code(str(plan.tasks),plan.task.meta.obj_addr,plan.dev.domain,plan.core_mask,pairs,index)

  def _finalize_merged_submits(self) -> None:
    # Patches are left in place across replay rebuilds: a different JIT (prefill)
    # may build its replay between two decode trials while these plans stay in
    # use, so restoring here would unlink a live merged replay. _MERGED_TAIL_ORIGINALS
    # keeps the first original of every patched word so a fallback restore is exact.
    for group in self._merge_groups:
      slots = [self._merge_slots[slot] for slot in group]
      code: str|None = None
      if len(slots)>1:
        try:
          info,keep = build_merged_submit([slot['plan'] for slot in slots])
        except Exception:
          info = None
        else:
          self.kept.extend(keep)
          code = self._submit_code(info['tasks'],info['obj_addr'],info['domain'],info['core_mask'],info['subcores'],slots[-1]['index'])
      if code is None:
        # Unsupported group: fall back to one ioctl per plan, exactly as the default path.
        for slot in slots:
          self.statements[self.statements.index(slot['token'])] = self._single_submit_code(slot['plan'],slot['index'])
          self.submits += 1
        continue
      for slot in slots[:-1]: self.statements[self.statements.index(slot['token'])] = ''
      self.statements[self.statements.index(slots[-1]['token'])] = code
      self.submits += 1

  @functools.cached_property
  def _output_guards(self) -> tuple[Any,...]:
    # Plans retain immutable binding tuples. Multiple output views can share
    # one allocation and therefore the same CPU-clean/version state.
    return tuple({id(buf.meta):buf.meta for _,buffers in self.plans
                  for buf in ((buffers[0],buffers[4],buffers[5]) if len(buffers)==6 else (buffers[0],))}.values())

  def _valid_deduplicated(self, inputs:list[UOp]) -> bool:
    if len(inputs)!=len(self.roots):return False
    for u,size in zip(inputs,self.input_sizes):
      if not isinstance(buf:=u.buffer,Buffer) or buf.device!='CPU' or buf.nbytes!=size:return False
    for buf,storage in self.guards.items():
      if buf._storage is not storage:return False
    for meta,version in self.weight_guards.values():
      if meta.version!=version:return False
    for program in self.attention_programs:
      if program.closed or not program.owned:return False
    for plan,buffers in self.plans:
      # Preparing again replaces this tuple even if the buffer values compare
      # equal; the old command addresses must then be rebuilt as well.
      if not plan.owned or plan.bufs is not buffers:return False
    for meta in self._output_guards:
      if meta.cpu_clean_version!=meta.version:return False
    return True

  def _valid_inputs(self, inputs:list[UOp]) -> bool:
    """Authoritative input shape check: the replay reads through these pointers on every call."""
    if len(inputs)!=len(self.roots):return False
    for u,size in zip(inputs,self.input_sizes):
      if not isinstance(buf:=u.buffer,Buffer) or buf.device!='CPU' or buf.nbytes!=size:return False
    return True

  def _valid_authoritative(self, inputs:list[UOp]) -> bool:
    if getenv('ONPU_REPLAY_DEDUP_GUARDS',1):return self._valid_deduplicated(inputs)
    return (len(inputs)==len(self.roots) and all(isinstance(u.buffer,Buffer) and u.buffer.device=='CPU'
      and u.buffer.nbytes==size for u,size in zip(inputs,self.input_sizes))
      and all(buf._storage is storage for buf,storage in self.guards.items())
      and all(meta.version==version for meta,version in self.weight_guards.values())
      and all(not program.closed and program.owned for program in self.attention_programs)
      and all(all(buf.meta.cpu_clean_version==buf.meta.version for buf in
        ((plan.bufs[0],plan.bufs[4],plan.bufs[5]) if len(plan.bufs)==6 else (plan.bufs[0],))) for plan,_ in self.plans if plan.bufs)
      and all(plan.owned and plan.bufs==buffers for plan,buffers in self.plans))

  def _valid_epoch_cached(self, inputs:list[UOp]) -> bool:
    """Fast path: authoritative input-shape check, then reuse the last success until the epoch moves."""
    if not self._valid_inputs(inputs):return False
    epoch = _FAST_VALID_EPOCH
    if epoch!=self._fast_valid_epoch:
      self._fast_valid_full_runs += 1
      if not self._valid_authoritative(inputs):return False
      self._fast_valid_epoch = epoch
      return True
    self._fast_valid_skips += 1
    return True

  def valid(self, inputs:list[UOp]) -> bool:
    if not self._fast_valid:
      if getenv('ONPU_REPLAY_DEDUP_GUARDS',1):return self._valid_deduplicated(inputs)
      return (len(inputs)==len(self.roots) and all(isinstance(u.buffer,Buffer) and u.buffer.device=='CPU'
        and u.buffer.nbytes==size for u,size in zip(inputs,self.input_sizes))
        and all(buf._storage is storage for buf,storage in self.guards.items())
        and all(meta.version==version for meta,version in self.weight_guards.values())
        and all(not program.closed and program.owned for program in self.attention_programs)
        and all(all(buf.meta.cpu_clean_version==buf.meta.version for buf in
          ((plan.bufs[0],plan.bufs[4],plan.bufs[5]) if len(plan.bufs)==6 else (plan.bufs[0],))) for plan,_ in self.plans if plan.bufs)
        and all(plan.owned and plan.bufs==buffers for plan,buffers in self.plans))
    fast = self._valid_epoch_cached(inputs)
    if not self._fast_valid_debug:return fast
    # Debug mode: the authoritative check runs on every token; a disagreement means
    # an invalidation path is not covered by _install_fast_valid_hooks (or that the
    # fast path is too strict, which is also a bug).
    full = self._valid_authoritative(inputs)
    if full==fast:return fast
    self._fast_valid_mismatch += 1
    raise AssertionError(f'ONPU_REPLAY_FAST_VALID_DEBUG mismatch on replay {id(self)}: '
                         f'fast={fast} authoritative={full} (an invalidation path is unhooked)')

  def __call__(self, inputs:list[UOp], values:dict[str,int]):
    ops,mem = sym_infer(self.estimates.ops,values),sym_infer(self.estimates.mem,values)
    pointers = [ctypes.c_void_p(cast(Buffer,u.buffer).ensure_allocated().get_buf('CPU')) for u in inputs]
    scalars = [ctypes.c_int64(values[name]) for name in self.variables]
    stats = (ctypes.c_uint64*(2+len(self.calls) if self.profiling else 2))()
    with self.dev.lock:
      self.dev.check_healthy()
      result = self.program.fxn(*pointers,*scalars,stats)
      self.dev.submit_count += stats[0]
      self.dev.task_count += stats[1]
      if result>=7000000: raise RuntimeError(f'NPU attention CPU dispatch failed at call {result-7000000}')
      if result>=6000000: raise ValueError(f'NPU attention length or active rows are out of range at call {result-6000000}')
      if result>=5000000: raise ValueError(f'Native replay requires finite dynamic matrix weights at call {result-5000000}')
      if result>=4000000:
        self.dev.poisoned = True
        raise RuntimeError(f'Native OpenCL FFN failed at call {result-4000000}')
      if result>=3000000: raise ValueError(f'NPU native replay expert routing metadata is out of range at call {result-3000000}')
      if result>=2000000:
        self.dev.poisoned = True
        raise RuntimeError(f'NPU native replay submission failed at call {result-2000000}')
      if result: raise ValueError(f'NPU native replay requires finite activations at call {result-1000000}')
      for key,meta in self.writes.items():
        meta.version += self.write_counts[key]
        meta.cpu_clean_version = meta.version
    GlobalCounters.kernel_count += len(self.calls)
    GlobalCounters.global_ops += ops
    GlobalCounters.global_mem += mem
    if self.profiling:
      for i,label in enumerate(self.labels):
        self.profile_ns[label] += stats[i+2]
        self.profile_counts[label] += 1
      # Per-call samples, opt-in. The aggregate above sums each label's calls, so
      # it can say a kernel averages X ms but not how its individual calls spread.
      # For a tile whose cost may depend on position in the layer (a tail tile
      # stalling behind a wide sibling) the distribution is the whole question.
      if path := os.getenv('ONPU_REPLAY_PROFILE_OUT'):
        with open(path,'a') as output:
          output.write(json.dumps({'labels':self.labels,'ns':dict(self.profile_ns),'counts':dict(self.profile_counts),
                                   'submits':stats[0],'tasks':stats[1]})+'\n')
      if path := os.getenv('ONPU_REPLAY_CALLS_OUT'):
        with open(path,'a') as output:
          output.write(json.dumps({'calls':[[label,stats[i+2]]
                                            for i,label in enumerate(self.labels)]})+'\n')


class RockchipCapturedJit(CapturedJit):
  def __init__(self, captured:CapturedJit, cleaner:int|None=None, *,
               program_transform:Callable[[UOp],UOp]|None=None):
    # The preparation-pool path rewrites the linear graph before capture (RWKV
    # state, GDN pre-pool, exact-row preparation). Accept and apply it here so
    # the caller does not have to special-case its absence.
    linear = program_transform(captured._linear) if program_transform is not None else captured._linear
    super().__init__(captured.ret,linear,captured.expected_names,captured.expected_input_info)
    self.native:NativeReplay|None = None
    self.warmed = False
    self.reason:str|None = None
    self.native_runs = 0
    self.cleaner = cleaner

  def __reduce__(self): return CapturedJit,(self.ret,self._linear,self.expected_names,self.expected_input_info)

  def free_intermediates(self):
    self.native,self.warmed,self.reason = None,False,None
    if _FAST_VALID_HOOKS_INSTALLED: _bump_fast_valid_epoch()
    super().free_intermediates()

  def __call__(self, input_uops:list[UOp], var_vals:dict[str,int]):
    if DEBUG>=2 or PROFILE or self.reason is not None: return super().__call__(input_uops,var_vals)
    warming = not self.warmed
    if warming:
      result = super().__call__(input_uops,var_vals)
      self.warmed = True
    concrete = [_copy_scalar_input(u) if u in self._jit_bufs else u for u in input_uops]
    if self.native is not None and not self.native.valid(concrete): self.native = None
    if self.native is None:
      try: self.native = NativeReplay(self.linear,concrete,var_vals,self.cleaner)
      except (UnsupportedReplay,KeyError,AssertionError) as exc:
        # A native replay that merged submits in place mutates prepared command tails;
        # the ordinary executor must never inherit those links.
        _restore_merged_tails()
        if _FAST_VALID_HOOKS_INSTALLED: _bump_fast_valid_epoch()  # executor fallback: replay state left the native path
        self.reason = str(exc)
        return result if warming else super().__call__(input_uops,var_vals)
    # Preparing a replay does not execute it. Keep the warmup's state update and
    # result exactly once, with compilation complete before the next request.
    if warming:return result
    self.native(concrete,var_vals)
    self.native_runs += 1
    for tensor,uop,values in self._symbolic_ret:
      tensor.uop = uop.substitute({v:v.bind(var_vals.get(v.expr,i)) for v,i in values.items()},walk=True)
    return self.ret


# Install the mutator hooks at import when the fast validity path is requested
# (env is set before the benchmark process starts, so import-time is the right place).
if getenv('ONPU_REPLAY_FAST_VALID',0) or getenv('ONPU_REPLAY_FAST_VALID_DEBUG',0): _install_fast_valid_hooks()
