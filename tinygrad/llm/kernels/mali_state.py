"""Bit-preserving FP32 Mali state snapshots with one pack/unpack dispatch."""
from __future__ import annotations
import functools, math, weakref
from dataclasses import replace
from tinygrad import Device,Tensor,TinyJit,dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.uop.ops import Ops,UOp


def _storage(state:Tensor) -> UOp:
  storage=state.uop
  while storage.op in (Ops.RESHAPE,Ops.AFTER):storage=storage.src[0]
  return storage


def supported(states:tuple[Tensor,...]) -> bool:
  if not 1<=len(states)<=96:return False
  device=states[0].device
  if not isinstance(device,str) or device.split(':')[0]!='CL':return False
  storages:list[UOp]=[]
  for state in states:
    if state.device!=device or state.dtype is not dtypes.float or not all(isinstance(n,int) and n>0 for n in state.shape):return False
    storage=_storage(state)
    if storage.op is not Ops.BUFFER or storage.numel()!=state.numel() or any(storage is old for old in storages):return False
    storages.append(storage)
  name=getattr(Device[device],'device_name','')
  return sum(math.prod(state.max_shape) for state in states)<2**31 and isinstance(name,str) and name.startswith('Mali')


def _branches(lengths:tuple[int,...], *, unpack:bool) -> str:
  offsets=[0]
  for length in lengths:offsets.append(offsets[-1]+length)
  def branch(first:int,end:int) -> str:
    if end-first==1:
      pointer=f'state{first}[i-{offsets[first]}]'
      return f'{pointer}=packed[i];' if unpack else f'packed[i]={pointer};'
    middle=(first+end)//2
    return f'if(i<{offsets[middle]}){{{branch(first,middle)}}}else{{{branch(middle,end)}}}'
  return f'if(i<{offsets[-1]}){{{branch(0,len(lengths))}}}'


@functools.cache
def _pack(packed:UOp, states:tuple[UOp,...], lengths:tuple[int,...]) -> UOp:
  parameters=''.join(f',__global const uint *state{i}' for i in range(len(states)))
  source=f'''__kernel void mali_state_pack(__global uint *packed{parameters}) {{
    uint i=get_global_id(0);
    {_branches(lengths,unpack=False)}
  }}'''
  return _program('mali_state_pack',source,(packed,*states),(ceildiv(sum(lengths),256),1,1),(256,1,1))


@functools.cache
def _unpack(states:tuple[UOp,...], packed:UOp, lengths:tuple[int,...]) -> UOp:
  parameters=','.join(f'__global uint *state{i}' for i in range(len(states)))
  source=f'''__kernel void mali_state_unpack({parameters},__global const uint *packed) {{
    uint i=get_global_id(0);
    {_branches(lengths,unpack=True)}
  }}'''
  program=_program('mali_state_unpack',source,(*states,packed),(ceildiv(sum(lengths),256),1,1),(256,1,1))
  return program.replace(arg=replace(program.arg,outs=tuple(range(len(states))),ins=(len(states),)))


class _StateCopyPlan:
  def __init__(self, states:tuple[Tensor,...]):
    self.states=states
    self.lengths=tuple(int(state.numel()) for state in states)
    self.pack=TinyJit(self._pack_into)
    self.unpack=TinyJit(self._restore_from)

  def _pack_into(self, backup:Tensor) -> None:
    # Return no captured input Tensor: every snapshot owns its caller-provided
    # backup, including after JIT capture and input rebinding.
    backup.custom_kernel(*self.states,fxn=lambda out,*args:_pack(out,args,self.lengths))[0].realize()

  def _restore_from(self, backup:Tensor) -> None:
    outputs=self.states[0].custom_kernel(*self.states[1:],backup,
      fxn=lambda *args:_unpack(tuple(args[:-1]),args[-1],self.lengths))
    Tensor.realize(*outputs[:len(self.states)])


_plans:weakref.WeakValueDictionary[tuple[UOp,...],_StateCopyPlan]=weakref.WeakValueDictionary()


class PackedStateSnapshot:
  """Own an independent backup; share only the copy plan, never a rollback slot."""
  def __init__(self, states:tuple[Tensor,...]):
    if not supported(states):raise ValueError('Native state snapshot requires whole independent F32 Mali buffers')
    device=states[0].device
    assert isinstance(device,str)
    self.device,self.states=device,states
    self._key=tuple(_storage(state) for state in states)
    plan=_plans.get(self._key)
    if plan is None:
      plan=_StateCopyPlan(states)
      _plans[self._key]=plan
    self._plan=plan
    self.backup=Tensor.empty(sum(plan.lengths),device=self.device,dtype=dtypes.float).realize()
    # Empty tensors have storage identities before allocation. The JIT binds
    # allocated inputs only, so allocate this independent destination first.
    backup_buffer=self.backup.uop.buffer
    assert isinstance(backup_buffer,Buffer)
    backup_buffer.ensure_allocated()
    plan.pack(self.backup)
    Device[self.device].synchronize()
    self.nbytes=self.backup.nbytes()

  def restore(self) -> None:
    if any(_storage(state) is not storage for state,storage in zip(self.states,self._key)):
      raise ValueError('State storage changed after the snapshot')
    self._plan.unpack(self.backup)
    Device[self.device].synchronize()
