from __future__ import annotations
import collections, ctypes, mmap, os, time, weakref
from dataclasses import replace
import numpy as np
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, suppress_finalizing, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RockchipRenderer, decode_image, patch_stage, emit_ew_stage,
  RKArg, RKGather, RKHostAddress, RKEWOp, _MAX_EW_ELEMS_FP16, _EW_CFG, _EW_CFG_FLOOR, _EW_STAGE_FP32_OUT)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer
from tinygrad.uop.ops import Ops

_PC_TAIL, _CMD_BUF_MIN, _TASK_BUF_MIN = 4, 65536, 16384
_CMD_PREFETCH_GUARD = mmap.PAGESIZE
_PC_DATA_AMOUNT_MAX = (1 << 16) - 1
_SUBMIT_TIMEOUT_MS = max(1, int(os.getenv("ROCKCHIP_SUBMIT_TIMEOUT_MS", "6000")))
_MAX_EW_GROUP_OPS = 48
_MAX_FP32_EW_GROUP_OPS = 16
_SAFE_TILE = 2048
_TASK_DESC_BYTES = ctypes.sizeof(rk.struct_rknpu_task)
_BO_FLAGS = rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
_TARGET_DPU, _TARGET_DPU_RDMA = 0x1001, 0x2001
_UNSAFE_FP16_EW_CHAINS:dict[tuple[int|None, ...], int] = {
  tuple(_EW_CFG[op] for op in (Ops.FDIV, Ops.ADD, Ops.MUL)):2,
  (_EW_CFG_FLOOR, *tuple(_EW_CFG[op] for op in (Ops.SUB, Ops.MUL, Ops.ADD, Ops.MUL, Ops.ADD, Ops.MUL, Ops.ADD, Ops.MUL,
    Ops.ADD, Ops.MUL, Ops.ADD, Ops.ADD, Ops.MAX, Ops.SUB, Ops.SUB, Ops.MAX, Ops.SUB, Ops.MUL))):1}

def _pc(target:int, reg:int, value:int=0) -> int: return (target << 48) | ((value & 0xffffffff) << 16) | reg
def _align_up(value:int, alignment:int) -> int: return (value + alignment - 1) & ~(alignment - 1)
def _task_command_bytes(body_qwords:int) -> int: return _align_up(body_qwords + _PC_TAIL, 2) * 8
def _rearm_body(body:tuple[int, ...]) -> tuple[int, ...]:
  """Clear hidden DPU/RDMA ping-pong phase before an independent physical submission."""
  precision = _body_precision(body)
  dst_stride = 1 << 4 if precision is not None and precision[1] in (4, 5) else 0
  return (_pc(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x30),
          _pc(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30),
          _pc(_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, dst_stride), *body)
def _body_precision(body:tuple[int, ...]) -> tuple[int, int]|None:
  value = next(((word >> 16) & 0xffffffff for word in body
                if word >> 48 == _TARGET_DPU and word & 0xffff == rk.REG_DPU_DATA_FORMAT), None)
  return None if value is None else (value & 7, value >> 29 & 7)
def _body_ew_cfg(body:tuple[int, ...]) -> int|None:
  return next(((word >> 16) & 0xffffffff for word in body
               if word >> 48 == _TARGET_DPU and word & 0xffff == rk.REG_DPU_EW_CFG), None)
def _pcchain_sizes(body_qwords:list[int]) -> tuple[int, int]:
  """Exact command and descriptor BO sizes for one PC-chain, including prefetch guard."""
  if not body_qwords or any(not 0 < amount <= _PC_DATA_AMOUNT_MAX for amount in body_qwords): raise ValueError("invalid EW PC-chain body")
  return sum(_task_command_bytes(amount) for amount in body_qwords)+_CMD_PREFETCH_GUARD, len(body_qwords)*_TASK_DESC_BYTES

class RockchipAllocator(LRUAllocator['RockchipDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._gpu_alloc(size)
  def _copyin(self, dest:HCQBuffer, src:memoryview):
    ctypes.memmove(int(dest.va_addr), from_mv(src), src.nbytes)
    self.dev._sync_buffer(dest, rk.RKNPU_MEM_SYNC_TO_DEVICE)
  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev._sync_buffer(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    ctypes.memmove(from_mv(dest), int(src.va_addr), dest.nbytes)
  def _as_buffer(self, src:HCQBuffer):
    self.dev._sync_buffer(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    return to_mv(int(src.va_addr), src.size)
  def _offset(self, buf:HCQBuffer, size:int, offset:int): return buf.offset(offset, size)
  def _free(self, buf:HCQBuffer, options:BufferSpec): self.dev._gpu_free(buf)

class RockchipProgram(Program['RockchipDevice']):
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.dev, self.name, self.image = dev, obj.name, decode_image(obj.lib)
    self._scratch_offsets:list[int] = []
    self._scratch_size = 0
    for spec in self.image.scratch:
      self._scratch_size = _align_up(self._scratch_size, 4096)
      self._scratch_offsets.append(self._scratch_size)
      self._scratch_size += spec.size
    self._scratch_arena:HCQBuffer|None = None
    self.scratch:tuple[HCQBuffer, ...] = ()
    self.submit_count = 0
    self._cmd_buf:HCQBuffer|None = None
    self._task_buf:HCQBuffer|None = None
    self._standalone_cmd_buf:HCQBuffer|None = None
    self._standalone_task_buf:HCQBuffer|None = None
    self._standalone_body:tuple[int, ...]|None = None
    self._pcchain_bodies:tuple[tuple[int, ...], ...]|None = None
    self._scratch_ew_bodies:dict[tuple[RKEWOp, ...], tuple[tuple[int, ...], ...]] = {}
    dev._touch_program(self)
    self._ensure_scratch()

  def _ensure_scratch(self) -> None:
    if self._scratch_arena is not None or not self._scratch_size: return
    self._scratch_arena = self.dev._gpu_alloc(self._scratch_size)
    self.scratch = tuple(self._scratch_arena.offset(offset, spec.size)
      for offset,spec in zip(self._scratch_offsets, self.image.scratch))

  def _release_resources(self) -> None:
    self.scratch = ()
    for attr in ("_scratch_arena", "_cmd_buf", "_task_buf", "_standalone_cmd_buf", "_standalone_task_buf"):
      if (buf:=getattr(self, attr, None)) is not None:
        setattr(self, attr, None)
        self.dev._gpu_free(buf)
    self._pcchain_bodies = None
    self._standalone_body = None
    getattr(self, "_scratch_ew_bodies", {}).clear()

  @suppress_finalizing
  def __del__(self):
    self._release_resources()
    self.dev._forget_program(self)

  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)

  def _ensure_buffer(self, attr:str, size:int, minimum:int, flags:int=0, replace_buffer:bool=False) -> HCQBuffer:
    if (buf:=getattr(self, attr)) is None or buf.size < size or replace_buffer:
      new = self.dev._gpu_alloc(max(size, minimum), flags)
      setattr(self, attr, new)
      if buf is not None: self.dev._gpu_free(buf)
      return new
    return buf

  def _submit(self, cmd:HCQBuffer, task:HCQBuffer, n:int, standalone:bool=False) -> None:
    subcores = ((0, n),) if standalone else ((0, n), (n, 0), (n, 0))
    for buffer in (cmd, task): self.dev._sync_buffer(buffer, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    try:
      rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
        flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=_SUBMIT_TIMEOUT_MS,
        task_start=0, task_number=n, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
        regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
        subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*x) for x in subcores)))
    except TimeoutError as exc:
      # The 0.9.8 driver already soft-resets on timeout. Retrying after a failed IOMMU reset can panic the kernel.
      self.dev._poisoned = True
      raise RuntimeError("RKNPU submit timed out; platform NPU reset or power cycle required") from exc
    self.submit_count += 1
    self.dev.submit_count += 1
    self.dev.task_count += n

  def _submit_pcchain(self, bodies:list[tuple[int, ...]]) -> None:
    """Submit contiguous FP16 EW tasks as one blocking PC chain."""
    if (split:=_UNSAFE_FP16_EW_CHAINS.get(tuple(map(_body_ew_cfg, bodies)))) is not None:
      for group in (bodies[:split], bodies[split:]): self._submit_pcchain(group)
      return
    first, last = _body_precision(bodies[0]), _body_precision(bodies[-1])
    if first is not None and (self.dev._ew_precision, first[0]) in ((4, 1), (4, 2)): self.dev.reset_npu()
    bodies = [_rearm_body(bodies[0]), *bodies[1:]]
    packed_bodies, n = tuple(bodies), len(bodies)
    if self._pcchain_bodies == packed_bodies and self._cmd_buf is not None and self._task_buf is not None:
      self._submit(self._cmd_buf, self._task_buf, n)
      if last is not None: self.dev._ew_precision = last[1]
      return
    cmd_size, task_need = _pcchain_sizes([len(body) for body in bodies])
    offsets:list[int] = []
    words = 0
    for body in bodies:
      offsets.append(words)
      words += _align_up(len(body) + _PC_TAIL, 2)
    need = words * 8
    assert cmd_size == need+_CMD_PREFETCH_GUARD
    # The RKNPU submission path can retain object/address state after a blocking submit. Keep an identical chain cached,
    # but never overwrite its command or descriptor GEM with different bytes.
    replace_buffer = self._pcchain_bodies is not None
    cmd = self._ensure_buffer("_cmd_buf", cmd_size, _CMD_BUF_MIN, replace_buffer=replace_buffer)
    task = self._ensure_buffer("_task_buf", task_need, _TASK_BUF_MIN, rk.RKNPU_MEM_KERNEL_MAPPING, replace_buffer)
    ctypes.memset(int(cmd.va_addr), 0, cmd_size)
    base_dma = self._dma(cmd)
    for i, body in enumerate(bodies):
      base = offsets[i]
      ctypes.memmove(int(cmd.va_addr) + base*8, (ctypes.c_uint64 * len(body))(*body), len(body)*8)
      # REGISTER_AMOUNTS=0 terminates the chain. Keep its speculative base-address fetch inside the mapped
      # zero-filled guard page: RK3588 can otherwise race completion with an IOMMU read from address zero.
      next_addr = (base_dma + (offsets[i+1] if i+1 < n else words)*8) & 0xfffffff0
      next_amount = len(bodies[i+1]) if i+1 < n else 0
      tail = (_pc(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, next_addr),
              _pc(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, next_amount),
              _pc(rk.TARGET_VERSION, 0), _pc(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
      ctypes.memmove(int(cmd.va_addr) + (base+len(body))*8, (ctypes.c_uint64 * _PC_TAIL)(*tail), _PC_TAIL*8)
      desc = rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(body)+_PC_TAIL, 0, base_dma+base*8)
      ctypes.memmove(int(task.va_addr) + i*_TASK_DESC_BYTES, ctypes.addressof(desc), _TASK_DESC_BYTES)
    self._pcchain_bodies = packed_bodies
    self._submit(cmd, task, n)
    if last is not None: self.dev._ew_precision = last[1]

  def _submit_standalone(self, body:tuple[int, ...]) -> None:
    """Submit one stateful DPU stage with the direct PC tail used by the vendor examples."""
    body = _rearm_body(body)
    if self._standalone_body == body and self._standalone_cmd_buf is not None and self._standalone_task_buf is not None:
      self._submit(self._standalone_cmd_buf, self._standalone_task_buf, 1, standalone=True)
      return
    commands = body + (_pc(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18),)
    cmd_size, task_need = len(commands)*8+_CMD_PREFETCH_GUARD, _TASK_DESC_BYTES
    replace_buffer = self._standalone_body is not None and self._standalone_body != body
    cmd = self._ensure_buffer("_standalone_cmd_buf", cmd_size, _CMD_BUF_MIN, replace_buffer=replace_buffer)
    task = self._ensure_buffer("_standalone_task_buf", task_need, _TASK_BUF_MIN, rk.RKNPU_MEM_KERNEL_MAPPING, replace_buffer)
    ctypes.memset(int(cmd.va_addr), 0, cmd_size)
    ctypes.memmove(int(cmd.va_addr), (ctypes.c_uint64 * len(commands))(*commands), len(commands)*8)
    desc = rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(commands), 0, self._dma(cmd))
    ctypes.memmove(int(task.va_addr), ctypes.addressof(desc), _TASK_DESC_BYTES)
    self._standalone_body = body
    self._submit(cmd, task, 1, standalone=True)

  def _run_int32_conversion(self, op:RKEWOp, address, buffer) -> None:
    """Convert aligned four-lane atoms on DPU; host movement preserves raw lane representations."""
    to_int32 = op.int32_output
    if op.rhs.kind is not RKBufferKind.SCRATCH or (to_int32 and op.lhs.kind is not RKBufferKind.SCRATCH):
      raise RuntimeError("INT32 EW conversion requires scratch input and tile arena")
    source, tiles, dest = buffer(op.lhs.kind, op.lhs.index), buffer(op.rhs.kind, op.rhs.index), buffer(op.dst.kind, op.dst.index)
    src_itemsize, dst_itemsize = (2, 1 if op.bool_output else 4) if to_int32 else (4, 2)
    source_need = op.lhs.addend+op.count*src_itemsize
    tile_need = op.rhs.addend+(op.count+3)//4*64
    dest_need = op.dst.addend+op.count*dst_itemsize
    if source_need > source.size or tile_need > tiles.size or dest_need > dest.size:
      raise RuntimeError(f"INT32 conversion exceeds buffer: source {source_need}/{source.size}, "
                         f"tiles {tile_need}/{tiles.size}, destination {dest_need}/{dest.size}; "
                         f"slots lhs={op.lhs.index} rhs={op.rhs.index} dst={op.dst.index}, "
                         f"scratch={tuple(spec.size for spec in self.image.scratch)}")
    self.dev._sync_buffer(source, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    ctypes.memset(int(tiles.va_addr), 0, tiles.size)
    for tile,start in enumerate(range(0, op.count, 4)):
      ctypes.memmove(int(tiles.va_addr)+op.rhs.addend+tile*64, int(source.va_addr)+op.lhs.addend+start*src_itemsize,
                     min(4, op.count-start)*src_itemsize)
    self.dev._sync_buffer(tiles, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self.dev.reset_npu()
    bodies:list[tuple[int, ...]] = []
    command_bytes = 0
    for start in range(0, op.count, 4):
      count = min(4, op.count-start)
      tile_arg = RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+start//4*64)
      stage = emit_ew_stage(tile_arg, tile_arg, tile_arg, count, op.ew_cfg, stateful=True,
                            int32_output=to_int32, int32_input=not to_int32)
      body = patch_stage(stage, address)
      if bodies and command_bytes+_task_command_bytes(len(body)) > mmap.PAGESIZE:
        self._submit_pcchain(bodies)
        self.dev.reset_npu()
        bodies.clear()
        command_bytes = 0
      bodies.append(body)
      command_bytes += _task_command_bytes(len(body))
    if bodies: self._submit_pcchain(bodies)
    self.dev._sync_buffer(tiles, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    for tile,start in enumerate(range(0, op.count, 4)):
      lanes = min(4, op.count-start)
      if op.bool_output:
        for lane in range(lanes):
          ctypes.memmove(int(dest.va_addr)+op.dst.addend+start+lane, int(tiles.va_addr)+op.rhs.addend+tile*64+lane*4, 1)
      else:
        ctypes.memmove(int(dest.va_addr)+op.dst.addend+start*dst_itemsize, int(tiles.va_addr)+op.rhs.addend+tile*64,
                       lanes*dst_itemsize)
    self.dev._sync_buffer(dest, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self.dev.reset_npu()

  def _run_ew_ops(self, address, buffer, ops:tuple[RKEWOp, ...]|None=None, *, tile_groups:bool=True) -> None:
    ops = self.image.ew_ops if ops is None else ops
    if not ops: return
    scratch_int16 = bool(ops) and all(op.int16_input and op.int16_output and not op.submit_barrier and not op.compare and
      op.dst.kind is RKBufferKind.SCRATCH and op.lhs.kind is RKBufferKind.SCRATCH and op.rhs.kind is RKBufferKind.SCRATCH for op in ops)
    if scratch_int16:
      if (cached:=self._scratch_ew_bodies.get(ops)) is None:
        stages = []
        for op in ops:
          for start in range(0, op.count, _MAX_EW_ELEMS_FP16):
            count, offset = min(_MAX_EW_ELEMS_FP16, op.count-start), start*2
            stages.append(patch_stage(emit_ew_stage(
              RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset), RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
              RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg,
              stateful=True, int16_output=True, int16_input=True), address))
        self._scratch_ew_bodies[ops] = cached = tuple(stages)
      for start in range(0, len(cached), _MAX_EW_GROUP_OPS): self._submit_pcchain(list(cached[start:start+_MAX_EW_GROUP_OPS]))
      return
    if tile_groups:
      groups:list[tuple[RKEWOp, ...]] = []
      start = 0
      for i,op in enumerate(ops):
        if i > start and op.submit_barrier:
          groups.append(ops[start:i])
          start = i
      groups.append(ops[start:])
      spatial = tuple(group[0].stateful and group[0].count > _MAX_EW_ELEMS_FP16 and
        all(op.count == group[0].count and not (op.compare or op.int16_input or op.int16_output or op.int32_input or
                                               op.int32_output or op.ew_cfg & _EW_STAGE_FP32_OUT) for op in group)
        for group in groups)
      sequential = tuple(group[0].stateful and len(group) > _MAX_EW_GROUP_OPS and max(op.count for op in group) <= _MAX_EW_ELEMS_FP16 and
                         not any(op.int32_input or op.int32_output or op.ew_cfg & _EW_STAGE_FP32_OUT for op in group)
                         for group in groups)
      if any(tiled or split for tiled,split in zip(spatial, sequential)):
        for group,tiled,split in zip(groups, spatial, sequential):
          if split:
            for chunk_start in range(0, len(group), _MAX_EW_GROUP_OPS):
              chunk = list(group[chunk_start:chunk_start+_MAX_EW_GROUP_OPS])
              chunk[0] = replace(chunk[0], submit_barrier=False, stateful=True)
              self._run_ew_ops(address, buffer, tuple(chunk), tile_groups=False)
          elif not tiled:
            self._run_ew_ops(address, buffer, group, tile_groups=False)
          else:
            for tile_start in range(0, group[0].count, _MAX_EW_ELEMS_FP16):
              count, offset = min(_MAX_EW_ELEMS_FP16, group[0].count-tile_start), tile_start*2
              tile_bodies = [patch_stage(emit_ew_stage(
                RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset),
                RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
                RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg,
                stateful=op.stateful or i == 0), address) for i,op in enumerate(group)]
              for start in range(0, len(tile_bodies), _MAX_EW_GROUP_OPS):
                self._submit_pcchain(tile_bodies[start:start+_MAX_EW_GROUP_OPS])
        return
    bodies:list[tuple[int, ...]] = []
    body_precision = 0
    def append_body(body:tuple[int, ...]) -> None:
      bodies.append(body)
      if len(bodies) == _MAX_EW_GROUP_OPS:
        self._submit_pcchain(bodies)
        bodies.clear()
    for i, op in enumerate(ops):
      if op.submit_barrier and bodies:
        self._submit_pcchain(bodies)
        bodies.clear()
        body_precision = 0
      if op.ew_cfg & _EW_STAGE_FP32_OUT:
        if any(not later.ew_cfg & _EW_STAGE_FP32_OUT for later in ops[i+1:]):
          raise RuntimeError("FP32 EW output must be terminal")
        if bodies: self._submit_pcchain(bodies)
        stages = [patch_stage(emit_ew_stage(later.dst, later.lhs, later.rhs, later.count, later.ew_cfg), address)
                  for later in ops[i:]]
        for start in range(0, len(stages), _MAX_FP32_EW_GROUP_OPS):
          self._submit_pcchain(stages[start:start+_MAX_FP32_EW_GROUP_OPS])
          self.dev.reset_npu()
        bodies.clear()
        break
      if op.int16_input and op.int32_output:
        if op.int16_output or op.int32_input: raise RuntimeError("conflicting INT16→INT32 EW precision")
        if op.dst.kind is RKBufferKind.ARG and i != len(ops)-1:
          raise RuntimeError("INT32 argument output must be terminal")
        if bodies and body_precision not in (0, 16):
          self._submit_pcchain(bodies)
          bodies.clear()
        stages = []
        for start in range(0, op.count, 8):
          count = min(8, op.count-start)
          stages.append(patch_stage(emit_ew_stage(
            RKArg(op.dst.kind, op.dst.index, op.dst.addend+start*4),
            RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+start*2),
            RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+start*2), count, op.ew_cfg,
            stateful=True, int32_output=True, int16_input=True), address))
        for body in stages: append_body(body)
        body_precision = 0
        continue
      if op.int16_input and op.int16_output or op.int32_input and op.int32_output:
        precision = 16 if op.int16_input else 32
        if bodies and body_precision != precision:
          self._submit_pcchain(bodies)
          bodies.clear()
        body_precision, itemsize = precision, precision//8
        limit = _MAX_EW_ELEMS_FP16 if precision == 16 else _MAX_EW_ELEMS_FP16//2
        for start in range(0, op.count, limit):
          count, offset = min(limit, op.count-start), start*itemsize
          stage = emit_ew_stage(RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset),
                                RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
                                RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg,
                                stateful=True, int32_output=precision == 32, int32_input=precision == 32,
                                int16_output=precision == 16, int16_input=precision == 16)
          append_body(patch_stage(stage, address))
        continue
      if op.int32_input or op.int32_output:
        if op.int32_output and op.dst.kind is RKBufferKind.ARG and i != len(ops)-1:
          raise RuntimeError("INT32 argument output must be terminal")
        if bodies:
          self._submit_pcchain(bodies)
          bodies.clear()
        self._run_int32_conversion(op, address, buffer)
        body_precision = 0
        continue
      if op.int16_input:
        raise RuntimeError("mixed INT16 EW conversion is unsupported")
      if body_precision:
        self._submit_pcchain(bodies)
        self.dev.reset_npu()
        bodies.clear()
        body_precision = 0
      if op.compare:
        if bodies:
          self._submit_pcchain(bodies)
          bodies.clear()
        for start in range(0, op.count, _SAFE_TILE):
          count = min(_SAFE_TILE, op.count-start)
          offset = start*2
          stage = emit_ew_stage(RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset),
                                RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
                                RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg,
                                compare=op.compare)
          self.dev.reset_npu()
          self._submit_standalone(patch_stage(stage, address))
          self.dev.reset_npu()
        continue
      for start in range(0, op.count, _MAX_EW_ELEMS_FP16):
        count = min(_MAX_EW_ELEMS_FP16, op.count-start)
        offset = start*2
        stage = emit_ew_stage(RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset),
                              RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
                              RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg,
                              stateful=op.stateful or op.int16_output or not bodies, int16_output=op.int16_output)
        append_body(patch_stage(stage, address))
    if bodies: self._submit_pcchain(bodies)

  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    self.dev._touch_program(self)
    self._ensure_scratch()
    def buffer(kind:RKBufferKind, index:int) -> HCQBuffer:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return bufs[index]
      if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
      return self.scratch[index]
    self.dev._sync_buffers(bufs, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    for i in range(len(self.image.constants)//2):
      if i >= len(self.scratch): break
      lane = self.image.constants[i*2:i*2+2]
      bits = lane * (self.scratch[i].size//len(lane))
      ctypes.memmove(int(self.scratch[i].va_addr), bits, len(bits))
    linear:dict[int, np.ndarray] = {}
    cleared_scratch:set[int] = set()
    def apply_host_addresses(ops:tuple[RKHostAddress, ...], scatter:bool) -> None:
      for op in ops:
        source, indices, dest = buffer(op.src.kind, op.src.index), buffer(op.index.kind, op.index.index), buffer(op.dst.kind, op.dst.index)
        lane_dtype = {1:np.uint8, 2:np.uint16, 4:np.uint32}[op.itemsize]
        index_dtype = {2:np.int16, 4:np.int32}[op.index_itemsize]
        if op.src.addend % op.itemsize or op.dst.addend % op.itemsize or op.index.addend % op.index_itemsize:
          raise RuntimeError("unaligned RKHostAddress")
        src = np.frombuffer(to_mv(int(source.va_addr), source.size), dtype=lane_dtype)[op.src.addend//op.itemsize:]
        repeat = 1 if scatter else max(op.index_repeat, 1)
        index_count = op.count if scatter else op.count//repeat
        idx = np.frombuffer(to_mv(int(indices.va_addr), indices.size), dtype=index_dtype)[
          op.index.addend//op.index_itemsize:op.index.addend//op.index_itemsize+index_count].astype(np.intp)
        dst = np.frombuffer(to_mv(int(dest.va_addr), dest.size), dtype=lane_dtype)[op.dst.addend//op.itemsize:]
        if len(idx) != index_count or len(src) < (op.count if scatter else op.src_count) or \
           len(dst) < (op.dst_count if scatter else op.count):
          raise RuntimeError("RKHostAddress exceeds buffer")
        limit = op.dst_count if scatter else op.index_limit or op.src_count
        valid = (idx >= 0) & (idx < limit)
        if not scatter:
          if repeat > 1: idx, valid = np.repeat(idx, repeat), np.repeat(valid, repeat)
          inner = np.arange(op.count, dtype=np.intp) if repeat == 1 else np.arange(op.count, dtype=np.intp)%repeat
          idx = op.base + inner*op.lane_stride + idx*op.index_scale
          valid &= (idx >= 0) & (idx < op.src_count)
        if scatter:
          for lane in range(op.count):
            if valid[lane]: dst[idx[lane]] = src[lane]
        else:
          dst[:op.count] = op.fill_bits
          dst[np.nonzero(valid)[0]] = src[idx[valid]]
    def apply_gathers(gathers:tuple[RKGather, ...]) -> None:
      for gather in gathers:
        dest = buffer(gather.dst_kind, gather.dst_index)
        if gather.dst_kind is RKBufferKind.SCRATCH and gather.dst_index not in cleared_scratch:
          ctypes.memset(int(dest.va_addr), 0, dest.size)
          cleared_scratch.add(gather.dst_index)
        lane_dtype = {1:np.uint8, 2:np.uint16, 4:np.uint32}[gather.itemsize]
        dst = np.frombuffer(to_mv(int(dest.va_addr), dest.size), dtype=lane_dtype)
        if gather.count not in linear: linear[gather.count] = np.arange(gather.count, dtype=np.intp)
        dst_index = gather.dst_addend + linear[gather.count] * gather.dst_stride
        if gather.values: dst[dst_index] = gather.values
        elif gather.offsets:
          source = buffer(gather.src_kind, gather.src_index)
          src = np.frombuffer(to_mv(int(source.va_addr), source.size), dtype=lane_dtype)
          index = np.asarray(gather.offsets, dtype=np.intp)
          valid = index >= 0
          if not gather.partial: dst[dst_index] = gather.fill_bits
          dst[dst_index[valid]] = src[index[valid]]
        else:
          source = buffer(gather.src_kind, gather.src_index)
          src = np.frombuffer(to_mv(int(source.va_addr), source.size), dtype=lane_dtype)
          index = np.full(gather.count, gather.base, dtype=np.intp)
          for divisor, limit, stride in gather.axes: index += (linear[gather.count]//divisor%limit)*stride
          dst[dst_index] = src[index]
    apply_gathers(self.image.gathers)
    apply_host_addresses(self.image.host_gathers, False)
    self.dev._sync_buffers((*bufs, *((self._scratch_arena,) if self._scratch_arena is not None else ())), rk.RKNPU_MEM_SYNC_TO_DEVICE)
    def address(kind:RKBufferKind, index:int) -> int:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return self._dma(bufs[index])
      if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
      return self._dma(self.scratch[index])
    start = time.perf_counter()
    def synchronized_gathers(gathers:tuple[RKGather, ...]) -> None:
      touched = {(g.src_kind, g.src_index) for g in gathers if not g.values}
      touched.update((g.dst_kind, g.dst_index) for g in gathers)
      self.dev._sync_buffers(tuple(buffer(kind, index) for kind,index in touched), rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      apply_gathers(gathers)
      self.dev._sync_buffers(tuple(buffer(kind, index) for kind,index in {(g.dst_kind, g.dst_index) for g in gathers}),
                             rk.RKNPU_MEM_SYNC_TO_DEVICE)
    if self.image.mid_gathers:
      cursor = 0
      by_point:dict[int, list[RKGather]] = {}
      for gather in self.image.mid_gathers: by_point.setdefault(gather.after, []).append(gather)
      for point,gathers in sorted(by_point.items()):
        self._run_ew_ops(address, buffer, self.image.ew_ops[cursor:point])
        synchronized_gathers(tuple(gathers))
        cursor = point
      self._run_ew_ops(address, buffer, self.image.ew_ops[cursor:])
    else: self._run_ew_ops(address, buffer)
    if self.image.host_scatters:
      touched = {(op.src.kind, op.src.index) for op in self.image.host_scatters} | \
                {(op.index.kind, op.index.index) for op in self.image.host_scatters} | \
                {(op.dst.kind, op.dst.index) for op in self.image.host_scatters}
      self.dev._sync_buffers(tuple(buffer(kind, index) for kind,index in touched), rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      apply_host_addresses(self.image.host_scatters, True)
      self.dev._sync_buffers(tuple(buffer(op.dst.kind, op.dst.index) for op in self.image.host_scatters), rk.RKNPU_MEM_SYNC_TO_DEVICE)
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(os.getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.submit_count = self.task_count = 0
    self._poisoned = False
    self._ew_precision = 0
    self.reset_npu()
    self._program_resource_limit = max(1, int(os.getenv("ROCKCHIP_PROGRAM_CACHE", "32")))
    self._program_resources:collections.OrderedDict[int, weakref.ReferenceType[RockchipProgram]] = collections.OrderedDict()
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)
  def _touch_program(self, program:RockchipProgram) -> None:
    self._program_resources.pop(id(program), None)
    self._program_resources[id(program)] = weakref.ref(program)
    while len(self._program_resources) > self._program_resource_limit:
      _, reference = self._program_resources.popitem(last=False)
      if (old:=reference()) is not None: old._release_resources()
  def _forget_program(self, program:RockchipProgram) -> None: self._program_resources.pop(id(program), None)
  def _check_healthy(self) -> None:
    if self._poisoned: raise RuntimeError("RKNPU is unavailable after a submit timeout; platform NPU reset or power cycle required")
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    self._check_healthy()
    alloc = max(4096, (size+4095)&-4096)
    try: meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=alloc, flags=flags|_BO_FLAGS)
    except OSError as exc: raise MemoryError(f"RKNPU GEM allocation failed for {alloc} bytes") from exc
    try:
      mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
      mapped = self.fd_ctl.mmap(0, alloc, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset)
    except Exception as exc:
      try: rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=meta.handle, reserved=0, obj_addr=meta.obj_addr)
      except (OSError, RuntimeError): pass
      raise MemoryError(f"RKNPU GEM mapping failed for {alloc} bytes") from exc
    return HCQBuffer(mapped, size, meta=meta)
  def _sync_buffer(self, buf:HCQBuffer, flags:int):
    self._check_healthy()
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd_ctl, flags=flags, reserved=0, obj_addr=buf.meta.obj_addr, offset=0, size=buf.meta.size)
  def _sync_buffers(self, bufs:tuple[HCQBuffer, ...], flags:int):
    seen:set[int] = set()
    for buf in bufs:
      if buf.meta.obj_addr in seen: continue
      seen.add(buf.meta.obj_addr)
      self._sync_buffer(buf, flags)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    if not self._poisoned: rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self):
    self._check_healthy()
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
    self._ew_precision = 0
  def synchronize(self): pass
