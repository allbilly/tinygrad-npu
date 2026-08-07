from __future__ import annotations
import ctypes, fcntl, mmap, os, time
import numpy as np
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, getenv, round_up, suppress_finalizing, temp, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RockchipRenderer, decode_image, patch_stage, emit_ew_stage,
  RKArg, rk_cmd, _MAX_EW_ELEMS_FP16, _EW_CFG_MAX, _EW_STAGE_ABS, _EW_STAGE_POSITIVE_MASK)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

# Ordinary ADD/MUL chains are capped at the measured 383-task boundary; mixed TwoProduct chains use 256.
_EW_CHAIN_MAX = 383
_PC_TAIL, _CMD_BUF_MIN, _TASK_BUF_MIN, _RKNPU_ACT_CLR_TOTAL_RW_AMOUNT = 4, 65536, 16384, 13

class RockchipAllocator(LRUAllocator['RockchipDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._gpu_alloc(size)
  def _copyin(self, dest:HCQBuffer, src:memoryview): ctypes.memmove(int(dest.va_addr), from_mv(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:HCQBuffer): ctypes.memmove(from_mv(dest), int(src.va_addr), dest.nbytes)
  def _as_buffer(self, src:HCQBuffer): return to_mv(int(src.va_addr), src.size)
  def _offset(self, buf:HCQBuffer, size:int, offset:int): return buf.offset(offset, size)
  def _free(self, buf:HCQBuffer, options:BufferSpec): self.dev._gpu_free(buf)

class RockchipProgram(Program['RockchipDevice']):
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.dev, self.name, self.image = dev, obj.name, decode_image(obj.lib)
    self.scratch = tuple(dev._gpu_alloc(x.size) for x in self.image.scratch)
    self._cmd_buf:HCQBuffer|None = None
    self._task_buf:HCQBuffer|None = None

  @suppress_finalizing
  def __del__(self):
    for buf in getattr(self, "scratch", ()): self.dev._gpu_free(buf)
    if (cmd:=getattr(self, "_cmd_buf", None)) is not None: self.dev._gpu_free(cmd)
    if (task:=getattr(self, "_task_buf", None)) is not None: self.dev._gpu_free(task)

  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)

  def _submit_buffer(self, buf:HCQBuffer|None, need:int, minimum:int, flags:int=0) -> HCQBuffer:
    if buf is None or buf.size < need:
      if buf is not None: self.dev._gpu_free(buf)
      buf = self.dev._gpu_alloc(max(need, minimum), flags)
    return buf

  def _submit_pcchain(self, bodies:list[tuple[int, ...]]) -> None:
    """Submit contiguous FP16 EW tasks as one blocking PC chain."""
    n = len(bodies)
    if not (1 <= n <= _EW_CHAIN_MAX): raise ValueError(f"EW PC-chain length {n} out of range")
    offsets:list[int] = []
    words = 0
    for body in bodies:
      offsets.append(words)
      words += round_up(len(body) + _PC_TAIL, 2)
    need = words * 8
    task_need = n * ctypes.sizeof(rk.struct_rknpu_task)
    self._cmd_buf = cmd = self._submit_buffer(self._cmd_buf, need, _CMD_BUF_MIN)
    self._task_buf = task = self._submit_buffer(self._task_buf, task_need, _TASK_BUF_MIN, rk.RKNPU_MEM_KERNEL_MAPPING)
    ctypes.memset(int(cmd.va_addr), 0, need)
    base_dma = self._dma(cmd)
    for i, body in enumerate(bodies):
      base = offsets[i]
      ctypes.memmove(int(cmd.va_addr) + base*8, (ctypes.c_uint64 * len(body))(*body), len(body)*8)
      next_addr = (base_dma + offsets[i+1]*8) & 0xfffffff0 if i+1 < n else 0
      next_amount = len(bodies[i+1]) if i+1 < n else 0
      tail = (rk_cmd(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, next_addr),
              rk_cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, next_amount),
              rk_cmd(rk.TARGET_VERSION, 0, 0), rk_cmd(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
      ctypes.memmove(int(cmd.va_addr) + (base+len(body))*8, (ctypes.c_uint64 * _PC_TAIL)(*tail), _PC_TAIL*8)
      desc = rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(body)+_PC_TAIL, 0, base_dma+base*8)
      ctypes.memmove(int(task.va_addr) + i*ctypes.sizeof(desc), ctypes.addressof(desc), ctypes.sizeof(desc))
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
      flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000,
      task_start=0, task_number=n, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
      regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
      subcore_task=(rk.struct_rknpu_subcore_task*5)(
        rk.struct_rknpu_subcore_task(0, n), rk.struct_rknpu_subcore_task(n, 0), rk.struct_rknpu_subcore_task(n, 0)))
    self.dev.submit_count += 1

  def _submit_standalone(self, body:tuple[int, ...]) -> None:
    """Submit a stateful DPU stage with the direct PC tail used by the vendor examples."""
    commands = body + (rk_cmd(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18),)
    need = len(commands) * 8
    self._cmd_buf = cmd = self._submit_buffer(self._cmd_buf, need, _CMD_BUF_MIN)
    self._task_buf = task = self._submit_buffer(self._task_buf, ctypes.sizeof(rk.struct_rknpu_task), _TASK_BUF_MIN, rk.RKNPU_MEM_KERNEL_MAPPING)
    ctypes.memmove(int(cmd.va_addr), (ctypes.c_uint64 * len(commands))(*commands), need)
    desc = rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(commands), 0, self._dma(cmd))
    ctypes.memmove(int(task.va_addr), ctypes.addressof(desc), ctypes.sizeof(desc))
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
      flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000,
      task_start=0, task_number=1, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
      regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
      subcore_task=(rk.struct_rknpu_subcore_task*5)(rk.struct_rknpu_subcore_task(0, 1)))
    self.dev.submit_count += 1

  def _run_ew_ops(self, address) -> None:
    bodies:list[tuple[int, ...]] = []
    mask_program = any(op.ew_cfg == _EW_STAGE_POSITIVE_MASK for op in self.image.ew_ops)
    needs_state = False
    def flush() -> None:
      if not bodies: return
      self._submit_pcchain(bodies)
      bodies.clear()
    for op in self.image.ew_ops:
      standalone = op.ew_cfg == _EW_STAGE_POSITIVE_MASK
      if standalone:
        flush()
        self.dev.reset_npu()
        needs_state = True
      for start in range(0, op.count, _MAX_EW_ELEMS_FP16):
        count = min(_MAX_EW_ELEMS_FP16, op.count-start)
        offset = start*2
        stage = emit_ew_stage(RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset),
                              RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend+offset),
                              RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend+offset), count, op.ew_cfg, mask_program or needs_state)
        body = patch_stage(stage, address)
        if standalone:
          self._submit_standalone(body)
          if op.ew_cfg == _EW_STAGE_POSITIVE_MASK: self.dev.reset_npu()
        else:
          bodies.append(body)
          if op.ew_cfg == _EW_STAGE_ABS:
            flush()
            self.dev.reset_npu()
            dst = RKArg(op.dst.kind, op.dst.index, op.dst.addend+offset)
            bodies.append(patch_stage(emit_ew_stage(dst, dst, dst, count, _EW_CFG_MAX, True), address))
            flush()
          needs_state = False
          if len(bodies) >= self.image.chain_limit: flush()
    flush()

  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    count = max((op.count for op in self.image.ew_ops), default=0)
    for i in range(len(self.image.constants)//2):
      if i >= len(self.scratch): break
      bits = self.image.constants[i*2:i*2+2] * count
      ctypes.memmove(int(self.scratch[i].va_addr), bits, len(bits))
    linear:dict[int, np.ndarray] = {}
    for gather in self.image.gathers:
      dst = np.frombuffer(to_mv(int(self.scratch[gather.dst_scratch].va_addr), self.scratch[gather.dst_scratch].size), dtype=np.uint16)
      if gather.count not in linear: linear[gather.count] = np.arange(gather.count, dtype=np.intp)
      dst_index = linear[gather.count] * gather.dst_stride
      if gather.values:
        dst[dst_index] = gather.values
        continue
      src = np.frombuffer(to_mv(int(bufs[gather.src_index].va_addr), bufs[gather.src_index].size), dtype=np.uint16)
      if gather.offsets:
        index = np.asarray(gather.offsets, dtype=np.intp)
        valid = index >= 0
        if not gather.partial: dst[dst_index] = gather.fill_bits
        dst[dst_index[valid]] = src[index[valid]]
      else:
        index = np.full(gather.count, gather.base, dtype=np.intp)
        for divisor, limit, stride in gather.axes: index += (linear[gather.count]//divisor%limit)*stride
        dst[dst_index] = src[index]
    def address(kind:RKBufferKind, index:int) -> int:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return self._dma(bufs[index])
      if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
      return self._dma(self.scratch[index])
    start = time.perf_counter()
    fcntl.flock(self.dev.npu_lock_fd, fcntl.LOCK_EX)
    try: self._run_ew_ops(address)
    finally: fcntl.flock(self.dev.npu_lock_fd, fcntl.LOCK_UN)
    if (fill:=self.image.fill) is not None:
      bits = self.image.constants[:2] * fill.count
      dest = bufs[fill.dst.index] if fill.dst.kind is RKBufferKind.ARG else self.scratch[fill.dst.index]
      ctypes.memmove(int(dest.va_addr), bits, len(bits))
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.npu_lock_fd = os.open(temp("rockchip_npu.lock"), os.O_RDWR|os.O_CREAT|os.O_CLOEXEC, 0o666)
    self.submit_count = 0
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc = max(4096, round_up(size, 4096))
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=alloc, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
    return HCQBuffer(self.fd_ctl.mmap(0, alloc, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset), size, meta=meta)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, round_up(buf.base.size, 4096)))
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self):
    try: rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=_RKNPU_ACT_CLR_TOTAL_RW_AMOUNT, value=0)
    except OSError: pass
    try: rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
    except OSError: pass
  def synchronize(self): pass
