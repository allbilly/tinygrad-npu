"""Blocking RK3588 FP16 matrix multiplication through the existing RKNPU driver.

This device executes explicit matrix kernels. Other model operations stay on the
chosen CPU/OpenCL device. Command buffers and packed weights survive JIT replay.
The register recipes are retained from the branch-1500 streaming experiments.
"""
from __future__ import annotations
import array, contextlib, ctypes, fcntl, functools, mmap, os, struct, threading, time, weakref
from dataclasses import dataclass, replace
from typing import Any, Protocol
from tinygrad.device import Allocator, BufferSpec, BufferStorage, Compiled, Device, Program
from tinygrad.renderer import TinyELF
from tinygrad.helpers import ceildiv, from_mv, getenv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.system import FileIOInterface
from tinygrad.renderer.cstyle import ClangRenderer

_FINITE_HALF_SOURCE = '''typedef unsigned short Half8 __attribute__((aligned(1),ext_vector_type(8)));
static __attribute__((always_inline)) int all_finite_half(const unsigned short *src,unsigned long count) {
  unsigned long i=0;
  Half8 invalid=(Half8)0;
  for(;i+8<=count;i+=8) {
    Half8 bits=*(const Half8*)(src+i);
    invalid|=(bits&(Half8)0x7c00)==(Half8)0x7c00;
  }
  if(__builtin_reduce_or(invalid))return 0;
  for(;i<count;i++)if((src[i]&0x7c00)==0x7c00)return 0;
  return 1;
}'''


@functools.cache
def _finite_program():
  cpu = Device['CPU']
  renderer = ClangRenderer(replace(cpu.renderer.target,renderer='CLANG'))
  source = _FINITE_HALF_SOURCE+'''int finite_half(const unsigned short *src,unsigned long count) {return all_finite_half(src,count);}'''
  program:Any = cpu.runtime(TinyELF(renderer.compiler.compile_cached(source),'finite_half',renderer.target,()))
  program.fxn.restype = ctypes.c_int
  return program


def _cmd(target:int, reg:int, value:int=0) -> int: return (target<<48)|((value&0xffffffff)<<16)|reg


# Complete FP16 matrix state, retaining the ordering of qualified RKLLM captures.
# Unused fields are cleared so batched/decode transitions need no platform reset.
_MATMUL_REGISTERS = (
  (0x201,(0x1040,0x1104,0x1100,0x100c)), (0x1001,(0x4004,)),
  (0x201,(0x100c,0x1010,0x1014,0x1020,0x1024,0x1028,0x102c,0x1030,0x1034,0x1038,0x1040,0x1044,0x104c,0x1050,
          0x1054,0x1058,0x105c,0x1060,0x1064,0x1068,0x1070,0x1074,0x1078,0x107c,0x1080,0x1084,0x1088,0x1100,
          0x1104,0x1110,*range(0x1140,0x1188,4))),
  (0x801,(0x3010,0x3014,0x3018,0x301c,0x3030)),
  (0x1001,(0x400c,0x4010,0x4014,0x4020,0x4024,0x4030,0x4034,0x4038,0x403c,*range(0x4040,0x408c,4),
           *range(0x4090,0x40b0,4),0x40c0,0x40c4,*range(0x4100,0x4130,4))))


def _matmul_body(m:int, n:int, k:int, lhs:int, rhs:int, dst:int, row_major:bool=False) -> tuple[int, ...]:
  """FP16 streaming with native weights; explicit row-major mode follows the captured RKLLM layout."""
  banks = ceildiv(m*k*2, 32768)
  if not (1 <= m <= (128 if row_major else 64) and 32 <= k <= (512 if row_major else 4096)
          and 32 <= n <= (8192 if m==1 else 4096) and k%32 == n%32 == 0 and banks <= 4):
    raise ValueError(f"Unsupported Rockchip matmul geometry {(m, n, k)}")
  if row_major and n>512:raise ValueError('Unqualified row-major FP16 output width')
  C, O, D = 0x201, 0x801, 0x1001
  # Qualified GPT-2 streaming geometries from the reference register sweeps.
  grains = {(768,256):32,(768,768):8,(768,1024):2,(3072,256):4,(768,8192):4}.get((k,n),2) \
    if m==1 and getenv('ONPU_GPT2_TUNE',1) else 2
  regs = ((D,rk.REG_DPU_S_POINTER,0xe), (C,rk.REG_CNA_CONV_CON1,(2<<4)|(2<<7)),
    (C,rk.REG_CNA_CONV_CON2,grains<<4), (C,rk.REG_CNA_CONV_CON3,9), (C,rk.REG_CNA_DATA_SIZE0,(m<<16)|1),
    (C,rk.REG_CNA_DATA_SIZE1,((k-1)<<16)|k), (C,rk.REG_CNA_DATA_SIZE2,m), (C,rk.REG_CNA_DATA_SIZE3,m),
    (C,rk.REG_CNA_WEIGHT_SIZE0,k*2*n), (C,rk.REG_CNA_WEIGHT_SIZE1,k*2),
    (C,rk.REG_CNA_WEIGHT_SIZE2,(1<<24)|(1<<16)|n), (C,rk.REG_CNA_CBUF_CON0,((12-banks)<<4)|banks),
    (C,rk.REG_CNA_CBUF_CON1,m*k//32), (C,rk.REG_CNA_CVT_CON0,11), (C,rk.REG_CNA_CVT_CON1,1<<16),
    (C,rk.REG_CNA_CVT_CON2,1<<16), (C,rk.REG_CNA_CVT_CON3,1<<16), (C,rk.REG_CNA_CVT_CON4,1<<16),
    (C,rk.REG_CNA_FEATURE_DATA_ADDR,lhs), (C,rk.REG_CNA_DMA_CON0,(15<<16)|15), (C,rk.REG_CNA_DMA_CON1,m*4),
    (C,rk.REG_CNA_DMA_CON2,(-3*m)&0x0fffffff), (C,rk.REG_CNA_FC_DATA_SIZE0,(m<<16)|1), (C,rk.REG_CNA_FC_DATA_SIZE1,k),
    (C,rk.REG_CNA_DCOMP_ADDR0,rhs), (O,rk.REG_CORE_MISC_CFG,(2<<8)|1), (O,rk.REG_CORE_DATAOUT_SIZE_0,m-1),
    (O,rk.REG_CORE_DATAOUT_SIZE_1,n-1), (O,rk.REG_CORE_RESERVED_3030,0), (D,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)),
    (D,rk.REG_DPU_DATA_FORMAT,(5<<29)|(2<<26)|2), (D,rk.REG_DPU_DST_BASE_ADDR,dst), (D,rk.REG_DPU_DST_SURF_STRIDE,m<<4),
    (D,rk.REG_DPU_DATA_CUBE_WIDTH,m-1), (D,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (D,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0), (D,rk.REG_DPU_DATA_CUBE_CHANNEL,((n-1)<<16)|(n-1)),
    (D,rk.REG_DPU_BS_CFG,0x53), (D,rk.REG_DPU_BS_OW_CFG,(3<<8)|(3<<5)|(3<<2)|2),
    (D,rk.REG_DPU_WDMA_SIZE_0,n-1), (D,rk.REG_DPU_WDMA_SIZE_1,m-1), (D,rk.REG_DPU_BN_CFG,0x53),
    (D,rk.REG_DPU_EW_CFG,0x383), (D,rk.REG_DPU_OUT_CVT_SCALE,1), (D,rk.REG_DPU_SURFACE_ADD,m*4<<4))
  values = {(target,reg):value for target,reg,value in regs}
  if row_major:
    capacity=1<<((131072//(k*2)).bit_length()-1)
    values.update({(C,0x100c):0x20000120,(C,0x1010):16*min(m+1,capacity),
      (C,0x1020):(1<<16)|m,(C,0x1028):1,(C,0x1044):k//32,
      (C,0x107c):k//8,(C,0x1080):0,(C,0x1084):(1<<16)|m,
      (O,0x3014):(m-1)<<16,(D,0x4024):16,(D,0x4030):0,(D,0x4034):m-1,
      (D,0x4038):((n//4-1)<<16)|(n//4-1),(D,0x405c):(m-1)<<16,(D,0x40c0):0x40})
  return tuple(_cmd(target,reg,values.get((target,reg),int((target,reg)==(D,rk.REG_DPU_EW_CVT_SCALE_VALUE))))
               for target,registers in _MATMUL_REGISTERS for reg in registers)


def _matmul_parts(m:int, n:int, k:int, cores:int, int8:bool=False) -> tuple[tuple[int,int],...]:
  if cores not in (1,3) or not 0<n<=(8192 if m==1 else 4096)*cores: raise ValueError('Invalid Rockchip core assignment')
  width = round_up(ceildiv(n,cores),32)
  parts = tuple((offset,min(width,n-offset)) for offset in range(0,n,width))
  if len(parts) not in (1,3): parts = ((0,n),)
  for _,columns in parts:
    _int8_matmul_body(m,columns,k,0,0,0,columns) if int8 else _matmul_body(m,columns,k,0,0,0)
  return parts


# INT8 K depths the CNA accepts without padding to a 512 multiple.
_INT8_NATURAL_DEPTHS = frozenset((576,768,1152,2304))


def _int8_padded_depth(k:int) -> int:
  """Physical K for an INT8 task: the natural depth when qualified, else a 512 multiple."""
  return k if k in _INT8_NATURAL_DEPTHS else round_up(k,512)


def _int8_supported_depth(k:int) -> bool:
  # 4096 is a real wall for this recipe, not the conservative cap the reference reports.
  # Measured: k=9216 at m=1, n=2560, cores=3 times out (DRM_IOCTL_RKNPU_SUBMIT, Errno 110)
  # because `banks` collapses to its 0x1b floor at 18 groups. So ffn_down stays split into
  # three K tiles, and 323 submits per forward is the floor for this recipe.
  return k in _INT8_NATURAL_DEPTHS or (512<=k<=4096 and k%512==0)


def _int8_matmul_body(m:int, n:int, k:int, lhs:int, rhs:int, dst:int, stride:int) -> tuple[int, ...]:
  """INT8 x INT8 -> INT32. `lhs` points at the activation payload, past its fp32 scale header.

  `n` is this program's output width; `stride` is the row stride of the output buffer, which
  must be the FULL width when several cores write column ranges into one buffer (0x403c packs
  stride in the high half and width-1 in the low half).
  """
  if not (1<=m<=(128 if k<=2048 else 64) and _int8_supported_depth(k)
          and 32<=n<=(8192 if m==1 else 4096) and n%32==0 and stride==n):
    raise ValueError(f"Unsupported Rockchip INT8 matmul geometry {(m, n, k)} stride={stride}")
  capacity = 1 << ((131072//k).bit_length()-1)
  groups = ceildiv(k,512)
  banks = max(0x1b, 177-15*(groups-1)-15*groups*(ceildiv(m,64)-1))
  # One-row wide-K decode needs a single data bank; retain eleven weight banks, which
  # matches the captured RKLLM single-row decode CBUF_CON0=0xb1 across k=768/2560/4096.
  if m==1 and k==4096 and (n==512 or n>=1024): banks = 0xb1
  C, O, D = 0x201, 0x801, 0x1001
  values = {
    (D,0x4004):0xe,(C,0x100c):1<<29,(C,0x1010):16*min(m+1,capacity),(C,0x1014):9,
    (C,0x1020):(1<<16)|m,(C,0x1024):((k-1)<<16)|k,(C,0x1028):1,(C,0x102c):m,
    (C,0x1030):k*n,(C,0x1034):k,(C,0x1038):0x1010000|n,(C,0x1040):banks,
    (C,0x1044):ceildiv(k,64),(C,0x104c):11,
    **{(C,r):1<<16 for r in (0x1050,0x1054,0x1058,0x105c)},
    (C,0x1070):lhs,(C,0x1078):0xf000f,(C,0x107c):k//16,(C,0x1084):(1<<16)|m,
    (C,0x1088):k,(C,0x1110):rhs,
    (O,0x3010):1,(O,0x3014):(m-1)<<16,(O,0x3018):n-1,
    (D,0x400c):0x1e4,(D,0x4010):0x80000000,(D,0x4020):dst,(D,0x4024):16,
    (D,0x4034):m-1,(D,0x4038):((stride//4-1)<<16)|(n//4-1),
    (D,0x403c):((stride-1)<<16)|(n-1),(D,0x4040):0x53,(D,0x4050):0x7fc,
    (D,0x4058):n-1,(D,0x405c):(m-1)<<16,(D,0x4060):0x53,(D,0x4070):0x383,
    (D,0x4078):1,(D,0x4084):1,(D,0x40c0):0x80}
  return tuple(_cmd(t,r,values.get((t,r),0)) for t,registers in _MATMUL_REGISTERS for r in registers)


def _encode_task_chain(dev:'RockchipDevice', bodies:tuple[tuple[int, ...], ...],
                       active_cores:int) -> tuple['RockchipBuffer', 'RockchipBuffer', int, int, tuple[tuple[int, int], ...]]:
  """Lay `bodies` out as one command buffer plus one task array.

  Returns (cmd, task, tasks, core_mask, subcores). Shared by `RockchipMatmul.prepare` and
  `int8_task_plan`, so the fused INT8 Linear submits a byte-identical chain from its own C
  function instead of re-deriving the encoding.
  """
  tasks = len(bodies)
  if tasks%active_cores: raise ValueError('Unbalanced Rockchip matrix task chains')
  per_core = tasks//active_cores
  total_words = sum(round_up(len(b)+4,2) for b in bodies)
  cmd = dev.allocate(round_up(total_words*8,4096))
  task = dev.allocate(max(4096,tasks*ctypes.sizeof(rk.struct_rknpu_task)),rk.RKNPU_MEM_KERNEL_MAPPING)
  ctypes.memset(cmd.addr,0,cmd.size)
  words, offset = to_mv(cmd.addr,cmd.size).cast('Q'), 0
  descs = (rk.struct_rknpu_task*tasks).from_address(task.addr)
  for i,body in enumerate(bodies):
    next_task = i+1 if (i+1)%per_core else None
    tail:tuple[int, ...] = ((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(cmd.dma+(offset+round_up(len(body)+4,2))*8)&0xfffffff0),
             _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[next_task]),2)+1)) if next_task is not None else
            (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
    tail += (_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
    words[offset:offset+len(body)+4] = array.array('Q',body+tail)
    descs[i] = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,cmd.dma+offset*8)
    offset += round_up(len(body)+4,2)
  subcores = ((0,tasks),) if active_cores==1 else ((0,0),(0,0),*((i*(tasks//3),tasks//3) for i in range(3)))
  dev.sync(cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  dev.sync(task,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  return cmd, task, tasks, (1<<active_cores)-1, subcores


def int8_task_plan(dev:'RockchipDevice', m:int, n:int, k:int, cores:int, out_dma:int, lhs_dma:int,
                   rhs_dma:int) -> tuple['RockchipBuffer', 'RockchipBuffer', int, int,
                                        tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
  """Command buffer, task array and part table for one INT8 matmul at explicit device addresses.

  `lhs_dma` points past the payload's fp32 scale header and `rhs_dma` already includes the
  output-column offset, exactly as `RockchipMatmul.prepare` computes them. The caller owns
  the returned buffers and must `dev.free` them.
  """
  parts = _matmul_parts(m,n,k,cores,int8=True)
  bodies = tuple(_int8_matmul_body(m,n,k,lhs_dma,rhs_dma+offset*k,out_dma+offset*m*4,n) for offset,n in parts)
  cmd, task, tasks, core_mask, subcores = _encode_task_chain(dev,bodies,1 if len(parts)==1 else 3)
  return cmd, task, tasks, core_mask, subcores, parts


def int8_task_plan_multi(dev:'RockchipDevice', sources:tuple[tuple[int,int,int,int,int,int], ...]
                         ) -> tuple['RockchipBuffer', 'RockchipBuffer', int, int,
                                    tuple[tuple[int, int], ...], tuple[tuple[tuple[int,int], ...], ...]]:
  """One submit carrying several INT8 matmuls that read *different* activations.

  `sources` is `(m, n, k, lhs_dma, rhs_dma, out_dma)` per Linear. This works because
  `_int8_matmul_body` bakes its own lhs/rhs/dst into the register program: verified that two bodies
  differing only in lhs differ in **1 word of 108**, and that each body contains its own lhs address.
  So no register or encoding change is needed -- `_encode_task_chain` already lays out arbitrary
  bodies, requiring only `tasks % active_cores == 0`.

  Every source must use 3 cores (3 parts each), so two sources give 6 bodies = 2 tasks per core.
  The saving is the measured **0.920 ms fixed cost per submit**: merging two submits into one is
  worth ~0.92 ms, and halving a 603-submit prefill is worth ~0.28 s.

  Each source's output goes to a distinct `out_dma`; callers that want one buffer must give disjoint
  column ranges. Parity is column-for-column, because each output column depends only on its own
  weight tile and its own K reduction order.
  """
  per_source: list[list[tuple[int, ...]]] = []
  part_tables: list[tuple[tuple[int, int], ...]] = []
  for m, n, k, lhs_dma, rhs_dma, out_dma in sources:
    parts = _matmul_parts(m, n, k, 3, int8=True)
    per_source.append([_int8_matmul_body(m, n, k, lhs_dma, rhs_dma + offset*k,
                                         out_dma + offset*m*4, n) for offset, _ in parts])
    part_tables.append(parts)
  # `_encode_task_chain` links task i -> i+1 while (i+1) % per_core, so bodies must be grouped BY CORE:
  # core 0's tasks for every source, then core 1's, ... Appending per source instead would chain two
  # column ranges of the same core, which is a different (and wrong) task sequence.
  bodies: list[tuple[int, ...]] = [per_source[src][core]
                                   for core in range(3) for src in range(len(per_source))]
  if len(bodies) % 3: raise ValueError(f'unbalanced merged chain: {len(bodies)} tasks over 3 cores')
  cmd, task, tasks, core_mask, subcores = _encode_task_chain(dev, tuple(bodies), 3)
  return cmd, task, tasks, core_mask, subcores, tuple(part_tables)


@dataclass(frozen=True)
class RockchipBuffer:
  addr: int
  dma: int
  size: int
  meta: Any


class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: return BufferStorage(self.dev.allocate(size))
  def _free(self, storage:BufferStorage, options:BufferSpec): self.dev.free(storage.buf)
  def _copyin(self, dest:RockchipBuffer, src:memoryview):
    self.dev.check_healthy()
    ctypes.memmove(dest.addr, from_mv(src), src.nbytes)
    dest.meta.version += 1
    self.dev.sync(dest, rk.RKNPU_MEM_SYNC_TO_DEVICE)
  def _copyout(self, dest:memoryview, src:RockchipBuffer):
    self.dev.sync(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    ctypes.memmove(from_mv(dest), src.addr, dest.nbytes)
  def _offset(self, buf:RockchipBuffer, size:int, offset:int): return replace(buf, addr=buf.addr+offset, dma=buf.dma+offset, size=size)


class RockchipMatmul:
  groups = 1
  dynamic_weights = False
  def __init__(self, dev:RockchipDevice, obj:TinyELF):
    self.dev = dev
    self.m, self.n, self.k, self.cores = struct.unpack('<4I', obj.lib[:16])
    self.rhs_offset = struct.unpack('<I',obj.lib[16:20])[0] if len(obj.lib)>=20 else 0
    self.lhs_offset = struct.unpack('<I',obj.lib[20:24])[0] if len(obj.lib)>=24 else 0
    self.split_k = bool(struct.unpack('<I',obj.lib[24:28])[0]) if len(obj.lib)>=28 else False
    self.int8 = bool(struct.unpack('<I',obj.lib[28:32])[0]) if len(obj.lib)>=32 else False
    self.itemsize = 1 if self.int8 else 2
    self.parts:tuple[tuple[int,int],...]
    if self.split_k:
      if self.cores!=3 or self.k%96: raise ValueError('Invalid Rockchip K split')
      _matmul_body(self.m,self.n,self.k//3,0,0,0)
      self.parts = ((0,self.n),)*3
    else: self.parts = _matmul_parts(self.m,self.n,self.k,self.cores,self.int8)
    self.bufs:tuple[RockchipBuffer, ...] = ()
    self.owned:list[RockchipBuffer] = []
    self.weight_version:tuple[int,int]|None = None
    dev.programs.add(self)

  def close(self):
    for buf in self.owned: self.dev.free(buf)
    self.owned.clear()
    self.bufs = ()

  def update_inputs(self, lhs:RockchipBuffer, rhs:RockchipBuffer): pass

  def prepare(self, out:RockchipBuffer, lhs:RockchipBuffer, rhs:RockchipBuffer):
    self.dev.check_domain(out,lhs,rhs)
    if (out.size<self.m*self.n*4*(3 if self.split_k else 1) or lhs.size<self.lhs_offset+self.m*self.k*self.itemsize
        or rhs.size<self.rhs_offset+self.n*self.k*self.itemsize):
      raise ValueError('Rockchip matrix kernel exceeds its bound buffers')
    self.close()
    self.bufs = out, lhs, rhs
    parts = self.parts
    if self.split_k:
      part = self.k//3
      bodies = tuple(_matmul_body(self.m,self.n,part,lhs.dma+self.lhs_offset+i*self.m*part*self.itemsize,
        rhs.dma+self.rhs_offset+i*self.n*part*self.itemsize,out.dma+i*self.m*self.n*4) for i in range(3))
    elif self.int8:
      # Each core writes a dense (m, part_width) plane at offset*m int32 elements; the host
      # scatters the planes into (m, n). There is no register that makes a column-split core
      # write into a wider row-major buffer: 0x4038/0x403c carry channel counts, not a stride.
      bodies = tuple(_int8_matmul_body(self.m,n,self.k,lhs.dma+self.lhs_offset,
        rhs.dma+self.rhs_offset+offset*self.k*self.itemsize,out.dma+offset*self.m*4,n) for offset,n in parts)
    else:
      bodies = tuple(_matmul_body(self.m,n,self.k,lhs.dma+self.lhs_offset,
        rhs.dma+self.rhs_offset+offset*self.k*self.itemsize,out.dma+offset*self.m*4) for offset,n in parts)
    self.tasks = len(bodies)
    self.cmd, self.task, self.tasks, self.core_mask, self.subcores = _encode_task_chain(
      self.dev, bodies, 1 if self.tasks==1 else 3)
    self.owned.append(self.cmd)
    self.owned.append(self.task)

  def __call__(self, *args:RockchipBuffer, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    out,lhs,rhs = args
    with self.dev.lock:
      self.dev.check_healthy()
      if (out,lhs,rhs) != self.bufs: self.prepare(out,lhs,rhs)
      check = _finite_program()
      self.update_inputs(lhs,rhs)
      version = rhs.meta.obj_addr,rhs.meta.version
      if not self.int8 and (self.dynamic_weights or version != self.weight_version):
        count = getattr(self,'weight_groups',self.groups)*self.n*self.k
        if not check.fxn(ctypes.c_void_p(rhs.addr+self.rhs_offset),ctypes.c_ulong(count)):
          raise ValueError('Rockchip matmul requires finite FP16 weights')
        self.weight_version = version
      # New cacheable GEM pages can carry dirty zeroes: clean before the NPU writes,
      # otherwise the following invalidation can overwrite its result with zeroes.
      if out.meta.cpu_clean_version!=out.meta.version: self.dev.sync(out,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      start = time.perf_counter()
      try:
        rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,timeout=6000,
          task_start=0,task_number=self.tasks,task_counter=0,priority=0,task_obj_addr=self.task.meta.obj_addr,
          iommu_domain_id=self.dev.domain,task_base_addr=0,hw_elapse_time=0,core_mask=self.core_mask,fence_fd=-1,
          subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*s) for s in self.subcores)))
      except OSError as exc:
        self.dev.poisoned = True
        if hasattr(exc,'add_note'):
          exc.add_note(f'Rockchip geometry M={self.m}, N={self.n}, K={self.k}, cores={self.cores}')
        raise
      self.dev.submit_count += 1
      self.dev.task_count += self.tasks
      elapsed = time.perf_counter()-start
      self.dev.submit_seconds += elapsed
      return elapsed if wait else None

  def __del__(self):
    if hasattr(self,'owned'): self.close()

class RockchipProgram(Program['RockchipDevice']):
  """A kernel can occur in many layers; retain each layer's bound command buffers."""
  def __init__(self, dev:RockchipDevice, obj:TinyELF, links:dict[str,Program]|None=None):
    self.dev, self.obj = dev, obj
    self.bindings:dict[tuple,RockchipMatmul] = {}

  def __call__(self, *args:RockchipBuffer, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    key = tuple((id(buf.meta),buf.addr,buf.size) for buf in args)
    if (plan := self.bindings.get(key)) is None: plan = self.bindings[key] = RockchipMatmul(self.dev,self.obj)
    return plan(*args,global_size=global_size,local_size=local_size,wait=wait,timeout=timeout)

class PreparedNPUResource(Protocol):
  def close(self) -> None: ...

class _RockchipHardware:
  def __init__(self):
    self.owner = open('/tmp/rk3588_npu_submit.lock','a')
    try:
      fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
      self.fd = FileIOInterface(os.getenv('ROCKCHIP_DRM','/dev/dri/card1'),os.O_RDWR)
      rk.DRM_IOCTL_RKNPU_ACTION(self.fd,flags=rk.RKNPU_ACT_RESET,value=0)
    except BaseException:
      self.owner.close()
      raise
    self.lock = threading.Lock()
    self.poisoned, self.users = False, 0
    self.submit_count = self.task_count = self.allocation_count = 0
    # Wall time spent inside blocking submits, so a run can separate device time from host time.
    self.submit_seconds = 0.0
    # The fused INT8 Linear submits from its own C function, so its counters live in this
    # process's memory and the C side accumulates into them: [submit_ns, submits, tasks].
    self.fused_stats = (ctypes.c_uint64 * 3)()

_hardware_lifetime_lock = threading.Lock()

@functools.cache
def _hardware() -> _RockchipHardware: return _RockchipHardware()


class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.domain = int(device.partition(':')[2] or '0')
    if not 0<=self.domain<16: raise ValueError('Rockchip memory domain must be between 0 and 15')
    with _hardware_lifetime_lock:
      try: self.hw = _hardware()
      except BlockingIOError: raise RuntimeError('Rockchip NPU is owned by another worker') from None
      self.hw.users += 1
    self.fd, self.lock, self.owner = self.hw.fd, self.hw.lock, self.hw.owner
    self.released, self.mapped_bytes = False, 0
    self.allocations:dict[int,RockchipBuffer] = {}
    self.programs:weakref.WeakSet[PreparedNPUResource] = weakref.WeakSet()
    try: super().__init__(device,RockchipAllocator(self),RockchipProgram)
    except BaseException:
      self.finalize()
      raise

  @property
  def poisoned(self) -> bool: return self.hw.poisoned
  @poisoned.setter
  def poisoned(self, value:bool): self.hw.poisoned = value
  @property
  def submit_count(self) -> int: return self.hw.submit_count + self.hw.fused_stats[1]
  @submit_count.setter
  def submit_count(self, value:int): self.hw.submit_count = value
  @property
  def task_count(self) -> int: return self.hw.task_count + self.hw.fused_stats[2]
  @task_count.setter
  def task_count(self, value:int): self.hw.task_count = value
  @property
  def allocation_count(self) -> int: return self.hw.allocation_count
  @allocation_count.setter
  def allocation_count(self, value:int): self.hw.allocation_count = value
  @property
  def submit_seconds(self) -> float: return self.hw.submit_seconds + self.hw.fused_stats[0] * 1e-9
  @submit_seconds.setter
  def submit_seconds(self, value:float): self.hw.submit_seconds = value
  @property
  def fused_stats_addr(self) -> int: return ctypes.addressof(self.hw.fused_stats)

  def check_domain(self, *bufs:RockchipBuffer):
    if any(buf.meta.iommu_domain_id!=self.domain for buf in bufs): raise ValueError('Rockchip matrix buffers cross memory domains')

  def check_healthy(self):
    if self.released: raise RuntimeError('This worker released its NPU ownership')
    if self.poisoned: raise RuntimeError('Rockchip NPU submission failed; reset the device before reloading')

  def reset(self): rk.DRM_IOCTL_RKNPU_ACTION(self.fd,flags=rk.RKNPU_ACT_RESET,value=0)

  def allocate(self, size:int, flags:int=0) -> RockchipBuffer:
    self.check_healthy()
    size = max(4096,round_up(size,4096))
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd,size=size,flags=flags|rk.RKNPU_MEM_NON_CONTIGUOUS|
                                      rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT,iommu_domain_id=self.domain)
    try:
      mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd,handle=meta.handle)
      addr = self.fd.mmap(0,size,mmap.PROT_READ|mmap.PROT_WRITE,mmap.MAP_SHARED,mapping.offset)
    except Exception:
      rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd,handle=meta.handle,obj_addr=meta.obj_addr)
      raise
    meta.version = 0
    meta.cpu_clean_version = -1
    self.allocation_count += 1
    self.mapped_bytes += size
    buf = RockchipBuffer(addr,int(meta.dma_addr),size,meta)
    self.allocations[meta.obj_addr] = buf
    return buf

  def sync(self, buf:RockchipBuffer, flags:int):
    self.check_healthy()
    # Initial pages can contain dirty zeroes anywhere in a shared allocation.
    # Later host writes are confined to the copied range and cleaned immediately.
    initial = bool(flags&rk.RKNPU_MEM_SYNC_TO_DEVICE and buf.meta.cpu_clean_version<0)
    offset,size = (0,buf.meta.size) if initial else (buf.dma-int(buf.meta.dma_addr),buf.size)
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd,flags=flags,obj_addr=buf.meta.obj_addr,offset=offset,size=size)
    if flags&rk.RKNPU_MEM_SYNC_TO_DEVICE: buf.meta.cpu_clean_version = buf.meta.version

  def free(self, buf:RockchipBuffer):
    if (owned:=self.allocations.get(buf.meta.obj_addr)) is None or owned.meta is not buf.meta: return
    FileIOInterface.munmap(owned.addr,owned.size)
    if not self.poisoned:
      with contextlib.suppress(OSError): rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd,handle=buf.meta.handle,obj_addr=buf.meta.obj_addr)
    del self.allocations[buf.meta.obj_addr]
    self.mapped_bytes -= owned.size

  def synchronize(self, timeout=None): self.check_healthy() # Every submission is blocking.

  def finalize(self):
    if self.released: return
    with self.lock:
      for program in list(self.programs): program.close()
      if hasattr(self,'allocator'): self.allocator.free_cache()
      # Live Tensor storage can outlast the device. Drop every VMA/GEM reference
      # while ownership is held; later allocator callbacks become harmless.
      for buf in list(self.allocations.values()): self.free(buf)
      with _hardware_lifetime_lock:
        self.hw.users -= 1
        self.released = True
        if self.hw.users==0:
          rk.DRM_IOCTL_RKNPU_ACTION(self.fd,flags=rk.RKNPU_SET_IOMMU_DOMAIN_ID,value=0)
          os.close(self.fd.fd)
          del self.fd.fd
          fcntl.flock(self.owner,fcntl.LOCK_UN)
          self.owner.close()
          _hardware.cache_clear()
