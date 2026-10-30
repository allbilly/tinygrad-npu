"""Blocking RK3588 FP16 matrix multiplication through the existing RKNPU driver.

This device executes explicit matrix kernels. Other model operations stay on the
chosen CPU/OpenCL device. Command buffers and packed weights survive JIT replay.
The register recipes are retained from the branch-1500 streaming experiments.
"""
from __future__ import annotations
import array, contextlib, ctypes, fcntl, functools, mmap, os, struct, threading, time, weakref
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast
from tinygrad.device import Allocator, BufferSpec, BufferStorage, Compiled, Device, Program, TinyELF
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
    if int8:
      from tinygrad.runtime.support.rockchip_int8 import int8_body
      int8_body(m,columns,k,0,0,0,columns)
    else: _matmul_body(m,columns,k,0,0,0)
  return parts


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
    self.sdk_prefill = struct.unpack('<I',obj.lib[32:36])[0] if len(obj.lib)>=36 else 0
    if self.sdk_prefill:
      from tinygrad.runtime.support.rockchip_int8 import sdk_prefill_supported
      if self.sdk_prefill!=1 or not self.int8 or not sdk_prefill_supported(self.m,self.n,self.k):
        raise ValueError('Invalid SDK INT8 prefill descriptor')
    self.itemsize = 1 if self.int8 else 2
    if self.int8 and (self.split_k or self.lhs_offset!=round_up(self.m*4,64)):
      raise ValueError('Invalid INT8 input scale header')
    self.scale_offset=struct.unpack('<I',obj.lib[36:40])[0] if self.int8 and len(obj.lib)==40 else 0
    if self.scale_offset%64:raise ValueError('Unaligned INT8 input scale header')
    self.lhs_offset+=self.scale_offset
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
    self.tasks = len(parts)
    active_cores = 1 if self.tasks==1 else 3
    self.core_mask = (1<<active_cores)-1
    if self.int8:
      from tinygrad.runtime.support.rockchip_int8 import int8_body, int8_sdk_body
      body_builder = int8_sdk_body if self.sdk_prefill else int8_body
      bodies = tuple(body_builder(self.m,n,self.k,lhs.dma+self.lhs_offset,rhs.dma+self.rhs_offset+offset*self.k,
                               out.dma+offset*self.m*4,n) for offset,n in parts)
    elif self.split_k:
      part = self.k//3
      bodies = tuple(_matmul_body(self.m,self.n,part,lhs.dma+self.lhs_offset+i*self.m*part*2,
        rhs.dma+self.rhs_offset+i*self.n*part*2,out.dma+i*self.m*self.n*4) for i in range(3))
    else:
      bodies = tuple(_matmul_body(self.m,n,self.k,lhs.dma+self.lhs_offset,rhs.dma+self.rhs_offset+offset*self.k*2,
        out.dma+offset*self.m*4) for offset,n in parts)
    total_words = sum(round_up(len(b)+4,2) for b in bodies)
    self.cmd = self.dev.allocate(round_up(total_words*8,4096))
    self.owned.append(self.cmd)
    self.task = self.dev.allocate(max(4096,self.tasks*ctypes.sizeof(rk.struct_rknpu_task)),rk.RKNPU_MEM_KERNEL_MAPPING)
    self.owned.append(self.task)
    ctypes.memset(self.cmd.addr,0,self.cmd.size)
    words, offset = to_mv(self.cmd.addr,self.cmd.size).cast('Q'), 0
    descs = (rk.struct_rknpu_task*self.tasks).from_address(self.task.addr)
    for i,body in enumerate(bodies):
      tail = (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_cmd(rk.TARGET_VERSION,0),
              _cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
      words[offset:offset+len(body)+4] = array.array('Q',body+tail)
      descs[i] = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,self.cmd.dma+offset*8)
      offset += round_up(len(body)+4,2)
    self.subcores = ((0,self.tasks),) if active_cores==1 else (
      (0,0),(0,0),*((i*(self.tasks//3),self.tasks//3) for i in range(3)))
    self.dev.sync(self.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self.dev.sync(self.task,rk.RKNPU_MEM_SYNC_TO_DEVICE)

  def __call__(self, *args:RockchipBuffer, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    out,lhs,rhs = args
    with self.dev.lock:
      self.dev.check_healthy()
      if (out,lhs,rhs) != self.bufs: self.prepare(out,lhs,rhs)
      check = _finite_program()
      if self.int8:
        scales = (ctypes.c_float*getattr(self,'storage_rows',self.groups*self.m)).from_address(lhs.addr+getattr(self,'scale_offset',0))
        if any(not 0<s<float('inf') for s in scales): raise ValueError('Rockchip INT8 requires finite positive activation scales')
      elif not check.fxn(ctypes.c_void_p(lhs.addr+self.lhs_offset),ctypes.c_ulong(self.groups*self.m*self.k)):
        raise ValueError('Rockchip matmul requires finite FP16 activations')
      self.update_inputs(lhs,rhs)
      version = rhs.meta.obj_addr,rhs.meta.version
      if self.dynamic_weights or version != self.weight_version:
        count = getattr(self,'weight_groups',self.groups)*self.n*self.k
        if not self.int8 and not check.fxn(ctypes.c_void_p(rhs.addr+self.rhs_offset),ctypes.c_ulong(count)):
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
          exc.add_note(f'Rockchip geometry M={self.m}, N={self.n}, K={self.k}, cores={self.cores}, int8={self.int8}')
        raise
      self.dev.submit_count += 1
      self.dev.task_count += self.tasks
      return time.perf_counter()-start if wait else None

  def __del__(self):
    if hasattr(self,'owned'): self.close()

class RockchipBatchedMatmul(RockchipMatmul):
  """One PC chain per core, retaining the ordinary immutable FP16 weight contract."""
  def __init__(self, dev:RockchipDevice, obj:TinyELF):
    self.dev = dev
    self.m,self.n,self.k,self.cores,rhs,lhs,split,integer,self.tile,marker = struct.unpack('<10I',obj.lib)
    if (marker!=0x42544d4d or rhs or lhs or split or integer or not 1<=self.m<=1024 or not 32<=self.k<=4096 or self.k%32
        or not 1<=self.tile<=min(64,65536//self.k) or self.cores not in (1,3)):
      raise ValueError('Unsupported batched FP16 matrix descriptor')
    self.rhs_offset=self.lhs_offset=0
    self.split_k=self.int8=False
    self.itemsize=2
    self.parts=_matmul_parts(self.tile,self.n,self.k,self.cores)
    self.bufs=()
    self.owned=[]
    self.weight_version=None
    dev.programs.add(self)

  def prepare(self, out:RockchipBuffer, lhs:RockchipBuffer, rhs:RockchipBuffer):
    self.dev.check_domain(out,lhs,rhs)
    if out.size<self.m*self.n*4 or lhs.size<self.m*self.k*2 or rhs.size<self.n*self.k*2:
      raise ValueError('Batched FP16 matrices exceed their bound buffers')
    self.close()
    self.bufs=out,lhs,rhs
    rows=list(range(0,self.m,self.tile))
    count=len(rows)
    bodies=[]
    for column,columns in self.parts:
      for row in rows:
        m=min(self.tile,self.m-row)
        bodies.append(_matmul_body(m,columns,self.k,lhs.dma+row*self.k*2,rhs.dma+column*self.k*2,
                                   out.dma+(row*self.n+column*m)*4))
    self.tasks=len(bodies)
    active=len(self.parts)
    self.core_mask=(1<<active)-1
    stride=round_up(len(bodies[0])+4,2)
    self.cmd=self.dev.allocate(stride*self.tasks*8)
    self.owned.append(self.cmd)
    self.task=self.dev.allocate(self.tasks*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
    self.owned.append(self.task)
    ctypes.memset(self.cmd.addr,0,self.cmd.size)
    for i,body in enumerate(bodies):
      next_task=i+1 if (i+1)%count else None
      tail=((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(self.cmd.dma+next_task*stride*8)&0xfffffff0),
             _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[next_task]),2)+1)) if next_task is not None else
            (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
      words=array.array('Q',body+tail+(_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)))
      ctypes.memmove(self.cmd.addr+i*stride*8,words.buffer_info()[0],len(words)*8)
      desc=rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,self.cmd.dma+i*stride*8)
      ctypes.memmove(self.task.addr+i*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
    self.subcores=((0,count),) if active==1 else ((0,0),(0,0),(0,count),(count,count),(count*2,count))
    self.dev.sync(self.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self.dev.sync(self.task,rk.RKNPU_MEM_SYNC_TO_DEVICE)


class RockchipGroupedMatmul(RockchipMatmul):
  """Grouped PC chains: 1 static FP16, 2 INT8, 7 dynamic FP16, 8 dynamic row-major FP16."""
  def __init__(self,dev:RockchipDevice,obj:TinyELF):
    self.dev=dev
    self.m,self.n,self.k,self.cores,self.groups,self.tile,marker,version=struct.unpack('<8I',obj.lib)
    self.weight_groups=self.groups
    integer=version==2
    self.dynamic_weights=version in (7,8)
    self.row_major=version==8
    limit=(128 if self.k<=2048 else 64) if integer else min(128 if self.row_major else 64,65536//max(1,self.k))
    if (marker!=0x47524d4d or version not in (1,2,7,8) or self.cores not in (1,3) or not 1<=self.groups<=128 or not 1<=self.m<=128
        or not 32<=self.k<=4096 or self.k%32 or not 32<=self.n<=4096 or self.n%32
        or not 1<=self.tile<=limit):
      raise ValueError('Unsupported grouped matrix descriptor')
    if integer:
      from tinygrad.runtime.support.rockchip_int8 import int8_body
      int8_body(min(self.m,self.tile),self.n,self.k,0,0,0,self.n)
    else:_matmul_body(min(self.m,self.tile),self.n,self.k,0,0,0,self.row_major)
    self.rhs_offset=0
    self.lhs_offset=round_up(self.groups*self.m*4,64) if integer else 0
    self.split_k=False
    self.int8=integer
    self.itemsize=1 if integer else 2
    self.bufs=()
    self.owned=[]
    self.weight_version=None
    dev.programs.add(self)

  def prepare(self,out:RockchipBuffer,lhs:RockchipBuffer,rhs:RockchipBuffer):
    self.dev.check_domain(out,lhs,rhs)
    if (out.size<self.groups*self.m*self.n*4 or lhs.size<self.lhs_offset+self.groups*self.m*self.k*self.itemsize
        or rhs.size<self.weight_groups*self.n*self.k*self.itemsize):raise ValueError('Grouped matrices exceed their bound buffers')
    self.close()
    self.bufs=out,lhs,rhs
    active=3 if self.cores==3 and self.groups>=3 else 1
    chains=[]
    for core in range(active):
      chain=[]
      for group in range(core,self.groups,active):
        for row in range(0,self.m,self.tile):
          a=lhs.dma+self.lhs_offset+(group*self.m+row)*self.k*self.itemsize
          b=rhs.dma+(group%self.weight_groups)*self.n*self.k*self.itemsize
          c=out.dma+(group*self.m+row)*self.n*4
          if self.int8:
            from tinygrad.runtime.support.rockchip_int8 import int8_body
            chain.append(int8_body(min(self.tile,self.m-row),self.n,self.k,a,b,c,self.n))
          else:chain.append(_matmul_body(min(self.tile,self.m-row),self.n,self.k,a,b,c,self.row_major))
      chains.append(chain)
    bodies=[body for chain in chains for body in chain]
    self.tasks=len(bodies)
    self.core_mask=(1<<active)-1
    stride=round_up(len(bodies[0])+4,2)
    self.cmd=self.dev.allocate(stride*self.tasks*8)
    self.owned.append(self.cmd)
    self.task=self.dev.allocate(self.tasks*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
    self.owned.append(self.task)
    ctypes.memset(self.cmd.addr,0,self.cmd.size)
    offset=0
    ranges=[]
    for chain in chains:
      ranges.append((offset,len(chain)))
      for j,body in enumerate(chain):
        index=offset+j
        tail=((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(self.cmd.dma+(index+1)*stride*8)&0xfffffff0),
               _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[index+1]),2)+1)) if j+1<len(chain) else
              (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
        words=array.array('Q',body+tail+(_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)))
        ctypes.memmove(self.cmd.addr+index*stride*8,words.buffer_info()[0],len(words)*8)
        desc=rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,self.cmd.dma+index*stride*8)
        ctypes.memmove(self.task.addr+index*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
      offset+=len(chain)
    self.subcores=tuple(ranges) if active==1 else ((0,0),(0,0),*ranges)
    self.dev.sync(self.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self.dev.sync(self.task,rk.RKNPU_MEM_SYNC_TO_DEVICE)


class RockchipRoutedMatmul(RockchipGroupedMatmul):
  """Single-token INT8 groups select immutable expert matrices using an input header."""
  def __init__(self,dev:RockchipDevice,obj:TinyELF):
    if len(obj.lib)!=36: raise ValueError('Invalid routed INT8 descriptor size')
    m,n,k,cores,groups,tile,marker,version,self.experts=struct.unpack('<9I',obj.lib)
    if (version!=3 or tile!=m or not 1<=groups<=8 or not 1<=self.experts<=128
        or marker!=0x47524d4d): raise ValueError('Unsupported routed INT8 descriptor')
    # Reuse the qualified integer recipes and grouped command chains.
    super().__init__(dev,replace(obj,lib=struct.pack('<8I',m,n,k,cores,groups,tile,marker,2)))
    self.selector_offset=self.groups*self.m*4
    self.lhs_offset=round_up((self.groups*self.m+self.groups)*4,64)
    self.weight_groups=self.experts
    self.weight_patches:list[tuple[int,int]]=[]

  def prepare(self,out:RockchipBuffer,lhs:RockchipBuffer,rhs:RockchipBuffer):
    if rhs.size<self.experts*self.n*self.k: raise ValueError('Routed expert weight atlas is too small')
    if rhs.dma<0 or rhs.dma+self.experts*self.n*self.k>1<<32: raise ValueError('Routed expert addresses exceed 32-bit IOVA')
    # Placeholder groups repeat within the atlas until validated selectors patch them.
    super().prepare(out,lhs,rhs)
    self.weight_patches=[]
    descs=(rk.struct_rknpu_task*self.tasks).from_address(self.task.addr)
    active=3 if self.cores==3 and self.groups>=3 else 1
    order=[g for core in range(active) for g in range(core,self.groups,active)]
    for index,group in enumerate(order):
      desc=descs[index]
      address=self.cmd.addr+desc.regcmd_addr-self.cmd.dma
      words=(ctypes.c_uint64*desc.regcfg_amount).from_address(address)
      slots=[i for i,w in enumerate(words) if (w>>48,w&0xffff)==(0x201,0x1110)]
      if len(slots)!=1: raise RuntimeError('Routed expert command has no unique weight address')
      self.weight_patches.append((address+slots[0]*8,group))

  def update_inputs(self,lhs:RockchipBuffer,rhs:RockchipBuffer):
    ids=(ctypes.c_int32*self.groups).from_address(lhs.addr+self.selector_offset)
    if any(not 0<=expert<self.experts for expert in ids): raise ValueError('Rockchip expert selector is out of range')
    for address,group in self.weight_patches:
      ctypes.c_uint64.from_address(address).value=_cmd(0x201,0x1110,rhs.dma+ids[group]*self.n*self.k)
    self.dev.sync(self.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)


class RockchipBucketedMatmul(RockchipGroupedMatmul):
  """Dynamic expert counts; ABI 4 has fixed slots, ABI 5 compact row storage."""
  def __init__(self,dev:RockchipDevice,obj:TinyELF):
    if len(obj.lib) not in (32,36): raise ValueError('Invalid bucketed INT8 descriptor size')
    m,n,k,cores,groups,tile,marker,version=struct.unpack('<8I',obj.lib[:32])
    if version not in (4,5,6) or len(obj.lib)!=(36 if version in (5,6) else 32):
      raise ValueError('Unsupported bucketed INT8 descriptor')
    maximum=128 if k<=2048 else 64
    if version==6:
      if not 1<=tile<=maximum or not tile<=m<=tile*2:raise ValueError('Unsupported chunked expert row capacity')
    elif tile!=m:raise ValueError('Unsupported bucketed INT8 tile')
    rows=struct.unpack('<I',obj.lib[32:])[0] if version in (5,6) else None
    if rows is not None and not 1<=rows<=groups*m: raise ValueError('Invalid compact expert row storage')
    super().__init__(dev,replace(obj,lib=struct.pack('<8I',tile,n,k,cores,groups,tile,marker,2)))
    self.m=m
    self.task_slots=groups*ceildiv(m,tile)
    self.storage_rows=self.groups*self.m if rows is None else rows
    self.count_offset=self.storage_rows*4
    self.lhs_offset=round_up(self.count_offset+self.groups*4,64)
    self.dynamic_state=(ctypes.c_uint32*12)()
    from tinygrad.runtime.support.rockchip_bucketed import updater
    self.updater=updater(self.groups,self.m,self.n,self.k,self.cores,rows,self.tile)

  def prepare(self,out:RockchipBuffer,lhs:RockchipBuffer,rhs:RockchipBuffer):
    self.dev.check_domain(out,lhs,rhs)
    for buf,required in ((out,self.storage_rows*self.n*4),
                         (lhs,self.lhs_offset+self.storage_rows*self.k),(rhs,self.groups*self.n*self.k)):
      if buf.size<required: raise ValueError('Bucketed expert matrix exceeds its bound buffers')
      if buf.dma<0 or buf.dma+required>1<<32: raise ValueError('Bucketed expert addresses exceed 32-bit IOVA')
    self.close()
    self.bufs=out,lhs,rhs
    from tinygrad.runtime.support.rockchip_int8 import int8_body
    length=len(int8_body(1,self.n,self.k,0,0,0,self.n))
    stride=round_up(length+4,2)
    self.cmd=self.dev.allocate(stride*self.task_slots*8)
    self.owned.append(self.cmd)
    self.task=self.dev.allocate(self.task_slots*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
    self.owned.append(self.task)
    ctypes.memset(self.cmd.addr,0,self.cmd.size)
    descs=(rk.struct_rknpu_task*self.task_slots).from_address(self.task.addr)
    for slot in range(self.task_slots):
      descs[slot]=rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,length,0,self.cmd.dma+slot*stride*8)
    self.dev.sync(self.task,rk.RKNPU_MEM_SYNC_TO_DEVICE)

  def update_inputs(self,lhs:RockchipBuffer,rhs:RockchipBuffer):
    result=self.updater.fxn(ctypes.c_void_p(self.cmd.addr),ctypes.c_uint64(self.cmd.dma),ctypes.c_uint64(lhs.dma),
      ctypes.c_uint64(rhs.dma),ctypes.c_uint64(self.bufs[0].dma),ctypes.c_void_p(lhs.addr+self.count_offset),
      ctypes.cast(self.dynamic_state,ctypes.c_void_p))
    if result: raise ValueError('Rockchip expert row counts are out of range or empty')
    self.tasks,self.core_mask=int(self.dynamic_state[0]),int(self.dynamic_state[1])
    self.subcores=tuple((int(self.dynamic_state[i]),int(self.dynamic_state[i+1])) for i in range(2,12,2))
    self.dev.sync(self.cmd,rk.RKNPU_MEM_SYNC_TO_DEVICE)


class RockchipProgram(Program['RockchipDevice']):
  """A kernel can occur in many layers; retain each layer's bound command buffers."""
  def __init__(self, dev:RockchipDevice, obj:TinyELF):
    self.dev, self.obj = dev, obj
    self.binding_limit = getenv('ROCKCHIP_BINDING_CACHE',256)
    if not 1<=self.binding_limit<=8192: raise ValueError('Rockchip binding cache must be between 1 and 8192')
    self.bindings:OrderedDict[tuple,RockchipMatmul] = OrderedDict()

  def __call__(self, *args:RockchipBuffer, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    key = tuple((id(buf.meta),buf.addr,buf.size) for buf in args)
    if key not in self.bindings:
      binary = getattr(self.obj,'lib',b'')
      plan = (RockchipRoutedMatmul if len(binary)==36 and struct.unpack('<II',binary[24:32])==(0x47524d4d,3) else
              RockchipBucketedMatmul if len(binary)==32 and struct.unpack('<II',binary[24:32])==(0x47524d4d,4) else
              RockchipBucketedMatmul if len(binary)==36 and struct.unpack('<II',binary[24:32])==(0x47524d4d,5) else
              RockchipBucketedMatmul if len(binary)==36 and struct.unpack('<II',binary[24:32])==(0x47524d4d,6) else
              RockchipGroupedMatmul if len(binary)==32 and struct.unpack('<I',binary[24:28])[0]==0x47524d4d else
              RockchipBatchedMatmul if len(binary)==40 and struct.unpack('<I',binary[-4:])[0]==0x42544d4d else RockchipMatmul)
      self.bindings[key] = plan(self.dev,self.obj)
      # A live replay can still own an evicted plan. Its final owner releases it.
      if len(self.bindings)>self.binding_limit: self.bindings.popitem(last=False)
    self.bindings.move_to_end(key)
    return self.bindings[key](*args,global_size=global_size,local_size=local_size,wait=wait,timeout=timeout)

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
    self.poisoned, self.users, self.weight_domain = False, 0, 0
    self.submit_count = self.task_count = self.allocation_count = 0

_hardware_lifetime_lock = threading.Lock()

@functools.cache
def _hardware() -> _RockchipHardware: return _RockchipHardware()


def int8_weight_device(size:int) -> str:
  """Opt-in sequential weight placement; leave room for command and activation buffers."""
  domains = getenv('ONPU_NPU_DOMAINS',1)
  budget = getenv('ONPU_NPU_DOMAIN_BYTES',3*1024**3)
  if not 1<=domains<=16 or not 4096<=budget<2**32 or not 0<size<=budget:
    raise ValueError('Invalid Rockchip weight domain configuration')
  base = cast(RockchipDevice,Device['ROCKCHIP'])
  if domains==1: return 'ROCKCHIP'
  while True:
    name = f'ROCKCHIP:{base.hw.weight_domain}' if base.hw.weight_domain else 'ROCKCHIP'
    dev = cast(RockchipDevice,Device[name])
    if dev.mapped_bytes+round_up(size,4096)<=budget: return name
    if base.hw.weight_domain+1>=domains: raise MemoryError('Configured Rockchip weight domains are full')
    base.hw.weight_domain += 1


class RockchipDevice(Compiled):
  def runtime(self, obj:TinyELF) -> Program:
    if len(obj.lib)==16 and struct.unpack('<I',obj.lib[12:16])[0]==0x4750464e:
      from tinygrad.runtime.support.rockchip_gpu import RockchipGPUProgram
      return RockchipGPUProgram(self,obj)
    return super().runtime(obj)

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
    try: super().__init__(device,RockchipAllocator(self),[],RockchipProgram)
    except BaseException:
      self.finalize()
      raise

  @property
  def poisoned(self) -> bool: return self.hw.poisoned
  @poisoned.setter
  def poisoned(self, value:bool): self.hw.poisoned = value
  @property
  def submit_count(self) -> int: return self.hw.submit_count
  @submit_count.setter
  def submit_count(self, value:int): self.hw.submit_count = value
  @property
  def task_count(self) -> int: return self.hw.task_count
  @task_count.setter
  def task_count(self, value:int): self.hw.task_count = value
  @property
  def allocation_count(self) -> int: return self.hw.allocation_count
  @allocation_count.setter
  def allocation_count(self, value:int): self.hw.allocation_count = value

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
