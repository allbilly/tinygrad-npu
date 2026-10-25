"""Blocking RK3588 FP16 matrix multiplication through the existing RKNPU driver.

This device executes explicit matrix kernels. Other model operations stay on the
chosen CPU/OpenCL device. Command buffers and packed weights survive JIT replay.
The register recipes are retained from the branch-1500 streaming experiments.
"""
from __future__ import annotations
import array, contextlib, ctypes, fcntl, functools, mmap, os, struct, threading, time, weakref
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Protocol
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


def _matmul_body(m:int, n:int, k:int, lhs:int, rhs:int, dst:int) -> tuple[int, ...]:
  """Full-K streaming: input (K/8,M,8), weight (N/16,K/32,16,32), output (N/4,M,4)."""
  banks = ceildiv(m*k*2, 32768)
  if not (1 <= m <= 64 and 32 <= k <= 4096
          and 32 <= n <= (8192 if m==1 else 4096) and k%32 == n%32 == 0 and banks <= 4):
    raise ValueError(f"Unsupported Rockchip matmul geometry {(m, n, k)}")
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
    ctypes.memmove(dest.addr, from_mv(src), src.nbytes)
    dest.meta.version += 1
    self.dev.sync(dest, rk.RKNPU_MEM_SYNC_TO_DEVICE)
  def _copyout(self, dest:memoryview, src:RockchipBuffer):
    self.dev.sync(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    ctypes.memmove(from_mv(dest), src.addr, dest.nbytes)
  def _offset(self, buf:RockchipBuffer, size:int, offset:int): return replace(buf, addr=buf.addr+offset, dma=buf.dma+offset, size=size)


class RockchipMatmul:
  def __init__(self, dev:RockchipDevice, obj:TinyELF):
    self.dev = dev
    self.m, self.n, self.k, self.cores = struct.unpack('<4I', obj.lib[:16])
    self.rhs_offset = struct.unpack('<I',obj.lib[16:20])[0] if len(obj.lib)>=20 else 0
    self.lhs_offset = struct.unpack('<I',obj.lib[20:24])[0] if len(obj.lib)>=24 else 0
    self.split_k = bool(struct.unpack('<I',obj.lib[24:28])[0]) if len(obj.lib)>=28 else False
    self.int8 = bool(struct.unpack('<I',obj.lib[28:32])[0]) if len(obj.lib)>=32 else False
    self.itemsize = 1 if self.int8 else 2
    if self.int8 and (self.split_k or self.lhs_offset!=round_up(self.m*4,64)):
      raise ValueError('Invalid INT8 input scale header')
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

  def prepare(self, out:RockchipBuffer, lhs:RockchipBuffer, rhs:RockchipBuffer):
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
      from tinygrad.runtime.support.rockchip_int8 import int8_body
      bodies = tuple(int8_body(self.m,n,self.k,lhs.dma+self.lhs_offset,rhs.dma+self.rhs_offset+offset*self.k,
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
        scales = (ctypes.c_float*self.m).from_address(lhs.addr)
        if any(not 0<s<float('inf') for s in scales): raise ValueError('Rockchip INT8 requires finite positive activation scales')
      elif not check.fxn(ctypes.c_void_p(lhs.addr+self.lhs_offset),ctypes.c_ulong(self.m*self.k)):
        raise ValueError('Rockchip matmul requires finite FP16 activations')
      version = rhs.meta.obj_addr,rhs.meta.version
      if version != self.weight_version:
        count = self.n*self.k
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
          regcfg_obj_addr=0,task_base_addr=0,user_data=0,core_mask=self.core_mask,fence_fd=-1,
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
      plan = RockchipBatchedMatmul if len(binary)==40 and struct.unpack('<I',binary[-4:])[0]==0x42544d4d else RockchipMatmul
      self.bindings[key] = plan(self.dev,self.obj)
      # A live replay can still own an evicted plan. Its final owner releases it.
      if len(self.bindings)>self.binding_limit: self.bindings.popitem(last=False)
    self.bindings.move_to_end(key)
    return self.bindings[key](*args,global_size=global_size,local_size=local_size,wait=wait,timeout=timeout)

class PreparedNPUResource(Protocol):
  def close(self) -> None: ...

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.owner = open('/tmp/rk3588_npu_submit.lock','a')
    try: fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Rockchip NPU is owned by another worker') from None
    self.fd = FileIOInterface(os.getenv('ROCKCHIP_DRM','/dev/dri/card1'),os.O_RDWR)
    self.lock = threading.Lock()
    self.poisoned, self.released, self.submit_count, self.task_count, self.allocation_count = False, False, 0, 0, 0
    self.programs:weakref.WeakSet[PreparedNPUResource] = weakref.WeakSet()
    self.reset()
    super().__init__(device,RockchipAllocator(self),[],RockchipProgram)

  def check_healthy(self):
    if self.released: raise RuntimeError('This worker released its NPU ownership')
    if self.poisoned: raise RuntimeError('Rockchip NPU submission failed; reset the device before reloading')

  def reset(self): rk.DRM_IOCTL_RKNPU_ACTION(self.fd,flags=rk.RKNPU_ACT_RESET,value=0)

  def allocate(self, size:int, flags:int=0) -> RockchipBuffer:
    self.check_healthy()
    size = max(4096,round_up(size,4096))
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd,size=size,flags=flags|rk.RKNPU_MEM_NON_CONTIGUOUS|
                                      rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    try:
      mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd,handle=meta.handle)
      addr = self.fd.mmap(0,size,mmap.PROT_READ|mmap.PROT_WRITE,mmap.MAP_SHARED,mapping.offset)
    except Exception:
      rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd,handle=meta.handle,obj_addr=meta.obj_addr)
      raise
    meta.version = 0
    meta.cpu_clean_version = -1
    self.allocation_count += 1
    return RockchipBuffer(addr,int(meta.dma_addr),size,meta)

  def sync(self, buf:RockchipBuffer, flags:int):
    self.check_healthy()
    # Initial pages can contain dirty zeroes anywhere in a shared allocation.
    # Later host writes are confined to the copied range and cleaned immediately.
    initial = bool(flags&rk.RKNPU_MEM_SYNC_TO_DEVICE and buf.meta.cpu_clean_version<0)
    offset,size = (0,buf.meta.size) if initial else (buf.dma-int(buf.meta.dma_addr),buf.size)
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd,flags=flags,obj_addr=buf.meta.obj_addr,offset=offset,size=size)
    if flags&rk.RKNPU_MEM_SYNC_TO_DEVICE: buf.meta.cpu_clean_version = buf.meta.version

  def free(self, buf:RockchipBuffer):
    FileIOInterface.munmap(buf.addr,buf.size)
    if not self.poisoned:
      with contextlib.suppress(OSError): rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd,handle=buf.meta.handle,obj_addr=buf.meta.obj_addr)

  def synchronize(self, timeout=None): self.check_healthy() # Every submission is blocking.

  def finalize(self):
    for program in list(self.programs): program.close()
    self.allocator.free_cache()
    if not self.released:
      fcntl.flock(self.owner,fcntl.LOCK_UN)
      self.owner.close()
      self.released = True
