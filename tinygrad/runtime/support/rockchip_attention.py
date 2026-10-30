"""Prepared causal FP16 NPU attention with CPU operands and checked replay.

Only the bounded Qwen3 (16/8/128) and TinyLLaMA (32/4/64) layouts are admitted
for experimentation. K/V are copied from the live CPU cache on every call;
projection-weight immutability is never relaxed. Nothing opens the NPU until
this explicit attention descriptor is instantiated.
"""
from __future__ import annotations
import ctypes, struct, time
from dataclasses import replace
from pathlib import Path
from typing import cast
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.helpers import unwrap
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _fast_softmax_source
from tinygrad.runtime.ops_cpu import CPUDevice, CPUProgram
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipDevice, RockchipGroupedMatmul
from tinygrad.runtime.autogen import rockchip as rk

ATTENTION_MAGIC = 0x41504e52


class RockchipAttentionProgram(CPUProgram):
  def __init__(self, dev:CPUDevice, obj:TinyELF):
    if len(obj.lib)!=32:raise ValueError('Invalid NPU attention descriptor size')
    magic,self.heads,self.kv_heads,self.dim,self.context,self.capacity,options,version=struct.unpack('<8I',obj.lib)
    if (magic!=ATTENTION_MAGIC or version!=1 or (self.heads,self.kv_heads,self.dim) not in ((16,8,128),(32,4,64))
        or self.context not in (64,128,256,512) or not 1<=self.capacity<=128 or options&~15 or dev.remote is not None):
      raise ValueError('Unsupported NPU prefill attention descriptor')
    self.closed=False
    self.owned:list[RockchipBuffer]=[]
    self.plans:list[tuple[int,RockchipGroupedMatmul,RockchipGroupedMatmul]]=[]
    self.npu=cast(RockchipDevice,Device['ROCKCHIP'])
    self.status=(ctypes.c_uint64*3)()
    self.fast_softmax=bool(options&2)
    self.row_major,self.token_major=bool(options&4),bool(options&8)
    self.pool=None
    try:
      if options&1:
        from tinygrad.llm.kernels.rockchip_pool import attention_pool
        self.pool=attention_pool()
      self.qa=self.allocate(self.heads*self.capacity*self.dim*2)
      self.ka,self.va=(self.allocate(self.kv_heads*self.context*self.dim*2) for _ in range(2))
      self.scores=self.allocate(self.heads*self.capacity*self.context*4)
      self.probabilities=self.allocate(self.heads*self.capacity*self.context*2)
      self.result=self.allocate(self.heads*self.capacity*self.dim*4)
      # Initial cacheable GEM pages may contain dirty zeroes. Clean once before
      # the first NPU write, as the ordinary matrix runtime does.
      for buf in (self.scores,self.result):self.npu.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      for padded in (64,128,256,512):
        if padded>self.context:continue
        qk=self.matrix(self.scores,self.qa,self.ka,padded,self.dim)
        pv=self.matrix(self.result,self.probabilities,self.va,self.dim,padded)
        self.plans.append((padded,qk,pv))
      self.source=self.make_source()
      renderer=_renderer()
      super().__init__(dev,replace(obj,lib=renderer.compiler.compile_cached(self.source)))
      self.fxn.restype=ctypes.c_int
      self.npu.programs.add(self)
    except BaseException:
      self.close()
      raise

  def allocate(self,size:int) -> RockchipBuffer:
    buf=self.npu.allocate(size)
    self.owned.append(buf)
    return buf

  def matrix(self,out:RockchipBuffer,lhs:RockchipBuffer,rhs:RockchipBuffer,n:int,k:int) -> RockchipGroupedMatmul:
    binary=struct.pack('<8I',self.capacity,n,k,3,self.heads,min(128 if self.row_major else 64,65536//k),0x47524d4d,8 if self.row_major else 7)
    plan=RockchipGroupedMatmul(self.npu,TinyELF(binary,'attention_matrix',_renderer().target,()))
    # Group order is (query-within-KV-group, KV head); each query group
    # reuses the same packed K/V matrices instead of expanding the cache.
    plan.weight_groups=self.kv_heads
    try:plan.prepare(out,lhs,rhs)
    except BaseException:
      plan.close()
      raise
    return plan

  def make_source(self) -> str:
    libc=ctypes.CDLL(None)
    ioctl=unwrap(ctypes.cast(libc.ioctl,ctypes.c_void_p).value)
    errno=unwrap(ctypes.cast(getattr(libc,'__errno_location'),ctypes.c_void_p).value)
    sync_request=(3<<30)|(ctypes.sizeof(rk.struct_rknpu_mem_sync)<<16)|(ord('d')<<8)|0x45
    submit_request=(3<<30)|(ctypes.sizeof(rk.struct_rknpu_submit)<<16)|(ord('d')<<8)|0x41
    def sync(buf:RockchipBuffer,flag:int,size:str) -> str:
      return (f'{{struct Sync s={{{flag},0,{buf.meta.obj_addr}ULL,0,{size}}};'
              f'if(request({sync_request}UL,&s))return 3;}}')
    def submit(plan:RockchipGroupedMatmul) -> str:
      subcores=[v for pair in (*plan.subcores,*((0,0),)*(5-len(plan.subcores))) for v in pair]
      return ('{struct Submit s={'+f'5,6000,0,{plan.tasks},0,0,{plan.task.meta.obj_addr}ULL,{self.npu.domain},0,0,'+
        f'{plan.core_mask},-1,'+'{'+','.join(map(str,subcores))+'}};'+
        f'if(request({submit_request}UL,&s))return 3;status[0]++;status[1]+={plan.tasks};}}')
    selections='\n'.join(f'''if(padded=={padded}){{
      {submit(qk)}
      {sync(self.scores,2,f'{self.heads}UL*{self.capacity}*padded*4')}
      code=normalize({self.capacity},active,total,padded,(const float*){self.scores.addr}ULL,(__fp16*){self.probabilities.addr}ULL);
      if(code)return code>=1000?4:3;
      {sync(self.probabilities,1,f'{self.heads}UL*{self.capacity}*padded*2')}
      {submit(pv)}
    }}''' for padded,qk,pv in self.plans)
    pool=self.pool.prefill_address if self.pool is not None else 0
    helpers=Path(__file__).parents[2]/'llm/kernels/rockchip_causal_attention.c'
    softmax=(_fast_softmax_source() if self.fast_softmax else '''extern float expf(float);
    static void rockchip_softmax_vector(float *values,int count,float maximum) {
      float sum=0;for(int t=0;t<count;t++){values[t]=expf(values[t]-maximum);sum+=values[t];}
      for(int t=0;t<count;t++)values[t]/=sum;
    }''')
    return f'''#pragma STDC FP_CONTRACT OFF
    #define HEADS {self.heads}
    #define KV_HEADS {self.kv_heads}
    #define DIM {self.dim}
    #define CONTEXT {self.context}
    #define SHARE_KV 1
    #define ROW_MAJOR {int(self.row_major)}
    #define TOKEN_MAJOR_OUTPUT {int(self.token_major)}
    #define ATTENTION_POOL_ADDRESS {pool}ULL
    '''+softmax+'\n'+helpers.read_text()+f'''
    typedef unsigned int U32;typedef unsigned long long U64;
    struct Sync {{U32 flags,reserved;U64 object,offset,size;}} __attribute__((packed));
    struct Submit {{U32 flags,timeout,start,number,counter;int priority;U64 object,regcfg,base,user;
      U32 mask;int fence;U32 subcores[10];}} __attribute__((packed));
    static U64 *status=(U64*){ctypes.addressof(self.status)}ULL;
    static int request(unsigned long code,void *arg) {{
      if(((int(*)(int,unsigned long,void*)){ioctl}ULL)({self.npu.fd.fd},code,arg)){{
        status[2]=*((int*(*)(void)){errno}ULL)();return 1;}}return 0;
    }}
    int rockchip_prefill_attention(float *out,const float *query,const __fp16 *cache,const int *length,const int *rows) {{
      status[0]=status[1]=status[2]=0;
      int total=*length,active=*rows,padded=64;
      if(active<1 || active>{self.capacity} || total<active || total>{self.context})return 1;
      while(padded<total)padded*=2;
      int code=pack({self.capacity},active,total,padded,query,cache,
        (__fp16*){self.qa.addr}ULL,(__fp16*){self.ka.addr}ULL,(__fp16*){self.va.addr}ULL);
      if(code)return code>=1000?4:2;
      {sync(self.qa,1,f'{self.heads}UL*{self.capacity}*{self.dim}*2')}
      {sync(self.ka,1,f'{self.kv_heads}UL*padded*{self.dim}*2')}
      {sync(self.va,1,f'{self.kv_heads}UL*{self.dim}*padded*2')}
      {selections}
      {sync(self.result,2,f'{self.heads}UL*{self.capacity}*{self.dim}*4')}
      return unpack_result({self.capacity},active,padded,(const float*){self.result.addr}ULL,out)?4:0;
    }}'''

  def __call__(self,*args:int,global_size=(1,1,1),local_size=(1,1,1),wait=False,timeout=None) -> float|None:
    with self.npu.lock:
      if self.closed:raise RuntimeError('NPU attention resources have been released')
      self.npu.check_healthy()
      start=time.perf_counter()
      code=self.fxn(*(ctypes.c_uint64(a) for a in args))
      self.npu.submit_count+=self.status[0]
      self.npu.task_count+=self.status[1]
      if code==1:raise ValueError('NPU attention length or active rows are out of range')
      if code==2:raise ValueError('NPU attention requires finite FP16-representable inputs')
      if code==4:raise RuntimeError('NPU attention CPU dispatch failed')
      if code:
        self.npu.poisoned=True
        raise RuntimeError(f'NPU attention submission, synchronization or score validation failed: {self.status[2]}')
      return time.perf_counter()-start if wait else None

  def emit(self,expressions:list[str],index:int) -> str:
    address=unwrap(ctypes.cast(self.fxn,ctypes.c_void_p).value)
    status=ctypes.addressof(self.status)
    return ('{int code=((int(*)(void*,const void*,const void*,const void*,const void*))'+
      f'{address}ULL)('+','.join(expressions)+');'+
      f'stats[0]+=((const unsigned long long*){status}ULL)[0];stats[1]+=((const unsigned long long*){status}ULL)[1];'+
      f'if(code==1)return {6000000+index};if(code==2)return {1000000+index};'+
      f'if(code==4)return {7000000+index};if(code)return {2000000+index};'+'}')

  def close(self):
    if self.closed:return
    for _,qk,pv in self.plans:
      qk.close()
      pv.close()
    for buf in reversed(self.owned):self.npu.free(buf)
    self.owned.clear()
    self.closed=True

  def __del__(self):
    if hasattr(self,'closed'):self.close()
