"""Prepared three-core vision attention with FP32 CPU softmax.

Each core executes a chain of 16 bounded matmuls for one head. K/V are dynamic
operator inputs and are repacked and validated on every call. This does not relax
the immutable-weight contract of the regular linear/replay implementation.
"""
from __future__ import annotations
import array, ctypes
from dataclasses import replace
from typing import cast
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import ceildiv, round_up, unwrap
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipDevice, _cmd, _matmul_body
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

class NPUVisionAttention:
  # Runtime resources are deliberately opaque to model parameter traversal.
  __slots__ = ('dev','owned','heads','rows','batch_heads','q','k','v','scores','probabilities','result','qk_tasks','pv_tasks',
               'status','libc','jit','closed','reset_context','profile','softmax','__weakref__')

  def __init__(self, heads:int=12, rows:int=1024, reset_context:bool=True, profile:bool=False, gpu_softmax:bool=False):
    if heads not in (3,12) or rows!=1024: raise ValueError('NPU vision attention requires 3 or 12 heads and 1024 patches')
    self.dev = cast(RockchipDevice,Device['ROCKCHIP'])
    self.owned:list[RockchipBuffer] = []
    self.heads, self.rows, self.closed = heads,rows,False
    self.batch_heads = heads if gpu_softmax else 3
    self.reset_context = reset_context
    self.profile = profile
    self.softmax = None
    self.status = (ctypes.c_uint64*10)()
    self.libc = ctypes.CDLL(None)
    try:
      self.q,self.k,self.v = (self.allocate(self.batch_heads*rows*64*2) for _ in range(3))
      self.scores = self.allocate(self.batch_heads*rows*rows*4)
      self.probabilities = self.allocate(self.batch_heads*rows*rows*2)
      self.result = self.allocate(self.batch_heads*rows*64*4)
      self.qk_tasks = self.chain(self.q,self.k,self.scores,64,rows)
      self.pv_tasks = self.chain(self.probabilities,self.v,self.result,rows,64)
      for buf in (self.scores,self.result): self.dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      if gpu_softmax:
        from tinygrad.llm.kernels.rockchip_softmax import SharedNPUVisionSoftmax
        self.softmax = SharedNPUVisionSoftmax(self.dev,self.scores,self.probabilities)
      self.jit = TinyJit(self.execute)
      self.dev.programs.add(self)
    except BaseException:
      self.close()
      raise

  def __reduce__(self): raise TypeError('Prepared NPU attention is process-local and cannot be serialized')

  def allocate(self, size, flags=0):
    buf = self.dev.allocate(size,flags)
    self.owned.append(buf)
    return buf

  def chain(self, lhs, rhs, out, depth, columns):
    count = self.rows//64*(self.batch_heads//3)
    bodies = [_matmul_body(64,columns,depth,lhs.dma+(head*self.rows+row)*depth*2,rhs.dma+head*columns*depth*2,
      out.dma+(head*self.rows+row)*columns*4)
      for core in range(3) for head in range(core,self.batch_heads,3) for row in range(0,self.rows,64)]
    stride = round_up(len(bodies[0])+4,2)
    commands = self.allocate(len(bodies)*stride*8)
    tasks = self.allocate(len(bodies)*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
    ctypes.memset(commands.addr,0,commands.size)
    for i,body in enumerate(bodies):
      next_task = i+1 if (i+1)%count else None
      tail = ((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(commands.dma+next_task*stride*8)&0xfffffff0),
               _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[next_task]),2)+1)) if next_task is not None else
              (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
      words = array.array('Q',body+tail+(_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)))
      ctypes.memmove(commands.addr+i*stride*8,words.buffer_info()[0],len(words)*8)
      desc = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,commands.dma+i*stride*8)
      ctypes.memmove(tasks.addr+i*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
    for buf in (commands,tasks): self.dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    return tasks

  def source(self):
    rows, tiles = self.rows,self.rows//64
    tasks = tiles*(self.batch_heads//3)
    ioctl = unwrap(ctypes.cast(self.libc.ioctl,ctypes.c_void_p).value)
    sync_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_mem_sync)<<16)|(ord('d')<<8)|0x45
    submit_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_submit)<<16)|(ord('d')<<8)|0x41
    reset_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_action)<<16)|(ord('d')<<8)|0x40
    def sync(buf, flag):
      return f'if(sync_mem({buf.meta.obj_addr}ULL,{buf.size}ULL,{flag}))return;'
    def submit(tasks):
      return f'if(submit({tasks.meta.obj_addr}ULL))return;'
    def timed(index,code): return f'{{U64 start=now();{code}status[{index}]+=now()-start;}}' if self.profile else code
    softmax = ('if(shared_softmax()){status[0]=4;return;}' if self.softmax else
               'if(parallel(1,out,q,k,v)){status[0]=3;return;}')
    # Shared intermediates are never touched by the CPU after initial cleaning.
    # Device completion, rather than CPU cache maintenance, orders their handoff.
    scores_to_cpu = '' if self.softmax else sync(self.scores,2)
    probabilities_from_cpu = '' if self.softmax else sync(self.probabilities,1)
    clock = ('extern int clock_gettime(int,void*);static U64 now(void){long t[2];clock_gettime(1,t);'
             'return (U64)t[0]*1000000000ULL+t[1];}') if self.profile else ''
    # Terms below exp(-20) round to zero in the FP16 PV input. Omitting their
    # denominator contribution loses at most 1024*exp(-20) < 2.12e-6 mass.
    return f'''{_vector_exp2_source()}
    typedef unsigned long long U64;typedef unsigned int U32;
    {clock}
    typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
    typedef unsigned long thread_t;
    extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
    extern int pthread_join(thread_t,void**);extern int *__errno_location(void);
    static U64 *status=(U64*){ctypes.addressof(self.status)}ULL;
    static __fp16 *qa=(__fp16*){self.q.addr}ULL,*ka=(__fp16*){self.k.addr}ULL,*va=(__fp16*){self.v.addr}ULL;
    static float *scores=(float*){self.scores.addr}ULL,*pv=(float*){self.result.addr}ULL;
    static __fp16 *prob=(__fp16*){self.probabilities.addr}ULL;
    static __attribute__((always_inline)) float4 bounded_exp2(float4 x){{
      int4 q=__builtin_convertvector(x-.5f,int4);
      float4 s=x-__builtin_convertvector(q,float4);
      float4 p=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
        0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
      return p*__builtin_bit_cast(float4,(q+127)<<23);
    }}
    static float exponentials(float *p,float maximum){{
      float4 partial=(float4)0;
      for(int c=0;c<{rows};c+=4){{
        float4 delta=*(const float4*)(p+c)-(float4)maximum;
        int4 zero=delta<(float4)-20.0f;
        float4 safe=zero?(float4)0:delta;
        float4 value=zero?(float4)0:bounded_exp2(safe*1.4426950216293335f);
        *(float4*)(p+c)=value;partial+=value;
      }}
      return 1.0f/(((partial[0]+partial[1])+partial[2])+partial[3]);
    }}
    static int request(unsigned long code,void *value){{
      if(((int(*)(int,unsigned long,void*)){ioctl}ULL)({self.dev.fd.fd},code,value)){{status[0]=2;status[2]=*__errno_location();return 1;}}return 0;
    }}
    static int sync_mem(U64 object,U64 size,U64 flag){{U64 args[4]={{flag,object,0,size}};return request({sync_request}UL,args);}}
    static int reset(void){{U64 args={rk.RKNPU_ACT_RESET};return request({reset_request}UL,&args);}}
    struct Submit{{U32 flags,timeout,start,count,counter;int priority;U64 task,reg,base,user;U32 mask;int fence;U32 subcores[10];}};
    typedef char check_submit_size[sizeof(struct Submit)=={ctypes.sizeof(rk.struct_rknpu_submit)}?1:-1];
    static int submit(U64 task){{
      struct Submit args={{5,6000,0,{tasks*3},0,0,task,0,0,0,7,-1,{{0,0,0,0,0,{tasks},{tasks},{tasks},{2*tasks},{tasks}}}}};
      if(request({submit_request}UL,&args))return 1;status[1]++;status[3]+={tasks*3};return 0;
    }}
    typedef struct{{float *out;const float *q,*k,*v;int phase,begin,end,invalid;}} Work;
    static void part(Work *w){{
      uint4 bad=(uint4)0;
      for(int index=w->begin;index<w->end;index++){{
        int head=index/{rows},row=index%{rows},tile=row/64,local=row%64;
        if(w->phase==0){{
          for(int c=0;c<64;c+=4){{
            int src=index*64+c;
            *(half4*)(qa+(head*{tiles}+tile)*4096+(c/8)*512+local*8+c%8)=__builtin_convertvector(*(const float4*)(w->q+src),half4);
            *(half4*)(ka+head*{rows*64}+(row/16)*1024+(c/32)*512+(row%16)*32+c%32)=
              __builtin_convertvector(*(const float4*)(w->k+src),half4);
            if(row%4==0){{
              float4 a=*(const float4*)(w->v+src),b=*(const float4*)(w->v+src+64),
                     d=*(const float4*)(w->v+src+128),e=*(const float4*)(w->v+src+192);
              __fp16 *target=va+head*{rows*64}+(c/16)*{rows*16}+(row/32)*512+(c%16)*32+row%32;
              *(half4*)(target)=__builtin_convertvector((float4){{a[0],b[0],d[0],e[0]}},half4);
              *(half4*)(target+32)=__builtin_convertvector((float4){{a[1],b[1],d[1],e[1]}},half4);
              *(half4*)(target+64)=__builtin_convertvector((float4){{a[2],b[2],d[2],e[2]}},half4);
              *(half4*)(target+96)=__builtin_convertvector((float4){{a[3],b[3],d[3],e[3]}},half4);
            }}
          }}
        }}else if(w->phase==1){{
          if(row%4)continue;
          float p[4][{rows}] __attribute__((aligned(16)));float4 maximum[4];
          for(int j=0;j<4;j++)maximum[j]=(float4)-__builtin_inff();
          for(int c=0;c<{rows};c+=4){{
            const float4 *source=(const float4*)(scores+(head*{tiles}+tile)*{64*rows}+(c/4)*256+local*4);
            for(int j=0;j<4;j++){{float4 value=source[j]*.125f;
              int4 invalid=(__builtin_bit_cast(uint4,value)&(uint4)0x7f800000)==(uint4)0x7f800000;
              bad|=__builtin_bit_cast(uint4,invalid);value=invalid?(float4)0:value;
              *(float4*)(p[j]+c)=value;maximum[j]=value>maximum[j]?value:maximum[j];}}
          }}
          float inverse[4];
          for(int j=0;j<4;j++){{float m=maximum[j][0];for(int i=1;i<4;i++)if(maximum[j][i]>m)m=maximum[j][i];
            inverse[j]=exponentials(p[j],m);}}
          for(int c=0;c<{rows};c+=4){{
            __fp16 *target=prob+(head*{tiles}+tile)*{64*rows}+(c/8)*512+local*8+c%8;
            for(int j=0;j<4;j++)*(half4*)(target+j*8)=__builtin_convertvector(*(const float4*)(p[j]+c)*(float4)inverse[j],half4);
          }}
        }}else{{
          for(int c=0;c<64;c+=4)*(float4*)(w->out+(row*{self.heads}+head)*64+c)=
            *(const float4*)(pv+(head*{tiles}+tile)*4096+(c/4)*256+local*4);
        }}
      }}
      w->invalid=__builtin_reduce_or(bad)!=0;
    }}
    static void *worker(void *opaque){{part((Work*)opaque);return (void*)0;}}
    static int parallel(int phase,float *out,const float *q,const float *k,const float *v){{
      Work work[4];thread_t threads[3];int created[3];
      for(int i=0;i<4;i++)work[i]=(Work){{out,q,k,v,phase,{self.batch_heads*rows}*i/4,{self.batch_heads*rows}*(i+1)/4}};
      for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;if(!created[i])part(&work[i]);}}
      part(&work[3]);for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
      int invalid=0;for(int i=0;i<4;i++)invalid|=work[i].invalid;return invalid;
    }}
    static int valid(const float *input){{
      uint4 bad=(uint4)0;
      for(int i=0;i<{self.heads*rows*64};i+=4){{uint4 bits=*(const uint4*)(input+i);float4 v=*(const float4*)(input+i);
        bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;bad|=(v>(float4)65504)|(v<(float4)-65504);}}
      return __builtin_reduce_or(bad)==0;
    }}
    {self.softmax.bridge() if self.softmax else ''}
    void rockchip_vision_attention_native(float *out,const float *q,const float *k,const float *v){{
      for(int i=0;i<10;i++)status[i]=0;
      {timed(4,'if(!valid(q)||!valid(k)||!valid(v)){status[0]=1;return;}'+('if(reset())return;' if self.reset_context else ''))}
      for(int first=0;first<{self.heads};first+={self.batch_heads}){{
        {timed(5,f'parallel(0,out,q+first*{rows*64},k+first*{rows*64},v+first*{rows*64});')}
        {timed(6,sync(self.q,1)+sync(self.k,1)+sync(self.v,1)+submit(self.qk_tasks)+scores_to_cpu)}
        {timed(7,softmax)}
        {timed(8,probabilities_from_cpu+submit(self.pv_tasks)+sync(self.result,2))}
        {timed(9,'parallel(2,out+first*64,q,k,v);')}
      }}
      {timed(4,'reset();') if self.reset_context else ''}
    }}'''

  def program(self, out:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
    source = self.source()
    inputs = (out,q,k,v)
    sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_vision_attention_native'))
    renderer = _renderer()
    info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
    return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
               UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))

  def execute(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
    result = Tensor.empty(1,self.rows,self.heads,64,device='CPU',dtype=dtypes.float).custom_kernel(q,k,v,fxn=self.program)[0].realize()
    return result.transpose(1,2)

  def __call__(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
    if self.closed: raise RuntimeError('Prepared NPU attention has been released')
    if any(t.device!='CPU' or t.dtype is not dtypes.float or t.shape!=(1,self.heads,self.rows,64) for t in (q,k,v)):
      raise ValueError('Prepared NPU attention requires matching CPU FP32 Q/K/V tensors')
    with self.dev.lock:
      self.dev.check_healthy()
      result = self.jit(q.contiguous().realize(),k.contiguous().realize(),v.contiguous().realize())
      self.dev.submit_count += self.status[1]
      self.dev.task_count += self.status[3]
      if self.status[0]==2:
        self.dev.poisoned = True
        raise OSError(self.status[2],'Prepared NPU attention submission/synchronization failed')
      if self.status[0]==3:
        self.dev.poisoned = True
        raise RuntimeError('Prepared NPU attention returned nonfinite scores')
      if self.status[0]==4:
        self.dev.poisoned = True
        raise RuntimeError(f'Shared GPU softmax failed: {self.status[2]}')
      if self.status[0]: raise ValueError('Prepared NPU attention requires finite FP16-representable inputs')
      return result

  def close(self):
    if self.closed: return
    if self.softmax is not None: self.softmax.close()
    for buf in reversed(self.owned): self.dev.free(buf)
    self.owned.clear()
    self.closed = True

  def phase_timings(self) -> dict[str,float]:
    return {name:self.status[i+4]/1e6 for i,name in enumerate(('guard_reset','pack','qk','softmax','pv','unpack'))}

  def __del__(self):
    if hasattr(self,'closed'): self.close()
