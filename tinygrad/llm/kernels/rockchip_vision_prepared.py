"""Prepared three-core FP16 vision MLP with GPU GELU and split-K down projection.

Weights are immutable after preparation. Private NPU intermediates are shared
with the GPU and never exposed as mutable model parameters.
"""
from __future__ import annotations
import array, ctypes, math
from dataclasses import replace
from typing import cast
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import ceildiv, round_up
from tinygrad.llm.kernels.rockchip import RockchipLinear, pack_split_weight
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_softmax import SharedNPUKernel
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipDevice, _cmd, _matmul_body
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

class PreparedVisionMLP:
  __slots__ = ('dev','owned','closed','softmax','status','libc','fc1','fc2','bias1','bias2','packed_weights','kept','weights',
               'guards','storage_guards','source_guards','biases','q','scores','probabilities','result','qk_tasks','pv_tasks','jit','__weakref__')

  def __init__(self, fc1:RockchipLinear, fc2:RockchipLinear):
    if ((fc1.depth,fc1.columns,fc2.depth,fc2.columns)!=(768,3072,3072,768) or fc1.prefill!='on' or fc2.prefill!='on'
        or fc1.weight.shape!=(3072,768) or fc2.weight.shape!=(768,3072)
        or any(bias is None or bias.device!='CPU' or bias.dtype is not dtypes.half for bias in (fc1.bias,fc2.bias))):
      raise ValueError('Prepared vision MLP requires qualified FP16 768/3072 projections')
    assert fc1.bias is not None and fc2.bias is not None
    if fc1.bias.shape!=(3072,) or fc2.bias.shape!=(768,):raise ValueError('Prepared vision MLP bias shape')
    Tensor.realize(fc1.weight,fc2.weight,fc1.bias,fc2.bias)
    fc1.prepare()
    fc2.prepare()
    if len(fc1.packed)!=1 or len(fc2.packed)!=1:raise ValueError('Prepared vision MLP requires unsplit output weights')
    if any(weight.dtype is not dtypes.half or weight.device!='ROCKCHIP' for weight in (fc1.packed[0],fc2.packed[0])):
      raise ValueError('Prepared vision MLP requires FP16 packed NPU weights')
    self.dev = cast(RockchipDevice,Device['ROCKCHIP'])
    self.owned:list[RockchipBuffer] = []
    self.closed = False
    self.softmax:SharedNPUKernel|None = None
    self.status, self.libc = (ctypes.c_uint64*10)(),ctypes.CDLL(None)
    self.fc1,self.fc2 = fc1,fc2
    self.bias1,self.bias2 = fc1.bias,fc2.bias
    self.packed_weights:tuple[Tensor,...] = (fc1.packed[0],pack_split_weight(fc2.weight).realize())
    self.kept = tuple(cast(Buffer,weight.uop.buffer) for weight in self.packed_weights)
    self.weights = tuple(buf._buf for buf in self.kept)
    self.guards = tuple((weight.uop,buf,buf._buf.meta.version) for weight,buf in zip(self.packed_weights,self.kept))
    self.source_guards = (fc1.packed[0].uop,fc2.packed[0].uop,fc1.weight.uop,fc2.weight.uop)
    self.biases = (self.bias1.uop,self.bias2.uop)
    parameters = (fc1.weight,fc2.weight,self.bias1,self.bias2,fc1.packed[0],fc2.packed[0],self.packed_weights[1])
    self.storage_guards = tuple((storage_buf:=cast(Buffer,param.uop.buffer),storage_buf.get_storage()) for param in parameters)
    for weight in self.packed_weights:
      if not weight.to('CPU').isfinite().all().item():raise ValueError('Nonfinite prepared weight')
    try:
      self.q = self.allocate(1024*768*2)
      self.scores = self.allocate(1024*3072*4)
      self.probabilities = self.allocate(1024*3072*2)
      self.result = self.allocate(3*1024*768*4)
      self.qk_tasks = self.make_chain(self.q,self.weights[0],self.scores,768,3072,64)
      self.pv_tasks = self.make_chain(self.probabilities,self.weights[1],self.result,3072,768,64,True)
      for buf in (self.scores,self.result):self.dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      bias = cast(list[float],self.bias1.float().tolist())
      if not all(map(math.isfinite,bias)) or not self.bias2.isfinite().all().item():raise ValueError('Nonfinite prepared bias')
      def literal(value):
        text = format(float(value),'.9g')
        return text+('f' if '.' in text or 'e' in text else '.0f')
      source = '''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
      __constant float bias[3072]={BIAS};
      static float4 bounded(float4 x) {
        x=clamp(x,(float4)-24.f,(float4)27.f);
        int4 q=convert_int4_rtz(x+select((float4).5f,(float4)-.5f,x<0));float4 s=x-convert_float4(q);
        float4 p=((((((.00015359208919107914f*s+.0013392627006396651f)*s+.009618384763598442f)*s+
          .055503472685813904f)*s+.24022644758224487f)*s+.6931471824645996f)*s+1.f);
        return p*as_float4((q+127)<<23);
      }
      __kernel void shared_gelu(__global const float *raw,__global half *packed,__global uint *invalid) {
        for(int index=get_global_id(0)*4;index<1024*3072;index+=12288*4) {
          int core=index/1048576,inside=index%1048576,tile=inside/65536,position=inside%65536;
          int row=tile*64+(position/4)%64,col=core*1024+(position/256)*4;
          float4 value=vload4(0,raw+index)+vload4(0,bias+col);
          if(!all(isfinite(value))) {atomic_or(invalid,1u);continue;}
          float4 cube=(value*value)*value,inner=(value+cube*.044715f)*.7978845608028654f;
          float4 sigmoid=1.f/(1.f+bounded((inner*2.f)*-1.4426950216293335f));
          float4 activated=(value*.5f)*(1.f+(2.f*sigmoid-1.f));
          if(!all(isfinite(activated)) || any(fabs(activated)>65504.f)) {atomic_or(invalid,1u);continue;}
          int dst=TILE_DST;
          vstore_half4_rte(activated,0,packed+dst);
        }
      }'''.replace('BIAS',','.join(map(literal,bias)))
      destination = '(col/1024)*1048576+(row/64)*65536+((col%1024)/8)*512+(row%64)*8+col%8'
      source = source.replace('TILE_DST',destination)
      self.softmax = SharedNPUKernel(self.dev,(self.scores,self.probabilities),source,'shared_gelu',12288)
      self.jit = TinyJit(self.execute)
      self.dev.programs.add(self)
    except BaseException:
      self.close()
      raise

  def make_chain(self, lhs, weight, out, depth, columns, tile, split=False):
    width = columns if split else columns//3
    depth = depth//3 if split else depth
    bodies = [_matmul_body(min(tile,1024-row),width,depth,lhs.dma+((core*1024 if split else 0)+row)*depth*2,
                          weight.dma+core*width*depth*2,out.dma+(core*1024*width+row*width)*4)
              for core in range(3) for row in range(0,1024,tile)]
    count = ceildiv(1024,tile)
    stride = round_up(max(map(len,bodies))+4,2)
    commands = self.allocate(len(bodies)*stride*8)
    tasks = self.allocate(len(bodies)*ctypes.sizeof(rk.struct_rknpu_task),rk.RKNPU_MEM_KERNEL_MAPPING)
    ctypes.memset(commands.addr,0,commands.size)
    for i,body in enumerate(bodies):
      following = i+1 if (i+1)%count else None
      tail = ((_cmd(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,(commands.dma+following*stride*8)&0xfffffff0),
               _cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,ceildiv(len(bodies[following]),2)+1)) if following is not None else
              (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS)))
      words = array.array('Q',body+tail+(_cmd(rk.TARGET_VERSION,0),_cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)))
      ctypes.memmove(commands.addr+i*stride*8,words.buffer_info()[0],len(words)*8)
      desc = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,commands.dma+i*stride*8)
      ctypes.memmove(tasks.addr+i*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
    for buf in (commands,tasks):self.dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    return tasks

  def source(self):
    assert self.softmax is not None
    ioctl = ctypes.cast(self.libc.ioctl,ctypes.c_void_p).value
    sync_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_mem_sync)<<16)|(ord('d')<<8)|0x45
    submit_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_submit)<<16)|(ord('d')<<8)|0x41
    def sync(buf,flag):return f'if(sync_mem({buf.meta.obj_addr}ULL,{buf.size}ULL,{flag}))return;'
    output = '''int offset=(row/64)*64*768+(c/4*64+row%64)*4;
          float4 value=(*(const float4*)(raw+offset)+*(const float4*)(raw+1024*768+offset))+
            *(const float4*)(raw+2*1024*768+offset);'''
    return f'''{_vector_exp2_source()}
    typedef unsigned long long U64;typedef unsigned U32;
    typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
    typedef unsigned long thread_t;
    extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
    extern int pthread_join(thread_t,void**);extern int *__errno_location(void);
    static U64 *status=(U64*){ctypes.addressof(self.status)}ULL;
    static __fp16 *packed=(__fp16*){self.q.addr}ULL;
    static float *raw=(float*){self.result.addr}ULL;
    static int request(unsigned long code,void *value){{
      if(((int(*)(int,unsigned long,void*)){ioctl}ULL)({self.dev.fd.fd},code,value)){{status[0]=2;status[2]=*__errno_location();return 1;}}return 0;
    }}
    static int sync_mem(U64 object,U64 size,U64 flag){{U64 args[4]={{flag,object,0,size}};return request({sync_request}UL,args);}}
    struct Submit{{U32 flags,timeout,start,count,counter;int priority;U64 task,reg,base,user;U32 mask;int fence;U32 subcores[10];}};
    typedef char check_submit[sizeof(struct Submit)=={ctypes.sizeof(rk.struct_rknpu_submit)}?1:-1];
    static int submit(U64 task,U32 count){{
      struct Submit args={{5,6000,0,count*3,0,0,task,0,0,0,7,-1,{{0,0,0,0,0,count,count,count,2*count,count}}}};
      if(request({submit_request}UL,&args))return 1;status[1]++;status[3]+=count*3;return 0;
    }}
    typedef struct{{float *out;const float *x;const __fp16 *bias;const float *residual;int phase,begin,end,invalid;}} Work;
    static void part(Work *w){{
      uint4 bad=(uint4)0;
      for(int row=w->begin;row<w->end;row++)for(int c=0;c<768;c+=4){{
        if(w->phase==0){{
          float4 value=*(const float4*)(w->x+row*768+c);
          bad|=(__builtin_bit_cast(uint4,value)&(uint4)0x7f800000)==(uint4)0x7f800000;
          bad|=(value>(float4)65504)|(value<(float4)-65504);
          bad|=(__builtin_bit_cast(uint4,*(const float4*)(w->residual+row*768+c))&(uint4)0x7f800000)==(uint4)0x7f800000;
          *(half4*)(packed+(row/64)*64*768+(c/8)*512+(row%64)*8+c%8)=__builtin_convertvector(value,half4);
        }}else{{
          {output}
          *(float4*)(w->out+row*768+c)=(value+__builtin_convertvector(*(const half4*)(w->bias+c),float4))+*(const float4*)(w->residual+row*768+c);
        }}
      }}
      w->invalid=__builtin_reduce_or(bad)!=0;
    }}
    static void *worker(void *opaque){{part((Work*)opaque);return (void*)0;}}
    static int parallel(int phase,float *out,const float *x,const __fp16 *bias,const float *residual){{
      Work work[4];thread_t threads[3];int created[3],bad=0;
      for(int i=0;i<4;i++)work[i]=(Work){{out,x,bias,residual,phase,i*256,(i+1)*256,0}};
      for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;if(!created[i])part(&work[i]);}}
      part(&work[3]);for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
      for(int i=0;i<4;i++)bad|=work[i].invalid;return bad;
    }}
    {self.softmax.bridge('shared_gelu')}
    void prepared_vision_mlp(float *out,const float *x,const __fp16 *bias,const float *residual){{
      for(int i=0;i<10;i++)status[i]=0;
      if(parallel(0,out,x,bias,residual)){{status[0]=1;return;}}
      {sync(self.q,1)}
      if(submit({self.qk_tasks.meta.obj_addr}ULL,16))return;
      if(shared_gelu()){{if(status[0]!=2)status[0]=status[2]==1?3:4;return;}}
      if(submit({self.pv_tasks.meta.obj_addr}ULL,16))return;
      {sync(self.result,2)}
      parallel(1,out,x,bias,residual);
    }}'''

  def program(self, out:UOp, x:UOp, bias:UOp, residual:UOp) -> UOp:
    source = self.source()
    inputs = (out,x,bias,residual)
    sink = UOp.sink(*inputs,arg=KernelInfo('prepared_vision_mlp'))
    renderer = _renderer()
    info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
    return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                       UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))

  def execute(self, x:Tensor, residual:Tensor) -> Tensor:
    return Tensor.empty(1,1024,768,device='CPU',dtype=dtypes.float).custom_kernel(x,self.bias2,residual,fxn=self.program)[0].realize()

  def __call__(self, x:Tensor, residual:Tensor) -> Tensor:
    if self.closed:raise RuntimeError('Prepared MLP closed')
    if any(t.device!='CPU' or t.dtype is not dtypes.float or t.shape!=(1,1024,768) for t in (x,residual)):raise ValueError('Unsupported MLP input')
    if any(buf._storage is not storage for buf,storage in self.storage_guards):
      raise RuntimeError('Prepared MLP immutable parameter allocation changed')
    if (len(self.fc1.packed)!=1 or len(self.fc2.packed)!=1 or
        (self.fc1.packed[0].uop,self.fc2.packed[0].uop,self.fc1.weight.uop,self.fc2.weight.uop)!=self.source_guards):
      raise RuntimeError('Prepared MLP source weight changed')
    for packed,guard in zip(self.packed_weights,self.guards):
      if packed.uop is not guard[0] or packed.uop.buffer is not guard[1] or guard[1]._buf.meta.version!=guard[2]:
        raise RuntimeError('Prepared MLP immutable weight changed')
    if (self.fc1.bias is not self.bias1 or self.fc2.bias is not self.bias2 or (self.bias1.uop,self.bias2.uop)!=self.biases):
      raise RuntimeError('Prepared MLP immutable bias changed')
    with self.dev.lock:
      self.dev.check_healthy()
      result = self.jit(x.contiguous().realize(),residual.contiguous().realize())
      self.dev.submit_count+=self.status[1]
      self.dev.task_count+=self.status[3]
      if self.status[0] in (2,4):
        self.dev.poisoned=True
        raise RuntimeError(f'Prepared MLP device failure: {list(self.status[:4])}')
      if self.status[0]:raise ValueError('Prepared MLP nonfinite or FP16 overflow')
      return result

  def allocate(self, size:int, flags:int=0) -> RockchipBuffer:
    buf = self.dev.allocate(size,flags)
    self.owned.append(buf)
    return buf

  def close(self):
    if self.closed:return
    if self.softmax is not None:self.softmax.close()
    for buf in reversed(self.owned):self.dev.free(buf)
    self.owned.clear()
    self.packed_weights,self.kept,self.weights,self.guards = (),(),(),()
    self.storage_guards = ()
    self.closed=True

  def __reduce__(self):raise TypeError('Prepared vision MLP is process-local')

  def __del__(self):
    if hasattr(self,'closed'):self.close()
