"""Opt-in FP16 GPT-2 adapter using the selective CPU/NPU backend."""
from __future__ import annotations
import contextlib, ctypes, functools, math, subprocess
from collections import OrderedDict
from dataclasses import replace
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip import RockchipLinear, configure
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.llm.kernels.rockchip_greedy import greedy_argmax as greedy_argmax
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _softmax_functions() -> str:
  from tinygrad.codegen import to_program
  sources = []
  for name in ('gpt2_exp','gpt2_probability'):
    output = UOp.param(0,dtypes.float if name=='gpt2_exp' else dtypes.half,(1,))
    value = UOp.param(1,dtypes.half if name=='gpt2_exp' else dtypes.float,(1,))
    other = UOp.param(2,dtypes.half,(1,))
    result = (((value[0].load()-other[0].load()).cast(dtypes.float)*(1/math.log(2))).exp2() if name=='gpt2_exp' else
              value[0].load().cast(dtypes.half)*other[0].load().reciprocal())
    sink = output[0].store(result).sink(arg=KernelInfo(name))
    program = to_program(sink,_renderer())
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    sources.append(source.replace(f'void {name}(',f'static __attribute__((always_inline)) void {name}('))
  return '\n'.join(sources)


@functools.cache
def _attention_stage(out:UOp, a:UOp, cache:UOp, length:UOp|None, heads:int, rows:int, dim:int, context:int, pv:bool|None,
                     static_length:int|None=None, pool:int|None=None) -> UOp:
  name = 'gpt2_attention' if pv is None else 'gpt2_pv' if pv else 'gpt2_qk'
  total = '*length' if length is not None else str(static_length)
  body = f'''for(int row=0;row<{rows};row++) {{
    int used=total-{rows}+row+1;
    float4 acc[{dim//4}]={{{{0}}}};
    for(int t=0;t<used;t++) {{
      float p=a[(head*{rows}+row)*{context}+t];
      const half4 *value=(const half4*)(cache+({heads}*{context}+head*{context}+t)*{dim});
      for(int c=0;c<{dim//4};c++)acc[c]=acc[c]+(float4)p*__builtin_convertvector(value[c],float4);
    }}
    for(int c=0;c<{dim//4};c++)((half4*)(out+(head*{rows}+row)*{dim}))[c]=__builtin_convertvector(acc[c],half4);
  }}''' if pv else f'''for(int row=0;row<{rows};row++) {{
    int used=total-{rows}+row+1;
    const half4 *q=(const half4*)(a+(head*{rows}+row)*{dim});
    for(int t=0;t<used;t++) {{
      const half4 *key=(const half4*)(cache+(head*{context}+t)*{dim});
      float4 acc={{0}};
      for(int c=0;c<{dim//4};c++)acc=acc+__builtin_convertvector(q[c],float4)*__builtin_convertvector(key[c],float4);
      out[(head*{rows}+row)*{context}+t]=(__fp16)((((acc[0]+acc[1])+acc[2])+acc[3])*{1/math.sqrt(dim):.17g}f);
    }}
    for(int t=used;t<{context};t++)out[(head*{rows}+row)*{context}+t]=(__fp16)(-__builtin_inff());
  }}'''
  if pv is None:
    body = f'''for(int row=0;row<{rows};row++) {{
      int used=total-{rows}+row+1;
      const half4 *q=(const half4*)(a+(head*{rows}+row)*{dim});
      __fp16 scores[{context}],maximum=(__fp16)(-__builtin_inff());float exponent[{context}],sum=0;
      for(int t=0;t<used;t++) {{
        const half4 *key=(const half4*)(cache+(head*{context}+t)*{dim});float4 acc={{0}};
        for(int c=0;c<{dim//4};c++)acc=acc+__builtin_convertvector(q[c],float4)*__builtin_convertvector(key[c],float4);
        scores[t]=(__fp16)((((acc[0]+acc[1])+acc[2])+acc[3])*{1/math.sqrt(dim):.17g}f);
        if(scores[t]>maximum)maximum=scores[t];
      }}
      for(int t=0;t<used;t++){{gpt2_exp(exponent+t,scores+t,&maximum);sum+=(float)(__fp16)exponent[t];}}
      __fp16 denominator=(__fp16)sum;
      float4 acc[{dim//4}]={{{{0}}}};
      for(int t=0;t<used;t++) {{
        __fp16 probability;gpt2_probability(&probability,exponent+t,&denominator);
        const half4 *value=(const half4*)(cache+({heads}*{context}+head*{context}+t)*{dim});
        for(int c=0;c<{dim//4};c++)acc[c]=acc[c]+(float4)(float)probability*__builtin_convertvector(value[c],float4);
      }}
      for(int c=0;c<{dim//4};c++)((half4*)(out+(head*{rows}+row)*{dim}))[c]=__builtin_convertvector(acc[c],half4);
    }}'''
  parallel = rows>=16 and heads>=4
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {__fp16 *out;const __fp16 *a,*cache;int total,begin,end;} Work;
  static void *worker(void *opaque) {
    Work *w=(Work*)opaque;part(w->out,w->a,w->cache,w->total,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,a,cache,{total},{heads}*i/4,{heads}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(out,a,cache,{total},work[i].begin,work[i].end);}}
    part(out,a,cache,{total},work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(out,a,cache,{total},0,{heads});'
  if pool is not None and rows==1 and heads>=4:
    declarations += f'''static void pool_part(void **args,int total,int group) {{
      part(args[0],args[1],args[2],total,{heads}*group/4,{heads}*(group+1)/4);
    }}'''
    dispatch = f'''if({total}>=16){{void *args[4]={{out,(void*)a,(void*)cache,0}};
      if(!((int(*)(void(*)(void**,int,int),void**,int)){pool}ULL)(pool_part,args,{total}))return;
    }}part(out,a,cache,{total},0,{heads});'''
  source = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {_softmax_functions() if pv is None else ''}
  static void part(__fp16 *out,const __fp16 *a,const __fp16 *cache,int total,int begin,int end) {{
    for(int head=begin;head<end;head++) {{ {body} }}
  }}
  {declarations}
  void {name}(__fp16 *out,const __fp16 *a,const __fp16 *cache{',const int *length' if length is not None else ''}) {{
    if({total}<{rows} || {total}>{context}) {{
      for(int i=0;i<{heads*rows*(context if pv is False else dim)};i++)out[i]=(__fp16)__builtin_nanf("");return;
    }}
    {dispatch}
  }}'''
  inputs = (out,a,cache)+((length,) if length is not None else ())
  sink = UOp.sink(*inputs,arg=KernelInfo(name))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def half_attention(query:Tensor, cache:Tensor, start:UOp|int, fused:bool=True, pool:int|None=None) -> Tensor:
  """Bounded causal QK/PV with stock FP16 softmax between the two stages."""
  if (query.device!='CPU' or cache.device!='CPU' or query.dtype is not dtypes.half or cache.dtype is not dtypes.half
      or query.ndim!=4 or cache.ndim!=5 or query.shape[0]!=1 or cache.shape[:3]!=(2,1,query.shape[1])
      or query.shape[-1]!=cache.shape[-1]): raise ValueError('Unsupported GPT-2 attention layout')
  _,heads,rows,dim = query.shape
  context = cache.shape[-2]
  if not all(isinstance(v,int) for v in (heads,rows,dim,context)): raise ValueError('GPT-2 attention requires static geometry')
  assert isinstance(heads,int) and isinstance(rows,int) and isinstance(dim,int) and isinstance(context,int)
  if not 1<=heads<=128 or not 1<=rows<=context<=4096 or not 4<=dim<=256 or dim%4: raise ValueError('Unsupported GPT-2 attention geometry')
  if isinstance(start,int) and not 0<=start<=context-rows: raise ValueError('GPT-2 attention exceeds context')
  total = start+rows
  lengths = (Tensor(total,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(total,UOp) else ()
  if fused:
    return Tensor.empty(1,heads,rows,dim,device='CPU',dtype=dtypes.half).custom_kernel(query.contiguous(),cache,*lengths,
      fxn=lambda o,q,c,*t:_attention_stage(o,q,c,t[0] if t else None,heads,rows,dim,context,None,total if isinstance(total,int) else None,pool))[0]
  scores = Tensor.empty(1,heads,rows,context,device='CPU',dtype=dtypes.half).custom_kernel(query.contiguous(),cache,*lengths,
    fxn=lambda o,q,c,*t:_attention_stage(o,q,c,t[0] if t else None,heads,rows,dim,context,False,total if isinstance(total,int) else None))[0]
  probability = scores[:,:,:,:total].softmax(-1).pad_to((1,heads,rows,context)).contiguous()
  return Tensor.empty(1,heads,rows,dim,device='CPU',dtype=dtypes.half).custom_kernel(probability,cache,*lengths,
    fxn=lambda o,p,c,*t:_attention_stage(o,p,c,t[0] if t else None,heads,rows,dim,context,True,total if isinstance(total,int) else None))[0]


class GPT2Linear(RockchipLinear):
  def __call__(self, x:Tensor) -> Tensor:
    return self.source(x) if any(size==0 for size in x.shape) else super().__call__(x)

  def npu(self, x:Tensor) -> Tensor:
    # GPT-2's eager KV assignment otherwise lets mixed-device copy dependencies
    # enter its CPU store kernel. Keep a concrete CPU activation boundary.
    return super().npu(x.contiguous().realize())


class GPT2Head(GPT2Linear):
  def __call__(self, x:Tensor) -> Tensor:
    # Custom kernels cannot inherit the stock graph's final-row pruning.
    return super().__call__(x if x.shape[-2]==0 else x[:,-1:])


class GPT2Attention:
  """Keep each head's KV rows adjacent while preserving stock FP16 attention."""
  def __init__(self, source, context:int, fast:bool, pool=None):
    self.c_attn,self.c_proj = source.c_attn,source.c_proj
    self.n_heads,self.dim,self.head_dim = source.n_heads,source.dim,source.head_dim
    self.context = context
    self.fast = fast
    self.pool = pool
    if hasattr(source,'cache_kv'): self.cache_kv = source.cache_kv.permute(0,1,3,2,4).contiguous().realize()

  def __call__(self, x:Tensor, start:UOp, mask:Tensor|None) -> Tensor:
    batch,rows,_ = x.shape
    qkv = self.c_attn(x.half()).reshape(batch,rows,3,self.n_heads,self.head_dim)
    query,key,value = [qkv[:,:,i].transpose(1,2) for i in range(3)]
    if not hasattr(self,'cache_kv'):
      self.cache_kv = Tensor.zeros(2,batch,self.n_heads,self.context,self.head_dim,device='CPU',dtype=dtypes.half).contiguous().realize()
    self.cache_kv[:,:,:,start:start+rows].assign(Tensor.stack(key,value)).realize()
    keys,values = self.cache_kv[0][:,:,:start+rows],self.cache_kv[1][:,:,:start+rows]
    pool = self.pool.address if self.pool is not None and self.pool.active else None
    out = half_attention(query,self.cache_kv,start,pool=pool) if self.fast and batch==1 and rows else \
      query.scaled_dot_product_attention(keys,values,mask)
    return self.c_proj(out.transpose(1,2).reshape(batch,rows,self.dim))


class PhaseJit:
  def __init__(self, jit, layers:list[RockchipLinear], phase:str, native:bool, helper=None, cleaner:int|None=None, pool=None):
    self.jit,self.layers,self.phase,self.native = jit,layers,phase,native
    self.helper = helper
    self.cleaner = cleaner
    self.pool = pool

  @contextlib.contextmanager
  def scope(self):
    with contextlib.ExitStack() as stack:
      if self.helper is not None and getattr(self.captured,'warmed',False): stack.enter_context(self.helper)
      if self.pool is not None: stack.enter_context(self.pool)
      yield

  @property
  def captured(self): return self.jit.captured

  @captured.setter
  def captured(self, value): self.jit.captured = value

  def __call__(self, *args):
    for layer in self.layers: layer.phase = self.phase
    try:
      with self.scope():
        result = self.jit(*args)
      if self.native and self.captured is not None and not isinstance(self.captured,RockchipCapturedJit):
        self.captured = RockchipCapturedJit(self.captured,self.cleaner)
      return result
    finally:
      for layer in self.layers: layer.phase = None


class PrefillJit:
  def __init__(self, forward, layers:list[RockchipLinear], native:bool, cleaner:int|None=None):
    self.forward,self.layers,self.native = forward,layers,native
    self.cleaner = cleaner
    self.variants:OrderedDict[tuple,PhaseJit] = OrderedDict()

  def __call__(self, tokens:Tensor|UOp, start:UOp, temperature:float=0.0):
    if not isinstance(tokens,Tensor) or tokens.shape[-1]==0: return self.forward(tokens,start,temperature)
    key = tokens.shape,temperature
    if key not in self.variants:
      self.variants[key] = PhaseJit(TinyJit(self.forward),self.layers,'prefill',self.native,cleaner=self.cleaner)
      if len(self.variants)>8: self.variants.popitem(last=False)
    self.variants.move_to_end(key)
    return self.variants[key](tokens,start,temperature)


def configure_gpt2(model, native_replay:bool=True, packed_cache:bool=True, fast_attention:bool=True, wake_helper:bool=True,
                   cache_cleaning:bool=True, cpu_pool:bool=True) -> dict:
  """Enable explicit three-core Linears, bounded prefill JITs, and native replay."""
  if getattr(model,'rockchip_gpt2',None) is not None: raise ValueError('GPT-2 NPU adapter is already configured')
  if model.wte.weight.device!='CPU' or model.wte.weight.dtype is not dtypes.half:
    raise ValueError('GPT-2 NPU inference requires DEV=CPU HALF=1')
  if not 1<=model.max_context<=model.wpe.weight.shape[0]: raise ValueError('GPT-2 context exceeds its learned position embeddings')
  placement = configure(model,prefill='on',decode='on',tile_columns=24576)
  pool = None
  if cpu_pool and packed_cache and fast_attention and any(block.attn.n_heads>=4 for block in model.h):
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    try:
      pool = attention_pool()
      placement['cpu_pool'] = {'enabled':True,'workers':[5,6,7],'main':4,'decode_min_context':16}
    except OSError as exc: placement['cpu_pool'] = {'enabled':False,'reason':str(exc)}
  if packed_cache:
    for block in model.h: block.attn = GPT2Attention(block.attn,model.max_context,fast_attention,pool)
  layers:list[RockchipLinear] = []
  sites = [(obj,name) for block in model.h for obj,names in [(block.attn,('c_attn','c_proj')),(block.mlp,('c_fc','c_proj'))] for name in names]
  for obj,name in [*sites,(model,'lm_head')]:
    layer = getattr(obj,name)
    if not isinstance(layer,RockchipLinear): raise ValueError('Unsupported GPT-2 projections')
    adapted = (GPT2Head if obj is model else GPT2Linear)(layer.source,layer.prefill,layer.decode,layer.tile_columns)
    adapted.packed = layer.packed
    setattr(obj,name,adapted)
    layers.append(adapted)
  cleaner = None
  if cache_cleaning and native_replay:
    from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
    try:
      library = cache_cleaner()
      cleaner = ctypes.cast(library.clean_to,ctypes.c_void_p).value
      placement['cache_cleaning'] = {'enabled':True,'range_limit':1048576,'line_bytes':64,'from_device':'driver'}
    except (OSError,subprocess.TimeoutExpired) as exc: placement['cache_cleaning'] = {'enabled':False,'reason':str(exc)}
  prefill = PrefillJit(model.forward,layers,native_replay,cleaner)
  helper = None
  if wake_helper and native_replay:
    from tinygrad.llm.kernels.rockchip_idle import idle_helper
    try:
      helper = idle_helper()
      placement['idle_helper'] = {'enabled':True,'cpu':0,'nice':19,'scope':'native decode calls only'}
    except (OSError,RuntimeError) as exc: placement['idle_helper'] = {'enabled':False,'reason':str(exc)}
  decode = PhaseJit(TinyJit(model.forward),layers,'decode',native_replay,helper,cleaner,pool)
  model.forward,model.forward_jit = prefill,decode
  model.rockchip_gpt2 = {'placement':placement,'prefill':prefill,'decode':model.forward_jit}
  model.rockchip_argmax = greedy_argmax
  from tinygrad.llm.kernels.rockchip_gpt2_decode import decode_many
  model.rockchip_decode_many = lambda token,position,count:decode_many(model,token,position,count)
  placement.update(native_replay=native_replay,packed_cache=packed_cache,fast_attention=packed_cache and fast_attention)
  return placement
