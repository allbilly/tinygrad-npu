"""CPU Q/K head normalization, rotary embeddings and FP16 KV writes for oNPU."""
import functools
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv, round_up
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_PREFILL_CPU_TUNING, shared_program
from tinygrad.llm.kernels.rockchip_padding import mark_zero_tail
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def _balanced_head_order(heads:int, kv_heads:int) -> tuple[int,...]:
  """Keep fixed worker ranges while distributing the extra K/V-store work."""
  total=heads+kv_heads
  capacity=[total*(i+1)//4-total*i//4 for i in range(4)]
  keys=[0]*4
  for _ in range(kv_heads):
    index=min((i for i in range(4) if keys[i]<capacity[i]),key=lambda i:(capacity[i]+keys[i],i))
    keys[index]+=1
  order:list[int]=[]
  q,k=0,heads
  for size,count in zip(capacity,keys):
    order.extend(range(q,q+size-count))
    q+=size-count
    order.extend(range(k,k+count))
    k+=count
  return tuple(order)


@shared_program
@functools.cache
def _prepare(out:UOp, cache:UOp, projected:UOp|tuple[UOp,...], qnorm:UOp, knorm:UOp, freqs:UOp, start:UOp|None, rows:UOp|None,
             heads:int, kv_heads:int, dim:int, context:int, capacity:int, eps:float, static_start:int,
             raw_scales:tuple[UOp,...]=(), weight_scales:UOp|None=None, tile_rows:tuple[int,...]=(),
             static_rows:int|None=None, normalize:bool=True, balance_heads:bool=False, bias:UOp|None=None) -> UOp:
  qtype,ktype = ('__fp16' if p.dtype is dtypes.half else 'float' for p in (qnorm,knorm))
  qload = '__builtin_convertvector(*(const half4*)(qn+c),float4)' if qtype=='__fp16' else '*(const float4*)(qn+c)'
  kload = '__builtin_convertvector(*(const half4*)(kn+c),float4)' if ktype=='__fp16' else '*(const float4*)(kn+c)'
  btype = '__fp16' if bias is not None and bias.dtype is dtypes.half else 'float'
  bias_load = '__builtin_convertvector(*(const half4*)(src->bias+column),float4)' if btype=='__fp16' else '*(const float4*)(src->bias+column)'
  bias_field = f'const {btype} *bias;' if bias is not None else ''
  bias_param = f',const {btype} *bias' if bias is not None else ''
  columns = (heads+2*kv_heads)*dim
  if isinstance(projected,tuple):
    assert weight_scales is not None and len(projected)==len(raw_scales)==len(tile_rows)
    width = round_up(ceildiv(round_up(columns,32),3),32)
    parts = [(first,min(width,round_up(columns,32)-first)) for first in range(0,round_up(columns,32),width)]
    if len(parts)!=3: parts = [(0,round_up(columns,32))]
    index = f'{parts[-1][0]}*m+local*{parts[-1][1]}+column-{parts[-1][0]}'
    for first,size in reversed(parts[:-1]): index = f'(column<{first+size}?{first}*m+local*{size}+column-{first}:{index})'
    source_type = 'const Sources *'
    definitions = f'''typedef int int4 __attribute__((aligned(16),ext_vector_type(4)));
    typedef struct {{const int *raw[{len(projected)}];const float *sa[{len(projected)}],*sw;
      {bias_field}}} Sources;
    static inline float4 read_projected(const Sources *src,int row,int column) {{
      {'#pragma clang fp contract(off)' if bias is not None else ''}
      int tile=row/{tile_rows[0]},local=row%{tile_rows[0]},m=tile=={len(tile_rows)-1}?{tile_rows[-1]}:{tile_rows[0]};
      int position={index};float4 v=__builtin_convertvector(*(const int4*)(src->raw[tile]+position),float4);
      return (v*(float4)src->sa[tile][local])*(*(const float4*)(src->sw+column)){f'+{bias_load}' if bias is not None else ''};
    }}'''
    parameters = ','.join([*(f'const int *r{i}' for i in range(len(projected))),
                           *(f'const float *a{i}' for i in range(len(raw_scales))),'const float *sw'])
    raw_array = ','.join(f'r{i}' for i in range(len(projected)))
    scale_array = ','.join(f'a{i}' for i in range(len(raw_scales)))
    initialize = 'Sources storage={{'+raw_array+'},{'+scale_array+'},sw'+(',bias' if bias is not None else '')+'};const Sources *src=&storage;'
    values = (*projected,*raw_scales,weight_scales)
  else:
    source_type = 'const float *'
    definitions = f'''static inline float4 read_projected(const float *src,int row,int column) {{
      return *(const float4*)(src+row*{columns}+column);
    }}'''
    parameters,initialize = 'const float *src',''
    values = (projected,)
  parallel = capacity>=16
  # Spread query heads and the more expensive key/value heads across all four workers.
  head_index = f'(work%{(heads+kv_heads)//4})*4+work/{(heads+kv_heads)//4}' if balance_heads and parallel and (heads+kv_heads)%4==0 else 'work'
  head_order_source = ''
  if balance_heads and parallel and (heads+kv_heads)%4:
    order=','.join(map(str,_balanced_head_order(heads,kv_heads)))
    head_order_source=f'static const int work_heads[{heads+kv_heads}]={{{order}}};'
    head_index='work_heads[work]'
  declarations = f'''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {{float *out;__fp16 *cache;{source_type}src;const {qtype} *qn;const {ktype} *kn;
    const float *freq;int first,rows,begin,end;}} Work;
  static void *prepare_worker(void*);''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,cache,src,qn,kn,freq,first,active,{heads+kv_heads}*i/4,{heads+kv_heads}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,prepare_worker,&work[i])==0;
      if(!created[i])prepare_part(out,cache,src,qn,kn,freq,first,active,work[i].begin,work[i].end);}}
    prepare_part(out,cache,src,qn,kn,freq,first,active,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else \
    f'prepare_part(out,cache,src,qn,kn,freq,first,active,0,{heads+kv_heads});'
  worker = '''static void *prepare_worker(void *opaque) {
    Work *w=(Work*)opaque;prepare_part(w->out,w->cache,w->src,w->qn,w->kn,w->freq,w->first,w->rows,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  normalization = f'''float sum=((acc[0]+acc[1])+acc[2])+acc[3];
      float inverse=1.0f/__builtin_sqrtf(sum*{1/dim:.17g}f+{eps:.17g}f);
      for(int c=0;c<{dim};c+=4) {{float4 v=*(const float4*)(normal+c),weight=h<{heads}?{qload}:{kload};
        *(float4*)(normal+c)=(v*(float4)inverse)*weight;}}''' if normalize else ''
  source = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {definitions}
  {head_order_source}
  static void prepare_part(float*,__fp16*,{source_type},const {qtype}*,const {ktype}*,const float*,int,int,int,int);
  {declarations}
  void rockchip_prepare_qkv(float *out,__fp16 *cache,{parameters},
    const {qtype} *qn,const {ktype} *kn,const float *freq{bias_param}{
    ',const int *start' if start is not None else ''}{',const int *rows' if rows is not None else ''}) {{
    {initialize}int first={'*start' if start is not None else static_start},active={
      '*rows' if rows is not None else capacity if static_rows is None else static_rows};
    if(first<0 || active<1 || active>{capacity} || first+active>{context}) {{
      for(int i=0;i<{heads*capacity*dim};i++)out[i]=__builtin_nanf("");return;
    }}
    {dispatch}
  }}
  static void prepare_part(float *out,__fp16 *cache,{source_type}src,const {qtype} *qn,const {ktype} *kn,const float *freq,
                            int first,int active,int begin,int end) {{
    for(int work=begin;work<end;work++)for(int r=0;r<{capacity};r++) {{
      int h={head_index};
      if(r>=active) {{if(h<{heads})for(int c=0;c<{dim};c+=4)*(float4*)(out+(h*{capacity}+r)*{dim}+c)=(float4)0;continue;}}
      float4 acc=(float4)0;float normal[{dim}] __attribute__((aligned(16)));
      for(int c=0;c<{dim};c+=4) {{float4 v=read_projected(src,r,h*{dim}+c);*(float4*)(normal+c)=v;{
        'acc=acc+v*v;' if normalize else ''}}}
      {normalization}
      for(int c=0;c<{dim//2};c+=4) {{
        float4 a=*(float4*)(normal+c),b=*(float4*)(normal+c+{dim//2});
        float4 cosine=*(const float4*)(freq+(first+r)*{dim}+c),sine=*(const float4*)(freq+(first+r)*{dim}+c+{dim//2});
        float4 left=a*cosine-b*sine,right=b*cosine+a*sine;
        if(h<{heads}) {{*(float4*)(out+(h*{capacity}+r)*{dim}+c)=left;*(float4*)(out+(h*{capacity}+r)*{dim}+c+{dim//2})=right;}}
        else {{int target=((h-{heads})*{context}+first+r)*{dim}+c;
          *(half4*)(cache+target)=__builtin_convertvector(left,half4);
          *(half4*)(cache+target+{dim//2})=__builtin_convertvector(right,half4);
          *(half4*)(cache+target+{kv_heads*context*dim})=__builtin_convertvector(read_projected(src,r,(h+{kv_heads})*{dim}+c),half4);
          *(half4*)(cache+target+{kv_heads*context*dim+dim//2})=__builtin_convertvector(
            read_projected(src,r,(h+{kv_heads})*{dim}+c+{dim//2}),half4);
        }}
      }}
    }}
  }}
  {worker}'''
  inputs = (out,cache,*values,qnorm,knorm,freqs)+((bias,) if bias is not None else ())
  inputs += ((start,) if start is not None else ())+((rows,) if rows is not None else ())
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_prepare_qkv'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,1),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _normalizers(qnorm:Tensor|None, knorm:Tensor|None, dim:int) -> tuple[Tensor,Tensor,bool]:
  if (qnorm is None)!=(knorm is None):raise ValueError('Q/K normalization weights must both be present or absent')
  normalize = qnorm is not None
  # Retain the existing kernel ABI; these constant operands are never loaded
  # when normalization is disabled in the generated source.
  if qnorm is None:qnorm=knorm=Tensor.ones(dim,device='CPU',dtype=dtypes.float)
  assert knorm is not None
  return qnorm,knorm,normalize


def prepare_qkv(projected:Tensor, cache:Tensor, qnorm:Tensor|None, knorm:Tensor|None, freqs:Tensor, start:int|UOp,
                heads:int, eps:float) -> tuple[Tensor,Tensor]:
  """Prepare rotary Q/K and cache V; absent Q/K weights skip normalization."""
  if (projected.device!='CPU' or projected.dtype is not dtypes.float or projected.ndim!=3 or projected.shape[0]!=1
      or cache.device!='CPU' or cache.dtype is not dtypes.half or cache.ndim!=5 or cache.shape[:2]!=(2,1)):
    raise ValueError('Unsupported QKV preparation layout')
  kv_heads,context,dim = cache.max_shape[2:]
  capacity = projected.max_shape[1]
  qnorm,knorm,normalize = _normalizers(qnorm,knorm,dim)
  if (not 1<=kv_heads<=heads<=128 or heads%kv_heads or not 8<=dim<=256 or dim%8 or not 1<=context<=4096
      or not 1<=capacity<=context or projected.shape[-1]!=(heads+2*kv_heads)*dim
      or any(w.device!='CPU' or w.shape!=(dim,) or w.dtype not in (dtypes.float,dtypes.half) for w in (qnorm,knorm))
      or freqs.device!='CPU' or freqs.dtype is not dtypes.float or freqs.shape!=(context,dim)):
    raise ValueError('Unsupported QKV preparation geometry')
  position = (Tensor(start,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(start,UOp) else ()
  count = projected.shape[1]
  rows = (Tensor(count,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(count,UOp) else ()
  result = Tensor.empty(1,heads,capacity,dim,device='CPU',dtype=dtypes.float).custom_kernel(
    cache,projected.pad_to(projected.max_shape).contiguous(),qnorm.contiguous(),knorm.contiguous(),freqs.contiguous(),*position,*rows,
    fxn=lambda o,k,p,qw,kw,f,*r:_prepare(o,k,p,qw,kw,f,r[0] if position else None,r[-1] if rows else None,
                                       heads,kv_heads,dim,context,capacity,eps,start if isinstance(start,int) else 0,
                                       normalize=normalize,balance_heads=bool(ONPU_PREFILL_CPU_TUNING)))
  return mark_zero_tail(result[0][:,:,:count],result[0],'qkv'),result[1]


def prepare_qkv_int8(raw:list[Tensor], sa:list[Tensor], sw:Tensor, sizes:list[int], cache:Tensor, qnorm:Tensor|None, knorm:Tensor|None,
                     freqs:Tensor, start:int|UOp, heads:int, eps:float, count:int|UOp, *, bias:Tensor|None=None) -> tuple[Tensor,Tensor]:
  if cache.device!='CPU' or cache.dtype is not dtypes.half or cache.ndim!=5 or cache.shape[:2]!=(2,1):
    raise ValueError('Unsupported INT8 QKV cache')
  kv_heads,context,dim=cache.max_shape[2:]
  capacity=sum(sizes)
  columns=(heads+kv_heads*2)*dim
  qnorm,knorm,normalize = _normalizers(qnorm,knorm,dim)
  if (not raw or len(raw)!=len(sa) or len(raw)!=len(sizes) or any(m!=sizes[0] for m in sizes[:-1])
      or not 0<sizes[-1]<=sizes[0] or not 1<=capacity<=context<=4096 or dim%8 or not 8<=dim<=256
      or isinstance(count,int) and not 1<=count<=capacity
      or not 1<=kv_heads<=heads<=128 or heads%kv_heads or sw.device!='CPU' or sw.dtype is not dtypes.float or sw.shape!=(columns,)
      or any(r.device!='CPU' or r.dtype is not dtypes.int32 or r.numel()!=m*round_up(columns,32) for r,m in zip(raw,sizes))
      or any(s.device!='CPU' or s.dtype is not dtypes.float or s.shape!=(m,) for s,m in zip(sa,sizes))
      or any(w.device!='CPU' or w.shape!=(dim,) or w.dtype not in (dtypes.float,dtypes.half) for w in (qnorm,knorm))
      or freqs.device!='CPU' or freqs.dtype is not dtypes.float or freqs.shape!=(context,dim)):
    raise ValueError('Unsupported INT8 QKV geometry')
  if bias is not None and (bias.device!='CPU' or bias.dtype not in (dtypes.half,dtypes.float) or bias.shape!=(columns,)):
    raise ValueError('INT8 QKV bias must be a matching CPU FP16/FP32 vector')
  bias_values=(bias.contiguous(),) if bias is not None else ()
  position=(Tensor(start,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(start,UOp) else ()
  active=(Tensor(count,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(count,UOp) else ()
  n=len(raw)
  def build(o:UOp,k:UOp,*args:UOp):
    other=args[n*2+4+len(bias_values):]
    return _prepare(o,k,tuple(args[:n]),args[n*2+1],args[n*2+2],args[n*2+3],other[0] if position else None,
                    other[-1] if active else None,heads,kv_heads,dim,context,capacity,eps,start if isinstance(start,int) else 0,
                    tuple(args[n:n*2]),args[n*2],tuple(sizes),count if isinstance(count,int) else None,
                    normalize=normalize,balance_heads=bool(ONPU_PREFILL_CPU_TUNING),bias=args[n*2+4] if bias_values else None)
  result=Tensor.empty(1,heads,capacity,dim,device='CPU',dtype=dtypes.float).custom_kernel(
    cache,*raw,*sa,sw,qnorm.contiguous(),knorm.contiguous(),freqs.contiguous(),*bias_values,*position,*active,fxn=build)
  return mark_zero_tail(result[0][:,:,:count],result[0],'qkv'),result[1]
