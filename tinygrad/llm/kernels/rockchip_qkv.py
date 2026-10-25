"""CPU Q/K head normalization, rotary embeddings and FP16 KV writes for oNPU."""
import functools
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv, round_up
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _prepare(out:UOp, cache:UOp, projected:UOp|tuple[UOp,...], qnorm:UOp, knorm:UOp, freqs:UOp, start:UOp|None, rows:UOp|None,
             heads:int, kv_heads:int, dim:int, context:int, capacity:int, eps:float, static_start:int,
             raw_scales:tuple[UOp,...]=(), weight_scales:UOp|None=None, tile_rows:tuple[int,...]=(),
             static_rows:int|None=None) -> UOp:
  qtype,ktype = ('__fp16' if p.dtype is dtypes.half else 'float' for p in (qnorm,knorm))
  qload = '__builtin_convertvector(*(const half4*)(qn+c),float4)' if qtype=='__fp16' else '*(const float4*)(qn+c)'
  kload = '__builtin_convertvector(*(const half4*)(kn+c),float4)' if ktype=='__fp16' else '*(const float4*)(kn+c)'
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
    typedef struct {{const int *raw[{len(projected)}];const float *sa[{len(projected)}],*sw;}} Sources;
    static inline float4 read_projected(const Sources *src,int row,int column) {{
      int tile=row/{tile_rows[0]},local=row%{tile_rows[0]},m=tile=={len(tile_rows)-1}?{tile_rows[-1]}:{tile_rows[0]};
      int position={index};float4 v=__builtin_convertvector(*(const int4*)(src->raw[tile]+position),float4);
      return (v*(float4)src->sa[tile][local])*(*(const float4*)(src->sw+column));
    }}'''
    parameters = ','.join([*(f'const int *r{i}' for i in range(len(projected))),
                           *(f'const float *a{i}' for i in range(len(raw_scales))),'const float *sw'])
    initialize = f'''Sources storage={{{{{','.join(f'r{i}' for i in range(len(projected)))}}},
      {{{','.join(f'a{i}' for i in range(len(raw_scales)))}}},sw}};const Sources *src=&storage;'''
    values = (*projected,*raw_scales,weight_scales)
  else:
    source_type = 'const float *'
    definitions = f'''static inline float4 read_projected(const float *src,int row,int column) {{
      return *(const float4*)(src+row*{columns}+column);
    }}'''
    parameters,initialize = 'const float *src',''
    values = (projected,)
  parallel = capacity>=16
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
  source = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {definitions}
  static void prepare_part(float*,__fp16*,{source_type},const {qtype}*,const {ktype}*,const float*,int,int,int,int);
  {declarations}
  void rockchip_prepare_qkv(float *out,__fp16 *cache,{parameters},const {qtype} *qn,const {ktype} *kn,const float *freq{
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
    for(int h=begin;h<end;h++)for(int r=0;r<{capacity};r++) {{
      if(r>=active) {{if(h<{heads})for(int c=0;c<{dim};c+=4)*(float4*)(out+(h*{capacity}+r)*{dim}+c)=(float4)0;continue;}}
      float4 acc=(float4)0;float normal[{dim}] __attribute__((aligned(16)));
      for(int c=0;c<{dim};c+=4) {{float4 v=read_projected(src,r,h*{dim}+c);*(float4*)(normal+c)=v;acc=acc+v*v;}}
      float sum=((acc[0]+acc[1])+acc[2])+acc[3];
      float inverse=1.0f/__builtin_sqrtf(sum*{1/dim:.17g}f+{eps:.17g}f);
      for(int c=0;c<{dim};c+=4) {{float4 v=*(const float4*)(normal+c),weight=h<{heads}?{qload}:{kload};
        *(float4*)(normal+c)=(v*(float4)inverse)*weight;}}
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
  inputs = (out,cache,*values,qnorm,knorm,freqs)+((start,) if start is not None else ())+((rows,) if rows is not None else ())
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_prepare_qkv'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,1),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def prepare_qkv(projected:Tensor, cache:Tensor, qnorm:Tensor, knorm:Tensor, freqs:Tensor, start:int|UOp,
                heads:int, eps:float) -> tuple[Tensor,Tensor]:
  if (projected.device!='CPU' or projected.dtype is not dtypes.float or projected.ndim!=3 or projected.shape[0]!=1
      or cache.device!='CPU' or cache.dtype is not dtypes.half or cache.ndim!=5 or cache.shape[:2]!=(2,1)):
    raise ValueError('Unsupported QKV preparation layout')
  kv_heads,context,dim = cache.max_shape[2:]
  capacity = projected.max_shape[1]
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
                                       heads,kv_heads,dim,context,capacity,eps,start if isinstance(start,int) else 0))
  return result[0][:,:,:count],result[1]


def prepare_qkv_int8(raw:list[Tensor], sa:list[Tensor], sw:Tensor, sizes:list[int], cache:Tensor, qnorm:Tensor, knorm:Tensor,
                     freqs:Tensor, start:int|UOp, heads:int, eps:float, count:int|UOp) -> tuple[Tensor,Tensor]:
  if cache.device!='CPU' or cache.dtype is not dtypes.half or cache.ndim!=5 or cache.shape[:2]!=(2,1):
    raise ValueError('Unsupported INT8 QKV cache')
  kv_heads,context,dim=cache.max_shape[2:]
  capacity=sum(sizes)
  columns=(heads+kv_heads*2)*dim
  if (not raw or len(raw)!=len(sa) or len(raw)!=len(sizes) or any(m!=sizes[0] for m in sizes[:-1])
      or not 0<sizes[-1]<=sizes[0] or not 1<=capacity<=context<=4096 or dim%8 or not 8<=dim<=256
      or isinstance(count,int) and not 1<=count<=capacity
      or not 1<=kv_heads<=heads<=128 or heads%kv_heads or sw.device!='CPU' or sw.dtype is not dtypes.float or sw.shape!=(columns,)
      or any(r.device!='CPU' or r.dtype is not dtypes.int32 or r.numel()!=m*round_up(columns,32) for r,m in zip(raw,sizes))
      or any(s.device!='CPU' or s.dtype is not dtypes.float or s.shape!=(m,) for s,m in zip(sa,sizes))
      or any(w.device!='CPU' or w.shape!=(dim,) or w.dtype not in (dtypes.float,dtypes.half) for w in (qnorm,knorm))
      or freqs.device!='CPU' or freqs.dtype is not dtypes.float or freqs.shape!=(context,dim)):
    raise ValueError('Unsupported INT8 QKV geometry')
  position=(Tensor(start,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(start,UOp) else ()
  active=(Tensor(count,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(count,UOp) else ()
  n=len(raw)
  def build(o:UOp,k:UOp,*args:UOp):
    other=args[n*2+4:]
    return _prepare(o,k,tuple(args[:n]),args[n*2+1],args[n*2+2],args[n*2+3],other[0] if position else None,
                    other[-1] if active else None,heads,kv_heads,dim,context,capacity,eps,start if isinstance(start,int) else 0,
                    tuple(args[n:n*2]),args[n*2],tuple(sizes),count if isinstance(count,int) else None)
  result=Tensor.empty(1,heads,capacity,dim,device='CPU',dtype=dtypes.float).custom_kernel(
    cache,*raw,*sa,sw,qnorm.contiguous(),knorm.contiguous(),freqs.contiguous(),*position,*active,fxn=build)
  return result[0][:,:,:count],result[1]
