"""Bounded causal CPU attention for mixed CPU/Rockchip inference."""
import functools, math, platform
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _renderer():
  arch = {'amd64':'x86_64','aarch64':'arm64'}.get(machine:=platform.machine().lower(),machine)
  return ClangRenderer(Target('CPU',arch=arch+',native',renderer='CLANG'))


@functools.cache
def _sigmoid_source() -> str:
  # Use the same transcendental legalization and rounding as the regular CPU path.
  from tinygrad.codegen import to_program
  out,source = UOp.param(0,dtypes.float,(1,)),UOp.param(1,dtypes.float,(1,))
  value = source[0].load()
  result = (value*(-1/math.log(2))).exp2().alu(Ops.ADD,UOp.const(1.0,dtypes.float)).reciprocal()
  sink = out[0].store(result).sink(arg=KernelInfo('rockchip_sigmoid'))
  program = to_program(sink,_renderer())
  text = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  assert 'void rockchip_sigmoid(' in text
  return text.replace('void rockchip_sigmoid(','static __attribute__((always_inline)) void rockchip_sigmoid(')


@functools.cache
def _decode_heads(heads:int, kv_heads:int, dim:int, context:int, fast_softmax:bool=False) -> str:
  group = heads//kv_heads
  tile = min(4,16//group)
  if dim%(4*tile): tile = 1
  queries = '\n'.join(f'const float4 *q{i}=(const float4*)(query+(kv*{group}+{i})*{dim});' for i in range(group))
  acc = ','.join(f'a{i}={{0}}' for i in range(group))
  dot = '\n'.join(f'a{i}=a{i}+q{i}[c]*key;' for i in range(group))
  scores = '\n'.join(f'''p[{i}][t]=(((a{i}[0]+a{i}[1])+a{i}[2])+a{i}[3])*{1/math.sqrt(dim):.17g}f;
    if(p[{i}][t]>maximum[{i}])maximum[{i}]=p[{i}][t];''' for i in range(group))
  maxima = ','.join('-__builtin_inff()' for _ in range(group))
  values = '\n'.join(f'float4 v{j}=__builtin_convertvector(value[c+{j}],float4);' for j in range(tile))
  pv_acc = ','.join(f'a{i}{j}={{0}}' for i in range(group) for j in range(tile))
  pv = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)p[{i}][t];' for i in range(group) for j in range(tile))
  stores = '\n'.join(f'((float4*)(out+(kv*{group}+{i})*{dim}))[c+{j}]=a{i}{j};' for i in range(group) for j in range(tile))
  normalize = (f'for(int i=0;i<{group};i++)rockchip_softmax_vector(p[i],total,maximum[i]);' if fast_softmax else f'''
    for(int i=0;i<{group};i++) {{float sum=0;
      for(int t=0;t<total;t++) {{p[i][t]=expf(p[i][t]-maximum[i]);sum+=p[i][t];}}
      for(int t=0;t<total;t++)p[i][t]/=sum;
    }}''')
  return f'''for(int kv=begin/{group};kv<end/{group};kv++) {{
    const __fp16 *keys=cache+kv*{context*dim},*values=keys+{kv_heads*context*dim};
    float p[{group}][{context}],maximum[{group}]={{{maxima}}};{queries}
    for(int t=0;t<total;t++) {{const half4 *source=(const half4*)(keys+t*{dim});float4 {acc};
      for(int c=0;c<{dim//4};c++) {{float4 key=__builtin_convertvector(source[c],float4);{dot}}}
      {scores}
    }}
    {normalize}
    for(int c=0;c<{dim//4};c+={tile}) {{float4 {pv_acc};
      for(int t=0;t<total;t++) {{const half4 *value=(const half4*)(values+t*{dim});{values}{pv}}}
      {stores}
    }}
  }}return;'''


@functools.cache
def _attention_rows(dim:int, context:int, max_rows:int, block:int=4, channel_tile:int=0, fast_softmax:bool=False,
                    causal:bool=True, kv_float:bool=False) -> str:
  """Reuse each KV load across queries without reordering dot sums."""
  if block not in (4,8,16) or channel_tile not in (0,1,2,4): raise ValueError('Unsupported CPU attention register tile')
  queries = '\n'.join(f'const float4 *q{i}=(const float4*)(query+(head*{max_rows}+row+{i})*{dim});' for i in range(block))
  acc = ','.join(f'a{i}={{0}}' for i in range(block))
  dot = '\n'.join(f'a{i}=a{i}+q{i}[c]*key;' for i in range(block))
  score = '\n'.join(f'''if(t<used[{i}]) {{float v=(((a{i}[0]+a{i}[1])+a{i}[2])+a{i}[3])*{1/math.sqrt(dim):.17g}f;
    p[{i}][t]=v;if(v>maximum[{i}])maximum[{i}]=v;}}''' for i in range(block))
  tile = channel_tile or (2 if dim%8==0 else 1)
  if dim%(4*tile): raise ValueError('CPU attention channel tile does not divide the head')
  kvtype = 'float4' if kv_float else 'half4'
  values = '\n'.join(f'float4 v{j}='+ (f'value[c+{j}]' if kv_float else f'__builtin_convertvector(value[c+{j}],float4)')+';'
                     for j in range(tile))
  pv_acc = ','.join(f'a{i}{j}={{0}}' for i in range(block) for j in range(tile))
  pv = '\n'.join('if(t<used['+str(i)+']) {'+''.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)p[{i}][t];' for j in range(tile))+'}'
                 for i in range(block))
  pv_common = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)p[{i}][t];' for i in range(block) for j in range(tile))
  stores = '\n'.join(f'((float4*)(out+(head*{max_rows}+row+{i})*{dim}))[c+{j}]=a{i}{j};'
                     for i in range(block) for j in range(tile))
  maxima = ','.join('-__builtin_inff()' for _ in range(block))
  normalize = (f'for(int i=0;i<{block};i++)rockchip_softmax_vector(p[i],used[i],maximum[i]);' if fast_softmax else f'''
    for(int i=0;i<{block};i++) {{
      float sum=0;for(int t=0;t<used[i];t++) {{p[i][t]=expf(p[i][t]-maximum[i]);sum+=p[i][t];}}
      for(int t=0;t<used[i];t++)p[i][t]/=sum;
    }}''')
  return f'''for(;row+{block}<=active_rows;row+={block}) {{
    int used[{block}];for(int i=0;i<{block};i++)used[i]={'total-active_rows+row+i+1' if causal else 'total'};
    float p[{block}][{context+(16 if block==8 and context>=4096 else 0)}],maximum[{block}]={{{maxima}}};
    {queries}
    for(int t=0;t<used[{block-1}];t++) {{
      const {kvtype} *source=(const {kvtype}*)(keys+t*{dim});float4 {acc};
      for(int c=0;c<{dim//4};c++) {{float4 key={'source[c]' if kv_float else '__builtin_convertvector(source[c],float4)'};{dot}}}
      {score}
    }}
    {normalize}
    for(int i=0;i<{block};i++) {{
      for(int t=used[i];t<used[{block-1}];t++)p[i][t]=0;
    }}
    for(int c=0;c<{dim//4};c+={tile}) {{
      float4 {pv_acc};
      for(int t=0;t<used[0];t++) {{const {kvtype} *value=(const {kvtype}*)(values+t*{dim});{values}{pv_common}}}
      for(int t=used[0];t<used[{block-1}];t++) {{const {kvtype} *value=(const {kvtype}*)(values+t*{dim});{values}{pv}}}
      {stores}
    }}
  }}'''


@functools.cache
def _attention(out:UOp, query:UOp, cache:UOp, length:UOp|None, heads:int, kv_heads:int, dim:int, context:int,
               rows:UOp|None=None, max_rows:int=1, static_length:int|None=None, blocked_rows:bool=True,
               query_rows:int|None=None, channel_tile:int=0, decode_grouped:bool=True, pool:int|None=None,
               fast_softmax:bool=False) -> UOp:
  if query_rows is None:
    query_rows = 8 if max_rows>=64 and (dim==128 and context<=1024 or dim==64 and context>=4096) else 4
  parallel = max_rows>=16 and heads>=4
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {float *out;const float *query;const __fp16 *cache;int total,rows,begin,end;} Work;
  static void *attention_worker(void*);''' if parallel else ''
  dispatch = f'''if(active_rows<16) attention_part(out,query,cache,total,active_rows,0,{heads});
    else {{
      Work work[4];thread_t threads[3];int created[3];
      for(int i=0;i<4;i++)work[i]=(Work){{out,query,cache,total,active_rows,{heads}*i/4,{heads}*(i+1)/4}};
      for(int i=0;i<3;i++){{
        created[i]=pthread_create(&threads[i],(void*)0,attention_worker,&work[i])==0;
        if(!created[i])attention_part(out,query,cache,total,active_rows,work[i].begin,work[i].end);
      }}
      attention_part(out,query,cache,total,active_rows,work[3].begin,work[3].end);
      for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
    }}''' if parallel else f'attention_part(out,query,cache,total,active_rows,0,{heads});'
  worker = '''static void *attention_worker(void *opaque) {
    Work *w=(Work*)opaque;attention_part(w->out,w->query,w->cache,w->total,w->rows,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  if pool is not None and max_rows==1 and (kv_heads>=4 or kv_heads>=2 and context>=4096):
    declarations += f'''static void pool_part(void **args,int total,int group) {{
      attention_part(args[0],args[1],args[2],total,1,({kv_heads}*group/4)*{heads//kv_heads},
                     ({kv_heads}*(group+1)/4)*{heads//kv_heads});
    }}'''
    dispatch = f'''if(total>={1024 if kv_heads<4 else 16}){{void *args[4]={{out,(void*)query,(void*)cache,0}};
      if(!((int(*)(void(*)(void**,int,int),void**,int)){pool}ULL)(pool_part,args,total))return;
    }}attention_part(out,query,cache,total,active_rows,0,{heads});'''
  elif pool is not None and parallel:
    declarations += f'''static void pool_prefill_part(void **args,int total,int group) {{
      attention_part(args[0],args[1],args[2],total,*(const int*)args[3],{heads}*group/4,{heads}*(group+1)/4);
    }}'''
    dispatch = f'''if(active_rows>=16){{void *args[4]={{out,(void*)query,(void*)cache,(void*)&active_rows}};
      if(!((int(*)(void(*)(void**,int,int),void**,int)){pool}ULL)(pool_prefill_part,args,total))return;
    }}{dispatch}'''
  source = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {_fast_softmax_source() if fast_softmax else ''}
  extern float expf(float);
  static void attention_part(float*,const float*,const __fp16*,int,int,int,int);
  {declarations}
  void rockchip_cpu_attention(float *out,const float *query,const __fp16 *cache{
      ',const int *length' if length is not None else ''}{',const int *rows' if rows is not None else ''}) {{
    const int total={'*length' if length is not None else str(static_length)},active_rows={'*rows' if rows is not None else str(max_rows)};
    if(active_rows<1 || active_rows>{max_rows} || total<active_rows || total>{context}) {{
      for(int i=0;i<{heads*max_rows*dim};i++)out[i]=__builtin_nanf("");return;
    }}
    {dispatch}
  }}
  static void attention_part(float *out,const float *query,const __fp16 *cache,int total,int active_rows,int begin,int end) {{
    {_decode_heads(heads,kv_heads,dim,context,fast_softmax) if decode_grouped and max_rows==1 and 1<heads//kv_heads<=8 else ''}
    for(int head=begin;head<end;head++) {{
      const __fp16 *keys=cache+(head/{heads//kv_heads})*{context*dim};
      const __fp16 *values=keys+{kv_heads*context*dim};
      int row=0;
      {_attention_rows(dim,context,max_rows,query_rows,channel_tile,fast_softmax) if blocked_rows and max_rows>=16 else ''}
      for(;row<active_rows;row++) {{
      const int used=total-active_rows+row+1;
      const float4 *q=(const float4*)(query+(head*{max_rows}+row)*{dim});
      float scores[{context}],maximum=-__builtin_inff(),denominator=0.0f;
      for(int t=0;t<used;t++) {{
        const half4 *key=(const half4*)(keys+t*{dim});float4 acc={{0}};
        for(int c=0;c<{dim//4};c++)acc=acc+q[c]*__builtin_convertvector(key[c],float4);
        float score=((acc[0]+acc[1])+acc[2])+acc[3];
        scores[t]=score*{1/math.sqrt(dim):.17g}f;
        if(scores[t]>maximum)maximum=scores[t];
      }}
      {'rockchip_softmax_vector(scores,used,maximum);denominator=1.0f;' if fast_softmax else
       'for(int t=0;t<used;t++){scores[t]=expf(scores[t]-maximum);denominator+=scores[t];}'}
      float4 accum[{dim//4}]={{{{0}}}};
      for(int t=0;t<used;t++) {{
        const half4 *value=(const half4*)(values+t*{dim});float p=scores[t]/denominator;
        for(int c=0;c<{dim//4};c++)accum[c]=accum[c]+(float4){{p,p,p,p}}*__builtin_convertvector(value[c],float4);
      }}
      for(int c=0;c<{dim//4};c++)((float4*)(out+(head*{max_rows}+row)*{dim}))[c]=accum[c];
      }}
    }}
  }}
  {worker}'''
  arch = {'amd64':'x86_64','aarch64':'arm64'}.get(machine:=platform.machine().lower(),machine)
  renderer = ClangRenderer(Target('CPU',arch=arch+',native',renderer='CLANG'))
  inputs = (out,query,cache)+((length,) if length is not None else ())+((rows,) if rows is not None else ())
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_attention'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),
             UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _cpu_attention(query:Tensor, cache:Tensor, length:int|UOp, decode_only:bool, pool:int|None=None, fast_softmax:bool=False) -> Tensor:
  if (query.device!='CPU' or cache.device!='CPU' or query.dtype is not dtypes.float or cache.dtype is not dtypes.half
      or query.ndim!=4 or cache.ndim!=5 or query.shape[0]!=1 or decode_only and query.shape[2]!=1
      or cache.shape[:2]!=(2,1) or query.shape[-1]!=cache.shape[-1]): raise ValueError('Unsupported CPU decode attention layout')
  heads,kv_heads,dim,context = query.shape[1],cache.shape[2],query.shape[3],cache.shape[3]
  if (not all(isinstance(value,int) for value in (heads,kv_heads,dim,context)) or heads%kv_heads or dim%4
      or not 4<=dim<=256 or not 1<=context<=8192): raise ValueError('Unsupported CPU decode attention geometry')
  assert isinstance(heads,int) and isinstance(kv_heads,int) and isinstance(dim,int) and isinstance(context,int)
  if isinstance(length,int) and not 1<=length<=context: raise ValueError('Invalid CPU decode attention length')
  max_rows = query.max_shape[2]
  if not 1<=max_rows<=4096: raise ValueError('Unsupported CPU attention query count')
  size = (Tensor(length,device='CPU',dtype=dtypes.int).reshape(1),) if isinstance(length,UOp) else ()
  rows = (Tensor(query.shape[2],device='CPU',dtype=dtypes.int).reshape(1),) if not isinstance(query.shape[2],int) and max_rows>1 else ()
  out = Tensor.empty(*query.max_shape,device='CPU',dtype=dtypes.float)
  result = out.custom_kernel(query.pad_to(query.max_shape).contiguous(),cache,*size,*rows,
    fxn=lambda o,q,k,*r:_attention(o,q,k,r[0] if size else None,heads,kv_heads,dim,context,
                                  r[-1] if rows else None,max_rows,length if isinstance(length,int) else None,
                                  pool=pool,fast_softmax=fast_softmax))[0]
  return result.shrink(tuple((0,size) for size in query.shape))


def decode_attention(query:Tensor, cache:Tensor, length:int|UOp, pool=None, fast_softmax:bool=False) -> Tensor:
  return _cpu_attention(query,cache,length,True,pool.address if pool is not None and pool.active else None,fast_softmax)
def prefill_attention(query:Tensor, cache:Tensor, length:int|UOp, fast_softmax:bool=False, pool=None) -> Tensor:
  address = pool.prefill_address if pool is not None and pool.active and query.max_shape[2]>=16 else None
  return _cpu_attention(query,cache,length,False,pool=address,fast_softmax=fast_softmax)


@functools.cache
def _unpack_tiles(out:UOp, parts:tuple[UOp,...], tile_rows:int, last_rows:int, columns:int, split:bool, parallel:bool|None=None) -> UOp:
  params = ','.join(f'const float *p{i}' for i in range(len(parts)))
  pointers = ','.join(f'p{i}' for i in range(len(parts)))
  rows = tile_rows*(len(parts)-1)+last_rows
  if parallel is None: parallel = rows*columns>=131072
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {float *out;const float *const *parts;int begin,end;} Work;
  static void *unpack_worker(void*);''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,parts,{(rows+3)//4}*i/4,{(rows+3)//4}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,unpack_worker,&work[i])==0;
      if(!created[i])unpack_part(out,parts,work[i].begin,work[i].end);}}
    unpack_part(out,parts,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else \
    f'unpack_part(out,parts,0,{(rows+3)//4});'
  worker = '''static void *unpack_worker(void *opaque) {
    Work *w=(Work*)opaque;unpack_part(w->out,w->parts,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  source = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  static void unpack_part(float*,const float *const*,int,int);
  {declarations}
  void rockchip_unpack_tiles(float *out,{params}) {{
    const float *parts[{len(parts)}]={{{pointers}}};
    {dispatch}
  }}
  static void unpack_part(float *out,const float *const *parts,int begin,int end) {{
    for(int block=begin*4;block<end*4;block+=4)for(int c=0;c<{columns};c+=4) {{
      for(int r=block;r<block+4 && r<{rows};r++) {{
        int i=r/{tile_rows},m=i=={len(parts)-1}?{last_rows}:{tile_rows};
        const float *src=parts[i]+(c/4*m+r-i*{tile_rows})*4;
        float4 value=*(const float4*)src;
        {'value=value+*(const float4*)(src+m*'+str(columns)+');value=value+*(const float4*)(src+2*m*'+str(columns)+');' if split else ''}
        *(float4*)(out+r*{columns}+c)=value;
      }}
    }}
  }}'''
  source += worker
  sink = UOp.sink(out,*parts,arg=KernelInfo('rockchip_unpack_tiles'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,1+len(parts))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,*parts)),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def unpack_tiles(parts:list[Tensor], split:bool=False) -> Tensor:
  """Unpack concatenated native row tiles, optionally adding three K partials."""
  if not parts or len(parts)>64 or parts[0].ndim!=(4 if split else 3): raise ValueError('Unsupported Rockchip output tile count')
  columns,tile_rows = parts[0].shape[-3]*4,parts[0].shape[-2]
  if (not isinstance(columns,int) or not isinstance(tile_rows,int) or not 1<=tile_rows<=64 or not 32<=columns<=24576 or columns%32
      or any(p.device!='CPU' or p.dtype is not dtypes.float or p.ndim!=(4 if split else 3)
             or p.shape[:-2]!=((3,columns//4) if split else (columns//4,)) or p.shape[-1]!=4
             or p.shape[-2]!=(parts[-1].shape[-2] if i==len(parts)-1 else tile_rows) for i,p in enumerate(parts))
      or not isinstance(parts[-1].shape[-2],int) or not 1<=parts[-1].shape[-2]<=tile_rows):
    raise ValueError('Unsupported Rockchip output layout')
  last_rows = parts[-1].shape[-2]
  out = Tensor.empty(tile_rows*(len(parts)-1)+last_rows,columns,device='CPU',dtype=dtypes.float)
  return out.custom_kernel(*parts,fxn=lambda o,*p:_unpack_tiles(o,p,tile_rows,last_rows,columns,split))[0]


@functools.cache
def _rms_norm(out:UOp, source:UOp, weight:UOp, rows:int, width:int, eps:float, parallel:bool|None=None) -> UOp:
  dtype = '__fp16' if weight.dtype is dtypes.half else 'float'
  load = '__builtin_convertvector(*(const half4*)(weight+c),float4)' if dtype=='__fp16' else '*(const float4*)(weight+c)'
  if parallel is None: parallel = rows*width>=131072
  declarations = f'''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {{float *out;const float *src;const {dtype} *weight;int begin,end;}} Work;
  static void *norm_worker(void*);''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,src,weight,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,norm_worker,&work[i])==0;
      if(!created[i])norm_part(out,src,weight,work[i].begin,work[i].end);}}
    norm_part(out,src,weight,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'norm_part(out,src,weight,0,{rows});'
  worker = '''static void *norm_worker(void *opaque) {
    Work *w=(Work*)opaque;norm_part(w->out,w->src,w->weight,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  text = f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  static void norm_part(float*,const float*,const {dtype}*,int,int);
  {declarations}
  void rockchip_cpu_rmsnorm(float *out,const float *src,const {dtype} *weight) {{{dispatch}}}
  static void norm_part(float *out,const float *src,const {dtype} *weight,int begin,int end) {{
    for(int r=begin;r<end;r++) {{
      float4 acc=(float4)0;
      for(int c=0;c<{width};c+=4) {{float4 value=*(const float4*)(src+r*{width}+c);acc=acc+value*value;}}
      float sum=((acc[0]+acc[1])+acc[2])+acc[3];
      float inverse=1.0f/__builtin_sqrtf(sum*{1/width:.17g}f+{eps:.17g}f);
      for(int c=0;c<{width};c+=4) {{float4 value=*(const float4*)(src+r*{width}+c),gamma={load};
        *(float4*)(out+r*{width}+c)=(value*(float4)inverse)*gamma;}}
    }}
  }}
  {worker}'''
  sink = UOp.sink(out,source,weight,arg=KernelInfo('rockchip_cpu_rmsnorm'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,source,weight)),UOp(Ops.SOURCE,arg=text),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(text))))


def rms_norm(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  width,rows = x.max_shape[-1],math.prod(x.max_shape[:-1])
  if (x.device!='CPU' or x.dtype is not dtypes.float or weight.device!='CPU' or weight.dtype not in (dtypes.float,dtypes.half)
      or weight.shape!=(width,) or not 4<=width<=8192 or width%4 or rows<1 or not math.isfinite(eps) or eps<=0):
    raise ValueError('Unsupported CPU normalization geometry')
  out = Tensor.empty(*x.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
    x.pad_to(x.max_shape).contiguous(),weight.contiguous(),fxn=lambda o,a,w:_rms_norm(o,a,w,rows,width,eps))[0]
  return out.shrink(tuple((0,s) for s in x.shape))


def _vector_exp2_source() -> str:
  # Same FP32 xexp2 polynomial, range reduction and exceptional cases as the
  # scalar CPU legalization above. Full half payloads are checked against it.
  return '''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 rockchip_exp2_vector(float4 x) {
    int4 nan=x!=x;
    float4 safe=((x==(float4)__builtin_inff())|(x==(float4)-__builtin_inff())|nan)?(float4)0:x;
    int4 q=__builtin_convertvector(safe+(safe<0?(float4)-.5f:(float4).5f),int4),q1=q>>1;
    float4 s=safe-__builtin_convertvector(q,float4);
    float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f)*
      __builtin_bit_cast(float4,(q1+127)<<23)*__builtin_bit_cast(float4,(q-q1+127)<<23);
    y=x>=128?(float4)__builtin_inff():y;
    y=x<-150?(float4)0:y;
    y=nan?(float4)__builtin_nanf(""):y;
    return y;
  }'''


def _vector_sigmoid_source() -> str:
  return _vector_exp2_source()+'''static __attribute__((always_inline)) float4 rockchip_sigmoid_vector(float4 value) {
    float4 x=value*-1.4426950216293335f;
    return 1.0f/(rockchip_exp2_vector(x)+1.0f);
  }'''


def _fast_softmax_source() -> str:
  return 'extern float expf(float);\n'+_vector_exp2_source()+'''static void rockchip_softmax_vector(float *values,int count,float maximum) {
    float sum=0;int t=0;
    for(;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);
      float4 y=rockchip_exp2_vector((x-(float4)maximum)*(float4)1.4426950216293335f);
      __builtin_memcpy(values+t,&y,16);
      sum+=y[0];sum+=y[1];sum+=y[2];sum+=y[3];
    }
    for(;t<count;t++){values[t]=expf(values[t]-maximum);sum+=values[t];}
    for(t=0;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);x=x/(float4)sum;__builtin_memcpy(values+t,&x,16);}
    for(;t<count;t++)values[t]/=sum;
  }'''


@functools.cache
def _swiglu_pack(out:UOp, combined:UOp|tuple[UOp,...], rows:int, depth:int, tile_rows:int, input_rows:int=0, vectorized:bool=True) -> UOp:
  tiled = isinstance(combined,tuple)
  inputs = combined if isinstance(combined,tuple) else (combined,)
  pointer_type = 'const float *const *' if tiled else 'const float *'
  parameters = ','.join(f'const float *src{i}' for i in range(len(inputs))) if tiled else 'const float *src'
  initialize = f'const float *src[{len(inputs)}]={{{",".join(f"src{i}" for i in range(len(inputs)))}}};' if tiled else ''
  parallel = rows*depth>=32768
  declarations = f'''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {{__fp16 *out;{pointer_type}src;int begin,end;}} Work;
  static void *pack_worker(void*);''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,src,{depth}*i/4,{depth}*(i+1)/4}};
    for(int i=0;i<3;i++){{
      created[i]=pthread_create(&threads[i],(void*)0,pack_worker,&work[i])==0;
      if(!created[i])pack_part(out,src,work[i].begin,work[i].end);
    }}
    pack_part(out,src,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
  ''' if parallel else f'pack_part(out,src,0,{depth});'
  worker = '''static void *pack_worker(void *opaque) {
    Work *w=(Work*)opaque;pack_part(w->out,w->src,w->begin,w->end);return (void*)0;
  }''' if parallel else ''
  indexing = (f'const int first=row/{input_rows}*{input_rows},im={rows}-first<{input_rows}?{rows}-first:{input_rows};'
              f'const int g=first*{depth*2}+(c/4*im+row-first)*4+c%4,u=g+{depth}*im;' if input_rows else
              f'const int g=row*{depth*2}+c,u=g+{depth};')
  vectorized = vectorized and bool(input_rows)
  if vectorized:
    loops = f'for(int c=begin;c<end;c+=4)for(int row=0;row<{rows};row++) {{'
    activation = f'''float4 gate=*(const float4*)(src+g),up=*(const float4*)(src+u);
      float4 reciprocal=rockchip_sigmoid_vector(gate);
      float4 activated=gate*reciprocal,value=activated*up;
      *(half4*)(out+start*{depth}+(c/8*m+row-start)*8+c%8)=__builtin_convertvector(value,half4);'''
  else:
    loops = f'for(int row=0;row<{rows};row++)for(int c=begin;c<end;c++) {{'
    activation = f'''float gate=src[g],up=src[u];
      float reciprocal;rockchip_sigmoid(&reciprocal,&gate);
      float activated=gate*reciprocal,value=activated*up;
      out[start*{depth}+(c/8*m+row-start)*8+c%8]=(__fp16)value;'''
  if tiled:
    if not input_rows or not vectorized: raise ValueError('Projection tiles require vector FP32 activation packing')
    loops = f'''for(int tile=0;tile<{len(inputs)};tile++) {{
      const float *src=sources[tile];
      const int first=tile*{input_rows},im=tile=={len(inputs)-1}?{rows-input_rows*(len(inputs)-1)}:{input_rows};
      for(int c=begin;c<end;c+=4)for(int row=first;row<first+im;row++) {{'''
    indexing = f'const int g=(c/4*im+row-first)*4,u=g+{depth}*im;'
  source = f'''{_vector_sigmoid_source() if vectorized else _sigmoid_source()}
  static void pack_part(__fp16*,{pointer_type},int,int);
  {declarations}
  void rockchip_swiglu_pack(__fp16 *out,{parameters}) {{{initialize}{dispatch}}}
  static void pack_part(__fp16 *out,{pointer_type}{'sources' if tiled else 'src'},int begin,int end) {{
    {loops}
      const int start=row/{tile_rows}*{tile_rows},m={rows}-start<{tile_rows}?{rows}-start:{tile_rows};
      {indexing}
      {activation}
    }}
    {'}' if tiled else ''}
  }}
  {worker}'''
  arch = {'amd64':'x86_64','aarch64':'arm64'}.get(machine:=platform.machine().lower(),machine)
  renderer = ClangRenderer(Target('CPU',arch=arch+',native',renderer='CLANG'))
  sink = UOp.sink(out,*inputs,arg=KernelInfo('rockchip_swiglu_pack'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs)+1)))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,*inputs)),
             UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def swiglu_pack(combined:Tensor, depth:int, tile_rows:int, input_rows:int=0, split_k:bool=False) -> Tensor:
  if (combined.device!='CPU' or combined.dtype is not dtypes.float or not input_rows and combined.shape[-1]!=depth*2
      or not 0<depth<=4096 or depth%32 or split_k and depth%96
      or not 1<=tile_rows<=min(64,65536//(depth//3 if split_k else depth))):
    raise ValueError('Unsupported CPU SwiGLU packing layout')
  rows = math.prod(combined.max_shape)//(depth*2)
  source = combined.pad_to(combined.max_shape).reshape(rows,depth*2).contiguous()
  out = Tensor.empty(rows*depth,device='CPU',dtype=dtypes.half)
  return out.custom_kernel(source,fxn=lambda o,x:_swiglu_pack(o,x,rows,depth,tile_rows,input_rows))[0]


def swiglu_pack_tiles(parts:list[Tensor], depth:int, tile_rows:int, split_k:bool=False) -> Tensor:
  if (not parts or len(parts)>64 or not 0<depth<=4096 or depth%32 or split_k and depth%96
      or not 1<=tile_rows<=min(64,65536//(depth//3 if split_k else depth))
      or any(p.device!='CPU' or p.dtype is not dtypes.float or p.ndim!=1 or p.max_shape[0]%(2*depth) for p in parts)):
    raise ValueError('Unsupported CPU projection tiles')
  sizes = [p.max_shape[0]//(2*depth) for p in parts]
  if not 1<=sizes[0]<=64 or any(m!=(sizes[-1] if i==len(sizes)-1 else sizes[0]) for i,m in enumerate(sizes)) \
      or not 1<=sizes[-1]<=sizes[0]: raise ValueError('Unsupported CPU projection row tiles')
  rows = sum(sizes)
  out = Tensor.empty(rows*depth,device='CPU',dtype=dtypes.half)
  return out.custom_kernel(*(p.contiguous() for p in parts),
    fxn=lambda o,*p:_swiglu_pack(o,p,rows,depth,tile_rows,sizes[0]))[0]
