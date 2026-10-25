"""Model-owned packed KV for bounded CPU prefill of the qualified 9/3-head decoder.

Keep the ordinary FP16 cache for decode. Pack only newly written positions and
any decode gap, sharing each KV load across grouped queries. Arithmetic stays
FP32 by default, preserving dot reductions and normalization. Optional FP16
QK uses FP32 softmax/PV and falls back on nonfinite intermediate scores. Packed FP32 values avoid
repeated FP16 conversion during prefill.
"""
import functools, platform
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _fast_softmax_source, decode_attention
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def _part_source(padded:int, capacity:int) -> str:
  tile,context,value_float,reciprocal = 2,padded,True,False
  acc = ','.join(f'a{i}_{j}={{0}}' for i in range(6) for j in range(4))
  dot = ''
  for j in range(4):
    dot += f'''{{int d=c*4+{j};float4 key=__builtin_convertvector(keys[(kv*{context//4}+t/4)*64+d],float4);
      float4 q0=((float4*)q[d])[0],q1=((float4*)q[d])[1];'''
    dot += '\n'.join(f'a{i}_{j}=a{i}_{j}+key*(float4)q{i//4}[{i%4}];' for i in range(6))+'}'
  scores = '\n'.join(f'''{{float4 score=(((a{i}_0+a{i}_1)+a{i}_2)+a{i}_3)*0.125f;
    for(int lane=0;lane<4;lane++)if(t+lane<used[1])p[(t+lane)*8+{i}]=t+lane<used[{i%2}]?score[lane]:-__builtin_inff();}}'''
                     for i in range(6))
  pv_acc = ','.join(f'a{i}{j}={{0}}' for i in range(6) for j in range(tile))
  values = '\n'.join(f'float4 v{j}='+(
    f'value[(kv*16+c+{j})*{context+16}+t]' if value_float else
    f'__builtin_convertvector(value[(kv*16+c+{j})*{context+16}+t],float4)')+';' for j in range(tile))
  common = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)pr{i//4}[{i%4}];' for i in range(6) for j in range(tile))
  tail = '\n'.join(f'if(t<used[{i%2}]){{'+''.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)pr{i//4}[{i%4}];'
                                                for j in range(tile))+'}' for i in range(6))
  stores = '\n'.join(f'''if({i%2}<count)((float4*)(out+((kv*3+{i//2})*{capacity}+row+{i%2})*64))[c+{j}]=a{i}{j};'''
                     for i in range(6) for j in range(tile))
  overrides = '\n'.join(f'''if(t>=tails[{i%2}]&&t<used[{i%2}])y{i//4}[{i%4}]=expf(p[t*8+{i}]-m{i//4}[{i%4}]);'''
                        for i in range(6))
  return f'''{_fast_softmax_source()}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{float *out;const float *query;const __fp16 *packed;const __fp16 *cache;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *packed,const __fp16 *cache,int total,int rows,int begin,int end){{
    const half4 *keys=(const half4*)(packed+32);
    const {'float4' if value_float else 'half4'} *value=(const {'float4' if value_float else 'half4'}*)(packed+32+{3*context*64});
    int tiles=(rows+1)/2;
    for(int job=begin;job<end;job++){{
      int kv=job/tiles,row=(job%tiles)*2,count=rows-row<2?rows-row:2;
      int used[2]={{total-rows+row+1,total-rows+row+count}},tails[2];
      tails[0]=used[0]/4*4;tails[1]=used[1]/4*4;int first_tail=tails[0]<tails[1]?tails[0]:tails[1];
      float q[64][8] __attribute__((aligned(16)));
      for(int d=0;d<64;d++)for(int i=0;i<8;i++)q[d][i]=i<6&&i%2<count?
        query[((kv*3+i/2)*{capacity}+row+i%2)*64+d]:0;
      float p[{8*(context+16)}] __attribute__((aligned(16)));
      for(int t=0;t<used[1];t+=4){{
        float4 {acc};for(int c=0;c<16;c++){{{dot}}}{scores}
        for(int lane=0;lane<4;lane++)if(t+lane<used[1]){{p[(t+lane)*8+6]=-__builtin_inff();p[(t+lane)*8+7]=-__builtin_inff();}}
      }}
      float4 m0=(float4)-__builtin_inff(),m1={{-__builtin_inff(),-__builtin_inff(),0,0}};
      for(int t=0;t<used[1];t++){{float4 v0=((float4*)(p+t*8))[0],v1=((float4*)(p+t*8))[1];m0=v0>m0?v0:m0;m1=v1>m1?v1:m1;}}
      float4 s0={{0}},s1={{0}};
      for(int t=0;t<used[1];t++){{
        float4 y0=rockchip_exp2_vector((((float4*)(p+t*8))[0]-m0)*(float4)1.4426950216293335f);
        float4 y1=rockchip_exp2_vector((((float4*)(p+t*8))[1]-m1)*(float4)1.4426950216293335f);
        if(t>=first_tail){{{overrides}}}
        ((float4*)(p+t*8))[0]=y0;((float4*)(p+t*8))[1]=y1;s0+=y0;s1+=y1;
      }}
      s1[2]=s1[3]=1;
      {'s0=1.0f/s0;s1=1.0f/s1;' if reciprocal else ''}
      for(int t=0;t<used[1];t++){{((float4*)(p+t*8))[0]{'*=' if reciprocal else '/='}s0;
        ((float4*)(p+t*8))[1]{'*=' if reciprocal else '/='}s1;}}
      for(int c=0;c<16;c+={tile}){{
        float4 {pv_acc};
        for(int t=0;t<used[0];t++){{{values}float4 pr0=((float4*)(p+t*8))[0],pr1=((float4*)(p+t*8))[1];{common}}}
        for(int t=used[0];t<used[1];t++){{{values}float4 pr0=((float4*)(p+t*8))[0],pr1=((float4*)(p+t*8))[1];{tail}}}
        {stores}
      }}
    }}
  }}
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->packed,w->cache,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  '''

def _half_part_source(context:int, capacity:int, padded:int) -> str:
  tile,half_pv = 2,False
  acc = ','.join(f'a{i}={{0}}' for i in range(8))
  dots = '\n'.join(f'a{i}=half_mla(a{i},qh[{i}][c],key);' for i in range(8))
  scores = '\n'.join(f'''{{float8 f=__builtin_convertvector(a{i},float8);
    float4 s=__builtin_shufflevector(f,f,0,1,2,3)+__builtin_shufflevector(f,f,4,5,6,7);
    float score=(((s[0]+s[1])+s[2])+s[3])*0.125f;
    if(!__builtin_isfinite(score)){{
      const float4 *fq=(const float4*)(query+(head*{capacity}+row+{i})*64);
      const half4 *fk=(const half4*)(cache+(kv*{context}+t)*64);float4 a={{0}};
      for(int c=0;c<16;c++)a+=fq[c]*__builtin_convertvector(fk[c],float4);
      score=(((a[0]+a[1])+a[2])+a[3])*0.125f;
    }}
    p[t*8+{i}]=t<used[{i}]?score:-__builtin_inff();}}''' for i in range(8))
  overrides = '\n'.join(f'if(t>=tails[{i}]&&t<used[{i}])y{i//4}[{i%4}]=expf(p[t*8+{i}]-m{i//4}[{i%4}]);'
                        for i in range(8))
  pv = ','.join(f'a{i}{j}={{0}}' for i in range(8) for j in range(tile))
  values = '\n'.join(f'float4 v{j}=value[(kv*16+c+{j})*{padded+16}+t];' for j in range(tile))
  common = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)pr{i//4}[{i%4}];' for i in range(8) for j in range(tile))
  tail = '\n'.join(f'if(t<used[{i}]){{'+''.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)pr{i//4}[{i%4}];'
                                             for j in range(tile))+'}' for i in range(8))
  stores = '\n'.join(f'if({i}<count)((float4*)(out+(head*{capacity}+row+{i})*64))[c+{j}]=a{i}{j};'
                     for i in range(8) for j in range(tile))
  pv_code = f'''for(int c=0;c<16;c+={tile}){{float4 {pv};
    for(int t=0;t<used[0];t++){{{values}float4 pr0=((float4*)(p+t*8))[0],pr1=((float4*)(p+t*8))[1];{common}}}
    for(int t=used[0];t<used[7];t++){{{values}float4 pr0=((float4*)(p+t*8))[0],pr1=((float4*)(p+t*8))[1];{tail}}}
    {stores}
  }}'''
  return f'''{_fast_softmax_source()}
  typedef __fp16 half8 __attribute__((aligned(16),ext_vector_type(8)));
  typedef float float8 __attribute__((aligned(32),ext_vector_type(8)));
  static inline half8 half_mla(half8 a,half8 b,half8 c){{__asm__("fmla %0.8h, %1.8h, %2.8h":"+w"(a):"w"(b),"w"(c));return a;}}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{float *out;const float *query;const __fp16 *packed;const __fp16 *cache;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *packed,const __fp16 *cache,int total,int rows,int begin,int end){{
    const {'half8' if half_pv else 'float4'} *value=(const {'half8' if half_pv else 'float4'}*)(packed+32+{3*padded*64});int tiles=(rows+7)/8;
    for(int job=begin;job<end;job++){{
      int head=job/tiles,kv=head/3,row=job%tiles*8,count=rows-row<8?rows-row:8;
      const __fp16 *keys=cache+kv*{context*64};half8 qh[8][8];
      for(int i=0;i<8;i++)for(int c=0;c<8;c++){{float8 f={{0}};
        if(i<count)__builtin_memcpy(&f,query+(head*{capacity}+row+i)*64+c*8,32);
        qh[i][c]=__builtin_convertvector(f,half8);}}
      int used[8],tails[8];for(int i=0;i<8;i++){{used[i]=total-rows+row+(i<count?i:count-1)+1;tails[i]=used[i]/4*4;}}
      int first_tail=tails[0];float p[{8*(context+16)}] __attribute__((aligned(16)));
      for(int t=0;t<used[7];t++){{half8 {acc};
        for(int c=0;c<8;c++){{half8 key=((const half8*)(keys+t*64))[c];{dots}}}{scores}
      }}
      float4 m0=(float4)-__builtin_inff(),m1=m0;
      for(int t=0;t<used[7];t++){{float4 v0=((float4*)(p+t*8))[0],v1=((float4*)(p+t*8))[1];m0=v0>m0?v0:m0;m1=v1>m1?v1:m1;}}
      float4 s0={{0}},s1={{0}};
      for(int t=0;t<used[7];t++){{
        float4 y0=rockchip_exp2_vector((((float4*)(p+t*8))[0]-m0)*(float4)1.4426950216293335f);
        float4 y1=rockchip_exp2_vector((((float4*)(p+t*8))[1]-m1)*(float4)1.4426950216293335f);
        if(t>=first_tail){{{overrides}}}
        ((float4*)(p+t*8))[0]=y0;((float4*)(p+t*8))[1]=y1;s0+=y0;s1+=y1;
      }}
      {'' if half_pv else 'for(int t=0;t<used[7];t++){((float4*)(p+t*8))[0]/=s0;((float4*)(p+t*8))[1]/=s1;}'}
      {pv_code}
    }}
  }}
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->packed,w->cache,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  '''

@functools.cache
def _packed(out:UOp, query:UOp, cache:UOp, packed:UOp, length:UOp, rows:UOp, context:int, capacity:int, fp16_qk:bool=False) -> UOp:
  dim,kv_heads = 64,3
  padded = round_up(context,4)
  value_stride = padded+16
  keys_size = kv_heads*padded*dim
  source = f'''{_half_part_source(context,capacity,padded) if fp16_qk else _part_source(padded,capacity)}
  static void small(float *out,const float *query,const __fp16 *cache,int total,int rows){{
    for(int head=0;head<9;head++)for(int row=0;row<rows;row++){{
      const __fp16 *keys=cache+(head/3)*{context*dim},*values=keys+{kv_heads*context*dim};
      const float4 *q=(const float4*)(query+(head*{capacity}+row)*64);
      int used=total-rows+row+1;float p[{context}],maximum=-__builtin_inff();
      for(int t=0;t<used;t++){{const half4 *key=(const half4*)(keys+t*64);float4 acc={{0}};
        for(int c=0;c<16;c++)acc+=q[c]*__builtin_convertvector(key[c],float4);
        float score=(((acc[0]+acc[1])+acc[2])+acc[3])*0.125f;
        p[t]=score;if(score>maximum)maximum=score;
      }}
      rockchip_softmax_vector(p,used,maximum);
      float4 acc[16]={{{{0}}}};
      for(int t=0;t<used;t++){{const half4 *value=(const half4*)(values+t*64);
        for(int c=0;c<16;c++)acc[c]+=__builtin_convertvector(value[c],float4)*(float4)p[t];}}
      for(int c=0;c<16;c++)((float4*)(out+(head*{capacity}+row)*64))[c]=acc[c];
    }}
  }}
  void rockchip_packed_attention(float *out,const float *query,const __fp16 *cache,__fp16 *packed,
                                const int *length,const int *count){{
    int total=*length,rows=*count;
    if(rows<1||rows>{capacity}||total<rows||total>{context}){{for(int i=0;i<{9*capacity*dim};i++)out[i]=__builtin_nanf("");return;}}
    int seen=*((int*)packed);if(seen<0||seen>{context})seen=0;
    int first=total-rows;if(seen<first)first=seen;
    if(rows<16){{
      int start=total-rows;if(seen>start)*((int*)packed)=start;
      ((int*)packed)[1]=0;
      small(out,query,cache,total,rows);return;
    }}
    __fp16 *key=packed+32;float *value=(float*)(packed+32+{keys_size});
    for(int kv=0;kv<3;kv++)for(int t=first;t<total;t++)for(int d=0;d<64;d++){{
      key[((kv*{padded//4}+t/4)*64+d)*4+t%4]=cache[(kv*{context}+t)*64+d];
      value[((kv*16+d/4)*{value_stride}+t)*4+d%4]=cache[{kv_heads*context*dim}+(kv*{context}+t)*64+d];
    }}
    *((int*)packed)=total;((int*)packed)[1]=total-first;
    int jobs={'9*((rows+7)/8)' if fp16_qk else '3*((rows+1)/2)'};
    if(rows<16){{part(out,query,packed,cache,total,rows,0,jobs);return;}}
    Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,query,packed,cache,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,packed,cache,total,rows,work[i].begin,work[i].end);}}
    part(out,query,packed,cache,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''
  inputs = (out,query,cache,packed,length,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_packed_attention'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,3),ins=(1,2,3,4,5))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


@functools.cache
def _invalidate(cache:UOp, packed:UOp, length:UOp, rows:UOp) -> UOp:
  source = '''void rockchip_packed_invalidate(const __fp16 *cache,__fp16 *packed,const int *length,const int *rows) {
    (void)cache;int first=*length-*rows;if(*((int*)packed)>first)*((int*)packed)=first;((int*)packed)[1]=0;
  }'''
  inputs = (cache,packed,length,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_packed_invalidate'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(1,),ins=(0,1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


class PackedCPUAttention:
  # Cache state is not a model parameter or a candidate for NPU weight packing.
  __slots__ = ('context','packed','fp16_qk')

  def __init__(self, context:int, fp16_qk:bool=False):
    if not 1<=context<=8192: raise ValueError('Unsupported packed CPU context')
    if fp16_qk and platform.machine().lower() not in ('aarch64','arm64'):
      raise ValueError('FP16 QK requires AArch64 FP16 SIMD')
    self.context,self.fp16_qk = context,fp16_qk
    padded = round_up(context,4)
    self.packed = Tensor.zeros(32+3*64*(padded+2*(padded+16)),device='CPU',dtype=dtypes.half).realize()

  def __call__(self, query:Tensor, cache:Tensor, length:int|UOp) -> Tensor:
    if (query.device!='CPU' or cache.device!='CPU' or query.dtype is not dtypes.float or cache.dtype is not dtypes.half
        or query.ndim!=4 or query.shape[:2]!=(1,9) or query.shape[3]!=64 or cache.shape!=(2,1,3,self.context,64)):
      raise ValueError('Unsupported packed CPU attention layout')
    capacity = query.max_shape[2]
    if not isinstance(capacity,int) or not 1<=capacity<=256: raise ValueError('Packed CPU attention needs bounded queries')
    if isinstance(length,int) and not 1<=length<=self.context: raise ValueError('Invalid packed CPU attention length')
    end = Tensor.full((1,),length,device='CPU',dtype=dtypes.int)
    count = query.shape[2]
    rows = Tensor.full((1,),count,device='CPU',dtype=dtypes.int)
    out = Tensor.empty(*query.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
      query.pad_to(query.max_shape).contiguous(),cache,self.packed,end,rows,
      fxn=lambda o,q,k,p,e,r:_packed(o,q,k,p,e,r,self.context,capacity,self.fp16_qk))[0]
    return out.shrink(tuple((0,size) for size in query.shape))

  def decode(self, query:Tensor, cache:Tensor, length:int|UOp, pool=None, fast_softmax:bool=True) -> Tensor:
    # A cached retry can recompute an earlier token. Mark that packed suffix
    # dirty before the regular decoder writes/reuses the ordinary KV cache.
    count = query.shape[2]
    end = Tensor.full((1,),length,device='CPU',dtype=dtypes.int)
    rows = Tensor.full((1,),count,device='CPU',dtype=dtypes.int)
    # AFTER effects need storage. Defer a COPY input without forcing execution
    # inside @function, and keep the model's existing KV buffer in place.
    if cache.uop.storage_base.op not in (Ops.BUFFER,Ops.ALLOC,Ops.PARAM,Ops.STAGE,Ops.INDEX):
      cache = Tensor(cache.uop.bufferize())
    cache = cache.custom_kernel(self.packed,end,rows,fxn=_invalidate)[0]
    return decode_attention(query,cache,length,pool=pool,fast_softmax=fast_softmax)
