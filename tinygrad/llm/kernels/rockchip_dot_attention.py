"""Opt-in per-row INT8 score dots with FP32 softmax/PV and ordinary FP16 KV.

Use the same score math for every query count, including cached prefill and
decode. Reuse the packed FP32 value allocation; quantized keys and their scales
fit within its existing key reservation. This route requires AArch64 SDOT.
"""
import functools, platform
from dataclasses import replace
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _fast_softmax_source
from tinygrad.llm.kernels.rockchip_int8 import _quantize_vector_source
from tinygrad.llm.kernels.rockchip_packed_attention import PackedCPUAttention
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def supports_dotprod() -> bool:
  try: return platform.machine().lower() in ('aarch64','arm64') and 'asimddp' in Path('/proc/cpuinfo').read_text()
  except OSError: return False


def _part_source(context:int, capacity:int, block:int) -> str:
  padded = round_up(context,4)
  queries,width = 3*block,round_up(3*block,4)
  integer_acc = ','.join(f'a{i}={{0}}' for i in range(queries))
  dots = ''.join(f'a{i}=dot(a{i},q[{i}][c],key);' for i in range(queries))
  scores = ''.join(f'''p[t*{width}+{i}]=t<used[{i}]?
    ((float)(((a{i}[0]+a{i}[1])+a{i}[2])+a{i}[3])*qs[{i}])*scale*0.125f:-__builtin_inff();''' for i in range(queries))
  maxima = ','.join(f'm{i}=(float4)-__builtin_inff()' for i in range(width//4))
  maximum = ''.join(f'{{float4 x=((float4*)(p+t*{width}))[{i}];m{i}=x>m{i}?x:m{i};}}' for i in range(width//4))
  sums = ','.join(f's{i}={{0}}' for i in range(width//4))
  exp = ''.join(f'float4 y{i}=rockchip_exp2_vector((((float4*)(p+t*{width}))[{i}]-m{i})*(float4)1.4426950216293335f);'
                for i in range(width//4))
  tails = ''.join(f'if(t>=tail[{i}]&&t<used[{i}])y{i//4}[{i%4}]=expf(p[t*{width}+{i}]-m{i//4}[{i%4}]);'
                  for i in range(queries))
  exp_store = ''.join(f'((float4*)(p+t*{width}))[{i}]=y{i};s{i}+=y{i};' for i in range(width//4))
  normalize = ''.join(f'((float4*)(p+t*{width}))[{i}]/=s{i};' for i in range(width//4))
  float_acc = ','.join(f'a{i}={{0}}' for i in range(queries))
  loads = ''.join(f'float4 pr{i}=((float4*)(p+t*{width}))[{i}];' for i in range(width//4))
  pv = ''.join(f'a{i}=a{i}+value*(float4)pr{i//4}[{i%4}];' for i in range(queries))
  pv_tail = ''.join(f'if(t<used[{i}])a{i}=a{i}+value*(float4)pr{i//4}[{i%4}];' for i in range(queries))
  stores = ''.join(f'''if({i%block}<count)((float4*)(out+((kv*3+{i//block})*{capacity}+row+{i%block})*64))[c]=
    bad[{i}]?(float4)__builtin_nanf(""):a{i};''' for i in range(queries))
  return f'''static void part{block}(float *out,const float *query,const unsigned char *packed,int total,int rows,int begin,int end){{
    const signed char *keys=(const signed char*)(packed+64);
    const float *scales=(const float*)(packed+64+{3*padded*64});
    const float4 *values=(const float4*)(packed+64+{3*padded*64*2});int tiles=(rows+{block-1})/{block};
    for(int job=begin;job<end;job++){{
      int kv=job/tiles,row=(job%tiles)*{block},count=rows-row<{block}?rows-row:{block};
      char16 q[{queries}][4];float qs[{queries}];int bad[{queries}],used[{queries}],tail[{queries}];
      for(int i=0;i<{queries};i++){{int r=i%{block};used[i]=total-rows+row+(r<count?r:count-1)+1;tail[i]=used[i]/4*4;
        if(r<count)bad[i]=quantize((signed char*)q[i],qs+i,(const float4*)(query+((kv*3+i/{block})*{capacity}+row+r)*64));
        else{{bad[i]=0;qs[i]=1;for(int c=0;c<4;c++)q[i][c]=(char16)0;}}
      }}
      float p[{width*(context+16)}] __attribute__((aligned(16)));
      for(int t=0;t<used[{queries-1}];t++){{int4 {integer_acc};float scale=scales[kv*{padded}+t];
        for(int c=0;c<4;c++){{char16 key=((const char16*)(keys+(kv*{padded}+t)*64))[c];{dots}}}
        {scores}for(int i={queries};i<{width};i++)p[t*{width}+i]=0;
      }}
      float4 {maxima};for(int t=0;t<used[{queries-1}];t++){{{maximum}}}
      float4 {sums};for(int t=0;t<used[{queries-1}];t++){{{exp}if(t>=tail[0]){{{tails}}}{exp_store}}}
      for(int t=0;t<used[{queries-1}];t++){{{normalize}}}
      for(int c=0;c<16;c++){{float4 {float_acc};
        for(int t=0;t<used[0];t++){{float4 value=values[(kv*16+c)*{padded+16}+t];{loads}{pv}}}
        for(int t=used[0];t<used[{queries-1}];t++){{float4 value=values[(kv*16+c)*{padded+16}+t];{loads}{pv_tail}}}
        {stores}
      }}
    }}
  }}'''


@functools.cache
def _packed(out:UOp, query:UOp, cache:UOp, packed:UOp, length:UOp, rows:UOp, context:int, capacity:int, pool_address:int=0) -> UOp:
  padded = round_up(context,4)
  source = f'''{_fast_softmax_source()}
  {_quantize_vector_source()}
  typedef signed char char16 __attribute__((aligned(16),ext_vector_type(16)));
  static inline int4 dot(int4 a,char16 b,char16 c){{
    __asm__("sdot %0.4s, %1.16b, %2.16b":"+w"(a):"w"(b),"w"(c));return a;}}
  static inline int quantize(signed char *out,float *scale_out,const float4 *input){{
    float4 maximum={{0}};int bad=0;
    for(int c=0;c<16;c++){{float4 x=input[c];for(int i=0;i<4;i++)bad|=!__builtin_isfinite(x[i]);
      x=x>=(float4)0?x:-x;maximum=x>maximum?x:maximum;}}
    float largest=maximum[0];for(int i=1;i<4;i++)if(maximum[i]>largest)largest=maximum[i];
    float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;*scale_out=bad?__builtin_nanf(""):scale;
    for(int c=0;c<16;c++)((char4*)out)[c]=rockchip_quantize_vector(bad?(float4)0:input[c],1.0f/scale);
    return bad;
  }}
  {_part_source(context,capacity,1)}
  {_part_source(context,capacity,4)}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef void (*part_fn)(float*,const float*,const unsigned char*,int,int,int,int);
  typedef struct{{part_fn part;float *out;const float *query;const unsigned char *packed;int total,rows,begin,end;}} Work;
  static void *worker(void *p){{Work *w=p;w->part(w->out,w->query,w->packed,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  static void pooled(void **args,int position,int group){{
    int total=*((int*)args[3]),rows=position;
    if(group<3)part1(args[0],args[1],args[2],total,rows,group*rows,(group+1)*rows);
  }}
  void rockchip_dot_attention(float *out,const float *query,const __fp16 *cache,unsigned char *packed,const int *length,const int *count){{
    int total=*length,rows=*count;
    if(rows<1||rows>{capacity}||total<rows||total>{context}){{for(int i=0;i<{9*capacity*64};i++)out[i]=__builtin_nanf("");return;}}
    int seen=*((int*)packed);if(seen<0||seen>{context})seen=0;
    int first=total-rows;if(seen<first)first=seen;
    signed char *keys=(signed char*)(packed+64);float *scales=(float*)(packed+64+{3*padded*64});
    float4 *values=(float4*)(packed+64+{3*padded*64*2});
    for(int kv=0;kv<3;kv++)for(int t=first;t<total;t++){{float4 key[16];
      const half4 *raw_key=(const half4*)(cache+(kv*{context}+t)*64);
      const half4 *raw_value=raw_key+{3*context*16};
      for(int c=0;c<16;c++){{key[c]=__builtin_convertvector(raw_key[c],float4);
        values[(kv*16+c)*{padded+16}+t]=__builtin_convertvector(raw_value[c],float4);}}
      quantize(keys+(kv*{padded}+t)*64,scales+kv*{padded}+t,key);
    }}
    *((int*)packed)=total;((int*)packed)[1]=total-first;
    if({pool_address}ULL&&rows<4){{void *args[4]={{out,(void*)query,packed,(void*)length}};
      if(((int(*)(void(*)(void**,int,int),void**,int)){pool_address}ULL)(pooled,args,rows)==0)return;}}
    int block=rows<4?1:4,jobs=3*((rows+block-1)/block);part_fn part=rows<4?part1:part4;
    Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{part,out,query,packed,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,packed,total,rows,work[i].begin,work[i].end);}}
    part(out,query,packed,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''
  inputs = (out,query,cache,packed,length,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_dot_attention'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,3),ins=(1,2,3,4,5))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


class Int8CPUAttention(PackedCPUAttention):
  __slots__ = ()

  def __init__(self, context:int):
    if not supports_dotprod(): raise ValueError('INT8 QK requires Linux AArch64 dot-product support')
    super().__init__(context)

  def _run(self, query:Tensor, cache:Tensor, length:int|UOp, pool_address:int=0) -> Tensor:
    if (query.device!='CPU' or cache.device!='CPU' or query.dtype is not dtypes.float or cache.dtype is not dtypes.half
        or query.ndim!=4 or query.shape[:2]!=(1,9) or query.shape[3]!=64 or cache.shape!=(2,1,3,self.context,64)):
      raise ValueError('Unsupported INT8 CPU attention layout')
    capacity = query.max_shape[2]
    if not isinstance(capacity,int) or not 1<=capacity<=256: raise ValueError('INT8 CPU attention needs bounded queries')
    if isinstance(length,int) and not 1<=length<=self.context: raise ValueError('Invalid INT8 CPU attention length')
    end = Tensor.full((1,),length,device='CPU',dtype=dtypes.int)
    rows = Tensor.full((1,),query.shape[2],device='CPU',dtype=dtypes.int)
    out = Tensor.empty(*query.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
      query.pad_to(query.max_shape).contiguous(),cache,self.packed,end,rows,
      fxn=lambda o,q,k,p,e,r:_packed(o,q,k,p,e,r,self.context,capacity,pool_address))[0]
    return out.shrink(tuple((0,size) for size in query.shape))

  def __call__(self, query:Tensor, cache:Tensor, length:int|UOp) -> Tensor: return self._run(query,cache,length)

  def decode(self, query:Tensor, cache:Tensor, length:int|UOp, pool=None, fast_softmax:bool=True) -> Tensor:
    return self._run(query,cache,length,pool.address if pool is not None and pool.active else 0)
