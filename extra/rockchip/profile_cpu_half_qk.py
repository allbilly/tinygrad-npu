"""Evaluate FP16 QK with FP32 softmax/PV on AArch64; serving is unchanged."""
import argparse, functools, json, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _attention, _renderer, _fast_softmax_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def half_source(context=8192, capacity=128, tile=2, half_pv=False, split=16):
  acc = ','.join(f'a{i}={{0}}' for i in range(8))
  dots = '\n'.join(f'a{i}=half_mla(a{i},qh[{i}][c],key);' for i in range(8))
  scores = '\n'.join(f'''{{float8 f=__builtin_convertvector(a{i},float8);
    float4 s=__builtin_shufflevector(f,f,0,1,2,3)+__builtin_shufflevector(f,f,4,5,6,7);
    float score=(((s[0]+s[1])+s[2])+s[3])*0.125f;
    p[t*8+{i}]=t<used[{i}]?score:-__builtin_inff();}}''' for i in range(8))
  overrides = '\n'.join(f'if(t>=tails[{i}]&&t<used[{i}])y{i//4}[{i%4}]=expf(p[t*8+{i}]-m{i//4}[{i%4}]);'
                        for i in range(8))
  pv = ','.join(f'a{i}{j}={{0}}' for i in range(8) for j in range(tile))
  values = '\n'.join(f'float4 v{j}=value[(kv*16+c+{j})*{context+16}+t];' for j in range(tile))
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
  if half_pv:
    global_acc=','.join(f'a{i}={{0}}' for i in range(8))
    local_acc=','.join(f'h{i}={{0}}' for i in range(8))
    common_half='\n'.join(f'h{i}=half_mla(h{i},v,(half8)probability[{i}]);' for i in range(8))
    tail_half='\n'.join(f'if(t<used[{i}])h{i}=half_mla(h{i},v,(half8)probability[{i}]);' for i in range(8))
    half_probability='float8 f;__builtin_memcpy(&f,p+t*8,32);half8 probability=__builtin_convertvector(f,half8);'
    reduce='\n'.join(f'a{i}+=__builtin_convertvector(h{i},float8);' for i in range(8))
    store='\n'.join(f'''if({i}<count){{float8 result=a{i}/(float8)denominator[{i}];
      __builtin_memcpy(out+(head*{capacity}+row+{i})*64+c*8,&result,32);}}''' for i in range(8))
    pv_code=f'''float denominator[8];__builtin_memcpy(denominator,&s0,16);__builtin_memcpy(denominator+4,&s1,16);
      for(int c=0;c<8;c++){{float8 {global_acc};
        for(int begin=0;begin<used[7];begin+={split}){{half8 {local_acc};
          int stop=begin+{split}<used[7]?begin+{split}:used[7],common=stop<used[0]?stop:used[0],t=begin;
          for(;t<common;t++){{half8 v=value[(kv*8+c)*{context+16}+t];{half_probability}{common_half}}}
          for(;t<stop;t++){{half8 v=value[(kv*8+c)*{context+16}+t];{half_probability}{tail_half}}}
          {reduce}
        }}
        {store}
      }}'''
  return f'''{_fast_softmax_source()}
  typedef __fp16 half8 __attribute__((aligned(16),ext_vector_type(8)));
  typedef float float8 __attribute__((aligned(32),ext_vector_type(8)));
  static inline half8 half_mla(half8 a,half8 b,half8 c){{__asm__("fmla %0.8h, %1.8h, %2.8h":"+w"(a):"w"(b),"w"(c));return a;}}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{float *out;const float *query;const __fp16 *cache;const void *value;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *cache,const void *raw,int total,int rows,int begin,int end){{
    const {'half8' if half_pv else 'float4'} *value=(const {'half8' if half_pv else 'float4'}*)raw;int tiles=(rows+7)/8;
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
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->cache,w->value,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  void half_qk_probe(float *out,const float *query,const __fp16 *cache,const void *value,const int *length,const int *count){{
    int total=*length,rows=*count,jobs=9*((rows+7)/8);Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,query,cache,value,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,cache,value,total,rows,work[i].begin,work[i].end);}}
    part(out,query,cache,value,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''


@functools.cache
def program(out,q,cache,value,end,rows,tile,half_pv,split):
  inputs = (out,q,cache,value,end,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('half_qk_probe'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4,5))
  code = half_source(tile=tile,half_pv=half_pv,split=split)
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=code),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(code))))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(104)
  query = rng.normal(0,.3,(1,9,128,64)).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
  cache[:,:,:,8191:] = np.nan
  value = np.pad(cache[1,0].astype(np.float32).reshape(3,8192,16,4).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  half_value = np.pad(cache[1,0].reshape(3,8192,8,8).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  q,kv,v = (Tensor(a,device='CPU').realize() for a in (query,cache,value))
  end = Tensor([8191],device='CPU',dtype=dtypes.int).realize()
  def original(q,k,v,e,r):
    return Tensor.empty(*query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,k,e,r,
      fxn=lambda o,q,k,e,r:_attention(o,q,k,e,9,3,64,8192,rows=r,max_rows=128,fast_softmax=True))[0].realize()
  hv=Tensor(half_value,device='CPU').realize()
  variants = {'original':(TinyJit(original),v)}
  for tile,hpv,split in ((2,False,16),(2,True,4),(2,True,8)):
    def execute(q,k,v,e,r,tile=tile,hpv=hpv,split=split):
      return Tensor.empty(*query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,k,v,e,r,
        fxn=lambda o,q,k,v,e,r:program(o,q,k,v,e,r,tile,hpv,split))[0].realize()
    variants[f'half_qk{tile}-pv{hpv}-split{split}'] = TinyJit(execute),hv if hpv else v
  checks = []
  for count in (128,65,7,127):
    rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
    expected = variants['original'][0](q,kv,v,end,rows).numpy()[:,:,:count]
    for label,(run,data) in variants.items():
      actual = run(q,kv,data,end,rows).numpy()[:,:,:count]
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=.02)
      checks.append({'variant':label,'rows':count,'max_abs_error':float(abs(actual-expected).max())})
  rows = Tensor([128],device='CPU',dtype=dtypes.int).realize()
  samples = {label:[] for label in variants}
  for index in range(9):
    for label in variants if index%2==0 else reversed(variants):
      begin=time.perf_counter()
      run,data=variants[label]
      run(q,kv,data,end,rows)
      if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
  report={'scope':'synthetic FP16 QK, optional blockwise FP16 PV, FP32 softmax/output; packing excluded; serving unchanged','checks':checks,
          'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
