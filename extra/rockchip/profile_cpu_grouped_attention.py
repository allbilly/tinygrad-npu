"""Test feature-vector FP32 attention sharing KV across grouped heads and rows.

Diagnostic only: compare bitwise output and primitive timing with the accepted
packed kernel. Value packing is excluded from experimental primitive timings.
"""
import argparse, functools, json, os, re, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _fast_softmax_source
from tinygrad.llm.kernels.rockchip_packed_attention import PackedCPUAttention, _part_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def source(block:int, context:int=8192, capacity:int=128) -> str:
  queries = 3*block
  width = (queries+3)//4*4
  accum = ','.join(f'a{i}={{0}}' for i in range(queries))
  dots = ''.join(f'a{i}=a{i}+q[{i}][c]*key;' for i in range(queries))
  scores = ''.join(f'''p[t*{width}+{i}]=t<used[{i}]?
    (((a{i}[0]+a{i}[1])+a{i}[2])+a{i}[3])*0.125f:-__builtin_inff();''' for i in range(queries))
  loads = ''.join(f'float4 pr{i}=((float4*)(p+t*{width}))[{i}];' for i in range(width//4))
  pv = ''.join(f'a{i}=a{i}+value*(float4)pr{i//4}[{i%4}];' for i in range(queries))
  pv_tail = ''.join(f'if(t<used[{i}])a{i}=a{i}+value*(float4)pr{i//4}[{i%4}];' for i in range(queries))
  stores = ''.join(f'''if({i%block}<count)((float4*)(out+((kv*3+{i//block})*{capacity}+row+{i%block})*64))[c]=a{i};'''
                   for i in range(queries))
  maxima = ','.join(f'm{i}=(float4)-__builtin_inff()' for i in range(width//4))
  sums = ','.join(f's{i}={{0}}' for i in range(width//4))
  maximum = ''.join(f'{{float4 x=((float4*)(p+t*{width}))[{i}];m{i}=x>m{i}?x:m{i};}}' for i in range(width//4))
  exp = ''.join(f'float4 y{i}=rockchip_exp2_vector((((float4*)(p+t*{width}))[{i}]-m{i})*(float4)1.4426950216293335f);'
                for i in range(width//4))
  tails = ''.join(f'if(t>=tail[{i}]&&t<used[{i}])y{i//4}[{i%4}]=expf(p[t*{width}+{i}]-m{i//4}[{i%4}]);'
                  for i in range(queries))
  exp_store = ''.join(f'((float4*)(p+t*{width}))[{i}]=y{i};s{i}+=y{i};' for i in range(width//4))
  normalize = ''.join(f'((float4*)(p+t*{width}))[{i}]/=s{i};' for i in range(width//4))
  return f'''{_fast_softmax_source()}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{float *out;const float *query;const __fp16 *cache;const float *value;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *cache,const float *values,int total,int rows,int begin,int end){{
    int tiles=(rows+{block-1})/{block};
    for(int job=begin;job<end;job++){{
      int kv=job/tiles,row=(job%tiles)*{block},count=rows-row<{block}?rows-row:{block};
      float4 q[{queries}][16];int used[{queries}],tail[{queries}];
      for(int i=0;i<{queries};i++){{int r=i%{block};used[i]=total-rows+row+(r<count?r:count-1)+1;tail[i]=used[i]/4*4;
        for(int c=0;c<16;c++)q[i][c]=r<count?((const float4*)(query+((kv*3+i/{block})*{capacity}+row+r)*64))[c]:(float4)0;}}
      float p[{width*(context+16)}] __attribute__((aligned(16)));
      for(int t=0;t<used[{queries-1}];t++){{float4 {accum};
        for(int c=0;c<16;c++){{float4 key=__builtin_convertvector(((const half4*)(cache+(kv*{context}+t)*64))[c],float4);{dots}}}
        {scores}for(int i={queries};i<{width};i++)p[t*{width}+i]=0;
      }}
      float4 {maxima};for(int t=0;t<used[{queries-1}];t++){{{maximum}}}
      float4 {sums};for(int t=0;t<used[{queries-1}];t++){{{exp}if(t>=tail[0]){{{tails}}}{exp_store}}}
      for(int t=0;t<used[{queries-1}];t++){{{normalize}}}
      for(int c=0;c<16;c++){{float4 {accum};
        for(int t=0;t<used[0];t++){{float4 value=((const float4*)values)[(kv*16+c)*{context+16}+t];{loads}{pv}}}
        for(int t=used[0];t<used[{queries-1}];t++){{float4 value=((const float4*)values)[(kv*16+c)*{context+16}+t];{loads}{pv_tail}}}
        {stores}
      }}
    }}
  }}
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->cache,w->value,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  void grouped_attention_probe(float *out,const float *query,const __fp16 *cache,const float *value,const int *end,const int *count){{
    int rows=*count,total=*end,jobs=3*((rows+{block-1})/{block});Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,query,cache,value,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,cache,value,total,rows,work[i].begin,work[i].end);}}
    part(out,query,cache,value,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''


def vector_scores(code:str) -> str:
  first = code.index('{float4 score=')
  last = code.index('\n      }\n      float4 m0',first)
  scores = ''.join(f'float4 s{i}=(((a{i}_0+a{i}_1)+a{i}_2)+a{i}_3)*0.125f;' for i in range(6))
  common = ''.join(f'''((float4*)(p+(t+{lane})*8))[0]=(float4){{s0[{lane}],s1[{lane}],s2[{lane}],s3[{lane}]}};
    ((float4*)(p+(t+{lane})*8))[1]=(float4){{s4[{lane}],s5[{lane}],-__builtin_inff(),-__builtin_inff()}};'''
                   for lane in range(4))
  tail = ''.join(f'p[(t+lane)*8+{i}]=t+lane<used[{i%2}]?s{i}[lane]:-__builtin_inff();' for i in range(6))
  return code[:first]+scores+f'''if(t+3<used[0]){{{common}}}else{{
    for(int lane=0;lane<4;lane++)if(t+lane<used[1]){{{tail}
      p[(t+lane)*8+6]=p[(t+lane)*8+7]=-__builtin_inff();}}
  }}'''+code[last:]


def blocked_pv(code:str, block:int, capacity:int=128) -> str:
  before = 'for(int c=0;c<16;c+=2){'
  assert code.count(before)==1
  code = code.replace(before,f'''for(int first=0;first<used[1];first+={block}){{
    int stop=first+{block}<used[1]?first+{block}:used[1];{before}''')
  def initialize(match):
    i,j = map(int,match.groups())
    return f'a{i}{j}=first&&{i%2}<count?((float4*)(out+((kv*3+{i//2})*{capacity}+row+{i%2})*64))[c+{j}]:(float4)0'
  code,count = re.subn(r'a([0-5])([0-1])=\{0\}',initialize,code)
  assert count==12
  code = code.replace('for(int t=0;t<used[0];t++)','for(int t=first;t<stop&&t<used[0];t++)')
  code = code.replace('for(int t=used[0];t<used[1];t++)','for(int t=used[0]>first?used[0]:first;t<stop;t++)')
  last = '\n      }\n    }\n  }\n  static void *worker'
  assert code.count(last)==1
  return code.replace(last,'\n      }}\n    }\n  }\n  static void *worker')


def weighted_source(ratio:int, context:int=8192, capacity:int=128, vector:bool=False, pv_block:int=0) -> str:
  prefix = _part_source(context,capacity)
  if vector: prefix = vector_scores(prefix)
  if pv_block: prefix = blocked_pv(prefix,pv_block,capacity)
  if ratio==0:
    return prefix+'''
    void grouped_attention_probe(float *out,const float *query,const __fp16 *cache,const float *value,const int *end,const int *count){
      int rows=*count,total=*end,jobs=3*((rows+1)/2);const __fp16 *packed=(const __fp16*)value;
      Work work[4];thread_t threads[3];int created[3];
      for(int i=0;i<4;i++)work[i]=(Work){out,query,packed,cache,total,rows,jobs*i/4,jobs*(i+1)/4};
      for(int i=0;i<3;i++){created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
        if(!created[i])part(out,query,packed,cache,total,rows,work[i].begin,work[i].end);}
      part(out,query,packed,cache,total,rows,work[3].begin,work[3].end);
      for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
    }'''
  return prefix+f'''
  extern int sched_getaffinity(int,unsigned long,void*);
  extern int sched_setaffinity(int,unsigned long,const void*);
  typedef struct{{Work work;int cpu;}} PinnedWork;
  static void *pinned_worker(void *p){{PinnedWork *w=p;unsigned long mask=1UL<<w->cpu;
    sched_setaffinity(0,sizeof(mask),&mask);return worker(&w->work);}}
  void grouped_attention_probe(float *out,const float *query,const __fp16 *cache,const float *value,const int *end,const int *count){{
    int rows=*count,total=*end,jobs=3*((rows+1)/2),position=0;
    const __fp16 *packed=(const __fp16*)value;
    unsigned long previous[16],mask=1UL<<4;int saved=sched_getaffinity(0,sizeof(previous),previous)==0;
    if(saved)sched_setaffinity(0,sizeof(mask),&mask);
    PinnedWork work[7];thread_t threads[7];int created[7];
    for(int i=0;i<7;i++){{int weight=i<4?1:{ratio},cpu=i<4?i:i+1;
      work[i]=(PinnedWork){{{{out,query,packed,cache,total,rows,jobs*position/{4+4*ratio},jobs*(position+weight)/{4+4*ratio}}},cpu}};
      position+=weight;
      created[i]=pthread_create(&threads[i],0,pinned_worker,&work[i])==0;
      if(!created[i])part(out,query,packed,cache,total,rows,work[i].work.begin,work[i].work.end);
    }}
    part(out,query,packed,cache,total,rows,jobs*position/{4+4*ratio},jobs);
    for(int i=0;i<7;i++)if(created[i])pthread_join(threads[i],0);
    if(saved)sched_setaffinity(0,sizeof(previous),previous);
  }}'''


@functools.cache
def program(out,q,cache,values,end,rows,block):
  inputs = (out,q,cache,values,end,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('grouped_attention_probe'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4,5))
  code = (weighted_source(0,pv_block=block-100000) if block>=100000 else
          source(block) if block>0 else weighted_source((-block)%100,vector=(-block)>=100))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=code),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(code))))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--weighted',action='store_true')
  parser.add_argument('--blocked',action='store_true')
  args = parser.parse_args()
  rng = np.random.default_rng(110)
  raw_query = rng.normal(0,.3,(1,9,128,64)).astype(np.float32)
  raw_cache = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
  raw_cache[:,:,:,8191:] = np.nan
  values = np.pad(raw_cache[1,0].astype(np.float32).reshape(3,8192,16,4).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  q,cache,v = (Tensor(a,device='CPU').realize() for a in (raw_query,raw_cache,values))
  packed = np.concatenate((np.zeros(32,np.float16),raw_cache[0,0].reshape(3,2048,4,64).transpose(0,1,3,2).ravel(),
                           values.ravel().view(np.float16)))
  if args.weighted or args.blocked: v = Tensor(packed,device='CPU').realize()
  end = Tensor([8191],device='CPU',dtype=dtypes.int).realize()
  state = PackedCPUAttention(8192)
  accepted = TinyJit(lambda q,k:state(q,k,8191).realize())
  variants = {}
  initial_affinity = os.sched_getaffinity(0)
  for block in ((100128,100256,100512,101024) if args.blocked else (-4,-6,-8,-100,-106,-108) if args.weighted else (2,4,6,8)):
    def execute(q,k,v,e,r,block=block):
      return Tensor.empty(*raw_query.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,k,v,e,r,
        fxn=lambda o,q,k,v,e,r:program(o,q,k,v,e,r,block))[0].realize()
    variants[f'grouped{block}'] = TinyJit(execute)
  checks = []
  for count in (128,65,7,127):
    expected = state(q[:,:,:count],cache,8191).numpy()
    rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
    for label,run in variants.items():
      actual = run(q,cache,v,end,rows).numpy()[:,:,:count]
      assert os.sched_getaffinity(0)==initial_affinity
      np.testing.assert_array_equal(actual,expected)
      checks.append({'variant':label,'rows':count,'bitwise_equal':True})
  rows = Tensor([128],device='CPU',dtype=dtypes.int).realize()
  samples = {name:[] for name in ['accepted',*variants]}
  for index in range(9):
    for label in samples if index%2==0 else reversed(samples):
      begin = time.perf_counter()
      accepted(q,cache) if label=='accepted' else variants[label](q,cache,v,end,rows)
      if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
  report = {'complete':True,'scope':'primitive only; experimental V packing excluded','checks':checks,
            'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__': main()
