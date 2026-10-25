"""Packed CPU attention experiment preserving the qualified FP32 operation order."""
import argparse, functools, json, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _attention, _renderer, _fast_softmax_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def exact_source(tile=4, context=8192, capacity=128, value_float=False, reciprocal=False):
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
  typedef struct{{float *out;const float *query;const __fp16 *packed;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *packed,int total,int rows,int begin,int end){{
    const half4 *keys=(const half4*)packed;
    const {'float4' if value_float else 'half4'} *value=(const {'float4' if value_float else 'half4'}*)(packed+{3*context*64});
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
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->packed,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  void packed_exact_probe(float *out,const float *query,const __fp16 *packed,const int *length,const int *count){{
    int rows=*count,total=*length,jobs=3*((rows+1)/2);
    if(rows<16){{part(out,query,packed,total,rows,0,jobs);return;}}
    Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,query,packed,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,packed,total,rows,work[i].begin,work[i].end);}}
    part(out,query,packed,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''


def compact_probability_source(code,context=8192):
  # Two unused lanes can read the next row, but only the six real query lanes
  # are stored. memcpy expresses the unaligned vector loads explicitly.
  pads='for(int lane=0;lane<4;lane++)if(t+lane<used[1]){p[(t+lane)*8+6]=-__builtin_inff();p[(t+lane)*8+7]=-__builtin_inff();}'
  assert code.count(pads)==1
  code=code.replace(pads,'')
  code=code.replace('float4 m0=(float4)-__builtin_inff()',
                    'p[used[1]*6]=p[used[1]*6+1]=0;float4 m0=(float4)-__builtin_inff()')
  for lane,size in ((0,16),(1,8)):
    offset=lane*4
    target=f'((float4*)(p+t*8))[{lane}]'
    code=code.replace(target+f'=y{lane};',f'__builtin_memcpy(p+t*6+{offset},&y{lane},{size});')
    code=code.replace(target+f'/=s{lane};',f'float4 n{lane}=packed_load(p+t*6+{offset})/s{lane};'
                      f'__builtin_memcpy(p+t*6+{offset},&n{lane},{size});')
    code=code.replace(target,f'packed_load(p+t*6+{offset})')
  code=code.replace(f'float p[{8*(context+16)}]',f'float p[{6*(context+16)}]')
  code=code.replace('*8+','*6+')
  prefix='static void part('
  helper='static inline float4 packed_load(const float *p){float4 v;__builtin_memcpy(&v,p,16);return v;}\n'
  assert code.count(prefix)==1
  return code.replace(prefix,helper+prefix)


@functools.cache
def program(out,q,packed,end,rows,tile,value_float,reciprocal,compact):
  inputs = (out,q,packed,end,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('packed_exact_probe'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4))
  code = exact_source(tile,value_float=value_float,reciprocal=reciprocal)
  if compact:code=compact_probability_source(code)
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=code),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(code))))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--wait-report',type=Path)
  args = parser.parse_args()
  if args.wait_report:
    for _ in range(1440):
      try:r=json.loads(args.wait_report.read_text())
      except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(5)
        continue
      if r.get('error'):raise RuntimeError(r['error'])
      if r.get('complete'):break
      time.sleep(5)
    else:raise TimeoutError(args.wait_report)
  rng = np.random.default_rng(101)
  qdata = rng.normal(0,.3,(1,9,128,64)).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
  cache[:,:,:,8191:] = np.nan
  keys = cache[0,0].reshape(3,2048,4,64).transpose(0,1,3,2).copy()
  values = np.pad(cache[1,0].reshape(3,8192,16,4).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  q = Tensor(qdata,device='CPU').realize()
  kv = Tensor(cache,device='CPU').realize()
  packed = Tensor(np.concatenate((keys.ravel(),values.ravel())),device='CPU').realize()
  packed_float = Tensor(np.concatenate((keys.ravel(),values.astype(np.float32).ravel().view(np.float16))),device='CPU').realize()
  end = Tensor([8191],device='CPU',dtype=dtypes.int).realize()
  def original(q,kv,end,rows):
    return Tensor.empty(*qdata.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,kv,end,rows,
      fxn=lambda o,q,k,e,r:_attention(o,q,k,e,9,3,64,8192,rows=r,max_rows=128,fast_softmax=True))[0].realize()
  variants = {'original':(TinyJit(original),kv)}
  for tile,vf,recip,compact in ((2,True,False,False),(2,True,False,True),(4,True,False,True)):
    def execute(q,pk,end,rows,tile=tile,vf=vf,recip=recip,compact=compact):
      return Tensor.empty(*qdata.shape,device='CPU',dtype=dtypes.float).custom_kernel(q,pk,end,rows,
        fxn=lambda o,q,p,e,r:program(o,q,p,e,r,tile,vf,recip,compact))[0].realize()
    variants[f'exact{tile}-float{vf}-recip{recip}-compact{compact}'] = TinyJit(execute),packed_float if vf else packed
  checks = []
  for count in (128,65,7,127):
    rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
    expected = variants['original'][0](q,kv,end,rows).numpy()[:,:,:count]
    for label,(run,data) in variants.items():
      actual = run(q,data,end,rows).numpy()[:,:,:count]
      if 'recipTrue' in label: np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
      else: np.testing.assert_array_equal(actual,expected)
      checks.append({'variant':label,'rows':count,'bitwise_equal':bool(np.array_equal(actual,expected)),
                     'max_abs_error':float(abs(actual-expected).max())})
  rows = Tensor([128],device='CPU',dtype=dtypes.int).realize()
  samples = {label:[] for label in variants}
  for index in range(9):
    for label in variants if index%2==0 else reversed(variants):
      run,data = variants[label]
      begin = time.perf_counter()
      run(q,data,end,rows)
      if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
  report = {'scope':'FP32 order-preserving packed primitive; packing excluded','checks':checks,
            'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
