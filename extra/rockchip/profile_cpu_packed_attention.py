"""Evaluate packed KV and shared GQA loads with FP32 attention arithmetic.

This experiment uses prepacked synthetic inputs. Serving code is unchanged.
"""
import argparse, functools, json, statistics, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _attention, _renderer, _fast_softmax_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def source(block, tile, context=8192, capacity=128, key_tile=4, paired=False, late_normalize=False,
           column_probs=False, column_queries=False):
  dim,heads,kv_heads,gqa = 64,9,3,3
  group = 8 if paired else block*gqa
  assert not column_probs or late_normalize and group%4==0
  groups = key_tile//4
  acc = ','.join(f'a{i}_{j}={{0}}' for i in range(group) for j in range(groups))
  dot = '\n'.join(f'a{i}_{j}=a{i}_{j}+(float4)q[{i}][d]*key{j};' for i in range(group) for j in range(groups))
  load_queries = ''
  if column_queries:
    assert group%4==0
    load_queries = '\n'.join(f'float4 query{i}=((float4*)q[d])[{i}];' for i in range(group//4))
    dot = '\n'.join(f'a{i}_{j}=a{i}_{j}+(float4)query{i//4}[{i%4}]*key{j};' for i in range(group) for j in range(groups))
  load_keys = '\n'.join(f'float4 key{j}=__builtin_convertvector(keys[(kv*{context//4}+t/4+{j})*64+d],float4);'
                        for j in range(groups))
  scores = '\n'.join(f'''for(int lane=0;lane<4;lane++)if(t+{j*4}+lane<used[{i}]){{
    float score=a{i}_{j}[lane]*0.125f;p[{i}][t+{j*4}+lane]=score;
    if(score>maximum[{i}])maximum[{i}]=score;}}''' for i in range(group) for j in range(groups))
  if column_probs:
    scores = '\n'.join(f'''for(int lane=0;lane<4;lane++)if(t+{j*4}+lane<used[{group-1}]){{
      float score=t+{j*4}+lane<used[{i}]?a{i}_{j}[lane]*0.125f:-__builtin_inff();
      p[(t+{j*4}+lane)*{group}+{i}]=score;if(score>maximum[{i}])maximum[{i}]=score;}}'''
                       for i in range(group) for j in range(groups))
  pv_acc = ','.join(f'a{i}{j}={{0}}' for i in range(group) for j in range(tile))
  values = '\n'.join(f'float4 v{j}=__builtin_convertvector(values[(kv*16+c+{j})*{context+16}+t],float4);' for j in range(tile))
  common = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)p[{i}][t];' for i in range(group) for j in range(tile))
  tail = '\n'.join(f'if(t<used[{i}]){{'+''.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)p[{i}][t];' for j in range(tile))+'}'
                   for i in range(group))
  stores = []
  for i in range(group):
    offset = f'((head_base+{i}/row_block)*{capacity}+row+{i}%row_block)*64'
    for j in range(tile):
      value = f'a{i}{j}'+(f'/(float4)denominator[{i}]' if late_normalize else '')
      stores.append(f'if({i}%row_block<count){{((float4*)(out+{offset}))[c+{j}]={value};}}')
  stores = '\n'.join(stores)
  normalize = (f'float denominator[{group}];for(int i=0;i<{group};i++)denominator[i]=packed_exp_sum(p[i],used[i],maximum[i]);'
               if late_normalize else f'for(int i=0;i<{group};i++)rockchip_softmax_vector(p[i],used[i],maximum[i]);')
  if column_probs:
    sums = ','.join(f's{i}={{0}}' for i in range(group//4))
    maximums = '\n'.join(f'float4 m{i}=((float4*)maximum)[{i}];' for i in range(group//4))
    exponentials = '\n'.join(f'''float4 y{i}=rockchip_exp2_vector((((float4*)(p+t*{group}))[{i}]-m{i})*(float4)1.4426950216293335f);
      ((float4*)(p+t*{group}))[{i}]=y{i};s{i}+=y{i};''' for i in range(group//4))
    normalize = f'''float denominator[{group}] __attribute__((aligned(16)));float4 {sums};{maximums}
      for(int t=0;t<used[{group-1}];t++){{{exponentials}}}
      '''+'\n'.join(f'((float4*)denominator)[{i}]=s{i};' for i in range(group//4))
    values += '\n'+'\n'.join(f'float4 probability{i}=((float4*)(p+t*{group}))[{i}];' for i in range(group//4))
    common = '\n'.join(f'a{i}{j}=a{i}{j}+v{j}*(float4)probability{i//4}[{i%4}];' for i in range(group) for j in range(tile))
    tail = '\n'.join(f'if(t<used[{i}]){{'+''.join(
      f'a{i}{j}=a{i}{j}+v{j}*(float4)probability{i//4}[{i%4}];' for j in range(tile))+'}' for i in range(group))
  declaration = f'float p[{group*(context+16)}];' if column_probs else f'float p[{group}][{context+16}];'
  query_declaration = f'float q[64][{group}]' if column_queries else f'float q[{group}][64]'
  query_copy = (f'''for(int i=0;i<{group};i++)for(int d=0;d<64;d++)q[d][i]=i%row_block<count?
    query[((head_base+i/row_block)*{capacity}+row+i%row_block)*64+d]:0;''' if column_queries else f'''
    for(int i=0;i<{group};i++){{
      if(i%row_block<count)__builtin_memcpy(q[i],query+((head_base+i/row_block)*{capacity}+row+i%row_block)*64,64*4);
      else for(int d=0;d<64;d++)q[i][d]=0;
    }}''')
  tile_count = '(rows+3)/4+(rows+7)/8' if paired else f'(rows+{block-1})/{block}'
  job = '''int kv=job/tiles,part=job%tiles,pairs=(rows+3)/4;
      int row_block=part<pairs?4:8,head_base=kv*3+(part<pairs?0:2),row=(part<pairs?part:part-pairs)*row_block;
      int count=rows-row<row_block?rows-row:row_block;''' if paired else f'''
      int kv=job/tiles,row=(job%tiles)*{block},row_block={block},head_base=kv*3,count=rows-row<{block}?rows-row:{block};'''
  return f'''typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  {_fast_softmax_source()}
  static float packed_exp_sum(float *values,int count,float maximum){{
    float sum=0;int t=0;
    for(;t+3<count;t+=4){{float4 x;__builtin_memcpy(&x,values+t,16);
      float4 y=rockchip_exp2_vector((x-(float4)maximum)*(float4)1.4426950216293335f);
      __builtin_memcpy(values+t,&y,16);sum+=y[0];sum+=y[1];sum+=y[2];sum+=y[3];}}
    for(;t<count;t++){{values[t]=expf(values[t]-maximum);sum+=values[t];}}
    return sum;
  }}
  typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{float *out;const float *query;const __fp16 *packed;int total,rows,begin,end;}} Work;
  static void part(float *out,const float *query,const __fp16 *packed,int total,int rows,int begin,int end){{
    const half4 *keys=(const half4*)packed,*values=(const half4*)(packed+{kv_heads*context*dim});
    int tiles={tile_count};
    for(int job=begin;job<end;job++){{
      {job}
      int used[{group}];for(int i=0;i<{group};i++)used[i]=total-rows+row+(i%row_block<count?i%row_block:count-1)+1;
      {query_declaration} __attribute__((aligned(16)));
      {query_copy}
      {declaration}float maximum[{group}] __attribute__((aligned(16)));
      for(int i=0;i<{group};i++)maximum[i]=-__builtin_inff();
      for(int t=0;t<used[{group-1}];t+={key_tile}){{
        float4 {acc};
        for(int d=0;d<64;d++){{
          {load_keys}
          {load_queries}
          {dot}
        }}
        {scores}
      }}
      {normalize}
      for(int c=0;c<16;c+={tile}){{
        float4 {pv_acc};
        for(int t=0;t<used[0];t++){{{values}{common}}}
        for(int t=used[0];t<used[{group-1}];t++){{{values}{tail}}}
        {stores}
      }}
    }}
  }}
  static void *worker(void *p){{Work *w=p;part(w->out,w->query,w->packed,w->total,w->rows,w->begin,w->end);return (void*)0;}}
  void packed_attention_probe(float *out,const float *query,const __fp16 *packed,const int *length,const int *count){{
    int total=*length,rows=*count;
    if(rows<1||rows>{capacity}||total<rows||total>{context}){{for(int i=0;i<{heads*capacity*dim};i++)out[i]=__builtin_nanf("");return;}}
    int jobs=3*({tile_count});
    if(rows<16){{part(out,query,packed,total,rows,0,jobs);return;}}
    Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,query,packed,total,rows,jobs*i/4,jobs*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],0,worker,&work[i])==0;
      if(!created[i])part(out,query,packed,total,rows,work[i].begin,work[i].end);}}
    part(out,query,packed,total,rows,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],0);
  }}'''


@functools.cache
def program(out,q,packed,end,rows,block,tile,key_tile,paired,late_normalize,column_probs,column_queries):
  code = source(block,tile,key_tile=key_tile,paired=paired,late_normalize=late_normalize,
                column_probs=column_probs,column_queries=column_queries)
  inputs = (out,q,packed,end,rows)
  sink = UOp.sink(*inputs,arg=KernelInfo('packed_attention_probe'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3,4))
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
  rng = np.random.default_rng(91)
  shape = (1,9,128,64)
  query = rng.normal(0,.3,shape).astype(np.float32)
  cache = rng.normal(0,.3,(2,1,3,8192,64)).astype(np.float16)
  cache[:,:,:,8191:] = np.nan
  keys = cache[0,0].reshape(3,2048,4,64).transpose(0,1,3,2).copy()
  values = np.pad(cache[1,0].reshape(3,8192,16,4).transpose(0,2,1,3),((0,0),(0,0),(0,16),(0,0)))
  packed = np.concatenate((keys.ravel(),values.ravel()))
  q,kv,pk = Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize(),Tensor(packed,device='CPU').realize()
  end = Tensor([8191],device='CPU',dtype=dtypes.int).realize()
  def original(q,kv,end,rows):
    return Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(q,kv,end,rows,
      fxn=lambda o,q,k,e,r:_attention(o,q,k,e,9,3,64,8192,rows=r,max_rows=128,fast_softmax=True))[0].realize()
  variants = {'original':(TinyJit(original),kv)}
  for block,tile,key_tile,paired,late,col,cq in ((4,2,4,False,True,True,False),(4,2,4,False,True,True,True),(4,2,8,True,True,True,True)):
    def execute(q,pk,end,rows,block=block,tile=tile,key_tile=key_tile,paired=paired,late=late,col=col,cq=cq):
      return Tensor.empty(*shape,device='CPU',dtype=dtypes.float).custom_kernel(q,pk,end,rows,
        fxn=lambda o,q,p,e,r:program(o,q,p,e,r,block,tile,key_tile,paired,late,col,cq))[0].realize()
    variants[f'packed{block}x{tile}k{key_tile}-pair{paired}-col{col}-query{cq}'] = TinyJit(execute),pk
  checks = []
  for count in (128,65,7,127):
    rows = Tensor([count],device='CPU',dtype=dtypes.int).realize()
    expected = np.empty((1,9,count,64),dtype=np.float32)
    for head in range(9):
      k,v = cache[:,0,head//3,:8191].astype(np.float32)
      scores = query[0,head,:count]@k.T/np.float32(8)
      scores[np.arange(8191)[None]>8191-count+np.arange(count)[:,None]] = -np.inf
      p = np.exp(scores-scores.max(-1,keepdims=True))
      p /= p.sum(-1,keepdims=True)
      expected[0,head] = p@v
    for label,(run,data) in variants.items():
      actual = run(q,data,end,rows).numpy()[:,:,:count]
      np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-4)
      checks.append({'variant':label,'rows':count,'max_abs_error':float(abs(actual-expected).max())})
  rows = Tensor([128],device='CPU',dtype=dtypes.int).realize()
  samples = {label:[] for label in variants}
  for index in range(9):
    for label in variants if index%2==0 else reversed(variants):
      run,data = variants[label]
      begin = time.perf_counter()
      run(q,data,end,rows)
      if index>=2:samples[label].append((time.perf_counter()-begin)*1000)
  report = {'scope':'synthetic prepacked inputs; packing excluded; no serving changes','checks':checks,
            'variants':{k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__':main()
