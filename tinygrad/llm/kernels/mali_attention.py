"""Causal cached OpenCL attention with bounded scratch and FP32 online softmax."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from typing import cast
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, ceildiv
from tinygrad.uop.ops import AddrSpace, KernelInfo, Ops, ProgramInfo, UOp, sint


@functools.cache
def _partial(out:UOp, query:UOp, cache:UOp, length:UOp, rows:UOp, heads:int, kv_heads:int, dim:int,
             context:int, capacity:int, chunk:int, query_rows:int, subgroups:bool) -> UOp:
  gqa = heads//kv_heads
  group,threads,parts,stride = gqa*query_rows,128 if query_rows>1 else 64,ceildiv(context,chunk),dim+4
  part_axis,kv_axis,row_axis = 0,1,2
  groups = (parts,kv_heads,ceildiv(capacity,query_rows))
  # Qwen3-0.6B/Mali2.1: share cached KV tiles across nearby queries. Decoder
  # capacity1 and unmeasured geometries retain their existing dispatch order.
  if subgroups and (heads,kv_heads,dim,capacity,query_rows)==(16,8,128,64,2):
    if context==16384:
      part_axis,kv_axis,row_axis,groups = 2,0,1,(kv_heads,ceildiv(capacity,query_rows),parts)
    elif context==40960:
      part_axis,kv_axis,row_axis,groups = 1,2,0,(ceildiv(capacity,query_rows),parts,kv_heads)
  dot_init = ''.join(f'float4 dot{h}=(float4)0;' for h in range(group))
  dot_math = ''.join(f'dot{h}=fma(vload4(0,q+{h}*D+c),k,dot{h});' for h in range(group))
  dot_store = ''.join(f'''prob[{h}*N+key]=row+{h//gqa}<count && first+key<min(total,total-count+row+{h//gqa}+1)?
    ((dot{h}.x+dot{h}.y)+(dot{h}.z+dot{h}.w))*{1/math.sqrt(dim):.17g}f:-INFINITY;''' for h in range(group))
  dot_kernel = f'''for(int key=lane;key<N;key+={threads}) {{
    {dot_init}
    if(first+key<last) {{for(int c=0;c<D;c+=4) {{float4 k=convert_float4(vload4(0,cache+(kv*{context}+first+key)*D+c));
      {dot_math}}}}}
    {dot_store}
  }}'''
  query_load = f'''__local float q[G*D];
    for(int i=lane;i<G*D;i+={threads}){{int h=i/D,t=row+h/{gqa},head=kv*{gqa}+h%{gqa};
      q[i]=t<count?query[(head*C+t)*D+i%D]:0;}}
    barrier(CLK_LOCAL_MEM_FENCE);'''
  if subgroups:
    channels = dim//16
    vector = 'float' if channels==1 else f'float{channels}'
    def load(pointer):return f'({pointer})[0]' if channels==1 else f'vload{channels}(0,{pointer})'
    def component(name,index):return name if channels==1 else f'{name}.s{index:x}'
    query_load = 'int subgroup=get_sub_group_id(),sublane=get_sub_group_local_id();'
    for h in range(group):
      pointer = f'query+((kv*{gqa}+{h%gqa})*C+row+{h//gqa})*D+sublane*{channels}'
      query_load += f'{vector} q{h}=row+{h//gqa}<count?{load(pointer)}:({vector})0;'
    dots = ''.join(f'float dot{h}=0;' for h in range(group))
    for c in range(channels):
      dots += ''.join(f'dot{h}=fma({component(f"q{h}",c)},{component("k",c)},dot{h});' for h in range(group))
    for h in range(group):
      dots += f'''{{float result=-INFINITY;
        if(row+{h//gqa}<count && first+key<min(total,total-count+row+{h//gqa}+1))
          result=sub_group_reduce_add(dot{h})*{1/math.sqrt(dim):.17g}f;
        if(sublane==0)prob[{h}*N+key]=result;}}'''
    pointer = f'cache+(kv*{context}+first+key)*D+sublane*{channels}'
    key_load = f'(float)({pointer})[0]' if channels==1 else f'convert_{vector}({load(pointer)})'
    dot_kernel = f'''for(int key=subgroup;key<N;key+=get_num_sub_groups()) {{
      {vector} k=first+key<last?{key_load}:({vector})0;{dots}
    }}'''
  max_init = ''.join(f'float maximum{h}=scratch[{h}*{threads}];' for h in range(group))
  probability = ''.join(f'''{{float sum=0,shift=isinf(maximum{h})?0:maximum{h};
    for(int key=lane;key<N;key+={threads}){{float p=exp(prob[{h}*N+key]-shift);prob[{h}*N+key]=p;sum+=p;}}
    scratch[{h}*{threads}+lane]=sum;}}''' for h in range(group))
  stats = ''.join(f'''if(lane=={h} && row+{h//gqa}<C) {{int index=((({h%gqa}+kv*{gqa})*C+row+{h//gqa})*P+part)*S;
    out[index+D]=scratch[{h}*{threads}]>0?maximum{h}:-INFINITY;
    out[index+D+1]=scratch[{h}*{threads}];out[index+D+2]=out[index+D+3]=0;}}''' for h in range(group))
  language = '// tinygrad-cl-std=CL2.0\n' if subgroups else ''
  source = language+f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #define G {group}
  #define D {dim}
  #define N {chunk}
  #define C {capacity}
  #define P {parts}
  #define S {stride}
  __kernel void mali_attention_partial(__global float *out,__global const float *query,
      __global const half *cache,__global const int *length,__global const int *rows) {{
    int lane=get_local_id(0),part=get_group_id({part_axis}),kv=get_group_id({kv_axis}),row=get_group_id({row_axis})*{query_rows};
    int total=*length,count=*rows,first=part*N,last=min(total,total-count+row+{query_rows});
    if(count<1||count>C||total<count||total>{context}||row>=count||first>=last) {{
      int invalid=count<1||count>C||total<count||total>{context};
      for(int i=lane;i<G*S;i+={threads}) {{int h=i/S,c=i%S,t=row+h/{gqa},head=kv*{gqa}+h%{gqa};
        if(t<C)out[((head*C+t)*P+part)*S+c]=invalid?NAN:(c==D?-INFINITY:0);}}
      return;
    }}
    __local float prob[G*N],scratch[G*{threads}];
    {query_load}
    {dot_kernel}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int h=0;h<G;h++) {{float m=-INFINITY;
      for(int key=lane;key<N;key+={threads})m=fmax(m,prob[h*N+key]);scratch[h*{threads}+lane]=m;}}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset={threads//2};offset;offset/=2) {{
      if(lane<offset)for(int h=0;h<G;h++)scratch[h*{threads}+lane]=fmax(scratch[h*{threads}+lane],scratch[h*{threads}+lane+offset]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {max_init}
    barrier(CLK_LOCAL_MEM_FENCE);
    {probability}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset={threads//2};offset;offset/=2) {{
      if(lane<offset)for(int h=0;h<G;h++)scratch[h*{threads}+lane]+=scratch[h*{threads}+lane+offset];
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
    for(int item=lane;item<G*D/4;item+={threads}) {{
      int h=item/(D/4),column=item%(D/4),t=row+h/{gqa},head=kv*{gqa}+h%{gqa},key=0;
      float4 a0=(float4)0,a1=(float4)0,a2=(float4)0,a3=(float4)0;
      for(;key+3<N && first+key+3<last;key+=4) {{
        float p0=prob[h*N+key],p1=prob[h*N+key+1],p2=prob[h*N+key+2],p3=prob[h*N+key+3];
        if(p0>0)a0=fma(convert_float4(vload4(0,cache+({kv_heads*context}+kv*{context}+first+key)*D+column*4)),(float4)p0,a0);
        if(p1>0)a1=fma(convert_float4(vload4(0,cache+({kv_heads*context}+kv*{context}+first+key+1)*D+column*4)),(float4)p1,a1);
        if(p2>0)a2=fma(convert_float4(vload4(0,cache+({kv_heads*context}+kv*{context}+first+key+2)*D+column*4)),(float4)p2,a2);
        if(p3>0)a3=fma(convert_float4(vload4(0,cache+({kv_heads*context}+kv*{context}+first+key+3)*D+column*4)),(float4)p3,a3);
      }}
      for(;key<N && first+key<last;key++) {{float p=prob[h*N+key];
        if(p>0)a0=fma(convert_float4(vload4(0,cache+({kv_heads*context}+kv*{context}+first+key)*D+column*4)),(float4)p,a0);}}
      float4 value=(a0+a1)+(a2+a3);
      if(t<C)vstore4(value,0,out+((head*C+t)*P+part)*S+column*4);
    }}
    {stats}
  }}'''
  return _program('mali_attention_partial',source,(out,query,cache,length,rows),groups,(threads,1,1))


@functools.cache
def _merge(out:UOp, partial:UOp, rows:UOp, heads:int, dim:int, capacity:int, parts:int) -> UOp:
  source = f'''__kernel void mali_attention_merge(__global float *out,__global const float *partial,__global const int *rows) {{
    int item=get_group_id(0),lane=get_local_id(0),row=item%{capacity};
    if(row>=*rows){{vstore4((float4)0,0,out+item*{dim}+lane*4);return;}}
    const __global float *data=partial+item*{parts*(dim+4)};
    float maximum=-INFINITY;
    for(int part=0;part<{parts};part++)maximum=fmax(maximum,data[part*{dim+4}+{dim}]);
    float4 value=(float4)0;float denominator=0;
    for(int part=0;part<{parts};part++) {{
      float mass=data[part*{dim+4}+{dim+1}];
      if(isnan(mass)){{denominator=NAN;break;}}
      if(mass>0) {{float alpha=exp(data[part*{dim+4}+{dim}]-maximum);
        value=fma(vload4(0,data+part*{dim+4}+lane*4),(float4)alpha,value);denominator+=alpha*mass;}}
    }}
    vstore4(value/denominator,0,out+item*{dim}+lane*4);
  }}'''
  return _program('mali_attention_merge',source,(out,partial,rows),(heads*capacity,1,1),(dim//4,1,1))


def _program(name:str, source:str, args:tuple[UOp,...], groups:tuple[sint,sint,sint], local:tuple[int,int,int]) -> UOp:
  sink = UOp.sink(*args,arg=KernelInfo(name))
  info = replace(ProgramInfo.from_sink(sink,Target('CL')),outs=(0,),
    ins=tuple(i for i,arg in enumerate(args) if i and arg.addrspace is not AddrSpace.ALU),
    global_size=cast(tuple[int|float,...],groups),local_size=local)
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=args),UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=source.encode())))


class SharedAttention:
  """One model-owned workspace; dependent decoder layers execute sequentially."""
  __slots__ = ('workspaces','subgroups','consistent_decode')

  def __init__(self, subgroups:bool=False, consistent_decode:bool=False):
    self.workspaces:dict[tuple[int,int,int,int],Tensor] = {}
    self.subgroups = subgroups
    self.consistent_decode = consistent_decode

  def prepare(self, heads:int, dim:int, context:int, capacity:int, device:str):
    key = (heads,dim,context,capacity)
    if key not in self.workspaces:
      self.workspaces[key] = Tensor.empty(heads,capacity,ceildiv(context,512),dim+4,device=device,dtype=dtypes.float).realize()

  def __call__(self, query:Tensor, cache:Tensor, length:int|UOp) -> Tensor:
    key = (query.max_shape[1],query.max_shape[3],cache.max_shape[3],query.max_shape[2])
    return attention(query,cache,length,workspace=self.workspaces.get(key),subgroups=self.subgroups,
                     consistent_decode=self.consistent_decode)


def attention(query:Tensor, cache:Tensor, length:int|UOp, workspace:Tensor|None=None, subgroups:bool=False,
              consistent_decode:bool=False) -> Tensor:
  if (not isinstance(query.device,str) or query.device.split(':')[0]!='CL' or cache.device!=query.device
      or query.dtype is not dtypes.float or cache.dtype is not dtypes.half or query.ndim!=4 or cache.ndim!=5
      or query.shape[0]!=1 or cache.shape[:2]!=(2,1) or query.shape[-1]!=cache.shape[-1]):
    raise ValueError('Unsupported Mali cached attention layout')
  heads,kv_heads,dim,context = query.shape[1],cache.shape[2],query.shape[3],cache.shape[3]
  if (not all(isinstance(size,int) for size in (heads,kv_heads,dim,context)) or heads%kv_heads
      or dim not in (16,32,64,128,256) or context<1 or not 1<=heads//kv_heads<=8):
    raise ValueError('Unsupported Mali cached attention geometry')
  assert isinstance(heads,int) and isinstance(kv_heads,int) and isinstance(dim,int) and isinstance(context,int)
  capacity = query.max_shape[2]
  if not 1<=capacity<=256: raise ValueError('Unsupported Mali query capacity')
  count = Tensor(query.shape[2],dtype=dtypes.int).reshape(1).clone(device=query.device)
  end = Tensor(length,dtype=dtypes.int).reshape(1).clone(device=query.device)
  out = Tensor.empty(*query.max_shape,device=query.device,dtype=dtypes.float)
  q = query.pad_to(query.max_shape).contiguous()
  chunk,parts = 512,ceildiv(context,512)
  query_rows = min(capacity,4,max(1,8//(heads//kv_heads)))
  use_subgroups = subgroups and (heads//kv_heads)*(dim//16+1)+dim//16<=64
  if use_subgroups:
    # At 128 threads, probabilities and reduction scratch use 2560 bytes/head.
    query_rows = min(capacity,4,max(1,((48-dim//16)//(dim//16+1))//(heads//kv_heads)),max(1,12//(heads//kv_heads)))
  # Batched speculative rows must use the same softmax reduction and KV
  # accumulation as a single ordinary token, including subgroup selection.
  if consistent_decode and capacity<=8:query_rows,use_subgroups=1,False
  partial = workspace if workspace is not None else Tensor.empty(heads,capacity,parts,dim+4,device=query.device,dtype=dtypes.float)
  partial = partial.custom_kernel(q,cache,end,count,
    fxn=lambda o,q,c,e,r:_partial(o,q,c,e,r,heads,kv_heads,dim,context,capacity,chunk,query_rows,use_subgroups))[0]
  result = out.custom_kernel(partial,count,fxn=lambda o,p,r:_merge(o,p,r,heads,dim,capacity,parts))[0]
  return result.shrink(tuple((0,size) for size in query.shape))
