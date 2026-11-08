"""CPU gate-half sigmoid with unchanged arithmetic and optional parked row workers."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_sigmoid_source
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@functools.cache
def _program(out:UOp, joint:UOp, count:UOp, capacity:int, depth:int, pool:int|None) -> UOp:
  source = Path(__file__).with_suffix('.c').read_text()+f'''
void rockchip_cpu_sigmoid_gate(float *out,const float *input,const int *rows) {{
  if(sigmoid_gate(out,input,*rows,{capacity},{depth},(sigmoid_parallel_fn){pool or 0}ULL))__builtin_trap();
}}
'''
  inputs = (out,joint,count)
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_sigmoid_gate'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


@functools.cache
def _program_mul(out:UOp, joint:UOp, count:UOp, capacity:int, depth:int, pool:int|None) -> UOp:
  source = '#include <stddef.h>\nextern void *memset(void*,int,size_t);\n' + _vector_sigmoid_source()+f'''
typedef int (*mul_parallel_fn)(void (*)(void**,int,int),void**,int);
typedef struct {{float *out;const float *input;int rows,depth,done[4];}} MulWork;
static void mul_part(float* restrict out,const float* restrict input,int rows,int depth,int group,int groups) {{
  for(int row=group;row<rows;row+=groups) {{
    const float *gate=input+(size_t)row*depth*2,*up=gate+depth;float *dest=out+(size_t)row*depth;
    for(int column=0;column<depth;column+=4) {{
      float4 g=*(const float4*)(gate+column),u=*(const float4*)(up+column);
      *(float4*)(dest+column)=(g*rockchip_sigmoid_vector(g))*u;
    }}
  }}
}}
static void mul_worker(void **args,int unused,int group) {{
  (void)unused;MulWork *w=(MulWork*)args[0];if(group<0||group>=4)return;
  mul_part(w->out,w->input,w->rows,w->depth,group,4);w->done[group]=1;
}}
static int sigmoid_gate_mul_fn(float *out,const float *input,int rows,int capacity,int depth,mul_parallel_fn parallel) {{
  if(!out||!input||rows<1||rows>capacity||capacity>128||depth<4||depth>32768||depth%4)return 1;
  if(!parallel||rows<4)mul_part(out,input,rows,depth,0,1);
  else {{
    MulWork work={{.out=out,.input=input,.rows=rows,.depth=depth,.done={{0,0,0,0}}}};void *args[4]={{&work,0,0,0}};
    int error=parallel(mul_worker,args,0),completed=0;for(int i=0;i<4;i++)completed+=work.done[i]!=0;
    if(completed==0&&error)mul_part(out,input,rows,depth,0,1);
    else if(error||completed!=4)return error?error:2;
  }}
  memset(out+(size_t)rows*depth,0,(size_t)(capacity-rows)*depth*sizeof(float));
  return 0;
}}
void rockchip_cpu_sigmoid_gate_mul(float *out,const float *input,const int *rows) {{
  if(sigmoid_gate_mul_fn(out,input,*rows,{capacity},{depth},(mul_parallel_fn){pool or 0}ULL))__builtin_trap();
}}
'''
  inputs = (out,joint,count)
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_sigmoid_gate_mul'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def sigmoid_gate_mul(joint:Tensor, *, pool:AttentionPool|None=None) -> Tensor:
  """Fuse sigmoid(gate) * gate * up for the second half of a gate/up projection."""
  if joint.device!='CPU' or joint.dtype is not dtypes.float or joint.ndim!=3 or joint.shape[0]!=1:
    raise ValueError('Fused CPU gate/up requires batch-one float32 projections')
  capacity,columns = joint.max_shape[1:]
  depth = columns//2
  if not 1<=capacity<=128 or columns%8 or not 4<=depth<=32768:
    raise ValueError('Unsupported fused CPU gate/up projection shape')
  rows = joint.shape[1]
  if isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity):raise ValueError('Invalid fused CPU gate/up row bounds')
  dense = joint.pad_to(joint.max_shape).contiguous()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  address = pool.prefill_address if pool is not None and pool.active else None
  result = Tensor.empty(1,capacity,depth,device='CPU',dtype=dtypes.float).custom_kernel(dense,count,
    fxn=lambda o,x,n:_program_mul(o,x,n,capacity,depth,address))[0]
  return result.shrink(((0,1),(0,rows),(0,depth)))


def sigmoid_gate(joint:Tensor, *, pool:AttentionPool|None=None) -> Tensor:
  """Apply sigmoid to the first half of a joint gate/up projection."""
  if joint.device!='CPU' or joint.dtype is not dtypes.float or joint.ndim!=3 or joint.shape[0]!=1:
    raise ValueError('CPU sigmoid requires batch-one float32 gate/up projections')
  capacity,columns = joint.max_shape[1:]
  depth = columns//2
  if not 1<=capacity<=128 or columns%8 or not 4<=depth<=32768:
    raise ValueError('Unsupported CPU sigmoid projection shape')
  rows = joint.shape[1]
  if isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity):raise ValueError('Invalid CPU sigmoid row bounds')
  dense = joint.pad_to(joint.max_shape).contiguous()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  address = pool.prefill_address if pool is not None and pool.active else None
  result = Tensor.empty(1,capacity,depth,device='CPU',dtype=dtypes.float).custom_kernel(dense,count,
    fxn=lambda o,x,n:_program(o,x,n,capacity,depth,address))[0]
  return result.shrink(((0,1),(0,rows),(0,depth)))

def _program_vector(out:UOp, joint:UOp, count:UOp, capacity:int, depth:int) -> UOp:
  """Sigmoid on the first half of a gate/up projection, float4 math.

  The existing `_program` compiles a C file whose body is scalar-per-lane with
  three NaN/Inf predicate chains repeated for each of the four lanes. This uses
  `rockchip_sigmoid_vector`, which operates on a float4 in one pass, so the
  emitted loop is genuinely vector.
  """
  # size_t is not declared in this freestanding compile; the existing kernels
  # include <stddef.h> explicitly, and so must this one.
  source = '#include <stddef.h>\n' + _vector_sigmoid_source() + f'''
void rockchip_cpu_sigmoid_gate_vector(float *out,const float *input,const int *rows) {{
  int active=*rows;
  if(active<1 || active>{capacity}){{for(int i=0;i<{capacity*depth};i++)out[i]=0;return;}}
  for(int row=0;row<active;row++) {{
    const float *gate=input+(size_t)row*{depth*2};
    float *dest=out+(size_t)row*{depth};
    for(int column=0;column<{depth};column+=4) {{
      float4 value=*((const float4*)(gate+column));
      *((float4*)(dest+column))=rockchip_sigmoid_vector(value);
    }}
  }}
}}'''
  inputs = (out,joint,count)
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_cpu_sigmoid_gate_vector'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def sigmoid_gate_vector(joint:Tensor, *, pool:AttentionPool|None=None) -> Tensor:
  """Vectorised sigmoid on the first half of a joint gate/up projection."""
  if joint.device!='CPU' or joint.dtype is not dtypes.float or joint.ndim!=3 or joint.shape[0]!=1:
    raise ValueError('CPU sigmoid requires batch-one float32 gate/up projections')
  capacity,columns = joint.max_shape[1:]
  depth = columns//2
  if not 1<=capacity<=128 or columns%8 or not 4<=depth<=32768 or depth%4:
    raise ValueError('Unsupported CPU sigmoid projection shape')
  rows = joint.shape[1]
  if isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity):raise ValueError('Invalid CPU sigmoid row bounds')
  dense = joint.pad_to(joint.max_shape).contiguous()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  result = Tensor.empty(1,capacity,depth,device='CPU',dtype=dtypes.float).custom_kernel(dense,count,
    fxn=lambda o,x,n:_program_vector(o,x,n,capacity,depth))[0]
  return result.shrink(((0,1),(0,rows),(0,depth)))

def _program_pair_vector(out:UOp, gate:UOp, up:UOp, count:UOp, capacity:int, depth:int) -> UOp:
  """gate * sigmoid(gate) * up in one float4 pass over a batch-one CPU row."""
  source = '#include <stddef.h>\n' + _vector_sigmoid_source() + f'''
void rockchip_gated_activation_vector(float *out,const float *gate,const float *up,const int *rows) {{
  int active=*rows;
  if(active<1 || active>{capacity}){{for(int i=0;i<{capacity*depth};i++)out[i]=0;return;}}
  for(int row=0;row<active;row++) {{
    const float *g=gate+(size_t)row*{depth},*u=up+(size_t)row*{depth};
    float *dest=out+(size_t)row*{depth};
    for(int column=0;column<{depth};column+=4) {{
      float4 gv=*((const float4*)(g+column));
      float4 uv=*((const float4*)(u+column));
      float4 sig=rockchip_sigmoid_vector(gv);
      *((float4*)(dest+column))=gv*sig*uv;
    }}
  }}
}}'''
  inputs = (out,gate,up,count)
  renderer = _renderer()
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_gated_activation_vector'))
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def sigmoid_pair_vector(gate:Tensor, up:Tensor) -> Tensor:
  """Fused `gate*sigmoid(gate)*up` for a batch-one CPU float tensor pair."""
  if (gate.device!='CPU' or up.device!='CPU' or gate.dtype is not dtypes.float
      or up.dtype is not dtypes.float or gate.ndim!=3 or up.ndim!=3
      or gate.shape!=up.shape):
    raise ValueError('Vectorised gated activation requires matching CPU float triples')
  capacity = math.prod(gate.max_shape[:-1])
  depth = gate.max_shape[-1]
  if not 1<=capacity<=128 or not isinstance(depth,int) or depth%4 or not 4<=depth<=32768:
    raise ValueError('Unsupported vectorised gated activation shape')
  rows = math.prod(gate.shape[:-1])
  dense_gate = gate.pad_to(gate.max_shape).contiguous()
  dense_up = up.pad_to(up.max_shape).contiguous()
  count = Tensor(rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  result = Tensor.empty(*gate.max_shape,device='CPU',dtype=dtypes.float).custom_kernel(
    dense_gate,dense_up,count,fxn=lambda o,g,u,n:_program_pair_vector(o,g,u,n,capacity,depth))[0]
  return result.shrink(tuple((0,s) for s in gate.shape))
