"""Bounded raw INT8 gated activations and per-K-group quantization for prefill."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from typing import NamedTuple, TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv, getenv, round_up
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_int8 import (RockchipInt8Linear, RockchipInt8SplitLinear, _quantize_vector_source,
                                              dequantize_tiles, quantize, raw_int8_matmul)
from tinygrad.runtime.support.rockchip_int8 import padded_depth
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool


class ProjectionTile(NamedTuple):
  first_row:int
  rows:int
  first_column:int
  columns:int
  scale_index:int
  sdk:bool=False


def payload_layout(rows:int, depths:tuple[int,...]) -> tuple[tuple[int,int,int,int],...]:
  """(byte offset, padded depth, row tile, byte size) for each independent K group."""
  result=[]
  offset=0
  for depth in depths:
    padded=padded_depth(depth)
    tile=128 if padded<=2048 else 64
    size=sum(round_up(min(tile,rows-first)*4,64)+min(tile,rows-first)*padded for first in range(0,rows,tile))
    result.append((offset,padded,tile,size))
    offset+=size
  return tuple(result)


def _exponent(activation:str) -> str:
  if activation=='gelu_tanh':return '(g+0.044714998453855515f*g*g*g)*-2.302208185195923f'
  if activation=='silu':return 'g*-1.4426950216293335f'
  if activation=='gelu':return '(g*g)*-0.7213475108146667f'
  raise ValueError('Unsupported fused gated activation')


def _activation_value(activation:str) -> str:
  if activation=='gelu':
    return r"""float4 erf_exp=rockchip_exp2_vector(exponent);
      float4 sign=(g*(float4)0.7071067690849304f)<(float4)0?(float4)-1:(float4)1;
      float4 denominator=(float4)1+(g*sign)*(float4)0.23164188861846924f;
      float4 polynomial=(float4)1.0614054203033447f/denominator+(float4)-1.453152060508728f;
      polynomial=polynomial/denominator+(float4)1.421413779258728f;
      polynomial=polynomial/denominator+(float4)-0.2844967246055603f;
      polynomial=polynomial/denominator+(float4)0.2548295855522156f;
      polynomial=polynomial/denominator;
      float4 value=((g*((float4)1+sign*((float4)1-polynomial*erf_exp)))*u)*(float4)0.5f;"""
  if activation=='silu':
    # The serving FFN materializes sigmoid in a custom kernel. Preserve that
    # reciprocal rounding before multiplying by gate, then up.
    return 'float4 sigmoid=1.0f/(1.0f+rockchip_exp2_vector(exponent));float4 value=(g*sigmoid)*u;'
  return 'float4 value=(g/(1.0f+rockchip_exp2_vector(exponent)))*u;'


def _static_projection_loop(tiles:tuple[ProjectionTile,...], rows:int, hidden:int, first:int, depth:int,
                             activation:str='gelu_tanh') -> str:
  """Resolve raw row/column fragments before emitting each contiguous channel loop."""
  fragments:list[tuple[int,ProjectionTile,int,int]]=[]
  for index,tile in enumerate(tiles):
    width=round_up(ceildiv(tile.columns,3),32)
    parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
    if len(parts)!=3:parts=[(0,tile.columns)]
    fragments.extend((index,tile,start,count) for start,count in parts)
  row_bounds=sorted({0,rows,*(tile.first_row for tile in tiles),*(tile.first_row+tile.rows for tile in tiles)})
  result=[]
  for row_begin,row_end in zip(row_bounds,row_bounds[1:]):
    active=[fragment for fragment in fragments if fragment[1].first_row<=row_begin<fragment[1].first_row+fragment[1].rows]
    bounds={0,depth}
    for _,tile,start,count in active:
      for base in (first,first+hidden):
        bounds.update(max(0,min(depth,column-base)) for column in (tile.first_column+start,tile.first_column+start+count))
    columns=sorted(bounds)
    loops=[]
    for begin,end in zip(columns,columns[1:]):
      loads=[]
      for base in (first,first+hidden):
        index,tile,start,count=next(fragment for fragment in active
          if fragment[1].first_column+fragment[2]<=base+begin<base+end<=fragment[1].first_column+fragment[2]+fragment[3])
        offset=start*tile.rows+base-tile.first_column-start
        position=(f'{start*tile.rows}+(({base-tile.first_column-start}+c)/4*{tile.rows}+row-{tile.first_row})*4' if tile.sdk else
                  f'{offset}+(row-{tile.first_row})*{count}+c')
        loads.append(f'''(__builtin_convertvector(*(const int4*)(raw[{index}]+({position})),float4)*
          (float4)sa[{tile.scale_index}][row-{tile.first_row}])*(*(const float4*)(sw+{base}+c))''')
      loops.append(f'''for(int c={begin};c<{end};c+=4) {{
        float4 g={loads[0]},u={loads[1]};
        float4 exponent={_exponent(activation)};
        {_activation_value(activation)}*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}''')
    result.append(f'if(row>={row_begin} && row<{row_end}) {{'+''.join(loops)+'}')
  return ''.join(result)


def _batched_projection_loop(tiles:tuple[ProjectionTile,...], rows:int, hidden:int, first:int, depth:int,
                             activation:str, batch:int) -> str:
  fragments:list[tuple[int,ProjectionTile,int,int]]=[]
  for index,tile in enumerate(tiles):
    width=round_up(ceildiv(tile.columns,3),32)
    parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
    if len(parts)!=3:parts=[(0,tile.columns)]
    fragments.extend((index,tile,start,count) for start,count in parts)
  bounds=sorted({0,rows,*(t.first_row for t in tiles),*(t.first_row+t.rows for t in tiles)})
  result=[]
  for row_begin,row_end in zip(bounds,bounds[1:]):
    active=[f for f in fragments if f[1].first_row<=row_begin<f[1].first_row+f[1].rows]
    columns={0,depth}
    for _,tile,start,count in active:
      for base in (first,first+hidden):
        columns.update(max(0,min(depth,column-base)) for column in (tile.first_column+start,tile.first_column+start+count))
    points=sorted(columns)
    loops=[]
    for begin,end in zip(points,points[1:]):
      lanes=[]
      for lane in range(batch):
        loads=[]
        for base in (first,first+hidden):
          index,tile,start,count=next(f for f in active
            if f[1].first_column+f[2]<=base+begin<base+end<=f[1].first_column+f[2]+f[3])
          offset=start*tile.rows+base-tile.first_column-start
          position=(f'{start*tile.rows}+(({base-tile.first_column-start}+c)/4*{tile.rows}+row+{lane}-{tile.first_row})*4' if tile.sdk else
                    f'{offset}+((row+{lane})-{tile.first_row})*{count}+c')
          loads.append(f'''(__builtin_convertvector(*(const int4*)(raw[{index}]+({position})),float4)*
            (float4)sa[{tile.scale_index}][(row+{lane})-{tile.first_row}])*(*(const float4*)(sw+{base}+c))''')
        lanes.append(f'''{{float4 g={loads[0]},u={loads[1]};float4 exponent={_exponent(activation)};
          {_activation_value(activation)}*(float4*)(batch_values[{lane}]+c)=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[{lane}]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[{lane}]=a>batch_maximum[{lane}]?a:batch_maximum[{lane}];}}''')
      loops.append(f'for(int c={begin};c<{end};c+=4) {{'+''.join(lanes)+'}')
    result.append(f'if(row>={row_begin} && row+{batch}<={row_end}) {{'+''.join(loops)+'}')
  return ''.join(result)


def _row_batch_body(tiles:tuple[ProjectionTile,...], rows:int, hidden:int, first_channel:int, depth:int,
                    activation:str, batch:int, offset:int, padded:int, row_tile:int, group:int) -> str:
  bounds=sorted({0,rows,*(t.first_row for t in tiles),*(t.first_row+t.rows for t in tiles)})
  safe=' || '.join(f'(row>={a} && row+{batch}<={b})' for a,b in zip(bounds,bounds[1:]))
  packing=[]
  for lane in range(batch):
    packing.append(f'''{{int row=batch_start+{lane};float *values=batch_values[{lane}];
      float4 maximum=batch_maximum[{lane}];uint4 bad=batch_bad[{lane}];
      int first=row/{row_tile}*{row_tile},count={rows}-first<{row_tile}?{rows}-first:{row_tile};
      unsigned char *base=out+{offset}+first*{padded}+first/{row_tile}*{round_up(row_tile*4,64)};
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*{padded};
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[{group*rows}+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<{depth};c+=4){{float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}}
      for(;c<{padded};c++)dest[c]=0;
    }}''')
  return f'''if(row+{batch}<=end && row+{batch}<=active && ({safe})) {{
    int batch_start=row;float batch_values[{batch}][{depth}] __attribute__((aligned(16)));
    float4 batch_maximum[{batch}]={{0}};uint4 batch_bad[{batch}]={{0}};
    {_batched_projection_loop(tiles,rows,hidden,first_channel,depth,activation,batch)}
    {''.join(packing)}row+={batch-1};continue;
  }}
      '''


@functools.cache
def _program(out:UOp, raw:tuple[UOp,...], sa:tuple[UOp,...], sw:UOp, active:UOp, scale_out:UOp,
             rows:int, hidden:int, depths:tuple[int,...], tiles:tuple[ProjectionTile,...], static_layout:bool=False,
             activation:str='gelu_tanh', pool:int|None=None, row_batch:int=1) -> UOp:
  if row_batch not in (1,2,4):raise ValueError('Unsupported raw SwiGLU row batch')
  if activation!='silu' or not static_layout or rows<16:row_batch=1
  exponent=_exponent(activation)
  names={'gelu':'rockchip_int8_erf_gelu_pack','gelu_tanh':'rockchip_int8_gelu_pack','silu':'rockchip_int8_swiglu_pack'}
  name=names[activation]
  project=[]
  for index,tile in enumerate(tiles):
    width=round_up(ceildiv(tile.columns,3),32)
    parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
    if len(parts)!=3:parts=[(0,tile.columns)]
    for start,count in parts:
      column=tile.first_column+start
      position=(f'{start*tile.rows}+((column-{column})/4*{tile.rows}+row-{tile.first_row})*4+column%4' if tile.sdk else
                f'{start*tile.rows}+(row-{tile.first_row})*{count}+column-{column}')
      project.append(f'''if(row>={tile.first_row} && row<{tile.first_row+tile.rows} && column>={column} && column<{column+count}) {{
        int position={position};
        float4 value=__builtin_convertvector(*(const int4*)(raw[{index}]+position),float4);
        return (value*(float4)sa[{tile.scale_index}][row-{tile.first_row}])*(*(const float4*)(sw+column));
      }}''')
  layout=payload_layout(rows,depths)
  loops=[]
  padding=[]
  first_channel=0
  for group,(depth,(offset,padded,row_tile,_)) in enumerate(zip(depths,layout)):
    projection_loop=_static_projection_loop(tiles,rows,hidden,first_channel,depth,activation) if static_layout else f'''
      for(int c=0;c<{depth};c+=4) {{
        float4 g=project(raw,sa,sw,row,c+{first_channel}),u=project(raw,sa,sw,row,c+{hidden+first_channel});
        float4 exponent={exponent};
        {_activation_value(activation)}*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}'''
    for first in range(0,rows,row_tile):
      count=min(row_tile,rows-first)
      base=offset+first*padded+(first//row_tile)*round_up(row_tile*4,64)
      padding.append(f'for(int i={count*4};i<{round_up(count*4,64)};i++)out[{base}+i]=0;')
    batch_body=_row_batch_body(tiles,rows,hidden,first_channel,depth,activation,row_batch,offset,padded,row_tile,group) if row_batch>1 else ''
    loops.append(f'''for(int row=begin;row<end;row++) {{
      {batch_body}int first=row/{row_tile}*{row_tile},count={rows}-first<{row_tile}?{rows}-first:{row_tile};
      unsigned char *base=out+{offset}+first*{padded}+first/{row_tile}*{round_up(row_tile*4,64)};
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*{padded};
      if(row>=active) {{((float*)base)[row-first]=scales[{group*rows}+row]=1e-8f;
        for(int c=0;c<{padded};c++)dest[c]=0;continue;}}
      float values[{depth}] __attribute__((aligned(16)));float4 maximum={{0}};uint4 bad={{0}};
      {projection_loop}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[{group*rows}+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<{depth};c+=4){{float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}}
      for(;c<{padded};c++)dest[c]=0;
    }}''')
    first_channel+=depth
  parameters=','.join([*(f'const int *r{i}' for i in range(len(raw))),*(f'const float *a{i}' for i in range(len(sa))),
                       'const float *sw','const int *active','float *scales'])
  initialize=f'''const int *raw[{len(raw)}]={{{','.join(f'r{i}' for i in range(len(raw)))}}};
    const float *sa[{len(sa)}]={{{','.join(f'a{i}' for i in range(len(sa)))}}};'''
  parallel=rows>=16 and (activation!='gelu' or pool is not None)
  declarations='''typedef unsigned long Thread;
  extern int pthread_create(Thread*,const void*,void*(*)(void*),void*);extern int pthread_join(Thread,void**);
  typedef struct {unsigned char *out;const int *const *raw;const float *const *sa;const float *sw;float *scales;int active,first,last;} Work;
  static void *worker(void *p){Work *w=p;part(w->out,w->raw,w->sa,w->sw,w->scales,w->active,w->first,w->last);return (void*)0;}''' if parallel else ''
  dispatch=f'''Work work[4];Thread ids[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,raw,sa,sw,scales,*active,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(ids+i,(void*)0,worker,work+i)==0;if(!created[i])worker(work+i);}}
    worker(work+3);for(int i=0;i<3;i++)if(created[i])pthread_join(ids[i],(void**)0);''' if parallel else \
      f'part(out,raw,sa,sw,scales,*active,0,{rows});'
  if parallel and pool is not None:
    declarations=f"""typedef struct {{unsigned char *out;const int *const *raw;const float *const *sa;
      const float *sw;float *scales;int active;}} Work;
    static void pool_part(void **args,int unused,int group){{
      Work *w=args[0];part(w->out,w->raw,w->sa,w->sw,w->scales,w->active,
                          {rows}*group/4,{rows}*(group+1)/4);
    }}"""
    dispatch=f"""Work work={{out,raw,sa,sw,scales,*active}};void *args[4]={{&work,0,0,0}};
      if(((int(*)(void(*)(void**,int,int),void**,int)){pool}ULL)(pool_part,args,0))__builtin_trap();"""
  source=_vector_exp2_source()+_quantize_vector_source()+f'''
  typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 project(const int *const *raw,const float *const *sa,const float *sw,int row,int column) {{
    {''.join(project)}return (float4)__builtin_nanf("");
  }}
  static void part(unsigned char *out,const int *const *raw,const float *const *sa,const float *sw,float *scales,int active,int begin,int end) {{
    {''.join(loops)}
  }}
  {declarations}
  void {name}(unsigned char *out,{parameters}) {{
    if(*active<1 || *active>{rows})__builtin_trap();{initialize}{''.join(padding)}{dispatch}
  }}'''
  inputs=(out,*raw,*sa,sw,active,scale_out)
  sink=UOp.sink(*inputs,arg=KernelInfo(name))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,len(inputs)-1),ins=tuple(range(1,len(inputs)-1)))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _pack_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                     hidden:int, depths:tuple[int,...], active_rows:int|UOp, activation:str, pool:AttentionPool|None=None) -> tuple[Tensor,Tensor]:
  _exponent(activation)
  if (not 1<=rows<=128 or not 512<=hidden<=12288 or hidden%32 or sum(depths)!=hidden
      or any(not 1<=depth<=4096 or depth%32 for depth in depths) or not raw or len(raw)!=len(tiles)
      or not sa or any(t.device!='CPU' or t.dtype is not dtypes.float for t in (*sa,sw)) or sw.shape!=(hidden*2,)
      or any(t.device!='CPU' or t.dtype is not dtypes.int32 for t in raw)):
    raise ValueError('Unsupported fused GELU projection layout')
  coverage=[]
  for value,tile in zip(raw,tiles):
    if (type(tile.sdk) is not bool or not 0<=tile.first_row<tile.first_row+tile.rows<=rows or tile.first_column%32 or tile.columns%32
        or not 0<=tile.first_column<tile.first_column+tile.columns<=hidden*2 or not 0<=tile.scale_index<len(sa)
        or value.numel()!=tile.rows*tile.columns or sa[tile.scale_index].shape!=(tile.rows,)):
      raise ValueError('Invalid raw GELU projection tile')
    coverage.append((tile.first_row,tile.first_row+tile.rows,tile.first_column,tile.first_column+tile.columns))
  for row in range(rows):
    intervals=sorted((a,b) for first,last,a,b in coverage if first<=row<last)
    if not intervals or intervals[0][0]!=0 or intervals[-1][1]!=hidden*2 or any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:])):
      raise ValueError('Raw GELU tiles must cover each row once')
  if isinstance(active_rows,int) and not 1<=active_rows<=rows or isinstance(active_rows,UOp) and not 1<=active_rows.vmin<=active_rows.vmax<=rows:
    raise ValueError('Invalid fused GELU active row count')
  address=pool.prefill_address if pool is not None and pool.active and rows>=16 else None
  layout=payload_layout(rows,depths)
  count=Tensor(active_rows,device='CPU',dtype=dtypes.int).reshape(1).clone()
  row_batch=getenv('ONPU_SWIGLU_PREFILL_ROWS',1) if activation=='silu' and rows>=16 else 1
  if row_batch not in (1,2,4):raise ValueError('Unsupported raw SwiGLU row batch')
  result=Tensor.empty(sum(x[3] for x in layout),device='CPU',dtype=dtypes.uint8).custom_kernel(
    *raw,*sa,sw,count,Tensor.empty(len(depths)*rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,*args:_program(o,tuple(args[:len(raw)]),tuple(args[len(raw):len(raw)+len(sa)]),args[-3],args[-2],args[-1],
                               rows,hidden,depths,tuple(tiles),
                               True if activation in ('silu','gelu') else bool(getenv('ONPU_GELU_PACK_STATIC_LAYOUT',1)),
                               activation,address,row_batch))
  return result[0],result[-1]


def pack_gelu_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                         hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None) -> tuple[Tensor,Tensor]:
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'gelu_tanh',pool)


def pack_swiglu_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                           hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None) -> tuple[Tensor,Tensor]:
  """Fuse raw SwiGLU with the down projection's independent K-group quantization."""
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'silu',pool)


def pack_erf_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                        hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None) -> tuple[Tensor,Tensor]:
  """Preserve per-K FP32 maxima/scales and packed codes while fusing raw erf-GELU."""
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'gelu',pool)


class _RockchipInt8GatedFeedForward:
  activation='gelu_tanh'
  decode_enabled=False
  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None):
    self.projections,self.down,self.pool=projections,down,pool

  def enabled(self,x:Tensor) -> bool:
    joint=self.projections.joint
    parts=self.down.parts if isinstance(self.down,RockchipInt8SplitLinear) else [self.down]
    phase=self.projections.layers[0].phase
    decoding=phase=='decode'
    return (isinstance(joint,RockchipInt8Linear) and x.device=='CPU' and x.dtype is dtypes.float and x.ndim==3 and x.shape[0]==1
      and ((self.activation in ('silu','gelu') and self.decode_enabled and x.max_shape[1]==1) if decoding else 16<=x.max_shape[1]<=128)
      and len(joint.packed)==len(joint.scales)==1 and 512<=self.down.depth<=12288 and self.down.depth%32==0
      and joint.columns==self.down.depth*2 and 1<=self.down.columns<=12288
      and joint.bias is self.down.bias is None and self.down.input_scale is None
      and bool(parts) and all(len(p.packed)==len(p.scales)==1 and p.input_scale is None and p.bias is None for p in parts)
      and all(getattr(layer,'decode' if decoding else 'prefill')=='on' for layer in (*self.projections.layers,self.down)))

  def __call__(self,x:Tensor) -> Tensor:
    joint=self.projections.joint
    assert isinstance(joint,RockchipInt8Linear)
    capacity=math.prod(x.max_shape[:-1])
    assert isinstance(capacity,int)
    dense=capacity_input(x).reshape(capacity,joint.depth)
    k=joint.packed[0].shape[1]*32
    assert isinstance(k,int)
    input_rows=128 if k<=2048 else 64
    raw=[]
    sa:list[Tensor]=[]
    tiles=[]
    for first in range(0,capacity,input_rows):
      part=dense[first:first+input_rows]
      rows=part.shape[0]
      assert isinstance(rows,int)
      sdk=(self.activation=='silu' and (rows,joint.columns,k)==(128,9728,896) and bool(getenv('ONPU_INT8_SDK_PREFILL',0)))
      payload,scale=quantize(joint.scaled_input(part),k,sdk_prefill=sdk)
      scale_index=len(sa)
      sa.append(scale)
      for column in range(0,joint.columns,24576 if rows==1 else 12288):
        count=min(24576 if rows==1 else 12288,joint.columns-column)
        raw.append(raw_int8_matmul(payload,joint.packed[0],rows,count,column,sdk_prefill=sdk))
        tiles.append(ProjectionTile(first,rows,column,count,scale_index,sdk))
    parts=self.down.parts if isinstance(self.down,RockchipInt8SplitLinear) else [self.down]
    depths=tuple(part.depth for part in parts)
    pack={'silu':pack_swiglu_projection,'gelu':pack_erf_projection,'gelu_tanh':pack_gelu_projection}[self.activation]
    packed,scales=pack(raw,sa,joint.scales[0],tiles,capacity,self.down.depth,depths,math.prod(x.shape[:-1]),pool=self.pool)
    shared_input=(capacity>=16 and getenv('ONPU_FFN_PACKED_UPLOAD',0) and len({layer.packed[0].device for layer in parts})==1)
    if shared_input:
      # All K groups share one fully written payload. Upload its allocation
      # once, then retain the original per-group/per-row scale boundaries.
      packed=packed.to(parts[0].packed[0].device)
    outputs=[]
    for index,(layer,(offset,padded,tile,_)) in enumerate(zip(parts,payload_layout(capacity,depths))):
      values=[]
      activation_scales=[]
      counts=[]
      for first in range(0,capacity,tile):
        rows=min(tile,capacity-first)
        begin=offset+first*padded+(first//tile)*round_up(tile*4,64)
        payload=packed if shared_input else packed[begin:begin+round_up(rows*4,64)+rows*padded]
        values.append(raw_int8_matmul(payload,layer.packed[0],rows,layer.columns,payload_offset=begin if shared_input else None))
        activation_scales.append(scales[index*capacity+first:index*capacity+first+rows])
        counts.append(rows)
      outputs.append(dequantize_tiles(values,activation_scales,layer.scales[0],counts,layer.columns))
    result=outputs[0]
    for value in outputs[1:]:result=result+value
    result=result.reshape(*x.max_shape[:-1],self.down.columns).shrink(tuple((0,s) for s in (*x.shape[:-1],self.down.columns)))
    return result.cast(x.dtype)


class RockchipInt8GeluFeedForward(_RockchipInt8GatedFeedForward):
  pass


class RockchipInt8SwiGLUFeedForward(_RockchipInt8GatedFeedForward):
  activation='silu'

  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None,decode:bool=False):
    if type(decode) is not bool:raise ValueError('Raw SwiGLU decode selection must be a bool')
    super().__init__(projections,down,pool=pool)
    self.decode_enabled=decode


class RockchipInt8ErfGeluFeedForward(_RockchipInt8GatedFeedForward):
  activation='gelu'

  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None,decode:bool=False):
    if type(decode) is not bool:raise ValueError('Raw erf GELU decode selection must be a bool')
    super().__init__(projections,down,pool=pool)
    self.decode_enabled=decode
