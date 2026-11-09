"""Bounded raw INT8 gated activations and per-K-group quantization for prefill."""
from __future__ import annotations
import functools, math
from dataclasses import replace
from typing import NamedTuple, TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar, ceildiv, getenv, round_up
from tinygrad.llm.kernels.rockchip import RockchipProjections, RockchipLinear
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.llm.kernels.rockchip_hadamard import ONPU_INT8_HADAMARD
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source
from tinygrad.llm.kernels.rockchip_int8 import (RockchipInt8Linear, RockchipInt8SplitLinear, _quantize_vector_source,
                                              dequantize_tiles, quantize, raw_int8_matmul, ONPU_GROUPED_DECODE_DOWN)
from tinygrad.llm.kernels.rockchip_grouped_int8 import dequantize as grouped_dequantize, raw_matmul as grouped_raw_matmul
from tinygrad.runtime.support.rockchip_int8 import padded_depth
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_SWIGLU_CHANNEL_SCRATCH = ContextVar('ONPU_SWIGLU_CHANNEL_SCRATCH',0)
ONPU_SWIGLU_C4_PREFETCH = ContextVar('ONPU_SWIGLU_C4_PREFETCH',0)
ONPU_SWIGLU_BATCHED_PACK = ContextVar('ONPU_SWIGLU_BATCHED_PACK',0)
ONPU_SWIGLU_K_SPLIT = ContextVar('ONPU_SWIGLU_K_SPLIT',0)
ONPU_WIDE_RAW_SWIGLU = ContextVar('ONPU_WIDE_RAW_SWIGLU',0)
_batched_pack_traces = 0


def batched_pack_stats() -> dict[str,int]:
  from tinygrad.llm.kernels.rockchip_swiglu_ksplit import split_prefill_stats
  split = split_prefill_stats()
  return {'traces':_batched_pack_traces+split['traces'],'rows':128,'hidden':split.get('hidden',0) or 4864}

SDK_SILU_DIV_PREFILL_SHAPES = frozenset(((128,6144,1024),(128,8192,1024)))
SDK_SILU_PREFILL_SHAPES = frozenset(((128,9728,896),(128,17920,1536),(128,9216,1024)))
SDK_ERF_GELU_PREFILL_SHAPES = frozenset(((128,13312,2048),))


class ProjectionTile(NamedTuple):
  first_row:int
  rows:int
  first_column:int
  columns:int
  scale_index:int
  sdk:bool=False


# Preserve the qualified SDK C4 tiles when batching four adjacent tanh-GELU rows.
SDK_TANH_PREFILL_SHAPES = frozenset(((128,12288,1536),(128,24576,1536)))
_SDK_TANH_ROW_TILES = frozenset(((ProjectionTile(0,128,0,12288,0,True),),
  (ProjectionTile(0,128,0,12288,0,True),ProjectionTile(0,128,12288,12288,0,True))))


def _row_batch_supported(activation:str, tiles:tuple[ProjectionTile,...]) -> bool:
  return (activation in ('silu','silu_div') or activation=='gelu' and bool(tiles) and all(tile.sdk for tile in tiles)
          or activation=='gelu_tanh' and tiles in _SDK_TANH_ROW_TILES)


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


def grouped_down_payload(packed:Tensor, scales:Tensor, rows:int, depths:tuple[int,...], groups:int) -> tuple[Tensor,Tensor]:
  """Reuse the fused FFN pack as one group-major payload for the grouped down projection.

  The fused pack stores an activation-scale header per row tile inside each K group.
  The grouped matrix consumes one header followed by contiguous group-major codes, so
  strip the per-tile headers and concatenate the codes in group order. Scales are
  returned group-major because the grouped dequantizer indexes them the same way.
  """
  if (packed.device!='CPU' or packed.dtype is not dtypes.uint8 or packed.ndim!=1 or scales.device!='CPU'
      or scales.dtype is not dtypes.float or scales.shape!=(len(depths)*rows,)
      or type(rows) is not int or not 1<=rows<=128 or type(groups) is not int or not 1<=groups<=len(depths)
      or any(type(depth) is not int or depth<=0 or depth%32 for depth in depths[:groups])
      or any(padded_depth(depth)!=padded_depth(depths[0]) for depth in depths[:groups])):
    raise ValueError('Unsupported grouped down-projection payload layout')
  layout=payload_layout(rows,depths[:groups])
  offset,padded,_,size=layout[-1]
  if packed.numel()<offset+size:raise ValueError('Grouped down-projection payload is truncated')
  header=round_up(groups*rows*4,64)
  activation_scales=Tensor.cat(*(scales[index*rows:(index+1)*rows] for index in range(groups))).contiguous().realize()
  pieces=[activation_scales.bitcast(dtypes.uint8).pad_to((header,))]
  for offset,padded,tile,_ in layout:
    for first in range(0,rows,tile):
      count=min(tile,rows-first)
      tile_header=round_up(count*4,64)
      begin=offset+first*padded+(first//tile)*tile_header
      pieces.append(packed[begin+tile_header:begin+tile_header+count*padded])
  return Tensor.cat(*pieces).contiguous().realize(),activation_scales


def _exponent(activation:str) -> str:
  if activation=='gelu_tanh':return '(g+0.044714998453855515f*g*g*g)*-2.302208185195923f'
  if activation in ('silu','silu_div'):return 'g*-1.4426950216293335f'
  if activation=='gelu':return '(g*g)*-0.7213475108146667f'
  raise ValueError('Unsupported fused gated activation')


def _activation_value(activation:str) -> str:
  if activation=='silu_div':
    # The single-K legacy FFN divides gate directly; do not introduce an
    # independently rounded reciprocal when replacing that kernel.
    return 'float4 value=(g/(1.0f+rockchip_exp2_vector(exponent)))*u;'
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
                             activation:str, batch:int, channel_scratch:bool=False, prefetch:int=0, batched_pack:bool=False) -> str:
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
      hints=[]
      declarations=[]
      for lane in range(batch):
        loads=[]
        for base in (first,first+hidden):
          index,tile,start,count=next(f for f in active
            if f[1].first_column+f[2]<=base+begin<base+end<=f[1].first_column+f[2]+f[3])
          offset=start*tile.rows+base-tile.first_column-start
          position=(f'{start*tile.rows}+(({base-tile.first_column-start}+c)/4*{tile.rows}+row+{lane}-{tile.first_row})*4' if tile.sdk else
                    f'{offset}+((row+{lane})-{tile.first_row})*{count}+c')
          if prefetch and lane==0:
            ahead=position.replace('+c',f'+(c+{prefetch})')
            hints.append(f'__builtin_prefetch(raw[{index}]+({ahead}),0,3);')
          loads.append(f'''(__builtin_convertvector(*(const int4*)(raw[{index}]+({position})),float4)*
            (float4)sa[{tile.scale_index}][(row+{lane})-{tile.first_row}])*(*(const float4*)(sw+{base}+c))''')
        store=f'batch_values[c/4][{lane}]' if channel_scratch else f'*(float4*)(batch_values[{lane}]+c)'
        if batched_pack:
          declarations.append(f'float4 g{lane}={loads[0]},u{lane}={loads[1]},x{lane}='+_exponent(activation).replace('g',f'g{lane}')+';')
          value=(f'float4 g=g{lane},u=u{lane};float4 value=(g/(1.0f+e{lane}))*u;' if activation=='gelu_tanh' else
                 f'float4 g=g{lane},u=u{lane};float4 sigmoid=1.0f/(1.0f+e{lane});float4 value=(g*sigmoid)*u;')
        else:
          value=f'float4 g={loads[0]},u={loads[1]};float4 exponent={_exponent(activation)};{_activation_value(activation)}'
        lanes.append(f'''{{{value}{store}=value;
          uint4 bits=__builtin_bit_cast(uint4,value);batch_bad[{lane}]|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=value>=(float4)0?value:-value;batch_maximum[{lane}]=a>batch_maximum[{lane}]?a:batch_maximum[{lane}];}}''')
      # Stay within this column fragment; all four row vectors share one C4 cache line.
      hint=f'if(c+{prefetch}<{end}){{'+''.join(hints)+'}' if hints else ''
      prepare=''.join(declarations)+'float4 e0,e1,e2,e3;rockchip_exp2_batch4(x0,x1,x2,x3,&e0,&e1,&e2,&e3);' if batched_pack else ''
      loops.append(f'for(int c={begin};c<{end};c+=4) {{'+hint+prepare+''.join(lanes)+'}')
    result.append(f'if(row>={row_begin} && row+{batch}<={row_end}) {{'+''.join(loops)+'}')
  return ''.join(result)


def _row_batch_body(tiles:tuple[ProjectionTile,...], rows:int, hidden:int, first_channel:int, depth:int,
                    activation:str, batch:int, offset:int, padded:int, row_tile:int, group:int, channel_scratch:bool=False,
                    prefetch:int=0, batched_pack:bool=False) -> str:
  bounds=sorted({0,rows,*(t.first_row for t in tiles),*(t.first_row+t.rows for t in tiles)})
  safe=' || '.join(f'(row>={a} && row+{batch}<={b})' for a,b in zip(bounds,bounds[1:]))
  packing=[]
  quantizer='rockchip_quantize_bounded_vector' if batched_pack else 'rockchip_quantize_vector'
  for lane in range(batch):
    values=f'(float*)batch_values+{lane*4}' if channel_scratch else f'batch_values[{lane}]'
    packing.append(f'''{{int row=batch_start+{lane};float *values={values};
      float4 maximum=batch_maximum[{lane}];uint4 bad=batch_bad[{lane}];
      int first=row/{row_tile}*{row_tile},count={rows}-first<{row_tile}?{rows}-first:{row_tile};
      unsigned char *base=out+{offset}+first*{padded}+first/{row_tile}*{round_up(row_tile*4,64)};
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*{padded};
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      ((float*)base)[row-first]=scales[{group*rows}+row]=invalid?__builtin_nanf(""):scale;
      float inverse=1.0f/scale;int c=0;
      for(;c<{depth};c+=4){{float4 value=invalid?(float4)0:*(const float4*)(values+c{f'*{batch}' if channel_scratch else ''});
        *(char4*)(dest+c)={quantizer}(value,inverse);}}
      for(;c<{padded};c++)dest[c]=0;
    }}''')
  # Adjacent row vectors share a cache line instead of four depth-strided streams.
  scratch=f'float4 batch_values[{depth//4}][{batch}]' if channel_scratch else f'float batch_values[{batch}][{depth}]'
  return f'''if(row+{batch}<=end && row+{batch}<=active && ({safe})) {{
    int batch_start=row;{scratch} __attribute__((aligned(16)));
    float4 batch_maximum[{batch}]={{0}};uint4 batch_bad[{batch}]={{0}};
    {_batched_projection_loop(tiles,rows,hidden,first_channel,depth,activation,batch,channel_scratch,prefetch,batched_pack)}
    {''.join(packing)}row+={batch-1};continue;
  }}
      '''


@functools.cache
def _program(out:UOp, raw:tuple[UOp,...], sa:tuple[UOp,...], sw:UOp, active:UOp, scale_out:UOp,
             rows:int, hidden:int, depths:tuple[int,...], tiles:tuple[ProjectionTile,...], static_layout:bool=False,
             activation:str='gelu_tanh', pool:int|None=None, row_batch:int=1, channel_scratch:bool=False, prefetch:int=0,
             batched_pack:bool=False, input_scale:UOp|None=None, hadamard:int=0) -> UOp:
  if row_batch not in (1,2,4):raise ValueError('Unsupported raw SwiGLU row batch')
  from tinygrad.llm.kernels.rockchip_hadamard import supported as hadamard_supported,rotate_ffn_source
  if hadamard not in (0,128):raise ValueError('Unsupported INT8 Hadamard block')
  if hadamard and any(hadamard_supported(depth) for depth in depths):row_batch=1
  if prefetch not in (0,16):raise ValueError('Unsupported C4 prefetch distance')
  if not _row_batch_supported(activation,tiles) or not static_layout or rows<16:row_batch=1
  sdk_batch4=(static_layout and rows==128 and row_batch==4 and channel_scratch and (
    hidden==4864 and depths==(4096,768) and activation=='silu' and tiles==(ProjectionTile(0,128,0,9728,0,True),)
    or activation=='gelu_tanh' and tiles in _SDK_TANH_ROW_TILES))
  if not sdk_batch4:prefetch=0
  from tinygrad.llm.kernels.rockchip_cpu import ONPU_EXP2_FAST_PATH,ONPU_EXP2_SCALE_BITS,_vector_exp2_batch4_source
  batched_pack=bool(batched_pack and sdk_batch4 and ONPU_EXP2_FAST_PATH and ONPU_EXP2_SCALE_BITS)
  if batched_pack:
    global _batched_pack_traces
    _batched_pack_traces+=1
  quantizer='rockchip_quantize_bounded_vector' if batched_pack else 'rockchip_quantize_vector'
  # Valid rows have |value| <= largest, scale=max(round(largest/127),1e-8).
  # FP32 rounding keeps |value*inverse| well below127.5. Invalid rows supply0.
  # Thus nearest-even conversion cannot exceed[-127,127]; no clamp is needed.
  bounded_quant='''static __attribute__((always_inline)) char4 rockchip_quantize_bounded_vector(float4 x,float inverse) {
    #if defined(__aarch64__)
    float4 scaled=x*(float4)inverse;int4 q;
    __asm__("fcvtns %0.4s, %1.4s":"=w"(q):"w"(scaled));return __builtin_convertvector(q,char4);
    #else
    return rockchip_quantize_vector(x,inverse);
    #endif
  }''' if batched_pack else ''
  exponent=_exponent(activation)
  names={'gelu':'rockchip_int8_erf_gelu_pack','gelu_tanh':'rockchip_int8_gelu_pack',
         'silu':'rockchip_int8_swiglu_pack','silu_div':'rockchip_int8_swiglu_pack'}
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
    if input_scale is not None:
      marker=_activation_value(activation)
      projection_loop=projection_loop.replace(marker,marker+f'value=value/(*(const float4*)(ds+{first_channel}+c));')
    for first in range(0,rows,row_tile):
      count=min(row_tile,rows-first)
      base=offset+first*padded+(first//row_tile)*round_up(row_tile*4,64)
      padding.append(f'for(int i={count*4};i<{round_up(count*4,64)};i++)out[{base}+i]=0;')
    batch_body=(_row_batch_body(tiles,rows,hidden,first_channel,depth,activation,row_batch,offset,padded,row_tile,group,
                                channel_scratch,prefetch,batched_pack)
                if row_batch>1 else '')
    if input_scale is not None:
      marker=_activation_value(activation)
      batch_body=batch_body.replace(marker,marker+f'value=value/(*(const float4*)(ds+{first_channel}+c));')
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
        *(char4*)(dest+c)={quantizer}(value,inverse);}}
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
  source=_vector_exp2_source()+(_vector_exp2_batch4_source() if batched_pack else '')+_quantize_vector_source()+bounded_quant+f'''
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
  if input_scale is not None:
    source=source.replace('const float *sw,float *scales','const float *sw,const float *ds,float *scales')
    source=source.replace('const float *sw;float *scales','const float *sw,*ds;float *scales')
    source=source.replace('w->sw,w->scales','w->sw,w->ds,w->scales')
    source=source.replace('out,raw,sa,sw,scales','out,raw,sa,sw,ds,scales')
    source=source.replace('const float *sw,const int *active','const float *sw,const float *ds,const int *active')
  if hadamard:source=rotate_ffn_source(source)
  inputs=(out,*raw,*sa,sw,*((input_scale,) if input_scale is not None else ()),active,scale_out)
  sink=UOp.sink(*inputs,arg=KernelInfo(name))
  renderer=_renderer()
  info=replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,len(inputs)-1),ins=tuple(range(1,len(inputs)-1)))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _pack_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                     hidden:int, depths:tuple[int,...], active_rows:int|UOp, activation:str, pool:AttentionPool|None=None,
                     input_scale:Tensor|None=None) -> tuple[Tensor,Tensor]:
  _exponent(activation)
  if (not 1<=rows<=128 or not (512<=hidden<=12288 or hidden==13696) or hidden%32 or sum(depths)!=hidden
      or any(not 1<=depth<=4096 or depth%32 for depth in depths) or not raw or len(raw)!=len(tiles)
      or not sa or any(t.device!='CPU' or t.dtype is not dtypes.float for t in (*sa,sw)) or sw.shape!=(hidden*2,)
      or any(t.device!='CPU' or t.dtype is not dtypes.int32 for t in raw)):
    raise ValueError('Unsupported fused GELU projection layout')
  if input_scale is not None and (input_scale.device!='CPU' or input_scale.dtype is not dtypes.float or input_scale.shape!=(hidden,)):
    raise ValueError('Input scales require one matching CPU FP32 channel vector')
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
  row_batch=getenv('ONPU_SWIGLU_PREFILL_ROWS',1) if _row_batch_supported(activation,tuple(tiles)) and rows>=16 else 1
  if row_batch not in (1,2,4):raise ValueError('Unsupported raw SwiGLU row batch')
  result=Tensor.empty(sum(x[3] for x in layout),device='CPU',dtype=dtypes.uint8).custom_kernel(
    *raw,*sa,sw,*((input_scale.contiguous(),) if input_scale is not None else ()),count,
    Tensor.empty(len(depths)*rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,*args:_program(o,tuple(args[:len(raw)]),tuple(args[len(raw):len(raw)+len(sa)]),args[len(raw)+len(sa)],args[-2],args[-1],
                               rows,hidden,depths,tuple(tiles),
                               True if activation in ('silu','silu_div','gelu') else bool(getenv('ONPU_GELU_PACK_STATIC_LAYOUT',1)),
                               activation,address,row_batch,bool(ONPU_SWIGLU_CHANNEL_SCRATCH),ONPU_SWIGLU_C4_PREFETCH.value,
                               bool(ONPU_SWIGLU_BATCHED_PACK),args[-3] if input_scale is not None else None,
                               ONPU_INT8_HADAMARD.value))
  return result[0],result[-1]


def pack_gelu_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                         hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None,
                         input_scale:Tensor|None=None) -> tuple[Tensor,Tensor]:
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'gelu_tanh',pool,input_scale)


def pack_swiglu_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                           hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None,
                         input_scale:Tensor|None=None) -> tuple[Tensor,Tensor]:
  """Fuse raw SwiGLU with the down projection's independent K-group quantization."""
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'silu',pool,input_scale)


def pack_legacy_swiglu_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                                hidden:int, depths:tuple[int,...], active_rows:int|UOp, *,
                                pool:AttentionPool|None=None, input_scale:Tensor|None=None) -> tuple[Tensor,Tensor]:
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'silu_div',pool,input_scale)


def pack_erf_projection(raw:list[Tensor], sa:list[Tensor], sw:Tensor, tiles:list[ProjectionTile], rows:int,
                        hidden:int, depths:tuple[int,...], active_rows:int|UOp, *, pool:AttentionPool|None=None,
                         input_scale:Tensor|None=None) -> tuple[Tensor,Tensor]:
  """Preserve per-K FP32 maxima/scales and packed codes while fusing raw erf-GELU."""
  return _pack_projection(raw,sa,sw,tiles,rows,hidden,depths,active_rows,'gelu',pool,input_scale)


def supported_down_input_scale(down:RockchipInt8Linear) -> bool:
  parts=down.parts if isinstance(down,RockchipInt8SplitLinear) else [down]
  scale=down.input_scale
  if scale is None:return all(part.input_scale is None for part in parts)
  return (bool(ONPU_INT8_HADAMARD) and scale.device=='CPU' and scale.dtype is dtypes.float and scale.shape==(down.depth,)
          and all(part.input_scale is not None and part.input_scale.device=='CPU' and part.input_scale.dtype is dtypes.float
                  and part.input_scale.shape==(part.depth,) for part in parts))


def supported_down_bias(down:RockchipInt8Linear) -> bool:
  bias=down.bias
  return bias is None or (isinstance(bias,Tensor) and bias.device=='CPU' and bias.dtype in (dtypes.half,dtypes.float)
                         and all(isinstance(size,int) for size in bias.shape) and bias.shape==(down.columns,))


def wide_swiglu_supported(joint:RockchipLinear, down:RockchipInt8Linear) -> bool:
  """Qualified two-weight-tile prefill geometry; retain unsplit decode weights."""
  return (bool(ONPU_WIDE_RAW_SWIGLU) and isinstance(joint,RockchipInt8Linear) and isinstance(down,RockchipInt8SplitLinear)
          and (joint.depth,joint.columns,joint.tile_columns,down.depth,down.columns)==(4096,27392,24576,13696,4096)
          and joint.bias is None and down.bias is None and down.input_scale is None
          and tuple(p.depth for p in down.parts)==(4096,4096,4096,1408)
          and tuple(p.shape for p in joint.packed)==((768,128,32,32),(88,128,32,32))
          and tuple(p.shape for p in joint.scales)==((24576,),(2816,)))


class _RockchipInt8GatedFeedForward:
  activation='gelu_tanh'
  decode_enabled=False
  supports_down_bias=False
  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None):
    self.projections,self.down,self.pool=projections,down,pool
    self.combined_scales:Tensor|None = None
    if wide_swiglu_supported(projections.joint,down):
      assert isinstance(projections.joint,RockchipInt8Linear)
      self.combined_scales=Tensor.cat(*projections.joint.scales).contiguous().realize()

  def enabled(self,x:Tensor) -> bool:
    joint=self.projections.joint
    parts=self.down.parts if isinstance(self.down,RockchipInt8SplitLinear) else [self.down]
    phase=self.projections.layers[0].phase
    decoding=phase=='decode'
    wide=self.activation=='silu' and wide_swiglu_supported(joint,self.down) and self.combined_scales is not None
    return (isinstance(joint,RockchipInt8Linear) and x.device=='CPU' and x.dtype is dtypes.float and x.ndim==3 and x.shape[0]==1
      and ((self.activation in ('silu','silu_div','gelu','gelu_tanh') and self.decode_enabled and x.max_shape[1]==1)
           if decoding else 16<=x.max_shape[1]<=128)
      and ((len(joint.packed)==len(joint.scales)==1 and 512<=self.down.depth<=12288) or wide and not decoding)
      and self.down.depth%32==0
      and joint.columns==self.down.depth*2 and 1<=self.down.columns<=12288
      and joint.bias is None and (self.down.bias is None or self.supports_down_bias and supported_down_bias(self.down))
      and supported_down_input_scale(self.down)
      and bool(parts) and all(len(p.packed)==len(p.scales)==1
                             and (p.bias is None or p is self.down and self.supports_down_bias) for p in parts)
      and all(getattr(layer,'decode' if decoding else 'prefill')=='on' for layer in (*self.projections.layers,self.down)))

  def __call__(self,x:Tensor) -> Tensor:
    split=getattr(self,'split_prefill',None)
    if split is not None and split.enabled(x) and self.enabled(x):return split(x)
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
      sdk=((self.activation=='gelu_tanh' and getattr(self,'sdk_prefill',False)
            and (rows,joint.columns,k) in SDK_TANH_PREFILL_SHAPES
            or self.activation=='silu' and (rows,joint.columns,k) in SDK_SILU_PREFILL_SHAPES
            or self.activation=='silu_div' and (rows,joint.columns,k) in SDK_SILU_DIV_PREFILL_SHAPES
            or self.activation=='gelu' and (rows,joint.columns,k) in SDK_ERF_GELU_PREFILL_SHAPES)
           and bool(getenv('ONPU_INT8_SDK_PREFILL',0)))
      payload,scale=quantize(joint.scaled_input(part),k,sdk_prefill=sdk)
      scale_index=len(sa)
      sa.append(scale)
      for wi,weight in enumerate(joint.packed):
        tile_columns=joint.columns if len(joint.packed)==1 else joint.tile_columns
        base=wi*tile_columns
        width=min(tile_columns,joint.columns-base)
        for column in range(0,width,24576 if rows==1 else 12288):
          count=min(24576 if rows==1 else 12288,width-column)
          raw.append(raw_int8_matmul(payload,weight,rows,count,column,sdk_prefill=sdk))
          tiles.append(ProjectionTile(first,rows,base+column,count,scale_index,sdk))
    parts=self.down.parts if isinstance(self.down,RockchipInt8SplitLinear) else [self.down]
    depths=tuple(part.depth for part in parts)
    pack={'silu':pack_swiglu_projection,'silu_div':pack_legacy_swiglu_projection,
          'gelu':pack_erf_projection,'gelu_tanh':pack_gelu_projection}[self.activation]
    weight_scales=joint.scales[0] if self.combined_scales is None else self.combined_scales
    packed,scales=pack(raw,sa,weight_scales,tiles,capacity,self.down.depth,depths,math.prod(x.shape[:-1]),
                       pool=self.pool,input_scale=self.down.input_scale)
    layout=payload_layout(capacity,depths)
    grouped_down=self.down if isinstance(self.down,RockchipInt8SplitLinear) else None
    grouped_payload:tuple[Tensor,Tensor]|None=None
    # Decode shares one activation row across equal-depth down groups; the grouped
    # PC chain replaces the three leading K=4096 submissions and the tail stays separate.
    if (ONPU_GROUPED_DECODE_DOWN and capacity==1 and grouped_down is not None
        and grouped_down.grouped_weight is not None and grouped_down.grouped_scale is not None
        and grouped_down.grouped_count==3 and grouped_down.columns==4096 and len(parts)>=4
        and tuple(part.depth for part in parts[:3])==(4096,4096,4096)
        and grouped_down.grouped_weight.shape==(3,grouped_down.columns//32,128,32,32)
        and grouped_down.grouped_scale.shape==(3,grouped_down.columns)):
      grouped_payload=grouped_down_payload(packed,scales,capacity,depths,grouped_down.grouped_count)
    shared_input=((capacity>=16 or capacity==1 and self.activation=='gelu_tanh' and getenv('ONPU_TANH_SINGLE_UPLOAD',0))
                  and getenv('ONPU_FFN_PACKED_UPLOAD',0) and len({layer.packed[0].device for layer in parts})==1)
    if shared_input:
      # All K groups share one fully written payload. Upload its allocation
      # once, then retain the original per-group/per-row scale boundaries.
      packed=packed.to(parts[0].packed[0].device)
    outputs:list[Tensor]=[]
    first_part=0
    if grouped_payload is not None:
      assert grouped_down is not None and grouped_down.grouped_weight is not None and grouped_down.grouped_scale is not None
      grouped=grouped_raw_matmul(grouped_payload[0],grouped_down.grouped_weight,capacity,grouped_down.columns)
      grouped_values=grouped_dequantize(grouped,grouped_payload[1],grouped_down.grouped_scale,grouped_down.grouped_count,
                                        capacity,grouped_down.columns,layout[0][1])
      outputs.extend(grouped_values[index] for index in range(grouped_down.grouped_count))
      first_part=grouped_down.grouped_count
    for index in range(first_part,len(parts)):
      layer=parts[index]
      offset,padded,tile,_=layout[index]
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
    result=result.cast(x.dtype)
    # Match SplitLinear: sum K groups, cast, then apply the original bias once.
    # Keep this a logical-row operation; padded rows are not bias-filled/marked zero.
    return result if self.down.bias is None else result+self.down.bias


class RockchipInt8GeluFeedForward(_RockchipInt8GatedFeedForward):
  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None,
               decode:bool=False,sdk_prefill:bool=False):
    if type(decode) is not bool or type(sdk_prefill) is not bool:raise ValueError('Raw GELU phase selection must be a bool')
    if sdk_prefill and decode:raise ValueError('SDK GELU prefill requires the baseline decode path')
    if sdk_prefill and (projections.joint.depth!=1536 or down.depth not in (6144,12288)):
      raise ValueError('Unqualified SDK tanh-GELU geometry')
    super().__init__(projections,down,pool=pool)
    self.decode_enabled,self.sdk_prefill=decode,sdk_prefill


class RockchipInt8SwiGLUFeedForward(_RockchipInt8GatedFeedForward):
  activation='silu'
  supports_down_bias=True

  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None,decode:bool=False,
               legacy_division:bool=False):
    if type(decode) is not bool:raise ValueError('Raw SwiGLU decode selection must be a bool')
    if type(legacy_division) is not bool:raise ValueError('Legacy SwiGLU division selection must be a bool')
    super().__init__(projections,down,pool=pool)
    self.decode_enabled=decode
    if legacy_division:self.activation='silu_div'
    from tinygrad.llm.kernels.rockchip_swiglu_ksplit import prepare_split_prefill
    self.split_prefill=prepare_split_prefill(self) if ONPU_SWIGLU_K_SPLIT else None


class RockchipInt8ErfGeluFeedForward(_RockchipInt8GatedFeedForward):
  activation='gelu'

  def __init__(self,projections:RockchipProjections,down:RockchipInt8Linear, *, pool:AttentionPool|None=None,decode:bool=False):
    if type(decode) is not bool:raise ValueError('Raw erf GELU decode selection must be a bool')
    super().__init__(projections,down,pool=pool)
    self.decode_enabled=decode
