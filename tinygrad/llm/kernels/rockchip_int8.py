"""Opt-in dynamic per-token/per-channel INT8 matrices with FP32 dequantization."""
from __future__ import annotations
import copy, functools, math, struct
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar, Target, ceildiv, getenv, round_up
from tinygrad.llm.kernels.rockchip import RockchipLinear, RockchipFeedForward, RockchipProjections, RockchipPreparedQKV
from tinygrad.uop.ops import resolve
from tinygrad.llm.kernels.rockchip_cpu import _renderer, ONPU_NORM_QUANT, ONPU_NORM_QUANT_DECODE
from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_PREFILL_CPU_TUNING, shared_program
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.runtime.support.rockchip_int8 import padded_depth, supported_depth, sdk_prefill_supported
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_INT8_GROUPED_GATE_UP = ContextVar('ONPU_INT8_GROUPED_GATE_UP',0)
ONPU_GROUPED_DECODE_DOWN = ContextVar('ONPU_GROUPED_DECODE_DOWN',0)
ONPU_INT8_REUSE_ACTIVATION = ContextVar('ONPU_INT8_REUSE_ACTIVATION',0)

def _quantize_vector_source() -> str:
  return '''typedef signed char char4 __attribute__((aligned(1),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 rockchip_round_even(float4 x) {
    #if defined(__aarch64__)
    float4 result;__asm__("frintn %0.4s, %1.4s":"=w"(result):"w"(x));return result;
    #else
    return (float4){__builtin_nearbyintf(x[0]),__builtin_nearbyintf(x[1]),__builtin_nearbyintf(x[2]),__builtin_nearbyintf(x[3])};
    #endif
  }
  static __attribute__((always_inline)) char4 rockchip_quantize_vector(float4 x,float inverse) {
    float4 q=rockchip_round_even(x*(float4)inverse);
    q=q>(float4)127?(float4)127:q;q=q<(float4)-127?(float4)-127:q;
    return __builtin_convertvector(q,char4);
  }'''


def _sdk_quantize_part(rows:int, depth:int, padded:int, header:int) -> str:
  return f'''static void part(unsigned char *raw,const float *input,float *scale_out,int begin,int end) {{
    float *scales=(float*)raw;signed char *output=(signed char*)(raw+{header});
    for(int first=begin;first<end;first+=4) {{
      int count=end-first<4?end-first:4;float inverse[4];int invalid[4];
      for(int r=0;r<count;r++) {{
        const float *x=input+(first+r)*{depth};float4 maximum={{0}};uint4 bad={{0}};int i=0;
        for(;i+3<{depth};i+=4) {{
          float4 v=*(const float4*)(x+i);uint4 bits=*(const uint4*)(x+i);
          bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
          float4 a=v>=(float4)0?v:-v;maximum=a>maximum?a:maximum;
        }}
        float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
        invalid[r]=__builtin_reduce_or(bad)!=0;
        for(;i<{depth};i++){{unsigned bits=*((const unsigned*)(x+i));
          invalid[r]|=(bits&0x7f800000)==0x7f800000;float a=__builtin_fabsf(x[i]);if(a>largest)largest=a;}}
        float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
        scales[first+r]=scale_out[first+r]=invalid[r]?__builtin_nanf(""):scale;inverse[r]=1.0f/scale;
      }}
      for(int c=0;c<{padded};c+=16)for(int r=0;r<count;r++) {{
        const float *x=input+(first+r)*{depth};signed char *dest=output+(c/16*{rows}+first+r)*16;
        for(int j=0;j<16;j+=4) {{
          if(c+j+3<{depth}) {{float4 value=invalid[r]?(float4)0:*(const float4*)(x+c+j);
            *(char4*)(dest+j)=rockchip_quantize_vector(value,inverse[r]);}}
          else for(int lane=0;lane<4;lane++) {{int i=c+j+lane;float q=0;
            if(i<{depth} && !invalid[r])q=__builtin_nearbyintf(x[i]*inverse[r]);
            if(q>127)q=127;if(q< -127)q=-127;dest[j+lane]=(signed char)q;
          }}
        }}
      }}
    }}
  }}'''


@shared_program
@functools.cache
def _quantize(out:UOp, inp:UOp, scale_out:UOp, rows:int, depth:int, padded:int, sdk_prefill:bool=False) -> UOp:
  header = round_up(rows*4,64)
  index = f'(i/16*{rows}+row)*16+i%16' if sdk_prefill else f'row*{padded}+i'
  parallel = rows*depth>=32768
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {unsigned char *raw;const float *input;float *scales;int begin,end;} Work;
  static void *worker(void *p){Work *w=p;part(w->raw,w->input,w->scales,w->begin,w->end);return (void*)0;}''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{raw,input,scale_out,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(raw,input,scale_out,work[i].begin,work[i].end);}}
    part(raw,input,scale_out,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(raw,input,scale_out,0,{rows});'
  part = f'''static void part(unsigned char *raw,const float *input,float *scale_out,int begin,int end) {{
    float *scales=(float*)raw;signed char *output=(signed char*)(raw+{header});
    for(int row=begin;row<end;row++) {{
      const float *x=input+row*{depth};float4 maximum={{0}};uint4 bad={{0}};int i=0;
      for(;i+3<{depth};i+=4) {{
        float4 v=*(const float4*)(x+i);uint4 bits=*(const uint4*)(x+i);
        bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=v>=(float4)0?v:-v;maximum=a>maximum?a:maximum;
      }}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;
      for(;i<{depth};i++){{unsigned bits=*((const unsigned*)(x+i));
        invalid|=(bits&0x7f800000)==0x7f800000;float a=__builtin_fabsf(x[i]);if(a>largest)largest=a;}}
      float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      scales[row]=scale_out[row]=invalid?__builtin_nanf(""):scale;float inverse=1.0f/scale;
      for(i=0;i+3<{depth};i+=4){{float4 value=invalid?(float4)0:*(const float4*)(x+i);
        *(char4*)(output+{index})=rockchip_quantize_vector(value,inverse);}}
      for(;i<{depth};i++){{float q=invalid?0:__builtin_nearbyintf(x[i]*inverse);
        if(q>127)q=127;if(q< -127)q=-127;output[{index}]=(signed char)q;}}
      for(;i<{padded};i++)output[{index}]=0;
    }}
  }}'''
  source = f'''typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef unsigned uint4 __attribute__((aligned(4),ext_vector_type(4)));
  {_quantize_vector_source()}
  {_sdk_quantize_part(rows,depth,padded,header) if sdk_prefill else part}
  {declarations}
  void rockchip_quantize(unsigned char *raw,const float *input,float *scale_out) {{
    for(int i={rows*4};i<{header};i++)raw[i]=0;{dispatch}
  }}'''
  inputs = (out,inp,scale_out)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_quantize'+('_sdk' if sdk_prefill else '')))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,2),ins=(1,))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def quantize(x:Tensor, padded:int, *, sdk_prefill:bool=False, weight:bool=False) -> tuple[Tensor,Tensor]:
  if x.ndim!=2 or x.device!='CPU' or not isinstance(x.shape[0],int) or not isinstance(x.shape[1],int):
    raise ValueError('INT8 quantization requires a static CPU matrix')
  rows,depth = x.shape
  assert isinstance(rows,int) and isinstance(depth,int)
  if not 1<=rows<=24576 or not 1<=depth<=padded or not supported_depth(padded): raise ValueError('Unsupported INT8 quantization geometry')
  if type(sdk_prefill) is not bool:raise ValueError('SDK INT8 quantization layout must be a bool')
  from tinygrad.llm.kernels.rockchip_hadamard import enabled,quantize as hadamard_quantize
  if type(weight) is not bool:raise ValueError('Weight quantization selection must be a bool')
  if enabled(depth):
    if sdk_prefill or ONPU_NORM_QUANT:raise ValueError('Hadamard quantization requires ordinary layout and separate CPU normalization')
    return hadamard_quantize(x,padded,weight=weight)
  if ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE:
    from tinygrad.llm.kernels.rockchip_norm_quant import try_quantize
    if (fused:=try_quantize(x,padded,sdk_prefill=sdk_prefill)) is not None:return fused
  result = Tensor.empty(round_up(rows*4,64)+rows*padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
    x.float().contiguous(),Tensor.empty(rows,device='CPU',dtype=dtypes.float),fxn=lambda o,a,s:_quantize(o,a,s,rows,depth,padded,sdk_prefill))
  return result[0],result[2]


@functools.cache
def _matmul(out:UOp, lhs:UOp, rhs:UOp, m:int, n:int, k:int, column_offset:int, cores:int, sdk_prefill:bool=False,
            payload_offset:int=0) -> UOp:
  sink = UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_int8_{m}_{n}_{k}'+('_sdk' if sdk_prefill else '')))
  info = replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),UOp(Ops.SOURCE,arg=f'INT8 {m}x{k}x{n}'),
    UOp(Ops.BINARY,arg=struct.pack('<10I' if payload_offset else '<9I' if sdk_prefill else '<8I',
                                 m,n,k,cores,column_offset*k,round_up(m*4,64),0,1,
                                 *((int(sdk_prefill),payload_offset) if payload_offset else (1,) if sdk_prefill else ())))))


@shared_program
@functools.cache
def _dequantize_tiles(out:UOp, raw:tuple[UOp,...], sa:tuple[UOp,...], sw:UOp, counts:tuple[int,...],
                      columns:int, n:int, offset:int=0, sdk_tiles:tuple[bool,...]=()) -> UOp:
  rows = sum(counts)
  tile_rows = counts[0]
  width = round_up(ceildiv(n,3),32)
  parts = [(start,min(width,n-start)) for start in range(0,n,width)]
  if len(parts)!=3: parts = [(0,n)]
  loops = []
  layouts = ','.join(str(int(flag)) for flag in (sdk_tiles or (False,)*len(counts)))
  for start,count in parts:
    used = min(count,columns-start)
    if used<=0: continue
    if any(sdk_tiles):
      first_row = 0
      for tile,(m,blocked) in enumerate(zip(counts,sdk_tiles)):
        if blocked:
          loops.append(f'''{{int first=begin>{first_row}?begin-{first_row}:0;
            int last=end<{first_row+m}?end-{first_row}:{m};
            const int *source=raw[{tile}]+{start*m};const float *scales=sw+{offset+start};
            for(int row=first;row<last;row+=4) {{int active=last-row<4?last-row:4;int c=0;
              for(;c+3<{used};c+=4) {{float4 w=*(const float4*)(scales+c);
                for(int r=0;r<active;r++) {{
                  float4 value=__builtin_convertvector(*(const int4*)(source+(c/4*{m}+row+r)*4),float4);
                  *(float4*)(out+({first_row}+row+r)*{columns}+{start}+c)=(value*(float4)sa[{tile}][row+r])*w;
                }}
              }}
              for(;c<{used};c++)for(int r=0;r<active;r++) {{
                int index=(c/4*{m}+row+r)*4+c%4;
                out[({first_row}+row+r)*{columns}+{start}+c]=((float)source[index]*sa[{tile}][row+r])*scales[c];
              }}
            }}
          }}''')
        else:
          loops.append(f'''for(int row=begin>{first_row}?begin:{first_row};row<end && row<{first_row+m};row++) {{
            const int *source=raw[{tile}]+{start*m}+(row-{first_row})*{count};
            float *dest=out+row*{columns}+{start};const float *scales=sw+{offset+start};float a=sa[{tile}][row-{first_row}];int c=0;
            for(;c+3<{used};c+=4){{float4 value=__builtin_convertvector(*(const int4*)(source+c),float4);
              *(float4*)(dest+c)=(value*(float4)a)*(*(const float4*)(scales+c));}}
            for(;c<{used};c++)dest[c]=((float)source[c]*a)*scales[c];
          }}''')
        first_row += m
      continue
    loops.append(f'''for(int row=begin;row<end;row++) {{
      int tile=row/{tile_rows},local=row%{tile_rows},m=tile=={len(counts)-1}?{counts[-1]}:{tile_rows};
      const int *source=raw[tile]+{start}*m;const int blocked[{len(counts)}]={{{layouts}}};
      float *dest=out+row*{columns}+{start};
      const float *scales=sw+{offset+start};float a=sa[tile][local];int c=0;
      for(;c+3<{used};c+=4){{int index=blocked[tile]?(c/4*m+local)*4:local*{count}+c;
        float4 value=__builtin_convertvector(*(const int4*)(source+index),float4);
        *(float4*)(dest+c)=(value*(float4)a)*(*(const float4*)(scales+c));}}
      for(;c<{used};c++){{int index=blocked[tile]?(c/4*m+local)*4+c%4:local*{count}+c;
        dest[c]=((float)source[index]*a)*scales[c];}}
    }}''')
  parallel = rows*columns>=131072
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {float *out;const int *const *raw;const float *const *sa;const float *sw;int begin,end;} Work;
  static void *worker(void *p){Work *w=p;part(w->out,w->raw,w->sa,w->sw,w->begin,w->end);return (void*)0;}''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,raw,sa,sw,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(out,raw,sa,sw,work[i].begin,work[i].end);}}
    part(out,raw,sa,sw,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(out,raw,sa,sw,0,{rows});'
  parameters=','.join([*(f'const int *r{i}' for i in range(len(raw))),*(f'const float *a{i}' for i in range(len(sa))),'const float *sw'])
  initialize=f'''const int *raw[{len(raw)}]={{{','.join(f'r{i}' for i in range(len(raw)))}}};
    const float *sa[{len(sa)}]={{{','.join(f'a{i}' for i in range(len(sa)))}}};'''
  source = f'''typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
  static void part(float *out,const int *const *raw,const float *const *sa,const float *sw,int begin,int end){{{''.join(loops)}}}
  {declarations}
  void rockchip_dequantize(float *out,{parameters}){{{initialize}{dispatch}}}'''
  inputs = (out,*raw,*sa,sw)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_dequantize'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def _dequantize(out:UOp, raw:UOp, sa:UOp, sw:UOp, rows:int, columns:int, n:int, offset:int, sdk_prefill:bool=False) -> UOp:
  return _dequantize_tiles(out,(raw,),(sa,),sw,(rows,),columns,n,offset,(sdk_prefill,))


def dequantize_tiles(raw:list[Tensor], sa:list[Tensor], sw:Tensor, counts:list[int], columns:int, *, sdk_tiles:tuple[bool,...]=()) -> Tensor:
  if (not raw or len(raw)!=len(sa) or len(raw)!=len(counts) or any(n!=counts[0] for n in counts[:-1])
      or any(n<1 for n in counts) or counts[-1]>counts[0] or columns<1
      or sw.device!='CPU' or sw.dtype is not dtypes.float or sw.numel()<columns
      or any(r.device!='CPU' or r.dtype is not dtypes.int32 or r.numel()!=m*round_up(columns,32) for r,m in zip(raw,counts))
      or any(s.device!='CPU' or s.dtype is not dtypes.float or s.shape!=(m,) for s,m in zip(sa,counts))
      or sdk_tiles and (len(sdk_tiles)!=len(raw) or any(type(flag) is not bool for flag in sdk_tiles))):
    raise ValueError('Invalid INT8 output tiles')
  count=len(raw)
  return Tensor.empty(sum(counts),columns,device='CPU',dtype=dtypes.float).custom_kernel(*raw,*sa,sw,
    fxn=lambda o,*args:_dequantize_tiles(o,tuple(args[:count]),tuple(args[count:count*2]),args[-1],
                                       tuple(counts),columns,round_up(columns,32),sdk_tiles=sdk_tiles))[0]


def raw_int8_matmul(payload:Tensor, weight:Tensor, rows:int, columns:int, column_offset:int=0, *,
                    sdk_prefill:bool=False, cpu_output:bool=True, payload_offset:int|None=None) -> Tensor:
  k = weight.shape[1]*32
  n = round_up(columns,32)
  if (not isinstance(k,int) or weight.ndim!=4 or weight.shape[2:]!=(32,32) or weight.dtype is not dtypes.int8
      or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP'
      or not isinstance(payload.device,str) or payload.device!='CPU' and payload.device.split(':')[0]!='ROCKCHIP'
      or payload.dtype is not dtypes.uint8
      or not 1<=rows<=(128 if k<=2048 else 64) or columns<1 or column_offset<0 or column_offset%32
      or column_offset+n>weight.shape[0]*32 or n>(24576 if rows==1 else 12288)
      or (payload_offset is None and payload.numel()!=round_up(rows*4,64)+rows*k)):
    raise ValueError('Unsupported raw INT8 operands')
  if payload_offset is not None and (type(payload_offset) is not int or payload_offset<0 or payload_offset>=2**32 or payload_offset%64
      or payload_offset+round_up(rows*4,64)+rows*k>payload.numel()):raise ValueError('Invalid packed INT8 payload offset')
  if type(sdk_prefill) is not bool or sdk_prefill and not sdk_prefill_supported(rows,n,k):
    raise ValueError('Unsupported SDK INT8 matrix geometry')
  if type(cpu_output) is not bool:raise ValueError('CPU output selection must be a bool')
  # Core mode: RKLLM issues three SINGLE-core jobs from three threads, while
  # oNPU issues one three-core job. Measured: 14078/14112 of RKLLM's decode
  # groups have the three cores overlapping, core_mask in {1,2,4}. Expose the
  # core count so the scheduling difference can be A/B'd.
  cores = int(getenv('ONPU_INT8_DECODE_CORES',0)) or 3
  if cores not in (1,3): raise ValueError('INT8 core count must be 1 or 3')
  result = Tensor.empty(rows*n,device=weight.device,dtype=dtypes.int32).custom_kernel(payload.to(weight.device),weight,
    fxn=lambda o,a,b:_matmul(o,a,b,rows,n,k,column_offset,cores,sdk_prefill,payload_offset or 0))[0]
  return result.to('CPU') if cpu_output else result


def packed_int8_matmul(x:Tensor, weight:Tensor, scales:Tensor, columns:int, column_offset:int=0) -> Tensor:
  if (x.ndim!=2 or x.device!='CPU' or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP' or weight.dtype is not dtypes.int8
      or weight.ndim!=4 or weight.shape[2:]!=(32,32) or scales.device!='CPU' or scales.dtype is not dtypes.float):
    raise ValueError('Unsupported INT8 matrix layout')
  rows,depth = x.shape
  k = weight.shape[1]*32
  if not isinstance(k,int): raise ValueError('INT8 weights require static geometry')
  n = round_up(columns,32)
  if (not isinstance(rows,int) or not 1<=rows<=(128 if k<=2048 else 64) or not 0<depth<=k or columns<1 or column_offset<0 or column_offset%32
      or column_offset+n>weight.shape[0]*32 or n>(24576 if rows==1 else 12288)):
    raise ValueError('Unsupported INT8 matrix geometry')
  sdk = bool(getenv('ONPU_INT8_SDK_PREFILL',0)) and sdk_prefill_supported(rows,n,k)
  payload,activation_scales = quantize(x,k,sdk_prefill=sdk)
  out = raw_int8_matmul(payload,weight,rows,columns,column_offset,sdk_prefill=sdk)
  return Tensor.empty(rows,columns,device='CPU',dtype=dtypes.float).custom_kernel(out,activation_scales,scales,
    fxn=lambda o,a,b,c:_dequantize(o,a,b,c,rows,columns,n,column_offset,sdk))[0]


def _k_split_parts(padded:int) -> tuple[tuple[int,int],...]:
  """Split a padded INT8 depth into qualified K<=2048 tasks."""
  if padded<=2048:return ((0,padded),)
  return tuple((start,min(2048,padded-start)) for start in range(0,padded,2048))


def _split_payload(payload:Tensor, rows:int, padded:int, parts:tuple[tuple[int,int],...]) -> list[Tensor]:
  """Return one valid INT8 payload per K part, reusing the original per-row scales."""
  header = round_up(rows*4,64)
  data = payload[header:].reshape(rows,padded)
  result = []
  for start,count in parts:
    result.append(Tensor.cat(payload[:header],data[:,start:start+count].contiguous().reshape(-1)).contiguous())
  return result


class RockchipInt8Linear(RockchipLinear):
  def __init__(self, *args):
    super().__init__(*args)
    if self.weight.device!='CPU': raise ValueError('INT8 acceleration requires CPU weights')
    self.scales:list[Tensor] = []
    # Dual packing: present only when decode_tile_columns differs from tile_columns.
    self.decode_packed:list[Tensor] = []
    self.decode_scales:list[Tensor] = []
    self.k_parts:list[tuple[Tensor,...]] = []
    self._prepared = False
    self.input_scale:Tensor|None = None
    self._grouped_chunks:list[tuple[Tensor,Tensor]] = []
    self._grouped_first = 0

  def scaled_input(self, x:Tensor) -> Tensor: return x if self.input_scale is None else x.float()/self.input_scale

  def prepare(self):
    if self.shared_projection is not None:return
    if self._prepared: return
    k = padded_depth(self.depth)
    weight = self.weight
    if self.input_scale is not None:
      scale = self.input_scale
      if scale.device!='CPU' or scale.dtype is not dtypes.float or scale.shape!=(self.depth,):
        raise ValueError('INT8 input channel scales require a matching CPU FP32 vector')
      values = scale.tolist()
      assert isinstance(values,list)
      if any(not 0<float(value)<float('inf') for value in values): raise ValueError('INT8 input channel scales must be finite and positive')
      weight = weight.float()*scale
    def pack(width:int) -> tuple[list[Tensor],list[Tensor]]:
      packed_out:list[Tensor] = []
      scales_out:list[Tensor] = []
      for start in range(0,self.columns,width):
        raw,scales = quantize(weight[start:start+width],k)
        scales = scales.contiguous().realize()
        values = scales.tolist()
        assert isinstance(values,list)
        if any(not 0<s<float('inf') for s in values): raise ValueError('INT8 weights must be finite')
        count = len(values)
        n = round_up(count,32)
        codes = raw[round_up(count*4,64):].bitcast(dtypes.int8).reshape(count,k).pad(((0,n-count),(0,0)))
        from tinygrad.runtime.ops_rockchip import int8_weight_device
        device = int8_weight_device(n*k)
        packed_out.append(codes.reshape(n//32,32,k//32,32).permute(0,2,1,3).contiguous().to(device).realize())
        scales_out.append(scales)
      return packed_out, scales_out

    self.packed, self.scales = pack(self.tile_columns)
    if self.decode_tile_columns!=self.tile_columns:
      self.decode_packed, self.decode_scales = pack(self.decode_tile_columns)
    self._grouped_chunks.clear()
    if (ONPU_INT8_GROUPED_GATE_UP and self.input_scale is None and self.bias is None
        and len(self.packed)==2 and self.packed[0].shape[1]*32==padded_depth(self.depth)
        and self.packed[1].shape[1]*32==padded_depth(self.depth)):
      first=self.packed[0].shape[0]*32
      blocks_total=self.packed[0].shape[0]
      assert isinstance(first,int) and isinstance(blocks_total,int)
      if first%12288==0 and 0<first<self.columns and self.packed[1].shape[0]*32==self.columns-first:
        blocks=4096//32
        for offset in range(0,blocks_total,3*blocks):
          weight=self.packed[0][offset:offset+3*blocks].reshape(3,blocks,self.packed[0].shape[1],32,32)
          scales=self.scales[0][offset*32:(offset+3*blocks)*32].reshape(3,4096)
          self._grouped_chunks.append((weight,scales))
        self._grouped_first=first
    self._prepared = True

  def release_packed(self):
    super().release_packed()
    self.scales.clear()
    self.k_parts.clear()
    self._grouped_chunks.clear()
    self._grouped_first = 0
    self._prepared = False

  def raw_projection(self, x:Tensor, *, sdk_prefill:bool=False, defer_bias:bool=False) -> tuple[list[Tensor],list[Tensor],list[int]]:
    if type(defer_bias) is not bool:raise ValueError('Deferred raw projection bias must be a bool')
    if x.device!='CPU' or not (len(self.packed)==1 or len(self.k_parts)==1) or self.columns>12288 or self.bias is not None and not defer_bias:
      raise ValueError('Unsupported raw INT8 projection')
    dense = capacity_input(x).reshape(-1,self.depth)
    k = padded_depth(self.depth)
    assert isinstance(k,int)
    tile_rows = 128 if len(self.k_parts)==1 or self.depth<=2048 else 64
    raw,scales,counts = [],[],[]
    for start in range(0,dense.shape[0],tile_rows):
      part = dense[start:start+tile_rows]
      count = part.shape[0]
      assert isinstance(count,int)
      sdk = sdk_prefill and not self.k_parts and sdk_prefill_supported(count,round_up(self.columns,32),k)
      payload,scale = quantize(self.scaled_input(part),k,sdk_prefill=sdk)
      if self.k_parts:
        payloads = _split_payload(payload,count,k,_k_split_parts(k))
        raw.append(self._sum_k_parts(payloads,count,0,0,self.columns))
      else:
        raw.append(raw_int8_matmul(payload,self.packed[0],count,self.columns,sdk_prefill=sdk))
      scales.append(scale)
      counts.append(count)
    return raw,scales,counts

  def _sum_k_parts(self, payloads:list[Tensor], rows:int, tile:int=0, column:int=0, columns:int|None=None) -> Tensor:
    if tile>=len(self.k_parts) or len(payloads)!=len(self.k_parts[tile]):raise ValueError('INT8 K-split layout changed')
    width = min(self.tile_columns,self.columns-tile*self.tile_columns)-column if columns is None else columns
    total:Tensor|None = None
    for payload,weight in zip(payloads,self.k_parts[tile]):
      value = raw_int8_matmul(payload,weight,rows,width,column_offset=column)
      total = value if total is None else total+value
    assert total is not None
    return total

  def _phase_tiles(self, phase:str|None) -> tuple[int,list[Tensor],list[Tensor]]:
    if phase=='decode' and self.decode_tile_columns!=self.tile_columns:
      return self.decode_tile_columns, self.decode_packed, self.decode_scales
    return self.tile_columns, self.packed, self.scales

  def npu(self, x:Tensor) -> Tensor:
    if self.shared_projection is not None:return self.shared_npu(x)
    if x.device!='CPU': raise ValueError('INT8 acceleration requires CPU activations')
    shape = x.shape
    dense = capacity_input(x).reshape(-1,self.depth)
    if (self._grouped_chunks and not self.k_parts and x.ndim==3 and x.shape[0]==1 and x.dtype is dtypes.float
        and isinstance(x.max_shape[1],int) and x.max_shape[1]>1):
      capacity=dense.shape[0]
      if isinstance(capacity,int) and 1<=capacity<=128:
        from tinygrad.llm.kernels.rockchip_grouped_int8 import shared_matmul
        pieces=[]
        for weight,scales in self._grouped_chunks:
          grouped=shared_matmul(dense,weight,scales,4096).realize()
          pieces.append(Tensor.cat(grouped[0],grouped[1],grouped[2],dim=1))
        tail_width=self.columns-self._grouped_first
        tail=[packed_int8_matmul(self.scaled_input(dense[start:start+64,self._grouped_first:self.columns]),
                                 self.packed[1],self.scales[1],tail_width) for start in range(0,capacity,64)]
        pieces.append(Tensor.cat(*tail,dim=0))
        out=Tensor.cat(*pieces,dim=1).reshape(*x.max_shape[:-1],self.columns)
        out=out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
        return out if self.bias is None else out+self.bias
    rows = []
    if self.k_parts:
      k = padded_depth(self.depth)
      parts = _k_split_parts(k)
      tile_rows = 128
      for start in range(0,dense.shape[0],tile_rows):
        part = dense[start:start+tile_rows]
        count = part.shape[0]
        assert isinstance(count,int)
        payload,scale = quantize(self.scaled_input(part),k)
        payloads = _split_payload(payload,count,k,parts)
        columns = []
        for tile in range(len(self.k_parts)):
          width = min(self.tile_columns,self.columns-tile*self.tile_columns)
          limit = 24576 if count==1 else 12288
          for col in range(0,width,limit):
            chunk = min(limit,width-col)
            raw = self._sum_k_parts(payloads,count,tile,col,chunk)
            columns.append(dequantize_tiles([raw],[scale],self.scales[tile][col:col+chunk],[count],chunk))
        rows.append(Tensor.cat(*columns,dim=1) if len(columns)>1 else columns[0])
      out = Tensor.cat(*rows,dim=0).reshape(*x.max_shape[:-1],self.columns)
      out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
      return out if self.bias is None else out+self.bias
    phase = self.phase or ('decode' if resolve(math.prod(x.shape[:-1])==1,default=False) else 'prefill')
    tile_columns, packed, scales = self._phase_tiles(phase)
    tile_rows = 128 if padded_depth(self.depth)<=2048 else 64
    if len(packed)==1 and self.columns<=12288 and dense.shape[0]>1 and self.bias is None:
      sdk_enabled = bool(getenv('ONPU_INT8_SDK_PREFILL',0))
      raw_tiles,tile_scales,counts = self.raw_projection(x,sdk_prefill=sdk_enabled)
      sdk_tiles = tuple(sdk_enabled and sdk_prefill_supported(m,round_up(self.columns,32),padded_depth(self.depth)) for m in counts)
      out = dequantize_tiles(raw_tiles,tile_scales,scales[0],counts,self.columns,sdk_tiles=sdk_tiles)
      out = out.reshape(*x.max_shape[:-1],self.columns)
      out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
      return out if self.bias is None else out+self.bias
    for start in range(0,dense.shape[0],tile_rows):
      part = dense[start:start+tile_rows]
      count = part.shape[0]
      assert isinstance(count,int)
      columns = []
      reuse_activation = bool(ONPU_INT8_REUSE_ACTIVATION) and count==1 and len(packed)>1
      if reuse_activation:
        payload,activation_scales = quantize(self.scaled_input(part),padded_depth(self.depth))
      for index,(weight,ws) in enumerate(zip(packed,scales)):
        width = min(tile_columns,self.columns-index*tile_columns)
        limit = 24576 if count==1 else 12288
        for col in range(0,width,limit):
          chunk = min(limit,width-col)
          if reuse_activation:
            raw = raw_int8_matmul(payload,weight,count,chunk,column_offset=col)
            columns.append(dequantize_tiles([raw],[activation_scales],ws[col:col+chunk],[count],chunk))
          else:
            columns.append(packed_int8_matmul(self.scaled_input(part),weight,ws,chunk,col))
      rows.append(Tensor.cat(*columns,dim=1))
    out = Tensor.cat(*rows,dim=0).reshape(*x.max_shape[:-1],self.columns)
    out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
    return out if self.bias is None else out+self.bias


class RockchipInt8SplitLinear(RockchipInt8Linear):
  """Blockwise INT8 projections beyond one NPU task's input-width limit."""
  def __init__(self, *args):
    super().__init__(*args)
    self.parts:list[RockchipInt8Linear] = []
    self.grouped_weight:Tensor|None = None
    self.grouped_scale:Tensor|None = None
    self.grouped_count = 0

  def prepare(self):
    if self.shared_projection is not None:return
    if self._prepared:return
    if not 4096<self.depth<=32768:raise ValueError('Unsupported split INT8 projection depth')
    for start in range(0,self.depth,4096):
      source = copy.copy(self.source)
      source.weight,source.bias = self.weight[:,start:start+4096],None
      part = RockchipInt8Linear(source,'on','on',self.tile_columns)
      if self.input_scale is not None:part.input_scale = self.input_scale[start:start+4096].contiguous().realize()
      part.prepare()
      self.parts.append(part)
      if not part.k_parts and len(part.packed)==1:
        self.packed.append(part.packed[0])
      self.scales.extend(part.scales)
    if len(self.packed)!=len(self.parts):self.packed.clear()
    if ((getenv('ONPU_INT8_GROUPED_SPLIT',0) or ONPU_GROUPED_DECODE_DOWN) and self.input_scale is None and self.bias is None
        and len(self.parts)>=4
        and all(part.depth==4096 and part.columns==self.columns and part.input_scale is None and part.bias is None
                and len(part.packed)==1 and len(part.scales)==1 for part in self.parts[:3])):
      # Stack host copies once; the per-part weights stay resident for decode.
      device = self.parts[0].packed[0].device
      self.grouped_weight = Tensor.stack(*(part.packed[0].to('CPU') for part in self.parts[:3]),dim=0).contiguous().to(device).realize()
      self.grouped_scale = Tensor.stack(*(part.scales[0] for part in self.parts[:3]),dim=0).contiguous().realize()
      self.grouped_count = 3
    self._prepared = True

  def release_packed(self):
    super().release_packed()
    self.scales.clear()
    self.grouped_weight = None
    self.grouped_scale = None
    self.grouped_count = 0
    self.parts.clear()

  def npu(self, x:Tensor) -> Tensor:
    if self.shared_projection is not None:return self.shared_npu(x)
    if (self.grouped_weight is not None and self.grouped_scale is not None and self.grouped_count
        and x.device=='CPU' and x.ndim==3 and x.shape[0]==1
        and isinstance(x.max_shape[1],int) and x.max_shape[1]>=1):
      from tinygrad.llm.kernels.rockchip_grouped_int8 import matmul as grouped_int8_matmul
      dense = capacity_input(x)
      capacity = dense.shape[1]
      if isinstance(capacity,int) and 1<=capacity<=128 and (capacity>1 or (ONPU_GROUPED_DECODE_DOWN and self.columns==4096)):
        groups = self.grouped_count
        first = dense[...,:groups*4096]
        grouped_x = first.reshape(1,-1,groups,4096).squeeze(0).permute(1,0,2).contiguous()
        # The grouped dequantizer is a custom program, so the cross-group adds cannot
        # contract its final multiply into an FMA and change the reference FP32 sum.
        grouped = grouped_int8_matmul(grouped_x,self.grouped_weight,self.grouped_scale,4096,materialize=True)
        outputs = [grouped[index].reshape(*dense.shape[:-1],self.columns) for index in range(groups)]
        result = outputs[0]
        for output in outputs[1:]:result = result+output
        offset = groups*4096
        for index in range(groups,len(self.parts)):
          part = self.parts[index]
          result = result+part.npu(dense[...,offset:offset+part.depth].float().contiguous())
          offset += part.depth
        result = result.shrink(tuple((0,s) for s in (*x.shape[:-1],self.columns))).cast(x.dtype)
        return result if self.bias is None else result+self.bias
    values = [part.npu(x[...,index*4096:index*4096+part.depth].float().contiguous()) for index,part in enumerate(self.parts)]
    result = values[0]
    for value in values[1:]:result = result+value
    result = result.cast(x.dtype)
    return result if self.bias is None else result+self.bias


@shared_program
@functools.cache
def _ffn_pack(out:UOp, raw:UOp, sa:UOp, sw:UOp, scale_out:UOp, rows:int, depth:int, tile_rows:int,
              static_layout:bool=False, tune_cpu:bool=False) -> UOp:
  from tinygrad.llm.kernels.rockchip_cpu import _vector_sigmoid_source
  columns = depth*2
  width = round_up(ceildiv(columns,3),32)
  parts = [(start,min(width,columns-start)) for start in range(0,columns,width)]
  if len(parts)!=3: parts = [(0,columns)]
  indices = f'row*{parts[-1][1]}+column-{parts[-1][0]}+{parts[-1][0]*rows}'
  for start,count in reversed(parts[:-1]): indices = f'(column<{start+count}?{start*rows}+row*{count}+column-{start}:{indices})'
  bounds = sorted({0,depth,*(max(0,min(depth,start-base)) for start,_ in parts for base in (0,depth))}) if static_layout else [0,depth]
  loops = []
  for begin,end in zip(bounds,bounds[1:]):
    loads = []
    for base in (0,depth):
      if static_layout:
        start,count = next((start,count) for start,count in parts if start<=base+begin<base+end<=start+count)
        loads.append(f'''(__builtin_convertvector(*(const int4*)(raw+{start*rows+base-start}+row*{count}+c),float4)*
          (float4)sa[row])*(*(const float4*)(sw+{base}+c))''')
      else: loads.append(f'project(raw,sa,sw,row,c+{base})')
    # Independent vector activations hide divide/polynomial latency; reductions keep their original order.
    loops.append(f'''
      {'#pragma clang loop unroll_count(4)' if tune_cpu else ''}
      for(int c={begin};c<{end};c+=4) {{
        float4 gate={loads[0]},up={loads[1]};
        float4 value=(gate/(1.0f+rockchip_exp2_vector(gate*-1.4426950216293335f)))*up;*(float4*)(values+c)=value;
        uint4 bits=__builtin_bit_cast(uint4,value);bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
        float4 a=value>=(float4)0?value:-value;maximum=a>maximum?a:maximum;
      }}''')
  parallel = rows*depth>=32768
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct {unsigned char *out;const int *raw;const float *sa,*sw;float *scales;int begin,end;} Work;
  static void *worker(void *p){Work *w=p;part(w->out,w->raw,w->sa,w->sw,w->scales,w->begin,w->end);return (void*)0;}''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,raw,sa,sw,scales,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(out,raw,sa,sw,scales,work[i].begin,work[i].end);}}
    part(out,raw,sa,sw,scales,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(out,raw,sa,sw,scales,0,{rows});'
  padding = []
  for first in range(0,rows,tile_rows):
    size = min(tile_rows,rows-first)
    base = first*depth+(first//tile_rows)*round_up(tile_rows*4,64)
    padding.append(f'for(int i={size*4};i<{round_up(size*4,64)};i++)out[{base}+i]=0;')
  source = f'''{_vector_sigmoid_source()}
  {_quantize_vector_source()}
  typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
  static __attribute__((always_inline)) float4 project(const int *raw,const float *sa,const float *sw,int row,int column) {{
    int position={indices};float4 value=__builtin_convertvector(*(const int4*)(raw+position),float4);
    return (value*(float4)sa[row])*(*(const float4*)(sw+column));
  }}
  static void part(unsigned char *out,const int *raw,const float *sa,const float *sw,float *scales,int begin,int end) {{
    for(int row=begin;row<end;row++) {{
      float values[{depth}] __attribute__((aligned(16)));float4 maximum=(float4)0;uint4 bad=(uint4)0;
      {''.join(loops)}
      float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
      int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
      int first=row/{tile_rows}*{tile_rows},count={rows}-first<{tile_rows}?{rows}-first:{tile_rows};
      unsigned char *base=out+first*{depth}+first/{tile_rows}*{round_up(tile_rows*4,64)};
      ((float*)base)[row-first]=scales[row]=invalid?__builtin_nanf(""):scale;float inverse=1.0f/scale;
      signed char *dest=(signed char*)(base+((count*4+63)/64)*64)+(row-first)*{depth};
      int c=0;
      for(;c+3<{depth};c+=4){{float4 value=invalid?(float4)0:*(const float4*)(values+c);
        *(char4*)(dest+c)=rockchip_quantize_vector(value,inverse);}}
      for(;c<{depth};c++) {{float q=invalid?0:__builtin_nearbyintf(values[c]*inverse);
        if(q>127)q=127;if(q< -127)q=-127;dest[c]=(signed char)q;}}
    }}
  }}
  {declarations}
  void rockchip_int8_ffn_pack(unsigned char *out,const int *raw,const float *sa,const float *sw,float *scales) {{
    {''.join(padding)}{dispatch}
  }}'''
  inputs = (out,raw,sa,sw,scale_out)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_int8_ffn_pack'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,4),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))


def ffn_pack(raw:Tensor, activation_scales:Tensor, weight_scales:Tensor, rows:int, depth:int,
             tile_rows:int=64) -> tuple[Tensor,Tensor]:
  if (raw.device!='CPU' or raw.dtype is not dtypes.int32 or raw.numel()!=rows*depth*2
      or depth%512 or not 1<=rows<=128 or not 1<=tile_rows<=128 or not 512<=depth<=4096
      or activation_scales.shape!=(rows,) or weight_scales.shape!=(depth*2,)
      or any(t.device!='CPU' or t.dtype is not dtypes.float for t in (activation_scales,weight_scales))):
    raise ValueError('Unsupported fused INT8 FFN layout')
  size = sum(round_up(min(tile_rows,rows-first)*4,64)+min(tile_rows,rows-first)*depth for first in range(0,rows,tile_rows))
  result = Tensor.empty(size,device='CPU',dtype=dtypes.uint8).custom_kernel(
    raw,activation_scales,weight_scales,Tensor.empty(rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,r,a,w,s:_ffn_pack(o,r,a,w,s,rows,depth,tile_rows,bool(getenv('ONPU_SWIGLU_PACK_STATIC_LAYOUT',0)),
                                 bool(ONPU_PREFILL_CPU_TUNING)))
  return result[0],result[4]


class RockchipInt8FeedForward(RockchipFeedForward):
  down:RockchipInt8Linear
  def __init__(self, projections:RockchipProjections, down:RockchipInt8Linear, pool:AttentionPool|None=None):
    self.projections,self.down = projections,down
    self.split_weight = None
    self.pool = pool

  def enabled(self, x:Tensor) -> bool:
    return self._packed_enabled(x) or self._float_enabled(x)

  def _packed_enabled(self, x:Tensor) -> bool:
    joint = self.projections.joint
    return (super().enabled(x) and isinstance(joint,RockchipInt8Linear) and self.down.depth%512==0 and self.down.input_scale is None
      and joint.columns==self.down.depth*2 and len(joint.packed)==len(self.down.packed)==1 and joint.bias is self.down.bias is None)

  def _float_enabled(self, x:Tensor) -> bool:
    return (super().enabled(x) and self.pool is not None and self.pool.active and x.ndim==3 and x.shape[0]==1
      and 16<=x.max_shape[1]<=128 and self.down.depth%4==0 and 4<=self.down.depth<=32768
      and self.projections.joint.columns==self.down.depth*2)

  def __call__(self, x:Tensor) -> Tensor:
    joint = self.projections.joint
    assert isinstance(joint,RockchipInt8Linear)
    if not self._packed_enabled(x):
      from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate
      projected = joint.npu(x)
      depth = self.down.depth
      if getenv('ONPU_FUSED_GATE_MUL',0):
        from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate_mul
        try:return self.down(sigmoid_gate_mul(projected,pool=self.pool))
        except ValueError:pass
      if getenv('ONPU_VECTOR_SIGMOID',0):
        # sigmoid_gate is a custom kernel whose C is still scalar-per-lane.
        # Replace only the sigmoid with a float4 helper, keeping the two
        # multiplies as-is so the arithmetic order is unchanged.
        from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate_vector
        return self.down((projected[...,:depth]*sigmoid_gate_vector(projected))*projected[...,depth:])
      return self.down((projected[...,:depth]*sigmoid_gate(projected,pool=self.pool))*projected[...,depth:])
    dense = capacity_input(x).reshape(-1,joint.depth)
    outputs,output_scales,output_counts = [],[],[]
    input_rows = 128 if padded_depth(joint.depth)<=2048 else 64
    output_rows = 128 if padded_depth(self.down.depth)<=2048 else 64
    for start in range(0,dense.shape[0],input_rows):
      part = dense[start:start+input_rows]
      rows = part.shape[0]
      assert isinstance(rows,int)
      k = joint.packed[0].shape[1]*32
      assert isinstance(k,int)
      payload,sa = quantize(joint.scaled_input(part),k)
      use_gpu = bool(getenv('ONPU_FFN_GPU',0)) and (rows,self.down.depth,output_rows)==(128,3072,64)
      packed_tiles:tuple[Tensor,...]
      if use_gpu:
        from tinygrad.llm.kernels.rockchip_gpu import gpu_ffn_pack
        assert isinstance(self.down.packed[0].device,str)
        payload = payload.to(joint.packed[0].device)
        projected = raw_int8_matmul(payload,joint.packed[0],rows,joint.columns,cpu_output=False)
        packed_tiles,scales = gpu_ffn_pack(projected,payload,joint.scales[0],self.down.packed[0].device)
      else:
        projected = raw_int8_matmul(payload,joint.packed[0],rows,joint.columns)
        packed,scales = ffn_pack(projected,sa,joint.scales[0],rows,self.down.depth,output_rows)
        packed_tiles = tuple(packed[first*self.down.depth+(first//output_rows)*round_up(output_rows*4,64):
          first*self.down.depth+(first//output_rows)*round_up(output_rows*4,64)+
          round_up(min(output_rows,rows-first)*4,64)+min(output_rows,rows-first)*self.down.depth]
          for first in range(0,rows,output_rows))
      for first,payload in zip(range(0,rows,output_rows),packed_tiles):
        count = min(output_rows,rows-first)
        output = raw_int8_matmul(payload,self.down.packed[0],count,self.down.columns)
        outputs.append(output)
        output_scales.append(scales[first:first+count])
        output_counts.append(count)
    out = dequantize_tiles(outputs,output_scales,self.down.scales[0],output_counts,self.down.columns)
    out = out.reshape(*x.max_shape[:-1],self.down.columns)
    return out.shrink(tuple((0,s) for s in (*x.shape[:-1],self.down.columns))).cast(x.dtype)


class RockchipInt8PreparedQKV(RockchipPreparedQKV):
  def __call__(self, x:Tensor, start:int|UOp, cache:Tensor, freqs:Tensor) -> tuple[Tensor,Tensor]:
    from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv_int8
    joint = self.projections.joint
    assert isinstance(joint,RockchipInt8Linear)
    raw,scales,counts = joint.raw_projection(x,defer_bias=True)
    return prepare_qkv_int8(raw,scales,joint.scales[0],counts,cache,self.qnorm,self.knorm,freqs,start,self.heads,self.eps,x.shape[1],
                            bias=joint.bias,rope_dim=self.rope_dim or 0)
