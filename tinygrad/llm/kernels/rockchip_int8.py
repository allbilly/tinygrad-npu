"""Opt-in dynamic per-token/per-channel INT8 matrices with FP32 dequantization."""
from __future__ import annotations
import copy, functools, struct
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, ceildiv, getenv, round_up
from tinygrad.llm.kernels.rockchip import RockchipLinear, RockchipFeedForward, RockchipProjections, RockchipPreparedQKV
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_PREFILL_CPU_TUNING, shared_program
from tinygrad.llm.kernels.rockchip_padding import capacity_input
from tinygrad.runtime.support.rockchip_int8 import padded_depth, supported_depth, sdk_prefill_supported
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

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


def quantize(x:Tensor, padded:int, *, sdk_prefill:bool=False) -> tuple[Tensor,Tensor]:
  if x.ndim!=2 or x.device!='CPU' or not isinstance(x.shape[0],int) or not isinstance(x.shape[1],int):
    raise ValueError('INT8 quantization requires a static CPU matrix')
  rows,depth = x.shape
  assert isinstance(rows,int) and isinstance(depth,int)
  if not 1<=rows<=24576 or not 1<=depth<=padded or not supported_depth(padded): raise ValueError('Unsupported INT8 quantization geometry')
  if type(sdk_prefill) is not bool:raise ValueError('SDK INT8 quantization layout must be a bool')
  result = Tensor.empty(round_up(rows*4,64)+rows*padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
    x.float().contiguous(),Tensor.empty(rows,device='CPU',dtype=dtypes.float),fxn=lambda o,a,s:_quantize(o,a,s,rows,depth,padded,sdk_prefill))
  return result[0],result[2]


@functools.cache
def _matmul(out:UOp, lhs:UOp, rhs:UOp, m:int, n:int, k:int, column_offset:int, cores:int, sdk_prefill:bool=False) -> UOp:
  sink = UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_int8_{m}_{n}_{k}'+('_sdk' if sdk_prefill else '')))
  info = replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),UOp(Ops.SOURCE,arg=f'INT8 {m}x{k}x{n}'),
    UOp(Ops.BINARY,arg=struct.pack('<9I' if sdk_prefill else '<8I',m,n,k,cores,column_offset*k,round_up(m*4,64),0,1,
                                 *((1,) if sdk_prefill else ())))))


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
                    sdk_prefill:bool=False, cpu_output:bool=True) -> Tensor:
  k = weight.shape[1]*32
  n = round_up(columns,32)
  if (not isinstance(k,int) or weight.ndim!=4 or weight.shape[2:]!=(32,32) or weight.dtype is not dtypes.int8
      or not isinstance(weight.device,str) or weight.device.split(':')[0]!='ROCKCHIP'
      or not isinstance(payload.device,str) or payload.device!='CPU' and payload.device.split(':')[0]!='ROCKCHIP'
      or payload.dtype is not dtypes.uint8
      or not 1<=rows<=(128 if k<=2048 else 64) or columns<1 or column_offset<0 or column_offset%32
      or column_offset+n>weight.shape[0]*32 or n>(24576 if rows==1 else 12288)
      or payload.numel()!=round_up(rows*4,64)+rows*k): raise ValueError('Unsupported raw INT8 operands')
  if type(sdk_prefill) is not bool or sdk_prefill and not sdk_prefill_supported(rows,n,k):
    raise ValueError('Unsupported SDK INT8 matrix geometry')
  if type(cpu_output) is not bool:raise ValueError('CPU output selection must be a bool')
  result = Tensor.empty(rows*n,device=weight.device,dtype=dtypes.int32).custom_kernel(payload.to(weight.device),weight,
    fxn=lambda o,a,b:_matmul(o,a,b,rows,n,k,column_offset,3,sdk_prefill))[0]
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


class RockchipInt8Linear(RockchipLinear):
  def __init__(self, *args):
    super().__init__(*args)
    if self.weight.device!='CPU': raise ValueError('INT8 acceleration requires CPU weights')
    self.scales:list[Tensor] = []
    self.input_scale:Tensor|None = None

  def scaled_input(self, x:Tensor) -> Tensor: return x if self.input_scale is None else x.float()/self.input_scale

  def prepare(self):
    if self.shared_projection is not None:return
    if self.packed: return
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
    for start in range(0,self.columns,self.tile_columns):
      raw,scales = quantize(weight[start:start+self.tile_columns],k)
      scales = scales.contiguous().realize()
      values = scales.tolist()
      assert isinstance(values,list)
      if any(not 0<s<float('inf') for s in values): raise ValueError('INT8 weights must be finite')
      count = len(values)
      n = round_up(count,32)
      codes = raw[round_up(count*4,64):].bitcast(dtypes.int8).reshape(count,k).pad(((0,n-count),(0,0)))
      from tinygrad.runtime.ops_rockchip import int8_weight_device
      device = int8_weight_device(n*k)
      self.packed.append(codes.reshape(n//32,32,k//32,32).permute(0,2,1,3).contiguous().to(device).realize())
      self.scales.append(scales)

  def release_packed(self):
    super().release_packed()
    self.scales.clear()

  def raw_projection(self, x:Tensor, *, sdk_prefill:bool=False, defer_bias:bool=False) -> tuple[list[Tensor],list[Tensor],list[int]]:
    if type(defer_bias) is not bool:raise ValueError('Deferred raw projection bias must be a bool')
    if x.device!='CPU' or len(self.packed)!=1 or self.columns>12288 or self.bias is not None and not defer_bias:
      raise ValueError('Unsupported raw INT8 projection')
    dense = capacity_input(x).reshape(-1,self.depth)
    k = self.packed[0].shape[1]*32
    assert isinstance(k,int)
    tile_rows = 128 if k<=2048 else 64
    raw,scales,counts = [],[],[]
    for start in range(0,dense.shape[0],tile_rows):
      part = dense[start:start+tile_rows]
      count = part.shape[0]
      assert isinstance(count,int)
      sdk = sdk_prefill and sdk_prefill_supported(count,round_up(self.columns,32),k)
      payload,scale = quantize(self.scaled_input(part),k,sdk_prefill=sdk)
      raw.append(raw_int8_matmul(payload,self.packed[0],count,self.columns,sdk_prefill=sdk))
      scales.append(scale)
      counts.append(count)
    return raw,scales,counts

  def npu(self, x:Tensor) -> Tensor:
    if self.shared_projection is not None:return self.shared_npu(x)
    if x.device!='CPU': raise ValueError('INT8 acceleration requires CPU activations')
    shape = x.shape
    dense = capacity_input(x).reshape(-1,self.depth)
    rows = []
    tile_rows = 128 if padded_depth(self.depth)<=2048 else 64
    if len(self.packed)==1 and self.columns<=12288 and dense.shape[0]>1 and self.bias is None:
      sdk_enabled = bool(getenv('ONPU_INT8_SDK_PREFILL',0))
      raw_tiles,tile_scales,counts = self.raw_projection(x,sdk_prefill=sdk_enabled)
      sdk_tiles = tuple(sdk_enabled and sdk_prefill_supported(m,round_up(self.columns,32),padded_depth(self.depth)) for m in counts)
      out = dequantize_tiles(raw_tiles,tile_scales,self.scales[0],counts,self.columns,sdk_tiles=sdk_tiles)
      out = out.reshape(*x.max_shape[:-1],self.columns)
      out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
      return out if self.bias is None else out+self.bias
    for start in range(0,dense.shape[0],tile_rows):
      part = dense[start:start+tile_rows]
      columns = []
      for index,(weight,scales) in enumerate(zip(self.packed,self.scales)):
        width = min(self.tile_columns,self.columns-index*self.tile_columns)
        limit = 24576 if part.shape[0]==1 else 12288
        for col in range(0,width,limit): columns.append(packed_int8_matmul(self.scaled_input(part),weight,scales,min(limit,width-col),col))
      rows.append(Tensor.cat(*columns,dim=1))
    out = Tensor.cat(*rows,dim=0).reshape(*x.max_shape[:-1],self.columns)
    out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
    return out if self.bias is None else out+self.bias


class RockchipInt8SplitLinear(RockchipInt8Linear):
  """Blockwise INT8 projections beyond one NPU task's input-width limit."""
  def __init__(self, *args):
    super().__init__(*args)
    self.parts:list[RockchipInt8Linear] = []

  def prepare(self):
    if self.shared_projection is not None:return
    if self.packed:return
    if not 4096<self.depth<=32768:raise ValueError('Unsupported split INT8 projection depth')
    for start in range(0,self.depth,4096):
      source = copy.copy(self.source)
      source.weight,source.bias = self.weight[:,start:start+4096],None
      part = RockchipInt8Linear(source,'on','on',self.tile_columns)
      if self.input_scale is not None:part.input_scale = self.input_scale[start:start+4096].contiguous().realize()
      part.prepare()
      self.parts.append(part)
      self.packed.extend(part.packed)
      self.scales.extend(part.scales)

  def release_packed(self):
    super().release_packed()
    self.scales.clear()
    self.parts.clear()

  def npu(self, x:Tensor) -> Tensor:
    if self.shared_projection is not None:return self.shared_npu(x)
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
    return prepare_qkv_int8(raw,scales,joint.scales[0],counts,cache,self.qnorm,self.knorm,freqs,start,self.heads,self.eps,x.shape[1],bias=joint.bias)
