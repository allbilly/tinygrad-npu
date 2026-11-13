"""Opt-in dynamic per-token/per-channel INT8 matrices with FP32 dequantization."""
from __future__ import annotations
import copy, functools, math, struct, weakref
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

# Opt-in: quantize a row window of the shared dense activation buffer instead of
# materialising each row tile into its own temporary (bit-exact, see quantize()).
ONPU_INT8_ROW_FOLD = ContextVar('ONPU_INT8_ROW_FOLD',1)
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_INT8_GROUPED_GATE_UP = ContextVar('ONPU_INT8_GROUPED_GATE_UP',0)
ONPU_GROUPED_DECODE_DOWN = ContextVar('ONPU_GROUPED_DECODE_DOWN',0)
ONPU_INT8_REUSE_ACTIVATION = ContextVar('ONPU_INT8_REUSE_ACTIVATION',0)

# ---------------------------------------------------------------------------
# Prefill structural quantize share (default OFF, see ONPU_PREFILL_QUANT_SHARE below).
#
# MLA `attn_q_a` and `attn_kv_a_mqa` are both INT8 linears with the SAME 2560-deep INT8
# activation, called on the SAME normalized hidden state, so for every 64-row tile they
# quantize identical bytes twice and upload the payload twice (2 redundant launches per
# block x 62 blocks). A quantize memo keyed on the *tile* can never see this: each
# projection first materialises its own `capacity_input(x)` padded buffer, so the two
# tiles are different UOps with the same bytes.
#
# This shares the padded dense activation AND the quantized (payload, scales) result
# between consumers that were handed the SAME input Tensor object in one forward. It
# cannot change any value: the second consumer reads exactly the tensor the first one
# produced (same buffer, no recompute), and the shared key is the input's UOp identity,
# so two different activations can never be confused even when shapes match. Decode
# (1 row) and all non-CPU paths are untouched. Cleared once per captured forward.
ONPU_PREFILL_QUANT_SHARE = ContextVar('ONPU_PREFILL_QUANT_SHARE',1)
_shared_dense:dict[tuple,Tensor] = {}
_shared_payload:dict[tuple,tuple[Tensor,Tensor]] = {}
_shared_upload:dict[tuple,Tensor] = {}
_share_counts:dict[str,int] = {}


def clear_quantize_share() -> None:
  """Drop the per-forward share caches. Called once per captured forward/token."""
  _shared_dense.clear()
  _shared_payload.clear()
  _shared_upload.clear()
  _share_counts.clear()


def quantize_share_counts() -> dict[str,int]: return dict(_share_counts)



def _share_key(x:Tensor, depth:int) -> tuple|None:
  """Identity key for a sharable prefill activation tile (None: do not share).

  `x.uop` is used as the dict key: UOp hashes on its structure but compares by identity
  (`UOp.__eq__ is`), and the dict holds the UOp strongly, so a hit means literally the same
  producer node - a different buffer with the same shape can never match. Only concrete
  multi-row CPU FP32 prefill activations are shared.
  """
  if not ONPU_PREFILL_QUANT_SHARE or x.device!='CPU' or x.dtype is not dtypes.float:return None
  if x.ndim!=3 or not isinstance(x.shape[-1],int) or int(x.shape[-1])!=depth:return None
  rows = x.max_shape[:2]
  if not all(isinstance(n,int) for n in rows) or math.prod(int(n) for n in rows)<=1:return None
  return (x.uop, depth)


def _shared_capacity(x:Tensor, depth:int, key:tuple) -> Tensor:
  """`capacity_input(x).reshape(-1,depth)`, computed once per activation per forward."""
  dense = _shared_dense.get(key)
  if dense is None:
    dense = capacity_input(x).reshape(-1,depth)
    _shared_dense[key] = dense
    _share_counts['dense'] = _share_counts.get('dense',0)+1
  else:
    _share_counts['dense_hit'] = _share_counts.get('dense_hit',0)+1
  return dense


def _shared_quantize(key:tuple, index:tuple, compute:'Callable[[],tuple[Tensor,Tensor]]') -> tuple[Tensor,Tensor]:
  """One quantize per (activation, row window) per forward; later consumers reuse the bytes."""
  full = (key, *index)
  hit = _shared_payload.get(full)
  if hit is not None:
    _share_counts['payload_hit'] = _share_counts.get('payload_hit',0)+1
    return hit
  result = compute()
  _shared_payload[full] = result
  _share_counts['payload'] = _share_counts.get('payload',0)+1
  return result


def _shared_payload_upload(payload:Tensor, device:str) -> Tensor:
  """Upload a shared payload once per (payload, device): a byte copy is idempotent."""
  if not ONPU_PREFILL_QUANT_SHARE:return payload.to(device)
  full = (payload.uop, device)
  hit = _shared_upload.get(full)
  if hit is not None:
    _share_counts['upload_hit'] = _share_counts.get('upload_hit',0)+1
    return hit
  result = payload.to(device)
  _shared_upload[full] = result
  return result

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


def _sdk_quantize_part(rows:int, depth:int, padded:int, header:int, row0:int=0) -> str:
  return f'''static void part(unsigned char *raw,const float *input,float *scale_out,int begin,int end) {{
    float *scales=(float*)raw;signed char *output=(signed char*)(raw+{header});
    for(int first=begin;first<end;first+=4) {{
      int count=end-first<4?end-first:4;float inverse[4];int invalid[4];
      for(int r=0;r<count;r++) {{
        const float *x=input+(first+r+{row0})*{depth};float4 maximum={{0}};uint4 bad={{0}};int i=0;
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
        const float *x=input+(first+r+{row0})*{depth};signed char *dest=output+(c/16*{rows}+first+r)*16;
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
def _quantize(out:UOp, inp:UOp, scale_out:UOp, rows:int, depth:int, padded:int, sdk_prefill:bool=False,
              row0:int=0) -> UOp:
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
      const float *x=input+(row+{row0})*{depth};float4 maximum={{0}};uint4 bad={{0}};int i=0;
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
  {_sdk_quantize_part(rows,depth,padded,header,row0) if sdk_prefill else part}
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


def quantize(x:Tensor, padded:int, *, sdk_prefill:bool=False, weight:bool=False,
             rows:int|None=None, row_offset:int=0) -> tuple[Tensor,Tensor]:
  if x.ndim!=2 or x.device!='CPU' or not isinstance(x.shape[0],int) or not isinstance(x.shape[1],int):
    raise ValueError('INT8 quantization requires a static CPU matrix')
  total,depth = x.shape
  assert isinstance(total,int) and isinstance(depth,int)
  if rows is None: rows = total
  if (type(rows) is not int or type(row_offset) is not int or rows<1 or row_offset<0
      or row_offset+rows>total or not 1<=rows<=24576):
    raise ValueError('Unsupported INT8 quantization row window')
  window = rows != total or row_offset != 0
  if not 1<=depth<=padded or not supported_depth(padded): raise ValueError('Unsupported INT8 quantization geometry')
  if type(sdk_prefill) is not bool:raise ValueError('SDK INT8 quantization layout must be a bool')
  from tinygrad.llm.kernels.rockchip_hadamard import enabled,quantize as hadamard_quantize
  if type(weight) is not bool:raise ValueError('Weight quantization selection must be a bool')
  if enabled(depth):
    if sdk_prefill or ONPU_NORM_QUANT:raise ValueError('Hadamard quantization requires ordinary layout and separate CPU normalization')
    if window:raise ValueError('Hadamard quantization does not support row windows')
    return hadamard_quantize(x,padded,weight=weight)
  if ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE:
    from tinygrad.llm.kernels.rockchip_norm_quant import try_quantize
    if not window and (fused:=try_quantize(x,padded,sdk_prefill=sdk_prefill)) is not None:return fused
  # rows/row_offset quantize one row window of a larger matrix in place, so the
  # caller can feed the shared dense buffer instead of materialising a slice per
  # tile (bit-exact: same per-row max/scale/rounding, same order, reads the same
  # rows; only the input pointer changes).
  result = Tensor.empty(round_up(rows*4,64)+rows*padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
    x.float().contiguous(),Tensor.empty(rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,a,s:_quantize(o,a,s,rows,depth,padded,sdk_prefill,row_offset))
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


# ---------------------------------------------------------------------------
# Prefill host fold: dequantize the ffn_down parts and add the block residual in one kernel.
#
# The prefill block output is `h + feed_forward*residual_scale` where feed_forward is the sum
# of the ffn_down part dequantizers. Today that is one `rockchip_dequantize` kernel per part
# (each writing a 128x2560 f32 tensor) plus one elementwise residual-add kernel that re-reads
# them (profile: CPU:E_28toks2A256029 35.9 ms / 123 calls, bandwidth-bound). Registering the
# part dequantizers lets one kernel dequantize every part in registers, sum them in the
# reference order, scale, and add the residual, so no f32 down output is ever materialized.
ONPU_PREFILL_RESIDUAL_FUSE = ContextVar('ONPU_PREFILL_RESIDUAL_FUSE',1)
_pending_dequant:weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _walk_registered(u:UOp):
  if (found:=_pending_dequant.get(u)) is not None:return found
  if u.op in (Ops.CAST,Ops.SHRINK,Ops.RESHAPE,Ops.PAD,Ops.CONTIGUOUS_BACKWARD,Ops.BITCAST):
    return _walk_registered(u.src[0])
  return None


def _register_prefill_dequant(result:Tensor, raws:list[Tensor], sas:list[Tensor], sw:Tensor,
                              counts:list[int], columns:int, *, layouts:tuple[int,...]|None=None) -> None:
  """Record one CPU prefill part-dequantizer so its f32 output can be folded away.

  `layouts` (one entry per raw tile) selects the source layout of each raw tensor: 0 is the
  blocked `_dequantize_tiles` layout, a positive value is the row-major `padded` stride of
  `rockchip_grouped_int8`'s dequantizer. Omitted layouts are all-blocked, the only case the
  earlier registrations ever produced.
  """
  if not ONPU_PREFILL_RESIDUAL_FUSE:return
  _pending_dequant[result.uop] = {'raws':[r.uop for r in raws],'sas':[a.uop for a in sas],'sw':sw.uop,
                                  'counts':tuple(counts),'columns':columns,
                                  'layouts':tuple(layouts) if layouts is not None else ()}


def _register_prefill_sum(result:Tensor, parts:list, columns:int, rows:int) -> None:
  """Record the sum of N registered part dequantizers (the ffn_down output)."""
  if not ONPU_PREFILL_RESIDUAL_FUSE or any(part is None for part in parts):return
  _pending_dequant[result.uop] = {'parts':parts,'columns':columns,'rows':rows}


def _register_prefill_grouped(outputs:list, raw:Tensor, activation_scales:Tensor, weight_scales:Tensor,
                             groups:int, rows:int, counts:tuple[int,...], columns:int, padded:int) -> bool:
  """Register the per-group dequantizers of one row-major grouped raw output.

  `rockchip_grouped_int8.dequantize` computes `((float)raw[g,row,c]*a[g,row])*w[g,c]` on a
  row-major buffer of `padded` columns per row. One descriptor per group lets the block-output
  residual add fold that dequantize in registers. The group's buffer is registered once as
  `(groups*rows, padded)` and each row tile of the shared partition is addressed by a
  compile-time base row, so nothing is sliced or copied and `counts` must be the SAME partition
  the other parts of the sum register (the fused kernel derives one tile per output row).
  """
  if not ONPU_PREFILL_RESIDUAL_FUSE:return False
  if (not outputs or len(outputs)!=groups or groups<1 or rows<1 or columns<1 or padded<columns
      or not counts or sum(counts)!=rows or any(count<=0 for count in counts)):return False
  starts,offset = [],0
  for count in counts:
    starts.append(offset);offset += count
  raws = raw.reshape(groups*rows,padded)
  scales = activation_scales.reshape(groups*rows)
  weight = weight_scales.reshape(groups,padded)
  for group,output in enumerate(outputs):
    _pending_dequant[output.uop] = {
      'raws':[raws.uop]*len(counts),'sas':[scales.uop]*len(counts),'sw':weight.uop,
      'counts':tuple(counts),'columns':columns,'layouts':(padded,),
      'raw_bases':tuple(group*rows+start for start in starts),
      'sa_bases':tuple(group*rows+start for start in starts),'sw_bases':(group*padded,)}
  return True


@shared_program
@functools.cache
def _dequant_residual(out:UOp, hidden:UOp, raw:tuple[UOp,...], sa:tuple[UOp,...], sw:tuple[UOp,...],
                      counts:tuple[int,...], columns:int, tile:int, nparts:int, scale:float,
                      layouts:tuple[int,...]=(), raw_bases:tuple[int,...]=(), sa_bases:tuple[int,...]=(),
                      sw_bases:tuple[int,...]=()) -> UOp:
  """out[r,c] = hidden[r,c] + ((d0 + d1 + ...) * scale) with each d_p dequantized in place.

  Mirrors `_dequantize_tiles` exactly: the raw INT32 tile stores the columns in ceil(n/3)
  blocks of `part_width` (that is how the NPU writes each raw output), so the source index is
  `start*m + local*part_width + (c-start)` for output column c, and the destination is
  row-major. Everything else (per-row activation scale, per-column weight scale, multiply
  order, part summation order) is the same expression as the standing dequantizer.

  `layouts[p]` selects the source layout of part p:
    0            = blocked per-part raw of `_dequantize_tiles`: column c lives at
                   `start*m + local*width + (c-start)` inside the ceil(n/3) block `start`;
    `padded>0`   = row-major raw of `rockchip_grouped_int8._dequantize_program`, which
                   stores `padded` (= k) columns per row: source index `local*padded + c`
                   and the weight row is that group's own `sw` row at offset 0.
  Defaults (all zero) reproduce the previous source byte-for-byte, so flag-off is unchanged.
  Every part keeps the standing expression `((float)raw*a)*w` and the same summation order.
  """
  ntiles,total,n = len(counts),sum(counts),round_up(columns,32)
  layouts = layouts or (0,)*nparts
  raw_bases = raw_bases or (0,)*(nparts*ntiles)
  sa_bases = sa_bases or (0,)*(nparts*ntiles)
  sw_bases = sw_bases or (0,)*nparts
  if (len(layouts)!=nparts or any(type(layout) is not int or layout<0 for layout in layouts)
      or len(raw_bases)!=nparts*ntiles or len(sa_bases)!=nparts*ntiles or len(sw_bases)!=nparts):
    raise ValueError('Unsupported residual part layout')
  # A row-major part indexes by ABSOLUTE output column (`local*stride + c`), so it works
  # under the same 3-way column split as the blocked layout; it only needs its row stride
  # to cover the whole output width.
  if any(layout>0 and layout<columns for layout in layouts):
    raise ValueError('Row-major residual part requires a stride covering the output width')
  width = round_up(ceildiv(n,3),32)
  column_parts = [(start,min(width,n-start)) for start in range(0,n,width)]
  if len(column_parts)!=3:column_parts = [(0,n)]
  parallel = total*columns>=131072
  # A grouped part registers its group's whole row-major buffer once and passes it for every
  # row tile with that tile's base row (so `local` stays the tile-local row); a blocked part
  # registers one tensor per row tile and needs no base.
  # A grouped part registers its group's whole row-major buffer once and passes it for every
  # row tile with that tile's base row, so `local` stays the tile-local row. The runtime tile
  # index picks the compiled per-tile base/scale offset out of a literal table.
  loads = ' '.join(
    (f'const int *r{p}=wk->raw[{p*ntiles}]+(size_t)(raw_base{p}[t])*{layouts[p]};'
     f'const float *a{p}=wk->sa[{p*ntiles}]+row_base{p}[t];const float *w{p}=wk->sw[{p}]+{sw_bases[p]};')
    if layouts[p] else
    (f'const int *r{p}=wk->raw[{p*ntiles}+t];const float *a{p}=wk->sa[{p*ntiles}+t];'
     f'const float *w{p}=wk->sw[{p}];') for p in range(nparts))
  grouped_tables = ' '.join(
    f'static const int raw_base{p}[{ntiles}]={{{",".join(str(raw_bases[p*ntiles+i]) for i in range(ntiles))}}};'
    f'static const int row_base{p}[{ntiles}]={{{",".join(str(sa_bases[p*ntiles+i]) for i in range(ntiles))}}};'
    for p in range(nparts) if layouts[p])
  sums = ' '.join(['float4 sum=d0;']+[f'sum=sum+d{p};' for p in range(1,nparts)])
  def source(p:int, start:int, count:int) -> str:
    return f'local*{layouts[p]}+c' if layouts[p] else f'{start}*m_+local*{count}+c-{start}'
  # a grouped part indexes its own group's rows from the base, so its tile-local row is the
  # global row minus that group's first row; the base offsets make `local` the tile row.
  loops = []
  for start,count in column_parts:
    used = min(count,columns-start)
    if used<=0:continue
    deqs = ' '.join(f'float4 g{p}=__builtin_convertvector(*(const int4*)(r{p}+{source(p,start,count)}),float4);'
                    f'float4 d{p}=(g{p}*(float4)a{p}[local])*(*(const float4*)(w{p}+c));' for p in range(nparts))
    loops.append(f'''for(int c={start};c<{start+used};c+=4) {{
        {deqs}
        {sums}
        o[c/4]=hs[c/4]+sum*(float4){scale:.17g}f;
      }}''')
  declarations = '''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);''' if parallel else ''
  dispatch = (f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,hidden,raw,sa,sw,{total}*i/4,{total}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(&work[i]);}}
    part(&work[3]);for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else
              f'Work one={{out,hidden,raw,sa,sw,0,{total}}};part(&one);')
  parameters = ','.join([*(f'const int *r{i}' for i in range(nparts*ntiles)),
                         *(f'const float *a{i}' for i in range(nparts*ntiles)),
                         *(f'const float *w{i}' for i in range(nparts))])
  text = f'''#include <stddef.h>
  typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(16),ext_vector_type(4)));
  typedef struct {{float *out;const float *hidden;const int *const *raw;const float *const *sa;
                   const float *const *sw;int begin,end;}} Work;
  {grouped_tables}
  static void part(Work *wk);
  static void *worker(void *p);
  {declarations}
  static void part(Work *wk) {{
    const int rows_per_tile[{ntiles}]={{{','.join(str(m) for m in counts)}}};
    for(int r=wk->begin;r<wk->end;r++) {{
      int t=r/{tile};if(t>={ntiles})t={ntiles-1};const int local=r-t*{tile},m_=rows_per_tile[t];
      const float4 *hs=(const float4*)(wk->hidden+(size_t)r*{columns});
      float4 *o=(float4*)(wk->out+(size_t)r*{columns});
      {loads}
      {''.join(loops)}
    }}
  }}
  static void *worker(void *p){{part((Work*)p);return (void*)0;}}
  void rockchip_dequantize_residual_add(float *out,const float *hidden,{parameters}) {{
    const int *raw[{nparts*ntiles}]={{{','.join(f'r{i}' for i in range(nparts*ntiles))}}};
    const float *sa[{nparts*ntiles}]={{{','.join(f'a{i}' for i in range(nparts*ntiles))}}};
    const float *sw[{nparts}]={{{','.join(f'w{i}' for i in range(nparts))}}};
    {dispatch}
  }}'''
  inputs = (out,hidden,*raw,*sa,*sw)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_dequantize_residual_add'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=text),
                                     UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(text))))

def try_prefill_residual(hidden:Tensor, value:Tensor, scale:float, *, site:str="") -> Tensor|None:
  """Emit the fused dequantize+residual-add kernel when `value` is a registered ffn_down sum."""
  dbg=bool(getenv('ONPU_DEBUG_LEVERS',0))
  want=getenv('ONPU_PREFILL_RESIDUAL_FUSE_SITE','') or ''
  def _skip(reason:str) -> None:
    if dbg:print(f'[LEV] residual skip: {reason}',flush=True)
    return None
  if want and site!=want:return _skip(f'site {site!r} != {want!r}')
  if not ONPU_PREFILL_RESIDUAL_FUSE or hidden.device!='CPU' or value.device!='CPU':return _skip(f'flag/dev {ONPU_PREFILL_RESIDUAL_FUSE} {hidden.device} {value.device}')
  if hidden.dtype is not dtypes.float or not isinstance(scale,(int,float)) or not math.isfinite(scale):return _skip(f'dtype/scale {hidden.dtype} {scale!r}')
  found = _walk_registered(value.uop)
  if found is None:return _skip(f'not-registered value.op={value.uop.op} shape={value.max_shape}')
  # a bare part dequantizer (attention output projection) or the sum of N of them (ffn_down)
  if 'parts' in found:parts,rows,columns = found['parts'],found['rows'],found['columns']
  else:parts,rows,columns = [found],sum(found['counts']),found['columns']
  if dbg:print(f'[LEV] residual found rows={rows} columns={columns} parts={len(parts)}',flush=True)
  # Symbolic prefill graphs carry `toks` as a bound variable: validate through max_shape
  # (the qualified single-chunk prefill has max rows == the registered capacity), never
  # evaluate a symbolic comparison. The custom kernel is generated at the capacity-size
  # row count; the underlying buffers are allocated at max_shape, so reading/writing the
  # padded rows is in-bounds and the caller slices the logical rows back out.
  if rows<2:return _skip(f'rows<2 rows={rows}')
  if hidden.ndim!=3 or value.ndim!=3:return _skip(f'ndim hidden={hidden.ndim} value={value.ndim} {hidden.max_shape} {value.max_shape}')
  if not isinstance(hidden.shape[0],int) or hidden.shape[0]!=1:return _skip(f'batch {hidden.shape}')
  if not isinstance(hidden.shape[-1],int) or hidden.shape[-1]!=columns:return _skip(f'columns {hidden.shape} vs {columns}')
  if int(hidden.max_shape[-1])!=columns or int(value.max_shape[-1])!=columns:return _skip(f'max columns {hidden.max_shape} {value.max_shape} vs {columns}')
  if int(hidden.max_shape[0])*int(hidden.max_shape[1])!=rows:return _skip(f'max rows hidden {hidden.max_shape} vs rows={rows} columns={columns}')
  if int(value.max_shape[0])*int(value.max_shape[1])!=rows:return _skip(f'max rows value {value.max_shape} vs rows={rows} columns={columns}')
  if dbg:print(f'[LEV] residual FUSE rows={rows} columns={columns} hidden={hidden.max_shape}',flush=True)
  raws:list[Tensor] = []
  sas:list[Tensor] = []
  sws:list[Tensor] = []
  layouts:list[int] = []
  raw_bases:list[int] = []
  sa_bases:list[int] = []
  sw_bases:list[int] = []
  counts:tuple[int,...]|None = None
  for part in parts:
    if counts is None:counts = tuple(part['counts'])
    elif tuple(part['counts'])!=counts:return None
    part_layouts = tuple(part.get('layouts') or ())
    tiles = len(part['raws'])
    # the layout is a per-PART property (every row tile of a part shares it); bases are per tile
    if len(part_layouts) not in (0,1):return None
    raws.extend(Tensor(u) for u in part['raws'])
    sas.extend(Tensor(u) for u in part['sas'])
    sws.append(Tensor(part['sw']))
    layouts.extend(part_layouts or (0,))
    part_raw_bases = tuple(part.get('raw_bases') or ())
    part_sa_bases = tuple(part.get('sa_bases') or ())
    part_sw_bases = tuple(part.get('sw_bases') or ())
    if len(part_raw_bases) not in (0,tiles) or len(part_sa_bases) not in (0,tiles) or len(part_sw_bases) not in (0,1):
      return None
    raw_bases.extend(part_raw_bases or (0,)*tiles)
    sa_bases.extend(part_sa_bases or (0,)*tiles)
    sw_bases.extend(part_sw_bases or (0,))
  if counts is None:return None
  # Every registered part must expose one layout/base per raw tile of every row tile.
  if (len(layouts)!=len(parts) or len(raw_bases)!=len(raws) or len(sa_bases)!=len(sas)
      or len(sw_bases)!=len(sws)):return None
  # The fused kernel is generated at the padded (max-shape) row count and reads the hidden
  # buffer flat: pass the symbolic tensor unchanged (its allocation is max-shape sized) and
  # slice the logical rows back out with a symbolic shrink, so no shape product is evaluated.
  # custom_kernel placeholders need concrete shapes: pad the residual to the padded
  # (max-shape) capacity first (a no-op copy when the logical rows already fill it), then
  # reshape it concretely. The result is sliced back to the logical rows symbolically.
  hin = capacity_input(hidden)
  result = Tensor.empty(rows,columns,device='CPU',dtype=dtypes.float).custom_kernel(
    hin.reshape(rows,columns).contiguous(),*[r.contiguous() for r in raws],*[a.contiguous() for a in sas],
    *[w.contiguous() for w in sws],
    fxn=lambda o,h,*args:_dequant_residual(o,h,tuple(args[:len(raws)]),
                                           tuple(args[len(raws):len(raws)+len(sas)]),
                                           tuple(args[len(raws)+len(sas):]),counts,columns,counts[0],
                                           len(parts),float(scale),tuple(layouts),tuple(raw_bases),
                                           tuple(sa_bases),tuple(sw_bases)))[0]
  out = result.reshape(1,rows,columns)
  depth = hidden.shape[1]
  return out if isinstance(depth,int) and depth==rows else out.shrink(((0,1),(0,depth),(0,columns)))


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
  result = Tensor.empty(rows*n,device=weight.device,dtype=dtypes.int32).custom_kernel(
    _shared_payload_upload(payload,weight.device),weight,
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
    # ONPU_PREFILL_QUANT_SHARE: share the padded dense tile AND its quantized payload with any
    # other projection fed the same activation this forward (MLA q_a + kv_a). Sharing is only
    # possible when the quantize is parameter-identical: no input scales, one packed weight tile
    # (so the payload layout depends only on rows/k), no K split and no SDK layout.
    share = _share_key(x,self.depth) if self.input_scale is None and not self.k_parts and len(self.packed)==1 else None
    dense = _shared_capacity(x,self.depth,share) if share is not None else capacity_input(x).reshape(-1,self.depth)
    k = padded_depth(self.depth)
    assert isinstance(k,int)
    tile_rows = 128 if len(self.k_parts)==1 or self.depth<=2048 else 64
    raw,scales,counts = [],[],[]
    for start in range(0,dense.shape[0],tile_rows):
      part = dense[start:start+tile_rows]
      count = part.shape[0]
      assert isinstance(count,int)
      sdk = sdk_prefill and not self.k_parts and sdk_prefill_supported(count,round_up(self.columns,32),k)
      def _quantize_tile(start:int=start,count:int=count,sdk:bool=sdk) -> tuple[Tensor,Tensor]:
        if (ONPU_INT8_ROW_FOLD and not self.k_parts and self.input_scale is None
            and isinstance(dense.shape[0],int) and dense.shape[0]>count):
          return quantize(self.scaled_input(dense),k,sdk_prefill=sdk,rows=count,row_offset=start)
        return quantize(self.scaled_input(part),k,sdk_prefill=sdk)
      if share is not None and not sdk:
        payload,scale = _shared_quantize(share,(start,count,k),_quantize_tile)
      else:
        payload,scale = _quantize_tile()
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
      if not any(sdk_tiles):_register_prefill_dequant(out,raw_tiles,tile_scales,scales[0],counts,self.columns)
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
      from tinygrad.llm.kernels.rockchip_grouped_int8 import matmul as grouped_int8_matmul, matmul_split as grouped_int8_split
      dense = capacity_input(x)
      capacity = dense.shape[1]
      if isinstance(capacity,int) and 1<=capacity<=128 and (capacity>1 or (ONPU_GROUPED_DECODE_DOWN and self.columns==4096)):
        groups = self.grouped_count
        first = dense[...,:groups*4096]
        grouped_x = first.reshape(1,-1,groups,4096).squeeze(0).permute(1,0,2).contiguous()
        # The grouped dequantizer is a custom program, so the cross-group adds cannot
        # contract its final multiply into an FMA and change the reference FP32 sum.
        split = grouped_int8_split(grouped_x,self.grouped_weight,self.grouped_scale,4096) \
          if ONPU_PREFILL_RESIDUAL_FUSE else None
        if split is None:
          grouped = grouped_int8_matmul(grouped_x,self.grouped_weight,self.grouped_scale,4096,materialize=True)
          outputs = [grouped[index].reshape(*dense.shape[:-1],self.columns) for index in range(groups)]
        else:
          # Same quantize + raw_matmul calls as `matmul(materialize=True)`; only the
          # dequantize is deferred so the consumer can fold it. The per-group dequantize
          # below is the first step of the fold's fallback path.
          from tinygrad.llm.kernels.rockchip_grouped_int8 import dequantize as grouped_dequantize
          raw,activation_scales,weight_scales,padded = split
          outputs = [grouped_dequantize(raw,activation_scales,weight_scales,groups,capacity,4096,padded)[index]
                       .reshape(*dense.shape[:-1],self.columns) for index in range(groups)]
        result = outputs[0]
        for output in outputs[1:]:result = result+output
        tail_values = []
        offset = groups*4096
        for index in range(groups,len(self.parts)):
          part = self.parts[index]
          tail = part.npu(dense[...,offset:offset+part.depth].float().contiguous())
          tail_values.append(tail)
          result = result+tail
          offset += part.depth
        result = result.shrink(tuple((0,s) for s in (*x.shape[:-1],self.columns))).cast(x.dtype)
        # Record the summed grouped down output so the block-output residual add can fold it.
        # The first `groups` terms are row-major grouped raws (layout = padded stride) whose
        # per-group dequantizer is the `rockchip_grouped_int8` program; the remaining terms are
        # the standing blocked per-part dequantizers, all in the loop's summation order.
        rows = math.prod(int(n) for n in x.max_shape[:-1])
        if split is not None and self.bias is None and rows>1:
          # The grouped dequantizer emits the SAME row-major `(1, capacity, columns)` value the
          # standing per-part dequantizer would, so the groups are registered as the leading
          # parts of one sum. A fold needs every part of a sum to share one row-tile partition
          # (`_dequant_residual` derives a single tile index per output row), so adopt whatever
          # partition the tail parts registered; no tail part means one tile over capacity.
          tail_counts = {tuple(found['counts']) for found in
                         (_walk_registered(value.uop) for value in tail_values) if found is not None}
          tile_counts = tail_counts.pop() if len(tail_counts)==1 else ((capacity,) if not tail_counts else None)
          registered = False
          if tile_counts is not None:
            registered = _register_prefill_grouped(outputs,raw,activation_scales,weight_scales,
                                                   groups,capacity,tile_counts,self.columns,padded)
          if registered:
            # `result` is the grouped sum plus every tail part, i.e. exactly the value the
            # block-output residual add consumes, and the parts are listed in the reference
            # summation order. The tails are already registered part dequantizers.
            _register_prefill_sum(result,[_walk_registered(value.uop) for value in outputs]
                                        +[_walk_registered(value.uop) for value in tail_values],
                                  self.columns,rows)
        return result if self.bias is None else result+self.bias
    values = [part.npu(x[...,index*4096:index*4096+part.depth].float().contiguous()) for index,part in enumerate(self.parts)]
    result = values[0]
    for value in values[1:]:result = result+value
    result = result.cast(x.dtype)
    # Record the summed down output so the block-output residual add can be folded into one
    # dequantize+sum+add kernel (nothing is materialized when the consumer takes the fold).
    # Symbolic prefill graphs carry `toks` as a bound variable: register against the padded
    # (max-shape) row count so the block-output residual fuse can match its fixed-size kernel.
    rows = math.prod(int(n) for n in x.max_shape[:-1])
    if self.bias is None and rows>1 and len(self.parts)>1:
      _register_prefill_sum(result,[_walk_registered(value.uop) for value in values],self.columns,rows)
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
