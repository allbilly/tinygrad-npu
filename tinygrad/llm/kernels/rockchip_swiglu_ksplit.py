"""Exact static128 SwiGLU with K2048 tasks and original K4096 quantization scales.

Only the guarded 896->4864->896 and 1024->4608->1024 geometries are prepared. Partial/dynamic chunks,
scalar decode and unsupported inputs retain the ordinary fused path. The two
INT32 partials are added before either FP32 conversion or dequantization.
"""

from __future__ import annotations
import functools, re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  from tinygrad.llm.kernels.rockchip_gelu_ffn import _RockchipInt8GatedFeedForward
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar, getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
from tinygrad.llm.kernels import rockchip_gelu_ffn as f
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_batch4_source
from tinygrad.llm.kernels.rockchip_int8 import _quantize_vector_source
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear, quantize, raw_int8_matmul, dequantize_tiles
from tinygrad.llm.kernels.rockchip_padding import capacity_input

SIZE = 624128
ONPU_SWIGLU_K_SPLIT_TAIL_SDK = ContextVar('ONPU_SWIGLU_K_SPLIT_TAIL_SDK',0)
_prepared_blocks = 0
_pack_traces = 0
_extra_weight_bytes = 0
_hidden = 0


def split_prefill_stats() -> dict[str, int]:
  return {"prepared_blocks": _prepared_blocks, "traces": _pack_traces, "extra_weight_bytes": _extra_weight_bytes, "hidden": _hidden}


def _packing_hidden(tiles: tuple[f.ProjectionTile, ...]) -> int:
  for hidden in (4864,4608):
    if tiles == (f.ProjectionTile(0,128,0,hidden*2,0,True),):return hidden
  raise ValueError("Unsupported full static split projection")


def _packet_size(hidden:int) -> int:return 525312+512+128*(hidden-4096)


def source(tiles: tuple[f.ProjectionTile, ...], address: int, max_bits: bool = False, sdk_tail:bool=False) -> str:
  hidden = _packing_hidden(tiles)
  if sdk_tail and hidden!=4608:raise ValueError("Unqualified SDK split tail")
  loops = []
  for group, (first, depth) in enumerate(((0, 4096), (4096, hidden-4096))):
    project = f._batched_projection_loop(tiles, 128, hidden, first, depth, "silu", 4, True, 16, True)
    packing = []
    for lane in range(4):
      segments = ((0, 0, 2048), (262656, 2048, 2048)) if group == 0 else ((525312, 0, hidden-4096),)
      stores = []
      for offset, begin, count in segments:
        position = f'((c-{begin})/16*128+r)*16+(c-{begin})%16' if sdk_tail and group==1 else f'r*{count}+c-{begin}'
        stores.append(
          f"((float*)(out+{offset}))[r]=header;for(int c={begin};c<{begin + count};c+=4){{"
          f"float4 v=invalid?(float4)0:*(const float4*)((const float*)batch_values+{lane * 4}+c*4);"
          f"*(char4*)(out+{offset + 512}+{position})=rockchip_quantize_bounded_vector(v,inverse);}}"
        )
      packing.append(
        f'{{int r=row+{lane};float4 maximum=batch_maximum[{lane}];uint4 bad=batch_bad[{lane}];'
        'float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];'
        'int invalid=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;'
        f'float header=invalid?__builtin_nanf(""):scale;scales[{group * 128}+r]=header;float inverse=invalid?1.0f:1.0f/scale;'
        + "".join(stores)
        + "}"
      )
    loops.append(
      f"for(int row=worker*32;row<(worker+1)*32;row+=4){{float4 batch_values[{depth // 4}][4] __attribute__((aligned(16)));"
      "float4 batch_maximum[4]={0};uint4 batch_bad[4]={0};"
      + project
      + "".join(packing)
      + "}"
    )
  text:str = (
    f._vector_exp2_source()
    + _vector_exp2_batch4_source()
    + _quantize_vector_source()
    + r"""
 static __attribute__((always_inline)) char4 rockchip_quantize_bounded_vector(float4 x,float inverse) {
   #if defined(__aarch64__)
   float4 scaled=x*(float4)inverse;int4 q;
   __asm__("fcvtns %0.4s, %1.4s":"=w"(q):"w"(scaled));return __builtin_convertvector(q,char4);
   #else
   return rockchip_quantize_vector(x,inverse);
   #endif
 }
  typedef unsigned uint4 __attribute__((aligned(16),ext_vector_type(4)));
 typedef struct{unsigned char *out;const int *r;const float *a,*sw;float *scales;} Work;
 static void part(void **args,int unused,int worker){Work *w=args[0];if((unsigned)worker>=4)__builtin_trap();
 unsigned char *out=w->out;const int *raw[1]={w->r};const float *sa[1]={w->a};const float *sw=w->sw;float *scales=w->scales;
 """
    + "".join(loops)
    + r"""
 }
 void rockchip_int8_swiglu_pack(unsigned char*out,const int*r,const float*a,const float*sw,float*scales){
 Work w={out,r,a,sw,scales};void *args[4]={&w,0,0,0};
 if(((int(*)(void(*)(void**,int,int),void**,int))ADDRESSULL)(part,args,0))__builtin_trap();}
 """.replace("ADDRESS", str(address))
  )
  if max_bits:
    text = text.replace("float4 batch_maximum", "uint4 batch_maximum").replace("float4 maximum", "uint4 maximum")
    pattern = (r"uint4 bits=__builtin_bit_cast\(uint4,value\);(batch_bad\[\d+\]|bad)\|="
               r"\(bits&\(uint4\)0x7f800000\)==\(uint4\)0x7f800000;\s*float4 a=value>=\(float4\)0\?value:-value;"
               r"(batch_maximum\[\d+\]|maximum)=a>(batch_maximum\[\d+\]|maximum)\?a:(batch_maximum\[\d+\]|maximum);")
    def maximum_bits(match:re.Match[str]) -> str:
      target=match[2]
      return 'uint4 bits=__builtin_bit_cast(uint4,value)&(uint4)0x7fffffff;'+target+'=bits>'+target+'?bits:'+target+';'
    text,count=re.subn(pattern,maximum_bits,text)
    assert count >= 8, count
    text = text.replace(
      "float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];",
      "unsigned largest_bits=__builtin_reduce_max(maximum);float largest=__builtin_bit_cast(float,largest_bits);",
    ).replace("int invalid=__builtin_reduce_or(bad)!=0;", "int invalid=largest_bits>=0x7f800000;")
  return text


@functools.cache
def _pack_program(out: UOp, raw: UOp, sa: UOp, sw: UOp, scales: UOp, tiles: tuple[f.ProjectionTile, ...], address: int,
                  max_bits: bool, sdk_tail:bool=False) -> UOp:
  global _pack_traces, _hidden
  _hidden = _packing_hidden(tiles)
  _pack_traces += 1
  text = source(tiles, address, max_bits, sdk_tail)
  inputs = (out, raw, sa, sw, scales)
  sink = UOp.sink(*inputs, arg=KernelInfo("rockchip_int8_swiglu_pack"))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0, 4), ins=(1, 2, 3))
  return UOp(
    Ops.PROGRAM,
    arg=info,
    src=(sink, UOp(Ops.LINEAR, src=inputs), UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(text))),
  )


def pack(
  raw: list[Tensor], sa: list[Tensor], sw: Tensor, tiles: list[f.ProjectionTile], pool: AttentionPool, *,
  max_bits: bool = True, sdk_tail:bool=False
) -> tuple[Tensor, Tensor]:
  hidden = _packing_hidden(tuple(tiles))
  if type(sdk_tail) is not bool or sdk_tail and hidden!=4608:raise ValueError("Unqualified SDK split tail")
  if (
    len(raw) != len(sa)
    or len(raw) != 1
    or not pool.active
    or raw[0].device != "CPU"
    or raw[0].dtype is not dtypes.int32
    or raw[0].numel() != 128 * hidden * 2
    or sa[0].device != "CPU"
    or sa[0].dtype is not dtypes.float
    or sa[0].shape != (128,)
    or sw.device != "CPU"
    or sw.dtype is not dtypes.float
    or sw.shape != (hidden*2,)
  ):
    raise ValueError("Unsupported full static split packing")
  result = Tensor.empty(_packet_size(hidden), device="CPU", dtype=dtypes.uint8).custom_kernel(
    raw[0],
    sa[0],
    sw,
    Tensor.empty(256, device="CPU", dtype=dtypes.float),
    fxn=lambda o, r, a, w, s: _pack_program(o, r, a, w, s, tuple(tiles), pool.prefill_address, max_bits, sdk_tail),
  )
  return result[0], result[4]


@functools.cache
def _combine_program(out: UOp, a: UOp, b: UOp, sa: UOp, sw: UOp, address: int, columns:int) -> UOp:
  if columns not in (896,1024):raise ValueError("Unsupported split output width")
  loops = []
  for first, width in (((0,320),(320,320),(640,256)) if columns==896 else ((0,352),(352,352),(704,320))):
    loops.append(
      f"for(int r=begin;r<end;r++)for(int c=0;c<{width};c+=4){{int i={first * 128}+r*{width}+c;"
      "int4 s=*(const int4*)(a+i)+*(const int4*)(b+i);float4 v=__builtin_convertvector(s,float4);"
      f"*(float4*)(out+r*{columns}+{first}+c)=(v*(float4)sa[r])*(*(const float4*)(sw+{first}+c));}}"
    )
  text = (
    "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));"
    "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));"
    "static void part(float*out,const int*a,const int*b,const float*sa,const float*sw,int begin,int end){"
    + "".join(loops)
    + "}"
  )
  text += ("typedef struct{float*out;const int*a,*b;const float*sa,*sw;} Work;"
           "static void callback(void **args,int unused,int worker){Work*w=args[0];"
           "part(w->out,w->a,w->b,w->sa,w->sw,worker*32,(worker+1)*32);}")
  dispatch = (
    f"Work w={{out,a,b,sa,sw}};void*args[4]={{&w,0,0,0}};"
    f"if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(callback,args,0))__builtin_trap();"
    if address
    else "part(out,a,b,sa,sw,0,128);"
  )
  text += "void rockchip_k2_dequantize(float*out,const int*a,const int*b,const float*sa,const float*sw){" + dispatch + "}"
  inputs = (out, a, b, sa, sw)
  sink = UOp.sink(*inputs, arg=KernelInfo("rockchip_k2_dequantize"))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2, 3, 4))
  return UOp(
    Ops.PROGRAM,
    arg=info,
    src=(sink, UOp(Ops.LINEAR, src=inputs), UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(text))),
  )


def combine(a: Tensor, b: Tensor, sa: Tensor, sw: Tensor, pool: AttentionPool | None) -> Tensor:
  columns = sw.shape[0] if sw.ndim==1 else 0
  if not isinstance(columns,int):raise ValueError("Split down shape changed")
  if (columns not in (896,1024) or a.shape != (128*columns,) or b.shape != (128*columns,) or sa.shape != (128,)
      or any(x.device!='CPU' for x in (a,b,sa,sw)) or a.dtype is not dtypes.int32 or b.dtype is not dtypes.int32
      or sa.dtype is not dtypes.float or sw.dtype is not dtypes.float):
    raise ValueError("Split down shape changed")
  return Tensor.empty(128, columns, device="CPU", dtype=dtypes.float).custom_kernel(
    a, b, sa, sw, fxn=lambda o, a, b, s, w: _combine_program(o, a, b, s, w, pool.prefill_address if pool is not None and pool.active else 0, columns)
  )[0]


class SplitSwiGLUPrefill:
  def __init__(self, ffn: _RockchipInt8GatedFeedForward):
    global _prepared_blocks, _extra_weight_bytes, _hidden
    assert isinstance(ffn.projections.joint, RockchipInt8Linear) and isinstance(ffn.down, RockchipInt8SplitLinear)
    assert ffn.pool is not None
    self.joint, self.parts, self.pool = ffn.projections.joint, tuple(ffn.down.parts), ffn.pool
    self.phase_layer = ffn.projections.layers[0]
    self.width, self.hidden = self.joint.depth, ffn.down.depth
    _hidden = self.hidden
    self.original = self.parts[0].packed[0]
    host = self.original.to("CPU").realize()
    self.weights = tuple(host[:, first : first + 64].contiguous().to(self.original.device).realize() for first in (0, 64))
    _prepared_blocks += 1
    for w in self.weights:
      count=w.numel()
      assert isinstance(count,int)
      _extra_weight_bytes+=count*w.dtype.itemsize

  def enabled(self, x: Tensor) -> bool:
    return (
      self.pool.active
      and self.phase_layer.phase == "prefill"
      and x.device == "CPU"
      and x.dtype is dtypes.float
      and all(type(v) is int for v in x.shape)
      and x.shape == (1, 128, self.width)
      and self.parts[0].packed[0] is self.original
    )

  def __call__(self, x: Tensor) -> Tensor:
    sdk_tail=bool(ONPU_SWIGLU_K_SPLIT_TAIL_SDK) and self.width==1024 and self.hidden==4608
    dense = capacity_input(x).reshape(128, self.width)
    payload, sa = quantize(self.joint.scaled_input(dense), self.width, sdk_prefill=True)
    raw = raw_int8_matmul(payload, self.joint.packed[0], 128, self.hidden*2, sdk_prefill=True)
    packed, scales = pack([raw], [sa], self.joint.scales[0], [f.ProjectionTile(0, 128, 0, self.hidden*2, 0, True)], self.pool,
                          sdk_tail=sdk_tail)
    packed = packed.to(self.parts[0].packed[0].device)
    a = raw_int8_matmul(packed, self.weights[0], 128, self.width, payload_offset=0)
    b = raw_int8_matmul(packed, self.weights[1], 128, self.width, payload_offset=262656)
    big = combine(a, b, scales[:128].contiguous(), self.parts[0].scales[0], self.pool)
    tail = raw_int8_matmul(packed, self.parts[1].packed[0], 128, self.width, payload_offset=525312,sdk_prefill=sdk_tail)
    last = dequantize_tiles([tail], [scales[128:].contiguous()], self.parts[1].scales[0], [128], self.width,
                            sdk_tiles=(True,) if sdk_tail else ())
    return (big + last).reshape(1, 128, self.width)


def prepare_split_prefill(ffn: _RockchipInt8GatedFeedForward) -> SplitSwiGLUPrefill | None:
  joint, down = ffn.projections.joint, ffn.down
  if (
    ffn.activation != "silu"
    or ffn.pool is None
    or not f.ONPU_SWIGLU_BATCHED_PACK
    or not f.ONPU_SWIGLU_CHANNEL_SCRATCH
    or f.ONPU_SWIGLU_C4_PREFETCH.value!=16
    or getenv('ONPU_SWIGLU_PREFILL_ROWS',1)!=4
    or not getenv('ONPU_INT8_SDK_PREFILL',0)
    or not getenv('ONPU_FFN_PACKED_UPLOAD',0)
    or not isinstance(joint, RockchipInt8Linear)
    or not isinstance(down, RockchipInt8SplitLinear)
    or (joint.depth,down.depth) not in ((896,4864),(1024,4608))
    or joint.columns != down.depth*2
    or down.columns != joint.depth
    or len(joint.packed) != 1
    or len(joint.scales) != 1
    or joint.bias is not None
    or down.bias is not None
    or down.input_scale is not None
    or len(down.parts) != 2
    or tuple(p.depth for p in down.parts) != (4096, down.depth-4096)
    or any(len(p.packed) != 1 or len(p.scales) != 1 or p.input_scale is not None or p.bias is not None for p in down.parts)
    or any(p.packed[0].device != joint.packed[0].device for p in down.parts)
  ):
    return None
  return SplitSwiGLUPrefill(ffn)
