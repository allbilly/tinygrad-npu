"""Qualify raw RK3588 INT8 matrices before selecting quantized model execution.

Register/layout reference: https://github.com/oRKLLM/ork-driver
Copyright (c) 2025-2026 Michael Fischer / oRKLLM
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.
THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
from tinygrad.helpers import ceildiv, getenv, round_up
from tinygrad.runtime.ops_rockchip import _cmd, _MATMUL_REGISTERS

NATURAL_DEPTHS = frozenset((576,768,1152,2304))
EXPERT_DEPTHS = frozenset((896,1280))
SMALL_DEPTHS = frozenset((64,128,256))
NARROW_DECODE_WIDTHS = frozenset((128,192,256,320,352,416,448,576,608,640,704,768,832,864,896,960))
SDK_PREFILL_SHAPES = frozenset(((64,2112,1024),(64,3072,2048),(64,12288,2048),(64,1024,2048),
                                (128,3072,1024),(128,6144,1024),(128,8192,1024),(128,9216,1024),(128,9728,896),
                                (128,12288,1536),(128,5632,1536),(128,12288,2048),(128,3072,2048),(128,1024,2048),
                                (128,1024,512)))

def sdk_prefill_supported(m:int,n:int,k:int) -> bool:return (m,n,k) in SDK_PREFILL_SHAPES

def int8_sdk_body(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[int,...]:
  """SDK prefill requires 16-channel input blocks and four-channel output blocks."""
  if (m,n,k) not in ((64,704,1024),(64,1024,2048),(128,1024,1024),(128,2048,1024),(128,2752,1024),(128,2688,1024),
                    (128,3072,1024),(128,3264,896),(128,3200,896)):
    raise ValueError('Unqualified SDK INT8 prefill geometry')
  changes={(0x201,0x100c):0,(0x201,0x1010):32,(0x201,0x1020):(m<<16)|1,
    (0x201,0x1028):m,(0x201,0x1044):m*k//64,(0x201,0x107c):k//4,
    (0x201,0x1080):(-3*m)&0xfffffff,(0x201,0x1084):(m<<16)|1,
    (0x1001,0x4024):m*16,(0x1001,0x4030):m-1,(0x1001,0x4034):0,
    (0x1001,0x4038):0,(0x1001,0x40c0):m*128}
  if k==896 or (m,n,k) in ((128,2048,1024),(128,2752,1024),(128,2688,1024),(128,3072,1024)):
    # Literal SDK M128 core/tail captures: the feature surface follows the
    # blocked M dimension; its stride is independent of K. Keep older recipes.
    changes.update({(0x201,0x107c):m*4,(0x801,0x3014):m-1,(0x1001,0x405c):m-1})
  return tuple(_cmd(w>>48,w&0xffff,changes[(w>>48,w&0xffff)])
               if (w>>48,w&0xffff) in changes else w for w in int8_body(m,n,k,lhs,rhs,dst,stride))

# Ordered SDK reset/configuration writes for the qualified split-row A16/C4 plans.
_SDK_SPLIT_REGISTERS = (
  (0x201,0x1040),(0x201,0x1104),(0x201,0x1100),(0x201,0x100c),(0x1001,0x4004),(0x201,0x100c),
  (0x201,0x1010),(0x201,0x1014),(0x201,0x1020),(0x201,0x1024),(0x201,0x1028),(0x201,0x102c),
  (0x201,0x1030),(0x201,0x1034),(0x201,0x1038),(0x201,0x1040),(0x201,0x1044),(0x201,0x104c),
  (0x201,0x1050),(0x201,0x1054),(0x201,0x1058),(0x201,0x105c),(0x201,0x1060),(0x201,0x1064),
  (0x201,0x1068),(0x201,0x1070),(0x201,0x1074),(0x201,0x1078),(0x201,0x107c),(0x201,0x1080),
  (0x201,0x1084),(0x201,0x1088),(0x201,0x1100),(0x201,0x1104),(0x201,0x1110),(0x201,0x1140),
  (0x201,0x1144),(0x201,0x1148),(0x201,0x114c),(0x201,0x1150),(0x201,0x1154),(0x201,0x1158),
  (0x201,0x115c),(0x201,0x1160),(0x201,0x1164),(0x201,0x1168),(0x201,0x116c),(0x201,0x1170),
  (0x201,0x1174),(0x201,0x1178),(0x201,0x117c),(0x201,0x1180),(0x201,0x1184),(0x801,0x3010),
  (0x801,0x3014),(0x801,0x3018),(0x801,0x301c),(0x801,0x3030),(0x1001,0x400c),(0x1001,0x4010),
  (0x1001,0x4014),(0x1001,0x4020),(0x1001,0x4024),(0x1001,0x4030),(0x1001,0x4034),(0x1001,0x4038),
  (0x1001,0x403c),(0x1001,0x4040),(0x1001,0x4044),(0x1001,0x4048),(0x1001,0x404c),(0x1001,0x4050),
  (0x1001,0x4054),(0x1001,0x4058),(0x1001,0x405c),(0x1001,0x4060),(0x1001,0x4064),(0x1001,0x4068),
  (0x1001,0x406c),(0x1001,0x4070),(0x1001,0x4074),(0x1001,0x4078),(0x1001,0x407c),(0x1001,0x4080),
  (0x1001,0x4084),(0x1001,0x4088),(0x1001,0x4090),(0x1001,0x4094),(0x1001,0x4098),(0x1001,0x409c),
  (0x1001,0x40a0),(0x1001,0x40a4),(0x1001,0x40a8),(0x1001,0x40ac),(0x1001,0x40c0),(0x1001,0x40c4),
  (0x1001,0x4100),(0x1001,0x4104),(0x1001,0x4108),(0x1001,0x410c),(0x1001,0x4110),(0x1001,0x4114),
  (0x1001,0x4118),(0x1001,0x411c),(0x1001,0x4120),(0x1001,0x4124),(0x1001,0x4128),(0x1001,0x412c),
)

def int8_sdk_decode_body(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[int,...]:
  """SDK A16/C4 register recipe for single-row decode.

  RKLLM's captured decode task (m=1,n=1536,k=4096) matches this recipe on
  104/104 command words, while the default int8_body() recipe differs on four
  registers (0x100c, 0x107c, 0x1080, 0x4038). Those four are exactly the
  A16/C4 surface controls the SDK prefill path already emits, so the same
  values apply at m=1; no new geometry is involved.

  Kept behind an env flag so it can be A/B'd against the default recipe: the
  register recipe being identical does not prove it is faster on this part.
  """
  if not (m==1 and supported_depth(k) and 32<=n<=8192 and n%32==0 and stride==n):
    raise ValueError('Unqualified SDK INT8 decode geometry')
  values={(w>>48,w&0xffff):(w>>16)&0xffffffff for w in int8_body(m,n,k,lhs,rhs,dst,stride)}
  values.update({(0x201,0x100c):0,(0x201,0x1010):32,(0x201,0x1020):(m<<16)|1,
    (0x201,0x1028):m,(0x201,0x1044):m*k//64,(0x201,0x107c):m*4,
    (0x201,0x1080):(-3*m)&0xfffffff,(0x201,0x1084):(m<<16)|1,
    (0x801,0x3014):m-1,(0x1001,0x4024):m*16,(0x1001,0x4030):m-1,
    (0x1001,0x4034):0,(0x1001,0x4038):0,(0x1001,0x405c):m-1,(0x1001,0x40c0):m*128})
  return tuple(_cmd(w>>48,w&0xffff,values[(w>>48,w&0xffff)])
               if (w>>48,w&0xffff) in values else w
               for w in int8_body(m,n,k,lhs,rhs,dst,stride))


def int8_sdk_bodies(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[tuple[int,...],...]:
  if (m,n,k) in ((64,4096,2048),(64,352,2048),(64,320,2048),(128,352,512),(128,320,512)):
    if stride!=n:raise ValueError('Unqualified half-row SDK INT8 output stride')
    # Actual M64 and M128/K512 captures use one task and an A16/C4 surface. Retain the
    # older M64/N1024 recipe: its full three-core shape is a separate cohort.
    values={(w>>48,w&0xffff):(w>>16)&0xffffffff for w in int8_body(m,n,k,lhs,rhs,dst,stride)}
    values.update({(0x201,0x100c):0,(0x201,0x1010):32,(0x201,0x1020):(m<<16)|1,
      (0x201,0x1028):m,(0x201,0x1044):m*k//64,(0x201,0x107c):m*4,
      (0x201,0x1080):(-3*m)&0xfffffff,(0x201,0x1084):(m<<16)|1,
      (0x801,0x3014):m-1,(0x1001,0x4024):m*16,(0x1001,0x4030):m-1,
      (0x1001,0x4034):0,(0x1001,0x4038):0,(0x1001,0x405c):m-1,(0x1001,0x40c0):m*128})
    return (tuple(_cmd(t,r,values.get((t,r),0)) for t,r in _SDK_SPLIT_REGISTERS),)
  if k not in (1536,2048) or k==2048 and m!=128:return (int8_sdk_body(m,n,k,lhs,rhs,dst,stride),)
  widths=(4096,1888,1856) if k==1536 else (4096,1024,352,320)
  if m!=128 or n not in widths or stride!=n:raise ValueError('Unqualified split SDK INT8 geometry')
  result=[]
  for offset,rows in (((84,44),(0,84)) if k==1536 else ((64,64),(0,64))):
    values={(w>>48,w&0xffff):(w>>16)&0xffffffff for w in int8_body(rows,n,k,lhs+offset*16,rhs,dst+offset*16,stride)}
    banks=ceildiv(rows*k,32768)
    values.update({(0x201,0x1040):((12-banks)<<4)|banks,
      (0x201,0x1010):32,(0x201,0x1020):(rows<<16)|1,(0x201,0x1028):rows,
      (0x201,0x1044):rows*k//64,(0x201,0x107c):m,(0x201,0x1080):m-rows,(0x201,0x1084):(rows<<16)|1,
      (0x801,0x3014):rows-1,(0x1001,0x4024):m*16,(0x1001,0x4030):rows-1,
      (0x1001,0x4034):0,(0x1001,0x4038):0,(0x1001,0x405c):rows-1,(0x1001,0x40c0):m*128})
    result.append(tuple(_cmd(t,r,values.get((t,r),0)) for t,r in _SDK_SPLIT_REGISTERS))
  return tuple(result)


def supported_depth(k:int) -> bool:
  return k in SMALL_DEPTHS or 512<=k<=4096 and (k%512==0 or k in NATURAL_DEPTHS or k in EXPERT_DEPTHS)

def padded_depth(k:int) -> int:
  if 1<=k<=256 and getenv('ONPU_INT8_SMALL_DEPTH',0):return next(depth for depth in sorted(SMALL_DEPTHS) if depth>=k)
  if k in EXPERT_DEPTHS and getenv('ONPU_INT8_EXPERT_DEPTHS',0):return k
  return k if k in NATURAL_DEPTHS and (k!=2304 or getenv('ONPU_INT8_NATURAL2304',1)) else round_up(k,512)

def int8_body(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[int,...]:
  if not (1<=m<=(128 if k<=2048 else 64) and supported_depth(k)
          and 32<=n<=(8192 if m==1 else 4096) and n%32==0 and stride==n):
    raise ValueError('Unqualified INT8 matrix geometry')
  capacity = 1 << ((131072//k).bit_length()-1)
  # The qualified 64-aligned depths retain the next 512-channel shape's CBUF
  # allocation. Integer golden tests cover narrow, wide and 128-row execution.
  groups = ceildiv(k,512)
  banks = max(0x1b,177-15*(groups-1)-15*groups*(ceildiv(m,64)-1))
  # One-row wide-K decode needs only one data bank. Retain eleven weight banks
  # for the qualified wide outputs (extra/rockchip/int8_decode_cbuf.md).
  # RKLLM target captures (2026-11-06) show CBUF_CON0=0xb1 for every single-row
  # decode matmul across k=768/2560/4096 and n=96..8128 (ChatGLM3 and MiniCPM3).
  # Match that allocation for the captured K set; 64-row and prefill rows keep
  # the earlier recipe.
  if m==1 and k==4096 and (n==512 or n>=1024 or n in NARROW_DECODE_WIDTHS and getenv('ONPU_INT8_DECODE_NARROW_CBUF',0)):
    banks = 0xb1
  C,O,D = 0x201,0x801,0x1001
  values = {
    (D,0x4004):0xe,(C,0x100c):1<<29,(C,0x1010):16*min(m+1,capacity),(C,0x1014):9,
    (C,0x1020):(1<<16)|m,(C,0x1024):((k-1)<<16)|k,(C,0x1028):1,(C,0x102c):m,
    (C,0x1030):k*n,(C,0x1034):k,(C,0x1038):0x1010000|n,(C,0x1040):banks,
    (C,0x1044):ceildiv(k,64),(C,0x104c):11,
    **{(C,r):1<<16 for r in (0x1050,0x1054,0x1058,0x105c)},
    (C,0x1070):lhs,(C,0x1078):0xf000f,(C,0x107c):k//16,(C,0x1084):(1<<16)|m,
    (C,0x1088):k,(C,0x1110):rhs,
    (O,0x3010):1,(O,0x3014):(m-1)<<16,(O,0x3018):n-1,
    (D,0x400c):0x1e4,(D,0x4010):0x80000000,(D,0x4020):dst,(D,0x4024):16,
    (D,0x4034):m-1,(D,0x4038):((stride//4-1)<<16)|(n//4-1),
    (D,0x403c):((stride-1)<<16)|(n-1),(D,0x4040):0x53,(D,0x4050):0x7fc,
    (D,0x4058):n-1,(D,0x405c):(m-1)<<16,(D,0x4060):0x53,(D,0x4070):0x383,
    (D,0x4078):1,(D,0x4084):1,(D,0x40c0):0x80}
  return tuple(_cmd(t,r,values.get((t,r),0)) for t,registers in _MATMUL_REGISTERS for r in registers)
