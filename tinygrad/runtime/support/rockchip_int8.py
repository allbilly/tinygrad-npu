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
from tinygrad.helpers import ceildiv, round_up
from tinygrad.runtime.ops_rockchip import _cmd, _MATMUL_REGISTERS

NATURAL_DEPTHS = frozenset((576,768,1152))

def supported_depth(k:int) -> bool: return 512<=k<=4096 and (k%512==0 or k in NATURAL_DEPTHS)

def padded_depth(k:int) -> int: return k if k in NATURAL_DEPTHS else round_up(k,512)

def int8_body(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[int,...]:
  if not (1<=m<=(128 if k<=2048 else 64) and supported_depth(k)
          and 32<=n<=(8192 if m==1 else 4096) and n%32==0 and stride==n):
    raise ValueError('Unqualified INT8 matrix geometry')
  capacity = 1 << ((131072//k).bit_length()-1)
  # The qualified 64-aligned depths retain the next 512-channel shape's CBUF
  # allocation. Integer golden tests cover narrow, wide and 128-row execution.
  groups = ceildiv(k,512)
  banks = max(0x1b,177-15*(groups-1)-15*groups*(ceildiv(m,64)-1))
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
