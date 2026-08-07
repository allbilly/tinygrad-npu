from __future__ import annotations
# ruff: noqa: E702
import base64, math, os, struct
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Callable
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import cdiv, cmod, floordiv, floormod
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 17
_HEADER = struct.Struct("<4sHHHHIII")  # magic, version, target, scratch, gathers, ops, constants, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBiHI"), struct.Struct("<IIi")
_FILL = struct.Struct("<BBHI")  # dst_kind, pad, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, pad, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg

class RKTarget(IntEnum): RK3588 = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True)
class RKScratch: size: int; alignment: int = 4096

@dataclass(frozen=True)
class RKGather:
  """Materialize an affine or fallback index map into contiguous FP16 scratch."""
  src_index: int; dst_scratch: int; count: int; base: int = 0
  axes: tuple[tuple[int, int, int], ...] = ()  # dst divisor, range limit, source stride
  offsets: tuple[int, ...] = ()
  fill_bits: int = 0
  values: tuple[int, ...] = ()  # compile-time FP16 vector, no source argument
  partial: bool = False  # skip negative offsets, preserving lanes written by another gather
  dst_stride: int = 1  # destination stride in FP16 lanes; scalar reductions use 32 for 64-byte alignment

@dataclass(frozen=True)
class RKMultiGather: gathers: tuple[RKGather, ...]

@dataclass(frozen=True)
class RKStatic: expr: UOp

RKLeaf = RKArg|RKStatic|RKGather|RKMultiGather|float|tuple[UOp, UOp, UOp|None, int]|None

@dataclass(frozen=True)
class RKFill: dst: RKArg; count: int

@dataclass(frozen=True)
class RKEWOp:
  """One contiguous FP16 DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); fill: RKFill|None = None; ew_ops: tuple[RKEWOp, ...] = ()
  chain_limit: int = 383

@dataclass(frozen=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  flags = (1 if image.fill is not None else 0) | (image.chain_limit << 8)
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(image.gathers),
                               len(image.ew_ops), len(image.constants), flags))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in image.gathers:
    kind = 3 if g.partial else 2 if g.values else 1 if g.offsets else 0
    out += _GATHER.pack(g.dst_scratch, g.src_index, g.count, kind, len(g.axes), g.base, g.fill_bits, g.dst_stride)
    if kind == 2: out += struct.pack(f"<{g.count}H", *g.values)
    elif kind in (1, 3): out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for op in image.ew_ops:
    out += _EWOP.pack(int(op.dst.kind), 0, op.dst.index, int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index)
    out += _EWOP2.pack(op.count, op.ew_cfg) + struct.pack("<iii", op.dst.addend, op.lhs.addend, op.rhs.addend)
  if image.fill is not None: out += _FILL.pack(int(image.fill.dst.kind), 0, image.fill.dst.index, image.fill.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nop, nconst, flags = _HEADER.unpack_from(blob)
  chain_limit = flags >> 8
  if magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or flags & 0xfe or chain_limit not in (256, 383):
    raise ValueError("invalid RKImage header")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_scratch, src_index, count, kind, naxes, base, fill_bits, dst_stride = _GATHER.unpack_from(blob, off); off += _GATHER.size
    if kind not in (0, 1, 2, 3) or (kind and naxes) or dst_stride < 1: raise ValueError("invalid RKGather")
    if kind == 2:
      values = struct.unpack_from(f"<{count}H", blob, off); off += 2*count
      gathers.append(RKGather(src_index, dst_scratch, count, fill_bits=fill_bits, values=values, dst_stride=dst_stride))
    elif kind in (1, 3):
      offsets = struct.unpack_from(f"<{count}i", blob, off); off += 4*count
      gathers.append(RKGather(src_index, dst_scratch, count, offsets=offsets, fill_bits=fill_bits, partial=kind == 3, dst_stride=dst_stride))
    else:
      axes = tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)); off += naxes*_GATHER_AXIS.size
      gathers.append(RKGather(src_index, dst_scratch, count, base, axes, fill_bits=fill_bits, dst_stride=dst_stride))
  ew_ops:list[RKEWOp] = []
  for _ in range(nop):
    dk, _, di, lk, li, rk_, ri = _EWOP.unpack_from(blob, off); off += _EWOP.size
    count, ew_cfg = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg))
  fill = None
  if flags & 1:
    dst_kind, _, dst_index, count = _FILL.unpack_from(blob, off); off += _FILL.size
    fill = RKFill(RKArg(RKBufferKind(dst_kind), dst_index), count)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  return RKImage(RKTarget(target), scratch, blob[off:], version, tuple(gathers), fill, tuple(ew_ops), chain_limit)

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for reloc in stage.relocs:
    word = commands[reloc.word]
    value = (address(reloc.arg.kind, reloc.arg.index) + reloc.arg.addend) & 0xffffffff
    commands[reloc.word] = (word & ~0xffffffff0000) | (value << 16)
  return tuple(commands)

_DPU, _RDMA = 0x1001, 0x2001
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_EW_DATA_MODE_FP16 = 1 << 28
_EW_EDATA_SIZE_FP16 = 2 << 22
_EW_ALU_MIN, _EW_ALU_ADD, _EW_ALU_SUB = 1 << 16, 2 << 16, 4 << 16
_EW_RELUX_EN, _EW_RELU_BYPASS, _EW_OP_CVT_BYPASS = 1 << 10, 1 << 9, 1 << 8
_EW_LUT_BYPASS, _EW_OP_SRC_DMA, _EW_MUL_PRELU = 1 << 7, 1 << 6, 1 << 5
_EW_OP_TYPE_MUL, _EW_OP_BYPASS, _EW_BYPASS = 1 << 2, 1 << 1, 1 << 0
_EW_CFG_COMMON = _EW_DATA_MODE_FP16 | _EW_EDATA_SIZE_FP16 | _EW_LUT_BYPASS | _EW_OP_SRC_DMA
_EW_CFG_RELU = _EW_CFG_COMMON
_EW_CFG_RELU6 = _EW_CFG_COMMON | _EW_RELUX_EN
_EW_CFG_MIN = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_MIN
_EW_CFG_SUB = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_SUB
_EW_CFG_LEAKY_RELU = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_MUL_PRELU | _EW_OP_TYPE_MUL
_EW_CFG_FDIV = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | (3 << 16)
_EW_CFG_MAX = _EW_CFG_COMMON | _EW_RELU_BYPASS
_EW_CFG_BYPASS = _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_LUT_BYPASS | _EW_OP_BYPASS | _EW_BYPASS
_EW_RELUX_CMP_RELU6 = struct.unpack("<I", struct.pack("<f", 6.0))[0]
_EW_STAGE_POSITIVE_MASK = 1 << 31  # software tag consumed by emit_ew_stage, never written to EW_CFG
_EW_STAGE_ABS = 1 << 30  # software tag for BS PReLU slope -1, never written to EW_CFG
_EW_CFG = {
  Ops.ADD: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ADD,
  Ops.MUL: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_OP_TYPE_MUL,
  Ops.MAX: _EW_CFG_MAX,
  Ops.FDIV: _EW_CFG_FDIV,
}
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)

_BS_BYPASS, _BS_ALU_BYPASS, _BS_MUL_BYPASS = 1, 1 << 1, 1 << 4
_BS_MUL_PRELU, _BS_RELU_BYPASS, _BS_RELUX_EN = 1 << 5, 1 << 6, 1 << 7
_BS_ALU_SUB = 4 << 16
_BS_BN_BYPASS = _BS_BYPASS | _BS_ALU_BYPASS | _BS_MUL_BYPASS | _BS_RELU_BYPASS
_BS_MASK_CFG = _BS_ALU_SUB | _BS_RELU_BYPASS
_BN_MASK_CFG = _BS_ALU_SUB | _BS_ALU_BYPASS | _BS_RELUX_EN
_BS_OW_OD_BYPASS = 1 << 1
_FP32_MIN_FP16_SUBNORMAL = struct.unpack("<I", struct.pack("<f", 2**-24))[0]
_FP32_ONE = struct.unpack("<I", struct.pack("<f", 1.0))[0]
_FP16_TWO_MUL = struct.unpack("<H", struct.pack("<e", 2.0))[0] << 16
_FP16_POS_INF_MUL = struct.unpack("<H", struct.pack("<e", float("inf")))[0] << 16
_FP16_NEG_ONE_MUL = struct.unpack("<H", struct.pack("<e", -1.0))[0] << 16

def _emit_positive_mask_stage(dst:RKArg, src:RKArg, count:int) -> RKStage:
  """Map positive FP16 lanes to 1 and non-positive lanes to 0 using DPU BS/BN."""
  width = (count + 7) // 8 - 1
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7),
    (_DPU,rk.REG_DPU_BS_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,_BS_OW_OD_BYPASS),(_DPU,rk.REG_DPU_WDMA_SIZE_0,7),(_DPU,rk.REG_DPU_WDMA_SIZE_1,width),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,0),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0),
    (_DPU,rk.REG_DPU_BS_CFG,_BS_MASK_CFG),(_DPU,rk.REG_DPU_BS_ALU_CFG,_FP32_MIN_FP16_SUBNORMAL),
    (_DPU,rk.REG_DPU_BS_MUL_CFG,_FP16_TWO_MUL),(_DPU,rk.REG_DPU_BN_CFG,_BN_MASK_CFG),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,_FP16_POS_INF_MUL),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,_FP32_ONE),
    (_DPU,rk.REG_DPU_EW_CFG,_EW_CFG[Ops.MAX]|_EW_BYPASS),(_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),
    (_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,(1<<16)|1),(_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),
    (_DPU,rk.REG_DPU_SURFACE_ADD,1<<6),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,src),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,src)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, (2<<15)|(15<<11)|(2<<5)|(1<<3)|1))
  return RKStage(tuple(commands), tuple(relocs))

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, stateful:bool=False) -> RKStage:
  """Build one contiguous FP16 DPU EW command body without its PC-chain tail."""
  if not (0 < count <= _MAX_EW_ELEMS_FP16): raise ValueError(f"EW fp16 count {count} out of range")
  if ew_cfg == _EW_STAGE_POSITIVE_MASK: return _emit_positive_mask_stage(dst, lhs, count)
  abs_stage = ew_cfg == _EW_STAGE_ABS
  width = (count + 7) // 8 - 1
  pipeline = ((_DPU,rk.REG_DPU_BS_CFG,(_BS_RELU_BYPASS|_BS_MUL_PRELU|_BS_ALU_BYPASS) if abs_stage else _BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BS_ALU_CFG,0),
    (_DPU,rk.REG_DPU_BS_MUL_CFG,_FP16_NEG_ONE_MUL if abs_stage else 0),(_DPU,rk.REG_DPU_BS_OW_CFG,_BS_OW_OD_BYPASS),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,7),(_DPU,rk.REG_DPU_WDMA_SIZE_1,width),(_DPU,rk.REG_DPU_BN_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0)) if stateful or abs_stage else ()
  conversion = ((_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,1<<6)) if stateful or abs_stage or ew_cfg == _EW_CFG_FDIV else ()
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7)) + pipeline + (
    (_DPU,rk.REG_DPU_EW_CFG,_EW_CFG_BYPASS if abs_stage else ew_cfg),
    (_DPU,rk.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6)) + conversion + (
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,1 if ew_cfg == _EW_CFG_FDIV else (1<<16)|1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
                       (2<<15)|(15<<11)|(2<<5)|(0 if ew_cfg == _EW_CFG_FDIV else 1<<3)|1))
  return RKStage(tuple(commands), tuple(relocs))

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM:
    if not u.src: return None
    u = u.src[0]
  return u

def _eval_cast(value:int|float|bool, dtype:DType) -> int|float|bool:
  if dtype.scalar() is dtypes.bool: return bool(value)
  if dtype.scalar() in dtypes.ints: return int(value)
  if dtype.scalar() is dtypes.half: return struct.unpack("<e", struct.pack("<e", float(value)))[0]
  if dtype.scalar() is dtypes.float: return struct.unpack("<f", struct.pack("<f", float(value)))[0]
  return float(value)

def _eval_expr(u:UOp, env:dict[UOp, int]) -> int|float|bool:
  if u.op is Ops.CONST: return _eval_cast(u.arg, u.dtype)
  if u.op is Ops.RANGE: return env[u]
  if u.op is Ops.CAST: return _eval_cast(_eval_expr(u.src[0], env), u.dtype)
  if u.op is Ops.WHERE: return _eval_cast(_eval_expr(u.src[1] if _eval_expr(u.src[0], env) else u.src[2], env), u.dtype)
  lhs = _eval_expr(u.src[0], env)
  if u.op is Ops.RECIPROCAL: return _eval_cast(1.0 / float(lhs), u.dtype)
  if u.op is Ops.TRUNC: return _eval_cast(int(lhs), u.dtype)
  rhs = _eval_expr(u.src[1], env)
  if u.op is Ops.ADD: return _eval_cast(lhs + rhs, u.dtype)
  if u.op is Ops.MUL: return _eval_cast(lhs * rhs, u.dtype)
  if u.op is Ops.SUB: return _eval_cast(lhs - rhs, u.dtype)
  if u.op is Ops.CDIV: return _eval_cast(cdiv(int(lhs), int(rhs)), u.dtype)
  if u.op is Ops.CMOD: return _eval_cast(cmod(int(lhs), int(rhs)), u.dtype)
  if u.op is Ops.FLOORDIV: return _eval_cast(floordiv(int(lhs), int(rhs)), u.dtype)
  if u.op is Ops.FLOORMOD: return _eval_cast(floormod(int(lhs), int(rhs)), u.dtype)
  if u.op is Ops.MAX: return _eval_cast(max(lhs, rhs), u.dtype)
  if u.op is Ops.CMPLT: return lhs < rhs
  if u.op is Ops.CMPNE: return lhs != rhs
  if u.op is Ops.AND: return _eval_cast(int(lhs) & int(rhs), u.dtype)
  if u.op is Ops.OR: return _eval_cast(int(lhs) | int(rhs), u.dtype)
  raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")

def _eval_int(u:UOp, env:dict[UOp, int]) -> int: return int(_eval_expr(u, env))

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.MAX}
def _is_static_expr(u:UOp, cache:dict[UOp, bool]|None=None) -> bool:
  if cache is not None and u in cache: return cache[u]
  ret = u.op in _STATIC_OPS and all(_is_static_expr(x, cache) for x in u.src)
  if cache is not None: cache[u] = ret
  return ret

@lru_cache(maxsize=16384)
def _cached_index_ranges(index:UOp) -> tuple[UOp, ...]: return tuple(u for u in index.toposort() if u.op is Ops.RANGE)
def _index_ranges(index:UOp) -> list[UOp]: return list(_cached_index_ranges(index))

def _iter_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  if not ranges: return [{}]
  order:list[UOp] = []
  seen:set[UOp] = set()
  def add(r:UOp) -> None:
    if r in seen: return
    for src in r.src[1:]:
      if src.op is Ops.RANGE: add(src)
    seen.add(r); order.append(r)
  for r in ranges: add(r)
  envs:list[dict[UOp, int]] = [{}]
  for r in order:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    envs = [{**env, r: i} for env in envs for i in range(int(r.src[0].arg))]
  return envs

def _static_expr_source(u:UOp, names:dict[UOp, str]) -> str:
  """Render a trusted geometry-only UOp as a flat Python expression for gather-map construction."""
  def cvt(src:str, dtype:DType) -> str:
    if dtype.scalar() is dtypes.bool: return f"bool({src})"
    if dtype.scalar() in dtypes.ints: return f"int({src})"
    if dtype.scalar() is dtypes.half: return f"half({src})"
    if dtype.scalar() is dtypes.float: return f"float32({src})"
    return f"float({src})"
  if u.op is Ops.CONST:
    value = bool(u.arg) if u.dtype.scalar() is dtypes.bool else int(u.arg) if u.dtype.scalar() in dtypes.ints else float(u.arg)
    return repr(value) if not isinstance(value, float) or math.isfinite(value) else f"float({str(value)!r})"
  if u.op is Ops.RANGE: return names[u]
  if u.op is Ops.CAST: return cvt(_static_expr_source(u.src[0], names), u.dtype)
  if u.op is Ops.WHERE:
    cond, lhs, rhs = (_static_expr_source(x, names) for x in u.src)
    return cvt(f"({lhs} if {cond} else {rhs})", u.dtype)
  lhs = _static_expr_source(u.src[0], names)
  if u.op is Ops.RECIPROCAL: return cvt(f"(1.0/float({lhs}))", u.dtype)
  if u.op is Ops.TRUNC: return cvt(f"int({lhs})", u.dtype)
  rhs = _static_expr_source(u.src[1], names)
  if u.op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.CMPLT, Ops.CMPNE):
    op = {Ops.ADD:"+", Ops.MUL:"*", Ops.SUB:"-", Ops.CMPLT:"<", Ops.CMPNE:"!="}[u.op]
    return cvt(f"({lhs}{op}{rhs})", u.dtype)
  if u.op in (Ops.AND, Ops.OR): return cvt(f"(int({lhs}){'&' if u.op is Ops.AND else '|'}int({rhs}))", u.dtype)
  if u.op is Ops.CDIV: return cvt(f"cdiv(int({lhs}),int({rhs}))", u.dtype)
  if u.op is Ops.CMOD: return cvt(f"cmod(int({lhs}),int({rhs}))", u.dtype)
  if u.op is Ops.FLOORDIV: return cvt(f"floordiv(int({lhs}),int({rhs}))", u.dtype)
  if u.op is Ops.FLOORMOD: return cvt(f"floormod(int({lhs}),int({rhs}))", u.dtype)
  if u.op is Ops.MAX: return cvt(f"max({lhs},{rhs})", u.dtype)
  raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")

class _GatherEval:
  """Per-kernel cache for fast, exact evaluation of static gather geometry."""
  def __init__(self):
    self.envs:dict[tuple[UOp, ...], tuple[tuple[int, ...], ...]] = {}
    self.funcs:dict[tuple[UOp, tuple[UOp, ...]], Callable[..., int|float|bool]] = {}
    self.dsts:dict[tuple[UOp, tuple[UOp, ...], int], tuple[int, ...]] = {}

  def environments(self, ranges:list[UOp]) -> tuple[tuple[int, ...], ...]:
    key = tuple(ranges)
    if key not in self.envs: self.envs[key] = tuple(tuple(env[r] for r in ranges) for env in _iter_range_env(ranges))
    return self.envs[key]

  def compile(self, u:UOp, ranges:list[UOp]) -> Callable[..., int|float|bool]:
    key = u, tuple(ranges)
    if key not in self.funcs:
      names = {r:f"r{i}" for i,r in enumerate(ranges)}
      scope = {"half":lambda x:struct.unpack("<e", struct.pack("<e", float(x)))[0],
               "float32":lambda x:struct.unpack("<f", struct.pack("<f", float(x)))[0],
               "cdiv":cdiv, "cmod":cmod, "floordiv":floordiv, "floormod":floormod}
      self.funcs[key] = eval(f"lambda {','.join(names.values())}: {_static_expr_source(u, names)}", scope)
    return self.funcs[key]

def _static_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  values:list[int|None] = [None] * count
  for env in _iter_range_env(ranges):
    dst = _eval_int(out_index, env)
    if not 0 <= dst < count: raise RuntimeError("RKPLAN_REJECT:static_index")
    bits = struct.unpack("<H", struct.pack("<e", float(_eval_expr(expr, env))))[0]
    if values[dst] is not None and values[dst] != bits: raise RuntimeError("RKPLAN_REJECT:static_index")
    values[dst] = bits
  if any(x is None for x in values): raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(x for x in values if x is not None)

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op is Ops.RANGE: return 0, {u: 1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _affine_index(u.src[0]), _affine_index(u.src[1])
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {r: coeff*scale for r, coeff in affine[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  coeffs = lhs[1].copy()
  for r, coeff in rhs[1].items():
    if (merged:=coeffs.get(r, 0) + sign*coeff): coeffs[r] = merged
    elif r in coeffs: del coeffs[r]
  return lhs[0] + sign*rhs[0], coeffs

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int, evaluator:_GatherEval|None=None) -> tuple[int, ...]:
  evaluator = evaluator or _GatherEval()
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(load_index) + ([] if gate is None else _index_ranges(gate))):
    raise RuntimeError("RKPLAN_REJECT:gather_index")
  envs, range_key = evaluator.environments(ranges), tuple(ranges)
  dst_key = out_index, range_key, count
  if dst_key not in evaluator.dsts: evaluator.dsts[dst_key] = tuple(int(evaluator.compile(out_index, ranges)(*env)) for env in envs)
  src_fxn, gate_fxn = evaluator.compile(load_index, ranges), evaluator.compile(gate, ranges) if gate is not None else None
  offsets = [-2] * count
  for env, dst in zip(envs, evaluator.dsts[dst_key]):
    src = int(src_fxn(*env))
    if not (0 <= dst < count): raise RuntimeError("RKPLAN_REJECT:gather_index")
    offsets[dst] = src if gate_fxn is None or gate_fxn(*env) else -1
    if offsets[dst] < -1: raise RuntimeError("RKPLAN_REJECT:gather_index")
  if any(offset == -2 for offset in offsets): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(offsets)

def _gather_plan(src_index:int, dst_scratch:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0,
                 evaluator:_GatherEval|None=None) -> RKGather:
  out_affine, load_affine = _affine_index(out_index), _affine_index(load_index)
  if gate is None and out_affine is not None and load_affine is not None and out_affine[0] == 0:
    expected = 1
    axes:list[tuple[int, int, int]] = []
    for r, dst_stride in sorted(out_affine[1].items(), key=lambda item: item[1]):
      if dst_stride != expected or r.src[0].op is not Ops.CONST: break
      limit = int(r.src[0].arg)
      if limit <= 0: break
      if (src_stride:=load_affine[1].get(r, 0)): axes.append((dst_stride, limit, src_stride))
      expected *= limit
    else:
      if expected == count and all(r in out_affine[1] for r in load_affine[1]):
        return RKGather(src_index, dst_scratch, count, load_affine[0], tuple(axes))
  return RKGather(src_index, dst_scratch, count, offsets=_gather_offsets(out_index, load_index, gate, count, evaluator), fill_bits=fill_bits)

def _selection_gather(u:UOp, out_index:UOp, count:int, oslot:int, selection_cache:dict[UOp, tuple[bool, bool]]|None=None,
                      static_cache:dict[UOp, bool]|None=None,
                      choice_cache:dict[tuple, tuple[tuple[int, int]|None, int|None]]|None=None) -> RKGather|RKMultiGather|None:
  """Collapse a static WHERE/ADD tree of mutually exclusive gated loads into raw gather offsets."""
  seen = selection_cache if selection_cache is not None else {}
  def selection_tree(x:UOp) -> tuple[bool, bool]:
    """Return (uses only selection-value ops, contains an actual selection) without walking index/condition graphs."""
    if x in seen: return seen[x]
    if x.op is Ops.CONST: ret = (x.dtype.scalar() is dtypes.half, False)
    elif x.op is Ops.CAST and x.dtype.scalar() is dtypes.half: ret = selection_tree(x.src[0])
    elif x.op is Ops.WHERE:
      lhs, rhs = selection_tree(x.src[1]), selection_tree(x.src[2]); ret = (lhs[0] and rhs[0], True)
    elif x.op is Ops.ADD:
      lhs, rhs = selection_tree(x.src[0]), selection_tree(x.src[1]); ret = (lhs[0] and rhs[0], lhs[1] or rhs[1])
    elif x.op is Ops.LOAD and x.src[0].op is Ops.INDEX:
      fallback = selection_tree(x.src[1]) if len(x.src) > 1 else (True, False)
      ret = (fallback[0], fallback[1] or len(x.src) > 2)
    else: ret = (False, False)
    seen[x] = ret
    return ret
  if selection_tree(u) != (True, True): return None
  out_ranges = _index_ranges(out_index)
  fill_bits:int|None = None
  choices = choice_cache if choice_cache is not None else {}
  range_key = tuple(out_ranges)
  def selected(x:UOp, env:dict[UOp, int]) -> tuple[tuple[int, int]|None, int|None]:
    key = x, range_key, tuple(env[r] for r in out_ranges)
    if key in choices: return choices[key]
    ret:tuple[tuple[int, int]|None, int|None]
    if x.op is Ops.CONST and x.dtype.scalar() is dtypes.half:
      bits = struct.unpack("<H", struct.pack("<e", float(x.arg)))[0]
      ret = None, bits
    elif x.op is Ops.CAST and x.dtype.scalar() is dtypes.half: ret = selected(x.src[0], env)
    elif x.op is Ops.WHERE:
      if not _is_static_expr(x.src[0], static_cache): raise ValueError
      ret = selected(x.src[1] if _eval_int(x.src[0], env) else x.src[2], env)
    elif x.op is Ops.ADD:
      (lhs,lbits), (rhs,rbits) = selected(x.src[0], env), selected(x.src[1], env)
      if lhs is not None and rhs is not None or lbits is not None and rbits is not None and lbits != rbits: raise ValueError
      ret = (lhs if lhs is not None else rhs), (lbits if lbits is not None else rbits)
    elif x.op is Ops.LOAD and x.src[0].op is Ops.INDEX and x.src[0].src[0].op is Ops.PARAM:
      param, index, gate = x.src[0].src[0], x.src[0].src[1], x.src[2] if len(x.src) > 2 else None
      if (param.dtype.scalar() is not dtypes.half or param.arg.slot == oslot or param.src[0].op is not Ops.CONST or
          any(r not in out_ranges for r in _index_ranges(index) + ([] if gate is None else _index_ranges(gate)))): raise ValueError
      if gate is not None:
        if not _is_static_expr(gate, static_cache): raise ValueError
        if not _eval_int(gate, env): ret = selected(x.src[1], env) if len(x.src) > 1 else (None, None)
        else: ret = ((param.arg.slot, _eval_int(index, env)), None)
      else: ret = ((param.arg.slot, _eval_int(index, env)), None)
    else: raise ValueError
    choices[key] = ret
    return ret
  slots, offsets = [-2] * count, [-2] * count
  try:
    for env in _iter_range_env(out_ranges):
      dst, (choice,bits) = _eval_int(out_index, env), selected(u, env)
      if not 0 <= dst < count: raise ValueError
      if bits is not None and fill_bits not in (None, bits): raise ValueError
      if bits is not None: fill_bits = bits
      slot, offset = (-1, -1) if choice is None else choice
      if slots[dst] not in (-2, slot) or offsets[dst] not in (-2, offset): raise ValueError
      slots[dst], offsets[dst] = slot, offset
  except (RuntimeError, ValueError): return None
  sources = sorted(set(slot for slot in slots if slot >= 0))
  if not sources or any(offset == -2 for offset in offsets): return None
  if len(sources) == 1: return RKGather(sources[0], 0, count, offsets=tuple(offsets), fill_bits=fill_bits or 0)
  if fill_bits is not None or any(slot < 0 for slot in slots): return None
  return RKMultiGather(tuple(RKGather(src, 0, count, offsets=tuple(offset if slot == src else -1 for slot, offset in zip(slots, offsets)),
                                      partial=True) for src in sources))

def _ew_leaf(u:UOp, out_index:UOp, count:int, oslot:int, static_cache:dict[UOp, bool]|None=None,
             selection_cache:dict[UOp, tuple[bool, bool]]|None=None,
             choice_cache:dict[tuple, tuple[tuple[int, int]|None, int|None]]|None=None) -> RKLeaf:
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.half: return float(u.arg)
  if u.dtype.scalar() is dtypes.half and _is_static_expr(u, static_cache): return RKStatic(u)
  if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half:
    return _ew_leaf(u.src[0], out_index, count, oslot, static_cache, selection_cache, choice_cache)
  if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
    param, index, gate = u.src[0].src[0], u.src[0].src[1], u.src[2] if len(u.src) > 2 else None
    if len(u.src) > 1 and u.src[1].op is not Ops.CONST:
      return _selection_gather(u, out_index, count, oslot, selection_cache, static_cache, choice_cache)
    fill_bits = struct.unpack("<H", struct.pack("<e", float(u.src[1].arg) if len(u.src) > 1 else 0.0))[0]
    if param.dtype.scalar() is not dtypes.half or param.arg.slot == oslot or param.src[0].op is not Ops.CONST: return None
    if gate is None and index.key == out_index.key and int(param.src[0].arg) == count: return RKArg(RKBufferKind.ARG, param.arg.slot)
    return param, index, gate, fill_bits
  return _selection_gather(u, out_index, count, oslot, selection_cache, static_cache, choice_cache) if u.dtype.scalar() is dtypes.half else None

def _unsupported_ew_ops(uops:list[UOp], out_index:UOp, count:int, oslot:int, supported:dict) -> list[str]:
  bad:list[str] = []
  for i, u in enumerate(uops):
    if u.op in (Ops.CONST, Ops.PARAM, Ops.RANGE, Ops.END, Ops.SINK, Ops.STORE, Ops.INDEX): continue
    if u.op in (Ops.ADD, Ops.MUL) and u.dtype.scalar() is dtypes.int: continue
    if u.op in supported and u.dtype.scalar() is dtypes.half: continue
    if u.op is Ops.LOAD or (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half):
      if _ew_leaf(u, out_index, count, oslot) is None: bad.append(f"{i}:{u.op.name}")
      continue
    if u.op is Ops.CAST or u.op is Ops.REDUCE or u.op in GroupOp.ALU: bad.append(f"{i}:{u.op.name}")
  return bad

def _mul_reduction_terms(u:UOp) -> tuple[list[UOp], int]|None:
  """Flatten a half ADD tree containing products and optional bias terms."""
  if u.dtype.scalar() is not dtypes.half: return None
  if u.op is Ops.MUL: return [u], 1
  if u.op is not Ops.ADD: return [u], 0
  lhs, rhs = _mul_reduction_terms(u.src[0]), _mul_reduction_terms(u.src[1])
  return None if lhs is None or rhs is None else (lhs[0] + rhs[0], lhs[1] + rhs[1])

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is not Ops.MAX or u.dtype.scalar() is not dtypes.half: return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None

def _const_operand(u:UOp, op:Ops, value:float|None=None) -> tuple[UOp, UOp]|None:
  if u.op is not op: return None
  for operand, const in (u.src, u.src[::-1]):
    if const.op is Ops.CONST and (value is None or float(const.arg) == value): return operand, const
  return None

def _negated_operand(u:UOp) -> UOp|None:
  if (scaled:=_const_operand(u, Ops.MUL, -1.0)) is not None: return scaled[0]
  return UOp.const(-float(u.arg), dtypes.half) if u.op is Ops.CONST else None

def _ew_plan(u:UOp) -> tuple[int, UOp, UOp]|None:
  if u.op is Ops.CUSTOMI and u.arg == "rockchip_positive_mask" and len(u.src) == 1:
    return _EW_STAGE_POSITIVE_MASK, u.src[0], u.src[0]
  if (negated:=_const_operand(u, Ops.MUL, -1.0)) is not None:
    return _EW_CFG_SUB, UOp.const(0.0, dtypes.half), negated[0]
  if (relu:=_relu_operand(u)) is not None: return _EW_CFG_RELU, relu, UOp.const(0.0, dtypes.half)
  if u.op is Ops.MAX:
    for direct, product in (u.src, u.src[::-1]):
      if (scaled:=_const_operand(product, Ops.MUL)) is None or scaled[0].key != direct.key: continue
      factor = float(scaled[1].arg)
      if factor == -1.0: return _EW_STAGE_ABS, direct, direct
      if 0.0 <= factor <= 1.0: return _EW_CFG_LEAKY_RELU, direct, scaled[1]
  if (outer:=_const_operand(u, Ops.MUL, -1.0)) is not None and outer[0].op is Ops.MAX:
    lhs, rhs = (_negated_operand(x) for x in outer[0].src)
    if lhs is not None and rhs is not None: return _EW_CFG_MIN, lhs, rhs
  if u.op is Ops.ADD:
    for positive, negative in (u.src, u.src[::-1]):
      source, scaled = _relu_operand(positive), _const_operand(negative, Ops.MUL, -1.0)
      if source is not None and scaled is not None and (upper:=_relu_operand(scaled[0])) is not None and \
         (shifted:=_const_operand(upper, Ops.ADD, -6.0)) is not None and shifted[0].key == source.key:
        return _EW_CFG_RELU6, source, UOp.const(0.0, dtypes.half)
  return (_EW_CFG[u.op], u.src[0], u.src[1]) if u.op in _EW_CFG and u.dtype.scalar() is dtypes.half else None

def _compensated_mul_sum(terms:list[UOp]) -> UOp:
  """Kahan sum built after symbolic simplification so compensation remains as DPU EW ops."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  total, correction = terms[0], zero
  for term in terms[1:]:
    adjusted = term.alu(Ops.ADD, correction.alu(Ops.MUL, neg_one))
    updated = total.alu(Ops.ADD, adjusted)
    correction = updated.alu(Ops.ADD, total.alu(Ops.MUL, neg_one)).alu(Ops.ADD, adjusted.alu(Ops.MUL, neg_one))
    total = updated
  return total

def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = x.alu(Ops.MUL, splitter)
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one)
  error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low)).alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  return total, lhs_error.alu(Ops.ADD, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  products, errors = tuple(x[0] for x in pairs), tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL)
  high, middle, low = products[0], zero, zero
  for part in products[1:] + errors:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = low.alu(Ops.ADD, error)
  middle = middle.alu(Ops.ADD, low)
  return high.alu(Ops.ADD, middle)

def _lower_scalar_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Turn a compact scalar register reduction into balanced FP16 DPU EW stages."""
  global_stores = [(store, _root_param(store.src[0])) for store in uops if store.op is Ops.STORE]
  global_stores = [(store, param) for store,param in global_stores if param is not None]
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if len(global_stores) != 1 or len(ranges) != 1: return None
  store, out_param = global_stores[0]
  assert out_param is not None
  if (out_param.dtype.scalar() is not dtypes.half or out_param.src[0].op is not Ops.CONST or int(out_param.src[0].arg) != 1 or
      store.src[0].op is not Ops.INDEX): return None
  reduce_range = ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) <= 0: return None
  loads:list[tuple[UOp, UOp]] = []
  for u in uops:
    if u.op is not Ops.LOAD or u.dtype.scalar() is not dtypes.half or not u.src or u.src[0].op is not Ops.INDEX: continue
    param = _root_param(u.src[0])
    if param is not None and param.arg.slot != out_param.arg.slot: loads.append((u, param))
  if not loads or len({param.key for _,param in loads}) != 1: return None
  in_param = loads[0][1]
  if in_param.src[0].op is not Ops.CONST: return None
  updates = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1: return None
  final_value = store.src[1]
  def local_load(x:UOp) -> UOp|None:
    while x.op is Ops.CAST: x = x.src[0]
    return x if x.op is Ops.LOAD and _root_param(x.src[0]) is None else None
  if local_load(final_value) is not None: post_scale = 1.0
  elif final_value.op is Ops.MUL and (final_load:=next((x for x in final_value.src if local_load(x) is not None), None)) is not None and \
       (scale:=final_value.src[1 if final_value.src[0] is final_load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else: return None
  reduce_ops = {u.op for u in updates[0].src[1].toposort()
                if u.dtype.scalar() is dtypes.half and u.op in (Ops.ADD, Ops.MUL, Ops.MAX)}
  negate_inputs = reduce_ops == {Ops.MUL, Ops.MAX} and any(u.op is Ops.CONST and u.dtype.scalar() is dtypes.half and
                                                            float(u.arg) == -1.0 for u in updates[0].src[1].toposort())
  if negate_inputs: reduce_op = Ops.MAX
  elif len(reduce_ops) == 1: reduce_op = next(iter(reduce_ops))
  else: return None
  if reduce_op not in _EW_CFG: return None
  gather_offsets = [tuple(_eval_int(load.src[0].src[1], {reduce_range:r}) for r in range(groups)) for load,_ in loads]
  input_count = int(in_param.src[0].arg)
  if input_count != groups*len(loads) or sorted(offset for offsets in gather_offsets for offset in offsets) != list(range(input_count)): return None

  if input_count < 2: return None
  const_values = tuple(dict.fromkeys(x for x in ((-1.0,) if negate_inputs else ()) + ((post_scale,) if post_scale != 1.0 else ())))
  const_slots = {value:i for i,value in enumerate(const_values)}
  data_slot = len(const_values)
  gathers = (RKGather(in_param.arg.slot, data_slot, input_count, 0, ((1, input_count, 1),), dst_stride=32),)
  cfg = _EW_CFG[reduce_op]
  ew_ops:list[RKEWOp] = []
  active = [i*64 for i in range(input_count)]
  if negate_inputs:
    neg_one = RKArg(RKBufferKind.SCRATCH, const_slots[-1.0])
    for offset in active:
      value = RKArg(RKBufferKind.SCRATCH, data_slot, offset)
      ew_ops.append(RKEWOp(value, value, neg_one, 1, _EW_CFG[Ops.MUL]))
  while len(active) > 1:
    next_active:list[int] = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs = active[i], active[i+1]
      dst = RKArg(RKBufferKind.ARG, out_param.arg.slot) if len(active) == 2 and post_scale == 1.0 else RKArg(RKBufferKind.SCRATCH, data_slot, lhs)
      ew_ops.append(RKEWOp(dst, RKArg(RKBufferKind.SCRATCH, data_slot, lhs), RKArg(RKBufferKind.SCRATCH, data_slot, rhs), 1, cfg))
      next_active.append(lhs)
    if len(active) & 1: next_active.append(active[-1])
    active = next_active
  if post_scale != 1.0:
    ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), RKArg(RKBufferKind.SCRATCH, data_slot, active[0]),
                         RKArg(RKBufferKind.SCRATCH, const_slots[post_scale]), 1, _EW_CFG[Ops.MUL]))
  constants = b"".join(struct.pack("<e", value) for value in const_values)
  scratch = tuple(RKScratch(2) for _ in const_values) + (RKScratch(input_count*64),)
  return RKImage(RKTarget.RK3588, scratch, constants, gathers=gathers, ew_ops=tuple(ew_ops), chain_limit=383)

def lower_ew(uops:list[UOp]) -> RKImage:
  if (loop_reduction:=_lower_scalar_loop_reduction(uops)) is not None: return loop_reduction
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out_param = outs[0]; assert out_param is not None
  count, oslot, store = int(out_param.src[0].arg), out_param.arg.slot, stores[0]
  if store.src[0].op is not Ops.INDEX: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_index, out, val = store.src[0].src[1], RKArg(RKBufferKind.ARG, oslot), store.src[1]
  static_cache:dict[UOp, bool] = {}
  selection_cache:dict[UOp, tuple[bool, bool]] = {}
  choice_cache:dict[tuple, tuple[tuple[int, int]|None, int|None]] = {}
  gather_eval = _GatherEval()
  leaf_cache:dict[UOp, RKLeaf] = {}
  def ew_leaf(u:UOp):
    if u not in leaf_cache: leaf_cache[u] = _ew_leaf(u, out_index, count, oslot, static_cache, selection_cache, choice_cache)
    return leaf_cache[u]
  if val.op is Ops.CONST and val.dtype.scalar() is dtypes.half:
    return RKImage(RKTarget.RK3588, constants=struct.pack("<e", float(val.arg)), fill=RKFill(out, count))
  if ew_leaf(val) is not None: val = val.alu(Ops.ADD, UOp.const(0.0, dtypes.half))
  supported = RockchipRenderer.code_for_op
  chain_limit = 383
  relu_input = _relu_operand(val)
  reduction = relu_input if relu_input is not None else val
  native_plan = _ew_plan(val)
  native_cfg = native_plan[0] if native_plan is not None else None
  if (native_cfg not in (_EW_CFG_MIN, _EW_CFG_RELU6) and (reduction_info:=_mul_reduction_terms(reduction)) is not None and
      (reduction_info[1] or len(reduction_info[0]) > 8)):
    terms = reduction_info[0]
    reduce_mode = os.getenv("ROCKCHIP_EW_REDUCE", "twoproduct").strip().lower()
    if reduce_mode == "kahan": reduced = _compensated_mul_sum(terms)
    elif reduce_mode == "twoproduct": reduced, chain_limit = _precise_mul_sum(terms), 256
    elif reduce_mode != "sequential": raise ValueError(f"invalid ROCKCHIP_EW_REDUCE={reduce_mode!r}")
    else: reduced = reduction
    val = reduced.alu(Ops.MAX, val.src[0] if val.src[1] is reduction else val.src[1]) if reduction is not val else reduced
  order:list[UOp] = []
  visited:dict[UOp, bool] = {}
  plans:dict[UOp, tuple[int, UOp, UOp]] = {}
  def visit(u:UOp) -> bool:
    if u in visited: return visited[u]
    if (plan:=_ew_plan(u)) is not None:
      visited[u] = all(visit(src) for src in plan[1:])
      if visited[u]: plans[u] = plan; order.append(u)
    else: visited[u] = ew_leaf(u) is not None
    return visited[u]
  if not visit(val) or not order or order[-1] is not val:
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  uses = {u: 0 for u in order}
  for node in order:
    for src in plans[node][1:]:
      if src in uses: uses[src] += 1
  values:dict[UOp, RKArg] = {}
  leaves:dict[UOp, RKArg] = {}
  free:list[int] = []
  const_scratch:dict[bytes, int] = {}
  static_scratch:dict[tuple[int, ...], int] = {}
  gather_scratch:dict[tuple, int] = {}
  gathers:list[RKGather] = []
  for expr in order:
    for src in plans[expr][1:]:
      if isinstance((leaf:=ew_leaf(src)), float):
        bits = struct.pack("<e", leaf)
        if bits not in const_scratch: const_scratch[bits] = len(const_scratch)
  scratch_count = len(const_scratch)
  ew_ops:list[RKEWOp] = []
  def operand(u:UOp) -> RKArg:
    nonlocal scratch_count
    if u in values: return values[u]
    if u in leaves: return leaves[u]
    leaf = ew_leaf(u)
    assert leaf is not None
    if isinstance(leaf, float): ret = RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", leaf)])
    elif isinstance(leaf, RKStatic):
      static = _static_vector(out_index, leaf.expr, count)
      if static not in static_scratch:
        static_scratch[static] = scratch_count
        gathers.append(RKGather(0, scratch_count, count, values=static))
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, static_scratch[static])
    elif isinstance(leaf, RKArg): ret = leaf
    else:
      if isinstance(leaf, RKMultiGather): gather_plans = leaf.gathers
      elif isinstance(leaf, RKGather): gather_plans = (leaf,)
      else:
        param, index, gate, fill_bits = leaf
        gather_plans = (_gather_plan(param.arg.slot, 0, out_index, index, gate, count, fill_bits, gather_eval),)
      for plan in gather_plans:
        source = next((x for x in u.toposort() if x.op is Ops.PARAM and x.arg.slot == plan.src_index), None)
        if source is None or source.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:gather_index")
        if plan.offsets: low, high = min(plan.offsets, default=0), max(plan.offsets, default=-1)
        else:
          low = high = plan.base
          for _, limit, stride in plan.axes:
            if stride < 0: low += (limit-1)*stride
            else: high += (limit-1)*stride
        if low < -1 or high >= int(source.src[0].arg): raise RuntimeError("RKPLAN_REJECT:gather_index")
      key = tuple((plan.src_index, plan.count, plan.base, plan.axes, plan.offsets, plan.fill_bits, plan.partial) for plan in gather_plans)
      if key not in gather_scratch:
        gather_scratch[key] = scratch_count
        gathers.extend(RKGather(plan.src_index, scratch_count, plan.count, plan.base, plan.axes, plan.offsets, plan.fill_bits,
                                partial=plan.partial) for plan in gather_plans)
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, gather_scratch[key])
    leaves[u] = ret
    return ret
  for expr in order:
    ew_cfg, lhs_u, rhs_u = plans[expr]
    lhs, rhs = operand(lhs_u), operand(rhs_u)
    if expr is val: dst = out
    elif (reuse:=next((values[src] for src in (lhs_u, rhs_u) if src in values and uses[src] == 1 and
                       values[src].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    ew_ops.append(RKEWOp(dst, lhs, rhs, count, ew_cfg))
    values[expr] = dst
    for dep in (lhs_u, rhs_u):
      if dep in uses:
        uses[dep] -= 1
        arg = values[dep]
        if uses[dep] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  constants = b""
  if const_scratch:
    by_slot = {slot: bits for bits, slot in const_scratch.items()}
    constants = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot) + 1))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(_scratch_bytes(count)) for _ in range(scratch_count)), constants,
                 gathers=tuple(gathers), ew_ops=tuple(ew_ops), chain_limit=chain_limit)

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

def _same_condition(lhs:UOp, rhs:UOp) -> bool:
  if lhs.key == rhs.key: return True
  if lhs.op is not Ops.AND or rhs.op is not Ops.AND or len(lhs.src) != 2 or len(rhs.src) != 2: return False
  return ((_same_condition(lhs.src[0], rhs.src[0]) and _same_condition(lhs.src[1], rhs.src[1])) or
          (_same_condition(lhs.src[0], rhs.src[1]) and _same_condition(lhs.src[1], rhs.src[0])))

def _opposite_condition(lhs:UOp, rhs:UOp) -> bool:
  def unwrap_not(x:UOp) -> UOp|None:
    if x.op is not Ops.CMPNE: return None
    if x.src[1].op is Ops.CONST and bool(x.src[1].arg): return x.src[0]
    if x.src[0].op is Ops.CONST and bool(x.src[0].arg): return x.src[1]
    return None
  lhs_not, rhs_not = unwrap_not(lhs), unwrap_not(rhs)
  return ((lhs_not is not None and _same_condition(lhs_not, rhs)) or
          (rhs_not is not None and _same_condition(lhs, rhs_not)))

def _fold_masked_load(gate:UOp, load:UOp, default:UOp) -> UOp|None:
  if len(load.src) <= 2 or load.src[1].op is not Ops.CONST: return None
  load_gate = load.src[2]
  same_default = float(load.src[1].arg) == float(default.arg)
  # If the outer condition implies the LOAD condition, the inner default is unreachable. This is the padded-pool form.
  outer_implies_inner = _same_condition(gate, load_gate) or (gate.op is Ops.AND and any(_same_condition(x, load_gate) for x in gate.src))
  if not same_default and not outer_implies_inner: return None
  return load.replace(src=(load.src[0], default, gate.alu(Ops.AND, load_gate) if same_default else gate))

def _fold_masked_max(gate:UOp, default:UOp, val:UOp, opposite:bool) -> UOp|None:
  if val.op is Ops.MAX:
    lhs = _fold_masked_max(gate, default, val.src[0], opposite)
    rhs = _fold_masked_max(gate, default, val.src[1], opposite)
    return None if lhs is None or rhs is None else val.replace(src=(lhs, rhs))
  if val.op is not Ops.LOAD or len(val.src) <= 2 or val.src[1].op is not Ops.CONST: return None
  condition_matches = _opposite_condition(gate, val.src[2]) if opposite else _same_condition(gate, val.src[2])
  if condition_matches:
    return val.replace(src=(val.src[0], default, val.src[2]))
  return None

def _is_const(u:UOp, value:float) -> bool: return u.op is Ops.CONST and float(u.arg) == value

def _minimum(lhs:UOp, rhs:UOp) -> UOp:
  neg_one = UOp.const(-1.0, dtypes.half)
  return lhs.alu(Ops.MUL, neg_one).alu(Ops.MAX, rhs.alu(Ops.MUL, neg_one)).alu(Ops.MUL, neg_one)

def _fold_ordered_where(x:UOp) -> UOp|None:
  gate, yes, no = x.src
  if gate.op is Ops.OR and yes.op is Ops.CONST:
    for upper, lower in ((gate.src[0], gate.src[1]), (gate.src[1], gate.src[0])):
      if (upper.op is Ops.CMPLT and upper.src[0].key == yes.key and upper.src[1].op is Ops.MAX and
          lower.op is Ops.CMPLT and lower.src[0].key == no.key and lower.src[1].key == yes.key and
          {u.key for u in upper.src[1].src} == {no.key, yes.key}): return _minimum(upper.src[1], yes)
  if gate.op is not Ops.CMPLT: return None
  lhs, rhs = gate.src
  if yes.key == rhs.key and no.key == lhs.key: return lhs.alu(Ops.MAX, rhs)
  if yes.key == lhs.key and no.key == rhs.key: return _minimum(lhs, rhs)
  # LeakyReLU with 0 <= slope <= 1 is max(x, slope*x).
  if no.key == lhs.key and _is_const(rhs, 0.0) and yes.op is Ops.MUL:
    scaled = yes.src[1] if yes.src[0].key == no.key else yes.src[0] if yes.src[1].key == no.key else None
    if scaled is not None and scaled.op is Ops.CONST and 0.0 <= float(scaled.arg) <= 1.0: return no.alu(Ops.MAX, yes)
  return None

def _unwrap_condition(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype.scalar() in (dtypes.bool, dtypes.half, dtypes.float): u = u.src[0]
  return u

def _positive_mask(u:UOp) -> UOp: return UOp(Ops.CUSTOMI, dtypes.half, (u,), "rockchip_positive_mask")

def _mask_expr(u:UOp) -> UOp|None:
  """Build an exact FP16 0/1 predicate using only DPU stages."""
  u = _unwrap_condition(u)
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    lhs, rhs = (_unwrap_condition(x) for x in u.src)
    # Boolean CMPNE is tinygrad's canonical NOT. Preserve it without trying to subtract bool values.
    for value, other in ((lhs, rhs), (rhs, lhs)):
      if value.op is Ops.CONST and value.dtype.scalar() is dtypes.bool:
        mask = _mask_expr(other)
        if mask is None: return None
        return UOp.const(1.0, dtypes.half).alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(-1.0, dtypes.half))) if bool(value.arg) else mask
    if lhs.dtype.scalar() not in (dtypes.half, dtypes.float) or rhs.dtype.scalar() not in (dtypes.half, dtypes.float): return None
    neg_one = UOp.const(-1.0, dtypes.half)
    lhs, rhs = lhs.cast(dtypes.half), rhs.cast(dtypes.half)
    positive = _positive_mask(rhs.alu(Ops.ADD, lhs.alu(Ops.MUL, neg_one)))
    if u.op is Ops.CMPLT: return positive
    reverse = _positive_mask(lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one)))
    return positive.alu(Ops.MAX, reverse)
  if u.op in (Ops.OR, Ops.AND):
    operands = tuple(_mask_expr(x) for x in u.src)
    if len(operands) != 2 or any(x is None for x in operands): return None
    mask_lhs, mask_rhs = operands
    assert mask_lhs is not None and mask_rhs is not None
    return mask_lhs.alu(Ops.MAX if u.op is Ops.OR else Ops.MUL, mask_rhs)
  return None

def _fold_general_where(x:UOp) -> UOp|None:
  mask = _mask_expr(x.src[0])
  if mask is None: return None
  yes, no = (arm.cast(dtypes.half) for arm in x.src[1:])
  neg_one, one = UOp.const(-1.0, dtypes.half), UOp.const(1.0, dtypes.half)
  inverse = one.alu(Ops.ADD, mask.alu(Ops.MUL, neg_one))
  gate = _unwrap_condition(x.src[0])
  if gate.op is Ops.CMPLT:
    lhs, rhs, yes_u, no_u = (_unwrap_condition(u) for u in (gate.src[0], gate.src[1], x.src[1], x.src[2]))
    if rhs.op is Ops.CONST and math.isfinite(float(rhs.arg)):
      threshold = float(rhs.arg)
      if yes_u.key == lhs.key and no_u.op is Ops.CONST and math.isfinite(float(no_u.arg)) and float(no_u.arg) != threshold:
        base = _minimum(lhs.cast(dtypes.half), UOp.const(threshold, dtypes.half))
        return base.alu(Ops.ADD, inverse.alu(Ops.MUL, UOp.const(float(no_u.arg)-threshold, dtypes.half)))
      if no_u.key == lhs.key and yes_u.op is Ops.CONST and math.isfinite(float(yes_u.arg)) and float(yes_u.arg) != threshold:
        base = lhs.cast(dtypes.half).alu(Ops.MAX, UOp.const(threshold, dtypes.half))
        return base.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(float(yes_u.arg)-threshold, dtypes.half)))
  nonfinite = tuple(arm.op is Ops.CONST and not math.isfinite(float(arm.arg)) for arm in x.src[1:])
  if any(nonfinite):
    if any(arm.op is Ops.CONST and math.isnan(float(arm.arg)) for arm in x.src[1:]): return None
    if all(nonfinite) and float(x.src[1].arg) == float(x.src[2].arg): return yes
    inf_index = next(i for i,is_nonfinite in enumerate(nonfinite) if is_nonfinite)
    if all(nonfinite): return None
    finite, denominator = (no, inverse) if inf_index == 0 else (yes, mask)
    sign = math.copysign(1.0, float(x.src[1+inf_index].arg))
    correction = UOp.const(sign, dtypes.half).alu(Ops.FDIV, denominator).alu(Ops.ADD, UOp.const(-sign, dtypes.half))
    return finite.alu(Ops.ADD, correction)
  return yes.alu(Ops.MUL, mask).alu(Ops.ADD, no.alu(Ops.MUL, inverse))

_abs_pattern = UPat(Ops.MUL, dtypes.half, src=[UPat.var("x"), UPat(Ops.WHERE, src=(
  UPat(Ops.CMPNE, src=(UPat.var("x"), UPat.const(0.0))),
  UPat(Ops.WHERE, src=(UPat(Ops.CMPLT, src=(UPat.var("x"), UPat.const(0.0))), UPat.const(-1.0), UPat.const(1.0))), UPat.const(0.0)))])
_pm_abs = PatternMatcher([(_abs_pattern, lambda x: x.alu(Ops.MAX, x.alu(Ops.MUL, UOp.const(-1.0, dtypes.half))))])

def _fp32_alu_to_fp16(x:UOp) -> UOp|None:
  return None if _is_static_expr(x) else x.src[0].cast(dtypes.half).alu(x.op, x.src[1].cast(dtypes.half))

def _fold_casted_relu(root:UOp) -> UOp|None:
  """Recover native half MAX from the float WHERE emitted for half ReLU inside a reduction."""
  if len(root.src) != 1 or (where:=root.src[0]).op is not Ops.WHERE or where.dtype.scalar() is not dtypes.float: return None
  gate, yes, no = where.src
  if (yes.op is not Ops.CAST or len(yes.src) != 1 or yes.src[0].dtype.scalar() is not dtypes.half or
      no.op is not Ops.CONST or float(no.arg) != 0.0): return None
  val = yes.src[0]
  if (gate.op is not Ops.CMPLT or gate.src[0].op is not Ops.CONST or float(gate.src[0].arg) != 0.0 or
      gate.src[1].key != val.key): return None
  return val.alu(Ops.MAX, UOp.const(0.0, dtypes.half))

_pm_fp32_to_fp16 = PatternMatcher([
  (UPat((Ops.ADD, Ops.MUL), dtypes.float, name="x"), _fp32_alu_to_fp16),
  (UPat(Ops.CAST, dtypes.half, name="root"), _fold_casted_relu),
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
  # Fold padding into the gather mask. This changes only host layout initialization; selected values still feed DPU EW.
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat(Ops.LOAD, dtypes.half, name="load"), UPat.cvar("default"))),
   lambda gate,load,default: _fold_masked_load(gate, load, default)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.var("val"), UPat.cvar("default"))),
   lambda gate,val,default: _fold_masked_max(gate, default, val, False)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.cvar("default"), UPat.var("val"))),
   lambda gate,default,val: _fold_masked_max(gate, default, val, True)),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_ordered_where),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_general_where),
])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  sink = graph_rewrite(sink, _pm_abs, name="rockchip abs")
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  code_for_op = {Ops.ADD: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_ew(_fp16_rewrite(uops)))).decode()
