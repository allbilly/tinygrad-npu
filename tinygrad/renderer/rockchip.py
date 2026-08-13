from __future__ import annotations
# ruff: noqa: E702
import base64, heapq, math, os, struct
import numpy as np
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Iterable, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes, float_to_fp16
from tinygrad.helpers import Target, ceildiv, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, UPat, PatternMatcher, consumer_map_from_toposort, exec_alu, graph_rewrite, python_alu
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 36
_HEADER = struct.Struct("<4sHHHHHIII")  # magic/version, scratch/gather/host counts, ops/constants, mid-gather count
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<I"), struct.Struct("<HHIBBBBBiIIii"), struct.Struct("<IIi")
_HOST_ADDRESS = struct.Struct("<BBBBBHHHIIIIIiiiiiii")
_EWOP = struct.Struct("<BBHIIIIIIiii")  # buffer fields, count/config, byte addends
_ITEM_FORMAT = {1:"B", 2:"H", 4:"I"}
_RKIMAGE_U16_MAX = (1 << 16) - 1

class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1
class RKLayout(IntEnum): FP16 = 0; INT16 = 1; BOOL_MASK = 2; INT32 = 3; BOOL_INT16 = 4; INT_FP16 = 5
class RKExecutionClass(IntEnum): NATIVE = 0; HOST_ADDRESS = 1
_BUFFER_KINDS = RKBufferKind.ARG, RKBufferKind.SCRATCH

@dataclass(frozen=True, slots=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True, slots=True)
class RKValue:
  """Physical ABI. BOOL_MASK and bounded exact INT_FP16 values occupy FP16 scratch lanes."""
  arg: RKArg; dtype: DType; count: int; layout: RKLayout

@dataclass(frozen=True, slots=True)
class RKScratch: size: int

@dataclass(frozen=True, slots=True)
class RKGather:
  """Materialize an affine or fallback raw-lane index map."""
  src_index: int; dst_index: int; count: int; base: int = 0
  axes: tuple[tuple[int, int, int], ...] = ()  # dst divisor, range limit, source stride
  offsets: tuple[int, ...] = ()
  fill_bits: int = 0
  values: tuple[int, ...] = ()  # compile-time FP16 vector, no source argument
  partial: bool = False  # preserve lanes populated by another gather when the offset is negative
  dst_stride: int = 1  # destination stride in lanes; scalar FP16 reductions use 32 for 64-byte spacing
  dst_addend: int = 0  # destination offset in lanes
  dst_kind: RKBufferKind = RKBufferKind.SCRATCH
  itemsize: int = 2
  src_kind: RKBufferKind = RKBufferKind.ARG
  after: int = -1  # EW-op split; -1 schedules the gather after the final stage

@dataclass(frozen=True, slots=True)
class RKHostAddress:
  """Host-calculated raw-lane movement. It never owns numeric or reduction semantics."""
  src: RKArg; index: RKArg; dst: RKArg; count: int; src_count: int; dst_count: int
  itemsize: int = 2; index_itemsize: int = 4; fill_bits: int = 0
  index_limit: int = 0; base: int = 0; index_scale: int = 1; lane_stride: int = 0; index_repeat: int = 1

@dataclass(frozen=True, slots=True)
class RKEWOp:
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False
  int16_output: bool = False; int16_input: bool = False

@dataclass(frozen=True, slots=True)
class RKImage:
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""
  gathers: tuple[RKGather, ...] = (); ew_ops: tuple[RKEWOp, ...] = ()
  mid_gathers: tuple[RKGather, ...] = ()
  host_gathers: tuple[RKHostAddress, ...] = (); host_scatters: tuple[RKHostAddress, ...] = ()

  @property
  def execution_class(self) -> RKExecutionClass:
    return RKExecutionClass.HOST_ADDRESS if self.host_gathers or self.host_scatters else RKExecutionClass.NATIVE

def _image_u16_safe(image:RKImage) -> bool:
  gathers, hosts = image.gathers+image.mid_gathers, image.host_gathers+image.host_scatters
  if any(not 0 <= count <= _RKIMAGE_U16_MAX for count in
         (len(image.scratch), len(gathers), len(image.mid_gathers), len(image.host_gathers), len(image.host_scatters))): return False
  return all(0 <= index <= _RKIMAGE_U16_MAX for gather in gathers for index in (gather.src_index, gather.dst_index)) and \
    all(0 <= arg.index <= _RKIMAGE_U16_MAX for host in hosts for arg in (host.src, host.index, host.dst)) and \
    all(0 <= arg.index <= _RKIMAGE_U16_MAX for op in image.ew_ops for arg in (op.dst, op.lhs, op.rhs))

def _map_image_args(image:RKImage, fn:Callable[[RKArg], RKArg], map_value_sources:bool=True) -> RKImage:
  def gather(value:RKGather) -> RKGather:
    source = RKArg(value.src_kind, value.src_index)
    src, dst = fn(source) if map_value_sources or not value.values else source, fn(RKArg(value.dst_kind, value.dst_index))
    return replace(value, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  def host(value:RKHostAddress) -> RKHostAddress: return replace(value, src=fn(value.src), index=fn(value.index), dst=fn(value.dst))
  def ew(op:RKEWOp) -> RKEWOp: return RKEWOp(fn(op.dst), fn(op.lhs), fn(op.rhs), op.count, op.ew_cfg, op.submit_barrier,
    op.compare, op.stateful, op.int32_output, op.int32_input, op.bool_output, op.int16_output, op.int16_input)
  return replace(image, gathers=tuple(gather(x) for x in image.gathers), mid_gathers=tuple(gather(x) for x in image.mid_gathers),
    ew_ops=tuple(ew(op) for op in image.ew_ops),
    host_gathers=tuple(host(x) for x in image.host_gathers), host_scatters=tuple(host(x) for x in image.host_scatters))

def _alias_image_args(image:RKImage, aliases:dict[int, RKArg]) -> RKImage:
  """Retarget logical argument slots while preserving their byte addends."""
  return _map_image_args(image, lambda arg:replace(aliases[arg.index], addend=aliases[arg.index].addend+arg.addend)
    if arg.kind is RKBufferKind.ARG and arg.index in aliases else arg)

def _hoist_leading_vector_materialization(image:RKImage) -> RKImage:
  """Compose a leading vector copy and its lane gathers so scalar execution starts in one NPU chain."""
  if len(image.ew_ops) < 2 or not image.mid_gathers: return image
  lead = image.ew_ops[0]
  if (lead.ew_cfg != _EW_CFG[Ops.MAX] or lead.lhs != lead.rhs or lead.lhs.kind is not RKBufferKind.SCRATCH or
      lead.dst.kind is not RKBufferKind.SCRATCH or lead.lhs.addend or lead.dst.addend or
      any((lead.compare, lead.stateful, lead.int32_output, lead.int32_input, lead.bool_output,
           lead.int16_output, lead.int16_input))): return image
  producers = [g for g in image.gathers if g.dst_kind is lead.lhs.kind and g.dst_index == lead.lhs.index]
  if len(producers) != 1: return image
  producer = producers[0]
  if (producer.values or producer.partial or producer.fill_bits or producer.count != lead.count or producer.dst_addend or
      producer.dst_stride != 1 or producer.itemsize != 2 or producer.src_kind is not RKBufferKind.ARG): return image
  moved = [g for g in image.mid_gathers if g.after == 1]
  if not moved or any(g.src_kind is not lead.dst.kind or g.src_index != lead.dst.index or g.itemsize != 2 for g in moved): return image
  written:set[tuple[int, int]] = set()
  for op in image.ew_ops[1:]:
    if any(arg.kind is lead.dst.kind and arg.index == lead.dst.index and (arg.addend, op.count) not in written for arg in (op.lhs, op.rhs)):
      return image
    if op.dst.kind is lead.dst.kind and op.dst.index == lead.dst.index: written.add((op.dst.addend, op.count))
  source_offsets = _plan_offsets(producer)
  indices = tuple(_plan_offsets(g) for g in moved)
  if any(index < 0 or index >= len(source_offsets) for row in indices for index in row): return image
  direct = tuple(replace(g, src_kind=producer.src_kind, src_index=producer.src_index, base=0, axes=(),
                         offsets=tuple(source_offsets[index] for index in row), after=-1) for g,row in zip(moved, indices))
  remaining = tuple(replace(g, after=max(0, g.after-1)) for g in image.mid_gathers if g.after != 1)
  return replace(image, gathers=image.gathers+direct, ew_ops=image.ew_ops[1:], mid_gathers=remaining)

def _reuse_linear_scratch(image:RKImage, constant_slots:dict[bytes, int]) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  ends, order = [-1] * len(image.scratch), list[tuple[int, int]]()
  def touch(arg:RKArg, event:int) -> None:
    if arg.kind is not RKBufferKind.SCRATCH: return
    if not 0 <= arg.index < len(ends): raise ValueError("invalid virtual scratch slot")
    if ends[arg.index] < 0: order.append((event, arg.index))
    ends[arg.index] = event
  def touch_gather(gather:RKGather, event:int) -> None:
    if not gather.values: touch(RKArg(gather.src_kind, gather.src_index), event)
    touch(RKArg(gather.dst_kind, gather.dst_index), event)
  def touch_host(host:RKHostAddress, event:int) -> None:
    touch(host.src, event); touch(host.index, event); touch(host.dst, event)
  for slot in constant_slots.values(): touch(RKArg(RKBufferKind.SCRATCH, slot), 0)
  event = 1
  for gather in image.gathers: touch_gather(gather, event); event += 1
  for host in image.host_gathers: touch_host(host, event); event += 1
  mid_by_point:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers:
    mid_by_point.setdefault(gather.after, []).append(gather)
  for index,op in enumerate(image.ew_ops):
    for gather in mid_by_point.get(index, ()): touch_gather(gather, event); event += 1
    touch(op.lhs, event); touch(op.rhs, event); touch(op.dst, event)
    event += 1
  for gather in mid_by_point.get(len(image.ew_ops), ()): touch_gather(gather, event); event += 1
  for host in image.host_scatters: touch_host(host, event); event += 1
  if not order: return replace(image, scratch=(), constants=b"")
  # Mid-program gathers may populate one logical slot in several partial phases. The runtime clears a
  # destination once per physical slot, so these stateful materialization slots must not alias.
  pinned = {gather.dst_index for gather in image.mid_gathers if gather.dst_kind is RKBufferKind.SCRATCH}
  intervals = ((start, ends[slot], slot) for start,slot in order)
  remap:dict[int, int] = {}
  physical:list[RKScratch] = []
  active:list[tuple[int, int]] = []
  available:list[int] = []
  for start,end,slot in intervals:
    while active and active[0][0] < start:
      _,target = heapq.heappop(active)
      heapq.heappush(available, target)
    spec = image.scratch[slot]
    if slot not in pinned and available:
      target = heapq.heappop(available)
      physical[target] = RKScratch(max(physical[target].size, spec.size))
    else:
      target = len(physical)
      physical.append(spec)
    if slot not in pinned: heapq.heappush(active, (end, target))
    remap[slot] = target
  physical_args = tuple(RKArg(RKBufferKind.SCRATCH, slot) for slot in range(len(physical)))
  def remap_arg(arg:RKArg) -> RKArg:
    if arg.kind is not RKBufferKind.SCRATCH: return arg
    return physical_args[remap[arg.index]] if not arg.addend else RKArg(arg.kind, remap[arg.index], arg.addend)
  image = _map_image_args(image, remap_arg, False)
  by_slot:dict[int, bytes] = {}
  for bits,slot in constant_slots.items():
    target = remap[slot]
    if target in by_slot and by_slot[target] != bits: raise ValueError("overlapping scratch constants")
    by_slot[target] = bits
  constants = b"" if not by_slot else b"".join(by_slot.get(slot, b"\0\0") for slot in range(max(by_slot)+1))
  return replace(image, scratch=tuple(physical), constants=constants)

@dataclass(frozen=True, slots=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True, slots=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.mid_gathers
  if any(not 0 <= g.after <= len(image.ew_ops) for g in image.mid_gathers):
    raise ValueError("invalid mid-gather split")
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, RKIMAGE_VERSION, len(image.scratch), len(gathers),
                               len(image.host_gathers), len(image.host_scatters),
                               len(image.ew_ops), len(image.constants), len(image.mid_gathers)))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size)
  for g in gathers:
    kind = 3 if g.partial else 2 if g.values else 1 if g.offsets else 0
    out += _GATHER.pack(g.dst_index, g.src_index, g.count, kind, len(g.axes), g.itemsize, int(g.dst_kind), int(g.src_kind),
                        g.base, g.fill_bits, g.dst_stride, g.dst_addend, g.after)
    if kind == 2: out += struct.pack(f"<{g.count}{_ITEM_FORMAT[g.itemsize]}", *g.values)
    elif kind in (1, 3): out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for host in image.host_gathers + image.host_scatters:
    if host.itemsize not in _ITEM_FORMAT or host.index_itemsize not in (2, 4) or host.index_repeat < 1 or \
       host.count % host.index_repeat: raise ValueError("invalid RKHostAddress item size")
    out += _HOST_ADDRESS.pack(int(host.src.kind), int(host.index.kind), int(host.dst.kind), host.itemsize, host.index_itemsize,
      host.src.index, host.index.index, host.dst.index, host.count, host.src_count, host.dst_count, host.fill_bits,
      host.index_limit, host.src.addend, host.index.addend, host.dst.addend,
      host.base, host.index_scale, host.lane_stride, host.index_repeat)
  for op in image.ew_ops:
    if op.bool_output and not op.int32_output: raise ValueError("bool output requires INT32 conversion")
    int16_to_int32 = op.int16_input and op.int32_output and not op.int16_output and not op.int32_input
    if (op.int16_output or op.int16_input) and (op.int32_output or op.int32_input) and not int16_to_int32:
      raise ValueError("conflicting integer precision")
    op_flags = (int(op.submit_barrier) | int(op.compare)<<1 | int(op.stateful)<<2 | int(op.int32_output)<<3 |
                int(op.int32_input)<<4 | int(op.bool_output)<<5 | int(op.int16_output)<<6 | int(op.int16_input)<<7)
    out += _EWOP.pack(int(op.dst.kind), op_flags, op.dst.index, int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index,
                      op.count, op.ew_cfg, op.dst.addend, op.lhs.addend, op.rhs.addend)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, nscratch, ngather, nhost_gather, nhost_scatter, nop, nconst, mid_count = \
    _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or mid_count > ngather:
    raise ValueError("invalid RKImage header")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_index, src_index, count, kind, naxes, itemsize, dst_kind, src_kind, base, fill_bits, dst_stride, dst_addend, after = \
      _GATHER.unpack_from(blob, off); off += _GATHER.size
    if (kind not in (0, 1, 2, 3) or (kind and naxes) or itemsize not in _ITEM_FORMAT or dst_kind not in (0, 1) or src_kind not in (0, 1) or
        dst_stride < 1 or dst_addend < 0): raise ValueError("invalid RKGather")
    axes:tuple[tuple[int, int, int], ...] = (); offsets:tuple[int, ...] = (); values:tuple[int, ...] = ()
    if kind == 2: values = struct.unpack_from(f"<{count}{_ITEM_FORMAT[itemsize]}", blob, off); off += itemsize*count
    elif kind in (1, 3): offsets = struct.unpack_from(f"<{count}i", blob, off); off += 4*count
    else: axes = tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)); off += naxes*_GATHER_AXIS.size
    gathers.append(RKGather(src_index, dst_index, count, base if kind == 0 else 0, axes, offsets, fill_bits, values, kind == 3,
      dst_stride, dst_addend, _BUFFER_KINDS[dst_kind], itemsize, _BUFFER_KINDS[src_kind], after))
  host_addresses:list[RKHostAddress] = []
  for _ in range(nhost_gather+nhost_scatter):
    src_kind, index_kind, dst_kind, itemsize, index_itemsize, src_index, index_index, dst_index, count, src_count, dst_count, \
      fill_bits, index_limit, src_addend, index_addend, dst_addend, base, index_scale, lane_stride, index_repeat = \
      _HOST_ADDRESS.unpack_from(blob, off)
    off += _HOST_ADDRESS.size
    if (src_kind not in (0, 1) or index_kind not in (0, 1) or dst_kind not in (0, 1) or itemsize not in _ITEM_FORMAT or
        index_itemsize not in (2, 4) or min(count, src_count, dst_count, index_limit) < 0 or index_repeat < 1 or count % index_repeat):
      raise ValueError("invalid RKHostAddress")
    host_addresses.append(RKHostAddress(RKArg(_BUFFER_KINDS[src_kind], src_index, src_addend),
      RKArg(_BUFFER_KINDS[index_kind], index_index, index_addend), RKArg(_BUFFER_KINDS[dst_kind], dst_index, dst_addend),
      count, src_count, dst_count, itemsize, index_itemsize, fill_bits, index_limit, base, index_scale, lane_stride, index_repeat))
  ew_ops:list[RKEWOp] = []
  for _ in range(nop):
    dk, op_flags, di, lk, li, rk_, ri, count, ew_cfg, da, la, ra = _EWOP.unpack_from(blob, off); off += _EWOP.size
    int16_to_int32 = op_flags & 0x88 == 0x88 and not op_flags & 0x50
    if dk not in (0, 1) or lk not in (0, 1) or rk_ not in (0, 1) or op_flags & 0x20 and not op_flags & 0x08 or \
       op_flags & 0xc0 and op_flags & 0x18 and not int16_to_int32:
      raise ValueError("invalid RKEWOp flags")
    ew_ops.append(RKEWOp(RKArg(_BUFFER_KINDS[dk], di, da), RKArg(_BUFFER_KINDS[lk], li, la), RKArg(_BUFFER_KINDS[rk_], ri, ra), count, ew_cfg,
                         bool(op_flags & 1), bool(op_flags & 2), bool(op_flags & 4), bool(op_flags & 8), bool(op_flags & 16),
                         bool(op_flags & 32), bool(op_flags & 64), bool(op_flags & 128)))
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  pre_count = ngather-mid_count
  if any(not 0 <= gather.after <= nop for gather in gathers[pre_count:]): raise ValueError("invalid mid-gather split")
  return RKImage(scratch, blob[off:], tuple(gathers[:pre_count]), tuple(ew_ops),
                 tuple(gathers[pre_count:]), tuple(host_addresses[:nhost_gather]), tuple(host_addresses[nhost_gather:]))

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for reloc in stage.relocs:
    word = commands[reloc.word]
    value = (address(reloc.arg.kind, reloc.arg.index) + reloc.arg.addend) & 0xffffffff
    commands[reloc.word] = (word & ~0xffffffff0000) | (value << 16)
  return tuple(commands)

_DPU, _RDMA = 0x1001, 0x2001
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_MAX_MAPPED_DOT_SCRATCH_BYTES = 352 << 20
_MAX_GENERIC_UNROLL = 1 << 14
_MAX_GENERIC_EXPANDED_NODES = 1 << 20
_MAX_OPTIONAL_RECIPE_NODES = 4096
_MIN_GENERIC_PRODUCT_RESIDUAL_TERMS = 64
_MAX_STATIC_LOCAL_STEPS = 1 << 20
_MAX_STATIC_RANGE_ENVS = 1 << 18
_EW_ELEMS_32BIT = 8*dtypes.half.itemsize//dtypes.float.itemsize
_FP16_EXACT_INTEGER = 1 << 11
_EW_DATA_MODE_FP16 = 1 << 28
_EW_EDATA_SIZE_FP16 = 2 << 22
_EW_ALU_MIN = 1 << 16
_EW_ALU_ADD = 2 << 16
_EW_ALU_FDIV = 3 << 16
_EW_ALU_SUB = 4 << 16
_EW_ALU_ABS = 5 << 16
_EW_ALU_NEG = 6 << 16
_EW_ALU_FLOOR = 7 << 16
_EW_ALU_CEIL = 8 << 16
_EW_RELU_BYPASS = 1 << 9
_EW_OP_CVT_BYPASS = 1 << 8
_EW_LUT_BYPASS = 1 << 7
_EW_OP_SRC_DMA = 1 << 6
_EW_MUL_PRELU = 1 << 5
_EW_OP_TYPE_MUL = 1 << 2
_EW_CFG_COMMON = _EW_DATA_MODE_FP16 | _EW_EDATA_SIZE_FP16 | _EW_LUT_BYPASS | _EW_OP_SRC_DMA
_EW_CFG_MIN = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_MIN
_EW_CFG_ABS = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ABS
_EW_CFG_NEG = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_NEG
_EW_CFG_FLOOR = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_FLOOR
_EW_CFG_CEIL = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_CEIL
_EW_CFG_LEAKY_RELU = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_MUL_PRELU | _EW_OP_TYPE_MUL
_EW_STAGE_FP32_OUT = 1 << 29  # software tags consumed before writing EW_CFG
_EW_STAGE_FP32_IN = 1 << 30
_DPU_DATA_FORMAT_FP16 = (2<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_OUT = (5<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_IN = (2<<29)|(5<<26)|2
_DPU_DATA_FORMAT_INT16_OUT = (1<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT16 = (1<<29)|(1<<26)|1
_DPU_DATA_FORMAT_INT16_TO_INT32 = (4<<29)|(1<<26)|1
_DPU_DATA_FORMAT_INT32_OUT = (4<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT32_IN = (2<<29)|(4<<26)|4
_DPU_DATA_FORMAT_INT32 = (4<<29)|(4<<26)|4
_BS_BN_BYPASS = 1|(1<<1)|(1<<4)|(1<<6)
_BS_OW_FP32_SCALAR = (1<<8)|(1<<5)|(1<<2)|(1<<1)
_BS_CFG_COMPARE = 0x40040
_BS_ALU_COMPARE = 0x33800000
_BS_MUL_COMPARE = 0x40000000
_BN_CFG_COMPARE = 0x40082
_BN_MUL_COMPARE = 0x7c000000
_BN_RELUX_COMPARE = 0x3f800000
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_FLOOR, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN,
 _NATIVE_POSITIVE_MASK, _NATIVE_PRECISE_ADD, _NATIVE_RAW_MIN, _NATIVE_SIGN) = (
   "rockchip_abs", "rockchip_ceil", "rockchip_floor", "rockchip_leaky_relu", "rockchip_mask_mul",
   "rockchip_min", "rockchip_positive_mask", "rockchip_precise_add", "rockchip_raw_min", "rockchip_sign")
_EW_CFG = {
  Ops.ADD: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ADD,
  Ops.SUB: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_SUB,
  Ops.MUL: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_OP_TYPE_MUL,
  Ops.MAX: _EW_CFG_COMMON | _EW_RELU_BYPASS,
  Ops.FDIV: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_ALU_FDIV,
}
_INT16_EW = dict(int16_input=True, int16_output=True)
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)
def _scratch_arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
def _reduction_stride(count:int) -> int: return round_up(count*2, 64)
def _int32_tiles_bytes(count:int) -> int: return ceildiv(count, 4) * 64
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff
def _int16_low_bytes(source:RKArg, out_slot:int, count:int, stride:int=2) -> RKGather:
  return RKGather(source.index, out_slot, count, base=source.addend, axes=((1, count, stride),),
                  dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)

def _finish_ew_stage(regs:tuple[tuple[int, int, int], ...], dst:RKArg, lhs:RKArg, rhs:RKArg, rdma_feature:int) -> RKStage:
  commands = tuple(_cmd(*x) for x in regs)
  bindings = ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),(_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs))
  relocs = tuple(RKReloc(len(commands)+i, arg) for i,(_,_,arg) in enumerate(bindings))
  return RKStage(commands+tuple(_cmd(target, reg, 0) for target,reg,_ in bindings)+
                 (_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature),), relocs)

def _emit_stateful_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                         int32_output:bool=False, int32_input:bool=False,
                         int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False, fp32_input:bool=False) -> RKStage:
  """Emit a self-contained DPU EW stage, optionally consuming or producing native integers."""
  native_int16, native_int32 = int16_input and int16_output, int32_input and int32_output
  int16_to_int32 = int16_input and int32_output and not int16_output and not int32_input
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if native_int32 else \
          _EW_ELEMS_32BIT if int32_output or int32_input or fp32_output or fp32_input else _MAX_EW_ELEMS_FP16
  if not 0 < count <= limit:
    raise ValueError(f"stateful EW count {count} out of range")
  lanes = 4 if int32_input or fp32_input else 8
  is_div = ew_cfg == _EW_CFG[Ops.FDIV]
  width = (count + lanes-1) // lanes - 1
  pipeline:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_BS_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR if int16_to_int32 or fp32_output and count == 1 else 2),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,0 if fp32_output and count == 1 else 3 if fp32_output else lanes-1),
    (_DPU,rk.REG_DPU_WDMA_SIZE_1,width),(_DPU,rk.REG_DPU_BN_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0))
  if compare: pipeline += ((_DPU,rk.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(_DPU,rk.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),
    (_DPU,rk.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_CFG,_BN_CFG_COMPARE),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE))
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP32_OUT if fp32_output else _DPU_DATA_FORMAT_FP32_IN if fp32_input else
                                  _DPU_DATA_FORMAT_INT16 if native_int16 else _DPU_DATA_FORMAT_INT32 if native_int32 else
                                  _DPU_DATA_FORMAT_INT16_TO_INT32 if int16_to_int32 else
                                  _DPU_DATA_FORMAT_INT32_OUT if int32_output else _DPU_DATA_FORMAT_INT16_OUT if int16_output else
                                  _DPU_DATA_FORMAT_INT32_IN if int32_input else _DPU_DATA_FORMAT_FP16)) + \
    (((_DPU,rk.REG_DPU_DST_SURF_STRIDE,1<<4),) if int16_to_int32 or fp32_output else ()) + (
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0 if fp32_output and count == 1 else ((lanes-1)<<16)|(lanes-1))) + pipeline + (
    (_DPU,rk.REG_DPU_EW_CFG,_EW_CFG_COMMON|1 if compare else
                               (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else
                               ew_cfg & ~_EW_OP_CVT_BYPASS if native_int16 or int16_to_int32 else ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,0 if fp32_output else 1 if int32_output or int16_output or is_div else (1<<16)|1),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),
    (_DPU,rk.REG_DPU_SURFACE_ADD,(2 if native_int16 or int16_to_int32 else 4)<<4),(_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,lanes-1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input or fp32_input else 2)<<2)))
  rdma_precision = 5 if fp32_input else 4 if int32_input else 1 if int16_input else 2
  rdma_feature = (rdma_precision<<15)|(15<<11)|(rdma_precision<<5)|(0 if is_div or int16_input or fp32_input else 1<<3)|1
  return _finish_ew_stage(regs, dst, lhs, rhs, rdma_feature)

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                  stateful:bool=False, int32_output:bool=False, int32_input:bool=False,
                  int16_output:bool=False, int16_input:bool=False) -> RKStage:
  """Build one DPU EW command body without its PC-chain tail."""
  if ew_cfg & _EW_STAGE_FP32_OUT:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_OUT, fp32_output=True)
  if ew_cfg & _EW_STAGE_FP32_IN:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_IN, fp32_input=True)
  if compare or stateful or int32_output or int32_input or int16_output or int16_input:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg, compare, int32_output, int32_input, int16_output, int16_input)
  if not (0 < count <= _MAX_EW_ELEMS_FP16): raise ValueError(f"EW fp16 count {count} out of range")
  is_div = ew_cfg == _EW_CFG[Ops.FDIV]
  width = (count + 7) // 8 - 1
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP16),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7),
    (_DPU,rk.REG_DPU_EW_CFG,ew_cfg)) + (((_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,1<<6)) if is_div else ()) + (
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,1 if is_div else (1<<16)|1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  return _finish_ew_stage(regs, dst, lhs, rhs, (2<<15)|(15<<11)|(2<<5)|(0 if is_div else 1<<3)|1)

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM:
    if not u.src: return None
    u = u.src[0]
  return u

def _strip_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _local_load(u:UOp) -> UOp|None:
  u = _strip_cast(u)
  return u if u.op is Ops.LOAD and _root_param(u.src[0]) is None else None

def _semantic_local_loads(u:UOp, cache:dict[UOp, tuple[UOp, ...]]|None=None) -> tuple[UOp, ...]:
  if cache is not None and u in cache: return cache[u]
  sources = () if u.op in (Ops.RANGE, Ops.SPECIAL) else u.src[:1] if u.op is Ops.AFTER else u.src
  result = (load,) if (load:=_local_load(u)) is not None else tuple(dict.fromkeys(y for x in sources for y in _semantic_local_loads(x, cache)))
  if cache is not None: cache[u] = result
  return result

def _eval_cast(value:int|float|bool, dtype:DType) -> int|float|bool:
  if dtype.scalar() is dtypes.bool: return bool(value)
  if dtype.scalar() in dtypes.ints: return int(value)
  if dtype.scalar() is dtypes.half: return float_to_fp16(value)
  if dtype.scalar() is dtypes.float: return struct.unpack("<f", struct.pack("<f", float(value)))[0]
  return float(value)

_StaticValue = int|float|bool|np.ndarray

def _static_cast(value:_StaticValue, dtype:DType, vector:bool=False) -> _StaticValue:
  return np.asarray(value, dtype=np.dtype(dtype.scalar().fmt) if dtype.scalar().fmt is not None else None) \
    if vector or isinstance(value, np.ndarray) else _eval_cast(value, dtype)

_STATIC_ALU_OPS = {Ops.RECIPROCAL, Ops.TRUNC, Ops.NEG, Ops.ADD, Ops.MUL, Ops.SUB, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV,
                   Ops.FLOORMOD, Ops.MAX, Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.WHERE}
def _static_alu(op:Ops, dtype:DType, values:tuple[_StaticValue, ...], vector:bool) -> _StaticValue:
  if op not in _STATIC_ALU_OPS: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {op.name}")
  if vector and op in (Ops.CDIV, Ops.CMOD):
    left, right = np.asarray(values[0]), np.asarray(values[1])
    with np.errstate(divide="ignore", invalid="ignore"): quotient = np.where(right != 0, np.trunc(left/right), 0)
    value = quotient if op is Ops.CDIV else left-quotient*right
  elif vector and op in (Ops.FLOORDIV, Ops.FLOORMOD):
    left, right = np.asarray(values[0]), np.asarray(values[1])
    quotient = np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=np.result_type(left, right))
    np.floor_divide(left, right, out=quotient, where=right != 0)
    value = quotient if op is Ops.FLOORDIV else left-quotient*right
  elif vector and op is Ops.MAX: value = np.maximum(values[0], values[1])
  elif vector and op is Ops.WHERE: value = np.where(*values)
  elif vector and op is Ops.TRUNC: value = np.trunc(values[0])
  elif op is Ops.RECIPROCAL: value = 1.0 / values[0]
  elif not vector and op is Ops.TRUNC: value = int(values[0])
  elif not vector and op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.AND, Ops.OR, Ops.XOR):
    value = exec_alu(op, dtype, tuple(int(x) for x in values), False)
  else: value = python_alu[op](*values) if vector else exec_alu(op, dtype, values, False)
  return _static_cast(value, dtype)

def _eval_expr(u:UOp, env:dict[UOp, _StaticValue], cache:dict[UOp, _StaticValue], vector:bool=False,
               load:Callable[[UOp], _StaticValue]|None=None) -> _StaticValue:
  if u in cache: return cache[u]
  if u.op is Ops.CONST: ret = _static_cast(u.arg, u.dtype, vector)
  elif u.op in (Ops.RANGE, Ops.SPECIAL): ret = env[u]
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
  elif u.op is Ops.AFTER: ret = _eval_expr(u.src[0], env, cache, vector, load)
  elif u.op is Ops.LOAD and load is not None: ret = load(u)
  elif u.op is Ops.CAST: ret = _static_cast(_eval_expr(u.src[0], env, cache, vector, load), u.dtype)
  elif u.op is Ops.WHERE and not vector:
    condition = _eval_expr(u.src[0], env, cache, load=load)
    ret = _static_cast(_eval_expr(u.src[1] if condition else u.src[2], env, cache, load=load), u.dtype)
  else: ret = _static_alu(u.op, u.dtype, tuple(_eval_expr(src, env, cache, vector, load) for src in u.src), vector)
  cache[u] = ret
  return ret

def _eval_vector(u:UOp, env:dict[UOp, np.ndarray], cache:dict[UOp, np.ndarray]) -> np.ndarray:
  return np.asarray(_eval_expr(u, typing_cast(dict[UOp, _StaticValue], env), typing_cast(dict[UOp, _StaticValue], cache), True))

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX}
def _is_static_expr(u:UOp) -> bool:
  return all(node.op in _STATIC_OPS and not (isinstance(node.arg, str) and node.arg.startswith("rockchip_")) for node in u.toposort())

def _index_ranges(index:UOp) -> list[UOp]:
  """Ranges used as index values, excluding AFTER/END ordering dependencies attached to a RANGE."""
  ranges:list[UOp] = []
  def walk(u:UOp) -> None:
    if u.op in (Ops.RANGE, Ops.SPECIAL):
      if u not in ranges: ranges.append(u)
      return
    for src in u.src: walk(src)
  walk(index)
  return ranges

RKOutput = tuple[UOp, UOp, int, UOp, UOp]
def _output_store(uops:list[UOp], dtype:DType|tuple[DType, ...], *, allow_local:bool=False) \
                  -> RKOutput|None:
  """Return the single statically-sized output store shared by specialized graph matchers."""
  stores = [u for u in uops if u.op is Ops.STORE]
  outputs = [(store, root) for store in stores if (root:=_root_param(store.src[0])) is not None]
  if len(outputs) != 1 or not allow_local and len(stores) != 1:
    return None
  store, out_param = outputs[0]
  accepted = dtype if isinstance(dtype, tuple) else (dtype,)
  if out_param.dtype.scalar() not in accepted or out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  return store, out_param, int(out_param.src[0].arg), store.src[0].src[1], store.src[1]

def _iter_range_env(ranges:list[UOp], max_envs:int=_MAX_STATIC_RANGE_ENVS) -> list[dict[UOp, int]]:
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
    bound = int(r.src[0].arg)
    if bound < 0 or bound and len(envs) > max_envs//bound: raise RuntimeError("RKPLAN_REJECT:static_index_budget")
    envs = [{**env, r: i} for env in envs for i in range(bound)]
  return envs

def _iter_selected_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  """Enumerate only the selected structural axes, preserving dependent output axes as vector lanes."""
  envs:list[dict[UOp, int]] = [{}]
  for r in ranges:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    envs = [{**env, r:i} for env in envs for i in range(int(r.src[0].arg))]
  return envs

def _static_int_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  """Evaluate a compile-time integer expression in compact output order."""
  return RKStaticIndexEvaluator(out_index, count).values(expr, int)

_LinearTerm = UOp|tuple[UOp, int]
def _linear_index(u:UOp, divided:bool=False) -> tuple[int, dict[_LinearTerm, int]]|None:
  """Represent static address arithmetic as a sum of scaled RANGE or RANGE//constant terms."""
  if u.op is Ops.CONST: return int(u.arg), {}
  if divided and u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() in (dtypes.int, dtypes.uint):
    return _linear_index(u.src[0], divided)
  term:_LinearTerm|None = (u, 1) if divided and u.op in (Ops.RANGE, Ops.SPECIAL) else u if u.op in (Ops.RANGE, Ops.SPECIAL) else None
  if divided and u.op is Ops.CDIV and len(u.src) == 2 and u.src[0].op in (Ops.RANGE, Ops.SPECIAL) and \
     u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0: term = (u.src[0], int(u.src[1].arg))
  if term is not None: return 0, {term:1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _linear_index(u.src[0], divided), _linear_index(u.src[1], divided)
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {key:coefficient*scale for key,coefficient in affine[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  terms = lhs[1].copy()
  for key,coefficient in rhs[1].items():
    if (merged:=terms.get(key, 0)+sign*coefficient): terms[key] = merged
    elif key in terms: del terms[key]
  return lhs[0]+sign*rhs[0], terms

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  return typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(u))

class RKStaticIndexEvaluator:
  """Share one compile-time output RANGE materialization across related static gather plans."""
  def __init__(self, out_index:UOp, count:int):
    self.out_index, self.count, self.ranges = out_index, count, _index_ranges(out_index)
    self._vectors:tuple[dict[UOp, np.ndarray], np.ndarray]|None = None
    self._cache:dict[UOp, np.ndarray] = {}

  def _prepare(self) -> tuple[dict[UOp, np.ndarray], np.ndarray]:
    if self._vectors is not None: return self._vectors
    envs = _iter_range_env(self.ranges)
    vector_env = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in self.ranges}
    if (out_affine:=_affine_index(self.out_index)) is None:
      dst = np.broadcast_to(_eval_vector(self.out_index, vector_env, {}), len(envs)).astype(np.int64)
    else:
      dst = np.full(len(envs), out_affine[0], dtype=np.int64)
      for r,stride in out_affine[1].items(): dst += vector_env[r]*stride
    if np.any((dst < 0) | (dst >= self.count)): raise RuntimeError("RKPLAN_REJECT:gather_index")
    self._vectors = vector_env, dst
    return self._vectors

  def values(self, expr:UOp, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
    if any(r not in self.ranges for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
    vector_env, dst = self._prepare()
    raw = np.broadcast_to(_eval_vector(expr, vector_env, self._cache), len(dst))
    if encode is int: encoded = raw.astype(np.int64)
    elif encode is _fp16_bits: encoded = raw.astype(np.float16).view(np.uint16).astype(np.int64)
    elif encode is _int16_bits: encoded = raw.astype(np.int64) & 0xffff
    else: encoded = np.fromiter((encode(x.item()) for x in raw), dtype=np.int64, count=len(dst))
    order = np.argsort(dst)
    ordered_dst, ordered_values = dst[order], encoded[order]
    if len(np.unique(ordered_dst)) != self.count or \
       np.any((ordered_dst[1:] == ordered_dst[:-1]) & (ordered_values[1:] != ordered_values[:-1])):
      raise RuntimeError("RKPLAN_REJECT:static_index")
    values = np.empty(self.count, dtype=np.int64)
    values[dst] = encoded
    return tuple(values.tolist())

  def offsets(self, load_index:UOp, gate:UOp|None) -> tuple[int, ...]:
    if any(r not in self.ranges for r in _index_ranges(load_index) + ([] if gate is None else _index_ranges(gate))):
      raise RuntimeError("RKPLAN_REJECT:gather_index")
    vector_env, dst = self._prepare()
    src = np.broadcast_to(_eval_vector(load_index, vector_env, self._cache), len(dst)).astype(np.int64)
    values = src if gate is None else np.where(np.broadcast_to(_eval_vector(gate, vector_env, self._cache), len(dst)), src, -1)
    if np.any(values < -1): raise RuntimeError("RKPLAN_REJECT:gather_index")
    offsets = np.full(self.count, -2, dtype=np.int64)
    offsets[dst] = values
    if np.any(offsets == -2): raise RuntimeError("RKPLAN_REJECT:gather_index")
    return tuple(offsets.tolist())

def _contiguous_output(out_index:UOp, count:int) -> bool:
  """Prove that an affine output index covers every destination lane once."""
  ranges, affine = _index_ranges(out_index), _affine_index(out_index)
  if affine is None or affine[0] != 0 or set(affine[1]) != set(ranges): return False
  extent = 1
  for r,stride in sorted(affine[1].items(), key=lambda item:item[1]):
    if stride != extent or not r.src or r.src[0].op is not Ops.CONST or (limit:=int(r.src[0].arg)) <= 0: return False
    extent *= limit
  return extent == count

def _typed_load_offsets(load:UOp, dtype:DType, out_index:UOp, count:int, allow_fill:bool=False) -> tuple[UOp, tuple[int, ...]]|None:
  """Resolve one typed global load to bounded static offsets."""
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX: return None
  param = _root_param(load.src[0])
  if param is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None
  try: offsets = RKStaticIndexEvaluator(out_index, count).offsets(load.src[0].src[1], load.src[2] if len(load.src) == 3 else None)
  except RuntimeError: return None
  if any(offset < (-1 if allow_fill else 0) or offset >= int(param.src[0].arg) for offset in offsets): return None
  return param, offsets

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0,
                 index_evaluator:RKStaticIndexEvaluator|None=None) -> RKGather:
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
        return RKGather(src_index, dst_index, count, load_affine[0], tuple(axes))
  if gate is None and out_affine is not None and out_affine[0] == 0 and \
     (load_divided:=typing_cast(tuple[int, dict[tuple[UOp, int], int]]|None, _linear_index(load_index, True))) is not None:
    expected = 1
    for r,dst_stride in sorted(out_affine[1].items(), key=lambda item:item[1]):
      if dst_stride != expected or r.src[0].op is not Ops.CONST or (limit:=int(r.src[0].arg)) <= 0: break
      expected *= limit
    else:
      if expected == count and all(r in out_affine[1] and divisor <= int(r.src[0].arg)
                                   for r,divisor in load_divided[1]):
        divided_axes = tuple((out_affine[1][r]*divisor, (int(r.src[0].arg)+divisor-1)//divisor, stride)
                             for (r,divisor),stride in load_divided[1].items() if stride)
        return RKGather(src_index, dst_index, count, load_divided[0], divided_axes)
  offsets = (index_evaluator or RKStaticIndexEvaluator(out_index, count)).offsets(load_index, gate)
  return RKGather(src_index, dst_index, count, offsets=offsets, fill_bits=fill_bits)

def _validate_gather_bounds(plan:RKGather, source_count:int) -> None:
  if plan.offsets: low, high = min(plan.offsets, default=0), max(plan.offsets, default=-1)
  else:
    low = high = plan.base
    for _,limit,stride in plan.axes:
      if stride < 0: low += (limit-1)*stride
      else: high += (limit-1)*stride
  if low < -1 or high >= source_count: raise RuntimeError("RKPLAN_REJECT:gather_index")

def _gather_cache_key(plans:Iterable[RKGather]) -> tuple:
  return tuple((p.src_index, p.count, p.base, p.axes, p.offsets, p.fill_bits, p.values, p.partial,
                p.dst_stride, p.dst_addend, p.itemsize, p.src_kind) for p in plans)

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is not Ops.MAX or u.dtype.scalar() is not dtypes.half: return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None


def _precise_add(lhs:UOp, rhs:UOp) -> UOp: return UOp(Ops.ADD, lhs.dtype, src=(lhs, rhs), arg=_NATIVE_PRECISE_ADD)
def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return _precise_add(lhs, UOp(Ops.MUL, dtypes.half, src=(rhs, neg_one)))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = UOp(Ops.MUL, dtypes.half, src=(x, splitter))
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(UOp(Ops.MUL, dtypes.half, src=(lhs_high, rhs_high)), term, neg_one)
  error = _precise_add(_precise_add(error, UOp(Ops.MUL, dtypes.half, src=(lhs_high, rhs_low))), UOp(Ops.MUL, dtypes.half, src=(lhs_low, rhs_high)))
  return term, _precise_add(error, UOp(Ops.MUL, dtypes.half, src=(lhs_low, rhs_low)))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = _precise_add(lhs, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  return total, _precise_add(lhs_error, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_add_parts(terms:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 addition residuals as a high lane plus a low correction lane."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  high, middle, low = terms[0], zero, zero
  for part in terms[1:]:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = _precise_add(low, error)
  return high, _precise_add(middle, low)

def _precise_sum_parts(terms:list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 product and addition residuals as a high lane plus a low correction lane."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  return _precise_add_parts(tuple(x[0] for x in pairs) + tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL))

def _tag_precise_adds(root:UOp) -> UOp:
  """Mark additions inside an already compensated physical recipe so they are not expanded again."""
  tagged:dict[UOp, UOp] = {}
  for node in root.toposort():
    tagged[node] = node.replace(src=tuple(tagged[src] for src in node.src),
      arg=_NATIVE_PRECISE_ADD if node.op is Ops.ADD and node.dtype.scalar() is dtypes.half and node.arg is None else node.arg)
  return tagged[root]

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  high, middle = _precise_sum_parts(list(_tag_precise_adds(UOp(Ops.SINK, src=tuple(terms))).src))
  return _precise_add(high, middle)

def _finish_mapped_add_reduction(mapped:RKImage, out_slot:int, rows:int, groups:int, post_scale:float,
                                 reduce_op:Ops=Ops.ADD, int16:bool=False,
                                 op_barriers:bool=False, compensated_limit:int=_reduction_stride(1)//2, kahan:bool=False) -> RKImage|None:
  """Retarget a vector map image into scratch, then append a row-wise ADD reduction."""
  if not mapped.ew_ops or any(gather.after == len(mapped.ew_ops) for gather in mapped.mid_gathers): return None
  lanes = rows*groups
  scratch_shift, value_slot = 1, len(mapped.scratch)+1
  def remap_arg(arg:RKArg) -> RKArg:
    if arg.kind is RKBufferKind.SCRATCH: return replace(arg, index=arg.index+scratch_shift)
    return replace(arg, kind=RKBufferKind.SCRATCH, index=value_slot) if arg.index == out_slot else arg
  mapped = _map_image_args(mapped, remap_arg)
  pre_ops, gathers, host_gathers, host_scatters = mapped.ew_ops, mapped.gathers, mapped.host_gathers, mapped.host_scatters
  ops = list(pre_ops)
  outer, inner = rows, groups
  stride, arena_slot = _reduction_stride(outer), value_slot+1
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, arena_slot, offset)
  scaled_slot = arena_slot+1
  destination = RKArg(RKBufferKind.ARG, out_slot) if post_scale == 1.0 else RKArg(RKBufferKind.SCRATCH, scaled_slot)
  mid:tuple[RKGather, ...] = tuple(RKGather(value_slot, arena_slot, rows, base=group*rows, axes=((1, rows, 1),),
                                            dst_addend=group*(stride//2), src_kind=RKBufferKind.SCRATCH) for group in range(groups))
  if reduce_op is not Ops.ADD:
    reduced = _reduce_arena(ops, [group*stride for group in range(inner)], outer, _EW_CFG[reduce_op], arena, destination,
                            op_barriers=op_barriers, int16=int16)
  elif kahan:
    def temporary(index:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, value_slot, index*stride)
    reduced = _kahan_add(ops, [group*stride for group in range(inner)], outer, arena, temporary, destination,
                         op_barriers=op_barriers)
  elif 2 <= inner <= compensated_limit:
    def temporary(index:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, value_slot, index*stride)
    reduced = _compensated_add(ops, [group*stride for group in range(inner)], outer, arena, temporary, destination,
                               op_barriers=op_barriers)
  else:
    reduced = _reduce_arena(ops, [group*stride for group in range(inner)], outer, _EW_CFG[Ops.ADD], arena, destination,
                            op_barriers=op_barriers)
  if post_scale != 1.0:
    ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, RKArg(RKBufferKind.SCRATCH, 0), rows, _EW_CFG[Ops.MUL],
                      submit_barrier=op_barriers, stateful=op_barriers))
  scratch = (RKScratch(_scratch_bytes(lanes)), *mapped.scratch, RKScratch(_scratch_bytes(lanes)), RKScratch(inner*stride))
  if post_scale != 1.0: scratch += (RKScratch(_scratch_bytes(rows)),)
  mid = mapped.mid_gathers+tuple(replace(gather, after=len(pre_ops)) for gather in mid)
  return RKImage(scratch, struct.pack("<e", post_scale)+mapped.constants,
                 gathers=gathers, ew_ops=tuple(ops), mid_gathers=mid,
                 host_gathers=host_gathers, host_scatters=host_scatters)

def _append_product_residual(mapped:RKImage, out_slot:int, out_index:UOp, product:UOp, lanes:int) -> RKImage|None:
  """Append a TwoProduct residual for `MUL(composed UOps, LOAD)` without recovering its tensor operation."""
  if product.op is not Ops.MUL or product.arg is not None or mapped.host_gathers or mapped.host_scatters or \
     any(gather.after == len(mapped.ew_ops) for gather in mapped.mid_gathers): return None
  direct = next((_strip_cast(src) for src in product.src if _strip_cast(src).op is Ops.LOAD), None)
  if (direct is None or direct.dtype.scalar() is not dtypes.half or len(direct.src) != 1 or direct.src[0].op is not Ops.INDEX or
      (param:=_root_param(direct.src[0])) is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST): return None
  base = len(mapped.scratch)
  lhs, rhs, lhs_high, rhs_high, splitter = (RKArg(RKBufferKind.SCRATCH, base+i) for i in range(5))
  try:
    rhs_gather = _gather_plan(param.arg.slot, rhs.index, out_index, direct.src[0].src[1], None, lanes)
    _validate_gather_bounds(rhs_gather, int(param.src[0].arg))
  except RuntimeError: return None
  product_arg, error_arg = RKArg(RKBufferKind.ARG, out_slot), RKArg(RKBufferKind.ARG, out_slot, lanes*2)
  stages = ((product_arg, lhs, rhs, Ops.MUL),
            (lhs_high, lhs, splitter, Ops.MUL), (rhs_high, lhs_high, lhs, Ops.SUB),
            (lhs_high, lhs_high, rhs_high, Ops.SUB), (lhs, lhs, lhs_high, Ops.SUB),
            (rhs_high, rhs, splitter, Ops.MUL), (error_arg, rhs_high, rhs, Ops.SUB),
            (rhs_high, rhs_high, error_arg, Ops.SUB), (rhs, rhs, rhs_high, Ops.SUB),
            (error_arg, lhs_high, rhs_high, Ops.MUL), (error_arg, error_arg, product_arg, Ops.SUB),
            (lhs_high, lhs_high, rhs, Ops.MUL), (error_arg, error_arg, lhs_high, Ops.ADD),
            (rhs_high, lhs, rhs_high, Ops.MUL), (error_arg, error_arg, rhs_high, Ops.ADD),
            (lhs, lhs, rhs, Ops.MUL), (error_arg, error_arg, lhs, Ops.ADD))
  mapped = _alias_image_args(mapped, {out_slot:lhs})
  return replace(mapped, scratch=(*mapped.scratch, *(RKScratch(_scratch_bytes(lanes)) for _ in range(5))),
    gathers=(*mapped.gathers, rhs_gather, RKGather(out_slot, splitter.index, lanes, values=(_fp16_bits(65.0),)*lanes)),
    ew_ops=(*mapped.ew_ops, *(RKEWOp(dst, left, right, lanes, _EW_CFG[op], submit_barrier=i == 0, stateful=True)
                              for i,(dst,left,right,op) in enumerate(stages))))


def _append_inplace_image(first:RKImage, second:RKImage) -> RKImage|None:
  """Append an in-place EW image, scheduling its input materialization after the first image completes."""
  if not second.ew_ops or second.host_gathers or second.host_scatters or \
     any(gather.after == len(first.ew_ops) for gather in first.mid_gathers): return None
  first_constants, second_constants = len(first.constants)//2, len(second.constants)//2
  def first_arg(arg:RKArg) -> RKArg:
    return replace(arg, index=arg.index+second_constants) \
      if arg.kind is RKBufferKind.SCRATCH and arg.index >= first_constants else arg
  def second_arg(arg:RKArg) -> RKArg:
    if arg.kind is not RKBufferKind.SCRATCH: return arg
    return replace(arg, index=first_constants+arg.index if arg.index < second_constants else len(first.scratch)+arg.index)
  first, second = _map_image_args(first, first_arg), _map_image_args(second, second_arg)
  first_ops, second_ops = first.ew_ops, list(second.ew_ops)
  second_ops[0] = replace(second_ops[0], submit_barrier=True, stateful=True)
  def second_gather(gather:RKGather, after:int) -> RKGather:
    return replace(gather, after=after)
  split = len(first_ops)
  second_mid = tuple(second_gather(gather, split) for gather in second.gathers)+tuple(
    second_gather(gather, split+gather.after) for gather in second.mid_gathers)
  scratch = (first.scratch[:first_constants] + second.scratch[:second_constants] + first.scratch[first_constants:] +
             second.scratch[second_constants:])
  return RKImage(scratch, first.constants+second.constants,
                 gathers=first.gathers, ew_ops=first_ops+tuple(second_ops), mid_gathers=first.mid_gathers+second_mid,
                 host_gathers=first.host_gathers, host_scatters=first.host_scatters)

def _lower_mapped_add_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Evaluate fused FP16 maps over each local ADD reduction, then reduce the materialized lanes."""
  if (output:=_output_store(uops, dtypes.half, allow_local=True)) is None: return None
  store, out, rows, out_index, root = output
  out_ranges = _index_ranges(out_index)
  reduce_ranges = [u for u in uops if u.op is Ops.RANGE and u not in out_ranges]
  if not reduce_ranges or any(r.src[0].op is not Ops.CONST or int(r.src[0].arg) <= 0 for r in reduce_ranges): return None
  if not _contiguous_output(out_index, rows): return None
  updates = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None and any(r in u.toposort() for r in reduce_ranges)]
  if not updates: return None
  post_root, post_scale, value = None, 1.0, root
  while value.op is Ops.CAST: value = value.src[0]
  if len(updates) == 1 and _local_load(value) is not None: pass
  elif len(updates) == 1 and value.op is Ops.MUL and (load:=next((x for x in value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=value.src[1 if value.src[0] is load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else:
    if not any(_local_load(u) is not None for u in root.toposort()): return None
    post_root = root
  if not math.isfinite(post_scale): return None
  next_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  range_substitutions = {axis:axis.replace(src=(axis.src[0],)) for axis in _index_ranges(out_index) if len(axis.src) > 1}
  post_index = store.src[0].substitute(range_substitutions) if range_substitutions else store.src[0]
  combined, replacements = None, dict[UOp, UOp]()
  for update_store in updates:
    update = _strip_cast(update_store.src[1])
    if update.dtype.scalar() is not dtypes.float or update.op is not Ops.ADD or \
       (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None: return None
    term = update.src[1 if update.src[0] is acc else 0]
    axes = [r for r in reduce_ranges if r in _index_ranges(term)] if len(updates) > 1 else reduce_ranges
    if not axes: return None
    if len(updates) > 1:
      clean = {r:r.replace(src=(r.src[0],)) for r in axes if len(r.src) > 1}
      if clean: term, axes = term.substitute(clean), [clean.get(r, r) for r in axes]
    flat, groups = _flat_static_ranges(axes)
    target = out.arg.slot if len(updates) == 1 else next_slot
    if len(updates) != 1: next_slot += 1
    lanes = rows*groups
    linear_index = flat.alu(Ops.MUL, flat.const_like(rows)).alu(Ops.ADD, out_index)
    fake_out = out.replace(src=(out.src[0].const_like(lanes),)) if target == out.arg.slot else UOp.param(target, dtypes.half, (lanes,))
    direct = _strip_cast(term)
    direct_load, kahan = None, groups <= _reduction_stride(1)//2
    mapped:RKImage|None
    if len(updates) == 1 and post_root is None and post_scale == 1.0 and \
       groups >= _MIN_GENERIC_PRODUCT_RESIDUAL_TERMS and \
       direct.op is Ops.MUL and direct.arg is None and direct.dtype.scalar() is dtypes.half and all(
         _strip_cast(src).op is Ops.LOAD and _strip_cast(src).dtype.scalar() is dtypes.half for src in direct.src) and \
       all(len(_index_ranges(_strip_cast(src).src[0].src[1])) <= len(out_ranges)+1 for src in direct.src):
      try:
        terms = [direct.substitute({axis:axis.const_like(value) for axis,value in env.items()})
                 for env in _iter_selected_range_env(axes)]
        precise_store = store.replace(src=(store.src[0], _precise_mul_sum(terms), *store.src[2:]))
        if (precise:=_lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(precise_store,)).toposort())),
                                       vectorize_reductions=False, recipes_ready=True)) is not None: return precise
      except RuntimeError: pass
    if direct.op is Ops.LOAD and direct.dtype.scalar() is dtypes.half and direct.src and direct.src[0].op is Ops.INDEX and \
       (param:=_root_param(direct.src[0])) is not None and param.src[0].op is Ops.CONST:
      try:
        plan = _gather_plan(param.arg.slot, target, linear_index, direct.src[0].src[1], direct.src[2] if len(direct.src) == 3 else None, lanes)
        _validate_gather_bounds(plan, int(param.src[0].arg))
      except RuntimeError: return None
      arg = RKArg(RKBufferKind.ARG, target)
      mapped = RKImage(gathers=(replace(plan, dst_kind=RKBufferKind.ARG),), ew_ops=tuple(
        RKEWOp(replace(arg, addend=start*2), replace(arg, addend=start*2), replace(arg, addend=start*2),
               min(_MAX_EW_ELEMS_FP16, lanes-start), _EW_CFG[Ops.MAX]) for start in range(0, lanes, _MAX_EW_ELEMS_FP16)))
      kahan = True
    else:
      direct_load = next((_strip_cast(src) for src in direct.src if _strip_cast(src).op is Ops.LOAD), None) if direct.op is Ops.MUL else None
      mapped_term = direct.src[1] if direct_load is not None and _strip_cast(direct.src[0]) is direct_load else \
        direct.src[0] if direct_load is not None else term
      fake_store = store.replace(src=(store.src[0].replace(src=(fake_out, linear_index)), mapped_term, *store.src[2:]))
      mapped = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(fake_store,)).toposort())),
                                  vectorize_reductions=False, recipes_ready=True)
      if mapped is not None and direct_load is not None:
        mapped = _append_product_residual(mapped, target, linear_index, direct, lanes)
    physical_groups = groups*(2 if direct_load is not None else 1)
    if mapped is None or (reduced:=_finish_mapped_add_reduction(mapped, target, rows, physical_groups,
        post_scale if post_root is None else 1.0, compensated_limit=physical_groups if direct_load is not None else _reduction_stride(1)//2,
        kahan=kahan)) is None: return None
    combined = reduced if combined is None else _append_inplace_image(combined, reduced)
    if combined is None: return None
    if post_root is None: continue
    buffer = _local_buffer(acc)
    post_local = next((u for u in reversed(list(post_root.toposort())) if _local_load(u) is not None and
                       (ref:=_local_buffer(u)) is not None and buffer is not None and ref.key == buffer.key), None)
    if post_local is None: return None
    load_slot = next_slot if len(updates) == 1 else target
    post_load = UOp(Ops.LOAD, dtypes.half, src=(post_index.replace(src=(UOp.param(load_slot, dtypes.half, (rows,)), *post_index.src[1:])),))
    replacements[post_local] = post_load if post_local.dtype.scalar() is dtypes.half else post_load.cast(post_local.dtype.scalar())
  if post_root is None: return combined
  if combined is None: return None
  post_store = store.replace(src=(post_index, post_root.substitute({**replacements, **range_substitutions}), *store.src[2:]))
  post = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(post_store,)).toposort())),
                            vectorize_reductions=False, recipes_ready=True)
  if post is None: return None
  first_fake = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  if len(updates) == 1: post = _alias_image_args(post, {next_slot:RKArg(RKBufferKind.ARG, out.arg.slot)})
  elif (image:=_append_inplace_image(combined, post)) is None: return None
  else:
    aliases, scratch = {}, list(image.scratch)
    for slot in range(first_fake, next_slot):
      aliases[slot] = RKArg(RKBufferKind.SCRATCH, len(scratch)); scratch.append(RKScratch(_scratch_bytes(rows)))
    return replace(_alias_image_args(image, aliases), scratch=tuple(scratch))
  return _append_inplace_image(combined, post)


def _lower_vectorized_mul_add_reduction(uops:list[UOp]) -> RKImage|None:
  """Execute repeated FP16 MUL UOps with product residuals, then compensate their physical ADD reduction."""
  if (output:=_output_store(uops, dtypes.half)) is None: return None
  _, out, rows, out_index, root = output
  bias:UOp|None = None
  summed, post_scale, relu = root, 1.0, False
  if root.op is Ops.WHERE and len(root.src) == 3 and root.src[0].op is Ops.CMPLT and \
     len(root.src[0].src) == 2 and root.src[0].src[0].op is Ops.CONST and float(root.src[0].src[0].arg) == 0.0 and \
     root.src[0].src[1].key == root.src[1].key and root.src[2].op is Ops.CONST and float(root.src[2].arg) == 0.0:
    summed, relu = root.src[1], True
  if summed.op is Ops.ADD:
    for dot,candidate in (summed.src, summed.src[::-1]):
      candidate = _strip_cast(candidate)
      if (candidate.op is Ops.LOAD and candidate.dtype.scalar() is dtypes.half) or \
         (candidate.dtype.scalar() in (dtypes.half, dtypes.float) and _is_static_expr(candidate)):
        summed, bias = dot, candidate; break
  while True:
    if summed.op is Ops.CAST: summed = summed.src[0]; continue
    value, scale = next(((a, b) for a,b in (summed.src, summed.src[::-1]) if b.op is Ops.CONST), (None, None)) \
      if summed.op is Ops.MUL else (None, None)
    if value is None or scale is None: break
    post_scale *= float(scale.arg); summed = value
  if summed.op is not Ops.ADD or not math.isfinite(post_scale): return None
  terms = tuple(_strip_cast(term) for term in _flatten_binary(summed, Ops.ADD))
  groups, lanes = len(terms), rows*len(terms)
  chunk_lanes = _MAX_EW_ELEMS_FP16*(_reduction_stride(1)//2)
  if groups != 8 and (groups < _MIN_GENERIC_PRODUCT_RESIDUAL_TERMS or groups < 256 and rows <= _MAX_EW_ELEMS_FP16) or \
     8*_scratch_bytes(lanes)+_scratch_bytes(min(chunk_lanes, lanes)) > _MAX_MAPPED_DOT_SCRATCH_BYTES: return None

  index_evaluator = RKStaticIndexEvaluator(out_index, rows)
  parsed:list[tuple[tuple[UOp, RKGather], tuple[UOp, RKGather]]] = []
  for term in terms:
    if term.op is not Ops.MUL or term.arg is not None: return None
    operands:list[tuple[UOp, RKGather]] = []
    for src in term.src:
      load = _strip_cast(src)
      if (load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or not load.src or load.src[0].op is not Ops.INDEX or
          (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtypes.half or
          not param.src or param.src[0].op is not Ops.CONST or len(load.src) > 1 and load.src[1].op is not Ops.CONST): return None
      gate = load.src[2] if len(load.src) > 2 else None
      fill_bits = _fp16_bits(load.src[1].arg if len(load.src) > 1 else 0)
      try:
        plan = _gather_plan(param.arg.slot, 0, out_index, load.src[0].src[1], gate, rows, fill_bits, index_evaluator)
        _validate_gather_bounds(plan, int(param.src[0].arg))
      except RuntimeError: return None
      operands.append((param, plan))
    parsed.append((operands[0], operands[1]))
  slots = (parsed[0][0][0].arg.slot, parsed[0][1][0].arg.slot)
  normalized:list[tuple[tuple[UOp, RKGather], tuple[UOp, RKGather]]] = []
  for pair in parsed:
    pair_slots = (pair[0][0].arg.slot, pair[1][0].arg.slot)
    if pair_slots == slots: normalized.append(pair)
    elif pair_slots[::-1] == slots: normalized.append((pair[1], pair[0]))
    else: return None
  gathers = tuple(replace(operand[1], dst_index=side+1, dst_addend=group*rows)
                  for group,pair in enumerate(normalized) for side,operand in enumerate(pair))
  mapped_ops:list[RKEWOp] = []
  splitter = RKArg(RKBufferKind.SCRATCH, 0)
  for start in range(0, lanes, chunk_lanes):
    count, offset = min(chunk_lanes, lanes-start), start*2
    lhs, rhs = RKArg(RKBufferKind.SCRATCH, 1, offset), RKArg(RKBufferKind.SCRATCH, 2, offset)
    lhs_high, rhs_high = RKArg(RKBufferKind.SCRATCH, 3, offset), RKArg(RKBufferKind.SCRATCH, 4, offset)
    product = RKArg(RKBufferKind.ARG, out.arg.slot, offset)
    error = RKArg(RKBufferKind.ARG, out.arg.slot, lanes*2+offset)
    stages = ((product, lhs, rhs, Ops.MUL),
              (lhs_high, lhs, splitter, Ops.MUL), (rhs_high, lhs_high, lhs, Ops.SUB),
              (lhs_high, lhs_high, rhs_high, Ops.SUB), (lhs, lhs, lhs_high, Ops.SUB),
              (rhs_high, rhs, splitter, Ops.MUL), (error, rhs_high, rhs, Ops.SUB),
              (rhs_high, rhs_high, error, Ops.SUB), (rhs, rhs, rhs_high, Ops.SUB),
              (error, lhs_high, rhs_high, Ops.MUL), (error, error, product, Ops.SUB),
              (lhs_high, lhs_high, rhs, Ops.MUL), (error, error, lhs_high, Ops.ADD),
              (rhs_high, lhs, rhs_high, Ops.MUL), (error, error, rhs_high, Ops.ADD),
              (lhs, lhs, rhs, Ops.MUL), (error, error, lhs, Ops.ADD))
    mapped_ops.extend(RKEWOp(dst, left, right, count, _EW_CFG[op], submit_barrier=i == 0 and bool(start), stateful=True)
                      for i,(dst,left,right,op) in enumerate(stages))
  mapped = RKImage(
    (RKScratch(_scratch_bytes(min(chunk_lanes, lanes))), *(RKScratch(_scratch_bytes(lanes)) for _ in range(4))),
    struct.pack("<e", 65.0), gathers=gathers, ew_ops=tuple(mapped_ops))
  physical_groups = groups*2
  if bias is not None:
    if bias.op is Ops.LOAD:
      bias_param = _root_param(bias.src[0]) if bias.src and bias.src[0].op is Ops.INDEX else None
      if bias_param is None or bias_param.src[0].op is not Ops.CONST: return None
      try: bias_offsets = index_evaluator.offsets(bias.src[0].src[1], bias.src[2] if len(bias.src) > 2 else None)
      except RuntimeError: return None
      if any(not 0 <= offset < int(bias_param.src[0].arg) for offset in bias_offsets): return None
      bias_gather = RKGather(bias_param.arg.slot, out.arg.slot, rows, offsets=bias_offsets,
                             dst_addend=physical_groups*rows, dst_kind=RKBufferKind.ARG)
    else:
      try: values = RKStaticIndexEvaluator(out_index, rows).values(bias, _fp16_bits)
      except RuntimeError: return None
      bias_gather = RKGather(0, out.arg.slot, rows, values=values, dst_addend=physical_groups*rows, dst_kind=RKBufferKind.ARG)
    mapped = replace(mapped, gathers=(*mapped.gathers, bias_gather)); physical_groups += 1
  finished = _finish_mapped_add_reduction(mapped, out.arg.slot, rows, physical_groups, post_scale,
                                           op_barriers=True, compensated_limit=physical_groups)
  if finished is None: return None
  if relu:
    relu_image = RKImage((RKScratch(_scratch_bytes(rows)),), struct.pack("<e", 0.0), ew_ops=(
      RKEWOp(RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.SCRATCH, 0),
             rows, _EW_CFG[Ops.MAX]),))
    if (finished:=_append_inplace_image(finished, relu_image)) is None: return None
  return finished

def _lower_repeated_add_reduction(uops:list[UOp]) -> RKImage|None:
  """Kahan-reduce either direct FP32 load sums or one repeated scalar UOp term."""
  if (output:=_output_store(uops, dtypes.half)) is None: return None
  store, out, count, out_index, root = output
  if (source:=_half_storage(root)) is not None and source.op is Ops.ADD:
    try: direct_terms = _fp32_add_terms(source)
    except _RKGenericReject: direct_terms = []
    groups, lanes = len(direct_terms), count*len(direct_terms)
    if 32 <= groups <= _RKIMAGE_U16_MAX//5 and lanes <= _MAX_EW_ELEMS_FP16:
      direct_gathers:list[RKGather] = []
      for group,term in enumerate(direct_terms):
        if (loaded:=_typed_load_offsets(term, dtypes.half, out_index, count)) is None: break
        param,offsets = loaded
        direct_gathers.append(RKGather(param.arg.slot, out.arg.slot, count, offsets=offsets, dst_addend=group*count,
                                       dst_kind=RKBufferKind.ARG))
      if len(direct_gathers) == groups:
        target = RKArg(RKBufferKind.ARG, out.arg.slot)
        mapped = RKImage(gathers=tuple(direct_gathers), ew_ops=(RKEWOp(target, target, target, lanes, _EW_CFG[Ops.MAX]),))
        return _finish_mapped_add_reduction(mapped, out.arg.slot, count, groups, 1.0, kahan=True, op_barriers=True)
  if count != 1: return None
  summed, post_scale = root, 1.0
  while True:
    if summed.op is Ops.CAST: summed = summed.src[0]; continue
    scaled = next(((value, scale) for value,scale in (summed.src, summed.src[::-1]) if scale.op is Ops.CONST), None) \
      if summed.op is Ops.MUL else None
    if scaled is None: break
    summed, scale = scaled; post_scale *= float(scale.arg)
  if summed.op is not Ops.ADD or not math.isfinite(post_scale): return None
  terms = tuple(_strip_cast(term) for term in _flatten_binary(summed, Ops.ADD))
  if not 128 <= len(terms) <= 512: return None
  loads = tuple(u for u in terms[0].toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.half and
                len(u.src) == 1 and u.src[0].op is Ops.INDEX and _root_param(u.src[0]) is not None)
  if not loads: return None
  peers:dict[UOp, list[UOp]] = {load:[] for load in loads}
  def pair(lhs:UOp, rhs:UOp, found:dict[UOp, UOp]) -> bool:
    if lhs in peers:
      if rhs.op is not Ops.LOAD or rhs.dtype.scalar() is not dtypes.half or len(rhs.src) != 1 or \
         rhs.src[0].op is not Ops.INDEX or _root_param(rhs.src[0]) is None: return False
      if lhs in found and found[lhs].key != rhs.key: return False
      found[lhs] = rhs
      return True
    return lhs.op is rhs.op and lhs.dtype == rhs.dtype and lhs.arg == rhs.arg and len(lhs.src) == len(rhs.src) and \
      all(pair(left, right, found) for left,right in zip(lhs.src, rhs.src))
  for term in terms:
    found:dict[UOp, UOp] = {}
    if not pair(terms[0], term, found) or set(found) != set(loads): return None
    for load in loads: peers[load].append(found[load])

  axis = max((r.arg[0] for r in _index_ranges(out_index) if isinstance(r.arg, tuple)), default=-1)+1
  lane = UOp(Ops.RANGE, out_index.dtype, src=(out_index.const_like(len(terms)),), arg=(axis, AxisType.LOOP))
  next_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  replacements:dict[UOp, UOp] = {}
  sources:list[tuple[int, UOp, tuple[int, ...]]] = []
  for slot,load in enumerate(loads, next_slot):
    parsed = [_typed_load_offsets(peer, dtypes.half, out_index, 1) for peer in peers[load]]
    if any(value is None for value in parsed): return None
    values = [value for value in parsed if value is not None]
    if len({value[0].key for value in values}) != 1: return None
    source = UOp.param(slot, dtypes.half, (len(terms),))
    replacements[load] = source.index(lane).load()
    sources.append((slot, values[0][0], tuple(value[1][0] for value in values)))
  vector = terms[0].substitute(replacements)
  direct = next((_strip_cast(src) for src in vector.src if _strip_cast(src).op is Ops.LOAD), None) if vector.op is Ops.MUL else None
  if direct is None: return None
  expression = vector.src[1] if _strip_cast(vector.src[0]) is direct else vector.src[0]
  fake_out = out.replace(src=(out.src[0].const_like(len(terms)),))
  fake_index = store.src[0].replace(src=(fake_out, lane))
  map_uops = _fp16_rewrite(list(UOp(Ops.SINK, src=(store.replace(src=(fake_index, expression, *store.src[2:])),)).toposort()))
  if (mapped_program:=_lower_uop_program(map_uops, vectorize_reductions=False, recipes_ready=True)) is None or \
     (mapped_result:=_append_product_residual(mapped_program, out.arg.slot, lane, vector, len(terms))) is None: return None
  mapped = mapped_result
  slots = {slot:len(mapped.scratch)+i for i,(slot,_,_) in enumerate(sources)}
  mapped = _alias_image_args(mapped, {slot:_scratch_arg(target) for slot,target in slots.items()})
  gathers = tuple(RKGather(param.arg.slot, slots[slot], len(offsets), offsets=offsets) for slot,param,offsets in sources)
  mapped = replace(mapped, scratch=(*mapped.scratch, *(RKScratch(_scratch_bytes(len(terms))) for _ in sources)),
                   gathers=gathers+mapped.gathers)
  return _finish_mapped_add_reduction(mapped, out.arg.slot, 1, 2*len(terms), post_scale, kahan=True)

def _flatten_binary(root:UOp, op:Ops, *, plain:bool=False) -> list[UOp]:
  leaves, stack = [], [root]
  while stack:
    node = stack.pop()
    if node.op is op and (not plain or node.arg is None): stack.extend(reversed(node.src))
    else: leaves.append(node)
  return leaves

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64
  return vector_bytes, vector_bytes//2, rows*vector_bytes//2

def _stripe_gathers(src_slot:int, dst_slot:int, count:int, rows:Iterable[Iterable[int]], vector_lanes:int, *,
                    values:bool=False, itemsize:int=2) -> tuple[RKGather, ...]:
  """Pack candidate or repeated-current rows into one aligned lane matrix."""
  return tuple(RKGather(src_slot, dst_slot, count, offsets=() if values else tuple(row), values=tuple(row) if values else (),
                        dst_addend=i*vector_lanes, itemsize=itemsize) for i,row in enumerate(rows))

def _physical_lists(minimum:int=0) -> tuple[list[int], Callable[[int], int], list[RKGather], list[RKEWOp]]:
  scratch_sizes:list[int] = []; gathers:list[RKGather] = []; ops:list[RKEWOp] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(minimum, size)); return len(scratch_sizes)-1
  return scratch_sizes, scratch, gathers, ops

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int) -> RKArg:
  return _reduce_arena(ops, list(range(len(active))), count, cfg, active.__getitem__, int16=True)


RKCoordinateRows = tuple[tuple[int, ...], ...]

def _reduce_arena(ops:list[RKEWOp], active:list[int], count:int, cfg:int, arena:Callable[[int], RKArg],
                  out:RKArg|None=None, op_barriers:bool=False, int16:bool=False) -> RKArg:
  """Append a balanced in-place arena reduction and optionally write its final stage directly to output."""
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs, final = active[i], active[i+1], len(active) == 2 and out is not None
      dst = out if final and out is not None else arena(lhs)
      ops.append(RKEWOp(dst, arena(lhs), arena(rhs), count, cfg, submit_barrier=op_barriers and bool(ops),
                        stateful=op_barriers, int16_input=int16, int16_output=int16))
      reduced.append(lhs)
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return out if out is not None else arena(active[0])

def _compensated_add(ops:list[RKEWOp], active:list[int], count:int, arena:Callable[[int], RKArg],
                     temporary:Callable[[int], RKArg], out:RKArg, op_barriers:bool=False) -> RKArg:
  """Reduce aligned FP16 vectors with TwoSum residuals retained in consumed arena lanes."""
  errors:list[int] = []
  while len(active) > 1:
    reduced:list[int] = []
    for lhs,rhs in zip(active[::2], active[1::2]):
      total, delta = temporary(0), temporary(1)
      ops.extend(RKEWOp(dst, lhs_arg, rhs_arg, count, cfg, submit_barrier=op_barriers, stateful=op_barriers)
                 for dst,lhs_arg,rhs_arg,cfg in ((total, arena(lhs), arena(rhs), _EW_CFG[Ops.ADD]),
                   (delta, total, arena(lhs), _EW_CFG[Ops.SUB]), (arena(rhs), arena(rhs), delta, _EW_CFG[Ops.SUB]),
                   (delta, total, delta, _EW_CFG[Ops.SUB]), (arena(lhs), arena(lhs), delta, _EW_CFG[Ops.SUB]),
                   (arena(rhs), arena(lhs), arena(rhs), _EW_CFG[Ops.ADD]), (arena(lhs), total, total, _EW_CFG[Ops.MAX])))
      reduced.append(lhs); errors.append(rhs)
    if len(active)&1: reduced.append(active[-1])
    active = reduced
  residual = _reduce_arena(ops, errors, count, _EW_CFG[Ops.ADD], arena, op_barriers=op_barriers)
  ops.append(RKEWOp(out, arena(active[0]), residual, count, _EW_CFG[Ops.ADD],
                    submit_barrier=op_barriers, stateful=op_barriers))
  return out

def _kahan_add(ops:list[RKEWOp], active:list[int], count:int, arena:Callable[[int], RKArg],
               temporary:Callable[[int], RKArg], out:RKArg, op_barriers:bool=False) -> RKArg:
  """Accumulate an FP16 high lane and reduce its rounding residuals as a balanced low lane."""
  if not active: raise ValueError("empty Kahan reduction")
  total, updated, delta = (temporary(i) for i in range(3))
  def emit(dst:RKArg, lhs:RKArg, rhs:RKArg, op:Ops) -> None:
    ops.append(RKEWOp(dst, lhs, rhs, count, _EW_CFG[op], submit_barrier=op_barriers, stateful=op_barriers))
  emit(total, arena(active[0]), arena(active[0]), Ops.MAX)
  errors:list[int] = []
  for offset in active[1:]:
    emit(updated, total, arena(offset), Ops.ADD)
    emit(delta, total, updated, Ops.SUB)
    emit(arena(offset), delta, arena(offset), Ops.ADD)
    total, updated = updated, total
    errors.append(offset)
  residual = _reduce_arena(ops, errors, count, _EW_CFG[Ops.ADD], arena, op_barriers=op_barriers)
  emit(out, total, residual, Ops.ADD)
  return out

def _int16_byte_bits(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     value:RKArg, lanes:int) -> tuple[RKArg, ...]:
  """Split unsigned byte lanes into eight exact native INT16 0/1 planes."""
  result:list[RKArg|None] = [None]*8
  remainder = value
  for bit in range(7, 0, -1):
    delta, positive, flag, scaled, next_remainder = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, remainder, constants[(1<<bit)-1], lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(flag, positive, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(scaled, flag, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(next_remainder, remainder, scaled, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
    result[bit], remainder = flag, next_remainder
  result[0] = remainder
  return typing_cast(tuple[RKArg, ...], tuple(result))

def _lower_raw_fp16_bitcast(output:RKOutput) -> RKImage|None:
  """Pair adjacent FP16 lane representations into an INT32 output without numeric conversion."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage()
  if value.op is not Ops.BITCAST or value.dtype.scalar() is not dtypes.int or len(value.src) != 1: return None
  packed = value.src[0]
  if packed.op is not Ops.ADD or packed.dtype.scalar() is not dtypes.uint: return None
  lanes:dict[int, UOp] = {}
  for term in packed.src:
    if (term.op is not Ops.SHL or len(term.src) != 2 or term.src[1].op is not Ops.CONST or
        (shift:=int(term.src[1].arg)) not in (0, 16)): return None
    cast = term.src[0]
    if (cast.op is not Ops.CAST or cast.dtype.scalar() is not dtypes.uint or len(cast.src) != 1 or
        cast.src[0].op is not Ops.BITCAST or cast.src[0].dtype.scalar() is not dtypes.ushort or len(cast.src[0].src) != 1): return None
    load = cast.src[0].src[0]
    if load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or len(load.src) != 1 or load.src[0].op is not Ops.INDEX: return None
    lanes[shift] = load
  if len(lanes) != 2: return None
  params = tuple(_root_param(lanes[shift].src[0]) for shift in (0, 16))
  if (any(param is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST for param in params) or
      params[0].arg.slot != params[1].arg.slot): return None  # type: ignore[union-attr]
  source = params[0]; assert source is not None
  try: low, high = (RKStaticIndexEvaluator(out_index, count).offsets(lanes[shift].src[0].src[1], None) for shift in (0, 16))
  except RuntimeError: return None
  source_count = int(source.src[0].arg)
  if any(offset < 0 or offset+1 >= source_count or offset & 1 or high[i] != offset+1 for i,offset in enumerate(low)): return None
  gather = RKGather(source.arg.slot, out_param.arg.slot, count, offsets=tuple(offset//2 for offset in low),
                    dst_kind=RKBufferKind.ARG, itemsize=4)
  return RKImage(gathers=(gather,))


def _int32_less_mask(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     lhs_components:list[RKArg], rhs_components:list[RKArg], lanes:int) -> RKArg:
  """Compare signed INT32 lanes represented as high-to-low widened bytes."""
  def biased_sign(value:RKArg) -> RKArg:
    delta, positive, high, scaled, biased = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, value, constants[127], lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(high, positive, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(scaled, high, constants[256], lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(biased, value, constants[128], lanes, _EW_CFG[Ops.ADD], **_INT16_EW),
                RKEWOp(biased, biased, scaled, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
    return biased
  lhs_components[0], rhs_components[0] = biased_sign(lhs_components[0]), biased_sign(rhs_components[0])
  return _ordered_byte_less(ops, allocate, constants[0], constants[1], lhs_components, rhs_components, lanes)

def _ordered_byte_less(ops:list[RKEWOp], allocate:Callable[[], RKArg], zero:RKArg, one:RKArg,
                       lhs_components:Iterable[RKArg], rhs_components:Iterable[RKArg], lanes:int) -> RKArg:
  less, equal = zero, one
  for lhs,rhs in zip(lhs_components, rhs_components):
    maximum, lhs_delta, rhs_delta, lhs_less, rhs_less, unequal, same, selected, next_less, next_equal = (allocate() for _ in range(10))
    ops.extend((RKEWOp(maximum, lhs, rhs, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(lhs_delta, maximum, lhs, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(rhs_delta, maximum, rhs, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(lhs_less, lhs_delta, one, lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(rhs_less, rhs_delta, one, lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(unequal, lhs_less, rhs_less, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(same, one, unequal, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(selected, equal, lhs_less, lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(next_less, less, selected, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(next_equal, equal, same, lanes, _EW_CFG[Ops.MUL], **_INT16_EW)))
    less, equal = next_less, next_equal
  return less

def _plan_offsets(plan:RKGather) -> tuple[int, ...]:
  if plan.offsets: return plan.offsets
  return tuple(plan.base + sum((lane//divisor % limit)*stride for divisor,limit,stride in plan.axes) for lane in range(plan.count))

def _candidate_gather(src:int, dst:int, rows:tuple[tuple[int, ...], ...], lanes:int) -> RKGather:
  """Gather candidate rows compactly when their active rectangle is affine."""
  n, base = len(rows[0]), rows[0][0]; dx, dy = (rows[0][1]-base if n > 1 else 0), (rows[1][0]-base if len(rows) > 1 else 0)
  affine = lanes%n == 0 and all(value == base+r*dy+c*dx for r,row in enumerate(rows) for c,value in enumerate(row))
  return RKGather(src, dst, len(rows)*lanes, base, ((1, n, dx), (lanes, len(rows), dy)), dst_stride=2, itemsize=1) if affine else \
    RKGather(src, dst, len(rows)*lanes, offsets=tuple(x for row in rows for x in (*row, *(-1,)*(lanes-n))), dst_stride=2, itemsize=1)

def _negative_normalized_index(root:UOp) -> tuple[UOp, int]|None:
  """Recognize `WHERE(index < 0, index + extent, index)` exactly."""
  if root.op is not Ops.WHERE or root.src[0].op is not Ops.CMPLT or root.src[0].src[1].op is not Ops.CONST or \
     int(root.src[0].src[1].arg) != 0: return None
  index = root.src[0].src[0]
  if index.op is not Ops.LOAD or index.dtype.scalar() is not dtypes.int or root.src[2].key != index.key or root.src[1].op is not Ops.ADD:
    return None
  indexes = [x for x in root.src[1].src if x.key == index.key]
  extents = [x for x in root.src[1].src if x.op is Ops.CONST and int(x.arg) > 0]
  return (index, int(extents[0].arg)) if len(indexes) == len(extents) == 1 else None

def _bounded_index_gate(gate:UOp, bounded:UOp, limit:int) -> bool:
  """Require the canonical nonnegative and strict-upper-bound checks within a conjunction."""
  if gate.op is not Ops.AND: return False
  nodes = gate.toposort()
  upper = any(u.op is Ops.CMPLT and u.src[0].key == bounded.key and u.src[1].op is Ops.CONST and
              int(u.src[1].arg) == limit for u in nodes)
  lower = any(u.op is Ops.CMPNE and any(marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and
              comparison.op is Ops.CMPLT and comparison.src[0].key == bounded.key and comparison.src[1].op is Ops.CONST and
              int(comparison.src[1].arg) == 0 for comparison,marker in (u.src, u.src[::-1])) for u in nodes)
  return lower and upper

def _bool_tautology(root:UOp, max_atoms:int=8) -> bool:
  """Prove a Boolean expression for all assignments without assuming relations between comparisons."""
  atoms:dict[UOp, int] = {}
  def evaluate(u:UOp, assignment:int) -> bool:
    if u.op is Ops.CONST and u.dtype is dtypes.bool: return bool(u.arg)
    if len(u.src) == 2 and all(src.dtype is dtypes.bool for src in u.src):
      lhs, rhs = evaluate(u.src[0], assignment), evaluate(u.src[1], assignment)
      if u.op in (Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ): return bool(python_alu[u.op](lhs, rhs))
    return bool(assignment & (1 << atoms.setdefault(u, len(atoms))))
  evaluate(root, 0)
  return len(atoms) <= max_atoms and all(evaluate(root, assignment) for assignment in range(1 << len(atoms)))

def _tautological_store_scope(uops:list[UOp], store:UOp) -> bool:
  """Accept a STORE only when every enclosing structured IF is unconditionally true."""
  stack:list[UOp] = []; gates:tuple[UOp, ...]|None = None
  for u in uops:
    if u.op is Ops.IF and (u.dtype is not dtypes.void or len(u.src) != 2 or u.src[0].dtype is not dtypes.bool): return False
    if u.op is Ops.IF: stack.append(u)
    elif u.op is Ops.ENDIF and (len(u.src) != 1 or not stack or u.src[0].key != stack.pop().key): return False
    elif u.op is Ops.STORE and u.key == store.key and gates is not None: return False
    elif u.op is Ops.STORE and u.key == store.key: gates = tuple(scope.src[0] for scope in stack)
  return gates is not None and not stack and all(_bool_tautology(gate) for gate in gates)

def _native_int16_byte_mask(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], index_slot:int,
                            index_offsets:tuple[int, ...]|tuple[tuple[int, ...], ...], coordinate_sets:tuple[RKCoordinateRows, ...],
                            count:int, vector_lanes:int) -> RKArg|None:
  """Compare arbitrary INT32 values exactly as four unsigned bytes using native INT16 DPU EW."""
  rows = len(coordinate_sets[0]) if coordinate_sets else 0
  if not rows or any(len(group) != rows or any(len(row) != count for row in group) for group in coordinate_sets): return None
  matrix_lanes = rows*vector_lanes
  if index_offsets and isinstance(index_offsets[0], int): offset_rows = (index_offsets,)*rows
  else: offset_rows = typing_cast(tuple[tuple[int, ...], ...], index_offsets)
  if len(offset_rows) != rows or any(len(offsets) != count for offsets in offset_rows): return None
  offset_arrays = tuple(np.asarray(offsets, dtype=np.int64) for offsets in offset_rows)
  one, diff, magnitude, unequal = (scratch(matrix_lanes*2) for _ in range(4))
  gathers.append(RKGather(index_slot, one, matrix_lanes, values=(1,)*matrix_lanes, itemsize=2))
  masks:list[RKArg] = []
  for coordinates in coordinate_sets:
    coordinate_matrix = np.zeros((rows, vector_lanes), dtype=np.int64)
    coordinate_matrix[:, :count] = np.asarray(coordinates, dtype=np.int64)
    byte_masks:list[RKArg] = []
    for byte in range(4):
      dynamic, static, equal = (scratch(matrix_lanes*2) for _ in range(3))
      gathers.extend(RKGather(index_slot, dynamic, count, offsets=tuple((offsets*4+byte).tolist()),
        dst_stride=2, dst_addend=row*vector_lanes*2, itemsize=1) for row,offsets in enumerate(offset_arrays))
      values = tuple(((coordinate_matrix >> (byte*8)) & 0xff).ravel().tolist())
      gathers.append(RKGather(index_slot, static, matrix_lanes, values=values, itemsize=2))
      ops.extend((RKEWOp(_scratch_arg(diff), _scratch_arg(dynamic), _scratch_arg(static), matrix_lanes,
                          _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), matrix_lanes, _EW_CFG_ABS, **_INT16_EW),
                  RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, **_INT16_EW),
                  RKEWOp(_scratch_arg(equal), _scratch_arg(one), _scratch_arg(unequal), matrix_lanes,
                          _EW_CFG[Ops.SUB], **_INT16_EW)))
      byte_masks.append(_scratch_arg(equal))
    mask = byte_masks[0]
    for byte_mask in byte_masks[1:]:
      dst = scratch(matrix_lanes*2); ops.append(RKEWOp(_scratch_arg(dst), mask, byte_mask, matrix_lanes,
        _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)); mask = _scratch_arg(dst)
    masks.append(mask)
  mask = masks[0]
  for alternate in masks[1:]:
    dst = scratch(matrix_lanes*2); ops.append(RKEWOp(_scratch_arg(dst), mask, alternate, matrix_lanes,
      _EW_CFG[Ops.MAX], int16_input=True, int16_output=True)); mask = _scratch_arg(dst)
  return mask

def _native_integer_nonzero_mask(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], source_slot:int,
                                 rows:tuple[tuple[int, ...], ...], count:int, vector_lanes:int, itemsize:int=4) -> RKArg|None:
  """Test opaque integer bytes for nonzero using unsigned-byte INT16 MIN/MAX stages."""
  if not rows or any(len(row) != count for row in rows): return None
  matrix_lanes = len(rows)*vector_lanes
  one = scratch(matrix_lanes*2)
  byte_masks = tuple(scratch(matrix_lanes*2) for _ in range(itemsize))
  gathers.append(RKGather(source_slot, one, matrix_lanes, values=(1,)*matrix_lanes, itemsize=2))
  for byte,slot in enumerate(byte_masks):
    gathers.extend(RKGather(source_slot, slot, count,
      offsets=tuple(offset*itemsize+byte if offset >= 0 else -1 for offset in row), dst_addend=i*vector_lanes*2,
      dst_stride=2, itemsize=1) for i,row in enumerate(rows))
  for slot in byte_masks:
    ops.append(RKEWOp(_scratch_arg(slot), _scratch_arg(slot), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, int16_input=True, int16_output=True))
  mask = _scratch_arg(byte_masks[0])
  for slot in byte_masks[1:]:
    ops.append(RKEWOp(mask, mask, _scratch_arg(slot), matrix_lanes, _EW_CFG[Ops.MAX], int16_input=True, int16_output=True))
  return mask

def _full_predicate_count(expr:UOp, out_index:UOp, count:int, dtype:DType, predicate:Callable[[UOp], UOp|None],
                          max_scale:int=1) -> tuple[UOp, int]|None:
  """Prove an unrolled sum covers every typed source predicate uniformly, possibly repeated or scaled."""
  terms, source, offsets, scales = _flatten_binary(expr, Ops.ADD), None, [], []
  try:
    for term in terms:
      scale = 1
      if max_scale > 1 and term.op is Ops.MUL:
        constants = [u for u in term.src if u.op is Ops.CONST and u.dtype.scalar() is dtypes.int]
        values = [u for u in term.src if u not in constants]
        if len(constants) != 1 or len(values) != 1: return None
        scale, term = int(constants[0].arg), values[0]
      if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.int or len(term.src) != 1 or \
         (load:=predicate(term.src[0])) is None or [u for u in term.toposort() if u.op is Ops.LOAD] != [load]: return None
      param = _root_param(load.src[0])
      if (param is None or param.dtype.scalar() is not dtype or param.src[0].op is not Ops.CONST or
          source is not None and param.arg.slot != source.arg.slot): return None
      source = param
      row = RKStaticIndexEvaluator(out_index, count).offsets(load.src[0].src[1], None)
      if len(set(row)) != 1: return None
      offsets.append(row[0]); scales.append(scale)
  except RuntimeError: return None
  if source is None or not scales or len(set(scales)) != 1: return None
  source_count = int(source.src[0].arg)
  if source_count <= 0 or len(terms)%source_count or any(offsets.count(i) != len(terms)//source_count for i in range(source_count)): return None
  effective_scale = scales[0]*(len(terms)//source_count)
  return (source, effective_scale) if 1 <= effective_scale <= max_scale else None


RKBoundedPredicateCoordinates = tuple[UOp, int, UOp, tuple[int, ...], tuple[tuple[int, ...], ...], int]

def _bounded_predicate_coordinate_plan(output:RKOutput, dtype:DType, predicate:Callable[[UOp], UOp|None],
                                       encodable:Callable[[int], bool]) -> RKBoundedPredicateCoordinates|None:
  """Prove a bounded predicate count plus dynamic-rank coordinate selection and fill program."""
  _, _, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or root.op is not Ops.WHERE or len(root.src) != 3: return None
  fill = root.src[2]
  if fill.op is not Ops.CONST or fill.dtype.scalar() is not dtypes.int: return None
  fill_value = int(fill.arg)
  totals = [(u.src[1], info) for u in root.toposort() if u.op is Ops.CMPLT and u.src[0].key == out_index.key and
            (info:=_full_predicate_count(u.src[1], out_index, count, dtype, predicate, 8)) is not None]
  if len(totals) != 1: return None
  total_expr, (source, rank) = totals[0]
  coordinate_count = int(source.src[0].arg)*rank
  if not 1 <= coordinate_count <= _FP16_EXACT_INTEGER: return None
  source_loads = {u.key for u in total_expr.toposort() if u.op is Ops.LOAD}
  index_loads = [u for u in root.toposort() if u.op is Ops.LOAD and u.key not in source_loads]
  if len(index_loads) != 1 or {u.key for u in root.toposort() if u.op is Ops.LOAD} != source_loads|{index_loads[0].key}: return None
  index_load = index_loads[0]
  index_param = _root_param(index_load.src[0]) if index_load.src and index_load.src[0].op is Ops.INDEX else None
  if index_param is None or index_param.dtype.scalar() is not dtypes.int or index_param.src[0].op is not Ops.CONST or \
     int(index_param.src[0].arg) != count: return None
  try:
    index_offsets = RKStaticIndexEvaluator(out_index, count).offsets(index_load.src[0].src[1], None)
    total_const = total_expr.const_like(count)
    coordinate_rows = tuple(_static_int_vector(out_index, root.substitute(
      {total_expr:total_const, index_load:index_load.const_like(i)}), count) for i in range(coordinate_count))
    first = coordinate_rows[0]
    for selected_count in range(count+1):
      got = _static_int_vector(out_index, root.substitute(
        {total_expr:total_expr.const_like(selected_count), index_load:index_load.const_like(0)}), count)
      if got != tuple(first[lane] if lane < selected_count else fill_value for lane in range(count)): return None
    if index_offsets != tuple(range(count)) or not encodable(fill_value) or \
       any(not encodable(value) for row in coordinate_rows for value in row): return None
  except (RuntimeError, OverflowError, struct.error): return None
  return source, rank, index_param, index_offsets, coordinate_rows, fill_value

def _lower_bounded_integer_predicate_coordinates(output:RKOutput, dtype:DType) -> RKImage|None:
  """Execute bounded typed predicate coordinates through native INT16 masks."""
  _, out_param, count, _, _ = output
  if (plan:=_bounded_predicate_coordinate_plan(output, dtype, lambda u:_nonzero_load(u, dtype),
                                               lambda value:-32768 <= value <= 32767)) is None: return None
  source, rank, index_param, index_offsets, coordinate_rows, fill_value = plan
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, coordinate_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  scratch_sizes, scratch, gathers, ops = _physical_lists(64)
  _, source_vector_lanes, source_matrix_lanes = _stripe_layout(1, source_count)
  source_rows = tuple((lane,) for lane in range(source_count))
  if (source_mask:=_native_integer_nonzero_mask(ops, gathers, scratch, source.arg.slot, source_rows, 1,
                                               source_vector_lanes, dtype.itemsize)) is None: return None
  total = _reduce_rows(ops, [replace(source_mask, addend=source_mask.addend+row*64) for row in range(source_count)],
                       1, _EW_CFG[Ops.ADD])
  if rank != 1:
    rank_slot = scratch(2)
    gathers.append(RKGather(source.arg.slot, rank_slot, 1, values=(_int16_bits(rank),)))
    ops.append(RKEWOp(total, total, _scratch_arg(rank_slot), 1, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
  candidates = tuple((candidate,)*count for candidate in range(coordinate_count))
  if (equal:=_native_int16_byte_mask(ops, gathers, scratch, index_param.arg.slot, index_offsets,
                                     (candidates,), count, vector_lanes)) is None: return None
  gather_after = len(ops)
  total_vector, output_coordinate, zero, one, coordinate_matrix = (scratch(matrix_lanes*2) for _ in range(5))
  mid = (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count,
                  src_kind=RKBufferKind.SCRATCH, after=gather_after),)
  gathers.extend(_stripe_gathers(source.arg.slot, coordinate_matrix, count,
    tuple(tuple(_int16_bits(value) for value in row) for row in coordinate_rows), vector_lanes, values=True))
  gathers.extend((RKGather(source.arg.slot, output_coordinate, count, values=tuple(range(count))),
                  RKGather(source.arg.slot, zero, matrix_lanes, values=(0,)*matrix_lanes),
                  RKGather(source.arg.slot, one, matrix_lanes, values=(1,)*matrix_lanes)))
  fill_slot = scratch(count*2)
  gathers.append(RKGather(source.arg.slot, fill_slot, count, values=(_int16_bits(fill_value),)*count))
  valid_delta, positive, valid, remaining = (scratch(count*2) for _ in range(4))
  selected, guarded, fill_part, result = (scratch(matrix_lanes*2) for _ in range(4))
  ops.extend((RKEWOp(_scratch_arg(valid_delta), _scratch_arg(total_vector), _scratch_arg(output_coordinate), count, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(_scratch_arg(positive), _scratch_arg(valid_delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(_scratch_arg(valid), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(_scratch_arg(remaining), _scratch_arg(one), _scratch_arg(valid), count, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(_scratch_arg(selected), equal, _scratch_arg(coordinate_matrix), matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW)))
  selected_value = _reduce_rows(ops, [_scratch_arg(selected, row*vector_bytes) for row in range(coordinate_count)],
                                count, _EW_CFG[Ops.ADD])
  ops.extend((RKEWOp(_scratch_arg(guarded), selected_value, _scratch_arg(valid), count, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(_scratch_arg(fill_part), _scratch_arg(fill_slot), _scratch_arg(remaining), count, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(_scratch_arg(result), _scratch_arg(guarded), _scratch_arg(fill_part), count, _EW_CFG[Ops.ADD], **_INT16_EW),
              RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), _scratch_arg(result), _scratch_arg(result), count, _EW_CFG[Ops.MAX],
                     int16_input=True, int32_output=True)))
  return RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid)


def _lower_dynamic_typed_load(output:RKOutput, dtype:DType) -> RKImage|None:
  """Parse and materialize one bounded dynamic typed LOAD as exact candidate-major raw bytes."""
  _, out_param, count, out_index, load = output
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or
      load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or load.src[1].arg != 0): return None
  data_param, data_index, gate = _root_param(load.src[0]), load.src[0].src[1], load.src[2]
  data_nodes, gate_nodes = data_index.toposort(), gate.toposort()
  bool_loads = tuple(u for u in gate_nodes if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.bool)
  normalized = tuple((u, *parsed) for u in data_nodes if (parsed:=_negative_normalized_index(u)) is not None)
  normalized_by_load = {dynamic.key:(root, extent) for root,dynamic,extent in normalized}; \
    loads = tuple({u.key:u for u in data_nodes if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}.values())
  if len(normalized_by_load) != len(normalized): return None
  axes:list[tuple[UOp, UOp, int, bool]] = []
  for dynamic in loads:
    if dynamic.key in normalized_by_load: root, extent, wrapped = *normalized_by_load[dynamic.key], True
    else:
      limits = {int(u.src[1].arg) for u in gate_nodes if u.op is Ops.CMPLT and u.src[0].key == dynamic.key and
                u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0}
      if len(limits) != 1: return None
      root, extent, wrapped = dynamic, next(iter(limits)), False
    if not _bounded_index_gate(gate, root, extent): return None
    axes.append((root, dynamic, extent, wrapped))
  if (data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or not axes or
      len(bool_loads) > 1 or {u.key for u in gate_nodes if u.op is Ops.LOAD} != {u.key for u in (*loads, *bool_loads)}): return None
  params = typing_cast(tuple[UOp, ...], tuple(_root_param(u.src[0]) if u.src and u.src[0].op is Ops.INDEX else None for u in loads))
  if any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in params): return None
  bool_load = bool_loads[0] if bool_loads else None
  bool_param = _root_param(bool_load.src[0]) if bool_load is not None and len(bool_load.src) == 1 and bool_load.src[0].op is Ops.INDEX else None
  if bool_load is not None and (bool_load not in _flatten_binary(gate, Ops.AND) or bool_param is None or
      bool_param.dtype.scalar() is not dtypes.bool or bool_param.src[0].op is not Ops.CONST): return None
  if any(not 0 <= param.arg.slot <= _RKIMAGE_U16_MAX for param in
         (out_param, data_param, *params, *((bool_param,) if bool_param is not None else ()))): return None
  candidates = math.prod(extent for _,_,extent,_ in axes); data_count = int(data_param.src[0].arg)
  alternative_count, external_count, pre_repeat = sum(1+wrapped for *_,wrapped in axes), int(bool_load is not None), min(8, count)
  pre_vl = _stripe_layout(count, 2)[1] if candidates > 1 else count
  if pre_vl > _MAX_EW_ELEMS_FP16: return None
  pre_blocks = ceildiv(candidates, _MAX_EW_ELEMS_FP16//pre_vl)
  if candidates > _MAX_STATIC_RANGE_ENVS or candidates > _MAX_STATIC_LOCAL_STEPS//(count*dtype.itemsize+len(axes)) or any(
     size > _RKIMAGE_U16_MAX for size in
     (pre_blocks*(4+12*alternative_count+2*pre_repeat*dtype.itemsize)+external_count,
      pre_blocks*(1+8*alternative_count+pre_repeat*dtype.itemsize)+external_count+pre_repeat*dtype.itemsize)): return None
  combinations:tuple[tuple[int, ...], ...] = ((),)
  for _,_,extent,_ in axes: combinations = tuple(prefix+(value,) for prefix in combinations for value in range(extent))
  evaluator, external = RKStaticIndexEvaluator(out_index, count), None
  if bool_load is not None and bool_param is not None:
    try: bool_offsets = evaluator.offsets(bool_load.src[0].src[1], None)
    except RuntimeError: return None
    if any(not 0 <= offset < int(bool_param.src[0].arg) for offset in bool_offsets): return None
    external = (bool_param.arg.slot, bool_offsets)
  try:
    index_offsets = tuple(evaluator.offsets(dynamic.src[0].src[1], None) for dynamic in loads)
    plans = tuple(_gather_plan(data_param.arg.slot, 0, out_index, data_index.substitute(mapping),
      gate.substitute(mapping | ({bool_loads[0]:bool_loads[0].const_like(True)} if bool_loads else {})), count, index_evaluator=evaluator)
      for values in combinations for mapping in ({dynamic:dynamic.const_like(value) for dynamic,value in zip(loads, values)},))
    plan_offsets = tuple(_plan_offsets(plan) for plan in plans)
  except RuntimeError: return None
  if (any(not 0 <= offset < int(param.src[0].arg) or offset*4+3 > dtypes.int.max
          for offsets,param in zip(index_offsets, params) for offset in offsets) or
      any(not 0 <= offset < data_count or offset*dtype.itemsize+dtype.itemsize-1 > dtypes.int.max
          for row in plan_offsets for offset in row) or
      external is not None and any(offset > dtypes.int.max for offset in external[1])): return None
  coordinates = tuple(tuple(values[axis] for values in combinations) for axis in range(len(axes)))
  repeat = next((width for width in range(min(8, count), 0, -1) if count%width == 0 and
    all(all(row[start:start+width] == (row[start],)*width for start in range(0, count, width))
        for row in (*index_offsets, *((external[1],) if external else ())))), 1)
  group_count = count//repeat; grouped_indices = tuple((param.arg.slot, offsets[::repeat]) for param,offsets in zip(params, index_offsets))
  gg = None if external is None else (external[0], external[1][::repeat]); vl = _stripe_layout(group_count, 2)[1] if candidates > 1 else group_count
  if vl > _MAX_EW_ELEMS_FP16: return None
  block_count = ceildiv(candidates, _MAX_EW_ELEMS_FP16//vl)
  if any(size > _RKIMAGE_U16_MAX for size in
         (block_count*(4+12*alternative_count+2*repeat*dtype.itemsize)+external_count,
          block_count*(1+8*alternative_count+repeat*dtype.itemsize)+external_count+repeat*dtype.itemsize)): return None
  scratch_sizes, scratch, gathers, ops = _physical_lists(); blocks:list[list[list[RKArg]]] = []
  for start in range(0, candidates, _MAX_EW_ELEMS_FP16//vl):
    stop = min(candidates, start+_MAX_EW_ELEMS_FP16//vl)
    rows, matrix_lanes, pad = stop-start, (stop-start)*vl, (-1,)*(vl-group_count)
    one, diff, mag, unequal = (scratch(matrix_lanes*2) for _ in range(4)); gathers.append(RKGather(0, one, matrix_lanes, values=(1,)*matrix_lanes))
    axis_masks:list[RKArg] = []
    for axis,((index_slot, offsets), values) in enumerate(zip(grouped_indices, coordinates)):
      alternatives = (values, *((tuple(value-axes[axis][2] for value in values),) if axes[axis][3] else ())); masks:list[RKArg] = []
      for alternative in alternatives:
        byte_masks:list[RKArg] = []
        for byte in range(4):
          dynamic_slot, static_slot, equal_slot = (scratch(matrix_lanes*2) for _ in range(3))
          dynamic_rows = ((tuple(offset*4+byte for offset in offsets)),)*rows
          static_bytes = tuple(x for value in alternative[start:stop] for x in (*(((value >> (byte*8)) & 0xff,)*group_count), *(0,)*len(pad)))
          gathers.extend((_candidate_gather(index_slot, dynamic_slot, dynamic_rows, vl),
                          RKGather(index_slot, static_slot, matrix_lanes, values=static_bytes)))
          ops.extend((RKEWOp(_scratch_arg(diff), _scratch_arg(dynamic_slot), _scratch_arg(static_slot), matrix_lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                      RKEWOp(_scratch_arg(mag), _scratch_arg(diff), _scratch_arg(diff), matrix_lanes, _EW_CFG_ABS, **_INT16_EW),
                      RKEWOp(_scratch_arg(unequal), _scratch_arg(mag), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, **_INT16_EW),
                      RKEWOp(_scratch_arg(equal_slot), _scratch_arg(one), _scratch_arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
          byte_masks.append(_scratch_arg(equal_slot))
        masks.append(_reduce_rows(ops, byte_masks, matrix_lanes, _EW_CFG[Ops.MUL]))
      axis_masks.append(_reduce_rows(ops, masks, matrix_lanes, _EW_CFG[Ops.MAX]))
    mask = _reduce_rows(ops, axis_masks, matrix_lanes, _EW_CFG[Ops.MUL])
    block:list[list[RKArg]] = []
    for channel in range(repeat):
      result:list[RKArg] = []
      for byte in range(dtype.itemsize):
        raw, selected = scratch(matrix_lanes*2), scratch(matrix_lanes*2)
        raw_rows = tuple(tuple(x*dtype.itemsize+byte for x in row[channel::repeat]) for row in plan_offsets[start:stop])
        gathers.append(_candidate_gather(data_param.arg.slot, raw, raw_rows, vl))
        value = _scratch_arg(selected); ops.append(RKEWOp(value, _scratch_arg(raw), mask, matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
        active = [_scratch_arg(selected, row*vl*2) for row in range(rows)]
        while len(active) > 1:
          ops.extend(RKEWOp(active[pair], active[pair], active[-pair-1], group_count, _EW_CFG[Ops.ADD], **_INT16_EW)
                     for pair in range(len(active)//2))
          active = active[:(len(active)+1)//2]
        result.append(active[0])
      block.append(result)
    blocks.append(block)
  results = blocks[0]
  for block in blocks[1:]:
    for result,other in zip(results, block):
      for value,part in zip(result, other): ops.append(RKEWOp(value, value, part, group_count, _EW_CFG[Ops.ADD], **_INT16_EW))
  if gg is not None:
    gate_slot = scratch(group_count*2)
    gathers.append(RKGather(gg[0], gate_slot, group_count, offsets=gg[1], dst_stride=2, itemsize=1))
    ops.extend(RKEWOp(value, value, _scratch_arg(gate_slot), group_count, _EW_CFG[Ops.MUL], **_INT16_EW)
               for result in results for value in result)
  terminal = tuple(RKGather(value.index, out_param.arg.slot, group_count, offsets=tuple(range(value.addend, value.addend+group_count*2, 2)),
    dst_stride=repeat*dtype.itemsize, dst_addend=channel*dtype.itemsize+byte, dst_kind=RKBufferKind.ARG,
    src_kind=RKBufferKind.SCRATCH, itemsize=1) for channel,result in enumerate(results) for byte,value in enumerate(result))
  image = RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                  mid_gathers=tuple(replace(gather, after=len(ops)) for gather in terminal))
  return image if _image_u16_safe(image) else None

def _nonzero_load(term:UOp, dtype:DType=dtypes.half) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtype and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and zero.arg == 0]
  return candidates[0] if len(candidates) == 1 else None


def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if root.op is Ops.CAST and root.dtype.scalar() is dtypes.bool and len(root.src) == 1 and root.src[0].dtype.scalar() is dtypes.half:
    root = root.src[0] != UOp.const(0.0, dtypes.half)
  if (load:=_nonzero_load(root)) is None: return None
  magnitude = UOp(Ops.MAX, dtypes.half, src=(load, load), arg=_NATIVE_ABS)
  return _positive_mask(magnitude)

def _half_backed(values:tuple[UOp, ...]) -> bool:
  return all(value.dtype.scalar() in (dtypes.half, dtypes.float) and all(load.src and load.src[0].op is Ops.INDEX and
    (param:=_root_param(load.src[0])) is not None and param.dtype.scalar() is dtypes.half
    for load in value.toposort() if load.op is Ops.LOAD) for value in values)

def _exact_int_range(root:UOp) -> tuple[int, int]|None:
  """Conservatively bound an integer UOp before choosing its exact physical scratch layout."""
  cache:dict[UOp, bool] = {}
  def supported(u:UOp) -> bool:
    if u in cache: return cache[u]
    if u.dtype.scalar() not in (dtypes.int, dtypes.weakint): valid = False
    elif u.op is Ops.CONST: valid = True
    elif u.op in (Ops.RANGE, Ops.SPECIAL) and u.src and u.src[0].op is Ops.CONST: valid = True
    elif u.op is Ops.CAST and len(u.src) == 1: valid = u.src[0].dtype.scalar() is dtypes.bool or supported(u.src[0])
    elif u.op is Ops.WHERE and len(u.src) == 3: valid = all(supported(src) for src in u.src[1:])
    elif u.op is Ops.XOR and len(u.src) == 2:
      valid = any(marker.op is Ops.CONST and marker.arg == -1 and supported(source) for marker,source in (u.src, u.src[::-1]))
    elif u.op in (Ops.CDIV, Ops.CMOD) and len(u.src) == 2:
      valid = supported(u.src[1]) and u.src[1].vmin == u.src[1].vmax != 0 and (u.op is Ops.CMOD or supported(u.src[0]))
    else: valid = u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2 and all(supported(src) for src in u.src)
    if valid:
      low, high, dtype = int(u.vmin), int(u.vmax), u.dtype.scalar()
      valid = dtype.min <= low <= high <= dtype.max
    cache[u] = valid
    return valid
  if not supported(root): return None
  return (0, max(0, int(root.vmax))) if root.op in (Ops.RANGE, Ops.SPECIAL) else (int(root.vmin), int(root.vmax))

def _int_fp16_expr(u:UOp) -> UOp:
  """Represent an integer UOp whose values are exactly carried in FP16 lanes as a half-valued recipe."""
  if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.int, dtypes.weakint): return UOp.const(float(int(u.arg)), dtypes.half)
  if u.dtype.scalar() not in (dtypes.int, dtypes.weakint): raise _RKGenericReject
  if u.op in (Ops.LOAD, Ops.RANGE, Ops.SPECIAL): return u.cast(dtypes.half)
  if u.op is Ops.CAST and len(u.src) == 1:
    if u.src[0].dtype.scalar() is dtypes.half: return _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=u.src))
    if u.src[0].dtype.scalar() is dtypes.bool: return _int_fp16_bool(u.src[0]).cast(dtypes.half)
    if u.src[0].dtype.scalar() in (dtypes.int, dtypes.weakint): return _int_fp16_expr(u.src[0])
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2:
    return UOp(u.op, dtypes.half, src=tuple(_int_fp16_expr(src) for src in u.src), arg=u.arg)
  if u.op is Ops.WHERE and len(u.src) == 3:
    return UOp(Ops.WHERE, dtypes.half, src=(_int_fp16_bool(u.src[0]), _int_fp16_expr(u.src[1]), _int_fp16_expr(u.src[2])), arg=u.arg)
  if u.op is Ops.CMOD and len(u.src) == 2:
    lhs, rhs = (_int_fp16_expr(src) for src in u.src)
    quotient = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(lhs.alu(Ops.FDIV, rhs),)))
    return lhs.alu(Ops.SUB, quotient.alu(Ops.MUL, rhs))
  if u.op is Ops.CDIV and len(u.src) == 2:
    lhs, rhs = (_int_fp16_expr(src) for src in u.src)
    return _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(lhs.alu(Ops.FDIV, rhs),)))
  raise _RKGenericReject(f"INT_FP16 recipe {u.op.name}")

def _int_fp16_bool(u:UOp) -> UOp:
  """Translate predicates internal to one proven-bounded integer recipe into FP16 comparisons."""
  if u.op is Ops.CONST and u.dtype is dtypes.bool: return u
  if u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() in (dtypes.half, dtypes.float) for src in u.src): return u
  if u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(
    src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src
  ): return UOp(u.op, dtypes.bool, src=tuple(_int_fp16_expr(src) for src in u.src), arg=u.arg)
  if u.op in (Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype is dtypes.bool for src in u.src):
    return UOp(u.op, dtypes.bool, src=tuple(_int_fp16_bool(src) for src in u.src), arg=u.arg)
  raise _RKGenericReject(f"INT_FP16 predicate {u.op.name}")

def _int16_expr(u:UOp) -> UOp:
  """Represent a proven-bounded integer UOp in signed INT16 lanes."""
  if u.dtype.scalar() is dtypes.int16: return u
  if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.int, dtypes.weakint):
    value = int(u.arg)
    if not -32768 <= value <= 32767: raise _RKGenericReject
    return UOp.const(value, dtypes.int16)
  if u.dtype.scalar() not in (dtypes.int, dtypes.weakint): raise _RKGenericReject
  if u.op in (Ops.LOAD, Ops.RANGE, Ops.SPECIAL): return u.cast(dtypes.int16)
  if u.op is Ops.CAST and len(u.src) == 1:
    if u.src[0].dtype.scalar() is dtypes.bool: return _int16_bool(u.src[0]).cast(dtypes.int16)
    if u.src[0].dtype.scalar() in (dtypes.int, dtypes.weakint, dtypes.int16): return _int16_expr(u.src[0])
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2:
    return UOp(u.op, dtypes.int16, src=tuple(_int16_expr(src) for src in u.src), arg=u.arg)
  if u.op is Ops.WHERE and len(u.src) == 3:
    return UOp(Ops.WHERE, dtypes.int16, src=(_int16_bool(u.src[0]), _int16_expr(u.src[1]), _int16_expr(u.src[2])), arg=u.arg)
  raise _RKGenericReject(f"INT16 recipe {u.op.name}")

def _int16_bool(u:UOp) -> UOp:
  if u.op is Ops.CONST and u.dtype is dtypes.bool: return u
  if u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() in (dtypes.half, dtypes.float) for src in u.src): return u
  if u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() in (dtypes.int, dtypes.weakint) for src in u.src):
    return UOp(u.op, dtypes.bool, src=tuple(_int16_expr(src) for src in u.src), arg=u.arg)
  if u.op in (Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype is dtypes.bool for src in u.src):
    return UOp(u.op, dtypes.bool, src=tuple(_int16_bool(src) for src in u.src), arg=u.arg)
  raise _RKGenericReject(f"INT16 predicate {u.op.name}")

def _bounded_int32_fp16_root(root:UOp) -> UOp|None:
  """Narrow one canonically bounded INT32 value program into exact composable FP16 integer recipes."""
  loads = tuple({u.key:u for u in root.toposort() if u.op is Ops.LOAD}.values())
  if root.op is not Ops.WHERE or root.dtype.scalar() is not dtypes.int or len(loads) != 1 or \
     loads[0].dtype.scalar() is not dtypes.int or root.src[2].op is not Ops.CONST: return None
  load = loads[0]
  if len(load.src) != 1 or load.src[0].op is not Ops.INDEX or (param:=_root_param(load.src[0])) is None or \
     param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: return None
  gates = _flatten_binary(root.src[0], Ops.AND)
  limits = {int(u.src[1].arg) for u in gates if u.op is Ops.CMPLT and u.src[0].key == load.key and
            u.src[1].op is Ops.CONST and 0 < int(u.src[1].arg) <= _FP16_EXACT_INTEGER}
  nonnegative = any(u.op is Ops.CMPNE and any(marker.op is Ops.CONST and marker.dtype is dtypes.bool and bool(marker.arg) and
    comparison.op is Ops.CMPLT and comparison.src[0].key == load.key and comparison.src[1].op is Ops.CONST and
    int(comparison.src[1].arg) == 0 for comparison,marker in (u.src, u.src[::-1])) for u in gates)
  if len(limits) != 1 or not nonnegative: return None
  bounded = UOp.special(next(iter(limits)), "rockchip_bound", dtypes.int)
  proof_nodes = (u for arm in (root.src[1].substitute({load:bounded}), root.src[2]) for u in arm.toposort()
                 if u.dtype.scalar() in (dtypes.int, dtypes.weakint))
  if any((bound:=_exact_int_range(u)) is None or not -_FP16_EXACT_INTEGER <= bound[0] <= bound[1] <= _FP16_EXACT_INTEGER
         for u in proof_nodes): return None
  try: value = UOp(Ops.WHERE, dtypes.half, src=(root.src[0], _int_fp16_expr(root.src[1]), _int_fp16_expr(root.src[2])))
  except _RKGenericReject: return None
  return value.cast(dtypes.int)

def _native_int16_comparison(root:UOp) -> UOp|None:
  """Express one signed INT16 comparison as a saturating integer ALU mask."""
  if root.op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) or any(src.dtype.scalar() is not dtypes.int16 for src in root.src): return None
  lhs, rhs = root.src
  delta = rhs.alu(Ops.SUB, lhs) if root.op is Ops.CMPLT else lhs.alu(Ops.SUB, rhs)
  magnitude = delta.alu(Ops.MAX, UOp.const(0, dtypes.int16)) if root.op is Ops.CMPLT else \
              UOp(Ops.MAX, dtypes.int16, src=(delta, delta), arg=_NATIVE_ABS)
  mask = UOp(Ops.MAX, dtypes.int16, src=(magnitude, UOp.const(1, dtypes.int16)), arg=_NATIVE_MIN)
  return UOp.const(1, dtypes.int16).alu(Ops.SUB, mask) if root.op is Ops.CMPEQ else mask

class _RKGenericReject(Exception): pass

def _runtime_index(u:UOp) -> tuple[UOp, UOp, UOp, int]|None:
  """Return the index LOAD, its parameter, lane-address expression, and raw index width."""
  u = _strip_cast(u)
  if (u.op is not Ops.LOAD or len(u.src) != 1 or u.src[0].op is not Ops.INDEX or
      (param:=_root_param(u.src[0])) is None or param.src[0].op is not Ops.CONST): return None
  dtype = param.dtype.scalar()
  if dtype not in (dtypes.int, dtypes.int16): return None
  return u, param, u.src[0].src[1], dtype.itemsize

def _has_runtime_address(root:UOp) -> bool:
  """True when a value LOAD obtains its address or gate from another runtime LOAD."""
  for load in root.toposort():
    if load.op is not Ops.LOAD or not load.src or load.src[0].op is not Ops.INDEX: continue
    address_nodes = load.src[0].src[1].toposort()
    gate_nodes = load.src[2].toposort() if len(load.src) > 2 else ()
    if any(_runtime_index(node) is not None for node in (*address_nodes, *gate_nodes)): return True
  return False

def _runtime_affine_index(u:UOp, out_index:UOp, count:int) -> tuple[UOp, UOp, int, int, int, int, int, int]|None:
  """Resolve `static_base + runtime_index * scale + (lane % repeat) * inner` without reading the runtime index."""
  loads = [node for node in u.toposort() if _runtime_index(node) is not None]
  if len(loads) != 1 or (info:=_runtime_index(loads[0])) is None: return None
  load, param, lane_index, itemsize = info
  try:
    lane_offsets = _static_int_vector(out_index, lane_index, count)
    zero = _static_int_vector(out_index, u.substitute({load:load.const_like(0)}), count)
    one = _static_int_vector(out_index, u.substitute({load:load.const_like(1)}), count)
  except RuntimeError: return None
  scales = tuple(a-b for a,b in zip(one, zero))
  if len(set(scales)) != 1: return None
  def packed(repeat:int, lane_offset:int, base:int, stride:int) -> tuple[UOp, UOp, int, int, int, int, int, int]|None:
    groups = count if repeat == 1 else count//repeat
    lanes = tuple(lane_offset + (lane if repeat == 1 else lane//repeat) for lane in range(count))
    bases = tuple(base + (lane if repeat == 1 else lane%repeat)*stride for lane in range(count))
    if (repeat < 1 or count % repeat or groups <= 0 or lane_offsets != lanes or
        not 0 <= lane_offset <= int(param.src[0].arg)-groups or zero != bases): return None
    return load, param, itemsize, lane_offset, base, scales[0], stride, repeat
  if (identity:=packed(1, lane_offsets[0], zero[0], (zero[1]-zero[0]) if count > 1 else 0)) is not None: return identity
  repeat = next((i for i in range(1, count) if lane_offsets[i] != lane_offsets[0]), count)
  return packed(repeat, lane_offsets[0], zero[0], (zero[1]-zero[0]) if repeat > 1 else 0)

def _half_storage(u:UOp) -> UOp|None:
  return u.src[0] if u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() is dtypes.half and \
    u.src[0].dtype.scalar() is dtypes.float else None

def _fp32_expr_to_half(u:UOp) -> UOp:
  """Represent a float ADD/MUL expression with a three-half expansion at its FP16 storage boundary."""
  if u.dtype.scalar() is dtypes.half: return u
  if u.dtype.scalar() is not dtypes.float: raise _RKGenericReject
  src_dt = u.src[0].dtype.scalar() if u.op is Ops.CAST and len(u.src) == 1 else None
  if src_dt is dtypes.half: return u.src[0]
  if src_dt in (dtypes.int, dtypes.int16, dtypes.bool): return u.src[0].cast(dtypes.half)
  if u.op is Ops.LOAD: return u.cast(dtypes.half)
  if u.op is Ops.CONST: return UOp.const(float(u.arg), dtypes.half)
  if _is_static_expr(u): return u.cast(dtypes.half)
  if u.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN, Ops.NEG) and len(u.src) == 1:
    return UOp(u.op, dtypes.half, src=(_fp32_expr_to_half(u.src[0]),), arg=u.arg)
  if (u.op in (Ops.MUL, Ops.SUB, Ops.MAX) and len(u.src) == 2) or (u.op is Ops.ADD and any(
      node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half, dtypes.float) and not math.isfinite(float(node.arg))
      for node in u.toposort())):
    return UOp(u.op, dtypes.half, src=tuple(_fp32_expr_to_half(src) for src in u.src), arg=u.arg)
  if u.op is Ops.WHERE and len(u.src) == 3:
    return UOp(Ops.WHERE, dtypes.half, src=(u.src[0], _fp32_expr_to_half(u.src[1]), _fp32_expr_to_half(u.src[2])), arg=u.arg)
  if u.op is Ops.TRUNC and len(u.src) == 1:
    return _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(_fp32_expr_to_half(u.src[0]),)))
  if u.op is Ops.ADD: return _precise_mul_sum(_fp32_add_terms(u))
  raise _RKGenericReject(f"FP32 storage {u.op.name}")

def _nested_fp32_storage_cast(x:UOp) -> UOp|None:
  try: return _fp32_expr_to_half(x)
  except _RKGenericReject: return None

_pm_half_storage_algebra = PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.float, name="x"),)), _nested_fp32_storage_cast),
  (UPat(Ops.FDIV, dtypes.half, src=(UPat.var("x"), UPat.var("y"))),
   lambda x,y:x.alu(Ops.MUL, UOp(Ops.RECIPROCAL, dtypes.half, src=(y,)))),
])

def _canonical_half_storage(source:UOp) -> UOp:
  """Commit one FP32 storage expression, then reuse Tinygrad's ordinary algebra on its now-identical half values."""
  converted = _fp32_expr_to_half(source)
  if len(source.toposort()) > 64: return converted
  simplified = graph_rewrite(converted, _pm_half_storage_algebra+sym,
                             name="rockchip half storage algebra")
  return _tag_precise_adds(graph_rewrite(simplified, pm_commit_weak, name="rockchip commit storage constants"))

def _fp32_add_terms(u:UOp) -> list[UOp]:
  return [_fp32_expr_to_half(term) for term in _flatten_binary(u, Ops.ADD)]

def _fp32_add_has_product_terms(u:UOp) -> bool:
  """Whether a floating ADD tree contains a direct floating or cast-half product term."""
  return any((term.op is Ops.MUL and term.dtype.scalar() is dtypes.float) or
             (term.op is Ops.CAST and len(term.src) == 1 and term.src[0].op is Ops.MUL and
              term.src[0].dtype.scalar() is dtypes.half) for term in _flatten_binary(u, Ops.ADD))

def _fp32_ratio_to_half(u:UOp) -> UOp|None:
  """Divide two FP32 ADD boundaries while retaining their high/low half expansions through FDIV."""
  if u.op is not Ops.FDIV or u.dtype.scalar() is not dtypes.half or len(u.src) != 2: return None
  sums:list[UOp] = []
  for source in u.src:
    if (src:=_half_storage(source)) is None or src.op is not Ops.ADD: return None
    sums.append(src)
  numerator_high,numerator_low = _precise_sum_parts(_fp32_add_terms(sums[0]))
  denominator_high,denominator_low = _precise_sum_parts(_fp32_add_terms(sums[1]))
  numerator = numerator_high.alu(Ops.ADD, numerator_low)
  denominator = denominator_high.alu(Ops.ADD, denominator_low)
  quotient = numerator.alu(Ops.FDIV, denominator)
  neg_one = UOp.const(-1.0, dtypes.half)
  residual = _sub_half(numerator_high, quotient.alu(Ops.MUL, denominator_high), neg_one).alu(Ops.ADD,
    _sub_half(numerator_low, quotient.alu(Ops.MUL, denominator_low), neg_one))
  root = quotient.alu(Ops.ADD, residual.alu(Ops.FDIV, denominator))
  return _tag_precise_adds(root)

def _accurate_add_recipe(u:UOp) -> UOp:
  terms:list[UOp] = []
  for x in _flatten_binary(u, Ops.ADD, plain=True):
    if (src:=_half_storage(x)) is not None and src.op is Ops.ADD: terms.extend(_fp32_add_terms(src))
    else: terms.append(x)
  if sum(term.op is Ops.MUL and term.arg is None for term in terms) < 2: raise _RKGenericReject
  if any(node.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN) for node in UOp(Ops.SINK, src=tuple(terms)).toposort()):
    raise _RKGenericReject
  return _precise_mul_sum(terms)

class RKContext:
  """Typed physical lowering context. UOps remain the only semantic IR."""
  def __init__(self, output:RKOutput, nodes:dict[UOp, None], *, accurate_adds:bool=True):
    _, self.out_param, self.count, self.out_index, self.root = output
    self.out = RKArg(RKBufferKind.ARG, self.out_param.arg.slot)
    self.values:dict[UOp, RKValue] = {}
    self.scratch:list[RKScratch] = []
    self.scratch_specs = RKScratch(_scratch_bytes(self.count)), RKScratch(self.count*4)
    self.constants:dict[bytes, int] = {}
    self.materialized_slots:dict[tuple, RKValue] = {}
    self.raw_components:dict[RKArg, tuple[RKValue, ...]] = {}
    self.int32_divmod:dict[tuple[UOp, UOp], tuple[tuple[RKArg, ...], tuple[RKArg, ...], RKArg, RKArg]] = {}
    self.int16_masks:dict[RKArg, int] = {}
    self.fp16_components:dict[RKArg, tuple[RKValue, RKArg, RKArg]] = {}
    self.fp16_ordered:dict[RKArg, tuple[RKArg, RKArg]] = {}
    self.gathers:list[RKGather] = []
    self.host_gathers:list[RKHostAddress] = []
    self.mid_gathers:list[RKGather] = []
    self.ew_ops:list[RKEWOp] = []
    self.mask_program = any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in nodes)
    int_range = _exact_int_range(self.root) if self.root.dtype.scalar() is dtypes.int else None
    packed_bool_load = any(node.op is Ops.LOAD and node.dtype.scalar() is dtypes.bool and _root_param(node.src[0]) is not None for node in nodes)
    native_bool = any(node.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in node.src) for node in nodes)
    embedded_half_int = any(node.op is Ops.CAST and node.dtype.scalar() is dtypes.int and len(node.src) == 1 and
                            node.src[0].dtype.scalar() in (dtypes.half, dtypes.bool) for node in nodes)
    dynamic_int_load = any(node.op is Ops.LOAD and node.dtype.scalar() in (dtypes.int, dtypes.uint) and node.src and
                           _root_param(node.src[0]) is not None for node in nodes)
    self.int_layout = (RKLayout.INT_FP16 if self.root.op is Ops.CAST and self.root.dtype.scalar() is dtypes.int and embedded_half_int else
                       RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int and dynamic_int_load else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and (packed_bool_load or native_bool) and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT_FP16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -2048 <= int_range[0] <= int_range[1] <= 2048 else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int else
                       RKLayout.INT32 if dynamic_int_load else
                       RKLayout.INT_FP16 if embedded_half_int else None)
    self.accurate_adds = accurate_adds
    self.nodes = nodes
    self.static_evaluator = RKStaticIndexEvaluator(self.out_index, self.count)
    self.static_nodes:dict[UOp, bool] = {}

  def _is_static(self, u:UOp) -> bool:
    return u.topovisit(lambda node:node.op in _STATIC_OPS and not (isinstance(node.arg, str) and node.arg.startswith("rockchip_")) and
      all(self.static_nodes[src] for src in node.src), self.static_nodes)

  def _scratch(self, dtype:DType, layout:RKLayout, size:int|None=None) -> RKValue:
    slot = len(self.scratch)
    self.scratch.append(self.scratch_specs[layout is RKLayout.INT32] if size is None else RKScratch(size))
    return RKValue(RKArg(RKBufferKind.SCRATCH, slot), dtype, self.count, layout)

  def _int16_arg(self) -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg

  def _dst(self, u:UOp, dtype:DType, layout:RKLayout) -> RKValue:
    if (u is self.root and self.out_param.dtype.scalar() is dtype and
        (dtype is dtypes.half and layout is RKLayout.FP16 or dtype is dtypes.int16 and layout is RKLayout.INT16 or
         dtype is dtypes.int and layout is RKLayout.INT32)):
      return RKValue(self.out, dtype, self.count, layout)
    return self._scratch(dtype, layout)

  def _materialized_slot(self, key:tuple, dtype:DType, layout:RKLayout, plan:RKGather, size:int|None=None) -> RKValue:
    if key not in self.materialized_slots:
      value = self._scratch(dtype, layout, size)
      self.gathers.append(replace(plan, dst_index=value.arg.index))
      self.materialized_slots[key] = value
    return RKValue(self.materialized_slots[key].arg, dtype, self.count, layout)

  def _static_slot(self, dtype:DType, layout:RKLayout, vector:tuple[int, ...]) -> RKValue:
    return self._materialized_slot(("static", layout, vector), dtype, layout,
      RKGather(0, 0, self.count, values=vector, itemsize=4 if layout is RKLayout.INT32 else 2))

  def _gather_slot(self, dtype:DType, layout:RKLayout, plan:RKGather, size:int) -> RKValue:
    return self._materialized_slot(("gather", layout, _gather_cache_key((plan,))), dtype, layout, plan, size)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> RKValue:
    dtype = dtype_hint or u.dtype.scalar()
    if dtype is dtypes.uint or dtype is dtypes.int and self.int_layout is RKLayout.INT32:
      return self._static_slot(dtype, RKLayout.INT32, (int(u.arg) & 0xffffffff,) * self.count)
    if dtype in (dtypes.half, dtypes.float): bits, layout = struct.pack("<e", float(u.arg)), RKLayout.FP16
    elif dtype in (dtypes.int16, dtypes.uchar): bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), RKLayout.INT16
    elif dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16:
      bits, layout = struct.pack("<e", float(u.arg)), self.int_layout
    elif dtype is dtypes.int and self.int_layout is RKLayout.INT16:
      bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), self.int_layout
    elif dtype is dtypes.bool: bits, layout = struct.pack("<e", float(bool(u.arg))), RKLayout.BOOL_MASK
    else: raise _RKGenericReject(f"constant {dtype}")
    if bits in self.constants:
      slot = self.constants[bits]
      return RKValue(RKArg(RKBufferKind.SCRATCH, slot), dtype, self.count, layout)
    value = self._scratch(dtype, layout)
    self.constants[bits] = value.arg.index
    return value

  def _operand(self, u:UOp, dtype:DType) -> RKValue:
    return self._constant(u, dtype) if u.op is Ops.CONST and \
      (u.dtype.scalar() in dtypes.weaks or dtype is dtypes.half and u.dtype.scalar() is dtypes.float) else self.lower(u)

  def _static(self, u:UOp, bool_layout:RKLayout=RKLayout.BOOL_MASK) -> RKValue:
    dtype = u.dtype.scalar()
    if not _index_ranges(u):
      scalar = typing_cast(int|float|bool, _eval_expr(u, {}, {}))
      if dtype is dtypes.bool and bool_layout is RKLayout.BOOL_INT16:
        value = self._constant(UOp.const(int(bool(scalar)), dtypes.int16))
        return RKValue(value.arg, dtype, self.count, bool_layout)
      return self._constant(UOp.const(scalar, dtype))
    if dtype is dtypes.half: vector, layout = self.static_evaluator.values(u, _fp16_bits), RKLayout.FP16
    elif dtype is dtypes.int16: vector, layout = self.static_evaluator.values(u, _int16_bits), RKLayout.INT16
    elif dtype in (dtypes.int, dtypes.uint):
      values = self.static_evaluator.values(u, int)
      if dtype is dtypes.uint: vector, layout = tuple(value & 0xffffffff for value in values), RKLayout.INT32
      elif self.int_layout is RKLayout.INT_FP16 and all(-2048 <= value <= 2048 for value in values):
        vector, layout = tuple(_fp16_bits(float(value)) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT16 and all(-32768 <= value <= 32767 for value in values):
        vector, layout = tuple(_int16_bits(value) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT32:
        vector, layout = tuple(value & 0xffffffff for value in values), self.int_layout
      else: raise _RKGenericReject
    elif dtype is dtypes.bool:
      if bool_layout is RKLayout.BOOL_INT16: vector, layout = self.static_evaluator.values(u, lambda x:int(bool(x))), bool_layout
      else: vector, layout = self.static_evaluator.values(u, lambda x:_fp16_bits(float(bool(x)))), RKLayout.BOOL_MASK
    else: raise _RKGenericReject
    return self._static_slot(dtype, layout, vector)

  def _load(self, u:UOp, fill_override:int|None=None) -> RKValue:
    dtype = u.dtype.scalar()
    if dtype not in (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) or not u.src or u.src[0].op is not Ops.INDEX or \
       (param:=_root_param(u.src[0])) is None or param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST:
      raise _RKGenericReject
    index, gate, default = u.src[0].src[1], u.src[2] if len(u.src) > 2 else None, u.src[1] if len(u.src) > 1 else None
    index_nodes, gate_nodes = index.toposort(), () if gate is None else gate.toposort()
    runtime_address = any(x.op is Ops.LOAD for x in (*index_nodes, *gate_nodes))
    if default is not None and default.op is not Ops.CONST:
      if dtype not in (dtypes.half, dtypes.int16, dtypes.int) or gate is None or runtime_address:
        raise _RKGenericReject
      expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
      itemsize = 4 if expected is RKLayout.INT32 else 2
      schedule = len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)
      fallback = self.lower(default)
      if fallback.layout is not expected or fallback.count != self.count or \
         schedule != (len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)):
        raise _RKGenericReject
      plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      value = self._scratch(dtype, expected, self.count*itemsize)
      self.gathers.append(RKGather(fallback.arg.index, value.arg.index, self.count,
        base=fallback.arg.addend//itemsize, axes=((1, self.count, 1),), src_kind=fallback.arg.kind, itemsize=itemsize))
      self.gathers.append(replace(plan, dst_index=value.arg.index, partial=True, itemsize=itemsize))
      return value
    if dtype is dtypes.float:
      if runtime_address: raise _RKGenericReject
      fill_bits = struct.unpack("<I", struct.pack("<f", float(0 if default is None else default.arg)))[0]
      plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count, fill_bits)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      groups = tuple(range(0, self.count, _EW_ELEMS_32BIT))
      raw_key = ("fp32_raw", _gather_cache_key((replace(plan, itemsize=4),)))
      raw = self._materialized_slot(raw_key, dtype, RKLayout.FP16, replace(plan, itemsize=4), len(groups)*16)
      aligned = self._scratch(dtypes.half, RKLayout.FP16, len(groups)*16)
      zero = self._scratch(dtype, RKLayout.FP16, 16)
      self.gathers.append(RKGather(0, zero.arg.index, _EW_ELEMS_32BIT, values=(0,)*_EW_ELEMS_32BIT, itemsize=4))
      for group,start in enumerate(groups):
        lanes = min(_EW_ELEMS_32BIT, self.count-start)
        source = replace(raw.arg, addend=group*16)
        self.ew_ops.append(RKEWOp(replace(aligned.arg, addend=group*16), source, zero.arg, lanes,
                                  _EW_CFG[Ops.ADD] | _EW_STAGE_FP32_IN, stateful=True))
      compact = self._scratch(dtypes.half, RKLayout.FP16, self.count*2)
      self.mid_gathers.append(RKGather(aligned.arg.index, compact.arg.index, self.count,
        offsets=tuple((lane//_EW_ELEMS_32BIT)*8+lane%_EW_ELEMS_32BIT for lane in range(self.count)),
        src_kind=RKBufferKind.SCRATCH, after=len(self.ew_ops)))
      return RKValue(compact.arg, dtype, self.count, RKLayout.FP16)
    if dtype is dtypes.bool:
      if runtime_address: raise _RKGenericReject
      offsets = RKStaticIndexEvaluator(self.out_index, self.count).offsets(index, gate)
      plan = RKGather(param.arg.slot, 0, self.count, offsets=offsets, fill_bits=int(bool(default.arg)) if default is not None else 0,
                      dst_stride=2, itemsize=1)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      return self._gather_slot(dtype, RKLayout.BOOL_INT16, plan, self.count*2)
    layout = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
    itemsize = 4 if layout is RKLayout.INT32 else 2
    fill_bits = fill_override if fill_override is not None else _fp16_bits(0 if default is None else default.arg) if dtype is dtypes.half else \
      _int16_bits(0 if default is None else default.arg) if dtype is dtypes.int16 else int(0 if default is None else default.arg) & 0xffffffff
    if runtime_address:
      if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1": raise _RKGenericReject
      runtime_index = _runtime_affine_index(index, self.out_index, self.count)
      runtime_loads = {node.key:node for node in (*index_nodes, *gate_nodes)
                       if _runtime_index(node) is not None}
      index_repeat = 1
      if runtime_index is not None:
        runtime_load, index_param, index_itemsize, index_offset, base, index_scale, lane_stride, index_repeat = runtime_index
      elif len(runtime_loads) == 1 and (runtime_info:=_runtime_index(next(iter(runtime_loads.values())))) is not None:
        runtime_load, index_param, index_lane, index_itemsize = runtime_info
        try: index_lanes = _static_int_vector(self.out_index, index_lane, self.count)
        except RuntimeError: raise _RKGenericReject from None
        index_offset = index_lanes[0]
        if index_lanes != tuple(index_offset+lane for lane in range(self.count)) or \
           not 0 <= index_offset <= int(index_param.src[0].arg)-self.count: raise _RKGenericReject
        base = index_scale = lane_stride = 0
      else: raise _RKGenericReject
      index_limit = int(param.src[0].arg)
      if gate is not None:
        limits = [int(node.src[1].arg) for node in gate_nodes if node.op is Ops.CMPLT and
                  node.src[0].key == runtime_load.key and node.src[1].op is Ops.CONST and int(node.src[1].arg) > 0]
        if len(set(limits)) != 1 or not _bounded_index_gate(gate, runtime_load, limits[0]) or \
           {node.key for node in gate_nodes if node.op is Ops.LOAD} != {runtime_load.key}: raise _RKGenericReject
        index_limit = limits[0]
      source = RKArg(RKBufferKind.ARG, param.arg.slot)
      source_count = int(param.src[0].arg)
      if runtime_index is None:
        if gate is None or index_limit <= 0 or self.count*index_limit > _MAX_STATIC_RANGE_ENVS: raise _RKGenericReject
        try:
          candidates = tuple(_static_int_vector(self.out_index, index.substitute({runtime_load:runtime_load.const_like(candidate)}), self.count)
                             for candidate in range(index_limit))
        except RuntimeError: raise _RKGenericReject from None
        offsets = tuple(candidates[candidate][lane] for lane in range(self.count) for candidate in range(index_limit))
        plan = RKGather(param.arg.slot, 0, len(offsets), offsets=offsets, itemsize=itemsize)
        _validate_gather_bounds(plan, source_count)
        source, source_count = self._gather_slot(dtype, layout, plan, len(offsets)*itemsize).arg, len(offsets)
        base, index_scale, lane_stride, index_repeat = 0, 1, index_limit, 1
      value = self._scratch(dtype, layout, self.count*itemsize)
      self.host_gathers.append(RKHostAddress(source,
        RKArg(RKBufferKind.ARG, index_param.arg.slot, index_offset*index_itemsize), value.arg,
        self.count, source_count, self.count,
        itemsize=itemsize, index_itemsize=index_itemsize, fill_bits=fill_bits, index_limit=index_limit,
        base=base, index_scale=index_scale, lane_stride=lane_stride, index_repeat=index_repeat))
      return value
    if gate is None and index.key == self.out_index.key and int(param.src[0].arg) == self.count:
      return RKValue(RKArg(RKBufferKind.ARG, param.arg.slot), dtype, self.count, layout)
    plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count, fill_bits)
    _validate_gather_bounds(plan, int(param.src[0].arg))
    return self._gather_slot(dtype, layout, replace(plan, itemsize=itemsize), self.count*itemsize)

  def _emit(self, dst:RKValue, lhs:RKValue, rhs:RKValue, cfg:int, *, compare:bool=False) -> RKValue:
    integer16, integer32 = dst.layout in (RKLayout.INT16, RKLayout.BOOL_INT16), dst.layout is RKLayout.INT32
    if integer16 and (lhs.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16) or
                      rhs.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16)): raise _RKGenericReject
    if integer32 and (lhs.layout is not RKLayout.INT32 or rhs.layout is not RKLayout.INT32): raise _RKGenericReject
    if not integer16 and not integer32 and lhs.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16) or \
       not integer16 and not integer32 and rhs.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16): raise _RKGenericReject
    barrier = not integer16 and not integer32 and cfg in (_EW_CFG_FLOOR, _EW_CFG[Ops.FDIV])
    self.ew_ops.append(RKEWOp(dst.arg, lhs.arg, rhs.arg, self.count, cfg, submit_barrier=barrier,
      compare=compare, stateful=integer32 or not integer16 and (self.mask_program and not compare or barrier),
      int16_output=integer16, int16_input=integer16, int32_output=integer32, int32_input=integer32))
    self.mask_program |= compare
    return dst

  def _masked_where(self, u:UOp, dtype:DType, layout:RKLayout, selector:RKValue, yes:RKValue, no:RKValue, one:RKValue) -> RKValue:
    selected_yes, inverse, selected_no = (self._scratch(one.dtype, one.layout) for _ in range(3))
    self._emit(selected_yes, selector, yes, _EW_CFG[Ops.MUL])
    self._emit(inverse, one, selector, _EW_CFG[Ops.SUB])
    self._emit(selected_no, inverse, no, _EW_CFG[Ops.MUL])
    return self._emit(self._dst(u, dtype, layout), selected_yes, selected_no, _EW_CFG[Ops.ADD])

  def _native_min(self, u:UOp, lhs:RKValue, rhs:RKValue) -> RKValue:
    zero = self.lower(UOp.const(0.0, dtypes.half))
    neg_lhs, neg_rhs = (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(2))
    self._emit(neg_lhs, zero, lhs, _EW_CFG[Ops.SUB])
    self._emit(neg_rhs, zero, rhs, _EW_CFG[Ops.SUB])
    self._emit(neg_lhs, neg_lhs, neg_rhs, _EW_CFG[Ops.MAX])
    dst = self._dst(u, dtypes.half, RKLayout.FP16) if u is self.root else neg_lhs
    return self._emit(dst, zero, neg_lhs, _EW_CFG[Ops.SUB])

  def _raw_parts(self, value:RKValue, *, cache:bool=True, copy_wide:bool=True) -> tuple[RKValue, ...]:
    if value.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT_FP16, RKLayout.INT32): raise _RKGenericReject
    if cache and value.arg in self.raw_components: return self.raw_components[value.arg]
    itemsize, source = (4 if value.layout is RKLayout.INT32 else 2), value
    if itemsize == 4 and copy_wide:
      source = self._scratch(dtypes.int, RKLayout.INT32)
      self._emit(source, value, value, _EW_CFG[Ops.MAX])
    parts = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(itemsize))
    split_after = len(self.ew_ops)
    for byte,part in enumerate(parts):
      self.mid_gathers.append(RKGather(source.arg.index, part.arg.index, self.count,
        base=source.arg.addend+byte, axes=((1, self.count, itemsize),), dst_stride=2,
        src_kind=source.arg.kind, itemsize=1, after=split_after))
    if cache: self.raw_components[value.arg] = parts
    return parts

  def _pack_raw(self, u:UOp, components:Iterable[RKValue|RKArg], layout:RKLayout, *,
                mask:int|None=None, dst:RKValue|None=None, cache:bool=True) -> RKValue:
    parts = tuple(part if isinstance(part, RKValue) else RKValue(part, dtypes.int16, self.count, RKLayout.INT16)
                  for part in components)
    itemsize = 4 if layout is RKLayout.INT32 else 2
    if len(parts) != itemsize: raise _RKGenericReject
    result = self._dst(u, u.dtype.scalar(), layout) if dst is None else dst
    pack_after = len(self.ew_ops)
    for byte,source in enumerate(parts):
      self.mid_gathers.append(RKGather(source.arg.index, result.arg.index, self.count,
        base=source.arg.addend, axes=((1, self.count, 2),), dst_stride=itemsize, dst_addend=byte,
        dst_kind=result.arg.kind, src_kind=source.arg.kind, itemsize=1, after=pack_after))
    if cache: self.raw_components[result.arg] = parts
    if mask is not None: self.int16_masks[result.arg] = mask
    return result

  def _max_operand(self, src:UOp, dtype:DType) -> RKValue:
    if src.op is Ops.CONST and math.isinf(float(src.arg)) and float(src.arg) < 0:
      return self._constant(UOp.const(-65504.0, dtypes.half))
    if (src.op is Ops.LOAD and len(src.src) > 2 and src.src[1].op is Ops.CONST and
        math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0 and self._is_static(src.src[2])):
      return self._load(src, _fp16_bits(-65504.0))
    return self._operand(src, dtype)

  def _alu(self, u:UOp) -> RKValue:
    if u.op is Ops.RECIPROCAL:
      src = self.lower(u.src[0]); one = self.lower(UOp.const(1.0, dtypes.half))
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), one, src, _EW_CFG[Ops.FDIV])
    if u.op is Ops.NEG:
      src = self.lower(u.src[0])
      dst = self._dst(u, u.dtype.scalar(), src.layout)
      return self._emit(dst, src, src, _EW_CFG_NEG)
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.ADD and u.arg is None and (recipe:=_fold_relu_cap(u)) is not None:
      return self.lower(recipe)
    if u.op is Ops.FDIV and (recipe:=_preserve_infinite_division_sign(u)) is not None:
      return self.lower(recipe)
    dtype = u.dtype.scalar()
    int_range = _exact_int_range(u) if dtype is dtypes.int else None
    bounded = (self.int_layout is RKLayout.INT_FP16 and int_range is not None and -2048 <= int_range[0] <= int_range[1] <= 2048 or
               self.int_layout is RKLayout.INT16 and int_range is not None and -32768 <= int_range[0] <= int_range[1] <= 32767 or
               self.int_layout is RKLayout.INT32)
    expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if bounded else None
    if expected is None: raise _RKGenericReject(f"alu {u.op.name} {dtype} bounds={int_range}")
    operand = self._max_operand if u.op is Ops.MAX and dtype is dtypes.half else self._operand
    lhs, rhs = operand(u.src[0], dtype), operand(u.src[1], dtype)
    compatible = (RKLayout.FP16, RKLayout.BOOL_MASK) if expected is RKLayout.FP16 else (expected,)
    if lhs.layout not in compatible or rhs.layout not in compatible: raise _RKGenericReject
    if dtype is dtypes.half and (u.arg is None or u.op is Ops.ADD and u.arg == _NATIVE_PRECISE_ADD):
      return self._emit(self._dst(u, dtype, expected), lhs, rhs, _EW_CFG[u.op])
    if u.op is Ops.SUB and u.arg == _NATIVE_SIGN:
      if expected is not RKLayout.FP16 or lhs.layout is not RKLayout.FP16: raise _RKGenericReject
      zero = self._constant(UOp.const(0.0, dtypes.half))
      negative = self._scratch(dtypes.half, RKLayout.FP16)
      negative_mask, positive_mask = (self._scratch(dtypes.bool, RKLayout.BOOL_MASK) for _ in range(2))
      self._emit(negative, zero, lhs, _EW_CFG[Ops.SUB])
      self._emit(negative_mask, negative, negative, _EW_CFG[Ops.MAX], compare=True)
      self._emit(positive_mask, lhs, lhs, _EW_CFG[Ops.MAX], compare=True)
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), positive_mask, negative_mask, _EW_CFG[Ops.SUB])
    if u.op is Ops.MAX and u.arg in (_NATIVE_MIN, _NATIVE_RAW_MIN):
      if expected is RKLayout.FP16 and u.arg == _NATIVE_MIN: return self._native_min(u, lhs, rhs)
      return self._emit(self._dst(u, dtype, RKLayout.INT16 if u.arg == _NATIVE_MIN else expected), lhs, rhs, _EW_CFG_MIN)
    cfg = {(Ops.MAX, _NATIVE_ABS):_EW_CFG_ABS, (Ops.MAX, _NATIVE_FLOOR):_EW_CFG_FLOOR, (Ops.MAX, _NATIVE_CEIL):_EW_CFG_CEIL,
           (Ops.MUL, _NATIVE_LEAKY_RELU):_EW_CFG_LEAKY_RELU}.get((u.op, u.arg), _EW_CFG[u.op])
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    layout = RKLayout.BOOL_MASK if compare else expected
    out_dtype = dtypes.bool if compare else dtype
    dst = self._dst(u, out_dtype, layout)
    return self._emit(dst, lhs, rhs, cfg, compare=compare)

  def _coerce_bool(self, value:RKValue, layout:RKLayout) -> RKValue:
    if value.layout is layout: return value
    if value.layout is not RKLayout.BOOL_MASK or layout is not RKLayout.BOOL_INT16: raise _RKGenericReject
    converted = self._scratch(dtypes.bool, RKLayout.BOOL_INT16)
    self.ew_ops.append(RKEWOp(converted.arg, value.arg, value.arg, self.count, _EW_CFG[Ops.MAX], submit_barrier=True,
                             stateful=True, int16_output=True))
    return converted

  def _bool_binary(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.CMPNE:
      for expression,marker in (u.src, u.src[::-1]):
        if (marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and
            expression.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in expression.src)):
          less = self.lower(expression)
          if less.layout is not RKLayout.BOOL_INT16: raise _RKGenericReject
          operands = tuple(self._operand(src, dtypes.half) for src in expression.src)
          nan = tuple(self._fp16_component_values(value)[2] for value in operands)
          one = self._constant(UOp.const(1, dtypes.int16))
          inverse, either_nan, numeric = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(3))
          self._emit(inverse, one, less, _EW_CFG[Ops.SUB])
          self._emit(either_nan, RKValue(nan[0], dtypes.int16, self.count, RKLayout.INT16),
                     RKValue(nan[1], dtypes.int16, self.count, RKLayout.INT16), _EW_CFG[Ops.MAX])
          self._emit(numeric, one, either_nan, _EW_CFG[Ops.SUB])
          return self._emit(self._dst(u, dtypes.bool, RKLayout.BOOL_INT16), inverse, numeric, _EW_CFG[Ops.MUL])
    values = [self.lower(src) if not (src.op is Ops.CONST and src.dtype.scalar() is dtypes.bool) else None for src in u.src]
    preferred = (RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in values) else
                 RKLayout.BOOL_MASK)
    lhs, rhs = (self._static(src, preferred) if value is None else self._coerce_bool(value, preferred)
                for src,value in zip(u.src, values))
    dst = self._dst(u, dtypes.bool, preferred)
    if u.op is Ops.AND: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MUL])
    if u.op is Ops.OR: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MAX])
    data_dtype, data_layout = (dtypes.int16, RKLayout.INT16) if preferred is RKLayout.BOOL_INT16 else (dtypes.half, RKLayout.FP16)
    delta = self._scratch(data_dtype, data_layout)
    self._emit(delta, lhs, rhs, _EW_CFG[Ops.SUB])
    if u.op in (Ops.XOR, Ops.CMPNE): return self._emit(dst, delta, delta, _EW_CFG_ABS)
    if u.op is Ops.CMPEQ:
      raw_one = self._constant(UOp.const(1, dtypes.int16)) if preferred is RKLayout.BOOL_INT16 else self.lower(UOp.const(True, dtypes.bool))
      one = RKValue(raw_one.arg, dtypes.bool, self.count, preferred)
      unequal = self._scratch(dtypes.bool, preferred)
      self._emit(unequal, delta, delta, _EW_CFG_ABS)
      return self._emit(dst, one, unequal, _EW_CFG[Ops.SUB])
    raise _RKGenericReject

  def _integer_bitwise(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.AND:
      for marker,source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or (mask:=int(marker.arg)&0xffff) not in (0x7fff, 0x8000): continue
        value = self.lower(source)
        if value.layout is not RKLayout.INT16: raise _RKGenericReject
        low, high = self._raw_parts(value)
        zero, one, const127, const128 = (self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128))
        delta, positive, sign, sign_scale = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(4))
        self._emit(delta, high, const127, _EW_CFG[Ops.SUB])
        self._emit(positive, delta, zero, _EW_CFG[Ops.MAX])
        self._emit(sign, positive, one, _EW_CFG_MIN)
        self._emit(sign_scale, sign, const128, _EW_CFG[Ops.MUL])
        if mask == 0x7fff:
          masked_high = self._scratch(dtypes.int16, RKLayout.INT16)
          self._emit(masked_high, high, sign_scale, _EW_CFG[Ops.SUB])
          return self._pack_raw(u, (low, masked_high), RKLayout.INT16, mask=mask)
        return self._pack_raw(u, (zero, sign_scale), RKLayout.INT16, mask=mask)
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.OR:
      values = tuple(self.lower(source) for source in u.src)
      masks = tuple(self.int16_masks.get(value.arg) for value in values)
      if all(mask is not None for mask in masks) and typing_cast(int, masks[0]) & typing_cast(int, masks[1]) == 0:
        parts = tuple(self._raw_parts(value) for value in values)
        low, high = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(2))
        self._emit(low, parts[0][0], parts[1][0], _EW_CFG[Ops.ADD])
        self._emit(high, parts[0][1], parts[1][1], _EW_CFG[Ops.ADD])
        return self._pack_raw(u, (low, high), RKLayout.INT16,
                              mask=typing_cast(int, masks[0]) | typing_cast(int, masks[1]))
    if u.op is Ops.XOR:
      for marker, source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or int(marker.arg) != -1: continue
        dtype = u.dtype.scalar()
        layout = RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if dtype is dtypes.int else None
        if layout is None: raise _RKGenericReject
        rhs = self.lower(source)
        if rhs.layout is not layout: raise _RKGenericReject
        if layout is RKLayout.INT32:
          components = self._raw_parts(rhs)
          const255 = self._constant(UOp.const(255, dtypes.int16))
          inverted = tuple(self._emit(self._scratch(dtypes.int16, RKLayout.INT16), const255, component, _EW_CFG[Ops.SUB])
                           for component in components)
          return self._pack_raw(u, inverted, RKLayout.INT32, dst=self._scratch(dtypes.int, RKLayout.INT32))
        lhs = self._constant(UOp.const(-1, dtype))
        return self._emit(self._dst(u, dtype, layout), lhs, rhs, _EW_CFG[Ops.SUB])
    dtype = u.dtype.scalar()
    layout = RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32 if dtype is dtypes.int and self.int_layout is RKLayout.INT32 else None
    if layout is None or u.op not in (Ops.AND, Ops.OR, Ops.XOR): raise _RKGenericReject
    values = tuple(self.lower(source) for source in u.src)
    if any(value.layout is not layout for value in values): raise _RKGenericReject
    lanes = self.count*(4 if layout is RKLayout.INT32 else 2)
    if not 1 <= lanes <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16, _scratch_bytes(lanes)).arg
    raw:list[RKArg] = []
    for value in values:
      expanded = allocate(); raw.append(expanded)
      self.mid_gathers.append(RKGather(value.arg.index, expanded.index, lanes, base=value.arg.addend,
        axes=((1, lanes, 1),), dst_stride=2, src_kind=value.arg.kind, itemsize=1, after=len(self.ew_ops)))
    constants:dict[int, RKArg] = {}
    for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128):
      constants[number] = constant = allocate()
      self.gathers.append(RKGather(self.out_param.arg.slot, constant.index, lanes, values=(number,)*lanes))
    lhs_bits, rhs_bits = (_int16_byte_bits(self.ew_ops, allocate, constants, value, lanes) for value in raw)
    weighted:list[RKArg] = []
    for bit,(left,right) in enumerate(zip(lhs_bits, rhs_bits)):
      combined = allocate()
      if u.op is Ops.XOR:
        self.ew_ops.extend((RKEWOp(combined, left, right, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                            RKEWOp(combined, combined, combined, lanes, _EW_CFG_ABS, **_INT16_EW)))
      else:
        self.ew_ops.append(RKEWOp(combined, left, right, lanes,
          _EW_CFG_MIN if u.op is Ops.AND else _EW_CFG[Ops.MAX], **_INT16_EW))
      if bit: self.ew_ops.append(RKEWOp(combined, combined, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
      weighted.append(combined)
    combined = _reduce_rows(self.ew_ops, weighted, lanes, _EW_CFG[Ops.ADD])
    result = self._dst(u, dtype, layout)
    self.mid_gathers.append(RKGather(combined.index, result.arg.index, lanes, base=combined.addend,
      axes=((1, lanes, 2),), dst_stride=1, dst_kind=result.arg.kind,
      src_kind=RKBufferKind.SCRATCH, itemsize=1, after=len(self.ew_ops)))
    return result

  def _int32_shift(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or u.dtype.scalar() not in (dtypes.int, dtypes.uint) or \
       u.src[1].dtype.scalar() not in (dtypes.int, dtypes.uint) or self.int_layout is not RKLayout.INT32:
      raise _RKGenericReject
    value = self.lower(u.src[0])
    if value.layout is not RKLayout.INT32: raise _RKGenericReject
    vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(self.count, 32)
    pre_lanes = vector_lanes*5
    if self.count < 1 or matrix_lanes > _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    def arena(size:int, stride:int) -> Callable[[], RKArg]:
      base, row = self._scratch(dtypes.int16, RKLayout.INT16, size).arg, 0
      def allocate() -> RKArg:
        nonlocal row
        result = replace(base, addend=row*stride); row += 1
        return result
      return allocate
    pre_allocate, post_allocate = arena(51*pre_lanes*2, pre_lanes*2), arena(20*matrix_lanes*2, matrix_lanes*2)
    raw, value_parts = pre_allocate(), self._raw_parts(value)
    shift = None if u.src[1].op is Ops.CONST else self.lower(u.src[1])
    if shift is not None and shift.layout is not RKLayout.INT32: raise _RKGenericReject
    shift_part = None if shift is None else self._raw_parts(shift)[0]
    raw_after = len(self.ew_ops)
    packed = (*value_parts, *((shift_part,) if shift_part is not None else ()))
    self.mid_gathers.extend(RKGather(part.arg.index, raw.index, self.count, base=part.arg.addend//2,
      axes=((1, self.count, 1),), dst_addend=byte*vector_lanes, src_kind=part.arg.kind, after=raw_after)
      for byte,part in enumerate(packed))
    if shift_part is None:
      self.mid_gathers.append(RKGather(self.out_param.arg.slot, raw.index, self.count,
        values=(int(u.src[1].arg)&0xff,)*self.count, dst_addend=4*vector_lanes, after=raw_after))
    constants = {number:pre_allocate() for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)}
    for number,dst in constants.items():
      self.gathers.append(RKGather(self.out_param.arg.slot, dst.index, pre_lanes,
        values=(number,)*pre_lanes, dst_addend=dst.addend//2))
    planes = _int16_byte_bits(self.ew_ops, pre_allocate, constants, raw, pre_lanes)

    bits, masks, sign, zero, weights = post_allocate(), tuple(post_allocate() for _ in range(5)), \
                                       post_allocate(), post_allocate(), post_allocate()
    post_after = len(self.ew_ops)
    for absolute_bit in range(32):
      plane,byte = absolute_bit&7, absolute_bit>>3
      self.mid_gathers.append(RKGather(planes[plane].index, bits.index, self.count,
        base=planes[plane].addend//2+byte*vector_lanes, axes=((1, self.count, 1),),
        dst_addend=absolute_bit*vector_lanes, src_kind=RKBufferKind.SCRATCH, after=post_after))
    for bit,mask in enumerate(masks):
      self.mid_gathers.append(RKGather(planes[bit].index, mask.index, matrix_lanes,
        base=planes[bit].addend//2+4*vector_lanes, axes=((1, vector_lanes, 1),),
        dst_addend=mask.addend//2, src_kind=RKBufferKind.SCRATCH, after=post_after))
    self.mid_gathers.append(RKGather(planes[7].index, sign.index, matrix_lanes,
      base=planes[7].addend//2+3*vector_lanes, axes=((1, vector_lanes, 1),),
      dst_addend=sign.addend//2, src_kind=RKBufferKind.SCRATCH, after=post_after))
    self.mid_gathers.append(RKGather(self.out_param.arg.slot, weights.index, matrix_lanes,
      values=tuple(1 << (row&7) if lane < self.count else 0 for row in range(32) for lane in range(vector_lanes)),
      dst_addend=weights.addend//2, after=post_after))
    current = bits
    for bit,amount in enumerate((1, 2, 4, 8, 16)):
      temp,result = post_allocate(),post_allocate()
      normal_rows,normal_dst,shifted_src = (32-amount, amount, 0) if u.op is Ops.SHL else (32-amount, 0, amount)
      boundary_rows,boundary_dst = (amount, 0) if u.op is Ops.SHL else (amount, 32-amount)
      for rows,dst_row,src,fill in ((normal_rows, normal_dst, current, shifted_src),
                                    (boundary_rows, boundary_dst, sign if u.op is Ops.SHR and u.dtype.scalar() is dtypes.int else zero, 0)):
        addend = dst_row*vector_bytes
        dst,old,selected = (replace(arg, addend=arg.addend+addend) for arg in (temp, current, result))
        source = replace(src, addend=src.addend+(fill*vector_bytes if src is current else addend))
        mask = replace(masks[bit], addend=masks[bit].addend+addend)
        count = rows*vector_lanes
        self.ew_ops.extend((RKEWOp(dst, source, old, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                            RKEWOp(dst, dst, mask, count, _EW_CFG[Ops.MUL], **_INT16_EW),
                            RKEWOp(selected, old, dst, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
      current = result
    weighted = post_allocate()
    self.ew_ops.append(RKEWOp(weighted, current, weights, matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
    byte_results = tuple(_reduce_rows(self.ew_ops,
      [replace(weighted, addend=weighted.addend+(byte*8+bit)*vector_bytes) for bit in range(8)],
      vector_lanes, _EW_CFG[Ops.ADD]) for byte in range(4))
    return self._pack_raw(u, byte_results, RKLayout.INT32)

  def _compare(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src): return self._bool_binary(u)
    if u.op is Ops.CMPNE and (nonzero:=_fp16_nonzero_mask(u)) is not None:
      value = self.lower(nonzero)
      if value.layout is not RKLayout.BOOL_MASK: raise _RKGenericReject
      return RKValue(value.arg, dtypes.bool, self.count, value.layout)
    if u.op in (Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_equality(u)
    if u.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_less(u)
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      sources = tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src)
      bounds = tuple(_exact_int_range(src) for src in sources)
      if self.int_layout is not RKLayout.INT32 and all(
        bound is not None and -2048 <= bound[0] <= bound[1] <= 2048 for bound in bounds
      ):
        value = self.lower(UOp(u.op, dtypes.bool, src=tuple(_int_fp16_expr(src) for src in sources), arg=u.arg))
        if value.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
        return value
      return self._int32_compare(u.replace(src=sources))
    if all(src.dtype.scalar() is dtypes.int16 for src in u.src):
      if (int16_recipe:=_native_int16_comparison(u)) is None: raise _RKGenericReject
      value = self.lower(int16_recipe)
      if value.layout is not RKLayout.INT16: raise _RKGenericReject
      return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_INT16)
    if _half_backed(u.src):
      return self._compare(u.replace(src=tuple(src if src.dtype.scalar() is dtypes.half else src.cast(dtypes.half) for src in u.src)))
    raise _RKGenericReject

  def _i16(self, lhs:RKArg, rhs:RKArg, cfg:int) -> RKArg:
    dst = self._int16_arg()
    self.ew_ops.append(RKEWOp(dst, lhs, rhs, self.count, cfg, **_INT16_EW))
    return dst

  def _i16_equal(self, lhs:RKArg, rhs:RKArg, one:RKArg) -> RKArg:
    diff = self._i16(lhs, rhs, _EW_CFG[Ops.SUB])
    return self._i16(one, self._i16(self._i16(diff, diff, _EW_CFG_ABS), one, _EW_CFG_MIN), _EW_CFG[Ops.SUB])

  def _i16_const(self, value:int) -> RKArg: return self._constant(UOp.const(value, dtypes.int16)).arg

  def _i16_clamp_one(self, value:RKArg) -> RKArg:
    return self._i16(self._i16(value, self._i16_const(0), _EW_CFG[Ops.MAX]), self._i16_const(1), _EW_CFG_MIN)

  def _i16_positive_over(self, value:RKArg, threshold:int) -> RKArg:
    return self._i16_clamp_one(self._i16(value, self._i16_const(threshold), _EW_CFG[Ops.SUB]))

  def _i16_xor(self, lhs:RKArg, rhs:RKArg) -> RKArg:
    delta = self._i16(lhs, rhs, _EW_CFG[Ops.SUB])
    return self._i16(delta, delta, _EW_CFG_ABS)

  def _i16_twos_complement(self, raw:tuple[RKArg, ...], sign:RKArg) -> tuple[RKArg, ...]:
    carry, result = sign, []
    for byte in raw:
      doubled = self._i16(byte, byte, _EW_CFG[Ops.ADD])
      inverted = self._i16(self._i16(self._i16_const(255), doubled, _EW_CFG[Ops.SUB]), sign, _EW_CFG[Ops.MUL])
      total = self._i16(self._i16(byte, inverted, _EW_CFG[Ops.ADD]), carry, _EW_CFG[Ops.ADD])
      carry = self._i16_positive_over(total, 255)
      result.append(self._i16(total, self._i16(carry, self._i16_const(256), _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB]))
    return tuple(result)

  def _int32_divmod(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or self.int_layout is not RKLayout.INT32 or not 1 <= self.count <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    key = u.src
    if key not in self.int32_divmod:
      values = tuple(self._operand(src, dtypes.int) for src in key)
      if any(value.layout is not RKLayout.INT32 for value in values): raise _RKGenericReject
      raw = tuple(tuple(part.arg for part in self._raw_parts(value)) for value in values)
      constants = {value:self._i16_const(value) for value in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256)}
      signs = tuple(self._i16_positive_over(value[3], 127) for value in raw)
      numerator, denominator = (self._i16_twos_complement(value, sign) for value,sign in zip(raw, signs))
      denominator_nonzero = _reduce_rows(self.ew_ops, [self._i16_clamp_one(value) for value in denominator],
                                         self.count, _EW_CFG[Ops.MAX])
      numerator_bits = tuple(bit for byte in numerator for bit in
                             _int16_byte_bits(self.ew_ops, self._int16_arg, constants, byte, self.count))
      remainder, quotient = [constants[0]]*4, [constants[0]]*4
      for bit_index in range(31, -1, -1):
        shifted:list[RKArg] = []
        incoming = numerator_bits[bit_index]
        for byte in remainder:
          carry = self._i16_positive_over(byte, 127)
          doubled = self._i16(byte, byte, _EW_CFG[Ops.ADD])
          wrapped = self._i16(doubled, self._i16(carry, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
          shifted.append(self._i16(wrapped, incoming, _EW_CFG[Ops.ADD])); incoming = carry
        remainder = shifted
        greater, equal = constants[0], constants[1]
        for left,right in zip(reversed(remainder), reversed(denominator)):
          diff = self._i16(left, right, _EW_CFG[Ops.SUB])
          positive = self._i16(self._i16(diff, constants[0], _EW_CFG[Ops.MAX]), constants[1], _EW_CFG_MIN)
          greater = self._i16(greater, self._i16(equal, positive, _EW_CFG_MIN), _EW_CFG[Ops.MAX])
          magnitude = self._i16(diff, diff, _EW_CFG_ABS)
          byte_equal = self._i16(constants[1], self._i16(magnitude, constants[1], _EW_CFG_MIN), _EW_CFG[Ops.SUB])
          equal = self._i16(equal, byte_equal, _EW_CFG_MIN)
        ge = self._i16(self._i16(greater, equal, _EW_CFG[Ops.MAX]), denominator_nonzero, _EW_CFG_MIN)
        borrow, reduced = constants[0], []
        for left,right in zip(remainder, denominator):
          partial = self._i16(left, self._i16(right, ge, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
          delta = self._i16(partial, borrow, _EW_CFG[Ops.SUB])
          borrow = self._i16_clamp_one(self._i16(constants[0], delta, _EW_CFG[Ops.SUB]))
          reduced.append(self._i16(delta, self._i16(borrow, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD]))
        remainder = reduced
        byte_index, weight = bit_index >> 3, 1 << (bit_index&7)
        quotient[byte_index] = self._i16(quotient[byte_index], self._i16(ge, constants[weight], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
      self.int32_divmod[key] = tuple(quotient), tuple(remainder), signs[0], self._i16_xor(signs[0], signs[1])
    quotient_raw, remainder_raw, remainder_sign, quotient_sign = self.int32_divmod[key]
    return self._pack_raw(u, self._i16_twos_complement(
      quotient_raw if u.op is Ops.CDIV else remainder_raw, quotient_sign if u.op is Ops.CDIV else remainder_sign), RKLayout.INT32)

  def _int32_compare(self, u:UOp) -> RKValue:
    def operand(src:UOp) -> RKValue:
      if self._is_static(src):
        raw = self.static_evaluator.values(src, int) if _index_ranges(src) else (int(_eval_expr(src, {}, {})),)*self.count
        value = self._static_slot(dtypes.int, RKLayout.INT32, tuple(number & 0xffffffff for number in raw))
      else: value = self.lower(src)
      if value.layout is not RKLayout.INT32: raise _RKGenericReject
      return value
    lhs, rhs = (operand(src) for src in u.src)
    lhs_bytes, rhs_bytes = self._raw_parts(lhs), self._raw_parts(rhs)
    constants = {value:self._constant(UOp.const(value, dtypes.int16)).arg for value in (0, 1, 127, 128, 256)}
    if u.op is Ops.CMPLT:
      mask = _int32_less_mask(self.ew_ops, self._int16_arg, constants, [value.arg for value in lhs_bytes[::-1]],
                              [value.arg for value in rhs_bytes[::-1]], self.count)
    else:
      equal = constants[1]
      for left,right in zip(lhs_bytes, rhs_bytes):
        byte_equal = self._i16_equal(left.arg, right.arg, constants[1])
        equal = self._i16(equal, byte_equal, _EW_CFG[Ops.MUL])
      if u.op is Ops.CMPEQ: mask = equal
      elif u.op is Ops.CMPNE: mask = self._i16(constants[1], equal, _EW_CFG[Ops.SUB])
      else: raise _RKGenericReject
    return RKValue(mask, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_equality(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 equality through raw bytes and native INT16 arithmetic, without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    lhs_low,lhs_high,lhs_nan = self._fp16_component_values(values[0])
    rhs_low,rhs_high,rhs_nan = self._fp16_component_values(values[1])
    low_equal = self._i16_equal(lhs_low.arg, rhs_low.arg, constants[1].arg)
    high_equal = self._i16_equal(lhs_high, rhs_high, constants[1].arg)
    either_nan = self._i16(lhs_nan, rhs_nan, _EW_CFG[Ops.MAX])
    numeric = self._i16(constants[1].arg, either_nan, _EW_CFG[Ops.SUB])
    bits_equal = self._i16(low_equal, high_equal, _EW_CFG[Ops.MUL])
    equal = self._i16(bits_equal, numeric, _EW_CFG[Ops.MUL])
    if u.op is Ops.CMPNE: equal = self._i16(constants[1].arg, equal, _EW_CFG[Ops.SUB])
    return RKValue(equal, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_component_values(self, value:RKValue) -> tuple[RKValue, RKArg, RKArg]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if value.arg in self.fp16_components: return self.fp16_components[value.arg]
    low, high = self._raw_parts(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 123, 124, 127, 128)}
    sign = self._i16_positive_over(high.arg, 127)
    sign_scale = self._i16(sign, constants[128].arg, _EW_CFG[Ops.MUL])
    magnitude = self._i16(high.arg, sign_scale, _EW_CFG[Ops.SUB])
    high_zero = self._i16_equal(magnitude, constants[0].arg, constants[1].arg)
    low_zero = self._i16_equal(low.arg, constants[0].arg, constants[1].arg)
    zero_value = self._i16(high_zero, low_zero, _EW_CFG[Ops.MUL])
    clean_high = self._i16(high.arg, self._i16(sign_scale, zero_value, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
    exponent_all = self._i16_positive_over(magnitude, 123)
    mantissa_high = self._i16_positive_over(magnitude, 124)
    mantissa = self._i16(mantissa_high, self._i16(low.arg, constants[1].arg, _EW_CFG_MIN), _EW_CFG[Ops.MAX])
    nan = self._i16(exponent_all, mantissa, _EW_CFG[Ops.MUL])
    self.fp16_components[value.arg] = low, clean_high, nan
    return self.fp16_components[value.arg]

  def _fp16_ordered_values(self, value:RKValue) -> tuple[RKArg, RKArg]:
    """Map a classified FP16 lane to two unsigned bytes whose lexical order is IEEE numeric order."""
    if value.arg in self.fp16_ordered: return self.fp16_ordered[value.arg]
    low, clean_high, _ = self._fp16_component_values(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128, 255)}
    sign = self._i16(self._i16(self._i16(clean_high, constants[127].arg, _EW_CFG[Ops.SUB]),
                               constants[0].arg, _EW_CFG[Ops.MAX]), constants[1].arg, _EW_CFG_MIN)
    positive_high = self._i16(clean_high, constants[128].arg, _EW_CFG[Ops.ADD])
    negative_high = self._i16(constants[255].arg, clean_high, _EW_CFG[Ops.SUB])
    ordered_high = self._i16(positive_high, self._i16(sign, self._i16(negative_high, positive_high, _EW_CFG[Ops.SUB]),
      _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
    negative_low = self._i16(constants[255].arg, low.arg, _EW_CFG[Ops.SUB])
    ordered_low = self._i16(low.arg, self._i16(sign, self._i16(negative_low, low.arg, _EW_CFG[Ops.SUB]),
      _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
    self.fp16_ordered[value.arg] = ordered_high, ordered_low
    return self.fp16_ordered[value.arg]

  def _fp16_less(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 less-than as an ordered raw-byte comparison without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    ordered = tuple(self._fp16_ordered_values(value) for value in values)
    nan = tuple(self._fp16_component_values(value)[2] for value in values)
    less = _ordered_byte_less(self.ew_ops, self._int16_arg, constants[0].arg, constants[1].arg,
                              ordered[0], ordered[1], self.count)
    either_nan = self._i16(nan[0], nan[1], _EW_CFG[Ops.MAX])
    numeric = self._i16(constants[1].arg, either_nan, _EW_CFG[Ops.SUB])
    result = self._i16(less, numeric, _EW_CFG[Ops.MUL])
    return RKValue(result, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _raw_where(self, u:UOp) -> RKValue:
    """Select typed values exactly, keeping nonfinite arms lazy until the selector layout is known."""
    nonfinite = [i for i,arm in enumerate(u.src[1:]) if arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.half and not math.isfinite(arm.arg)]
    arms:list[RKValue|None] = [None, None]; selector:RKValue|None = None
    if len(nonfinite) == 1:
      special, finite_index = nonfinite[0], 1-nonfinite[0]; arms[finite_index] = finite = self.lower(u.src[1+finite_index])
      if finite.layout is not RKLayout.FP16: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if self._is_static(u.src[0]) else self.lower(u.src[0])
      if selector.layout is RKLayout.BOOL_MASK:
        one, denominator = self._constant(UOp.const(1.0, dtypes.half)), selector
        if special == 0: denominator = self._scratch(dtypes.half, RKLayout.FP16); self._emit(denominator, one, selector, _EW_CFG[Ops.SUB])
        if math.isnan(float(u.src[1+special].arg)):
          zero, correction = self._constant(UOp.const(0.0, dtypes.half)), self._scratch(dtypes.half, RKLayout.FP16)
          self._emit(correction, zero, denominator, _EW_CFG[Ops.FDIV])
          return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, correction, _EW_CFG[Ops.ADD])
        sign = self._constant(UOp.const(math.copysign(1.0, float(u.src[1+special].arg)), dtypes.half))
        quotient, correction = self._scratch(dtypes.half, RKLayout.FP16), self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(quotient, sign, denominator, _EW_CFG[Ops.FDIV]); self._emit(correction, quotient, sign, _EW_CFG[Ops.SUB])
        return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, correction, _EW_CFG[Ops.ADD])
    yes, no = (self.lower(src) if value is None else value for src,value in zip(u.src[1:], arms))
    if yes.layout is not no.layout or yes.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT_FP16, RKLayout.INT32): raise _RKGenericReject
    mask_layout = RKLayout.BOOL_INT16 if yes.layout in (RKLayout.INT16, RKLayout.INT32) else RKLayout.BOOL_MASK
    if selector is None: selector = self._static(u.src[0], mask_layout) if self._is_static(u.src[0]) else self.lower(u.src[0])
    if selector.layout is not mask_layout and not (yes.layout in (RKLayout.FP16, RKLayout.INT_FP16, RKLayout.INT32) and
                                                    selector.layout in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16)): raise _RKGenericReject
    dtype = dtypes.half if yes.layout is RKLayout.FP16 else dtypes.int16 if yes.layout is RKLayout.INT16 else dtypes.int
    if dtype is dtypes.int16: return self._masked_where(u, dtype, yes.layout, selector, yes, no, self._constant(UOp.const(1, dtypes.int16)))
    if selector.layout is RKLayout.BOOL_MASK:
      mask = self._scratch(dtypes.int16, RKLayout.BOOL_INT16)
      self.ew_ops.append(RKEWOp(mask.arg, selector.arg, selector.arg, self.count, _EW_CFG[Ops.MAX], submit_barrier=True,
                               stateful=True, int16_output=True))
    else: mask = RKValue(selector.arg, dtypes.int16, self.count, RKLayout.INT16)
    yes_bytes, no_bytes = (self._raw_parts(source, cache=False, copy_wide=False) for source in (yes, no))
    selected_bytes:list[RKValue] = []
    for yes_byte,no_byte in zip(yes_bytes, no_bytes):
      delta, selected, result = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(3))
      self._emit(delta, yes_byte, no_byte, _EW_CFG[Ops.SUB])
      self._emit(selected, RKValue(mask.arg, dtypes.int16, self.count, RKLayout.INT16), delta, _EW_CFG[Ops.MUL])
      selected_bytes.append(self._emit(result, no_byte, selected, _EW_CFG[Ops.ADD]))
    return self._pack_raw(u, selected_bytes, yes.layout, dst=self._scratch(u.dtype.scalar(), yes.layout), cache=False)

  def _where(self, u:UOp) -> RKValue:
    if len(u.src) != 3: raise _RKGenericReject
    if u.dtype.scalar() is dtypes.bool:
      dynamic = [None if self._is_static(src) else self.lower(src) for src in u.src]
      preferred = (RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in dynamic) else
                   RKLayout.BOOL_MASK)
      values = [self._static(src, preferred) if value is None else self._coerce_bool(value, preferred)
                for src,value in zip(u.src, dynamic)]
      selector, yes, no = values
      data_dtype = dtypes.int16 if preferred is RKLayout.BOOL_INT16 else dtypes.half
      one = self._constant(UOp.const(1, data_dtype))
      return self._masked_where(u, dtypes.bool, preferred, selector, yes, no, one)
    if u is self.root and u.dtype.scalar() is dtypes.int and all(
      arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.int for arm in u.src[1:]
    ):
      yes_int, no_int = (int(arm.arg) for arm in u.src[1:])
      try: exact = all(_eval_cast(value, dtypes.half) == value for value in (no_int, yes_int-no_int))
      except (OverflowError, struct.error): exact = False
      if not exact: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if self._is_static(u.src[0]) else self.lower(u.src[0])
      if selector.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
      arithmetic_dtype, arithmetic_layout = ((dtypes.int16, RKLayout.INT16) if selector.layout is RKLayout.BOOL_INT16 else
                                              (dtypes.half, RKLayout.FP16))
      delta, baseline = (self._constant(UOp.const(value, arithmetic_dtype)) for value in (yes_int-no_int, no_int))
      selected, result = (self._scratch(arithmetic_dtype, arithmetic_layout) for _ in range(2))
      self._emit(selected, selector, delta, _EW_CFG[Ops.MUL])
      self._emit(result, baseline, selected, _EW_CFG[Ops.ADD])
      if arithmetic_layout is RKLayout.INT16: return self._widen_int16(u, result)
      tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
      value = RKValue(self.out, dtypes.int, self.count, RKLayout.INT32)
      self.ew_ops.append(RKEWOp(value.arg, result.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
      return value
    if u is self.root and u.dtype.scalar() in (dtypes.half, dtypes.int16) and self._is_static(u.src[0]):
      dtype = u.dtype.scalar()
      routes:dict[UOp, list[bool]] = {}
      def route(node:UOp, active:tuple[bool, ...]) -> None:
        if node.op is Ops.WHERE and self._is_static(node.src[0]):
          selector = tuple(bool(x) for x in self.static_evaluator.values(node.src[0], lambda x:int(bool(x))))
          route(node.src[1], tuple(live and take for live,take in zip(active, selector)))
          route(node.src[2], tuple(live and not take for live,take in zip(active, selector)))
          return
        mask = routes.setdefault(node, [False]*self.count)
        for i,live in enumerate(active): mask[i] |= live
      route(u, (True,)*self.count)
      def exact_operand(src:UOp) -> RKValue:
        if (src.op is Ops.LOAD and dtype is dtypes.half and len(src.src) > 2 and src.src[1].op is Ops.CONST and
            math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0.0 and
            (param:=_root_param(src.src[0])) is not None and param.src[0].op is Ops.CONST and int(param.src[0].arg) < self.count):
          return self._load(src, _fp16_bits(-65504.0))
        return self.lower(src)
      expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16
      itemsize = dtype.itemsize
      for partial,(leaf,mask) in enumerate(routes.items()):
        value = exact_operand(leaf)
        if value.layout is not expected: raise _RKGenericReject
        offsets = tuple(value.arg.addend//itemsize+i if take else -1 for i,take in enumerate(mask))
        self.mid_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count, offsets=offsets,
          partial=bool(partial), dst_kind=RKBufferKind.ARG, src_kind=value.arg.kind, itemsize=itemsize))
      return RKValue(self.out, dtype, self.count, expected)
    for fold in (_fold_where_abs, _fold_ordered_where, _fold_threshold_where):
      if (recipe:=fold(u)) is not None: return self.lower(recipe)
    return self._raw_where(u)

  def _widen_int16(self, u:UOp, source:RKValue) -> RKValue:
    if source.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16) or \
       (u is not self.root and self.int_layout is not RKLayout.INT32) or \
       (u is self.root and self.out_param.dtype.scalar() is not dtypes.int):
      raise _RKGenericReject
    zero = self._constant(UOp.const(0, dtypes.int16))
    value = RKValue(self.out if u is self.root else self._scratch(dtypes.int, RKLayout.INT32).arg,
                    dtypes.int, self.count, RKLayout.INT32)
    self.ew_ops.append(RKEWOp(value.arg, source.arg, zero.arg, self.count, _EW_CFG[Ops.ADD], int16_input=True, int32_output=True))
    return value

  def _widen_exact_int(self, source:RKValue) -> None:
    if self.out_param.dtype.scalar() is not dtypes.int or source.layout is not RKLayout.INT_FP16: raise _RKGenericReject
    tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
    self.ew_ops.append(RKEWOp(self.out, source.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True))

  def _narrow_int32(self, source:RKValue) -> RKValue:
    if source.layout is not RKLayout.INT32: raise _RKGenericReject
    value = self._scratch(dtypes.half, RKLayout.FP16)
    tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
    self.ew_ops.append(RKEWOp(value.arg, source.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], int32_input=True))
    return value

  def _math(self, u:UOp) -> RKValue:
    if len(u.src) != 1 or u.dtype.scalar() is not dtypes.half: raise _RKGenericReject
    if u.op is Ops.SQRT:
      if (recipe:=_dpu_sqrt(u.src[0])) is None: raise _RKGenericReject
    elif u.op is Ops.EXP2: recipe = _dpu_exp2(u.src[0])
    elif u.op is Ops.LOG2: recipe = _dpu_log2(u.src[0])
    elif u.op is Ops.SIN: recipe = _dpu_sin(u.src[0])
    else: raise _RKGenericReject
    recipe = _tag_precise_adds(recipe)
    value = self.lower(recipe)
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if u is self.root and value.arg != self.out:
      value = self._emit(RKValue(self.out, dtypes.half, self.count, RKLayout.FP16), value, value, _EW_CFG[Ops.MAX])
    return value

  def lower(self, u:UOp) -> RKValue:
    if u in self.values: return self.values[u]
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) and self._is_static(u):
      value = self._static(u)
    elif u.op is Ops.LOAD: value = self._load(u)
    elif u.op is Ops.BITCAST and len(u.src) == 1:
      source, src_dt = self.lower(u.src[0]), u.src[0].dtype.scalar()
      if dtype is dtypes.int16 and src_dt is dtypes.half and source.layout is RKLayout.FP16: layout = RKLayout.INT16
      elif dtype is dtypes.half and src_dt is dtypes.int16 and source.layout is RKLayout.INT16: layout = RKLayout.FP16
      else: raise _RKGenericReject(f"bitcast {src_dt}->{dtype}")
      value = RKValue(source.arg, dtype, self.count, layout)
      if u is self.root and value.arg != self.out:
        self.mid_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count,
          base=value.arg.addend//2, axes=((1, self.count, 1),), dst_kind=RKBufferKind.ARG,
          src_kind=value.arg.kind, itemsize=2))
        value = RKValue(self.out, dtype, self.count, value.layout)
    elif u.op is Ops.CAST and len(u.src) == 1:
      source_dtype = u.src[0].dtype.scalar()
      int_range = _exact_int_range(u.src[0]) if source_dtype is dtypes.int else None
      if dtype is dtypes.half and source_dtype is dtypes.float:
        source = self._load(u.src[0]) if u.src[0].op is Ops.LOAD else self.lower(_fp32_expr_to_half(u.src[0]))
      elif dtype is dtypes.half and source_dtype is dtypes.int and int_range is not None and \
           -_FP16_EXACT_INTEGER <= int_range[0] <= int_range[1] <= _FP16_EXACT_INTEGER:
        source = self.lower(_int_fp16_expr(u.src[0]))
      else: source = self.lower(u.src[0])
      if dtype is dtypes.uchar and source_dtype is dtypes.half and source.layout is RKLayout.FP16:
        cast_source = u.src[0]
        if (relu:=_relu_operand(cast_source)) is not None: cast_source = relu.alu(Ops.MAX, UOp.const(0.0, dtypes.half))
        truncated = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(cast_source,)))
        quotient = _native_floor(truncated.alu(Ops.MUL, UOp.const(1.0/256.0, dtypes.half)))
        converted, value = self.lower(truncated.alu(Ops.SUB, quotient.alu(Ops.MUL, UOp.const(256.,dtypes.half)))), self._scratch(dtype,RKLayout.INT16)
        self.ew_ops.append(RKEWOp(value.arg, converted.arg, converted.arg, self.count, _EW_CFG[Ops.MAX],
                                  submit_barrier=True, stateful=True, int16_output=True))
      elif dtype is dtypes.bool and source_dtype is dtypes.half and source.layout is RKLayout.FP16:
        value = self.lower(_positive_mask(UOp(Ops.MAX, dtypes.half, src=(u.src[0], u.src[0]), arg=_NATIVE_ABS)))
      elif source.layout is RKLayout.INT32 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.int):
        value = self._narrow_int32(source)
      elif source.layout is RKLayout.BOOL_INT16 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.bool or
                                                     dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16):
        value = self.lower(u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half)))
        if dtype is dtypes.int:
          if value.layout is not RKLayout.FP16: raise _RKGenericReject
          value = RKValue(value.arg, dtype, self.count, RKLayout.INT_FP16)
      elif (source.layout is RKLayout.FP16 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.half) or
            dtype is dtypes.half and source.layout in (RKLayout.BOOL_MASK, RKLayout.INT_FP16)):
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      elif (dtype is dtypes.int16 and source.layout in (RKLayout.INT16, RKLayout.BOOL_INT16) or
            dtype is dtypes.int and source.layout is RKLayout.BOOL_INT16 and self.int_layout is RKLayout.INT16):
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.int16 and source.layout is RKLayout.BOOL_MASK:
        value = self._scratch(dtype, RKLayout.INT16)
        self.ew_ops.append(RKEWOp(value.arg, source.arg, source.arg, self.count, _EW_CFG[Ops.MAX],
                                  stateful=True, int16_output=True))
      elif dtype is dtypes.int and source.layout in (RKLayout.FP16, RKLayout.BOOL_MASK):
        if self.int_layout is RKLayout.INT_FP16:
          value = RKValue(source.arg if source.layout is RKLayout.BOOL_MASK else self.lower(_int_fp16_expr(u)).arg,
                          dtype, self.count, self.int_layout)
        elif self.int_layout is RKLayout.INT16:
          value = self._scratch(dtype, self.int_layout)
          self.ew_ops.append(RKEWOp(value.arg, source.arg, source.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int16_output=True))
        else: raise _RKGenericReject
      elif dtype is dtypes.int:
        value = RKValue(source.arg, dtype, self.count, source.layout) if source_dtype is dtypes.uint and source.layout is RKLayout.INT32 \
          else self._widen_int16(u, source)
      else: raise _RKGenericReject(f"cast {source.layout.name}->{dtype}")
    elif u.op is Ops.ADD and dtype is dtypes.half and u.arg is None and self.accurate_adds:
      try: value = self.lower(_accurate_add_recipe(u))
      except _RKGenericReject: value = self._alu(u)
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL):
      value = self.lower(_fp32_expr_to_half(u)) if dtype is dtypes.float else self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype is dtypes.bool: value = self._bool_binary(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype in (dtypes.int16, dtypes.int): value = self._integer_bitwise(u)
    elif u.op in (Ops.SHL, Ops.SHR) and dtype in (dtypes.int, dtypes.uint): value = self._int32_shift(u)
    elif u.op in (Ops.CDIV, Ops.CMOD) and dtype is dtypes.int:
      value = RKValue(self.lower(_int_fp16_expr(u)).arg, dtype, self.count, self.int_layout) \
        if self.int_layout is RKLayout.INT_FP16 else self._int32_divmod(u)
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN): value = self._math(u)
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    self.values[u] = value
    return value

  def finish(self) -> RKImage:
    nodes = self.nodes
    if len(nodes) > 800 and not any(node.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.WHERE) for node in nodes):
      consumers = consumer_map_from_toposort(nodes)
      precise = {node for node in nodes if node.op is Ops.ADD and node.dtype.scalar() is dtypes.half and node.arg is None and
                 not any(user.op is Ops.ADD and user.dtype.scalar() is dtypes.half and user.arg is None for user in consumers[node])}
      deferred:set[UOp] = set()
      for node in nodes:
        if node in precise or any(src in deferred for src in node.src): deferred.add(node)
      accurate_adds, self.accurate_adds = self.accurate_adds, False
      for node in nodes:
        if node not in deferred and node.dtype.scalar() in (dtypes.half, dtypes.int16, dtypes.bool) and \
           node.op in (Ops.CONST, Ops.LOAD, Ops.CAST, *GroupOp.ALU):
          self.lower(node)
      self.accurate_adds = accurate_adds
    result = self.lower(self.root)
    dtype = self.out_param.dtype.scalar()
    if (dtype is dtypes.half and result.layout in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16) or
        dtype is dtypes.int16 and result.layout is RKLayout.INT16):
      if result.arg != self.out:
        self._emit(RKValue(self.out, dtype, self.count, RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.FP16),
                   result, result, _EW_CFG[Ops.MAX])
    elif dtype is dtypes.uchar and result.layout is RKLayout.INT16 or dtype is dtypes.bool and result.layout is RKLayout.BOOL_INT16:
      self.mid_gathers.append(_int16_low_bytes(result.arg, self.out_param.arg.slot, self.count))
    elif dtype is dtypes.bool and result.layout is RKLayout.BOOL_MASK:
      tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
      self.ew_ops.append(RKEWOp(self.out, result.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX],
        stateful=True, int32_output=True, bool_output=True))
    elif dtype is dtypes.int and result.layout is RKLayout.INT32:
      if result.arg != self.out:
        self.mid_gathers.append(RKGather(result.arg.index, self.out_param.arg.slot, self.count,
          base=result.arg.addend//4, axes=((1, self.count, 1),), dst_kind=RKBufferKind.ARG,
          src_kind=result.arg.kind, itemsize=4))
    elif dtype is dtypes.int and result.layout is RKLayout.INT_FP16: self._widen_exact_int(result)
    elif dtype is dtypes.int and result.layout is RKLayout.INT16: self._widen_int16(self.root, result)
    elif dtype is dtypes.float and result.layout is RKLayout.FP16:
      groups = tuple(range(0, self.count, _EW_ELEMS_32BIT))
      aligned = self._scratch(dtypes.half, RKLayout.FP16, len(groups)*16)
      split = len(self.ew_ops)
      for group,start in enumerate(groups):
        lanes = min(_EW_ELEMS_32BIT, self.count-start)
        self.mid_gathers.append(RKGather(result.arg.index, aligned.arg.index, lanes,
          offsets=tuple(result.arg.addend//2+lane for lane in range(start, start+lanes)), dst_addend=group*8,
          src_kind=result.arg.kind, after=split))
        source = replace(aligned.arg, addend=group*16)
        self.ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, self.out_param.arg.slot, start*4), source, source, lanes,
                                  _EW_CFG[Ops.MAX] | _EW_STAGE_FP32_OUT))
    else: raise _RKGenericReject
    constants = b""
    if self.constants:
      by_slot = {slot:bits for bits,slot in self.constants.items()}
      constants = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot)+1))
    image = RKImage(tuple(self.scratch), constants, gathers=tuple(self.gathers), ew_ops=tuple(self.ew_ops),
                    mid_gathers=tuple(replace(gather, after=len(self.ew_ops)) if gather.after < 0 else gather
                                      for gather in self.mid_gathers),
                    host_gathers=tuple(self.host_gathers))
    return _reuse_linear_scratch(image, self.constants)

def _structural_reduce(reduce_op:Ops, dtype:DType, terms:list[UOp]) -> UOp:
  if reduce_op is Ops.ADD and dtype.scalar() is dtypes.half:
    nonzero = [term for term in terms if not (term.op is Ops.CONST and float(term.arg) == 0.0)]
    if nonzero and all(term.op is Ops.MUL and term.dtype.scalar() is dtypes.half for term in nonzero):
      return _precise_mul_sum(nonzero)
  while len(terms) > 1:
    terms = [UOp(reduce_op, dtype, src=(terms[i], terms[i+1])) for i in range(0, len(terms)-1, 2)] + \
      (terms[-1:] if len(terms) & 1 else [])
  return terms[0]

def _expand_math_uops(root:UOp, nodes:dict[UOp, None], *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  bounded_recipes = len(nodes) <= _MAX_OPTIONAL_RECIPE_NODES
  if not bounded_recipes and not any(u.op in (Ops.WHERE, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.TRUNC) or
    _half_storage(u) is not None for u in nodes): return root
  composite_math = (folded if (folded:=_fold_inverse_hyperbolic(root)) is not None else _fold_atan(root)) if bounded_recipes else None
  cache:dict[UOp, UOp] = {}
  exact_static_selection = root.op is Ops.WHERE and _is_static_expr(root.src[0]) and not any(
    node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half, dtypes.float) and not math.isfinite(float(node.arg))
    for node in nodes)
  def physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp:
    placeholders = {source:UOp.param(-index-1, source.dtype, ()) for index,source in enumerate(opaque)}
    rewritten = _fp16_rewrite(list(UOp(Ops.SINK, src=(recipe.substitute(placeholders),)).toposort()))
    if not rewritten or rewritten[-1].op is not Ops.SINK or len(rewritten[-1].src) != 1: raise _RKGenericReject
    tagged_root = _tag_precise_adds(rewritten[-1].src[0])
    return tagged_root.substitute({placeholder:source for source,placeholder in placeholders.items()})
  if composite_math is not None: root = physical_recipe(composite_math)
  def rewrite(u:UOp) -> UOp:
    if u in cache: return cache[u]
    if (src:=_half_storage(u)) is not None:
      cache[u] = rewrite(physical_recipe(_dpu_sin(src.src[0]), (src.src[0],))) if src.op is Ops.SIN else _canonical_half_storage(src)
      return cache[u]
    if accurate_adds and bounded_recipes and u.op is Ops.ADD and u.dtype.scalar() is dtypes.half and u.arg is None:
      try: cache[u] = _accurate_add_recipe(u); return cache[u]
      except _RKGenericReject: pass
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.WHERE and (absolute:=_fold_where_abs(mapped)) is not None: mapped = rewrite(absolute)
    if exact_static_selection and mapped.op is Ops.MUL and (minimum:=_fold_minimum(mapped)) is not None:
      mapped = minimum.replace(arg=_NATIVE_RAW_MIN)
    if mapped.op is Ops.SQRT:
      if (recipe:=_dpu_sqrt(mapped.src[0])) is None: raise _RKGenericReject
      mapped = rewrite(physical_recipe(recipe, (mapped.src[0],)))
    elif mapped.op in (Ops.EXP2, Ops.LOG2, Ops.SIN):
      if mapped.op is Ops.LOG2 and mapped.src[0].op is Ops.WHERE: raise _RKGenericReject
      recipe = _dpu_exp2(mapped.src[0]) if mapped.op is Ops.EXP2 else _dpu_log2(mapped.src[0]) if mapped.op is Ops.LOG2 else \
        _dpu_sin(mapped.src[0])
      mapped = rewrite(physical_recipe(recipe, (mapped.src[0],)))
    elif mapped.op is Ops.TRUNC and mapped.dtype.scalar() is dtypes.half and not _is_static_expr(mapped):
      mapped = rewrite(_fold_trunc(mapped))
    cache[u] = mapped
    return mapped
  return rewrite(root)

_pm_fp32_sin_storage = PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat(Ops.SIN, dtypes.float),)),
   lambda root:_expand_math_uops(root, root.toposort())),
])

def _finite_max_neutrals(root:UOp, nodes:dict[UOp, None]) -> tuple[UOp, dict[UOp, None]]:
  """Canonicalize finite physical neutrals for FP selectors and exact INT32 MAX arithmetic."""
  int_min = any(u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg) == dtypes.int.min for u in nodes)
  if root.op is not Ops.MAX and not int_min: return root,nodes
  finite_selectors = root.op is Ops.MAX
  cache:dict[tuple[UOp, bool], UOp] = {}
  stack:list[tuple[UOp, bool, bool]] = [(root, False, False)]
  while stack:
    u,under_max,ready = stack.pop()
    key = (u, under_max)
    if key in cache: continue
    active = under_max or u.op is Ops.MAX and u.dtype.scalar() is dtypes.int
    if active and u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg) == dtypes.int.min:
      cache[key] = u.const_like(-2048)
    elif ready:
      src = tuple(cache[(source, active)] for source in u.src)
      if (finite_selectors and u.op is Ops.WHERE and src[1].op is Ops.CONST and
          src[1].dtype.scalar() in (dtypes.half, dtypes.float) and math.isinf(float(src[1].arg)) and float(src[1].arg) < 0.0):
        src = (src[0], src[1].const_like(-65504.0), src[2])
      cache[key] = u.replace(src=src)
    else:
      stack.append((u, under_max, True))
      stack.extend((src, active, False) for src in reversed(u.src))
  return cache[(root, False)], cache[(root, False)].toposort()

def _substitute_static_ranges(root:UOp, replacements:dict[UOp, UOp]) -> UOp:
  cache:dict[UOp, UOp] = {}
  for u in root.toposort(): cache[u] = replacements[u] if u in replacements else u.replace(src=tuple(cache[src] for src in u.src))
  return cache[root]

def _unroll_static_reduces(root:UOp) -> UOp:
  """Interpret static REDUCE/RANGE structure into ordinary semantic UOps."""
  cache:dict[UOp, UOp] = {}
  def rewrite(u:UOp) -> UOp:
    if u in cache: return cache[u]
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.REDUCE:
      reduce_op = mapped.arg[0] if isinstance(mapped.arg, tuple) else mapped.arg
      ranges = list(mapped.src[1:])
      if reduce_op not in (Ops.ADD, Ops.MAX, Ops.MUL) or not ranges or any(
        r.op is not Ops.RANGE or r.src[0].op is not Ops.CONST for r in ranges): raise _RKGenericReject
      iterations = math.prod(int(r.src[0].arg) for r in ranges)
      if iterations > _MAX_GENERIC_UNROLL or iterations*len(mapped.src[0].toposort()) > _MAX_GENERIC_EXPANDED_NODES:
        raise _RKGenericReject
      envs = _iter_selected_range_env(ranges)
      if not envs: raise _RKGenericReject
      terms = [_substitute_static_ranges(mapped.src[0], {r:r.const_like(env[r]) for r in ranges}) for env in envs]
      mapped = _structural_reduce(reduce_op, u.dtype, terms)
    cache[u] = mapped
    return mapped
  return rewrite(root)

def _local_buffer(u:UOp) -> UOp|None:
  u = _strip_cast(u)
  if u.op in (Ops.LOAD, Ops.STORE): u = u.src[0]
  if u.op is Ops.INDEX: u = u.src[0]
  while u.op is Ops.AFTER: u = u.src[0]
  return u if u.op is Ops.BUFFER else None

def _unroll_static_local(uops:list[UOp], root:UOp) -> UOp:
  """Execute static local accumulators and identity-indexed local bridges without recovering a tensor operation."""
  local_load_cache:dict[UOp, tuple[UOp, ...]] = {}
  local_loads = _semantic_local_loads(root, local_load_cache)
  buffers = {_local_buffer(load) for load in local_loads}
  if not local_loads: return root
  if None in buffers: raise _RKGenericReject
  definitions:dict[UOp, _RKStaticLocalDef] = {}
  expanded:dict[tuple[UOp, tuple[tuple[UOp, int], ...]], UOp] = {}
  active:set[tuple[UOp, tuple[tuple[UOp, int], ...]]] = set()
  budget = [_MAX_STATIC_LOCAL_STEPS]
  def expand_dependencies(expr:UOp, owner:UOp, env:dict[UOp, int]) -> UOp:
    substitutions = {axis:axis.const_like(value) for axis,value in env.items()}
    substitutions.update({load:expand_load(load, env) for load in _semantic_local_loads(expr, local_load_cache)
                          if _local_buffer(load) is not owner})
    return _substitute_static_ranges(expr, substitutions)
  def expand_load(load:UOp, env:dict[UOp, int]) -> UOp:
    buffer = _local_buffer(load)
    if buffer is None or buffer.src[0].op is not Ops.CONST: raise _RKGenericReject
    if int(buffer.src[0].arg) == 1: return expand_buffer(buffer, env)
    stores = [u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]
    if len(stores) != 1 or stores[0].src[0].op is not Ops.INDEX: raise _RKGenericReject
    store_index = stores[0].src[0].src[1]
    load_index = _substitute_static_ranges(load.src[0].src[1], {axis:axis.const_like(value) for axis,value in env.items()})
    axes = _index_ranges(store_index)
    if len(axes) != 1 or store_index.key != axes[0].key or axes[0].op is not Ops.SPECIAL or \
       axes[0].src[0].op is not Ops.CONST or int(axes[0].src[0].arg) != int(buffer.src[0].arg): raise _RKGenericReject
    return _substitute_static_ranges(expand_dependencies(stores[0].src[1], buffer, env), {axes[0]:load_index})
  def expand_buffer(buffer:UOp, env:dict[UOp, int]) -> UOp:
    key = buffer, tuple(sorted(env.items(), key=lambda item:item[0].key))
    if key in expanded: return expanded[key]
    if key in active: raise _RKGenericReject
    active.add(key)
    if buffer not in definitions: definitions.update(_static_local_defs(uops, {buffer}))
    definition = definitions[buffer]
    if not definition.loops or any(loop.src[0].op is not Ops.CONST or not 0 <= int(loop.src[0].arg) <= _MAX_GENERIC_UNROLL
                                   for loop in definition.loops): raise _RKGenericReject
    accumulator = expand_dependencies(definition.initial, buffer, env)
    steps = math.prod(int(loop.src[0].arg) for loop in definition.loops)
    if steps > budget[0]: raise _RKGenericReject
    budget[0] -= steps
    for loop_env in _iter_selected_range_env(list(definition.loops)):
      term = expand_dependencies(definition.term, buffer, {**env, **loop_env})
      accumulator = UOp(definition.update_op, buffer.dtype, src=(accumulator, term))
    expanded[key] = accumulator
    active.remove(key)
    if len(expanded[key].toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
    return expanded[key]
  substitutions = {load:expand_load(load, {}) for load in local_loads}
  return _substitute_static_ranges(root, substitutions)

@dataclass(frozen=True, slots=True)
class _RKStaticLocalDef:
  initial:UOp; update_op:Ops; term:UOp; loops:tuple[UOp, ...]

def _flat_static_ranges(loops:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, int]:
  flat, groups = UOp.const(0, dtypes.int), 1
  for loop in reversed(loops):
    flat = flat.alu(Ops.ADD, loop.alu(Ops.MUL, loop.const_like(groups)))
    groups *= int(loop.src[0].arg)
  return flat, groups

def _static_local_defs(uops:list[UOp], buffers:set[UOp]) -> dict[UOp, _RKStaticLocalDef]:
  """Parse scalar local accumulators without assigning tensor-operation meaning to their loops."""
  definitions:dict[UOp, _RKStaticLocalDef] = {}
  for buffer in buffers:
    if buffer.src[0].op is not Ops.CONST or int(buffer.src[0].arg) != 1: raise _RKGenericReject
    stores = [u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]
    initializers:list[UOp] = []
    updates:list[tuple[Ops, UOp, tuple[UOp, ...]]] = []
    for store in stores:
      value = _strip_cast(store.src[1])
      accumulator = [(i, _local_load(src)) for i,src in enumerate(value.src)] \
        if value.op in (Ops.ADD, Ops.MAX, Ops.MUL, Ops.AND, Ops.OR) else []
      accumulator = [(i, load) for i,load in accumulator if load is not None and _local_buffer(load) is buffer]
      if len(accumulator) == 1:
        term = value.src[1-accumulator[0][0]]
        if any(_local_buffer(load) is buffer for node in term.toposort() if (load:=_local_load(node)) is not None):
          raise _RKGenericReject
        base = store.src[0].src[0] if store.src and store.src[0].op is Ops.INDEX else None
        loops = tuple(src for src in base.src[1:] if src.op is Ops.RANGE) if base is not None and base.op is Ops.AFTER else ()
        if not loops: loops = tuple(r for r in _index_ranges(term) if r.arg[1] is AxisType.REDUCE)
        if not loops: raise _RKGenericReject
        updates.append((value.op, term, loops))
      elif not any(_local_buffer(load) is buffer for node in value.toposort() if (load:=_local_load(node)) is not None):
        initializers.append(store.src[1])
    if len(initializers) != 1 or len(updates) != 1: raise _RKGenericReject
    definitions[buffer] = _RKStaticLocalDef(initializers[0], *updates[0])
  return definitions

def _mapped_max_descriptor(uops:list[UOp], root:UOp) -> tuple[_RKStaticLocalDef, UOp]:
  local_loads = _semantic_local_loads(root)
  if len(local_loads) == 1 and (buffer:=_local_buffer(local_loads[0])) is not None:
    return _static_local_defs(uops, {buffer})[buffer], local_loads[0]
  if local_loads: raise _RKGenericReject
  nodes = root.toposort()
  plain_max = {node for node in nodes if node.op is Ops.MAX and node.arg is None and node.dtype.scalar() is dtypes.int}
  nested = {source for node in plain_max for source in node.src if source in plain_max}
  candidates = [(node, terms) for node in nodes if node in plain_max-nested and
                128 <= len(terms:=_flatten_binary(node, Ops.MAX, plain=True)) <= 512]
  if len(candidates) != 1: raise _RKGenericReject
  reduced, terms = candidates[0]
  axis = UOp.range(len(terms), len(nodes), AxisType.REDUCE, dtype=dtypes.int)
  def parameterize(lhs:UOp, rhs:UOp) -> UOp:
    if lhs.key == rhs.key: return lhs
    if lhs.op is rhs.op is Ops.CONST and lhs.dtype == rhs.dtype and lhs.dtype.scalar() is dtypes.int:
      return lhs.const_like(2*int(lhs.arg)-int(rhs.arg)) + axis*(int(rhs.arg)-int(lhs.arg))
    if lhs.op is not rhs.op or lhs.dtype != rhs.dtype or lhs.arg != rhs.arg or len(lhs.src) != len(rhs.src): raise _RKGenericReject
    return lhs.replace(src=tuple(parameterize(left, right) for left,right in zip(lhs.src, rhs.src)))
  term = parameterize(terms[1], terms[2])
  if any(graph_rewrite(term.substitute({axis:axis.const_like(k)}), sym, name="rockchip repeated MAX proof").key != peer.key
         for k,peer in enumerate(terms)): raise _RKGenericReject
  return _RKStaticLocalDef(UOp.const(0, dtypes.int), Ops.MAX, term, (axis,)), reduced

def _map_loop_term_image(term:UOp, axes:list[UOp], rows:int, out_index:UOp, slot:int, reduce_op:Ops) -> RKImage|None:
  """Map one loop-indexed FP16 term onto physical lanes, then reduce those lanes."""
  clean = {r:r.replace(src=(r.src[0],)) for r in axes if len(r.src) > 1}
  if clean: term, axes = term.substitute(clean), [clean.get(r, r) for r in axes]
  if term.dtype.scalar() is dtypes.float: term = term.cast(dtypes.half)
  if term.dtype.scalar() is not dtypes.half: return None
  flat, groups = _flat_static_ranges(axes)
  lanes = rows*groups
  linear_index = flat.alu(Ops.MUL, flat.const_like(rows)).alu(Ops.ADD, out_index)
  mapped = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(
    UOp.param(slot, dtypes.half, (lanes,)).index(linear_index).store(term),)).toposort())),
                              vectorize_reductions=False, recipes_ready=True)
  if mapped is None: return None
  return _finish_mapped_add_reduction(mapped, slot, rows, groups, 1.0, reduce_op=reduce_op)

def _lower_mapped_scalar_local_reduction(uops:list[UOp]) -> RKImage|None:
  """Map one bounded scalar local reduction over physical lanes, then execute its post-UOps normally."""
  if (output:=_output_store(uops, dtypes.int, allow_local=True)) is None: return None
  store, out, rows, out_index, root = output
  if not _contiguous_output(out_index, rows): return None
  try: definition, reduced_node = _mapped_max_descriptor(uops, root)
  except _RKGenericReject: return None
  if definition.update_op not in (Ops.ADD, Ops.MAX) or definition.initial.op is not Ops.CONST or not definition.loops or any(
     loop.src[0].op is not Ops.CONST or int(loop.src[0].arg) <= 0 for loop in definition.loops): return None
  clean = {r:r.replace(src=(r.src[0],)) for r in definition.loops if len(r.src) > 1}
  loops = [clean.get(r, r) for r in definition.loops]
  term_base = definition.term.substitute(clean) if clean else definition.term
  flat, groups = _flat_static_ranges(loops)
  term_range = _exact_int_range(definition.term)
  initial = int(definition.initial.arg)
  if groups > _MAX_STATIC_RANGE_ENVS or term_range is None or initial > term_range[0]: return None
  local_range = (initial+groups*term_range[0], initial+groups*term_range[1]) if definition.update_op is Ops.ADD else \
    (max(initial, term_range[0]), max(initial, term_range[1]))
  use_int16, limit = True, 32767
  if not -limit <= term_range[0] <= term_range[1] <= limit: return None
  if local_range[0] != 0 or local_range[1] > limit: return None
  bounded_local = UOp.special(local_range[1]+1, "rockchip_local_bound", dtypes.int)
  proof = root.substitute({reduced_node:bounded_local})
  if any((bound:=_exact_int_range(node)) is None or not -limit <= bound[0] <= bound[1] <= limit
         for node in proof.toposort() if node.dtype.scalar() in (dtypes.int, dtypes.weakint)): return None
  next_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  nested_images:list[RKImage] = []; nested_slots:list[int] = []
  def premap(expr:UOp) -> UOp|None:
    nonlocal next_slot
    rewritten = expr
    for buf in dict.fromkeys(b for n in expr.toposort() if n.op is Ops.LOAD and (b:=_local_buffer(n)) is not None):
      try: nested = _static_local_defs(uops, {buf})[buf]
      except _RKGenericReject: return None
      if nested.update_op not in (Ops.MAX, Ops.ADD) or nested.initial.dtype.scalar() not in (dtypes.half, dtypes.float): return None
      inner = premap(nested.term)
      if inner is None: return None
      slot = next_slot; next_slot += 1
      mapped = _map_loop_term_image(inner, list(nested.loops), rows, out_index, slot, nested.update_op)
      if mapped is None: return None
      nested_images.append(mapped); nested_slots.append(slot)
      reduced_load = UOp.param(slot, dtypes.half, (rows,)).index(out_index).load()
      rewritten = rewritten.substitute({n:reduced_load for n in rewritten.toposort()
                                        if n.op is Ops.LOAD and _local_buffer(n) is buf})
    return rewritten
  term_src = premap(term_base)
  if term_src is None: return None
  try:
    if definition.update_op is Ops.ADD and term_src.op is Ops.WHERE and all(src.op is Ops.CONST for src in term_src.src[1:]):
      term = UOp(Ops.WHERE, dtypes.int16, src=(term_src.src[0], *(_int16_expr(src) for src in term_src.src[1:])))
    elif definition.update_op is Ops.ADD and term_src.op is Ops.CAST and term_src.src[0].dtype is dtypes.bool:
      term = term_src.src[0].where(UOp.const(1, dtypes.int16), UOp.const(0, dtypes.int16))
    else: term = _int16_expr(term_src) if use_int16 else _int_fp16_expr(term_src)
  except _RKGenericReject: return None

  out_axes, reduce_axes = _index_ranges(out_index), list(loops)
  if set(out_axes) & set(reduce_axes): return None
  lanes = rows*groups
  axis = max((u.arg[0] for u in (*out_axes, *reduce_axes) if isinstance(u.arg, tuple)), default=-1)+1
  lane = UOp.range(lanes, axis)
  fake_out_slot, next_slot = next_slot, next_slot+1
  replacements:dict[UOp, UOp] = {}
  materializations:list[tuple[int, int, RKGather, tuple[int, int, int], int]] = []
  out_evaluator, reduce_evaluator = RKStaticIndexEvaluator(out_index, rows), RKStaticIndexEvaluator(flat, groups)
  lane_dtype, encode = (dtypes.int16, _int16_bits) if use_int16 else (dtypes.half, _fp16_bits)

  def materialize(node:UOp, count:int, plan:RKGather, broadcast:tuple[int, int, int], dtype:DType=dtypes.half) -> None:
    nonlocal next_slot
    source = UOp.param(next_slot, dtype, (lanes,)); next_slot += 1
    replacements[node] = source.index(lane).load()
    materializations.append((source.arg.slot, count, replace(plan, itemsize=dtype.itemsize), broadcast, dtype.itemsize))

  axis_dtype, axis_encode = (dtypes.int, lambda value:value&0xffffffff) if definition.update_op is Ops.ADD else (lane_dtype, encode)
  for value in out_axes:
    materialize(value if axis_dtype is dtypes.int else value.cast(axis_dtype), rows,
      RKGather(0, 0, rows, values=out_evaluator.values(value, axis_encode)), (1, rows, 1), axis_dtype)
  for value in reduce_axes:
    materialize(value if axis_dtype is dtypes.int else value.cast(axis_dtype), groups,
      RKGather(0, 0, groups, values=reduce_evaluator.values(value, axis_encode)), (rows, groups, 1), axis_dtype)
  out_set, reduce_set = set(out_axes), set(reduce_axes)
  for load in dict.fromkeys(node for node in term_src.toposort() if node.op is Ops.LOAD and _root_param(node.src[0]) is not None):
    load_dtype = load.dtype.scalar()
    if load_dtype not in ((dtypes.int, dtypes.half) if definition.update_op is Ops.ADD else (dtypes.half,)) or not load.src or \
       load.src[0].op is not Ops.INDEX or len(load.src) not in (1, 3) or len(load.src) == 3 and load.src[1].op is not Ops.CONST or \
       (param:=_root_param(load.src[0])) is None or param.src[0].op is not Ops.CONST: return None
    dependencies = set(_index_ranges(load.src[0].src[1])) | (set(_index_ranges(load.src[2])) if len(load.src) == 3 else set())
    if dependencies <= out_set:
      count, base_index, broadcast = rows, out_index, (1, rows, 1)
    elif dependencies <= reduce_set:
      count, base_index, broadcast = groups, flat, (rows, groups, 1)
    else: return None
    try:
      fill = 0 if len(load.src) == 1 else int(load.src[1].arg) & ((1 << (load_dtype.itemsize*8))-1)
      plan = _gather_plan(param.arg.slot, 0, base_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count, fill)
      _validate_gather_bounds(plan, int(param.src[0].arg))
    except RuntimeError: return None
    materialize(load, count, plan, broadcast, load_dtype)

  term = term.substitute(replacements)
  mapped = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(
    UOp.param(fake_out_slot, lane_dtype, (lanes,)).index(lane).store(term),)).toposort())),
                              vectorize_reductions=False, recipes_ready=True)
  if mapped is None: return None
  scratch = list(mapped.scratch)
  gathers = list(mapped.gathers)
  slots:dict[int, int] = {}
  for fake_slot,count,plan,broadcast,itemsize in materializations:
    base_slot, vector_slot = len(scratch), len(scratch)+1
    scratch.extend((RKScratch(_scratch_bytes(count*itemsize//2)), RKScratch(_scratch_bytes(lanes*itemsize//2))))
    gathers.extend((replace(plan, dst_index=base_slot), RKGather(base_slot, vector_slot, lanes, axes=(broadcast,),
                                                                 src_kind=RKBufferKind.SCRATCH, itemsize=itemsize)))
    slots[fake_slot] = vector_slot
  mapped = _alias_image_args(replace(mapped, scratch=tuple(scratch), gathers=tuple(gathers)),
                             {slot:_scratch_arg(target) for slot,target in slots.items()})
  if (reduced:=_finish_mapped_add_reduction(mapped, fake_out_slot, rows, groups, 1.0, reduce_op=definition.update_op,
                                            int16=use_int16)) is None: return None
  combined:RKImage|None = None
  for image in (*nested_images, reduced):
    combined = image if combined is None else _append_inplace_image(combined, image)
    if combined is None: return None
  fake_source = UOp.param(fake_out_slot, lane_dtype, (rows,))
  bound = bounded_local.cast(lane_dtype)
  try:
    post_expr = _int16_expr(proof) if use_int16 else _int_fp16_expr(proof)
    post_value = post_expr.substitute({bound:fake_source.index(out_index).load()}).cast(dtypes.int)
  except _RKGenericReject: return None
  post = _lower_uop_program(list(UOp(Ops.SINK, src=(store.replace(src=(store.src[0], post_value, *store.src[2:])),)).toposort()),
                            vectorize_reductions=False)
  if post is None or combined is None or (appended:=_append_inplace_image(combined, post)) is None: return None
  aliases, extra = dict[int, RKArg](), list[RKScratch]()
  for slot in (*nested_slots, fake_out_slot):
    aliases[slot] = _scratch_arg(len(appended.scratch)+len(extra)); extra.append(RKScratch(_scratch_bytes(rows)))
  return replace(_alias_image_args(appended, aliases), scratch=(*appended.scratch, *extra))

def _lower_host_scatter(uops:list[UOp]) -> RKImage|None:
  """Lower a direct dynamic STORE as raw last-writer host address materialization."""
  if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1" or \
     (output:=_output_store(uops, (dtypes.half, dtypes.int16))) is None or len(output[0].src) != 2: return None
  _, out_param, out_count, dynamic_index, value = output
  if (index_info:=_runtime_index(dynamic_index)) is None: return None
  _, index_param, lane_index, index_itemsize = index_info
  value = _strip_cast(value)
  if (value.op is not Ops.LOAD or len(value.src) != 1 or value.src[0].op is not Ops.INDEX or
      (source:=_root_param(value.src[0])) is None or source.src[0].op is not Ops.CONST or
      value.src[0].src[1].key != lane_index.key or source.dtype.scalar() is not out_param.dtype.scalar()): return None
  count = int(index_param.src[0].arg)
  if int(source.src[0].arg) < count: return None
  address = RKHostAddress(RKArg(RKBufferKind.ARG, source.arg.slot), RKArg(RKBufferKind.ARG, index_param.arg.slot),
    RKArg(RKBufferKind.ARG, out_param.arg.slot), count, int(source.src[0].arg), out_count,
    itemsize=out_param.dtype.scalar().itemsize, index_itemsize=index_itemsize)
  return RKImage(host_scatters=(address,))

def _lower_uop_program(uops:list[UOp], *, vectorize_reductions:bool=True, recipes_ready:bool=False) -> RKImage|None:
  """Lower a composable typed UOp program; return None for the legacy correctness oracle."""
  output_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is not None]
  if len(output_stores) > 1:
    combined:RKImage|None = None
    for store in output_stores:
      child = _lower_uop_program(list(UOp(Ops.SINK, src=(store,)).toposort()),
                                 vectorize_reductions=vectorize_reductions, recipes_ready=recipes_ready)
      if child is None: return None
      combined = child if combined is None else _append_inplace_image(combined, child)
      if combined is None: return None
    return combined
  if vectorize_reductions and (mapped_local:=_lower_mapped_scalar_local_reduction(uops)) is not None: return mapped_local
  if (scatter:=_lower_host_scatter(uops)) is not None: return scatter
  if (int_output:=_output_store(uops, dtypes.int)) is not None:
    if (raw_bitcast:=_lower_raw_fp16_bitcast(int_output)) is not None: return raw_bitcast
  for dtype in (dtypes.half, dtypes.int16, dtypes.int):
    if (direct_load:=_output_store(uops, dtype)) is None: continue
    if (image:=_lower_dynamic_typed_load(direct_load, dtype)) is not None: return image
    root = direct_load[4]
    if root.op is Ops.WHERE and root.src[1].op is Ops.LOAD and root.src[2].op is Ops.CONST and \
       (folded_load:=_fold_masked_load(root.src[0], root.src[1], root.src[2], allow_additional_gate_loads=True)) is not None and \
       (image:=_lower_dynamic_typed_load((*direct_load[:4], folded_load), dtype)) is not None: return image
  storage_uops:list[UOp]|None = None
  storage_product_adds = False
  if any(u.dtype.scalar() is dtypes.float for u in uops):
    sink = next((u for u in uops if u.op is Ops.SINK), None)
    if sink is not None:
      storage_sink = sink
      if (storage_output:=_output_store(uops, dtypes.half, allow_local=True)) is not None:
        storage_root = storage_output[4]
        root_storage = _half_storage(storage_root) is not None
        storage_product_adds = any((src:=_half_storage(boundary)) is not None and _fp32_add_has_product_terms(src) and
          (boundary is not storage_root or len(src.toposort()) > 64) for boundary in storage_root.toposort())
        # A later half FDIV/WHERE/etc. can own several independent FP32 reduction boundaries. Commit each CAST
        # before the bottom-up generic rewrite erases the semantic FP32 ADD tree, including pure denominator sums.
        if (ratio:=_fp32_ratio_to_half(storage_root)) is not None: storage_sink = storage_sink.substitute({storage_root:ratio})
        else:
          nested_storage:dict[UOp, UOp] = {}
          for boundary in storage_root.toposort():
            if boundary is storage_root or (src:=_half_storage(boundary)) is None or src.op is not Ops.ADD: continue
            try: nested_storage[boundary] = _canonical_half_storage(src)
            except _RKGenericReject: pass
          if nested_storage: storage_sink = storage_sink.substitute(nested_storage)
        if root_storage:
          try:
            source = storage_root.src[0]
            if not _has_runtime_address(source):
              converted = _expand_math_uops(storage_root, storage_root.toposort()) if source.op is Ops.SIN else _canonical_half_storage(source)
              storage_sink = sink.substitute({storage_root:converted})
          except _RKGenericReject: pass
      storage_sink = graph_rewrite(storage_sink, _pm_fp32_sin_storage, name="rockchip fp32 sin storage")
      storage_uops = list(graph_rewrite(storage_sink, _pm_generic_storage_precision,
                                        name="rockchip generic storage precision").toposort())
  if vectorize_reductions and (mul_add:=_lower_vectorized_mul_add_reduction(uops)) is not None: return mul_add
  if vectorize_reductions and (repeated_add:=_lower_repeated_add_reduction(uops)) is not None: return repeated_add
  mapped_loop = _lower_mapped_add_loop_reduction(uops) if vectorize_reductions else None
  if mapped_loop is not None: return mapped_loop
  if storage_uops is not None: uops = storage_uops
  if (output:=_output_store(uops, (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool, dtypes.uchar), allow_local=True)) is None or \
     len(output[0].src) != 2:
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage()
  if not _tautological_store_scope(uops, output[0]): return None
  if output[1].dtype.scalar() is dtypes.int:
    for source_dtype in (dtypes.half, dtypes.int16, dtypes.int):
      if (coordinates:=_lower_bounded_integer_predicate_coordinates(output, source_dtype)) is not None: return coordinates
    if (bounded:=_bounded_int32_fp16_root(output[4])) is not None: output = (*output[:4], bounded)
  try:
    if (not _contiguous_output(output[3], output[2]) and
        _static_int_vector(output[3], output[3], output[2]) != tuple(range(output[2]))): return None
    reduced = _unroll_static_reduces(output[4]) if any(u.op is Ops.REDUCE for u in uops) else output[4]
    local_root = _unroll_static_local(uops, reduced)
    root = graph_rewrite(local_root, _pm_masked_loads, name="rockchip masked load materialization")
    root_nodes = root.toposort()
    root,root_nodes = _finite_max_neutrals(root, root_nodes)
    defer_math = len(root_nodes) > 256
    if not recipes_ready and not defer_math and \
       (expanded:=_expand_math_uops(root, root_nodes, accurate_adds=storage_uops is None or storage_product_adds)) is not root:
      root,root_nodes = expanded,expanded.toposort()
    expanded_nodes = root_nodes
    if len(expanded_nodes) > _MAX_GENERIC_EXPANDED_NODES:
      if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(expanded_nodes)}")
      return None
    if root is not output[4]:
      output = (*output[:4], root)
    image = RKContext(output, expanded_nodes, accurate_adds=(not recipes_ready and (storage_uops is None or storage_product_adds) and
                                             len(expanded_nodes) <= _MAX_OPTIONAL_RECIPE_NODES and
                                             not _has_runtime_address(output[4]))).finish()
    if not _image_u16_safe(image):
      if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1":
        raise _RKGenericReject(f"image u16 fields, ew_ops={len(image.ew_ops)}")
      return None
    return image
  except (_RKGenericReject, RuntimeError, ValueError, KeyError):
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise
    return None


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

def _fold_masked_mul(x:UOp) -> UOp|None:
  """Push a WHERE through MUL so inactive factors become identities before native DPU multiplication."""
  gate, yes, no = x.src
  if yes.op is Ops.WHERE and _same_condition(gate, yes.src[0]): return UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[1], no))
  if no.op is Ops.WHERE and _same_condition(gate, no.src[0]): return UOp(Ops.WHERE, x.dtype, src=(gate, yes, no.src[2]))
  one = UOp.const(1.0, dtypes.half)
  if yes.op is Ops.MUL and no.op is Ops.CONST:
    return UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[0], no)).alu(Ops.MUL, UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[1], one)))
  if no.op is Ops.MUL and yes.op is Ops.CONST:
    return UOp(Ops.WHERE, x.dtype, src=(gate, yes, no.src[0])).alu(Ops.MUL, UOp(Ops.WHERE, x.dtype, src=(gate, one, no.src[1])))
  return None

def _const_operand(u:UOp, op:Ops, value:float|None=None) -> tuple[UOp, UOp]|None:
  if u.op is not op: return None
  for operand, const in (u.src, u.src[::-1]):
    if const.op is Ops.CONST and (value is None or float(const.arg) == value): return operand, const
  return None

def _native_min(lhs:UOp, rhs:UOp) -> UOp:
  return UOp(Ops.MAX, lhs.dtype, src=(lhs, rhs), arg=_NATIVE_MIN)

def _fold_ordered_where(x:UOp) -> UOp|None:
  """Turn ordered clamp WHEREs into native DPU EW MIN/MAX stages."""
  gate, yes, no = x.src
  if gate.op is Ops.OR and yes.op is Ops.CONST:
    for upper, lower in ((gate.src[0], gate.src[1]), (gate.src[1], gate.src[0])):
      if (upper.op is Ops.CMPLT and upper.src[0].key == yes.key and upper.src[1].op is Ops.MAX and
          lower.op is Ops.CMPLT and lower.src[0].key == no.key and lower.src[1].key == yes.key and
          {u.key for u in upper.src[1].src} == {no.key, yes.key}): return _native_min(upper.src[1], yes)
  if gate.op is not Ops.CMPLT: return None
  lhs, rhs = gate.src
  if yes.key == rhs.key and no.key == lhs.key: return lhs.alu(Ops.MAX, rhs)
  if yes.key == lhs.key and no.key == rhs.key: return _native_min(lhs, rhs)
  return None

def _unwrap_condition(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype.scalar() in (dtypes.bool, dtypes.half, dtypes.float): u = u.src[0]
  return u

def _positive_mask(u:UOp) -> UOp:
  return UOp(Ops.MAX, dtypes.half, src=(u, u), arg=_NATIVE_POSITIVE_MASK)

def _mask_mul(lhs:UOp, rhs:UOp) -> UOp:
  return UOp(Ops.MUL, dtypes.half, src=(lhs, rhs), arg=_NATIVE_MASK_MUL)

def _finite_positive_mask(u:UOp) -> UOp:
  """Map finite binary16 values to `u > 0` without the stateful DPU compare path."""
  magnitude = u.alu(Ops.MAX, UOp.const(0.0, dtypes.half))
  for _ in range(2): magnitude = magnitude.alu(Ops.MUL, UOp.const(256.0, dtypes.half))
  return _native_min(magnitude, UOp.const(1.0, dtypes.half))

def _mask_expr(u:UOp) -> UOp|None:
  """Build an FP16 0/1 predicate using DPU positive-mask stages."""
  u = _unwrap_condition(u)
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    lhs, rhs = (_unwrap_condition(x) for x in u.src)
    for value, other in ((lhs, rhs), (rhs, lhs)):
      if value.op is Ops.CONST and value.dtype.scalar() is dtypes.bool:
        mask = _mask_expr(other)
        if mask is None: return None
        return UOp.const(1.0, dtypes.half).alu(Ops.SUB, mask) if bool(value.arg) else mask
    if lhs.dtype.scalar() not in (dtypes.half, dtypes.float) or rhs.dtype.scalar() not in (dtypes.half, dtypes.float): return None
    lhs, rhs = lhs.cast(dtypes.half), rhs.cast(dtypes.half)
    positive = _positive_mask(rhs.alu(Ops.SUB, lhs))
    if u.op is Ops.CMPLT: return positive
    return positive.alu(Ops.MAX, _positive_mask(lhs.alu(Ops.SUB, rhs)))
  if u.op in (Ops.OR, Ops.AND):
    mask_lhs, mask_rhs = (_mask_expr(x) for x in u.src)
    if mask_lhs is None or mask_rhs is None: return None
    return mask_lhs.alu(Ops.MAX, mask_rhs) if u.op is Ops.OR else _mask_mul(mask_lhs, mask_rhs)
  return None

def _fold_threshold_where(x:UOp) -> UOp|None:
  """Select a compared value or finite constant without multiplying an inactive nonfinite value."""
  gate = _unwrap_condition(x.src[0])
  if gate.op is not Ops.CMPLT or gate.src[1].op is not Ops.CONST or not math.isfinite(float(gate.src[1].arg)): return None
  lhs, threshold = gate.src[0], float(gate.src[1].arg)
  yes, no = (_unwrap_condition(u) for u in x.src[1:])
  if (mask:=_mask_expr(x.src[0])) is None: return None
  inverse = UOp.const(1.0, dtypes.half).alu(Ops.SUB, mask)
  if yes.key == lhs.key and no.op is Ops.CONST and math.isfinite(float(no.arg)) and float(no.arg) != threshold:
    return _native_min(lhs.cast(dtypes.half), UOp.const(threshold, dtypes.half)).alu(
      Ops.ADD, _mask_mul(inverse, UOp.const(float(no.arg)-threshold, dtypes.half)))
  if no.key == lhs.key and yes.op is Ops.CONST and math.isfinite(float(yes.arg)) and float(yes.arg) != threshold:
    return lhs.cast(dtypes.half).alu(Ops.MAX, UOp.const(threshold, dtypes.half)).alu(
      Ops.ADD, _mask_mul(mask, UOp.const(float(yes.arg)-threshold, dtypes.half)))
  return None

def _fold_relu_cap(x:UOp) -> UOp|None:
  """Recognize relu(source)-relu(source-cap), the canonical ReLU6/clamp expansion."""
  def relu(u:UOp) -> UOp|None:
    if (source:=_relu_operand(u)) is not None: return source
    return _relu_operand(folded) if u.op is Ops.WHERE and (folded:=_fold_ordered_where(u)) is not None else None
  def shifted(u:UOp) -> tuple[UOp, float]:
    return (term[0], float(term[1].arg)) if (term:=_const_operand(u, Ops.ADD)) is not None else (u, 0.0)
  for positive, negative in (x.src, x.src[::-1]):
    source, scaled = relu(positive), _const_operand(negative, Ops.MUL, -1.0)
    if source is None or scaled is None or (upper:=relu(scaled[0])) is None: continue
    source_base, source_shift = shifted(source)
    upper_base, upper_shift = shifted(upper)
    if source_base.key != upper_base.key or (cap:=source_shift-upper_shift) < 0.0: continue
    return _native_min(positive, UOp.const(cap, dtypes.half))
  return None

def _fold_abs(x:UOp) -> UOp|None:
  """Recognize tinygrad's signed-zero-aware ABS graph and select native DPU EW ABS."""
  for value, sign in (x.src, x.src[::-1]):
    if sign.op is not Ops.WHERE: continue
    nonzero, signed, zero = sign.src
    if (nonzero.op is not Ops.CMPNE or nonzero.src[0].key != value.key or nonzero.src[1].op is not Ops.CONST or
        float(nonzero.src[1].arg) != 0.0 or zero.op is not Ops.CONST or float(zero.arg) != 0.0 or signed.op is not Ops.WHERE): continue
    negative, minus_one, plus_one = signed.src
    if (negative.op is Ops.CMPLT and negative.src[0].key == value.key and negative.src[1].op is Ops.CONST and
        float(negative.src[1].arg) == 0.0 and minus_one.op is Ops.CONST and float(minus_one.arg) == -1.0 and
        plus_one.op is Ops.CONST and float(plus_one.arg) == 1.0):
      return UOp(Ops.MAX, x.dtype, src=(value, value), arg=_NATIVE_ABS)
  return None

def _fold_where_abs(x:UOp) -> UOp|None:
  """Recognize `WHERE(x < 0, -x, x)` before an unselected infinity can contaminate a mask blend."""
  if x.op is not Ops.WHERE or len(x.src) != 3 or x.dtype.scalar() is not dtypes.half: return None
  condition = _strip_cast(x.src[0])
  negative = _strip_cast(x.src[1])
  source = condition.src[0] if condition.op is Ops.CMPLT else None
  negated = source is not None and negative.op is Ops.NEG and len(negative.src) == 1 and negative.src[0].key == source.key
  if source is not None and (scaled:=_const_operand(negative, Ops.MUL, -1.0)) is not None:
    negated |= scaled[0].key == source.key
  if (source is not None and source.op is Ops.FDIV and negative.op is Ops.FDIV and
      source.src[1].key == negative.src[1].key and source.src[0].op is Ops.CONST and negative.src[0].op is Ops.CONST):
    negated |= float(source.src[0].arg) == -float(negative.src[0].arg)
  if (condition.op is not Ops.CMPLT or condition.src[1].op is not Ops.CONST or float(condition.src[1].arg) != 0.0 or
      x.src[2].key != condition.src[0].key or not negated): return None
  return UOp(Ops.MAX, x.dtype, src=(condition.src[0], condition.src[0]), arg=_NATIVE_ABS)

def _fold_trunc(x:UOp) -> UOp:
  """Compose truncation from native floor/ceil without mask multiplication on infinities."""
  source, zero = x.src[0], UOp.const(0.0, dtypes.half)
  positive = source.alu(Ops.MAX, zero)
  negative = zero.alu(Ops.SUB, zero.alu(Ops.SUB, source).alu(Ops.MAX, zero))
  floor = UOp(Ops.MAX, x.dtype, src=(positive, positive), arg=_NATIVE_FLOOR)
  ceil = UOp(Ops.MAX, x.dtype, src=(negative, negative), arg=_NATIVE_CEIL)
  return floor.alu(Ops.ADD, ceil)

def _fold_minimum(x:UOp) -> UOp|None:
  """Recognize -max(-x,-y); native ALU-MIN mishandles infinities, so lowering expands it through SUB and MAX."""
  outer = _const_operand(x, Ops.MUL, -1.0)
  if outer is None or outer[0].op is not Ops.MAX: return None
  operands = [_const_operand(u, Ops.MUL, -1.0) for u in outer[0].src]
  if len(operands) != 2 or any(u is None for u in operands): return None
  lhs, rhs = (u for u in operands if u is not None)
  return _native_min(lhs[0], rhs[0])

def _fold_masked_load(gate:UOp, load:UOp, default:UOp, allow_additional_gate_loads:bool=False) -> UOp|None:
  if len(load.src) <= 2 or load.src[1].op is not Ops.CONST: return None
  load_gate = load.src[2]
  if not allow_additional_gate_loads and \
     {u.key for u in gate.toposort() if u.op is Ops.LOAD} - {u.key for u in load_gate.toposort() if u.op is Ops.LOAD}: return None
  same_default = float(load.src[1].arg) == float(default.arg)
  # If the outer condition implies the LOAD condition, the inner default is unreachable. This is the padded-pool form.
  outer_implies_inner = _same_condition(gate, load_gate) or (gate.op is Ops.AND and any(_same_condition(x, load_gate) for x in gate.src))
  if not same_default and not outer_implies_inner: return None
  return load.replace(src=(load.src[0], default, gate.alu(Ops.AND, load_gate) if same_default else gate))

_pm_masked_materialization = PatternMatcher([
  (UPat(Ops.WHERE, src=(UPat.var("gate"), UPat(Ops.LOAD, name="load"), UPat.cvar("default"))),
   lambda gate,load,default: _fold_masked_load(gate, load, default)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.var("val"), UPat.cvar("default"))),
   lambda gate,val,default: _fold_masked_max(gate, default, val, False)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.cvar("default"), UPat.var("val"))),
   lambda gate,default,val: _fold_masked_max(gate, default, val, True)),
])
_pm_masked_loads = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_masked_mul)]) + _pm_masked_materialization

def _fold_masked_max(gate:UOp, default:UOp, val:UOp, opposite:bool) -> UOp|None:
  if val.op is Ops.MAX:
    lhs = _fold_masked_max(gate, default, val.src[0], opposite)
    rhs = _fold_masked_max(gate, default, val.src[1], opposite)
    return None if lhs is None or rhs is None else val.replace(src=(lhs, rhs))
  if val.op is Ops.MUL:
    for source,factor in (val.src, val.src[::-1]):
      if factor.op is not Ops.CONST or not math.isfinite(scale:=float(factor.arg)) or scale == 0.0 or math.isnan(float(default.arg)):
        continue
      folded = _fold_masked_max(gate, default.const_like(float(default.arg)/scale), source, opposite)
      if folded is not None: return val.replace(src=(folded, factor))
  if val.op is not Ops.LOAD or len(val.src) <= 2 or val.src[1].op is not Ops.CONST: return None
  def matches(condition:UOp) -> bool: return _opposite_condition(gate, condition) if opposite else _same_condition(gate, condition)
  condition_matches = matches(val.src[2]) or (val.src[2].op is Ops.AND and any(matches(x) for x in val.src[2].src))
  if condition_matches:
    return val.replace(src=(val.src[0], default, val.src[2]))
  return None

def _replace_infinite_multiply(x:UOp) -> UOp|None:
  """DPU MUL maps finite infinity products to NaN; signed finite/zero FDIV has the required result."""
  for value, factor in (x.src, x.src[::-1]):
    if factor.op is not Ops.CONST or not math.isinf(scale:=float(factor.arg)): continue
    if scale < 0: value = value.alu(Ops.MUL, UOp.const(-1.0, dtypes.half))
    return value.alu(Ops.FDIV, UOp.const(0.0, dtypes.half))
  return None

def _preserve_infinite_division_sign(x:UOp) -> UOp|None:
  """RK3588 FDIV ignores the denominator sign for an infinite numerator; rebuild it with finite DPU intermediates."""
  numerator, denominator = x.src
  if numerator.op is not Ops.CONST or not math.isinf(value:=float(numerator.arg)): return None
  unit = UOp.const(-1.0 if value < 0 else 1.0, dtypes.half)
  return unit.alu(Ops.FDIV, denominator).alu(Ops.FDIV, UOp.const(0.0, dtypes.half))

_pm_storage_common = PatternMatcher([
  (UPat((Ops.ADD, Ops.MUL), dtypes.float, name="x"),
   lambda x:None if _is_static_expr(x) else x.src[0].cast(dtypes.half).alu(x.op, x.src[1].cast(dtypes.half))),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
  (UPat(Ops.MUL, dtypes.half, name="x"), _replace_infinite_multiply),
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
])
_pm_generic_storage_precision = PatternMatcher([(UPat(Ops.WHERE, dtypes.float, name="x"),
  lambda x:None if _is_static_expr(x) else
  UOp(Ops.WHERE, dtypes.half, src=(x.src[0], x.src[1].cast(dtypes.half), x.src[2].cast(dtypes.half)), arg=x.arg))]) + _pm_storage_common

def _unit_ratio_source(root:UOp) -> UOp|None:
  """Match Tinygrad's x/sqrt(1+x*x) normalization."""
  candidates = ((root.src[0], root.src[1]),) if root.op is Ops.FDIV else tuple((source, inverse.src[0])
    for source,inverse in (root.src, root.src[::-1]) if inverse.op is Ops.RECIPROCAL and len(inverse.src) == 1)
  for source, denominator in candidates:
    if (denominator.op is not Ops.SQRT or len(denominator.src) != 1 or
        (radicand:=denominator.src[0]).op is not Ops.ADD): continue
    square = next((u for u in radicand.src if u.op is Ops.MUL and len(u.src) == 2 and
                   u.src[0].key == source.key and u.src[1].key == source.key), None)
    unit = next((u for u in radicand.src if u.op is Ops.CONST and float(u.arg) == 1.0), None)
    if square is not None and unit is not None: return source
  return None

def _fold_atan(root:UOp) -> UOp|None:
  """Replace Tinygrad's asin-based atan with a compact range-reduced DPU polynomial."""
  nodes = root.toposort()
  sources:dict[bytes, UOp] = {}
  for u in nodes:
    if u.op in (Ops.MUL, Ops.FDIV) and (source:=_unit_ratio_source(u)) is not None: sources[source.key] = source
  constants = {float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.half, dtypes.float)}
  if (len(sources) != 1 or not any(abs(v-math.pi/2) < 1e-12 for v in constants) or
      not any(abs(v-1.570796305) < 1e-10 for v in constants) or
      not any(abs(v+0.0012624911) < 1e-8 for v in constants)): return None
  source = next(iter(sources.values())).cast(dtypes.half)
  one = UOp.const(1.0, dtypes.half)
  magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS)
  reduced = _native_min(magnitude, one.alu(Ops.FDIV, magnitude))
  angle = reduced.alu(Ops.MUL, UOp.const(math.pi/4, dtypes.half).alu(Ops.ADD,
    one.alu(Ops.SUB, reduced).alu(Ops.MUL, UOp.const(0.2447, dtypes.half).alu(
      Ops.ADD, reduced.alu(Ops.MUL, UOp.const(0.0663, dtypes.half))))))
  large = _finite_positive_mask(magnitude.alu(Ops.SUB, one))
  reflected = UOp.const(math.pi/2, dtypes.half).alu(Ops.SUB, angle)
  selected = angle.alu(Ops.ADD, large.alu(Ops.MUL, reflected.alu(Ops.SUB, angle)))
  sign = source.alu(Ops.FDIV, magnitude.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)))
  return selected.alu(Ops.MUL, sign)

def _hyperbolic_log_source(root:UOp) -> tuple[UOp, int]|None:
  """Match log(x + sqrt(x*x +/- 1)) after natural log expands to LOG2 times ln(2)."""
  if root.op is not Ops.MUL: return None
  for logarithm, scale in (root.src, root.src[::-1]):
    if (scale.op is not Ops.CONST or abs(float(scale.arg)-math.log(2)) > 1e-12 or logarithm.op is not Ops.LOG2 or
        len(logarithm.src) != 1 or (argument:=logarithm.src[0]).op is not Ops.ADD): continue
    for source, radical in (argument.src, argument.src[::-1]):
      if radical.op is not Ops.SQRT or len(radical.src) != 1 or (radicand:=radical.src[0]).op is not Ops.ADD: continue
      square = next((u for u in radicand.src if u.op is Ops.MUL and len(u.src) == 2 and
                     u.src[0].key == source.key and u.src[1].key == source.key), None)
      offset = next((u for u in radicand.src if u.op is Ops.CONST and float(u.arg) in (-1.0, 1.0)), None)
      if square is not None and offset is not None: return source, int(float(offset.arg))
  return None

def _poly_horner(source:UOp, coefficients:tuple[float, ...]) -> UOp:
  result = UOp.const(coefficients[-1], dtypes.half)
  for coefficient in reversed(coefficients[:-1]):
    result = result.alu(Ops.MUL, source).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  return result

def _hyperbolic_tail(magnitude:UOp, coefficients:tuple[float, ...]) -> UOp:
  """Approximate log(2*x) plus an inverse-even-power correction for large positive x."""
  one, ln2 = UOp.const(1.0, dtypes.half), UOp.const(math.log(2), dtypes.half)
  inverse = one.alu(Ops.FDIV, magnitude); inverse_square = inverse.alu(Ops.MUL, inverse)
  correction = inverse_square.alu(Ops.MUL, _poly_horner(inverse_square, coefficients))
  return magnitude.alu(Ops.LOG2).alu(Ops.MUL, ln2).alu(Ops.ADD, ln2).alu(Ops.ADD, correction)

def _fold_inverse_hyperbolic(root:UOp) -> UOp|None:
  """Stabilize Tinygrad's FP16 asinh/acosh expansions without LUT or CMAC."""
  if (matched:=_hyperbolic_log_source(root)) is None: return None
  source, offset = matched; source = source.cast(dtypes.half)
  one, ln2 = UOp.const(1.0, dtypes.half), UOp.const(math.log(2), dtypes.half)
  if offset == 1:
    magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS)
    bounded = _native_min(magnitude, UOp.const(1.5, dtypes.half))
    square = bounded.alu(Ops.MUL, bounded)
    small = bounded.alu(Ops.MUL, _poly_horner(square,
      (0.99989513, -0.16376462, 0.06135906, -0.01879756, 0.00268578)))
    safe = magnitude.alu(Ops.MAX, UOp.const(1.5, dtypes.half))
    large = _hyperbolic_tail(safe, (0.25, -3/32, 5/96))
    gate = _finite_positive_mask(magnitude.alu(Ops.SUB, UOp.const(1.5, dtypes.half)))
    selected = small.alu(Ops.ADD, gate.alu(Ops.MUL, large.alu(Ops.SUB, small)))
    sign = source.alu(Ops.FDIV, magnitude.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)))
    return selected.alu(Ops.MUL, sign)
  bounded = _native_min(source, UOp.const(2.0, dtypes.half)).alu(Ops.MAX, UOp.const(-2.0, dtypes.half))
  square = bounded.alu(Ops.MUL, bounded)
  small = bounded.alu(Ops.ADD, square.alu(Ops.SUB, one).sqrt()).alu(Ops.LOG2).alu(Ops.MUL, ln2)
  safe = source.alu(Ops.MAX, UOp.const(2.0, dtypes.half))
  large = _hyperbolic_tail(safe, (-0.25, -3/32, -5/96))
  gate = _finite_positive_mask(source.alu(Ops.SUB, UOp.const(2.0, dtypes.half)))
  return small.alu(Ops.ADD, gate.alu(Ops.MUL, large.alu(Ops.SUB, small)))

def _dpu_sqrt(source:UOp, reciprocal:bool=False) -> UOp|None:
  """Approximate FP16 sqrt/rsqrt with range-independent Babylonian iterations on DPU EW."""
  if any(_local_load(u) is not None for u in source.toposort()): return None
  source = source.cast(dtypes.half)
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  negative = _positive_mask(zero.alu(Ops.SUB, source))
  finite = _native_min(source.alu(Ops.MAX, zero), UOp.const(65504.0, dtypes.half))
  safe = finite.alu(Ops.MAX, UOp.const(2**-24, dtypes.half))
  estimate = safe.alu(Ops.MAX, one)
  for _ in range(14): estimate = estimate.alu(Ops.ADD, safe.alu(Ops.FDIV, estimate)).alu(Ops.MUL, UOp.const(0.5, dtypes.half))
  valid = one.alu(Ops.SUB, negative)
  invalid_factor = valid.alu(Ops.FDIV, valid)
  result = estimate.alu(Ops.FDIV, source) if reciprocal else source.alu(Ops.FDIV, estimate)
  return result.alu(Ops.ADD, invalid_factor.alu(Ops.SUB, one))

def _native_floor(source:UOp) -> UOp:
  return UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_FLOOR)

def _dpu_periodic_reduce(source:UOp, reciprocal_period:float, split:tuple[float, ...], half_period:float) -> tuple[UOp, UOp, UOp]:
  """Reduce a finite FP16 angle with split constants so large products do not erase the residual."""
  one = UOp.const(1.0, dtypes.half)
  bounded = _native_min(source.cast(dtypes.half).alu(Ops.MAX, UOp.const(-10000.0, dtypes.half)),
                        UOp.const(10000.0, dtypes.half))
  quotient = bounded.alu(Ops.MUL, UOp.const(reciprocal_period, dtypes.half))
  magnitude = UOp(Ops.MAX, dtypes.half, src=(quotient, quotient), arg=_NATIVE_ABS)
  multiple = _native_floor(magnitude.alu(Ops.ADD, UOp.const(0.5, dtypes.half))).alu(
    Ops.MUL, _positive_mask(quotient).alu(Ops.MUL, UOp.const(2.0, dtypes.half)).alu(Ops.SUB, one))
  reduced = bounded
  for coefficient in split: reduced = reduced.alu(Ops.SUB, multiple.alu(Ops.MUL, UOp.const(coefficient, dtypes.half)))
  # The rounded FP16 quotient can be a few periods off at large magnitudes. Normalize the small residual instead.
  for _ in range(3):
    correction = _positive_mask(reduced.alu(Ops.SUB, UOp.const(half_period, dtypes.half))).alu(
      Ops.SUB, _positive_mask(UOp.const(-half_period, dtypes.half).alu(Ops.SUB, reduced)))
    multiple = multiple.alu(Ops.ADD, correction)
    for coefficient in split: reduced = reduced.alu(Ops.SUB, correction.alu(Ops.MUL, UOp.const(coefficient, dtypes.half)))
  return bounded, multiple, reduced

def _dpu_periodic_reduce_parts(source:UOp, reciprocal_period:float, split:tuple[float, ...], half_period:float) -> tuple[UOp, UOp]:
  """Return a periodic reduction as an FP16 high lane plus its arithmetic residual."""
  bounded, multiple, _ = _dpu_periodic_reduce(source, reciprocal_period, split, half_period)
  return _precise_add_parts([bounded, *(multiple.alu(Ops.MUL, UOp.const(-coefficient, dtypes.half)) for coefficient in split)])

def _dpu_reflected_angle(reduced:UOp, one:UOp) -> tuple[UOp, UOp, UOp]:
  magnitude = UOp(Ops.MAX, dtypes.half, src=(reduced, reduced), arg=_NATIVE_ABS)
  reflected = _positive_mask(magnitude.alu(Ops.SUB, UOp.const(math.pi/2, dtypes.half)))
  pi_minus = UOp.const(3.0, dtypes.half).alu(Ops.SUB, magnitude).alu(Ops.ADD, UOp.const(0.140625, dtypes.half)).alu(
    Ops.ADD, UOp.const(math.pi-3.140625, dtypes.half))
  angle = _mask_mul(magnitude, one.alu(Ops.SUB, reflected)).alu(Ops.ADD, _mask_mul(pi_minus, reflected))
  return angle, angle.alu(Ops.MUL, angle), reflected

def _dpu_sin(source:UOp) -> UOp:
  """Approximate FP16 SIN without LUTs using Cody-Waite reduction and an odd polynomial."""
  one = UOp.const(1.0, dtypes.half)
  period_split = (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125)
  if source.dtype.scalar() is dtypes.float:
    terms:list[UOp] = []
    residuals:list[UOp] = []
    for u in _flatten_binary(source, Ops.ADD):
      if u.op is Ops.CONST:
        high = struct.unpack("<e", struct.pack("<e", float(u.arg)))[0]
        terms.append(UOp.const(high, dtypes.half))
        if (low:=float(u.arg)-high) != 0.0: residuals.append(UOp.const(low, dtypes.half))
      else: terms.append(_fp32_expr_to_half(u))
    reduced_parts = [_dpu_periodic_reduce_parts(term, 1/(2*math.pi), period_split, math.pi) for term in terms]
    reduced_terms, extra = zip(*reduced_parts); residuals.extend(extra)
    reduced, addition_residual = _precise_sum_parts(list(reduced_terms)); residuals.append(addition_residual)
    for _ in range(3):
      correction = _positive_mask(reduced.alu(Ops.SUB, UOp.const(math.pi, dtypes.half))).alu(
        Ops.SUB, _positive_mask(UOp.const(-math.pi, dtypes.half).alu(Ops.SUB, reduced)))
      reduced, normalization_residual = _precise_add_parts(
        [reduced, *(correction.alu(Ops.MUL, UOp.const(-coefficient, dtypes.half)) for coefficient in period_split)])
      residuals.append(normalization_residual)
    invalid = terms[0].alu(Ops.MUL, UOp.const(0.0, dtypes.half))
    for term in terms[1:]: invalid = invalid.alu(Ops.ADD, term.alu(Ops.MUL, UOp.const(0.0, dtypes.half)))
  else:
    source = source.cast(dtypes.half)
    _, _, reduced = _dpu_periodic_reduce(source, 1/(2*math.pi), period_split, math.pi)
    invalid = source.alu(Ops.MUL, UOp.const(0.0, dtypes.half))
  angle, square, reflected = _dpu_reflected_angle(reduced, one)
  polynomial = _poly_horner(square, (1.0, -1/6, 1/120, -1/5040, 1/362880))
  sign = one.alu(Ops.SUB, _positive_mask(UOp.const(0.0, dtypes.half).alu(Ops.SUB, reduced)).alu(
    Ops.MUL, UOp.const(2.0, dtypes.half)))
  result = angle.alu(Ops.MUL, polynomial).alu(Ops.MUL, sign)
  if source.dtype.scalar() is dtypes.float and residuals:
    residual = residuals[0]
    for term in residuals[1:]: residual = residual.alu(Ops.ADD, term)
    cosine = _poly_horner(square, (1.0, -1/2, 1/24, -1/720, 1/40320)).alu(
      Ops.MUL, one.alu(Ops.SUB, reflected.alu(Ops.MUL, UOp.const(2.0, dtypes.half))))
    result = result.alu(Ops.ADD, residual.alu(Ops.MUL, cosine))
  return result.alu(Ops.ADD, invalid)

def _dpu_pow2_integer(exponent:UOp) -> UOp:
  """Build `2**exponent` for the FP16 exponent range with exact native DPU arithmetic."""
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  shifted = _native_min(exponent.alu(Ops.ADD, UOp.const(24.0, dtypes.half)).alu(Ops.MAX, zero), UOp.const(39.0, dtypes.half))
  scale, quotient = UOp.const(2**-24, dtypes.half), shifted
  for factor,repeats in ((2.0, 1), (4.0, 1), (16.0, 1), (256.0, 1), (256.0, 2), (256.0, 4)):
    half = _native_floor(quotient.alu(Ops.MUL, UOp.const(0.5, dtypes.half)))
    bit = quotient.alu(Ops.SUB, half.alu(Ops.MUL, UOp.const(2.0, dtypes.half)))
    multiplier = one.alu(Ops.ADD, bit.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
    for _ in range(repeats): scale = scale.alu(Ops.MUL, multiplier)
    quotient = half
  return scale

def _dpu_exp2_bounded(bounded:UOp) -> UOp:
  integer = _native_floor(bounded)
  return _poly_horner(bounded.alu(Ops.SUB, integer),
    (1.0, 0.6931471806, 0.2402265069, 0.0555041087, 0.0096181291, 0.0013333558)).alu(Ops.MUL, _dpu_pow2_integer(integer))

def _dpu_exp2(source:UOp) -> UOp:
  """Approximate FP16 EXP2 without LUTs using native FLOOR, Horner arithmetic, and exact exponent scaling."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  one = UOp.const(1.0, dtypes.half)
  bounded = _native_min(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(15.9921875, dtypes.half))
  result = _dpu_exp2_bounded(bounded)
  below = mask_fn(UOp.const(-24.0, dtypes.half).alu(Ops.SUB, source))
  above = mask_fn(source.alu(Ops.SUB, UOp.const(15.9921875, dtypes.half)))
  finite = _mask_mul(result, one.alu(Ops.SUB, below))
  return finite.alu(Ops.ADD, one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one))

def _dpu_log2(source:UOp) -> UOp:
  """Approximate FP16 LOG2 without LUTs using threshold exponent extraction and an atanh polynomial."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  mantissa = _native_min(source.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)), UOp.const(65504.0, dtypes.half))
  exponent = zero
  def rescale(pairs:tuple[tuple[float, float], ...], divide:bool) -> None:
    nonlocal mantissa, exponent
    for factor,shift in pairs:
      threshold = struct.unpack("<e", struct.pack("<H", _fp16_bits(factor)-1))[0] if divide else 2.0/factor
      mask = _finite_positive_mask((mantissa if divide else UOp.const(threshold, dtypes.half)).alu(
        Ops.SUB, UOp.const(threshold, dtypes.half) if divide else mantissa))
      scale = one.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
      mantissa = mantissa.alu(Ops.FDIV if divide else Ops.MUL, scale)
      exponent = exponent.alu(Ops.ADD if divide else Ops.SUB, mask.alu(Ops.MUL, UOp.const(shift, dtypes.half)))
  rescale(((256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0)), True)
  rescale(((256.0, 8.0), (256.0, 8.0), (256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0)), False)
  z = mantissa.alu(Ops.SUB, one).alu(Ops.FDIV, mantissa.alu(Ops.ADD, one))
  result = exponent.alu(Ops.ADD, z.alu(Ops.MUL, _poly_horner(z.alu(Ops.MUL, z), (1.0, 1/3, 1/5, 1/7, 1/9))).alu(
    Ops.MUL, UOp.const(2.0/math.log(2.0), dtypes.half)))
  nonzero = mask_fn(source).alu(Ops.MAX, mask_fn(zero.alu(Ops.SUB, source)))
  zero_correction = UOp.const(-1.0, dtypes.half).alu(Ops.FDIV, nonzero).alu(Ops.ADD, one)
  valid = one.alu(Ops.SUB, mask_fn(zero.alu(Ops.SUB, source)))
  negative_correction = valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one)
  above = mask_fn(source.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
  inf_correction = one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one)
  return result.alu(Ops.ADD, zero_correction).alu(Ops.ADD, negative_correction).alu(Ops.ADD, inf_correction)

_pm_fp16_rewrite = PatternMatcher([(UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.bool, name="predicate"),)),
  lambda predicate:_fp16_nonzero_mask(predicate))])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  return list(graph_rewrite(next(u for u in uops if u.op is Ops.SINK), _pm_fp16_rewrite, name="rockchip fp16 recipes").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4, shared_max = True, False, False, 16
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None,
                 Ops.FDIV: lambda: None, Ops.SQRT: lambda: None, Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_uop_program(uops)
    if image is None: image = _lower_uop_program(_fp16_rewrite(uops), recipes_ready=True)
    if image is None: raise RuntimeError("RKPLAN_REJECT:generic_uops " + repr([(i, u.op.name, str(u.dtype)) for i,u in enumerate(uops)]))
    return base64.b64encode(encode_image(_hoist_leading_vector_materialization(image))).decode()
