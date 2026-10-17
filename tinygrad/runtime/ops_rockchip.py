# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, sys, ctypes, os, mmap, struct, math

from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface
from dataclasses import replace
from tinygrad.dtype import bitcast, DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
from tinygrad.helpers import all_same, getenv, Target, IMAGE, is_image_shape, to_mv, mv_address, round_up
from tinygrad.device import HostAllocator, BufferStorage, BufferSpec, Compiled, Compiler, Program, TinyELF
from tinygrad.renderer import tc
from tinygrad.uop.ops import Ops, UOp, GroupOp, PatternMatcher, UPat, graph_rewrite
from tinygrad.renderer import Renderer
from tinygrad.codegen.decomp.transcendental import xexp2, xpow, ilogb2k, ldexp3k

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  if (w:=m.nbytes // len(m)) >= dtype.itemsize: return from_storage_scalar(m[i], dtype)
  return sum(m[i+k] << (8*w*k) for k in range(dtype.itemsize // w)) # a bitcast can read wider than the buffer, _store splits it the same way

def _step(m, dtype: DType): return max(1, dtype.itemsize // (m.nbytes // len(m))) # storage elements per lane

def load(inp, j, dtype: DType):
  if len(inp) >= 3: return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) if gate else alt for (m,x),alt,gate in zip(*inp[:3])]
  return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) for m,x in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  if (w:=m.nbytes // len(m)) >= dtype.itemsize: m[i] = to_storage_scalar(v, dtype)
  else:
    for k in range(dtype.itemsize // w): m[i+k] = (v >> 8*w*k) & ((1 << 8*w) - 1)

supported_ops = {Ops.ADD, Ops.MUL, Ops.SUB, Ops.NEG, Ops.FDIV, Ops.MAX, Ops.RECIPROCAL, Ops.SHL, Ops.SHR,
                 Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.WHERE, Ops.TRUNC, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN}

ew_alu_algo = {"MAX": 0, "MIN": 1, "ADD": 2, "DIV": 3, "SUB": 4, "ABS": 5, "NEG": 6, "FLOOR": 7, "CEIL": 8}
op_to_ew = {Ops.ADD: "ADD", Ops.SUB: "SUB", Ops.MAX: "MAX", Ops.FDIV: "DIV", Ops.NEG: "NEG"}

def fp16(value:float) -> int: return bitcast(value, dtypes.half, dtypes.uint16)

class RockchipProgram(Program['RockchipDevice']):
  def run_wmma(self, arg, inp, warp_size:int, dtype:DType) -> list:
    if arg[:3] != ((1, 1, 8), dtypes.half, 1) or dtype not in (dtypes.half, dtypes.float):
      raise NotImplementedError(f"ROCKCHIP WMMA does not support {arg} with {dtype}")
    assert [len(x) for x in inp] == [8, 8, 1]
    result = []
    for lane in range(warp_size):
      # A and B are eight-half dot-product fragments; unused channels/kernels are zero.
      to_mv(self.dev.input_buf, 32)[:] = struct.pack("<16e", *(x[lane] for x in inp[0]), *([0.0]*8))
      to_mv(self.dev.weight_buf, 512)[:] = struct.pack("<16e", *(x[lane] for x in inp[1]), *([0.0]*8)) + bytes(480)
      self.build_wmma_registers(inp[2][0][lane], dtype)
      self.submit(cna=True)
      result.append(struct.unpack_from("<e" if dtype == dtypes.half else "<f", to_mv(self.dev.output_buf, 64))[0])
    return [result]

  def build_wmma_registers(self, accumulator:float, dtype:DType) -> None:
    E = self.EMIT
    # Reuse convolution setup; override the UINT8 converter, tile shape and FP16/FP32 output.
    self.build_conv_uint8_registers(1, self.dev.output_mem.dma_addr)
    size_e = 1 if dtype == dtypes.half else 3
    self.npu_regs += [
      E(rk.CNA, rk.REG_CNA_CONV_CON1,
        (2 << rk.CNA_CONV_CON1_IN_PRECISION__SHIFT) | (2 << rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT)),
      E(rk.CNA, rk.REG_CNA_DATA_SIZE1, (7 << rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT) | 16),
      E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, 512),
      E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
        (1 << rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT) | (1 << rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT) | 16),
      E(rk.CNA, rk.REG_CNA_CBUF_CON1, 1),
      E(rk.CNA, rk.REG_CNA_CVT_CON0, 1 << rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT),
      E(rk.CNA, rk.REG_CNA_CVT_CON1, 1 << rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT),
      E(rk.CNA, rk.REG_CNA_CVT_CON2, 1 << rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT),
      E(rk.CNA, rk.REG_CNA_CVT_CON3, 1 << rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT),
      E(rk.CNA, rk.REG_CNA_CVT_CON4, 1 << rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT),
      E(rk.CNA, rk.REG_CNA_CVT_CON5, 7),
      E(rk.CNA, rk.REG_CNA_DMA_CON1, 4),
      E(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1, 16),
      E(rk.CORE, rk.REG_CORE_MISC_CFG, 2 << rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT),
      E(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1, 15),
      E(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        ((2 if dtype == dtypes.half else 5) << rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT) |
        (2 << rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT) | (2 << rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, (15 << rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT) | 15),
      E(rk.DPU, rk.REG_DPU_BS_CFG, (2 << rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT) |
        (1 << rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BS_ALU_CFG, bitcast(accumulator, dtypes.float, dtypes.uint)),
      E(rk.DPU, rk.REG_DPU_BS_OW_CFG, (size_e << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
        (size_e << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) | (size_e << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) |
        (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, 15),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
        ((1 << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) | 1) if dtype == dtypes.half else 0),
    ]

  def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
    assert (src_dtype, dtype) in ((dtypes.int, dtypes.half), (dtypes.half, dtypes.int),
                                  (dtypes.int, dtypes.int16), (dtypes.int16, dtypes.int))
    if src_dtype == dtypes.half:
      if any(not math.isfinite(x) for x in a):
        raise NotImplementedError("ROCKCHIP FP16 to INT32 CAST requires finite inputs")
      floor = self.run_npu(Ops.CUSTOM, a, arg=("FLOOR", dtypes.half))
      ceil = self.run_npu(Ops.CUSTOM, a, arg=("CEIL", dtypes.half))
      negative = self.run_npu(Ops.NEG, self.run_npu(Ops.MAX, self.run_npu(Ops.NEG, ceil), [0.0]*len(a)))
      a = self.run_npu(Ops.MAX, floor, negative)
    result:list = []
    for start in range(0, len(a), 8):
      lanes = a[start:start+8]
      raw = struct.pack("<8"+storage_fmt_for_dtype(src_dtype), *(lanes+[0]*(8-len(lanes))))
      to_mv(self.dev.input_buf, 128)[:] = raw+bytes(128-len(raw))
      if dtype == dtypes.int16:
        # NPU byte selection keeps each low word, including wrapping outside INT16 range.
        weights = [[int(j == 4*(i//2)+i%2) for j in range(32)] for i in range(16)]
        self.conv_shl_subtask(weights, self.dev.output_mem.dma_addr, scratch_input=True, channels=32)
      else:
        self.build_cast_registers(src_dtype, dtype)
        self.submit()
      raw = bytes(to_mv(self.dev.output_buf, 32))
      # FP16 output uses two four-lane surfaces; remove padding, not numeric bits.
      if dtype == dtypes.half: raw = raw[:8]+raw[16:24]
      result.extend(struct.unpack("<8"+storage_fmt_for_dtype(dtype), raw[:8*dtype.itemsize])[:len(lanes)])
    return result
  def build_cast_registers(self, src_dtype:DType, dtype:DType) -> None:
    E = self.EMIT
    precision = {dtypes.int16: 1, dtypes.half: 2, dtypes.int: 4}[src_dtype]
    output = {dtypes.half: 2, dtypes.int: 4}[dtype]
    self.build_registers(Ops.NEG)
    self.npu_regs += [
      E(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        (output << rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT) |
        (precision << rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT) | (precision << rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT)),
      E(rk.DPU, rk.REG_DPU_EW_CFG,
        (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
        ((dtype == dtypes.half) << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) |
        (1 << rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BS_OW_CFG,
        ((src_dtype.itemsize == 2 and dtype.itemsize == 4) << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
        ((src_dtype.itemsize == 2 and dtype.itemsize == 4) << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) |
        ((src_dtype.itemsize == 2 and dtype.itemsize == 4) << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) |
        (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 1 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, 1 << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
        (1 << rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT) | (1 << rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT) |
        (1 << rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT) | (1 << rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT)),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT) |
        (15 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT) |
        ((src_dtype == dtypes.half) << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT) |
        (1 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG, 1),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_NRDMA_CFG, 1),
    ]
  def run_u32_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
    assert op in (Ops.SHL, Ops.SHR)
    if not b or not all_same(b) or not 0 <= b[0] <= 31:
      raise NotImplementedError("ROCKCHIP 32-bit shift requires one uniform shift count in 0..31")
    fmt = "<I" if dtype == dtypes.uint else "<i"
    return [struct.unpack(fmt, self.conv_shift(op, struct.pack(fmt, x), int(b[0]), signed=dtype == dtypes.int))[0] for x in a]
  def conv_shift(self, op:Ops, raw:bytes, amount:int, signed:bool=False) -> bytes:
    assert op in (Ops.SHL, Ops.SHR)
    left = op is Ops.SHL
    signed = signed and not left
    output_addr = self.dev.output_mem.dma_addr
    # One initial upload; original bytes stay in lanes 0..3 of the scratch input.
    to_mv(self.dev.input_buf, 128)[:] = raw+bytes(128-len(raw))
    residual, displacement = amount%8, amount//8
    if residual == 0 and not signed: # Shifts by 0, 8, 16 or 24 bits: only whole-byte routing, no carry calculation.
      # Whole-byte shift: one convolution selects bytes and inserts zeros.
      self.conv_shl_subtask([[int(j==(i-displacement if left else i+displacement)) for j in range(4)] for i in range(4)], output_addr)
    else:
      if residual:
        k = 8-residual if left else residual
        # Task 1: the NPU writes q copies at scratch offsets 16, 20 and 24.
        qweights = [[2*int(j==i%4) for j in range(4)] for i in range(12)]
        self.conv_shl_subtask(qweights, self.dev.input_mem.dma_addr+16, offset=-1 if k == 1 else 1, shift=k+1)
        # Task 2: the NPU writes unsigned carry at scratch offset 48.
        self.conv_shl_subtask([[2*int(j==i) for j in range(4)] for i in range(4)],
                           self.dev.input_mem.dma_addr+48, offset=256-(2**k-1), shift=k+1, unsigned=True)
      if signed:
        # Read byte 3 and write four copies of its sign, -1 or 0.
        self.conv_shl_subtask([[0, 0, 0, 2] for _ in range(4)], self.dev.input_mem.dma_addr+80, offset=-127, shift=8)
      channels = 96 if signed else 64
      # Task 3: select s, q and the preceding carry, then write INT8 (table steps 3.1–3.3).
      weights = [[0]*channels for _ in range(4)]
      for i in range(4):
        if left:
          # Select the source byte; negative j leaves a zero output row.
          j = i-displacement
          if j < 0: continue
          if residual == 7:
            weights[i][j] = -128
            # Three q copies let INT8 weights express +256 as 127+127+2.
            weights[i][16+j], weights[i][20+j], weights[i][24+j] = 127, 127, 2
          else:
            weights[i][j] = 2**residual
            # Two q copies express -256 as -128-128.
            weights[i][16+j] = weights[i][20+j] = -128
          # Select the preceding byte's carry; byte 0 has none.
          if j > 0: weights[i][48+j-1] = 1
          continue
        # Output byte i gets source byte j and incoming bits from j+1.
        j = i+displacement
        if j >= 4:
          if signed: weights[i][80] = 1
          continue
        if residual == 0:
          weights[i][j] = 1
          continue
        weights[i][48+j] = 1
        if j < 3:
          if residual == 1:
            # +128 cannot be an INT8 weight: use -128*s + (127+127+2)*q.
            weights[i][j+1] = -128
            weights[i][17+j], weights[i][21+j], weights[i][25+j] = 127, 127, 2
          else:
            # Two q copies let INT8 weights supply -256*q.
            weights[i][j+1] = 2**(8-residual)
            weights[i][17+j] = weights[i][21+j] = -128
        elif signed:
          # +128 does not fit an INT8 weight; use two copies of sign.
          if residual == 1: weights[i][80] = weights[i][81] = 64
          else: weights[i][80] = 2**(8-residual)
      self.conv_shl_subtask(weights, output_addr, scratch_input=True, channels=channels)
    # The four output bytes are already in UINT32 order. No packing/rearrangement.
    return bytes(to_mv(self.dev.output_buf, 4))
  def conv_shl_subtask(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0,
                       unsigned:bool=False, scratch_input:bool=False, channels:int=64) -> None:
    E = self.EMIT
    # weights/offset/shift encode the task formula; zero offset/shift leave the sum unchanged before INT8 saturation.
    # Defaults select signed input and the four-byte layout; scratch_input=True selects channels scratch lanes.
    # Scratch means temporary NPU storage: SHL keeps the original bytes plus q/carry from tasks 1/2 for task 3 to read.
    k = channels if scratch_input else 32 # SHL/UINT32 SHR use 64 scratch lanes; signed SHR needs 96 including sign.
    tile = bytearray(32*k)
    # weights[o][i] multiplies input i into output o; pack constants in 32x32-byte tiles.
    for o, row in enumerate(weights):
      # Input tile, output row, input lane; &255 stores the signed weight's byte.
      for i, w in enumerate(row): tile[(i//32)*1024+o*32+i%32] = w & 255
    to_mv(self.dev.weight_buf, len(tile))[:] = tile
    # All three tasks use height × width = 1 × 1; input bytes are channels.
    self.build_conv_uint8_registers(1, out_addr)
    self.npu_regs += [
      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, offset),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) | (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
    ]
    # Task 1/3 read signed bytes; Task 2 keeps UINT8 conversion with offset -128.
    if not unsigned:
      self.npu_regs.append(E(rk.CNA, rk.REG_CNA_CVT_CON0, (1 << rk.CNA_CVT_CON0_CVT_TYPE__SHIFT) | (1 << rk.CNA_CVT_CON0_DATA_SIGN__SHIFT) |
        (scratch_input << rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT)))
      for reg in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4):
        self.npu_regs.append(E(rk.CNA, reg, 1 << rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT))
    # Task 3 reads signed scratch channels instead of the initial 32.
    if scratch_input:
      self.npu_regs += [
        E(rk.CNA, rk.REG_CNA_CONV_CON1, 1 << rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT),
        E(rk.CNA, rk.REG_CNA_DATA_SIZE1, ((k-1) << rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT) | k),
        E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, 32*k), E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1, k),
        E(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1, k), E(rk.CNA, rk.REG_CNA_DMA_CON1, k//16),
        E(rk.CNA, rk.REG_CNA_CBUF_CON1, k//32),
      ]
    self.submit(cna=True)
  def build_conv_uint8_registers(self, rows:int, output_addr:int) -> None:
    E = self.EMIT
    assert 1 <= rows <= 64
    # Ops.MUL supplies integer initialization only; below we select CNA/CORE and bypass EW, with no extra MUL task.
    self.build_registers(Ops.MUL, int16_mode=True, output_addr=output_addr)
    self.npu_regs += [
      # Input shape and weights.
      E(rk.CNA, rk.REG_CNA_CONV_CON1, (1 << rk.CNA_CONV_CON1_NONALIGN_DMA__SHIFT) |
        (1 << rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT) | (11 << rk.CNA_CONV_CON1_ARGB_IN__SHIFT)),
      E(rk.CNA, rk.REG_CNA_CONV_CON2, (rows+1) << rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT),
      E(rk.CNA, rk.REG_CNA_CONV_CON3, (1 << rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT) |
        (1 << rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT)),
      E(rk.CNA, rk.REG_CNA_DATA_SIZE0, (1 << rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT) | rows),
      E(rk.CNA, rk.REG_CNA_DATA_SIZE1, (3 << rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT) | 32),
      E(rk.CNA, rk.REG_CNA_DATA_SIZE2, 1),
      E(rk.CNA, rk.REG_CNA_DATA_SIZE3, rows),
      E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, 1024),
      E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1, 32),
      E(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2, (1 << rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT) |
        (1 << rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT) | 32),
      E(rk.CNA, rk.REG_CNA_CBUF_CON0, (11 << rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT) | 1),
      E(rk.CNA, rk.REG_CNA_CBUF_CON1, 4*rows),
      # UINT8 converter and DMA: scale 1, offset -128.
      E(rk.CNA, rk.REG_CNA_CVT_CON0, 1 << rk.CNA_CVT_CON0_CVT_TYPE__SHIFT), # unsigned, conversion enabled
      E(rk.CNA, rk.REG_CNA_CVT_CON1, (1 << rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT) | (-128 & rk.CNA_CVT_CON1_CVT_OFFSET0__MASK)),
      E(rk.CNA, rk.REG_CNA_CVT_CON2, (1 << rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT) | (-128 & rk.CNA_CVT_CON2_CVT_OFFSET1__MASK)),
      E(rk.CNA, rk.REG_CNA_CVT_CON3, (1 << rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT) | (-128 & rk.CNA_CVT_CON3_CVT_OFFSET2__MASK)),
      E(rk.CNA, rk.REG_CNA_CVT_CON4, (1 << rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT) | (-128 & rk.CNA_CVT_CON4_CVT_OFFSET3__MASK)),
      E(rk.CNA, rk.REG_CNA_CVT_CON5, 0xffffffff),
      E(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR, self.dev.input_mem.dma_addr),
      E(rk.CNA, rk.REG_CNA_DMA_CON0, (15 << rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT) | 15),
      E(rk.CNA, rk.REG_CNA_DMA_CON1, 1),
      E(rk.CNA, rk.REG_CNA_DMA_CON2, 0),
      E(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0, (1 << rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT) | rows),
      E(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1, 32),
      E(rk.CNA, rk.REG_CNA_DCOMP_ADDR0, self.dev.weight_mem.dma_addr),
      # Integer CORE and INT8 output layout.
      E(rk.CORE, rk.REG_CORE_MISC_CFG, 1 << rk.CORE_MISC_CFG_QD_EN__SHIFT),
      E(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0, (rows-1) << rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT),
      E(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1, 31),
      E(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15 << rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
        (2 << rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DATA_FORMAT, 0),
      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 1 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, rows-1),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (1 << rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT) | 1),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, (31 << rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT) | 31),
      E(rk.DPU, rk.REG_DPU_BS_OW_CFG, (1 << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
        (1 << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) | (1 << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT)),
      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, 31),
      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, (rows-1) << rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT),
      E(rk.DPU, rk.REG_DPU_EW_CFG, (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0),
      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, 32),
    ]
  def run_npu(self, op:Ops, a:list, b:list|None=None, arg:tuple[str, DType]|None=None, *, dtype:DType=dtypes.half) -> list:
    if op is Ops.RECIPROCAL:
      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in a):
        raise NotImplementedError("ROCKCHIP NPU RECIPROCAL does not preserve the sign of negative zero or negative infinity")
      return self.run_npu(Ops.FDIV, [1.0] * len(a), a)
    if b is None:
      assert op in GroupOp.Unary | {Ops.CAST, Ops.CUSTOM}
    else:
      assert len(a) == len(b)
    if op is Ops.SHL:
      assert b is not None
      if dtype != dtypes.int16 or not b or not all_same(b) or not 0 <= b[0] <= 14:
        raise NotImplementedError("ROCKCHIP SHL requires INT16 and one uniform shift count in 0..14")
      shift = b[0]
      # Select a constant multiplier instead of using CPU << inside the SHL implementation.
      powers = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
      b = [powers[shift]] * len(a)
    byte_output = False
    if op is Ops.CAST:
      byte_output = dtype in (dtypes.int8, dtypes.bool)
      if byte_output and any(x not in (0.0, 1.0) for x in a):
        raise NotImplementedError("ROCKCHIP FP16 to byte CAST currently requires a 0/1 mask")
      # result = mask * fp16(1.0)
      self.build_registers(Ops.MUL, int16_mode=not byte_output, byte_output=byte_output)
      # Bool-to-half needs the FP16 1.0 bits as an INT16 multiplier; byte output uses this buffer for the second eight input lanes.
      if not byte_output: to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<H", fp16(1.0)) * 8
    elif op is Ops.SHL: self.build_registers(Ops.MUL, int16_mode=True)
    elif dtype == dtypes.int16 and op in (Ops.MUL, Ops.ADD, Ops.SUB): self.build_registers(op, int16_mode=True)
    else: self.build_registers(op, arg=arg)
    result:list = []
    # The captured register sequence processes eight FP16 elements per submission.
    for start in range(0, len(a), 8):
      lanes = a[start:start+8]
      # SHL and bool-to-half CAST pack INT16 lanes (8h); other paths pack FP16 (8e).
      packed = struct.pack("<8h" if dtype == dtypes.int16 or (op is Ops.CAST and not byte_output) else "<8e", *(lanes + [0] * (8-len(lanes))))
      to_mv(self.dev.input_buf, 16)[:] = packed
      if byte_output: to_mv(self.dev.weight_buf, 16)[:] = packed
      if b is not None:
        rhs = b[start:start+8]
        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8h" if dtype == dtypes.int16 else "<8e", *(rhs + [0] * (8-len(rhs))))
      self.submit()
      # The 16-byte output holds 16 bool/INT8 values or 8 INT16/FP16 values; keep only len(lanes).
      fmt = "16?" if dtype == dtypes.bool else "16b" if dtype == dtypes.int8 else "8h" if dtype == dtypes.int16 else "8e"
      result.extend(struct.unpack("<" + fmt, to_mv(self.dev.output_buf, 16))[:len(lanes)])
    return result
  @staticmethod
  def EMIT(target:int, reg:int, value:int) -> int: return ((target + 1) << 48) | ((value & 0xFFFFFFFF) << 16) | reg

  def submit(self, cna:bool=False) -> None:
    E = self.EMIT
    guard_offset = round_up((len(self.npu_regs) + 3)*8, 16)
    assert guard_offset + mmap.PAGESIZE <= self.dev.regcmd_mem.size
    regs = self.npu_regs + [
      E(rk.PC, rk.REG_PC_BASE_ADDRESS, (self.dev.regcmd_mem.dma_addr + guard_offset) & rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK),
      E(rk.PC, rk.REG_PC_REGISTER_AMOUNTS, 0),
      E(0x80, rk.REG_PC_OPERATION_ENABLE,
        rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | (rk.GLOBAL_OPERATION_ENABLE_CNA_OP_EN__MASK |
        rk.GLOBAL_OPERATION_ENABLE_CORE_OP_EN__MASK if cna else rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK)),
    ]
    # copy the registers to the C array regcmd
    ctypes.memset(self.dev.regcmd_buf, 0, self.dev.regcmd_mem.size)
    regcmd = (ctypes.c_uint64 * len(regs)).from_address(self.dev.regcmd_buf)
    regcmd[:] = list(regs)

    # set the regcmd DMA address in a task
    tasks = ctypes.cast(self.dev.task_buf, ctypes.POINTER(rk.struct_rknpu_task))
    tasks[0] = rk.struct_rknpu_task(
      flags=0,
      op_idx=1 if cna else 4,
      enable_mask=0xd if cna else 0x18,
      int_mask=0x300,
      int_clear=0x1ffff,
      int_status=0,
      regcfg_amount=len(regs),
      regcfg_offset=0,
      regcmd_addr=self.dev.regcmd_mem.dma_addr
    )

    # prepare the submut struct
    submit = rk.struct_rknpu_submit(
      flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
      timeout=6000,
      task_start=0,
      task_number=1,
      task_counter=0,
      priority=0,
      task_obj_addr=self.dev.task_mem.obj_addr,
      core_mask=1,
      fence_fd=-1
    )
    submit.subcore_task[0] = rk.struct_rknpu_subcore_task(task_start=0, task_number=1) # assigned to core 0
    submit.subcore_task[1] = rk.struct_rknpu_subcore_task(task_start=1, task_number=0)
    submit.subcore_task[2] = rk.struct_rknpu_subcore_task(task_start=2, task_number=0)

    # submit with DRM_IOCTL_RKNPU_SUBMIT
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl, __payload=submit)
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.dev = dev
    self.supported_ops = supported_ops
    self.uops: list[UOp] = pickle.loads(obj.lib)
    self.tensor_cores = RockchipRenderer(obj.target).tensor_cores
    self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
    self.npu_regs:list[int] = []

  def build_registers(self, op:Ops, int16_mode:bool=False, arg:tuple[str, DType]|None=None, byte_output:bool=False,
                      input_addr:int|None=None, weight_addr:int|None=None, output_addr:int|None=None) -> None:
    E = self.EMIT
    if arg == ("PRELU", dtypes.half): op = Ops.MUL
    if arg == ("RELUX", dtypes.half):
      self.build_registers(op, int16_mode, ("fp16_exponent_shift_minus(16)", dtypes.half), byte_output,
                           input_addr, weight_addr, output_addr)
      self.npu_regs += [
        E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0),
        E(rk.DPU, rk.REG_DPU_BN_CFG,
          (1 << rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) |
          (1 << rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT)),
        E(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
          bitcast(1.0, dtypes.float, dtypes.uint) << rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT),
      ]
      return
    exp_shift = arg == ("fp16_exponent_shift_minus(16)", dtypes.half)
    unary = op in GroupOp.Unary or arg in (("FLOOR", dtypes.half), ("CEIL", dtypes.half))
    # MUL uses the multiplier, so its ALU selector stays 0.
    if op is Ops.CUSTOM:
      assert arg is not None
      alu_algo = ew_alu_algo.get(arg[0], 0)
    else:
      alu_algo = ew_alu_algo[op_to_ew[op]] if op in op_to_ew else 0
    if op not in op_to_ew and op not in (Ops.MUL, Ops.NEG, Ops.CUSTOM):
      raise NotImplementedError(f"ROCKCHIP requires lowering or a dedicated handler for {op}")
    precision = 1 if int16_mode else 2
    self.npu_regs = [
      E(rk.DPU, rk.REG_DPU_S_POINTER, 0xE),
      E(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        (byte_output << rk.DPU_FEATURE_MODE_CFG_COMB_USE__SHIFT) |
        (15 << rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
        (2 << rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT) |
        (1 << rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        ((0 if byte_output else precision) << rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT) |
        (precision << rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT) |
        (precision << rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
      E(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        ((15 if byte_output else 7) << rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT) |
        ((15 if byte_output else 7) << rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BS_CFG,
        ((0 if byte_output else 2) << rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT) |
        (1 << rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT) |
        (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT) if exp_shift or byte_output else
        (1 << rk.DPU_BS_CFG_BS_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT) |
        (1 << rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BN_CFG,
        (2 << rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT) | (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) |
        (1 << rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT) if byte_output else
        (1 << rk.DPU_BN_CFG_BN_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT) |
        (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BS_ALU_CFG, 0),
      E(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        (fp16(1.0) << rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT) if exp_shift or byte_output else 0),
      E(rk.DPU, rk.REG_DPU_BS_OW_CFG, 1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT),
      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, 15 if byte_output else 7),
      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, 0),
      E(rk.DPU, rk.REG_DPU_BN_MUL_CFG, 0),
      E(rk.DPU, rk.REG_DPU_BN_ALU_CFG, 0),
      E(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0),
      E(rk.DPU, rk.REG_DPU_EW_CFG,
        (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) if exp_shift or byte_output else
        (1 << rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT) |
        (2 << rk.DPU_EW_CFG_EDATA_SIZE__SHIFT) |
        ((arg == ("PRELU", dtypes.half)) << rk.DPU_EW_CFG_EW_MUL_PRELU__SHIFT) |
        (alu_algo << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) |
        ((not int16_mode and op in (Ops.MUL, Ops.FDIV)) << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
        ((not unary) << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
        ((op is Ops.MUL) << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT)),
      E(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, (16 if exp_shift else 0) << rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT),
      E(rk.DPU, rk.REG_DPU_SURFACE_ADD,
        (1 if byte_output else 2 if int16_mode or exp_shift else 4) << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
        ((not int16_mode and not byte_output and op is not Ops.FDIV) << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) |
        (1 << rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT)),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xE),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, 0),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 15 if byte_output else 7),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT) |
        (2 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT) |
        ((exp_shift or unary) << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT)),
      E(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, self.dev.output_mem.dma_addr if output_addr is None else output_addr),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, self.dev.input_mem.dma_addr if input_addr is None else input_addr),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        ((3 if byte_output else 0) << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__SHIFT) |
        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT) |
        (15 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT) |
        ((not int16_mode and op is not Ops.FDIV) << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT) |
        (1 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SURF_NOTCH,
        (1 if byte_output else 0) << rk.DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__SHIFT),
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
        (1 if byte_output else 0) << rk.DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__SHIFT),
    ]
    if exp_shift or byte_output:
      self.npu_regs += [
        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG, 0 if byte_output else 1),
        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_NRDMA_CFG, 0 if byte_output else 1),
        E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, (1 if byte_output else 2) << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 2 << rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT),
      ]
    if not unary:
      self.npu_regs.append(E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
                            self.dev.weight_mem.dma_addr if weight_addr is None else weight_addr))
  def copy_device(self, input_addr:int, output_addr:int, size:int) -> None:
    # Copy raw 16-byte atoms as INT16, so FP16 NaN payloads never enter float arithmetic.
    regions = [(mem.dma_addr, mem.size) for addr,mem in self.dev._gpu_bufs
               if addr not in (self.dev.weight_buf, self.dev.task_buf, self.dev.regcmd_buf)]
    if size <= 0 or input_addr % 16 or output_addr % 16 or size % 16: raise ValueError("device copy requires aligned 16-byte atoms")
    if not all(any(base <= addr and addr+size <= base+length for base,length in regions) for addr in (input_addr, output_addr)):
      raise ValueError("device copy exceeds the input/output allocation")
    if input_addr == output_addr: return
    if input_addr < output_addr+size and output_addr < input_addr+size: raise ValueError("overlapping device copy")
    to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8h", *([1] * 8))
    for offset in range(0, size, 16):
      self.build_registers(Ops.MUL, int16_mode=True, input_addr=input_addr+offset, output_addr=output_addr+offset)
      self.submit()

  def device_bitcast(self, bufs:tuple[int, ...], global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> bool:
    # Keep a contiguous BITCAST graph in mapped device storage, including its final STORE.
    arithmetic = (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.NEG)
    if not any(u.op is Ops.BITCAST and u.addrspace is AddrSpace.ALU for u in self.uops): return False
    if any(u.op not in (Ops.PARAM, Ops.CONST, Ops.CAST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.BITCAST, Ops.STORE, Ops.SINK, *arithmetic)
           for u in self.uops): return False
    stores = [u for u in self.uops if u.op is Ops.STORE]
    if len(stores) != 1 or local_size != (1,1,1): return False
    indexed = UPat(Ops.INDEX, src=(UPat(Ops.PARAM), UPat.var("lane")))
    if not UPat(Ops.STORE, src=(indexed, UPat.var("value"))).match(stores[0], {}): return False
    index, value = stores[0].src
    dest_param, lane = index.src
    if value.max_numel() != 1 or value.dtype.itemsize not in (2,4): return False
    if dest_param.dtype.itemsize != value.dtype.itemsize: return False
    count = dest_param.max_numel()
    if global_size != (count,1,1): return False
    position = lane
    while position.op is Ops.CAST: position = position.src[0]
    if not ((position.op is Ops.SPECIAL and position.arg == 'gidx0') or
            (count == 1 and position.op is Ops.CONST and position.arg == 0)): return False
    # Round only into each allocation's padding; never overrun a logical buffer view.
    logical_size = count*value.dtype.itemsize
    if self.dev.allocator.sizes.get(bufs[dest_param.arg.slot]) != logical_size: return False
    size = round_up(logical_size, 16)
    dst = self.dev.dma_address(bufs[dest_param.arg.slot], size)
    if dst is None or dst % 16: return False
    # LOAD is a leaf here: its INDEX describes addressing, not tensor arithmetic.
    program = [u for u in value.toposort(gate=lambda u: u.op is not Ops.INDEX) if u.dtype not in dtypes.weaks]
    if len(program)*mmap.PAGESIZE > self.dev.output_mem.size: return False
    values = {u:self.dev.output_mem.dma_addr+i*mmap.PAGESIZE for i,u in enumerate(program)}
    inputs:dict[UOp, int] = {}
    for i,u in enumerate(program):
      if u.op is Ops.CONST or (u.op is Ops.CAST and u.src[0].op is Ops.CONST):
        if u.dtype.itemsize not in (2,4): return False
        constant = u.dtype.const(u.val)
        to_mv(self.dev.output_buf+i*mmap.PAGESIZE, 16)[:] = struct.pack("<"+storage_fmt_for_dtype(u.dtype), constant)*(16//u.dtype.itemsize)
      elif u.op is Ops.LOAD:
        if not UPat(Ops.LOAD, src=(indexed,)).match(u, {"lane": lane}): return False
        param = u.src[0].src[0]
        if u.dtype.itemsize != value.dtype.itemsize: return False
        if param.max_numel() != count or param.dtype.itemsize != u.dtype.itemsize: return False
        addr = bufs[param.arg.slot]
        device_addr = self.dev.dma_address(addr, size)
        if device_addr is None or device_addr % 16: return False
        inputs[u] = device_addr
      elif u.op is Ops.BITCAST:
        if u.dtype.itemsize != u.src[0].dtype.itemsize: return False
      elif u.op in arithmetic and u.dtype == dtypes.half:
        if len(u.src) != (1 if u.op is Ops.NEG else 2): return False
        if any(s.dtype != dtypes.half for s in u.src): return False
      else: return False
    if not inputs: return False
    if getenv("TRACE"): print("device BITCAST graph:", [u.op.name for u in program], "bytes:", logical_size)
    atom = 16//value.dtype.itemsize
    for start in range(0, count, atom):
      values.update((u,addr+start*u.dtype.itemsize) for u,addr in inputs.items())
      for u in program:
        if u.op in (Ops.LOAD, Ops.CONST, Ops.CAST): continue
        if u.op is Ops.BITCAST: values[u] = values[u.src[0]]
        else:
          self.build_registers(u.op, input_addr=values[u.src[0]], output_addr=values[u],
                               weight_addr=values[u.src[1]] if len(u.src) == 2 else None)
          self.submit()
      self.copy_device(values[value], dst+start*value.dtype.itemsize, 16)
    return True

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    st = time.perf_counter()
    if self.device_bitcast(bufs, global_size, local_size): return time.perf_counter()-st
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[UOp, Any] = {}
      pbufs: list[int] = list(bufs)
      pvals: list[int] = list(vals)
      exec_masks = [[True] * warp_size]
      i = 0
      while i < len(self.uops):
        u = self.uops[i]
        src_values = [values[v] for v in u.src if v.dtype is not dtypes.void]
        src_dtypes = [v.dtype for v in u.src if v.dtype is not dtypes.void]
        if getenv("TRACE"): print(i, u.op, u.dtype, u.arg, src_values, src_dtypes)
        if u.op is Ops.END:
          if len(u.src) == 3:
            # conditional backedge on a loop: jump back while the condition is true
            if values[u.src[2]][0]: i = self.uop_to_index[u.src[1]]
            else: i += 1
          else: i = self.uop_to_index[u.src[1]]
          continue
        if u.op is Ops.IF:
          exec_masks.append([x and y for x,y in zip(exec_masks[-1], src_values[0])])
          i += 1
          continue
        if u.op is Ops.ENDIF:
          exec_masks.pop()
          i += 1
          continue
        if u.op in (Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.CUSTOM_FUNCTION) or (u.op is Ops.RANGE and u.dtype == dtypes.void):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        if u.op is Ops.STORE:
          assert len(src_values) == 2, f"STORE must be lowered to 2 srcs, got {len(src_values)}"
          store_gate = exec_masks[-1]
          for j,val in enumerate(src_values[1] if u.max_numel() > 1 else [src_values[1]]):
            for (m,o),v,g in zip(src_values[0], val, store_gate):
              if g: _store(m, o+j*_step(m, src_dtypes[1]), v, src_dtypes[1])
          i += 1
          continue
        if u.op is Ops.AFTER or (u.op is Ops.BITCAST and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL)): values[u] = src_values[0]
        elif u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU: values[u] = [pvals.pop(0)] * warp_size
        elif u.op in {Ops.PARAM, Ops.BUFFER}:
          storage_fmt = storage_fmt_for_dtype(u.dtype)
          if storage_fmt is None: raise RuntimeError(f"dtype={u.dtype} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if u.addrspace == AddrSpace.REG:
            # REGs are per thread
            values[u] = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            size = u.max_numel() * u.dtype.itemsize
            buf = memoryview(bytearray(size)) if u.op is not Ops.PARAM else to_mv(pbufs.pop(0), size)
            values[u] = [buf.cast(storage_fmt)] * warp_size
        elif u.op is Ops.SPECIAL:
          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])]] * warp_size
          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for x in warp]
        elif u.op is Ops.CONST: values[u] = [u.val] * warp_size
        elif u.op in {Ops.INDEX, Ops.SHRINK}:
          ret:list = []
          if u.src[0].addrspace == AddrSpace.ALU:
            ret = [src_values[0][i][t] for t,i in enumerate(src_values[1])]
          elif is_image_shape(u.src[0]._shape):
            for m,oy,ox in zip(*src_values):
              if ox < 0 or ox >= u.src[0]._shape[1] or oy < 0 or oy >= u.src[0]._shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*u.src[0]._shape[1]*4))
          else:
            scale = u.src[0].dtype.itemsize // u.src[0].src[0].dtype.itemsize if u.src[0].op is Ops.BITCAST else 1
            for m,o in zip(src_values[0], src_values[1]): ret.append((m[0], m[1]+o*scale) if isinstance(m, tuple) else (m, o*scale))
          values[u] = ret
        elif u.op is Ops.RANGE:
          if u not in values: values[u] = [0] * warp_size
          else:
            for j in range(len(values[u])):
              values[u][j] += 1
          if values[u][0] == src_values[0][0]:
            del values[u]
            i = self.loop_ends[u] + 1
            continue
        elif u.op is Ops.STACK: values[u] = src_values
        elif u.op is Ops.BITCAST: values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
        elif u.op is Ops.CAST:
          if (src_dtypes[0], u.dtype) in ((dtypes.int, dtypes.half), (dtypes.half, dtypes.int),
                                          (dtypes.int, dtypes.int16), (dtypes.int16, dtypes.int)):
            values[u] = self.run_cast(src_values[0], src_dtypes[0], u.dtype)
          elif (src_dtypes[0], u.dtype) in ((dtypes.bool, dtypes.half), (dtypes.half, dtypes.int8), (dtypes.half, dtypes.bool)):
            values[u] = self.run_npu(Ops.CAST, src_values[0], dtype=u.dtype)
          else:
            if (src_dtypes[0], u.dtype) == (dtypes.half, dtypes.bool):
              print(f"warning: {u.op} from {src_dtypes[0]} to {u.dtype} is not supported on ROCKCHIP NPU, emulating in python")
            values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
        elif u.op is Ops.LOAD:
          if (load_sz := u.max_numel()) > 1:
            # buf and gate are not vecs
            values[u] = [load([src_values[k] if k in [0,2] else src_values[k][j] \
                               for k in range(len(src_values))], j, u.dtype) for j in range(load_sz)]
          else:
            values[u] = load(src_values, 0, u.dtype)
        elif u.op is Ops.CALL:
          restype = None if u.dtype is dtypes.void else getattr(ctypes, f"c_{'u' if u.dtype in dtypes.uints else ''}int{u.dtype.bitsize}")
          cfunc = ctypes.CFUNCTYPE(restype, *[ctypes.c_uint64] * len(src_values))
          values[u] = []
          for fptr,args,gate in zip(values[u.src[0].src[0]], zip(*src_values), exec_masks[-1]):
            call_args = [(mv_address(x[0]) + x[1]*dt.itemsize) if isinstance(x, tuple) else x for x,dt in zip(args, src_dtypes)]
            values[u].append(cfunc(fptr)(*call_args) if gate else None)
        elif u.op is Ops.WMMA: values[u] = self.run_wmma(u.arg, src_values, warp_size, u.dtype)
        elif u.op is Ops.CUSTOM:
          if u.arg == ("PRELU", dtypes.half) and u.dtype == dtypes.half and src_dtypes == [dtypes.half]*2: pass
          elif u.arg == ("fp16_exponent_shift_minus(16)", dtypes.half) and u.dtype == dtypes.half and src_dtypes == [dtypes.half] * 3:
            # Equal infinities intentionally produce an intermediate NaN; check original inputs only.
            if any(math.isnan(x) for xs in src_values[1:] for x in xs):
              raise NotImplementedError("ROCKCHIP NPU FP16 comparisons do not support NaN inputs")
          # Only these unary CUSTOM modes with one FP16 input and FP16 output reach the NPU.
          elif u.arg in (("RELUX", dtypes.half), ("FLOOR", dtypes.half), ("CEIL", dtypes.half),
                         ("fp16_exponent_shift_minus(16)", dtypes.half)) and \
              u.dtype == dtypes.half and src_dtypes == [dtypes.half]: pass
          else:
            raise NotImplementedError(f"ROCKCHIP NPU does not support CUSTOM {u.arg}")
          values[u] = self.run_npu(Ops.CUSTOM, src_values[0], src_values[1] if u.arg == ("PRELU", dtypes.half) else None, arg=u.arg)
        elif u.op in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {u.op}"
          assert all_same([u.dtype] + src_dtypes) or u.op in {*GroupOp.Comparison, Ops.WHERE, Ops.SHL, Ops.SHR}, f"dtype mismatch on {u.op}"
          allowed_dtypes = {Ops.SHL: (dtypes.int16, dtypes.int, dtypes.uint),
                            Ops.MUL: (dtypes.half, dtypes.int16), Ops.ADD: (dtypes.half, dtypes.int16),
                            Ops.SUB: (dtypes.half, dtypes.int16),
                            Ops.SHR: (dtypes.int16, dtypes.int, dtypes.uint)}.get(u.op, (dtypes.half,))
          if u.op not in self.supported_ops or u.dtype not in allowed_dtypes:
            raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
          if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
            values[u] = self.run_u32_shift(Ops.SHL, src_values[0], src_values[1], u.dtype)
          elif u.op is Ops.SHR and u.dtype in (dtypes.int16, dtypes.int, dtypes.uint):
            values[u] = self.run_u32_shift(Ops.SHR, src_values[0], src_values[1], dtypes.int if u.dtype == dtypes.int16 else u.dtype)
          else: values[u] = self.run_npu(u.op, *src_values, dtype=u.dtype)
        assert u in values, u
        i += 1
    return time.perf_counter() - st

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class RockchipRenderer(Renderer):
  tensor_cores = [tc.TensorCore(dtypes.half, dtype, ((), ("k0", "k1", "k2")),
                  ((), ("k0", "k1", "k2")), ((), ())) for dtype in (dtypes.float, dtypes.half)]

  @staticmethod
  def _pm_sin(x:UOp) -> UOp:
    def minimum(a:UOp, b:UOp) -> UOp: return a.alu(Ops.NEG).maximum(b.alu(Ops.NEG)).alu(Ops.NEG)
    def absolute(a:UOp) -> UOp: return a.maximum(a.alu(Ops.NEG))
    def positive(a:UOp) -> UOp:
      mask = a.maximum(a.const_like(0))
      for _ in range(3): mask = mask.alu(Ops.MUL, a.const_like(256))
      return minimum(mask, a.const_like(1))
    one = x.const_like(1)
    reduced = minimum(x.maximum(x.const_like(-10000)), x.const_like(10000))
    error = x.const_like(0)
    for _ in range(2):
      quotient = reduced.alu(Ops.MUL, x.const_like(1/(2*math.pi)))
      rounded = UOp(Ops.CUSTOM, src=(absolute(quotient).alu(Ops.ADD, x.const_like(0.5)),), arg=("FLOOR", dtypes.half))
      sign = positive(quotient).alu(Ops.MUL, x.const_like(2)).alu(Ops.SUB, one)
      multiple = rounded.alu(Ops.MUL, sign)
      terms = [reduced, error]
      for coefficient in (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125):
        terms.append(multiple.alu(Ops.MUL, x.const_like(-coefficient)))
      reduced, error = RockchipRenderer._pm_sum_parts(terms)
    magnitude = absolute(reduced)
    reflected = positive(magnitude.alu(Ops.SUB, x.const_like(math.pi/2)))
    pi_minus = x.const_like(3).alu(Ops.SUB, magnitude).alu(Ops.ADD, x.const_like(0.140625)).alu(Ops.ADD, x.const_like(math.pi-3.140625))
    angle = magnitude.alu(Ops.MUL, one.alu(Ops.SUB, reflected)).alu(Ops.ADD, pi_minus.alu(Ops.MUL, reflected))
    square = angle.alu(Ops.MUL, angle)
    polynomial = x.const_like(1/362880)
    for coefficient in (-1/5040, 1/120, -1/6, 1): polynomial = polynomial.alu(Ops.MUL, square).alu(Ops.ADD, x.const_like(coefficient))
    sign = one.alu(Ops.SUB, positive(x.const_like(0).alu(Ops.SUB, reduced)).alu(Ops.MUL, x.const_like(2)))
    return angle.alu(Ops.MUL, polynomial).alu(Ops.MUL, sign).alu(Ops.ADD, x.alu(Ops.MUL, x.const_like(0)))

  @staticmethod
  def _pm_two_sum(a:UOp, b:UOp) -> tuple[UOp, UOp]:
    total = a.alu(Ops.ADD, b)
    virtual_b = total.alu(Ops.SUB, a)
    error_a = a.alu(Ops.SUB, total.alu(Ops.SUB, virtual_b))
    return total, error_a.alu(Ops.ADD, b.alu(Ops.SUB, virtual_b))

  @staticmethod
  def _pm_sum_parts(terms:list[UOp]) -> tuple[UOp, UOp]:
    high, middle, low = terms[0], terms[0].const_like(0), terms[0].const_like(0)
    for term in terms[1:]:
      high, error = RockchipRenderer._pm_two_sum(high, term)
      middle, error = RockchipRenderer._pm_two_sum(middle, error)
      low = low.alu(Ops.ADD, error)
    return high, middle.alu(Ops.ADD, low)

  @staticmethod
  def _pm_sqrt(x:UOp) -> UOp:
    safe = x.maximum(x.const_like(2**-24)).alu(Ops.NEG).maximum(x.const_like(-65504)).alu(Ops.NEG)
    k = ilogb2k(safe).alu(Ops.SHR, UOp.const(1, dtypes.int16)).cast(dtypes.half)
    normalized = ldexp3k(safe, k.alu(Ops.MUL, k.const_like(-2)))
    root = ldexp3k(xpow(normalized, normalized.const_like(0.5)), k)
    special = (x < 0).where(x.const_like(math.nan), x)
    result = ((x > 0) & (x < math.inf)).where(root, special)
    return graph_rewrite(result, PatternMatcher([
      # Protect every FP16 selection inside native SQRT, including nested POW branches.
      (UPat(Ops.WHERE, dtypes.half, name="u"),
       lambda u: UOp(Ops.CUSTOM, src=u.src, arg=("FLOAT_SELECT", dtypes.half))),
    ]))

  @staticmethod
  def _pm_cdiv_half(u:UOp) -> UOp:
    a, b = (x.cast(dtypes.half) for x in u.src)
    return RockchipRenderer._pm_lower_trunc(a.alu(Ops.FDIV, b)).cast(u.dtype)

  @staticmethod
  def _pm_cmod_half(u:UOp) -> UOp:
    a, b = (x.cast(dtypes.half) for x in u.src)
    q = RockchipRenderer._pm_lower_trunc(a.alu(Ops.FDIV, b))
    return a.alu(Ops.SUB, q.alu(Ops.MUL, b)).cast(u.dtype)

  @staticmethod
  def _pm_exp2(x:UOp) -> UOp:
    isnan = UOp(Ops.CUSTOM, src=(x,), arg=("EXP2_ISNAN", dtypes.bool))
    safe = UOp(Ops.CUSTOM, src=(isnan, x.const_like(0), x), arg=("FLOAT_SELECT", dtypes.half))
    # Retain tinygrad's decomposition, but keep its selections away from arithmetic WHERE.
    result = graph_rewrite(xexp2(safe), PatternMatcher([
      (UPat(Ops.WHERE, dtypes.half, name="u"),
       lambda u: UOp(Ops.CUSTOM, src=u.src, arg=("FLOAT_SELECT", dtypes.half))),
    ]))
    return UOp(Ops.CUSTOM, src=(isnan, x.const_like(math.nan), result), arg=("FLOAT_SELECT", dtypes.half))

  @staticmethod
  def _pm_float_select(mask:UOp, a:UOp, b:UOp) -> UOp:
    mask = mask.cast(dtypes.half)
    # SUB creates +0 for the unselected arm; NEG would create -0 and multiply NaN.
    yes = UOp(Ops.CUSTOM, src=(mask.const_like(0).alu(Ops.SUB, mask), a), arg=("PRELU", dtypes.half))
    no = UOp(Ops.CUSTOM, src=(mask.alu(Ops.SUB, mask.const_like(1)), b), arg=("PRELU", dtypes.half))
    return yes.alu(Ops.ADD, no).alu(Ops.NEG).alu(Ops.ADD, mask.const_like(0))

  @staticmethod
  def _pm_lower_trunc(x:UOp) -> UOp:
    # trunc(x) = max(floor(x), -max(-ceil(x), 0))
    floor, ceil = (UOp(Ops.CUSTOM, src=(x,), arg=(mode, dtypes.half)) for mode in ("FLOOR", "CEIL"))
    return floor.maximum(ceil.alu(Ops.NEG).maximum(x.const_like(0)).alu(Ops.NEG))

  @staticmethod
  def _pm_lower_where(u:UOp) -> UOp:
    x, a, b = u.src
    a, b = a.cast(dtypes.half), b.cast(dtypes.half)
    mask = x.cast(dtypes.half)
    positive = a.alu(Ops.MUL, mask)
    inverse = b.alu(Ops.MUL, mask.const_like(1).alu(Ops.SUB, mask))
    return positive.alu(Ops.ADD, inverse).cast(u.dtype)

  @staticmethod
  def _pm_lower_cmplt(u:UOp) -> UOp:
    a, b = (x.cast(dtypes.half) for x in u.src)
    delta = b.alu(Ops.ADD, a.alu(Ops.NEG))
    biased = delta.alu(Ops.MUL, delta.const_like(2)).alu(Ops.ADD, delta.const_like(-2**-24))
    product = biased.alu(Ops.MUL, delta.const_like(float("inf")))
    mask = UOp(Ops.CUSTOM, src=(product,), arg=("RELUX", dtypes.half))
    # Hardware ReLU-X maps NaN to 1; clear equal inputs, including equal infinities.
    unequal = RockchipRenderer._pm_lower_compare(a.alu(Ops.CMPNE, b)).cast(dtypes.half)
    return mask.alu(Ops.MUL, unequal).cast(dtypes.bool)

  @staticmethod
  def _pm_log2_polynomial(x:UOp) -> UOp:
    # Bound only the working input; restore domain and infinity results below.
    bounded = x.maximum(x.const_like(2**-24)).alu(Ops.NEG).maximum(x.const_like(-65504)).alu(Ops.NEG)
    denormal = bounded < bounded.const_like(2**-14)
    a = denormal.where(bounded * 1024, bounded)
    exponent = ilogb2k(a * math.sqrt(2)).cast(dtypes.half)
    r = ldexp3k(a, exponent.alu(Ops.NEG)) - 1
    exponent = denormal.where(exponent - 10, exponent)
    coefficients = [(-1)**k / ((k+1)*math.log(2)) for k in range(16)]
    # One FP16 step toward zero removes the remaining error in the all-positive-half simulation.
    coefficients[1] = -0.720703125
    polynomial = r.const_like(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]): polynomial = polynomial * r + coefficient
    result = exponent + r * polynomial
    result = x.ne(0).where(result, x.const_like(-math.inf))
    result = (x < 0).where(x.const_like(math.nan), result)
    return x.ne(math.inf).where(result, x.const_like(math.inf))

  @staticmethod
  def _pm_log2(x:UOp) -> UOp:
    isnan = UOp(Ops.CUSTOM, src=(x,), arg=("LOG2_ISNAN", dtypes.bool))
    safe = UOp(Ops.CUSTOM, src=(isnan, x.const_like(1), x), arg=("FLOAT_SELECT", dtypes.half))
    result = graph_rewrite(RockchipRenderer._pm_log2_polynomial(safe), PatternMatcher([
      (UPat(Ops.WHERE, dtypes.half, name="u"),
       lambda u: UOp(Ops.CUSTOM, src=u.src, arg=("FLOAT_SELECT", dtypes.half))),
    ]))
    return UOp(Ops.CUSTOM, src=(isnan, x.const_like(math.nan), result), arg=("FLOAT_SELECT", dtypes.half))

  @staticmethod
  def _pm_isnan(x:UOp) -> UOp:
    # MUL turns negative NaNs positive before MAX; keep finite inputs in the safe exponent range.
    bounded = x.alu(Ops.MUL, x.const_like(1)).maximum(x.const_like(32768))
    tag = UOp(Ops.CUSTOM, src=(bounded,), arg=("fp16_exponent_shift_minus(16)", dtypes.half))
    return tag.alu(Ops.SUB, tag.const_like(1)).maximum(tag.const_like(0)).alu(Ops.MUL, tag.const_like(1024)).cast(dtypes.bool)

  @staticmethod
  def _pm_lower_compare(u:UOp) -> UOp:
    a, b = (x.cast(dtypes.half) for x in u.src)
    # EW SUB mishandles infinity pairs; ADD with NPU NEG preserves their signs.
    product = a.alu(Ops.ADD, b.alu(Ops.NEG)).alu(Ops.MUL, a.const_like(float("inf")))
    tag = UOp(Ops.CUSTOM, src=(product, a, b), arg=("fp16_exponent_shift_minus(16)", dtypes.half))
    mask = tag.alu(Ops.SUB, tag.const_like(1)).alu(Ops.MUL, tag.const_like(1024)).maximum(tag.const_like(0))
    # CMPEQ keeps the equality mask; CMPNE inverts it.
    return (mask.const_like(1).alu(Ops.SUB, mask) if u.op is Ops.CMPNE else mask).cast(dtypes.bool)

  # Native-op markers, as in LLVM/NIR; RockchipProgram performs the hardware dispatch.
  code_for_op = {op:lambda:None for op in supported_ops}
  extra_matcher = PatternMatcher([
    # Keep native SQRT's formula, but protect its nested selections.
    (UPat(Ops.SQRT, dtypes.half, src=(UPat.var("x", dtypes.half),)),
     lambda x: RockchipRenderer._pm_sqrt(x)),
    # Trial: round the FP16 quotient toward zero, then restore INT32 output.
    (UPat(Ops.CDIV, dtypes.int, name="u"), lambda u: RockchipRenderer._pm_cdiv_half(u)),
    # Trial: FP16 quotient/remainder arithmetic, then restore the integer output dtype.
    (UPat(Ops.CMOD, dtypes.int, name="u"), lambda u: RockchipRenderer._pm_cmod_half(u)),
    # POW can select NaN outside EXP2; this branch already normalizes zero, so reuse its selector.
    (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("m"), UPat.cvar("c").or_casted(),
      UPat(Ops.CUSTOM, arg=("FLOAT_SELECT", dtypes.half), name="value"))),
     lambda m,c,value: UOp(Ops.CUSTOM, src=(m, c.cast(dtypes.half), value), arg=("FLOAT_SELECT", dtypes.half))
     if math.isnan(c.arg) else None),
    # LOG2 uses a division-free reduced polynomial, with NaN-safe input and PReLU selections.
    (UPat(Ops.LOG2, dtypes.half, src=(UPat.var("x", dtypes.half),)), lambda x: RockchipRenderer._pm_log2(x)),
    # Keep native EXP2 arithmetic, replacing only its selections.
    (UPat(Ops.EXP2, dtypes.half, src=(UPat.var("x", dtypes.half),)), lambda x: RockchipRenderer._pm_exp2(x)),
    # Bool OR is MAX of the FP16 0/1 inputs.
    (UPat(Ops.OR, dtypes.bool, name="u"),
     lambda u: u.src[0].cast(dtypes.half).maximum(u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
  ])
  comparison_matcher = PatternMatcher([
    # A scalar result from this dot tile can absorb its outer reduction accumulator into BS ADD.
    (UPat(Ops.ADD, dtypes.float, src=[UPat(Ops.INDEX, src=(UPat(Ops.WMMA, name="w",
      src=(UPat(), UPat(), UPat(Ops.STACK, src=(UPat.const(0).cast(dtypes.float),)))), UPat.var("idx"))), UPat.var("acc")]),
     lambda w,idx,acc: w.replace(src=(w.src[0], w.src[1], UOp.stack(acc))).index(idx)
     if w.arg[:3] == ((1, 1, 8), dtypes.half, 1) else None),
    # Expand SIN late: preserve the FP16 rounding steps in compensated range reduction.
    (UPat(Ops.SIN, dtypes.half, src=(UPat.var("x", dtypes.half),)), lambda x: RockchipRenderer._pm_sin(x)),
    # Bool XOR is inequality of the FP16 0/1 inputs; reuse comparison lowering.
    (UPat(Ops.XOR, dtypes.bool, name="u"),
     lambda u: u.src[0].cast(dtypes.half).ne(u.src[1].cast(dtypes.half))),
    # Bool AND multiplies 0/1 masks; keep the final CAST after general rewrites.
    (UPat(Ops.AND, dtypes.bool, name="u"),
     lambda u: u.src[0].cast(dtypes.half).alu(Ops.MUL, u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
    # LOG2 shares the verified NaN detector and float selector, not ordinary arithmetic WHERE.
    (UPat(Ops.CUSTOM, arg=("LOG2_ISNAN", dtypes.bool), src=(UPat.var("x"),)), lambda x: RockchipRenderer._pm_isnan(x)),
    # Native LOG2 extracts five exponent bits; signed floor division gives the exact low-bit remainder.
    (UPat(Ops.AND, dtypes.int16, src=[UPat.var("x", dtypes.int16), UPat(Ops.CAST, dtypes.int16, src=(UPat.const(31),))]),
     lambda x: x.alu(Ops.SUB, x.alu(Ops.SHR, x.const_like(5)).alu(Ops.MUL, x.const_like(32)))),
    # Expand the detector late so its MUL(x, 1) survives simplification.
    (UPat(Ops.CUSTOM, arg=("EXP2_ISNAN", dtypes.bool), src=(UPat.var("x"),)), lambda x: RockchipRenderer._pm_isnan(x)),
    # x != x detects NaN, including either NaN sign; emit after arithmetic simplification.
    (UPat(Ops.CMPNE, src=(UPat.var("x", dtypes.half), UPat.var("x", dtypes.half))),
     lambda x: RockchipRenderer._pm_isnan(x)),
    # Expand the shared selector late, preserving SUB(0, mask)'s +0 bypass input.
    (UPat(Ops.CUSTOM, arg=("FLOAT_SELECT", dtypes.half), src=(UPat.var("m"), UPat.var("a"), UPat.var("b"))),
     lambda m,a,b: RockchipRenderer._pm_float_select(m, a, b)),
    # Round toward zero using native FP16 FLOOR/CEIL
    (UPat(Ops.TRUNC, dtypes.half, src=(UPat.var("x", dtypes.half),)),
     lambda x: RockchipRenderer._pm_lower_trunc(x)),
    # Arithmetic selection; INT32 casts are lossy, and non-finite values/signed zero need separate handling.
    (UPat(Ops.WHERE, (dtypes.half, dtypes.int32), src=(UPat(dtype=dtypes.bool),
      UPat(dtype=(dtypes.half, dtypes.int32, dtypes.weakint, dtypes.weakfloat)),
      UPat(dtype=(dtypes.half, dtypes.int32, dtypes.weakint, dtypes.weakfloat))), name="u"),
     lambda u: RockchipRenderer._pm_lower_where(u)),
    # Scale before subtracting epsilon so the smallest positive FP16 delta stays positive.
    (UPat(Ops.CMPLT, src=(UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.int32, dtypes.bool)),
                         UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.int32, dtypes.bool))), name="u"),
     lambda u: RockchipRenderer._pm_lower_cmplt(u)),
    # Experimental: narrow integer arithmetic; operands and results must fit INT16.
    (UPat((Ops.MUL, Ops.ADD, Ops.SUB), (dtypes.int32, dtypes.weakint), name="u"),
     lambda u: u.src[0].cast(dtypes.int16).alu(u.op, u.src[1].cast(dtypes.int16)).cast(u.dtype)),
    # Bool inputs convert exactly to FP16 0/1 on the NPU.
    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=dtypes.bool), UPat(dtype=dtypes.bool)), name="u"),
     lambda u: RockchipRenderer._pm_lower_compare(u)),
    # Lossy INT32 comparison: FP16 conversion can make distinct integers equal.
    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=dtypes.int32), UPat(dtype=dtypes.int32)), name="u"),
     lambda u: RockchipRenderer._pm_lower_compare(u)),
    # Lower comparisons after general rewrites so the final mask CAST stays a CAST.
    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=(dtypes.half, dtypes.weakfloat)),
                                     UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
     lambda u: RockchipRenderer._pm_lower_compare(u)),
  ])
  compiler = RockchipCompiler()

  def __init__(self, target:Target):
    assert (emu:=getenv("EMULATE", "")) == "", ("EMULATE is deprecated, use DEV=PYTHON::" +
      {"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_MFMA":"gfx950", "CUDA":"sm_80", "CUDA_SM75":"sm_75", "CUDA_SM89":"sm_89"}.get(emu, emu))
    target = replace(target, renderer="PYTHON")
    if target.arch == "METAL": self.target, self.tensor_cores = replace(target, device="METAL"), tc.metal
    elif target.arch.startswith("gfx"):
      self.target = replace(target, device="AMD")
      self.tensor_cores = tc.get_amd(target.arch)
    elif target.arch.startswith("sm"):
      self.target = replace(target, device="CUDA")
      self.tensor_cores = tc.get_cuda(target.arch)
    elif IMAGE and not target.arch: self.target = replace(target, arch="IMAGE_PITCH_ALIGNMENT=1")
    else: self.target = target

  def render(self, uops:list[UOp]) -> str:
    # Keep the original UOp order; remove only the temporary sink after lowering.
    sink = graph_rewrite(UOp.sink(*uops), self.comparison_matcher)
    return base64.b64encode(pickle.dumps(list(sink.toposort())[:-1])).decode()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d != dtypes.half or sys.version_info >= (3, 12)}

class RockchipAllocator(HostAllocator):
  def __init__(self, dev):
    super().__init__(dev)
    self.sizes:dict[int, int] = {}

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.external_ptr is not None: return super()._alloc(size, options)
    addr, mem = self.dev._gpu_alloc(size)
    self.sizes[addr] = size
    return BufferStorage(addr, mem, self._view(addr, size))

  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.buf in self.sizes:
      del self.sizes[storage.buf]
      self.dev._gpu_free(storage.buf, storage.meta)

class RockchipDevice(Compiled):
  allocator: RockchipAllocator

  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
    self._gpu_bufs:list[tuple[int, rk.struct_rknpu_mem_create]] = []
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)
    self.task_buf, self.task_mem = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING)
    self.regcmd_buf, self.regcmd_mem = self._gpu_alloc(8192)
    self.input_buf, self.input_mem = self._gpu_alloc(4194304)
    self.weight_buf, self.weight_mem = self._gpu_alloc(4194304)
    self.output_buf, self.output_mem = self._gpu_alloc(4194304)

  def dma_address(self, addr:int, size:int) -> int|None:
    for base,mem in self._gpu_bufs:
      if base <= addr and addr+size <= base+mem.size: return mem.dma_addr + addr-base
    return None

  def _gpu_alloc(self, size:int, flags:int=0) -> tuple[int, rk.struct_rknpu_mem_create]:
    mem = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    try:
      mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem.handle)
      addr = self.fd_ctl.mmap(0, mem.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset)
    except Exception:
      rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.handle, obj_addr=mem.obj_addr)
      raise
    self._gpu_bufs.append((addr, mem))
    return addr, mem

  def _gpu_free(self, addr:int, mem:rk.struct_rknpu_mem_create) -> None:
    FileIOInterface.munmap(addr, mem.size)
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.handle, obj_addr=mem.obj_addr)
    self._gpu_bufs.remove((addr, mem))
