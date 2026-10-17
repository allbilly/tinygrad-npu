"""Standalone LUT probe; not a Tensor activation implementation.

The host constructs constant tables and reference answers. LUT evaluation and
SiLU/GELU multiplication run on the NPU. The GELU tail currently misses tolerance.
Run serially: python extra/rockchip/probe_lut.py sigmoid SiLU
"""
import math, struct, sys
import numpy as np
from tinygrad.uop.ops import Ops
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram, fp16, rk, to_mv

dev = RockchipDevice('ROCKCHIP')
p = RockchipProgram.__new__(RockchipProgram)
p.dev = dev
dev.regcmd_buf, dev.regcmd_mem = dev._gpu_alloc(16384)
E = p.EMIT
values = [-4., -2., -1., -0.25, 0., 0.25, 1., 2., 4.]
def sigmoid(x): return 1/(1+math.exp(-x))
def gelu_gate(x): return (1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x*x*x)))/2

def lut(fn):
  result = []
  for start in range(0, len(values), 8):
    lanes = values[start:start+8]
    p.build_registers(Ops.MUL)
    p.npu_regs += [
      # LUT has no EW operand: do not start an unused operand DMA stream.
      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT),
      E(rk.DPU, rk.REG_DPU_BN_CFG, (2 << rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT) |
        (1 << rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000),
      E(rk.DPU, rk.REG_DPU_BN_MUL_CFG, fp16(2596.) << rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT),
      E(rk.DPU, rk.REG_DPU_EW_CFG, (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) |
        (1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT)),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) |
        (15 << rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT)),
      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 1),
      E(rk.DPU, rk.REG_DPU_LUT_CFG, (1 << rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT) |
        (1 << rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT) | (2 << rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT)),
      E(rk.DPU, rk.REG_DPU_LUT_INFO, (5 << rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT) |
        (5 << rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT)),
      E(rk.DPU, rk.REG_DPU_LUT_LE_START, -16384), E(rk.DPU, rk.REG_DPU_LUT_LE_END, 0),
      E(rk.DPU, rk.REG_DPU_LUT_LO_START, 0), E(rk.DPU, rk.REG_DPU_LUT_LO_END, 16384),
      E(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE, 0), E(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT, 0),
      E(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 0), E(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 0),
    ]
    for table in (0,1):
      p.npu_regs.append(E(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
        (1 << rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT) | (table << rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT)))
      for i in range(513):
        x = (i-512 if table==0 else i)*32/2596
        p.npu_regs.append(E(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA, max(1,min(32767,round(fn(x)*32768)))))
    to_mv(dev.input_buf,16)[:] = struct.pack('<8e',*(lanes+[0.]*(8-len(lanes))))
    p.submit()
    result.extend(struct.unpack('<8e',to_mv(dev.output_buf,16))[:len(lanes)])
  return result

passed = True
selected = sys.argv[1:] or ['sigmoid', 'SiLU', 'GELU(tanh)']
functions = {'sigmoid': (sigmoid, False), 'SiLU': (sigmoid, True), 'GELU(tanh)': (gelu_gate, True)}
for name in selected:
  fn, multiply = functions[name]
  got = lut(fn)
  if multiply: got = p.run_npu(Ops.MUL, values, got)
  ref = np.array([fn(x)*x if multiply else fn(x) for x in values], dtype=np.float16)
  passed &= bool(np.allclose(got,ref,rtol=0.001,atol=1e-6))
  print(name, 'inputs',values,'NPU',got,'reference',ref.tolist(),
        'max_abs',float(np.max(np.abs(np.array(got)-ref))),
        'close',bool(np.allclose(got,ref,rtol=0.001,atol=1e-6)),flush=True)

raise SystemExit(0 if passed else 1)
