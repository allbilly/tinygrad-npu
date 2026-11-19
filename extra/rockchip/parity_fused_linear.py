#!/usr/bin/env python3
"""Bitwise parity of the fused INT8 Linear at its own output boundary.

`generated_ids` is NOT a parity gate: the model samples, and the baseline's own three
runs diverge at token 0 (`threadsplit/scanpool-3plus3` run0 [141274, ...] vs run1
[182662, ...] vs run2 [185053, ...]). The deterministic gate this tree already uses is
`logits_sha256_16` (see `docs/rknpu/results/parity/nosplit`).

This narrows it further and is cheaper: one fused Linear, one fixed input row, run once
with the serial dequantise emitter and once with the pooled one, then compared bit for
bit. No model load, no JIT, no sampling.

Geometry m=64 n=2560 k=4096 with row_stride == k, so it deliberately stays on the plain
`int8_task_plan` path and never touches the `m_rows == 2*m` branch (which has a separate
suspected out-of-bounds at `npu_linear.py:609`).

Usage: PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/parity_fused_linear.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tinygrad import Device, Tensor  # noqa: E402
import npu_linear as NL  # noqa: E402

M, N, K = 64, 2560, 4096


def build(dev, seed: int = 0):
  rng = np.random.default_rng(seed)
  codes = rng.integers(-127, 128, size=N * K, dtype=np.int16).astype(np.int8)
  weight = dev.allocate(N * K)
  dev.allocator._copyin(weight, memoryview(bytearray(codes.tobytes())))
  scales = (np.abs(rng.standard_normal(N)).astype(np.float32) / 127.0)
  w_scale = dev.allocate(N * 4)
  dev.allocator._copyin(w_scale, memoryview(bytearray(scales.tobytes())))
  plan = NL._FusedInt8Plan(dev, m=M, n=N, k=K, depth_in=K, cores=3, column_offset=0,
                           row_stride=K, row_offset=0, k_offset=0,
                           weight_dma=weight.dma, w_scale_addr=w_scale.addr)
  return plan, weight, w_scale


def main() -> int:
  dev = Device["ROCKCHIP:0"]

  # A fixed, non-symmetric activation row so a transposed or shifted write cannot pass.
  x_np = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 97) / 97.0 - 0.5
  x = Tensor(x_np, device="CPU").contiguous().realize()

  NL.PARALLEL_QUANT = False
  NL._FUSED_PROGRAMS.clear()
  plan_serial, w1, s1 = build(dev)
  out_serial = plan_serial(x).realize().numpy()
  print(f"serial   plan built: PARALLEL_QUANT={NL.PARALLEL_QUANT} "
        f"programs={len(NL._FUSED_PROGRAMS)} finite={np.isfinite(out_serial).all()}", flush=True)

  import thread_split
  thread_split.install()                     # registers rkpool in the DLL table
  print(f"pool address: {hex(NL._pool_addr() or 0)}", flush=True)

  NL.PARALLEL_QUANT = True
  NL._FUSED_PROGRAMS.clear()
  plan_par, w2, s2 = build(dev)
  out_par = plan_par(x).realize().numpy()
  print(f"parallel plan built: PARALLEL_QUANT={NL.PARALLEL_QUANT} "
        f"programs={len(NL._FUSED_PROGRAMS)} finite={np.isfinite(out_par).all()}", flush=True)

  same = np.array_equal(out_serial, out_par)
  print(f"\nshapes {out_serial.shape} vs {out_par.shape}")
  print(f"bitwise identical: {same}")
  if not same:
    diff = np.abs(out_serial.astype(np.float64) - out_par.astype(np.float64))
    print(f"  max_abs_diff={diff.max():.6f}  n_diff={(diff>0).sum()} / {diff.size}")
    rows = np.where((diff > 0).any(axis=1))[0]
    print(f"  differing rows: {rows[:8]} ... total {len(rows)}")
  print(f"serial   checksum {float(np.abs(out_serial).sum()):.6f}")
  print(f"parallel checksum {float(np.abs(out_par).sum()):.6f}")
  return 0 if same else 1


if __name__ == "__main__":
  raise SystemExit(main())
