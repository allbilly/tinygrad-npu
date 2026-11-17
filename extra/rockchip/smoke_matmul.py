#!/usr/bin/env python3
"""Smoke-test the RK3588 FP16 matmul device: one 64x64x64 GEMM vs numpy.

Layouts mirror tinygrad/llm/kernels/rockchip.py:
  lhs    (m, k)      -> (k/8, m, 8)
  weight (n, k)      -> (n/16, k/32, 16, 32)
  out    (n/4, m, 4) -> (m, n)
"""
import struct, sys
import numpy as np
from tinygrad import Device
from tinygrad.helpers import Target
from tinygrad.renderer import TinyELF

M = N = K = 64
rng = np.random.default_rng(0)
lhs = rng.standard_normal((M, K)).astype(np.float16)
rhs = rng.standard_normal((N, K)).astype(np.float16)
lhs_packed = lhs.reshape(M, K // 8, 8).transpose(1, 0, 2).copy()
rhs_packed = rhs.reshape(N // 16, 16, K // 32, 32).transpose(0, 2, 1, 3).copy()

dev = Device["ROCKCHIP"]


def upload(array: np.ndarray):
    storage = dev.allocator.alloc(array.nbytes)
    dev.allocator._copyin(storage.buf, memoryview(bytearray(array.tobytes())))
    return storage.buf


lhs_buf, rhs_buf, out_buf = upload(lhs_packed), upload(rhs_packed), upload(np.zeros((N // 4, M, 4), np.float32))
prog = dev.runtime(TinyELF(struct.pack("<4I", M, N, K, 1), "rockchip_matmul", Target("ROCKCHIP"), ()))
prog(out_buf, lhs_buf, rhs_buf)

tiles = np.empty((N // 4, M, 4), np.float32)
dev.allocator._copyout(memoryview(tiles).cast("B"), out_buf)
out = tiles.transpose(1, 0, 2).reshape(M, N)
ref = lhs.astype(np.float32) @ rhs.astype(np.float32).T
err = np.abs(out - ref).max()
rel = err / np.abs(ref).max()
print(f"max_abs_err={err:.4f} rel={rel:.5f} out[0,0]={out[0,0]:.4f} ref[0,0]={ref[0,0]:.4f}")
sys.exit(0 if rel < 0.02 else 1)
