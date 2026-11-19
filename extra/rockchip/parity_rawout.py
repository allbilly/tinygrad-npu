#!/usr/bin/env python3
"""Does raw_out leave int32 planes that dequantise to the identical fp32 result?

FFN tail fusion needs gate/up to stop after the planes and be consumed by one pack kernel, so
those planes must be exactly what the fused dequantise would have produced. This checks it
directly: run the normal plan, run the raw_out plan, dequantise the raw planes in numpy with the
same expression the emitter uses (`(float)raw * a * w`), and compare bitwise against the fp32 the
normal plan produced.
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


def build(dev, weight, w_scale, raw_out: bool):
  # Give each arm its OWN plane and payload allocations. `_plane_buffer` pools on
  # (id(dev), m, n), so without this both arms share one buffer and the readback is a mixture.
  NL._FUSED_PROGRAMS.clear()
  NL._PLANE_POOL.clear()
  NL._PAYLOAD_POOL.clear()
  return NL._FusedInt8Plan(dev, m=M, n=N, k=K, depth_in=K, cores=3, column_offset=0,
                           row_stride=K, row_offset=0, k_offset=0, weight_dma=weight.dma,
                           w_scale_addr=w_scale.addr, raw_out=raw_out)


def unpack_planes(flat: np.ndarray, parts, m: int, n: int) -> np.ndarray:
  """Rebuild (m, n) from the emitter's layout: part block at offset*m, then row*width."""
  out = np.empty((m, n), np.float32)
  for offset, width in parts:
    block = flat[offset * m: offset * m + m * width].reshape(m, width)
    out[:, offset:offset + width] = block
  return out


def read_f32(dev, buf, count: int) -> np.ndarray:
  out = np.empty(count, np.float32)
  dev.allocator._copyout(memoryview(out).cast("B"), buf)
  return out


def read_i32(dev, buf, count: int) -> np.ndarray:
  out = np.empty(count, np.int32)
  dev.allocator._copyout(memoryview(out).cast("B"), buf)
  return out


def main() -> int:
  dev = Device["ROCKCHIP:0"]
  rng = np.random.default_rng(0)
  codes = rng.integers(-127, 128, size=N * K, dtype=np.int16).astype(np.int8)
  weight = dev.allocate(N * K)
  dev.allocator._copyin(weight, memoryview(bytearray(codes.tobytes())))
  wsc = (np.abs(rng.standard_normal(N)).astype(np.float32) / 127.0)
  w_scale = dev.allocate(N * 4)
  dev.allocator._copyin(w_scale, memoryview(bytearray(wsc.tobytes())))
  x = Tensor(((np.arange(M * K, dtype=np.float32) % 97) / 97.0 - 0.5).reshape(M, K),
             device="CPU").contiguous().realize()

  # arm A: normal fused path (dequantise inside the kernel)
  pa = build(dev, weight, w_scale, False)
  out_fp32 = pa(x).realize().numpy()

  # arm B: raw_out (planes left as int32)
  pb = build(dev, weight, w_scale, True)
  pb(x).realize()
  src_b = NL._fused_source(pb.key)
  has_dequant = "dst[j]" in src_b
  flat = read_i32(dev, pb.out, M * N)
  planes = unpack_planes(flat, pb.key.parts, M, N)
  scales = read_f32(dev, pb.payload, M)          # fp32 scale header, first M floats

  # same expression the emitter uses, evaluated in numpy
  expect = (planes.astype(np.float32) * scales[:, None]) * wsc[None, :]
  identical = np.array_equal(expect, out_fp32)

  print(f"raw_out key           : {pb.key.raw_out}   source has dequantise: {has_dequant}")
  print(f"parts                 : {pb.key.parts}")
  print(f"planes                : {planes.shape} {planes.dtype}  nonzero={int((planes != 0).sum())}")
  print(f"scales[:3]            : {scales[:3]}")
  print(f"fp32 out shape        : {out_fp32.shape}")
  print(f"bitwise identical     : {identical}")
  if not identical:
    d = np.abs(expect.astype(np.float64) - out_fp32.astype(np.float64))
    print(f"  max_abs_diff={d.max():.6g}  n_diff={(d > 0).sum()}/{d.size}")
  return 0 if identical else 1


if __name__ == "__main__":
  raise SystemExit(main())
