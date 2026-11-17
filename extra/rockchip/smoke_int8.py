#!/usr/bin/env python3
"""Smoke-test the RK3588 INT8 matmul across shapes vs numpy.

Layouts (from the reference's rockchip_int8.py):
  lhs  (m, k)      -> [m fp32 scales, padded to 64 bytes] then row-major int8
  rhs  (n, k)      -> (n/32, k/32, 32, 32) int8
  out  (m, n)      -> int32 row-major, dequantized as out * a_scale[m] * w_scale[n]

The descriptor is <8I: m, n, k, cores, rhs_offset, lhs_offset, split_k, int8.
"""
import struct, sys
import numpy as np
from tinygrad import Device
from tinygrad.helpers import Target, round_up
from tinygrad.renderer import TinyELF

# (m, n, k, cores) covering Qwen3.5 4B's projection shapes.
# Device limits: k<=4096, m<=64 when k>2048 (128 when k<=2048), n<=4096/core (8192 when m==1).
SHAPES = [
  (1, 2560, 2560, 1),    # q/k/v/o at decode
  (1, 2560, 2560, 3),    # same, three cores
  (1, 9216, 2560, 3),    # ffn gate/up at decode
  (64, 2560, 2560, 3),   # prefill row tile
  (64, 9216, 2560, 3),   # ffn gate/up at prefill
  (64, 2560, 4096, 3),   # ffn down at prefill: m=64 is legal for INT8 at k>2048
  (128, 2560, 2048, 3),  # 128 rows allowed only when k<=2048
]

dev = Device["ROCKCHIP"]


def upload(arr: np.ndarray):
  storage = dev.allocator.alloc(arr.nbytes)
  dev.allocator._copyin(storage.buf, memoryview(bytearray(arr.tobytes())))
  return storage.buf


def run(m: int, n: int, k: int, cores: int, seed: int = 0) -> tuple[float, float]:
  rng = np.random.default_rng(seed)
  a = rng.integers(-127, 128, (m, k), dtype=np.int8)
  w = rng.integers(-127, 128, (n, k), dtype=np.int8)
  a_scale = rng.random(m).astype(np.float32) + 0.5
  w_scale = rng.random(n).astype(np.float32) + 0.5

  header = round_up(m * 4, 64)
  lhs = np.zeros(header + m * k, dtype=np.uint8)
  lhs[:m * 4] = np.frombuffer(a_scale.tobytes(), dtype=np.uint8)
  lhs[header:] = a.reshape(-1).view(np.uint8)
  rhs = w.reshape(n // 32, 32, k // 32, 32).transpose(0, 2, 1, 3).copy()

  lhs_buf, rhs_buf = upload(lhs), upload(rhs)
  out_buf = upload(np.zeros((m, n), np.int32))
  desc = struct.pack("<8I", m, n, k, cores, 0, header, 0, 1)
  prog = dev.runtime(TinyELF(desc, "rockchip_int8", Target("ROCKCHIP"), ()))
  prog(out_buf, lhs_buf, rhs_buf)

  raw = np.empty(m * n, np.int32)
  dev.allocator._copyout(memoryview(raw).cast("B"), out_buf)
  ref = (a.astype(np.int32) @ w.astype(np.int32).T)
  # Two candidate layouts: one shared row-major (m,n), or dense per-core planes.
  width = round_up(-(-n // cores), 32) if cores > 1 else n
  parts = [(o, min(width, n - o)) for o in range(0, n, width)] if cores > 1 else [(0, n)]
  if len(parts) not in (1, 3): parts = [(0, n)]
  try:
    scattered = np.concatenate([raw[o * m:(o + c) * m].reshape(m, c) for o, c in parts], axis=1)
  except ValueError:
    scattered = None
  rowmajor = raw.reshape(m, n)
  got = scattered if (scattered is not None and np.array_equal(scattered, ref)) else rowmajor
  layout = "planes" if got is scattered else "rowmajor"
  return float(np.abs(got - ref).max()), float(np.abs(ref).max()), layout


def main() -> int:
  bad = 0
  for m, n, k, cores in SHAPES:
    err, scale, layout = run(m, n, k, cores)
    rel = err / scale
    ok = rel < 1e-5
    bad += not ok
    print(f"m={m:<4} n={n:<5} k={k:<5} cores={cores}  max_abs_err={err:>10.4f} rel={rel:.3e}  {layout:<9} {'ok' if ok else 'FAIL'}")
  print("all exact" if not bad else f"{bad} shape(s) failed")
  return 0 if not bad else 1


if __name__ == "__main__":
  sys.exit(main())
