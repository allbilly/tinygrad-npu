#!/usr/bin/env python3
"""Is the `m_rows == 2*m` merged path numerically correct?

`_fused_plan` merges the two row-halves of a 128-row prefill call when
`m == 128 and k > 2048` (that is `ffn_down`: k tile 4096, row_stride 9216). The merged
plan then places the second half's activation at

    payload.dma + header + m * row_stride          (npu_linear.py:609)

while the payload was allocated as

    round_up(rows_total*4,64) + rows_total * k       (npu_linear.py:604)

For the real geometry those disagree badly: payload = 512 + 128*4096 = 524,800 bytes,
but the second half starts at 512 + 64*9216 = 590,336 — 65,536 bytes past the end, and
its 64*4096 = 262,144 bytes are read entirely outside the payload.

If that is a bug, the merged plan's rows 64-127 must differ from running the same rows
as a plain m=64 plan. If the two agree bitwise, the offset is harmless (for example the
allocator happens to place a correct copy there) and the suspicion is refuted.

Both arms use only validated geometry (m=64, k=4096 <= 4096, n=2560 % 32 == 0).

Usage: PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/parity_mrows.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tinygrad import Device, Tensor  # noqa: E402
import npu_linear as NL  # noqa: E402

M, N, K, ROW_STRIDE = 64, 2560, 4096, 9216
ROWS = 2 * M


def main() -> int:
  assert getattr(NL, "PARALLEL_DEQUANT", False) is False, "run this on the serial emitter"
  dev = Device["ROCKCHIP:0"]
  rng = np.random.default_rng(0)

  codes = rng.integers(-127, 128, size=N * K, dtype=np.int16).astype(np.int8)
  weight = dev.allocate(N * K)
  dev.allocator._copyin(weight, memoryview(bytearray(codes.tobytes())))
  scales = (np.abs(rng.standard_normal(N)).astype(np.float32) / 127.0)
  w_scale = dev.allocate(N * 4)
  dev.allocator._copyin(w_scale, memoryview(bytearray(scales.tobytes())))

  x_np = ((np.arange(ROWS * ROW_STRIDE, dtype=np.float32) % 211) / 211.0) - 0.5
  x = Tensor(x_np.reshape(ROWS, ROW_STRIDE), device="CPU").contiguous().realize()

  def plan(m: int, row_offset: int, m_rows: int):
    return NL._FusedInt8Plan(dev, m=m, n=N, k=K, depth_in=K, cores=3, column_offset=0,
                             row_stride=ROW_STRIDE, row_offset=row_offset, k_offset=0,
                             weight_dma=weight.dma, w_scale_addr=w_scale.addr, m_rows=m_rows)

  NL._FUSED_PROGRAMS.clear()
  merged = plan(M, 0, ROWS)
  out_merged = merged(x).realize().numpy()
  merged_key = merged.key
  print(f"merged plan : m={merged_key.m} m_rows={merged_key.m_rows} "
        f"row_stride={merged_key.row_stride} k={merged_key.k} tasks={merged_key.tasks}")
  print(f"merged plan : payload_size={merged_key.payload_size} bytes")
  # The emitter stores quantised rows k-wide, so the second half starts at header + m*k.
  # (The key also records `row_stride`, which is the SOURCE stride used only to read x.)
  used = merged_key.header + merged_key.m * merged_key.k
  overshoot = used - merged_key.payload_size
  print(f"merged plan : second-half lhs offset={used} bytes "
        f"(header + m*k); source row_stride={merged_key.row_stride}")
  print(f"merged plan : overshoot = {overshoot} bytes"
        f" {'<-- OUT OF BOUNDS' if overshoot > 0 else '(in bounds)'}", flush=True)

  NL._FUSED_PROGRAMS.clear()
  lo = plan(M, 0, 0)
  out_lo = lo(x).realize().numpy()
  NL._FUSED_PROGRAMS.clear()
  hi = plan(M, M, 0)
  out_hi = hi(x).realize().numpy()

  same_lo = np.array_equal(out_merged[:M], out_lo)
  same_hi = np.array_equal(out_merged[M:], out_hi)
  print(f"\nmerged rows  0-63  vs plain m=64 row_offset=0  : identical={same_lo}")
  print(f"merged rows 64-127 vs plain m=64 row_offset=64 : identical={same_hi}")
  for tag, a, b in (("lo", out_merged[:M], out_lo), ("hi", out_merged[M:], out_hi)):
    if not (a == b).all():
      d = np.abs(a.astype(np.float64) - b.astype(np.float64))
      print(f"  {tag}: max_abs_diff={d.max():.6f}  n_diff={(d > 0).sum()}/{d.size}  "
            f"median_rel={float(np.median(d / (np.abs(b) + 1e-9))):.4f}")
  verdict = "CORRECT" if (same_lo and same_hi) else "MISMATCH"
  print(f"\nverdict: {verdict}")
  return 0 if verdict == "CORRECT" else 1


if __name__ == "__main__":
  raise SystemExit(main())
