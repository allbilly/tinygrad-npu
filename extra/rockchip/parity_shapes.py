#!/usr/bin/env python3
"""Bit-exactness of the fused INT8 Linear at the Qwen3.5 4B's real geometries.

`parity_token.py` proves the fused path on Qwen2 0.5B, whose shapes are n=896, k=896
with no K split. The 4B is different in three ways that test does not reach: n up to
9216, k=4096 tiles, and ffn_down's k=9216 split into three K tiles with k_offset
0/4096/8192. A megakernel that is right at one geometry and wrong at another is the
exact failure mode this file exists to catch, so it compares the fused call against
`packed_matmul_int8` per case instead of trusting the token parity.

Usage: DEV=CPU .venv/bin/python extra/rockchip/parity_shapes.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from tinygrad import Tensor, nn

from npu_linear import (K_TILE, _int8_padded_depth, pack_weight_int8, packed_matmul_int8,
                        packed_matmul_int8_fused)

# (n, k, m, rows, column_offset) — the 4B's projections, decode rows and prefill tiles.
CASES = [
    (9216, 2560, 1, 1, 0),      # ffn gate/up, decode
    (9216, 2560, 64, 64, 0),    # ffn gate/up, prefill tile
    (8192, 2560, 1, 1, 0),      # attn_qkv, decode
    (2560, 4096, 1, 1, 0),      # ffn down K tile, decode
    (2560, 4096, 64, 64, 0),    # ffn down K tile, prefill
    (2560, 9216, 1, 1, 0),      # ffn down whole K: three K tiles, k_offset 0/4096/8192
    (2560, 2560, 1, 1, 256),    # column_offset, the lm_head tile path
]


def run(n: int, k: int, m: int, rows: int, column_offset: int) -> list[tuple[str, bool, float]]:
    # The weight must be column_offset wider than the output: the dequantize indexes
    # w_scale at column_offset + j, exactly as the real lm_head tile path does.
    linear = nn.Linear(k, n + column_offset)
    dense = Tensor.rand(rows, k).float() * 2 - 1
    results = []
    for k0 in range(0, k, K_TILE):
        kt = min(K_TILE, k - k0)
        padded = _int8_padded_depth(kt)
        tile = linear.weight[:, k0:k0 + kt].contiguous()
        codes, scales = pack_weight_int8(tile.pad(((0, 0), (0, padded - kt))), "ROCKCHIP")
        codes, scales = codes.realize(), scales.realize()

        old = packed_matmul_int8(dense[:, k0:k0 + kt], codes, scales, n, "ROCKCHIP", 3, column_offset).realize()
        new = packed_matmul_int8_fused(dense, codes, scales, n, "ROCKCHIP", m, 0, k0, column_offset, 3).realize()
        a, b = old.numpy(), new.numpy()
        same = bool(np.array_equal(a, b))
        delta = float(np.abs(a - b).max()) if not same else 0.0
        # Time both, so a bit-exact-but-slower kernel is visible rather than assumed faster.
        def med(fn, reps=5):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); fn().realize(); ts.append((time.perf_counter() - t0) * 1e3)
            return sorted(ts)[len(ts) // 2]
        t_old = med(lambda: packed_matmul_int8(dense[:, k0:k0 + kt], codes, scales, n, "ROCKCHIP", 3, column_offset))
        t_new = med(lambda: packed_matmul_int8_fused(dense, codes, scales, n, "ROCKCHIP", m, 0, k0, column_offset, 3))
        results.append((f"k_offset={k0}", same, delta, t_old, t_new))
    return results


def main() -> int:
    bad = 0
    print(f"{'n':>6} {'k':>5} {'m':>3} {'rows':>4} {'coloff':>6}  {'result':<10} {'old ms':>8} {'fused ms':>9} {'ratio':>6}")
    for n, k, m, rows, column_offset in CASES:
        for label, same, delta, t_old, t_new in run(n, k, m, rows, column_offset):
            bad += not same
            note = "exact" if same else f"MISMATCH {delta:.2e}"
            print(f"{n:>6} {k:>5} {m:>3} {rows:>4} {column_offset:>6}  {note:<10} {t_old:>8.3f} {t_new:>9.3f} {t_old/t_new:>5.2f}x")
    print("\nall exact" if not bad else f"\n{bad} case(s) mismatch")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
