#!/usr/bin/env python3
"""Row-tiling invariance: the INT8 matmul must be independent of how rows are grouped.

`NPULinear` tiles rows at 64 for INT8 (128 when k<=2048). That is only safe if the
matmul is row-independent, so this compares the tile the model uses against half
of it. The integer result and the activation scales must be bit-identical; the
float dequantize is allowed float32 rounding because the multiply kernel is
vectorized differently at different row counts.

Usage: DEV=CPU .venv/bin/python extra/rockchip/smoke_rows.py
"""
from __future__ import annotations

import sys

import numpy as np

from tinygrad import Tensor, dtypes, nn

from npu_linear import _matmul_int8, _matmul_parts, int8_payload, pack_weight_int8

# (rows, columns, depth) covering both INT8 row-tile bounds.
SHAPES = [
    (128, 2560, 2560),  # k=2560 > 2048, so the tile is 64: two tiles
    (128, 2560, 4096),  # ffn_down depth, same 64-row tile
    (128, 2560, 2048),  # k<=2048, so the tile is 128: one tile
    (192, 2560, 2560),  # three tiles, last one ragged
]


def raw(x: Tensor, codes: Tensor, rows: int, columns: int, depth: int, tile: int):
    """The matmul without dequantize, so the integer result can be compared exactly."""
    outs, scales = [], []
    for i in range(0, rows, tile):
        part = x[i:i + tile]
        m = part.shape[0]
        payload, a_scale = int8_payload(part, "ROCKCHIP")
        out = Tensor.empty(m * columns, device="ROCKCHIP", dtype=dtypes.int32)
        out = out.custom_kernel(payload, codes,
                                fxn=lambda o, a, w: _matmul_int8(o, a, w, m, columns, depth, 3, 0))[0].to("CPU")
        parts = _matmul_parts(m, columns, depth, 3, int8=True)
        planes = [out[o * m:(o + c) * m].reshape(m, c) for o, c in parts]
        outs.append(Tensor.cat(*planes, dim=1).numpy())
        scales.append(a_scale.numpy())
    return np.concatenate(outs, axis=0), np.concatenate(scales)


def run(rows: int, columns: int, depth: int) -> tuple[bool, bool]:
    linear = nn.Linear(depth, columns)
    codes, _ = pack_weight_int8(linear.weight, "ROCKCHIP")
    x = Tensor.rand(rows, depth).float() * 2 - 1
    big = 128 if depth <= 2048 else 64
    a_int, a_scale = raw(x, codes, rows, columns, depth, big)
    b_int, b_scale = raw(x, codes, rows, columns, depth, big // 2)
    return bool((a_int == b_int).all()), bool((a_scale == b_scale).all())


def main() -> int:
    bad = 0
    for rows, columns, depth in SHAPES:
        int_ok, scale_ok = run(rows, columns, depth)
        ok = int_ok and scale_ok
        bad += not ok
        print(f"rows={rows:<4} n={columns:<5} k={depth:<5} tile={128 if depth <= 2048 else 64:<4} "
              f"int32={'exact' if int_ok else 'MISMATCH':<9} scales={'exact' if scale_ok else 'MISMATCH':<9} "
              f"{'ok' if ok else 'FAIL'}")
    print("row tiling is exact" if not bad else f"{bad} shape(s) mismatch")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
