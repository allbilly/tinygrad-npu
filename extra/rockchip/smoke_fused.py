#!/usr/bin/env python3
"""Bit-exactness gate for the fused INT8 Linear: one C call against the six-launch path.

The fused function does the quantize, both ioctls and the dequantize itself, so the only
thing that can drift is arithmetic. This compares raw fp32 output words, not a tolerance:
same per-row max, same 1e-8 floor, same `__builtin_nearbyintf` codes, same unfused
`(float)src * a * w_scale` chain.

The reference is the existing path run with `column_offset=0` on an explicit contiguous
tile, which is unambiguously correct. The fused call is given the whole activation and the
row/K/column offsets instead, so it exercises the baked-offset path.

Cases cover the decode and prefill shapes the models issue, a ragged K tile
(`depth_in < k`, the zero-fill path), a row tile at a nonzero row offset, and a nonzero
output-column offset.

Usage: DEV=CPU .venv/bin/python extra/rockchip/smoke_fused.py [--dump DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from tinygrad import Tensor

from npu_linear import packed_matmul_int8, packed_matmul_int8_fused, pack_weight_int8

# (dense_rows, depth, m, row_offset, k_offset, k, n, column_offset, cores, note)
CASES = [
    (1, 2560, 1, 0, 0, 2560, 2560, 0, 3, "q/k/v/o at decode"),
    (1, 2560, 1, 0, 0, 2560, 9216, 0, 3, "ffn gate/up at decode"),
    (64, 2560, 64, 0, 0, 2560, 9216, 0, 3, "ffn gate/up at prefill"),
    (64, 4096, 64, 0, 0, 4096, 2560, 0, 3, "ffn down at prefill"),
    (1, 2560, 1, 0, 0, 2560, 1280, 1280, 3, "nonzero column offset"),
    (64, 2560, 64, 0, 0, 2560, 1280, 1280, 3, "nonzero column offset, 64 rows"),
    (128, 4864, 64, 64, 4096, 768, 2560, 0, 3, "ragged K tile at a nonzero row offset"),
    (1, 896, 1, 0, 0, 1024, 896, 0, 3, "depth_in < k: the zero-fill path"),
    (1, 2560, 1, 0, 0, 2560, 2560, 0, 1, "single core"),
]


def bits(t: Tensor) -> np.ndarray:
    return t.numpy().view(np.uint32)


def run(case: tuple[int, int, int, int, int, int, int, int, int, str], seed: int) -> dict:
    dense_rows, depth, m, row_offset, k_offset, k, n, column_offset, cores, note = case
    rng = np.random.default_rng(seed)
    weight = rng.random((column_offset + n, k), dtype=np.float32) * 2 - 1
    codes, scales = pack_weight_int8(Tensor(weight), "ROCKCHIP")
    codes, scales = codes.realize(), scales.realize()
    # The reference packs its own copy of the same rows: packing is row-independent, so the
    # codes and scales match, and the old path never has to take a view of an NPU buffer.
    ref_codes, ref_scales = pack_weight_int8(Tensor(weight[column_offset:].copy()), "ROCKCHIP")
    ref_codes, ref_scales = ref_codes.realize(), ref_scales.realize()
    dense = Tensor(rng.random((dense_rows, depth), dtype=np.float32) * 2 - 1)

    fused = packed_matmul_int8_fused(dense, codes, scales, n, "ROCKCHIP", m, row_offset=row_offset,
                                     k_offset=k_offset, column_offset=column_offset, cores=cores)
    # Reference: the same weight columns through the old path at offset 0 on a materialised tile.
    tile = dense[row_offset:row_offset + m, k_offset:k_offset + k].contiguous()
    ref = packed_matmul_int8(tile, ref_codes, ref_scales, n, "ROCKCHIP", cores, 0)
    # What the old path does when it is given the offset instead: documents the latent scale-index bug.
    old = packed_matmul_int8(tile, codes, scales, n, "ROCKCHIP", cores, column_offset)

    got, want = bits(fused), bits(ref)
    return {"note": note, "shape": f"m={m} n={n} k={k}", "exact": bool(np.array_equal(got, want)),
            "differing": int((got != want).sum()), "elements": int(got.size),
            "old_offset_exact": bool(np.array_equal(bits(old), want)),
            "fused": fused, "ref": ref}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, default=None,
                    help="write the reference outputs here as .npy for a later comparison")
    args = ap.parse_args()
    if args.dump is not None:
        args.dump.mkdir(parents=True, exist_ok=True)

    print(f"{'case':<42} {'shape':<22} {'words':>8} {'differ':>7}  old-offset")
    bad = 0
    for index, case in enumerate(CASES):
        result = run(case, seed=index)
        bad += not result["exact"]
        print(f"{result['note']:<42} {result['shape']:<22} {result['elements']:>8} "
              f"{result['differing']:>7}  {'exact' if result['old_offset_exact'] else 'differs'}"
              f"  {'ok' if result['exact'] else 'FAIL'}")
        if args.dump is not None:
            np.save(args.dump / f"ref-{index}.npy", result["ref"].numpy())
            np.save(args.dump / f"fused-{index}.npy", result["fused"].numpy())
    print("all exact" if not bad else f"{bad} case(s) failed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
