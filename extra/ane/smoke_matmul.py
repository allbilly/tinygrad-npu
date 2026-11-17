#!/usr/bin/env python3
"""Check the ANE FP16 matmul against NumPy before loading a model.

The device must be available (`/dev/accel/accel0`); this is a hardware gate,
not a CPU fallback. Exit status is non-zero on a missing device or a mismatch.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import ane_linear

    rng = np.random.default_rng(args.seed)
    weight = rng.standard_normal((args.n, args.k), dtype=np.float32).astype(np.float16)
    x = rng.standard_normal((args.rows, args.k), dtype=np.float32)
    expected = x @ weight.astype(np.float32).T

    ane = ane_linear.ane_device()
    plan = ane.create(weight)
    actual = ane.run(plan, x, args.n)
    error = np.abs(actual - expected)
    relative = error / np.maximum(np.abs(expected), 1e-6)
    print(f"shape {args.rows}x{args.k} @ {args.k}x{args.n}  "
          f"max_abs={error.max():.6g}  max_rel={relative.max():.6g}")
    ok = np.allclose(actual, expected, rtol=args.rtol, atol=args.atol)
    ane.free(plan)
    ane_linear.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
