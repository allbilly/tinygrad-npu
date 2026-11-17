#!/usr/bin/env python3
"""Break one INT8 NPULinear call into its stages, at the Qwen3.5 4B's real shapes.

Decode was 662 ms/token with 151 ms in the NPU, so ~510 ms was host work across 323 submits,
i.e. ~1.55 ms per Linear. This times each stage separately so the port targets the real cost
rather than the assumed one, then times the fused replacement against it.

Two sections:

1. Stage breakdown of the six-launch path, mirroring `packed_matmul_int8`: pad, quantize
   (kernel + H2D), alloc, submit, D2H, then the fused tail (plane scatter + int32 -> fp32 +
   both multiplies, one CPU kernel). Every stage is timed without a realize, so the rows are
   lazy graph-construction cost; execution is not in this table. The fused tail runs in
   13-22 us at m=1, n=2560, so construction is what the model pays per call.

2. Whole-call comparison, `construct` (lazy only, what the model pays 323 times per token) and
   `construct+realize` (one call executed on its own, so it also carries the NPU submit). The
   second column is an upper bound on the per-call cost: in the model all 323 calls are
   realized in one scheduler pass, which is why the decode figure has to come from bench_llm.

Usage: DEV=CPU .venv/bin/python extra/rockchip/profile_int8.py [--reps N]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import round_up

from npu_linear import (_dequantize, _matmul_int8, _matmul_parts, int8_payload, pack_weight_int8,
                        packed_matmul_int8, packed_matmul_int8_fused)

# (n, k) from the 4B: ffn gate/up, attn_qkv, ffn down, q/k/v.
SHAPES = [(9216, 2560), (8192, 2560), (2560, 4096), (2560, 2560)]
STAGES = ("pad", "quantize", "alloc", "submit", "d2h", "dequant")


def stages(x: Tensor, weight: Tensor, scales: Tensor, columns: int, cores: int = 3) -> dict[str, float]:
    m, original_k = x.shape
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    t: dict[str, float] = {}

    def timed(name: str, fn):
        begin = time.perf_counter()
        out = fn()
        t[name] = (time.perf_counter() - begin) * 1000
        return out

    padded = timed("pad", lambda: x.pad(((0, 0), (0, k - original_k))))
    payload, a_scale = timed("quantize", lambda: int8_payload(padded, "ROCKCHIP"))
    out = timed("alloc", lambda: Tensor.empty(m * n, device="ROCKCHIP", dtype=dtypes.int32))
    out = timed("submit", lambda: out.custom_kernel(
        payload, weight, fxn=lambda o, a, w: _matmul_int8(o, a, w, m, n, k, cores, 0))[0])
    host = timed("d2h", lambda: out.to(x.device))
    parts = _matmul_parts(m, n, k, cores, int8=True)
    timed("dequant", lambda: _dequantize(host, a_scale, scales, m, n, parts))
    return t


def whole(x: Tensor, weight: Tensor, scales: Tensor, columns: int, m: int, reps: int) -> tuple[float, float]:
    """(lazy construction ms, construction + realization ms) per call, medians."""
    calls = {
        "six-launch": lambda: packed_matmul_int8(x, weight, scales, columns, "ROCKCHIP", 3, 0),
        "fused": lambda: packed_matmul_int8_fused(x, weight, scales, columns, "ROCKCHIP", m),
    }
    construct, total = [], []
    for name in ("six-launch", "fused"):
        lazy, full = [], []
        for _ in range(reps):
            begin = time.perf_counter()
            out = calls[name]()
            middle = time.perf_counter()
            out.realize()
            lazy.append((middle - begin) * 1000)
            full.append((time.perf_counter() - begin) * 1000)
        construct.append(statistics.median(lazy))
        total.append(statistics.median(full))
    return construct, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()

    print(f"reps={args.reps}, median ms per stage (lazy construction only)\n")
    print(f"{'n':>6} {'k':>5} {'m':>4} " + " ".join(f"{s:>9}" for s in STAGES) + f" {'total':>9} {'host%':>6}")
    for n, k in SHAPES:
        linear = nn.Linear(k, n)
        weight, scales = pack_weight_int8(linear.weight, "ROCKCHIP")
        weight, scales = weight.realize(), scales.realize()
        for m in (1, 64):
            x = Tensor.rand(m, k).float() * 2 - 1
            rows = [stages(x, weight, scales, n) for _ in range(args.reps)]
            med = {s: statistics.median(r[s] for r in rows) for s in STAGES}
            total = sum(med.values())
            host = total - med["submit"]
            print(f"{n:>6} {k:>5} {m:>4} " + " ".join(f"{med[s]:>9.3f}" for s in STAGES) +
                  f" {total:>9.3f} {100*host/total:>5.1f}%")

    print(f"\nwhole call, median ms ({args.reps} reps)\n")
    print(f"{'n':>6} {'k':>5} {'m':>4} {'six lazy':>9} {'fused lazy':>11} {'x':>6} "
          f"{'six full':>9} {'fused full':>11} {'x':>6}")
    for n, k in SHAPES:
        linear = nn.Linear(k, n)
        weight, scales = pack_weight_int8(linear.weight, "ROCKCHIP")
        weight, scales = weight.realize(), scales.realize()
        for m in (1, 64):
            x = Tensor.rand(m, k).float() * 2 - 1
            construct, total = whole(x, weight, scales, n, m, args.reps)
            print(f"{n:>6} {k:>5} {m:>4} {construct[0]:>9.3f} {construct[1]:>11.3f} "
                  f"{construct[0]/construct[1]:>5.2f}x {total[0]:>9.3f} {total[1]:>11.3f} "
                  f"{total[0]/total[1]:>5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
