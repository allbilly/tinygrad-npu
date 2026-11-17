#!/usr/bin/env python3
"""Time one raw INT8 submit at the shapes Qwen3.5 4B actually issues.

The question this answers: decode costs 654 ms/token, of which ~124 ms is the
ideal weight stream (4.2 GB at the 33.79 GB/s DDR ceiling) and ~88 ms is the
measured host tail. That leaves ~410 ms unexplained. Either the NPU streams
weights well below the bus ceiling, or each submit carries a fixed latency.

Both hypotheses predict differently, so one sweep separates them: hold k fixed
and vary n. Time is a + b*n. The intercept `a` is the per-submit fixed cost
(ioctl + task setup + kernel launch); the slope `b` gives effective bandwidth.

Every submission is RKNPU_JOB_BLOCK, so `prog(...)` returns after the NPU has
finished. `wait=True` additionally returns the ioctl's own duration, which
separates host prep from device time.

Usage: DEV=ROCKCHIP .venv/bin/python extra/rockchip/bench_submit.py [--reps N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from tinygrad import Device
from tinygrad.helpers import Target, round_up
from tinygrad.renderer import TinyELF

# RK3588 has a 64-bit LPDDR bus at 2112 MHz: 2112e6 * 8 bytes * 2 (DDR) = 33.79 GB/s.
DDR_CEILING_GBPS = 33.79

# (m, n, k, cores, note). Decode shapes are m=1; prefill uses 64-row tiles.
SWEEP = [
    # Fixed-cost probe: as small as the recipe allows.
    (1, 32, 512, 1, "probe: minimum task"),
    # n sweep at the FFN depth, three cores. Slope -> bandwidth, intercept -> latency.
    (1, 256, 2560, 3, "n sweep"),
    (1, 512, 2560, 3, "n sweep"),
    (1, 1024, 2560, 3, "n sweep"),
    (1, 2048, 2560, 3, "n sweep"),
    (1, 4096, 2560, 3, "n sweep"),
    (1, 8192, 2560, 3, "n sweep"),
    (1, 9216, 2560, 3, "ffn gate/up, decode"),
    # Same shape on one core: does the third core add throughput or only latency?
    # cores=1 caps n at 8192 when m==1, so compare there rather than at 9216.
    (1, 8192, 2560, 1, "n sweep, one core"),
    (1, 2560, 2560, 3, "q/k/v/o, decode"),
    (1, 2560, 2560, 1, "q/k/v/o, one core"),
    # The k=4096 path takes the special 0xb1 bank recipe.
    (1, 512, 4096, 3, "k=4096, narrow n"),
    (1, 1024, 4096, 3, "k=4096, n sweep"),
    (1, 2560, 4096, 3, "ffn down, decode"),
    # Prefill row tiles.
    (64, 2560, 2560, 3, "q/k/v/o, 64-row tile"),
    (64, 9216, 2560, 3, "ffn gate/up, 64-row tile"),
    (64, 2560, 4096, 3, "ffn down, 64-row tile"),
]


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def clocks() -> dict:
    """Device state at call time, so the artifact carries its own provenance."""
    return {
        "npu_min_hz": _read("/sys/class/devfreq/fdab0000.npu/min_freq"),
        "npu_max_hz": _read("/sys/class/devfreq/fdab0000.npu/max_freq"),
        "npu_cur_hz": _read("/sys/class/devfreq/fdab0000.npu/cur_freq"),
        "ddr_cur_hz": _read("/sys/class/devfreq/dmc/cur_freq"),
        "cpu6_cur_hz": _read("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"),
        "cpu6_governor": _read("/sys/devices/system/cpu/cpufreq/policy6/scaling_governor"),
        "temperature_c": round(int(_read("/sys/class/thermal/thermal_zone0/temp") or 0) / 1000, 1),
    }


def upload(dev, arr: np.ndarray):
    storage = dev.allocator.alloc(arr.nbytes)
    dev.allocator._copyin(storage.buf, memoryview(bytearray(arr.tobytes())))
    return storage.buf


def build(dev, m: int, n: int, k: int, cores: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, (m, k), dtype=np.int8)
    w = rng.integers(-127, 128, (n, k), dtype=np.int8)
    header = round_up(m * 4, 64)
    lhs = np.zeros(header + m * k, dtype=np.uint8)
    lhs[: m * 4] = np.frombuffer(np.ones(m, np.float32).tobytes(), dtype=np.uint8)
    lhs[header:] = a.reshape(-1).view(np.uint8)
    rhs = w.reshape(n // 32, 32, k // 32, 32).transpose(0, 2, 1, 3).copy()
    lhs_buf, rhs_buf = upload(dev, lhs), upload(dev, rhs)
    out_buf = upload(dev, np.zeros((m, n), np.int32))
    desc = struct.pack("<8I", m, n, k, cores, 0, header, 0, 1)
    prog = dev.runtime(TinyELF(desc, "rockchip_int8", Target("ROCKCHIP"), ()))
    return prog, lhs_buf, rhs_buf, out_buf, (lhs_buf, rhs_buf, out_buf)


def measure(dev, m: int, n: int, k: int, cores: int, reps: int) -> dict:
    prog, lhs, rhs, out, bufs = build(dev, m, n, k, cores)
    # Bind once so `prepare` is off the hot path, exactly as it is after a JIT capture.
    prog(out, lhs, rhs)
    totals, internals = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        inner = prog(out, lhs, rhs, wait=True)
        totals.append((time.perf_counter() - t0) * 1e3)
        internals.append((inner or 0.0) * 1e3)
    for b in bufs:
        dev.free(b)
    weight_bytes = n * k
    total = statistics.median(totals)
    return {
        "m": m, "n": n, "k": k, "cores": cores,
        "total_ms": total,
        "internal_ms": statistics.median(internals),
        "host_ms": total - statistics.median(internals),
        "ideal_ms": weight_bytes / (DDR_CEILING_GBPS * 1e9) * 1e3,
        "gbps": weight_bytes / (total / 1e3) / 1e9,
        "pct_ceiling": 100 * (weight_bytes / (total / 1e3)) / (DDR_CEILING_GBPS * 1e9),
    }


def fit(points: list[dict]) -> dict:
    """Least squares of total_ms against n at fixed k: intercept is per-submit cost."""
    xs = np.array([p["n"] for p in points], dtype=np.float64)
    ys = np.array([p["total_ms"] for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    k = points[0]["k"]
    # slope is ms per output column; one column costs k weight bytes
    gbps = (k / 1e9) / (slope / 1e3)
    return {"k": k, "intercept_ms": float(intercept), "slope_ms_per_col": float(slope),
            "implied_gbps": float(gbps), "n_points": len(points)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="docs/rknpu/results/submit-timing.json")
    args = ap.parse_args()

    dev = Device["ROCKCHIP"]
    clocks_start = clocks()
    rows = []
    print(f"{'m':>3} {'n':>5} {'k':>5} {'c':>2} {'total':>9} {'ioctl':>9} {'host':>7} "
          f"{'ideal':>8} {'GB/s':>7} {'%ceil':>6}  note")
    for m, n, k, cores, note in SWEEP:
        r = measure(dev, m, n, k, cores, args.reps)
        r["note"] = note
        rows.append(r)
        print(f"{m:>3} {n:>5} {k:>5} {cores:>2} {r['total_ms']:>9.3f} {r['internal_ms']:>9.3f} "
              f"{r['host_ms']:>7.3f} {r['ideal_ms']:>8.3f} {r['gbps']:>7.2f} "
              f"{r['pct_ceiling']:>6.1f}  {note}")

    fits = []
    for k in (2560, 4096):
        pts = [r for r in rows if r["k"] == k and r["m"] == 1 and r["cores"] == 3 and r["n"] >= 256]
        if len(pts) >= 3:
            fits.append(fit(pts))
    print()
    for f in fits:
        print(f"fit k={f['k']}: fixed {f['intercept_ms']:.3f} ms + "
              f"{f['implied_gbps']:.2f} GB/s effective ({f['n_points']} points)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reps": args.reps, "ddr_ceiling_gbps": DDR_CEILING_GBPS,
                               "clocks_start": clocks_start, "clocks_end": clocks(),
                               "rows": rows, "fits": fits}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
