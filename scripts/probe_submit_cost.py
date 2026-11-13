#!/usr/bin/env python3
"""Microbenchmark: what exactly does one extra NPU submit cost, and is it idle-dependent?

Uses real prepared INT8 kernels (same UOp PROGRAM path as scripts/probe_single_gemm.py
in /home/orangepi/oNPU) and times BLOCKING submits, so the number is
`submit + execute + return`, exactly the per-submit interval the replay fit is built on.

Arms (all use the same core_mask=7, one task per core unless stated):
  A  n=8192 k=4096, cores=3 : 1 submit, 33.55 MB          (back-to-back)
  B  n=8192 k=2048, cores=3 : 2 submits per iteration, same 33.55 MB
  C  n=8192 k=1024, cores=3 : 4 submits per iteration, same 33.55 MB
  D  n=2752 k=4096, cores=1 : 1 submit, 11.26 MB, 1 task/1 core
  E  arm A preceded by a busy-spin idle gap of 0/100/400/1000 us

What it answers (hypotheses H-submit / H-idle):
  * per-submit cost: (time_B - time_A) = one extra ioctl (plus 3 extra body
    fetches, so it is an upper bound if the flat cost is per task; A vs C gives
    3 extra ioctls);
  * idle dependence: compare E[gap] - E[0]. A nonzero slope means the 34.6 us
    includes NPU wake-up after idle, which a pipeline could hide; a flat line
    means it is inherent per-submit cost and only fewer submits help;
  * core scaling: D's per-submit cost vs A's, and GB/s of a 1-core submit.

It also reports the per-call distribution (p10/p90/outliers): if isolated multi-ms
stalls appear here with no CPU work interleaved, they are NPU/driver-internal.

HARDWARE: uses the NPU. The Lead runs this (single-owner), never concurrently with
a benchmark. Typical runtime <15 s after the device is free.

    PYTHONPATH=/home/orangepi/tg-exp-stream /home/orangepi/tinygrad/.venv/bin/python \
      /home/orangepi/tg-exp-stream/scripts/probe_submit_cost.py --json-out /tmp/submit_cost.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import time

BUS_GBPS = 33.79  # LPDDR4x 2112 MHz 64-bit theoretical ceiling, used only as a sanity floor


def round_up(v: int, a: int) -> int:
  return (v + a - 1) // a * a


def build_plan(n: int, k: int, cores: int):
  """Prepared INT8 Rockchip plan for m=1, n, k, one task per core."""
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
  from tinygrad.helpers import Target
  from tinygrad.runtime.ops_rockchip import RockchipProgram
  from dataclasses import replace
  payload = Tensor.zeros(round_up(1 * 4, 64) + 1 * k, device="CPU", dtype=dtypes.uint8).realize()
  weight = Tensor.zeros(n // 32, k // 32, 32, 32, device="ROCKCHIP", dtype=dtypes.int8).realize()
  out = Tensor.empty(1 * n, device="ROCKCHIP", dtype=dtypes.int32).realize()
  sink = UOp.sink(out.uop, payload.to("ROCKCHIP").uop, weight.uop, arg=KernelInfo(f"probe_submit_{n}_{k}_{cores}"))
  info = replace(ProgramInfo.from_sink(sink, Target("ROCKCHIP")), outs=(0,), ins=(1, 2))
  descriptor = UOp(Ops.BINARY, arg=struct.pack("<8I", 1, n, k, cores, 0, round_up(1 * 4, 64), 0, 1))
  plan = RockchipProgram(UOp(Ops.PROGRAM, arg=info,
                             src=(sink, UOp(Ops.LINEAR, src=(out.uop, payload.to("ROCKCHIP").uop, weight.uop)),
                                  UOp(Ops.SOURCE, arg=f"INT8 1x{k}x{n}"), descriptor)))
  args = (out, payload.to("ROCKCHIP"), weight)
  if hasattr(plan, "prepare"): plan.prepare(*args)
  for _ in range(3): plan(*args, wait=True)  # warm: allocations, command build, cache
  return plan, args


def time_plan(plan, args, iters: int, batch: int = 1, gap_us: float = 0.0) -> dict:
  samples = []
  for _ in range(iters):
    if gap_us:
      target = time.perf_counter() + gap_us / 1e6
      while time.perf_counter() < target: pass
    t0 = time.perf_counter()
    for _ in range(batch): plan(*args, wait=True)
    samples.append((time.perf_counter() - t0) * 1e3)
  ordered = sorted(samples)
  median = statistics.median(samples)
  return {"iters": iters, "batch": batch, "gap_us": gap_us,
          "median_ms": round(median, 5), "mean_ms": round(statistics.mean(samples), 5),
          "p10_ms": round(ordered[len(ordered) // 10], 5), "p90_ms": round(ordered[9 * len(ordered) // 10], 5),
          "min_ms": round(ordered[0], 5), "max_ms": round(ordered[-1], 5),
          "outliers_gt_1_2x_median": sum(1 for x in samples if x > 1.2 * median),
          "per_submit_ms": round(median / batch, 5)}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=25)
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()
  import os
  os.environ.setdefault("DEV", "ROCKCHIP")
  import sys
  sys.path.insert(0, "/home/orangepi/tg-exp-stream")
  from tinygrad.device import Device  # noqa: F401  (device must exist before plans are built)

  out = {"bus_gbps_floor": BUS_GBPS, "arms": {}}
  plans = {}
  for name, (n, k, cores, batch) in {
      "A_1submit_3task_33MB": (8192, 4096, 3, 1),
      "B_2submits_3task_33MB": (8192, 2048, 3, 2),
      "C_4submits_3task_33MB": (8192, 1024, 3, 4),
      "D_1submit_1task_11MB": (2752, 4096, 1, 1)}.items():
    plans[name] = (*build_plan(n, k, cores), batch)
  for gap in (0.0, 100.0, 400.0, 1000.0):
    plans[f"E_1submit_gap{gap:.0f}us"] = (*build_plan(8192, 4096, 3), 1)

  for name, (plan, plan_args, batch) in plans.items():
    gap_us = 0.0
    if name.startswith("E_"):
      gap_us = float(name.split("gap")[1].split("us")[0])
    row = time_plan(plan, plan_args, args.iters, batch=batch, gap_us=gap_us)
    mb_per_iteration = {"A_1submit_3task_33MB": 33.55, "B_2submits_3task_33MB": 33.55,
                        "C_4submits_3task_33MB": 33.55, "D_1submit_1task_11MB": 11.26,
                        "E_1submit_gap0us": 33.55, "E_1submit_gap100us": 33.55,
                        "E_1submit_gap400us": 33.55, "E_1submit_gap1000us": 33.55}[name]
    row["mb_per_iteration"] = mb_per_iteration
    row["gbps"] = round(mb_per_iteration / row["median_ms"] * 1000, 3) if row["median_ms"] else 0
    row["floor_ms"] = round(mb_per_iteration / BUS_GBPS * 1000, 4)
    row["below_physical_floor"] = row["median_ms"] < row["floor_ms"] * 0.5
    out["arms"][name] = row

  a = out["arms"]
  derived = {}
  if "B_2submits_3task_33MB" in a and "A_1submit_3task_33MB" in a:
    derived["us_per_extra_submit_B_minus_A"] = round(
        (a["B_2submits_3task_33MB"]["median_ms"] - a["A_1submit_3task_33MB"]["median_ms"]) * 1000, 2)
  if "C_4submits_3task_33MB" in a and "A_1submit_3task_33MB" in a:
    derived["us_per_extra_submit_C_minus_A_over_3"] = round(
        (a["C_4submits_3task_33MB"]["median_ms"] - a["A_1submit_3task_33MB"]["median_ms"]) * 1000 / 3, 2)
  gap0 = a.get("E_1submit_gap0us", {}).get("median_ms")
  if gap0:
    derived["idle_gap_us_delta"] = {
        g: round((a[f"E_1submit_gap{g:.0f}us"]["median_ms"] - gap0) * 1000, 2) for g in (100, 400, 1000)}
  out["derived"] = derived
  print(json.dumps(out, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
