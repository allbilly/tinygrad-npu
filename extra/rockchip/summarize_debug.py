#!/usr/bin/env python3
"""Summarise a DEBUG=2 tinygrad log: where the time goes, by kernel and by function.

`DEBUG=2` prints two useful line shapes:

  function <key> in <ms> ms: <name>          <- trace + rewrite + schedule, not execution
  *** <dev> <n> <op> <bytes>, <src> <- <dst> arg .. mem .. tm <ms>/<ms> (.. GFLOPS .. GB/s)

The first is the `@function` wrapper cost, which is what a per-block host stall
looks like. The second is actual kernel execution. Splitting them decides whether
a slow block is re-tracing its graph or running slow ops.

Usage: python summarize_debug.py <log> [--top N]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

FUNCTION = re.compile(r"function ([0-9a-f]+) in\s+([0-9.]+) ms: (.+)$")
KERNEL = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+\S+\s+([\d.]+ [KMG]?B),\s+(\S+) <- (\S+).*?tm\s+([0-9.]+)(?:us|ms|s)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def to_ms(value: str, unit_hint: str | None = None) -> float:
    return float(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    functions: list[tuple[float, str, str]] = []
    kernels: dict[tuple[str, str], list[float]] = defaultdict(list)
    kernel_total = 0.0

    with open(args.log, errors="replace") as handle:
        for raw in handle:
            line = ANSI.sub("", raw)
            if (m := FUNCTION.search(line)):
                functions.append((float(m.group(2)), m.group(1), m.group(3)))
                continue
            if "***" not in line:
                continue
            # tm <value><unit> is the authoritative time; the unit varies with magnitude
            tm = re.search(r"tm\s+([0-9.]+)(us|ms|s)\b", line)
            if not tm:
                continue
            value, unit = float(tm.group(1)), tm.group(2)
            ms = value / 1000 if unit == "us" else (value * 1000 if unit == "s" else value)
            head = re.search(r"\*\*\*\s+(\S+)\s+\d+\s+(\S+)", line)
            if not head:
                continue
            kernels[(head.group(1), head.group(2))].append(ms)
            kernel_total += ms

    print(f"function wrapper calls: {len(functions)}")
    if functions:
        total = sum(f[0] for f in functions)
        print(f"  total wrapper time: {total:9.1f} ms   ({total/1000:.2f} s)")
        print(f"  {'ms':>10}  {'key':<10} name")
        for ms, key, name in sorted(functions, reverse=True)[:args.top]:
            print(f"  {ms:10.1f}  {key:<10} {name}")

    print(f"\nkernels: {sum(len(v) for v in kernels.values())} across {len(kernels)} kinds")
    print(f"  total kernel time: {kernel_total:9.1f} ms   ({kernel_total/1000:.2f} s)")
    print(f"  {'ms':>10}  {'count':>6}  {'ms/call':>8}  device / op")
    for (dev, op), times in sorted(kernels.items(), key=lambda kv: -sum(kv[1]))[:args.top]:
        print(f"  {sum(times):10.1f}  {len(times):>6}  {sum(times)/len(times):8.3f}  {dev} / {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
