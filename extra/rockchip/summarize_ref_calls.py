#!/usr/bin/env python3
"""Summarise the reference replay's per-call profile.

`ONPU_REPLAY_PROFILE=1` wraps every statement in the generated C with `now()` and
accumulates per-label nanoseconds, but the text path never prints them. The only
channel that survives the worker subprocess is `ONPU_REPLAY_CALLS_OUT`, which writes
one JSON line per replay call: {"calls": [[label, ns], ...]}.

This aggregates by label: total, call count, mean, and share. Use it to see which
labels own the replay, not to quote a rate — profiling adds a clock read around every
op, so the timings are inflated.

Usage: python extra/rockchip/summarize_ref_calls.py <calls.jsonl> [--top N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    lines = 0
    with open(args.path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lines += 1
            for label, ns in record.get("calls", []):
                totals[label] += int(ns)
                counts[label] += 1

    if not lines:
        print(f"no records in {args.path}")
        return 1

    grand = sum(totals.values())
    print(f"{lines} replay calls, {len(totals)} labels, {grand/1e9:.3f} s total\n")
    print(f"{'share':>6} {'total ms':>10} {'calls':>7} {'mean ms':>9}  label")
    for label, ns in sorted(totals.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{100*ns/grand:5.1f}% {ns/1e6:10.1f} {counts[label]:7d} "
              f"{ns/counts[label]/1e6:9.3f}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
