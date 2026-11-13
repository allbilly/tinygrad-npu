#!/usr/bin/env python3
"""Passive NPU clock / load / interrupt sampler for a benchmark the Lead runs.

Does NOT touch the NPU: it only reads /sys, /proc/interrupts and the thermal zone.
Run it in a second shell while an official benchmark runs:

    python3 scripts/probe_npu_clock_sampler.py --seconds 180 --out /tmp/sample.json
    python3 scripts/probe_npu_clock_sampler.py --while-pid <benchmark_pid> --out /tmp/sample.json

Prints JSON: per-domain frequency histogram, every 800 MHz episode (start/duration),
load stats, DRAM (dmc) frequency, CPU policy frequencies, thermal samples, and the
NPU GIC interrupt delta per second. That answers, cheaply:

  H-clock: are there real devfreq 800 MHz windows during decode, how long, and what
           fraction of the run? (The recorded per-token profile says slow tokens are
           LOCALIZED single-submit stalls, not uniform 2x-frequency windows, so the
           expected answer is "no sustained 800 MHz windows".)

Use with --while-pid so sampling starts/stops exactly with the measured run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

NPU_LOAD = Path("/sys/class/devfreq/fdab0000.npu/load")
NPU_FREQ = Path("/sys/class/devfreq/fdab0000.npu/cur_freq")
NPU_MIN = Path("/sys/class/devfreq/fdab0000.npu/min_freq")
NPU_MAX = Path("/sys/class/devfreq/fdab0000.npu/max_freq")
DMC_FREQ = Path("/sys/class/devfreq/dmc/cur_freq")
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
CPU_FREQS = [Path(f"/sys/devices/system/cpu/cpufreq/policy{i}/scaling_cur_freq") for i in (0, 4, 6)]


def read_int(path: Path) -> int:
  try: return int(path.read_text().strip())
  except (OSError, ValueError): return -1


def read_npu_load() -> tuple[int, int]:
  """(max per-core load %, freq Hz). The driver formats load as '<pct>[@<freq>Hz]'."""
  try: text = NPU_LOAD.read_text().strip()
  except OSError: return -1, -1
  body, _, tail = text.partition("@")
  loads = []
  for part in body.replace(",", " ").split():
    try: loads.append(int(float(part.rstrip("%"))))
    except ValueError: pass
  freq = -1
  if tail:
    try: freq = int(tail.replace("Hz", "").strip())
    except ValueError: pass
  return (max(loads) if loads else -1), freq


def npu_interrupts() -> int:
  total = 0
  try:
    for line in Path("/proc/interrupts").read_text().splitlines():
      if "fdab0000.npu" not in line: continue
      total += sum(int(x) for x in line.split(":", 1)[1].split("GICv3")[0].split() if x.isdigit())
  except OSError: return -1
  return total


def pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError: return False
  except PermissionError: return True


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--interval-ms", type=float, default=10.0)
  ap.add_argument("--while-pid", type=int, default=None, help="sample until this process exits")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  interval = max(1.0, args.interval_ms) / 1000.0
  samples = []  # (t, npu_load, npu_freq, dmc_freq, temp_c, cpu0, cpu4, cpu6, npu_irqs)
  start = time.monotonic()
  irq_start = npu_interrupts()
  while True:
    t = time.monotonic() - start
    if args.while_pid is not None and not pid_alive(args.while_pid): break
    if args.while_pid is None and t >= args.seconds: break
    if args.while_pid is not None and t >= args.seconds * 4: break  # guard against a stuck pid
    load, freq = read_npu_load()
    samples.append((round(t, 4), load, freq, read_int(DMC_FREQ), read_int(ZONE) // 1000,
                    read_int(CPU_FREQS[0]), read_int(CPU_FREQS[1]), read_int(CPU_FREQS[2]), npu_interrupts()))
    time.sleep(interval)
  wall = time.monotonic() - start
  irq_end = npu_interrupts()

  freqs: dict[str, int] = {}
  dmc: dict[str, int] = {}
  loads = []
  temps = []
  cpu_tail = {"policy0": set(), "policy4": set(), "policy6": set()}
  episodes = []
  for load, freq, dmc_freq, temp, c0, c4, c6 in ((s[1], s[2], s[3], s[4], s[5], s[6], s[7]) for s in samples):
    freqs[str(freq)] = freqs.get(str(freq), 0) + 1
    dmc[str(dmc_freq)] = dmc.get(str(dmc_freq), 0) + 1
    if load >= 0: loads.append(load)
    if temp >= 0: temps.append(temp)
    cpu_tail["policy0"].add(c0)
    cpu_tail["policy4"].add(c4)
    cpu_tail["policy6"].add(c6)
  # 800 MHz episodes by index scan
  sfreq = [s[2] for s in samples]
  for i, f in enumerate(sfreq):
    if f == 800_000_000 and (i == 0 or sfreq[i - 1] != 800_000_000): episodes.append({"start_s": samples[i][0], "samples": 0})
    if f == 800_000_000 and episodes: episodes[-1]["samples"] += 1
  for ep in episodes: ep["duration_ms"] = round(ep["samples"] * interval * 1000, 1)
  n = len(samples)
  out = {
    "samples": n, "wall_s": round(wall, 3), "interval_ms": args.interval_ms,
    "npu_freq_share": {k: round(v / n, 4) for k, v in sorted(freqs.items())},
    "dmc_freq_share": {k: round(v / n, 4) for k, v in sorted(dmc.items())},
    "npu_freq_800_episodes": episodes,
    "npu_freq_800_share": round(sum(1 for f in sfreq if f == 800_000_000) / n, 4),
    "npu_load_mean": round(sum(loads) / len(loads), 1) if loads else -1,
    "npu_load_nonzero_share": round(sum(1 for x in loads if x > 0) / len(loads), 4) if loads else -1,
    "temp_c": {"min": min(temps) if temps else -1, "max": max(temps) if temps else -1,
               "mean": round(sum(temps) / len(temps), 1) if temps else -1},
    "cpu_freq_seen": {k: sorted(v) for k, v in cpu_tail.items()},
    "npu_irq_per_s": round((irq_end - irq_start) / wall, 1) if irq_start >= 0 and wall > 0 else -1,
    "pins": {"npu_min_hz": read_int(NPU_MIN), "npu_max_hz": read_int(NPU_MAX)},
  }
  print(json.dumps(out, indent=1))
  if args.out: Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
