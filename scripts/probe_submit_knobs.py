#!/usr/bin/env python3
"""Passive dump of every NPU driver/sysfs knob that could affect per-submit latency.

Never opens the NPU: only reads /sys, /proc/interrupts, /proc/cmdline, modinfo and
(if permitted) dmesg/debugfs. Answers for the 34.6 us/submit investigation:
  * which devfreq governor/frequencies are active and whether cur_freq is pinned;
  * the platform device runtime-PM state: control, runtime_status, autosuspend delay
    (the driver's custom power_put_delay is 3000 ms and is NOT this knob - see
    rknpu_dev->power_put_delay in rknpu_drv.c);
  * module params exposed by rknpu (modinfo) and the built-in driver version;
  * the three NPU GIC interrupt lines and their idle rate;
  * any debugfs rknpu nodes (power/delayms/freq) and whether they are writable by the
    current user (the driver's rknpu_debugger.c exposes power and delayms controls);
  * iommu/domain configuration evidence from the kernel command line / dmesg.

  python3 /home/orangepi/tg-exp-stream/scripts/probe_submit_knobs.py --json-out /tmp/submit_knobs.json
Runtime: <2 s. Safe to run at any time (passive).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

NPU_DEV = Path("/sys/class/devfreq/fdab0000.npu")
PMPATH = Path("/sys/devices/platform/fdab0000.npu/power")
MODULE = Path("/sys/module/rknpu")
DEBUGFS = Path("/sys/kernel/debug/rknpu")
DMC = Path("/sys/class/devfreq/dmc")


def read(path: Path, limit: int = 4000) -> str | None:
  try: return path.read_text(errors="replace").strip()[:limit]
  except OSError: return None


def dump_dir(path: Path) -> dict:
  out = {}
  try: names = sorted(entry.name for entry in path.iterdir())
  except OSError: return out
  for name in names:
    text = read(path / name)
    if text is not None: out[name] = text
  return out


def npu_irq_counts() -> dict[str, int]:
  out = {}
  try: lines = Path("/proc/interrupts").read_text().splitlines()
  except OSError: return out
  for line in lines:
    if "fdab0000.npu" not in line: continue
    name = line.split(":", 1)[1].split("GICv3")[-1].strip() if "GICv3" in line else "?"
    head = line.split(":", 1)[1].split() if "GICv3" not in line else line.split(":", 1)[1].split("GICv3")[0].split()
    counts = [int(token) for token in head if token.isdigit()]
    out[f"{line.split(':')[0].strip()} {name}"] = sum(counts)
  return out


def run(cmd: list[str]) -> str | None:
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    return (result.stdout + result.stderr).strip()[:4000]
  except (OSError, subprocess.TimeoutExpired):
    return None


def debugfs_report() -> dict:
  try:
    exists = DEBUGFS.exists()
  except OSError:
    return {"exists": None, "accessible": False, "entries": [], "contents": {},
            "error": "permission denied (debugfs mounted but not readable by this user)"}
  if not exists:
    return {"exists": False, "accessible": False, "entries": [], "contents": {}}
  try:
    entries = sorted(entry.name for entry in DEBUGFS.iterdir())
  except OSError as exc:
    return {"exists": True, "accessible": False, "entries": [], "contents": {}, "error": f"{type(exc).__name__}: {exc}"}
  return {"exists": True, "accessible": True, "entries": entries, "contents": dump_dir(DEBUGFS)}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()
  irq0 = npu_irq_counts()
  time.sleep(0.25)
  irq1 = npu_irq_counts()
  report = {
    "npu_devfreq": dump_dir(NPU_DEV),
    "npu_platform_power": dump_dir(PMPATH),
    "dmc_devfreq": dump_dir(DMC),
    "rknpu_module_params": dump_dir(MODULE / "parameters") if MODULE.exists() else {},
    "rknpu_module_version": read(MODULE / "version") if MODULE.exists() else None,
    "debugfs_rknpu": debugfs_report(),
    "module_params_modinfo": run(["modinfo", "rknpu"]),
    "interrupts_npu": {"before": irq0, "after_250ms": irq1,
                       "idle_rate_per_s": {key: round((irq1.get(key, 0) - value) / 0.25, 1) for key, value in irq0.items()}},
    "dmesg_rknpu": run(["sh", "-c", "dmesg 2>/dev/null | grep -i -E 'rknpu|iommu' | tail -40"]),
    "cmdline": read(Path("/proc/cmdline")),
    "thermal_zone0_c": read(Path("/sys/class/thermal/thermal_zone0/temp")),
    "notes": [
      "power_put_delay is a rknpu_driver variable (3000 ms default), not power/autosuspend_delay_ms",
      "rknpu_debugger exposes /sys/kernel/debug/rknpu/{power,delayms,freq} when debugfs is mounted",
      "devfreq load is '<pct>@<freq>Hz' per core or aggregate",
    ],
  }
  print(json.dumps(report, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(report, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
