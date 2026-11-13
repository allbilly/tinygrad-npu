#!/usr/bin/env python3
"""Is the per-submit fixed cost idle-dependent (NPU power/clock/ramp) or intrinsic?

The driver keeps the NPU powered for `power_put_delay` = 3000 ms after the last ioctl
(rknpu_drv.c: rknpu_dev->power_put_delay = 3000, rknpu_power_put_delay queues the
power-off work only when the refcount is 1). In decode the NPU is used every ~200 us,
so it should never power down mid-run - but if any part of the 34.6 us/submit fixed
cost is wake-up/ramp after an idle gap, a long gap would show it. This probe issues a
real INT8 kernel (same submit path) and measures the NEXT submit latency as a function
of the idle gap, using both busy-spin (CPU hot) and sleep (CPU idle) gaps, and samples
the devfreq/runtime-pm state before and after each next submit.

Arms: gap list in microseconds via busy spin (0, 10, 50, 200, 1000, 5000) and via
sleep (0, 100, 1000, 10000, 3200000 = beyond the 3 s power-off delay). A busier NPU
devfreq state or a >3 s gap that makes the next submit much slower both point at
driver/hardware wake-up rather than per-ioctl work.

HARDWARE: the Lead runs this (single owner); takes ~15 s (2 long-gap iterations).
  python3 /home/orangepi/tg-exp-stream/scripts/probe_submit_idle.py --json-out /tmp/submit_idle.json
"""
from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

TREE = "/home/orangepi/tg-exp-stream"
sys.path.insert(0, TREE)
os.environ.setdefault("DEV", "ROCKCHIP")

from tinygrad import Device  # noqa: E402
from tinygrad.helpers import ceildiv, round_up  # noqa: E402
from tinygrad.runtime.autogen import rockchip as rk  # noqa: E402
from tinygrad.runtime.ops_rockchip import _cmd  # noqa: E402
from tinygrad.runtime.support.rockchip_int8 import int8_body  # noqa: E402

PC, NONBLOCK, PINGPONG = (1 << 0), (1 << 1), (1 << 2)
DEVFD = Path("/sys/class/devfreq/fdab0000.npu")
PMPATH = Path("/sys/devices/platform/fdab0000.npu/power")


def read(path: Path) -> str | None:
  try: return path.read_text().strip()
  except OSError: return None


def sample_state() -> dict:
  return {"npu_cur_freq": read(DEVFD / "cur_freq"), "npu_load": read(DEVFD / "load"),
          "runtime_status": read(PMPATH / "runtime_status"), "autosuspend_delay_ms": read(PMPATH / "autosuspend_delay_ms")}


class Kernel:
  def __init__(self, dev, n: int = 4096, k: int = 4096):
    self.dev = dev
    width = round_up(ceildiv(n, 3), 32)
    parts = tuple((start, min(width, n - start)) for start in range(0, n, width))
    if len(parts) not in (1, 3): parts = ((0, n),)
    lhs, rhs, out = dev.allocate(round_up(4, 64) + k), dev.allocate(n * k), dev.allocate(n * 4)
    bodies = [int8_body(1, count, k, lhs.dma + 64, rhs.dma + start * k, out.dma + start * 4, count)
              for start, count in parts]
    stride = round_up(max(len(body) for body in bodies) + 4, 2)
    cmd = dev.allocate(stride * len(bodies) * 8)
    task = dev.allocate(max(4096, len(bodies) * ctypes.sizeof(rk.struct_rknpu_task)), rk.RKNPU_MEM_KERNEL_MAPPING)
    descs = (rk.struct_rknpu_task * len(bodies)).from_address(task.addr)
    terminal = (_cmd(1, 0), _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS),
                _cmd(rk.TARGET_VERSION, 0), _cmd(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
    for index, body in enumerate(bodies):
      words = array.array("Q", body + terminal)
      ctypes.memmove(cmd.addr + index * stride * 8, words.buffer_info()[0], len(words) * 8)
      descs[index] = rk.struct_rknpu_task(0, 0, 0xd, 0x300, 0x1ffff, 0, len(body), 0, cmd.dma + index * stride * 8)
    self.lhs, self.rhs, self.out, self.cmd, self.task = lhs, rhs, out, cmd, task
    self.tasks, self.cores = len(bodies), len(parts)
    self.core_mask = (1 << self.cores) - 1
    self.subcores = ((0, len(bodies)),) if self.cores == 1 else ((0, 0), (0, 0), (0, 1), (1, 1), (2, 1))
    for buf in (lhs, rhs, out, cmd, task): ctypes.memset(buf.addr, 0, buf.size)
    for buf in (lhs, rhs, out, cmd, task): dev.sync(buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)

  def submit(self) -> float:
    pairs = list(self.subcores) + [(0, 0)] * (5 - len(self.subcores))
    t0 = time.perf_counter()
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd, flags=PC | PINGPONG, timeout=6000, task_start=0,
                              task_number=self.tasks, task_counter=0, priority=0, task_obj_addr=self.task.meta.obj_addr,
                              iommu_domain_id=self.dev.domain, task_base_addr=0, hw_elapse_time=0,
                              core_mask=self.core_mask, fence_fd=-1,
                              subcore_task=(rk.struct_rknpu_subcore_task * 5)(*(rk.struct_rknpu_subcore_task(*p) for p in pairs)))
    return (time.perf_counter() - t0) * 1e3

  def free(self) -> None:
    for buf in (self.cmd, self.task, self.out, self.rhs, self.lhs): self.dev.free(buf)


def busy_wait(seconds: float) -> None:
  end = time.perf_counter() + seconds
  while time.perf_counter() < end: pass


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json-out", default=None)
  ap.add_argument("--iters", type=int, default=12)
  args = ap.parse_args()
  dev = Device["ROCKCHIP"]
  kernel = Kernel(dev)
  report: dict = {"iters": args.iters, "arms": {}, "state_before": sample_state(), "notes": []}
  try:
    for _ in range(3): kernel.submit()  # warm
    arms = report["arms"]
    for gap_us in (0, 10, 50, 200, 1000, 5000):
      values, state = [], None
      for _ in range(args.iters):
        busy_wait(gap_us / 1e6)
        values.append(kernel.submit())
        state = sample_state()
      arms[f"busy_gap_{gap_us}us"] = {"gap_us": gap_us, "median_ms": round(statistics.median(values), 5),
                                      "median_us": round(statistics.median(values) * 1000, 2),
                                      "min_us": round(min(values) * 1000, 2), "max_us": round(max(values) * 1000, 2),
                                      "state_after": state}
    for gap_us in (0, 100, 1000, 10000):
      values, state = [], None
      for _ in range(args.iters):
        time.sleep(gap_us / 1e6)
        values.append(kernel.submit())
        state = sample_state()
      arms[f"sleep_gap_{gap_us}us"] = {"gap_us": gap_us, "median_ms": round(statistics.median(values), 5),
                                       "median_us": round(statistics.median(values) * 1000, 2),
                                       "min_us": round(min(values) * 1000, 2), "max_us": round(max(values) * 1000, 2),
                                       "state_after": state}
    # longer than the driver's 3 s power_put_delay: does the next submit pay power-on?
    values, state = [], None
    for _ in range(2):
      time.sleep(3.3)
      before = sample_state()
      value = kernel.submit()
      values.append(value)
      state = {"before": before, "after": sample_state()}
    arms["sleep_gap_3300000us_power_off"] = {"gap_us": 3300000, "median_ms": round(statistics.median(values), 5),
                                             "median_us": round(statistics.median(values) * 1000, 2),
                                             "min_us": round(min(values) * 1000, 2), "max_us": round(max(values) * 1000, 2),
                                             "state": state}
    baseline = arms["busy_gap_0us"]["median_us"]
    report["derived"] = {
      "baseline_US": baseline,
      "busy_gap_slope_max_us": round(max(a["median_us"] for key, a in arms.items() if key.startswith("busy_gap")) - baseline, 2),
      "sleep_10ms_delta_us": round(arms["sleep_gap_10000us"]["median_us"] - arms["sleep_gap_0us"]["median_us"], 2),
      "after_power_off_delta_us": round(arms["sleep_gap_3300000us_power_off"]["median_us"] - baseline, 2),
      "interpretation": "all deltas ~0 => fixed cost is intrinsic per submit, not idle/wake-up",
    }
    report["state_after"] = sample_state()
  finally:
    kernel.free()
    try: dev.reset()
    except Exception: pass
    dev.finalize()
  print(json.dumps(report, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(report, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
