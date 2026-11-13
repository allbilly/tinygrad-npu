#!/usr/bin/env python3
"""Decompose the per-NPU-submit fixed cost: blocking ioctl vs non-blocking + fence wait.

Every submit today is one BLOCKING ioctl (RKNPU_JOB_PC|PINGPONG, flags=5), and the
fitted fixed cost is ~34.6 us/submit (113 submits/token = ~3.9 ms/token). This probe
splits that number using real prepared INT8 kernels (same register recipes as the
replay) and the SAME driver path:

  * tiny kernel (n=32,k=512, 1 task, 1 core, 16 KB of weights): almost no streaming,
    so its blocking time is the pure per-submit floor (ioctl + driver job setup +
    NPU task start/stop + completion IRQ + wakeup + cleanup).
  * medium kernel (n=4096,k=4096, 3 tasks, 3 cores, 16.8 MB): ~0.52 ms of streaming,
    the real replay submit shape.
  * tiny3 (n=96,k=512, 3 tasks) to separate task-dispatch from the floor.

Arms per kernel:
  block                 PC|PINGPONG blocking ioctl (exactly what the replay does)
  block_no_pingpong     PC only (task_pp_en=0 in the driver's PC_TASK_CONTROL)
  block_priority1       blocking with priority=1 (the driver source ignores priority;
                        this confirms it)
  async_submit_only     NONBLOCK|FENCE_OUT ioctl, record only the ioctl return
                        (= pure host submit path, NPU still running)
  async_fence_busy      NONBLOCK|FENCE_OUT, then userspace busy-polls the dma-fence fd
                        with poll(timeout=0) until readable, then closes it
                        (removes scheduler sleep/wakeup latency entirely)
  async_fence_sleep     NONBLOCK|FENCE_OUT, then blocking select() on the fence fd
                        (same wakeup path as blocking ioctl, but fence-based)
  two_inline            one ioctl carrying TWO independent 3-task kernels chained per
                        core (6 tasks) -> the per-submit cost that our merge removes
  two_separate          the two kernels as two separate blocking ioctls (control)

Output JSON per arm: median/p10/p90/min/max ms and us/submit; for the async arms the
ioctl-only and wait-only components; plus derived terms (submit-path cost, wait
path cost, per-inline-merge saving). Interrupt/power/driver interpretation belongs to
the analysis, this script measures.

HARDWARE: the Lead runs this (single owner). No other benchmark may be live.
Python: /home/orangepi/tinygrad/.venv/bin/python, run from anywhere:
  python3 /home/orangepi/tg-exp-stream/scripts/probe_submit_modes.py \
      --json-out /tmp/submit_modes.json
Runtime: a few seconds after import; each tiny arm 40 iters, medium arms 25.
"""
from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import select
import statistics
import sys
import time

TREE = "/home/orangepi/tg-exp-stream"
sys.path.insert(0, TREE)
os.environ.setdefault("DEV", "ROCKCHIP")

from tinygrad import Device  # noqa: E402
from tinygrad.helpers import ceildiv, round_up  # noqa: E402
from tinygrad.runtime.autogen import rockchip as rk  # noqa: E402
from tinygrad.runtime.ops_rockchip import _cmd  # noqa: E402
from tinygrad.runtime.support.rockchip_int8 import int8_body  # noqa: E402

PC, NONBLOCK, PINGPONG, FENCE_OUT = (1 << 0), (1 << 1), (1 << 2), (1 << 4)


class Kernel:
  """Prepared INT8 matmul: m=1, n columns over 3 (or 1) cores, optional inline chains."""

  def __init__(self, dev, n: int, k: int, chains: int = 1):
    self.dev, self.n, self.k, self.chains = dev, n, k, chains
    width = round_up(ceildiv(n, 3), 32)
    parts = tuple((start, min(width, n - start)) for start in range(0, n, width))
    if len(parts) not in (1, 3): parts = ((0, n),)
    cores = len(parts)
    lhs = dev.allocate(round_up(4, 64) + k)
    rhs = dev.allocate(n * k)
    out = dev.allocate(n * 4 * chains)
    stride = None
    bodies: list[tuple[int, ...]] = []
    for chain in range(chains):
      for start, count in parts:
        bodies.append(int8_body(1, count, k, lhs.dma + 64, rhs.dma + start * k,
                                out.dma + (chain * n + start) * 4, count))
    stride = round_up(max(len(body) for body in bodies) + 4, 2)
    cmd = dev.allocate(stride * len(bodies) * 8)
    task = dev.allocate(max(4096, len(bodies) * ctypes.sizeof(rk.struct_rknpu_task)), rk.RKNPU_MEM_KERNEL_MAPPING)
    # core-major order: [c0-chain0, c0-chain1, ..., c1-chain0, ...]
    self.order = [(core, chain) for core in range(cores) for chain in range(chains)]
    descs = (rk.struct_rknpu_task * len(bodies)).from_address(task.addr)
    for index, (core, chain) in enumerate(self.order):
      body = bodies[chain * cores + core]
      next_same_core = chain + 1 < chains
      tail = ((_cmd(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, (cmd.dma + (index + 1) * stride * 8) & 0xfffffff0),
               _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, ceildiv(len(bodies[(chain + 1) * cores + core]), 2) + 1))
              if next_same_core else
              (_cmd(1, 0), _cmd(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS)))
      words = array.array("Q", body + tail + (_cmd(rk.TARGET_VERSION, 0), _cmd(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd)))
      ctypes.memmove(cmd.addr + index * stride * 8, words.buffer_info()[0], len(words) * 8)
      descs[index] = rk.struct_rknpu_task(0, 0, 0xd, 0x300, 0x1ffff, 0, len(body), 0, cmd.dma + index * stride * 8)
    self.cores, self.cmd, self.task, self.lhs, self.rhs, self.out = cores, cmd, task, lhs, rhs, out
    self.tasks = len(bodies)
    self.core_mask = (1 << cores) - 1
    if cores == 1:
      self.subcores = ((0, self.tasks),)
    else:
      per = chains
      self.subcores = ((0, 0), (0, 0), (0, per), (per, per), (2 * per, per))
    ctypes.memset(lhs.addr, 0, lhs.size)
    ctypes.memset(rhs.addr, 0, rhs.size)
    for buf in (lhs, rhs, out, cmd, task):
      if buf is not lhs and buf is not rhs: ctypes.memset(buf.addr, 0, buf.size)
      dev.sync(buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)

  def subcore_struct(self):
    pairs = list(self.subcores) + [(0, 0)] * (5 - len(self.subcores))
    return (rk.struct_rknpu_subcore_task * 5)(*(rk.struct_rknpu_subcore_task(*pair) for pair in pairs))

  def submit(self, flags: int = PC | PINGPONG, timeout: int = 6000, priority: int = 0):
    return rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd, flags=flags, timeout=timeout, task_start=0,
                                     task_number=self.tasks, task_counter=0, priority=priority,
                                     task_obj_addr=self.task.meta.obj_addr, iommu_domain_id=self.dev.domain,
                                     task_base_addr=0, hw_elapse_time=0, core_mask=self.core_mask, fence_fd=-1,
                                     subcore_task=self.subcore_struct())

  def blocking(self, flags: int = PC | PINGPONG, priority: int = 0) -> float:
    t0 = time.perf_counter()
    self.submit(flags=flags, priority=priority)
    return (time.perf_counter() - t0) * 1e3

  def async_fence(self, busy: bool, timeout_ms: int = 6000) -> tuple[float, float]:
    """(ioctl_ms, wait_ms); closes the fence fd."""
    t0 = time.perf_counter()
    made = self.submit(flags=PC | PINGPONG | NONBLOCK | FENCE_OUT)
    ioctl_ms = (time.perf_counter() - t0) * 1e3
    fd = int(made.fence_fd)
    if fd < 0: raise RuntimeError(f'fence_fd not returned ({fd}) - RKNPU fence config may be disabled')
    t1 = time.perf_counter()
    if busy:
      poller = select.poll()
      poller.register(fd, select.POLLIN)
      deadline = t1 + timeout_ms / 1e3
      while True:
        if poller.poll(0):
          break
        if time.perf_counter() > deadline:
          os.close(fd)
          raise TimeoutError('fence poll timeout')
    else:
      ready, _, _ = select.select([fd], [], [], timeout_ms / 1e3)
      if not ready:
        os.close(fd)
        raise TimeoutError('fence select timeout')
    wait_ms = (time.perf_counter() - t1) * 1e3
    os.close(fd)
    return ioctl_ms, wait_ms

  def free(self) -> None:
    for buf in (self.cmd, self.task, self.out, self.rhs, self.lhs): self.dev.free(buf)


def stats(values: list[float]) -> dict:
  ordered = sorted(values)
  n = len(ordered)
  return {"iters": n, "median_ms": round(statistics.median(values), 5),
          "mean_ms": round(statistics.mean(values), 5),
          "p10_ms": round(ordered[max(0, n // 10)], 5), "p90_ms": round(ordered[min(n - 1, 9 * n // 10)], 5),
          "min_ms": round(ordered[0], 5), "max_ms": round(ordered[-1], 5),
          "outliers_gt_1_2x_median": sum(1 for value in values if value > 1.2 * statistics.median(values))}


def time_repeat(fn, warmup: int, iters: int) -> dict:
  for _ in range(warmup): fn()
  values = [fn() for _ in range(iters)]
  row = stats(values)
  row["us_per_submit"] = round(row["median_ms"] * 1000, 2)
  return row


def time_async(kernel: Kernel, busy: bool, warmup: int, iters: int) -> dict:
  for _ in range(warmup): kernel.async_fence(busy=busy)
  ioctls, waits = [], []
  for _ in range(iters):
    ioctl_ms, wait_ms = kernel.async_fence(busy=busy)
    ioctls.append(ioctl_ms)
    waits.append(wait_ms)
  totals = [a + b for a, b in zip(ioctls, waits)]
  row = stats(totals)
  row.update({"ioctl_median_us": round(statistics.median(ioctls) * 1000, 2),
              "wait_median_us": round(statistics.median(waits) * 1000, 2),
              "us_per_submit": round(row["median_ms"] * 1000, 2)})
  return row


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json-out", default=None)
  ap.add_argument("--tiny-iters", type=int, default=40)
  ap.add_argument("--med-iters", type=int, default=25)
  ap.add_argument("--tree", default=TREE)
  args = ap.parse_args()
  if args.tree != TREE:
    sys.path.insert(0, args.tree)
  dev = Device["ROCKCHIP"]
  kernels: list[Kernel] = []
  report: dict = {"tree": TREE, "arms": {}, "notes": []}
  try:
    tiny1 = Kernel(dev, n=32, k=512)
    tiny3 = Kernel(dev, n=96, k=512)
    med = Kernel(dev, n=4096, k=4096)
    kernels += [tiny1, tiny3, med]
    for name, kernel, iters in (("tiny1", tiny1, args.tiny_iters), ("tiny3", tiny3, args.tiny_iters),
                                ("med", med, args.med_iters)):
      arms: dict[str, dict] = {}
      arms["block"] = time_repeat(kernel.blocking, 3, iters)
      arms["block_no_pingpong"] = time_repeat(lambda k=kernel: k.blocking(flags=PC), 3, iters)
      arms["block_priority1"] = time_repeat(lambda k=kernel: k.blocking(priority=1), 3, iters)
      try:
        arms["async_fence_busy"] = time_async(kernel, True, 3, iters)
        arms["async_fence_sleep"] = time_async(kernel, False, 3, iters)
      except Exception as exc:  # fence unsupported: record and continue
        report["notes"].append(f"{name}: fence path failed: {type(exc).__name__}: {exc}")
      report["arms"][name] = arms
    # 2-inline vs 2-separate: 6 tasks in one ioctl vs two 3-task ioctls (same domain)
    two = Kernel(dev, n=4096, k=4096, chains=2)
    kernels.append(two)
    inline = time_repeat(lambda: two.blocking(), 3, args.med_iters)
    separate_values = []
    for _ in range(3):
      med.blocking()
      med.blocking()
    for _ in range(args.med_iters):
      t0 = time.perf_counter()
      med.blocking()
      med.blocking()
      separate_values.append((time.perf_counter() - t0) * 1e3)
    separate = stats(separate_values)
    report["arms"]["two_inline_vs_two_separate"] = {
      "two_inline": {**inline, "us_per_submit": round(inline["median_ms"] * 500, 2)},
      "two_separate": {**separate, "us_per_submit": round(separate["median_ms"] * 500, 2)},
      "saving_ms_per_merged_pair": round(separate["median_ms"] - inline["median_ms"], 5)}
    # derived decomposition (tiny1 = no streaming, so it is the per-submit floor)
    def micro(row): return row["us_per_submit"]
    t = report["arms"]["tiny1"]
    derived = {
      "empty_submit_blocking_us": micro(t["block"]),
      "no_pingpong_delta_us": round(micro(t["block_no_pingpong"]) - micro(t["block"]), 2),
      "priority_delta_us": round(micro(t["block_priority1"]) - micro(t["block"]), 2),
    }
    if "async_fence_busy" in t:
      derived["async_ioctl_only_us"] = t["async_fence_busy"]["ioctl_median_us"]
      derived["async_fence_busy_us"] = micro(t["async_fence_busy"])
      derived["async_fence_sleep_us"] = micro(t["async_fence_sleep"])
      derived["wakeup_path_saving_us"] = round(micro(t["block"]) - micro(t["async_fence_busy"]), 2)
      derived["notification_us"] = round(t["async_fence_busy"]["wait_median_us"], 2)
      derived["async_total_minus_blocking_us"] = round(micro(t["async_fence_busy"]) - micro(t["block"]), 2)
    derived["tiny3_tasks3_minus_tiny1_task1_us"] = round(micro(report["arms"]["tiny3"]["block"]) - micro(t["block"]), 2)
    derived["med_blocking_us"] = micro(report["arms"]["med"]["block"])
    derived["med_minus_tiny1_us"] = round(derived["med_blocking_us"] - derived["empty_submit_blocking_us"], 2)
    report["derived"] = derived
  finally:
    for kernel in reversed(kernels): kernel.free()
    try: dev.reset()
    except Exception: pass
    dev.finalize()
  print(json.dumps(report, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(report, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
