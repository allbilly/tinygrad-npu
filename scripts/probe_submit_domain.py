#!/usr/bin/env python3
"""Does an iommu-domain switch between consecutive submits cost time? (the 34.6 us suspect)

The driver (`rknpu_iommu_domain_get_and_switch` in rknpu_iommu.c) only re-attaches the
iommu domain when the requested `iommu_domain_id` differs from the current one AND
the domain refcount is zero. Re-attach is `iommu_detach_device` + `iommu_attach_device`
(arm-smmu domain switch, TLB handling), so if it is expensive it is paid once per
submit whose buffers live in a different iommu domain than the previous submit.

The shipped decode configuration uses 3 weight domains (ONPU_NPU_DOMAINS=3) and, per
layer, the submit order is qkv(dom0) -> attn_out(dom1) -> gate+up(dom0) -> ffn_down(dom1):
~113 submits/token that alternate domain on almost every submit. This probe measures
the cost directly: identical INT8 kernels are prepared in domain 0/1/2 and issued in
same-domain vs alternating-domain sequence, with the same blocking ioctl the replay uses.

Arms (kernel = n=4096,k=4096 over 3 cores = 16.8 MB; tiny = n=32,k=512 = 16 KB):
  same_med         [0,0,0,...]                 no switch                (control)
  alt_med01        [0,1,0,1,...]               switch every submit
  alt_med012       [0,1,2,0,1,2,...]           switch every submit, 2 different targets
  pair_med01       [0,0,1,1,0,0,1,1,...]       switch every 2nd submit
  same_tiny        [t0,t0,...]                 tiny payload, no switch   (control)
  alt_tiny01       [t0,t1,t0,t1,...]           tiny payload, switch every submit
  alt_tiny01_async [t0,t1,...] nonblocking+fence busy-poll wait
Derived: switch cost = median(alternating) - median(same) for the same kernel shape.
`--same-iters` repeats each arm; per-submit medians are reported (no averaging games).

HARDWARE: the Lead runs this (single owner), no benchmark may be live.
  python3 /home/orangepi/tg-exp-stream/scripts/probe_submit_domain.py --json-out /tmp/submit_domain.json
Runtime: ~10-20 s (3 domains open, ~16.8 MB per domain, few hundred submits).
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
  def __init__(self, dev, n: int, k: int):
    self.dev, self.n, self.k = dev, n, k
    width = round_up(ceildiv(n, 3), 32)
    parts = tuple((start, min(width, n - start)) for start in range(0, n, width))
    if len(parts) not in (1, 3): parts = ((0, n),)
    cores = len(parts)
    lhs = dev.allocate(round_up(4, 64) + k)
    rhs = dev.allocate(n * k)
    out = dev.allocate(n * 4)
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
    self.bufs, self.cmd, self.task, self.lhs, self.rhs, self.out = (lhs, rhs, out, cmd, task), cmd, task, lhs, rhs, out
    self.tasks, self.cores = len(bodies), cores
    self.core_mask = (1 << cores) - 1
    self.subcores = ((0, self.tasks),) if cores == 1 else ((0, 0), (0, 0), (0, 1), (1, 1), (2, 1))
    for buf in self.bufs: ctypes.memset(buf.addr, 0, buf.size)
    for buf in self.bufs: dev.sync(buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)

  def subcore_struct(self):
    pairs = list(self.subcores) + [(0, 0)] * (5 - len(self.subcores))
    return (rk.struct_rknpu_subcore_task * 5)(*(rk.struct_rknpu_subcore_task(*pair) for pair in pairs))

  def submit(self, flags: int = PC | PINGPONG):
    return rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd, flags=flags, timeout=6000, task_start=0,
                                     task_number=self.tasks, task_counter=0, priority=0,
                                     task_obj_addr=self.task.meta.obj_addr, iommu_domain_id=self.dev.domain,
                                     task_base_addr=0, hw_elapse_time=0, core_mask=self.core_mask, fence_fd=-1,
                                     subcore_task=self.subcore_struct())

  def blocking(self) -> float:
    t0 = time.perf_counter()
    self.submit()
    return (time.perf_counter() - t0) * 1e3

  def async_busy(self) -> float:
    t0 = time.perf_counter()
    made = self.submit(flags=PC | PINGPONG | NONBLOCK | FENCE_OUT)
    fd = int(made.fence_fd)
    if fd < 0: raise RuntimeError('no fence fd (fence config disabled?)')
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while not poller.poll(0): pass
    os.close(fd)
    return (time.perf_counter() - t0) * 1e3

  def free(self) -> None:
    for buf in (self.cmd, self.task, self.out, self.rhs, self.lhs): self.dev.free(buf)


def run_sequence(kernels: list[Kernel], pattern: list[int], rounds: int, mode: str = "blocking") -> dict:
  """Round-robin through `pattern` (indices into kernels), timing each submit."""
  names = {id(kernel): name for name, kernel in ((f"k{index}", kernel) for index, kernel in enumerate(kernels))}
  times: list[list[float]] = [[] for _ in range(len(pattern))]
  for _ in range(2):  # warmup pass over the whole pattern
    for index in pattern:
      kernels[index].blocking() if mode == "blocking" else kernels[index].async_busy()
  for _ in range(rounds):
    for slot, index in enumerate(pattern):
      kernel = kernels[index]
      times[slot].append(kernel.blocking() if mode == "blocking" else kernel.async_busy())
  medians = [round(statistics.median(samples), 5) for samples in times]
  return {"pattern_len": len(pattern), "rounds": rounds, "mode": mode,
          "median_ms_per_slot": medians,
          "median_ms": round(statistics.median([value for samples in times for value in samples]), 5),
          "us_per_submit": round(statistics.median([value for samples in times for value in samples]) * 1000, 2),
          "per_slot_us": [round(m * 1000, 2) for m in medians], "names": [names[id(kernels[index])] for index in pattern]}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json-out", default=None)
  ap.add_argument("--rounds", type=int, default=30)
  args = ap.parse_args()
  devices = [Device["ROCKCHIP"], Device["ROCKCHIP:1"], Device["ROCKCHIP:2"]]
  report: dict = {"domains": [dev.domain for dev in devices], "arms": {}, "notes": []}
  kernels: list[Kernel] = []
  try:
    med = [Kernel(dev, n=4096, k=4096) for dev in devices]
    tiny = [Kernel(dev, n=32, k=512) for dev in devices]
    kernels += med + tiny
    arms = report["arms"]
    arms["same_med"] = run_sequence(med, [0] * 4, args.rounds)
    arms["alt_med01"] = run_sequence(med, [0, 1] * 2, args.rounds)
    arms["alt_med012"] = run_sequence(med, [0, 1, 2, 0, 1, 2], args.rounds)
    arms["pair_med01"] = run_sequence(med, [0, 0, 1, 1] * 2, args.rounds)
    arms["same_tiny"] = run_sequence(tiny, [0] * 4, args.rounds)
    arms["alt_tiny01"] = run_sequence(tiny, [0, 1] * 2, args.rounds)
    try:
      arms["alt_tiny01_async"] = run_sequence(tiny, [0, 1] * 2, max(10, args.rounds // 2), mode="async")
    except Exception as exc:
      report["notes"].append(f"async domain arm failed: {type(exc).__name__}: {exc}")
    derived = {
      "switch_cost_med_us": round(arms["alt_med01"]["us_per_submit"] - arms["same_med"]["us_per_submit"], 2),
      "switch_cost_tiny_us": round(arms["alt_tiny01"]["us_per_submit"] - arms["same_tiny"]["us_per_submit"], 2),
      "switch_cost_med012_us": round(arms["alt_med012"]["us_per_submit"] - arms["same_med"]["us_per_submit"], 2),
      "switch_cost_pair_us": round(arms["pair_med01"]["us_per_submit"] - arms["same_med"]["us_per_submit"], 2),
      "estimated_switches_per_token_shipped": 113,
      "estimated_saving_ms_if_2_switches_per_token": round((113 - 2) / 1000 *
        (arms["alt_med01"]["us_per_submit"] - arms["same_med"]["us_per_submit"]), 3),
    }
    report["derived"] = derived
  finally:
    for kernel in reversed(kernels): kernel.free()
    try: devices[0].reset()
    except Exception: pass
    for dev in reversed(devices):
      try: dev.finalize()
      except Exception: pass
  print(json.dumps(report, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(report, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
