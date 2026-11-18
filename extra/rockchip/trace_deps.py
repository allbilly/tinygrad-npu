#!/usr/bin/env python3
"""Trace one prefill's CPU programs, then simulate dependency-aware multi-worker schedules.

Why this exists: the in-model split (`results/progcost/report.json`) shows 2,391 of
2,396 ms is *inside* the compiled kernels, on one core, while the reference keeps 4.26
cores busy at 1,201 ms. Before writing any async dispatcher, this measures the floor such
a dispatcher could reach: the graph's own data dependencies decide it, not the scheduler.

The hook is on `exec_kernel`, which is the one place that holds both the `ProgramInfo`
(declared `ins`/`outs` slots) and the concrete argument pointers it is about to pass. A
second hook on `CPUProgram.__call__` times each kernel. Joining the two gives, per call:
name, read pointers, written pointers, and measured duration.

The simulation is an **upper bound on available parallelism**:

  - buffer identity is the raw argument pointer, so two logically distinct buffers that
    ever share an address are treated as one buffer, which only *adds* ordering;
  - programs are assumed to be pure functions of their declared ins/outs;
  - `STORE` copies are not programs and are not modelled; measured non-program time
    (154 ms) is added as a serial tail by the caller;
  - NPU-calling programs (`rockchip_linear_*`) are serialised against each other.

Usage (one NPU process at a time; this owns the device during the run):
  DEV=CPU:CLANG .venv/bin/python extra/rockchip/trace_deps.py \
      --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
"""
from __future__ import annotations

import argparse, collections, json, pathlib, time

PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def build_deps(progs: list[dict]) -> list[list[int]]:
  """Sequential-consistency edges against the previous conflicting access per buffer."""
  last: dict[int, tuple[str, int]] = {}
  deps: list[list[int]] = []
  for i, p in enumerate(progs):
    d: list[int] = []
    for b in p["ins"]:
      kind, pid = last.get(b, ("", -1))
      if kind == "w": d.append(pid)
    for b in p["outs"]:
      kind, pid = last.get(b, ("", -1))
      if kind in ("w", "r"): d.append(pid)
    for b in p["ins"]:
      if last.get(b, ("",))[0] != "w": last[b] = ("r", i)
    for b in p["outs"]:
      last[b] = ("w", i)
    deps.append(sorted(set(d)))
  return deps


def critical_path(progs: list[dict], deps: list[list[int]]) -> float:
  best = [0.0] * len(progs)
  for i in range(len(progs)):  # deps always point backwards, so index order is topological
    best[i] = progs[i]["dur"] + max((best[d] for d in deps[i]), default=0.0)
  return max(best)


def simulate(progs: list[dict], deps: list[list[int]], workers: int, npu_serial: bool = True) -> float:
  n = len(progs)
  finish: list[float | None] = [None] * n
  free = [0.0] * workers
  npu_free = 0.0
  is_npu = [p["name"].startswith("rockchip_linear") for p in progs]
  remaining = set(range(n))
  while remaining:
    ready = [i for i in remaining if all(finish[d] is not None for d in deps[i])]
    # longest-ready-first, the standard list-scheduling heuristic
    i = max(ready, key=lambda x: progs[x]["dur"])
    dep_done = max((finish[d] for d in deps[i]), default=0.0)
    w = min(range(workers), key=lambda k: free[k])
    start = max(free[w], dep_done)
    if npu_serial and is_npu[i]: start = max(start, npu_free)
    finish[i] = start + progs[i]["dur"]
    free[w] = finish[i]
    if npu_serial and is_npu[i]: npu_free = finish[i]
    remaining.remove(i)
  return max(f for f in finish if f is not None)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=pathlib.Path, required=True)
  p.add_argument("--prompt", type=int, default=128)
  p.add_argument("--gen", type=int, default=1)
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/rknpu/results/progcost"))
  args = p.parse_args()

  import tinygrad.engine.realize as R
  from tinygrad.runtime.ops_cpu import CPUProgram

  current = {"ins": (), "outs": ()}
  state = {"on": False}
  trace: list[dict] = []

  original_exec = R.exec_kernel
  def exec_kernel_hooked(ctx, call, ast, devices=None):
    current["ins"], current["outs"] = tuple(ast.arg.ins), tuple(ast.arg.outs)
    return original_exec(ctx, call, ast, devices)
  R.exec_kernel = exec_kernel_hooked

  original_call = CPUProgram.__call__
  def recording(self, *fargs, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=False, timeout=None):
    if not state["on"]:
      return original_call(self, *fargs, global_size=global_size, local_size=local_size, wait=wait, timeout=timeout)
    ins, outs = current["ins"], current["outs"]
    t0 = time.perf_counter_ns()
    out = original_call(self, *fargs, global_size=global_size, local_size=local_size, wait=wait, timeout=timeout)
    dur = (time.perf_counter_ns() - t0) / 1e6
    trace.append({"i": len(trace), "name": self.name, "dur": dur,
                  "ins": [fargs[s] for s in ins], "outs": [fargs[s] for s in outs]})
    return out
  CPUProgram.__call__ = recording

  from tinygrad import Device, Tensor
  print(f"CPU renderer: {type(Device['CPU'].renderer).__name__}", flush=True)
  from tinygrad.llm.gguf import gguf_parse
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.tokenizer import SimpleTokenizer
  from tinygrad.uop.ops import UOp

  kv, _ = gguf_parse(str(args.ckpt))
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  ids = tokenizer.encode(PROMPT)[:args.prompt]
  max_context = args.prompt + args.gen + 16
  model, _ = Transformer.from_gguf(str(args.ckpt), max_context=max_context)
  from npu_linear import prepare_linears, replace_linears, fuse_linears
  fuse_linears(model); replace_linears(model); prepare_linears(model)

  temperature = Tensor([0.0])
  prompt_tensor = Tensor(ids, dtype="int32").reshape(1, args.prompt)
  v_start = UOp.variable("start_pos", 0, max_context - 1)
  pos_zero = v_start.bind(0)

  model(prompt_tensor, (pos_zero,), temperature).realize()          # capture
  trace.clear(); state["on"] = True
  wall = time.perf_counter()
  model(prompt_tensor, (pos_zero,), temperature).realize()          # replay, traced
  wall = time.perf_counter() - wall
  state["on"] = False

  out_dir = args.out if args.out.is_absolute() else pathlib.Path("/home/orangepi/pr/tinygrad") / args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "trace.jsonl").write_text("\n".join(json.dumps(t) for t in trace))

  total = sum(t["dur"] for t in trace)
  deps = build_deps(trace)
  cp = critical_path(trace, deps)
  print(f"\nprograms {len(trace)}  traced wall {wall*1e3:.1f} ms  kernel total {total:.1f} ms  critical path {cp:.1f} ms")
  by_name = collections.defaultdict(lambda: [0, 0.0])
  for t in trace:
    by_name[t["name"]][0] += 1; by_name[t["name"]][1] += t["dur"]
  print(f"{'kernel':44s} {'calls':>6s} {'ms':>9s}")
  for name, (calls, ms) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"{name:44s} {calls:6d} {ms:9.1f}")

  print(f"\n{'workers':>8s} {'makespan ms':>12s} {'speedup':>8s} {'busy':>6s}")
  results = {}
  for w in (1, 2, 3, 4, 6):
    ms = simulate(trace, deps, w)
    results[w] = ms
    print(f"{w:8d} {ms:12.1f} {total/ms:8.2f} {total/ms/w*100:5.0f}%")
  (out_dir / "schedule.json").write_text(json.dumps(
    {"programs": len(trace), "traced_wall_ms": wall * 1e3, "kernel_total_ms": total,
     "critical_path_ms": cp, "makespan_ms": results}, indent=2))
  print(f"\nwrote {out_dir/'trace.jsonl'} and {out_dir/'schedule.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
