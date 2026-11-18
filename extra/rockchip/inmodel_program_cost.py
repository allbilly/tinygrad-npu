#!/usr/bin/env python3
"""Attribute one real prefill between compiled-kernel time and CPU-device dispatch.

The claim under test is the one in FINDINGS.md: "~1.3 ms per CPU program x ~1,031
programs ~= the whole gap". That 1.3 ms was measured through `fast_sigmoid`, which
rebuilds a Python graph on every call (`reshape/contiguous/empty/custom_kernel`), and
none of that work exists inside a captured JIT. So the in-model number has to be
measured in-model (AGENTS.md measurement rule 1).

Each instrumented prefill splits every `CPUProgram.__call__` into:

  pack    -- building the packed argument buffer (ctypes create_string_buffer + pack_into)
  fxn     -- the compiled kernel itself (ctypes CFUNCTYPE into RWX pages)
  outer   -- the whole call, so `outer - pack - fxn` is Python bookkeeping around them

and prefill wall time minus the sum of `outer` is everything that is not a CPU program:
executor iteration, Buffer copies, allocations, NPU submits and waits.

The patch is installed once but gated by a flag, so an un-instrumented prefill runs the
*original* `__call__` verbatim. Trials alternate instrumented/un-instrumented, which
measures the observer effect instead of assuming it away (a previous attempt reimplemented
`__call__` unconditionally and reported a 4.78 s prefill against a 2.53 s baseline).

Usage:
  DEV=CPU:CLANG .venv/bin/python extra/rockchip/inmodel_program_cost.py \
      --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
"""
from __future__ import annotations

import argparse, collections, json, pathlib, time

PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=pathlib.Path, required=True)
  p.add_argument("--prompt", type=int, default=128)
  p.add_argument("--gen", type=int, default=1)
  p.add_argument("--out", type=pathlib.Path, default=None)
  args = p.parse_args()

  import ctypes, struct
  from tinygrad.helpers import data64_le, unwrap, cpu_profile
  from tinygrad.runtime.ops_cpu import CPUProgram

  state = {"on": False}
  stats = {"outer_ns": 0, "pack_ns": 0, "fxn_ns": 0, "thread_ns": 0, "calls": 0}
  by_name: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # calls, outer_ns, fxn_ns, thread_ns
  original = CPUProgram.__call__

  def timed_call(self, *fargs, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=False, timeout=None):
    if not state["on"]:
      return original(self, *fargs, global_size=global_size, local_size=local_size, wait=wait, timeout=timeout)
    assert self.dev.remote is None, "this probe only instruments the local CPU device"
    t0 = time.perf_counter_ns()
    t3 = time.thread_time_ns()
    hdr = 12 if self.lvp else 0
    kernargs = ctypes.create_string_buffer(hdr + len(fargs) * 8)
    if self.lvp:
      struct.pack_into('<3I', kernargs, 0, *data64_le(ctypes.addressof(kernargs) + hdr), len(fargs) * 2)
    self.packer.pack_into(kernargs, hdr, *fargs)
    t1 = time.perf_counter_ns()
    with cpu_profile(self.name, self.dev.device, profile_key=self.profile_key) as prof:
      self.fxn(kernargs)
    t2 = time.perf_counter_ns()
    t4 = time.thread_time_ns()
    stats["calls"] += 1
    stats["pack_ns"] += t1 - t0
    stats["fxn_ns"] += t2 - t1
    stats["outer_ns"] += t2 - t0
    stats["thread_ns"] += t4 - t3
    row = by_name[self.name]
    row[0] += 1; row[1] += t2 - t0; row[2] += t2 - t1
    if len(row) < 4: row.append(0)
    row[3] += t4 - t3
    return float(unwrap(prof.en) - prof.st) * 1e-6 if wait else None

  CPUProgram.__call__ = timed_call

  from tinygrad import Device, Tensor
  print(f"CPU renderer: {type(Device['CPU'].renderer).__name__} arch={Device['CPU'].arch}", flush=True)
  from tinygrad.llm.gguf import gguf_parse
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.tokenizer import SimpleTokenizer
  from tinygrad.uop.ops import UOp

  kv, _ = gguf_parse(str(args.ckpt))
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  ids = tokenizer.encode(PROMPT)[:args.prompt]
  max_context = args.prompt + args.gen + 16
  load_started = time.time()
  model, _ = Transformer.from_gguf(str(args.ckpt), max_context=max_context)
  from npu_linear import packed_weight_bytes, prepare_linears, replace_linears, fuse_linears
  fuse_linears(model)
  n_linears = replace_linears(model)
  prepare_linears(model)
  nbytes = packed_weight_bytes(model)
  load_seconds = time.time() - load_started

  temperature = Tensor([0.0])
  prompt_tensor = Tensor(ids, dtype="int32").reshape(1, args.prompt)
  v_start = UOp.variable("start_pos", 0, max_context - 1)
  pos_zero = v_start.bind(0)

  # 1 captures the JIT; then instrument / bare / instrument, so the observer effect is a
  # measured ratio between adjacent trials rather than an assumption.
  plan = [("capture", False), ("instrumented-a", True), ("bare", False), ("instrumented-b", True)]
  trials = []
  for tag, instrument in plan:
    stats.update(outer_ns=0, pack_ns=0, fxn_ns=0, thread_ns=0, calls=0)
    by_name.clear()
    state["on"] = instrument
    npu = Device["ROCKCHIP"]
    s0, w0 = npu.submit_count, npu.submit_seconds
    t0 = time.perf_counter()
    logits = model(prompt_tensor, (pos_zero,), temperature).realize()
    wall = time.perf_counter() - t0
    # Bitwise identity at the model boundary: two runs of the same configuration must agree here
    # before a transformed run can be compared with an untransformed one.
    import hashlib
    import numpy as _np
    arr = logits.numpy()
    logits_hash = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    print(f"logits finite: {bool(_np.isfinite(arr).all())}", flush=True)
    state["on"] = False
    row = {
      "tag": tag, "instrumented": instrument,
      "wall_ms": round(wall * 1e3, 1),
      "cpu_programs": stats["calls"],
      "dispatch_outer_ms": round(stats["outer_ns"] / 1e6, 1),
      "dispatch_pack_ms": round(stats["pack_ns"] / 1e6, 1),
      "kernel_fxn_ms": round(stats["fxn_ns"] / 1e6, 1),
      "thread_cpu_ms": round(stats["thread_ns"] / 1e6, 1),
      "non_program_ms": round(wall * 1e3 - stats["outer_ns"] / 1e6, 1),
      "logits_sha256_16": logits_hash,
      "npu_submits": npu.submit_count - s0,
      "temp_c": round(int(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1),
      "npu_seconds": round(npu.submit_seconds - w0, 3),
      "top": sorted(({"name": n, "calls": v[0], "outer_ms": round(v[1] / 1e6, 1), "fxn_ms": round(v[2] / 1e6, 1),
                      "thread_ms": round((v[3] if len(v) > 3 else 0) / 1e6, 1)}
                     for n, v in by_name.items()), key=lambda r: -r["outer_ms"]),
    }
    trials.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "top"}, indent=2), flush=True)

  out = {"checkpoint": str(args.ckpt), "npu_linears": n_linears, "npu_weight_mib": round(nbytes / 2**20, 1),
         "load_seconds": round(load_seconds, 2), "trials": trials}
  path = args.out or pathlib.Path("docs/rknpu/results/progcost/report.json")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(out, indent=2))
  print(f"wrote {path}")
  try:
    import ctypes, pathlib as _p
    lib = ctypes.CDLL(str(_p.Path(__file__).resolve().parent / "librkpool.so"))
    lib.rk_calls.restype = ctypes.c_long; lib.rk_refused.restype = ctypes.c_long
    print(f"[rkpool] dispatch calls {lib.rk_calls()}  refused/serial {lib.rk_refused()}", flush=True)
  except OSError as exc:
    print(f"[rkpool] not loaded: {exc}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
