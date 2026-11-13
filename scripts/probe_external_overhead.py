#!/usr/bin/env python3
"""One instrumented decode run: where does the per-token wall time actually go?

The replay profile measures only what runs INSIDE the generated replay C function.
The shipped soak wall is ~1.3 ms/token larger than the sum of its labels, and nobody
owns that slice. This probe splits it in one run with no NPU-probe code of its own:

    token wall (benchmark decode_tps)
      = [ Rust/ctypes: build inputs + NativeReplay.valid + version bookkeeping ]   jit_overhead
      + [ replay C function: sum of per-call label ns ]                           labels
      + [ inside NativeReplay.__call__ but outside its C call ]                    native_prepost
      + [ model.generate Python between tokens (shrink/bind/item/...) ]            external

It monkeypatches RockchipCapturedJit.__call__ and NativeReplay.__call__/valid with
perf_counter wrappers, enables ONPU_REPLAY_PROFILE=1 (in-replay timing, no per-token
file writes because ONPU_REPLAY_PROFILE_OUT is unset), runs the SAME benchmark
invocation as scripts/probe_decode.py, then prints JSON with the decomposition.

Outputs (per decoded token, ms): wall, jit_total, valid, native_total, labels,
jit_overhead, native_prepost, external; plus the top-10 label rows from the decode
replay, and the benchmark's decode_submits/token as a sanity check.

HARDWARE: uses the NPU. The Lead runs this (single-owner). Runtime ~70 s
(~48 s of it is model preparation).

    /home/orangepi/tinygrad/.venv/bin/python \
      /home/orangepi/tg-exp-stream/scripts/probe_external_overhead.py \
      --tokens 16 --warmups 1 --runs 2 --report /tmp/ext_overhead_backend.json \
      --json-out /tmp/ext_overhead.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

TREE = "/home/orangepi/tg-exp-stream"
CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/ChatGLM3-6B/chatglm3-6b.Q8_0.gguf"
TOKENS = ("/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/"
          "staged/chatglm-mapped-weights-v1/chatglm3-6b.tokens")

SHIPPED_ENV = {
    "ONPU_WIDE_RAW_SWIGLU": "1", "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": "3221225472",
    "ONPU_VECTOR_SIGMOID": "1", "ONPU_CPU_DECODE_POOL": "1", "ONPU_HOST_POOL_WFE": "1",
    "ONPU_EXP2_FAST_PATH": "1", "ONPU_GROUPED_DECODE_DOWN": "1", "ONPU_CPU_RMS_ALL_ROWS": "1",
    "ONPU_INT8_REUSE_ACTIVATION": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
    "DEV": "CPU", "HALF": "1", "JIT": "1", "CAPTURING": "1", "PARALLEL": "0", "DEBUG": "0",
    "PROFILE": "0", "VIZ": "0", "PYTHONPATH": TREE, "PYTHONDONTWRITEBYTECODE": "1",
    "ONPU_REPLAY_PROFILE": "1",
}


def benchmark_argv(tokens: int, warmups: int, runs: int, report: Path) -> list[str]:
  return ["benchmark_inference", "--checkpoint", CKPT, "--input", TOKENS, "--device", "CPU",
          "--prefill", "on", "--decode", "on", "--context", "512", "--tokens", str(tokens),
          "--warmups", str(warmups), "--runs", str(runs), "--precision", "int8",
          "--tile-columns", "24576", "--chunk-size", "128", "--fuse-projections",
          "--cpu-attention", "--native-replay", "--fused-qkv", "--cpu-norm",
          "--split-down", "--cpu-fast-softmax", "--cache-cleaning", "--wake-helper",
          "--share-projection-weights", "--cpu-pool", "--cpu-prefill-pool",
          "--fused-ffn", "--defer-weight-realization", "--file-backed-cpu-weights",
          "--fused-swiglu-ffn", "--cpu-q8-embedding", "--report", str(report)]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tokens", type=int, default=16)
  ap.add_argument("--warmups", type=int, default=1)
  ap.add_argument("--runs", type=int, default=2)
  ap.add_argument("--report", default="/tmp/ext_overhead_backend.json")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  report_path = Path(args.report)
  report_path.unlink(missing_ok=True)
  os.environ.update(SHIPPED_ENV)
  sys.path.insert(0, TREE)

  import extra.rockchip.benchmark_inference as bi  # noqa: E402  (env must be set first)
  from tinygrad.llm.kernels import rockchip_replay as rr  # noqa: E402

  native: dict[int, dict] = {}
  jits: dict[int, dict] = {}
  orig_native_call, orig_valid, orig_jit_call = (rr.NativeReplay.__call__, rr.NativeReplay.valid,
                                                 rr.RockchipCapturedJit.__call__)

  def native_call(self, inputs, values):
    profiling = bool(getattr(self, "profiling", False))
    before = sum(self.profile_ns.values()) if profiling else 0
    t0 = time.perf_counter()
    result = orig_native_call(self, inputs, values)
    dt = time.perf_counter() - t0
    st = native.setdefault(id(self), {"calls": 0, "native_s": 0.0, "label_s": 0.0, "valid_s": 0.0,
                                      "instance": self})
    st["calls"] += 1
    st["native_s"] += dt
    if profiling: st["label_s"] += sum(self.profile_ns.values()) - before
    return result

  def valid(self, inputs):
    t0 = time.perf_counter()
    result = orig_valid(self, inputs)
    st = native.setdefault(id(self), {"calls": 0, "native_s": 0.0, "label_s": 0.0, "valid_s": 0.0,
                                      "instance": self})
    st["valid_s"] += time.perf_counter() - t0
    return result

  def jit_call(self, input_uops, var_vals):
    t0 = time.perf_counter()
    result = orig_jit_call(self, input_uops, var_vals)
    st = jits.setdefault(id(self), {"calls": 0, "jit_s": 0.0, "instance": self})
    st["calls"] += 1
    st["jit_s"] += time.perf_counter() - t0
    return result

  rr.NativeReplay.__call__, rr.NativeReplay.valid, rr.RockchipCapturedJit.__call__ = native_call, valid, jit_call
  sys.argv = benchmark_argv(args.tokens, args.warmups, args.runs, report_path)
  t_start = time.perf_counter()
  try:
    bi.main()
  finally:
    rr.NativeReplay.__call__, rr.NativeReplay.valid, rr.RockchipCapturedJit.__call__ = (orig_native_call, orig_valid, orig_jit_call)
  run_seconds = time.perf_counter() - t_start

  report = json.loads(report_path.read_text()) if report_path.exists() else {"trials": []}
  trials = [t for t in report.get("trials", []) if not t.get("warmup") and t.get("decode_tps")]
  if not trials: raise SystemExit(f"no measured decode trials in {report_path}")
  wall_ms = statistics.median(1000.0 / t["decode_tps"] for t in trials)
  nid = max(native, key=lambda key: native[key]["calls"])
  jid = max(jits, key=lambda key: jits[key]["calls"])
  nst, jst = native[nid], jits[jid]
  calls = max(1, nst["calls"])
  def ms(seconds: float) -> float: return round(seconds / calls * 1000, 4)

  labels = {}
  instance = nst["instance"]
  if getattr(instance, "profiling", False):
    labels = {k: round(v / 1e6 / calls, 4) for k, v in instance.profile_ns.items()}
  top = dict(sorted(labels.items(), key=lambda kv: -kv[1])[:10])

  out = {
    "run_seconds": round(run_seconds, 1),
    "tools": {"tokens": args.tokens, "warmups": args.warmups, "runs": args.runs,
              "tree": TREE, "profiling": True},
    "trials": [{"decode_tps": round(t["decode_tps"], 4),
                "decode_submits": t.get("decode_submits"),
                "decode_submits_per_step": round(t.get("decode_submits", 0) / max(1, len(t.get("generated_ids", [])) - 1), 2)
                if t.get("decode_submits") else None} for t in trials],
    "per_token_ms": {
      "wall": round(wall_ms, 3),
      "jit_total": ms(jst["jit_s"]),
      "valid": ms(nst["valid_s"]),
      "native_total": ms(nst["native_s"]),
      "labels_sum": ms(nst["label_s"]),
      "jit_overhead": round(ms(jst["jit_s"]) - ms(nst["native_s"]) - ms(nst["valid_s"]), 4),
      "native_prepost": round(ms(nst["native_s"]) - ms(nst["label_s"]), 4),
      "external_model_generate": round(wall_ms - ms(jst["jit_s"]), 4),
    },
    "counts": {"decode_replay_calls": nst["calls"], "decode_jit_calls": jst["calls"],
               "native_replays_built": len(native), "jits_instrumented": len(jits)},
    "labels_top10_ms_per_token": top,
    "identity_check": {"labels_sum_matches_profile_expectation": round(ms(nst["label_s"]), 2)},
  }
  print("\n=== external-overhead decomposition (JSON) ===")
  print(json.dumps(out, indent=1))
  if args.json_out: Path(args.json_out).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
