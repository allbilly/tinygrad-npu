#!/usr/bin/env python3
"""Where does the per-token wall time go OUTSIDE the replay? One instrumented run.

The replay profile measures 200.31 ms/token (ROCKCHIP 193.39 + CPU 6.08 + COPY 0.84)
while the unprofiled shipped soak wall is 201.57 ms/token: ~1.26 ms/token is outside
the replay C function (possibly partly profiling artefact/thermal - this run decides).

This probe runs the SAME official benchmark invocation as scripts/probe_decode.py,
from the PRISTINE tree /home/orangepi/tg-decode-all with the shipped flags, with
ONPU_REPLAY_PROFILE **off** (so the replay is not instrumented and the comparison
with the real shipped run is clean). It monkeypatches, per token:

  wall            = time between consecutive tokens yielded by Transformer.generate
  jit_total       = RockchipCapturedJit.__call__  (input build, valid, native call, ret subst)
  valid           = NativeReplay.valid            (deduplicated guard checks)
  native_total    = NativeReplay.__call__         (Python around the replay C call)
  replay_c        = self.program.fxn(...)         (the replay C function itself)
  sym_infer       = symbol inference calls inside NativeReplay.__call__
  buffer_ops      = Buffer.ensure_allocated / Buffer.get_buf inside the JIT/native path
  input_copies    = _copy_input / _copy_scalar_input in the JIT path
Derived (non-overlapping, sums to wall):
  external        = wall - jit_total              (model.generate Python, out.item(), benchmark loop)
  jit_pre         = jit_total - valid - native_total - input_copies
  native_pre      = native_total - replay_c - sym_infer - buffer_ops_native
The graph and the benchmark code are untouched; only wrapper functions add timers.

HARDWARE: uses the NPU. The Lead runs this (single-owner, ~80 s; ~50 s is model prep).

    /home/orangepi/tinygrad/.venv/bin/python \
      /home/orangepi/tg-exp-stream/scripts/probe_where_time_goes.py \
      --report /tmp/wtg_backend.json --json-out /tmp/wtg.json

Optional, in a second shell during the run (passive, no NPU):
    python3 /home/orangepi/tg-exp-stream/scripts/probe_npu_clock_sampler.py --while-pid <pid>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PRISTINE = "/home/orangepi/tg-decode-all"
CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/ChatGLM3-6B/chatglm3-6b.Q8_0.gguf"
TOKENS = ("/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/"
          "staged/chatglm-mapped-weights-v1/chatglm3-6b.tokens")

# Exactly the shipped soak-v13 configuration (same dict as probe_decode.SHIPPED_ENV),
# but ONPU_REPLAY_PROFILE is deliberately NOT set.
SHIPPED_ENV = {
    "ONPU_WIDE_RAW_SWIGLU": "1", "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": "3221225472",
    "ONPU_VECTOR_SIGMOID": "1", "ONPU_CPU_DECODE_POOL": "1", "ONPU_HOST_POOL_WFE": "1",
    "ONPU_EXP2_FAST_PATH": "1", "ONPU_GROUPED_DECODE_DOWN": "1", "ONPU_CPU_RMS_ALL_ROWS": "1",
    "ONPU_INT8_REUSE_ACTIVATION": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
    "DEV": "CPU", "HALF": "1", "JIT": "1", "CAPTURING": "1", "PARALLEL": "0", "DEBUG": "0",
    "PROFILE": "0", "VIZ": "0", "PYTHONPATH": PRISTINE, "PYTHONDONTWRITEBYTECODE": "1",
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


class Timers:
  """Additive buckets, snapshotted at every token boundary."""

  def __init__(self) -> None:
    self.bucket: dict[str, float] = {}
    self.snapshots: list[tuple[int, dict[str, float]]] = []  # (generate-call id, cumulative buckets)
    self.intervals: list[tuple[int, float]] = []             # (generate-call id, seconds since prev token)
    self.call_id = 0
    self.in_native = 0
    self.in_jit = 0

  def add(self, name: str, dt: float) -> None: self.bucket[name] = self.bucket.get(name, 0.0) + dt
  def snapshot(self) -> dict[str, float]: return dict(self.bucket)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tokens", type=int, default=32)
  ap.add_argument("--warmups", type=int, default=1)
  ap.add_argument("--runs", type=int, default=2)
  ap.add_argument("--report", default="/tmp/wtg_backend.json")
  ap.add_argument("--json-out", default=None)
  ap.add_argument("--selftest", action="store_true", help="only verify the patch targets exist (no NPU)")
  args = ap.parse_args()

  report_path = Path(args.report)
  report_path.unlink(missing_ok=True)
  os.environ.update(SHIPPED_ENV)
  sys.path.insert(0, PRISTINE)

  import tinygrad.llm.model as llm_model  # noqa: E402
  from tinygrad.device import Buffer  # noqa: E402
  from tinygrad.uop.ops import UOp  # noqa: E402
  from tinygrad.llm.kernels import rockchip_replay as rr  # noqa: E402
  import extra.rockchip.benchmark_inference as bi  # noqa: E402

  missing = [name for name in ("RockchipCapturedJit", "NativeReplay") if not hasattr(rr, name)]
  if not hasattr(llm_model.Transformer, "generate"): missing.append("Transformer.generate")
  for name in ("_copy_input", "_copy_scalar_input", "sym_infer"):
    if not hasattr(rr, name): missing.append(f"rockchip_replay.{name}")
  if missing:
    print(json.dumps({"error": "patch targets missing", "missing": missing}))
    return 2
  if args.selftest:
    print(json.dumps({"ok": True, "tree": PRISTINE,
                      "targets": ["Transformer.generate", "RockchipCapturedJit.__call__", "NativeReplay.__call__",
                                  "NativeReplay.valid", "program.fxn", "Buffer.ensure_allocated", "Buffer.get_buf",
                                  "rockchip_replay.sym_infer", "rockchip_replay._copy_scalar_input"]}))
    return 0

  timers = Timers()
  originals = {}

  # ---- per-token wall + cumulative-bucket snapshots (model.generate wrapper) ----
  original_generate = llm_model.Transformer.generate

  def generate(self, *fargs, **fkwargs):
    timers.call_id += 1
    call_id = timers.call_id
    last = time.perf_counter()
    for token in original_generate(self, *fargs, **fkwargs):
      now = time.perf_counter()
      timers.intervals.append((call_id, now - last))
      timers.snapshots.append((call_id, timers.snapshot()))
      last = now
      yield token

  llm_model.Transformer.generate = generate
  originals["generate"] = original_generate

  # ---- JIT dispatch wrapper ----
  original_jit_call = rr.RockchipCapturedJit.__call__

  def jit_call(self, input_uops, var_vals):
    timers.in_jit += 1
    t0 = time.perf_counter()
    try: return original_jit_call(self, input_uops, var_vals)
    finally:
      timers.in_jit -= 1
      timers.add("jit_total", time.perf_counter() - t0)

  rr.RockchipCapturedJit.__call__ = jit_call

  # ---- guard checks ----
  original_valid = rr.NativeReplay.valid

  def valid(self, inputs):
    t0 = time.perf_counter()
    try: return original_valid(self, inputs)
    finally: timers.add("valid", time.perf_counter() - t0)

  rr.NativeReplay.valid = valid

  # ---- NativeReplay.__call__: time the whole call and the inner C replay ----
  original_native_call = rr.NativeReplay.__call__

  def native_call(self, inputs, values):
    if not getattr(self, "_wtg_fxn_wrapped", False):
      original_fxn = self.program.fxn

      def timed_fxn(*fargs):
        t0 = time.perf_counter()
        try: return original_fxn(*fargs)
        finally: timers.add("replay_c", time.perf_counter() - t0)

      self.program.fxn = timed_fxn
      self._wtg_fxn_wrapped = True
    timers.in_native += 1
    t0 = time.perf_counter()
    try: return original_native_call(self, inputs, values)
    finally:
      timers.in_native -= 1
      timers.add("native_total", time.perf_counter() - t0)

  rr.NativeReplay.__call__ = native_call

  # ---- component wrappers active only inside the measured paths ----
  original_sym_infer = rr.sym_infer

  def sym_infer(*fargs):
    t0 = time.perf_counter()
    try: return original_sym_infer(*fargs)
    finally:
      if timers.in_native: timers.add("sym_infer", time.perf_counter() - t0)

  rr.sym_infer = sym_infer

  for buffer_method in ("ensure_allocated", "get_buf"):
    original = getattr(Buffer, buffer_method)

    def wrapped(self, *fargs, _original=original, **fkwargs):
      t0 = time.perf_counter()
      try: return _original(self, *fargs, **fkwargs)
      finally:
        dt = time.perf_counter() - t0
        if timers.in_native: timers.add("buffer_ops_native", dt)
        elif timers.in_jit: timers.add("buffer_ops_jit", dt)

    setattr(Buffer, buffer_method, wrapped)

  # Only the outer per-token copy helper is wrapped: _copy_input is called inside it.
  original_copy = rr._copy_scalar_input

  def copy_wrapped(*fargs):
    t0 = time.perf_counter()
    try: return original_copy(*fargs)
    finally:
      if timers.in_jit: timers.add("input_copies", time.perf_counter() - t0)

  rr._copy_scalar_input = copy_wrapped

  # The symbolic-return substitution runs per token inside the JIT wrapper.
  original_substitute = UOp.substitute

  def substitute(self, *fargs, **fkwargs):
    t0 = time.perf_counter()
    try: return original_substitute(self, *fargs, **fkwargs)
    finally:
      if timers.in_jit: timers.add("symbolic_subst", time.perf_counter() - t0)

  UOp.substitute = substitute

  sys.argv = benchmark_argv(args.tokens, args.warmups, args.runs, report_path)
  t_start = time.perf_counter()
  try:
    bi.main()
  finally:
    llm_model.Transformer.generate = originals["generate"]
    rr.RockchipCapturedJit.__call__ = original_jit_call
    rr.NativeReplay.valid = original_valid
    rr.NativeReplay.__call__ = original_native_call
    rr.sym_infer = original_sym_infer
    UOp.substitute = original_substitute
  run_seconds = time.perf_counter() - t_start

  # ---------- per-token deltas: first interval of each generate call is prefill ----------
  buckets = ("jit_total", "valid", "native_total", "replay_c", "sym_infer", "symbolic_subst",
             "buffer_ops_native", "buffer_ops_jit", "input_copies")
  rows = []
  prev_snapshot: dict[str, float] = {}
  prev_call = None
  for (call_id, interval), (snap_call, snap) in zip(timers.intervals, timers.snapshots):
    delta = {name: snap.get(name, 0.0) - prev_snapshot.get(name, 0.0) for name in buckets}
    first = call_id != prev_call
    prev_snapshot, prev_call = snap, call_id
    if first: continue  # prefill interval (also contains the prefill JIT)
    rows.append({"call_id": call_id, "wall": interval, **delta})

  if not rows: raise SystemExit("no decode tokens captured")
  normal = [r for r in rows if r["replay_c"] > 0]
  executor_fallback_rows = [r for r in rows if r["replay_c"] == 0]
  if not normal: raise SystemExit("no replayed decode tokens captured")
  med = statistics.median(r["wall"] for r in normal)
  outliers = [r for r in normal if r["wall"] > 2 * med]
  rows = [r for r in normal if r["wall"] <= 2 * med]
  n = len(rows)
  def mean(getter) -> float: return statistics.mean(getter(r) for r in rows) * 1e3
  total = {
      "wall": mean(lambda r: r["wall"]),
      "jit_total": mean(lambda r: r["jit_total"]),
      "replay_c": mean(lambda r: r["replay_c"]),
      "valid": mean(lambda r: r["valid"]),
      "native_total": mean(lambda r: r["native_total"]),
      "sym_infer": mean(lambda r: r["sym_infer"]),
      "buffer_ops_native": mean(lambda r: r["buffer_ops_native"]),
      "buffer_ops_jit": mean(lambda r: r["buffer_ops_jit"]),
      "input_copies": mean(lambda r: r["input_copies"]),
      "symbolic_subst": mean(lambda r: r["symbolic_subst"]),
      "jit_pre": mean(lambda r: r["jit_total"] - r["valid"] - r["native_total"] - r["input_copies"]
                               - r["symbolic_subst"]),
      "native_pre": mean(lambda r: r["native_total"] - r["replay_c"] - r["sym_infer"] - r["buffer_ops_native"]),
      "external_model_generate": mean(lambda r: r["wall"] - r["jit_total"]),
  }
  def p10(getter) -> float: return round(sorted(getter(r) for r in rows)[n // 10] * 1e3, 4)
  def p90(getter) -> float: return round(sorted(getter(r) for r in rows)[9 * n // 10] * 1e3, 4)

  report = json.loads(report_path.read_text()) if report_path.exists() else {"trials": []}
  trials = [t for t in report.get("trials", []) if not t.get("warmup") and t.get("decode_tps")]
  out = {
    "run_seconds": round(run_seconds, 1),
    "tree": PRISTINE,
    "profiling": False,
    "tokens_measured": n,
    "per_token_ms": {k: round(v, 4) for k, v in total.items()},
    "per_token_p10_p90_ms": {k: [p10(lambda r, kk=k: r[kk]), p90(lambda r, kk=k: r[kk])] for k in total},
    "accounted_sum_ms": round(sum(total[k] for k in ("valid", "native_total", "jit_pre", "external_model_generate",
                                                    "input_copies", "symbolic_subst")), 4),
    "benchmark": {
      "decode_tps_median": round(statistics.median(t["decode_tps"] for t in trials), 4) if trials else None,
      "wall_from_decode_tps_ms": round(statistics.median(1000.0 / t["decode_tps"] for t in trials), 4) if trials else None,
      "decode_submits_per_token": round(trials[0].get("decode_submits", 0) / max(1, args.tokens - 1), 2) if trials else None,
      "ids_first_token": (trials[0].get("generated_ids") or [None])[0] if trials else None,
      "clocks_before_npu_hz": (trials[0].get("clocks_before") or {}).get("/sys/class/devfreq/fdab0000.npu/cur_freq") if trials else None,
      "clocks_after_npu_hz": (trials[-1].get("clocks_after") or {}).get("/sys/class/devfreq/fdab0000.npu/cur_freq") if trials else None,
    },
    "check_sums": {
      "intervals_vs_decode_tps_pct_diff": round(100 * (total["wall"] /
        statistics.median(1000.0 / t["decode_tps"] for t in trials) - 1), 3) if trials else None,
      "delta_sum_vs_wall_ms": round(sum(total[k] for k in ("valid", "native_total", "jit_pre",
                                                          "external_model_generate")) - total["wall"], 4),
    },
    "per_generate_call_mean_ms": {
      str(call): {
        "wall": round(statistics.mean(r["wall"] for r in rows if r["call_id"] == call) * 1e3, 3),
        "replay_c": round(statistics.mean(r["replay_c"] for r in rows if r["call_id"] == call) * 1e3, 3),
        "external": round(statistics.mean(r["wall"] - r["jit_total"] for r in rows if r["call_id"] == call) * 1e3, 3),
      } for call in sorted({r["call_id"] for r in rows})},
    "outlier_and_fallback": {"decode_tokens_total": len(normal) + len(executor_fallback_rows),
                             "outlier_tokens_gt_2x_median": len(outliers),
                             "executor_fallback_tokens": len(executor_fallback_rows)},
    "reducible_guide": ("jit_pre and native_pre are Python; external_model_generate is model.generate/benchmark "
                        "loop; replay_c is NPU+CPU kernels and is NOT reducible from here."),
  }
  print("\n=== where-time-goes (JSON) ===")
  print(json.dumps(out, indent=1))
  if args.json_out: Path(args.json_out).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
