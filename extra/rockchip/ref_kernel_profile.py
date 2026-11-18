#!/usr/bin/env python3
"""Decompose the reference's 1,201 ms Qwen3.5 4B prefill per kernel, and dump its kernel C.

REA step that was missing: the reference's prefill was measured to 1,201.4 ms and its
*program counts* were captured (2,072 = 1,597 CPU + 475 NPU), but never its per-kernel time.
Without that, "port the fast implementation" has no target: it is unknown whether their
kernels are individually faster, or the same kernels spread over four cores.

Two hooks, both on the reference's own code paths, neither requiring an edit to the target:

  1. `benchmark_inference.observe` -- the harness already calls observe('begin', index) before
     generation and observe('first', index) at the first token, i.e. exactly the prefill
     boundary, and documents itself as "no production execution changes". Assigning it gives a
     per-trial prefill window.
  2. `CPUProgram.__call__` and `ClangCompiler.compile` -- every CPU kernel of theirs is a
     tinygrad custom kernel compiled by ClangCompiler, so the same instrumentation used on our
     tree works here and yields per-kernel wall time plus the emitted C source.

Run from the reference tree (python -m would put cwd first on sys.path; a script run by path
puts its own directory first, so PYTHONPATH decides and must resolve to the reference):

  cd /home/orangepi/tg-qwen35-pack-signature-fix
  /home/orangepi/tinygrad/.venv/bin/python \
      /home/orangepi/pr/tinygrad/extra/rockchip/ref_kernel_profile.py

Writes to /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-kernels/ (never into the archive).
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, re, sys, time

TREE = "/home/orangepi/tg-qwen35-pack-signature-fix"
CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
INPUT = "/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/w8a8-sweep-v1/inputs/qwen35-4b.txt"
OUT = pathlib.Path("/home/orangepi/pr/tinygrad/docs/rknpu/results/ref-kernels")
CHECKPOINT_SHA256 = "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1"

# Verbatim from docs/rknpu/results/ref-profile/run.sh, which reproduces 1,200.4 / 1,201.4 /
# 1,201.2 / 1,203.1 ms. Do not retype flags: five dropped flags once made prefill run eager
# (35,200 submits instead of 475, mistake.md M22).
ENV = {
  "ONPU_REPLAY_PROFILE": "1",
  "DEV": "CPU", "HALF": "1", "JIT": "1", "CAPTURING": "1", "PARALLEL": "0",
  "PROFILE": "0", "VIZ": "0", "DEBUG": "0", "PYTHONDONTWRITEBYTECODE": "1",
  "PYTHONPATH": TREE,
  "ONPU_BENCHMARK_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
  "ONPU_CPU_DECODE_POOL": "0", "ONPU_CPU_GQA_PREFILL": "0", "ONPU_CPU_PREFILL_LAYOUT": "1",
  "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_EXP2_FAST_PATH": "1", "ONPU_FFN_PACKED_UPLOAD": "1",
  "ONPU_GDN_DECODE_PREP_POOL": "1", "ONPU_GDN_DECODE_SPIN": "1", "ONPU_GDN_DIRECT_DECODE": "1",
  "ONPU_GDN_PREP_POOL": "1", "ONPU_HOST_POOL_WFE": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
  "ONPU_INT8_EXPERT_DEPTHS": "0", "ONPU_INT8_NATURAL2304": "1", "ONPU_INT8_SDK_PREFILL": "0",
  "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": "3221225472", "ONPU_PREFILL_CPU_TUNING": "1",
  "ONPU_REPLAY_BORROW_BATCH": "1", "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_DEDUP_GUARDS": "1",
  "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_REPLAY_NATIVE_FROM_LARGE": "1", "ONPU_REPLAY_SCALAR_COPY": "1",
  "ONPU_SHARED_HOST_POOL": "1", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1", "ONPU_SWIGLU_PREFILL_ROWS": "4",
  "ONPU_ZERO_PAD_VIEWS": "1",
}

ARGS = [
  "--checkpoint", CKPT, "--input", INPUT,
  "--device", "CPU", "--precision", "int8", "--prefill", "on", "--decode", "on",
  "--context", "512", "--tokens", "8", "--warmups", "2", "--runs", "3",
  "--chunk-size", "128", "--tile-columns", "24384",
  "--fuse-projections", "--cpu-attention", "--native-replay", "--fused-ffn", "--split-down",
  "--fused-qkv", "--cpu-norm", "--cpu-fast-softmax", "--cache-cleaning", "--wake-helper",
  "--cpu-pool", "--cpu-prefill-pool", "--share-projection-weights", "--cpu-recurrent-scan",
  "--fused-swiglu-ffn", "--fused-swiglu-decode", "--fused-qkv-without-norm",
  "--fused-recurrent-projections", "--cpu-q8-embedding", "--scalar-decode-loop",
  "--trim-preparation-cache", "--file-backed-cpu-weights", "--defer-weight-realization",
]


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, default=OUT)
  p.add_argument("--runs", type=int, default=3)
  args = p.parse_args()

  os.environ.update(ENV)
  cwd = pathlib.Path.cwd().resolve()
  if (cwd / "tinygrad" / "llm" / "model.py").exists() is False or cwd != pathlib.Path(TREE):
    print(f"must run from {TREE}, got {cwd}", file=sys.stderr)
    return 2
  sys.path.insert(0, TREE)

  from tinygrad.runtime.ops_cpu import CPUProgram
  from tinygrad.runtime.support.compiler_cpu import ClangCompiler

  out_dir: pathlib.Path = args.out
  src_dir = out_dir / "src"
  src_dir.mkdir(parents=True, exist_ok=True)

  # ---- per-kernel timing -------------------------------------------------------------
  stats: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])   # name -> [calls, ms]
  phases: list[dict] = []
  armed = {"on": False}
  original_call = CPUProgram.__call__

  def timed_call(self, *a, **kw):
    if not armed["on"]:
      return original_call(self, *a, **kw)
    t0 = time.perf_counter_ns()
    out = original_call(self, *a, **kw)
    dt = (time.perf_counter_ns() - t0) / 1e6
    row = stats[self.name]
    row[0] += 1
    row[1] += dt
    return out

  CPUProgram.__call__ = timed_call

  # ---- kernel source dump ------------------------------------------------------------
  sources: dict[str, str] = {}
  original_compile = ClangCompiler.compile

  def dump_compile(self, src: str) -> bytes:
    m = re.search(r"void\s+([A-Za-z_]\w*)\s*\(", src)
    name = m.group(1) if m else "unnamed"
    if name not in sources:
      sources[name] = src
      (src_dir / f"{name}.c").write_text(src)
    return original_compile(self, src)

  ClangCompiler.compile = dump_compile

  import extra.rockchip.benchmark_inference as bi

  def observe(event: str, index: int) -> None:
    if event == "begin":
      stats.clear()
      armed["on"] = True
      phases.append({"index": index, "event": "begin"})
    elif event == "first":
      armed["on"] = False
      phases.append({"index": index, "event": "first",
                     "kernels": {n: {"calls": v[0], "ms": round(v[1], 3)} for n, v in stats.items()},
                     "total_ms": round(sum(v[1] for v in stats.values()), 3)})

  bi.observe = observe

  base = [a for a in ARGS]
  # replace the recorded --runs with the requested one; keep --warmups 2 verbatim
  i = base.index("--runs")
  base[i + 1] = str(args.runs)
  sys.argv = ["benchmark_inference", *base, "--report", str(out_dir / "report.json")]

  rc = bi.main()
  armed["on"] = False

  graded = [ph for ph in phases if ph["event"] == "first"]
  best = min(graded, key=lambda ph: ph["total_ms"]) if graded else None
  summary = {
    "tree": TREE, "checkpoint": CKPT, "checkpoint_sha256": CHECKPOINT_SHA256,
    "note": "per-kernel prefill window from observe('begin')..observe('first'); fastest prefill "
            "trial is the one quoted, matching the 1,201 ms stabilised protocol",
    "trials": [{k: v for k, v in ph.items() if k != "kernels"} for ph in graded],
    "fastest_prefill": best,
    "kernel_sources_dumped": len(sources),
  }
  (out_dir / "kernel-profile.json").write_text(json.dumps(summary, indent=2))

  if best:
    print(f"\nfastest prefill trial: {best['total_ms']:.1f} ms of kernel time, "
          f"{sum(k['calls'] for k in best['kernels'].values())} programs")
    print(f"{'kernel':52s} {'calls':>6s} {'ms':>9s} {'share':>6s}")
    for name, k in sorted(best["kernels"].items(), key=lambda kv: -kv[1]["ms"])[:24]:
      print(f"{name:52s} {k['calls']:6d} {k['ms']:9.1f} {100*k['ms']/best['total_ms']:5.1f}%")
  print(f"\nwrote {out_dir/'kernel-profile.json'} and {len(sources)} kernel sources to {src_dir}")
  return rc or 0


if __name__ == "__main__":
  raise SystemExit(main())
