#!/usr/bin/env python3
"""Run the 12 PASS models through the generic NPU GEMM baseline.

One subprocess per model: the NPU is single-owner and this releases RAM between
models. A model whose report.json already exists is skipped, so an interrupted
sweep resumes instead of restarting.

Order is the HuggingFace download ranking, PASS rows only.
"""
from __future__ import annotations

import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
PYTHON = REPO / ".venv/bin/python"
RESULTS = REPO / "docs/rknpu/results"
BASE = pathlib.Path("/home/orangepi/vlm-bench/baseline-20261021")

# (name, slug, checkpoint, RKLLM published prefill tok/s, decode tok/s)
MODELS = [
  ("Qwen3.5 4B",    "qwen35-4b",    BASE/"checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf",                 61.87,  6.22),
  ("Qwen2.5 1.5B",  "qwen25-15b",   BASE/"qwen2.5-1.5b-instruct-q8_0.gguf",                             338.35, 16.69),
  ("Qwen3.5 2B",    "qwen35-2b",    BASE/"checkpoints/Qwen3.5-2B/Qwen3.5-2B-Q8_0.gguf",                164.75, 13.59),
  ("Qwen3.5 0.8B",  "qwen35-08b",   BASE/"checkpoints/Qwen3.5-0.8B/Qwen3.5-0.8B-Q8_0.gguf",            217.78, 27.05),
  ("Qwen3-VL 2B",   "qwen3vl-2b",   BASE/"checkpoints/Qwen3-VL-2B/Qwen3VL-2B-Instruct-Q8_0.gguf",      333.66, 14.98),
  ("Qwen3 0.6B",    "qwen3-06b",    pathlib.Path("/home/orangepi/oNPU/.data/models/72717a59205944dc-Qwen3-0.6B-Q8_0.gguf"), 642.93, 32.91),
  ("Gemma2 2B",     "gemma2-2b",    BASE/"checkpoints/Gemma2-2B/gemma-2-2b-it-Q8_0.gguf",              213.90, 10.37),
  ("TinyLLAMA 1.1B","tinyllama-11b",BASE/"tinyllama-1.1b-chat-v1.0.Q8_0.gguf",                         524.74, 24.43),
  ("TeleChat2 3B",  "telechat2-3b", BASE/"checkpoints/TeleChat2-3B/TeleChat2-3B-f16.gguf",             206.96, 10.18),
  ("Qwen2 0.5B",    "qwen2-05b",    BASE/"Qwen2-0.5B-Instruct-Q8_0.gguf",                               877.31, 41.58),
  ("MiniCPM4 0.5B", "minicpm4-05b", BASE/"checkpoints/MiniCPM4-0.5B/MiniCPM4-0.5B.Q8_0.gguf",          946.12, 45.34),
  ("InternLM2 1.8B","internlm2-18b",BASE/"checkpoints/InternLM2-1.8B/internlm2-chat-1_8b.Q8_0.gguf",   336.72, 15.45),
]


def temperature() -> float | None:
  try: return int(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
  except Exception: return None


def main() -> int:
  RESULTS.mkdir(parents=True, exist_ok=True)
  log = RESULTS / "sweep.log"
  only = sys.argv[1] if len(sys.argv) > 1 else None
  for name, slug, ckpt, _, _ in MODELS:
    if only and only != slug: continue
    out = RESULTS / slug
    if (out / "report.json").exists():
      print(f"skip {slug} (report exists)", flush=True)
      continue
    if (out / "failed.txt").exists():
      print(f"skip {slug} (marked failed: {(out / 'failed.txt').read_text().strip()[:120]})", flush=True)
      continue
    if not ckpt.exists():
      print(f"MISSING {slug}: {ckpt}", flush=True)
      continue
    start_temp = temperature()
    begin = time.time()
    with log.open("a") as handle:
      handle.write(f"{time.strftime('%F %T')} start {slug} temp={start_temp}C\n")
    proc = subprocess.run([str(PYTHON), "bench_llm.py", "--ckpt", str(ckpt), "--name", name,
                           "--out", str(out), "--npu"], cwd=HERE, env={**os.environ, "DEV": "CPU"},
                          capture_output=True, text=True, timeout=7200)
    seconds = round(time.time() - begin, 1)
    with log.open("a") as handle:
      handle.write(f"{time.strftime('%F %T')} end   {slug} rc={proc.returncode} {seconds}s "
                   f"temp={temperature()}C\n")
      if proc.returncode: handle.write(proc.stderr[-4000:] + "\n")
    print(f"{'ok  ' if proc.returncode == 0 else 'FAIL'} {slug} {seconds}s", flush=True)
    if proc.returncode: print(proc.stderr[-2000:], flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
