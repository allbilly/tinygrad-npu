#!/usr/bin/env python3
"""Run bench_llm with the four-way kernel split installed.

Thin wrapper so the timing protocol stays bench_llm's own (3 warm-ups + 3 measured, 128-token
prompt, 64 greedy tokens, clocks recorded): this only installs
`thread_split`, then hands every remaining argument to bench_llm verbatim.

Usage:
  DEV=CPU:CLANG .venv/bin/python extra/rockchip/bench_threaded.py -- \
      --ckpt <gguf> --name qwen35-4b-threaded --out <dir> --npu
"""
from __future__ import annotations

import pathlib, runpy, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import thread_split  # noqa: E402

def main() -> int:
  argv = sys.argv[1:]
  if argv[:1] == ["--"]: argv = argv[1:]
  thread_split.install()
  print(f"thread_split: {len(thread_split.THREAD_SPLIT)} kernels, {thread_split.THREAD_SPLIT_THREADS} threads: "
        f"{sorted(thread_split.THREAD_SPLIT)}", flush=True)
  sys.argv = [str(HERE / "bench_llm.py"), *argv]
  runpy.run_path(str(HERE / "bench_llm.py"), run_name="__main__")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
