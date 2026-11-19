#!/usr/bin/env python3
"""bench_llm's protocol with the four-way CPU transforms and the compiled-C replay installed.

Three things must be true for a number from this file to be comparable to
results/replay-ab/replay-parity (ttft [1256.18, 1259.09, 1270.38]):

1. `thread_split.install()` -- without it every four-way CPU kernel runs serially. Measured cost of
   forgetting it: 1,858.49 ms against 1,259.09, a 599 ms "regression" that was entirely the
   missing transform set (mistake.md M72). The install prints `transforms applied: {...}`.
2. `replay_c.install()` -- the compiled-C replay, installed after the transforms so the plans are
   emitted for the transformed programs.
3. `--npu` on the command line -- without it the model runs pure CPU and the log shows
   `prefill_submits: 0` (mistake.md M71).

All three are asserted below rather than left to the caller.
"""
from __future__ import annotations

import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import thread_split  # noqa: E402

thread_split.install()

from tinygrad.llm.kernels import replay_c  # noqa: E402

replay_c.install()

assert "rkpool" in __import__("tinygrad.runtime.support.c", fromlist=["DLL"]).DLL._loaded_, (
  "rkpool is not registered: the fused Linear's quantise would silently fall back to serial")

from bench_llm import main  # noqa: E402

if __name__ == "__main__":
  if "--npu" not in sys.argv:
    raise SystemExit("bench_replay_c.py requires --npu, or every trial reports 0 submits (M71)")
  print("[preflight] thread_split installed, rkpool registered, replay_c installed, --npu present", flush=True)
  rc = main()
  print(f"[replay_c] {replay_c.stats}", flush=True)
  raise SystemExit(rc)
