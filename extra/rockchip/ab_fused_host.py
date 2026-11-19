#!/usr/bin/env python3
"""Temperature-fair ABBA A/B of the fused host-half dequantise, in one process.

Why this exists: the two earlier A/Bs (`par-dq-3plus3`, `par-dq-big-3plus3`) were separate
processes compared wall-to-wall, and both had >20% spread between their own trials
(1740.96/1430.34/1415.66 and 1417.77/1843.50/1422.15). AGENTS.md says a pair that differs by
more than ~20% must not be used. So those verdicts are withdrawn, not confirmed.

The fix is the method the archive uses: interleave the arms in ABBA order inside a single
thermal window and compare **paired** numbers, so a monotonic temperature ramp cancels out
instead of being attributed to the arm. Temperature is recorded for every trial so the
fairness claim is checkable rather than asserted.

Arms:
    off     PARALLEL_DEQUANT = False, serial emitter (the committed baseline behaviour)
    on      PARALLEL_DEQUANT = True,  row-split dequantise through librkpool

Usage:
  PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/ab_fused_host.py [--rounds 3]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import thread_split  # noqa: E402
from tinygrad import Tensor  # noqa: E402
from tinygrad.llm.gguf import gguf_parse  # noqa: E402
from tinygrad.llm.model import Transformer  # noqa: E402
from tinygrad.llm.tokenizer import SimpleTokenizer  # noqa: E402
import npu_linear as NL  # noqa: E402

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def temp_c() -> float:
  try:
    return float(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
  except OSError:
    return float("nan")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rounds", type=int, default=3)
  ap.add_argument("--prompt", type=int, default=128)
  ap.add_argument("--gen", type=int, default=1)
  args = ap.parse_args()

  thread_split.install()
  kv, _ = gguf_parse(CKPT)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  ids = tok.encode(PROMPT)[:args.prompt]
  model, _ = Transformer.from_gguf(CKPT, max_context=args.prompt + 32)
  NL.fuse_linears(model)
  NL.replace_linears(model)
  NL.prepare_linears(model)
  print(f"loaded, fused host arm A/B; pool={hex(NL._pool_addr() or 0)}", flush=True)

  # Same convention as bench_llm.py: positions are bound UOps (bound ONCE, because TinyJit
  # keys capture on the UOp), temperature is a Tensor.
  from tinygrad.uop.ops import UOp
  max_context = args.prompt + 32
  prompt = Tensor(ids, dtype="int32").reshape(1, args.prompt)
  temperature = Tensor([0.0])
  v_start = UOp.variable("start_pos", 0, max_context - 1)
  pos_zero = v_start.bind(0)

  def prefill(parallel: bool) -> tuple[float, float]:
    # The variant is baked into the generated source, so the PROGRAM cache must be dropped.
    NL._FUSED_PROGRAMS.clear()
    NL._FUSED_PLANS.clear()
    NL.PARALLEL_QUANT = parallel
    t = temp_c()
    t0 = time.perf_counter()
    model(prompt, (pos_zero,), temperature).realize()
    return (time.perf_counter() - t0) * 1e3, t

  # Warm-up, both arms, so JIT capture and compile are outside the measurement.
  for p in (False, True, False, True):
    prefill(p)
  print("warm-up done", flush=True)

  rows: list[tuple[int, str, float, float]] = []
  for r in range(args.rounds):
    for arm in (False, True, True, False):          # ABBA
      ms, t = prefill(arm)
      rows.append((r, "on" if arm else "off", ms, t))
      print(f"round {r} arm {'on ' if arm else 'off'} {ms:8.2f} ms  temp {t:5.1f} C", flush=True)

  print(f"\n{'round':5} {'off ms':>8} {'on ms':>8} {'ratio on/off':>13} {'dT(C)':>7}")
  ratios = []
  for r in range(args.rounds):
    off = [m for rr, a, m, _ in rows if rr == r and a == "off"]
    on = [m for rr, a, m, _ in rows if rr == r and a == "on"]
    rm = st.mean(off), st.mean(on)
    ratio = rm[1] / rm[0]
    ratios.append(ratio)
    print(f"{r:5} {rm[0]:8.2f} {rm[1]:8.2f} {ratio:12.4f}x {temp_c()-temp_c():7.1f}")
  off_all = [m for _, a, m, _ in rows if a == "off"]
  on_all = [m for _, a, m, _ in rows if a == "on"]
  print(f"\noff median {st.median(off_all):.2f} ms   on median {st.median(on_all):.2f} ms"
        f"   on/off {st.median(on_all)/st.median(off_all):.4f}x")
  print(f"paired rounds where on is faster: {sum(1 for x in ratios if x < 1.0)}/{len(ratios)}")
  print(f"temperature range across all trials: {min(t for *_, t in rows):.1f} - {max(t for *_, t in rows):.1f} C")
  spread = (max(on_all) - min(on_all)) / st.median(on_all)
  print(f"on-arm spread {100*spread:.1f}% (AGENTS: >20% invalidates)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
