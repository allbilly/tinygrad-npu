#!/usr/bin/env python3
"""In-process ABBA for a decode-path change, with per-trial temperature.

`bench_llm` loads the model once per process (63 s) and records no per-trial temperature inside a
run, so an 8 ms effect cannot be separated from a 1%/C thermal drift. Here both arms run in one
process, alternating A,B,B,A, with the JIT re-captured per arm and temperature read before every
block. Report paired ratios, not independent medians.

Usage: ab_decode.py [rounds] [steps]
"""
import pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tinygrad.codegen as CG
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
import tinygrad.llm.model as M
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(flag: bool) -> tuple[float, float, float]:
  """Set the flag, re-capture both JITs, warm, then time STEPS decode steps."""
  M.GREEDY_SAMPLING = flag
  CG.to_program_cache.clear()   # M83: the JIT reset alone reuses the other arm's binary
  model.prefill_jit.reset(); model.rollout_jit.reset()
  pos = 128
  def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  # Four warm steps, not two: this harness clears `to_program_cache`, which forces ~1000 kernels
  # to recompile per arm, and with two warm steps some of that landed in the measured block
  # (bimodal 171 vs 248 ms with the median corrupted).
  for _ in range(4): step()                                   # capture + warm
  t0, t1 = temp_c(), 0.0
  ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  t1 = temp_c()
  return st.median(ts), t0, t1

res = {"A": [], "B": []}
for r in range(ROUNDS):
  for name, flag in (("A", True), ("B", False), ("B", False), ("A", True)):
    ms, t0, t1 = arm(flag)
    res[name].append(ms)
    print(f"round {r+1} arm {name} ({'greedy' if flag else 'gumbel'}): {ms:.2f} ms/token "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C", flush=True)

a, b = res["A"], res["B"]
print(f"\nA greedy : {[round(x,1) for x in a]}  median {st.median(a):.2f} ms  ({1000/st.median(a):.3f} tok/s)")
print(f"B gumbel : {[round(x,1) for x in b]}  median {st.median(b):.2f} ms  ({1000/st.median(b):.3f} tok/s)")
pairs = [(x, y) for x, y in zip(a, b)]
print(f"paired A/B ratios (greedy/gumbel): {[round(x/y,4) for x, y in pairs]}")
print(f"median ratio {st.median([x/y for x, y in pairs]):.4f}  (below 1.0 = greedy faster)")
