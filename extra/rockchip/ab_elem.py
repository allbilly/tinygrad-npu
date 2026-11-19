#!/usr/bin/env python3
"""In-process ABBA for the elementwise four-way transforms (thread_split.THREAD_SPLIT).

Arm A keeps the transform table, arm B empties it (all those kernels run single-threaded). Both arms
run in one process, A,B,B,A, JIT re-captured per arm, temperature per block. `thread_split.HITS`
is read before/after each capture so an arm that did not actually transform anything is visible.
"""
import pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split as TS
TS.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")
FULL = dict(TS.THREAD_SPLIT)

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(on: bool) -> tuple[float, float, float, int]:
  TS.THREAD_SPLIT.clear(); TS.THREAD_SPLIT.update(FULL if on else {})
  model.prefill_jit.reset(); model.rollout_jit.reset()
  before = sum(TS.HITS.values()) if hasattr(TS, "HITS") else 0
  pos = 128
  def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  step(); step()
  hits = (sum(TS.HITS.values()) if hasattr(TS, "HITS") else 0) - before
  t0 = temp_c(); ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  return st.median(ts), t0, temp_c(), hits

res = {"A": [], "B": []}
for r in range(ROUNDS):
  for name, on in (("A", True), ("B", False), ("B", False), ("A", True)):
    ms, t0, t1, hits = arm(on)
    res[name].append(ms)
    print(f"round {r+1} arm {name} ({'4-way' if on else 'serial'}): {ms:.2f} ms/token "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C  transform_hits={hits}", flush=True)

a, b = res["A"], res["B"]
print(f"\nA 4-way : {[round(x,1) for x in a]}  median {st.median(a):.2f} ms ({1000/st.median(a):.3f} tok/s)")
print(f"B serial: {[round(x,1) for x in b]}  median {st.median(b):.2f} ms ({1000/st.median(b):.3f} tok/s)")
print(f"paired A/B ratios: {[round(x/y,4) for x, y in zip(a, b)]}")
print(f"median ratio {st.median([x/y for x, y in zip(a, b)]):.4f}  (below 1.0 = 4-way faster)")
