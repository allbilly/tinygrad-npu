#!/usr/bin/env python3
"""In-process ABBA for the word-load rewrite (thread_split.WORD_LOAD).

Arm A compiles the emitted kernels unmodified, arm B rewrites every "four consecutive byte loads
assembled into one f32" into a single 4-byte load (bit-exact on little-endian). Both arms run in
one process, A,B,B,A, JIT re-captured and `to_program_cache` cleared per arm -- resetting the JIT
alone leaves the first arm's binaries in place (mistake.md M83) -- and the temperature is read
before every trial.

  python extra/rockchip/ab_wordload.py <rounds> <steps>
"""
import pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split as TS
TS.install()
import tinygrad.codegen as CG
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")
MODE = sys.argv[3] if len(sys.argv) > 3 else "decode"   # "decode" or "prefill"

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(on: bool) -> tuple[float, float, float, int]:
  TS.WORD_LOAD = on
  CG.to_program_cache.clear()
  model.prefill_jit.reset(); model.rollout_jit.reset()
  before = TS.WORD_LOAD_HITS.get("sites", 0)
  pos = 128
  if MODE == "prefill":
    # One full 128-token prefill per step: this is the frozen TTFT target, and the rewrite also
    # lands on prefill kernels (E_2_16_32_32_4_4 16 sites, E_1024_32_4_4 4, E_32_32_128_4 4).
    def step():
      model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()
  else:
   def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  step(); step()
  hits = TS.WORD_LOAD_HITS.get("sites", 0) - before
  t0 = temp_c(); ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  return st.median(ts), t0, temp_c(), hits

res = {"A": [], "B": []}
hits_a, hits_b = [], []
for r in range(ROUNDS):
  for name, on in (("A", False), ("B", True), ("B", True), ("A", False)):
    ms, t0, t1, hits = arm(on)
    res[name].append(ms); (hits_a if name == "A" else hits_b).append(hits)
    print(f"round {r+1} arm {name} ({'word-load' if on else 'byte-load'}): {ms:.2f} ms/{MODE} "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C  rewrite_sites={hits}", flush=True)

if all(h == 0 for h in hits_b): raise SystemExit("ARM B DID NOT REWRITE ANYTHING -- placebo")
if any(h != 0 for h in hits_a): raise SystemExit("ARM A REWROTE SOMETHING -- arms are not distinct")

a, b = res["A"], res["B"]
print(f"\nA byte-load: {[round(x,1) for x in a]}  median {st.median(a):.2f} ms ({1000/st.median(a):.3f} tok/s)")
print(f"B word-load: {[round(x,1) for x in b]}  median {st.median(b):.2f} ms ({1000/st.median(b):.3f} tok/s)")
r = [x/y for x, y in zip(a, b)]
print(f"paired A/B ratios: {[round(x,4) for x in r]}")
print(f"median ratio {st.median(r):.4f}  ({sum(1 for x in r if x > 1)}/{len(r)} pairs favour word-load)")
