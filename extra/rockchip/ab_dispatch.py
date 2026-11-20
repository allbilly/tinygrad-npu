#!/usr/bin/env python3
"""In-process ABBA for the per-call CPU profiling overhead (ops_cpu.PROFILE).

Arm A skips the never-selected zero branch of `where(0, recurrent_state)` at decode; arm B keeps it.
The branch emits `E_131072_4`, a pure 2 MB zero fill, 48 times per token = 96 MB of zeros for
8.76 ms/token, so this is the local form of the reference's `ONPU_ZERO_PAD_VIEWS`: stop
materialising padding that is never read.

Both arms clear `to_program_cache` (mistake.md M83) and the temperature is read before every trial.

  python extra/rockchip/ab_nopad.py <rounds> <steps>
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
import tinygrad.llm.model as M
import tinygrad.runtime.ops_cpu as ops_cpu
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(skip: bool) -> tuple[float, float, float]:
  ops_cpu.PROFILE = skip   # True = the old path that builds a ProfileRangeEvent per call
  CG.to_program_cache.clear()
  model.prefill_jit.reset(); model.rollout_jit.reset()
  pos = 128
  def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  step(); step()
  t0 = temp_c(); ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  return st.median(ts), t0, temp_c()

res = {"A": [], "B": []}
for r in range(ROUNDS):
  for name, skip in (("A", True), ("B", False), ("B", False), ("A", True)):
    ms, t0, t1 = arm(skip)
    res[name].append(ms)
    print(f"round {r+1} arm {name} ({'no-pad' if skip else 'zero-pad'}): {ms:.2f} ms/token "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C", flush=True)

a, b = res["A"], res["B"]
r = [x/y for x, y in zip(a, b)]
print(f"\nA no-pad  : {[round(x,1) for x in a]}  median {st.median(a):.2f} ms ({1000/st.median(a):.3f} tok/s)")
print(f"B zero-pad: {[round(x,1) for x in b]}  median {st.median(b):.2f} ms ({1000/st.median(b):.3f} tok/s)")
print(f"paired A/B ratios: {[round(x,4) for x in r]}")
print(f"median ratio {st.median(r):.4f}  ({sum(1 for x in r if x < 1)}/{len(r)} pairs favour no-pad)")
print(f"wall saved by bypassing cpu_profile: {st.median(b)-st.median(a):+.2f} ms")
