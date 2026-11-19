#!/usr/bin/env python3
"""In-process ABBA for the attention group split (attn_native.SPLIT_DECODE).

Both arms in one process, A,B,B,A, JIT re-captured per arm, temperature per block. Each arm clears
`attn_native._PROGRAM` and flips the flag, then prints the arm's generated `#if RK_SPLIT` line so a
silent non-flip is visible (the failure mode that invalidated an earlier fill test).
"""
import pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
import attn_native as AN
import tinygrad.codegen as CG
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

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(flag: bool) -> tuple[float, float, float, str]:
  AN.SPLIT_DECODE = flag
  AN._PROGRAM.clear()                     # force a rebuild with the new #if
  # Rebuilding the source is not enough: `to_program_cache` memoises the PROGRAM uop *including
  # the compiled binary*, keyed on a structurally identical ast, so without this the second arm
  # runs the first arm's binary. Both arms clear it.
  CG.to_program_cache.clear()
  model.prefill_jit.reset(); model.rollout_jit.reset()
  src = AN._source(1, 160)
  line = next((l for l in src.splitlines() if l.strip().startswith("#if RK_SPLIT") or
               l.strip() in ("#if 1", "#if 0")), "?")
  pos = 128
  def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  step(); step()
  t0 = temp_c(); ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  return st.median(ts), t0, temp_c(), line

res = {"A": [], "B": []}
for r in range(ROUNDS):
  for name, flag in (("A", True), ("B", False), ("B", False), ("A", True)):
    ms, t0, t1, line = arm(flag)
    res[name].append(ms)
    print(f"round {r+1} arm {name} ({'split 4' if flag else 'serial'}): {ms:.2f} ms/token "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C  src={line!r}", flush=True)

a, b = res["A"], res["B"]
print(f"\nA split : {[round(x,1) for x in a]}  median {st.median(a):.2f} ms ({1000/st.median(a):.3f} tok/s)")
print(f"B serial: {[round(x,1) for x in b]}  median {st.median(b):.2f} ms ({1000/st.median(b):.3f} tok/s)")
print(f"paired A/B ratios: {[round(x/y,4) for x, y in zip(a, b)]}")
print(f"median ratio {st.median([x/y for x, y in zip(a, b)]):.4f}  (below 1.0 = split faster)")
