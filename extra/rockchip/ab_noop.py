#!/usr/bin/env python3
"""In-process ABBA that ablates a set of CPU kernels by replacing their bodies with a no-op.

Purpose: measure the **transfer factor** from kernel time to wall time. A row's cost in the
manifest (`ms` = sum of per-kernel self time) is not a promise about the wall: removing the fills
took 15 ms of kernel and 5 ms of wall. So before optimising a row, ablate it and see what the
wall does.

Numerics are deliberately garbage in arm B -- this is an ablation, not a candidate change, so the
pre-sampling parity gate does not apply and must not be run against it.

Both arms clear `to_program_cache` (see mistake.md M83): resetting the JIT alone leaves the first
arm's compiled binaries in place and the ABBA compares an arm with itself.

  python extra/rockchip/ab_noop.py <rounds> <steps> <name>[,<name>...]
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
TARGETS = set((sys.argv[3] if len(sys.argv) > 3 else "").split(",")) - {""}
if not TARGETS: raise SystemExit("give at least one kernel name to ablate")
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

def temp_c() -> float: return float(TEMP.read_text()) / 1000

def arm(noop: bool) -> tuple[float, float, float, int]:
  TS.NOOP.clear()
  if noop: TS.NOOP.update(TARGETS)
  CG.to_program_cache.clear()
  model.prefill_jit.reset(); model.rollout_jit.reset()
  before = sum(TS.NOOP_HITS.values())
  pos = 128
  def step():
    nonlocal pos
    t = Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1)
    model(t, (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize(); pos += 1
  step(); step()
  hits = sum(TS.NOOP_HITS.values()) - before
  t0 = temp_c(); ts = []
  for _ in range(STEPS):
    a = time.perf_counter(); step(); ts.append((time.perf_counter() - a) * 1e3)
  return st.median(ts), t0, temp_c(), hits

res = {"A": [], "B": []}
hits_a, hits_b = [], []
for r in range(ROUNDS):
  for name, noop in (("A", False), ("B", True), ("B", True), ("A", False)):
    ms, t0, t1, hits = arm(noop)
    res[name].append(ms); (hits_a if name == "A" else hits_b).append(hits)
    print(f"round {r+1} arm {name} ({'noop' if noop else 'real'}): {ms:.2f} ms/token "
          f"({1000/ms:.3f} tok/s) temp {t0:.1f}->{t1:.1f} C  noop_hits={hits}", flush=True)

if all(h == 0 for h in hits_b): raise SystemExit("ARM B DID NOT ABLATE ANYTHING -- placebo")
if any(h != 0 for h in hits_a): raise SystemExit("ARM A ABLATED SOMETHING -- arms are not distinct")

a, b = res["A"], res["B"]
print(f"\nA real: {[round(x,1) for x in a]}  median {st.median(a):.2f} ms ({1000/st.median(a):.3f} tok/s)")
print(f"B noop: {[round(x,1) for x in b]}  median {st.median(b):.2f} ms ({1000/st.median(b):.3f} tok/s)")
print(f"paired A/B ratios: {[round(x/y,4) for x, y in zip(a, b)]}")
print(f"median ratio {st.median([x/y for x, y in zip(a, b)]):.4f}  (above 1.0 = ablation faster)")
print(f"wall saved by ablating {sorted(TARGETS)}: {st.median(a)-st.median(b):+.2f} ms")
