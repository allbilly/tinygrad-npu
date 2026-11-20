#!/usr/bin/env python3
"""Decode-only A/B: fused C wrapper vs the raw packed_matmul_int8 path, same shape.

The model only ever uses the fused wrapper (`npu_linear.FUSED_CALL = True`), and the raw path cannot
run prefill at all (`_int8_matmul_body` rejects m=128), so prefill happens once with the fused path
and the arms are measured on decode steps only.

Purpose: decide whether the k-proportional excess on `m=1, n=2560, k=4096` (418 us against the
reference's 345) lives in the fused wrapper -- quantise, payload, skip handling -- or in the device
submit. Same model, same weights, same steps; only the call form changes.

  python extra/rockchip/raw_vs_fused.py [steps]

LIMITATION (measured, not assumed): the per-name attribution does not fire for either arm. The fused
plan binds its CALL uop directly and `replay_c` invokes the compiled entry through a function
pointer, so a `CPUProgram.__call__` hook sees nothing -- the same blind spot as
`reconcile_wall.py`'s first attempt. What this script does measure reliably is the arm's wall, which
is enough to refute the raw path as an alternative but not to localise a per-call cost.
"""
import collections, pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split as TS
TS.install()
import tinygrad.codegen as CG
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

acc = collections.defaultdict(lambda: [0, 0.0])
on = [False]
orig = CPUProgram.__call__

def wrapped(self, *a, **k):
    if not on[0]: return orig(self, *a, **k)
    t0 = time.perf_counter(); r = orig(self, *a, **k)
    b = acc[self.name]; b[0] += 1; b[1] += (time.perf_counter()-t0)*1e3
    return r

CPUProgram.__call__ = wrapped
pos = 128
def step():
    global pos
    model(Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1),
          (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize()
    pos += 1

for arm in (True, False, False, True):     # ABBA, both arms in one process
    NL.FUSED_CALL = arm
    CG.to_program_cache.clear()
    model.prefill_jit.reset(); model.rollout_jit.reset()
    for _ in range(3): step()               # capture with this call form
    acc.clear()
    on[0] = True
    ts = []
    for _ in range(STEPS):
        a = time.perf_counter(); step(); ts.append((time.perf_counter()-a)*1e3)
    on[0] = False
    rows = sorted(((v[1], v[0], k) for k, v in acc.items()), reverse=True)[:3]
    shape = [(ms, c, n) for ms, c, n in rows if "2560" in n and "4096" in n]
    print(f"FUSED_CALL={int(arm)}  wall median {st.median(ts):.1f} ms  "
          f"2560x4096 -> {[(f'{ms/c*1000:.0f} us/call x{c}') for ms, c, n in shape] or 'not seen'}",
          flush=True)
    for ms, c, n in rows[:2]:
        print(f"    {n:<34}{c:>4} calls {ms:>7.2f} ms", flush=True)
