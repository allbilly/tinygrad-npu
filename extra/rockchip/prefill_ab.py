#!/usr/bin/env python3
"""Prefill A/B for the attn_qkv+attn_gate fusion (npu_linear.FUSE_QKV_GATE).

The decode side of this fusion is already measured (-0.71 ms, 283->259 submits, parity unchanged,
24.86 ms against the reference's 24.52 for the same 12288-wide shape). It was reverted on a single
1+1 prefill spot-check that moved +2.2%, which one sample cannot distinguish from noise.

This runs prefill in the *native* configuration -- the one whose 1,198.87 ms target is frozen --
three times, and prints every trial so the caller can pair runs across processes. The flag is set
from argv, so the two arms are the same code path with one boolean differs.

  python extra/rockchip/prefill_ab.py <0|1> [trials]
"""
import pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split as TS
TS.install()
import tinygrad.codegen as CG
from tinygrad.llm.kernels import replay_c as RC
RC.install()
import npu_linear as NL
NL.FUSE_QKV_GATE = sys.argv[1] == "1"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
TEMP = pathlib.Path("/sys/class/thermal/thermal_zone0/temp")
kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
fused = NL.fuse_linears(model)
NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
prompt = Tensor(ids, dtype="int32").reshape(1, len(ids))
pos = UOp.variable("start_pos", 0, 150).bind(0)
model(prompt, (pos,), tp).realize()          # capture + first run
ts = []
for _ in range(TRIALS):
    t0 = float(TEMP.read_text())/1000
    a = time.perf_counter(); model(prompt, (pos,), tp).realize(); ts.append((time.perf_counter()-a)*1e3)
    print(f"  trial {len(ts)}: {ts[-1]:.1f} ms  temp {t0:.1f} C", flush=True)
print(f"FUSE_QKV_GATE={int(NL.FUSE_QKV_GATE)} fused_groups={fused} "
      f"trials={[round(x,1) for x in ts]} median {st.median(ts):.1f} ms")
