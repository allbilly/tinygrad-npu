#!/usr/bin/env python3
"""The fused Linear's own C-side counters, read across ONE real warm prefill.

`dequant_per_geom.py` drives each geometry standalone and multiplies by the call count, which is
the multiplication AGENTS.md forbids: the in-model cost of the same loop was measured 4x its
isolated cost on the scan. The fused C already accumulates submit / quantise / dequantise
nanoseconds into fused_stats[0], [4] and [6], so the honest numbers need one model run and zero
arithmetic.

Residual = wall - submit - quantise - dequantise is what the fused C spends outside its three
counted phases; it is compared against the per-name table's non-fused total, which is the rest of
the forward.
"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Device, Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT = ("The history of the printing press begins with movable type in East Asia. " * 12)
kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode(PROMPT)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt = Tensor(ids, dtype="int32").reshape(1, 128); tp = Tensor([0.0])
vs = UOp.variable("start_pos", 0, 150).bind(0)
hw = Device["ROCKCHIP:0"].hw
for _ in range(3): model(pt, (vs,), tp).realize()
b = [hw.fused_stats[i] for i in range(7)]
t0 = time.perf_counter(); model(pt, (vs,), tp).realize(); wall = (time.perf_counter() - t0) * 1e3
d = [(hw.fused_stats[i] - b[i]) for i in range(7)]
submit, calls, tasks, quant, deq = d[0] / 1e6, d[1], d[2], d[4] / 1e6, d[6] / 1e6
by_name = json.loads((pathlib.Path(__file__).resolve().parent.parent.parent /
                      "docs/rknpu/results/attr-by-name/inmodel_by_name.json").read_text())
non_linear = by_name["non_linear_ms"]
print(f"wall                     {wall:8.2f} ms")
print(f"fused calls / tasks      {calls:5} / {tasks}")
print(f"  submit (device)        {submit:8.2f} ms   {100*submit/wall:5.1f}%")
print(f"  quantise (host)        {quant:8.2f} ms   {100*quant/wall:5.1f}%")
print(f"  dequantise (host)      {deq:8.2f} ms   {100*deq/wall:5.1f}%")
fused_counted = submit + quant + deq
print(f"  counted inside fused   {fused_counted:8.2f} ms")
print(f"fused Linear wall (by name) {by_name['fused_linear_ms']:8.2f} ms  over {by_name['fused_linear_calls']} calls")
print(f"  fused outside counters {by_name['fused_linear_ms']-fused_counted:8.2f} ms")
print(f"non-fused kernels        {non_linear:8.2f} ms  (one-call-segment table, same protocol)")
print(f"sum accounted            {fused_counted+non_linear:8.2f} ms")
print(f"outside every counter    {wall-fused_counted-non_linear:8.2f} ms")
print(f"quant pool hits (st[3])  {d[3]} of {calls} calls")
print(f"deq pool hits   (st[5])  {d[5]} of {calls} calls")
try:
    import ctypes
    lib = ctypes.CDLL(str(pathlib.Path(__file__).resolve().parent / "librkpool.so"))
    print(f"rkpool: dispatch calls {lib.rk_calls()}  refused/serial {lib.rk_refused()}")
except Exception as e:
    print(f"rkpool counters unavailable: {type(e).__name__} {e}")
