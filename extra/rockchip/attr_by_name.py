#!/usr/bin/env python3
"""Per-kernel in-model time under the replay, by name.

With MAX_CALLS_PER_SEG = 1 every emitted call is its own C segment, so timing the segment times
that one kernel in-model -- no isolated repro, no multiplication. Numerics and call order are
unchanged, so the logits hash is unaffected; only the segment boundaries move.

This is the measurement AGENTS.md demands in place of `cpu_kernels_by_name.py`, whose numbers are
isolated and therefore not the in-model cost (measured 4x discrepancy on the scan in an earlier
session).
"""
import collections, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODE = sys.argv[1] if len(sys.argv) > 1 else "prefill"
if MODE != "decode-nothreads":
    import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT = ("The history of the printing press begins with movable type in East Asia. " * 12)
kv, _ = gguf_parse(CKPT)
tok = SimpleTokenizer.from_gguf_kv(kv)
ids = tok.encode(PROMPT)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt = Tensor(ids, dtype="int32").reshape(1, 128)
tp = Tensor([0.0])
vs = UOp.variable("start_pos", 0, 150).bind(0)

seen = []
_orig = RC.Compiled.run
def run(self, ctx, opaque_runner):
    if not seen or seen[-1] is not self: seen.append(self)
    return _orig(self, ctx, opaque_runner)
RC.Compiled.run = run

for _ in range(3): model(pt, (vs,), tp).realize()
if MODE in ("decode", "decode-nothreads"):
    # one warm decode step so the m=1 plans are captured, then the measured one
    for step in range(6):
        model(Tensor([100 + step], dtype="int32").reshape(1, 1),
              (UOp.variable("start_pos", 0, 150).bind(128 + step),), tp).realize()
RC.stats["by_seg"].clear(); RC.stats["c_ms"] = 0.0; RC.stats["segs"] = 0
if MODE in ("decode", "decode-nothreads"):
    t0 = time.perf_counter()
    model(Tensor([200], dtype="int32").reshape(1, 1), (UOp.variable("start_pos", 0, 150).bind(135),), tp).realize()
    wall = (time.perf_counter() - t0) * 1e3
else:
    t0 = time.perf_counter(); model(pt, (vs,), tp).realize(); wall = (time.perf_counter() - t0) * 1e3
comp = seen[-1]   # the decode JIT when MODE=decode, the prefill JIT otherwise
# one call per segment, so plan order maps 1:1 onto comp.names
mapping, i = {}, 0
for kind, payload in comp.plan:
    if kind == "c":
        mapping[payload] = comp.names[i] if i < len(comp.names) else f"call{i}"
        i += 1
per = collections.Counter(); cnt = collections.Counter()
for (cid, seg), ms in RC.stats["by_seg"].items():
    if cid != id(comp): continue
    name = mapping.get(seg, seg)
    per[name] += ms; cnt[name] += 1
total = sum(per.values())
print(f"mode                 {MODE}")
print(f"wall                 {wall:9.2f} ms")
print(f"segments c_ms        {RC.stats['c_ms']:9.2f} ms over {RC.stats['segs']} segments")
print(f"sum by name          {total:9.2f} ms over {sum(cnt.values())} calls")
print(f"\n{'kernel':28} {'total ms':>9} {'calls':>6} {'us/call':>9} {'%':>6}")
for name, ms in per.most_common(18):
    print(f"{name[:28]:28} {ms:9.2f} {cnt[name]:6} {ms/cnt[name]*1000:9.1f} {100*ms/total:5.1f}%")
print(f"\nremaining names      {len(per)-18} distinct, {total-sum(m for _,m in per.most_common(18)):.2f} ms")
import json
out = pathlib.Path(__file__).resolve().parent.parent.parent / "docs/rknpu/results/attr-by-name"
out.mkdir(parents=True, exist_ok=True)
lin = sum(ms for name, ms in per.items() if name.startswith("rockchip_linear"))
(out / "inmodel_by_name.json").write_text(json.dumps({
    "wall_ms": wall, "segments_c_ms": RC.stats["c_ms"], "segments": RC.stats["segs"],
    "total_by_name_ms": total, "outside_run_linear_ms": wall - total,
    "fused_linear_ms": lin,
    "fused_linear_calls": sum(c for n, c in cnt.items() if n.startswith("rockchip_linear")),
    "non_linear_ms": total - lin,
    "by_name": {n: {"ms": round(ms, 3), "calls": cnt[n]} for n, ms in per.most_common()},
}, indent=1))
print(f"wrote {out/'inmodel_by_name.json'}")
