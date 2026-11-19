#!/usr/bin/env python3
"""Dump every CPU kernel of one decode step: its generated C source and its measured time.

Produces docs/rknpu/results/decode-kernels/:
  <kernel>.c      the exact source clang compiled for that kernel
  MANIFEST.md     per-kernel calls, total ms, us/call, and the class rollup

Timing is in-model: one call per replay segment, so a segment's duration is that kernel's duration
(including its blocking NPU submit where it has one). CPU time is duration minus the submit delta.
"""
import collections, pathlib, re, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad import Device, Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SRC: dict[str, str] = {}
_orig = ClangCompiler.compile_cached
def cap(self, src: str):
  m = re.search(r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", src)
  if m: SRC.setdefault(ANSI.sub("", m.group(1)), src)
  return _orig(self, src)
ClangCompiler.compile_cached = cap

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()
for s in range(5):
  model(Tensor([100+s], dtype="int32").reshape(1, 1), (UOp.variable("start_pos", 0, 150).bind(128+s),), tp).realize()
hw = Device["ROCKCHIP:0"].hw
RC.TRACE = []
t0 = time.perf_counter()
model(Tensor([200], dtype="int32").reshape(1, 1), (UOp.variable("start_pos", 0, 150).bind(133),), tp).realize()
wall = (time.perf_counter() - t0) * 1e3
tr = RC.TRACE; RC.TRACE = None

per = collections.Counter(); cnt = collections.Counter(); sub = collections.Counter()
for name, d, s_ in tr:
  n = ANSI.sub("", str(name)); per[n] += d; cnt[n] += 1; sub[n] += s_
out = HERE.parent.parent / "docs/rknpu/results/decode-kernels"
out.mkdir(parents=True, exist_ok=True)
for n in per:
  if n in SRC: (out / f"{n}.c").write_text(SRC[n])

def klass(n: str) -> str:
  if n.startswith("rockchip_linear"): return "fused INT8 Linear"
  if n.startswith("memset_"): return "fill (memset)"
  if n.startswith("E_131072") or n.startswith("E_81920"): return "fill (kernel)"
  if "scan" in n: return "GDN scan"
  if "attn" in n: return "attention"
  if n.startswith("E"): return "elementwise"
  if n.startswith("r"): return "reduce"
  return "other"

cls = collections.Counter()
for n, ms in per.items(): cls[klass(n)] += ms
rows = sorted(per.items(), key=lambda kv: -kv[1])
lines = [f"# Decode kernels: source and in-model timing", "",
         f"One warm decode token, `MAX_CALLS_PER_SEG = 1`. Wall **{wall:.2f} ms**, "
         f"{len(tr)} calls, {len(per)} distinct kernels.", "",
         "| kernel | class | calls | ms | us/call | submit ms | cpu ms | source |",
         "| --- | --- | --: | --: | --: | --: | --: | --- |"]
for n, ms in rows[:40]:
  lines.append(f"| `{n}` | {klass(n)} | {cnt[n]} | {ms:.2f} | {ms/cnt[n]*1000:.1f} | "
               f"{sub[n]:.2f} | {ms-sub[n]:.2f} | `{n}.c` |")
lines += ["", "## Class rollup", "", "| class | ms | share |", "| --- | --: | --: |"]
tot = sum(per.values())
for c, ms in cls.most_common():
  lines.append(f"| {c} | {ms:.2f} | {100*ms/tot:.1f}% |")
lines += ["", f"sum of kernels {tot:.2f} ms; wall {wall:.2f} ms; outside {wall-tot:.2f} ms", ""]
(out / "MANIFEST.md").write_text("\n".join(lines))
print(f"wrote {len([n for n in per if n in SRC])} sources, {len(per)} kernels, wall {wall:.2f} ms")
print("\n".join(lines[-4-len(cls):-2]))
