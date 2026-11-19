#!/usr/bin/env python3
"""Which lines of the shipped Rockchip path never execute, for one full prefill + decode.

Uses sys.monitoring (PEP 669), not coverage.py: each line location fires once and is then disabled,
so the tracing cost is paid only on first touch. Executable lines come from the AST, so docstrings
and comments are excluded from the denominator.

A line that does not execute in this run is not automatically dead: a decode-only or a
larger-model path may simply not be reached here. The output separates the two by listing the
unexecuted *ranges* with their source, so a reader can judge rather than trust a count.
"""
import ast, collections, pathlib, sys, time

TARGETS = {
  "extra/rockchip/npu_linear.py",
  "extra/rockchip/thread_split.py",
  "extra/rockchip/scan_native.py",
  "tinygrad/runtime/ops_rockchip.py",
  "tinygrad/llm/kernels/replay_c.py",
  "tinygrad/llm/model.py",
}
root = pathlib.Path(__file__).resolve().parent.parent.parent
resolved = {str((root / t).resolve()): t for t in TARGETS}

seen: set[tuple[str, int]] = set()
def on_line(code, line):
  f = code.co_filename
  if f in resolved:
    seen.add((f, line))
    return sys.monitoring.DISABLE
  return None

TOOL = sys.monitoring.PROFILER_ID
sys.monitoring.use_tool_id(TOOL, "deadcode")
sys.monitoring.register_callback(TOOL, sys.monitoring.events.LINE, on_line)
sys.monitoring.set_events(TOOL, sys.monitoring.events.LINE)

sys.path.insert(0, str(root / "extra/rockchip"))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC; RC.install()
from tinygrad import Tensor
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

t0 = time.perf_counter()
model(pt, (vs,), tp).realize()                      # prefill
for step in range(4):                               # decode: m=1 paths
  model(Tensor([100 + step], dtype="int32").reshape(1, 1),
        (UOp.variable("start_pos", 0, 150).bind(128 + step),), tp).realize()
sys.monitoring.set_events(TOOL, 0)
print(f"\nrun time under tracing: {time.perf_counter()-t0:.1f}s")

def executable_lines(path: pathlib.Path) -> set[int]:
  tree = ast.parse(path.read_text())
  lines: set[int] = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.stmt,)) and hasattr(node, "lineno"):
      if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant): continue   # docstring
      lines.add(node.lineno)
  return lines

for f, short in sorted(resolved.items()):
  p = pathlib.Path(f)
  if not p.exists(): continue
  want = executable_lines(p)
  got = {ln for (ff, ln) in seen if ff == f}
  miss = sorted(want - got)
  src = p.read_text().splitlines()
  print(f"\n{short}: {len(got)}/{len(want)} executable lines executed "
        f"({100*len(got)/max(1,len(want)):.0f}%), {len(miss)} never")
  # compress the missed lines into ranges
  ranges, start, prev = [], None, None
  for ln in miss:
    if start is None: start = prev = ln
    elif ln == prev + 1: prev = ln
    else: ranges.append((start, prev)); start = prev = ln
  if start is not None: ranges.append((start, prev))
  ranges.sort(key=lambda r: r[0] - r[1])
  for a, b in ranges[:6]:
    first = src[a-1].strip()[:70] if a-1 < len(src) else ""
    print(f"   {a:5}-{b:<5} ({b-a+1:3d} lines)  {first}")
