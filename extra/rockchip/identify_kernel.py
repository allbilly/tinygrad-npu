#!/usr/bin/env python3
"""Print the C source of named CPU kernels, captured at compile time (usage: identify_kernel.py NAME...)."""
import pathlib, re, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC; RC.install()
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
WANT = set(sys.argv[1:])
SRC: dict[str, str] = {}
_orig = ClangCompiler.compile_cached
ANSI = re.compile(r"\x1b\[[0-9;]*m")
def cap(self, src: str):
    m = re.search(r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", src)
    if m:
        name = ANSI.sub("", m.group(1))
        if name not in SRC and any(name.startswith(w) for w in WANT):
            import traceback
            SRC[name] = src
            print(f"--- first compile of {name}, stack:")
            for line in traceback.format_stack()[:40]:
                f = line.strip().split(chr(10))[0]
                if 'site-packages' in f or '/.venv/' in f: continue
                print('   ', f)
    return _orig(self, src)
ClangCompiler.compile_cached = cap
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL
CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
kv, _ = gguf_parse(CKPT); ids = SimpleTokenizer.from_gguf_kv(kv).encode("x " * 128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0]); pt = Tensor(ids, dtype="int32").reshape(1, len(ids))
model(pt, (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()
for s in range(2):
    model(Tensor([100 + s], dtype="int32").reshape(1, 1), (UOp.variable("start_pos", 0, 150).bind(len(ids) + s),), tp).realize()
print(f"captured {len(SRC)} kernels; names: {sorted(SRC)[:8]}")
for name, src in sorted(SRC.items()):
    if not any(name.startswith(w) for w in WANT): continue
    print(f"===== {name} ({len(src.splitlines())} lines)")
    print(src[:1200])
