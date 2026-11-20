#!/usr/bin/env python3
"""Per-kernel CPU time over N warm decode steps, bucketed by emitted kernel name.

`cpu_kernels_by_name.py` does this for a 128-token prefill; decode is a different kernel set, and
`MANIFEST.md` is produced by a different (segment) protocol, so per-kernel decode numbers need
their own itemiser. Wrapping `CPUProgram.__call__` covers every CPU kernel, including the ones
inside a fused Linear.

`WORD_LOAD=0|1` selects the thread_split word-load rewrite, so the same script measures the row
before and after the change. Run each setting in its own process: the compile is keyed on source,
and a fresh process is the cleanest way to guarantee the arm actually compiles.

  WORD_LOAD=1 python extra/rockchip/decode_kernels_by_name.py [steps]
"""
import collections, os, pathlib, statistics as st, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split
thread_split.install()
thread_split.WORD_LOAD = os.environ.get("WORD_LOAD", "1") == "1"
from tinygrad import Tensor
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PASS_EMBD = len(sys.argv) > 2 and sys.argv[2] == "1"   # caller supplies the embedding row


def _rows(weight):
  import ctypes
  u = weight.realize().uop
  n = u.max_numel()
  return np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(u.buffer._buf)).view(np.float16).reshape(weight.shape)
kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

_EMB = _rows(model.token_embd.weight) if PASS_EMBD else None
acc = collections.defaultdict(lambda: [0, 0.0])
on = [False]
orig = CPUProgram.__call__

def wrapped(self, *a, **k):
  if not on[0]: return orig(self, *a, **k)
  t0 = time.perf_counter(); r = orig(self, *a, **k); d = (time.perf_counter() - t0) * 1e3
  b = acc[self.name]; b[0] += 1; b[1] += d
  return r

CPUProgram.__call__ = wrapped
pos = 128
def step():
  global pos
  tok = 100 + (pos % 20)
  embd = Tensor(_EMB[tok].astype(np.float32)).reshape(1, 1, -1) if PASS_EMBD else None
  model(Tensor([tok], dtype="int32").reshape(1, 1), (UOp.variable("start_pos", 0, 150).bind(pos),), tp,
        embd=embd).realize()
  pos += 1
step(); step()                                    # warm / capture
on[0] = True
for _ in range(STEPS): step()
on[0] = False

total = sum(v[1] for v in acc.values()) / STEPS
print(f"WORD_LOAD={int(thread_split.WORD_LOAD)} steps={STEPS} "
      f"rewrite_sites={thread_split.WORD_LOAD_HITS.get('sites', 0)} total CPU {total:.2f} ms/token")
print(f"{'kernel':<18}{'calls/token':>12}{'ms/token':>10}{'us/call':>9}")
for name, (n, ms) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
  print(f"{name:<18}{n/STEPS:>12.0f}{ms/STEPS:>10.2f}{ms/n*1000:>9.1f}")
