"""Is the run-to-run non-determinism an uninitialised-buffer read?

Three prefills in ONE process with the KV cache pre-filled to a known value:
  A1 = fill 0.0,  B = fill 0.5,  A2 = fill 0.0 again
Interpretation:
  A1 == A2 and A1 != B  -> the model reads cache bytes it never wrote  => uninitialised read
  A1 != A2             -> non-determinism within one process, independent of the fill (race/other)
  A1 == A2 == B         -> the cache fill is irrelevant here; the earlier cross-process variation
                           must come from something else (weights, allocator addresses, NPU state)
Only the sampled tokens are observable, so this is coarse on purpose: a repeat is strong evidence,
a difference is only a hint.
"""
import sys
sys.path.insert(0, "/home/orangepi/pr/tinygrad/extra/rockchip")
from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from tinygrad.uop.ops import UOp

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
GEN = 8
model, _ = Transformer.from_gguf(CKPT, max_context=160)
toks = Tensor([[1] * 128], dtype="int32").realize()
pos = UOp.variable("start_pos", 0, 159).bind(0)
temp = Tensor([0.0])
out = model(toks, (pos,), temp).realize()          # first call primes structures

# cache_kv lives on the blocks that own one (attention blocks), not on Transformer.
holders = [(b, name) for b in model.blk for name in ("cache_kv",) if hasattr(b, name)]
print(f"blocks with cache_kv: {len(holders)} of {len(model.blk)}", flush=True)
if not holders:
  attrs = sorted({a for b in model.blk for a in vars(b) if "cache" in a})
  print("cache-ish attributes found on blocks:", attrs, flush=True)
  raise SystemExit(1)
shapes = {getattr(b, n).shape for b, n in holders}
print("cache shapes:", shapes, flush=True)


def run(fill: float, tag: str) -> None:
  for b, n in holders:
    ref = getattr(b, n)
    setattr(b, n, Tensor.full(ref.shape, fill, dtype=ref.dtype, device=ref.device).realize())
  ids, cur = [], out
  for step in range(GEN):
    cur = model(cur, (UOp.variable("start_pos", 0, 159).bind(128 + step),), temp).realize()
    ids.append(int(cur.item()))
  print(f"  {tag}: fill={fill}  ids={ids}", flush=True)


run(0.0, "A1")
run(0.5, "B ")
run(0.0, "A2")
