#!/usr/bin/env python3
"""Pre-sampling logits hash for a DECODE step (T=1), which `logits_gate.py` does not cover.

`logits_gate.py` runs one 128-token prefill at start_pos 0 and prints that hash. Every decode change in
this session -- greedy sampler, argmax, zero-pad skip, alpha/beta fusion, embedding hoist -- was
therefore validated against a *prefill* number, which cannot see a decode-side numerical change.

This gate does the opposite: prefills 128 tokens (so the caches are in the decode state), then runs one
T=1 step at a fixed position and hashes the pre-sampling logits.

  python extra/rockchip/decode_gate.py [--embd]
"""
import hashlib, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import thread_split; thread_split.install()
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

PASS_EMBD = "--embd" in sys.argv
CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT = ("The history of the printing press begins with movable type in East Asia. " * 12)
kv, _ = gguf_parse(CKPT); tok = SimpleTokenizer.from_gguf_kv(kv)
ids = tok.encode(PROMPT)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)


def _rows(weight):
  import ctypes
  u = weight.realize().uop
  n = u.max_numel()
  return np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(u.buffer._buf)).view(np.float16).reshape(weight.shape)


_ROWS = _rows(model.token_embd.weight) if PASS_EMBD else None


def body(tokens:Tensor, start_pos:tuple, embd:Tensor|None=None):
  t = tokens.to(model.token_embd.weight.device)
  x = (model.token_embd(t) if embd is None else embd.to(t.device)).float()
  hc = model.config.hc
  if hc is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
  elif model.config.attn_res_block_size: x = x.unsqueeze(2)
  idx_mask = None
  for block in model.blk: x, idx_mask = block(x, start_pos, idx_mask, t)
  if hc is not None: x = model.output_hc.mix(x)[0] if hasattr(model, 'output_hc') else x.mean(2)
  elif model.config.attn_res_block_size: x = model.output_res_score.mix(x)
  return model.output(model.output_norm(x[:, -1:]))[:, -1, :]

gate = TinyJit(body)
pt = Tensor(ids, dtype="int32").reshape(1, 128)
v0 = UOp.variable("start_pos", 0, 150).bind(0)
gate(pt, (v0,)).realize()                                    # prefill: fills the caches
STEP_TOK = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit() else 100
STEP_POS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 128
v1 = UOp.variable("start_pos", 0, 150).bind(STEP_POS)
st = Tensor([STEP_TOK], dtype="int32").reshape(1, 1)
emb = Tensor(_ROWS[STEP_TOK].astype(np.float32)).reshape(1, 1, -1) if PASS_EMBD else None
out = gate(st, (v1,), emb).realize().numpy().astype(np.float32)
h = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()[:16]
print(f"DECODE{' embd' if PASS_EMBD else ' mask'}  presample_logits_sha256_16 {h}  "
      f"shape {out.shape}  finite={np.isfinite(out).all()}")
