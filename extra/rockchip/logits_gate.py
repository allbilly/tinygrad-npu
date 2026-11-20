#!/usr/bin/env python3
"""Deterministic parity gate: hash the PRE-SAMPLING logits.

`Transformer.forward` applies the Gumbel-max trick (model.py:908) on every call, so its output is a
sample and no hash of it can gate anything (M63). This reproduces forward's body exactly, stops
before the sampling step, and hashes the logits tensor. Wrapped in its own TinyJit so the JIT path
(jit=True) is exercised, which is what the replay hooks.
"""
import hashlib, os, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
PASS_EMBD = "--embd" in sys.argv          # caller supplies the embedding row (hoisted lookup)
import thread_split; thread_split.install()
mod = None
if os.environ.get("NO_REPLAY") != "1":
    from tinygrad.llm.kernels import replay_c as mod
    mod.install()
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL
CKPT="/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT=("The history of the printing press begins with movable type in East Asia. "*12)
kv,_=gguf_parse(CKPT); tok=SimpleTokenizer.from_gguf_kv(kv); ids=tok.encode(PROMPT)[:128]
model,_=Transformer.from_gguf(CKPT,max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)

def body(tokens:Tensor, start_pos:tuple):
    x = model.token_embd(t:=tokens.to(model.token_embd.weight.device)).float()
    hc = model.config.hc
    if hc is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    elif model.config.attn_res_block_size: x = x.unsqueeze(2)
    idx_mask = None
    for block in model.blk: x, idx_mask = block(x, start_pos, idx_mask, t)
    if hc is not None: x = model.output_hc.mix(x)[0] if hasattr(model, 'output_hc') else x.mean(2)
    elif model.config.attn_res_block_size: x = model.output_res_score.mix(x)
    return model.output(model.output_norm(x[:, -1:]))[:, -1, :]
gate = TinyJit(body)

pt=Tensor(ids,dtype="int32").reshape(1,128); vs=UOp.variable("start_pos",0,150).bind(0)
gate(pt, (vs,)).realize()                   # warm-up / capture
out = gate(pt, (vs,)).realize().numpy().astype(np.float32)
h = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()[:16]
# The gate hashed an all-zero tensor for the whole session (895365e27ed4255b IS sha256 of zeros),
# so it certified nothing. Refuse to print a verdict unless the logits are real and non-degenerate.
_zero = hashlib.sha256(np.ascontiguousarray(np.zeros_like(out)).tobytes()).hexdigest()[:16]
if h == _zero or not np.any(out != 0):
  print(f"GATE INVALID: logits are all zero (hash {h}); this gate cannot certify anything")
  raise SystemExit(2)
print(f"ARM {'replay' if mod else 'stock'}  presample_logits_sha256_16 {h}  shape {out.shape}  finite={np.isfinite(out).all()}")
if mod is not None:
    s=mod.stats; print(f"     replay built={s['built']} segs={s['segments']} c_calls={s['calls_c']} py_calls={s['calls_py']}")
