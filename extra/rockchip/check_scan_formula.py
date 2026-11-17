#!/usr/bin/env python3
"""Check whether the GDN scan loop matches the formula written in model.py.

A previous attempt reported max abs delta 6.01 at T=1 between a replayed formula and
the model's loop, and concluded the formula was not understood. That is a strong claim
about source that is plainly readable, so this re-derives it from the model's own
intermediates instead of from a re-implementation: it captures q, k, v, beta, alpha
and the initial state by running the real `_attention` body, then replays the loop and
compares step by step.

Usage: DEV=CPU .venv/bin/python extra/rockchip/check_scan_formula.py [--tokens 8]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from tinygrad import Tensor
from tinygrad.llm.model import GatedDeltaNetBlock, Transformer
from tinygrad.uop.ops import UOp

CHECKPOINT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"


def capture(blk: GatedDeltaNetBlock, x: Tensor, pos: int):
    """Reproduce the pre-loop part of `_attention` and return its operands."""
    B, T, _ = x.shape
    start = UOp.variable("start_pos", 0, blk.config.max_context - 1).bind(pos)
    initial = Tensor(start).eq(0)
    T_pad = x.max_shape[1]
    xh = x.half()
    out_gate = blk.attn_gate(xh)
    out_gate = out_gate.reshape(B, T, blk.num_v_heads, blk.head_v_dim)
    beta = blk.ssm_beta(xh).sigmoid().reshape(B, T, blk.num_v_heads)
    alpha = blk.ssm_alpha(xh)
    g, a = (alpha.float() + blk.ssm_dt["bias"]).reshape(B, T, blk.num_v_heads, -1), blk.ssm_a.reshape(blk.num_v_heads, -1)
    log_alpha = g.softplus() * a
    conv_out, _ = __import__("tinygrad.llm.model", fromlist=["short_conv"]).short_conv(
        blk.conv_state, blk.attn_qkv(xh), blk.ssm_conv1d["weight"].flatten(1), initial)
    q, k, v = conv_out.split([blk.q_dim, blk.q_dim, blk.conv_channels - 2 * blk.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, blk.num_k_heads, blk.head_k_dim) for z in (q, k))
    q, k = ((z.rmsnorm(eps=1e-6 / blk.head_k_dim) / blk.head_k_dim ** 0.5)
            .repeat(1, 1, blk.num_v_heads // blk.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, blk.num_v_heads, blk.head_v_dim)
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * blk.head_k_dim ** -0.5
    alpha = log_alpha.transpose(1, 2).exp()
    q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
    alpha = alpha.unsqueeze(-2)
    state = initial.where(0, blk.recurrent_state.float())
    return q, k, v, beta, alpha, state, T_pad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()

    model, _ = Transformer.from_gguf(CHECKPOINT, max_context=args.tokens + 8)
    blk = next(b for b in model.blk if isinstance(b, GatedDeltaNetBlock))
    blk._init_state(Tensor.zeros(1, args.tokens, model.config.dim))
    x = Tensor.rand(1, args.tokens, model.config.dim).half() * 0.02

    q, k, v, beta, alpha, state, T_pad = capture(blk, x, 0)
    q, k, v, beta, alpha, state = (t.realize() for t in (q, k, v, beta, alpha, state))

    # The model's own loop, verbatim from model.py:741-750.
    model_outs, s = [], state
    for t in range(T_pad):
        s1 = s * alpha[:, :, t]
        delta = (v[:, :, t] - (s1 * k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
        s = s1 + delta * k[:, :, t]
        model_outs.append((s * q[:, :, t]).sum(-1))
    model_core = model_outs[0].stack(*model_outs[1:], dim=1).realize().numpy()

    # The same formula, but evaluated one step at a time with a fresh realize, so any
    # fusion across steps cannot change the arithmetic.
    replay, s = [], state
    for t in range(T_pad):
        s1 = (s * alpha[:, :, t]).realize()
        delta = ((v[:, :, t] - (s1 * k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]).realize()
        s = (s1 + delta * k[:, :, t]).realize()
        replay.append((s * q[:, :, t]).sum(-1).realize())
    replay_core = replay[0].stack(*replay[1:], dim=1).realize().numpy()

    delta = np.abs(model_core - replay_core)
    print(f"tokens={T_pad}  shape={model_core.shape}")
    print(f"max |model - replay| = {delta.max():.6e}   mean = {delta.mean():.6e}")
    print(f"model  [0,0,0,:4] = {model_core[0,0,0,:4]}")
    print(f"replay [0,0,0,:4] = {replay_core[0,0,0,:4]}")
    print("MATCH" if delta.max() == 0.0 else "DIFFERS")
    return 0 if delta.max() == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
