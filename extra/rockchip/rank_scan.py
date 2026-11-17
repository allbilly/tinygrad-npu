#!/usr/bin/env python3
"""Find the op that explodes the graph rank during a prefill.

A DEBUG=2 profile of the Qwen3.5 4B prefill shows one reshape lowered to a
139-dimensional kernel: 128 of its dims are size 128 and T_pad is 128, so
something accumulates a dim per token instead of collapsing into the batch axis.
That kernel is 52.7% of all kernel time.

This walks the *lazy* graph and reports every UOp whose rank exceeds the model's
natural maximum (4 for this architecture), so the offending op is named without
running anything. Uses T=8 by default: the explosion scales with T, so the same
bug shows up as 8 dims of 8 and costs far less to construct.

Usage: DEV=CPU .venv/bin/python extra/rockchip/rank_scan.py [--tokens N] [--max-rank N]
"""
from __future__ import annotations

import argparse
import sys

from tinygrad import Tensor
from tinygrad.llm.model import Transformer
from tinygrad.uop.ops import UOp

CHECKPOINT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--max-rank", type=int, default=4, help="natural rank; anything above is suspect")
    ap.add_argument("--show", type=int, default=6, help="how many distinct ranks to print")
    args = ap.parse_args()

    context = args.tokens + 8
    model, _ = Transformer.from_gguf(CHECKPOINT, max_context=context)
    start = UOp.variable("start_pos", 0, context - 1)
    out = model(Tensor([[1] * args.tokens], dtype="int32"), (start.bind(0),), Tensor([0.0]))
    print(f"built graph for {args.tokens} tokens", flush=True)

    seen: dict[int, list[UOp]] = {}
    for u in out.uop.backward_slice:
        shape = u.shape
        if shape is not None and len(shape) > args.max_rank:
            seen.setdefault(len(shape), []).append(u)

    total = sum(len(v) for v in seen.values())
    print(f"ops with ndim > {args.max_rank}: {total}", flush=True)
    if not total:
        print("no rank explosion at this token count")
        return 0

    for rank in sorted(seen, reverse=True)[:args.show]:
        u = seen[rank][0]
        shape = tuple(u.shape)
        print(f"\nndim={rank} op={u.op} count={len(seen[rank])}", flush=True)
        print(f"  shape={shape[:6]}...{shape[-3:]}", flush=True)
        for s in u.src[:4]:
            sshape = tuple(s.shape) if s.shape else None
            print(f"  src op={s.op} ndim={len(sshape) if sshape else 0} "
                  f"shape={sshape[:6] if sshape else None}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
