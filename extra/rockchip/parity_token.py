#!/usr/bin/env python3
"""Greedy token parity for the fused INT8 Linear, on a real model.

A 4-token continuation is run through the fused C call and through the six-launch path in
the same process, on the same checkpoint, with the same prompt. The token ids must be
identical: the fused function is meant to be bit-exact, not merely close, so a different
token means an arithmetic change and not sampling noise.

Usage: DEV=CPU .venv/bin/python extra/rockchip/parity_token.py \
           --ckpt /home/orangepi/vlm-bench/baseline-20261021/Qwen2-0.5B-Instruct-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

from tinygrad import Tensor

import npu_linear
from npu_linear import packed_matmul_int8, prepare_linears, replace_linears

PROMPT = "The history of the printing press begins with movable type in East Asia."


def _old_path(dense: Tensor, weight: Tensor, w_scale: Tensor, columns: int, device: str, m: int,
              row_offset: int = 0, k_offset: int = 0, column_offset: int = 0, cores: int = 3) -> Tensor:
    """The six-launch path behind the fused signature, so one model can be run both ways."""
    k = weight.shape[1] * 32
    tile = dense[row_offset:row_offset + m, k_offset:k_offset + k]
    return packed_matmul_int8(tile, weight, w_scale, columns, device, cores, column_offset)


def build(ckpt: pathlib.Path):
    from tinygrad.llm.gguf import gguf_parse
    from tinygrad.llm.model import Transformer
    from tinygrad.llm.tokenizer import SimpleTokenizer

    kv, _ = gguf_parse(str(ckpt))
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    model, _ = Transformer.from_gguf(str(ckpt), max_context=64)
    count = replace_linears(model)
    prepare_linears(model)
    return model, tokenizer, count


def greedy(model, ids: list[int], steps: int) -> tuple[list[int], float]:
    from tinygrad.uop.ops import UOp

    temperature = Tensor([0.0])
    start = UOp.variable("start_pos", 0, 63)
    out = Tensor(ids, dtype="int32").reshape(1, len(ids))
    begin = time.perf_counter()
    out = model(out, (start.bind(0),), temperature).realize()
    tokens = []
    for step in range(steps):
        out = model(out, (start.bind(len(ids) + step),), temperature).realize()
        tokens.append(int(out.item()))
    return tokens, steps / (time.perf_counter() - begin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--path", choices=("fused", "old", "both"), default="both")
    args = ap.parse_args()

    results: dict[str, list[int]] = {}
    order = ["fused", "old"] if args.path == "both" else [args.path]
    for path in order:
        if path == "old":
            npu_linear.packed_matmul_int8_fused = _old_path
        else:
            npu_linear.packed_matmul_int8_fused = _fused_entry
        model, tokenizer, count = build(args.ckpt)
        ids = tokenizer.encode(PROMPT)[:8]
        tokens, rate = greedy(model, ids, args.tokens)
        results[path] = tokens
        print(f"{path:<6} {count} linears  prompt {ids}  ->  {tokens}  "
              f"({rate:.2f} tok/s over {args.tokens} steps)", flush=True)
        del model
        gc.collect()

    if len(results) == 2:
        same = results["fused"] == results["old"]
        print("tokens match" if same else f"MISMATCH fused={results['fused']} old={results['old']}")
        return 0 if same else 1
    return 0


_fused_entry = npu_linear.packed_matmul_int8_fused

if __name__ == "__main__":
    sys.exit(main())
