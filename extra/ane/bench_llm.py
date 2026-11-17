#!/usr/bin/env python3
"""CPU-host + ANE-Linear prefill/decode timing for one GGUF.

Protocol: N-token prompt, M greedy decode tokens, optional warm-up passes, then
measured passes. Every trial is printed. The ANE path replaces every eligible
``nn.Linear`` with ``ANELinear`` and disables TinyJit, because the ANE call is a
host round-trip rather than a graph kernel.

Examples:
  python extra/ane/bench_llm.py --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf
  python extra/ane/bench_llm.py --model ... --ane
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

# The baseline is a CPU host with only nn.Linear on the ANE. The default device
# on this board is CL, which does not compile the model here.
os.environ.setdefault("DEV", "CPU")

PROMPT = (
    "The history of the printing press begins with movable type in East Asia. "
    "By the fifteenth century, metal type and oil-based ink made books cheap enough "
    "for ordinary readers, and the resulting spread of literacy reshaped religion, "
    "science, and politics across Europe. "
) * 8


def prompt_ids(kv: dict, count: int) -> tuple[list[int], str]:
    from tinygrad.llm.cli import SimpleTokenizer

    try:
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)
        ids = tokenizer.encode(PROMPT)[:count]
        source = "tokenizer"
    except Exception as exc:  # noqa: BLE001 - report the fallback, do not hide it
        vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else kv[
            f"{kv['general.architecture']}.vocab_size"]
        ids = [1] + [(i * 37 + 11) % max(1, vocab - 1) + 1 for i in range(count - 1)]
        source = f"synthetic ({type(exc).__name__}: {exc})"
    if not ids:
        raise RuntimeError("tokenizer returned no tokens")
    return (ids * (count // len(ids) + 1))[:count], source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local GGUF path or a name from tinygrad.llm.cli.models")
    parser.add_argument("--ane", action="store_true", help="replace nn.Linear with ANELinear")
    parser.add_argument("--max-weight", type=int, default=None, help="skip ANE Linears above this numel")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.decode_tokens < 0 or args.runs < 1 or args.chunk_size < 1:
        parser.error("prompt-tokens >= 1, decode-tokens >= 0, runs >= 1, chunk-size >= 1")

    from tinygrad import Tensor
    from tinygrad.helpers import fetch
    from tinygrad.llm.cli import models
    from tinygrad.llm.model import Transformer

    candidate = pathlib.Path(args.model).expanduser()
    path = str(candidate) if candidate.exists() else fetch(models.get(args.model, args.model))
    started = time.perf_counter()
    model, kv = Transformer.from_gguf(path, args.max_context)
    load_seconds = time.perf_counter() - started
    ids, prompt_source = prompt_ids(kv, args.prompt_tokens)
    prompt_len = len(ids)
    if prompt_len + args.decode_tokens + 8 > model.max_context:
        raise SystemExit(f"max-context {model.max_context} is too small for {prompt_len}+{args.decode_tokens} tokens")

    linears = 0
    prepare_seconds = 0.0
    ane = None
    if args.ane:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from extra.ane import ane_linear

        ane = ane_linear.ane_device()
        started = time.perf_counter()
        linears = ane_linear.replace_linears(model, ane, args.max_weight)
        ane_linear.prepare_linears(model)
        model.greedy = True
        for block in model.blk:
            block.fast_attention = True
        prepare_seconds = time.perf_counter() - started

    # The ANE path is eager, so a symbolic prefill chunk would reach ANELinear
    # as a BIND shape. One row per call keeps every shape static.
    chunk = 1 if args.ane or getattr(model, "has_recurrent_block", False) else args.chunk_size

    def reset() -> None:
        model._cached_tokens = []
        resets = [op for block in model.blk for op in block._state_reset_ops()]
        if resets:
            Tensor.realize(*resets)

    def one() -> tuple[float, float, list[int]]:
        reset()
        gen = model.generate(list(ids), chunk_size=chunk)
        start = time.perf_counter()
        first = next(gen)
        prefill = time.perf_counter() - start
        output = [first]
        start = time.perf_counter()
        for _ in range(args.decode_tokens):
            output.append(next(gen))
        decode = time.perf_counter() - start
        return prefill, decode, output

    try:
        for _ in range(args.warmup):
            one()
        trials = [one() for _ in range(args.runs)]
        prefill = statistics.median(t[0] for t in trials)
        decode = statistics.median(t[1] for t in trials)
        report = {
            "model": str(path),
            "arch": kv["general.architecture"],
            "device": "ANE+CPU" if args.ane else "CPU",
            "ane_linears": linears,
            "ane_submissions": ane.submissions if ane is not None else 0,
            "prompt_source": prompt_source,
            "prompt_tokens": prompt_len,
            "decode_tokens": args.decode_tokens,
            "chunk_size": chunk,
            "load_seconds": load_seconds,
            "prepare_seconds": prepare_seconds,
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "prefill_tok_s": prompt_len / prefill if prefill else 0.0,
            "decode_tok_s": args.decode_tokens / decode if decode else 0.0,
            "trials": [{"prefill_seconds": p, "decode_seconds": d, "output": o} for p, d, o in trials],
        }
        print(json.dumps(report, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if args.ane:
            import ane_linear

            ane_linear.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
