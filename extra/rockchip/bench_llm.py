#!/usr/bin/env python3
"""Baseline prefill/decode timing for one GGUF on the native tinygrad path.

Protocol: 128-token prompt, 64 greedy decode steps, 1 warm-up pass, 3 measured
passes. Reports every trial plus the median, so a warm-up artifact stays visible.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, sys, time

PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--npu", action="store_true", help="replace nn.Linear with the generic NPU GEMM")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from tinygrad import Device, Tensor
    from tinygrad.llm.gguf import gguf_parse
    from tinygrad.llm.model import Transformer
    from tinygrad.llm.tokenizer import SimpleTokenizer

    kv, _ = gguf_parse(str(args.ckpt))
    try:
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)
        ids = tokenizer.encode(PROMPT)[:args.prompt]
        prompt_source = "tokenizer"
    except Exception:
        vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else kv[f"{kv['general.architecture']}.vocab_size"]
        ids = [1] + [(i * 37 + 11) % (vocab - 1) + 1 for i in range(args.prompt - 1)]
        prompt_source = "synthetic"
    if len(ids) < args.prompt:
        ids = (ids * (args.prompt // max(1, len(ids)) + 1))[:args.prompt]
    max_context = args.prompt + args.gen + 16

    started = time.time()
    model, _ = Transformer.from_gguf(str(args.ckpt), max_context=max_context)
    from tinygrad.nn.state import get_parameters
    for parameter in get_parameters(model):
        parameter.realize()
    npu_linears = 0
    if args.npu:
        from npu_linear import prepare_linears, replace_linears
        npu_linears = replace_linears(model)
        prepare_linears(model)
    load_seconds = time.time() - started

    temperature = Tensor([0.0])
    prompt_tensor = Tensor(ids, dtype="int32").reshape(1, args.prompt)
    from tinygrad.uop.ops import UOp
    v_start = UOp.variable("start_pos", 0, max_context - 1)

    def trial() -> tuple[float, float]:
        begin = time.perf_counter()
        out = model(prompt_tensor, (v_start.bind(0),), temperature).realize()
        ttft_ms = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        for step in range(args.gen):
            out = model(out, (v_start.bind(args.prompt + step),), temperature).realize()
        decode_seconds = time.perf_counter() - begin
        return ttft_ms, args.gen / decode_seconds

    for _ in range(args.warmups):
        trial()
    trials = [trial() for _ in range(args.runs)]
    report = {
        "kind": "npu-baseline-timing" if args.npu else "native-baseline-timing",
        "name": args.name,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256(args.ckpt),
        "arch": kv.get("general.architecture"),
        "device": Device.DEFAULT,
        "npu": args.npu,
        "npu_linears": npu_linears,
        "prompt_tokens": args.prompt,
        "prompt_source": prompt_source,
        "generated_tokens": args.gen,
        "warmups": args.warmups,
        "runs": args.runs,
        "load_seconds": round(load_seconds, 2),
        "ttft_ms": [round(t, 2) for t, _ in trials],
        "decode_tokens_per_second": [round(d, 4) for _, d in trials],
        "median_ttft_ms": round(statistics.median(t for t, _ in trials), 2),
        "median_decode_tokens_per_second": round(statistics.median(d for _, d in trials), 4),
        "total_seconds": round(time.time() - started, 1),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("name", "median_ttft_ms", "median_decode_tokens_per_second")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
