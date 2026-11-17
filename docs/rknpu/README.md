# RK3588 NPU work — docs index

This directory holds the results and reference tables for the RK3588 work on
this checkout. Code lives in `extra/rockchip/`; this is the written record.

## What is here

| file                   | contents                                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npu-gemm-baseline.md` | 9 of the 12 PASS models on the generic NPU GEMM baseline: protocol, checkpoints, measured prefill/decode, RKLLM comparison, and the 3 blocked rows |
| `model-support.md`     | The 21-row download selection with `native` PASS/FAIL and each FAIL's root cause                                                                   |
| `results/`             | Raw `report.json` per model, `sweep.log`, `sweep-run.log`, and `failed.txt` for the 4B row (the other two blocked rows carry no marker)            |

## Baseline definition

The baseline is **the CPU host with every `nn.Linear` offloaded to the NPU
GEMM**. Everything else — attention, norms, softmax, rope, embedding lookup,
sampling, KV cache — stays on the CPU.

`DEV=ROCKCHIP` is the NPU device itself, and `DEV=CPU` without `--npu` is a
reference only. Neither is the baseline.

`DEV=CPU` is required: the default device on this board is `CL`, and the model
does not load there (see `npu-gemm-baseline.md`, "Why `DEV=CPU`").

## Reproducing

```bash
cd extra/rockchip

# device health, must print max_abs_err=0.0000 rel=0.00000
../../.venv/bin/python smoke_matmul.py

# one model
DEV=CPU ../../.venv/bin/python bench_llm.py \
  --ckpt <checkpoint.gguf> --name "<name>" --out <dir> --npu

# all 12 PASS models, skips any that already have a report or a failed.txt marker
../../.venv/bin/python sweep_npu.py
```

The NPU is single-owner: it takes `/tmp/rk3588_npu_submit.lock` and opens
`/dev/dri/card1`. Run one process at a time.

## Timing protocol

128-token prompt, 64 greedy decode tokens, 1 warm-up + 3 measured passes. Every
trial is reported alongside the median. Prefill is the proxy
`128000 / median_ttft_ms`, matching the definition used in the oNPU reference
`benchmark.md`, so the prefill column is comparable *in definition*.

The RKLLM columns are a different basis — whole-model INT8 at maximum clocks,
against FP16 linears-only here at unrecorded clocks — so a gap is not an
overhead-only measurement. 3 of 9 rows also fell back to a synthetic prompt
because the tokenizer failed; they are flagged in `npu-gemm-baseline.md`.

Cold-start revalidation (52–55 °C) is a separate pass applied only to rows being
published as final. **It has not been run.** The 9 measured rows started at
59.2–72.1 °C, so none are cold-start revalidated, and `report.json` records
neither clocks nor temperature.

