# RK3588 NPU — index

Code is in `extra/rockchip/` (NPU Linear, CPU kernels) and `tinygrad/runtime/ops_rockchip.py`
(the device). This directory is the written record.

## Current state

| | |
| --- | --- |
| prefill, Qwen3.5-4B, 3 warm-ups + 3 measured | **1,198.87 ms** median (replicated at 1,193.30) |
| prefill proxy, `128000 / ttft` | 106.79 tok/s |
| decode | 4.57-5.07 tok/s (varies run to run) |
| parity | pre-sampling logits `895365e27ed4255b`, identical to the stock tree |
| clocks | NPU 1 GHz, DDR 2112 MHz, CPU6 2256000, pinned; trials 71-79 °C |

Against the reference: oNPU documented **1,263.09 ms** (101.34 tok/s), archived `ref-profile` last
trial **1,210.6 ms**, RKLLM **2,068.88 ms** (61.87 tok/s). Decode is the one row still behind: 6.48
tok/s reference.

## Baseline definition

The baseline is **the CPU host with every `nn.Linear` offloaded to the NPU** (`bench_llm.py --npu`).
Attention, norms, softmax, rope, embedding, sampling and the KV cache stay on the CPU.
`DEV=ROCKCHIP` is the NPU device itself; `DEV=CPU` without `--npu` is a reference only.

## Reproduce

```bash
# device health: must print max_abs_err=0.0000 rel=0.00000
PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/smoke_matmul.py

# parity gate: must print 895365e27ed4255b
PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/logits_gate.py

# 3+3 prefill: reproduces the 1,198.87 ms row
PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/bench_replay_c.py \
  --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  --name pq-3x3 --out <dir> --warmups 3 --runs 3 --npu
```

`bench_replay_c.py` asserts its own configuration (thread_split installed, `rkpool` registered,
replay installed, `--npu` present). A run missing any of them is a different configuration, not a
slower machine: without `--npu` it reports `0 submits`, and without `thread_split` it reports
1,858 ms instead of 1,199 ms.

The NPU is single-owner: `/tmp/rk3588_npu_submit.lock`, `/dev/dri/card1`. One process at a time.

## Canonical documents

| file | contents |
| --- | --- |
| `results/prefill-parallel-quantise.md` | **the current result**: both 3+3 runs, temperatures, the comparison table, the dequantise verdict |
| `refuted-levers.md` | one line per landed/refuted/unsafe lever, and the numbers that were wrong with what replaced them |
| `reference-qwen35-4b.md` | the reference: identity, artifacts, verbatim command, per-kernel split, port status |
| `reference-run.md` | how to run the reference today, and the four driver defects that made it lie |
| `results/int8-name-reconciliation.md` | §1-§3 only: the 457.4793 GMAC name-level reconciliation |
| `results/decode-sublimits.md` | why decode is submit-count-bound, and the traps |
| `model-support.md` | the 21-row model selection with `native` PASS/FAIL and root causes |
| `results/repro-paths-{a,b,c}.md` | archive provenance: exact argv, tree identity, per-probe gates |

## Historical — read for method, not for numbers

These predate the INT8/replay work; every "ours" figure in them is from the 2,532 ms or 1,402 ms
tree. They are kept for the measurement method and the reference-side data.

| file | why it is still here |
| --- | --- |
| `npu-gemm-baseline.md` | the 9-model generic-GEMM baseline, and the device findings that still stand |
| `reference-qwen35-4b.md` §"ours" column | superseded by `prefill-parallel-quantise.md` |
| `results/decode-rea.md`, `decode-port-list.md` | the reference's 39-row decode profile and the mechanism inventory |
| `results/threadsplit/report.md`, `loop-classification.md`, `thread-split-audit.md`, `splitcheck/README.md` | how the four-way transforms were validated, and the pool's four rules |
| `results/ref-tinygrad-extract.md`, `ref-onpu-extract.md`, `ref-kernels/compare.md` | archive internals: GDN scan, NativeReplay, the reference's per-kernel table |
| `results/load-cost-plan.md` | the load-time decomposition (prepare 62.27 s of 70.9 s; `quantize_rows` 58.38 s) |
| `results/loader-call-analysis.md` | `R_AARCH64_ABS64` unhandled — do not bake addresses |

Result JSONs and raw traces are **not tracked** (`.gitignore:73`); the numbers live in the documents.
