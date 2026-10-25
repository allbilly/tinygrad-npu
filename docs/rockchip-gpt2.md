# GPT-2 acceleration for oNPU

Useful GPT-2 inference code from `tinygrad-1500` and `tinygrad-1500-clean` is now
ported into this checkout's `oNPU` branch. It runs without either reference
checkout. The archive copies created during the audit have been removed at the
user's request. Neither original folder was deleted. Changes are in the working
tree; they have not been committed or pushed.

This explicit CPU/NPU adapter uses FP16 matrix inputs, FP32 NPU accumulation,
and CPU attention, normalization, activation and sampling. OpenCL remains the
app default. GPT-2 acceleration is opt-in with `--rockchip`; this task does not
add GPT-2 model loading to the UI. Existing Qwen/SmolLM app integration is covered
in [rockchip-onpu.md](rockchip-onpu.md).

## Retained code

| Reference improvement | Ported implementation | Scope / limits |
| --- | --- | --- |
| Prepared commands, persistent packed weights, streaming, three cores | `ops_rockchip.py`, `rockchip.py` | Existing selective backend; real matrix tasks on all three cores. |
| Wider vocabulary/output tiles | `configure_gpt2`, `GPT2Head` | 24576-column tiles; only the final hidden row is projected. |
| Combined QKV and gate/up | Existing optional fusion | GPT-2 already combines QKV and has no gated FFN. |
| Sharing packed copies | GPT-2 wrappers reuse original packed objects | General combined/separate projection deduplication remains outside this GPT-2 port. |
| Packed KV and fused attention | `rockchip_gpt2.py` | Head-major cache, causal bounds, cached exponents, current compiler's FP16 rounding. |
| Old key swizzles / paired-head reductions | Not selected | The retained head-major per-head implementation is qualified; the complete old specialized attention pipeline was not copied. |
| Parallel CPU attention | `rockchip_pool.py`, `rockchip_pool.c` | CPU5–7 workers plus caller on CPU4 when allowed; decode length >=16; workers park between calls. Prefill distributes heads for >=16 rows. |
| Native CPU-to-NPU cleaning | `rockchip_cache.py`, `rockchip_cache.c`, `rockchip_replay.py` | Child-probed ARM `dc cvac`, exact cacheable/aligned input ranges <=1 MiB; driver fallback. Every FROM_DEVICE invalidation uses the driver. |
| Combined replay and generation loop | `rockchip_replay.py`, `rockchip_gpt2_decode.py` | Prepared CPU/NPU graph plus true greedy generation, bounded to32 tokens per native call; retains buffer/weight guards and execution counts. |
| Fast greedy sampling | `greedy_argmax` | One-pass SIMD first-maximum reduction, including ties and tail. |
| CPU wake helper | `rockchip_idle.py`, `rockchip_idle.c` | Nice19 CPU0 thread only around warmed native decode; parked otherwise. |
| Shape-specific registers | `ops_rockchip.py` | GPT-2 decode feature-grain settings; complete register initialization on every submission. |
| Write-combining allocation | Not selected | Current cacheable allocations plus bounded direct cleaning supply the retained synchronization optimization. |
| Old parallel GELU and generic renderer | Not selected | No demonstrated gain needed for this explicit path; no runtime dependency on old implementations. |

The adapter bounds prefill JIT variants to8, materializes CPU activation boundaries
before NPU calls, and replaces stale CPU decode capture when switching a warmed
model. The normal model path remains available without `--rockchip`.

## Measured performance

GPT-2 124M, context1024, 32 greedy output tokens, input lengths8/32/128. Prefill
includes producing the first output token; decode times the remaining31. Loading
and compilation are excluded. Four fresh processes run in ABBA order, each with
five complete warmups and five measurements per length. Control retains native
generation and SIMD sampling, with driver input synchronization and serial decode
attention. Candidate enables the ported cache cleaner and attention pool.

| Input tokens | CPU diagnostic prefill tok/s | CPU diagnostic decode tok/s | Ported prefill tok/s | Ported decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5.04 | 5.03 | 492.19 | 97.01 |
| 32 | 10.23 | 5.00 | 955.66 | 96.05 |
| 128 | 13.23 | 4.84 | 1006.76 | 96.42 |

The CPU diagnostic uses two warmups/two measurements and is not a paired CPU
speedup study. The final paired candidate improves decode over its paired
control by4.0%,2.8%,10.0%; prefill stays similar. Every one of3840 generated IDs
matches the CPU reference across all120 complete requests. Trials and hashes:
`/tmp/onpu-gpt2-20261018/final-paired/comparison.json`. An earlier independent
ABBA run also matches all3840 IDs, measuring492.77/948.99/1008.15 prefill tok/s
and98.12/100.26/95.00 decode tok/s. Its trials remain at
`/tmp/onpu-gpt2-20261018/paired-ported/comparison.json`.

All three cores have identical interrupt increments:200640 each over the final
paired experiment. Debugfs one-second busy-time windows average52.9%,54.1%,55.1%
and peak at64%,66%,67% in measured candidate decode intervals. These are driver busy-time percentages, not MAC
occupancy. A separate idle check records0% and zero interrupts. Devfreq `load` is
excluded: this driver's busy/total counters are fixed at1. Sampled clocks are
NPU1GHz, CPU4–7 at1.8GHz and DDR2.112GHz. No global clock, IRQ affinity, module or
driver settings were changed.

RKLLM1.3.1's installed importer returns-1 with `Not support GPT2LMHeadModel!`.
No RKLLM GPT-2 timing exists, so a speed advantage cannot be claimed. Probe:
`/tmp/onpu-gpt2-rkllm-probe.log`. This agrees with the
[versioned SDK model list](https://github.com/airockchip/rknn-llm/blob/release-v1.3.1/README.md#support-models).
The older specialized launcher recorded about103–106 decode tok/s; the current
port does not claim to exceed that historical result.

## Correctness and limitations

- Eight adapter tests and18 surrounding hardware/CPU/replay tests pass with
  `-n12`, covering complete outputs, rebinding, reset, native generation beyond32
  tokens, helper parking, affinity restoration, masking, inactive-cache NaNs and
  FP16 normalization rounding.
- The normal GPT-2 real-world test passes. Mypy checks224 source files; Ruff and
  Git whitespace checks pass.
- An independent64-token run matches fresh CPU output while an audit hook rejects
  reference-checkout/archive access, including canonical paths. It records59
  native-loop tokens,3264 submissions,9792 tasks and zero forbidden accesses:
  `/tmp/onpu-gpt2-20261018/final-independent-port.json`.
- Five32-token continuations match all160 CPU IDs. Twenty complete50257-value
  probability comparisons pass (maximum KL2.19e-5). Independent comparisons of
  1566720 complete matrix-output values also pass.
- **Strict raw-logit parity fails:**14 comparisons exceed unchanged max-absolute
  .1 and/or RMSE .025 gates, largely through a shared score offset. Centered
  scores and probabilities are closer, but the raw gates still fail.
  `verify_gpt2.py` exits1 and records `complete:false`; no threshold was raised
  and no score normalization was added. Evidence:
  `/tmp/onpu-gpt2-20261018/quality-rounding-v4.json`.

Only124M batch-one greedy inference has the full performance/token qualification
above. Sampled/batched requests retain the regular generation loop; larger GPT-2
variants and other contexts need separate qualification. The native loop uses
each actual greedy result as the next input. This is not pure-NPU execution.

Compact command buffers and a blocking-submit helper were tried and removed:
the measured gains did not justify added state/lifetime rules. The retained
runtime writes full register state and submits directly.

## Reproduction

Use one NPU owner at a time. Unload an idle NPU app worker before standalone
hardware runs and restore it afterward. A GPU-only worker can remain loaded.

```sh
DEV=CPU HALF=1 MAX_CONTEXT=128 .venv/bin/python examples/gpt2.py \
  --model_size gpt2 --rockchip --temperature 0 --count 64 \
  --prompt 'The quick brown fox jumps over the lazy'

.venv/bin/python -m extra.rockchip.check_gpt2_port --report /tmp/gpt2-independent.json

DEV=CPU ROCKCHIP_TEST=1 .venv/bin/python -m pytest test/runtime/test_rockchip_gpt2.py -x -q -n12
.venv/bin/python -m mypy tinygrad/
.venv/bin/python -m ruff check .
```

`extra/rockchip/monitor_npu.py` samples real per-core debugfs load and interrupts.
Its optional local converter-image reader mounts NPU debugfs read-only and uses
no network.

## Work record

- [x] Audit both reference checkouts for useful inference code.
- [x] Port and exercise selected code without reference imports.
- [x] Measure throughput and actual three-core activity.
- [x] Check installed RKLLM GPT-2 support; record unsupported import.
- [x] Run focused and surrounding regressions, mypy and Ruff.
- [x] Remove task-created archive copies, keeping the working port.
- [x] Finish final repeated performance/independence checks and restore the app NPU session.

Final app state: Qwen3 GPU session `55a95dc3c1b93b72` is ready on CL and selected;
SmolLM2 session `2e5429a5e3753bf6` is ready on CPU+ROCKCHIP. NPU cold preparation
took146.6 seconds. The tested inference/performance port and usage investigation
are complete; the strict raw-logit limitation above remains explicitly unqualified.

INTENT: GPT-2 currently uses ordinary CPU execution without the oNPU adapter; the task expects faster checked GPT-2 inference and measured three-core usage; the selective-port specification permits explicit NPU matmuls with surrounding CPU operations.

TWINS: searched disk-backed weight transposes in examples and tinygrad/llm - found 1 other site: examples/llm.c/train_gpt2.py (training path, left unchanged).

TWINS: searched retained half softmax exponent-cache denominators - found 0 other sites: none.

AUTH: user said "i dont want archives, either port the code or i will remove it"
