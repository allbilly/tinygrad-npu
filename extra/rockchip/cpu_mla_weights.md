# Prepared CPU MLA weights

`configure(..., cpu_mla_weights=True)` and the benchmark's
`--cpu-mla-weights` prepare contiguous FP16 per-head key/value projection tables
before JIT tracing. The default is off. CPU-only benchmarks with both NPU phases
off also perform the preparation and report `cpu_mla_weights_prepared`.

MiniCPM3's generic query and value projection kernels previously decoded packed
Q8 weights repeatedly inside their dot products. Preparation materializes the
same rounded FP16 values once. It preserves state-dict keys and tensor identities
without adding hidden derived tensors to model attributes. Disabling the option
skips new preparation; it does not reconstruct lazy packed-weight expressions.
Recreate captures after changing weights or preparing an already captured model.

Only CPU FP16 MLA tables with matching model geometry are accepted. All dimensions
are bounded to 4096 and total logical table storage to 256 MiB; unsupported models
retain their existing path. The measured MiniCPM3 has 124 tables totaling 155 MiB.

| MiniCPM3 4B, 128 input / 64 output / context 512 | Control | Prepared |
|---|---:|---:|
| Prefill proxy tok/s | 3.94 | 5.63 |
| Decode tok/s | 2.19 | 2.89 |
| Captured Q8-reading projection programs per phase | 124 | 0 |

Both variants use three full warmups and three measured requests. All twelve
64-token sequences match each other and the previous reference. NPU packed weights
are identical. Independent CPU tests preserve decoded weight bits, changed-input
projection outputs and small-model prefill/cache/reset/generation behavior. Captured
sources demonstrate that the prepared projection no longer receives packed Q8 bytes.

The original validator used a stale NPU byte count and stopped before the candidate.
The raw control and failure were retained; a corrected audit uses the current
profile's byte count. The candidate then ran under the same numerical source cohort
in a separate lease. Adaptive clocks and this extra lease are recorded. A later
CPU-only CLI fix adds preparation for off/off and a reporting field; the timed NPU
path and numerical kernels are unchanged.

Prefill remains below the published RKLLM 90.88 tok/s and decode below 5.94.
Full-model logits, held-out perplexity and comparable RKLLM quality remain open.
Both original serving models were restored and answered fresh requests.
[Raw results, CPU checks, correction and restoration](../../../oNPU/docs/benchmark-minicpm3-mla-weights-verification-v2.json).
