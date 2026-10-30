# Batched FP32 CPU MLA projections

`--cpu-mla-projections` opts into CPU query/value projections with prepared
FP16 tables. It reuses each four-channel weight vector across four query rows
and keeps the captured CPU kernel's sequential reduction order. The first
integration uses serial dispatch; the isolated prototype's serial medians were
lower than its parked-worker medians under adaptive clocks.

Tables must be static CPU FP16 with 1–128 heads and dimensions 64–4096 in
multiples of four. The combined table budget is 256 MiB. Batch-one FP32 inputs
use bounded capacities of 1–128 rows; unsupported input layouts retain the
ordinary Tensor expression. Packing preserves the original state-dict tensors.
Derived tables live in closures to avoid implicit-argument/state-walk copies.
Explicit CPU and NPU MLA projection requests are mutually exclusive.

The isolated MiniCPM3 prototype passed 32 bitwise cases and 640 timed output
checks. Its query oracle also computes 32 RoPE channels, so the recorded ratios
are primitive measurements. Actual-compiler tests and full-model inference
must qualify this integration before changing the preferred model timings.
[Primitive evidence](../../../oNPU/docs/benchmark-mla-latent-projection-cpu-verification-v1.json).

Integrated actual-compiler checks passed 33 tests (3 skips), mypy on 281
files and repository Ruff. Coverage includes changed-input symbolic-row JIT
replay, real FP16 table layouts, exact model prefill/cached decode/reset and
bounded generation, state-dict identity, and CPU-only CLI activation.
[Integrated CPU evidence](../../../oNPU/docs/benchmark-cpu-mla-projections-cpu-verification-v2.json).

Full MiniCPM3 ABBA measurements completed using 128 input IDs, 64 output
tokens, context 512, three warmups and three measured requests per process.
All 24 generated sequences and all four sets of 32 full-vocabulary teacher
vectors match exactly. The conservative preferred process measures
15.72 prefill proxy tok/s and 4.61 decode tok/s.
This improves the previous 10.62/3.86 row. Published RKLLM 90.88/5.94 remains
faster, and comparable HF/RKLLM quality remains unqualified. Both original
serving sessions were restored and answered fresh requests.
[Full-model evidence](../../../oNPU/docs/benchmark-cpu-mla-projections-models-verification-v1.json).
