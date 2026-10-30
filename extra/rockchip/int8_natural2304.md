# INT8 depth 2304 without zero padding

The mixed INT8 path previously rounded 2,304 input channels up to 2,560.
Those added channels contain zeros in both activations and weights. The new
natural-depth recipe removes that padding. Per-row activation maxima,
per-channel weight maxima, quantized codes, integer products and FP32 scale
multiplication retain their previous values.

Depth 2304 is now enabled by default. `ONPU_INT8_NATURAL2304=0` retains the
old padded planner for comparison. The measured candidate explicitly selected
value 1; the later default promotion selects the same tested branch and does
not change numerical kernels. The proof records both source hashes and the
post-promotion CPU/static checks.

Seven projection shapes covered one-row outputs of 4,096, 12,288, 18,432 and
24,576 columns and 64-row outputs of 4,096, 6,144 and 12,288 columns. Natural
depth gave 1.078–1.102× complete decode-projection speedups and 1.027–1.067×
prefill-projection speedups. Timings include quantization, transfers, matmul,
FP32 dequantization and native dispatch, with 24 balanced blocks and 48 calls
per variant.

The shared experiment also tested narrow-output K=4096 bank allocations.
All 72 independent integer checks, output tail guards and 168 FP32 projection
checks passed. The extra bank variants showed little additional benefit and
were not enabled by this change. A separate hardware regression passed 21
changed-input/native-replay comparisons, including the actual 10,240-column
vocabulary tail, partial 17-row input and a 9,216-column prefill output.

Full Gemma2-2B runs use 128 input tokens, 64 output tokens, context 512, three
full warmups and three measured requests per variant. Both variants use the
same capped CPU attention, vector GELU and final-logit softcap options.

| Metric | Padded control | Natural depth |
|---|---:|---:|
| Prefill proxy tok/s | 136.94 | 140.29 |
| Decode tok/s | 9.30 | 9.79 |
| Packed weight bytes | 2,829,582,336 | 2,614,099,968 |
| Peak RSS MiB | 6,549.62 | 6,298.61 |

All twelve 64-token sequences match each other and the prior host-kernel
candidate. Packed weights save 205.5 MiB. CPU/DMC/NPU clocks vary; the
recorded RKLLM 213.90/10.37 tok/s reference remains faster. Broader output
quality, fixed-clock RKLLM comparison and all-options API qualification remain
open. The earlier local host-quality sample is not a new quality evaluation
of this source cohort.

Both normal workers were restored and answered fresh requests. CPU tests,
mypy and Ruff passed under enforced memory/CPU limits. [Primitive samples,
full reports, source/default promotion and restoration evidence](../../../oNPU/docs/benchmark-gemma2-natural-verification-v1.json).
