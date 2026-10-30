# SDK INT8 small-depth register capture

A fresh GDB capture using the local RKNN matmul API and existing NPU dump tools
records four single-core M=1, N=2048 operations. All 8,192 output values match
independent INT32 matrix products exactly. Each task has 108 register words.

| Logical K | SDK physical K | Current model physical K | SDK CBUF |
|---:|---:|---:|---:|
| 64 | 64 | 512 | 0xb1 |
| 128 | 128 | 512 | 0xb1 |
| 256 | 256 | 512 | 0xb1 |
| 512 | 512 | 512 | 0xb1 |

Precision fields match the existing qualified INT8 recipe. Input grouping,
feature stride, group offset and output-surface fields differ from tinygrad's
row-major recipe. The SDK API selected a single core; this does not qualify
our three-core partitioning or larger row counts. Separate qualification now
covers physical K=64/128/256 with 63 exact INT32 and 63 exact FP32 checks across
M=1/17/128, including native replay. Logical rank 96 maps to physical K=128 in
the tested candidate. [Complete projection measurements](int8_small_depth.md)
record the speedups and limits. The planner remains opt-in.

The corrected RWKV7 full-model comparison measures 66.89/66.75 prefill proxy
tok/s and 12.65/13.10 decode tok/s for padded/smaller-depth variants, saving
68.125 MiB of packed weights. All twelve timing sequences match, and 192
fixed-prefix logit vectors match the existing INT8 path exactly. This local
regression check does not establish FP16 or RKLLM output quality; the published
RKLLM 201.78/13.82 tok/s reference remains faster in both phases.
[Model reports and serving restoration](../../../oNPU/docs/benchmark-rwkv7-small-depth-verification-v2.json).

The observer records submit packets before/after ioctl completion and preserves
SDK automatic core selection. A CPU-only fixture first checked the callback
and decoder paths without accessing the device. The first failed attempts are
retained in the evidence. Original NPU serving was restored and both models
answered fresh requests; the GPU worker remained loaded throughout.
[Captured register streams, numerical checks and restoration](../../../oNPU/docs/benchmark-sdk-lowk-registers-v1.json).
