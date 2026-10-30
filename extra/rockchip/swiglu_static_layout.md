# Resolve raw SwiGLU projection fragments before channel loops

`ONPU_SWIGLU_PACK_STATIC_LAYOUT=1` selects direct CPU loads in the fused INT8
SwiGLU packer. The default remains 0; local full-model results are below.
This does not change which projections run on the NPU.

The three-core projection returns contiguous fragments with independent row
strides. The generator splits the hidden-channel loop wherever either the
gate or up projection crosses a fragment boundary, then emits its exact
base and row stride. FP32 dequantization order, sigmoid arithmetic, per-row
maxima and scales, nearest-even INT8 quantization, nonfinite handling, tiled
headers and padding remain shared with the previous implementation.

The CPU-only prototype compares the complete generated C function, including
worker creation and joins. Nine geometries cover all eight supported hidden
widths, one-row operation, partial tiles and non-divisor tile sizes. Six input
patterns per geometry include zeros, extreme INT32 values, fragment boundary
impulses and nonfinite scales/weights. All 54 comparisons preserve payload
bytes and scale bits exactly. Each geometry uses five warmups and forty
counterbalanced ABBA/BAAB blocks, eighty samples per variant.

| Rows | Hidden width | Output tile rows | Median paired speedup |
|---:|---:|---:|---:|
| 1 | 512 | 1 | 1.000x |
| 17 | 512 | 17 | 1.036x |
| 65 | 1024 | 64 | 1.014x |
| 128 | 1536 | 65 | 1.026x |
| 64 | 2048 | 17 | 1.036x |
| 128 | 2560 | 64 | 1.030x |
| 128 | 3072 | 64 | 1.056x |
| 17 | 3584 | 128 | 1.012x |
| 128 | 4096 | 64 | 1.040x |

These are CPU operator measurements under adaptive clocks, not LLM or RKLLM
speedups. The repository tests additionally compare both implementations
through normal tensor execution and check zero header padding and invalid
row scales. CPU verification passed 80 tests and 74 subtests, with nine
explicit hardware tests skipped; mypy checked 280 source files and Ruff passed.
An initial 700 MiB test job exhausted its cap during pytest worker startup;
the successful retry used a 2 GiB cap after unloading only the GPU serving
worker. Both original models answered fresh requests after restoration.

The matched Qwen3 comparison completed four processes in ABBA order, each
with three warmups and three measured requests at 128 inputs, 64 outputs,
context 512. Only the layout option changes; native replay and packed NPU
weights stay identical.

| Aggregate role (six measured requests) | TTFT ms | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|---:|
| Existing layout | 300.61 | 425.81 | 37.7555 |
| Direct fragment loads | 285.04 | 449.06 | 37.7783 |

All 24 generated sequences and all four sets of 32 full-vocabulary teacher
vectors match bit for bit. Captured C confirms the selected implementation
in all 28 pack calls per phase. Adaptive clocks limit causal attribution;
these measurements do not improve on the older preferred Qwen3 prefill row
or establish comparable RKLLM quality. The original owner hit a stdout
BrokenPipeError while reporting restoration. Its failed job is preserved,
its final exit status is unknown, and a separate successful recovery verified
both original sessions, options, selection, settings and fresh API replies.
[Raw results and recovery evidence](../../../oNPU/docs/benchmark-qwen3-swiglu-static-verification-v1.json).
