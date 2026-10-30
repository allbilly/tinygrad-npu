# Small physical INT8 depths

The model planner can opt into physical depths 64, 128 and 256 with
`ONPU_INT8_SMALL_DEPTH=1`. The default remains off. Logical depths select the
smallest of those sizes that fits; larger depths keep their existing planner.
The existing row-major register recipe and three-core split are retained.
Only zero padding changes, preserving per-row codes/scales and INT32 arithmetic.

CPU packing checks preserve headers, scales and valid codes against padding to
512. Exclusive NPU checks cover M=1/17/128 and unequal/equal column splits; 63
raw INT32 and 63 FP32 checks pass with changed inputs and native replay.

Paired complete projection timings include quantization, transfers, NPU work,
FP32 conversion and native dispatch with resident inputs/weights. Five warmups
and forty alternating ABBA/BAAB blocks give eighty calls per variant. The
independent references and numerical readback are outside timed regions.
Actual descriptor depths and disabled profiling are checked.

| Rows | Logical K | New physical K | Padded median ms | Smaller median ms | Paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 64 | 0.2306 | 0.1219 | 1.878x | 40/40 |
| 1 | 96 | 128 | 0.3057 | 0.2119 | 1.434x | 40/40 |
| 1 | 128 | 128 | 0.2309 | 0.1391 | 1.662x | 40/40 |
| 1 | 192 | 256 | 0.3493 | 0.2858 | 1.230x | 37/40 |
| 1 | 256 | 256 | 0.3097 | 0.2529 | 1.249x | 39/40 |
| 128 | 64 | 64 | 0.6694 | 0.5991 | 1.126x | 37/40 |
| 128 | 96 | 128 | 0.6902 | 0.6377 | 1.089x | 38/40 |
| 128 | 128 | 128 | 1.3503 | 1.2133 | 1.113x | 38/40 |
| 128 | 256 | 256 | 0.7774 | 0.6832 | 1.056x | 37/40 |

Each comparison has fourteen exact FP32 reference checks and zero steady-state
NPU allocations. Clocks vary; paired block ratios reduce drift but do not
establish a model-level RKLLM win. Full-model timing and output-quality gates
remain open. Both original sessions were restored and answered fresh requests;
the GPU worker remained loaded.
[Raw samples](int8_small_depth_results.json) and
[source, numerical and restoration evidence](../../../oNPU/docs/benchmark-int8-small-depth-performance-v1.json).
