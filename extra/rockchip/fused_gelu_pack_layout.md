# Resolve raw GELU projection fragments before channel loops

The current fused Gemma2 profile attributes 143.55 ms per 128-token prompt
to raw GELU packing, 20.6% of native prefill time. NPU calls account for
39.1% of native prefill and 89.0% of decode. These component measurements
exclude three full warmups and cover three requests; profiling adds overhead.
[Component profile and restored-worker checks](../../../oNPU/docs/benchmark-gemma2-fused-profile-v1.json).

The packer previously located each four-channel gate/up fragment through
conditional lookups inside the channel loop. The new generator partitions
rows and channels at raw-tile and core-fragment boundaries, then emits direct
loads for each contiguous interval. It retains FP32 multiplication order,
the GELU approximation, independent down-projection-group maxima and scales,
row-tiled headers, padding and nonfinite-row handling.

`ONPU_GELU_PACK_STATIC_LAYOUT=1` selects this layout and is now the default
inside the optional fused GELU FFN. Value 0 retains the previous implementation.
The full fused GELU FFN remains optional and limited to its qualified dense
prefill geometry. This change does not expand placement or precision coverage.

Complete CPU pack measurements use three geometries, five warmups and forty
alternating ABBA/BAAB blocks, eighty calls per variant. Timings include TinyJit
dispatch, worker creation and synchronization. The harness clears cached
environment options before capture and verifies the actual generated sources
differ and contain the expected direct-load or conditional-lookup code.

| Rows | Hidden width | Active rows | Paired pack speedup | Faster blocks |
|---:|---:|---:|---:|---:|
| 128 | 9,216 | 128 | 1.068x | 40/40 |
| 65 | 9,216 | 17 | 1.026x | 26/40 |
| 128 | 6,400 | 113 | 1.051x | 36/40 |

Payload bytes and FP32 scale bits match exactly. Independent multi-stage
references also cover fragmented projections, partial rows, dynamic active
rows, poisoned padding and nonfinite values. Exclusive NPU tests compare
complete FFNs at 17 and 128 rows through seven changed inputs and native replay.

Full Gemma2-2B measurements use 128 inputs, 64 outputs, context 512, three
full warmups and three measured requests per variant, with identical options
except the layout switch. Both graphs execute 26 fused pack kernels in prefill,
none in decode; generated source inspection verifies the selected variant.

| Metric | Existing layout | Static layout |
|---|---:|---:|
| TTFT ms | 753.26 | 713.58 |
| Prefill proxy tok/s | 169.93 | 179.38 |
| Decode tok/s | 9.86 | 9.93 |

All twelve 64-token sequences match the previous candidate. Clocks vary, so
the 5.6% higher prefill rate in this pair is not a fixed-clock attribution to
this kernel alone. Earlier 183.86 tok/s samples used a different process/clock
state and must not be multiplied by the new ratio. The published RKLLM
213.90/10.37 tok/s reference remains faster; comparable-quality and serving API
qualification remain open. Both original sessions were restored and answered
fresh requests. [Raw samples, source identities, checks and restoration](../../../oNPU/docs/benchmark-gemma2-gelu-static-verification-v1.json).
