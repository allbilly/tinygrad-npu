# Grouped CPU expert projections

`configure(..., cpu_experts=True)` and `--cpu-experts` select an experimental
CPU path for routed expert projections. It groups active token/route slots by
expert, reuses packed FP16 weights across up to four rows, and computes sixteen
independent output channels together. Each output retains sequential FP32
feature accumulation. Stable scattering restores the original token/top-k order;
routing probabilities, activations, shared experts and the final sum stay in the
model implementation.

The helper supports CPU FP32 batch-one activations, CPU int32 routing IDs, up to
128 token rows and static FP16 expert tables. Channels are divisible by sixteen,
feature widths by four, with widths up to 4096 and up to 128 experts/eight routes.
Runtime IDs must be valid expert indices; malformed IDs produce non-finite
outputs without dereferencing expert storage. Normal model routers select valid
indices. Unsupported geometry/dtypes retain the original Tensor path.

To preserve compiler fusion, lazy arithmetic inputs keep the original expression.
The fast path consumes explicit storage/staging boundaries, including the shared
unscaled FP32 input of an active INT8 NPU router. FP16 router packing and calibrated
input scaling are different boundaries and retain the original Tensor expression.
The original weight tensors, layouts, identities and state-dict keys remain intact.
A separate packed copy lives in a closure outside recursive state walks; preparation
memory must therefore be measured. The helper is configured after loading and
before JIT captures, with immutable weights during serving.

An isolated prototype passes 17 tests against the captured DeepSeekOCR gate/up
and down C implementations at E64, N896/K1280 and N1280/K896, with changed routing,
1/17/128 rows, serial and four-worker execution, mutation checks and invalid-input
handling. Prepared-kernel measurements show these paired medians:

| Projection | Rows | Captured C ms | Grouped serial ms | Grouped four-worker ms |
|---|---:|---:|---:|---:|
| Gate/up | 1 | 6.765 | 1.772 | 1.861 |
| Gate/up | 128 | 886.632 | 104.406 | 31.437 |
| Down | 1 | 9.157 | 2.072 | 1.997 |
| Down | 128 | 1251.466 | 104.694 | 33.052 |

All six paired blocks improve over the captured implementation. These timings
exclude weight packing and selector materialization and do not establish model
or RKLLM throughput. The four-worker measurements include pool entry/exit and
parking, and clocks were adaptive.

Integrated CPU checks pass 46 tests with one explicit hardware skip, mypy on
274 files and Ruff. Generic and materialized-normalization complete-model logits
match bit for bit across prefix, cache decode and resets. A small actual-NPU
qualification also passes with unchanged dense INT8 layers and native replay;
all checked logits match, with three grouped expert programs in both phases.
Earlier fixture, storage-op and fusion-boundary failures remain in the run logs;
no numerical tolerance was relaxed to accept the integration.

The trained DeepSeekOCR pair now passes: all 32 fixed-teacher full-vocabulary
vectors match bit for bit, with unchanged generated IDs. It has one leading dense
block and eleven MoE blocks, with 33 Q8_0 source expert matrices. Both arms use
identical dense NPU options, 128 input/64 output tokens, context 512, three warmups
and three measurements. The candidate executes all 33 expert programs per phase;
dense NPU packed weights remain 505,446,400 bytes in each arm.

| Variant | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|
| Deferred original-expert control | 1.56 | 1.47 |
| Grouped expert candidate | 95.11 | 23.29 |
| Earlier realized-weight candidate | 3.43 | 3.15 |
| Published RKLLM reference | 182.54 | 31.60 |

Both new arms defer weight realization to fit the packed-weight experiment. Saved
control sources decode raw Q8 blocks within expert loops; the candidate prepares
FP16 tables once. Its gains therefore include preparation and grouped execution.
The matched control is distinct from the earlier realized-weight candidate.
Candidate lifetime peak RSS is 8,856 MiB, including the separate CPU packed copy.
It still trails RKLLM in both phases. These are text-decoder measurements; the
vision encoder and connector are excluded. The repeated-prompt regression does
not establish held-out, FP16 or comparable RKLLM quality. Both original serving
models were restored and answered fresh requests. The original runner's tool
handle expired; its exit is recorded as unknown, with child exits and restored
service independently verified.
[Reports, source hashes and exact-logit comparison](../../../oNPU/docs/benchmark-deepseek-cpu-experts-verification-v1.json).

A subsequent diagnostic uses identical placement, three warmups and one profiled
128/64 request. All four generated sequences match the candidate above. Grouped
experts still account for 81.17% of recorded prefill kernel time and 50.42% of
decode time. NPU execution alone accounts for 4.46% and 39.24%, respectively;
copies are separate. These percentages describe the grouped path, unlike the
earlier generic-expert profile. They include instrumentation overhead and do not
replace canonical throughput. Both serving models answered fresh requests after
restoration. The next experiment varies CPU tile dimensions while retaining the
same packed weights and sequential feature accumulation.
[Current component profile](../../../oNPU/docs/benchmark-deepseek-cpu-experts-profile-verification-v1.json).

An isolated tile comparison retains the same N16 packed storage and tests
four-row/sixteen-output and eight-row/eight-output tiles, with and without a
separate branch for full row tiles. The eight-row/eight-output full-tile variant
is 1.089x faster for gate/up and 1.085x for down in eight interleaved paired
blocks. All twelve shape/length/routing cases match current output bits, including
1/17/128 rows and repeated expert IDs. Timing uses 128 rows, prepared inputs and
an active four-worker pool; selector creation and weight packing are excluded.
The subsequent integration passed 49 CPU tests with one hardware skip, mypy,
Ruff, actual NPU model checks and exact equality of 32 trained full-vocabulary
vectors. Its complete-model timing nevertheless lost: 96.91/23.83 prefill/decode
tok/s versus 98.84/24.71 for the matched four-row control. The runtime option
and its temporary tests were removed, restoring all four affected files byte
for byte to their previously qualified versions. The isolated gain did not
transfer to this trained workload. The preferred 95.11/23.29 row remains.
[Paired samples, exact checks and generated sources](cpu_expert_tile_results.json).
[Trained experiment and rejection](../../../oNPU/docs/benchmark-deepseek-expert-tile-verification-v1.json).

## Six-row prefill tiles

`cpu_expert_prefill_rows=6` and `--cpu-expert-prefill-rows 6` retain the N16
packed layout, compute sixteen channels across six rows, and use a separate
full-tile loop without row-count branches inside the feature loop. Tails keep
bounds checks. Each output still accumulates features sequentially in FP32.
Capacity-one decode uses the original four-row kernel. The default stays four.

Prepared four-worker comparisons at the actual gate/up and down geometries
measure paired speedups of 1.127–1.195 across balanced and captured routing,
with twelve bitwise output cases.
[Operator records and sources](../../../oNPU/docs/benchmark-cpu-expert-phase-tiles-operator-v1.json).
Wider single-token tiles were neutral or
slower and were removed. CPU qualification passes 100 tests and 89 subtests,
mypy on 278 files and Ruff. Two actual-NPU synthetic model modes preserve all
tested prefix/decode/reset output bits; the vocabulary integer checks also pass.

The full trained pair preserves all 32 teacher vectors and all twelve generated
sequences exactly. Four/six rows measure 106.49/112.88 prefill proxy tok/s and
25.84/25.21 decode tok/s. The prefill improvement is about 6%; the decode median
is 2.4% lower despite the unchanged decode kernel. Therefore the candidate does
not replace the preferred timing variant and remains opt-in. Both remain below
the published RKLLM 182.54/31.60 targets. No full VLM or matched RKLLM quality
claim follows from this local exact-bit regression.
[Measured pair, source checks and restoration](../../../oNPU/docs/benchmark-deepseek-cpu-prefill-r6-verification-v1.json).
