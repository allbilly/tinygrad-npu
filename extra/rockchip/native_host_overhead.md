# Native replay host overhead: scalar snapshots and guard deduplication

This note documents two small host-side costs in Rockchip `NativeReplay`. Neither change modifies the generated native command stream. Their model-level timings are limited to paired worker diagnostics; the CPU microbenchmark is synthetic and does not establish an API or whole-model decode gain.

## Scalar CPU input snapshots

`_copy_scalar_input` in `tinygrad/llm/kernels/rockchip_replay.py` handles a narrow case: a local CPU `BUFFER` containing one `int32` or `int64` scalar. It allocates an independent one-element CPU buffer and copies the exact bytes through CPU memoryviews. Remote CPU buffers, other dtypes, multiple elements, and other ineligible inputs continue through the generic `_copy_input` path.

The paired worker report observed 62 scalar-copy calls per request. The generic-copy control recorded 62 generic calls; the direct-copy arm recorded zero generic calls. Both arms submitted 11,284 NPU operations per request, produced the same 64 token IDs across all 18 requests, and ran six measured requests per arm after three warmups. Median prefill/decode was 334.53/15.966 tok/s for control and 337.20/16.144 tok/s for direct copy. This small cohort does not establish a repeatable full-request decode improvement.

A CPU-only microbenchmark of the actual scalar snapshot measured 49.9 μs median per call versus 496.4 μs for the generic copy in its harness. This isolates the copy operation; it is not a model-throughput estimate.

## Deduplicated validity guards

`NativeReplay._output_guards` deduplicates CPU-clean/version checks for outputs by metadata identity. Multiple plan outputs that refer to one allocation share one guard. `_valid_deduplicated` still checks input sizes/devices, storage identity, weight versions, attention program ownership and plan ownership. It requires `plan.bufs is buffers`, so a newly prepared binding tuple invalidates the old replay even if the tuple's buffer values compare equal.

The synthetic guard benchmark used 526 storage guards and 175 plans sharing one output allocation. Median validation time was 57.5 μs for the deduplicated path and 211.1 μs for the full-scan path. This topology is a CPU-only microbenchmark, not an LLM throughput test.

In the combined paired worker screen, both arms produced the same 64 IDs across 18 requests, with six measured rows per arm and 11,284 NPU submissions per request. Control measured 338.142 prefill / 16.061 decode tok/s; direct copy plus deduplicated guards measured 346.543 / 16.0848. The roughly 0.15% decode median difference is too small to claim a whole-decode gain. Both worker reports have throughput qualification disabled.

## Current defaults and evidence limits

`ONPU_REPLAY_SCALAR_COPY` and `ONPU_REPLAY_DEDUP_GUARDS` both currently default to `1`. The API-stage source snapshot records only `tinygrad/llm/kernels/rockchip_replay.py` as changed from the measured paired source and attributes that delta to these two default arguments. The API host-overhead follow-up has not yet completed its report and restoration receipt, so its result remains pending.

The profiled request recorded 103.725 ms cumulative in `_copy_input` across 62 calls and 67.696 ms cumulative in `NativeReplay.valid` across 64 calls. Those are instrumented, inclusive cProfile values. `NativeReplay.__call__`'s 69.105 ms self time includes the profiler's replay counters; these figures must not be read as production overhead or subtracted from overlapping cumulative totals to infer a causal bottleneck.

The paired runs preserved identical generated IDs and command counts, but they are bounded worker comparisons, not independent output-quality evaluations or HTTP/API speed qualification. Detailed receipts, exact hashes, counters and medians are in [the host-overhead evidence record](../../../oNPU/docs/benchmark-qwen25-host-overhead-v1.json).
