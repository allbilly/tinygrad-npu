# Qwen2 prefill layouts and grouped-query attention

The exact Qwen2-0.5B Q8_0 checkpoint at context 512 uses INT8 projections with FP32 CPU helpers. These changes preserve the existing quantization boundaries and float arithmetic. Decode keeps its original graph. Defaults remain off in tinygrad; the app selects its checked checkpoint/context recipe.

## Blocked gate/up projection

`ONPU_INT8_SDK_PREFILL=1` admits the qualified M/N/K=128/9728/896 matrix. Passive RKNN SDK captures cover both per-core widths3264 and3200. All 108 ordered non-address register values match the new emitter. Unlike the old generic SDK formula, the K896 recipe uses feature-surface stride `M*4`, plus the captured low-dimension output fields. Copying fetch grain alone into the row-major path produced incorrect integer outputs and was rejected.

Input codes use 16-channel blocks and raw output uses 4-column blocks. `ProjectionTile.sdk` lets raw SwiGLU consume that output directly. There is no intervening transpose, re-quantization or changed scale. CPU tests cover full/partial rows and row batches 1/2/4. A serialized FFN test compares changed inputs bitwise through ordinary and native replay.

## One packed input and safe borrowing

`ONPU_FFN_PACKED_UPLOAD=1` uploads the complete SwiGLU payload once when all down-projection K groups use the same NPU device. Each original K group retains its own scale header and row tile. Explicit `payload_offset` values locate those headers in the shared allocation, avoiding unsupported general-purpose movement kernels on the NPU. Offsets are aligned and bounds checked; both ordinary execution and native replay validate the selected scale header.

The packer fully initializes payload bytes, inactive rows, padded channels and scale-header padding. Replay's full-write proof checks the exact extent before redirecting its private CPU output to the NPU mapping. Input/output borrowing can coexist only if every redirected NPU write range is disjoint from the borrowed source. A shared allocation alone does not imply overlap. Overlap, intervening writes, future reads and ordinary replay keep their conservative checks and fallbacks.

A prior prototype lost output borrowing when the same CPU kernel wrote directly to another NPU range. That introduced large copies and erased the input-upload saving. Synthetic overlap/disjoint tests and a hardware test requiring both aliases guard this contract.

## Grouped-query CPU prefill

`ONPU_CPU_GQA_PREFILL=1` reuses the established decode GQA reductions across query heads sharing one KV head. Rows are distributed over four persistent workers. It retains every dot-product and PV accumulation order and the original per-query softmax. Prepared INT8 QKV writes token-major query storage for the consumer; the public logical query shape and FP16 KV layout stay unchanged. One-token decode, unsupported geometry, windows and softcaps retain the ordinary path.

CPU tests compare every output bit with the original attention across dynamic rows, cached prefixes, length 512 and NaN-poisoned future cache. They also compare logical QKV queries and all cache bytes. Full-model tests require the previous 64-token sequence and 32 full-vocabulary teacher vectors to match bitwise.

## Evidence and rejected paths

Raw SDK gate/up execution improved about 1.4x in 20 paired blocks, with all independent integer outputs exact. This is not an LLM speed claim. The first full-model SDK/GQA process median was 888 tok/s and a repeat was 848 tok/s; their spread prevents treating one process as a repeatable published-reference win. The app/API cohort is reported separately in oNPU's benchmark.

FP16 NPU attention passed its operator oracle but was slower for this small head layout. Its data preparation and submission costs outweighed its compute saving; the prototype was removed. Smaller fetch grain, buffer shrinking, SDK down-projection tiling, eight-row temporary padding, an approximate exponential, and O3 compilation did not establish the required complete-model gain. A main-CPU wake helper prevented sleep transitions but improved paired prefill only about 2%; it remains an external experiment.

Run CPU and serialized hardware checks using the repository instructions. Raw captures, source cohorts, all losing trials, and API restoration receipts live under `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/` and are indexed by `~/oNPU/benchmark.md`.
