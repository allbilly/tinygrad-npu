# Spark half-row SDK prefill

The explicit SDK path supports `(M,N,K)=(64,12288,2048)` and `(64,1024,2048)`.
The first shape uses three 4096-column cores. The second uses 352, 352 and
320 columns. Each core executes one task, retaining the SDK's 108 ordered
register writes. The input is A16, the output is C4, the feature surface
register `0x107c` is 256, and output surface pitch `0x4024` is 1024.

These are observed M64 commands, rather than truncated M128 commands. Four
single-core SDK captures (including width 1024) pass their integer oracles.
The three widths selected by the new full shapes match every ordered capture
word in CPU tests. Existing M64/N1024-per-core generation, used by the older
full N3072 shape, retains its separate recipe.

Both the experimental literal commands and the production generator pass
28 full INT32 comparisons, guarded output checks and 29 alternating FP16
health calls. The production device-call medians, ordinary versus SDK, are
1.659/1.175 ms for N12288 and 0.582/0.465 ms for N1024. These omit packing and
do not measure complete inference. Tensor graph/native FFN replay, streaming
overlap, quality and API qualification are still required.

The intended next experiment overlaps independent row halves:

1. Produce gate/up for rows 0–63, then start gate/up for rows 64–127.
2. Pack the first half on the CPU while the NPU produces the second half.
3. Start the first half's down projections while packing the second half.
4. Complete the second half's down projections and preserve the original
   per-K-group dequantization and FP32 addition order.

CPU prerequisites already pass at active row counts 1, 17, 64, 65, 127 and
128: independently quantized A16 halves match the full input, SDK C4 raw
halves produce byte-identical packed activations, and reassembled scale bits
match the original 128-row operation. An entirely inactive half is skipped
and receives zero codes with minimum scales.

Use separate GEM allocations for concurrently accessed halves. A blocking
submission completing one half does not authorize CPU access to an allocation
that the NPU is still writing. Retain explicit cache synchronization, buffer
lifetimes, worker ownership and blocking completion. First prove a complete
mixed pipeline with changed inputs and an independent integer oracle, then
measure paired unprofiled complete requests. Do not infer a full-model gain
from the matrix-call timings or the CPU layout proof.

Evidence is under
`/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/`:
`spark-half-sdk-capture-20261102-v1`,
`spark-half-sdk-primitive-20261102-v1`,
`spark-half-sdk-production-20261102-v1`, and
`staged/spark-half-ffn-layout-20261102-v1`.
