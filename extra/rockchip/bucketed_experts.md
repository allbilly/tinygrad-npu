# Dynamic expert row counts

`rockchip_bucketed_int8` accepts grouped activations with a fixed row capacity
and an int32 count per expert. ABI 4 stores G*M FP32 activation scales, G counts,
64-byte header padding and G*M row-major INT8 activation rows. Weights use the
existing immutable grouped INT8 atlas. Counts must be in [0,M], with at least
one active expert. Inactive output rows are unspecified and must not be used.

The runtime prepares the task descriptor slots once. A compiled CPU updater
validates every count before changing commands, copies the qualified row-count
register template for each active expert, sets input/weight/output addresses and
links contiguous command chains. Experts are distributed by active ordinal,
not expert index, so sparse selections still balance across three cores. Fewer
than three active experts use one core. Command buffers are cleaned before the
blocking submission. Native replay reads dynamic task/subcore metadata and
reports the actual task count. Invalid counts fail without poisoning the device.

CPU checks pass 53 tests and 100 subtests. The only subsequent source change
before static qualification was an `Any` annotation for the CPU function pointer;
mypy on 277 files and Ruff pass. Hardware checks cover changed counts, empty
experts, signed extremes, exact integer outputs/task counts, invalid-count
recovery and native-to-ordinary transitions. Actual-depth cases use row capacity
128 at K896 and K1280. Twelve changed projection cases match fixed-NPU results
bit for bit, with no steady NPU allocations.

The complete operator still gathers, quantizes and copies all reserved rows.
At E64, 128 tokens/top-6, combined width 1792 and depth 1280:

| Routing | Prepared CPU ms | Fixed NPU ms | Dynamic NPU ms | Paired dynamic/fixed speedup |
|---|---:|---:|---:|---:|
| Balanced | 51.87 | 40.72 | 44.23 | 0.925x |
| Skewed | 51.97 | 42.93 | 35.36 | 1.188x |
| Mixed | 55.99 | 39.86 | 42.43 | 0.852x |

Speedups are medians of paired block ratios. Route-map/count construction is
outside timing; the CPU baseline includes bucketing in its prepared combined
kernel. Dynamic NPU work alone does not consistently improve the pipeline.
Compact active-route storage is the next target: 768 routed rows currently use
8,192 reserved rows. This API is not selected by app/model placement, and these
operator checks do not qualify full-model quality or speed against RKLLM.
Both serving models were restored and answered fresh requests.
[Raw reports and source/restoration evidence](../../../oNPU/docs/benchmark-bucketed-experts-verification-v1.json).

## Compact row storage

ABI 5 extends the descriptor with the total reserved row count R. Its header
contains R activation scales and G counts, then 64-byte padding and R INT8
rows. Input and output addresses use exclusive prefix sums of counts instead
of `expert * capacity`. Every count must fit its per-expert capacity, their sum
must be positive and at most R. Unused compact tail rows are unspecified.
`compact_matmul` derives expert IDs from counts for channel-scale selection;
the caller supplies input rows in expert order. Task descriptors remain fixed,
while commands and submission metadata update in ordinary and native replay.

CPU checks pass 58 tests and 110 subtests, with 20 hardware skips, plus mypy on
277 files and Ruff. Sequential hardware suites pass seven compact-format tests
and five fixed-slot tests. Actual-depth compact raw tests use G8, row capacity
128 and R512 at K896/K1280, with exact active integers and task counts. Twelve
changed balanced/skewed/mixed full projection cases match fixed outputs bit for
bit, including expert-scale selection and scatter. No steady NPU allocations
were observed during the native timing loops.

At the same E64, T128/top-6, width 1792 and depth 1280, R is 768 instead of
8,192 fixed slots:

| Routing | Prepared CPU ms | Fixed NPU ms | Dynamic slots NPU ms | Compact NPU ms | Paired compact/fixed speedup |
|---|---:|---:|---:|---:|---:|
| Balanced | 48.91 | 39.34 | 40.98 | 10.77 | 3.575x |
| Skewed | 50.79 | 49.68 | 39.12 | 6.54 | 7.491x |
| Mixed | 56.40 | 40.35 | 42.46 | 11.03 | 3.660x |

These new paired samples supersede the earlier dynamic-only experiment for
assessing compact storage. Speedups are paired block ratios. NPU measurements
include gather, format-specific quantization/copies, command updates, expert
scale selection and scatter; map/count construction is outside timing. The CPU
reference includes bucketing in a prepared combined-width kernel. Runtime maps,
model hooks, trained quality and repeated model timing remain open. These are
operator gains, with no preferred model-row or application placement changes.
[Compact samples and restoration evidence](../../../oNPU/docs/benchmark-compact-experts-verification-v1.json).

## Paired activation rows

The optional activation-residual path interleaves two rows per original route
and doubles each expert count. ABI 6 permits a logical capacity up to twice the
physical row tile. The updater splits each active count into qualified physical
matrix tasks, preserving compact prefix addresses and reusing the same expert
weight address. It validates all counts before modifying commands or state.
Task slots cover `groups * ceildiv(capacity, tile)`; native replay uses the
actual chunk count and per-core ranges. ABI 4 and 5 retain their earlier bounds.

After readback, one CPU kernel combines each pair using the explicit FMA
recipe documented in [routed experts](routed_experts.md). Channel-scale lookup
uses original counts and one expert ID per original route. The packed weight
atlas is shared, without a second packed copy for activation correction.

Hardware qualification passes changed compact counts at logical capacity 130,
with physical tile 128, and exact raw routed row-pair checks at real expert
depths. Synthetic model checks pass, but the trained residual candidate still
fails its local logit gate. Application placement remains disabled by default.

Weight-residual mode adds a second compact submission with original counts and
the original quantized A0 rows, using a separate W1 atlas. It reuses the same
route map and expert IDs. One CPU kernel combines A0W0+A1W0+A0W1 in the defined
FMA order; W0 is not packed twice. This passes operator and synthetic-model
qualification but still fails the trained DeepSeekOCR logit gate.
[Current three-product measurements and limitations](../../../oNPU/docs/benchmark-npu-experts-weight-verification-v1.json).
