# Grouped NPU projections

`pack_grouped_weight` packs a separate `[N,K]` weight for every group.
`grouped_matmul` consumes CPU `[groups,rows,K]` activations and returns CPU
FP32 `[groups,rows,N]` results. Inputs convert to FP16 before multiplication.
Packing immutable weights happens before captured execution.

The new runtime assigns whole heads round-robin across three cores. Each core
executes its own register-command chain; forty heads use 14/13/13 heads per core.
Rows beyond a hardware tile become additional tasks in the same chain. All tasks
share one submission, one input buffer, one weight buffer and one output buffer.
One-core execution is also available. Ordinary matrix and row-batched plans
retain a group count of one.

Input/weight finite checks cover every group, including native replay inputs.
Task preparation checks buffer extents and memory domains. Chain construction
keeps each core's command links inside its assigned task range.

The explicit API requires static geometry: groups up to 128, rows up to 128,
and padded K/N up to 4096. The hardware evidence below qualifies two projection
shapes at forty heads, rather than every shape admitted by those bounds.

| Projection, all 40 heads | Rows | CPU ms | Grouped NPU ms | CPU / NPU |
|---|---:|---:|---:|---:|
| Query latent, K64/N256 | 1 | 0.597 | 0.354 | 1.69× |
| Query latent, K64/N256 | 17 | 7.534 | 1.008 | 7.48× |
| Query latent, K64/N256 | 128 | 34.493 | 7.560 | 4.56× |
| Value, K256/N64 | 1 | 0.593 | 0.313 | 1.89× |
| Value, K256/N64 | 17 | 7.534 | 0.802 | 9.39× |
| Value, K256/N64 | 128 | 36.971 | 7.084 | 5.22× |

These are complete synthetic projection operators, including input conversion,
layout, transfers, native dispatch and output assembly. Eight interleaved timing
blocks contain sixteen measurements per variant; the grouped three-core path
wins every paired block. CPU frequency and memory clocks were not fixed.
The control is the current CPU Tensor projection with prepared FP16 weights.

All six changed inputs per shape passed an independent NumPy reference using
FP16 input conversion and FP32 accumulation. Grouped one-core, grouped
three-core and separate NPU outputs matched bit for bit. The separate NPU
comparison also uses native replay: reducing Python overhead alone does not
account for the grouped result.

CPU tests passed 32 tests and 50 subtests with five explicit hardware skips.
The hardware run passed all four grouped tests and 29 subtests, including
changed inputs, native replay and non-finite values in the final head. Mypy
passed 271 files and Ruff passed. The initial mypy narrowing error and three
test formatting errors are retained in the check logs and were corrected.

The optional MLA model hook now handles symbolic token counts through bounded
tiles and independent prefill/decode selection. Its trained mixed INT8-dense /
FP16-grouped candidate fails a finite-activation guard in captured decode.
The FP16 synthetic operator results above therefore do not establish trained
model correctness or a full-model/RKLLM win.

## Grouped INT8 model experiment

`rockchip_grouped_int8.py` adds per-row activation and per-output-channel weight
scales, signed INT8 matrix operands, INT32 accumulators and FP32 dequantization.
It packs immutable weights once and preserves original model tensor identities.
All group/row activation scales are checked, including during native replay.
The INT8 ABI uses group-major row-major inputs and outputs, a shared aligned
scale header and one per-core command chain. It is distinct from the FP16 layout.

Changed signed inputs at forty heads, rows 1/17/128 and both K64/N256 and
K256/N64 pass independent NumPy integer references and native replay checks.
A small complete model also passes changed-prefix cached generation against a
CPU integer-dot reference. These checks establish the operator's integer
contract; they do not establish that quantizing trained MLA projections
preserves model quality.

The trained MiniCPM3 run completes at 19.84 prefill proxy tok/s and 4.81 decode
tok/s, versus 10.62 and 3.97 for its matched CPU-MLA control. It **fails** the
predeclared local full-vocabulary logit gates: relative RMS 0.16412 exceeds
0.01, and mean probability total variation 0.06455 exceeds 0.02. Its 0.96179
sampled perplexity ratio passes the 1.02 bound, but does not override those
failures. The 32 targets repeat the prompt and are not a held-out quality test.
The candidate is rejected for preferred-row promotion and remains opt-in;
RKLLM's 90.88/5.94 tok/s and comparable output quality remain open. CPU attention
and surrounding FP32 arithmetic remain. Both normal serving models were restored
with their options and answered fresh requests.
[Trained reports, failed gates and restoration](../../../oNPU/docs/benchmark-minicpm3-mla-int8-verification-v1.json).

## Residual correction experiment

The optional `int8_residual` precision approximates both operands twice. With
`A = A0 + Ar` and `W = W0 + Wr`, it computes `A0W0 + A1W0 + A0W1`, where
`A1` and `W1` are separately scaled INT8 approximations of the residuals. The
second-order residual product is omitted. Each residual element uses an explicit
FP32 fused multiply/subtract so compiler fusion choices cannot change its scale
or codes. Original model tensors and the single-product INT8 path are preserved.

Weight groups are `[W0,W0,W1]` and activation groups are `[A0,A1,A0]`; all three
products share one submission. Scales remain per row and per output channel.
This requires three times the matrix work and packed weight bytes. The group
limit is 42 original heads, or at most 126 physical groups. It is a precision
correction experiment, not a claim of native 16-bit integer arithmetic.

A CPU-only study of the first, middle and last trained MiniCPM3 MLA tables,
using random normal and outlier inputs, reduced relative projection RMS error
from 0.7–4.5% to 0.003–0.043%. These are sampled projections, not trained model
activations or full-model quality. CPU packing/payload checks pass, along with
mypy and Ruff. Actual NPU qualification passes six tests and 32 subtests,
including changed inputs, outliers, native replay and one submission per
projection. Both complete small-model generation tests pass against CPU integer
projection references, with changed prefixes and cache resets.

The trained matched run measures **16.20 prefill proxy tok/s and 4.41 decode
tok/s**, versus **10.64 and 3.98** for its CPU-MLA control. It nevertheless
**fails all three unchanged local numerical limits**: relative logit RMS
0.159280 exceeds 0.01, mean probability total variation 0.063834 exceeds 0.02,
and sampled perplexity ratio 1.028491 exceeds 1.02. Argmax agreement is 93.750%.
The candidate is rejected for preferred-row promotion. Projection primitives
passing does not establish correctness or quality of the complete trained path.
The similar whole-model error after residual correction leaves its cause open;
these results do not establish quantization precision as the sole cause.

All six timing requests complete under the original 10 GiB soft/11 GiB hard
memory limits. The later non-greedy logit diagnostic repeatedly hits the soft
limit during compilation; raising only that diagnostic's soft limit to 10.5 GiB
allows completion, with the 11 GiB hard limit and 2 GiB board reserve retained.
Saved timing trials and medians are unchanged. Both normal serving models are
restored and answer fresh requests.
[Trained residual results, failed gates and memory-policy evidence](../../../oNPU/docs/benchmark-minicpm3-mla-int8-residual-verification-v2.json).

A subsequent trained trace captures the inputs and outputs of all 124 query/value
projections at the final fixed-teacher decode position. All 32 full-vocabulary
logit vectors remain bitwise identical to the untraced run. All source Half
weights, packed INT8 codes and FP32 scales match independent reconstruction.
Every captured projection output matches the independent scaled integer recipe
bit for bit; the largest RMS difference from the original CPU projection is
0.2254%. This checks that single decode position, not prefill or earlier cache
history. The complete model's failed quality gates remain open.
[Actual projection trace and independent analysis](../../../oNPU/docs/benchmark-minicpm3-projection-trace-verification-v1.json).

`configure(..., npu_mla_projections='query')` and the benchmark option
`--npu-mla-projections query` now select query projections independently from
value projections; `value` and the existing default `both` are also supported.
Only selected weights are packed, and unselected projections retain their
original CPU dot. CPU checks pass 36 tests and 37 subtests; NPU operator checks
and all three complete small-model cases pass. The matched trained query-only
run measures 12.69 prefill proxy/4.27 decode tok/s versus 10.74/4.03 for its
control, but fails every unchanged local numerical limit: relative logit RMS
0.164209, mean probability variation above 0.02, and perplexity ratio 1.021769.
It is rejected for preferred-row promotion. Keeping values on CPU does not
resolve the complete trained-model error. Both original serving models are
restored and answer fresh requests.
[Query-only timing and failed gates](../../../oNPU/docs/benchmark-minicpm3-mla-query-only-verification-v1.json).

Historical residual raw reports contain a stale FP16 suffix in their `precision`
text. Their `grouped_mla_precision='int8_residual'`, integer program ABI and
independently checked packed codes establish the actual INT8 execution. The
suffix is corrected for new reports; historical reports and hashes are retained.

[Raw operator results](../../../oNPU/docs/benchmark-minicpm3-grouped-npu-operator-v1.json)
and [verification](../../../oNPU/docs/benchmark-minicpm3-grouped-npu-operator-verification-v1.json)
retain source hashes, per-call measurements, submission counts and restoration.
