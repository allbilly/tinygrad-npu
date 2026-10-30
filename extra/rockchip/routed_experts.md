# Selected expert INT8 matrices

The experimental `rockchip_routed_int8` API executes one token across up to eight
selected experts. It prepares an immutable INT8 expert atlas and FP32 channel
scales once. Each call supplies per-route activations, activation scales and
int32 expert IDs. The existing grouped integer register recipe executes one
matrix per route across three core command chains. Duplicate IDs are allowed.

The 36-byte ABI-3 descriptor extends the grouped matrix descriptor with the
atlas expert count. The input header stores G*M FP32 scales, G int32 IDs, padding
to 64 bytes, then G*M row-major INT8 activation rows. Ordinary decode uses M=1;
the optional activation correction uses M=2 to share each expert matrix between
the original and residual activation rows. Selector offsets account for M.
The command buffers retain their shapes and task chains; each replay validates
all IDs, patches the weight address of each task and cleans the modified command
buffer before submission. Ordinary and native replay both perform this step.
The atlas buffer must cover all experts and fit the 32-bit IOVA address range.

This extends the existing per-row activation/per-output-channel weight INT8
arithmetic. It does not preserve original FP16 model logits by construction.
Integer golden tests, complete projection checks, model quality and full-model
timing are separate requirements. The application worker does not enable this
path. Explicit backend configuration can select supported model expert blocks.
No model or RKLLM throughput claim follows from adding the API.

The separate [bucketed prefill API](bucketed_experts.md) now accepts external
per-expert row counts and skips empty experts. Model integration now constructs
stable runtime route maps and uses compact active-route buffers for prefill.

Initial CPU qualification passed 49 tests and 87 subtests, with 17 hardware skips,
plus mypy on 275 files and Ruff. A hardware run passes six tests covering raw
grouped 17/128-row matrices and routed decode at physical depths 896 and 1280.
Changed IDs, duplicates, signed extremes and ordinary/native replay are checked.

`ONPU_INT8_EXPERT_DEPTHS=1` selects these unpadded depths; its default is zero.
Per-row/channel scales and non-padding codes remain unchanged. Complete routed
decode projections match their padded counterparts bit for bit across six
changed inputs. Median paired speedups are 1.134x for combined gate/up width
1792, depth 1280, and 1.081x for down width 1280, depth 896. The test atlas has
eight experts and six selected routes. Paired speedups are medians of per-block
ratios, rather than ratios of separate latency medians.

A separate balanced prefill probe shares the atlas with the existing grouped
INT8 API. At 128 tokens/top-6/E64, width 1792 and depth 1280, twelve rows per
expert measure 10.34 ms on NPU versus 51.89 ms in the prepared combined-width
CPU kernel, a paired 5.090x speedup. Reserving/computing 128 rows per expert
measures 41.82 versus 51.72 ms, a paired 1.225x. These include NPU gather,
quantization, copies, padded computation and dequantization/scatter; route-map
creation is excluded. Routing remains balanced in both tests. The padded case
models allocation/computation capacity for unique top-k routes, without proving
performance for arbitrary routing.

Operator differences from original FP16 projections are about 1.1% relative
RMS; model quality remains unqualified. The initial prefill harness asserted
native execution one call too early. A four-call diagnostic established native
support, and the corrected six-change probe records 22 native runs per capacity.
At that stage, dynamic counts, runtime maps and model hooks remained open.
Both serving models were restored and answered
fresh requests. No application placement or preferred model row changed.
[Hardware samples and source/restoration evidence](../../../oNPU/docs/benchmark-expert-depths-prefill-verification-v1.json).

## Activation residuals and model qualification

`activation_residual=True` evaluates two activation products against one packed
weight atlas. The second input is quantized from a correctly rounded FMA
remainder. Dequantization explicitly computes
`fma(float(dot0)*a0, weight_scale, (float(dot1)*a1)*weight_scale)` on CPU.
This matches the independent CPU expert recipe. Separately rounding both weight
products differed by up to 2.38e-7 at one gate/up projection; a later INT8
vocabulary projection amplified the difference enough to fail the model test.
The explicit FMA fixes that failure without changing its tolerances.

Current qualification passes 95 CPU tests and 86 subtests, mypy and Ruff, two
hardware operator tests with nine subtests, and all four synthetic model modes
(with/without residuals, all-NPU/CPU-down hybrid). Raw tests include K64, K896
and K1280, changed selectors/counts, signed extremes, chunking and native replay.

The trained precision-protected DeepSeekOCR residual candidate measures 165.93
prefill proxy tok/s and 33.28 decode tok/s, versus the published RKLLM 182.54 and
31.60. Its local relative-logit RMS is 0.054358, exceeding the unchanged 0.01
limit. Thus it is not the preferred timing variant and remains off by default.
This is text-decoder evidence, not complete VLM or comparable-quality
qualification. [Trained result, gates and restoration](../../../oNPU/docs/benchmark-npu-experts-activation-verification-v5.json).

## Weight residuals

`pack_weight_residual` prepares separate W0 and W1 atlases from an FP32 FMA
remainder of the original weight. It retains one W0 copy, shared by A0W0 and
A1W0. `weight_residual=(W1, scales1)` requires activation correction; a second
raw submission reuses A0 codes against W1. The CPU kernel combines the activation
pair with an explicit FMA, then fuses the third product into that rounded result.
Prepared atlases use configured sequential memory domains. Counts and selected
expert IDs retain their existing contracts. Original state tensors remain.

The model option `npu_expert_weight_residual=True` and benchmark flag
`--npu-expert-weight-residual` require activation residuals too. They remain off
by default. Qualification passes 98 CPU tests and 86 subtests, mypy, Ruff, two
operator tests with twelve subtests, six synthetic model modes and wide
vocabulary integer checks. The indexed-device observer is audited separately;
it originally missed expert labels on `ROCKCHIP:1`, while all 44 expert NPU
programs per phase appear in the saved full labels.

The all-MoE trained candidate measures 161.79 prefill proxy tok/s and 27.10 decode
tok/s, with about 11 GiB peak RSS. Its relative-logit RMS remains 0.055134,
exceeding 0.01; it is rejected and not the preferred timing variant. A CPU-only
projection probe shows substantially lower local arithmetic errors with three
products, but that does not establish trained-model quality. The remaining
discrepancy needs an intermediate model trace. The original observer failure
and warmup memory-limit adjustment are retained in the evidence.
[Trained result and audit](../../../oNPU/docs/benchmark-npu-experts-weight-verification-v1.json).

## Natural-text quality sample

A new single-product model pair measures 267.33 prefill proxy tok/s and 37.38
decode tok/s for the NPU candidate, versus the published 182.54/31.60 targets.
The model then scores 512 English WikiText-2 test targets from sixteen separate
prefixes. Candidate/control perplexity ratio is 0.996349, with an empirical 95%
sequence-bootstrap upper estimate of 1.011648, passing the predeclared local
1.02 PPL limit. Every logit dump is independently finite-checked and rescored.
Repeating the first sequence after the others reproduces all its output bits.

This adds evidence, not a broader quality certificate. Natural-vector mean TV
is 0.046617 and argmax agreement is 94.73%; outputs are not identical. The
original repeated-prompt logit RMS gate still fails at 0.065740 against 0.01.
The control has FP16 CPU experts with the same INT8 dense path, rather than a
full FP16 model or matched RKLLM export. Image/OCR and other languages are not
tested. The numerical gates, preferred row and application defaults remain.
[Natural-text comparison, uncertainty and restoration](../../../oNPU/docs/benchmark-deepseek-natural-quality-verification-v1.json).
