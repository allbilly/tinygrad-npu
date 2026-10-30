# Fused raw SwiGLU and split-K packing

`--fused-swiglu-ffn` selects an optional CPU packer for wide INT8 SwiGLU FFNs.
Use it with forced CPU/NPU prefill and fused projections. Supported hidden
widths are greater than 4,096 and at most 12,288, in multiples of 32. Inputs
must be batch-one CPU FP32 with a bounded prefill capacity of 16–128 rows.
Biased projections, down-input scaling, clipping and nonzero up-bias keep
their existing path.

The implementation shares raw projection tiling and split-K reduction with
the existing GELU packer. It combines INT32 dequantization, FP32 SwiGLU,
independent maxima/scales for each down-projection K group, and INT8 packing.
Each K group retains its existing weights and quantization. Row and column
fragments, partial rows, header padding and invalid-row behavior are preserved.
The model hook precedes the existing SwiGLU hook only when the option is
selected and supported; decode retains the existing implementation by default.

`--fused-swiglu-decode` additionally opts into this packer for a one-row decode
phase. It requires `--fused-swiglu-ffn`, CPU INT8 inference and an active NPU
decode phase. Every gate/up member and the down projection must select NPU
decode. Other capacities, dtypes, placements and unsupported layouts retain
their existing path. The one-row kernel is serial and keeps the materialized
sigmoid rounding and each independent K-group quantization.

The decode extension passed 44 related CPU tests, with three explicit hardware
skips, mypy on 281 files and repository Ruff. New tests compare full payload
bytes and scale bits with the ordinary Tensor sigmoid path, including zero
and nonfinite inputs. The exclusive NPU/native replay test passed at both
896/4864/896 and 1024/4608/1024 input/hidden/output widths with seven changed
inputs each. Full-model Qwen2 results are recorded below; other models still
need full-model measurement.
Installed decode hooks and reported counts require every gate/up and down
projection to resolve to NPU decode; mixed placement keeps the ordinary path.
[Decode CPU evidence](../../../oNPU/docs/benchmark-raw-swiglu-decode-cpu-verification-v2.json).

The Qwen2 decode comparison keeps natural K896, the same prefill worker pools
and all other options in both arms. Four ABBA processes use 128 input IDs,
64 output tokens, context 512 and three warmups/three measured requests.

| Recipe | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|
| Control | 505.21 | 40.563 |
| Raw decode candidate | 473.27 | **41.748** |
| Published RKLLM reference | **877.31** | 41.580 |

These are six-request role medians. The two candidate process medians are
41.796 and 41.700 decode tok/s. All 24 generated sequences and all four sets
of 32 full-vocabulary teacher vectors match each other and the prior preferred
output bitwise. Candidate decode is 2.9% faster than control, with only a small
margin above the published RKLLM decode result; prefill remains slower.
Both candidates failed to improve prior preferred TTFT, so the balanced
preferred row is retained and the decode option remains off by default.
Clocks were adaptive; comparable RKLLM quality remains unqualified.
Both original serving models were restored and answered fresh requests.
[Decode model and hardware evidence](../../../oNPU/docs/benchmark-qwen2-raw-decode-models-verification-v1.json).

`ONPU_SWIGLU_PREFILL_ROWS=2/4` opts into direct row batching during raw
SwiGLU prefill. The default is 1; one-row decode and GELU retain their existing
program. Fully active rows within one raw-tile interval share channel-loop
loads. Odd tails and inactive or boundary-crossing groups keep the original
row path, preserving per-row maxima, invalid flags and per-K quantization.

The direct generator passed 174 payload/scale comparisons and 640 timed
output checks. Its four-row Qwen-width operator won 36 of 40 paired blocks
(1.229x median paired speedup). The liquid-model-width result was less
consistent (21/40), so an operator gain is not assumed for every model.
Integration checks passed 51 tests, with three hardware skips, plus mypy and
Ruff. The actual NPU/native full-FFN test then passed 28 changed-input checks
across SiLU/GELU at 17 and 128 rows, including four-row source selection.

Qwen2 full-model ABBA candidates preserve all 24 generated sequences and
all four full-vocabulary teacher arrays bitwise against the prior preferred
output. Both candidates improve prior preferred TTFT and decode. The
conservative selected process measures 242.80 ms TTFT, 527.19 prefill proxy
tok/s and 41.919 decode tok/s. Role aggregates are 515.65/41.992 control and
532.21/42.002 candidate; those aggregates are not the selected process.
Published RKLLM prefill remains faster at 877.31 tok/s; the local decode
point is slightly above its 41.58 tok/s. Comparable RKLLM quality remains
unqualified. Both normal API sessions were restored and answered fresh requests.
[Row-batched model and qualification evidence](../../../oNPU/docs/benchmark-qwen2-row-batch-models-verification-v1.json).

TinyLLAMA/1.1B also completed four ABBA processes with fused SwiGLU decode
and four-row prefill batching. All 24 generated sequences and four
32-by-32,000 FP32 teacher arrays match the prior output bitwise. Both
candidates improved prior preferred TTFT and decode; the slower-TTFT candidate
records 366.06 ms, 349.67 prefill proxy tok/s and 25.565 decode tok/s.
Six-request role aggregates are 347.20/24.997 control and 358.06/25.295
candidate. These combined-option measurements do not isolate either option.
RKLLM's published prefill remains faster at 524.74 tok/s, while its decode
is 24.43 tok/s. Comparable quality remains unqualified. Both serving models
were restored and answered fresh requests.
[TinyLlama model evidence](../../../oNPU/docs/benchmark-tinyllama-row-batch-models-verification-v2.json).

Static fragment addresses compute the complete integer offset before adding
it to a raw buffer pointer. This permits negative constant offsets whose
valid loop indices bring the final address into the selected fragment.

CPU verification passed 91 tests and 35 subtests, with nine explicit hardware
skips, plus mypy on 280 files and repository Ruff. The first NPU test passed
an algebraically fused tensor reference, but full-model checking exposed a
rounding mismatch against the materialized sigmoid used by serving. Division
by the sigmoid denominator changed payload and scale bits. The corrected
kernel materializes reciprocal rounding before gate/up multiplication, and
the test reference now calls the serving sigmoid helper. The corrected
full-FFN NPU/native-replay test passed at 17 and 128 rows with seven changed
inputs. Corrected model speed measurements are recorded below.

The earlier CPU-only prototype also preserved payload and scale bits in eight
cases across widths 4,864, 5,632 and 8,960, including partial rows and raw tiles.
These checks establish local arithmetic/layout equivalence, not comparable
HF/RKLLM quality or an inference speed win.

The failed first sweep and its arithmetic trace are preserved in
[the integration record](../../../oNPU/docs/benchmark-grouped-swiglu-integration-status-v1.json).

Full-model corrected v2 measurements completed with four ABBA processes per model.
All 32 teacher vectors and 24 generated sequences per model match exactly.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 392.70 / 507.22 | 37.995 / 37.978 |
| TinyLLAMA/1.1B | 280.40 / 337.78 | 24.054 / 24.112 |
| Qwen2.5/1.5B | 158.72 / 230.75 | 15.905 / 15.857 |

Preferred rows use conservative candidates only when both phases improve; clocks remain adaptive and comparable HF/RKLLM quality is unqualified. Both original serving models answered fresh requests.
[Full evidence](../../../oNPU/docs/benchmark-grouped-swiglu-models-verification-v2.json).
