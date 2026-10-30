# QKV preparation without per-head normalization

Add `--fused-qkv-without-norm` to a CPU/NPU inference benchmark using
`--prefill on --fuse-projections --cpu-attention --fused-qkv` to extend prepared
QKV to ordinary transformer blocks whose `qk_norm` is zero. The equivalent
`configure` option is `fused_qkv_without_norm=True`, alongside those existing
placement and fusion options. The extension remains opt-in.

The generated CPU kernel skips normalization reductions and weight loads,
then applies the existing rotary transform and writes FP16 keys and values.
Absent Q/K normalization weights must be specified together. Constant operands
retain the existing kernel ABI but are not read in this mode. Models with
per-head normalization keep their existing implementation.

INT8 projections feed raw three-core tiles to the preparation kernel, which
performs FP32 dequantization there. Biased projections add bias after the two
FP32 scale multiplies, preserving the dense path's rounding order. Bias is not
folded into integer arithmetic, and weight quantization is unchanged.

Existing guards remain: full-head RoPE, compatible attention, batch one,
supported head dimensions, CPU FP32 activation, forced prefill placement and
allocated context at most 4,096. Gates, attention softcaps, incompatible blocks
and larger contexts retain their regular path. Decode is unchanged.

CPU tests cover head layouts 14/2/64, 16/2/64 and 32/4/64; symbolic row and
position rebinding; exact preservation outside cache writes; V payload bits;
raw partial tiles; dense/raw equality; and both missing-weight mismatches.
Model tests exercise normalized-free preparation with and without biases,
cached continuations, explicit selection and unsupported context/gate fallbacks.
The suite passed 77 tests and 35 subtests, with seven explicit hardware skips.
Mypy checked 280 source files and Ruff passed.

Two initial test attempts are retained: symbolic query readback required
padding before NumPy conversion, and direct TinyJit return of a mutated input
alias required the precompiled function boundary used by the existing QKV
test. A focused probe verified that the kernel updated the rebound input
cache correctly in both cases; the supported function boundary also returned
the correct cache. Neither tolerance nor output checks were relaxed.

Full-model Qwen2, MiniCPM4 and TinyLlama measurements completed with four
ABBA processes each. Each process uses 128 inputs, 64 outputs, context 512,
three full warmups and three measured requests, followed by 32 full-vocabulary
teacher vectors. All teacher vectors and all 24 generated sequences per model
match exactly. The measured role aggregates are below.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 252.30 / 255.74 | 37.244 / 37.618 |
| MiniCPM4/0.5B | 539.71 / 607.20 | 44.807 / 45.982 |
| TinyLLAMA/1.1B | 183.06 / 189.40 | 24.216 / 24.338 |

Clocks remain adaptive, older preferred prefill rates remain faster, and the
option remains opt-in. These local regression checks do not establish comparable
HF/RKLLM quality. Both original sessions were restored and answered fresh requests.
[Raw reports and verification](../../../oNPU/docs/benchmark-qkv-without-norm-models-verification-v1.json).
