# Explicit prefill pooling for wide FFNs

`--cpu-pool --cpu-attention --cpu-prefill-pool` now honors the explicit prefill
request even with two KV heads at context 512. The earlier creation guard
considered decode attention, recurrent, MLA and expert eligibility only.
Existing defaults and startup-failure fallbacks remain unchanged.

Wide INT8 FFNs that cannot use one packed input group can now use the existing
parked-worker sigmoid path while the prefill scope is active. Dense projection,
activation multiplication order, split-K quantization and NPU registers are
unchanged. Pool creation also enables its existing decode affinity policy.

The three-model ABBA comparison uses 128 inputs, 64 outputs, context 512,
three warmups and three measured requests per process. All 24 generated
sequences and all four sets of 32 teacher vectors per model match exactly.

| Model | Prefill proxy tok/s, control / candidate | Decode tok/s, control / candidate |
|---|---:|---:|
| Qwen2/0.5B | 262.83 / 373.38 | 37.429 / 37.934 |
| TinyLLAMA/1.1B | 183.97 / 267.10 | 24.353 / 24.418 |
| Qwen2.5/1.5B | 109.27 / 158.52 | 15.493 / 15.866 |

Preferred CPU+NPU rows use the slower of the two candidate process TTFTs;
both candidates improve both phases over the previous preferred timings.
Clocks remain adaptive; comparable HF/RKLLM quality is unqualified.
CPU checks passed 82 tests and 35 subtests, mypy and Ruff. Both original API
models were restored and answered fresh requests.
[Raw evidence](../../../oNPU/docs/benchmark-explicit-prefill-pool-models-verification-v1.json).
