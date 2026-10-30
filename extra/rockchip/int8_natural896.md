# Natural K896 for Qwen2 INT8 projections

`ONPU_INT8_EXPERT_DEPTHS=1` selects physical K=896 instead of padding to
1024 for Qwen2/0.5B's input-width projections. It removes 12.5% of their
packed weight and activation channels. Quantization still uses the logical
input width; zero padding does not change codes, scales or integer sums.
The existing switch also selects K1280. This model study covers K896.

Six raw three-core geometries passed 48 independent NumPy INT32 comparisons:
four changed-input patterns, including signed extremes, in natural and padded
arms. Outputs were poisoned before every submit; all actual/expected hashes
match, with zero maximum error and no remaining poison.

Four full-model ABBA processes use CPU INT8, NPU projections, 128 input IDs,
64 output tokens, context 512, three warmups and three measured requests.
Both arms retain CPU prefill and raw SwiGLU worker pooling, 24 attention/FFN
blocks and 49 CPU norms. Packed projection sharing is disabled. Only the
expert-depth switch changes between arms. Actual command inspection confirms
79 input-width plans / 237 core-task records per phase, with K1024 in control
and K896 in candidate, including all six full vocabulary tiles and its tail.

| Role | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|
| Padded control | 514.47 | 37.895 |
| Natural candidate | 496.03 | 40.625 |

Each role aggregates six measured requests. Measured decode improves 7.2%; the
prefill proxy decreases. All 24 generated sequences and all four FP32 arrays of 32
full-vocabulary teacher vectors are bitwise equal, including to the prior
preferred run. Float64 mean NLL is 0.7093233022609162 in all four processes.

The default remains zero and the preferred benchmark row is retained: both
candidate processes did not individually improve both phases over the prior
preferred report. Published RKLLM remains faster at 877.31 prefill proxy tok/s
and 41.58 decode tok/s. Comparable HF/RKLLM quality and the all-model speed
goal remain unqualified. Clocks were adaptive. Both original serving models
were restored and answered fresh requests.
[Full evidence](../../../oNPU/docs/benchmark-qwen2-natural896-models-verification-v1.json).
