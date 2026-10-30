# Raw INT8 GELU feed-forward prefill

`--fused-gelu-ffn` combines raw gate/up projection scaling, GELU-tanh and
quantization for the down projection. The kernel reads the existing per-core
INT32 output layout directly. It avoids FP32 projection concatenation, gate/up
copies and an intermediate activation matrix. Each down-projection K group
retains its own maximum, scale, padded width and row-tiled header. Partial
integer products retain their existing FP32 dequantization and addition order.

The implementation is off by default and applies to the ordinary dense
`TransformerBlock` feed-forward method. Supported prefill capacity is 16–128
rows, hidden width is 512–12,288 in multiples of 32, and the joint weights must
fit one packed weight tile. Unsupported shapes, other activations, biases,
calibrated down-input scales and decode use the existing path. The custom
Gemma3n/Gemma4 decode methods are excluded from installation.

Raw projection metadata includes row and column fragments and their activation
scales. This handles Gemma2's 12,288/6,144-column projection fragments and
4,096/4,096/1,024-channel down groups. A dynamic active-row input writes
zero codes with the usual minimum scale for padded rows. All helper functions
are private; only the CPU wrapper is exported as the JIT entry.

CPU payload tests cover row/column fragments, a final one-row projection tile,
mixed down-group padding, active-row rebinding, nonfinite rows, invalid coverage
and exact scale/header/code bytes against the existing multi-stage path.
Sixteen runtime tests passed; four selection/fallback tests passed, along with
mypy and Ruff. An exclusive NPU regression compared complete fused and unfused
FFNs at 17 and 128 rows through seven changed inputs and native replay; every
FP32 result matched bit for bit.

Full Gemma2-2B requests use 128 input tokens, 64 outputs, context 512, three
full warmups and three measured requests per variant. Both use natural depth
2304 and the same capped attention, GELU and final-softcap options.

| Metric | Unfused control | Fused candidate |
|---|---:|---:|
| TTFT ms | 911.90 | 696.17 |
| Prefill proxy tok/s | 140.37 | 183.86 |
| Decode tok/s | 9.93 | 9.93 |
| Peak RSS MiB | 6,298.58 | 6,247.41 |

All twelve 64-token sequences match the prior candidate. Native execution
contains 26 fused pack calls in prefill and none in decode. Prefill throughput
is 31.0% higher against the current control; decode is effectively unchanged.
Clocks vary. The published RKLLM 213.90/10.37 tok/s reference remains faster.
Broader quality, fixed-clock parity and all-options serving API qualification
remain open. The earlier local host-quality sample is not a new evaluation of
this source cohort.

Both normal workers were restored and answered fresh requests. [Full reports,
exact payload/native checks, source identities and restoration](../../../oNPU/docs/benchmark-gemma2-gelu-verification-v1.json).

## Erf-GELU fusion

`--fused-erf-gelu-ffn` opts into raw INT8 projection scaling, erf GELU and
per-K-group quantization for ordinary dense Transformer FFNs. Add
`--fused-erf-gelu-decode` to enable the one-row decode path. Both default to
off. The equivalent configuration keywords are `fused_erf_gelu_ffn` and
`fused_erf_gelu_decode`. They require INT8 CPU/NPU projections; decode also
requires the base option and every gate/up/down projection forced onto NPU.
Unsupported shapes, biases and calibrated down-input scales retain the
existing path. Tanh-GELU and SwiGLU keep their separate selectors.

The implementation preserves the tensor erf polynomial's FP32 division and
multiplication order. It reads the raw per-core projection fragments directly,
avoiding dense gate/up copies, an intermediate activation tensor and separate
down-input quantization. Prefill can use the existing parked-worker pool;
decode is serial. Each down K group keeps independent scales and packed
headers, including partial and inactive rows.

The private CPU prototype passed 40 exact-payload cases, two dynamic-row cases
and 960 checked timing calls. Paired pipeline speedups were approximately
1.97–2.07× for pooled 128-row prefill and 2.58–2.89× for one-row serial packing
at Spark widths 6,656 and 10,240. These are operator timings, not LLM rates.
The port passed 169 tests with 12 skips, mypy on 282 files and Ruff. Full-model
qualification remains in progress.
[Operator evidence](../../../oNPU/docs/benchmark-raw-erf-gelu-operator-verification-v3.json)
and [CPU integration checks](../../../oNPU/docs/benchmark-raw-erf-gelu-integration-cpu-verification-v2.json).
