# CPU GELU-tanh gating

Gemma4's generic GELU gate/up computation consumed approximately 669.5 ms
across 90 prefill calls. The optional `--cpu-gelu` path uses the same FP32
exp2 polynomial, range reduction, exceptional cases and operation grouping in
vector form. Large prefill arrays use four workers. Known slower one-row and
small side-input shapes retain the generic path.

The helper accepts separate gate/up tensors, including the noncontiguous
halves of a joint projection. Its measured input-copy cost is included. At
128 rows, paired fixture speedups were 4.10× for width 4096 and 5.27× for
width 12288, with bitwise-identical outputs. The selected decoder and width-128
fallback controls were within approximately 1% of parity.

Kernel and model checks cover NaN/infinity propagation, signed zeros, dynamic
row counts, zero padding, sparse Gemma3n thresholds, side inputs, cached decode
and reset. The final combined CPU suites passed 65 tests and three subtests;
mypy checked 266 source files and Ruff passed.

The complete Gemma4 candidate records 146.95 prefill proxy tok/s and 9.05
decode tok/s at the 128-input/64-output/context-512 protocol, with three full
warmups and three measured requests. All six sequences match the prior
candidate. Both rates remain below the published RKLLM reference; comparable
quality remains unqualified.

[Raw model/primitive samples and source cohorts](../../../oNPU/docs/benchmark-gemma-host-kernels-verification-v4.json)
also verify fresh replies from both restored normal serving workers.

## Erf GELU prefill

Spark uses `gelu(approximate='none')`, which follows the tensor erf polynomial
and differs from the tanh path above. `--cpu-erf-gelu` or
`configure(..., cpu_erf_gelu=True)` opts into its serial vector implementation.
The kernel preserves division order, FP32 operation grouping and multiplication
order. It supports matched CPU FP32 gate/up tensors with capacity 16–128 rows
and widths 512–32,768 divisible by four. Other shapes and dtypes use the
generic tensor formula; one-row decode remains generic. The option defaults
to false and does not change the tanh option.

Native Spark profiles identify erf GELU as approximately 313 ms of 1.7B
prefill and 667 ms of 4B prefill. The private operator prototype matched finite
output bits, special-value classes and dynamic-row outputs, including poisoned
inactive rows. Its 960 checked timing calls use actual tensor GELU as baseline.
At 128 rows, serial vector medians are 5.93 versus 11.48 ms for width 6,656
and 8.78 versus 17.41 ms for width 10,240; both win all 40 paired blocks.
Pooling gave no consistent extra benefit, and one-row timing regressed, so the
production helper uses serial prefill only.

The integration suite passed 80 tests with three explicit hardware skips;
mypy checked 282 source files and repository Ruff passed. Full-model NPU
qualification and end-to-end speed measurements remain pending.
[Diagnostic profiles](../../../oNPU/docs/benchmark-spark-phase-profiles-verification-v1.json)
and [operator evidence](../../../oNPU/docs/benchmark-cpu-erf-gelu-operator-verification-v1.json).
