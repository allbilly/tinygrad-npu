# Prepared CPU static projections

Gemma3n's six AltUp input/unembedding projections previously converted BF16
checkpoint values to FP16 inside generic CPU matrix products. Their profile
recorded 5.532 seconds across six prefill calls and 42.7 ms per decoded token.

The optional prepared path snapshots the same FP16 weight values once, packing
eight output columns together with their complete reduction dimension. Four
CPU workers split columns; a four-row/eight-column tile reuses weight loads.
Activations and accumulation remain FP32. The cache is held by partial
callables, so original state-dict keys and tensor identities are unchanged.
Weights are immutable inference snapshots; recreate preparations and JIT
captures after changing them.

The first K-major layout made decode slower. Contiguous column tiles instead
measured 7.29× decode and 14.52× prefill speedups against the dense FP16 CPU
fixture at 2048 input/output channels, including thread startup. These are
primitive measurements. All twelve one-row/multirow checks of the actual
checkpoint's six matrices matched generic CPU outputs bit for bit.

The end-to-end `--cpu-static-gemm --cpu-gelu` candidate records Gemma3n at
116.56 prefill proxy tok/s and 8.26 decode tok/s. It uses three full warmups
and three measured 128-input/64-output requests at context 512; all six
sequences match the prior candidate. The prefill proxy is slightly above the
published RKLLM 113.83 tok/s, while decode remains below 9.91 tok/s. Output
quality and matched clock/workload qualification remain open.

[Model, primitive, regression and restoration evidence](../../../oNPU/docs/benchmark-gemma-host-kernels-verification-v4.json)
retains both the selected CPU projection path and the separate register probe.
