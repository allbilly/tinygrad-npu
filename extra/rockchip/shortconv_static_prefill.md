# Optional static first chunks for short convolution

`ONPU_SHORTCONV_STATIC_PREFILL=1` extends the existing
`ONPU_STATIC_PREFILL=1` CPU-token route to models containing only ordinary
attention and short-convolution blocks. Both flags default to zero. Serial
recurrent scans, MLA and other block types retain the dynamic route.

The route still requires greedy generation, position zero and a complete first
chunk. Partial chunks, cached continuations, sampling and scalar decode keep
the existing route. The model retains `has_recurrent_block` and its conservative
prefix reuse, failure and completed-state publication rules.

For a literal zero position, short convolution constructs zero history directly.
The dynamic path previously created a bound position variable even for this
constant case. The new route retains the convolution store and its dependency
on the output; it does not discard or reset the continuation state afterward.

CPU tests compare complete output and history bytes for FP32/FP16, kernel sizes
three/four, changed initial chunks and single-token continuations. A real small
short-convolution Transformer exercises full, partial, cached and early-closed
generation against the original dynamic path.

The November 2 private LFM2.5-VL 450M screen uses exact API input IDs, context
512, 128 uncached inputs, 64 outputs and 200 ms between requests. Its balanced
screen improves median TTFT from 134.39 to 125.83 ms, with three of four blocks
positive and unchanged generated IDs. This is a backend screen, not an API
speed claim or RKLLM parity. Both the prototype and actual port preserve the
saved 32 full-vocabulary vectors and all 1,024 natural-target vector hashes.
The exact LFM/context-512 app profile now enables this route after the balanced API comparison (9.51% lower median TTFT) and two fresh default workers. See `~/oNPU/docs/benchmark-lfm450-static-default-20261102.json`; broader quality and model maximum remain open.

Raw cohorts under `~/vlm-bench/baseline-20261021/main-task-20261023/`:

- `lfm450-static-screen-20261102-v1`
- `lfm450-static-quality-20261102-v1`
- `lfm450-static-port-quality-20261102-v1`

The separate balanced idle diagnostic finds 82.26 ms without idle versus
130.29 ms after 200 ms idle. That warm-backend result cannot replace the declared
API workload. Independent RKLLM quality, model maximum and image processing
remain unqualified. No board governor, voltage, IRQ or idle setting is changed.
