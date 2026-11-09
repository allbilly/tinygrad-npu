# Shared CPU dispatch for Spark

The checked head-gated erf-GELU graph can use the existing shared host pool
without prepared QKV fusion. It requires context 512, both projection phases
on, one CPU attention/erf activation/raw FFN/decode FFN/head gate per ordinary
TransformerBlock, and no static CPU projections. Existing outer guards still
reject recurrent, expert and MLA graphs and require the normalization and
prefill-pool coverage.

Every threaded helper in the saved Spark prefill source inventory is generated
by a shared-program factory: activation quantization, dequantization or RMS
normalization. The pool rewrite retains the original work partitions and
arithmetic. Existing attention and raw FFN helpers already use the same pool.
The caller runs on CPU4, workers on CPUs5–7, and workers park outside the
inference scope. Sampled/cached and API paths still need separate qualification.

Four raw-token backend processes, control/shared/shared/control, use identical
128-input/64-output/context-512 settings and stock adaptive governors. Their
prefill medians are 306.26/399.41/403.79/311.71 tok/s. Decode medians are
15.86/15.81/15.84/15.80 tok/s. Each process matches all 32 full-vocabulary
regression vectors exactly. These measurements qualify the dispatch change at
backend scope; independent original/RKLLM quality and maximum context remain
open. API qualification is tracked separately in oNPU.

The packing-alias experiment remains disabled for this model comparison. Its
separate native-FFN tests do not establish an additional stable model gain.

Raw evidence:
`/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/spark-shared-model-20261102-v1/`.
