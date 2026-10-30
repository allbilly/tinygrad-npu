# CPU latent attention dots

`configure(..., cpu_mla_attention=True)` and the benchmark's
`--cpu-mla-attention` select bounded FP32 query/key and weighted-value kernels.
The option defaults off and works with CPU-only benchmarks too. Generic Tensor
softmax remains between the two kernels.

The QK kernel transposes the valid shared latent cache, pads its stride to
`round_up(context,16)+4`, and computes sixteen independent key outputs while
advancing each output's reduction in order. The extra stride avoids a power-of-two
cache layout. The PV kernel handles four value channels and up to four query rows
at once. Existing workers split independent heads, park between prefill kernels,
and remain awake within a decode call; the outer context parks them on exit.

The helper accepts batch-one CPU FP32 operands, up to 128 heads, query capacity
128, context 1024, feature/value widths divisible by four up to 512, and at most
1 MiB of packed-cache scratch. Bounds can therefore reject some combinations
below those individual maxima. Indexer-based attention retains its generic path.
Small widths and literal static reductions retain generic matmuls because their
fully unrolled contraction order can differ. Model cache writes remain explicit
dependencies of subsequent reads. Helpers are static C functions with one global
entry wrapper per program.

| MiniCPM3 4B, 128 input / 64 output / context 512 | Weights-only control | New attention |
|---|---:|---:|
| Prefill proxy tok/s | 5.63 | 10.62 |
| Decode tok/s | 2.89 | 3.86 |
| New QK/PV kernels in each captured phase | 0 / 0 | 62 / 62 |

Both arms use the same 124 prepared FP16 tables and packed NPU weights. Three
full warmups and three measured requests per arm give an 88.7% prefill and 33.7%
decode improvement over this control. All twelve 64-token sequences match the
previous reference. These remain below the published RKLLM 90.88/5.94 tok/s.

The final CPU suite passed 54 tests with three OpenCL hardware skips, including
dynamic changed inputs, poisoned unused cache, odd context 33, pooled/serial
agreement, state-dict behavior and captured model logits/cache decode at the
trained attention dimensions. Mypy passed 271 files and Ruff passed. Earlier
symbolic readback and static-contraction failures are retained in the evidence;
the exact checks were kept and the affected reductions use the generic fallback.

CPU-only paired layout probes precede the model comparison. Their setup and
readback scope is recorded; they do not establish a model or RKLLM win.
Full-model trained logits, held-out perplexity, maximum context and comparable
RKLLM quality remain unqualified. Both original serving models were restored
and answered fresh requests.
[Model reports, source evidence, CPU checks and restoration](../../../oNPU/docs/benchmark-minicpm3-mla-attention-verification-v1.json).

## Causal work reduction

The CPU kernels now avoid query/key outputs that are guaranteed to be masked
and skip zero-probability value tails. The shortcut is limited to bounded finite
queries/caches and a safe scale; non-finite or huge inputs retain the original
calculation. Value tails are skipped only after their probabilities are checked
as zero. Negative-zero and non-finite accumulators replay the original tail.
Single-row decode avoids the new safety scan.

An isolated comparison with the frozen original C passes 14 bitwise tests,
including noncausal probabilities, NaN/Inf, overflow, underflow and signed zero.
Integrated tests pass 25 cases with two hardware-specific skips; mypy and Ruff
pass. At forty heads and 128 query rows, complete serial operators improve
QK from 33.28 to 20.08 ms and PV from 33.41 to 21.05 ms, winning every one of
eight interleaved paired blocks. These are operator timings, not RKLLM results.
The full trained pair measures 10.69/11.68 prefill proxy tok/s and
4.011430/4.011272 decode tok/s for control/candidate. All timing sequences
match the earlier reference, and all 32 fixed-teacher full-vocabulary logit
vectors match bit for bit: RMS/TV are zero and the perplexity ratio is 1.0.
Prefill improves 9.2%; the 0.004% decode difference establishes no decode gain.
The existing preferred row is retained by the predeclared both-phase rule.
This exact local regression does not qualify held-out or RKLLM quality.
Both original serving models were restored and answered fresh requests.
`ONPU_MLA_CAUSAL_WORK=0` selects the original work for the paired run; the
default is 1, and the benchmark records the selected mode.
[Trained results, exact-vector comparison and restored serving](../../../oNPU/docs/benchmark-minicpm3-cpu-mla-causal-verification-v1.json).
