# CPU head-gate denominator materialization

`--cpu-head-gate` opts into the `configure(..., cpu_head_gate=True)` hook for models with head-gated attention. The default remains off. It computes the small sigmoid denominator before broadcasting division across each head's channels.

Both operands must be CPU FP32 values backed by a buffer or explicit kernel output. Unevaluated CPU expressions retain the original graph. A CPU matmul regression exposed a one-ULP difference when a new boundary changed reduction fusion; the guard and two lazy-expression regression tests preserve that original path. Half inputs also retain the original computation.

A synthetic CPU fixture already hoisted the denominator outside its inner loop and was slower with materialization (0.447 to 0.504 ms). That fixture did not represent Spark's generated flattened kernel. The actual captured source recomputes the denominator for every four-channel group. A linked test using that source improves 1.005 to 0.262 ms, with finite bits and nonfinite classes checked on changed inputs. Both results are retained.

Separate native profiles distinguish kernels by source hash. Head-gating prefill costs are 43.69 ms before and 7.84 ms after; CPU ELF objects and disassembly are retained. Other stage timings also vary between these diagnostic processes and should not be attributed to this change. The actual decoder retains identical head-gate source and operation labels.

Spark-X2.5 1.7B's actual backend control/candidate medians are 274.95 / 297.64 prefill tok/s and 15.76 / 15.48 decode tok/s. All 32 full-vocabulary vectors and 24 fixed-count sequences match across the prototype and actual cohorts. The fresh decode candidate median trails the published 15.61 reference. Prefill also trails 331.41; no complete speed win, independent quality equivalence, stock API or maximum-context qualification is claimed.

The focused CPU suite passes 50 tests, with mypy across 289 source files and whole-tree Ruff passing. Evidence and individual trials: `/home/orangepi/oNPU/docs/benchmark-spark-head-gate-20261031.json`.
