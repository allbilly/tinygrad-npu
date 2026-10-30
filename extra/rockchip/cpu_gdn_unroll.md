# CPU recurrent scan value lanes

The CPU GatedDeltaNet profile recorded 604.72 ms across 24 scan calls in
Qwen3.5-4B prefill. The scan previously processed four independent value lanes
per serial key loop. For value dimensions divisible by 16, it now processes
four such vectors together, sharing key/query loads and using adjacent state
data. Each lane retains its key reduction order and multiply/add grouping.
Other value dimensions keep the existing partial-vector loop. Pool dispatch,
reset, cached continuation, history layout and padding are unchanged.

The CPU-only comparison used identical inputs and the production Clang compiler
settings, three warmups, then 40 alternating ABBA/BAAB blocks. Each variant has
80 timing samples. The first two rows use 16 heads and 128 key/value channels.

| Fixture | Control ms | Candidate ms | Median paired speedup |
|---|---:|---:|---:|
| 128-token prefill | 50.260 | 18.364 | 2.735× |
| Single-token decode | 0.945 | 0.691 | 1.369× |
| Per-key decay, K=17/V=32, seven tokens | 0.060 | 0.046 | 1.314× |
| Scalar decay, K=17/V=16, seven tokens | 0.041 | 0.034 | 1.203× |
| Fallback K=17/V=9, seven tokens | 0.039 | 0.040 | 0.974× |

All 40 paired blocks favored the candidate for each optimized fixture. The
small fallback fixture regressed by approximately one microsecond. These are
standalone scan timings; whole-model gains and RKLLM parity require separate
measurement.

Output, final state and available history matched the original C kernel bit for
bit for reset and nonzero positions. The existing scan, pool and model suites,
including added scalar-reference cases at V=16/32, passed 25 tests and two
subtests. Mypy and Ruff passed. This does not qualify RKLLM output quality.

[Raw samples and source identities](cpu_gdn_unroll_results.json) retain the
measured standalone control/candidate sources. The production source adds only
a comment and indentation to the measured candidate.

The subsequent [three-model benchmark](../../../oNPU/docs/benchmark-cpu-gdn-models-v1-verification.json)
records Qwen3.5 0.8B/2B/4B at 145.12/114.04/47.80 prefill proxy tok/s and
21.02/11.13/5.29 decode tok/s. These complete configurations also use mapped
CPU loading and current attention hooks; comparisons with earlier runs do not
isolate the unroll. All 18 sequences match their prior candidates, but all three
still trail the published RKLLM rates and output quality remains unqualified.
