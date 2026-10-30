# Native CPU RWKV7 recurrence

RWKV7 previously limited generation prefill to one token to avoid unrolling its
recurrence into a large graph. The optional CPU recurrent-scan setting now
installs a native FP32 RWKV7 scan before chunk selection. Batching is permitted
only when every RWKV7 block has a scan; capacity is bounded to 256 tokens.
The generic fallback and OpenCL path retain their previous chunk policy.

The [RWKV7 reference equations](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo_fast.py)
apply the rank-one correction to the old state, before decay. The new kernel
implements that order directly; it does not reuse GatedDeltaNet's equations.
It transposes each head's state into local column scratch, vectorizes across
four independent value rows, advances only valid token rows, then commits the
FP32 state. Position zero resets the state, including poisoned initial values.
Inactive output rows are zero and padded inputs are never read.

The scan accepts contiguous CPU FP32 operands in batch/token/head/channel
layout and a whole matrix-state buffer. Static and symbolic row/position
bounds are checked. A custom kernel marks its state as both input and output,
so cached state updates survive function boundaries and native replay.
Existing parked CPU workers can execute independent heads. Dispatch errors
trap rather than retry a partially advanced state.

Use `--cpu-recurrent-scan` with the CPU/NPU benchmark. The feature remains
optional; no checkpoint whitelist or serving default is expanded by this work.

The bounded CPU suite passed 20 tests with one OpenCL hardware skip, including:

- Independent float64 matrix equations, odd widths, reset and continuation.
- Poisoned padding, dynamic rows, changed-input JIT and state rebinding.
- Bitwise agreement between whole/chunked scans and pooled/serial execution.
- Complete small-model PyTorch logits, cached chunks, reset and generation.
- Hook installation, full-block chunk gating and existing GatedDeltaNet policy.

Mypy and Ruff passed. The reference suite ran with a 6 GiB limit, no swap,
two-core CPU quota and one-thread reference libraries; only the normal GPU
session was unloaded, then restored and checked with fresh requests.

The trained 1.5B comparison uses 24 blocks, 32 heads of width 64, a 128-token
prompt, 64 outputs, context 512, three full warmups and three measured requests.
Both variants use the same deferred/mapped CPU weights and INT8 NPU projections.

| Metric | Generic one-token scan | Native batched scan |
|---|---:|---:|
| Prefill proxy tok/s | 9.12 | 67.20 |
| Decode tok/s | 10.25 | 12.73 |
| Prompt NPU submissions | 46,208 | 409 |
| Peak RSS MiB | 3,689.57 | 3,775.30 |

The candidate executes 24 native scan kernels in each captured phase. Every
variant is stable across its six runs. Control matches the previous candidate;
native output differs at two positions, outputs 23 and 57 (token 629 becomes
630). The primitive tests do not establish full-model quality equivalence.
Fixed-prefix/held-out quality, serving API and model-maximum context remain
unqualified. Adaptive clocks vary. The published RKLLM 201.78/13.82 tok/s
reference is still faster in both phases.

Both original sessions were restored with their previous options and selection
and answered fresh requests. [Reports, CPU checks, output differences and
restoration evidence](../../../oNPU/docs/benchmark-rwkv7-scan-verification-v1.json).

The later ordered-normalization fix supersedes the v1 quality result. The
generic fused reduction computes `sum(state * raw_key * weight) / norm`;
materializing a normalized key first changes FP32 rounding. The native ordered
path preserves that reduction and the preformed correction key. Independent
CPU checks include formed half weights, symbolic rows, poisoned padding and
pooled execution. The reference suite passes 23 tests with one OpenCL skip;
mypy and Ruff pass.

In the trained follow-up, all 192 fixed-prefix full logit vectors now match the
generic INT8 path exactly, including 128 held-out targets. Sampled perplexity
is 8.8876 for both, passing the unchanged local 2% gate. Corrected-scan timing
control/candidate use the same scan, differing only in opt-in small NPU depths:
66.89/66.75 prefill proxy tok/s and 12.65/13.10 decode tok/s. All twelve timing
sequences match. This does not qualify FP16/RKLLM output quality, maximum context
or serving API; both phases still trail the published RKLLM reference.
[Current reports, quality comparison and restored serving](../../../oNPU/docs/benchmark-rwkv7-small-depth-verification-v2.json).

`ONPU_RWKV_DECODE_SPIN=1` additionally selects the pool's spinning callback for
static single-row scan kernels. Larger-capacity prefill kernels retain parked
dispatch; the outer inference context parks the workers on exit. The default
remains off. The updated CPU suite passes 27 tests with one OpenCL skip,
including changed-input captured execution, ordered output/state bit agreement,
callback behavior, affinity restoration and worker parking. Mypy and Ruff pass.

Matched full-model control/candidate measure 13.05/14.51 decode tok/s, about
11.2% faster, and 65.77/65.20 prefill proxy tok/s. Captured scan sources confirm
the intended callback in all 24 blocks of each phase. All twelve timing
sequences match, and all 192 logit dumps match both the paired control and the
previous local INT8 cohort byte for byte. Separate profiling records 663.6 versus
230.7 ms for 1,512 decode scan calls; this is diagnostic kernel time, not an
additional throughput measurement. The CPU-only sleep-gap proxy used roughly
four times as much CPU time with spinning; full-model power was not measured.

The reported decode rate exceeds RKLLM's published 13.82 tok/s; prefill remains
below 201.78. This does not qualify matched RKLLM quality or clocks, broader
contexts, or the serving API. Both original serving sessions were restored and
answered fresh requests. [Full reports and restoration](../../../oNPU/docs/benchmark-rwkv7-pool-spin-verification-v1.json).
