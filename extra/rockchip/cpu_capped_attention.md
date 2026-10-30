# CPU capped-score attention

Gemma2 applies its score scale, then `cap * tanh(score / cap)`, then the
causal/sliding mask and softmax. The bounded CPU attention kernel now applies
that transform before computing the maximum and normalization. Scalar rows,
blocked prefill rows and grouped-query decode all use the same cap.

The default cap is zero and retains uncapped attention. The optional capped
path accepts finite caps between 0.125 and 128. Placement keeps sinks,
unsupported caps and incompatible packed/prepared-QKV paths on their previous
implementations. A block records the cap supported by its installed adapter;
an uncapped adapter or a stale cap cannot bypass the generic capped path.

Tests compare against independent float64 attention and the small Gemma2
PyTorch decoder. They cover cached prefill, decode, reset, saturation, grouped
heads, vector/block tails, symbolic rows, poisoned masked cache entries and
invalid caps. Model timings and comparable-quality RKLLM qualification remain
separate work.

After the board rebooted during verification, CPU checks were rerun with
enforced process-group resource limits: 6 GiB memory, no swap and a two-core
CPU quota, with reference libraries limited to one thread. Do not run a
12-worker reference suite alongside multi-GiB serving models or another
compiler/typecheck job. Reserve at least 8 GiB available RAM for that test
group. Keep verification logs on persistent storage.

A complete trained Gemma2-2B comparison used 128 input tokens, 64 output
tokens, context 512, three full warmups and three measured requests per
variant. The current control measured 42.34 prefill-proxy tok/s and 7.27
decode tok/s. The candidate, also enabling vector GELU and final-logit softcap,
measured 138.74 and 9.24 tok/s: 3.28× and 1.27× the current control rates.
The combined change does not isolate any one host kernel's contribution.

Actual execution used 26 CPU attention kernels in each phase, 26 GELU kernels
during prefill and one final softcap kernel in each phase. Candidate sequences
were stable across all six requests, but first differed from control at the
14th generated token; 29 of 64 positions differed. Comparable output quality
is therefore unqualified. CPU/memory clocks varied, and later candidate NPU
samples ended at 800 MHz while control samples ended at 1 GHz. These rates
remain below the published RKLLM 213.90/10.37 tok/s reference; no fixed-clock
or comparable-quality RKLLM win is claimed.

Both original workers were restored and answered fresh requests. The full
candidate options have not been qualified through the serving API.
[Raw reports, validation, output comparison and restoration evidence](../../../oNPU/docs/benchmark-gemma2-host-verification-v1.json).

The trained fixed-prefix follow-up separates forced continuation after EOS
from natural-text regression. Both benchmark variants first emit EOS at output
8, before their free-sequence divergence at output 14. The diagnostic reproduces
all 64 control predictions and the candidate's first differing prediction.

On 128 natural-text targets from 16 held-out WikiText-2 prefixes, sampled
conditional perplexity is 28.6413 for control and 27.6732 for candidate. The
0.96620 ratio passes the declared local 1.02 host-regression limit. Both paths
still use INT8 NPU projections. This is not an FP16 teacher comparison, full
corpus evaluation, or RKLLM quality qualification.

Outputs are not numerically equivalent: natural-target argmax agreement is
90.625%, mean probability total variation is 0.09149, and maximum absolute
logit difference is 6.0789. The small sample's lower perplexity does not justify
claiming harmless rounding or broad quality parity.

A separate direct-logit profile covers one 128-token prefill and 63 fixed-prefix
decode steps. Candidate NPU matmuls account for 31.9% of native prefill time and
89.7% of decode time. Prefill includes 157 ms in vector GELU, 81 ms in
quantization, 80 ms in attention, 57 ms in a projection concatenation and 55 ms
in dequantization. These are diagnostic component times, not a new throughput
benchmark. They point to CPU fusion for prefill and NPU weight/dispatch work
for decode. [Full-vector comparisons, samples, profiles and restored-worker
checks](../../../oNPU/docs/benchmark-gemma2-host-quality-v1.json).
