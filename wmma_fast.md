# Why `Ops.WMMA` alone does not make LLM inference fast

Our RK3588 inference work accelerates the complete token path: matrix kernels,
weight and activation layouts, attention and recurrent state, host scheduling,
cache maintenance, output projection and sampling. Mapping matrix multiplication
to a fast instruction is one part of that work. It does not remove the other
costs, and it does not guarantee that the matrix hardware is well utilized.

The current oNPU design is OpenCL GPU first. CPU/NPU execution is an optional
placement choice when it improves the relevant phase; separate workers can also
run a GPU model and a smaller CPU/NPU model independently. This document describes
the original 2026-10-23 work and the later qualified cohorts identified below. The current cross-model
benchmark has **zero qualified wins over RKLLM in both phases at comparable
output quality**. The improvements below are real local improvements, not a claim
that this objective has already been reached.

## Latest scoped results

The [causal CPU MLA work reduction](extra/rockchip/cpu_mla_attention.md) improves
matched trained prefill from 10.69 to 11.68 proxy tok/s while preserving all 32
fixed-teacher full-vocabulary logit vectors bit for bit. Decode remains about
4.01 tok/s. Complete serial QK/PV operators improve 1.66×/1.59× after 14 frozen-C
bitwise tests and 25 integrated tests. This is a validated local prefill gain;
the existing preferred row remains because both phase medians did not improve.
Held-out and RKLLM quality qualification remain open.

The [grouped NPU projection API](extra/rockchip/grouped_mla_npu.md)
uses one submission for forty distinct head weights, distributing whole heads
across three register-command chains. Complete synthetic query/value projections
are 1.69–1.89× faster than CPU at one row and 4.56–5.22× at 128 rows, including
conversion, packing, transfers and native replay. Outputs match separate NPU
calls bit for bit. Symbolic-token integration now exists, but trained mixed
FP16 execution fails a finite-activation guard. The subsequent grouped INT8
model candidate completes at 19.84 prefill proxy/4.81 decode tok/s versus
10.62/3.97 for its matched control. It fails the unchanged local logit RMS and
probability-distribution limits, so its faster timings are rejected for
preferred-row promotion. CPU attention remains; these results establish neither
comparable model quality nor an RKLLM win.

The subsequent three-product residual INT8 experiment improves its matched
control from 10.64/3.98 to 16.20/4.41 prefill-proxy/decode tok/s. Independent
NPU operator and small-model tests pass, but the trained run still fails all
three local quality limits: relative logit RMS 0.159280, mean probability total
variation 0.063834 and perplexity ratio 1.028491. It is rejected for preferred-row
promotion. The complete model error remains close to the single-product result
despite much smaller sampled projection error; the full-path cause needs
isolation. [Residual experiment and failed gates](../oNPU/docs/benchmark-minicpm3-mla-int8-residual-verification-v2.json).

An invariant trace then checks all 124 projections at the final fixed-teacher
decode position: actual outputs match the independent scaled integer recipe
bit for bit. Source weights, packed codes and scales match independent hashes.
This does not check prefill or earlier cache history. A query-only follow-up
retains original CPU value projections and measures 12.69/4.27 tok/s versus
10.74/4.03 for its control, but still fails every local numerical gate. It is
also rejected for preferred-row promotion. Neither register correctness for
one position nor faster matrix paths establish whole-model quality.
[Actual traces](../oNPU/docs/benchmark-minicpm3-projection-trace-verification-v1.json)
and [query-only failed gates](../oNPU/docs/benchmark-minicpm3-mla-query-only-verification-v1.json).

The [CPU MLA attention kernels](extra/rockchip/cpu_mla_attention.md) now improve
MiniCPM3 from 5.63 to 10.62 prefill proxy tok/s and from 2.89 to 3.86 decode tok/s.
Both runs use prepared per-head FP16 tables; only the candidate adds vectorized,
pooled FP32 QK/PV dots. Generic softmax remains, and all twelve generated sequences
match. Captured phases contain 62 QK and 62 PV kernels per candidate call. The
54-test CPU suite checks exact dynamic projections, cached model logits and pool
behavior. Full-model quality qualification and RKLLM's 90.88/5.94 tok/s target
remain open.

The earlier [MLA/MoE profiles](extra/rockchip/mla_moe_profile.md) identify CPU work
behind the largest remaining prefill gaps. NPU kernels account for only 4.6% of
MiniCPM3's recorded native prefill time and 0.3% of DeepSeekOCR's. MiniCPM3's
generic per-head projections repeatedly decode Q8 weights inside dot products;
latent attention adds substantial CPU cost. DeepSeekOCR spends 98.8% of native
prefill time in two routed expert-matmul kernels. These are measured component
attributions with exact generated sources, not new throughput rows or wins.

The subsequent [grouped CPU expert path](extra/rockchip/cpu_experts.md) measures
95.11 prefill proxy tok/s and 23.29 decode tok/s for DeepSeekOCR, versus the
earlier realized-weight candidate's 3.43/3.15. All 32 checked full-vocabulary
logit vectors match the new matched control bit for bit. Both new arms defer
weight realization; that control decodes Q8 inside expert loops, so its
1.56/1.47 rates are distinct from the earlier baseline. The grouped path prepares
FP16 tables and executes all 33 expert projections per captured phase, while
dense NPU packed weights remain unchanged. It still trails RKLLM's
182.54/31.60, uses 8,856 MiB peak RSS and has no held-out/RKLLM quality
qualification. The earlier profile percentages do not describe this new path.
Its subsequent component profile still attributes 81.17% of recorded prefill
kernel time and 50.42% of decode time to CPU experts; NPU execution alone is
4.46% and 39.24%. Copies are separate. Larger CPU row tiles subsequently passed
exact model-logit checks but lost in complete-model timing: 96.91/23.83 versus
98.84/24.71 tok/s for their control. The experimental runtime option was removed;
the four-row implementation and preferred benchmark remain unchanged.
[Current profile](../oNPU/docs/benchmark-deepseek-cpu-experts-profile-verification-v1.json).

The next [selected-expert NPU API](extra/rockchip/routed_experts.md) changes weight
addresses from guarded runtime routing IDs while retaining the atlas and task
chains. Exact integer and ordinary/native replay tests pass. Removing padding
at depths 896/1280 gives 1.081x/1.134x paired complete decode-projection speedups,
with bitwise outputs against the padded path. The option remains off by default.
A balanced prefill probe measures a 5.090x paired NPU speedup with twelve rows
per expert, but only 1.225x when 128 rows are reserved/computed for each expert.
Route-map construction is outside these timings. This identifies dynamic row
counts and padded work as implementation targets; it does not certify model
quality or an RKLLM win. Application placement and preferred model rows are
unchanged. [Operator samples](../oNPU/docs/benchmark-expert-depths-prefill-verification-v1.json).

The subsequent [dynamic-row implementation](extra/rockchip/bucketed_experts.md)
passes exact output and task-count checks while rebuilding active command
chains in native replay. It helps skewed routing but does not consistently beat
fixed-capacity NPU execution: balanced/mixed cases remain slower. Full reserved
row buffers still flow through gather, quantization and copies. Compact input
and output storage is therefore the next target before model integration.

The [compact-storage follow-up](extra/rockchip/bucketed_experts.md#compact-row-storage)
now passes exact output and task-count tests with 768 routed rows instead of
8,192 reserved slots. It measures 10.77/6.54/11.03 ms for balanced/skewed/mixed
routing, with paired 3.575x/7.491x/3.660x speedups over fixed NPU storage.
Route-map construction is excluded; model integration and trained quality remain
open. This shows why reducing NPU matrix work alone was insufficient: the
compact format also reduces host gather, quantization and transfer volume.
[Complete operator proof](../oNPU/docs/benchmark-compact-experts-verification-v1.json).

The first [MLA preparation experiment](extra/rockchip/cpu_mla_weights.md) now
improves MiniCPM3 from 3.94 to 5.63 prefill proxy tok/s and from 2.19 to 2.89
decode tok/s. Preparing its 124 per-head FP16 tables removes Q8-reading projection
programs from each captured phase while preserving all twelve output sequences.
The CPU suite also checks decoded weight bits and captured projection outputs.
The original stale byte-count validation failure and its correction are preserved
with the resumed measurement. Full-model logits/perplexity, comparable RKLLM
quality and the remaining attention/expert optimizations are still open.

GDB first captured the first twelve submissions per phase of a real
Qwen3-0.6B RKLLM W8A8 workload: 27 tasks and 2,799 register words. It led to the qualified
[decode buffer allocation](extra/rockchip/int8_decode_cbuf.md) and optional
[prefill layout](extra/rockchip/int8_sdk_prefill.md). Several prefill layouts
lose once CPU packing and conversion are included, despite faster NPU calls.
A subsequent complete bounded 128-input/2-output pass captures 2,657 submissions,
3,751 tasks and 367,308 register words, including active per-core task ranges
and matrix-address bindings. Prefill includes FP16-family matrices and DPU
conversions; decode uses INT8-family matrices without observed FP16-family
matrices. Register-pattern families do not by themselves establish operator
roles. This is scheduling evidence, not debugger throughput or a matched-quality
benchmark. [Full trace and verification](../oNPU/docs/benchmark-rkllm-full-register-capture-v2.json).

A separate controlled SDK capture qualified smaller physical INT8 depths.
[Complete projections](extra/rockchip/int8_small_depth.md) improve by
1.23–1.88x for single-row cases, but the matched RWKV7 model improves decode
only from 12.65 to 13.10 tok/s; prefill remains about 66.8 tok/s. Correcting the
native recurrence's normalized-key reduction order also restores exact agreement
on all 192 fixed-prefix logit vectors against existing INT8 inference. This is
local regression coverage, not FP16 or RKLLM quality qualification. The latest
profile attributes 12.2% of native prefill time and 72.2% of decode time to NPU
kernels. CPU recurrence and other CPU kernels therefore remain major prefill
targets. [Model measurements, raw quality and restored serving](../oNPU/docs/benchmark-rwkv7-small-depth-verification-v2.json).

The subsequent [RWKV7 worker-dispatch comparison](../oNPU/docs/benchmark-rwkv7-pool-spin-verification-v1.json)
improves decode from 13.05 to 14.51 tok/s by keeping CPU workers awake between
single-row scans and parking them at inference-call exit. All twelve timing
sequences and all 192 full logit vectors match the checked local INT8 cohort.
Prefill remains about 65 tok/s. The decode number exceeds the published RKLLM
13.82 tok/s reference; prefill and matched RKLLM output quality remain open.
The subsequent API check below covers the pinned 512-context serving path.
This improvement changes CPU dispatch rather than NPU registers.
It remains opt-in because the checked scope is limited and spinning consumes
more CPU time in the separate proxy experiment.

The [fresh API qualification](../oNPU/docs/benchmark-rwkv7-api-verification-v1.json)
selects the measured RWKV7 configuration by checkpoint identity with Auto NPU
prefill and a 128-token chunk limit. Eighteen uncached greedy chat requests
across counting, story and code workloads reproduce the independent backend
reference, usage, stopping behavior and NPU submission counts. Served decode
medians are 14.44, 14.30 and 14.03 tok/s; prefill remains about 65–66 tok/s.
These first-visible-text API metrics are separate from the canonical backend
row. GPU-first defaults remain in place. Broader context and matched RKLLM
quality remain unqualified; the existing normal server was preserved and both
original models answered fresh requests after the test.

The optional raw INT8 GELU feed-forward path now fuses projection scaling,
activation and independent down-group quantization. Complete Gemma2 requests
measure 183.86/9.93 prefill-proxy/decode tok/s against the current unfused
control's 140.37/9.93. All twelve 64-token sequences match the prior candidate.
The graph executes 26 fused pack calls in prefill and none in decode. This
31.0% prefill gain preserves CPU payload bytes and native FFN outputs; the
[implementation and evidence](extra/rockchip/fused_gelu_ffn.md) describe its
bounds and fallback. The option remains off by default pending serving API
qualification. RKLLM remains faster at 213.90/10.37 tok/s.

Gemma2 now avoids padding 2,304 input channels to 2,560. Full 128/64/512 runs
measure 140.29/9.79 prefill-proxy/decode tok/s against the identically configured
padded control's 136.94/9.30. All twelve sequences match the prior host-kernel
candidate, and packed weights save 205.5 MiB. Independent integer, FP32 and
native-replay boundary tests passed. The [natural-depth proof](extra/rockchip/int8_natural2304.md)
records the default promotion; broader quality, fixed-clock RKLLM parity and
all-options API qualification remain open. The published RKLLM rates are
still higher at 213.90/10.37 tok/s.

Gemma2's capped-score attention now uses the bounded CPU kernel, preserving
the scale, softcap, mask and softmax order. With vector GELU and final-logit
softcap, complete 128/64/512 runs measured 138.74/9.24 prefill-proxy/decode
tok/s, against the current control's 42.34/7.27. Both variants used three full
warmups and three measured requests. Candidate outputs are stable but differ
from control after the 13-token prefix, and CPU/memory/NPU clocks vary. The
[full-model evidence](../oNPU/docs/benchmark-gemma2-host-verification-v1.json)
does not qualify comparable quality or an RKLLM win; published reference rates
remain higher at 213.90/10.37 tok/s.

The [trained fixed-prefix follow-up](../oNPU/docs/benchmark-gemma2-host-quality-v1.json)
finds that the free-generation divergence occurs after EOS. Its 128 natural-text
targets pass the local host-regression limit with sampled perplexity ratio
0.96620 versus the existing INT8 path, but argmax agreement is 90.625% and
distributions differ substantially. Broader quality and RKLLM parity remain
unqualified. The separate profile attributes only 31.9% of native prefill time
to NPU matmuls, versus 89.7% of decode. CPU GELU, quantization, attention and
conversion therefore remain important prefill targets; register tuning has a
different practical benefit in decode.

The opt-in SDK-derived INT8 prefill layout now includes its matching input
packing and output conversion. A complete projection comparison found only
one of the original three shapes faster: M=64, N=3072, K=2048 improved 6.9%.
The other two were near parity, and six expanded widths were correct but
slower in every paired block. All 126 numerical outputs matched the independent
reference bit for bit. These results include packing, transfers, conversion
and native dispatch, which earlier NPU-call-only timings excluded. The
[pipeline evidence](extra/rockchip/int8_sdk_prefill.md) explains why the
register gains alone do not establish faster model inference. The option
remains off by default; no model timing row changes from this experiment.
The v2 four-row CPU conversion repeat preserves 140 bitwise outputs across ten
cases. The 64/3072/2048 case improves 9.7% against its current control, but wider
dispatch and splitting a 128-row projection into two SDK calls remain slower.
The [repeat and diagnostic profiles](extra/rockchip/int8_sdk_prefill_v2_results.json)
support retaining the narrow experimental policy and returning to full-model
CPU bottlenecks.

Prepared CPU AltUp projections now unpack and tile their FP16 weights once,
retaining FP32 activations and accumulation. A four-row/eight-column CPU tile
shares weight loads, and four workers split output columns. A separate vector
GELU-tanh kernel accelerates large prefill activations; measured slower decode
and small side-input shapes retain their original path. These optional paths
are selected by `--cpu-static-gemm` and `--cpu-gelu`.

The newer `--cpu-softcap` option vectorizes the final CPU logit transform while
preserving the generic FP32 results and greedy ties. A separately qualified
decode buffer allocation now also covers total N=1536, K=4096 on three cores.

| Model | Prefill proxy tok/s | Decode tok/s |
|---|---:|---:|
| Gemma4 E2B | 146.53 | 9.58 |
| Gemma3n 2B | 117.27 | 8.57 |

Both rows use three full warmups and three measured 128-input/64-output
requests at context 512; all twelve sequences match their recorded prior
candidates. Gemma3n's prefill proxy exceeds the published 113.83 tok/s reference
by about 3.0%; its decode remains below 9.91 tok/s. Gemma4 still trails in both
phases. Comparable quality and matched clock/workload qualification remain
open. [The v5 evidence](../oNPU/docs/benchmark-gemma-decode-verification-v5.json)
includes the signed-integer native replay check, CPU regressions and fresh replies
from both restored normal workers. Compared with v4, Gemma4 decode rose 5.9%
while its prefill proxy fell 0.28%; Gemma3n decode rose 3.8% and prefill rose
0.61%. Historical comparisons do not isolate individual changes or clock drift.
The [preceding v4 evidence](../oNPU/docs/benchmark-gemma-host-kernels-verification-v4.json)
contains the learned-weight checks and the 8.7% narrow CBUF primitive gain.
Other tested CBUF shapes showed no consistent benefit and retain their recipes.

Gemma's shared-KV forward method now consumes its CPU attention hooks, including
unit score scale, 512-channel heads and correct sliding masks. Profiling then
exposed a larger decode cost: precompiled block calls made unused original CPU
weights contiguous, unpacking whole Q8 matrices on every token. Pure unused
arguments are now pruned with simultaneous parameter renaming; effects, custom
gradients and device inference are preserved. A token-changing early prototype
was rejected, and same-shaped-slot and placement regressions cover its cause.

| Model | Previous prefill proxy tok/s | New | Previous decode tok/s | New |
|---|---:|---:|---:|---:|
| Gemma4 E2B | 19.72 | 89.59 | 0.35 | 9.03 |
| Gemma3n 2B | 11.16 | 18.47 | 0.33 | 6.20 |

Each new row uses three full warmups and three measured 128-input/64-output
requests at context 512; all twelve sequences match the recorded prior
candidates. Both phases still trail the published RKLLM rates, and comparable
quality remains unqualified. The
[model and rejection evidence](../oNPU/docs/benchmark-gemma-pruned-verification-v3.json),
[attention kernels](extra/rockchip/gemma_cpu_attention.md) and
[argument fix](extra/rockchip/unused_function_inputs.md) record their scope.
The remaining Gemma3n prefill profile is dominated by six CPU AltUp projections;
Gemma4's generic GELU path is its largest remaining CPU prefill cost.

The CPU recurrent scan now processes 16 independent value lanes per key loop
when the value dimension permits it, retaining each lane's arithmetic order.
An isolated paired comparison measured 2.735× speedup for a 128-token fixture
and 1.369× for single-token decode, with bitwise output/state equivalence.
The [scan evidence](extra/rockchip/cpu_gdn_unroll.md) separates these primitive
results from whole-model timings and records the small fallback regression.

The subsequent complete Qwen3.5 0.8B/2B/4B candidates record 145.12/114.04/47.80
prefill proxy tok/s and 21.02/11.13/5.29 decode tok/s. Each uses three full
warmups and three measured 128-input/64-output requests at context 512; all 18
sequences match their recorded prior candidates. These configurations also use
mapped CPU loading and current attention hooks, so differences from historical
runs do not isolate the scan change. All three remain below the recorded RKLLM
rates, with comparable quality unqualified. The
[full-model evidence](../oNPU/docs/benchmark-cpu-gdn-models-v1-verification.json)
records source cohorts, memory, terminal process cleanup and fresh replies from
both restored normal workers.

A measured Qwen3.5-4B CPU/NPU profile explains why matrix instructions alone
cannot solve prefill. INT8 NPU submissions account for **437.01 of 3,942.81 ms
(11.08%)** of native prefill call time. CPU scan and generic attention/reduction
work dominate. Decode has a different balance: NPU INT8 calls account for
76.27% of profiled native time. This diagnostic uses one native prefill after
three warmups, 128 input tokens and eight generated tokens at context 512. It
does not replace the full 64-token benchmark.

The CPU-attention installer excluded both output-gated and head-gated blocks.
The shared attention kernel can compute the same attention while each gate stays
in its original position afterward; prepared QKV remains excluded for gates. A
window covering the entire allocated context is also equivalent to the full
causal mask. Smaller windows, sinks, softcaps and unsupported geometries retain
their fallbacks. These are operator/feature rules, not model-name dispatch.

| Model | Previous prefill proxy tok/s | New | Previous decode tok/s | New |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 32.20 | 41.74 | 4.84 | 5.23 |
| Spark-X2.5-1.7B | 53.17 | 159.37 | 9.10 | 13.10 |

Each row uses three full warmups and three measured 128-input/64-output requests
at context 512. Sequences are stable within the new candidate but differ from
the previous path. Output quality remains unqualified, and both rows still lose
to the recorded RKLLM rates. CPU equivalence checks include cache history,
gate combinations, both softmax modes, the full-context window boundary and an
independent float64 reference at 256 channels. The
[gated-attention evidence](../oNPU/docs/benchmark-gated-attention-verification-v1.json)
preserves the profile, raw timing/source cohorts, failed attempts and fresh
GPU/NPU serving checks. No clock-policy change was used.

The newer optional cross-request prefix cache passed CPU checks and a fixed-case
[matched-seed trained GPU diagnostic](../oNPU/docs/speculative-prefix-reuse-matched-v4.json).
Across six repetitions, all 42 state buffers and common probe logits matched
ordinary cached continuation bit for bit; all 18 temperature comparisons had
zero probability error and total variation. Cold replay differed equally from
ordinary and speculative cached execution. The earlier
[cold-replay gate failure](../oNPU/docs/speculative-prefix-reuse-core-v1.json)
is preserved. Different prefill/decode reduction groupings exist for some
projections, but this diagnostic does not identify the first divergent kernel.
The [V40 API bridge](../oNPU/docs/speculative-prefix-reuse-api-v40.json) now enables
this cache for the pinned experimental worker: forced follow-ups report the
actual reused prefix, while automatic cached requests stay ordinary. Its real
chat test reused 31 of 55 input tokens and passed cache disabling, sampled
cancellation and greedy recovery. The pinned template now preserves its empty
think header across turns. Corrected offline validation and an independent
audit resolved an obsolete controller summary-field expectation; original
records remain unchanged. These checks do not qualify model maximum context,
general quality or a speculative speed advantage.

The frozen-source [V41 long API check](../oNPU/docs/speculative-prefix-reuse-long-api-v41.json)
then exercised 8K conversation state: forced reuse of 8,049 tokens, cache-enabled
sampled cancellation and cold recovery, the cached ordinary 8,191+1 boundary,
and rejection of an over-budget reply. Cached first-output times were 1.98 and
5.23 seconds; cold requests still needed about seven minutes. This distinguishes
serving/cache progress from the remaining cold-prefill performance problem.

The newer [prepared draft memory option](../oNPU/docs/speculative-draft-feature-history-component-v1.json)
keeps append-only K/V and position metadata without full projected-feature
history. An actual trained-draft/shared-head component check streamed 8,192
synthetic feature rows: K/V, proposal logits/hidden/IDs and three sampled
temperature cases matched exactly. Retained history fell from 32 MiB to zero.
Native observation-copy counters fell from 2,406,746,112 to 1,284,069,376 bytes
and from 1,152 to 1,024 commands. These are native-copy counters, not total memory
or model-speed measurements. Default feature retention and eager/deep-rollback
support remain available. The [V42 API activation](../oNPU/docs/speculative-draft-feature-history-api-v42.json)
passed the short cached-force/ordinary controls, cache disabling and sampled
cancellation/recovery with zero projected-history retention reported. The pinned
experimental loader now enables it. The earlier V41 long timings are for the
retaining layout; the new layout still needs larger-context API qualification.
V38 below is the preceding frozen-source API cohort.

The [reusable proposal-input component check](../oNPU/docs/speculative-draft-reusable-inputs-component-v2.json)
shows another cost outside matrix multiplication. The draft previously cloned
and zero-padded every K/V input on every proposal. An opt-in layout keeps owned
fixed-capacity inputs, preserves unchanged full-attention rows and refreshes
sliding windows. Tensor assignment alone was slower. Native rectangular copies
on the existing OpenCL queue reduced repeated-prefix input staging from
14.62–16.55 ms to 1.45–2.87 ms across four prefixes. Actual trained draft K/V,
padded inputs, logits, IDs and sampled probabilities remained identical through
8,192 synthetic feature rows; three Mali tests include partial-copy recovery.
These timings measure input preparation only. The later
[full-target comparison](../oNPU/docs/speculative-draft-reusable-inputs-model-v1.json)
matched all 54 output sequences but found no complete-request gain; candidate
medians were 0.3–1.1% higher than control. The app keeps this option off. GPU
clock variation limits attribution of those small differences. Total RSS,
model maximum and RKLLM wins remain unqualified.

The [short-row Q8 projection](../oNPU/docs/benchmark-mali-q8-short-head-operator-v4.json)
reduces weight-storage traffic while preserving FP16-rounded values and each
FP32 GEMV reduction. Real-head paired tests measured 53.70 → 34.77 ms for one
row. Larger row groups improved the new Q8 block kernel, but the direct public
eight-row test remained slower than dense (119.06 versus 113.43 ms), despite a
win in a mixed-variant grid. All 54 paired outputs matched bit for bit. This
supports a one-row operator benefit, not an LLM or W8A8 win. Current model/App
dispatch remains unchanged; a shared raw embedding/head representation is
needed to release the old dense tied matrix rather than add another copy.

The [shared raw table](../oNPU/docs/benchmark-mali-q8-shared-table-model-v1.json)
removes the dense embedding/head ownership that a head-only attachment kept.
A valid lazy FP16 view preserves metadata access, while native lookup/head use
one raw allocation. All 72 trained-model sequences matched; observed ordinary
latency was 4.4–6.2% shorter, with small speculative deltas and workload-dependent
fallback. The [V44 worker/API check](../oNPU/docs/speculative-shared-q8-table-api-v44.json)
passed the current pinned profile's short cache, sampling-stop and recovery
controls. General SDK speed/quality and model maximum remain unqualified;
context/admission are unchanged and clock/process-order limits remain explicit.

The trained Qwen3.5-2B/DFlash2 OpenCL profile now includes the integrated
two-row tiled FP32 GEMV path. The [V38 API cohort](../oNPU/docs/speculative-api-tiled-v38.json)
verified 36 short off/force greedy requests and 27 fresh probability comparisons
with zero measured error/total variation, plus the actual 8,176-input + 16-output
boundary in off/force/auto modes, sampled temperature-0.6 cancellation and exact
greedy recovery. Regression checks totaled 351 methods and 156 subtests. Both
original workers were restored and answered fresh requests. These are scoped
checks, not model-maximum, RKLLM-quality or general-speed qualification. V37
remains the prior-source routing/API cohort; earlier API-v2 statements below
describe their dated sources.

A later [prefill tile experiment](../oNPU/docs/speculative-prefill-fp32-tiles-v1.json)
preserved ascending FP32 FMA arithmetic and checked actual gate/down weights in
both layouts at 27, 36 and 128 valid rows. Smaller row tiles and larger K tiles
were slower on most cases. Workgroup-axis remapping changed row-major medians
by only about 1–2%; an isolated packed 36-row projection improved from 16.97 to
15.63 ms. These results do not establish faster full-model prefill, so the
production kernels remain unchanged. The experiment passed two GPU tests and
150 subtests, Ruff and mypy; both original workers answered fresh replies afterward.

The CPU/NPU path also has an opt-in parked worker pool for independent recurrent
heads. Its [C operator measurements](../oNPU/docs/benchmark-cpu-gdn-head-parallel-perf-v1.json)
reduced the recorded 16-head/key128/value128 scan from 56.41 to 12.74 ms at 128
rows while preserving output and state bits. The generated CPU binding keeps
helper functions private so the correct kernel wrapper remains the JIT entry.

The [whole-model comparison](../oNPU/docs/benchmark-cpu-gdn-head-parallel-model-v1.json)
shows the practical limit of that operator result. With CPU scan enabled in both
variants, Qwen3.5-2B rose from 63.92/10.00 to 71.15/10.80 prefill/decode tok/s when
pooling was enabled. The full model gain is much smaller than the scan's isolated
gain. The surrounding projections, attention, transfers and dispatch still need
profiling. All twelve requests matched the earlier INT8 output sequence; output
quality against FP16/RKLLM remains unassessed. The app worker has not automatically
enabled this backend profile. The recorded RKLLM reference remains faster at
164.75/13.59 tok/s. Both normal workers were restored and answered fresh requests.

The [native replay profile](../oNPU/docs/benchmark-native-replay-warmup-model-v1.json)
now separates the remaining costs: CPU kernels account for about 88% of captured
native prefill time, while Rockchip calls account for about 78% of native decode
time. The latter includes driver/submission work. The captured 6144-wide sigmoid
alone took about 16% of native prefill time before the pooled sigmoid change.
It expands a per-lane exponential polynomial and special-value checks in one CPU
program; the matrix engine does not execute those instructions.

The [pooled sigmoid comparison](../oNPU/docs/benchmark-cpu-sigmoid-backend-model-v1.json)
then improved Qwen3.5-2B prefill from 71.26 to 79.87 tok/s with the same pooled
scan in both variants. Decode was 10.69/10.73 tok/s with overlapping ranges;
there is no established decode gain. The isolated 128-row sigmoid was about
3.89× faster, but the whole-model prefill gain was 12.1%. The C helper retains
the captured arithmetic and materialization boundary, changes row ownership,
and keeps split-down quantization unchanged. All twelve model requests retained
the prior INT8 IDs. This opt-in backend profile remains below the recorded
RKLLM 164.75/13.59 tok/s and is not automatically enabled by the app worker.
Remaining CPU normalization and attention work needs separate measurements;
decode needs matrix and submission tuning.

The [current trained OpenCL profile](../oNPU/docs/speculative-trained-gpu-profile-v2.json)
also identifies a concrete GPU target. The 248,320×2,048 vocabulary head uses
FP32 activations and FP16 stored weights and accounts for 17–23% of recorded
GPU kernel time across counting, code and explanation. In counting, verification
spent another 0.965 seconds in 8-row 6,144×2,048 projections and 0.500 seconds in
8-row 2,048×6,144 projections. These event times come from serialized diagnostics;
they exclude transfers, CPU work and gaps and do not establish a request speedup.
Frozen warmup routing keeps profiling overhead from changing fallback decisions.
All twelve requests matched frozen outputs. The earlier workgroup test covered
only the smaller 6,144×2,048 shape and produced mixed small differences.

The [qualified row-tiling operator](../oNPU/docs/speculative-vocabulary-row-tiles-v2.json)
then reduced live row accumulators while preserving every output's arithmetic.
It keeps two rows per work item and interleaves neighboring row tiles by column
group. On the actual weights, the selected 8-row vocabulary head improved from
203.58 to 109.14 ms and down projection from 5.65 to 3.10 ms. No extra packed
weight copy is needed. Dispatch stays limited to the measured shapes and Mali
device/driver; scalar decode and other shapes retain their existing paths.

The [whole-model comparison](../oNPU/docs/speculative-vocabulary-row-tiling-model-v1.json)
measured 10.3% shorter counting requests, 9.1% shorter code requests and 3.4%
shorter explanation requests than the legacy speculative kernels. All 54 frozen
sequences matched. Counting and code also beat fresh ordinary decoding;
explanation did not. These short greedy results identify an inference benefit
from kernel geometry; current-source API and model-maximum qualification remain
open.

Profiling also found native replay compilation after three ordinary warmups.
Preparation now completes during the existing warmup update without applying
that update twice. The real-model check kept all output IDs and moved the first
measured TTFT from an 11.65-second setup outlier to about 1.80 seconds. This fixes
measurement readiness; the later pooled-sigmoid comparison supplies the current
unprofiled throughput row.

## First, distinguish the IR operation from our Rockchip implementation

[`Ops.WMMA`](tinygrad/uop/ops.py) represents a matrix multiply-accumulate operation
that a supported renderer lowers to its hardware instructions. The
[`TensorCore` description](tinygrad/renderer/tc.py) and
[tensor-core matching](tinygrad/codegen/opt/postrange.py) describe supported shapes,
operand types and layouts. Selecting that operation still leaves tiling, memory
movement, surrounding operations and runtime execution to implement.

**Our explicit Rockchip matmul path does not use `Ops.WMMA`.**
[`rockchip.py`](tinygrad/llm/kernels/rockchip.py) builds an `Ops.PROGRAM` with a
binary geometry descriptor. [`ops_rockchip.py`](tinygrad/runtime/ops_rockchip.py)
turns it into register commands submitted to the RKNPU driver. The renderer does
not offer general Rockchip graph lowering. Likewise, our Mali kernels use
OpenCL vector FMA, local memory and subgroup reductions; they are not a WMMA
implementation. Direct register control gives us control over the matrix recipe,
not free execution of an entire transformer on the NPU.

## What limits a token

A useful cost model for a sequential path is:

```text
token time = matrix arithmetic + weight/activation/KV traffic
           + attention or recurrent state work + normalization/activation
           + layout conversion and device/cache synchronization
           + submission, host scheduling and sampling
```

Some work can overlap, so this is an accounting guide rather than a universal
timing identity. Improving one term can expose another bottleneck.

| Cost | Why a fast matrix primitive does not solve it |
| --- | --- |
| Prefill versus decode | Prefill can reuse weights across many prompt rows. Ordinary decode has one new row; streaming weights and launching work can dominate a narrow matrix operation. The best kernel and placement can differ by phase. |
| Memory layout and capacity | Matrix hardware consumes specific packed layouts and bounded tiles. Repacking, padding, partial outputs and excessive copies can outweigh arithmetic savings. Retaining both original and packed weights also consumes RAM. |
| Attention and state | QK, softmax, PV, RoPE, KV writes and reads still run. Longer context increases attention traffic. Hybrid models additionally update recurrent/convolution state; speeding only their projections leaves that state path serial. |
| CPU/GPU/NPU boundaries | Shared system RAM does not imply coherent, zero-cost access. The current allocator includes copies and explicit driver cache synchronization; returning an NPU result to CPU or CL has a cost. |
| Small operations and launches | Norms, activations, quantization, copies and many small kernels add dispatch overhead and temporary buffers between large projections. |
| Vocabulary and sampling | A large output projection, logits transfer and greedy/sampled selection occur for every generated token. Body matmul throughput alone omits them. |
| Model correctness | Quantization, different reduction order, padding and state reuse can change logits or corrupt later requests. A speed result needs output and lifecycle checks. |

For example, if matrix arithmetic were half of a measured token's time,
making only that arithmetic twice as fast would improve the whole token by
`1 / (0.5 + 0.5/2) = 1.33x`. This is an illustration, not a measured breakdown.

## What we implemented around the matrix kernels

Implementation availability is separate from automatic serving selection.
Experimental options remain disabled unless explicitly selected or qualified for
the worker's exact checkpoint and configuration.

| Area | Concrete implementation | Current scope |
| --- | --- | --- |
| NPU matrix execution | Persistent packed weights, reusable bound command/task buffers, full-K streaming, one/three-core column sharding, wider decode output tiles and selected GPT-2 register-grain tuning. | Explicit supported shapes with CBUF/alignment checks; not arbitrary operator coverage. [Runtime](tinygrad/runtime/ops_rockchip.py), [FP16 kernels](tinygrad/llm/kernels/rockchip.py). |
| Phase placement | Separate prefill/decode policies; automatic trials include transfers, validate results and require a 10% median layer improvement. | Optional NPU assistance; forced modes are useful for experiments but do not establish a speed benefit. [Placement](tinygrad/llm/kernels/rockchip.py). |
| Fewer projection boundaries | Shared-input QKV and gate/up projections; native consumers of projection tiles avoid unnecessary joins and layout conversions. | Explicit options and model/layout checks. [Projection wrappers](tinygrad/llm/kernels/rockchip.py), [CPU kernels](tinygrad/llm/kernels/rockchip_cpu.py). |
| Quantized NPU path | Per-output-channel INT8 weights, per-token activation scales, INT32 products, fused dequantization with Q/K normalization/RoPE/KV writes and SwiGLU/input packing. | Optional quantization recipe; qualify output quality independently and report timing workload, dtype and clock scope. [INT8 kernels](tinygrad/llm/kernels/rockchip_int8.py), [register recipe](tinygrad/runtime/support/rockchip_int8.py). |
| Attention and host work | CPU decode attention, packed-cache variants, vector softmax, greedy argmax, reusable CPU pools and cache-cleaning helpers. | Selective modes; their value depends on model, phase and context. [Attention](tinygrad/llm/kernels/rockchip_attention.py), [packed attention](tinygrad/llm/kernels/rockchip_packed_attention.py), [softmax](tinygrad/llm/kernels/rockchip_softmax.py), [argmax](tinygrad/llm/kernels/rockchip_greedy.py). |
| Less Python dispatch | Guarded native CPU/NPU replay and a native decode loop over prepared graphs. | Ownership, buffer binding and state rules still apply; unsupported graphs use the ordinary path. [Replay](tinygrad/llm/kernels/rockchip_replay.py), [decode](tinygrad/llm/kernels/rockchip_decode.py). |
| GPU projections and embeddings | Separate GEMV for one-row decode and GEMM for bulk prefill; vector loads, FP32 accumulation, optional packed weights, a tiled row-major fallback, direct embedding lookup and optional QKV/gate-up fusion. | Mali-specific prepared path with ordinary fallbacks. [Linear kernels](tinygrad/llm/kernels/mali_linear.py), [configuration](tinygrad/llm/kernels/mali.py). |
| GPU attention and output head | Prepared attention with bounded masks/scratch and an optional native Q8_0 output-head kernel that reproduces the FP16 stored operands. | Worker selects the qualified Qwen3-0.6B checkpoint/configuration. Native Q8 head storage is not W8A8 body inference. [Attention](tinygrad/llm/kernels/mali_attention.py), [Q8 GEMV](tinygrad/llm/kernels/mali_linear.py), [worker policy](../oNPU/onpu/worker.py). |
| Avoided output work | Ordinary prefill projects only the last hidden token to the vocabulary, rather than materializing prompt-length-by-vocabulary logits. | Normal generation path; block verification deliberately needs a separate interface. [Model](tinygrad/llm/model.py). |
| Hybrid recurrent prefill | One OpenCL scan processes the valid rows while retaining FP32 GatedDeltaNet state; subgroup reductions replace repeated generic per-token recurrent graphs. Eligible projections use prepared kernels too. | Qualified Mali hardware; worker enables exact Qwen3.5 checkpoint hashes only with realized weights, GPU-only inference and context <=512. [Scan](tinygrad/llm/kernels/mali_recurrent.py), [model integration](tinygrad/llm/model.py). |
| Memory bounds | Streaming GGUF decoding, bounded loading temporaries, direct row-major recurrent projections without a second packed copy, and a no-weight-packing option for larger GPU models. | Avoiding OOM is essential to usable inference; less packing can cost prefill speed. [Loader](tinygrad/llm/gguf.py), [Mali policy](tinygrad/llm/kernels/mali.py). |
| Trained speculative decoding | DFlash2 draft, target feature/block-logit interface, greedy block verification and recurrent/convolution rollback with accepted-prefix replay. | Explicit greedy experimental API is HTTP-tested at context 512; separate backend diagnostics match 64 ordinary tokens. Prepared draft/observation, dynamic verification and exact replay remain API-disabled; speed and wider-context qualification remain pending. [Draft](tinygrad/llm/dflash2.py), [coordinator](tinygrad/llm/speculative.py), [API evidence](../oNPU/docs/speculative-api-verification-v2.json). |

The NPU FP16 recipe currently checks at most 64 rows per core, K<=4096,
32-aligned K/output columns and at most four input CBUF banks. Inputs use
`(K/8, M, 8)`, weights `(N/16, K/32, 16, 32)` and outputs `(N/4, M, 4)` with FP32
accumulation. These concrete choices explain why an abstract matmul mapping does
not by itself deliver a usable LLM backend. INT8 has its own checked geometry.

## Measured end-to-end gains, with limits

The following compares initial versus optimized **oNPU OpenCL** runs on this
RK3588S board. Each uses context 512, 128 input tokens, 64 generated tokens,
three complete warmups and three measurements, greedy decoding, fresh history
and ignored EOS. GPU FP16 operands originate from Q8_0 GGUF checkpoints. Loading
and warmup are excluded. Prefill is the proxy `128000 / TTFT(ms)`, including the
first output token; it is not isolated matrix throughput.

| Model | TTFT (ms), initial -> current | Prefill proxy (tok/s), initial -> current | Decode (tok/s), initial -> current |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 41215.11 -> **2735.69** | 3.11 -> **46.79** | 3.14 -> **4.54** |
| Qwen3.5-2B | 101525.93 -> **7096.23** | 1.26 -> **18.04** | 1.27 -> **2.40** |
| Qwen3.5-4B | 480831.24 -> **23436.57** | 0.27 -> **5.46** | 0.27 -> **0.60** |

All 64 generated IDs agree with the corresponding initial baseline for every
warmup and measured request. Measured timing ranges separate in both phases;
CPU affinity was unchanged. These gains combine the recurrent scan, bulk prefill
and prepared projections: they do not isolate the benefit of one kernel.
They do not establish maximum-context performance or RKLLM-equivalent quality.
See [family measurements](../oNPU/docs/benchmark-recurrent-family.json) and the
[full comparison](../oNPU/benchmark.md), which retains checkpoint/report hashes
and the remaining slower models.

Packing has a measurable tradeoff. The earlier packed Qwen3.5-0.8B experiment
achieved **56.79 prefill tok/s at 2910.21 MiB steady allocation**. The current
recurrent row-major path gives **46.79 tok/s at 2169.96 MiB**, saving about
740 MiB while decode stays about 4.54 tok/s. The 4B run disables global weight
packing as well. App RAM admission recognizes this exact 4B checkpoint and
GPU configuration with an additional startup/workspace reserve; other layouts
retain conservative estimates. A complete 4B API/UI load has not been rerun as
part of that admission fix, so this benchmark does not certify 4B UI usability.

Other models show why both phases must be measured: tiled GPU tuning improved
Gemma4-E2B prefill from 2.02 to 10.31 tok/s while decode changed from 2.88 to
2.87; ChatGLM3-6B prefill improved from 0.43 to 3.61 while decode changed from
1.51 to 1.39 tok/s. ChatGLM3 GPU clocks also varied, limiting attribution.
[Measurements](../oNPU/docs/benchmark-gpu-tuning.json) retain these regressions.

A separate GPT-2 124M CPU/NPU experiment measured 1006.76 prefill tok/s and
96.42 decode tok/s at context 1024, 128 input tokens and 32 greedy output tokens.
Its paired ABBA control already used NPU matmuls and native generation; adding
cache cleaning and the attention pool improved decode by 10.0% for that length,
while prefill stayed similar. This is evidence that work outside matmul matters.
All 3840 generated IDs across the full multi-length experiment matched CPU,
but 14 strict raw-logit comparisons still failed their unchanged gates. There is
no RKLLM GPT-2 timing in that experiment. [Scope and reproduction](docs/rockchip-gpt2.md).

An output-copy borrowing experiment is implemented behind
`ONPU_REPLAY_BORROW_OUTPUTS=1`, default off. Replay tests pass, but the measured
Qwen3 graph had **zero eligible eliminated output copies**. It has no demonstrated
full-model speed gain. Likewise, an opt-in CPU prefill pool has not demonstrated
an overall improvement and is not an automatic serving optimization.

The trained DFlash2 prototype now runs the verified 708,494,328-byte
`rasyosef/Qwen3.5-2B-DFlash2` checkpoint with the Qwen3.5-2B target on OpenCL.
For one 128-token prompt at context 512, all eight generated IDs match ordinary
greedy decoding. The draft proposed 17 tokens, five were accepted, and three
verification blocks required two rejection replays. Cold wall time was
**22.88 seconds ordinary versus 86.02 seconds speculative**, including compilation;
this is a correctness probe, with no demonstrated speed benefit. The target
embedding and output head are shared with the draft.

The follow-up real-model diagnostic uses cached draft prefix K/V, 37 prepared
Mali projections, a prepared shared output head and exact-row target graphs.
All eight greedy IDs match across three ordinary and three speculative requests
for the same prompt. On the final request, ordinary generation took
**10.07 seconds** (7.14 seconds to first token, 2.39 decode tok/s) versus
**39.37 seconds** speculative (30.17 seconds to first token, 0.76 decode tok/s
including all remaining costs). Compilation/capture is included; this short
diagnostic demonstrates no speed gain. See the
[diagnostic record](../oNPU/docs/benchmark-dflash2-diagnostic.json).

The v3 diagnostic changes prepared target prefill to dynamic rows while retaining
exact-row verification. On the final request, speculative prefill takes
**7.127 seconds**, versus 24.276 seconds in v2, and the whole speculative request
takes **22.12 seconds** versus 39.37 seconds. That observed reduction is 43.8%;
ordinary generation still takes only **10.01 seconds**, so speculation remains
**2.21× slower**. All eight greedy IDs match across three requests per engine;
peak process RSS is 7434.64 MiB. These separate runs include compilation/capture,
without the formal three-warmup/three-measurement protocol. The
[v3 record](../oNPU/docs/benchmark-dflash2-dynamic-prefill-diagnostic.json) pins the
measured source cohort; it does not qualify subsequent code changes.

The v4 cohort adds a bounded prepared draft graph and kernels for small row
counts. Its final request takes **21.11 seconds speculative versus 10.10 seconds
ordinary**, so speculation remains **2.09× slower**. Draft time is 2.199 seconds,
versus 3.392 seconds in v3, while verification takes about 8.78 seconds. All eight
IDs match across three requests per engine; one captured draft graph handles
nine calls. Prepared draft proposals remain disabled in the app. The
[v4 record](../oNPU/docs/benchmark-dflash2-prepared-draft-diagnostic.json) pins the
measured source and preserves the short-diagnostic limits. A separate optional
head diagnostic could not qualify the resident OpenCL view during layout
preflight and produced no head timing. That inspector was corrected for v5;
the old rejection did not establish a bad layout or invalidate the model probe.

V5 uses dynamic verifier rows and retains exact IDs on three eight-token requests
plus one 64-token request per engine. Its final short request is **17.82 seconds
speculative versus 10.07 seconds ordinary**. The single longer pair is **106.79
versus 33.36 seconds**: verification takes 33.95 seconds, rejected-prefix replay
31.53, draft work 19.07 and observation/cache work 13.08. This exposes costs
that a faster matrix primitive cannot remove. The
[v5 record](../oNPU/docs/benchmark-dflash2-dynamic-verifier-diagnostic.json) pins
the measured source and the limited validation; neither pair is a formal win.

The corrected supplemental test now qualifies the actual resident
`248320×2048` FP16 head without a packed weight copy. All output bits match the
legacy projection at 3/8/16 rows. Kernel-event medians improve about 28.8% at
three rows and 15.5% at eight, with three warmups and seven rotating measurement
rounds. These isolated projection gains do not establish whole-model gains.
Live head-gate qualification is still pending; dynamic verification, prepared
draft proposals and exact-row replay remain API-disabled. V5 does not qualify
subsequent source changes.

V6 adds prepared observation, exact replay and the qualified 3/8-row head gates.
Its strict short/64-token outputs all match, but the final short pair is
**16.14 seconds speculative versus 10.10 ordinary**, and the single long pair is
**115.09 versus 33.49 seconds**. Long replay takes 46.63 seconds, verification
32.56, drafting 16.80 and observation 9.88. Additional exact-row graph capture
is included; these timings do not establish an all-warm regression.
The optional counting/code pairs match fixed 32-token output, yet are slower;
the explanation pair was not run under the soft budget. EOS was ignored, so
these are not completed-answer latency measurements. The
[v6 record](../oNPU/docs/benchmark-dflash2-prepared-observation-diagnostic.json)
separates the complete strict cohort, incomplete natural pairs and verified
recovery after the wrapper's final evidence-write failure. Advanced API flags
remain unpromoted; state-history tracing and NPU draft proposals were staged at
that v6 checkpoint.

V7 implements native GDN prefix-state tracing for the measured cohort. All
strict outputs and three fixed-32-token natural pairs match. The last short
pair is **14.40 seconds speculative versus 10.02 ordinary**; the single long pair
is **70.90 versus 33.38 seconds**. Long replay is zero, replaced by 22 trace
writes and 20 restores with a 155,860,992-byte workspace. Verification still
takes 34.20 seconds, drafting 16.21 and observation 9.96. The
[v7 record](../oNPU/docs/benchmark-dflash2-prefix-trace-diagnostic.json) preserves
source and restoration pins. Compilation/capture and fixed-output EOS rules
still prevent a formal warmed or completed-answer speed claim.

Optional NPU placement can help a large projection: a separate synthetic
eight-row FP16 head test measures **350.43 ms GPU versus 62.90 ms CL/NPU/CL**,
including all transfers and 21 submissions. That 5.57× micro speedup costs an
extra 970 MiB packed head and is not a trained-model or target-quality win.
V7 did not use this NPU candidate; trained integration was then unqualified and
the option remains off by default. The [micro record](../oNPU/docs/benchmark-npu-draft-head-microdiagnostic.json)
keeps the failed original lease separate from verified supplemental recovery.

V8 exercises the NPU draft head in the trained model: strict and natural IDs
match, the GPU target head stays unchanged, and submission counters verify the
NPU work. The single long request falls from v7's 70.90 seconds to **65.11
seconds**, with draft time 16.21→10.74 seconds, yet ordinary takes **33.38
seconds**. Verification still takes 34.47 seconds. This
[trained diagnostic](../oNPU/docs/benchmark-dflash2-npu-head-diagnostic.json)
includes request transfers/submissions; the extra 970 MiB pack and 2.92-second
preparation are separate. It establishes limited correctness and NPU use,
without a formal speed, broad-quality or API win.

An earlier GPU-only packed-head micro measured 348.82→165.61 ms (2.106×)
with matching FP16/FP32 output bits for FP16 input, at an extra 970 MiB and
0.902-second pack; its flag was then staged and trained-LLM benefit untested
([raw 48340636…](../vlm-bench/baseline-20261021/main-task-20261023/GPU-packed-head-r8-f16-v1.json)).

V9 tests one GPU-packed head shared by target and draft: all strict/natural
output gates pass without opening the NPU. The single long request is
**63.18 seconds speculative versus 33.51 ordinary**, compared with 70.90 seconds
in unpacked v7; all three natural pairs also remain slower. Packing costs
970 MiB and 0.767 seconds, while single-row decode retains the original GEMV
weights. Ordinary short trials now share draft residency, so the cross-cohort
comparison is diagnostic. The [v9 record](../oNPU/docs/benchmark-dflash2-gpu-packed-head-diagnostic.json)
separates the preserved preflight metadata failure from the successful
same-interpreter retry and verified restoration. The flag remains default-off
with no API promotion or formal win.

V10 combines the shared head, 36 packed GDN FFN gate/up projections and batched
observation copies. Exact strict/natural outputs pass, yet the single long pair
is **57.21 seconds speculative versus 32.53 ordinary**. Verification falls to
25.30 seconds from v9's 29.83; observation is effectively unchanged at
**9.92 versus 9.94 seconds**. The observed whole-request reduction does not
isolate either change's contribution. Admission runs after both models and
unused-cache cleanup, reserving **970+864 MiB plus 2 GiB headroom** before any
packing/capture. The [v10 record](../oNPU/docs/benchmark-dflash2-combined-packing-diagnostic.json)
keeps setup, lifecycle and separate harness failures explicit. All pairs still
lose to ordinary; the options remain default-off without API promotion.

The recorded central app suite has 226 passing tests. The second isolated HTTP
diagnostic verifies dynamic prefill, matching nonstream/SSE text, usage and
experimental mode, sampled ordinary fallback and exact greedy recovery.
Query-scoped Stop rejects a stale request ID and ends the matching request with
an error and `[DONE]`; the worker becomes ready before disconnection, about
9.46 seconds after Stop acknowledgment in this run. The test server on port 8092
left no orphan workers; the original GPU/NPU sessions and settings were restored
and checked with real requests. The existing port-8091 service was not restarted.
See the [HTTP v2 record](../oNPU/docs/speculative-api-verification-v2.json).
The [older corrected proof](../oNPU/docs/speculative-api-verification.json) covers
selected-worker Stop only. Prepared draft proposals remain disabled in the app;
this HTTP proof does not qualify the v4 proposal-graph execution path.
The target graph cache now retains at most 16 graphs, including two prefill
graphs; its four cache tests pass, including two actual Mali cases. Early-anchor
emission remains disabled by default. Sampled speculation, automatic API cost
routing and model-maximum context/quality qualification remain open. The
[feature record](../oNPU/docs/omlx-feature-comparison.json) records the limits.
EAGLE-4 has no matching trained checkpoint for our current small
targets in the dated inspection; an independent cost policy is not an EAGLE-4
runtime. See [EAGLE-4 support](docs/eagle4-support.md).

## What still separates this from a fast general inference engine

- Compare measured prefill and decode directly with the published RKLLM speed
  targets, reporting workload, dtype and clock scope. Qualify output quality
  independently. Matched local SDK/export reproduction is optional when useful
  and currently deferred; “faster than published” does not certify a matched
  local comparison.
- Reduce actual weight/KV traffic and boundary costs identified in whole-model
  profiles. A fast operator microbenchmark is insufficient evidence.
- Validate each enabled path at the model's required context, including cache
  growth, cancellation, repeated requests and simultaneous workers. The CPU/NPU
  recurrent model candidate is measured at context 512; the pinned GPU DFlash2
  API reaches 8,192. Model maximum remains open.
- Resolve the recorded import/order-sensitive existing Mali attention-suite
  failure. Isolated and controlled-startup passes are useful evidence, but do
  not establish that the default suite is clean.
- Extend and accelerate the scoped trained speculative implementation. Warm measurements
  must include draft computation, target verification, rollback, accepted-token
  counts and all cache/host costs; repeated eight-token diagnostic requests do
  not establish acceleration or general output quality.

Expert residual dequantization also needs a stable floating-point contract.
An intermediate trace reproduced a synthetic decode failure bit for bit: the
CPU recipe fused one multiply-add while the NPU readback path rounded both
products separately. A 2.38e-7 projection difference later crossed an INT8
vocabulary quantization boundary. Explicit FMA dequantization now passes the
unchanged model tests. The trained residual candidate still fails its local
logit gate and reaches 165.93/33.28 prefill-proxy/decode tok/s versus RKLLM's
182.54/31.60. Correct registers and a passing operator are therefore separate
from a qualified model win. [Residual model evidence](../oNPU/docs/benchmark-npu-experts-activation-verification-v5.json).

Adding a third product for weight residuals passes six synthetic model modes
and reduces error in the tested isolated projections to about 0.007–0.025%.
The trained DeepSeekOCR run still has 0.055134 relative-logit RMS against its
CPU-expert control, above the unchanged 0.01 gate, and measures 161.79/27.10
prefill-proxy/decode tok/s. Therefore the remaining discrepancy cannot be
explained or qualified solely from those projection probes. Indexed-device
counting and the warmup memory limit also required explicit audit records.
[Three-product run and limitations](../oNPU/docs/benchmark-npu-experts-weight-verification-v1.json).

An intermediate trace now identifies a concrete amplification boundary. The
next attention input differs by only 0.0105%, but dynamic INT8 quantization flips
131 of 163,840 prefill codes. Independent CPU reconstruction predicts all
491,520 changed QKV integer sums exactly, and FP32 dequantization matches. The
attention output difference grows to 0.2751%; a later quantizer grows it further.
The same decode boundary changes one input code and all 3,840 integer deltas
are independently verified. This is quantization sensitivity, with no incorrect
NPU products at that boundary. The existing model logit gate remains failed.
[Bitwise-verified boundary trace and integer proof](../oNPU/docs/benchmark-npu-boundary-firstdiff-v2-verification.json).

An exact CPU alternative keeps the N16 Half weight pack and sequential FP32
feature order while reusing each weight across six prefill rows. It improves
the trained prefill proxy from 106.49 to 112.88 tok/s with bitwise-identical
teacher logits. Decode measures 25.84/25.21 tok/s, so the new option remains
off by default and the preferred timing row is retained. Wider decode tiles
lost their operator comparison and were removed. This supplies a measured
prefill improvement without claiming a two-phase or RKLLM win.
[Six-row trained results](../oNPU/docs/benchmark-deepseek-cpu-prefill-r6-verification-v1.json).

The fast single-product expert candidate now measures 267.33/37.38
prefill-proxy/decode tok/s and passes a separate local 512-target English PPL
check: ratio 0.996349, empirical sequence-bootstrap upper estimate 1.011648,
against 1.02. Its original raw-logit RMS gate still fails, while natural-text
argmax agreement is 94.73% and mean TV is 0.046617. Speed, logit drift and
task-oriented quality evidence must be reported separately. This sample uses a
mixed-precision control and does not certify FP16, RKLLM or vision-task quality;
no automatic promotion occurs.
[Scoped natural-text evaluation](../oNPU/docs/benchmark-deepseek-natural-quality-verification-v1.json).

The engineering target is lower time to first token and lower time per accepted
generated token at acceptable quality and memory use. `Ops.WMMA`, direct NPU
registers and custom OpenCL kernels are tools for reaching that target; the
complete measured request is the result that matters.
