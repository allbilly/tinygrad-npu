# CPU preparation for hybrid GatedDeltaNet models

The October 30 Qwen3.5 0.8B/2B port improves raw-token prefill at context 512. It combines four compatible recurrent projections, shares CPU workers through the request, and parallelizes four costly CPU preparation shapes. It preserves existing INT8 boundaries and FP32 operation order. This is a backend performance result, not independent RKLLM quality certification or model-maximum-context qualification.

`fused_recurrent_projections=True` combines QKV, output gate, beta and alpha for ordinary GDN blocks. They consume the same FP16 input. Their independent output-channel scales, FP16 projection rounding and subsequent sigmoid/decay operations remain intact. KDA's extra projection stages keep their original path. The combined 8,224-column matrices at K=1,024/2,048 passed eight independent INT32 oracle cases, followed by full-model regression.

The native scan processes 64 independent value lanes when available. Each lane retains the original serial key reduction and update grouping. Smaller widths retain the prior 16-lane or partial-vector implementation. Reset, valid-row bounds, canonical state layout and prefix-history writes remain unchanged. CPU coverage includes scalar/per-key decay, odd key dimensions, history and padding.

`ONPU_GDN_DECODE_SPIN=1` keeps the existing workers active between recurrent calls during a token operation. The enclosing pool scope parks them afterward. `ONPU_SHARED_HOST_POOL=1` also admits the bounded context-512 hybrid graph once all threaded helpers use the same pool; KDA, experts and unrelated graph families remain excluded. These flags change request dispatch, not governors, IRQ settings or clock limits.

`ONPU_GDN_PREP_POOL=1` splits independent outer loops in six reviewed source variants representing four CPU preparation shapes. Their output regions are disjoint across workers and reduction scratch is private. Admission checks the full original source hash, CPU target and separate output/input slots. Unrecognized source falls back. The rewritten SOURCE and BINARY are both retained in the captured program; production does not monkeypatch the compiler. The generic CPU renderer's empty default selector is valid alongside explicit CLANG; an integration test exercises an ordinary CPU JIT with that metadata.

The source gate is intentionally conservative. A compiler/layout change requires reviewing the new outer-loop dependencies and repeating qualification. It must not silently parallelize an unrelated kernel with a similar name. Serving checks the actual preparation records after warmup, so an unavailable fast path fails its selected recipe rather than reporting an unexercised feature.

Narrow SwiGLU FFNs need a separate rounding contract. The older single-K fused kernel divides gate directly by the sigmoid denominator; the wide pipeline materializes a rounded reciprocal first. `legacy_division` retains the former when replacing an enabled narrow fused FFN. Both packers initialize the same headers, padding and INT8 payloads. CPU tests compare all payload/scale bits against the original packer across row counts and row batches. A prototype using the other contract preserved generated IDs but changed full logits and was rejected.

Actual port cohorts use 128 raw input IDs, 64 actual output tokens, three full warmups and three unprofiled measurements. All 32 full-vocabulary teacher vectors and generated sequences match the same-day controls bit for bit. Each model reports six rewritten preparation sources and positive native/NPU work. All original serving sessions were restored with fresh replies; the original GPU remained resident. Decode still trails the published reference.

[Measurements, rejected prototypes and restoration receipts](../../../oNPU/docs/benchmark-qwen-remaining-20261030.json) distinguish the actual port from staged experiments. The failed first microbenchmark is excluded because it called the wrong ELF entry; its replacement verifies entry offset, actual output writes and the production CPU operator before timing.

## October 31: decode preparation and host waiting

The later port adds separate approvals for one-row decode convolution, SiLU and
normalization. `ONPU_GDN_DECODE_PREP_POOL=1` applies these approvals only to the
captured decode graph. The approved outer bounds are 2,16,32,512 and1024; two
outer iterations use two workers. Every per-head reduction keeps its serial
order. Multi-output/state-update programs and programs that read output slot0
remain excluded. Prefill and decode records are separate, including when an
identical source is independently approved for both phases.

The scalar decoder enters the model's shared scope, so preparation callbacks
reuse active workers through a bounded native batch. An ExitStack unwinds the
pool and shared flag if the IRQ helper fails to enter. Regression tests check
that a subsequent request can reacquire the pool.

`ONPU_HOST_POOL_WFE=1` uses ARM event waits between jobs. Submission, completion,
parking and shutdown notify waiting workers; the producer repeats notifications
until completion/shutdown acknowledgements, and all workers reach the condition
wait outside requests. This retains the stock adaptive governors. A paired2B
cohort retained13.68 vs13.67decode tok/s while request-end temperatures fell from
about83C to68C. Futex sleeping reduced the rate to10.17tok/s as worker frequencies
fell; it was rejected. No fan, governor, IRQ or clock-limit settings changed.

The4B preparation layouts passed independent CPU bit/ABI checks before admission.
The final staged decode treatment reports2B14.08 and4B6.34decode tok/s, with all32
full-vocabulary vectors equal to controls. Actual-port numerical and API checks
are recorded separately; staging timings must not be relabeled as current API
results. The current recipes are limited to their pinned checkpoints/context512.

`trim_cpu_preparation_cache()` releases unused CPU allocation-cache entries after
configuration returns. The4B worker also trims after each warmup shape, retaining
live model, captured and NPU buffers. This reclaimed roughly1.1GB of preparation
scratch and allowed testing alongside the original40k-context GPU model.

[October31 evidence and qualification status](../../../oNPU/docs/benchmark-qwen-remaining-20261031.json)
retains rejected runs and distinguishes backend and HTTP measurements.

The final actual-port API cohorts report2B242.44/14.16 and4B82.52/6.32
prefill/decode tok/s, with all ten requests above both published references for
each model. Both pass18serving checks and exact local full-logit regression.
These are context512 text results, with independent RKLLM quality still pending.

The current-source margin refresh also enables the five approved decode
preparation programs and event waits for Qwen3.5 0.8B: backend decode increased
from27.52 to29.54tok/s, and the actual API reports29.47tok/s. Full logits match.
For Qwen3-VL text decoding, the scalar loop alone increased API decode to15.80
tok/s. Event waits lost VL prefill in the controlled screen and remain disabled
for that profile. Waiting policy must therefore be qualified per workload.
[Refresh trials and source scope](../../../oNPU/docs/benchmark-qwen-margin-refresh-20261031.json)
retain the slower combination and all controls.


## Four-worker head partition, November 3, 2026

Two exact source hashes for the 128-row, 32-value-head convolution/SiLU/normalization layout previously ran serially. Their outer repeat has extent two; partitioning the next 16-head axis gives four independent worker ranges. Each head owns 16,384 output floats per repeat. The two repeats cover all 524,288 floats; all other buffers are read-only. The source hash guard keeps every other generated layout on its existing path.

The transformation preserves the complete arithmetic of every output. Both actual-source fixtures cover changed inputs, signed zero, subnormals, wide values, nonfinite values, active workers and parked fallback. Production compiled ELF bytes match the prototype at the same dispatcher address. The CPU component improves about 8.1 to 2.05 ms. Four full-model ABBA prototype processes preserve all 32 full-vocabulary vectors each and all generated sequences, with 20.42% and 19.93% lower backend TTFT. Decode is unchanged within measurement variation.

The production API result is 101.34 prefill-proxy and 6.48 decode tok/s over 20 requests across two workers, with 36 serving checks. Context is 512 with 128 uncached inputs and 64 actual outputs. This does not qualify independent original/RKLLM output quality or model-maximum context. See `/home/orangepi/oNPU/docs/benchmark-qwen-inner-heads-20261103.json` for all raw cohorts, source epochs, gates and restoration.
