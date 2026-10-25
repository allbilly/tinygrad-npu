# Near-maximum-context API fixes

INTENT: cancelled discovery can leave subprocess transports unreaped; the goal requires stable repeatable service; app lifespan cancels discovery and awaits cleanup during shutdown.
INTENT: prepared GPU matrices need a coalesced weight layout as well as the decode layout; the goal requires faster prefill with accurate admission; weights are immutable during serving, so preparation and both layouts must be accounted for once.
INTENT: the OpenCL compiler defaults to a language mode without subgroup builtins; the goal requires efficient cooperative reductions; the measured Mali driver supports OpenCL C2.0 and explicit language selection must affect only kernels requesting it.

Goal: make normal API requests near each loaded model's advertised maximum
complete correctly and keep the service usable, with GPU-first serving and an
independent NPU model. Qwen3-0.6B declares40,960; SmolVLM declares8,192 tokens.
Prompt, image positions and reply share that capacity. Prior negative tests are
preserved in rockchip-api-stability-progress.md; they are not completion evidence
for this fix.

- [x] Profile and fix the GPU prefill/attention bottleneck and excessive workspace; cold model-max latency remains limited.
- [x] Support model-declared context limits through API and UI with prepared-weight/KV memory admission.
- [x] Fix startup progress/timeouts and preserve failed-worker identity and recovery.
- [x] Make the VLM decoder efficient at its8K context without reducing output correctness against the fixed RKLLM quality gates (intrinsic long-context grounding remains weak).
- [x] Verify real near-maximum requests and retain answer-quality failures: beginning/middle/end retrieval, streaming,
      repeated and growing histories, exact reply budget, overflow, Stop/disconnect
      recovery and simultaneous independent GPU/NPU serving.
- [x] Verify latency and memory, meaningful regression tests with-n12, mypy,
      lint, browser settings flow and current-state completion audit.

Scope: tinygrad inference kernels and integration, app context/memory/lifecycle
handling, focused tests, live-API verification scripts and reports. Existing local
work is preserved. No archives, commits/pushes, driver/kernel-module changes or
global clock/IRQ changes are part of this work.

INTENT: symbolic token rows reach generic OpenCL matrix multiplication; the goal requires usable near-maximum prefill; Transformer.generate binds true row counts, so padding may add only masked unused computation while preserving those counts.
INTENT: API and UI impose a32768 cap independent of the checkpoint; the goal requires model-declared maxima; the load endpoint already validates context against model metadata.
INTENT: memory admission charges FP32 KV storage; the goal requires feasible maximum-context loads; TransformerBlock._init_state explicitly allocates FP16 KV caches.
INTENT: startup timeout counts total preparation time and drops model identity; the goal requires usable preparation and recovery; RuntimePool documents independent owners routed by retained model ID.
INTENT: optimized CPU attention rejects contexts above4096; the goal includes the8192-position VLM; the attention kernel bounds all accesses by its supplied cache capacity.
INTENT: generic symbolic mask indices use a cumulative-sum expression with quadratic capacity; the goal requires bounded maximum-context memory; the decoder specifies a lower-right causal mask over only the valid prefix.
INTENT: generic GPU attention materializes scores and runs inefficient dynamic reductions; the goal requires usable near-maximum context; cached attention must apply the same scaled causal softmax using FP32 accumulation over the valid FP16 KV prefix.
INTENT: GPU token lookup scans vocabulary and decode uses generic matrix-vector execution; the goal requires responsive serving; nn.Embedding specifies direct lookup and nn.Linear specifies the same matrix product and bias.
INTENT: every image turn discards encoded features and valid KV history; the goal requires usable growing conversations near maximum context; prefix reuse is valid only while the image identity and preceding tokens both match, and metrics must count only recomputed tokens.
INTENT: attention scratch is independently retained for every layer and serving graph; the goal requires maximum-context memory to fit with a separate NPU model; dependent decoder layers and requests execute sequentially, allowing one model-owned workspace.
INTENT: the long FP16 CPU/NPU path omits existing vector softmax and cache helpers; the goal requires usable long prefill; these helpers already have numerical/error bounds and buffer-synchronization guards independent of the linear-weight precision.
INTENT: each Mali attention group rereads the value cache for only two query rows; the goal requires usable full-context prefill; causal attention permits sharing each key/value tile among more queries while retaining independent FP32 softmax statistics.
INTENT: subgroup query selection considers private registers but can exceed 32 KiB of local scratch for small-head high-GQA shapes; the kernel advertises those geometries; its query grouping must satisfy both limits.
INTENT: cancellation clears all KV metadata even after completed chunks; long-context serving must avoid repeating valid work; Transformer.get_start_pos reuses only the matching processed prefix, so generation must publish that prefix after each successful chunk and cancellation must preserve it.
INTENT: the vendor decoder exporter hardcodes 512 positions; the user's matched model-maximum comparison requires 8192 for SmolVLM; the installed toolkit permits context multiples of 32 through 16384, so export an explicitly named baseline at the requested supported capacity.
INTENT: optional NPU placement measures generic GPU linears before Mali preparation; GPU-first placement must measure the actual fallback; prepare Mali first and retain separate GPU projections until combined GPU fallbacks are prepared, accounting for the additional NPU weight layout in admission.
INTENT: placement uses 32 rows for hidden prefill projections and adjusts model.output to one row only after looking up a shape-only cache key; configure documents measured per-phase choices; use the serving chunk for hidden projections and include the effective row count in that key.
INTENT: short GPU probes measured a 64-row projection at 12.4 ms while a sustained matching kernel measured 4.6 ms; GPU-first placement must compare representative execution; add bounded GPU warmup before timing either alternative without changing clocks.
INTENT: long 64-channel CPU prefill reuses KV across four query rows; eight-row grouping with separated score rows measured 84 ms versus 102 ms with the same numerical result; reuse eight rows only for the tested long-context geometry while preserving each query's arithmetic order.

The wider-query experiments were rejected and reverted: local-query reuse measured
254 ms and a larger private-register query group measured 180 ms versus the
retained 175 ms block at position 8,064, 128 rows, context 8,192. Correctness alone
was insufficient to select either candidate.

## Current measured results (verification still running)

Both declared maxima now load together: Qwen 40,960 on OpenCL, SmolVLM 8,192 on
CPU/NPU. Qwen reaches ready in 53.37 s, 8,265,654,272 bytes RSS. SmolVLM reaches
ready in about 346 s while Qwen remains resident; available RAM is about 1.15 GB.
This corrects the prior schema rejection, duplicated attention scratch and
startup-progress timeout. It does not establish maximum-context interactivity.

The real Qwen 40,952-token prompt plus eight-token reply receives no text within
180 s. The diagnostic cancels it; Stop/disconnect recover in 0.38/0.36 s. The full
cold/cached/growing maximum-context conversation test is queued after the VLM
matrix. Its longer deadline measures completion separately from the failed
three-minute interactive-use check.

The current VLM 8,184-token request completes in 104.72 s, prefill 77.95 tok/s,
decode 10.52 tok/s. Its answer is `The secret word is "O".` rather than ORCHID:
transport/admission pass, retrieval quality fails. Its input SHA256 is
6a12648aa2deb54f5649fc10867ba0278f03b6a415313530e571668d508a15d8. The earlier FP32
reference's `Secret word: 13` matches a DIFFERENT saved payload; it cannot be used
to assert numerical parity for this new request. Independent verification of the
new input is still required. Image features were cached; only 73 text-prefix
positions were reused, so this remains a substantially uncached decoder prefill.

Browser validation accepts each declared maximum and rejects maximum+1, with no
JavaScript exceptions or saved-setting changes. Regression: 58 passed, three
hardware-only skips with -n12; separate Mali numerical runs passed five tests.
Mypy passes 239 source files; Ruff and tracked whitespace checks pass. Final
checks will be repeated only where later edits require them.

TWINS: searched 32768/32_768 schema/UI caps, fixed-512 context restrictions,
load_started/300 timeout checks and CPU attention 4096 limits across app code,
tests and both backend attention selectors - found no remaining schema/UI cap;
the 512 precision qualification remains intentional and independent of capacity.
Historical verification docs retain old limits and will be distinguished from
current evidence; matrix dimension limits of 4096 remain unrelated to context.

The final VLM bounded matrix completes 17 checks in 612.35 seconds with zero API
failures and zero health-poll failures. All 13 scored answers are retained: eight
correct, five incorrect. A four-turn history reaches 8,184 prompt + eight reply
positions and answers ORCHID correctly. The identical cached 8,184-token prompt
reuses 8,183 positions, reaches first text in 143.48 ms, completes in 0.79 s and
returns exactly the preceding uncached answer. Stop/disconnect recover in
0.211/0.209 s; both fresh short recovery requests answer correctly. Long mostly
uncached VLM prefill ranges 61.83–77.95 tok/s under the measured thermal conditions.
The quality score is not promoted to a pass merely because the API works.

Independent verification of the new length-8,184 payload completes. Its input
hash exactly matches the API run; the original FP32 checkpoint generates token
IDs [378, 4911, 2229, 314, 476, 63, 2227, 49279], decoding to the same
`The secret word is "O".` plus the normal end-of-utterance marker. Thus this
particular retrieval failure is present in the original checkpoint. The other
four failed probes are still scored as failures and are not individually proven
against FP32. The independent implementation first validates against the earlier
Transformers image/logit fixture (NRMSE 4.94e-6/4.40e-6).

The final Mali hardware regression, including high-GQA small-head scratch bounds,
passes all five tests with -n12 (22.38 seconds). Superseded streaming-attention
experimental code was removed. The live Qwen worker is reloaded after that cleanup
and before the remaining cold/cached/growing test, so the final serving source is
used rather than a stale imported module.

The 180-second threshold is a diagnostic chat budget, not a user-specified
latency SLA. The user has been asked for an optional latency preference while
work continues; no answer or approval is inferred from elapsed time. The longer
GPU test remains separate and does not turn the earlier deadline failure into a
pass. Both models have been restored to their maxima on the final source. During
GPU cold prefill, the independent NPU model answers a fresh short request in
0.126 seconds. Detailed immutable results and source hashes are recorded in
rockchip-long-context-benchmark.json; the ongoing GPU report remains explicitly
pending until it actually completes.

A later audit found another usability defect: Stop discarded all KV metadata,
so retrying a long conversation would pay cold prefill again. Core generation now
truncates metadata at the matching prefix before writing divergent history and
publishes each completed chunk. Image prefill publishes the same progress, and
worker cancellation preserves it; genuine errors and image changes still clear
history. Two focused tests (plus two subtests) pass with -n12: cancellation before
the first divergent chunk or after two chunks, and interrupted image prefill;
resumed generation matches a fresh full recomputation. The new image fixture
initially returned a device-less constant; it was corrected to explicit CPU
storage before passing. No product failure is claimed from that fixture error.

This cancellation fix is newer than the workers' imports during the ongoing
GPU cold measurement. The original source hashes are retained; pending source
hashes are explicit in the JSON report. Both workers still need a reload and live
Stop/retry verification before this later fix can be called applied. The normal
arithmetic kernels did not change. Additional searches find only intentional
warmup, scene-change and genuine-error history resets; the cancellation reset
has been removed in both text and image paths.

User clarification: usable means **faster than RKLLM** for uncached prompts near
the model maximum, with the earlier comparable-quality W8A8 target retained.
The arbitrary 180-second diagnostic is not the acceptance target. The Qwen cold
measurement completes in 4,884 seconds, prefill 8.36 tok/s, correct ORCHID; cached
repeat first text 1.55 s; a growing 40,952-token prompt reuses 40,828 positions and
answers NEPTUNE correctly in 27.60 s to first text. This establishes bounded
transport and prefix reuse, not the user's speed target. A matched long-context
RKLLM baseline and further performance work are required before completion.

The installed RKLLM 1.3.1 toolkit's build API explicitly limits max_context to
16,384 (multiples of 32). The existing Qwen W8A8 artifact has that cap; the shared
FP16 artifact has a 512 cap. Both reject requested context 40,960 before inference.
A speed claim at Qwen's 40,960 maximum is therefore unavailable for this installed
SDK. Measure its largest common supported context (16,384) and report the oNPU
40,960 capacity separately. SmolVLM's 8,192 maximum is within the toolkit limit,
but its current vendor artifact was compiled at 512 and needs a new 8,192 build.
No checkpoint maximum or test expectation is reduced to manufacture a win.

A new SDK runner accepts explicit context and verifies both SDK prefill count and
retained KV size, so silent truncation cannot be mistaken for faster inference.
The first W8A8 diagnostic uses the exact 16,376-token raw prompt, eight generated
IDs, empty KV history, and the installed three-core artifact. It is diagnostic,
not a qualified paired timing/quality win. API cancellation regression now passes
60 tests plus two subtests with three hardware-only skips, using -n12.

The first matched common-context candidate completes three uncached 16,376-token
requests with identical raw output IDs. Median prefill is 20.55 tok/s versus the
single SDK diagnostic's 20.09; median decode is 3.87 versus 0.657 tok/s. The 2.3%
prefill lead is too small for a strong win claim without matched repetition and
quality qualification. No completion or universal speed claim is made.

Input-stride padding was tested in the prepared Mali matrix kernel, including
the padding copy itself. All padded outputs are bitwise identical, but timings
are effectively unchanged for 3072x1024 and slightly worse for the other two
shapes. That candidate is rejected; no serving kernel was changed. The diagnostic
also shows sustained GPU matrix times differ from short cold-device calibration,
so representative calibration warmup remains a concern rather than a proven fix.

The 8K VLM vendor conversion is legitimate toolkit export from the same decoder
and calibration data. Its emulated optimization stage takes about two minutes
per layer. The private converter container is paused during hardware timings to
avoid competing CPU/thermal load; its 1.1 GiB resident state is retained. It must
be resumed after profiling; no baseline artifact is patched to bypass limits.

Current-source regression: 61 passed plus two subtests with -n12; mypy passes 239 source files; whole-backend Ruff and whitespace checks pass. The main API was restarted onto the new admission logic while both workers were unloaded. The vendor quality job owns the NPU and most RAM, so admission correctly rejects loading another large model; this is not an API stability failure. Live cancellation/retry verification follows exclusive benchmark release.

INTENT: the VLM benchmark hardcodes 512 and does not verify consumed prompt positions; the user requires comparison at the real 8192 maximum; accept explicit context and bounded short warmups, and verify SDK prefill and KV counts on every measured full prompt.

The long-quality inputs are fixed held-out WikiText-2 tokens from the PyTorch examples mirror, SHA256 d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0. Each model scores 256 late-context positions with the unchanged 1.02 perplexity-ratio gate. Qwen uses the actual Q8 GGUF and measured GPU/NPU placement. VLM uses the original checkpoint and image-conditioned prefix; vendor image features are saved separately. The quality job is not used as timing evidence while conversion runs concurrently. Its small seed-mode probe exactly matches ordinary GET_LOGITS subsequent logits. Fixture preparation required explicit bit conversion of BF16 storage on this ARMv8.2 compiler; those fixture errors do not constitute serving regressions.

The cold API VLM timing helper uses valid JPEG comments to change the image identity while preserving identical decoded RGB bytes. Every measured request must report zero cached text positions and uncached vision; SDK timing verifies consumed prompt and retained KV counts. This prevents accidental cache reuse from being reported as faster cold inference.

The first long SDK quality seed was stopped after 19m20s because hidden-state mode allocated full-prefix output storage: RSS 10.5 GB, process swap 1.77 GB and about 427K major faults. This unfinished run is not timing or quality evidence. Chunked hidden-state seeding was rejected: subsequent logits differed from the full-prefix reference (max error 2.97, NRMSE 0.229). A normal GENERATE seed with exactly one uncommitted output instead matches subsequent reference logits bitwise, with verified KV length. The active quality run now uses that full-prefix path with a 7200s supervisor limit. The rejected chunk option was removed from the source.

Current-source live GPU cancellation passes at context 40960: Stop recovery 0.969s, a shortened retry retains 384 processed prefix positions and answers NEPTUNE correctly in 4.118s. Raw report: /home/orangepi/vlm-bench/fix-context-current-gpu-cancel.json. The helper wrapper initially called a nonexistent model-specific unload route (404); the test itself completed successfully, and cleanup then used the documented /api/models/unload?model= route. GPU worker is unloaded again for exclusive quality/performance runs.

The full-context VLM W8A8 route is still disabled pending qualification. Evaluation tools now support the real context for both the existing 20 image cases and 256 held-out late-context positions. No production precision whitelist has been expanded.

The next bounded timing comparison will retain the original repeated-records input and add a naturally varied held-out WikiText input, both exactly 16376 raw tokens plus eight generated IDs at context 16384. The latter is derived mechanically from the already-fixed quality sample; no candidate timing has been observed for it. Order is SDK records, oNPU records, oNPU WikiText, SDK WikiText, each with three 64-token graph warmups and initially one measured cold request. This controls both workload selection and engine order; repeat if a lead is within observed variability. These timings wait until conversion and quality jobs have exited.

A separate maximum-context check now covers the default GPU VLM profile. It loads context 8192 in 165.29s, RSS 3.24GB; the exact original 8184-token image payload completes in 131.50s, zero cached tokens, uncached vision 4.21s, text prefill 65.71 tok/s and decode 6.78 tok/s. It returns the same O answer as FP32. This is a diagnostic under concurrent SDK quality/conversion load, not a matched speed win. The image encoder is not the bottleneck. Head-geometry-specific Mali attention experiments all pass NumPy checks but do not improve the retained 49.60ms default: q1 59.23ms, q3 49.92ms, other variants 52.38–55.14ms. No serving attention kernel changed. Live GPU VLM Stop/retry now runs after an explicit scene/history reset and checks the request is still active before Stop; its retry must exactly match the completed baseline answer.

Current-source GPU VLM cancellation also passes at context8192. The request is observed active before Stop; recovery takes1.389s. The retry reuses713 positions and exactly matches the completed baseline answer. It takes175.43s under a simultaneous separate GPU quality workload; this is correctness/cancellation evidence, not fair timing evidence. Its app worker is then unloaded, and a pure-GPU full-context quality dump starts.

The SDK 8K artifact finishes legitimate export: toolkit1.3.1, source74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e, decoderfdba81a8fcb7db6af443a04bcb0451fc9bcccd2f1756345e4e55fb76743baa5f, artifacta989af8c4e9726a4f59f907b854b296e5eb233f5953790bdbb3cb131ddf2a962. Provenance is /home/orangepi/vlm-bench/vendor/text-ctx8192-provenance.json. The private converter container exits and is removed; no container remains paused. The pure-GPU VLM 256-position long-context dump also finishes, with no errors. Its comparison waits for the SDK dump; a matching 20-case image quality evaluation now checks actual GPU greedy answers.

The first candidate Qwen quality seed was stopped with SIGINT before its incompatible fixed-shape final chunk. A direct JIT probe confirms the mismatch; the evaluator now uses Transformer.generate, exactly like serving, rather than a hand-written fixed-size loop. A191-token small-model probe with64-token chunks retains all191 KV positions; subsequent teacher logits match full recomputation with max absolute error2.98e-7. No serving kernel changed for this harness correction. The incomplete candidate-v2 report carries KeyboardInterrupt and is not used as quality evidence. VLM comparison proceeds independently first. The new SDK VLM probe at context512 confirms exact128→129 KV lengths and only the expected49279 EOS logit suppression; no extra EOS mask was introduced.

INTENT: decode pooling excludes fewer than four KV heads, leaving SmolVLM serial at8K; the measured three-head pool reduces median2.641ms to1.112ms with bitwise-equal output and no idle-board outliers; allow two/three-head pooling only for long configured contexts and at least1024 valid positions, preserving FP32 arithmetic and FP16 KV.
INTENT: SmolVLM INT8 is restricted to512 by prior qualification; the fixed8192 evaluation passes256 late-context teacher positions and20 image cases (8 correct vs SDK7, perplexity ratio0.97313); extend only this exact checkpoint hash through8192, retaining the512 default for other qualified checkpoints and FP16 for the GPU profile.
The prefill row-job and wider-register candidates were all bitwise correct but slower than the original kernel (95.84ms baseline vs104.88/117.27/102.09ms). They are rejected. The accepted decode-pool change is independently repeated idle: median2.641→1.112ms, mean2.629→1.104ms, maximum2.694→1.219ms. Current source still needs the new targeted regression and actual API measurements; historical quality hashes are retained.

The first actual INT8 API timing reports cpu_pool:false: setup still excluded three KV heads, so the primitive-only fix was not active. That run is retained as an unpooled baseline, not a successful end-to-end optimization. The setup guard is corrected too, with an integration test asserting the actual block receives its pool and short contexts retain their original path.
TWINS: searched kv_heads>=4, n_kv_heads>=4 and kv_heads<4 across tinygrad/llm/kernels and onpu - found one other exclusion in tinygrad/llm/kernels/rockchip.py; both setup and execution now allow the qualified long two/three-KV geometry.

INTENT: long prefill repeatedly loads ordinary KV rows and normalizes every score before PV; the target requires more than the measured95.5 tok/s versus SDK160.2; port only the measured packed KV, feature-major query/score and late-normalization path for the9/3-head64-channel decoder, behind an explicit opt-in until cache/state/native and quality gates pass. The best prepacked primitive measures46.23ms vs84.14ms, with FP32 arithmetic. Packing costs are not yet included in that primitive number.
The packed-cache tests initially found literal scalar inputs lacked required buffer state; those arguments now use buffered lists. Their CPU-only replay test incorrectly expected mixed NPU-native replay; the implementation explicitly returns no NPU work, so the test now verifies that documented fallback. Actual mixed-model native execution remains a required hardware check. A separate decode-rewrite test exposed an AFTER on a lazy COPY; the cache is now materialized before the invalidation effect. No failed check is counted as passed.

The first packed full-model seed uses59 native replay calls and retains the correct7935/127 state in all30 layers. It then catches an eager cache.realize inside the decoder, which @function forbids. The invalidation path now uses a deferred STAGE bufferization and its regression traces through @function. That failed run remains complete:false with the exception, and no packed production policy has been enabled.

The function-tracing regression now mirrors TransformerBlock exactly: @function(precompile=True, allow_implicit=True), which permits its model-owned buffers while still forbidding eager device execution. The first test omitted allow_implicit and rejected the intended owned cache. This fixture correction does not relax device-usage or numerical checks. Deferred bufferization is restricted to inputs without storage; the model keeps its existing KV allocation.

The first API packed run is a negative result: Auto selected only91 of211 prefill projections, versus211 in qualification. NPU calibration reported13–23ms rather than about1ms and rejected Q/K/V/output projections; fused CPU fallbacks then took3.67s for128 positions and285.23s for8184. That measured run is retained, and the following timing was stopped.
INTENT: placement calibration omits the wake helper used by warm native inference and uses fixed rows while serving uses symbolic rows; the GPU-first policy requires measuring the actual candidates; initialize and scope the existing per-process helper during CPU calibration and bind the serving row range, without changing clocks or IRQ settings.
The new placement regression checks helper activation during NPU probes and that hidden prefill inputs carry symbolic row bounds. The existing runtime helper remains scoped to inference calls; no global policy changes are made.

The placement/runtime mismatch is now covered explicitly. Calibration inputs use the serving symbolic row range, and the existing CPU0 wake helper is initialized before calibration and scoped around both candidates. A regression checks that NPU probes execute with the helper active and that hidden prefill carries symbolic row bounds. Three placement tests pass, mypy240 and configured Ruff pass. The next API run requires all211 NPU prefill projections and compares execution configuration fields with the accepted quality run before timing.

The accepted FP32 packed attention passes both quality suites: long-context ratio1.005871 (256 positions) and20 image cases8 correct vs SDK7. GPU-vision variants also pass (long ratio1.012988). Actual calibrated API timing selects all211 prefill/decode projections and30 packed blocks. Three cold8184-token trials measure136.33/124.07/106.53 prefill tok/s and18.52/15.82/18.41 decode tok/s; medians124.07/18.41 vs SDK160.17/17.68. Sustained CPU throttling from1.8GHz to1.416GHz accompanies85.9C; no clocks are changed. The prefill target is not achieved.

INTENT: QK remains a large CPU prefill cost; an opt-in FP16 SIMD dot with FP32 softmax/PV has a faster local primitive, but requires unchanged end-to-end quality gates before enabling. The first integration test caught a dispatch/source mismatch (half job partition called FP32 part); source selection is corrected. The app keeps the experiment disabled.

Experimental FP16-QK now passes full/partial near-max, masked-future and large-query fallback unit tests (five tests, -n12, DEV=CPU, OPENBLAS_NUM_THREADS=1). The first combined invocation lacked explicit DEV=CPU and failed CPU-only fixtures against the default device; it is retained as a harness/config failure. The corrected combined CPU suite passes20 tests, app suite25, mypy240, Ruff.
The FP16-QK CPU-vision full-model gate passes:256 late-context positions ratio1.0028221546, native replay59 runs and all30 cache headers7935/127. Its20 image cases pass (8 vs SDK7; PPL ratio1.0130409145). GPU-vision qualification remains running; the API still uses accepted FP32 attention.

FP16-QK GPU-vision qualification also passes unchanged gates: long ratio0.9949606007,20 images ratio0.9923764243 with7 correct vs SDK7. The app enables it only alongside the qualified INT8 long-context CPU/NPU packed route. Fresh API controller waits for ready metadata requiring211/211 projections,30 packed blocks, pool andcpu_fp16_qk before three cold timings, active Stop/retry and boundary/cache/history checks. Qwen GPU is unloaded during fair standalone timings, and must be restored at40960 afterward.

The qualified FP16-QK API worker reports211/211 projections,30 packed blocks, pool andhalf-QK active. Three cold8184-token requests each verify zeroKV reuse anduncached identical decoded pixels: prefill144.31/131.42/124.50 tok/s (median131.42), first textmedian63.73s. Prior exactFP32median124.07; SDKmedian160.17 and51.93s. The target remains unmet. Decode24.53/23.16/11.04 varies substantially and counts differ fromSDKrawIDs, so no qualified decode-win claim. Temperatures reach85.9C with variable CPU/NPU clocks, unchanged policy. Stop/retry andboundary/cache matrix now running; Qwen40960 GPU reload starts after standalone timings finish.

Final audit catches a serving regression not covered by the four numerical quality gates: exact-context-total uncached answers Secret word:44, but identical cached boundary input answers Secret word:13 attemperature0. FP16-QK is disabled inworker again. The stability tool now requires exact cached/uncached greedy output equality in addition to cache-position reuse. The14 transport checks andactual8192 usage passed, but do not qualify this candidate for serving. Its131.42 tok/s result remains experimental negative evidence. Accepted FP32 packed attention is reloading for the strengthened cache check; both model maxima must be ready afterward.

Final accepted FP32 rollback verification:10 API checks,0 API failures,0 health errors. Cold/cached boundary answers match exactly (bothO; wrong retrieval), cached8183 positions, first text0.196s. Actual8191+1=8192 usage passes; overflow, Stop, disconnect and recovery pass. Retrieval3/5 correct is retained as a limitation. Current independent-serving proof: GPUQwen40960 prefill active while NPU8192 HELLO returns0.112s; GPUStop0.805s. Both workers finish ready with GPU selected and saved settings unchanged. Reportsfix-context-final-fp32-stability.json, fix-context-final-fp32-independent.json andfix-context-final-restored-status.json. FP16-QK remains disabled duecache-output regression. Accepted FP32 three-run baseline124.07 vs SDK160.17 still misses the user target; one fresh FP32 verification request135.82 is not a substitute for that matched baseline. Overall goal remains incomplete for performance.

Continuation: prior turn is progress (cache regression found, rollback and10 accepted-path checks observed). Current sources confirm both maxima ready andhalf-QK off before profiling. New native attribution records53.0 of63.0 seconds in CPU attention; NPU matrices take only a few seconds. Feature-vector grouped FP32 is bitwise equal but only54.4→51.5ms. Weighted little-core variants and time-blocked PV are slower; no serving changes from those experiments. Integer score dots retain FP32 softmax/PV and show54.8→37.8ms with fixed primitiveNRMSE gate0.02 (observed0.000780). This is not a model-quality or speed qualification.

- [x] Attribute accepted8K native execution.
- [x] Reject measured non-winning FP32/threading variants.
- [x] Port INT8 score-dot as opt-in, uniformly across prefill/decode with stateful cache tests.
- [x] Run unchanged held-out late-context/image gates and cache-output equality.
- [x] Compare sustained public-API performance to RKLLM and restore both maxima.
INTENT: accepted FP32 attention consumes53 seconds of63-second8K prefill; the user requires faster-than-RKLLM W8A8 at comparable quality; the attention contract requires causal FP16 KV inputs, FP32 softmax/value accumulation and consistent cached output. Add an opt-in per-row INT8 score dot with identical semantics in prefill/decode; serving stays FP32 until all gates pass.

The ported INT8 score-dot primitive includes incremental packing and measures54.17→37.19ms, fixed syntheticNRMSE0.000732. Its25 backend runtime/placement tests pass;25 app regressions pass; mypy241 andRuff pass. The first pooled test exposed a five-pointer callback against the existing four-pointer pool; fixed by carrying row count through its integer position argument, thenallfournewruntime tests pass including actual pool-call count andbit-identical batched/single/pooled outputs.
TWINS: searched args[4], args[5] andvoid*args arrays inRockchip kernels - found0 other five-pointer attention-pool call sites; rockchip_cpu.py androckchip_gpt2.py use four pointers.
Full-model late-context INT8 score-dot gate passes256 positions atPPLratio0.9995863941 with59 native runs and30 cache headers7935/127. All211 prefill/decode projections and30 packed blocks are active. Image cases andGPU-vision gates are stillrunning. An explicit ONPU_EXPERIMENTAL_INT8_QK=1 worker opt-in is available for isolated API qualification; normal serving defaults remain FP32.

INT8 score-dot final qualification: allfour unchanged quality gates pass (longCPU0.999586/GPUvision1.006175; imageCPU0.993698/GPUvision1.007301,9/7 correct versusSDK7). New matched pixel fixtures use the exact API tensor expression; the copied old fixture failed bitwise validation by oneFP32ULP andwas replaced before timing. Two fixed8184+8 paragraph workloads each use3 short128 warmups+3 uncached measurements per engine, exact8 natural output IDs andfullKV counts. Records mediansoNPU173.82/31.72 vsSDK157.77/17.11 tok/s; WikiText165.80/30.50 vs158.36/16.46. Every candidate prefill exceeds every reference rate onits workload. Clocks/temperature recorded; no policy changed. Both engines hallucinate a song for the astronaut image under WikiText distraction: model grounding remains a limitation.
The normal worker enables INT8 QK automatically only onthe existing qualified long-context INT8 packed route; the temporary environment opt-in is removed. All25 backend tests (including physicalend8192) and25 app tests pass with-n12; mypy241 andRuff pass. Main API completes8192-total cold/cached identical caption answers, preserves8183 cachedpositions, serves independentNPUHELLO0.106s duringQwen40960 GPU prefill, andGPUStop recovers0.706s. Both max workers ready, GPUselected, saved settings unchanged; two-second idle model-subtree sample0%CPU. Feature evidence:docs/rockchip-dot-attention-benchmark.json. Full goal remainsactive forQwen common16K qualification andthe remaining overall audit.

Qwen continuation: previous turn is verified progress (qualified INT8 score-dot, matched8K speed wins, normal API cache/dual-owner verification). Current API confirms both max workers ready andnewscore mode active before work.
INTENT: Qwen quality evaluation unconditionally enables NPU Auto while the app default GPU profile has both NPU flags off; the task requires evidence for the actual serving path; onpu.worker configures optional NPU only when requested. Honor the existing --gpu-npu flag for Qwen too, keeping the1.02 PPL gate unchanged.
TWINS: searched placement=configure(model) andGPU placement branches inquality_long_context.py - found one other GPU branch forVLM, already conditional on--gpu-npu.

Qwen GPU-only16K quality passes256 late-context positions: PPL21.760706 vs RKLLM21.716699, ratio1.0020264066 <=1.02. Report placementNone andgpu_npuFalse prove the actual default path. Integer-dot variants are numerically close but no useful speed gain; FP32 split is bit-identical butslower(132.99 vs131.04ms). Its stage attribution is70.43msscores/softmax and60.12msPV. Fullreal-model late-step attribution is3.589s attention of4.456swall, with0.565sGEMM. FP16partial dots improve132.19→120.79ms but introduce precision loss. Workgroup ordering improves130.91→124.29ms with bit-identical output; validating40Kbeforechoosing the exact-math change.
INTENT: benchmark_inference wraps linears in disabled NPU wrappers even when both phases are off; the task requires timing the actual default GPU serving path; onpu.worker skips NPU configure entirely when both options are off. Skip that optional configuration in this benchmark too, without changing timing/quality gates.

Qwen defaultGPU quality isnowobserved:256 late16K positions, PPLratio1.0020264066. Workgroup-order experiment isbit-identical for64/63/7/1 rows atboth16K and40960. At16K KV/query/part order measures124.29 vs130.91ms; at40960 query/part/KV measures313.97 vs332.01ms. FP16score experiment onlyabout9% faster andchangesprecision, so it isnotselected overthe small exact-math dispatch improvement.
INTENT: qualified Qwen attention dispatches KV parts before nearby query tiles; the task requires faster near-max serving; mali_attention.py specifies causal FP32 online softmax with bounded scratch. Use the measured workgroup orders only at the two tested Qwen geometries, preserving arithmetic, allocation andsingle-token decode ordering.

The Qwen dispatch change passes six hardware Mali tests (-n12), mypy (241 files) and Ruff. Its 256 full-vocabulary late-16K logit files are byte-identical to the preceding qualified GPU path (perplexity ratio 1.0020264066). The timing pilots below each have one measured cold run per engine/workload.
INTENT: the model-max API harness checks only an end-of-prompt needle; the user requires usable long context. Add an optional shared-prefix beginning/middle/end sequence, cached-answer equality, actual 40,960-token usage and overflow rejection, preserving the cold measurement and independent-model query. This test waits for exclusive SDK timing and both restored workers. No serving numerical change accompanies it.

Fresh reversed-order Qwen SDK pilots complete. Exact input-ID equality is verified for both 16,376-token fixtures, with eight raw outputs and SDK KV length 16,383. Records: oNPU 23.1858/3.8051 versus RKLLM 19.7913/0.56897; WikiText: oNPU 22.3820/3.9483 versus 20.0041/0.69244 tok/s (prefill/decode). These are single pilots; no repeated-median or 40,960-speed claim is made. The quality gate passes at 1.0020264066. SDK exits before main max-worker restoration starts. Raw paired report: qwen-current-paired-pilots.json.

Current 8K NPU retrieval/recovery checks finish: seven API passes, zero failures; long retrieval is 1/3 (end correct, beginning/middle wrong), both short recovery replies correct. Exact payload hashes reconstruct both failed cases. RKLLM also fails both, with different wording; these concurrent accuracy diagnostics are not speed evidence. The SDK processes exit normally. NPU reload is accepted with HTTP202; a wrapper incorrectly asserting200 stops, then a separate readiness observer follows the already-started load. Qwen40K cold remains active throughout.

Current-source 40K matrix completes seven API checks with exact40960 usage andmatching cold/cached output. Firsttext3763.16s cold,1.601s cached; prefill10.851tok/s, decode2.450tok/s. IndependentNPUHELLO0.107s. Beginning/end multi-needle answers are wrong; middle answer is truncated atNEPT bythe8-token budget, so that case is inconclusive. The next32-output-token fixture retains these failures ratherthan replacing them withapass.
INTENT: the worker clears cached history for a ValueError raised by reply-budget validation before inference. The task requires usable long-context recovery; validation has not submitted any KV writes. Preserve cached metadata when started isNone; inference errors still invalidate it. The final nonstream replay after overflow exposed the bug bybecoming cold. It was stopped, preserving its partial work. Add cached-after-overflow checks toboth live harnesses. Numerical kernels andquality gates stayunchanged.
TWINS: inspected worker.py ValueError/Exception handlers, model.get_start_pos andruntime release/cancel paths. The pre-inference ValueError handler is the cache-loss site; genuine generation exceptions retain invalidation.

Both workers reload onthevalidation fix. LiveGPU atcontext40960 rejects theactual40952+9 overbudget prompt inboth stream/nonstream forms, thenreuses the545-position cached reply withidentical NEPTUNE output in1.495/1.448s. The new40K matrix starts witha32-token reply budget; independentNPUHELLO takes0.103s duringit. App25 tests pass(-n12); changedPython Ruff passes. Reversingonlythe newguard exactly reproduces theprevious qualified worker SHA d283b696..., so normal numerical serving code isunchanged.

The32-token fixture needed more inter-turn space; this was caught during the first few minutes andthe attempt was stopped before scoring. Its partial report remains qwen-fixture-spacing-attempt.json. The helper now preflights all turns with maximum-length dummy assistant replies before submitting expensive inference. Fresh GPU reload clears partial state. Actual tokenizer preflight passes40704/40832/40928/40959 prompt positions; final40959+1 andoverflow/replay cases are unchanged. Fresh run starts withzero populated model cache; independentNPUHELLO0.118s.

Current validation fix completes both live matrices: eight GPU checks and eleven NPU checks, all API passes. GPU actual usage is 40959+1=40960; after both rejected overflow forms, the replay retains 40958 cached positions and answers identically in 1.594s. Cold 40704-token first text takes 3336.74s (55.6min), prefill 12.200 tok/s; identical cached first text 1.579s, full nonstream replay 5.579s. Stop 0.272s, shortened retry 1.666s, disconnect 0.394s; independent NPU HELLO 0.118s. Maximum-context retrieval is 1/3 for each model. Qwen replies are untruncated with 32 output positions available; its 40K failure cause is not attributed without an independent reference.

Final source audit finds zero numerical mismatches versus measured cohorts. The worker differs only by the pre-inference validation guard, verified by reverse-patch hash. Twenty-five focused app and 31 backend/Mali tests pass with -n12; current mypy (241 files), backend/changed-app Ruff and both whitespace checks pass. Both advertised maxima finish ready, GPU selected, saved settings unchanged; 20 model processes idle at 0% CPU over 2.018s, and both workers have zero swap. All owned benchmark/SDK/recovery jobs are terminal.

The API assessment is complete. The full user target remains unqualified: the installed RKLLM artifact/toolkit has the same 16384 cap recorded across earlier quality/throughput goal turns, so direct 40960 speed/quality qualification requires a supported reference. Existing 8K VLM three-run medians and 16K Qwen pilots remain scoped wins, not a full model-max claim. Current evidence is qwen-final-assessment.json and the current section of rockchip-long-context-benchmark.json; earlier attempts remain explicitly historical.
