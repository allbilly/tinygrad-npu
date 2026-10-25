# oNPU inference and RKLLM parity

The user clarified the target: port only code needed for oNPU to match or beat
RKLLM in both prefill and decode. Full operator parity with the blog/1500 branches
and an arbitrary pure-NPU tensor backend are not required for this target.

OpenCL remains the default. NPU assistance should improve measured inference,
and GPU-only serving must coexist with a smaller NPU-assisted model.

INTENT: the working matmul integration has modest application gains and no matched RKLLM parity result; the task requires competitive end-to-end LLM prefill and decode; the user's clarified scope calls for selective inference code rather than porting every reference operator.

- [x] Establish matched current-tinygrad and RKLLM baselines for both phases.
- [x] Attribute time to kernels, packing, transfers, synchronization and dispatch.
- [x] Port and qualify performance-critical kernels, layouts, fusion and replay.
- [x] Match or beat RKLLM on repeated matched prefill and decode workloads.
- [x] Validate outputs, GPU-first placement, independent sessions and UI behavior.
- [x] Run applicable regressions, mypy, Ruff and reference-independence checks.

The initial stock FP16 three-core sample measured median 14.315 decode tokens/s
and 508.758 ms first-token latency for 128 input/32 output tokens, two full
warmups and three measurements. Timings drifted; this is not a controlled paired
comparison. Optimized C-engine results are references, not tinygrad results.
W8A8 stock performance must be reported separately from matched FP16 results.

The unfinished general renderer experiment was removed from the working backend
after the scope clarification. Its source is preserved in the temporary archive
/tmp/onpu-full-20261018/general-port-attempt.tar.gz. Both reference worktrees
remain intact. The last broad trial stopped with 20 failures, 31 passes, two
existing skips and 28 passing subtests; no full-operator acceptance is claimed.
Five host UOp tests failed identically on untouched HEAD and the experiment.

extra/rockchip/benchmark_inference.py measures raw-token requests with explicit
placement, source hashes, checkpoint hash, clock snapshots, full warmups and
optional comparison against stock-generated IDs. The idle app NPU worker remains
unloaded for serial hardware experiments; its GPU worker is preserved.

Selective implementation progress:

- The current 128-input/32-output CPU+NPU baseline completed two full warmups and
  three measurements. All five generated sequences matched the stock IDs.
  Median first-token latency was 128324.075 ms and decode was 4.0207 tokens/s.
  This establishes the current gap, not parity or a controlled paired result.
- Opt-in wider decode tiles and shared-input QKV/gate-up projections pass the
  complete-output hardware checks on CPU/OpenCL inputs, JIT rebinding, mode
  transitions and phase fallback. Batched prefill uses explicit packed-weight
  byte offsets rather than requiring a general NPU movement renderer.
- Opt-in CPU decode attention passes symbolic length and rebinding checks against
  independent FP32 goldens. An isolated 128-position sample measured median
  0.582 ms versus 1.387 ms for the original CPU path, with matching outputs.
  Full-model correctness and performance with this path remain pending.
- Current checks: 19 host/tokenizer tests pass; two CPU attention tests pass;
  three NPU hardware tests pass; existing CPU custom-kernel/JIT regressions give
  115 passes, 25 skips, six expected failures and three passing subtests.
  These runs use -n12. The stock comparison and production application still
  need qualification after selecting optimized paths.

TWINS: searched attn_qkv/ffn_gate_up/decode_attention and reference-folder imports
across retained LLM kernels/model/runtime; the existing attn_qkv belongs to the
separate gated-delta block, and no retained inference code reads a reference folder.

The complete captured FP16 register initialization passes 30 reset-free full-output
probe cases across M=1/16/64 and K=1024/3072. The retained runtime now generates
that 108-write state sequence with runtime geometry and DMA addresses, clearing
unused fields and avoiding resets between matrix submissions. The complete
hardware suite passes again with this sequence (three tests, 14.02 seconds).
The optimized 128-token model comparison is running with wide tiles, projection
fusion and CPU decode attention; no end-to-end performance gain is claimed yet.

The optimized 128-input/32-output run completed two full warmups and three
measurements, matching every stock-generated ID in all five requests. Median
first-token latency is 2864.696 ms and decode is 5.6779 tokens/s. This improves
the initial tinygrad sample by about 44.8x in prefill and 1.41x in decode, while
remaining slower than the preliminary FP16 stock sample (508.758 ms, 14.315 tok/s).
Raw results and complete register probe data are retained in
`docs/rockchip-inference-progress.json`; exact executed source hashes were checked
and snapshots retained under /tmp/onpu-full-20261018/{baseline-sources,selective-128-sources}.

A separate eight-output-token diagnostic attributes inclusive host costs. Its
steady prefill has 2013 ms in CPU kernels, including two attention reductions
that total 1566 ms; NPU programs account for 418 ms. Decode's seven steps have
394 ms in NPU programs, 149 ms in CPU kernels and 58 ms in NPU copies, within a
roughly 1266 ms total span. Nested submit/sync counters overlap these totals and
must not be added again. The remaining host scheduling/argument work warrants
compiled replay, while prefill attention needs its own optimized route. The
current captured step contains 403 CPU kernels, 119 NPU kernels and 233 copies.
The profiling observer's initial atexit export failed; export now runs before
interpreter teardown and the repeated diagnostic retains the complete profile.

The CPU attention compiler now builds its target without accessing Device during
precompiled model tracing. Three CPU attention tests, including a precompiled
model function, pass. Current mypy and Ruff checks pass. Application integration,
controlled bracketing, the 256-token workload and RKLLM parity remain pending.

INTENT: temperature-zero generation still computes full-vocabulary Gumbel noise; the existing temperature-zero test requires greedy deterministic generation; the performance task requires the same greedy policy as RKLLM. Separate greedy JITs can select argmax directly while preserving the sampled path and per-operation cancellation checks.

Temperature-zero generation now selects separate greedy JITs instead of drawing
Gumbel noise. The sampled path and the public three-argument model call remain
available. All four JIT entry points retain oNPU's cancellation checkpoints and
phase placement. Fourteen existing LLM generation tests and 22 app tests pass.
The preceding greedy benchmark snapshot matches all five stock ID sequences and
measures 2876.294 ms first token / 5.8750 decode tokens/s; this is a small decode
change, with unpaired timing noise, not RKLLM parity.

A 128-row/eight-bank experimental matrix recipe failed full-output comparisons
and was rejected. Production retains its 64-row/four-bank bounds. A separate
30-case probe qualifies three-core batched execution within those bounds; the
retained hardware suite also passes (three tests, 17.09 seconds), including actual
K=2048/3072 prefill shapes. Three-core prefill now shares packed inputs and writes
disjoint output columns. The current 256-token model comparison is running.
Its preliminary stock FP16 control measured 653.097 ms first token / 16.527 tok/s;
the variation relative to the earlier control further requires paired bracketing.

Fused causal prefill attention now shares the checked CPU attention implementation,
including cached prefixes, a symbolic active query count and correctly masked
future positions. A single-thread full-model 128-token trial measures 1061.793 ms
first token / 5.9254 decode tokens/s, matching all stock IDs in all five requests.
The 256-token three-core trial before this attention change measured 10593.566 ms
first token / 5.4755 decode tokens/s, also matching all five stock sequences.

The prefill attention implementation now distributes independent head ranges
across four CPU threads for at least 16 query rows, joining every worker before
returning. Smaller work stays serial; thread creation failure uses the same math
in the caller. Full-output independent goldens cover symbolic query sizes, cached
prefixes, the parallel path and rebinding (four tests pass). A diagnostic without
competing BLAS work measured attention medians of 3.607/12.802 ms at 128/256 rows,
versus 18.050/48.894 ms for the single-thread version. The parallel full-model
comparison is running; isolated timings are not attributed to end-to-end serving.

Native CPU/Rockchip replay now runs the already prepared CPU functions, raw copies
and NPU submissions in one C call. It retains exclusive device ownership, mapped
buffer guards, scalar/input rebinding, finite activation checks, immutable-weight
version guards, task synchronization, blocking submission flags, submission/task
counts and normal fallback for unsupported graphs. Scratch release invalidates
its prepared state. CPU copies use memmove semantics; evicted matrix plans cannot
remain eligible for native replay.

A full 128-input/32-output model run with four complete warmups and three measured
requests reports 703.475 ms first token and 16.3579 decode tokens/s. All seven
32-ID sequences match the stock IDs. Actual native counters show four prefill
replays and 214 decode replays, with no unsupported-graph fallback. The additional
warmups include native C compilation (about 23 seconds for the prefill graph);
loading and compilation remain outside steady-state timing. This is promising
for decode, but prefill still trails the earlier stock control and fresh matched
bracketing remains required before a parity claim.

Four hardware/replay tests pass, covering fresh inputs/scalars, released scratch,
finite rejection without a submission, weight mutation fallback, complete matrix
outputs and resource release. The CPU-only replay fallback also uses the normal
executor. Mypy and Ruff pass. A PV tiling experiment gave only noise-level isolated
timing differences and was rejected.

Label correction: the first fused-prefill model result (1061.793 ms) already used
the parallel attention source. Its source hash equals the later parallel trial.
It must not be presented as an independent single-thread full-model control; the
isolated CPU experiments were separate diagnostics. The validated full-model
result remains an improvement from the fused attention implementation overall.

Next work: fuse the CPU SwiGLU and FP16 input packing for FFN-down to reduce the
remaining ~170 ms prefill kernel and repeated packing/copy costs; then rerun both
128/256 workloads with four warmups per engine, controlled bracketing, quality
checks and real application integration. Native replay and the model-specific
optimizations are still opt-in; the default OpenCL application path is preserved.

INTENT: FFN activation creates a full FP32 SwiGLU tensor and separately repacks each FP16 matmul tile; the task requires lower prefill overhead with unchanged inference math; the existing gated_activation uses FP32 sigmoid and multiplication, and the NPU interface specifies (K/8,M,8) input tiles. A checked fused CPU activation/packing kernel can produce those same input bytes once per FFN invocation.

The fused SwiGLU/FP16 packer now reuses tinygrad's generated scalar sigmoid,
including its EXP2 approximation and rounding. The initial system-exp2 prototype
differed in 14 of 393216 half values and was replaced. Complete uint16 payloads
now match the regular CPU path for small and large prefill geometries. Hardware
FFN comparisons also match every FP32 output bit, including concatenated input
tiles with LHS byte offsets. Nine hardware/CPU/replay tests pass with -n12.
The fused FFN remains opt-in and falls back for unsupported phase/device/layout
configurations. Full-model performance and stock-ID verification are running.

The first immutable-source paired comparison has completed, using four full
warmups and three measurements per fresh process in ABBA/reverse order. Every
frequency sample has CPU4/6 at 1.8 GHz, NPU at 1 GHz and DDR at 2.112 GHz.
All token IDs agree across engines and requests. At 128 inputs, tinygrad/stock
median first-token latencies are 585.333/314.024 ms and decode 16.683/16.961 tok/s.
At 256 they are 1743.363/661.591 ms and 15.269/16.260 tok/s. Neither phase has
established parity yet. Results are in /tmp/onpu-full-20261018/paired-fused-v1.

The native path was redundantly flushing immutable task descriptors and every
output mapping. The original runtime and native-C reference only clean prepared
output pages initially. CPU clean-version tracking now preserves that initial
barrier and detects later allocator writes. Native replay admits only CPU kernels
on host buffers and COPY-to-NPU writes followed by TO_DEVICE barriers. It also
redirects eligible sole-use FP16 scratch pack outputs into NPU upload mappings,
retaining the following cache clean while omitting the intermediate memcpy.
Nine hardware/CPU/replay tests pass after these changes. Performance qualification
of the new cache/alias path is pending.

The cache-only 128-token trial matches all seven stock sequences and measures
553.919 ms first token / 17.0500 decode tokens/s. Its first direct-input search
selected no aliases because COPY operands are byte bitcasts. The search now
checks the writer's half dtype and traces scratch BUFFER tags through those
views. It rejects later reads before overwrite and any intervening use of the
NPU destination. The hardware/replay test requires an actual direct-input alias,
checks rebindings/scratch release/finite rejection/weight fallback, and all nine
targeted tests pass. An end-to-end direct-input qualification is running.

The raw FFN experiment consumes concatenated native (N/4,M,4) projection outputs
inside the activation packer, avoiding a separate row-major transpose. A separate
split-down candidate follows the native-C reference: each core contracts one
third of K and CPU sums its three FP32 partial results. This changes FP32 reduction
order, so it remains a separate opt-in experiment requiring independent full-output,
logit and generated-ID validation; no bitwise equivalence is claimed for K splitting.

The raw-layout/split-down 128-token model trial completed four warmups and three
measurements with every stock ID matching. Its preliminary medians are 510.895 ms
first token and 17.0112 decode tokens/s. This remains slower in prefill than the
fresh ~314 ms stock control. The application worker now prepares wide/fused/native
paths when optional NPU placement is requested, pins CPU/NPU workers to available
big cores, and warms both greedy graphs before reporting readiness. GPU-only
loading keeps its existing default path. Real dual-model app verification is in
progress; split-down is not enabled by the app pending broader quality validation.

The real CPU/NPU application passed streaming, one-token limits, context rejection,
Stop, busy admission, disconnect cancellation and generation after cancellation.
The independent GPU and NPU workers were also observed generating concurrently.
The idle NPU worker was then unloaded to release ownership for hardware experiments.

INTENT: prefill attention currently performs both dense contractions on CPU; the task requires prefill parity with RKLLM; the reference attention_backend.h uses independent three-core FP16 attention matmuls with FP32 accumulation and CPU causal softmax. A bounded opt-in version will be checked against full independent attention outputs before model benchmarking.

INTENT: each synchronization currently flushes the entire shared NPU allocation; the task requires lower transfer overhead while preserving coherent outputs; the allocator writes and copies expose exact buffer byte ranges, and the reference synchronizes those ranges. Preserve the initial whole-allocation clean and synchronize subsequent buffer ranges, including native replay.

The three-head attention experiment passed full-output comparisons with fresh
queries, fresh KV caches, dynamic rows, cached prefixes and actual native replay.
It was rejected on timing: 27.49/60.01 ms per layer at 128/256 query rows versus
3.85/13.09 ms for CPU attention, including transfers and a 512-token KV capacity.
Profiling also exposed whole-shared-allocation synchronization. The attention
prototype and its extra runtime modes were archived outside the working tree;
the focused port retains CPU attention and tests range synchronization separately.

Ten hardware/CPU/replay tests passed after buffer-range synchronization, including
initial whole-allocation cleaning and complete split-K matrix outputs. The new
128-token trial measures 484.755 ms first token / 18.2971 decode tokens/s, with all
seven stock sequences matching. This is still a preliminary, unpaired sample.

Full-vector quality checks compare 151935 finite logits per request against RKLLM
and inspect the one SDK-masked EOS entry separately. GGUF metadata confirms EOS
ID 151645, which SDK ignore_eos_token sets to -infinity. Both full-K and split-K
paths pass the predefined limits (max absolute .25, RMSE .02, cosine >= .9999,
same top token) for 128/256 prefill and a stateful one-row decode at position128.
Observed stock comparison maxima are below .017 and RMSE below .0041. This does
not establish bitwise equivalence for K splitting.

INTENT: raw FFN activation packing computes scalar sigmoid lanes and costs about 96 ms per prefill; the task requires lower prefill cost without changing operands; tinygrad's FP32 EXP2 legalization defines the polynomial and rounding. Use four-lane CPU math over the native projection layout and require complete FP16 payload equality against the original tensor expression.

The vector activation path passed ten hardware/CPU/replay tests. Payloads match
the original CPU tensor expression bitwise for row-major and raw NPU output
layouts, including split-K packing. Its 128-token model trial measures 441.642 ms
first token / 17.5856 decode tokens/s; all seven complete stock sequences match.
This remains preliminary: decode drift between independent trials reinforces the
need for fresh paired measurements before any speed claim.

INTENT: CPU attention reloads and converts each FP16 key/value for every query; the task requires faster prefill while retaining CPU attention when it wins; the existing kernel specifies FP32 dot products, causal softmax and PV accumulation. Reuse loads across four query rows, preserving each row's dot and softmax summation order, and check complete outputs against the scalar path and independent NumPy.

Six CPU tests passed after four-row attention, including bitwise comparisons with
the prior scalar kernel and independent NumPy goldens. The diagnostic native
model replay now reports 394.367 ms first token (two output tokens, one measured
trial, profiling enabled); this is a diagnostic, not a paired performance claim.
Generated C shows the remaining FFN output merge loads both row tiles before
selecting one, and similarly loads all four attention-output tiles per row.

INTENT: lazy output concatenation repeatedly reads nonselected raw NPU tiles and scalar-reduces split-K outputs; the task requires lower prefill layout overhead; the matrix interface specifies native (N/4,M,4) outputs and ordered addition of three FP32 K partials. Select the correct row tile before loading, unpack four lanes together, and check full outputs and partial sums against the existing tensor expression.

INTENT: attention-output K=2048 currently limits full-K input tiles to32 rows; the task requires a faster prefill path; the qualified K-split interface supports64 rows per core and ordered FP32 partial summation. An opt-in attention-output projection pads K with zeroes to a multiple96 and splits K across three cores during prefill; decode retains its ordinary full-K path.

The immutable-source paired unpack comparison completed. At128 tokens, tinygrad
versus stock medians are362.359/329.909 ms first token and17.8007/16.3545 decode
tokens/s. At256 they are846.956/692.838 ms and16.4029/15.3960 tokens/s. Every
complete generated sequence agrees. Decode wins on these paired samples, while
prefill is still slower. Samples and source image: paired-unpack-v2.

The attention-output K-split experiment passed13 hardware/CPU/replay tests but
measured379.771/893.995 ms at128/256 tokens in the follow-up trial. It was rejected
and removed from the active code; its immutable source and reports remain in
/tmp/onpu-full-20261018/split-output-candidate. Splitting FFN-down remains a
separate qualified option. The four-row CPU attention additionally preserves
causal isolation from nonfinite future values; a branch-free common token prefix
and guarded three-token tail retain bitwise results in the CPU tests.

INTENT: Q/K normalization, RoPE and KV writes currently traverse separate CPU intermediates; the task requires faster prefill; RMSNorm, apply_rope and the native-C normalize_rotate specify those same operations in sequence. An opt-in fused CPU preparation kernel will consume the shared projection once, normalize/rotate heads, and write only the active FP16 cache range. CPU tests check queries, keys, values, cached prefixes and JIT rebindings before enabling it in model benchmarks.

Fused QKV preparation passes CPU precompiled-function/JIT tests with FP16/F32
normalization weights, symbolic rows/positions and fresh buffers. Every cache
entry outside the active range is unchanged. Complete model logit checks also
pass against stock for128/256 prefill and the stateful decode step; candidate
max errors are .01344/.01574/.01157, with matching top tokens. A two-token native
diagnostic measures341.901 ms first token; full32-token paired verification is
running. All14 hardware/CPU/replay tests pass, and mypy219 files/Ruff pass.

TWINS: searched shared-token PV loops in retained attention kernels - found 0 other unsafe sites: none. The scalar attention loops already stop at each row's causal boundary; the shared four-row loop now guards its three-token tail.

The app unit suite passes22 tests. The broader tinygrad server test file cannot
initialize its SDK-based fixtures because the optional openai client package is
absent from both existing virtual environments (11 setup errors,7 earlier passes).
No dependency was added and no test was weakened. Generation/parser/tokenizer
checks are run separately; real HTTP product checks are also available above.

The generation/parser/tokenizer subset passes24 tests. Fresh paired fused-QKV
results are340.283/323.584 ms first token at128 inputs and823.720/655.395 ms at256,
with tinygrad/stock decode17.7909/16.2975 and16.5874/16.6245 tokens/s. Every full
32-ID sequence matches. Prefill parity is still unachieved, and the256 decode
sample is effectively tied rather than a consistent win.

The bounded CPU register grid found an8-query/2-vector tile faster than4-query
default for the tested128-dimensional heads:2.206 vs2.424 ms at128 rows and6.727
vs7.633 ms at256, with bitwise scalar-path equality. That tile is selected for
128-dimensional heads, >=64 maximum query rows and <=1024-token capacity; other
geometries keep their existing tile. Larger register tiles were slower.

INTENT: decode attention converts and reloads identical KV entries for each grouped-query head; the task requires consistent decode parity as well as prefill gains; the existing causal FP32 attention defines one independent softmax and ordered PV sum per head. An opt-in CPU experiment reuses each KV load across heads in the same GQA group while preserving each head's arithmetic order.

INTENT: finite-activation guards scan every FP16 lane with a scalar branch; the task requires lower prefill overhead while retaining rejection before submission; the runtime rejects every half value whose exponent bits are all ones. Check eight half encodings together with the same predicate and retain the scalar tail, verifying all65536 encodings plus every vector lane, unaligned starts and remainder lengths.

The finite guard accumulates vector invalid masks before one final reduction;
all65536 encodings, every lane, unaligned starts and tails pass. The complete
15-test backend suite also passes. Grouped-query decode reuses KV loads with
bitwise outputs for GQA2/3/4 in isolated checks and is now selected on supported
single-row geometries. Updated full-model qualification remains pending.

Tall small-K matrices are independently qualified in30 complete-output trials
at M128/256 while retaining the four-bank bound. The earlier M128/K1024 failure
was an eight-bank case and does not imply that every M128 matrix is unsupported.
Chained independent CMAC bodies need explicit PC base/amount links: the first
unlinked two-task attempt timed out, reset recovered the device, and a one-task
control passed. Linked2-task,18-task and36-task submissions pass every output
across three fresh inputs. The36-task three-core batch takes about .32 ms.

INTENT: independent attention matrices currently require separate driver calls; the task requires lower prefill overhead; the reference PC tail supplies the next command address/amount, and hardware probes validate linked CMAC tasks. Add a bounded packed matrix-batch mode, preserve existing single-matrix bounds, and allow freshly uploaded shared RHS weights only with complete finite checks and byte-range upload guards.

The latest CPU-attention/guard candidate completes four full warmups and three
32-output measurements at each input length, matching every stock sequence.
Medians are305.432 ms/17.9280 decode tokens/s at128 and688.679 ms/17.7706 tokens/s
at256. This is an unpaired follow-up, so fresh stock bracketing remains required.
The long-input prefill result is still above the previous655 ms stock control.

The complete linked attention pipeline passes full-output, dynamic-row, cached
prefix and native-replay checks (15-test suite), but stays disconnected from the
model: fresh-prefix total cost is still higher than CPU. Known256-token length
bounds reduce its dimensions and reach5.757 vs5.988 ms in isolation, below the
10% selection threshold. A cached512-token total favors NPU more strongly.

INTENT: CPU normalization performs repeated division and a serial scalar reduction; the task requires lower prefill cost; Tensor.rmsnorm specifies multiplication by inverse root mean square. Use one FP32 inverse per row with a vector reduction and row-parallel CPU work, preserving GPU/default paths and verifying numeric outputs and full-model logits before qualification.

Sixteen backend tests pass with the normalization candidate, including symbolic
rows, FP16/F32 normalization weights, independent RMS output checks, QKV/cache
checks and full matrix/native replay tests. Full-vocabulary stock comparisons
pass for128/256 prefill and stateful decode: candidate maxima .01716/.01647/.01222,
RMSE <=.00413 and matching top tokens. Fresh paired32-token timing is running in
/tmp/onpu-full-20261018/paired-norm-v5. Mypy220 files and Ruff pass. These numerical
results do not by themselves establish the remaining prefill performance target.

INTENT: native finite checks read uploaded activations after cache cleaning, forcing another cold read; the task requires lower prefill overhead while preserving every rejection; COPY establishes an exact uploaded byte range and the next NPU writer invalidates that range. Validate covered half ranges immediately before their TO_DEVICE barrier, always complete that barrier before returning a finite error, and retain checks at the kernel for uncovered data.

The pre-barrier finite-validation experiment passed16 serial tests but did not
improve the follow-up timing (685.671 ms at256 inputs). Its source was archived
and the simpler kernel-side validation retained. The new256 profile attributes
about38 ms to concatenating four raw gate/up projection tiles before SwiGLU.

INTENT: FFN packing first copies every projection tile into one FP32 tensor; the task requires faster long prefill with identical operands; the qualified packer specifies exact FP32 SwiGLU and concatenated FP16 (K/8,M,8) tiles. Consume the original projection pointers directly, selecting one input tile before channel work, and require complete FP16 byte equality and full hardware outputs.

The direct-tile packer passes complete uint16 comparisons for1/17/128/193/256
rows, including partial input tiles and split/unsplit output packing. Hardware
FFN outputs also remain bitwise identical at256 rows. Its full256-input/32-output
trial completes all seven matching sequences and measures649.439 ms first token
and17.2447 decode tokens/s. This follow-up still needs fresh stock bracketing.

Output-unpack tests show parallel work helps the large projection layouts while
small layouts should stay serial; the row arithmetic is bitwise identical. The
parallel threshold is131072 output cells. RMS threading uses the same measured
cell threshold. Sixteen backend tests, mypy219 sources and Ruff pass after cleanup.

The unused NPU attention pipeline and its batch-only runtime extensions were
archived in /tmp/onpu-full-20261018/experimental-linked-attention.tar.gz and removed
from the retained port. Only the inference paths that are actually selected remain.
The app CPU/NPU worker now requests fused QKV, CPU normalization and split-down;
GPU-only defaults retain the normal OpenCL path. Clean paired timing of the final
source is running in /tmp/onpu-full-20261018/paired-final-v6.

INTENT: the app always uses32-token prefill chunks even when the qualified long-prompt path uses128/256; the task requires the fast inference path to be usable by oNPU; generate exposes chunk_size while each TinyJit captures one maximum input shape. Cache bounded prefill JIT variants in the optional native integration and select/warm suitable CPU/NPU chunks in the worker; preserve GPU-only batching and per-chunk cancellation checks.

INTENT: binding-cache eviction explicitly closes command buffers even when a live native replay owns the plan; the task requires warmed prompt variants to remain executable; RockchipProgram retains prepared commands and RockchipMatmul.__del__ releases them when ownership ends. Drop the evicted cache reference and let remaining replay ownership govern lifetime, with a failing-before/passing-after regression that also checks final release.

TWINS: searched popitem-plus-close and binding-cache removals across tinygrad - found 0 other unsafe sites: none.

The clean v6 paired source wins both phases:128 inputs276.801/323.520 ms and
18.5497/16.4445 tokens/s;256 inputs628.396/657.894 ms and17.5194/16.6145 tokens/s
(tinygrad/stock). All complete sequences match. App batching then gained bounded
prefill variants and adaptive chunks so it can use the qualified long-prompt path.

Real adaptive SmolLM2 loading prepared all four prompt sizes for both greedy and
sampled inference before readiness in151.795 seconds. This cold-compilation cost
is separate from steady-state inference. Real HTTP streaming, Stop, disconnect,
busy rejection and generation after cancellation passed. That warmup exposed the
binding ownership defect: the new eviction regression fails on the old code and
passes when eviction drops only the cache reference. The final owner still frees
the plan. Eighteen backend tests, mypy219 and Ruff pass after the fix. The NPU app
session was unloaded before clean final-source timing, now running in
/tmp/onpu-full-20261018/paired-final-v7. Fresh post-fix app/browser checks remain.

INTENT: the Rockchip load form hides prefill assistance together with the GPU-only decode control; the task requires optional accelerated prefill to be usable from oNPU; the API already accepts per-phase assistance and the Rockchip profile supplies NPU decoding. Expose prefill Auto/Off for Rockchip sessions, default it to Auto, and keep GPU-only defaults Off.

## Final current-state verification

The final v7 immutable source beats RKLLM in both measured phases. At128 inputs,
tinygrad/stock median first token is280.353/326.310 ms and decode18.7022/16.6705
tokens/s. At256 it is631.830/655.672 ms and17.0492/16.5525 tokens/s. All32-ID
sequences agree across every warmup and measurement; all current source hashes
match the measured snapshot. Big-core CPU/NPU/DDR sampled frequencies remain
1.8/1.0/2.112 GHz. This qualifies the matched Qwen3-0.6B FP16 workload, notW8A8
or universal model/context performance.

Current-source full-vocabulary checks also pass128/256 prefill and a stateful
decode step, with max errors below .018 and RMSE below .0042. Eighteen backend
tests,24 generation/parser/tokenizer checks,23 app unit tests, mypy219 files,
Ruff, JavaScript syntax and whitespace checks pass. Optional SDK-based server
fixtures remain unavailable because the openai client is absent; real HTTP
checks cover the application's compatible API. No dependency or check was changed
to conceal that limitation.

Fresh GPU and NPU HTTP generation, SSE, context/limit checks, busy rejection,
Stop/disconnect cancellation and recovery pass. Real overlapping GPU/NPU model
requests pass with independent workers importing this checkout. The browser
passes chat/history, Stop, model selection, import/unload/reload while preserving
the other session, themes and desktop/mobile navigation without console errors.
The browser exposed hidden Rockchip prefill controls; the corrected form defaults
to Auto and passed an actual NPU load while preserving the GPU model. GPU defaults
remain OpenCL with assistance Off.

Both final sessions are left ready. Cold preparation compiles four prompt sizes
for greedy and sampled execution; the observed NPU preparation is about152s.
Runtime code has no reference-worktree dependency, and both reference folders
remain intact with their existing uncommitted work. Nothing was committed, pushed
or removed from those folders.

The compact manifest is docs/rockchip-inference-progress.json. Exported raw logs,
compressed benchmark/quality reports, screenshots, source manifest and reviewable
source archive are in ~/oNPU/docs/rk3588-verification-20261018/. The earlier full
research history is preserved in /tmp/onpu-full-20261018/progress-history.json.gz.
