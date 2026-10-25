# oNPU VLM qualification

The current prepared GPU/NPU path is faster than matched RKNN vision plus RKLLM
W8A8 on the qualified SmolVLM workload, with comparable output quality under the
fixed 20-case gate. The main app at http://127.0.0.1:8091 serves this path.
The final requirement-by-requirement audit passes; see [verification](rockchip-vlm-verification.json).

- [x] Shared checkpoint, vendor conversion, and held-out reference inputs.
- [x] Complete SigLIP, connector, and original image-conditioned Llama decoder.
- [x] Full matched RKNN+RKLLM W8A8 baseline.
- [x] Fixed quality gate and complete 20-case current-source evaluation.
- [x] Faster complete responses at both declared output lengths.
- [x] Frozen-source comparisons with reversed engine order and clock samples.
- [x] Image upload/streaming, sampled generation, Stop, disconnect and recovery.
- [x] Concurrent GPU and NPU requests in the same main app.
- [x] Backend/app regression, hardware, type, lint and syntax checks.
- [x] Final requirement-by-requirement completion audit.

## Qualified scope and measurements

HuggingFaceTB/SmolVLM-256M-Instruct, revision
7e3e67edbbed1bf9888184d9df282b700a323964. Checkpoint SHA256:
74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e.
One square 512x512 RGB image, image splitting off, 1024 patches, 64 image tokens;
82 total prompt tokens and context 512. Both decoders keep the original 30 blocks.

| Fixed output length | oNPU full response | RKNN+RKLLM full response | Less latency |
|---|---:|---:|---:|
| 32 tokens | 1133.63 ms | 1460.98 ms | 22.4% |
| 128 tokens | 2043.05 ms | 3138.18 ms | 34.9% |

The subsequent prefill fix removes graph-setup and external selection overhead.
Prefill is now 1301.56 versus 931.10 tok/s on the 32-output workload, and 1292.39
versus 914.73 tok/s at 128 outputs. First-token medians are 857.67/859.47 ms versus
926.93/929.40 ms. Decode remains 112.35/107.21 versus 58.07/57.57 tok/s. See the
[prefill completion report](rockchip-prefill-progress.md) for current app checks.

Each engine/output length ran in two fresh processes with three warmups and
three measured trials per process. The 32-token order was oNPU/vendor/vendor/oNPU;
128 used vendor/oNPU/oNPU/vendor. Timing starts from prepared pixels and includes
vision, connector, text prefill, decode, synchronization and the required oNPU
image-boundary reset. Image file decoding/preprocessing, loading and cold
preparation are excluded for both engines. EOS is ignored for equal fixed-output
work; this does not measure the length of a naturally terminating answer.

Clock samples retain CPU big cores at 1.8 GHz, NPU at 1 GHz and DDR at 2.112 GHz.
GPU clocks follow the existing governor. No clock, IRQ or kernel-module policy
was changed. The stock executable loads librkllmrt 1.3.1 and the existing RKNN
runtime. Source, input, binary and raw-trial hashes are in
[rockchip-vlm-benchmark.json](rockchip-vlm-benchmark.json). Raw evidence is under
/home/orangepi/vlm-bench/prepared-final-order.

## Quality

The 20 vendor-provided MMBench images/questions are separate from the demo
calibration/performance image. Of 320 saved reference positions, 213 occur before
or at the first reference EOS. Teacher-forced likelihood and separate free greedy
answers are evaluated; forced argmaxes are not presented as generated answers.

| Metric | FP32 reference | RKNN+RKLLM W8A8 | oNPU |
|---|---:|---:|---:|
| Conditional perplexity, 213 positions | 1.20410 | 1.24319 | 1.25030 |
| Correct multiple-choice answers, 20 cases | 6 | 7 | 7 |
| Mean projected-image NRMSE | 0 | 5.521% | 0.1505% |
| Maximum projected-image NRMSE | 0 | 8.379% | 0.7336% |

Acceptance was fixed before evaluation: candidate perplexity at most 1.02 times
vendor, no fewer correct answers, each image NRMSE at most 1.1 times vendor plus
1e-5, and finite features/logits. The candidate/vendor perplexity ratio is about
1.0057; every gate passes. This is a limited quality proxy, not a full VQA suite.
Current evidence: /home/orangepi/vlm-bench/onpu-quality-prepared-final.
The vendor harness matches RKNN's queried NHWC input contract; early incorrect
NCHW measurements were invalidated and are not used in these comparisons.

## Implementation and app

The encoder uses prepared three-core FP16 matmuls, shared-buffer GPU softmax and
GELU, split-K down projection, and fused CPU residual addition. Prepared MLPs use
two submissions per block instead of 65. Attention batches all 12 heads. NPU
submissions and clFinish establish exclusive device handoffs; CPU-visible buffers
retain required synchronization. Image-boundary reset remains because returning
from INT8 text to FP16 vision otherwise failed full-model tests. The failed FIFO
reset experiment was removed.

Weights are immutable after preparation. Guards cover graph identity, allocation
identity and NPU versions. Replaced allocations, bad inputs, GPU activation
overflow and resource release have actual hardware checks. A newly added storage
replacement test first failed, then passed after matching NativeReplay's storage
identity rule. No correctness/quality tolerance was relaxed.

Direct register commands and explicit task chains provide tile boundaries, so no
SHARD dummy compiler barriers are needed. Legal tiling, memory layouts, precision,
synchronization and scheduling remain essential. The article's approximately 15x
result compares its encoder with a CPU baseline, not RKLLM.

The app preserves OpenCL as the default. The Rockchip hybrid profile exposes
"Use GPU for images when available" for vision models. Checkpoint and driver
capability gates restrict acceleration to the qualified path; disabling it retains
CPU/NPU vision and keeps the GPU available for another model. Actual execution is
reported as CPU+CL+ROCKCHIP. The app warms image encoding, bounded embedding-input
prefill and native greedy decoding before ready. Prefill uses adaptive chunks up to 128 rows and
retains cancellation checkpoints and strict context bounds.

The real browser passes profile controls, image upload and streaming with zero
exceptions. Its first 32-token image request took 1.434 seconds including image
preparation. The API passes sampling, Stop (1.396 seconds), disconnect/recovery,
and concurrent GPU/NPU requests observed active in the same server. App timings
are diagnostic and separate from the controlled engine comparison. The original
GPU model and the VLM are ready at port 8091, with the GPU model selected. SmolLM
is unloaded while the VLM occupies the single NPU slot.

## Checks and limits

- CPU regressions: 28 passed, 5 explicit hardware skips, using -n12.
- Dedicated NPU/GPU attention and prepared-MLP suites pass, including finite,
  rebinding, task-count, allocation, immutability and release checks.
- Mypy: 236 source files pass. Whole tinygrad Ruff and whitespace checks pass.
- App tests: 36 pass. New VLM files lint and JavaScript syntax checks pass.
  Whole-app lint has pre-existing findings; no whole-app clean claim is made.
- Native HF prompt IDs match the saved reference. Runtime code has no dependency
  on the reference tinygrad worktrees. No archives, commits, pushes or reference
  repository removals were performed.

The result is specific to this board, checkpoint, resolution and tested workload.
Pure CPU/NPU mode, other models, image splitting and larger contexts do not inherit
this speed claim. Temporary browser state was removed after verification.

## Implementation record

INTENT: NPU attention used CPU softmax over large intermediate matrices; the performance task requires lower image-to-answer latency; StagedVision explicitly permits heterogeneous attention stages while retaining the CPU/OpenCL reference path.
INTENT: GELU used a general exponential with exceptional-range work; the task requires lower encoder latency; the existing FP32 sigmoid/tanh expression saturates outside exponent [-24,27], allowing the same result with bounded range reduction.
INTENT: device-only intermediate buffers still received CPU cache maintenance; the task requires lower encoder latency; their exclusive GPU/NPU ownership and blocking completion permit handing off without touching CPU aliases.
INTENT: the vision MLP used 65 small NPU submissions and CPU GELU/packing; the task requires lower first-token latency; RockchipLinear declares prepared weights immutable and StagedVision permits heterogeneous stages, allowing two guarded PC-chain submissions with GPU activation packing and split-K reduction.
INTENT: prepared MLP cache guards checked UOps and version counters but not allocation identity; the task requires preserving guarded replay; NativeReplay.valid explicitly checks BufferStorage identity before executing cached addresses.
TWINS: searched prepared weight/storage guards - found 1 other cached replay site: rockchip_replay.py already checks storage identity; the new MLP now guards all seven immutable source/packed parameter allocations.
TWINS: searched score/probability cache sync sites - found 0 other unconditional steady-state CPU sync sites for those shared intermediates; initial cleaning and CPU-visible flag/result synchronization remain.
INTENT: text-only code needed image-conditioned inference; the task requires a faster complete VLM; the model configuration specifies SigLIP, a pixel-shuffle connector, and image substitution into the Llama decoder.
INTENT: projections padded 576/768-channel inputs to 1024; the task requires efficient small-model inference; exact integer golden checks now qualify only K=576,768,1152 with conservative CBUF allocation.
INTENT: Python drove every decode token; the task requires lower complete-response latency; the retained bounded scalar replay loop executes unchanged guarded decoder steps and preserves token IDs.
TWINS: searched BF16 checkpoint casts - found 1 other direct cast site: extra/models/llama.py:286; unchanged outside the new VLM loader.
TWINS: searched VLM RKNN input-format assignments - found 1 other site in the full response loop; both stock paths match queried input format.
TWINS: searched VLM encode timing boundaries - found 0 other retained sites missing GPU completion; stock calls block and diagnostic probes read numpy output.
INTENT: image prefill warmup used shape-specific prompts and could leave first requests compiling; the task requires usable image serving; the app README promises ready workers and context-bounded requests.
INTENT: a contiguous prompt slice could retain its parent-buffer view; real JIT inputs require independent buffers; tinygrad's JIT explicitly directs callers to clone such inputs.
TWINS: searched image prefill pad_to/contiguous inputs - found 0 other sliced embedding inputs in onpu/worker.py; the remaining warmup input is a full zero tensor.
INTENT: image sampling supplied a position variable named position; the warmed decoder requires start_pos; Transformer.generate defines start_pos over [0,max_context-1].
TWINS: searched onpu/ for UOp.variable("position") - found 0 other sites; image prefill already uses start_pos with the decoder's bounds.
TWINS: searched encoder invocations - found 2 caller sites, worker warmup and image generation; both now use the shared reset/encode/synchronize entry point.
