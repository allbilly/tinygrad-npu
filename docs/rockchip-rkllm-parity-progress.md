# RKLLM parity and performance work

Target: beat RKLLM in both prefill and decode at comparable quality for every model in the official RK3588 benchmark. Individual measurements, missing architecture support, and losses must remain explicit. Export/harness changes alone are not performance wins.

Scope: active tinygrad Rockchip/Mali kernels, export/benchmark tools and focused tests; oNPU integration/qualification documentation as findings require. Preserve existing model files, historical measurements and unrelated work. No new dependencies, commits/pushes, archives, kernel modules or global clock/governor changes.

- [x] Freeze the 21-model published matrix and reference protocol (128 input / 64 output, W8A8, level0); record the published clock policy separately from matched local runs.
- [x] Fix active reference exporters to level0 and complete build provenance; retain legacy artifacts for reproducibility.
- [ ] Build and qualify level0 Qwen/SmolVLM references using existing toolkit1.3.1.
- [ ] Run repeated short-context baseline comparisons with matching inputs, output count, configured capacity and clock policies.
- [ ] Profile each measured loss and retain only verified end-to-end improvements with unchanged quality gates.
- [ ] Extend model loading/architecture coverage and measurements across every published row; no unmeasured row is a win.
- [ ] Requalify long-context and concurrent serving after numerical changes; distinguish Qwen's documented32K window from40K configured capacity.
- [ ] Finish API/UI progress and sustained recovery/memory checks, then restore both current sessions and report remaining failures honestly.

Existing evidence: published RKLLM recipe specifies optimization_level0. Active VLM exporter hardcodes1; old artifact provenance omits the setting. Earlier local measurements remain historical results, not a certificate against the published recipe. Cached converter image1.3.1 is available;115GiB disk space and16GiB RAM require sequential model staging.

INTENT: active VLM export hardcodes optimization_level1; the task expects the published RKLLM reference; Rockchip benchmark specifies optimization_level0. Default new exports to0, record full settings and source hashes, and preserve legacy artifacts.

Exporter correction:5 regression tests pass andchanged-code Ruff passes. SDK1.3.1 documentation confirms level0 prioritizes speed/memory andlevel1 accuracy; the16K build cap applies toboth. Verified cachedQwen source against officialHF LFS SHA f47f7117... before reuse. Initial Q8 primitive probe reversed the parser's already-normalized shape; it was caught, retained as an incorrect-shape attempt andcorrected before qualification.

INTENT: Mali dispatches QKV and gate/up as separate projections; the task needs lower prefill/decode overhead; TransformerBlock already accepts joint QKV and gate/up callables. Add opt-in GPU joint projections with shared weights and an exact-FP16 Q8 output-head path; keep both disabled until whole-model verification and timing.
The corrected Q8 probe matches dense FP16 GEMV bitwise. It is slower for small hidden matrices but reduces the vocabulary projection17.30→10.94ms in a concurrent-conversion diagnostic. Restrict that candidate to the output head; these are not end-to-end isolated results.

INTENT: oNPU's hardware status reads only the DRM minor's uevent, which contains no driver identity on this board, and reports npu_present=false while inference works. Read the parent device's driver metadata too, case-insensitively, without opening or locking the NPU.

GPU candidate qualification: all 256 held-out positions produce bitwise identical complete 151936-entry logits with projection fusion and the Q8 output head enabled. This establishes equivalence to the existing GPU path on that sample; qualification against the new level0 SDK reference remains pending. Hardware detection and runtime tests:14 passed.

INTENT: the quality tools hardcode Qwen/Smol vocabularies and disallow CPU+NPU text candidates, preventing honest qualification of other checkpoints and the existing INT8 text pipeline. Derive vocabulary from each checkpoint, distinguish image embeddings explicitly, and add the CPU+NPU text configuration with the same 256-position/1.02 quality gate.

INTENT: INT8 NPU placement skips every projection with input width above4096. Qwen2's4864 andQwen2.5's8960 feed-forward widths therefore fall back to CPU. Split those optional INT8 projections into supported input-width blocks, quantize each block and accumulate FP32 partial outputs; test actual hardware against an independent block-quantized NumPy reference. Existing qualified checkpoints have narrower projections and retain their current path.

INTENT: repeated Mali configuration can report a requested Q8 head without attaching it to an already wrapped output layer, and reports zero fused groups for existing joint projections. Make the reported configuration reflect active wrappers and attach the requested head on repeated configuration; test idempotence and long-context projection boundaries.

The current Qwen Q8-checkpoint INT8 candidate has a held-out perplexity ratio1.04464 versus the GPU FP16 path on256 positions. This is not yet a comparison against SDK level0, and is insufficient evidence to activate that INT8 candidate. GPU arithmetic equivalence passes after the final source changes; the API now correctly reports npu_present=true after restart.

INTENT: evaluate existing input-channel scaling to reduce the observed INT8 quality loss, using only the separate eight-prompt calibration file. Share scales across fused projection groups, preserve the256-position/1.02 held-out gate and keep the candidate out of serving until qualification. Default calibration omits FFN-down scaling so the existing fused activation/packing path stays available.

INTENT: make the measured GPU improvement usable in oNPU only for a qualified exact checkpoint, within its tested capacity, with both NPU assistance phases Off. Preserve the existing assistance configuration and unknown-checkpoint behavior. Keep the qualification map empty until the full-model boundary checks pass.

The final-source GPU candidate matches all256 complete held-out vocabulary dumps bitwise. A full Qwen model also matches at32736+32,32767+1,40928+32 and40959+1 with a synthetic past; this checks arithmetic/cache boundaries, not semantic quality. GPU allocations are6.82GB baseline/6.98GB optimized. Enable the exact Qwen Q8_0 checksum in the oNPU GPU-only qualification map through configured capacity40960. Six measured trials per variant:88.38→94.18 prefill tok/s and6.73→7.88 decode tok/s; prefill ranges overlap, decode ranges separate. CPU/NPU INT8 pilot:572.02 prefill tok/s,34.25 decode tok/s; still unqualified against level0 SDK. Alternative exact CPU attention tiles were tested; they do not offer a useful improvement and remain diagnostic only.

Full input scaling worsens the held-out ratio to1.03291 versus GPU FP16, compared with1.02194 when retaining the fused FFN path. Neither experimental configuration is activated in serving. The GPU improvement is active for its exact checksum; both original serving capacities are being restored.

INTENT: the SDK's final export creates temporary files in its working directory. The initial SmolVLM level0 build finished optimization but failed because the container checkout was read-only. Put SDK operations in a dedicated writable temporary workspace and restore the caller's directory on all exits; reproduce the failure mode in a regression test and verify a real small SDK export before retrying the full conversion. The running Qwen job retains its original compiled exporter; its own container bind was made writable to prevent the same final-export failure without discarding that build. No host mount flags were changed.

INTENT: two different-profile loads can both pass admission before the first worker allocates its pending model buffers. A concurrent40K GPU/8K NPU restart caused swap pressure and an NPU preparation timeout. Reject a second model preparation while another is loading/preparing, inside the pool's existing lifecycle lock; retain simultaneous inference after both workers are ready and allow restarting the same session. Test rejection before creating/stopping a worker.

Final checkpoint: 58 focused tests pass with `-n12`; mypy checks 241 files and backend/changed-app Ruff checks pass. The new admission guard is live and returns HTTP409 before creating a second worker. Both original sessions are ready at configured capacities40960/8192, with GPU selected and saved settings unchanged. Live regression checks pass for cold/cached GPU equivalence, both overflow forms preserving21 cached positions, and a0.136-second independent NPU reply during GPU prefill followed by Stop and GPU recovery.

The export workspace fix passes a real synthetic128-wide SDK1.3.1 W8A8 level0 export from a read-only checkout; its scratch directory is removed afterward. This is an exporter test, not a model performance result. The first full SmolVLM reference failed before that fix. The unfinished Qwen conversion was stopped to recover serving memory. No converters remain paused. A local sequential retry waits for at least3GiB available RAM and does not change API sessions. Full level0 comparisons, unsupported architectures, gated Gemma checkpoint access and maximum-context semantic quality remain unfinished. See `rockchip-rkllm-parity-results.json` for current evidence; no blanket RKLLM win is recorded.

INTENT CHECK: changes implement the documented, observed fixes; the broader performance objective is incomplete.
TWINS CHECK: active reference exporters, Mali configuration call sites and the public model-load path were searched. The sole remaining explicit optimization_level1 is a negative qualification test; legacy external conversion scripts remain historical and are excluded by the level0 validator.
