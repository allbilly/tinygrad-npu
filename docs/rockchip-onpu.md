# Selective Rockchip inference for oNPU

OpenCL GPU serving is the default. Optional NPU assistance is selected separately
for prefill and decode, and independent workers can serve a GPU model alongside
a smaller CPU/NPU model. The retained code does not depend on either reference
worktree. Full operator coverage and arbitrary pure-NPU execution are outside
this inference-focused port.

## Retained paths

- `tinygrad/runtime/ops_rockchip.py` executes bounded FP16/FP32-accumulate and
  opt-in INT8/INT32-accumulate matrix operations, three-core column sharding, reusable command buffers and
  per-buffer cache barriers. Inputs and weights must be finite; all 65536 half
  encodings, vector lanes, unaligned starts and tails are tested. A worker holds
  exclusive NPU ownership for its lifetime.
- `tinygrad/llm/kernels/rockchip.py` provides independent phase placement, shared
  QKV/gate-up projections, optional FFN-down K splitting and CPU normalization.
  Auto placement includes transfers and requires valid outputs plus a 10% median
  layer improvement. Other operations stay on the original CPU/OpenCL device.
- CPU kernels implement causal attention, grouped-query decode, Q/K normalization,
  RoPE and KV writes. SwiGLU reads native projection tiles directly and produces
  the same FP16 packed activation bytes as the normal tensor expression.
- Guarded native replay removes Python dispatch from prepared CPU/NPU graphs.
  It retains ownership, storage, command and immutable-weight checks, rejects
  nonfinite inputs before submission, and falls back for unsupported graphs.
- Optional native integration caches bounded prefill graph variants. The app
  chooses 1/32/64/128/256-token chunks that fit the context, warms both greedy and
  sampled paths, and checks cancellation between chunks and decode operations.
  GPU-only serving retains its normal batching.

The NPU is a dense FP16 matrix path with K up to4096, input tiles within four
CBUF banks and at most64 rows per core. Split-K FFN-down sums three FP32 partials
on CPU. These limits do not qualify every model or context. The unused linked
NPU-attention prototype was archived after transfer-inclusive timing failed to
justify selection for fresh prompts.

## Matched FP16 results

The `paired-final-v7` source snapshot compares Qwen3-0.6B at context512 against
RKLLM1.3.1 using the same effective FP16 checkpoint, raw input IDs, empty history,
32 greedy output IDs and ignored EOS. Loading and four complete warmups are
excluded; each fresh process supplies three measurements. Engine order is ABBA
at128 inputs and reversed at256. CPU affinity is4–7; sampled big-core CPU, NPU and
DDR frequencies are1.8 GHz,1 GHz and2.112 GHz. No global clock settings changed.

| Input tokens | tinygrad first token | RKLLM first token | tinygrad decode | RKLLM decode |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 280.353 ms | 326.310 ms | 18.7022 tokens/s | 16.6705 tokens/s |
| 256 | 631.830 ms | 655.672 ms | 17.0492 tokens/s | 16.5525 tokens/s |

Every complete generated sequence agrees across engines and requests. Independent
full-output hardware checks and full-vocabulary logit comparisons also pass.
RKLLM masks EOS ID151645 when ignore_eos_token is enabled; all151935 remaining
logits are compared numerically, with that mask recorded separately.

These results establish the tested FP16 snapshot, not W8A8 performance or a
universal claim across models. Adaptive-worker and source qualification passed, including real GPU/NPU HTTP,
concurrent sessions and Chromium lifecycle/navigation checks. Details are
tracked in [the work record](rockchip-full-port.md). Full trials, checksums and
clock samples are retained in [the evidence record](rockchip-inference-progress.json).
The earlier [GPU auto-placement study](rockchip-onpu-benchmark.json) uses Q8 GGUF
and is a separate comparison.

## INT8 path and RKLLM W8A8 qualification

The INT8 path uses per-output-channel weight scales and per-token activation
scales, with FP32 host arithmetic. It keeps packed weights and prepared three-core
commands. Decode supports wide output tiles; K<=2048 supports128 input rows per
task and larger K uses bounded64-row tiles. CPU quantization writes a checked
scale header and rejects nonfinite inputs before any NPU submission.

The selected CPU kernels fuse integer-output dequantization with Q/K head
normalization, RoPE and KV writes, and with SwiGLU plus down-projection input
quantization. Multi-tile dequantization avoids intermediate floating-point joins.
Optional FP32 vector softmax, a parked per-KV-group decode pool, guarded input
cache cleaning and a nice19 CPU0 helper reduce host overhead. The helper runs
only around warmed native calls; no global clock or IRQ settings change.

The SHARD article describes a SigLIP vision-encoder speedup over PyTorch CPU,
not an LLM comparison against RKLLM. Its useful transferable ideas are explicit
memory bounds, tiling and scheduling. This backend already controls its kernels
directly, so it does not need the article's nonzero numerical “poison pill.”
The source and benchmark scope are at
<https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/>.
A tested FP16 CPU attention prototype supplied no material speed gain and was
removed. The production attention path continues to accumulate in FP32.

On256 teacher-forced WikiText-2 test positions (64 prefixes, each followed by
three decode positions), sampled conditional perplexity is17.6934 for the FP16
reference,18.1034 for RKLLM W8A8 and18.0784 for oNPU INT8. The candidate/reference
runtime code is hashed and every valid vocabulary score is compared. The fixed
quality gate permits at most2% higher overall perplexity than RKLLM; the measured
ratio is0.998617. This is a sampled text-quality proxy, not a full-corpus
perplexity or instruction/task-accuracy qualification. The phase ratios are
0.9067 for prefill and1.0313 for decode; these differences are retained rather
than hidden by the overall average. Argmax agreement with FP16 is88.28% for
oNPU INT8 and92.19% for RKLLM W8A8.

Both candidates declare Qwen3-0.6B, but their quantization differs. oNPU quantizes
the retained FP16 GGUF at load time. The downloaded Firefly RKLLM W8A8 model was
converted with toolkit1.3.0 and runs on runtime1.3.1; its exact original checkpoint
revision is unverified. These are practical model/runtime comparisons with
measured output quality, not identical-quantization operator comparisons.
RKLLM uses its faster tested `embed_flash=1` mode and domain1; domain0 failed its
large contiguous weight allocation on this board. The final in-place, source-
guarded ABBA comparison completes four full warmups and three32-output-token
measurements in every fresh process. CPU4–7, NPU and DDR samples remain at
1.8GHz,1GHz and2.112GHz. oNPU also runs its nice19 CPU0 helper during warmed
native calls; the four big cores perform the inference/attention work.

| Input tokens | oNPU INT8 prefill tok/s | RKLLM W8A8 prefill tok/s | oNPU INT8 decode tok/s | RKLLM W8A8 decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 556.33 | 509.91 | 33.7892 | 23.4555 |
| 256 | 484.24 | 476.28 | 32.2656 | 22.6525 |

Prefill is9.1% faster at128 inputs and1.7% faster at256; the latter is a narrow
margin. Decode is44.1% and42.4% faster. Prefill includes the first output token;
decode times the remaining31. Each engine's complete token sequences are stable
across its warmups, measurements and fresh processes. Different quantizers are
allowed to generate different sequences; their quality is checked separately.

An earlier paired attempt was rejected because one RKLLM process returned only
zero token IDs and the next returned normal IDs. The final runner explicitly
resets the exclusively owned NPU before entering the vendor runtime. Tinygrad
already resets on device initialization. Repeating all256 RKLLM quality dumps
with reset produces identical score hashes, preserving the quality result.
The rejected run remains labelled incomplete; no output check was weakened.

Evidence lives under `/tmp/onpu-shard-20261018/`; reproduction tools are
`extra/rockchip/quality_inference.py`, `quality_rkllm.cpp` and `compare_rkllm.py`.
They use the retained checkout with source hashes; no source archive is required.

The final report is `/tmp/onpu-shard-20261018/paired-w8a8-reset/comparison.json`;
quality is `/tmp/onpu-shard-20261018/quality-reset-result.json`. To repeat timing
after a completed quality run against the same runtime source:

```sh
.venv/bin/python -m extra.rockchip.compare_rkllm --report-dir /tmp/onpu-w8a8-repeat \
  --precision int8 --stock-precision w8a8 --stock-embed-flash 1 \
  --quality-report /tmp/onpu-shard-20261018/quality-reset-result.json \
  --fused-qkv --cpu-norm --cpu-pool --cpu-fast-softmax --cache-cleaning --wake-helper
```

The app worker selects INT8 automatically only for the qualified F16 GGUF hash
`6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035`,
with the explicit Rockchip profile and context<=512. GPU and other checkpoints
retain their existing precision policy. The existing installed model is named
`Fp16 Hf`, ID `a06be3967dd7c548`; its file is the matched Qwen3-0.6B F16 GGUF.
Its real app load selected all197 prefill/decode projections, INT8, the decode
pool, vector softmax, cache cleaning and wake helper; cold preparation took125s.
Greedy chat returned `2 + 2 = 4.`, sampled SSE completed, and a simultaneous
OpenCL GPU request returned `Three plus three is six.`. These HTTP checks use
actual independent workers, not mocked backend availability.

After qualification, the original GPU Qwen3 session and SmolLM2 CPU/NPU session
were restored ready, with the GPU selected. The qualified INT8 model remains
installed and can be loaded with the Rockchip profile. Changes in both checkouts
remain uncommitted and unpushed.

INTENT: the retained backend supports FP16 matrices only; the user requires faster inference than RKLLM W8A8 with comparable quality; the selective backend permits explicit matrix kernels and the independently checked INT8 recipe gives exact INT32 products. Add an opt-in guarded INT8 path while preserving FP16 execution.

TWINS: searched gate*reciprocal*up and cached scale denominators - found 1 other site: tinygrad/llm/kernels/rockchip_cpu.py (FP16 packer; its complete half payloads are separately qualified).

## Verification and reproduction

Current checks:32 backend/GPT-2 tests,31 generation/tokenizer checks,24 app unit
tests, mypy227 source files, Ruff and Git whitespace checks. Tests use -n12 in the
tinygrad checkout. The optional OpenAI-client server fixtures cannot initialize
because that client package is absent; real application HTTP/browser verification
is recorded separately. No test or dependency was altered to conceal that limit.

To repeat the matched board experiment after installing the pinned checkpoint
and RKLLM artifacts at the paths in the runner:

```sh
.venv/bin/python extra/rockchip/compare_rkllm.py --report-dir /tmp/onpu-comparison --split-down --fused-qkv --cpu-norm
```

Use one NPU owner at a time. Unload an idle NPU app session before standalone
hardware measurements. Existing GPU sessions can remain loaded, but run timing
without concurrent generation or hardware tests. Both reference worktrees remain
intact and their uncommitted changes have not been discarded.

The local app remains available at http://127.0.0.1:8091 with GPU and CPU/NPU
sessions ready. Cold preparation of the NPU worker took about152 seconds while
warming all prompt sizes and both sampling modes. Exported evidence and source
review artifacts are in ~/oNPU/docs/rk3588-verification-20261018/.
