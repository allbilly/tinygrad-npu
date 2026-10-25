# SmolVLM decoder prefill improvement

Complete: the same prefill phase that measured 788 tok/s now measures 1301.56
versus fresh matched RKLLM W8A8 at 931.10 tok/s, approximately 40% faster.
The live app measures a 1107.71 tok/s median across three quiet requests.

A later [sampled-prefill improvement](rockchip-sampling-progress.md) also captures
Gumbel-max first-token selection, raising live sampled prefill from 725.69 to
975.87 tok/s. The measurements below remain the original greedy qualification.

- [x] Reproduce and profile the same decoder prefill boundary.
- [x] Fix measured overhead without changing model arithmetic or outputs.
- [x] Beat a fresh vendor reference with frozen-source reversed-order repeats.
- [x] Preserve held-out quality and complete VLM behavior.
- [x] Verify backend/app checks and restore both model workers.
- [x] Requirement-by-requirement current-state completion audit.

## Cause and change

The decoder took approximately 62 ms, while embedding graph setup added about
19 ms and external first-token selection added about 20 ms. The metric includes
all of these costs. Selection is now captured in the prepared greedy embedding
prefill graph. CPU prompt lookup uses a reusable compiled routine with fresh
owned output buffers; trace/training contexts retain the prior custom kernel.
Raw logits remain available for quality checks and sampling.

The app also copies CPU chunks into fresh padded storage directly and uses
warmed 32/64/96/128-row variants. The 82-position prompt uses 96 rows, while
128-position prompts retain one 128-row chunk. Real-transformer tests cover
82/128/159/333 positions, full-logit agreement and retained output ownership.

## Evidence

| Measurement | oNPU | RKLLM W8A8 |
|---|---:|---:|
| Decoder prefill, 32-output workload | 1301.56 tok/s | 931.10 tok/s |
| Decoder prefill, 128-output workload | 1292.39 tok/s | 914.73 tok/s |
| Complete 32-token response | 1133.63 ms | 1460.98 ms |
| Complete 128-token response | 2043.05 ms | 3138.18 ms |

The prefill boundary is unchanged: vision encoder time is excluded, while prompt
embedding, decoder execution and greedy first-token selection are included.
The 82 decoder positions include 64 image embeddings. Same SmolVLM-256M
checkpoint, 512-square image and context 512. Each engine/output length ran in
two fresh processes, three warmups and three measurements per process, with
ABBA/BAAB ordering and source/clock records. Timings exclude model loading and
image preprocessing; fixed output counts ignore EOS for equal work.

The 20-case quality gate passes with unchanged perplexity 1.2502969348 and 7/20
answers, matching vendor 7/20. All 40 saved image/logit files are byte-identical
to the preceding qualified implementation. Every new greedy first token agrees
with full-logit argmax. This is a limited quality proxy, not a full VQA suite.

The live browser passes image upload/streaming without exceptions and measures
1115.38 tok/s prefill. Three quiet API repeats measure 1136.55/1107.71/1101.31.
Sampling, Stop, disconnect/recovery and same-server overlapping GPU/NPU requests
pass. The app at http://127.0.0.1:8091 leaves both workers ready and the original
GPU model selected; OpenCL remains the default. SmolLM stays unloaded while the
VLM occupies the single NPU slot.

Checks: 28 backend tests pass with 5 explicit hardware skips; 36 app tests pass;
236 files pass mypy; whole-backend Ruff and changed-app lint/whitespace checks
pass. Tests cover changed IDs/images, independent outputs, capture fallback,
raw/greedy equivalence and adaptive chunk math. No clocks, IRQs, kernel modules,
archives, commits, pushes or reference repositories were changed. Owned browser
state is stopped and removed.

[Benchmark evidence](rockchip-prefill-benchmark.json) and
[completion audit](rockchip-prefill-verification.json) record current source,
inputs, raw reports, hashes, quality gates and live model status. Raw data:
/home/orangepi/vlm-bench/prefill-fixed-order and onpu-quality-prefill-fixed.

## Implementation record

INTENT: embedding-input prefill returns logits and the caller schedules selection separately; the task requires lower first-token prefill latency; Transformer.greedy_forward already captures selection with decoder execution.
INTENT: CPU prompt embedding reschedules an eager custom kernel per request; the task requires lower prefill overhead; the embedding entry validates one contiguous image region and must return independently owned outputs.
INTENT: app prefill padded every short prompt to 128 rows; the task requires faster real serving; native prefill already caches variants by input shape, allowing smaller staging buckets without splitting long chunks.
TWINS: searched embedding-prefill first-token consumers - found 2 generation consumers, benchmark_vlm.py and onpu/vlm.py; both use the prepared greedy entry. Raw-logit teacher/parity and sampling paths remain intentional.
