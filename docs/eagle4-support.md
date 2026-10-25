# EAGLE-4 and oNPU

Checked 2026-10-23. EAGLE-4 is released: `eagle-harness` 0.1.0 appeared on October 20 and 0.1.1 on October 21.
This assessment pins the official repository at
[`ea02a3bdf29a20d7d635a635a2f1cbec1a443049`](https://github.com/eagle-hq/EAGLE-4/tree/ea02a3bdf29a20d7d635a635a2f1cbec1a443049).
The previous EAGLE repository is not the release location. See the official
[changelog](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/CHANGELOG.md).

**oNPU does not run an EAGLE-4 trained draft yet.** The official published drafts target much larger models than
our benchmark set. We can prepare shared verification and measurement machinery while finishing the actual,
downloaded Qwen3.5-2B DFlash2 path. That work must remain named DFlash2; it does not establish EAGLE-4 support.

## Available trained drafts

The [official publisher's model API](https://huggingface.co/api/models?author=EAGLE-Corp&limit=100&full=true) listed
these two drafts at inspection. Weight sizes below come from each repository's LFS metadata; no weights were
downloaded for this investigation.

| Trained target | Draft revision | Draft weights | Fit for the current board and benchmark targets |
|---|---|---:|---|
| [Qwen3.8-27B](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4) | `f58a5985223040552ad404fba98f0dd27f5f676a` | 6,178,152,384 bytes; 3,089,075,200 BF16 parameters | No matching target in our 22-model baseline. Target plus draft cannot fit in 16 GB at the published BF16 precision. |
| [Nemotron-3.5-Super-VL-120B-A12B](https://huggingface.co/EAGLE-Corp/Nemotron-3.5-Super-VL-120B-A12B-EAGLE4) | `41489e4ba72e079f9ddfd583b5e3fa2155d29cf0` | 8,230,464,792 bytes | No matching target; target size also exceeds the board's BF16 capacity. |

A public Hugging Face search for `EAGLE4` also found a community GGUF conversion of the 27B-target draft,
not a trained draft for our smaller targets. No compatible EAGLE-4 checkpoint for our Qwen3/Qwen3.5 models
was found in this inspection. This is a dated search result, not a claim that no additional checkpoint can appear.

The pinned [Qwen draft config](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4/blob/f58a5985223040552ad404fba98f0dd27f5f676a/config.json)
requires hidden width 5120, vocabulary 248320, auxiliary layer outputs `[3, 31, 59]`, one draft layer,
a 4096-token sliding window, and trained rank-128 candidate factors selecting 128 candidates.
Matching a tokenizer or model family is insufficient. A 2B/4B target has different hidden features and layers;
renaming these weights or substituting untrained candidate factors cannot make it compatible.

## What changes relative to EAGLE-3 and DFlash2

| Component | EAGLE-4 release | Consequence for oNPU |
|---|---|---|
| Draft generation | Autoregressive feature-conditioned drafting; the released Qwen draft has one layer. | This is a distinct trained model from our parallel masked DFlash2 draft. |
| Draft attention | Sliding window over committed context and the draft branch's ancestors. | Bounds draft attention work at long context; does not remove the target's context or KV costs. |
| Draft output head | A trained low-rank estimate selects candidates, followed by exact scoring of their full-head rows. | Could reduce a large vocabulary head's bandwidth cost, but requires its trained factors and a correct sparse-head implementation. |
| Speculative work | A measured-cost harness adapts draft depth and verification width. | Select by committed tokens per complete round time on this board, including rejection and cache-state costs. |
| Verification | Tree masks for attention and parent-specific convolution/recurrent states for hybrid targets. | A causal block forward is sufficient for a chain, but cannot verify sibling branches correctly. |

These distinctions follow the official
[architecture](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/docs/architecture.md)
and [draft structure](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/eagle_harness/models/eagle4_config.py).
Our downloaded DFlash2 checkpoint instead uses parallel masked block drafting, grouped dynamic convolutions,
and predecessor-conditioned candidate corrections; its target taps and weights must be implemented according
to that checkpoint's own training contract.

The published EAGLE-4 speedups use server GPUs and specific engines; they are not RK3588 results or RKLLM
comparisons. The release's [known limits](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/docs/known-limits.md)
also report high-load regressions and some greedy divergences. They explicitly refuse images on the SGLang
path. A draft's `VL` name alone therefore does not establish a working VLM serving path.

## Concrete preparation in this checkout

`tinygrad/llm/speculative_policy.py` provides an independent, opt-in measured-round budget selector.
It is a small throughput policy, not an EAGLE-4 harness port or a trained drafter. No EAGLE-4 source code is
incorporated. The upstream [license](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/LICENSE)
is source available with commercial and competitive restrictions, unlike tinygrad's MIT license.

The selector:

- Keys measurements by target/draft revisions, runtime fingerprint, precision, batch size and measured context band.
- Requires three qualified ordinary rounds and three qualified candidate rounds by default; excludes warmup.
- Uses total committed tokens divided by total round time, including snapshot/restore, rejected replay,
  observation, synchronization and other round work.
- Selects only drafter-supported budgets with a measured improvement above 5% by default; otherwise returns zero
  for ordinary decoding. It keeps bounded history and supports invalidating a failed budget.
- Does not initiate calibration, enable a worker flag, prove output quality or guarantee a future speedup.

For a DFlash2 implementation trained with seven speculative slots, construct
`MeasuredSpeculativePolicy((7,))`; do not assume it supports arbitrary shorter shapes. Record budget-zero
ordinary rounds under the same runtime key, then complete budget-seven rounds only after correctness qualification.
The caller must time synchronized, successful end-to-end rounds and invalidate evidence after a correctness
failure or relevant runtime change. It must supply a new key for a different load, kernel configuration or model.
Pass the smaller of remaining request output tokens and available model-context slots as `remaining_tokens`;
the policy does not infer model limits from a calibration band.

Remaining EAGLE-4 work is target-specific trained weights, an admitted draft loader and forward, sparse head,
ancestor attention masks, per-branch hybrid-state handling, accepted-path commit, and on-board correctness and
end-to-end timing. A compatible checkpoint is a prerequisite to a useful trained EAGLE-4 run. The official
[new-model guide](https://github.com/eagle-hq/EAGLE-4/blob/ea02a3bdf29a20d7d635a635a2f1cbec1a443049/docs/new-model.md)
describes training and export contracts but does not provide a reproducibly pinned complete training path for
an arbitrary new target. No EAGLE-4 speed or RKLLM win is claimed here.
