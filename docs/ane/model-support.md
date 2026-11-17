# Model support — small models on the ANE baseline

The selection starts from the 21-row HuggingFace download matrix on the
`rkllm` branch (`docs/rknpu/model-support.md`, counts read 2026-11-17). That
matrix establishes which architectures load and run four CPU tokens in this
checkout; it says nothing about ANE execution. The ANE column below is the
separate result.

`native` = load + 4-token CPU forward in this checkout. `ANE reference` = a
model-specific path in `~/ane`, not just the generic `ane_matmul.c` runtime.
`ANE baseline` = the generic `ANELinear` path in `extra/ane/ane_linear.py`.

## Smallest candidates

Order is parameter count, not download rank. All rows except GPT-2 are `native`
PASS on the `rkllm` matrix; GPT-2 is not in that matrix.

| model | params | Q8_0 GGUF | tinygrad arch | ANE reference | ANE baseline | note |
| --- | ---: | ---: | --- | --- | --- | --- |
| GPT-2 | 124M | — | none | `~/ane/gpt2` (validated) | not wired | absolute smallest; `tinygrad/llm` has no GPT-2 loader |
| Qwen2 0.5B | 0.5B | ~0.5 GB | `qwen2` | none | generic | smallest PASS row (tied with MiniCPM4) |
| MiniCPM4 0.5B | 0.5B | ~0.5 GB | `minicpm` | none | generic | tokenizer failed on the `rkllm` sweep |
| Qwen3 0.6B | 0.6B | ~0.64 GB | `qwen3` | none | generic | smallest Qwen3-generation model with a working tokenizer |
| Qwen3.5 0.8B | 0.75B (named 0.8B) | 812 MB | `qwen35` | `~/ane/qwen35` (validated) | generic | **recommended first run** |
| TinyLlama 1.1B | 1.1B | ~1.1 GB | `llama` | none | generic | tokenizer failed on the `rkllm` sweep |
| Qwen2.5 1.5B | 1.5B | ~1.6 GB | `qwen2` | none | generic | fits, but larger than the recommended first run |

The Qwen3.5 0.8B size is the actual `content-length` of
`unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf` (811,843,840 bytes). The
other sizes are nominal Q8_0 estimates, not measured downloads.

## Recommendation

**Run Qwen3.5 0.8B Q8_0 first.** It is not the absolute smallest model, but it
is the only sub-1B model in the `rkllm` matrix whose ANE path is already
validated in `~/ane`: `qwen35/` runs all 96 body projections on the M1 ANE with
a measured maximum full-logit NRMSE of 0.381% against its CPU reference. The
generic `ANELinear` path reuses the same `ane_matmul.c` plan and submission
code, so the hardware and driver are exercised on a known-good family before
testing other architectures. The 0.381% figure does **not** transfer to the
generic path: it uses plain FP16 with power-of-two input scaling, not the
reference's Hadamard basis and compensated multi-plane policy, and it keeps
projections below 768 elements on the CPU. The 8 GB host (7.3 GiB usable) has
enough room for the 812 MB Q8_0 checkpoint and the packed FP16 plans, including
the chunked vocabulary head. The optimized generic path ran end-to-end on
this host: 139 linears (the head is chunked onto the ANE), the same token IDs as
the CPU row, and 11.54 tok/s decode versus 1.97 on the CPU at 32 prompt tokens,
matching the `~/ane` 11.57 reference; a 4-token run reached 11.80. The remaining
prefill gap is the CPU recurrent/norm graphs, not the ANE matmul.

**Run Qwen3 0.6B second.** It is the smallest Qwen3-generation model with a
working tokenizer in the `rkllm` matrix and a native `qwen3` loader here. It has
no model-specific ANE reference, so it is the first real test of the generic
matmul path across a different architecture. Its width (1024) and FFN (3072)
are both above the 768-element ANE template floor.

**Qwen2 0.5B and MiniCPM4 0.5B are the smallest PASS rows**, but they are older
architectures and have no ANE reference. Use them only if the 0.6B run exposes
an architecture bug and you need another data point.

**GPT-2 124M is the absolute smallest and already runs on the ANE** in
`~/ane/gpt2` (58.18 generated tok/s on base M1 Asahi), but `tinygrad/llm` has no
GPT-2 loader. Wiring it would add a second model path for no ANE-runtime gain,
so it is out of scope for this branch.

## Model files already on this host

The ANE and `qwen3.c` caches do not contain a ready-to-run tinygrad GGUF for the
small models:

| path | format | tinygrad/llm support |
| --- | --- | --- |
| `~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf` | GGUF, `qwen35`, Q8_0 (812 MB) | **yes** — used by this baseline |
| `~/.cache/Qwen3.8-2B-Q4_K_M.gguf` | GGUF, `qwen35`, Q4_K_M (1.31 GB) | **yes** — 139 ANE Linears (the 508M head is chunked), 6.17 prefill / 6.36 decode tok/s pinned, exact CPU tokens, ~23 s plan preparation |
| `~/.cache/qwen3.c/Qwen3-0.6B/qwen3-0.6b-q8.bin` | qwen3.c custom Q8 container (magic `1cja`) | no — not GGUF |
| `~/.cache/qwen3.c/Qwen3-0.6B/model.safetensors` | HF safetensors, Qwen3 0.6B (1.5 GB) | no — this branch's `tinygrad/llm` loads GGUF only |
| `~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16` | MLX safetensors (1.2 GB) | no — MLX layout, not GGUF |
| `~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit` | MLX 4-bit (336 MB) | no — MLX layout, not GGUF |
| `~/.cache/huggingface/hub/models--trymirai--Qwen3.5-0.8B-M` | metadata only | no — the checkpoint weights are not downloaded |
| `~/.cache/huggingface/hub/models--openai-community--gpt2` | refs only | no — the checkpoint weights are not downloaded |
| `~/.cache/huggingface/hub/models--openai--whisper-tiny` | refs only | no — the checkpoint weights are not downloaded |

The 2B GGUF is writable now (`0644`). tinygrad's disk loader opens the file
`O_RDWR`, so a read-only copy needs `chmod u+w` before `gguf_load`. Its logical
state is 7.77 GB in FP32 and about 3.9 GB after the default FP16 cast. A
one-token CPU forward completed in 15.0 s with 2.5 GB peak RSS and no swap, so
it fits, but it is 2B and not the smallest first run.

## What is not claimed

- No ANE result in this checkout is a quality-parity result. The generic path
  uses plain FP16 with power-of-two input scaling; it does not apply the
  reference's Hadamard basis or compensated multi-plane policy.
- No row is a performance result against `rkllm`, RKLLM, CoreML, or the
  `~/ane` Qwen3.5 numbers. The protocols and hardware differ.
- The `rkllm` NPU baseline numbers are not reproduced here. They are on the
  `rkllm` branch and apply to the RK3588 NPU, not the Apple ANE.
