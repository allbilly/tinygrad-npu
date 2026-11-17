# Model support matrix — 21 models by HuggingFace downloads

Counts are the HF API `downloads` field (last 30 days), read 2026-11-17.

`native` = load + 4-token CPU forward in this checkout. Exact errors are in
`~/oNPU/analysis/upstream-support-20261117/results.jsonl`; the runner is
`~/oNPU/scripts/bench/run_upstream_support_matrix.py`.

FAIL notes give the root cause, the fix, and an estimated line count for load +
4-token forward (not quality parity). Reference code is in
`~/tinygrad/tinygrad/llm/`.

| rank | model           | HF repo                                      | downloads | arch          | native   | note                                          |
| ---: | --------------- | -------------------------------------------- | --------: | ------------- | -------- | --------------------------------------------- |
|    1 | Qwen3.5 4B      | unsloth/Qwen3.5-4B-GGUF                      | 1,607,513 | qwen35        | **PASS** | —                                             |
|    2 | Qwen2.5 1.5B    | Qwen/Qwen2.5-1.5B-Instruct-GGUF              |   585,214 | qwen2         | **PASS** | —                                             |
|    3 | Qwen3.5 2B      | unsloth/Qwen3.5-2B-GGUF                      |   512,840 | qwen35        | **PASS** | —                                             |
|    4 | Qwen3.5 0.8B    | unsloth/Qwen3.5-0.8B-GGUF                    |   336,991 | qwen35        | **PASS** | —                                             |
|    5 | Qwen3-VL 2B     | Qwen/Qwen3-VL-2B-Instruct-GGUF               |   255,752 | qwen3vl       | **PASS** | —                                             |
|    6 | Spark-X2.5 1.7B | XHToken/Spark-X2.5-1.7B-GGUF                 |   248,214 | spark2_5      | **FAIL** | bool-list sliding -> sliding_layers (~18)     |
|    7 | Qwen3 0.6B      | Qwen/Qwen3-0.6B-GGUF                         |   213,632 | qwen3         | **PASS** | —                                             |
|    8 | Gemma2 2B       | bartowski/gemma-2-2b-it-GGUF                 |    68,952 | gemma2        | **PASS** | —                                             |
|    9 | TinyLLAMA 1.1B  | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF       |    67,810 | llama         | **PASS** | —                                             |
|   10 | Gemma4 E2B      | bartowski/google_gemma-4-E2B-it-GGUF         |    59,189 | gemma4        | **FAIL** | no gemma4 -> gemma4.py (~226)                 |
|   11 | LFM2.5-VL 450M  | LiquidAI/LFM2.5-VL-450M-GGUF                 |    23,160 | lfm2          | **FAIL** | per-layer conv -> ShortConvBlock (~15)        |
|   12 | Phi3 3.8B       | bartowski/Phi-3-mini-4k-instruct-GGUF        |     9,613 | phi3          | **FAIL** | fused qkv+ffn_up -> split (~10)               |
|   13 | DeepSeekOCR 3B  | ggml-org/DeepSeek-OCR-GGUF                   |     5,984 | deepseek2-ocr | **FAIL** | `rope_dim=0` -> guard (~2)                    |
|   14 | Gemma3n 2B      | bartowski/google_gemma-3n-E2B-it-GGUF        |     3,122 | gemma3n       | **FAIL** | no gemma3n -> gemma3n.py (~176, needs gemma4) |
|   15 | ChatGLM3 6B     | mradermacher/chatglm3-6b-GGUF                |     2,084 | chatglm       | **FAIL** | fused gate/up + rotary -> chatglm.py (~67)    |
|   16 | TeleChat2 3B    | Tele-AI/TeleChat2-3B                         |       936 | telechat2     | **PASS** | —                                             |
|   17 | Qwen2 0.5B      | bartowski/Qwen2-0.5B-Instruct-GGUF           |       712 | qwen2         | **PASS** | —                                             |
|   18 | MiniCPM4 0.5B   | mradermacher/MiniCPM4-0.5B-GGUF              |       555 | minicpm       | **PASS** | —                                             |
|   19 | MiniCPM3 4B     | mradermacher/MiniCPM3-4B-GGUF                |       516 | minicpm3      | **FAIL** | fused attn_kv_b -> split + longrope (~25)     |
|   20 | RWKV7 1.5B      | mradermacher/RWKV7-Goose-World3-1.5B-HF-GGUF |       322 | rwkv7         | **FAIL** | no rwkv7 -> rwkv.py + dispatch (~120)         |
|   21 | InternLM2 1.8B  | mradermacher/internlm2-chat-1_8b-GGUF        |       215 | internlm2     | **PASS** | —                                             |

Spark-X2.6-4B is not a scoreboard row; by downloads it would rank second, so this
table is a selection, not a complete ranking. The counts were read from the HF API
on 2026-11-17 and are not stored in `results.jsonl`, so they cannot be re-derived
offline — treat them as a snapshot.

The 12 PASS rows are covered in `npu-gemm-baseline.md`: 9 measured, 3 blocked.
