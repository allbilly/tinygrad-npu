# NPU GEMM baseline — 11 of 12 models measured

Status: 11 of 12 measured; Qwen3.5 4B is blocked by host RAM. Raw reports in
`results/<slug>/report.json`, driver in `extra/rockchip/sweep_npu.py`.

**No row below is cold-start revalidated.** Start temperatures for the measured
rows were 46.2–72.1 °C, none inside the 52–55 °C band, and `report.json` records
neither clocks nor temperature. These are baseline numbers, not publishable
claims.

## What is measured

Every `nn.Linear` is replaced by `NPULinear` (`extra/rockchip/npu_linear.py`),
which packs the weight once into the NPU layout and issues one FP16 matmul per
**(row, column, K) tile** — `row_tile` on rows, 24576 columns when the row tile is
1 (decode) and 12288 otherwise, 4096 on K. Attention, norms, softmax, rope,
embedding, sampling and the KV cache stay on the CPU host.

Protocol: 128-token prompt, 64 greedy decode tokens, 1 warm-up + 3 measured
passes, every trial reported. Prefill is the proxy `128000 / median_ttft_ms`.

3 of the 9 measured rows fell back to a **synthetic** prompt because the tokenizer
failed: InternLM2 1.8B, MiniCPM4 0.5B, TinyLLAMA 1.1B. They are marked `*` in the
results table. Their prefill numbers are not comparable to the tokenizer rows at
the same prompt length — note that MiniCPM4's 110.87 tok/s, the best prefill in
the table, is one of them.

## Models

Order is the HuggingFace download ranking from the reference tree, PASS rows
only. RKLLM columns are the published reference rates.

|   # | model          | arch      | checkpoint                                                 | RKLLM prefill | RKLLM decode |
| --: | -------------- | --------- | ---------------------------------------------------------- | ------------: | -----------: |
|   1 | Qwen3.5 4B     | qwen35    | `checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf`              |         61.87 |         6.22 |
|   2 | Qwen2.5 1.5B   | qwen2     | `qwen2.5-1.5b-instruct-q8_0.gguf`                          |        338.35 |        16.69 |
|   3 | Qwen3.5 2B     | qwen35    | `checkpoints/Qwen3.5-2B/Qwen3.5-2B-Q8_0.gguf`              |        164.75 |        13.59 |
|   4 | Qwen3.5 0.8B   | qwen35    | `checkpoints/Qwen3.5-0.8B/Qwen3.5-0.8B-Q8_0.gguf`          |        217.78 |        27.05 |
|   5 | Qwen3-VL 2B    | qwen3vl   | `checkpoints/Qwen3-VL-2B/Qwen3VL-2B-Instruct-Q8_0.gguf`    |        333.66 |        14.98 |
|   6 | Qwen3 0.6B     | qwen3     | `oNPU/.data/models/72717a59205944dc-Qwen3-0.6B-Q8_0.gguf`  |        642.93 |        32.91 |
|   7 | Gemma2 2B      | gemma2    | `checkpoints/Gemma2-2B/gemma-2-2b-it-Q8_0.gguf`            |        213.90 |        10.37 |
|   8 | TinyLLAMA 1.1B | llama     | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`                       |        524.74 |        24.43 |
|   9 | TeleChat2 3B   | telechat2 | `checkpoints/TeleChat2-3B/TeleChat2-3B-f16.gguf`           |        206.96 |        10.18 |
|  10 | Qwen2 0.5B     | qwen2     | `Qwen2-0.5B-Instruct-Q8_0.gguf`                            |        877.31 |        41.58 |
|  11 | MiniCPM4 0.5B  | minicpm   | `checkpoints/MiniCPM4-0.5B/MiniCPM4-0.5B.Q8_0.gguf`        |        946.12 |        45.34 |
|  12 | InternLM2 1.8B | internlm2 | `checkpoints/InternLM2-1.8B/internlm2-chat-1_8b.Q8_0.gguf` |        336.72 |        15.45 |

Checkpoint root: `/home/orangepi/vlm-bench/baseline-20261021/`, except Qwen3 0.6B
which is `/home/orangepi/oNPU/.data/models/`.

## Results

11 of 12 measured. Prefill is `128000 / median_ttft_ms`; the `x` columns are how
far behind RKLLM the baseline sits. `*` marks a synthetic prompt.

|   # | model           | prefill tok/s | decode tok/s | RKLLM prefill | RKLLM decode | prefill gap | decode gap | linears |
| --: | --------------- | ------------: | -----------: | ------------: | -----------: | ----------: | ---------: | ------: |
|   1 | Qwen3.5 4B      |             — |            — |         61.87 |         6.22 |           — |          — |       — |
|   2 | Qwen2.5 1.5B    |         20.66 |         3.75 |        338.35 |        16.69 |       16.4x |       4.4x |     197 |
|   3 | Qwen3.5 2B      |          2.07 |         3.03 |        164.75 |        13.59 |       79.7x |       4.5x |     187 |
|   4 | Qwen3.5 0.8B    |          1.99 |         4.09 |        217.78 |        27.05 |      109.6x |       6.6x |     187 |
|   5 | Qwen3-VL 2B     |         50.26 |         3.80 |        333.66 |        14.98 |        6.6x |       3.9x |     197 |
|   6 | Qwen3 0.6B      |         38.80 |         5.63 |        642.93 |        32.91 |       16.6x |       5.8x |     197 |
|   7 | Gemma2 2B       |         30.52 |         3.07 |        213.90 |        10.37 |        7.0x |       3.4x |     183 |
|   8 | TinyLLAMA 1.1B* |         53.23 |         5.61 |        524.74 |        24.43 |        9.9x |       4.4x |     155 |
|   9 | TeleChat2 3B    |         22.71 |         3.12 |        206.96 |        10.18 |        9.1x |       3.3x |     169 |
|  10 | Qwen2 0.5B      |         81.33 |         6.23 |        877.31 |        41.58 |       10.8x |       6.7x |     169 |
|  11 | MiniCPM4 0.5B*  |        110.87 |         7.73 |        946.12 |        45.34 |        8.5x |       5.9x |     169 |
|  12 | InternLM2 1.8B* |         27.01 |         4.24 |        336.72 |        15.45 |       12.5x |       3.6x |     169 |

### What the numbers say

**Decode sits in a narrow band, 3.03–7.73 tok/s, regardless of architecture.**
Dividing the measured token time by the Linear count gives an **upper bound** of
0.77–1.90 ms per Linear, because token time also contains CPU attention, norms,
sampling and the KV cache. Against a 0.044 ms median raw submit (5 trials,
0.042–0.071 ms) that bounds overhead at 94.3–97.7%.

Three caveats. The probe was M=64, N=64, K=896 with zero-filled operands and
`cores=1` — a submit-cost floor, not the M=1 decode shape; a retained
real-workload artifact measures **0.1105 ms** for a `[1,896,896]` INT8 GEMM
(`~/oNPU/docs/benchmark-natural896-wide-golden-probe-v1.json`), 2.5x higher. Its
only record is the session transcript (`ses_e2ecd054effeSPzmuGpHUtCi2X`,
`msg_1e8ae6b2f0013Vs79ug20M7YbW`), not a results file. And the RKLLM columns are
whole-model INT8 at maximum clocks while this baseline is FP16 linears-only at
unrecorded clocks, so the gap mixes scheme, clock and overhead.

**Prefill is architecture-dependent and much worse.** The two Qwen3.5 rows reach
~2.0 tok/s, 80x–110x behind RKLLM, which is the SSM scan explained below. Every
other row is 6.6x–16.6x behind; within that group Qwen2.5 1.5B is the slowest at
20.66 tok/s and MiniCPM4 0.5B the fastest at 110.87 (synthetic prompt).

**First-trial TTFT is 1.25x–3.75x the later trials**, and 9 of 11 rows are above
2x: MiniCPM4 3.75x, Qwen2 0.5B 3.74x, Qwen3-VL 3.41x, TinyLLAMA 2.98x, Gemma2
2.82x, TeleChat2 2.31x, Qwen3 0.6B 2.22x, Qwen2.5 1.5B 2.19x, InternLM2 2.09x,
Qwen3.5 2B 1.34x, Qwen3.5 0.8B 1.25x. That is consistent with JIT capture, and it
is why the protocol keeps a warm-up pass and reports the median rather than a
single trial.

### Blocked row — Qwen3.5 4B never loaded

Only the 4B is blocked, and on RAM, not IOVA. Gemma2 2B and TeleChat2 3B used to
fail at `MEM_CREATE` with errno 14; multi-domain weight placement fixed both, and
they now measure 30.52 / 3.07 and 22.71 / 3.12 tok/s.

The 4B still hits IOVA on a load-only probe — `failed to allocate IOVA: -12`,
`dma map 47185920 fail` — but with domains in place it dies on RAM first. Both
limits are real; which one fires depends on how far packing gets.

| model                                  | FP16 packed linears | outcome                        |
| -------------------------------------- | ------------------: | ------------------------------ |
| Qwen3.5 4B                             |             8.41 GB | 4x OOM kill, one IOVA refusal  |
| TeleChat2 3B (f16)                     |             5.34 GB | now measures, 2 domains        |
| Gemma2 2B                              |             5.23 GB | now measures, 2 domains        |
| largest single-domain row (Qwen3.5 2B) |             3.76 GB | fits in one domain             |

Packed sizes are the 2-D weights at 2 bytes, plus a packed copy of `token_embd`
when the head is tied — which it is for Qwen3.5 4B, Qwen3.5 2B and Gemma2. The
single-domain boundary sits between 3.76 GB and 5.23 GB. The ~4 GB per-domain
window is independently supported by the reference's git-tracked allocation
inventory (`~/oNPU/docs/benchmark-w8a8-allocation-context-v3.json`, commit
`cf07c7c`), which records `span_bytes: 4293414912` — 4.29 GB of tracked DMA
ranges — and an errno-14 refusal of a 20 MB buffer while 5 GiB was free.

### Qwen3.5 4B — every attempt, in order

The RSS figures come from the kernel ring buffer, captured to
`results/oom-stack.log` before the 19:54 reboot that would have lost them. A 30 s
dmesg snapshot now persists to `/var/log.hdd/dmesg-crash.log` (host setup:
`~/scripts/setup-crash-log.sh`), because `/var/log` is `/dev/zram1` and the
journal dies with the machine.

| attempt                | outcome                                    | evidence                          |
| ---------------------- | ------------------------------------------ | --------------------------------- |
| 1                      | OOM kill, `shmem-rss` 3.56 GB              | `RKNPU: failed to get pages: -12` |
| 2                      | OOM kill, `shmem-rss` 3.50 GB              | `RKNPU: failed to get pages: -12` |
| 3                      | OOM kill, `shmem-rss` 6.56 GB              | `RKNPU: failed to get pages: -12` |
| 4 (load-only probe)    | allocation refused, IOVA exhausted         | `failed to allocate IOVA: -12`    |
| 5 (with IOMMU domains) | OOM kill, `shmem-rss` 4.67 GB              | `RKNPU: failed to get pages: -12` |

The OOM stack was `rknpu_gem_object_create → drm_gem_get_pages →
shmem_read_mapping_page_gfp` — RAM exhaustion inside NPU buffer allocation, not
IOVA. It is quoted verbatim in `results/oom-stack.log`, which also carries the
`page allocation failure` dump from the allocation path itself. So the 4B is
blocked by both limits depending on when it is measured: attempts 1–3 and 5 died
on RAM, attempt 4 on IOVA.

Attempt 5 is the informative one. It ran with IOMMU domains in place, so the IOVA
wall was gone, and it still died on RAM at 4.67 GB of NPU buffers. Domains fix
addressing, not residency: 8.41 GB of FP16 weights must be resident at once. The
reference's INT8 packed total for the same model is 3.92 GiB, less than half.

What holds that RAM is the packed FP16 itself, not the CPU-side weights:
`anon-rss` was 33 MB at the kill while `shmem-rss` was 4.67 GB, so the Q8 source
is file-backed page cache. The co-consumer is zram — its compressed pages live in
RAM, and the second OOM's `Mem-Info` showed 7.7 GB anon (mostly zram) + 8.4 GB
shmem + 1.5 GB unevictable against a 15.9 GB board. Swapping to zram therefore
accelerates the OOM instead of relieving it. Deferring CPU weight realization
would not help; halving the packed weights (INT8) or freeing zram would.

`prepare()` realizes each tile as it is packed and drops `self.weight`
afterwards, because a lazy graph would hold every tile's temporaries at once.
That lowers peak RAM on every other model; no before/after RSS was measured.

### The reference stack already solves the 4B

The reference tree hit the same wall and its fix has since moved well past it:

| reference stage                                            | prefill tok/s | decode tok/s |
| ---------------------------------------------------------- | ------------: | -----------: |
| Oct 26, two memory domains + deferred weight realization   |         32.20 |         4.84 |
| gated-attention follow-up                                  |         41.74 |         5.23 |
| **current selected row**                                   |    **101.34** |     **6.48** |
| RKLLM published                                            |         61.87 |         6.22 |

So the current reference row is numerically ahead of RKLLM on both phases. Read
that with the reference's own caveats: the comparison is against a maximum-clock
published claim, the 101.34 run used adaptive governors, and independent
output-quality equivalence remains unqualified. The two-domain fix is the entry
ticket — it makes the model load at all — and the rest of the gap is closed by
kernel work.

### Qwen3.5 prefill is gated by the CPU SSM scan, not by the linears

Both Qwen3.5 rows land at ~2 tok/s prefill regardless of size, while the
same-size-class Qwen2.5 1.5B reaches 20.66 tok/s:

| model         | arch   | blocks | SSM layers | prefill tok/s | decode tok/s |
| ------------- | ------ | -----: | ---------: | ------------: | -----------: |
| Qwen2.5 1.5B  | qwen2  |     28 |          0 |         20.66 |         3.75 |
| Qwen3.5 0.8B  | qwen35 |     24 |         18 |          1.99 |         4.09 |
| Qwen3.5 2B    | qwen35 |     24 |         18 |          2.05 |         3.04 |

Qwen3.5 is a hybrid: `qwen35.full_attention_interval = 4`, so 18 of 24 blocks
are gated-delta-net SSM layers and only 6 are full attention. In
`tinygrad/llm/model.py` the SSM scan takes the fused-kernel path only when
`head_k_dim % 32 == 0 and head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device)`
(both shape conditions hold for Qwen3.5); on the CPU it falls back to a Python
loop over every prompt token:

```python
for t in range(T_pad):
    s1 = state * alpha[:, :, t]
    delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
    state = s1 + delta * k[:, :, t]
    outs.append((state * q[:, :, t]).sum(-1))
```

18 SSM layers × 128 prompt tokens = 2304 Python-level iterations of small tensor
ops. **Attribution, not ablation:** Qwen2.5 1.5B does the same 128-token prefill
in 6.2 s, so the scan accounts for roughly 58 of the 64 s, but no run isolates it.
The two sizes coincide because the loop count depends on the layer count and
`full_attention_interval`, which are identical.

The reference tree already carries the kernel for this scan —
`rockchip_recurrent.c` (491 lines) plus `rockchip_recurrent.py` (115). NPU
attention alone would not move these rows.

## Why `DEV=CPU`

The baseline is the CPU host plus NPU linears, and `DEV=CPU` selects that host.
It is not optional on this board: the default device is `CL`, and a run without
`DEV=CPU` dies at `parameter.realize()` with an LLVM IR renderer error on a
2-element vector index:

```
RuntimeError: src:4:42: error: '%reg_0' defined with type '[2 x i64]*' but expected 'i64*'
  %v0 = getelementptr inbounds i64, i64* %reg_0, i32 0
```

The failure is not root-caused; only the correlation with `DEV=CL` is observed.
It is retained verbatim in `results/dev-cl-llvm-error.log` and
`results/token-parity.log:116`, both of which open with the Mali/OpenCL loader
line (`arm_release_ver of this libmali is 'g6p0-01eac0'`) and end at
`bench_llm.py:61 parameter.realize()`. The model in those logs is MiniCPM4 0.5B,
not Qwen3.5 4B, but the claim is model-agnostic.
