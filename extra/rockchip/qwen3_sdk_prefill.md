# Qwen3 0.6B gate/up prefill

The optional SDK prefill path now accepts `(M,N,K)=(128,6144,1024)`, split across three NPU cores as `(128,2048,1024)`. This is a bounded shape qualification, not a generalization to other widths or depths.

An observation-only GDB capture of the installed RKNN Matmul SDK produced one task for the exact per-core shape and checked all 262,144 integer outputs. The capture tracks GEM mappings, pre/post-submit structures and ordered command words; it does not patch the inferior or read live MMIO. It is a standalone SDK operation, not a dump of a complete RKLLM model.

The existing SDK A16/C4 command recipe needs these additional M128 fields for this particular width:

| Register | Value | Contract |
|---|---:|---|
| CNA 0x107c | 512 | Feature surface stride follows the blocked M dimension. |
| CORE 0x3014 | 127 | M128 output orientation. |
| DPU 0x405c | 127 | Corresponding output surface geometry. |

Addresses are freshly allocated and relocated by the normal runtime. The implementation retains the existing qualified CBUF configuration, command ordering, cache synchronization and blocking completion. Input quantization writes A16; the CPU consumer must read C4 output. Reusing the row-major consumer would be incorrect.

The raw SwiGLU consumer keeps the original direct division rounding (`silu_div`) and uses four-row batches with the persistent CPU pool. Earlier raw-FFN measurements without `cpu_prefill_pool=True` created temporary workers and do not establish the performance of this path. The separate exact normal-range exp2 path retains the existing polynomial and rounding contract. Decode retains the previous graph.

Evidence workspace: `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/`.

- `qwen3-gate-sdk-capture-v1/results/capture.json`: passive capture, SDK/library hashes and raw task evidence.
- `qwen3-gate-sdk-primitive-v1/results/primitive.json`: seven changed-input cases for both layouts, all integer outputs checked, native replay exercised. Paired primitive medians were 1.5084 ms ordinary / 1.2212 ms SDK.
- `qwen3-sdk-ffn-cpu-v1/results/cpu.json`: exact payload/header/scale checks, including invalid scales, infinite weights and zeros. CPU packing costs are measured separately from NPU gains.
- `qwen3-sdk-pooled-model-v1/results/`: raw-pooled control and SDK candidate. Candidate first-token median 139.41 ms; full 32-vector regression exactly matches the prior local recipe.
- `qwen3-sdk-production-v1/results/qwen3.json`: actual port, 140.99 ms first-token median / 40.79 decode tok/s; all 32 full-vocabulary vectors match the control.
- `qwen3-sdk-api-v1/results/qwen3-api.json`: separate real HTTP worker qualification. Use its completed report for serving numbers; backend timings do not establish API performance.

These are local numerical regression and published-reference speed comparisons. Independent RKLLM quality equivalence, model-maximum context and VLM image processing remain outside this qualification. Failed and losing token-block, affinity and CPU-unroll experiments remain in the same workspace.
