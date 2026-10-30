# MLA and MoE component profiles

Two model runs use three complete warmups and one diagnostic request with 128
input IDs, 64 raw outputs and context 512. Native replay component profiling is
enabled while global profiling stays off. Only the fourth request contributes
component timings; generated CPU sources and argument bindings are saved after
the timed region. These instrumented runs do not replace canonical timing rows.

| Model | Recorded prefill kernel time | NPU share | Main CPU costs |
|---|---:|---:|---|
| MiniCPM3 4B | 32.46 s | 4.6% | Query latent projection 11.93 s; latent QK 10.23 s; value projection 4.03 s; weighted values 2.02 s |
| DeepSeekOCR 3B(A570M) | 33.21 s | 0.3% | Routed expert gate/up 19.63 s; routed expert down 13.17 s |

MiniCPM3 has 62 blocks, 40 heads, query head width 96, rotary width 32 and
latent KV rank 256. Its generated query-projection C decodes Q8 block bytes and
scales, rounds decoded weights to FP16, and multiplies them inside every token's
dot products. Preparing contiguous FP16 per-head tables before tracing is a
direct experiment. Its latent QK dot over 288 channels is another major CPU
cost. The current generic softmax and reduction order provide an explicit
reference for subsequent attention work.

DeepSeekOCR has 12 attention blocks and 64 experts, with six selected per token.
Its 96 placed linears are not a count of transformer blocks. Two generic CPU
expert-matmul kernels account for 98.8% of recorded prefill time and 87.2% of
decode time. They gather selected FP16 expert weights and perform ordinary
batched dot products. Faster attention alone would leave almost all of this
prefill cost in place. Grouped expert execution, compact weights and vector
integer kernels are experiment targets, with separate numerical and quality
qualification required.

These percentages describe the original expert implementation. The later
[grouped CPU expert experiment](cpu_experts.md) measures 95.11 prefill proxy
tok/s and 23.29 decode tok/s, with all 32 checked full-vocabulary vectors bitwise
equal to its matched control. It still trails RKLLM's 182.54/31.60. A subsequent
diagnostic of this grouped path attributes 81.17% of recorded prefill kernel time
and 50.42% of decode time to CPU expert projections. NPU execution alone accounts
for 4.46% and 39.24%; copies are separate. The percentages in the earlier table
must not be reused as measurements of the grouped path.
[Grouped-path profile and restoration](../../../oNPU/docs/benchmark-deepseek-cpu-experts-profile-verification-v1.json).

Exact CPU source hashes, per-call bindings, profiles, checkpoint identities,
output comparisons and original serving restoration are in the
[diagnostic proof](../../../oNPU/docs/benchmark-mla-moe-profile-verification-v1.json).
NPU share refers to recorded native kernel-call time, excluding setup and host
work outside those calls. Both original models were restored with their prior
options and answered fresh requests. No model speed or comparable-quality
RKLLM win is established by these profiles.
