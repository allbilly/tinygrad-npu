# Static Qwen2 prefill with K-split down projections

The oNPU recipe for checkpoint
`a07bd33b0e52babc1083e97d9bb6018197d7af25697552cc4d0b751e035158ff`
enables `ONPU_SWIGLU_K_SPLIT=1` at context 512. The backend capability is
limited to static 128-row SwiGLU with dimensions 896→4864→896, a shared CPU
pool, SDK gate/up packing, and the qualified batched activation producer.
Partial or symbolic chunks and scalar decode use the ordinary fused path.
Other app profiles keep their existing defaults.

The 4096-channel down group formerly used two M64/K4096 tasks. The split path
uses two M128/K2048 tasks with distinct halves of the same original INT8
weight codes. This reduces the scheduled weight bytes for that group while
preserving the original **4096-channel** activation maximum and scale. It
does not quantize each 2048-channel half independently.

CPU code adds the two INT32 partials before conversion to FP32 and before
dequantization. Converting/dequantizing the halves separately and adding
floats would change rounding. The original 768-channel group is dequantized
and added afterward, retaining the original group-addition order.

The combined upload packet occupies 624128 bytes:

| Part | Scale header offset | Payload offset | Payload shape |
|---|---:|---:|---|
| First half of K4096 | 0 | 512 | 128×2048 |
| Second half of K4096 | 262656 | 263168 | 128×2048 |
| Original K768 tail | 525312 | 525824 | 128×768 |

The first two headers contain identical scale bytes. The producer fully
writes headers and payloads, preserving the existing packed-upload and native
borrowing contracts. Its exponential polynomial, normal-range guard and
exceptional fallback match the production producer; nearest-even conversion
is bounded by the same row maximum. Tests compare every packet byte and
scale, including zeros, nonfinite inputs and overflow.

Preparation slices the original packed K-block axis and materializes two
standalone weight tensors. This adds 88080384 bytes (84 MiB) across 24 blocks.
The original weights remain available to decode. These are prepared inference
artifacts; weight changes require reconfiguration, as for other prepared
projections.

`split_prefill_stats()` reports preparation and program construction, not
per-request execution counts. oNPU additionally inspects the warmed static
capture before ready: positive native runs, no runtime variables, and 48
K2048 matmul calls. The actual backend/API qualification retains 32 full
vocabulary vectors and generated IDs exactly. CPU tests also cover the
integer addition, packed-weight slice order, shape fallback and rebinding.

Measurement receipts and source epochs are recorded in
`~/oNPU/docs/benchmark-qwen2-ksplit-production-20261101.json` and
`~/oNPU/docs/benchmark-qwen2-ksplit-source-20261101.json`. The two HTTP workers
each pass streaming, cached continuation, cancellation/recovery and the
448+64 context boundary. Independent original/RKLLM quality, model maximum,
and broader repeatability remain open.
