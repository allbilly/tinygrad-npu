# Shared host workers for prepared INT8 prefill

`ONPU_SHARED_HOST_POOL=1` opts into the shared dispatcher when `configure`
admits a fully prepared dense INT8 prefill graph. It requires CPU workers,
prepared QKV, fused FFN, CPU normalization and supported NPU prefill attention
for every attention block. Placement reports both the request and whether the
path was admitted. Unsupported graphs keep their existing dispatch. The
option is off by default; enable it before configuring a fresh model.

The five large CPU kernel families are RMS normalization, activation
quantization, output dequantization, QKV preparation and raw SwiGLU packing.
Their arithmetic and four work partitions are unchanged. Only programs built
in the eligible prefill scope use the shared dispatcher. The caller uses CPU4
when available; the three persistent workers use CPU5–7. They remain active
during prefill and park on exit, including exceptional exits. The caller's
affinity and the program-construction context are restored. Decode keeps its
existing worker scope.

Model and JIT metadata use weak handles so weight discovery does not traverse
captured graphs or native runtime objects. Debugging code can retrieve the
state with `model_shared_host_pool(model)`.

A small native bridge selects active dispatch inside prefill and parked
dispatch for cached programs called outside it. If dispatch is unavailable,
the same four work partitions run sequentially. A stopped pool cannot enter a
new active scope. The bridge remains owned by the cached shared-pool state so
compiled callbacks do not outlive it. Repeated configuration unwraps phase
wrappers before creating a new one, avoiding nested pool locks.

The C source transformation refuses unfamiliar worker templates rather than
silently leaving local-thread work behind a single-core caller. Serial small
shapes keep their original program. With the feature disabled, the kernel
factories return their original cached programs.

The staged implementation measured 221.84–230.00 ms TTFT on Qwen3-0.6B versus
253.47–267.57 ms controls at 128 input/64 output/context512, with adaptive
clocks. All 32 teacher vectors and 1,024 held-out vectors matched bitwise.
Those are prototype measurements; current-source model requalification is
recorded separately. They do not establish RKLLM prefill parity or all-model,
VLM, API or maximum-context qualification.
[Prototype model evidence](../../../oNPU/docs/benchmark-qwen3-shared-host-pool-v1.json).
