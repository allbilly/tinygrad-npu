# Qwen2 first-chunk and IRQ-helper work

`ONPU_STATIC_PREFILL=1` gives greedy, uncached full first chunks a separate capture with constant input extent and cache position zero. Partial chunks, cached prefixes and sampling keep the dynamic path. The CPU token buffer remains owned and realized; the new capture has no scalar bindings. Rockchip supplies the normal prefill scope and persistent workers. oNPU warms both full and partial captures before reporting ready.

`ONPU_ASYNC_IRQ_WAKE=1` makes IRQ-helper wake and stop requests nonblocking. The helper accesses only its own atomic state. A stop request cancels work immediately; reactivation can reuse a pending worker, and final close still joins it. The worker sets `busy` under its mutex so cancellation cannot race with a late activation. Switching back to synchronous operation drains any pending stop.

Under a bounded normal-priority CPU0 load, the former stop acknowledgement reached 281 ms. Nonblocking stop measured about 0.03 ms and passed 10,000 rapid re-entry scopes. Lifecycle tests also cover cancellation, restart, synchronous mode, mixed modes and final join. These are CPU scheduling measurements, not inference speed claims.

The selected Qwen2 profile also uses decode event waits. Their API cohort reduced request temperatures from roughly 78–81°C to 66–67°C. No board governor, clock limit, IRQ affinity or idle configuration was changed.

Evidence workspace: `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/`.

- `qwen2-static-prefill-production-v2/results/`: control 122.96 ms versus static 113.17 ms backend first-token medians, with exact 32-vector regression.
- `qwen2-deferred-irq-production-v1/results/qwen2.json`: final helper behavior; exact 32 full-vocabulary vectors and matching 64-token requests.
- `qwen2-deferred-irq-api-v1/results/qwen2-api.json` and `qwen2-deferred-irq-api-v2/results/qwen2-api.json`: two fresh workers, ten measured requests and 18 functional checks each.

The final pooled API result is **150.84 ms TTFT / 848.58 prefill-proxy tok/s / 48.85 decode tok/s**, for 128 uncached input and 64 actual output tokens at context 512. Prefill still trails published RKLLM 877.31 tok/s; decode leads 41.58 tok/s. Four of 20 requests beat the prefill reference. An earlier single-process 910 tok/s result did not repeat and is retained separately, not selected as the final result.

Smaller CPU FFN row batches, scratch padding, alternate CPU GQA tiles and NPU attention were evaluated and rejected. The NPU-attention proposal required a corrected non-multiple-of-four head partition, then passed its numerical error check but was over twice as slow. None of those prototypes was promoted. Diagnostic wrapper import/CPU faults remain recorded separately from production inference.

The exact M64/K4096 SDK down-projection captures were completed. Their M-split replay was about31%slower; grouped K-split was neutral. Neither was ported. Further output-copy, cache-maintenance, pointwise-dispatch and eight-row SwiGLU proposals also failed complete-model/worker gates. See `/home/orangepi/oNPU/docs/benchmark-qwen2-prefill-followup-20261031.json`. That exact fusion is now implemented and qualified as described below; it still misses the prefill target. The next experiment is exact packed-key/FP32-value attention with CPU bounds, numerical and complete-serving gates. Independent RKLLM quality and model-maximum context remain unqualified.

`ONPU_NORM_QUANT=1` now enables exact RMSNorm/row-quantization fusion for static128-row/896-wide inputs in the active shared scope. Other layouts use the ordinary path. Both packed layouts preserve every scale, header and padding byte; gamma remains a live input. The pinned app profile verifies both layouts before ready. Current API medians are858.73prefill /48.75decode tok/s, so prefill still misses877.31. See `/home/orangepi/oNPU/docs/benchmark-qwen2-norm-quant-20261031.json`.
