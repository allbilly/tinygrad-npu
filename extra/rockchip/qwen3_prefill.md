# Qwen3-0.6B prefill optimization

Target: exceed the recorded RKLLM W8A8 reference of 642.93 tok/s (128 tokens / 199.09 ms), with output quality checked separately. Report whole-model uncached prompt-to-first-token throughput, including the output head. Preserve GPU-first serving and explicit NPU selection.

- [x] Read the latest full-model component profiles and previous trial failures.
- [x] Complete zero-tail reuse in the packed FFN input path; CPU integration regression passes.
- [x] Measure CPU scheduling changes on identical inputs: FFN unroll four reduces 0.75–0.76 ms to 0.52 ms; balanced QKV heads reduces 0.33–0.36 ms to 0.29–0.30 ms. All compared output bytes match. These are standalone kernel timings, not model throughput.
- [x] Verify integrated kernels with CPU tests, mypy and Ruff.
- [ ] Run matched control/candidate/candidate/control model processes, then held-out quality checks. Reuse the existing benchmark/restore supervisor.
- [ ] Record every trial and compare against RKLLM; retain changes only when correctness and repeated throughput support them.
- [ ] Enable a model-scoped configuration only after qualification and verify the serving path.

`ONPU_ZERO_PAD_VIEWS=1` reuses only explicitly registered zero-tailed producer storage. `ONPU_SHARED_HOST_POOL=1` reuses four pinned CPU workers for the supported prepared INT8 prefill graph. `ONPU_PREFILL_CPU_TUNING=1` unrolls independent FFN activation vectors and balances query/key/value heads across the workers. The flags default off while full-model qualification is pending.

The tuning preserves per-vector arithmetic and reduction order. QKV changes assign disjoint heads to different workers without changing normalization, rotary embeddings or cache indexing. Generic head geometries that do not divide into four equal groups retain the original head order.

## First full-model result (2026-10-29)

Four unprofiled processes completed. Control process medians: 222.47 / 212.28 ms; tuned candidate: 214.06 / 195.64 ms (597.96 / 654.26 prefill tok/s). The aggregate of six measured candidate requests is 204.85 ms / 624.85 tok/s. Only one process median exceeds the RKLLM reference; repeatable parity is still unachieved. All 24 generated sequences and the 32-target teacher vectors match.

All 1,024 saved held-out target vectors independently match the prior candidate bitwise; perplexity remains 30.9449981977. The quality worker exited 1 after scoring because its frozen cleanup script attempted `.restore()` on a weak reference. That failed raw result is preserved alongside a separate independent verification receipt. The supervisor restored both serving models and both answered fresh requests.

Evidence: `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/qwen3-zero-pad-views-v3/` and `qwen3-zero-pad-views-v3-quality-postprocess-v1.json` in its parent directory.

Next bounded experiment: `ONPU_REPLAY_BORROW_BATCH=1` lets a synchronous CPU reader consume several private NPU outputs directly across intervening copies. Host-range accesses, overlapping NPU writes, CPU kernels, mutable consumers and later live reads prevent borrowing. Native invalidation retains the existing clean, aligned mapping checks and forgets pending outputs after CPU kernels or overlapping uploads. The option defaults off until full-model qualification.

## Host-tuning review

The batch-transfer experiment completed all five workers successfully: aggregate six-request candidate median 198.75 ms / 644.04 prefill tok/s versus matched control 212.31 ms / 602.89 tok/s. All 1,024 target vectors remain bitwise identical. Its margin over published RKLLM is only 0.17%, so stable parity is still unproven.

A scoped utilization-hint experiment passed lifecycle/numerical checks but was based on the wrong CPU-governor premise: this board uses ondemand, while uclamp controls frequency through schedutil. That code was removed. A fused RMSNorm/quantization prototype preserved 24 CPU cases but saved only ~8 microseconds per pair; it was also removed before full-model testing. The 297 numerical source hashes now exactly match the completed batch-transfer cohort.

The host audit in `~/oNPU/docs/benchmark-rk3588-host-tuning-audit-v1.json` records all three NPU IRQs effectively on CPU0, dynamic CPU/DMC governors, and a proposed temporary stock-frequency/IRQ comparison. No board settings have been changed. The existing CPU0 wake helper already mitigates interrupt-CPU sleep, so the external repository's idle-IRQ speedups are not directly transferable.

## Ready token inputs and app integration

A separate cProfile request showed about 21.94 ms in `_prepare_jit_inputs` under instrumentation, while native replay was reused and its validation took about 1.21 ms. The new opt-in `ONPU_CPU_TOKEN_INPUTS` path creates owned, ready CPU int32 token storage and preserves multi-token prefix views. Single-token inputs retain the original contiguous contract so cached continuation can reuse the decode JIT.

The full-model ABBA improves aggregate TTFT from 204.24 to 199.31 ms (626.72 to 642.22 prefill-proxy tok/s), with all 1,024 held-out full logit vectors exact. This backend cohort precedes the one-token guard; CPU regression and the API cover the guard afterward.

The oNPU worker selects the recipe only for the pinned Qwen3 F16 checkpoint at context 512 in Rockchip NPU+CPU mode with Auto/On prefill. It resolves environment options before importing model/backend modules and reports the eight resolved feature flags. API median first-text latency is 196.77 ms (650.49 prefill-proxy tok/s), with natural 59-token output from a 64-token budget. Five measured prefill rates span 627.42–711.39; a consistent RKLLM win is still unproven. Uncached/cached/streamed/context-boundary/cancel/recovery checks pass, and both original models are restored.

Records: `~/oNPU/docs/benchmark-qwen3-cpu-token-inputs-v1.json` and `~/oNPU/docs/benchmark-qwen3-api-fast-prefill-v1.json`.

## Score preparation candidate (pending full-model measurement)

`ONPU_ATTENTION_SCORE_VECTOR=1` vectorizes FP32 NPU-score loading, scaling and finite checks before unchanged softmax. It covers aligned four-score groups and retains scalar causal tails; finite maxima are reduced before the original softmax. Default remains off. Scalar/vector probability payloads match exactly across blocked and row-major layouts, partial rows and context boundaries; visible NaN/Inf lanes still reject. CPU-only normalization ABBA measured 1.262/1.088/1.087/1.261 ms on CPU4 for 16 heads and 128 active tokens. This standalone saving is not a model benchmark or a win claim.

Record: `~/oNPU/docs/benchmark-qwen3-score-vector-cpu-v1.json`. Full-model source-pinned ABBA is staged separately as `qwen3-score-vector-v1`.

The score-vector experiment finished its four timing workers: current recipe 667.79 tok/s versus candidate 655.35 tok/s. Rejected and removed the candidate. The 1,024-target worker failed at import before model load; do not claim that gate passed. Original sessions and fresh replies were restored. See `~/oNPU/docs/benchmark-qwen3-score-vector-v1.json`.

## Fixed-stock operating profile result

The retained numerical recipe measured 856.18 prefill-proxy tok/s and 38.58 decode tok/s across 12 requests in 4 processes with supported stock CPU/NPU/DDR performance governors, NPU IRQs on CPU6 and CPU6 deep idle disabled. All 12 beat the published 642.93/32.91 reference. No score-vector code is retained. The result is conditional on that temporary profile, context 512 and 128-input/64-output workload; it is not an independent quality-equivalence or model-max claim. Raw and restoration evidence: `~/oNPU/docs/benchmark-qwen3-stock-tuning-v1.json`.
