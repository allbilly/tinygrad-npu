# Sampled image prefill improvement

Complete: sampled image prefill in the live app improves from 725.69 to
975.87 tok/s, 34.5% higher throughput. Median prefill latency falls from 113.00
to 84.03 ms. The complete first-token boundary improves from 904.82 to 875.36 ms;
image encoding remains the largest part of that boundary.

- [x] Measure the existing ready worker with greedy and sampled image requests.
- [x] Capture the existing Gumbel-max sampling expression in embedding prefill.
- [x] Verify random state advances, temperature rebinding and chunked prompt behavior.
- [x] Warm the serving path, measure it again, and check streaming/cancellation/concurrent GPU serving.
- [x] Run backend/app tests, type checking and lint; record results and leave both workers ready.

Scope: Transformer embedding sampling, Rockchip phase wrapping, app VLM prefill
and worker preparation, their tests, and measurement scripts/reports. Decoder
weights, vision arithmetic and placement policy remain the qualified versions.

INTENT: sampled image prefill schedules Gumbel-max selection after decoder execution; the task requires faster serving; Transformer.forward documents Gumbel-max sampling from logits divided by temperature, which must retain fresh random draws and runtime temperature inputs.

The new sampling entry captures the original expression together with the decoder.
Temperature is a Tensor input so changing it reuses the prepared graph. Intermediate
chunks use the prepared greedy entry to fill the same KV cache without drawing
random numbers; only the final chunk samples. The worker warms greedy and sampled
embedding paths instead of the unused raw-logit serving path.

The isolated qualified INT8 decoder measures 96.54 ms with external sampling and
66.53 ms with captured sampling (849.35 to 1232.47 tok/s). Ten paired trials use
reversed execution order, identical RNG state and several temperatures. All ten
tokens and RNG updates agree exactly, and all ten captured calls use native replay.
This probe has 82 precomputed embedding positions and excludes image encoding and
prompt embedding lookup. It is distinct from the app's full prefill boundary.

The ready app before this change measured median sampled prefill 112.997 ms,
725.686 tok/s, versus greedy 73.554 ms, 1114.834 tok/s. Same uploaded JPEG,
82 decoder positions (including 64 image embeddings), temperature 0.6, five
measured calls per temperature after two warmups, alternating mode order.

| App measurement | Before | After |
|---|---:|---:|
| Sampled prefill (tok/s) | 725.686 | 975.868 |
| Sampled prefill (ms) | 112.997 | 84.028 |
| Sampled first token, including vision (ms) | 904.820 | 875.363 |
| Greedy prefill (tok/s) | 1114.834 | 1110.716 |

These are medians of five measured requests per mode, with the same boundary
and JPEG. The comparison uses one ready worker before and one fresh worker after;
the separate same-process native probe establishes the causal sampling overhead.
It is an oNPU before/after comparison, not a fresh sampled RKLLM comparison. Sampled
decode remains about 44 tok/s; this change targets the first sampled token.
Individual request latency varies; the functional 32-token smoke is diagnostic,
not part of the five-call median. No CPU tests or other inference run during
the measured timing calls.

Sampled streaming, Stop (0.8275 seconds), disconnect/recovery and same-server
overlapping GPU/NPU requests pass. Both workers finish ready, with the GPU model
selected and OpenCL still the default. Worker preparation takes 266.32 seconds;
these are warmed serving measurements. The unused raw-logit image path remains
available to callers but is no longer warmed by the serving worker.

Checks: 27 backend regression tests pass with one explicit hardware skip; all 37
app tests pass. Mypy checks 236 source files and whole-backend Ruff passes, as does
changed-app lint and whitespace validation. The native probe separately exercises
real Rockchip hardware with the full SmolVLM decoder. No frontend changes required
a new browser run. The previous browser evidence remains historical.

[Machine-readable evidence](rockchip-sampling-benchmark.json) records source
hashes, raw measurements, limits and the live app state. Raw evidence resides in
/home/orangepi/vlm-bench/sampling-*. The owned temporary status file has been moved
into that evidence directory; no benchmark process or temporary browser remains.

The existing 20-case RKLLM quality comparison remains historical evidence. This
change does not modify vision operations, quantization, weights or decoder logits.
New verification compares the unchanged sampling expression and RNG transitions
directly rather than claiming a new held-out VQA evaluation.

TWINS: searched image-prefill consumers for external Gumbel-max selection - found 0 remaining serving sites; ordinary token sampling in Transformer.forward was already captured and remains unchanged.
