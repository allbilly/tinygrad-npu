# API context and stability verification

Testing complete: **advertised model-maximum serving is not qualified under the
current settings**. The HTTP API remains responsive; the tested 512-token VLM
path passes its bounded matrix. Both original 512-token workers are restored,
answer fresh API requests, and leave OpenCL selected.

- [x] Exercise the ready VLM at short, long and exact-limit prompts, including multi-turn and repeated requests.
- [x] Check VLM overflow rejection, streaming usage, Stop, disconnect/recovery and worker/resource health.
- [x] Verify advertised model maxima: Qwen 40,960 versus API 32,768 and SmolVLM 8,192.
- [x] Attempt SmolVLM maximum startup; record that it does not become ready before the normal startup timeout.
- [x] Record actual prompt lengths, latency, correctness and limits separately from transport stability.
- [x] Restore the original model configuration and leave workers ready.

Scope: a reusable live-API test script and verification reports. Current VLM
acceleration is qualified only at context 512. Maximum-context testing temporarily
reloaded both workers in sequence. Global settings were preserved and both workers
were restored to 512 afterward.

These are bounded soak and boundary tests, not a claim of day-long reliability
or a comprehensive long-context quality benchmark.

The user specified **model max** after the initial 8K target. Qwen declares 40,960,
but the API schema currently rejects loads above 32,768. SmolVLM declares 8,192.
The GPU at 8,192 takes more than 222 seconds without completing an uncached 1,024-token
request. Its 8,184-token streaming probe receives no text before the 180-second
deadline (17 keepalives). Therefore the full 4K/8K soak was shortened after the
observed latency failure; those unrun retrieval cases are not counted as passes.

## Maximum-context results

| Target | Observation | Outcome |
|---|---|---|
| Qwen3-0.6B, declared 40,960 | Load returns 422; API LoadOptions caps context at 32,768 | Model maximum unavailable through the current API |
| Qwen3-0.6B, API 32,768 with the VLM resident | RAM admission returns 400; estimate 9,884,610,208 bytes exceeds the projected available upper bound 9,378,344,960 | Guard prevents this co-load; standalone 32K was not attempted |
| Qwen3-0.6B, context 8,192, prompt 8,184 + reply 8 | Stream opens but emits no content before the 180-second deadline;17 keepalives | Fails the bounded interactive-use check |
| SmolVLM, declared 8,192 | Remains preparing through image encoding near 300 seconds, then worker stops; never reports ready | Maximum-context inference remains unqualified |

The app's startup maintenance stops preparing workers after 300 seconds and clears
their model identity. The controller observes `stopping` followed by a missing
model ID, so its raw exception is `StopIteration`; this is not a completed
inference test. A separate isolated reproduction of the exact timeout policy
confirms state `error`, message `Model preparation timed out...`, and `model_id` `None`.
The 8K VLM prompt matrix was therefore not run. Product source was not changed.

At GPU context 8,192, subsequent 128-token prompts still complete correctly but
need approximately 37–39 seconds to first text. Stop recovers in 7.81 seconds and
the tested streaming disconnect recovers in 0.20 seconds. All three recovery
answers are correct. No API-health polling failures occur in the GPU runs.

The 512-token VLM matrix completes 28 checks with zero API failures and 24/24 correct
toy retrieval answers. It includes exact prompt+reply admission, both overflow
modes, four conversation turns, eight alternating independent requests, streaming,
Stop and disconnect. RSS rises approximately 3 MiB. During the long GPU probe,
the separate CPU/NPU model returns a correct short answer in 0.19 seconds.

## Verification and limits

The GPU 8,192 matrix completes eight selected checks: one first-text deadline
failure and seven admission/recovery checks pass. The interrupted 1,024-token probe
is recorded separately and is not counted as a completed response. Zero health
poll failures occur in the VLM, interrupted GPU and bounded GPU runs. Maximum
observed status latency is 54.9 ms in the VLM matrix and 51.1 ms in the bounded GPU
matrix. GPU worker RSS remains approximately 5,933.5 MiB; VLM RSS rises about 3 MiB
across its 42.5-second bounded matrix. This does not establish leak-free day-long
service or comprehensive language/vision accuracy.

The 8,192-token fixture originally oscillated by one token before submitting an
API request. Its padding was corrected and that fixture-only scratch output was
removed. Actual successful/failed API calls retain independently checked token
counts, input hashes, status and output evidence. The first VLM matrix used the
earlier fixture builder; the final script supports exact padding, label selection
and absolute streaming deadlines. Role events and keepalives never count as first
text.

Mypy checks 236 source files; whole-backend Ruff, new-harness Ruff and whitespace
checks pass. Product source matches the preceding sampling verification hashes;
this task adds a live test harness and reports without changing inference or API
implementation. The startup identity loss is reproduced but not repaired here.
No startup limit, clocks, drivers, global settings, archives, commits, pushes or
reference repositories are changed. No test inference remains running.

[Machine-readable evidence](rockchip-api-stability-benchmark.json) records the
raw reports, source/evidence hashes, individual outcomes and restored app state.
