# Reproduce paths — TeleChat2 3B, Qwen2 0.5B, MiniCPM4 0.5B, InternLM2 1.8B

Read-only forensic trace of the four models as measured in `~/oNPU`. Nothing was run on
the NPU, no model was loaded, no checkpoint was re-hashed end-to-end, no repo file was
edited except this one. Every value below is cited to a `file:line` or to a directory
listing.

**Headline:** the four published numbers share **one** reproduce path — a staged HTTP-API
probe driving `python -m onpu` against backend `~/tinygrad`, 3 warm-ups + 10 measured
requests, pooled median over 1–2 processes. They do **not** use `retime_with_profile.py`,
they do **not** use `extra.rockchip.benchmark_inference`, and they do **not** come from
`~/tg-decode-all`. The tree that `retime_with_profile.py` and `retime_model.sh` default to
is not the tree any of these four numbers came from.

---

## 0. How the numbers were located

The chain is three hops, and every hop is on disk:

```
benchmark.md selected table          (the published value)
  -> docs/benchmark-*.json           (the aggregation: summary + per-process report list)
    -> <stage>/results/<slug>.json   (the per-process raw report: trials, checks, placement)
      -> <stage>/probe.py            (the harness that produced it)
        -> <stage>/build_command.py  (the exact argv + env)
```

1. **Published value.** `~/oNPU/benchmark.md:929` opens the section
   `## Latest selected model timings — November 4, 2026`. The four rows are
   `benchmark.md:935` (Qwen2), `:936` (MiniCPM4), `:941` (InternLM2), `:944` (TeleChat2).
   Prefill proxies are at `:961`, `:962`, `:967`, `:970`.
2. **Aggregation.** `~/oNPU/docs/benchmark-model-status.json` `models[]` carries, per model,
   a `current_preferred` block with `slug`, `report`, `report_sha256`, `source_count`,
   `source_manifest`. That is the pointer to the dated artifact.
3. **Per-process raw report.** Each aggregation names one or two `results/*.json` files.
   Those files carry `trials[]` (the 10 measured requests with `metrics.ttft_ms` and
   `metrics.decode_tps`), `checks[]` (18–19 functional checks), `placement` (the recipe that
   was actually applied), `source_before`/`source_after` (the pinned source hashes), and
   `protocol`.
4. **Harness.** Each `results/` directory sits beside a `staged/<stage>/` directory holding
   `probe.py`, `build_command.py`, `manifest.json`, `frozen-pins.json`, `validate_report.py`
   and the prompt/boundary fixtures.

**Independent confirmation that the artifacts are the real ones.** I recomputed the pooled
median from the raw `trials[]` of every per-process report and it reproduces each published
number to the last digit:

| model     | pooled n | recomputed TTFT (ms) | published TTFT | recomputed decode (tok/s) | published decode |
| --------- | -------: | -------------------: | -------------: | ------------------------: | ---------------: |
| Qwen2     |       10 |             141.6205 |         141.62 |                   48.7783 |            48.78 |
| MiniCPM4  |       20 |             130.0670 |         130.07 |                   57.5578 |            57.56 |
| InternLM2 |       20 |             339.1908 |         339.19 |                   16.1582 |            16.16 |
| TeleChat2 |       20 |             536.6984 |         536.70 |                   10.4468 |            10.45 |

And every per-process report's sha256 matches the sha256 recorded in its aggregation
(`sha256sum` run against each path in §1–§4).

---

## 1. TeleChat2 3B

| field                 | value |
| --------------------- | ----- |
| **checkpoint path**   | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/TeleChat2-3B/TeleChat2-3B-f16.gguf` (6,144,189,088 B) |
| **checksum**          | `e7e7f746fcc2ee75217a722ce873c533b2c23d8db124af9b86fb945827eb4cbb` |
| **code tree**         | `/home/orangepi/tinygrad` — backend `R` in `staged/telechat2-hadamard-api-20261103-v1/probe.py:4` |
| **interpreter**       | probe + server: `/home/orangepi/oNPU/.venv/bin/python` (Python 3.12.12); worker: `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner**            | `staged/telechat2-hadamard-api-20261103-v1/probe.py`, twice (`ONPU_CASE=api-a`, `ONPU_CASE=api-b`) |
| **entry point**       | `python -m onpu` (FastAPI app) → `python -m onpu.worker`. **Not** `extra.rockchip.benchmark_inference` |
| **recipe**            | `rockchip_options.py:221-278`, `telechat2-3b-hadamard-context512` — full dict in §1.4 |
| **gates**             | quality screen + external-GPU preservation + 2 GiB RAM + checkpoint sha + 326 source pins + placement + single worker pid — §1.3 |
| **artifact**          | `…/telechat2-hadamard-api-20261103-v1/results/api-a.json` (sha `aadea2d7…`) and `api-b.json` (sha `323b132f…`) |
| **protocol**          | **3 warm-ups + 10 measured**, 128 prompt / 64 gen, ctx 512, 0.2 s fixed idle, `taskset -c 0-3` for the probe / `-c 0-7` for the server |

Published: TTFT **536.70 ms**, decode **10.45 tok/s**, prefill proxy **238.50 tok/s**
(`benchmark.md:944`, `:970`).

### 1.1 Provenance chain

- Checkpoint checksum: `~/oNPU/checkpoints/manifests/TeleChat2-3B.conversion.json` →
  `"sha256": "e7e7f746…"`, `"bytes": 6144189088`, `"complete": true`. The same file records
  `source_repo: Tele-AI/TeleChat2-3B`, `source_revision: b364de95…`, `dtype: FP16`,
  `architecture: telechat2`, `tensor_count: 267`.
- On disk: `ls -la …/checkpoints/TeleChat2-3B/` shows `TeleChat2-3B-f16.gguf` at exactly
  6,144,189,088 B plus `TeleChat2-3B-f16.conversion.json` (713 B).
- Registry constant: `~/oNPU/onpu/rockchip_options.py:8` —
  `TELECHAT2_CHECKPOINT = "e7e7f746…"`. Scales asset hash at `:9`.
- Aggregation: `~/oNPU/docs/benchmark-telechat-hadamard-20261103.json`, schema
  `onpu.telechat-hadamard-production.v1`, `summary.ttft_ms = 536.6984040301759`,
  `summary.decode_tokens_per_second = 10.446758610772138`,
  `summary.processes = 2`, `summary.measured_requests = 20`, `summary.serving_checks = 38`.
  `api_reports[]` lists the two per-process files with their sha256.
- Manifest entry: `staged/telechat2-hadamard-api-20261103-v1/manifest.json` `entries[0]` —
  `slug: api-a`, `checkpoint`, `checkpoint_sha256`, `profile:
  telechat2-3b-hadamard-context512`, `expected_tile_columns: 24384`, `embedding: false`,
  `decode_wait: true`, `event_wait: false`, `recurrent: false`, `prior_report: null`.

### 1.2 Exact invocation

From `staged/telechat2-hadamard-api-20261103-v1/build_command.py`:

```
taskset -c 0-3
  /home/orangepi/oNPU/.venv/bin/python
  …/staged/telechat2-hadamard-api-20261103-v1/probe.py
  …/telechat2-hadamard-api-20261103-v1/results/api-a.json          # <- argv[1], report path
```

with environment (verbatim from `build_command.py`):

```
ONPU_EVIDENCE_RUN=telechat-had-api-6bf9fcbbc8da
ONPU_CASE=api-a                       # api-b for the second process
ONPU_REPLAY_PROFILE=0
DEBUG=0  VIZ=0  PROFILE=0
PYTHONPATH=/home/orangepi/tinygrad:/home/orangepi/oNPU
PYTHONUNBUFFERED=1
PARALLEL=8  OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1
```

`probe.py` then launches the server itself (`probe.py:26`):

```
taskset -c 0-7
  /home/orangepi/oNPU/.venv/bin/python -m onpu
  --host 127.0.0.1 --port 8092
  --backend /home/orangepi/tinygrad
  --data-dir …/results/api-a/data
```

and drives it over HTTP: `POST /api/models/import`, `POST /api/models/<id>/load` with
`{"profile":"ROCKCHIP","context":512,"realize":true,"npu_prefill":"auto","npu_decode":"auto","vision_gpu":false}`
(`probe.py:35`), then `POST /v1/chat/completions` per trial (`probe.py:75`).

The worker is a **second** interpreter: `runtime.py:61-63` resolves
`self.backend / ".venv/bin/python"`, i.e. `/home/orangepi/tinygrad/.venv/bin/python`, and
`runtime.py:151-158` spawns `-m onpu.worker --model <gguf> --context 512 --realize 1
--speculation off` with `cwd=/home/orangepi/tinygrad`.

### 1.3 Gates enforced before starting

Ordered as the probe executes them:

| # | gate | where |
| -: | ---- | ----- |
| 1 | Independent quality screen must be `complete` **and** `limited_screen_pass`; reads `telechat2-hadamard-quality-20261103-v1/results/comparison.json` | `probe.py:15-16` |
| 2 | External original GPU on port 8091: exactly 1 session, `selected_profile == CL`, `state == ready` | `probe.py:17` |
| 3 | `guard()` re-checks that session every request: same `model_id`, same `options`, ready, no active request; `/api/settings` unchanged | `probe.py:18-19` |
| 4 | `MemAvailable >= 2 GiB` at every guard | `probe.py:20` |
| 5 | Checkpoint sha256 == manifest `checkpoint_sha256` | `probe.py:21` |
| 6 | Port 8092 must be bindable (no stale server) | `probe.py:22` |
| 7 | All 326 pinned source hashes must match before starting | `probe.py:23` |
| 8 | Server reports `CPU`, `CL`, `ROCKCHIP` profiles available | `probe.py:31` |
| 9 | `placement.scaled_layers == 169` and `placement.hadamard_quantization.block == 128` | `probe.py:43` |
| 10 | `placement.hadamard_quantization.down_bias_blocks == 24` | `probe.py:44` |
| 11 | `placement.scalar_decode_loop.nonstream_batch == 32` | `probe.py:45` |
| 12 | `placement.tile_columns == 24384` | `probe.py:46` |
| 13 | `decode_host_event_wait is True`, `host_event_wait is False`, `prefill_profile == telechat2-3b-hadamard-context512` | `probe.py:54` |
| 14 | `placement.scalar_decode_loop.ready` (native scalar loop initialized) | `probe.py:56` |
| 15 | Exactly **one** `onpu.worker` process for this checkpoint, affinity `[4,5,6,7]` | `probe.py:64` |
| 16 | Per request: metrics id changed, `prompt_tokens == len(input_ids) == 128`, `completion_tokens == usage`, `npu_submissions > 0`, `cached_tokens == 0` when uncached | `probe.py:76` |
| 17 | All 10 measured responses byte-identical to the first warm-up | `probe.py:87` |
| 18 | Cached single token == uncached single token | `probe.py:92` |
| 19 | All 326 source hashes unchanged after the run | `probe.py:111` |
| 20 | Target unloaded, child terminated, external GPU still preserved | `probe.py:115-121` |

Post-hoc acceptance: `staged/telechat2-hadamard-api-20261103-v1/validate_report.py` requires
`complete`, `source_before == source_after`, 10 trials, `prior_response_exact`,
`cached_single_token_exact`, `target_unloaded`, `external_gpu_preserved`,
`scaled_layers == 169`, `hadamard block == 128`, `down_bias_blocks == 24`, non-empty
`prefill_captures` plus a `decode` capture, and per-capture
`activation_quantizers > 0`, `fused_ffn_calls == 24`, `weight_preparation_calls == 0`,
`fallback_reason is None`.

### 1.4 The recipe, in full

Branch: `~/oNPU/onpu/rockchip_options.py:221-278`, guarded by
`context == 512 and profile == "ROCKCHIP" and prefill in ("auto","on") and decode == "on"`.
Verified by evaluating the registry (pure-Python module, no NPU, no model):

```python
{'prefill': 'on',
 'mapped_cpu': True,
 'share_projection_weights': True,
 'prefill_chunk': 128,
 'tile_columns': 24384,
 'dense_swiglu': True,
 'scalar_decode_loop': True,
 'uncached_prefill': True,
 'trim_preparation_cache': True,
 'prefill_profile': 'telechat2-3b-hadamard-context512',
 'input_scales_asset': 'telechat2-full-cpu-20261103.json.gz',
 'input_scales_asset_sha256': 'b502b990e1a2baf2e10a96ca82f5cd4c7b6800b9d88ac13682b105975860408b',
 'input_scales_layers': 169,
 'hadamard_block': 128,
 'down_bias_blocks': 24,
 'memory_reserve': 2147483648,
 'environment': {
   'ONPU_NPU_DOMAINS': '3',
   'ONPU_NPU_DOMAIN_BYTES': '3221225472',
   'ONPU_INT8_SDK_PREFILL': '0',
   'ONPU_HOST_POOL_WFE': '0',
   'ONPU_DECODE_HOST_WFE': '1',
   'ONPU_SWIGLU_PACK_STATIC_LAYOUT': '1',
   'ONPU_INT8_EXPERT_DEPTHS': '0',
   'ONPU_INT8_SMALL_DEPTH': '0',
   'ONPU_CPU_GQA_PREFILL': '0',
   'ONPU_NORM_QUANT': '0',
   'ONPU_EXP2_SCALE_BITS': '1',
   'ONPU_ASYNC_IRQ_WAKE': '1',
   'ONPU_STATIC_PREFILL': '1',
   'ONPU_REPLAY_SCALAR_COPY': '1',
   'ONPU_REPLAY_DEDUP_GUARDS': '1',
   'ONPU_SHARED_HOST_POOL': '1',
   'ONPU_CPU_TOKEN_INPUTS': '1',
   'ONPU_ZERO_PAD_VIEWS': '1',
   'ONPU_REPLAY_BORROW_OUTPUTS': '1',
   'ONPU_REPLAY_BORROW_BATCH': '1',
   'ONPU_REPLAY_NATIVE_FROM': '1',
   'ONPU_REPLAY_NATIVE_FROM_LARGE': '1',
   'ONPU_CPU_DECODE_POOL': '1',
   'ONPU_PREFILL_CPU_TUNING': '1',
   'ONPU_CPU_PREFILL_LAYOUT': '1',
   'ONPU_EXP2_FAST_PATH': '1',
   'ONPU_FFN_PACKED_UPLOAD': '1',
   'ONPU_INT8_DECODE_NARROW_CBUF': '1',
   'ONPU_SWIGLU_CHANNEL_SCRATCH': '0',
   'ONPU_SWIGLU_C4_PREFETCH': '0',
   'ONPU_SWIGLU_BATCHED_PACK': '0',
   'ONPU_SWIGLU_K_SPLIT': '0',
   'ONPU_SWIGLU_K_SPLIT_TAIL_SDK': '0',
   'ONPU_REPLAY_EXTRA_PACK_ALIAS': '0',
   'ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN': '1',
   'ONPU_SHORTCONV_STATIC_PREFILL': '0',
   'ONPU_INT8_HADAMARD': '128',
   'ONPU_INT8_NATURAL2304': '1',
   'ONPU_SWIGLU_PREFILL_ROWS': '4'}}
```

39 environment variables. Cross-checked against the recorded `placement` in `api-a.json`:
`scaled_layers 169` ✓, `tile_columns 24384` ✓, `share_projection_weights true` ✓,
`decode_host_event_wait true` ✓, `host_event_wait false` ✓, `uncached_prefill true` ✓,
`scalar_decode_loop` present ✓, `cpu_preparation_cache_policy
"disabled-during-configuration"` (= `trim_preparation_cache`) ✓, `hadamard_quantization`
block 128 / down_bias_blocks 24 ✓, no `cpu_q8_embedding` ✓ (`embedding: false`).

### 1.5 Protocol

`probe.py:79-83` and `probe.py:80`:

```python
raw['protocol'] = {'warmups': 3, 'measured': 10, 'fixed_idle_seconds': 0.2,
                   'report_serialization': 'buffered until requests complete',
                   'stock_app': True}
```

- 3 warm-ups (`warmup-0..2`), then 10 measured (`measured-0..9`), each `max_tokens=64`,
  `temperature=0`, `speculation="off"`, `cache_prompt=False`, `stream=False`.
- 128-token prompt from `staged/telechat2-hadamard-api-20261103-v1/prompt.json`
  (`input_tokens: 128`, `template_sha256: fdffcd15…`).
- Context 512. `taskset -c 0-3` for the probe, `-c 0-7` for the server, worker pinned to
  `[4,5,6,7]`.
- After the 10 measured: `cached-followup` (1 token, cache on), `uncached-single-token`,
  streaming 16 tokens, `context-boundary` (448+64), streaming 64 with cancel,
  `cancel-recovery` (2 tokens). Those are the extra `checks[]` entries — 19 per process,
  38 total.
- Clocks recorded per trial from `/sys/class/devfreq/fdab0000.npu/cur_freq`,
  `/sys/class/devfreq/dmc/cur_freq`, `policy4`, `policy6` (`probe.py:73`), plus
  `/sys/class/thermal/thermal_zone*/temp` (`probe.py:72`). **No clock gate** — the scope
  string says "stock adaptive governors, no sudo tuning".

---

## 2. Qwen2 0.5B

| field                 | value |
| --------------------- | ----- |
| **checkpoint path**   | `/home/orangepi/vlm-bench/baseline-20261021/Qwen2-0.5B-Instruct-Q8_0.gguf` (531,065,536 B) — **top level, not under `checkpoints/`** |
| **checksum**          | `a07bd33b0e52babc1083e97d9bb6018197d7af25697552cc4d0b751e035158ff` |
| **code tree**         | `/home/orangepi/tinygrad` — `staged/qwen-current-api-20261102-v2/probe.py:4` |
| **interpreter**       | probe + server: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner**            | `staged/qwen-current-api-20261102-v2/probe.py`, once (`ONPU_CASE=qwen2`) |
| **entry point**       | `python -m onpu` → `python -m onpu.worker`. **Not** `benchmark_inference` |
| **recipe**            | `rockchip_options.py:403` + `:413-492` dense branch, `qwen2-0.5b-context512` — full dict in §2.4 |
| **gates**             | backend numeric gate + external GPU + 2 GiB + checkpoint sha + 321 source pins + K-split/c4/batched-pack placement + single worker pid — §2.3 |
| **artifact**          | `…/qwen-current-api-20261102-v2/results/qwen2.json` (sha `18f64538…`) |
| **protocol**          | **3 warm-ups + 10 measured**, 128 / 64, ctx 512, 0.2 s fixed idle, single process |

Published: TTFT **141.62 ms**, decode **48.78 tok/s**, prefill proxy **903.82 tok/s**
(`benchmark.md:935`, `:961`).

### 2.1 Provenance chain

- Checkpoint checksum: `~/vlm-bench/baseline-20261021/Qwen2-0.5B-Instruct-Q8_0.download.json`
  → `"sha256": "a07bd33b…"`, `"expected_bytes": 531065536`, `"complete": true`,
  `gguf_repo: bartowski/Qwen2-0.5B-Instruct-GGUF`, `revision: a641934b…`.
- On disk: `ls -la …/baseline-20261021/` shows `Qwen2-0.5B-Instruct-Q8_0.gguf` at
  531,065,536 B. There is **no** `checkpoints/Qwen2-0.5B/` directory.
- Registry constant: `~/oNPU/onpu/rockchip_options.py:19` — `QWEN2_CHECKPOINT = "a07bd33b…"`.
- Aggregation: `~/oNPU/docs/benchmark-qwen-current-api-20261102.json`, schema
  `tinygrad.qwen-current-api-results.v1`. `rows[]` entry `model: "Qwen2/0.5B"` carries
  `summary.ttft_ms = 141.62048100843094`,
  `summary.decode_tokens_per_second = 48.7783241657931`,
  `summary.measured_requests = 10`, `summary.processes = 1`,
  `summary.serving_checks = 18`, `report_path`, `report_sha256 = 18f64538…`.
- Manifest entry: `staged/qwen-current-api-20261102-v2/manifest.json` `entries[0]` —
  `slug: qwen2`, `profile: qwen2-0.5b-context512`, `expected_tile_columns: 24576`,
  `embedding: true`, `scalar: true`, `decode_wait: true`, `event_wait: false`,
  `norm_quantization: true`, `prefetch_channels: 16`, and
  `prior_report: …/qwen2-lazy-temperature-api-v1/results/qwen2-api.json`.

### 2.2 Exact invocation

From `staged/qwen-current-api-20261102-v2/build_command.py`:

```
taskset -c 0-3
  /home/orangepi/oNPU/.venv/bin/python
  …/staged/qwen-current-api-20261102-v2/probe.py
  …/qwen-current-api-20261102-v2/results/qwen2.json
```

environment:

```
ONPU_EVIDENCE_RUN=qwen-current-api-20261102-v2
ONPU_CASE=qwen2
PYTHONPATH=/home/orangepi/tinygrad:/home/orangepi/oNPU
PYTHONUNBUFFERED=1
PARALLEL=8  OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1
```

`build_command.py` additionally refuses to build the command unless the backend numeric gate
passes — see §2.3 gate 1.

`probe.py:24` launches the server identically to §1.2 (`-m onpu --port 8092 --backend
/home/orangepi/tinygrad`).

### 2.3 Gates

| # | gate | where |
| -: | ---- | ----- |
| 1 | Backend numeric gate: `qwen2-ksplit-production-backend-20261101-v1/results/backend.json` must be `complete: true`, `teacher_sha256 == 159bfc9174f7ef6d211b7f0a49683d2cdfeff8ea2020dc4fad58ea186edafe39`, `prefetch_native_calls == 24` | `build_command.py:6` |
| 2 | All 321 source hashes match | `build_command.py:8` |
| 3 | External GPU on 8091: 1 session, CL, ready | `probe.py:15` |
| 4 | `guard()`: session and settings unchanged; `MemAvailable >= 2 GiB` | `probe.py:16-18` |
| 5 | Checkpoint sha256 == manifest | `probe.py:19` |
| 6 | Port 8092 bindable | `probe.py:20` |
| 7 | 321 source pins match before start | `probe.py:21` |
| 8 | CPU / CL / ROCKCHIP profiles available | `probe.py:29` |
| 9 | `placement.tile_columns == 24576` | `probe.py:41` |
| 10 | `placement.prefill_features.c4_prefetch_channels == 16` | `probe.py:43` |
| 11 | `placement.prefill_features.ksplit_prefill` and `ksplit_prefill.prepared_blocks == 24` and `traces > 0` | `probe.py:44` |
| 12 | `ksplit_prefill.native_execution.k2048_calls == 48` and `native_runs > 0` | `probe.py:45` |
| 13 | `placement.prefill_features.batched_pack` and `batched_pack.traces > 0` | `probe.py:46` |
| 14 | `decode_host_event_wait is True`, `host_event_wait is False`, `prefill_profile == qwen2-0.5b-context512`, `cpu_q8_embedding.fast_traces > 0` | `probe.py:49` |
| 15 | `placement.scalar_decode_loop.ready` | `probe.py:51` |
| 16 | Exactly one `onpu.worker`, affinity `[4,5,6,7]` | `probe.py:59` |
| 17 | Per request: id changed, `prompt_tokens == 128`, `completion_tokens == usage`, `npu_submissions > 0`, `cached_tokens == 0` | `probe.py:71-72` |
| 18 | All 10 measured responses identical to the prior report's first response (`qwen2-lazy-temperature-api-v1/results/qwen2-api.json`) | `probe.py:79` |
| 19 | Cached follow-up reuses tokens (`cached_tokens > 0`) | `probe.py:81-82` |
| 20 | 321 source pins unchanged after run | `probe.py:100` |
| 21 | Unload + external GPU preserved | `probe.py:103-110` |

Post-hoc: `validate_report.py` requires `complete`, `external_gpu_preserved`,
`target_unloaded`, `source_before == source_after`, affinity `[4,5,6,7]`, 10 trials,
18 checks, `prior_response_exact`, and per-trial 128/64/0-cached/`npu_submissions > 0`.

`preflight.json` records `no_device_opened: true`,
`prior_failure: "v1 universal tile assertion failed Qwen2 before any timed request;
source/runtime unchanged"` — i.e. v1 was discarded before timing and v2 is the run.

### 2.4 The recipe, in full

Branch: `rockchip_options.py:403` (`QWEN2_CHECKPOINT: ("qwen2-0.5b-context512", 4, 24576)`)
inside the dense block `:413-492`, with the Qwen2-specific updates at `:424-425`,
`:435-440`, `:479-483`, `:488-491`.

```python
{'prefill': 'on',
 'mapped_cpu': True,
 'share_projection_weights': False,
 'prefill_chunk': 128,
 'tile_columns': 24576,
 'dense_swiglu': True,
 'prefill_profile': 'qwen2-0.5b-context512',
 'environment': {
   'ONPU_SWIGLU_PREFILL_ROWS': '4',
   'ONPU_INT8_EXPERT_DEPTHS': '1',
   'ONPU_INT8_NATURAL2304': '1',
   'ONPU_INT8_DECODE_NARROW_CBUF': '1',
   'ONPU_NPU_DOMAINS': '1',
   'ONPU_NPU_DOMAIN_BYTES': '3221225472',
   'ONPU_CPU_TOKEN_INPUTS': '1',
   'ONPU_ZERO_PAD_VIEWS': '1',
   'ONPU_REPLAY_BORROW_OUTPUTS': '1',
   'ONPU_REPLAY_BORROW_BATCH': '1',
   'ONPU_REPLAY_NATIVE_FROM': '1',
   'ONPU_SHARED_HOST_POOL': '1',
   'ONPU_PREFILL_CPU_TUNING': '1',
   'ONPU_CPU_PREFILL_LAYOUT': '1',
   'ONPU_REPLAY_NATIVE_FROM_LARGE': '1',
   'ONPU_CPU_DECODE_POOL': '1',
   'ONPU_EXP2_FAST_PATH': '1',
   'ONPU_SWIGLU_PACK_STATIC_LAYOUT': '1',
   'ONPU_SWIGLU_CHANNEL_SCRATCH': '1',
   'ONPU_INT8_SDK_PREFILL': '1',
   'ONPU_FFN_PACKED_UPLOAD': '1',
   'ONPU_CPU_GQA_PREFILL': '1',
   'ONPU_ASYNC_IRQ_WAKE': '1',
   'ONPU_STATIC_PREFILL': '1',
   'ONPU_NORM_QUANT': '1',
   'ONPU_SWIGLU_C4_PREFETCH': '16',
   'ONPU_SWIGLU_BATCHED_PACK': '1',
   'ONPU_SWIGLU_K_SPLIT': '1',
   'ONPU_HOST_POOL_WFE': '0',
   'ONPU_DECODE_HOST_WFE': '1'},
 'scalar_decode_loop': True,
 'uncached_prefill': True,
 'cpu_q8_embedding': True}
```

30 environment variables. Cross-checked against `qwen2.json` `placement`:
`share_projection_weights false` ✓, `tile_columns 24576` ✓, `norm_quantization` present ✓,
`batched_pack` present ✓, `ksplit_prefill.prepared_blocks 24` ✓, `async_irq_wake true` ✓,
`uncached_prefill true` ✓, `decode_host_event_wait true` ✓, `host_event_wait false` ✓,
`cpu_q8_embedding` present ✓.

This is the **only** one of the four whose recipe sets `ONPU_NORM_QUANT=1`,
`ONPU_SWIGLU_K_SPLIT=1`, `ONPU_SWIGLU_BATCHED_PACK=1`, `ONPU_SWIGLU_C4_PREFETCH=16` and
`ONPU_INT8_EXPERT_DEPTHS=1`.

### 2.5 Protocol

`probe.py:74-78`:

```python
raw['protocol'] = {'warmups': 3, 'measured': 10, 'fixed_idle_seconds': 0.2,
                   'report_serialization': 'buffered until requests complete',
                   'stock_app': True}
```

- 3 warm-ups + 10 measured, `max_tokens=64`, `temperature=0`, `speculation=off`,
  `cache_prompt=False`, `stream=False`, 0.2 s sleep before each measured request.
- 128-token prompt from `staged/qwen-current-api-20261102-v2/qwen2-prompt.json`
  (`input_ids` length 128, `worker_prepare_prompt_verified: true`).
- Context 512. Single process. 18 checks per process.
- Clocks recorded per trial; **no clock gate** (stock adaptive governors).

---

## 3. MiniCPM4 0.5B

| field                 | value |
| --------------------- | ----- |
| **checkpoint path**   | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/MiniCPM4-0.5B/MiniCPM4-0.5B.Q8_0.gguf` (462,829,184 B) |
| **checksum**          | `d69653113b3559808359cdac50644c54f7aa18bd31204c69310dc8a22725548f` |
| **code tree**         | `/home/orangepi/tinygrad` — `staged/minicpm4-current-api-20261101-v1/probe.py:4` |
| **interpreter**       | probe + server: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner**            | `staged/minicpm4-current-api-20261101-v1/probe.py`, twice (`ONPU_CASE=control-a`, `ONPU_CASE=control-b`) |
| **entry point**       | `python -m onpu` → `python -m onpu.worker`. **Not** `benchmark_inference` |
| **recipe**            | `rockchip_options.py:405` + `:413-492` dense branch, `minicpm4-0.5b-context512` — full dict in §3.4 |
| **gates**             | backend gate (source equality + teacher sha) + external GPU + 2 GiB + checkpoint sha + 313 source pins + `PARALLEL=8` in worker environ + placement + single worker pid — §3.3 |
| **artifact**          | `…/minicpm4-current-api-20261101-v1/results/control-a.json` (sha `eda181fe…`) and `control-b.json` (sha `2ff3318c…`) |
| **protocol**          | **3 warm-ups + 10 measured**, 128 / 64, ctx 512, 0.2 s fixed idle, two processes |

Published: TTFT **130.07 ms**, decode **57.56 tok/s**, prefill proxy **984.11 tok/s**
(`benchmark.md:936`, `:962`).

### 3.1 Provenance chain

- Checkpoint checksum: `…/checkpoints/MiniCPM4-0.5B/download.json` →
  `"sha256": "d6965311…"`, `"expected_bytes": 462829184`, `"complete": true`,
  `repo: mradermacher/MiniCPM4-0.5B-GGUF`, `revision: 232b2c20…`.
- On disk: `ls -la …/checkpoints/MiniCPM4-0.5B/` shows `MiniCPM4-0.5B.Q8_0.gguf` at
  462,829,184 B plus `download.json`.
- Registry constant: `~/oNPU/onpu/rockchip_options.py:13` — `MINICPM4_CHECKPOINT = "d6965311…"`.
- Aggregation: `~/oNPU/docs/benchmark-minicpm4-current-20261101.json`, schema
  `minicpm4-current-20261101-v1`. `api.processes[]` has two entries with
  `report`, `sha256`, `summary`, `timed_requests: 10`, `functional_checks: 18`,
  `prior_reply_exact: true`, `source_before_after_exact: true`, and per-process `ranges`.
  `control-a` TTFT 127.5225, `control-b` TTFT 132.8214 → pooled 130.0670.
- Manifest entry: `staged/minicpm4-current-api-20261101-v1/manifest.json` `entries[0]` —
  `slug: control-a`, `profile: minicpm4-0.5b-context512`, `embedding: true`, `scalar: true`,
  `decode_wait: true`, `event_wait: false`, `compiler_workers: 8`,
  `backend_stage: minicpm4-sdk-production-v1`, `control_stage: minicpm4-sdk-model-v1`,
  `prior_report: …/minicpm4-sdk-api-v1/results/mini-api.json`,
  `backend_gate: …/minicpm4-current-backend-20261101-v1/results/control.json`.
  Note this manifest has **no** `expected_tile_columns` — the probe does not assert
  `tile_columns` for this model.

### 3.2 Exact invocation

From `staged/minicpm4-current-api-20261101-v1/build_command.py`:

```
taskset -c 0-3
  /home/orangepi/oNPU/.venv/bin/python
  …/staged/minicpm4-current-api-20261101-v1/probe.py
  …/minicpm4-current-api-20261101-v1/results/control-a.json
```

environment:

```
ONPU_EVIDENCE_RUN=minicpm4-current-api-20261101-v1
ONPU_CASE=control-a                  # control-b for the second process
PYTHONPATH=/home/orangepi/tinygrad:/home/orangepi/oNPU
PARALLEL=8  PYTHONUNBUFFERED=1
OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1
```

`probe.py:23` launches the server identically to §1.2. This is the only one of the four
whose probe additionally asserts `PARALLEL=8` is present in the **worker's** `/proc/<pid>/environ`
(`probe.py:52`) and records `compiler_workers_at_exec: 8`.

### 3.3 Gates

| # | gate | where |
| -: | ---- | ----- |
| 1 | Backend gate: `minicpm4-current-backend-20261101-v1/results/control.json` must be `complete: true`, its `source_sha256 == source_sha256_after` and equal to the manifest's tinygrad-relative hashes, and `teacher_sha256 == 0fd363d419ebdc0687a4da11e1d59eb9e5369481e5207d5ea80519b8b92e8847` | `build_command.py:6-7` |
| 2 | All 313 source hashes match | `build_command.py:8` |
| 3 | External GPU on 8091: 1 session, CL, ready | `probe.py:15` |
| 4 | `guard()`: unchanged session/settings; `MemAvailable >= 2 GiB` | `probe.py:16-18` |
| 5 | Checkpoint sha256 == manifest | `probe.py:19` |
| 6 | 313 source pins match before start | `probe.py:20` |
| 7 | CPU / CL / ROCKCHIP profiles available | `probe.py:28` |
| 8 | `decode_host_event_wait is True`, `host_event_wait is False`, `prefill_profile == minicpm4-0.5b-context512`, `cpu_q8_embedding.fast_traces > 0` | `probe.py:42` |
| 9 | `placement.scalar_decode_loop.ready` | `probe.py:44` |
| 10 | Exactly one `onpu.worker`, affinity `[4,5,6,7]`, and `b'PARALLEL=8'` in its environ | `probe.py:52` |
| 11 | Per request: id changed, `prompt_tokens == 128`, `completion_tokens == usage`, `npu_submissions > 0`, `cached_tokens == 0` | `probe.py:64-65` |
| 12 | All 10 measured responses identical to the prior report's first response (`minicpm4-sdk-api-v1/results/mini-api.json`) | `probe.py:72` |
| 13 | Cached follow-up reuses tokens | `probe.py:74-75` |
| 14 | 313 source pins unchanged after run | `probe.py:93` |
| 15 | Unload + external GPU preserved | `probe.py:96-103` |

**No `tile_columns` assertion** and **no port-8092 pre-bind check** in this probe (unlike
Qwen2 and TeleChat2). Post-hoc `validate_report.py` requires `complete`,
`external_gpu_preserved`, `target_unloaded`, `source_before == source_after`, affinity,
10 trials, 18 checks, `prior_response_exact`, and per-trial 128/64/0-cached/submits.

### 3.4 The recipe, in full

Branch: `rockchip_options.py:405`
(`MINICPM4_CHECKPOINT: ("minicpm4-0.5b-context512", 4, 24576)`), dense block `:413-492`,
with `:426-429` (shared with TinyLlama), `:430-432`, `:479-483`, `:486-487`, `:488-489`,
`:490-491`.

```python
{'prefill': 'on',
 'mapped_cpu': True,
 'share_projection_weights': False,
 'prefill_chunk': 128,
 'tile_columns': 24576,
 'dense_swiglu': True,
 'prefill_profile': 'minicpm4-0.5b-context512',
 'environment': {
   'ONPU_SWIGLU_PREFILL_ROWS': '4',
   'ONPU_INT8_EXPERT_DEPTHS': '0',
   'ONPU_INT8_NATURAL2304': '1',
   'ONPU_INT8_DECODE_NARROW_CBUF': '1',
   'ONPU_NPU_DOMAINS': '1',
   'ONPU_NPU_DOMAIN_BYTES': '3221225472',
   'ONPU_CPU_TOKEN_INPUTS': '1',
   'ONPU_ZERO_PAD_VIEWS': '1',
   'ONPU_REPLAY_BORROW_OUTPUTS': '1',
   'ONPU_REPLAY_BORROW_BATCH': '1',
   'ONPU_REPLAY_NATIVE_FROM': '1',
   'ONPU_SHARED_HOST_POOL': '1',
   'ONPU_PREFILL_CPU_TUNING': '1',
   'ONPU_CPU_PREFILL_LAYOUT': '1',
   'ONPU_REPLAY_NATIVE_FROM_LARGE': '1',
   'ONPU_CPU_DECODE_POOL': '1',
   'ONPU_EXP2_FAST_PATH': '1',
   'ONPU_EXP2_SCALE_BITS': '1',
   'ONPU_FFN_PACKED_UPLOAD': '1',
   'ONPU_ASYNC_IRQ_WAKE': '1',
   'ONPU_STATIC_PREFILL': '1',
   'ONPU_CPU_GQA_PREFILL': '1',
   'ONPU_INT8_SDK_PREFILL': '1',
   'ONPU_HOST_POOL_WFE': '0',
   'ONPU_DECODE_HOST_WFE': '1'},
 'scalar_decode_loop': True,
 'sdk_gateup': (128, 8192, 1024),
 'uncached_prefill': True,
 'cpu_q8_embedding': True}
```

25 environment variables. Cross-checked against `control-a.json` `placement`:
`share_projection_weights false` ✓, `tile_columns 24576` ✓, `async_irq_wake true` ✓,
`uncached_prefill true` ✓, `decode_host_event_wait true` ✓, `host_event_wait false` ✓,
`cpu_q8_embedding` present ✓, and **no** `norm_quantization` / `batched_pack` /
`ksplit_prefill` keys ✓ (this branch does not set them).

### 3.5 Protocol

`probe.py:67-71`:

```python
raw['protocol'] = {'warmups': 3, 'measured': 10, 'fixed_idle_seconds': 0.2,
                   'report_serialization': 'buffered until requests complete',
                   'stock_app': True}
```

- 3 warm-ups + 10 measured, `max_tokens=64`, temperature 0, speculation off, cache off,
  stream off, 0.2 s sleep before each measured request.
- 128-token prompt from `staged/minicpm4-current-api-20261101-v1/mini-128.json`
  (`input_ids` length 128, `worker_prepare_prompt_verified: true`); boundary fixture is
  `mini-448.json`.
- Context 512, two sequential processes, 18 checks each = 36 total.
- Clocks recorded per trial; no clock gate.

---

## 4. InternLM2 1.8B

| field                 | value |
| --------------------- | ----- |
| **checkpoint path**   | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/InternLM2-1.8B/internlm2-chat-1_8b.Q8_0.gguf` (2,009,613,632 B) |
| **checksum**          | `97e3d100a75859d751809e543efa50b2d5db1a9ae27df0469f8ee30a1bf131db` |
| **code tree**         | `/home/orangepi/tinygrad` — `staged/internlm2-generic-api-v2/probe.py:4` |
| **interpreter**       | probe + server: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner**            | `staged/internlm2-generic-api-v1/probe.py` **and** `staged/internlm2-generic-api-v2/probe.py` — byte-identical, sha256 `114c7ba9…`; one run each, `ONPU_CASE=internlm2-api` |
| **entry point**       | `python -m onpu` → `python -m onpu.worker`. **Not** `benchmark_inference` |
| **recipe**            | `rockchip_options.py:406` + `:413-492` dense branch, `internlm2-1.8b-context512` — full dict in §4.4 |
| **gates**             | backend/control teacher-sha gate + external GPU + 2 GiB + checkpoint sha + 311 source pins + share/pool/async/uncached placement + single worker pid — §4.3 |
| **artifact**          | `…/internlm2-generic-api-v1/results/internlm2-api.json` (sha `5693907d…`) **and** `…/internlm2-generic-api-v2/results/internlm2-api.json` (sha `d6a11154…`) |
| **protocol**          | **3 warm-ups + 10 measured** per process, 128 / 64, ctx 512, two processes; **no** 0.2 s idle, **no** `protocol` dict written |

Published: TTFT **339.19 ms**, decode **16.16 tok/s**, prefill proxy **377.37 tok/s**
(`benchmark.md:941`, `:967`).

### 4.1 Provenance chain

- Checkpoint checksum: `…/checkpoints/InternLM2-1.8B/download.json` →
  `"sha256": "97e3d100…"`, `"expected_bytes": 2009613632`, `"complete": true`,
  `repo: mradermacher/internlm2-chat-1_8b-GGUF`, `revision: 32bfd162…`.
- On disk: `ls -la …/checkpoints/InternLM2-1.8B/` shows `internlm2-chat-1_8b.Q8_0.gguf` at
  2,009,613,632 B plus `download.json`.
- Registry constant: `~/oNPU/onpu/rockchip_options.py:15` — `INTERNLM2_CHECKPOINT = "97e3d100…"`.
- Aggregation: `~/oNPU/docs/benchmark-internlm2-generic-runtime-20261031.json`, schema
  `internlm2-generic-runtime-api-v1`. Top-level `ttft_ms = 339.1908350022277`,
  `prefill_proxy_tps = 377.36868686077366`, `decode_tps = 16.158242737223837`.
  `cohorts[]` lists the two per-process reports with sha256 and per-process summaries
  (v1: 334.6246 / 16.5087; v2: 345.8815 / 15.8541). `workload` records
  `context 512, input_tokens 128, actual_output_tokens 64, cache_prompt false,
  temperature 0, processes 2, warmups_per_process 3, measured_per_process 10`.
- Manifest entry: `staged/internlm2-generic-api-v2/manifest.json` `entries[0]` —
  `slug: internlm2-api`, `profile: internlm2-1.8b-context512`, `embedding: true`,
  `scalar: true`, `decode_wait: false`, `event_wait: false`, `norm_quantization: false`,
  `backend_stage: internlm2-generic-runtime-v1`, `control_stage: internlm2-generic-runtime-v1`.
- **Two cohorts, not one.** The published 339.19 is the pooled median of 20 trials drawn
  from two separate runs on two separate source-pin sets. Reproducing it needs **both**
  `internlm2-generic-api-v1` and `internlm2-generic-api-v2`.

### 4.2 Exact invocation

From `staged/internlm2-generic-api-v2/build_command.py` (v1 is byte-identical,
sha256 `26d108f3…`):

```
taskset -c 0-3
  /home/orangepi/oNPU/.venv/bin/python
  …/staged/internlm2-generic-api-v2/probe.py
  …/internlm2-generic-api-v2/results/internlm2-api.json
```

environment:

```
ONPU_EVIDENCE_RUN=internlm2-generic-api-v2
ONPU_CASE=internlm2-api
PYTHONPATH=/home/orangepi/tinygrad:/home/orangepi/oNPU
PYTHONUNBUFFERED=1
OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1
```

(no `PARALLEL` set for this one). `probe.py:22` launches the server identically to §1.2.

### 4.3 Gates

| # | gate | where |
| -: | ---- | ----- |
| 1 | Backend/control gate: reads `<backend_stage>/results/<backend_slug>.json` and `<control_stage>/results/<control_slug>.json`; requires `raw['complete']`, `raw['teacher_sha256'] == control['teacher_sha256']`, and `raw['scalar_decode_loop']['tokens'] > 0` | `build_command.py:6` |
| 2 | All 311 source hashes match | `build_command.py:7` |
| 3 | External GPU on 8091: 1 session, CL, ready | `probe.py:14` |
| 4 | `guard()`: unchanged session/settings; `MemAvailable >= 2 GiB` | `probe.py:15-17` |
| 5 | Checkpoint sha256 == manifest | `probe.py:18` |
| 6 | 311 source pins match before start | `probe.py:19` |
| 7 | CPU / CL / ROCKCHIP profiles available | `probe.py:27` |
| 8 | `placement.share_projection_weights` **and** `prepared_qkv_blocks == 24` **and** `shared_host_pool` | `probe.py:38` |
| 9 | `placement.async_irq_wake` **and** `placement.uncached_prefill` | `probe.py:38` |
| 10 | `decode_host_event_wait is False`, `host_event_wait is False`, `prefill_profile == internlm2-1.8b-context512`, `cpu_q8_embedding.fast_traces > 0` | `probe.py:38` |
| 11 | `placement.scalar_decode_loop.ready` | `probe.py:40` |
| 12 | Exactly one `onpu.worker`, affinity `[4,5,6,7]` | `probe.py:48` |
| 13 | Per request: id changed, `prompt_tokens == 128`, `completion_tokens == usage`, `npu_submissions > 0`, `cached_tokens == 0` | `probe.py:60-61` |
| 14 | Cached follow-up reuses tokens | `probe.py:67-68` |
| 15 | 311 source pins unchanged after run | `probe.py:86` |
| 16 | Unload + external GPU preserved | `probe.py:90-96` |

**No prior-response gate** for InternLM2 (no `prior_report` in the manifest, no
`prior_response_exact` assertion in the probe). Post-hoc `validate_report.py` requires the
same as Qwen2/MiniCPM4 **minus** `prior_response_exact`.

### 4.4 The recipe, in full

Branch: `rockchip_options.py:406`
(`INTERNLM2_CHECKPOINT: ("internlm2-1.8b-context512", 4, 24576)`), dense block `:413-492`,
with the InternLM2-specific block at `:448-454`. Note InternLM2 is **not** in the
`:479` list, so `ONPU_DECODE_HOST_WFE` stays `"0"` — that is why
`decode_host_event_wait` is `false` here and `true` for the other three.

```python
{'prefill': 'on',
 'mapped_cpu': True,
 'share_projection_weights': True,
 'prefill_chunk': 128,
 'tile_columns': 24576,
 'dense_swiglu': True,
 'prefill_profile': 'internlm2-1.8b-context512',
 'environment': {
   'ONPU_SWIGLU_PREFILL_ROWS': '4',
   'ONPU_INT8_EXPERT_DEPTHS': '0',
   'ONPU_INT8_NATURAL2304': '1',
   'ONPU_INT8_DECODE_NARROW_CBUF': '1',
   'ONPU_NPU_DOMAINS': '1',
   'ONPU_NPU_DOMAIN_BYTES': '3221225472',
   'ONPU_CPU_TOKEN_INPUTS': '1',
   'ONPU_ZERO_PAD_VIEWS': '1',
   'ONPU_REPLAY_BORROW_OUTPUTS': '1',
   'ONPU_REPLAY_BORROW_BATCH': '1',
   'ONPU_REPLAY_NATIVE_FROM': '1',
   'ONPU_SHARED_HOST_POOL': '1',
   'ONPU_PREFILL_CPU_TUNING': '1',
   'ONPU_CPU_PREFILL_LAYOUT': '1',
   'ONPU_REPLAY_NATIVE_FROM_LARGE': '1',
   'ONPU_CPU_DECODE_POOL': '1',
   'ONPU_EXP2_FAST_PATH': '1',
   'ONPU_EXP2_SCALE_BITS': '1',
   'ONPU_FFN_PACKED_UPLOAD': '1',
   'ONPU_ASYNC_IRQ_WAKE': '1',
   'ONPU_STATIC_PREFILL': '1',
   'ONPU_SWIGLU_PACK_STATIC_LAYOUT': '1',
   'ONPU_CPU_GQA_PREFILL': '0',
   'ONPU_INT8_SDK_PREFILL': '0',
   'ONPU_NORM_QUANT': '0',
   'ONPU_HOST_POOL_WFE': '0',
   'ONPU_DECODE_HOST_WFE': '0'},
 'scalar_decode_loop': True,
 'uncached_prefill': True,
 'cpu_q8_embedding': True}
```

27 environment variables. Cross-checked against `internlm2-api.json` `placement`:
`share_projection_weights true` ✓, `shared_projection_layers 120` ✓,
`prepared_qkv_blocks 24` ✓, `shared_host_pool true` ✓, `async_irq_wake true` ✓,
`uncached_prefill true` ✓, `decode_host_event_wait false` ✓, `host_event_wait false` ✓,
`cpu_q8_embedding` present ✓, `tile_columns 24576` ✓.

### 4.5 Protocol

The InternLM2 probe writes **no** `protocol` dict — `probe.py:63-65` is just
`for i in range(3): chat('warmup-'+str(i))` then `for i in range(10): chat('measured-'+str(i))`.
The protocol is documented only in the aggregation's `workload` field (quoted in §4.1):

- 3 warm-ups + 10 measured per process, 2 processes = 20 measured trials.
- `max_tokens=64`, temperature 0, speculation off, `cache_prompt=False`, stream off.
- **No 0.2 s fixed idle** between measured requests (unlike the other three).
- 128-token prompt from `staged/internlm2-generic-api-v2/prompt-128.json`
  (`input_tokens` length 128, `prompt_tokens: 128`, `reply_budget: 64`, `context: 512`,
  `input_ids_sha256: ff77610f…`, `worker_prepare_prompt_verified: true`).
- Boundary fixture `prompt-448.json` (448+64 boundary check).
- 18 checks per process, 36 total.

---

## 5. What the four have in common

| dimension     | shared value |
| ------------- | ------------ |
| **code tree** | `/home/orangepi/tinygrad` — `R = Path('/home/orangepi/tinygrad')` at `probe.py:4` in all four stages |
| **app tree**  | `/home/orangepi/oNPU` — `A = Path('/home/orangepi/oNPU')`, `cwd=A` for the server |
| **runner**    | the same `probe.py` shape, copied across all four stages (121 / 110 / 103 / 96 lines for TeleChat2 / Qwen2 / MiniCPM4 / InternLM2); InternLM2 v1/v2 are byte-identical (`114c7ba9…`) |
| **invocation**| `taskset -c 0-3 /home/orangepi/oNPU/.venv/bin/python probe.py <report-path>` with `PYTHONPATH=/home/orangepi/tinygrad:/home/orangepi/oNPU` |
| **entry point** | `python -m onpu --port 8092 --backend /home/orangepi/tinygrad` → `python -m onpu.worker`. **Never** `extra.rockchip.benchmark_inference` |
| **worker interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` (`runtime.py:61-63`) |
| **protocol**  | 3 warm-ups + 10 measured, 128 prompt / 64 gen, context 512, `temperature=0`, `speculation=off`, `cache_prompt=False` |
| **transport** | real HTTP `POST /v1/chat/completions`; metrics read back from `/api/status` `last_metrics` |
| **load call** | `{"profile":"ROCKCHIP","context":512,"realize":true,"npu_prefill":"auto","npu_decode":"auto","vision_gpu":false}` |
| **common gates** | external original GPU on 8091 preserved unchanged; `MemAvailable >= 2 GiB`; checkpoint sha256 == manifest; all source pins match before **and** after; exactly one `onpu.worker` with affinity `[4,5,6,7]`; per-request `prompt_tokens == 128`, `cached_tokens == 0`, `npu_submissions > 0`; target unloaded at the end |
| **clock policy** | clocks **recorded** per trial, **not gated** — scope string: "stock adaptive governors, no sudo tuning" |
| **recipe source** | one function, `onpu/rockchip_options.py:text_execution_profile`, keyed by GGUF sha256 |
| **aggregation** | pooled median over the raw `trials[]` of the per-process reports, `prefill_proxy = 128000 / median_ttft_ms` |
| **stale status** | all four rows are still flagged stale in `PUBLISH.md`; none has independent quality sign-off |

## 6. What differs

| dimension        | TeleChat2 3B | Qwen2 0.5B | MiniCPM4 0.5B | InternLM2 1.8B |
| ---------------- | ------------ | ---------- | ------------- | -------------- |
| processes        | 2 (`api-a`,`api-b`) | 1 (`qwen2`) | 2 (`control-a`,`control-b`) | 2 (v1, v2 — **different stages**) |
| recipe branch    | dedicated `:221-278` | dense `:403` | dense `:405` | dense `:406` |
| env vars         | 39 | 30 | 25 | 27 |
| `tile_columns`   | **24384** | 24576 | 24576 | 24576 |
| `share_projection_weights` | True | False | False | True |
| `decode_host_event_wait` | True | True | True | **False** |
| `ONPU_INT8_HADAMARD` | **128** | — | — | — |
| `ONPU_NORM_QUANT` | 0 | **1** | — | 0 |
| `ONPU_SWIGLU_K_SPLIT` / `BATCHED_PACK` / `C4_PREFETCH` | 0 / 0 / 0 | **1 / 1 / 16** | — | — |
| `ONPU_INT8_EXPERT_DEPTHS` | 0 | **1** | 0 | 0 |
| `sdk_gateup`     | — | — | **(128, 8192, 1024)** | — |
| extra scales asset | **`telechat2-full-cpu-20261103.json.gz`** (`b502b990…`) | — | — | — |
| quality gate     | **yes** — `limited_screen_pass` required before start | no | no | no |
| prior-response gate | first warm-up | prior report json | prior report json | **none** |
| port-8092 pre-bind | yes | yes | **no** | **no** |
| `PARALLEL=8` in worker environ asserted | no | no | **yes** | no |
| `protocol` dict written | yes | yes | yes | **no** |
| 0.2 s fixed idle | yes | yes | yes | **no** |
| `checks[]` per process | **19** | 18 | 18 | 18 |
| source pins      | 326 | 321 | 313 | 311 |
| published TTFT   | 536.70 | 141.62 | 130.07 | 339.19 |
| published decode | 10.45 | 48.78 | 57.56 | 16.16 |

The single most consequential difference: **TeleChat2 is the only one that refuses to start
without a passing independent quality screen**, and **InternLM2 is the only one whose
published number is a pool of two different source-pin sets**.

---

## 7. Things that look like the reproduce path but are not

Three other harnesses exist in the archive and all four share the word "retime". None of
them produced any of the four published numbers.

### 7.1 `scripts/bench/retime_with_profile.py` — different tree, different entry point

- Defaults `TREE = /home/orangepi/tg-decode-all` (`retime_with_profile.py:19`) and
  `PY = ~/tinygrad/.venv/bin/python` (`:20`).
- Drives `extra.rockchip.benchmark_inference` (`:100-103`), not the HTTP app.
- Protocol 3 warm-ups + 5 runs (`:52-57`), not 3 + 10.
- Its own docstring says it exists to re-derive the Nov-4 numbers from the app profile —
  but it is a *different* execution path, so it cannot reproduce a number measured through
  the API.
- **It has no retained successful artifact for any of these four.** The only
  `analysis/retime-20261113/` directory for these models is `qwen2-0.5b/`, and its
  `report.json` has `complete: false`, `trials: 0`,
  `error: "RuntimeError: cycle detected in assign graph"` (raised inside
  `/home/orangepi/tg-decode-all/tinygrad/schedule/__init__.py:78`). See
  `analysis/retime-20261113/qwen2-0.5b/run.log`.

### 7.2 `scripts/bench/retime_model.sh` — different tree, different protocol

- `PY=/home/orangepi/tinygrad/.venv/bin/python`, `TREE=${TREE:-/home/orangepi/tg-decode-all}`
  (`retime_model.sh:7-8`).
- Protocol comment says "3 warmups + 5 runs" (`:5`), argv says `--warmups 3 --runs 5`
  (`:119`).
- Has its own hardcoded env block (`:106-116`) that is **not** the `rockchip_options.py`
  recipe — e.g. it sets `ONPU_INT8_SDK_PREFILL=0` and `ONPU_CPU_RMS_ALL_ROWS=1` globally,
  which contradicts the Qwen2 and MiniCPM4 registry recipes.
- No retained artifact under `analysis/retime-20261113/` for TeleChat2, MiniCPM4 or
  InternLM2.

### 7.3 The Nov-16/17 "candidate retime" cohort — third tree, 1+3, still not published

`analysis/release-20261115/r2-lfm-20261116/candidate-retime-20261117-minicpm4/` and
`…-internlm2/` hold `receipt.json` + `benchmark.json` for a **1 warm-up + 3 measured**
direct retime:

- runner `run_candidate_minicpm4_timing.py` (sha `d299aa21…`) /
  `run_candidate_internlm2_timing.py` (sha `b72b3145…`)
- child `candidate_timing_child.py` (sha `b2e556b4…`), which imports
  `extra.rockchip.benchmark_inference` and calls `.main()`
- backend `/home/orangepi/tg-minicpm4-retime-candidate`, rev `1107c67b…`, two dirty files
  (`tinygrad/llm/kernels/rockchip_gelu_ffn.py`, `test/models/test_rockchip_gelu_ffn.py`),
  diff sha `b98eee00…`
- app `/home/orangepi/oNPU-release-rk3588-20261115-app`, rev `36354ae9…` —
  **this path no longer exists on disk** (`ls` → No such file or directory), so this path
  is not re-runnable as recorded
- results: MiniCPM4 decode 57.8397 tok/s / TTFT 1611.156 ms; InternLM2 decode 16.7351 /
  TTFT 2029.191 ms
- `plan.md:88-92` and `:100-106`, `PUBLISH.md:12-19` all say these leave the full rows
  **stale** — they are evidence, not replacements

TeleChat2's equivalent is `candidate-timing-20261116-telechat2/` + `-retry1/`, which
produced **zero** warm-ups and **zero** measured trials (blocked on a missing `numpy` in the
child interpreter, then on an inactive `shared_host_pool`). `plan.md:358-378` marks it
`BLOCKED BEFORE WARM-UP; RETRY BUDGET EXHAUSTED`.

### 7.4 The `~/tg-decode-all` claim

The repo `AGENTS.md` table says `~/tg-decode-all` is "the tree the published numbers were
measured from", citing `retime_with_profile.py`'s default. **For these four models that is
false.** The pinned `source_sha256` maps in all four manifests are dominated by
`/home/orangepi/tinygrad/` paths (305 / 300 / 299 / 305 of 321 / 313 / 311 / 326), and
`probe.py:4` hardcodes `R = Path('/home/orangepi/tinygrad')`. `~/tg-decode-all` appears
only in the failing `retime_with_profile.py` path (§7.1).

---

## 8. Fields that could not be determined, and one hard caveat

### 8.1 The source epoch is not reproducible from the current trees

Every probe asserts `raw['source_before'] == manifest['source_sha256']` before it will
start. **That assertion fails today.** I recomputed sha256 for every pinned path:

| cohort | pinned | match now | differ | missing |
| ------ | -----: | --------: | -----: | ------: |
| `qwen-current-api-20261102-v2` | 321 | 292 | 29 | 0 |
| `minicpm4-current-api-20261101-v1` | 313 | 280 | 33 | 0 |
| `internlm2-generic-api-v2` | 311 | 280 | 31 | 0 |
| `telechat2-hadamard-api-20261103-v1` | 326 | 300 | 26 | 0 |

So the **command** in §1.2/§2.2/§3.2/§4.2 is exact and re-runnable, but the **source
epoch** it pins is not on disk any more. A reproducer today would either (a) fail gate 7 of
each probe, or (b) have to relax that gate and accept a different backend build. Neither
yields the published number without re-measuring.

The drift is concentrated in the same files for all four — the shared rockchip kernel set:

```
tinygrad/tinygrad/llm/model.py
tinygrad/tinygrad/runtime/support/rockchip_int8.py
tinygrad/tinygrad/llm/kernels/rockchip.py
tinygrad/tinygrad/llm/kernels/rockchip_int8.py
tinygrad/tinygrad/llm/kernels/rockchip_gelu_ffn.py
tinygrad/tinygrad/llm/kernels/rockchip_prepare_pool.py
tinygrad/tinygrad/llm/kernels/rockchip_replay.py
tinygrad/tinygrad/llm/kernels/rockchip_shared_pool.py
tinygrad/tinygrad/llm/kernels/rockchip_mla_attention.{c,py}
tinygrad/tinygrad/llm/kernels/rockchip_mla_cpu.{c,py}
tinygrad/tinygrad/llm/kernels/rockchip_sigmoid.py
tinygrad/tinygrad/llm/kernels/rockchip_softcap.py
tinygrad/tinygrad/llm/kernels/rockchip_static.py
tinygrad/tinygrad/llm/kernels/rockchip_cpu.py
tinygrad/tinygrad/llm/kernels/rockchip_embedding.py
tinygrad/tinygrad/llm/kernels/rockchip_gelu.py
tinygrad/tinygrad/llm/kernels/rockchip_grouped_int8.py
```

plus `oNPU/onpu/{app,library,rockchip_options,worker}.py` in all four (and
`preparation.py`, `mla.py`, `templates.py`, `runtime.py` in some).

`~/tinygrad` is a git repo at `99cc2ffe2` with a dirty working tree, but the pinned hashes
are not any single commit — they are a working-tree snapshot from 2026-11-01..03. There is
no recorded commit id for that snapshot in any of the four manifests. **This is the one
field I could not determine: the commit the published numbers were built from.**

### 8.2 The recipe registry has also moved

`onpu/rockchip_options.py` is in the "differ" list for all four cohorts, so the file that
produced these numbers is not the file on disk now. I verified the *current* registry still
yields the *same* recipe for all four checksums (§1.4, §2.4, §3.4, §4.4) and that each
result matches the `placement` recorded in the measured report. The recipe is stable; the
file hash is not.

### 8.3 Smaller unknowns

| field | status |
| ----- | ------ |
| exact commit of `/home/orangepi/tinygrad` at measurement | **UNKNOWN** — no commit id recorded in any of the four manifests; only per-file sha256 |
| exact commit of `/home/orangepi/oNPU` at measurement | **UNKNOWN** — same reason |
| who aggregated per-process reports into `docs/benchmark-*.json` | **UNKNOWN** — no aggregator script retained; the four `docs/` files match the `benchmark-model-status.json` `current_preferred` blocks exactly, but the code that wrote them is not in `~/oNPU/scripts/` or `~/oNPU/` (searched `grep -rln "telechat-hadamard-production" ~/oNPU/` → only the json itself) |
| `prefetch_native_calls` for Qwen2 | **determined**: 24, in `qwen2-ksplit-production-backend-20261101-v1/results/backend.json` (the manifest's `backend_gate` path, not the `…-backend-20261101-v1` path I first tried) |
| InternLM2 `protocol` dict | **absent by design** — the probe never writes one; use the aggregation's `workload` block |

### 8.4 What a reproducer can actually do today

1. **Verify the arithmetic** — recompute the pooled median from the retained `trials[]`.
   This reproduces all four published numbers exactly (table in §0) and needs no hardware.
2. **Verify the recipe** — call
   `text_execution_profile(<checksum>, 512, "ROCKCHIP", "on", decode="on")` and compare to
   §1.4/§2.4/§3.4/§4.4 and to the `placement` in the retained report. This also needs no
   hardware.
3. **Re-run the measurement** — blocked on the source epoch. Either recover the
   `~/tinygrad` working tree as of 2026-11-01..03 (not recorded), or accept that a fresh run
   measures a different backend and label it as such.
