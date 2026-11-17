# Reproduce paths — Qwen2.5 1.5B, Qwen3 0.6B, Gemma2 2B, TinyLLAMA 1.1B

Read-only audit of the oNPU archive, 2026-11-18. Nothing was run on the NPU, no model was
loaded, no archive file was written.

**"The published number"** below means the `oNPU` column of the 21-model selected table
(`/home/orangepi/oNPU/benchmark.md:933-955`, prefill proxy at `:957-981`, download ranking at
`:985-1005`). All four rows were produced by the **HTTP API harness** (`python -m onpu`), *not*
by `extra.rockchip.benchmark_inference` directly. The standalone runner
(`scripts/bench/retime_with_profile.py`) is a **later, different** measurement — see
"§5 The other runner" at the end.

Verification actually performed (all read-only):

```
$ cd /home/orangepi && sha256sum \
    qwen3-bench/matched/roofline/gpu/Qwen3-0.6B-matched-f16.gguf \
    vlm-bench/baseline-20261021/qwen2.5-1.5b-instruct-q8_0.gguf \
    vlm-bench/baseline-20261021/checkpoints/Gemma2-2B/gemma-2-2b-it-Q8_0.gguf \
    vlm-bench/baseline-20261021/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035  qwen3-bench/matched/roofline/gpu/Qwen3-0.6B-matched-f16.gguf
d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8  vlm-bench/baseline-20261021/qwen2.5-1.5b-instruct-q8_0.gguf
2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe  vlm-bench/baseline-20261021/checkpoints/Gemma2-2B/gemma-2-2b-it-Q8_0.gguf
a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59  vlm-bench/baseline-20261021/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

Each of the four report JSONs was also re-hashed against its recorded `source_before` pin map
(§"Source-pin re-check" at the end). Each report's `entry` dict was compared byte-for-byte with
the `entries[]` of the staged manifest that produced it, and each `summary` was recomputed from
the raw trials.

---

## 1. Qwen2.5 1.5B

| field | value | proof |
| --- | --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/qwen2.5-1.5b-instruct-q8_0.gguf` (1,894,532,128 B) | `checkpoints/manifests/Qwen2.5-1.5B.download.json:10`; `qwen-current-api-20261102-v2/results/qwen25.json` → `entry.checkpoint`; `ls -la` shows the file |
| **checksum** | `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8` | manifest `:8` and `:12`; `entry.checkpoint_sha256`; re-hashed on disk (above) |
| **code tree** | `/home/orangepi/tinygrad` | `staged/qwen-current-api-20261102-v2/probe.py:4` — `R=Path('/home/orangepi/tinygrad')`; every `source_before` pin is under `/home/orangepi/tinygrad/…` or `/home/orangepi/oNPU/onpu/…` |
| **interpreter** | harness: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` | `staged/qwen-current-api-20261102-v2/probe.py:24` (`A/'.venv/bin/python' -m onpu --backend R`), then `onpu/runtime.py:61-63` (`self.backend / ".venv/bin/python"`) used at `onpu/runtime.py:150-157` |
| **runner** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/qwen-current-api-20261102-v2/probe.py` | report `entry` == this manifest's `entries[slug=qwen25]` exactly (verified); `build_command.py` in the same dir builds the argv |
| **entry point** | `python -m onpu` HTTP server → `POST /api/models/import`, `POST /api/models/<id>/load {profile:ROCKCHIP, context:512, realize:true, npu_prefill:auto, npu_decode:auto}`, then `POST /v1/chat/completions` | `probe.py:24`, `:32-33`, `:69-70` |
| **recipe** | `onpu/rockchip_options.py:403-411` `dense_profiles[QWEN25_CHECKPOINT] = ("qwen2.5-1.5b-context512", 4, 24576)`, branch at `:413`, plus `:424-425` (`REPLAY_NATIVE_FROM_LARGE`, `CPU_DECODE_POOL`, `EXP2_FAST_PATH`), `:433-434` (`INT8_SDK_PREFILL`, `FFN_PACKED_UPLOAD`), `:479-483` (`scalar_decode_loop`, `HOST_POOL_WFE=0`, `DECODE_HOST_WFE=1`), `:484-485` (`sdk_split_prefill`). Full dict evaluated from the module — see below | `onpu/rockchip_options.py` |
| **gates** | see list below | `probe.py:15,18,19,21,41,49,59,79,100`; `build_command.py` asserts manifest source hashes |
| **artifact** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/qwen-current-api-20261102-v2/results/qwen25.json` (sha256 `a94579f660920af9951432be74987ceca92b18ad2724ab84dc4ccbab65d4c304`) | `docs/benchmark-model-status.json:1910`; `docs/benchmark-qwen-current-api-20261102.json:365,403` |
| **protocol** | **3 warm-ups + 10 measured** HTTP requests; 128 prompt tokens, `max_tokens=64`, `temperature=0`, `cache_prompt=false`, `stream=false`, 200 ms fixed idle between measured requests; context 512 | `probe.py:74-76`; report `protocol` = `{warmups:3, measured:10, fixed_idle_seconds:0.2, …}` |

Measured (report `summary`, recomputed from the 10 trials): TTFT **355.9728 ms**, decode
**17.4171 tok/s**, prefill proxy **359.5781 tok/s** → published `355.97 / 17.42 / 359.58`
(`benchmark.md:939`, `:965`, `:993`).

Full recipe dict for this checksum (`text_execution_profile(ck, 512, "ROCKCHIP", "on", decode="on")`):

```python
{"prefill": "on", "mapped_cpu": True, "share_projection_weights": False, "prefill_chunk": 128,
 "tile_columns": 24576, "dense_swiglu": True, "prefill_profile": "qwen2.5-1.5b-context512",
 "scalar_decode_loop": True, "sdk_split_prefill": True,
 "environment": {
   "ONPU_SWIGLU_PREFILL_ROWS": "4", "ONPU_INT8_EXPERT_DEPTHS": "0", "ONPU_INT8_NATURAL2304": "1",
   "ONPU_INT8_DECODE_NARROW_CBUF": "1", "ONPU_NPU_DOMAINS": "1",
   "ONPU_NPU_DOMAIN_BYTES": "3221225472", "ONPU_CPU_TOKEN_INPUTS": "1",
   "ONPU_ZERO_PAD_VIEWS": "1", "ONPU_REPLAY_BORROW_OUTPUTS": "1",
   "ONPU_REPLAY_BORROW_BATCH": "1", "ONPU_REPLAY_NATIVE_FROM": "1",
   "ONPU_SHARED_HOST_POOL": "1", "ONPU_PREFILL_CPU_TUNING": "1",
   "ONPU_CPU_PREFILL_LAYOUT": "1", "ONPU_REPLAY_NATIVE_FROM_LARGE": "1",
   "ONPU_CPU_DECODE_POOL": "1", "ONPU_EXP2_FAST_PATH": "1",
   "ONPU_INT8_SDK_PREFILL": "1", "ONPU_FFN_PACKED_UPLOAD": "1",
   "ONPU_HOST_POOL_WFE": "0", "ONPU_DECODE_HOST_WFE": "1"}}
```

Gates enforced before/while starting (`staged/qwen-current-api-20261102-v2/probe.py`):

1. `:15` external original GPU on port 8091 must be `ready` with exactly one session on `CL`.
2. `:18` `MemAvailable >= 2 GiB`, sampled repeatedly (`minimum_available` recorded).
3. `:19` on-disk GGUF sha256 must equal `entry.checkpoint_sha256`.
4. `:21` all 321 pinned source files must hash to the manifest map, before the run.
5. `:41` `placement.tile_columns == entry.expected_tile_columns` (24576 for this model).
6. `:49` `placement.prefill_profile == "qwen2.5-1.5b-context512"`, `decode_host_event_wait is True`,
   `host_event_wait is False`, and `cpu_q8_embedding.fast_traces > 0` when `embedding`.
7. `:59` exactly one `onpu.worker` process for this checkpoint, affinity `[4,5,6,7]`.
8. `:79` every measured reply must equal the first reply of the prior report (bit-exact text).
9. `:100` source pins unchanged after the run (`source_after == source_before`).

---

## 2. Qwen3 0.6B

**This row does not use a checkpoint under `baseline-20261021/`.** The published number was
measured from an F16 GGUF in a different directory, and the checksum in
`checkpoints/manifests/Qwen3-0.6B.download.json` is a *different file* that never produced this
row.

| field | value | proof |
| --- | --- | --- |
| **checkpoint path** | `/home/orangepi/qwen3-bench/matched/roofline/gpu/Qwen3-0.6B-matched-f16.gguf` (1,509,347,168 B) — **outside** `/home/orangepi/vlm-bench/baseline-20261021/` | `qwen-current-api-20261102-v1/results/qwen3.json` → `entry.checkpoint`, `registered_model.path`, `registered_model.size`; `ls -la` shows the file. A `find /home/orangepi/vlm-bench/baseline-20261021 -iname '*qwen3*gguf*'` returns only Qwen3.5-0.8B/2B/4B and Qwen3-VL-2B — no Qwen3-0.6B |
| **checksum** | `6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035` | `entry.checkpoint_sha256`; equals `QWEN3_PREFILL_CHECKPOINT` at `onpu/rockchip_options.py:26`; re-hashed on disk (above) |
| **code tree** | `/home/orangepi/tinygrad` | `staged/qwen-current-api-20261102-v1/probe.py:4` — `R=Path('/home/orangepi/tinygrad')`; all pins under `/home/orangepi/tinygrad/…` and `/home/orangepi/oNPU/onpu/…` |
| **interpreter** | harness: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` | `staged/qwen-current-api-20261102-v1/probe.py:24`; `onpu/runtime.py:61-63`, `:150-157` |
| **runner** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/qwen-current-api-20261102-v1/probe.py` | report `entry` == that manifest's `entries[slug=qwen3]` exactly (verified). **Note:** the report's `entry.harness_origin` field points at `staged/qwen-current-source-final-api-20261101-v1`, but that manifest's entry differs (`prompt: qwen3-128.json` vs `qwen3-prompt.json`, `boundary: qwen3-448.json` vs `qwen3-boundary.json`) — `harness_origin` is a stale pointer, the v1 stage is the real one |
| **entry point** | `python -m onpu` HTTP server → import → load `ROCKCHIP`/ctx 512 → `POST /v1/chat/completions` | `probe.py:24`, `:32-33`, `:69-70` |
| **recipe** | **not** `text_execution_profile` — that returns `{}` for this checksum (it is absent from `TEXT_INT8_CONTEXTS`). It is `qwen3_prefill_profile`, `onpu/rockchip_options.py:57-71`, applied separately at `onpu/worker.py:318-319` | `onpu/rockchip_options.py:57-71`; `onpu/worker.py:318` |
| **gates** | see list below (plus a hard `tile_columns == 24384` assert) | `probe.py:15,18,19,21,41,49,59,79,100` |
| **artifact** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/qwen-current-api-20261102-v1/results/qwen3.json` (sha256 `b8e65e1fc0c925d7da86714a72dd53ce42f6c019ac8325e88140c61be85825c0`) | `docs/benchmark-model-status.json:1568`; `docs/benchmark-qwen-current-api-20261102.json:193,231` |
| **protocol** | **3 warm-ups + 10 measured** HTTP requests; 128 prompt tokens, `max_tokens=64`, `temperature=0`, `cache_prompt=false`, `stream=false`, 200 ms idle; context 512 | `probe.py:74-76`; report `protocol` |
| **status today** | **STALE-Nov4** — no retime exists; `scripts/bench/retime_with_profile.py:24-51` has no `qwen3-0.6b` key | `analysis/scoreboard-20261113/SCOREBOARD.md:67`; `handoff.md` "Qwen3 0.6B — no runner" |

Measured: TTFT **184.8299 ms**, decode **40.4850 tok/s**, prefill proxy **692.5288 tok/s** →
published `184.83 / 40.48 / 692.53` (`benchmark.md:937`, `:963`, `:991`).

Full recipe dict (`qwen3_prefill_profile(6782193…, 512, "ROCKCHIP", "on", "on")`):

```python
{"prefill": "on", "mapped_cpu": True, "share_projection_weights": True, "prefill_chunk": 128,
 "tile_columns": 24384, "qwen3_fast_prefill": True, "scalar_decode_loop": True,
 "environment": {
   "ONPU_HOST_POOL_WFE": "0", "ONPU_DECODE_HOST_WFE": "1",
   "ONPU_INT8_SDK_PREFILL": "1", "ONPU_SWIGLU_PREFILL_ROWS": "4",
   "ONPU_FFN_PACKED_UPLOAD": "1", "ONPU_EXP2_FAST_PATH": "1",
   "ONPU_SHARED_HOST_POOL": "1", "ONPU_ZERO_PAD_VIEWS": "1",
   "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_TOKEN_INPUTS": "1",
   "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
   "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1"}}
```

Gates (`staged/qwen-current-api-20261102-v1/probe.py`): same 1–9 as Qwen2.5, except

* `:41` asserts `placement.tile_columns == 24384` (hard-coded, not from the entry), and
* `:49` asserts `prefill_profile == "qwen3-0.6b-context512"` — a literal set at
  `onpu/worker.py:630`, not a value returned by the recipe dict.

### What the two Qwen3 checkpoints are

| file | sha256 | role |
| --- | --- | --- |
| `/home/orangepi/qwen3-bench/matched/roofline/gpu/Qwen3-0.6B-matched-f16.gguf` | `6782193c…` | **the published row.** F16, `profile=qwen3-0.6b-context512`, has a recipe |
| `/home/orangepi/oNPU/.data/models/72717a59205944dc-Qwen3-0.6B-Q8_0.gguf` | `9465e63a…` | the *official Q8_0* named by `checkpoints/manifests/Qwen3-0.6B.download.json:10`. Used only for the CL baseline (`qwen3-0.6b-cl-baseline.json`) and for a historical 2026-10-22 profile. **No recipe**: absent from `TEXT_INT8_CONTEXTS` and from `qwen3_prefill_profile`'s guard |

`analysis/release-20261115/r2-lfm-20261116/qwen3-06b-source-audit.md` records the Q8 recipe as
recovered-but-unreproducible (source epoch missing, current tree matches 1/15 recorded hashes) and
concludes "BLOCKED before hardware timing". So the Q8 file has no reproduce path at all; the F16
file is the only Qwen3-0.6B path that reproduces the published number.

---

## 3. Gemma2 2B

| field | value | proof |
| --- | --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Gemma2-2B/gemma-2-2b-it-Q8_0.gguf` (2,784,495,456 B) | `checkpoints/manifests/Gemma2-2B.json:10`; `gemma2-current-api-20261101-v3/results/worker-a.json` → `entry.checkpoint`; `ls -la checkpoints/Gemma2-2B/` |
| **checksum** | `2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe` | manifest `:8`, `:14`; `entry.checkpoint_sha256`; re-hashed on disk (above) |
| **code tree** | `/home/orangepi/tinygrad` | `staged/gemma2-current-api-20261101-v3/probe.py:4` — `R=Path('/home/orangepi/tinygrad')`; 316 pins, all under `/home/orangepi/tinygrad/…` (301) and `/home/orangepi/oNPU/onpu/…` (15) |
| **interpreter** | harness: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` | `staged/gemma2-current-api-20261101-v3/probe.py:24`; `onpu/runtime.py:61-63`, `:150-157` |
| **runner** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/gemma2-current-api-20261101-v3/probe.py`, run twice (`ONPU_CASE=worker-a`, then `worker-b`), then pooled by `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/publish-gemma2-api-20261101.py` | both `worker-a.json` and `worker-b.json` `entry` match that manifest's `entries[]` exactly (verified); pooling script asserts `len(trials)==10`, `len(checks)==18`, `source_before==source_after`, `tile_columns==24384`, checkpoint sha, and identical reply text across both workers |
| **entry point** | `python -m onpu` HTTP server → import → load `ROCKCHIP`/ctx 512 (`realize:true`, `npu_prefill:auto`, `npu_decode:auto`) → `POST /v1/chat/completions` | `probe.py:24`, `:31-32`, `:62-63` |
| **recipe** | `onpu/rockchip_options.py:364-378` — its own branch, **not** the `dense_profiles` table | `onpu/rockchip_options.py:364` |
| **gates** | see list below, including a Gemma2-specific placement gate | `probe.py:15,18,19,21,39,49,59,75,96` |
| **artifact** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/gemma2-current-api-20261101-v3/pooled20.json` (sha256 `17c1977b3a1c86ac432205a49a5952b26f3b65de7c8103e8617dc6d35df5ba88`); per-worker: `results/worker-a.json` (`07f55809…`), `results/worker-b.json` (`79aeff27…`) | `docs/benchmark-model-status.json:2435`; `benchmark.md:2029-2035` |
| **protocol** | **2 independent fresh processes × (3 warm-ups + 10 measured)** = 20 measured requests pooled; 128 uncached prompt tokens, `max_tokens=64`, `temperature=0`, `cache_prompt=false`, 200 ms idle; context 512 | `probe.py:67-69`; `pooled20.json` → `scope`; `docs/benchmark-model-status.json:2430` (`measurement_scope`) |

Measured (`pooled20.json` `summary`, recomputed over all 20 trials): TTFT **543.2106 ms**, decode
**10.6766 tok/s**, prefill proxy **235.6360 tok/s** → published `543.21 / 10.68 / 235.64`
(`benchmark.md:942`, `:968`, `:1002`).

Full recipe dict (`text_execution_profile(2d448a9a…, 512, "ROCKCHIP", "on", decode="on")`):

```python
{"prefill": "on", "mapped_cpu": True, "share_projection_weights": True, "prefill_chunk": 128,
 "tile_columns": 24384, "gelu_fast": True, "scalar_decode_loop": True, "cpu_q8_embedding": True,
 "uncached_prefill": True, "trim_preparation_cache": True,
 "prefill_profile": "gemma2-2b-context512",
 "environment": {
   "ONPU_NPU_DOMAINS": "3", "ONPU_NPU_DOMAIN_BYTES": "3221225472",
   "ONPU_INT8_NATURAL2304": "1", "ONPU_GELU_PACK_STATIC_LAYOUT": "1",
   "ONPU_INT8_SDK_PREFILL": "0", "ONPU_SHARED_HOST_POOL": "1",
   "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1",
   "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
   "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_REPLAY_NATIVE_FROM_LARGE": "1",
   "ONPU_CPU_DECODE_POOL": "1", "ONPU_PREFILL_CPU_TUNING": "1",
   "ONPU_CPU_PREFILL_LAYOUT": "1", "ONPU_EXP2_FAST_PATH": "1",
   "ONPU_EXP2_SCALE_BITS": "1", "ONPU_FFN_PACKED_UPLOAD": "1",
   "ONPU_ASYNC_IRQ_WAKE": "1", "ONPU_STATIC_PREFILL": "1",
   "ONPU_INT8_DECODE_NARROW_CBUF": "1", "ONPU_HOST_POOL_WFE": "0",
   "ONPU_DECODE_HOST_WFE": "1", "ONPU_CPU_GQA_PREFILL": "0", "ONPU_NORM_QUANT": "0"}}
```

Gates (`staged/gemma2-current-api-20261101-v3/probe.py`):

1. `:15` external GPU on 8091 `ready`, one `CL` session.
2. `:18` `MemAvailable >= 2 GiB`.
3. `:19` GGUF sha256 == `entry.checkpoint_sha256`.
4. `:21` all 316 source pins match the manifest.
5. `:39` **Gemma2-specific:** `placement.softcap_shared_pool is True` **and**
   `placement.fused_gelu_ffn_blocks == 26` **and** `placement.tile_columns == 24384`.
6. `:49` `prefill_profile == "gemma2-2b-context512"`, `decode_host_event_wait is True`,
   `host_event_wait is False`, `cpu_q8_embedding.fast_traces > 0`.
7. `:59` one `onpu.worker`, affinity `[4,5,6,7]`, `PARALLEL=8` in its environ.
8. `:75` reply text identical across all 10 trials (worker-b additionally against worker-a).
9. `:96` `source_after == source_before`.

---

## 4. TinyLLAMA 1.1B

| field | value | proof |
| --- | --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/tinyllama-1.1b-chat-v1.0.Q8_0.gguf` (1,170,781,568 B) | `checkpoints/manifests/TinyLLAMA-1.1B.download.json:10`; `tinyllama-generic-api-v1/results/tiny-api.json` → `entry.checkpoint` |
| **checksum** | `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59` | manifest `:8`, `:12`; `entry.checkpoint_sha256`; re-hashed on disk (above) |
| **code tree** | `/home/orangepi/tinygrad` | `staged/tinyllama-generic-api-v1/probe.py:4` — `R=Path('/home/orangepi/tinygrad')`; 311 pins, all under `/home/orangepi/tinygrad/…` (299) and `/home/orangepi/oNPU/onpu/…` (12) |
| **interpreter** | harness: `/home/orangepi/oNPU/.venv/bin/python`; worker: `/home/orangepi/tinygrad/.venv/bin/python` | `staged/tinyllama-generic-api-v1/probe.py:22`; `onpu/runtime.py:61-63`, `:150-157` |
| **runner** | `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/tinyllama-generic-api-v1/probe.py` **and** `…/staged/tinyllama-generic-api-v2/probe.py` (two cohorts, pooled by `/home/orangepi/oNPU/docs/benchmark-tinyllama-generic-runtime-20261031.json`) | both reports' `entry` match their own manifests exactly (verified) |
| **entry point** | `python -m onpu` HTTP server → import → load `ROCKCHIP`/ctx 512 → `POST /v1/chat/completions` | `probe.py:22`, `:30-31`, `:58-59` |
| **recipe** | `onpu/rockchip_options.py:403-411` `dense_profiles[TINYLLAMA_CHECKPOINT] = ("tinyllama-1.1b-context512", 4, 24384)`, branch `:413`, plus `:426-432` (TinyLlama/MiniCPM4 group: `REPLAY_NATIVE_FROM_LARGE`, `CPU_DECODE_POOL`, `EXP2_FAST_PATH`, `EXP2_SCALE_BITS`, `FFN_PACKED_UPLOAD`, `ASYNC_IRQ_WAKE`, `STATIC_PREFILL`), `:479-483`, `:488-489` (`uncached_prefill`), `:490-491` (`cpu_q8_embedding`) | `onpu/rockchip_options.py` |
| **gates** | see list below | `probe.py:14,17,18,19,38,48,86` |
| **artifact** | `/home/orangepi/oNPU/docs/benchmark-tinyllama-generic-runtime-20261031.json` (the pooled 20-request record); component reports `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/tinyllama-generic-api-v1/results/tiny-api.json` (sha256 `e800ce04…`) and `…/tinyllama-generic-api-v2/results/tiny-api.json` | `docs/benchmark-model-status.json:1833`; `docs/benchmark-tinyllama-generic-runtime-20261031.json:17,19,24`; `benchmark.md:2478-2490` |
| **protocol** | **2 independent fresh processes × (3 warm-ups + 10 measured)** = 20 measured requests pooled; 128 uncached prompt tokens, `max_tokens=64`, `temperature=0`, `cache_prompt=false`; context 512 | `docs/benchmark-tinyllama-generic-runtime-20261031.json:2222` (`workload`); `probe.py:63-64` |

Measured (median over the 20 pooled trials, recomputed): TTFT **235.1772 ms**, decode
**27.0415 tok/s**, prefill proxy **544.2705 tok/s** → published `235.18 / 27.04 / 544.27`
(`benchmark.md:938`, `:964`, `:999`).

Full recipe dict (`text_execution_profile(a4c9bb1d…, 512, "ROCKCHIP", "on", decode="on")`):

```python
{"prefill": "on", "mapped_cpu": True, "share_projection_weights": False, "prefill_chunk": 128,
 "tile_columns": 24384, "dense_swiglu": True, "prefill_profile": "tinyllama-1.1b-context512",
 "scalar_decode_loop": True, "uncached_prefill": True, "cpu_q8_embedding": True,
 "environment": {
   "ONPU_SWIGLU_PREFILL_ROWS": "4", "ONPU_INT8_EXPERT_DEPTHS": "0",
   "ONPU_INT8_NATURAL2304": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
   "ONPU_NPU_DOMAINS": "1", "ONPU_NPU_DOMAIN_BYTES": "3221225472",
   "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1",
   "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
   "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_SHARED_HOST_POOL": "1",
   "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_PREFILL_LAYOUT": "1",
   "ONPU_REPLAY_NATIVE_FROM_LARGE": "1", "ONPU_CPU_DECODE_POOL": "1",
   "ONPU_EXP2_FAST_PATH": "1", "ONPU_EXP2_SCALE_BITS": "1",
   "ONPU_FFN_PACKED_UPLOAD": "1", "ONPU_ASYNC_IRQ_WAKE": "1",
   "ONPU_STATIC_PREFILL": "1", "ONPU_HOST_POOL_WFE": "0", "ONPU_DECODE_HOST_WFE": "1"}}
```

Gates (`staged/tinyllama-generic-api-v1/probe.py`; v2 identical at the cited lines):

1. `:14` external GPU on 8091 `ready`, one `CL` session.
2. `:17` `MemAvailable >= 2 GiB`.
3. `:18` GGUF sha256 == `entry.checkpoint_sha256`.
4. `:19` all 311 source pins match the manifest.
5. `:38` **TinyLlama-specific:** `placement.async_irq_wake is True` **and**
   `placement.uncached_prefill is True`, plus `prefill_profile == "tinyllama-1.1b-context512"`,
   `decode_host_event_wait is True`, `host_event_wait is False`,
   `cpu_q8_embedding.fast_traces > 0`.
6. `:48` one `onpu.worker`, affinity `[4,5,6,7]`.
7. `:86` `source_after == source_before`.

Note: this cohort's `probe.py` is the older variant — it has **no** `protocol` key in the report
and **no** `prior_response_exact` oracle. The protocol is only recoverable from the pooled record's
`workload` block (`docs/benchmark-tinyllama-generic-runtime-20261031.json:2222`).

---

## 5. The other runner — `retime_with_profile.py` (does **not** reproduce these rows)

`/home/orangepi/oNPU/scripts/bench/retime_with_profile.py` is the standalone retime harness, and
the only place where `TREE=/home/orangepi/tg-decode-all` appears as a default. That default is not
the tree the four published numbers came from.

| item | value | proof |
| --- | --- | --- |
| tree | `/home/orangepi/tg-decode-all` (default, `TREE` env override) | `scripts/bench/retime_with_profile.py:19` |
| interpreter | `/home/orangepi/tinygrad/.venv/bin/python` | `:20` |
| entry point | `python -m extra.rockchip.benchmark_inference` | `:99-101` |
| protocol | 3 warm-ups + **5** runs, 128/64, ctx 512 | `:52-57` |
| model keys | `qwen2-0.5b, minicpm4-0.5b, tinyllama-1.1b, qwen2.5-1.5b, rwkv7, internlm2, gemma2, gemma3n, telechat2, phi3, qwen3vl, deepseek-ocr, qwen35-0.8b, qwen35-2b, qwen35-4b, gemma4, lfm25vl, spark25, minicpm3, chatglm3` | `:24-51` |
| gates | NPU 1 GHz, DDR 2112 MHz, CPU4/6 `performance`, temp ≤ 60 °C, no `~/oNPU/.hw-lock` | `:59-70` |

Four facts that make it unusable as the reproduce path for these four rows:

1. **No `qwen3-0.6b` key** (`:24-51`) — Qwen3 0.6B cannot be retimed by it at all.
2. **`/home/orangepi/tg-decode-all` has no `.git` and no `.venv`** (`ls -la` shows neither; `git -C`
   reports "not a git repository"), so it cannot be fingerprinted and cannot supply an interpreter.
   The interpreter comes from `~/tinygrad` regardless — i.e. the tree and the interpreter are
   different checkouts.
3. **It fails outright on this model class.** The one recorded attempt through it,
   `analysis/retime-20261113/qwen2-0.5b/report.json`, is `complete: false`, `trials: 0`,
   `error: "RuntimeError: cycle detected in assign graph"` (3 warm-ups + 5 runs requested,
   none produced). `analysis/retime-20261113/README.md` keeps that as a documented negative and
   routes retimes through the app API instead.
4. The four published rows were measured from **`/home/orangepi/tinygrad`**, proven by the
   `source_before` pin maps (below), not from `tg-decode-all`.

`/home/orangepi/oNPU/scripts/bench/retime_model.sh` has the same shape (`:7` `PY=~/tinygrad/.venv`,
`:8` `TREE=~/tg-decode-all`, `:119` `--warmups 3 --runs 5`), plus a fail-closed qualifier
(`:129-131` → `qualify_retime_report.py`, which requires exactly 3 warm-ups + 5 runs, 128 input
IDs, context 512, and the `model._cached_tokens = []` reset in the runner).

### Newer exact-protocol retimes that exist for three of the four

These are a **different measurement** (1 warm-up + 3 runs, standalone runner, different tree) and
are what `analysis/scoreboard-20261113/SCOREBOARD.md` currently shows for TinyLLAMA, Gemma2 and
Qwen2.5. They do not reproduce the Nov-4 published rows.

| model | runner | tree | interpreter | protocol | artifact | result |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 1.5B | `scripts/bench/retime_model.sh` | `/home/orangepi/tg-retime-pack-fix` @ `1107c67b` + 2 dirty files | `/home/orangepi/tinygrad/.venv/bin/python` | 3 + 5 | `analysis/retime-20261114/qwen2.5-1.5b/report.json`, `run_manifest.json` | 366.766 ms / 17.8547 tok/s (median) |
| TinyLLAMA 1.1B | `analysis/release-20261115/r2-lfm-20261116/run_candidate_tinyllama_timing.py` | `/home/orangepi/tg-minicpm4-retime-candidate` @ `1107c67b` | `/home/orangepi/oNPU/.venv/bin/python` → `candidate_timing_child.py` | 1 + 3 | `candidate-retime-20261117-tinyllama/{benchmark.json,receipt.json}` | median 1652.862 ms; **last trial 204.337 ms**; decode median 26.7379 |
| Gemma2 2B | `…/run_candidate_gemma2_timing.py` | `/home/orangepi/tg-minicpm4-retime-candidate` @ `1107c67b` | `/home/orangepi/oNPU/.venv/bin/python` → `candidate_timing_child.py` | 1 + 3 | `candidate-retime-20261117-gemma2/{benchmark.json,receipt.json}` | median 3064.344 ms; **last trial 458.070 ms**; decode median 10.9515 |
| Qwen3 0.6B | — | — | — | — | — | **none.** `SCOREBOARD.md:67` = `STALE-Nov4` |

Both retimes report `timing_validity: PASS` but `ttft_max_min_ratio` 10.591× (TinyLlama) and
17.335× (Gemma2); `PUBLISH.md` says to retain the decode evidence only and keep the full rows stale.
The scoreboard's `204.3` and `458.1` are the **last measured trial**, not medians — verified against
`benchmark.json` trial `index=3` in both cases.

---

## 6. What the four have in common, and what differs

### Common

| dimension | shared value |
| --- | --- |
| **runner family** | the same `probe.py` HTTP-API harness, copied per stage into `…/main-task-20261023/staged/<stage>/`; each drives `python -m onpu` on port 8092 and posts `/v1/chat/completions` |
| **code tree** | `/home/orangepi/tinygrad` — proven by the `source_before` pin paths in all four reports |
| **harness interpreter** | `/home/orangepi/oNPU/.venv/bin/python` |
| **worker interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` (`onpu/runtime.py:61-63`) |
| **recipe source** | one file, `/home/orangepi/oNPU/onpu/rockchip_options.py`, keyed by checkpoint sha256 |
| **entry point** | `python -m onpu` → import → `load{profile:ROCKCHIP, context:512, realize:true, npu_prefill:auto, npu_decode:auto}` → chat completions. `extra.rockchip.benchmark_inference` is *not* in this path |
| **shared gates** | external GPU on 8091 `ready`/one `CL` session; `MemAvailable ≥ 2 GiB`; GGUF sha256 == entry; all source pins match before **and** after; one `onpu.worker` with affinity `[4,5,6,7]`; `prefill_profile` == entry; `decode_host_event_wait` == entry; `tile_columns` asserted |
| **workload** | 128 prompt tokens, 64 output tokens, `temperature=0`, `cache_prompt=false`, `stream=false`, context 512, `prefill_chunk=128` |
| **clock basis** | adaptive governors, **no sudo pinning** — proven by the reports' own samples: `qwen3.json` `checks[0]` (phase `warmup-0`) has `dmc/cur_freq = 528000000` and `policy4/6 = 408000` **before**, `2112000000` / `1416000`–`1800000` after. Stated for TinyLLAMA at `benchmark.md:2480` and for Gemma2 at `docs/benchmark-model-status.json:2430` |
| **status** | `quality_qualified: false` and `maximum_context_qualified: false` for all four |

### Different

| dimension | Qwen2.5 1.5B | Qwen3 0.6B | Gemma2 2B | TinyLLAMA 1.1B |
| --- | --- | --- | --- | --- |
| checkpoint under `baseline-20261021/` | **yes**, repo root | **no** — `~/qwen3-bench/matched/roofline/gpu/` | **yes**, `checkpoints/Gemma2-2B/` | **yes**, repo root |
| quantisation | Q8_0 | **F16** | Q8_0 | Q8_0 |
| recipe branch | `dense_profiles` (`:403-411`, `:413`) | **`qwen3_prefill_profile` (`:57-71`)** — `text_execution_profile` returns `{}` | **own branch (`:364-378`)** | `dense_profiles` (`:404`, `:413`) |
| `share_projection_weights` | **False** | True | True | **False** |
| `tile_columns` | **24576** | 24384 | 24384 | 24384 |
| `ONPU_NPU_DOMAINS` | **1** | 1 | **3** | **1** |
| `ONPU_INT8_SDK_PREFILL` | 1 | 1 | **0** | 1 |
| activation | SwiGLU (`dense_swiglu`) | SwiGLU + `qwen3_fast_prefill` | **GELU + softcap** (`gelu_fast`) | SwiGLU (`dense_swiglu`) |
| extra flags | `sdk_split_prefill` | `qwen3_fast_prefill` | `uncached_prefill`, `trim_preparation_cache`, `cpu_q8_embedding` | `uncached_prefill`, `cpu_q8_embedding` |
| stage dir | `staged/qwen-current-api-20261102-v2` | `staged/qwen-current-api-20261102-v1` | `staged/gemma2-current-api-20261101-v3` | `staged/tinyllama-generic-api-v1` + `-v2` |
| processes | 1 | 1 | **2** | **2** |
| measured requests | 10 | 10 | **20 pooled** | **20 pooled** |
| protocol | 3 + 10 | 3 + 10 | 2 × (3 + 10) | 2 × (3 + 10) |
| model-specific placement gate | `tile_columns == 24576` (`probe.py:41`) | `tile_columns == 24384` (`probe.py:41`) | `softcap_shared_pool`, `fused_gelu_ffn_blocks == 26`, tile 24384 (`probe.py:39`) | `async_irq_wake`, `uncached_prefill` (`probe.py:38`) |
| response oracle | prior report, bit-exact (`:79`) | prior report, bit-exact (`:79`) | worker-a first warm-up, then cross-worker (`:74-75`) | **none** (older probe) |
| `protocol` key in report | yes | yes | yes | **absent** |
| newer exact retime exists | yes, 3+5, `tg-retime-pack-fix` | **no** | yes, 1+3, `tg-minicpm4-retime-candidate` | yes, 1+3, `tg-minicpm4-retime-candidate` |
| current scoreboard verdict | MEASURED (exact retime) | **STALE-Nov4** | MEASURED (retime, last trial) | MEASURED (retime, last trial) |

---

## 7. Source-pin re-check (why "code tree = `~/tinygrad`" is a measurement, not an inference)

Each report stores a `source_before` map of absolute path → sha256. Re-hashing those paths on
2026-11-18:

| report | pins | still match | differ | of the differing, `~/tinygrad` / `~/oNPU` |
| --- | --: | --: | --: | --- |
| `qwen-current-api-20261102-v1/results/qwen3.json` | 321 | 292 | 29 | 22 / 7 |
| `qwen-current-api-20261102-v2/results/qwen25.json` | 321 | 292 | 29 | 22 / 7 |
| `gemma2-current-api-20261101-v3/results/worker-a.json` | 316 | 285 | 31 | 24 / 7 |
| `tinyllama-generic-api-v1/results/tiny-api.json` | 311 | 279 | 32 | 27 / 5 |

Every pinned path begins `/home/orangepi/tinygrad/…` or `/home/orangepi/oNPU/onpu/…`; **no pin
points into `/home/orangepi/tg-decode-all`**. The files that differ today are exactly the
`tinygrad/llm/kernels/rockchip*.py`, `tinygrad/llm/model.py`,
`tinygrad/runtime/support/rockchip_int8.py`, `extra/rockchip/benchmark_inference.py` set plus the
`onpu/{app,worker,library,rockchip_options,runtime,preparation,mla,templates}.py` modules — i.e.
the shared source drifted after 2026-11-01/02. **A byte-identical reproduction of these four rows
is not possible from the current `/home/orangepi/tinygrad`**: 22–27 of the numerical files have
moved.

## 8. Things that could not be determined

| item | status | what was searched |
| --- | --- | --- |
| Qwen3 0.6B checkpoint under `baseline-20261021/` | **does not exist** | `find /home/orangepi/vlm-bench/baseline-20261021 -iname '*qwen3*gguf*'` → only Qwen3.5-0.8B/2B/4B and Qwen3-VL-2B; `ls` of the repo root and `checkpoints/` |
| reproduce path for the **Q8_0** Qwen3-0.6B (`9465e63a…`) | **UNKNOWN / blocked** | `onpu/rockchip_options.py` (absent from `TEXT_INT8_CONTEXTS`, `TEXT_INT8_EXACT_CONTEXTS`, and the `qwen3_prefill_profile` guard); `analysis/release-20261115/r2-lfm-20261116/qwen3-06b-source-audit.md` records the recovered command and concludes the source epoch is unrecoverable |
| exact outer orchestrator command that invoked `build_command.py` for each stage | **not recovered as a single script** | the queue entries `minicpm3-wfe-lease-20261102-v1/queue/003-qwen3.json` and `…/004-qwen25.json` and `qwen2-down-lease-v1/queue/009-tiny-api.json` name the stage + slug; `build_command.py` in each stage returns `argv`/`env`; the driver that consumes that return value was not located by name |
| a tree that reproduces the four rows byte-identically | **does not exist on this box** | `/home/orangepi/tg-decode-all` has no `.git`; `/home/orangepi/tinygrad` has drifted 22–27 files; `tg-retime-pack-fix` and `tg-minicpm4-retime-candidate` are at `1107c67b` (Nov 14/16), later than the Nov 1/2 epochs |
