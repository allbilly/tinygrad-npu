# Reproduce paths — Qwen3.5 4B / 2B / 0.8B, Qwen3-VL 2B

Read-only forensic trace of the four models as measured in `~/oNPU`. Nothing was run, no
NPU was touched, no checkpoint was loaded. Every value below is cited to a `file:line`
or to a directory listing.

**Headline:** the four do **not** share one reproduce path. Counting the published numbers
plus the 0.8B fallback evidence, they span **three code trees**
(`tg-qwen35-pack-signature-fix`, `tg-commit`, `tinygrad`), **five runner scripts**, and
**three protocols** (1+3, 3+3, 3+10). The tree that `retime_with_profile.py` and
`retime_model.sh` default to (`/home/orangepi/tg-decode-all`) is **not** the tree any of
these four numbers came from.

---

## 0. How the numbers were located

The published values are in `~/oNPU/PUBLISH.md:12-19` and `:32-34`, and in the scoreboard
at `~/oNPU/analysis/scoreboard-20261113/SCOREBOARD.md:79` and `:81-83`. Each published row
links to a dated artifact directory. The artifact directory's `receipt.json` records the
`command` array verbatim, plus `backend.path`, `backend.revision`, `runner.path`,
`runner.sha256`, `checkpoint`, `input`, `protocol`, and the gate snapshots. Those receipts
are the authority used here — not the prose.

Two independent confirmations that the artifacts are the real ones:

- `sha256sum` of each runner script matches the `runner.sha256` in its receipt
  (`run_candidate_warmed_retime.py` = `a87fc158…`, `run_candidate_qwen35_warm_retime.py`
  = `f6191aa9…`, `run_candidate_qwen3vl_timing.py` = `ea19fbd6…`,
  `candidate_timing_child.py` = `b2e556b4…`).
- `git -C /home/orangepi/tg-qwen35-pack-signature-fix diff --binary HEAD | sha256sum`
  = `7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77`, which equals
  `backend.dirty_diff_sha256` in both Qwen3.5 receipts.

---

## 1. Qwen3.5 4B — the only one of the four with a full published retime

| field | value |
| --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf` (4,482,403,488 B) |
| **checksum** | `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |
| **code tree** | `/home/orangepi/tg-qwen35-pack-signature-fix` — branch `codex/qwen35-pack-signature-fix`, rev `1b2a5db332e1198b3ace8c7559a19c7d73717dfb`, **4 dirty files**, diff sha `7f87f221…` |
| **interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner** | `/home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116/run_candidate_warmed_retime.py --model qwen35-4b --out <fresh-dir>` (sha256 `a87fc158…`) |
| **entry point** | `extra.rockchip.benchmark_inference`, driven by `candidate_timing_child.py` (imports it and calls `benchmark_inference.main()`) |
| **recipe** | `rockchip_options.py:379-402` Qwen3.5 fast branch, `large=True` |
| **gates** | see §1.3 |
| **artifact** | `…/r2-lfm-20261116/candidate-retime-20261117-qwen35-4b/benchmark.json` |
| **protocol** | **1 warm-up + 3 measured**, 128 prompt / 64 gen, ctx 512, `taskset -c 4-7` |

Published value: decode **6.3555 tok/s** median, TTFT proxy **9335.780 ms**
(`SCOREBOARD.md:83`; `PUBLISH.md:18-19`). `timing_validity=PASS`,
`execution_status=COMPLETE`.

### 1.1 Provenance chain

- Checkpoint manifest: `~/oNPU/checkpoints/manifests/Qwen3.5-4B.json:8,14` — `expected_sha256`
  and `sha256` both `10cc391b…`, `expected_bytes` 4482403488, `complete: true`.
- Checkpoint on disk: `ls -la …/checkpoints/Qwen3.5-4B/` shows
  `Qwen3.5-4B-Q8_0.gguf` at exactly 4,482,403,488 B.
- Input fixture: `…/main-task-20261023/staged/w8a8-sweep-v1/inputs/qwen35-4b.txt`,
  565 B, 128 whitespace tokens, sha256 `921df69467179a5ca2fc57e0a94544e29c774882ff1ced16eeb18c2ff10fd201`.
- Runner constant: `run_candidate_warmed_retime.py:20-24` pins `BACKEND`, `BACKEND_REVISION`,
  `BACKEND_BRANCH`, `BACKEND_DIFF_SHA256`, `BACKEND_DIRTY_PATHS`.
- Runner model table: `run_candidate_warmed_retime.py:61-71` — `qwen35-4b` with the
  checksum, input, `profile_name: "qwen35-4b-context512"`, and
  `projection_options: {recurrent_scan, swiglu, recurrent_fast}`.
- App profile pin: `run_candidate_warmed_retime.py:30-31` — `APP_REVISION`
  `5229660b…`, `PROFILE_SHA256` `3394f6df…`. Receipt `app_profile.sha256` = `3394f6df…`,
  which is the current `~/oNPU/onpu/rockchip_options.py`.

### 1.2 Exact command (from `receipt.json.command`, 54 argv elements)

```
/usr/bin/taskset -c 4-7
  /home/orangepi/tinygrad/.venv/bin/python
  …/r2-lfm-20261116/candidate_timing_child.py
  --checkpoint …/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
  --input      …/inputs/qwen35-4b.txt
  --device CPU --precision int8 --prefill on --decode on
  --context 512 --tokens 64 --warmups 1 --runs 3
  --chunk-size 128 --tile-columns 24384
  --fuse-projections --cpu-attention --native-replay --fused-ffn --split-down
  --fused-qkv --cpu-norm --cpu-fast-softmax --cache-cleaning --wake-helper
  --cpu-pool --cpu-prefill-pool --share-projection-weights --cpu-recurrent-scan
  --fused-swiglu-ffn --fused-swiglu-decode --fused-qkv-without-norm
  --fused-recurrent-projections --cpu-q8-embedding --scalar-decode-loop
  --trim-preparation-cache --file-backed-cpu-weights --defer-weight-realization
  --report <out>/benchmark.json
```

`cwd` = `/home/orangepi/tg-qwen35-pack-signature-fix`; `PYTHONPATH` = same tree
(`run_candidate_warmed_retime.py:303`).

### 1.3 Gates enforced before and after the run

Start gates (`run_candidate_warmed_retime.py:370-394`):

| gate | required |
| --- | --- |
| start temperature | **52 000–55 000 m°C** (52–55 °C), sampled every 2 s until in band |
| NPU min_freq | `1_000_000_000` |
| NPU cur_freq | `1_000_000_000` |
| DDR cur_freq | `2_112_000_000` |
| policy4 / policy6 governor | `performance` |
| policy4 / policy6 cur_khz | `2_256_000` |
| `~/oNPU/.hw-lock` | must not exist |
| DRM owner (`fuser /dev/dri/renderD128 renderD129`) | must be clear |
| `MemAvailable` | ≥ 8 GiB |

Pre-run identity gates (`run_candidate_warmed_retime.py:169-196`, `245-263`): interpreter is
`~/tinygrad/.venv/bin/python`; Python ≥ 3.11; backend branch + revision + dirty-path set +
diff sha all match the pinned constants; `rockchip_options.py` sha matches `3394f6df…`;
oNPU git HEAD matches `5229660b…`; imported module path resolves to that file; checkpoint
sha and input sha match; model key ∈ `WARM_MODEL_KEYS` (`:47`).

Post-run gates (`run_candidate_warmed_retime.py:524-690`): exactly 1+3 trials;
`checkpoint_sha256` and the 308-entry `source_sha256` map match; `context == 512` and
`input_ids` equal the fixture; every trial is exactly 128 prompt / 64 generated; measured
trials produce **identical** token IDs; every `configure_options` value is observed in the
report; `native_execution.prefill.runs ≥ 1` and `.decode.runs ≥ 1`; both `PROGRAM:ROCKCHIP`
and `PROGRAM:CPU` calls present in both phases; per-trial temperature begin/end present;
per-trial `clocks_after` has DDR 2112000000 and CPU4/6 2256000 on every measured trial.

### 1.4 The recipe, in full

`rockchip_options.py:379-402`, with `large = checksum == QWEN35_4B_CHECKPOINT` (`:381`):

```python
{"prefill": "on", "mapped_cpu": True, "share_projection_weights": True,
 "cpu_recurrent_scan": True, "prefill_chunk": 128,
 "tile_columns": 24384,                 # 24384 because large
 "gdn_fast": True, "cpu_q8_embedding": True, "scalar_decode_loop": True,
 "trim_preparation_cache": True,        # True because large
 "preparation_programs": 4,             # 4 because large
 "decode_preparation_programs": 5,
 "prefill_profile": "qwen35-4b-context512",
 "environment": {
   "ONPU_SWIGLU_PREFILL_ROWS": "4", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1",
   "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1", "ONPU_SHARED_HOST_POOL": "1",
   "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_PREFILL_LAYOUT": "1",
   "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
   "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_REPLAY_NATIVE_FROM_LARGE": "1",
   "ONPU_GDN_DECODE_SPIN": "1", "ONPU_GDN_PREP_POOL": "1",
   "ONPU_FFN_PACKED_UPLOAD": "1",
   "ONPU_EXP2_FAST_PATH": "1",          # "1" because large
   "ONPU_CPU_DECODE_POOL": "0", "ONPU_HOST_POOL_WFE": "1",
   "ONPU_GDN_DECODE_PREP_POOL": "1", "ONPU_GDN_DIRECT_DECODE": "1",
   "ONPU_CPU_GQA_PREFILL": "0", "ONPU_INT8_SDK_PREFILL": "0", "ONPU_INT8_EXPERT_DEPTHS": "0",
   "ONPU_INT8_DECODE_NARROW_CBUF": "1", "ONPU_REPLAY_SCALAR_COPY": "1",
   "ONPU_REPLAY_DEDUP_GUARDS": "1", "ONPU_INT8_NATURAL2304": "1",
   "ONPU_NPU_DOMAINS": "3",             # "3" because large
   "ONPU_NPU_DOMAIN_BYTES": "3221225472"}}
```

Effective environment actually recorded in the receipt is the above plus
`ONPU_BENCHMARK_CHECKPOINT_SHA256`, `DEV=CPU HALF=1 JIT=1 CAPTURING=1 PARALLEL=0
DEBUG=0 PROFILE=0 VIZ=0`, `PYTHONPATH=/home/orangepi/tg-qwen35-pack-signature-fix`,
`PYTHONDONTWRITEBYTECODE=1`, and the two observer hooks
`ONPU_TIMING_READY_GATE_FIFO` / `ONPU_TIMING_TEMPERATURES_JSONL`
(`run_candidate_warmed_retime.py:294-305`). One extra invariant is enforced at `:306-307`:
`ONPU_CPU_DECODE_POOL=1` requires `ONPU_HOST_POOL_WFE=1`, else the run is blocked.

---

## 2. Qwen3.5 2B — same tree, same runner, one profile difference

| field | value |
| --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-2B/Qwen3.5-2B-Q8_0.gguf` (2,012,012,800 B) |
| **checksum** | `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1` |
| **code tree** | `/home/orangepi/tg-qwen35-pack-signature-fix` — identical branch/rev/dirty-diff to the 4B |
| **interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner** | same script: `run_candidate_warmed_retime.py --model qwen35-2b --out <fresh-dir>` |
| **entry point** | `extra.rockchip.benchmark_inference` via `candidate_timing_child.py` |
| **recipe** | `rockchip_options.py:379-402`, `large=False` |
| **gates** | identical to §1.3 (same code path) |
| **artifact** | `…/candidate-retime-20261117-qwen35-2b/benchmark.json` |
| **protocol** | **1 warm-up + 3 measured**, 128 / 64, ctx 512, `taskset -c 4-7` |

Published value: decode **14.3259 tok/s** median, TTFT proxy **3572.561 ms**
(`SCOREBOARD.md:82`; `PUBLISH.md:16-17`). `timing_validity=PASS`.

Differences from the 4B, all traceable to `large=False`:

| item | 4B | 2B |
| --- | --- | --- |
| `--tile-columns` | 24384 | **24576** |
| `--trim-preparation-cache` | present | **absent** |
| `preparation_programs` | 4 | **6** |
| `ONPU_EXP2_FAST_PATH` | `1` | **`0`** |
| `ONPU_NPU_DOMAINS` | `3` | **`1`** |

Manifest: `~/oNPU/checkpoints/manifests/Qwen3.5-2B.json:8,14`.
Input: `…/inputs/qwen35-2b.txt`, sha256 `921df694…` — **byte-identical to the 4B and 0.8B
fixtures** (verified with `sha256sum`; all three are 565 B / 128 tokens).

---

## 3. Qwen3.5 0.8B — **no qualified timing exists**

This is the one row of the four with no reproducible measured number at the current
protocol. Three separate artifacts, and only the first is a "published" value.

| field | value |
| --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-0.8B/Qwen3.5-0.8B-Q8_0.gguf` (811,843,840 B) |
| **checksum** | `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c` |
| **code tree** | `/home/orangepi/tg-qwen35-pack-signature-fix` (same rev `1b2a5db`, same dirty diff `7f87f221…`) |
| **interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner** | see §3.1 — three R2 scripts plus the Nov-4 `probe.py`; none produced a valid R2 number |
| **entry point** | `extra.rockchip.benchmark_inference` via `candidate_timing_child.py` |
| **recipe** | `rockchip_options.py:379-402`, `large=False` (same as 2B) |
| **gates** | see §3.2 |
| **artifact** | see §3.1 |
| **protocol** | see §3.1 |

### 3.1 The three artifacts

**(a) The published scoreboard row is Nov-4, not R2.** `SCOREBOARD.md:81` marks the row
`STALE-Nov4` with **427.96 ms / 30.94 tok/s**. `PUBLISH.md:15-16` states plainly: *"The
0.8B warmed retime was blocked before warm-up when its 52–55 °C gate timed out (maximum
51.769 °C; zero in-band samples), so it produced no timing result."*

That Nov-4 number comes from the app-API path, not from `benchmark_inference`:

- Registry: `~/oNPU/docs/benchmark-qwen-current-api-20261102.json`, row `Qwen3.5/0.8B`,
  `summary.ttft_ms = 427.963518013712`, `summary.decode_tps = 30.93687676788589`,
  `report_path = …/qwen-current-api-20261102-v2/results/gdn08.json`,
  `report_sha256 = 50769c0f2ebd9137d90d645ac930e0c8f71fdf490f9b73b09177f2ba45ded54e`.
- Raw report on disk: `…/main-task-20261023/qwen-current-api-20261102-v2/results/gdn08.json`.
- Harness: `…/main-task-20261023/staged/qwen35-small-direct-api-20261101-v1/probe.py`,
  launched by that directory's `build_command.py:12` as
  `taskset -c 0-3 /home/orangepi/oNPU/.venv/bin/python probe.py <report>`.
- App launched by `probe.py` as
  `taskset -c 0-7 /home/orangepi/oNPU/.venv/bin/python -m onpu --host 127.0.0.1 --port 8092 --backend /home/orangepi/tinygrad --data-dir <run>/data`.
- Load call: `{'profile':'ROCKCHIP','context':512,'realize':True,'npu_prefill':'auto','npu_decode':'auto','vision_gpu':False}`.
- Protocol: `gdn08.json:protocol` = 3 warm-ups + **10 measured**, `fixed_idle_seconds 0.2`.
- Backend tree for this Nov-4 run is **`/home/orangepi/tinygrad`** — the 321-entry
  `source_before` map in `gdn08.json` is 305 paths under `/home/orangepi/tinygrad/` plus
  16 app modules under `/home/orangepi/oNPU/onpu/`.
- Gates in `probe.py`: checkpoint sha256 assert; `source_before == source_sha256` pin set;
  external GPU on port 8091 must be `ready` with `selected_profile == 'CL'` and must be
  preserved unchanged; `MemAvailable ≥ 2 GiB`; exactly one `onpu.worker` for this
  checkpoint with affinity `[4,5,6,7]` and `PARALLEL=8` in its environ;
  `placement.cpu_gdn_direct_decode is True`; `scalar_decode_loop.ready`; ≥ 6 preparation
  programs and ≥ 5 decode-phase ones; `placement.prefill_profile == 'qwen35-0.8b-context512'`;
  `cpu_q8_embedding.fast_traces > 0`.

**(b) Two R2 cold observations, both `timing_validity=FAIL`.** Runner
`run_candidate_qwen35_08b_timing.py` (sha256 `7956e67a…`), protocol `WARMUPS, RUNS = 0, 1`
(`:50`):

| dir | TTFT ms | decode tok/s | receipt status |
| --- | --: | --: | --- |
| `candidate-quick-20261116-qwen35-08b/`           | 12351.599 | 4.9463 | `FAIL` / `FAILED` |
| `candidate-quick-coldstart-20261116-qwen35-08b/` | 10364.855 | 5.2565 | `FAIL` / `FAILED` |

Both receipts carry `error: "ValueError: Qwen3.5 prefill native replay did not execute"`
with `benchmark_returncode: 0` — the benchmark completed; the *path* gate failed. Both
started above the band (`ready_temperature_millic` 62846 and 61923).
`TIMING_INVENTORY.md:28` records them as *"COLD-ONLY EVIDENCE; WARMED RETIME BLOCKED
BEFORE WARM-UP"*.

**(c) The warmed 1+3 retime was BLOCKED before warm-up.** Runner
`run_candidate_qwen35_warm_retime.py` (sha256 `f6191aa9…`), `WARMUPS, RUNS = 1, 3` (`:50`).
Receipt `…/candidate-retime-20261117-qwen35-08b/receipt.json`:
`execution_status = "BLOCKED"`, `timing_validity = "NOT_RUN"`, and

```
ready_temperature_gate: {accepted_band_millic: [52000, 55000], passed: false,
  sample_count: 1439, wait_seconds: 7202.559, last_sample: {temperature_millic: 47153}}
```

`benchmark.json` in that directory has `warmups: 1, runs: 3, trials: 0` — the file exists
but holds no measurement. This is a **hardware-availability** block, not a code defect:
the board would not cool into 52–55 °C.

### 3.2 The recipe is the 2B recipe

`rockchip_options.py:379-402` with `large=False`, so `tile_columns 24576`,
`ONPU_NPU_DOMAINS "1"`, `ONPU_EXP2_FAST_PATH "0"`, `preparation_programs 6`,
no `trim_preparation_cache`, `prefill_profile "qwen35-0.8b-context512"`
(`rockchip_options.py:30`). Identical to §2's dict.

### 3.3 The current recommended path

`~/oNPU/handoff.md:36` names the next action: run
`run_candidate_qwen35_warm_retime_stabilized.py` with `/home/orangepi/tinygrad/.venv/bin/python`.
`handoff.md:42` says the four `*_stabilized.py` runners were made by changing
`WARMUPS = 1` to `WARMUPS = 3`. Verified by diff — the **only** difference between
`run_candidate_qwen35_warm_retime.py` and `…_stabilized.py` is line 50:

```
< WARMUPS, RUNS = 1, 3
> WARMUPS, RUNS = 3, 3
```

So the 0.8B path is a **3+3** protocol, and no 0.8B number under either protocol exists
yet. Invocation: `…/run_candidate_qwen35_warm_retime_stabilized.py --out <fresh-dir>`
(sha256 `d674c565…`).

---

## 4. Qwen3-VL 2B — different tree, different runner, same 1+3 protocol

| field | value |
| --- | --- |
| **checkpoint path** | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3-VL-2B/Qwen3VL-2B-Instruct-Q8_0.gguf` (1,834,427,424 B) |
| **checksum** | `1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2` |
| **code tree** | `/home/orangepi/tg-commit` — branch `codex/decode-parity-20261110`, rev `1b2a5db332e1198b3ace8c7559a19c7d73717dfb`, **clean** (empty diff sha `e3b0c442…`) |
| **interpreter** | `/home/orangepi/tinygrad/.venv/bin/python` |
| **runner** | `…/r2-lfm-20261116/run_candidate_qwen3vl_timing.py --out <fresh-dir>` (sha256 `ea19fbd6…`) |
| **entry point** | `extra.rockchip.benchmark_inference` via `candidate_timing_child.py` |
| **recipe** | `rockchip_options.py:409` + `:413-492` dense branch, Qwen3-VL overlay at `:441-445` |
| **gates** | see §4.2 |
| **artifact** | `…/candidate-retime-20261117-qwen3vl-textonly/benchmark.json` |
| **protocol** | **1 warm-up + 3 measured**, 128 / 64, ctx 512, `taskset -c 4-7`, text-only workload |

Published value: decode **16.2546 tok/s** median, TTFT proxy **2879.698 ms**
(`SCOREBOARD.md:79`; `PUBLISH.md:32-34`). `timing_validity=PASS`.

### 4.1 Exact command (from `receipt.json.command`, 48 argv elements)

```
/usr/bin/taskset -c 4-7
  /home/orangepi/tinygrad/.venv/bin/python
  …/r2-lfm-20261116/candidate_timing_child.py
  --checkpoint …/checkpoints/Qwen3-VL-2B/Qwen3VL-2B-Instruct-Q8_0.gguf
  --input      …/inputs/qwen3-vl-2b.txt
  --device CPU --precision int8 --prefill on --decode on
  --context 512 --tokens 64 --warmups 1 --runs 3
  --fuse-projections --cpu-attention --native-replay --fused-qkv --cpu-norm
  --split-down --cpu-fast-softmax --cache-cleaning --wake-helper --cpu-pool
  --cpu-prefill-pool --fused-ffn --scalar-decode-loop --cpu-q8-embedding
  --chunk-size 128 --tile-columns 24576
  --file-backed-cpu-weights --defer-weight-realization --fused-swiglu-ffn
  --report <out>/benchmark.json
```

Note there is **no** `--share-projection-weights`, **no** `--cpu-recurrent-scan`, **no**
`--fused-swiglu-decode`, **no** `--fused-qkv-without-norm`, **no**
`--fused-recurrent-projections`, and **no** `--trim-preparation-cache` — Qwen3-VL is a
dense model, not a GDN/recurrent one. `--fused-swiglu-ffn` is present but
`fused_swiglu_decode_blocks = 0` in the report.

Input: `…/inputs/qwen3-vl-2b.txt`, sha256
`ea0368b5a22fb71bf8f6be738fcc7f50381f6d57529c4dfcd14c7d1ed21a0efa` — **different** from
the three Qwen3.5 fixtures.

### 4.2 Gates

Start gates come from the shared guard `run_candidate_chatglm3_timing.py`
(sha256 `f79a2a20…`, pinned at `run_candidate_qwen3vl_timing.py:48-49`), function
`validate_startup` at `:568-582`:

| gate | required |
| --- | --- |
| NPU min_freq | `1_000_000_000` |
| DDR cur_freq | `2_112_000_000` |
| cpu4 / cpu6 | `performance` **and** `2_256_000` kHz |
| start temperature | **≤ 60 000 m°C** — *not* the 52–55 °C band |
| `MemAvailable` | ≥ **12 GiB** (not 8) |
| `~/oNPU/.hw-lock` | must not exist |

Plus `validate_ownership` (`:557-565`): no competing process matching
`benchmark_inference.py`, `candidate_timing_child.py`, `run_candidate_`,
`run_soak_coldstart.py`, `run_rkllm_bench.py`, `pytest`, `py_compile`, `ruff check`,
`clang`, `gcc`, `cc1`, `ninja`, `cmake`, `make`; no unreadable `/proc` entries; every
`/dev/dri/*` node must be owner-free.

Model-specific gates in `run_candidate_qwen3vl_timing.py`: manifest sha
`40e55bef…` (`:29-30`); `QWEN3VL_CHECKPOINT` constant must equal the pinned GGUF (`:164`);
`EXPECTED_PROFILE_ENV` — 24 env keys must match exactly (`:68-93`); `EXPECTED_PLACEMENT`
(`:95-106`) including `tile_columns 24576`, `cpu_attention_blocks 28`,
`prepared_qkv_blocks 28`, `fused_ffn_blocks 28`, `fused_swiglu_ffn_blocks 28`,
`share_projection_weights False`; `cli_help_sha256` recorded and re-checked;
`ATTENTION_BLOCKS = 28` (`:51`).

### 4.3 The recipe, in full

`rockchip_options.py:403-410` selects the dense row
`QWEN3VL_CHECKPOINT: ("qwen3vl-2b-text-context512", 4, 24576)`; the generic dense
environment is at `:416-423`; the Qwen3-VL overlay at `:441-445`; the result dict at
`:446-447`; the post-hoc `scalar_decode_loop` / WFE block at `:479-483`; the
`cpu_q8_embedding` block at `:490-491`. Assembled:

```python
{"prefill": "on", "mapped_cpu": True,
 "share_projection_weights": False,     # dense row default
 "prefill_chunk": 128, "tile_columns": 24576,
 "dense_swiglu": True,
 "prefill_profile": "qwen3vl-2b-text-context512",
 "scalar_decode_loop": True,            # :480
 "cpu_q8_embedding": True,              # :491
 "environment": {
   # base dense, :416-423
   "ONPU_SWIGLU_PREFILL_ROWS": "4", "ONPU_INT8_EXPERT_DEPTHS": "0",
   "ONPU_INT8_NATURAL2304": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
   "ONPU_NPU_DOMAINS": "1", "ONPU_NPU_DOMAIN_BYTES": "3221225472",
   "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1",
   "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
   "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_SHARED_HOST_POOL": "1",
   "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_PREFILL_LAYOUT": "1",
   # Qwen3-VL overlay, :441-445
   "ONPU_REPLAY_NATIVE_FROM_LARGE": "1", "ONPU_CPU_DECODE_POOL": "1",
   "ONPU_CPU_GQA_PREFILL": "1", "ONPU_EXP2_FAST_PATH": "1",
   "ONPU_EXP2_SCALE_BITS": "1", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1",
   "ONPU_FFN_PACKED_UPLOAD": "1", "ONPU_INT8_SDK_PREFILL": "0",
   "ONPU_HOST_POOL_WFE": "0",
   # :481-483
   "ONPU_DECODE_HOST_WFE": "1"}}
```

This is verbatim what `receipt.json.profile` contains.

---

## 5. What the four have in common, and what differs

### 5.1 In common

| dimension | shared value | evidence |
| --- | --- | --- |
| checkpoint root | `/home/orangepi/vlm-bench/baseline-20261021/checkpoints/<Model>/` | all four manifests |
| format | `Q8_0` GGUF, `--precision int8` | all four `command` arrays |
| device flag | `--device CPU` (CPU host + NPU GEMM) | all four `command` arrays |
| workload | 128 prompt / 64 generated, ctx 512 | all four `protocol` blocks |
| affinity | `taskset -c 4-7` | all four |
| interpreter | `/home/orangepi/tinygrad/.venv/bin/python` | all four `command[3]` |
| child wrapper | `candidate_timing_child.py`, sha `b2e556b4…` | all four `command[4]` |
| entry point | `extra.rockchip.benchmark_inference` | child does `from extra.rockchip import benchmark_inference` |
| clock gate | NPU 1 GHz, DDR 2112 MHz, CPU4/6 `performance` @ 2256000 | all four |
| source pinning | 308-file `source_sha256` map re-checked post-run | all four `benchmark.json` |
| sampling | `temperature=0; raw IDs; no template; EOS ignored; empty KV history each request` | all four `benchmark.json` |
| clock bucket | every measured trial finished at NPU `1000000000` | all four `benchmark.json` `trials[].clocks_after` |
| replay (three retimes only) | native prefill runs 1, native decode runs 249 | the three `candidate-retime-20261117-*` receipts |
| replay (0.8B cold only) | **prefill replay absent**, decode runs 60 | the two `candidate-quick*` `benchmark.json` |

### 5.2 What differs

| dimension | 4B | 2B | 0.8B | Qwen3-VL 2B |
| --- | --- | --- | --- | --- |
| **tree** | `tg-qwen35-pack-signature-fix` | `tg-qwen35-pack-signature-fix` | `tg-qwen35-pack-signature-fix` | **`tg-commit`** |
| tree dirty | 4 files, diff `7f87f221…` | same | same | **clean** |
| **runner** | `run_candidate_warmed_retime.py --model qwen35-4b` | same, `--model qwen35-2b` | `run_candidate_qwen35_warm_retime.py` (blocked) / `…_stabilized.py` (next) | **`run_candidate_qwen3vl_timing.py`** |
| **protocol** | 1+3 | 1+3 | 1+3 blocked → **3+3** next | 1+3 |
| **start temp gate** | 52–55 °C band | 52–55 °C band | 52–55 °C band | **≤ 60 °C only** |
| **mem gate** | 8 GiB | 8 GiB | 8 GiB | **12 GiB** |
| **ready-FIFO gate** | yes (`ready_gate.fifo`) | yes | yes on the warmed attempt; **no** on the cold ones | **no** |
| **tile-columns** | **24384** | 24576 | 24576 | 24576 |
| `--trim-preparation-cache` | yes | no | no | no |
| `--cpu-recurrent-scan` | yes | yes | yes | **no** |
| `--fused-recurrent-projections` | yes | yes | yes | **no** |
| `--fused-swiglu-decode` | yes | yes | yes | **no** |
| `--fused-qkv-without-norm` | yes | yes | yes | **no** |
| `--share-projection-weights` | yes | yes | yes | **no** |
| `ONPU_NPU_DOMAINS` | **3** | 1 | 1 | 1 |
| `ONPU_EXP2_FAST_PATH` | **1** | 0 | 0 | 1 |
| `ONPU_HOST_POOL_WFE` | 1 | 1 | 1 | **0** |
| `ONPU_DECODE_HOST_WFE` | — | — | — | **1** |
| profile name | `qwen35-4b-context512` | `qwen35-2b-context512` | `qwen35-0.8b-context512` | `qwen3vl-2b-text-context512` |
| **published number exists** | **yes**, 6.3555 tok/s | **yes**, 14.3259 tok/s | **no** — row is Nov-4 | **yes**, 16.2546 tok/s |
| measured temp range | 69.307–76.692 °C | 64.692–70.230 °C | n/a (blocked) | 68.384–73.923 °C |
| TTFT max/min spread | 15.804× | 11.038× | n/a | 12.597× |

### 5.3 The single most important finding

**`/home/orangepi/tg-decode-all` is not the tree that produced any of these four numbers.**
It is the default `TREE` in `retime_with_profile.py:19` and `retime_model.sh:8`, and it is
a plain directory with **no `.git`** (`git -C /home/orangepi/tg-decode-all rev-parse HEAD`
→ `fatal: not a git repository`), so it cannot even be revision-pinned. The actual trees
are `tg-qwen35-pack-signature-fix` (three models) and `tg-commit` (Qwen3-VL).

Both real trees sit at the same commit `1b2a5db3…`, but `tg-qwen35-pack-signature-fix`
carries a 4-file dirty patch (`extra/rockchip/benchmark_inference.py`,
`test/models/test_cpu_gdn_benchmark.py`, `test/models/test_rockchip_gelu_ffn.py`,
`tinygrad/llm/kernels/rockchip_gelu_ffn.py`) whose diff sha256 `7f87f221…` is the
identifier that matters. A clean checkout of `1b2a5db3…` is **not** the measured
configuration for the three Qwen3.5 models.

---

## 6. Things that look like the reproduce path but are not

Recorded so they are not re-tried.

| candidate | why it is not the path |
| --- | --- |
| `scripts/bench/retime_with_profile.py` | Uses `--warmups 3 --runs 5` (`:53`), default `TREE=/home/orangepi/tg-decode-all` (`:19`), and appends only `--share-projection-weights` / `--scalar-decode-loop` / `--cpu-q8-embedding` (`:104-106`). It **never** passes `--cpu-recurrent-scan`, `--fused-recurrent-projections`, `--fused-qkv-without-norm`, `--fused-swiglu-decode`, or `--trim-preparation-cache`, all of which the measured Qwen3.5 runs used. Different configuration, different protocol. |
| `scripts/bench/retime_model.sh` | Same problem plus a hardcoded MLA/RWKV env block (`:106-116`) that has nothing to do with these models; 3+5 (`:5`, `:119`); `TREE` defaults to `tg-decode-all` (`:8`). |
| `scripts/bench/retime_api_exact.py` | API-level, `--repeats 3` with **no warm-up** (`:235`), and it measures prefill via the server's own `ttft_ms`. Different measurement surface entirely. |
| `scripts/bench/qualify_retime_report.py` | Defaults to `--warmups 3 --runs 5` (`:94-95`). Only usable for a 3+5 report; would reject a 1+3 one. |
| `scripts/bench/arm_reprobe.py` | MiniCPM3/MLA-specific logits probe (`:17-21` hardcodes `tile_columns=24384` and `rockchip_mla_attention.configure`). Unrelated to all four models. |
| `run_candidate_warmed_retime.py --model qwen35-0.8b` | Impossible: `WARM_MODEL_KEYS = {"qwen35-2b", "qwen35-4b", "spark25"}` (`:47`) and `load_config` raises `Blocked` for any other key (`:253-254`). |
| `/home/orangepi/oNPU/.venv/bin/python` | Rejected by three of the four runners, which compare `Path(sys.prefix).resolve()` against `VENV = /home/orangepi/tinygrad/.venv` (`run_candidate_qwen35_08b_timing.py:329-330`, `run_candidate_qwen35_warm_retime.py:373-374`, `run_candidate_qwen3vl_timing.py:276-277`). `run_candidate_warmed_retime.py` only checks `sys.version_info >= (3, 11)` (`:246`) but always spawns the child with its pinned `PYTHON` (`:39`). It has no numpy — see `handoff.md:82-83`. The app-API Nov-4 path *does* use it — see `probe.py` — so the interpreter differs by path, not by model. |

---

## 7. Per-model one-line reproduce answers

| model | the command that reproduces the published number |
| --- | --- |
| Qwen3.5 4B | `cd /home/orangepi/tg-qwen35-pack-signature-fix && /home/orangepi/tinygrad/.venv/bin/python ~/oNPU/analysis/release-20261115/r2-lfm-20261116/run_candidate_warmed_retime.py --model qwen35-4b --out <fresh-dir>` |
| Qwen3.5 2B | same, `--model qwen35-2b` |
| Qwen3.5 0.8B | **no command reproduces a published number.** Nov-4 value needs the app-API harness `…/staged/qwen35-small-direct-api-20261101-v1/probe.py`; the current-protocol attempt is `run_candidate_qwen35_warm_retime_stabilized.py --out <fresh-dir>` (3+3, needs 52–55 °C) and has never completed |
| Qwen3-VL 2B | `cd /home/orangepi/tg-commit && /home/orangepi/tinygrad/.venv/bin/python ~/oNPU/analysis/release-20261115/r2-lfm-20261116/run_candidate_qwen3vl_timing.py --out <fresh-dir>` |

All four require, first: `sudo /home/orangepi/oNPU/scripts/prepare_clocks_rk3588.sh`, no
`~/oNPU/.hw-lock`, no DRM owner, and a board cool enough for that model's gate (52–55 °C
for the three Qwen3.5 models, ≤ 60 °C for Qwen3-VL). Each runner takes an exclusive
`~/oNPU/.hw-lock` for the duration; run one at a time.

---

## 8. Fields that could not be determined

| field | status |
| --- | --- |
| Qwen3.5 0.8B measured number at the current protocol | **UNKNOWN — does not exist.** Searched `analysis/retime-20261113/`, `analysis/retime-20261114/`, all 123 dirs under `analysis/release-20261115/r2-lfm-20261116/`, `docs/benchmark-*.json`, `plan.md`, `PUBLISH.md`, `SCOREBOARD.md`, `TIMING_INVENTORY.md`. The only 0.8B artifacts are the two cold `FAIL` runs and the `BLOCKED` 1+3 attempt. |
| The outer shell command that invoked `run_candidate_warmed_retime.py` | **UNKNOWN.** Receipts record the *child* command and the runner path/hash, but not the parent shell line. The `--out` value is inferable from the artifact directory name; `--start-wait-log` is recorded as `start_wait_log` in the receipt. |
| `tg-decode-all` revision | **UNKNOWN — not a git repo.** No `.git` directory; `PROVENANCE.txt` exists at the top level but was not treated as a revision pin. |
| Whether `run_candidate_qwen35_warm_retime_stabilized.py` has ever completed | **No.** No `stabilized-qwen35*` or equivalent directory exists under `analysis/release-20261115/r2-lfm-20261116/`; only `stabilized-minicpm4-20261117/` exists. |
