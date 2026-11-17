# opencode session database — reference-run command archaeology

Read-only search of `/home/orangepi/.local/share/opencode/opencode.db`
(SQLite, 1.24 GB, opened `mode=ro`). No `sqlite3` CLI on this host, so every
query ran through Python's `sqlite3` module.

Scope: what command actually ran the reference on **Qwen3.5 4B**, which tree and
patch it pinned, whether `tg-decode-all` ever had a `.git`, and which command
produced the recorded 4B numbers.

---

## 1. Schema

The database carries **two generations** of message storage. Both were searched.
The earlier failed attempt guessed a `sessionID` column; the real column is
`session_id`.

### v1 tables (`part` holds the content)

```sql
CREATE TABLE `part` (
	`id` text PRIMARY KEY,
	`message_id` text NOT NULL,
	`session_id` text NOT NULL,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL,
	`data` text NOT NULL,
	CONSTRAINT `fk_part_message_id_message_id_fk`
		FOREIGN KEY (`message_id`) REFERENCES `message`(`id`) ON DELETE CASCADE
)

CREATE TABLE `message` (
	`id` text PRIMARY KEY,
	`session_id` text NOT NULL,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL,
	`data` text NOT NULL
)

CREATE TABLE `session` (
	`id` text PRIMARY KEY,
	`project_id` text NOT NULL,
	`parent_id` text,
	`slug` text NOT NULL,
	`directory` text NOT NULL,
	`title` text NOT NULL,
	`version` text NOT NULL,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL
)
```

`part.data` is JSON with `$.type` in
`tool | step-start | step-finish | reasoning | text | patch | file | compaction | subtask`.
A tool part carries `$.state.input.command`.

### v2 tables (`session_message` holds the content)

```sql
CREATE TABLE `session_v2` (
	`id` text PRIMARY KEY,
	`project_id` text NOT NULL,
	`parent_id` text,
	`fork_session_id` text,
	`fork_boundary` text,
	`slug` text NOT NULL,
	`directory` text NOT NULL,
	`path` text,
	`title` text,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL
)

CREATE TABLE `session_message` (
	`id` text PRIMARY KEY,
	`session_id` text NOT NULL,
	`type` text NOT NULL,          -- assistant | user | synthetic | idle | system | compaction | model-switched
	`seq` integer NOT NULL,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL,
	`data` text NOT NULL
)

CREATE TABLE `session_part` (
	`message_id` text NOT NULL,
	`session_id` text NOT NULL,
	`ordinal` integer NOT NULL,
	`type` text NOT NULL,
	`call_id` text,
	`time_created` integer NOT NULL,
	`time_updated` integer NOT NULL,
	`data` text NOT NULL,
	PRIMARY KEY(`message_id`, `ordinal`)
)
```

**`session_part` is empty — 0 rows.** All v2 content lives in
`session_message.data`, where an assistant message is
`{"agent":…,"model":{…},"content":[{"type":"tool","name":"bash","state":{"input":{"command":"…"}}}]}`.

### Row counts

| table              |    rows |
| ------------------ | ------: |
| `project`          |      22 |
| `message`          |  22,331 |
| `part`             |  93,121 |
| `session`          |     328 |
| `session_v2`       |     540 |
| `session_message`  |  47,951 |
| `session_part`     |  **0** |
| `todo`             |     317 |
| `event`            |  **0** |

### Method

`LIKE '%kw%'` over a 1.2 GB `text` column timed out at 120 s per keyword
(14 keywords, ~14 full scans). Replaced with **one streaming pass** over
`part` and one over `session_message`, testing all keywords per row with a
lowercased substring check:

```python
KWS = ['run_candidate_warmed_retime','tg-qwen35-pack-signature-fix','qwen35-4b',
       'Qwen3.5-4B','6.3555','9335','5229660b','7f87f221','tg-decode-all',
       'tg-commit','1b2a5db','retime_with_profile','benchmark_inference',
       'onpu.worker','bench_llm']
```

Result: **6,382 matching rows**, 5,631 of them tool calls.

| keyword                          | rows |
| -------------------------------- | ---: |
| `benchmark_inference`            | 2181 |
| `tg-decode-all`                  | 2068 |
| `qwen35-4b`                      |  908 |
| `Qwen3.5-4B`                     |  750 |
| `bench_llm`                      |  737 |
| `9335`                           |  519 |
| `tg-commit`                      |  404 |
| `1b2a5db`                        |  386 |
| `onpu.worker`                    |  375 |
| `retime_with_profile`            |  208 |
| `7f87f221`                       |  203 |
| `tg-qwen35-pack-signature-fix`   |  176 |
| `run_candidate_warmed_retime`    |  160 |
| `6.3555`                         |  116 |
| `5229660b`                       |   98 |

---

## 2. The tree and patch, pinned in the runner's own header

This is the decisive finding. `run_candidate_warmed_retime.py` — the runner that
produced the 4B number — **hard-codes its backend**. Read at
`2026-11-17 01:32:22`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`:

```python
19: APP = Path("/home/orangepi/oNPU")
20: BACKEND = Path("/home/orangepi/tg-qwen35-pack-signature-fix")
21: BACKEND_REVISION = "1b2a5db332e1198b3ace8c7559a19c7d73717dfb"
22: BACKEND_BRANCH = "codex/qwen35-pack-signature-fix"
23: BACKEND_DIFF_SHA256 = "7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77"
24: BACKEND_DIRTY_PATHS = {
25:     "extra/rockchip/benchmark_inference.py",
26:     "test/models/test_cpu_gdn_benchmark.py",
27:     "test/models/test_rockchip_gelu_ffn.py",
28:     "tinygrad/llm/kernels/rockchip_gelu_ffn.py",
29: }
30: APP_REVISION = "5229660be83d3ee8e155e31109518fb90d641e80"
31: PROFILE = APP / "onpu/rockchip_options.py"
32: PROFILE_SHA256 = "3394f6dffb81529bb567e2e5982c8ca306ae611b0004e5d703e8b641acbd85b6"
```

Docstring, line 1: `"""Source-pinned, warmed 1+3 retimes for Qwen3.5 2B/4B and Spark-X2.5."""`

So the 4B reference configuration is:

| element | value |
| --- | --- |
| backend tree | `/home/orangepi/tg-qwen35-pack-signature-fix` |
| commit | `1b2a5db332e1198b3ace8c7559a19c7d73717dfb` |
| branch | `codex/qwen35-pack-signature-fix` |
| dirty diff sha256 | `7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77` |
| dirty file count | **4** |
| app tree | `/home/orangepi/oNPU` @ `5229660be83d3ee8e155e31109518fb90d641e80` |
| profile | `onpu/rockchip_options.py`, sha256 `3394f6df…` |

### How that tree was created

`2026-11-16 18:51:26`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`:

```bash
git -C /home/orangepi/tg-commit worktree add -b codex/qwen35-pack-signature-fix \
  /home/orangepi/tg-qwen35-pack-signature-fix 1b2a5db332e1198b3ace8c7559a19c7d73717dfb
```

Output: `Preparing worktree (new branch 'codex/qwen35-pack-signature-fix')`.

**`tg-qwen35-pack-signature-fix` is a git worktree of `tg-commit`.** Its `.git`
is therefore a *file*, not a directory — which is exactly the trap that made an
earlier `[ -d "$t/.git" ]` test report it as "no git".

The four-file patch was hashed at `2026-11-16 19:06:37`, workdir
`/home/orangepi/tg-qwen35-pack-signature-fix`, verbatim:

```bash
git diff --binary HEAD -- extra/rockchip/benchmark_inference.py \
  test/models/test_cpu_gdn_benchmark.py test/models/test_rockchip_gelu_ffn.py \
  tinygrad/llm/kernels/rockchip_gelu_ffn.py | sha256sum
# -> 7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77  -
```

Re-confirmed with `git status --short` alongside it at `2026-11-16 19:15:53`,
which re-printed the same `7f87f221…` digest.

`5229660b` is **not** a backend revision. It is `APP_REVISION` — the `~/oNPU`
harness commit. Confirmed at `2026-11-16 17:40:23` in the same session:

```
ROOT /home/orangepi/oNPU | rev-parse 0 5229660be83d3ee8e155e31109518fb90d641e80 | branch 0 main
```

---

## 3. Every command that ran the reference on Qwen3.5 4B

All in session `ses_e2ecd054effeSPzmuGpHUtCi2X`
(`oNPU benchmark.md update vs rewrite`, dir `/home/orangepi/tinygrad`),
unless noted.

### 3.1 The run that produced `6.3555` — 2026-11-17 03:52:22

```bash
/home/orangepi/oNPU/.venv/bin/python \
  /home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116/run_candidate_warmed_retime.py \
  --model qwen35-4b \
  --out /home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116/candidate-retime-20261117-qwen35-4b
```

Preceded 22 s earlier by its gate:

```bash
/home/orangepi/oNPU/.venv/bin/python .../run_candidate_warmed_retime.py --model qwen35-4b --validate-only
```

### 3.2 Earlier validate-only attempts (same session)

| timestamp | command |
| --- | --- |
| 2026-11-17 01:20:46 | `/home/orangepi/tinygrad/.venv/bin/python .../run_candidate_warmed_retime.py --model qwen35-4b --validate-only` |
| 2026-11-17 01:29:42 | same, `--validate-only` |
| 2026-11-17 03:52:00 | `/home/orangepi/oNPU/.venv/bin/python .../run_candidate_warmed_retime.py --model qwen35-4b --validate-only` |

Note the interpreter changes from `~/tinygrad/.venv` to `~/oNPU/.venv` between
01:29 and 03:52. The run that counts uses **`~/oNPU/.venv`**.

### 3.3 The 2026-11-18 re-run — session `ses_e173ec309ffekqNKkwWJflq7XK`

(`handoff.md plan`, dir `/home/orangepi/pr/tinygrad`)

```bash
# 2026-11-18 04:12:59
cd /home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116 && \
  timeout 300 /home/orangepi/oNPU/.venv/bin/python run_candidate_warmed_retime.py \
  --model qwen35-4b --validate-only 2>&1 | tail -14
```

```bash
# 2026-11-18 04:14:52
cd /home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116 && \
  setsid nohup /home/orangepi/oNPU/.venv/bin/python run_candidate_warmed_retime.py \
  --model qwen35-4b \
  --out /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-warmed-4b \
  --start-wait-log /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-warmed-4b-start.log \
  > /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-warmed-4b.log 2>&1 < /dev/null & disown
```

This is the **current** attempt, writing into this checkout rather than into
`~/oNPU`. Same runner, same pinned backend, different output directory.

### 3.4 A separate, different 4B run on `tg-decode-all` — 2026-11-18 03:56:57

Session `ses_e173ec309ffekqNKkwWJflq7XK`. **This is not the reference retime.**
It invokes `benchmark_inference` directly against `tg-decode-all`:

```bash
mkdir -p /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-retime && \
cd /home/orangepi/tg-decode-all && \
export DEV=CPU HALF=1 JIT=1 CAPTURING=1 PARALLEL=0 \
  PYTHONPATH=/home/orangepi/tg-decode-all \
  ONPU_BENCHMARK_CHECKPOINT_SHA256=10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1 \
  ONPU_SWIGLU_PREFILL_ROWS=4 ONPU_SWIGLU_PACK_STATIC_LAYOUT=1 ONPU_CPU_TOKEN_INPUTS=1 \
  ONPU_ZERO_PAD_VIEWS=1 ONPU_SHARED_HOST_POOL=1 ONPU_PREFILL_CPU_TUNING=1 \
  ONPU_CPU_PREFILL_LAYOUT=1 ONPU_REPLAY_BORROW_OUTPUTS=1 ONPU_REPLAY_BORROW_BATCH=1 \
  ONPU_REPLAY_NATIVE_FROM=1 ONPU_REPLAY_NATIVE_FROM_LARGE=1 ONPU_GDN_DECODE_SPIN=1 \
  ONPU_GDN_PREP_POOL=1 ONPU_FFN_PACKED_UPLOAD=1 ONPU_EXP2_FAST_PATH=1 \
  ONPU_CPU_DECODE_POOL=0 ONPU_HOST_POOL_WFE=1 ONPU_GDN_DECODE_PREP_POOL=1 \
  ONPU_GDN_DIRECT_DECODE=1 ONPU_CPU_GQA_PREFILL=0 ONPU_INT8_SDK_PREFILL=0 \
  ONPU_INT8_EXPERT_DEPTHS=0 ONPU_INT8_DECODE_NARROW_CBUF=1 \
  ONPU_REPLAY_SCALAR_COPY=1 ONPU_REPLAY_DEDUP_GUARDS=1 ONPU_INT8_NATURAL2304=1 \
  ONPU_NPU_DOMAINS=3 ONPU_NPU_DOMAIN_BYTES=3221225472 && \
setsid nohup taskset -c 4-7 /home/orangepi/tinygrad/.venv/bin/python \
  -m extra.rockchip.benchmark_inference \
  --checkpoint /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  --input /home/orangepi/vlm-bench/baseline-20261021/qwen35-4b.tokens \
  --device CPU --precision int8 --prefill on --decode on --context 512 --tokens 64 \
  --warmups 3 --runs 5 --chunk-size 128 --fuse-projections --cpu-attention \
  --native-replay --fused-qkv --cpu-norm --split-down --cpu-fast-softmax \
  --cache-cleaning --wake-helper --cpu-pool --cpu-prefill-pool --fused-ffn \
  --defer-weight-realization --file-backed-cpu-weights --fused-swiglu-ffn \
  --tile-columns 24384 --share-projection-weights --scalar-decode-loop \
  --cpu-q8-embedding \
  --report /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-retime/report.json \
  > /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-retime/run.log 2>&1 < /dev/null & disown
```

Protocol here is **3 warm-ups + 5 measured** — different from the 1+3 of
`run_candidate_warmed_retime.py`. These two runs are not interchangeable and
their numbers must not be compared as if they were the same configuration.

### 3.5 `retime_with_profile.py` on 4B — dry-run only

```bash
# 2026-11-18 03:55:48, ses_e173ec309ffekqNKkwWJflq7XK
cd ~/oNPU && ~/tinygrad/.venv/bin/python scripts/bench/retime_with_profile.py \
  qwen35-4b /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-retime --dry-run 2>&1 | tail -8
```

Only `--dry-run`. No measured 4B run through `retime_with_profile.py` appears
anywhere in the database.

### 3.6 Sibling models through the same runner

```bash
# 2026-11-17, ses_e2ecd054effeSPzmuGpHUtCi2X
/home/orangepi/oNPU/.venv/bin/python .../run_candidate_warmed_retime.py --model qwen35-2b \
  --out .../candidate-retime-20261117-qwen35-2b
/home/orangepi/oNPU/.venv/bin/python .../run_candidate_warmed_retime.py --model spark25 \
  --out .../candidate-retime-20261117-spark25
/home/orangepi/oNPU/.venv/bin/python .../run_candidate_warmed_retime.py --model qwen35-08b \
  --out .../candidate-retime-20261117-qwen35-08b
```

The 0.8B arm is the one that later timed out at the temperature gate.

---

## 4. Recorded measurements of the 4B, with the command that produced them

### 4.1 Raw report fields — `2026-11-17`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`

Verbatim from the tool output of reading
`candidate-retime-20261117-qwen35-4b/benchmark.json`:

```
'warmups': 1, 'runs': 3, 'context': 512, 'device': 'CPU',
'median_decode_tps': 6.355455332644719,
'median_first_token_ms': 9335.779833025299,
'load_seconds': 3.2332767579937354, 'prepare_seconds': 31.7186715460266

'model': 'Qwen3.5 4B',
'native_decode_replay_runs': 249, 'native_prefill_replay_runs': 1,
'median_decode_tps': 6.355455332644719,
'median_ttft_ms': 9335.779833025299,
'prompt_tokens': 128,
'ttft_max_min_ratio': 15.803517292661397,
'ttft_ms': [9335.779833025299, 19132.280238030944, 1210.6343090417795],
'quality_assessed': False, 'timing_validity': 'PASS'

'npu_finish_clock_groups': {'1000000000': {
    'decode_tps': [6.341803676339021, 6.375844819706709, 6.355455332644719],
    'trial_indices': [1, 2, 3],
    'ttft_ms': [9335.779833025299, 19132.280238030944, 1210.6343090417795]}}
```

Per-trial records, same output:

```
(1, False, 128, 64,  9335.779833025299, 6.341803676339021, {...})
(3, False, 128, 64,  1210.6343090417795, 6.355455332644719, {...})
```

Clocks recorded inside each trial: `npu/cur_freq 1000000000`,
`gpu/cur_freq 1000000000`, `dmc/cur_freq 2112000000`,
`policy6/scaling_cur_freq 2256000`, `policy0/scaling_cur_freq 1800000`.

### 4.2 The interpretation written into the session

`2026-11-17`, same session, verbatim:

> Decode median **6.3555 tok/s** (6.3418/6.3758/6.3555); TTFT proxy median
> **9335.780 ms** (9335.780/19132.280/1210.634; max/min **15.804×**).
> Measured-trial temperature boundaries **69.307–76.692 °C**; all measured clocks
> matched NPU 1 GHz, DDR 2112 MHz, CPU4/6 2256 MHz; ready gate passed at
> 53.615 °C after 25.179 s. Pinned backend revision `1b2a5db…`

> The source-pinned Qwen3.5 4B 1+3 retime also completed with native
> prefill/decode and matching measured IDs: median decode **6.3555 tok/s**,
> TTFT-proxy median **9,335.780 ms**, max/min **15.804×**; measured temperatures
> were 69.307–76.692 °C and clocks matched. Its TTFT remains unresolved and
> quality was not assessed, so the full official row stays stale.

> The 4B warmed retime measured 6.3555 tok/s decode, but its TTFT-proxy spread is
> 15.804×, so its full row also remains stale.

### 4.3 What this number is and is not

- **Decode 6.3555 tok/s is valid** — `timing_validity=PASS`, all three measured
  trials on the same NPU clock group, measured IDs identical across trials.
- **TTFT is not usable as a prefill figure.** The three measured TTFTs are
  9335.780 / 19132.280 / 1210.634 ms — a 15.804× spread. The 1210.634 ms trial is
  the stabilized one; the median is a first-trial artifact. The session's own
  conclusion was to keep the full row stale for exactly this reason.
- Protocol is **1 warm-up + 3 measured**, 128 prompt tokens / 64 decode tokens.
  Not 3+10, not 3+3.

### 4.4 `1210` in context

`1210.6343090417795` is the third measured trial's TTFT, the stabilized value.
It appears only in the 4B report above. All other `1210` hits in the database are
unrelated line numbers in `sed`/`grep` output (e.g. `ses_e55d6a9c1ffe…`,
`ses_1fff102cfffe…`) and are not measurements.

---

## 5. `tg-decode-all` and its `.git`

### 5.1 Direct evidence it has no `.git`

`2026-11-16 18:09:47`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`, verbatim tool
output from a multi-tree revision probe:

```
1b2a5db332e1198b3ace8c7559a19c7d73717dfb | codex/decode-parity-20261110
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
5229660be83d3ee8e155e31109518fb90d641e80
```

Three trees reported `fatal: not a git repository` while `tg-commit`
(`1b2a5db`, branch `codex/decode-parity-20261110`) and `~/oNPU`
(`5229660b`) resolved. `tg-decode-all` is one of the three.

`2026-11-18 03:50:50`, session `ses_e14f016cfffdvwVPWHVufAu4xX`
(`Locate reproduce paths B`), verbatim reasoning:

> `/home/orangepi/tg-decode-all` DOES exist — my earlier test `[ -d "$t/.git" ]`
> failed, meaning it has no `.git` directory (it's a plain directory, not a git
> repo, or a worktree with `.git` file). Let me check.
>
> Also `tg-minicpm4-retime-candidate` and `tg-retime-pack-fix` exist but no
> `.git` dir. Let me check for `.git` file (worktrees use a `.git` file).

And the conclusion recorded in the sibling session:

> **`tg-decode-all` — the default `TREE` in `retime_with_profile.py:19` and
> `retime_model.sh:8` — is not the tree any of these four numbers came from.**
> It has no `.git` at all (`fatal: not a git repository`), so it cannot even be
> revision-pinned.

### 5.2 Was it removed, or never present?

**No `rm -rf .git` command exists anywhere in the database.** The query

```python
re.compile(r'rm -rf[^\n]{0,40}\.git')
```
over every `part` and `session_message` row returned **only** the text of the
task prompts that *asked* for this search (sessions
`ses_e14e542e4ffetk99k46PmZeJ4z`, `ses_e14e542e4ffd61lkPpbC2pBygD`). No
executed command, and no report of one, matches.

The positive evidence points to **never present**:

`2026-11-17 17:05`, session `ses_e16921d8fffeZKmwM90fg0gBif`
(`Splitting the current git diff into commits`, dir `~/oNPU`), verbatim:

> `git init`-ing three 3 GB trees and committing their diffs is a large,
> novelty-free operation

> The cheapest insurance: `git init` + commit in each tree, then push to a
> private repo or as branches.

> for t in tg-exp-attnnpu tg-exp-h5 tg-decode-all tg-exp-h1 tg-exp-h4x; do
>   git checkout --orphan "$t"
>   rsync -a --delete --exclude=.git --exclude=.venv --exclude=__pycache__ …

A tree that already had a `.git` would not be proposed for `git init`, and the
rsync would not need `--exclude=.git`.

**Caveat, and it matters:** the user has stated that `tg-decode-all`'s `.git`
may have been removed by hand (`ses_e173ec309ffekqNKkwWJflq7XK`, 2026-11-18):

> "has no .git at all, maybe just because i removed it" — the `tg-decode-all`
> tree may have had .git removed by the user, so "no .git" doesn't mean it's not
> the measured tree.

The database contains **no command that removed it**. That is an absence of
evidence, not evidence of absence — the removal, if it happened, was done
outside a recorded session.

### 5.3 The `[ -d "$t/.git" ]` trap

`2026-11-18`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`, correction recorded
verbatim:

> Correction: the source trees weren't absent. My first check incorrectly
> required `.git` to be a directory; Git worktrees store it as a file.
> `/home/orangepi/tg-commit`, `/home/orangepi/tg-decode-all`, and
> `/home/orangepi/tg-exp-stream` are present, so I'm now checking their exact
> revisions and whether one matches the recorded ChatGLM3 sources.

This means a `[ -d "$t/.git" ]` test cannot distinguish "no git" from "git
worktree". For `tg-qwen35-pack-signature-fix` the answer is definitively
"worktree of `tg-commit`" — proven by the `git worktree add` command in §2.

### 5.4 `tg-decode-all` is the default TREE of a *different* harness

`2026-11-17`, session `ses_e2ecd054effeSPzmuGpHUtCi2X`:

> `tg-decode-all` likewise needs freezing, since it is the benchmark tree behind
> the retime

But `retime_with_profile.py` is the harness that failed for these models. From
`qwen2-0.5b/report.json`: `complete: false`, `trials: 0`,
`error: "cycle detected in assign graph"`. Its own docstring says a retime that
invents its own env measures a different configuration, which is not a retime.

---

## 6. `onpu.worker` — the reference harness entry point

`onpu.worker` is the worker module of the `~/oNPU` app. Launched as
`python -m onpu --backend <TREE>`; the app imports `onpu.worker` in-process.
Representative invocations, all `ses_e2ecd054effeSPzmuGpHUtCi2X` unless noted:

```bash
# import check against a release tree
PYTHONPATH=/home/orangepi/tinygrad-release-rk3588-20261115 DEV=CPU ROCKCHIP_TEST=0 \
  /home/orangepi/oNPU/.venv/bin/python -c 'import onpu.worker; print("onpu.worker import OK")'

# app server bound to a specific backend tree
cd /home/orangepi/oNPU && nohup .venv/bin/python -m onpu \
  --backend /home/orangepi/tg-decode-all --port 8097 \
  --data-dir /home/orangepi/oNPU/.data-bench3 > /tmp/opencode/app97.log 2>&1 &
```

The `--backend` flag is what selects the tinygrad tree. `tg-decode-all` appears
as the `--backend` argument in the majority of these, which is the origin of the
"tg-decode-all is the reference tree" belief — but the 4B retime did **not** go
through `onpu.worker` at all. It called `benchmark_inference` inside the pinned
worktree via `run_candidate_warmed_retime.py`.

---

## 7. Queries that returned nothing

Each of these was run and produced no result. Absence is reported, not concluded.

| query | scope | result |
| --- | --- | --- |
| `rm -rf` within 40 chars of `.git` | all `part` + `session_message` | **nothing** except the task-prompt text requesting this search |
| `retime_with_profile.py qwen35-4b` **without** `--dry-run` | all rows | **nothing** — only `--dry-run` invocations exist |
| `run_candidate_warmed_retime.py --model qwen35-4b` from `~/tinygrad/.venv` **with** `--out` | all rows | **nothing** — every `--out` run uses `~/oNPU/.venv` |
| `git init` in `/home/orangepi/tg-decode-all` | all rows | **nothing** — `git init` appears only as a *proposal* in `ses_e16921d8fffeZKmwM90fg0gBif`, never as an executed command against that path |
| `tg-decode-all/.git` HEAD revision | all rows | **nothing** — no command ever printed a revision for that tree |
| session `ses_e14e016cfffeQEJRKmQhZT1OBR` | `session` + `session_v2` | **not found** — the id appears in `part`/`session_message` rows but has no session row, so no title or directory can be recovered |

---

## 8. Summary — the reproducible 4B reference configuration

| element | value | evidence |
| --- | --- | --- |
| runner | `~/oNPU/analysis/release-20261115/r2-lfm-20261116/run_candidate_warmed_retime.py` | §2, §3.1 |
| interpreter | `/home/orangepi/oNPU/.venv/bin/python` | §3.1 |
| backend tree | `/home/orangepi/tg-qwen35-pack-signature-fix` | runner line 20 |
| backend commit | `1b2a5db332e1198b3ace8c7559a19c7d73717dfb` | runner line 21 |
| backend branch | `codex/qwen35-pack-signature-fix` | runner line 22 |
| dirty patch | 4 files, diff sha256 `7f87f221…` | runner lines 23–29 |
| app tree | `~/oNPU` @ `5229660be83d3ee8e155e31109518fb90d641e80` | runner line 30 |
| profile | `onpu/rockchip_options.py` sha256 `3394f6df…` | runner line 32 |
| protocol | 1 warm-up + 3 measured, 128 / 64 tokens | report fields |
| decode | **6.3555 tok/s** (6.3418 / 6.3758 / 6.3555) | §4.1 |
| TTFT | 9335.780 / 19132.280 / **1210.634** ms — 15.804× spread, **unusable** | §4.3 |
| output dir | `…/candidate-retime-20261117-qwen35-4b` | §3.1 |
| command time | 2026-11-17 03:52:22 | §3.1 |

The single most important correction this search establishes: **`tg-decode-all`
is not the tree the 4B reference number came from.** The 4B retime ran in
`tg-qwen35-pack-signature-fix`, a git worktree of `tg-commit` at `1b2a5db` plus a
four-file dirty patch. `tg-decode-all` is the default `TREE` of
`retime_with_profile.py`, a different harness that failed on these models, and it
carries no `.git` with which to pin it.
