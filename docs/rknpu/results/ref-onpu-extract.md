# oNPU reference archive — Qwen3.5 4B extract and optimisation methodology

Read-only extraction from `/home/orangepi/oNPU/`. Every claim carries a `file:line`.
`[M]` = measured on this board. `[S]` = stated intent / plan / policy, not a measurement.

---

## 1. Every measured number for Qwen3.5 4B, with conditions

Workload for every row below unless stated: **128 uncached input tokens, 64 actual output
tokens, context 512, greedy** (`plan.md:247`, `benchmark.md:1169`). Published RKLLM
reference: **TTFT 2068.88 ms, 6.22 tok/s decode, 61.87 prefill-proxy tok/s**
(`benchmark.md:952`, `benchmark.md:978`), measured at maximum CPU/NPU clocks
(`benchmark.md:1169`).

### 1a. The Nov 3 API cohort (the "production head preparation" row)

This is the row the scoreboard and `table.md` publish.

| quantity | value | source |
|---|---|---|
| TTFT median | **1263.09 ms** | `benchmark.md:1329`, `benchmark.md:1333` |
| prefill proxy | **101.34 tok/s** (`128000/1263.09`) | `benchmark.md:1335` |
| decode | **6.48 tok/s** | `benchmark.md:1329` |
| measured requests | 20 (2 processes × 10) | `benchmark.md:1329` |
| serving checks | 36 | `benchmark.md:1329` |
| per-request leads | **20/20 prefill, 20/20 decode** | `benchmark.md:1335`, `benchmark.md:1189` |
| process medians | 1263.09 / 101.34 / 6.4687 and 1271.38 / 100.68 / 6.4857 | `docs/benchmark-qwen-inner-heads-20261103.json` (`process_medians`) |
| measured ranges | TTFT 1232.19–1303.86 ms; decode 6.4230–6.5125 | same file (`summary.measured_ranges`) |

Conditions, verbatim in effect: "two fresh ordinary API workers", "Rates use the worker timing
telemetry returned by completed HTTP requests. Local governors remain adaptive, without sudo
tuning, and the original GPU model stays resident" (`benchmark.md:1329`). Metric boundary is
**worker first-token telemetry, not HTTP first-byte latency** (`docs/benchmark-qwen-inner-heads-20261103.json`,
`metric_boundary`). Temperature is **not recorded** for this cohort — it is an API cohort, not a
thermally gated one. Clock state: **adaptive governors**, i.e. *not* pinned (`benchmark.md:1169`).
Native replay: backend ABBA companions ran `native_execution.prefill.runs=3`,
`decode.runs=375` with `profiled=false` and `reason=null` (same JSON, `backend_abba`).

Backend ABBA that qualifies the port (same JSON, `backend_acceptance`):
control 1538.61 ms → candidate 1224.42 ms, **prefill latency saving 20.42 %**;
second pair **19.93 %**; decode change **+0.28 %** and **−0.09 %**; `logits_sha256`
identical across all four processes; `all_six_sequences_exact_per_process: true`.
Also `plan.md:1310` and `benchmark.md:1327`.

### 1b. The source-pinned 1+3 retime (Nov 17) — a *different* cohort

| quantity | value | source |
|---|---|---|
| decode median | **6.3555 tok/s** (6.3418 / 6.3758 / 6.3555) | `candidate-retime-20261117-qwen35-4b/timing_review.json:17-24` |
| decode interval | 9934.08 / 9881.04 / 9912.74 ms | same, `:12-16` |
| TTFT trials | **9335.780 / 19132.280 / 1210.634 ms** | same, `:28-32` |
| TTFT median | **9335.780 ms** | same, `:25` |
| TTFT max/min | **15.804×** | same, `:27` |
| prefill proxy, median | 13.71 tok/s | derived, `128000/9335.78` |
| prefill proxy, **last trial** | **105.73 tok/s** (`128000/1210.634`) | derived |
| native replay | prefill 1, decode **249** (= 4 requests × 63 + 1) | `timing_review.json:35-36` |
| measured IDs identical | true, 64 IDs/trial | `timing_review.json:22-23` |
| `timing_validity` | **PASS** | `timing_review.json:57` |
| NPU finish bucket | all 3 trials at 1 GHz | `timing_review.json:37-55` |
| measured-trial temperatures | **69.307–76.692 °C** | `timing_review.json:9-10` |
| ready temperature | **53.615 °C** (52–55 °C band, 25.179 s wait) | `receipt.json` `ready_temperature_gate` |
| clocks | NPU 1 GHz, DDR 2112 MHz, CPU4/6 2256 MHz, `performance` | `receipt.json` `preflight_hardware`, `cleanup_hardware` |
| quality assessed | **false** | `timing_review.json:56` |
| backend | `1b2a5db…`, branch `codex/qwen35-pack-signature-fix`, 4 dirty paths | `receipt.json` `backend` |
| checkpoint | `10cc391b…`, 4,482,403,488 B | `receipt.json` `checkpoint` |

Per-trial temperature trace (`receipt.json` → `trial_temperatures.jsonl`): warmup 54.538→70.230;
measured 69.307→73.923, 73.000→74.846, 74.846→76.692 °C.

**This row is the source of the disagreement in §1e.** Note the median (9335.78 ms) is *worse*
than the published RKLLM 2068.88 ms; only the last trial (1210.63 ms) beats it.

### 1c. The Nov 16 cold one-request screen (superseded)

TTFT **19,293.771 ms** / prefill proxy **6.6343 tok/s** / decode **2.4383 tok/s**; 475 prefill and
12,789 decode submits; native decode replay 60, **native prefill replay absent**; gate 52.692 °C,
post-lock 53.615 °C, in-run peak 66.538 °C, cleanup 60.076 °C; clocks matched
(`analysis/release-20261115/r2-lfm-20261116/TIMING_INVENTORY.md:30`, `STATUS.md:460-469`).
Explicitly "not a fair warmed comparison" (`STATUS.md:468`).
A documentation defect was found in how its temperature was reported: 62.846 °C is an *in-run*
sample, not a trial boundary (`doc-audit-quick-cold-20261116.md:16-31`).

### 1d. Earlier backend-path numbers (all 3 warmups + 3 measured, 128/64/512)

All from `docs/benchmark-*.json`, trials listed; decode is stable across trials, TTFT is not.

| cohort | TTFT measured (ms) | prefill proxy | decode | source |
|---|---|---|---|---|
| gated-attention v1 | 3066.32 / 3066.71 / 3070.63 | **41.74** | **5.23** | `benchmark-gated-attention-qwen35-4b-v1.json`; `benchmark.md:2921-2922` |
| CPU GDN scan v1 | 2677.79 / 2689.35 / 2677.72 | **47.80** | **5.29** | `benchmark-cpu-gdn-qwen35-4b-v1.json`; `benchmark.md:2992` |
| NPU domains v3 (Oct 26) | 3971.03 / 3974.55 / 3979.23 | **32.20** | **4.84** | `benchmark-npu-domains-qwen35-4b-v3.json`; `benchmark.md:2880-2884` |

Note these three are *slower* than the API cohort even though they had 3 warmups: different
timing boundary (backend `first_token_ms` includes preparation, vs API worker telemetry).
Warmups in the same files read 12.7–42.7 s, which is the same trial-order artifact as §4.

Other published 4B rows, all superseded, all 128/64/512:

| date | TTFT / decode / prefill-proxy | conditions | source |
|---|---|---|---|
| Oct 31 API | 1551.08 ms / 6.32 / 82.52 | 10 requests, 18 checks | `benchmark.md:2527`, `:2534` |
| Nov 1 pre-K-split | 1588.10 ms / 6.45 / 80.60 | 10 requests/process, stock governors | `benchmark.md:2018`, `:2023` |
| Nov 1 pinned family | 1631.58 ms / 6.43 / 78.45 | R300/A14, 314-file cohort, 9/10 wins | `benchmark.md:2147`, `:2137` |
| Nov 2 current-source | 1689.74 ms / 6.29 / 75.75 | 10 requests, 18 checks, 321-file epoch | `benchmark.md:1741`, `:1753`, `:1755` |
| GPU FP16 baseline | 23,436.57 ms / 0.60 / 5.46 | `--gpu-recurrent-scan`, GPU-only | `benchmark.md:2835`, `:2847`, `:4030` |

### 1e. Status: three documents disagree — report all three

| document | verdict for the 4B row | line |
|---|---|---|
| `plan.md` header (written before the correction) | "Its TTFT remains unresolved and quality was not assessed, so **the full official row stays stale**" | `plan.md:35-38` |
| `plan.md` §TTFT trial-order correction | "The nine retimed rows are **re-scored as MEASURED (stabilized)**" — 4B listed with last ratio **1.709** | `plan.md:230-236`, `:221` |
| `PUBLISH.md` | "keep that full row stale too" | `PUBLISH.md:18-19` |
| `SCOREBOARD.md` row 18 | "**WIN both (stabilized last trial)** … **MEASURED**", 1210.6 ms / 6.3555, ratios 1.709 / 1.022 | `SCOREBOARD.md:83` |
| `TIMING_INVENTORY.md` | "TTFT UNRESOLVED … keep the official full row stale" | `TIMING_INVENTORY.md:30` |
| `stale-scoreboard-evidence-matrix` | "Retain decode evidence only; keep the full `STALE-Nov4` row until a TTFT acceptance rule exists" | `…evidence-matrix-20261116.md:57` |

`handoff.md:13` summarises the whole board as "9 stabilized, 5 exact-protocol, 2 API, **5 stale**,
0/21 quality". `SCOREBOARD.md:95` agrees: the 5 remaining stale rows are Qwen3, RWKV7, TeleChat2,
Phi3, **Qwen3.5 0.8B**. So the scoreboard counts the 4B as stabilized while `plan.md:35-38`,
`PUBLISH.md:18-19` and `TIMING_INVENTORY.md:30` still call it stale. **Unresolved in the archive.**

---

## 2. The exact recipe — the `large = True` branch of `rockchip_options.py`

Gate: `checksum in QWEN35_FAST_CHECKPOINTS and context == 512 and profile == "ROCKCHIP" and
prefill in ("auto","on") and decode == "on"` (`onpu/rockchip_options.py:379-380`).
`large = checksum == QWEN35_4B_CHECKPOINT` (`:381`), checkpoint `10cc391b…` (`:27`),
profile name `qwen35-4b-context512` (`:29`). Body at `:384-402`.

| key | value (4B) | what it does |
|---|---|---|
| `prefill` | `"on"` | enable NPU prefill path |
| `mapped_cpu` | `True` | CPU weights mapped, not decoded into RAM (`:547-551`) |
| `share_projection_weights` | `True` | one copy of projection weights for prefill+decode |
| `cpu_recurrent_scan` | `True` | GDN scan runs on CPU natively, not per-token NPU |
| `prefill_chunk` | `128` | 128-row prefill chunks |
| `tile_columns` | **`24384`** (large) vs 24576 | head tile width. Comment `:382-383`: "Keep the K2560 vocabulary head below the 8192-column/core transition that corrupts the next cached FP16 operation in the primitive check." |
| `gdn_fast` | `True` | fast GDN path |
| `cpu_q8_embedding` | `True` | Q8 embedding lookup on CPU (248320×2560, 675,430,400 B raw — `receipt.json` `path_evidence`) |
| `scalar_decode_loop` | `True` | scalar (1-row) decode loop |
| `trim_preparation_cache` | **`True`** (large only) | frees 1,204,457,728 B of preparation cache (`receipt.json` `placement.cpu_preparation_cache_trim_bytes`) |
| `preparation_programs` | **`4`** (large) vs 6 | fewer prepared programs to fit 16 GB |
| `decode_preparation_programs` | `5` | decode-side prepared programs |
| `prefill_profile` | `"qwen35-4b-context512"` | profile identity string |

Environment (`:390-401`), verbatim values:

| env | 4B value | note |
|---|---|---|
| `ONPU_SWIGLU_PREFILL_ROWS` | `4` | 4-row SwiGLU prefill |
| `ONPU_SWIGLU_PACK_STATIC_LAYOUT` | `1` | static layout for the pack kernel |
| `ONPU_CPU_TOKEN_INPUTS` | `1` | token IDs consumed on CPU |
| `ONPU_ZERO_PAD_VIEWS` | `1` | zero-pad via views |
| `ONPU_SHARED_HOST_POOL` | `1` | shared host worker pool |
| `ONPU_PREFILL_CPU_TUNING` | `1` | prefill CPU tuning |
| `ONPU_CPU_PREFILL_LAYOUT` | `1` | CPU prefill layout |
| `ONPU_REPLAY_BORROW_OUTPUTS` / `_BATCH` | `1` / `1` | borrow replay buffers |
| `ONPU_REPLAY_NATIVE_FROM` / `_LARGE` | `1` / `1` | native replay, large variant |
| `ONPU_GDN_DECODE_SPIN` | `1` | spin wait in GDN decode |
| `ONPU_GDN_PREP_POOL` / `ONPU_GDN_DECODE_PREP_POOL` | `1` / `1` | GDN preparation pools |
| `ONPU_GDN_DIRECT_DECODE` | `1` | direct GDN decode scan |
| `ONPU_FFN_PACKED_UPLOAD` | `1` | packed FFN upload |
| `ONPU_EXP2_FAST_PATH` | **`1`** (large) vs `0` | fast exp2, 4B only |
| `ONPU_CPU_DECODE_POOL` | `0` | no CPU decode pool |
| `ONPU_HOST_POOL_WFE` | `1` | WFE in host pool |
| `ONPU_CPU_GQA_PREFILL` | `0` | no CPU GQA prefill |
| `ONPU_INT8_SDK_PREFILL` | `0` | no SDK prefill |
| `ONPU_INT8_EXPERT_DEPTHS` | `0` | no experts |
| `ONPU_INT8_DECODE_NARROW_CBUF` | `1` | narrow CBUF for decode tails |
| `ONPU_REPLAY_SCALAR_COPY` / `_DEDUP_GUARDS` | `1` / `1` | replay copy + guard dedup |
| `ONPU_INT8_NATURAL2304` | `1` | natural 2304 depth |
| `ONPU_NPU_DOMAINS` | **`3`** (large) vs `1` | three IOMMU domains |
| `ONPU_NPU_DOMAIN_BYTES` | `3221225472` (3 GiB) | per-domain bytes |

Effective flags actually passed at the retime (`receipt.json` `selected_recipe.flags`):
`--fuse-projections --cpu-attention --native-replay --fused-ffn --split-down --fused-qkv
--cpu-norm --cpu-fast-softmax --cache-cleaning --wake-helper --cpu-pool --cpu-prefill-pool
--share-projection-weights --cpu-recurrent-scan --fused-swiglu-ffn --fused-swiglu-decode
--fused-qkv-without-norm --fused-recurrent-projections --cpu-q8-embedding --scalar-decode-loop
--trim-preparation-cache --file-backed-cpu-weights --defer-weight-realization`
plus `--chunk-size 128 --tile-columns 24384 --warmups 1 --runs 3 --tokens 64 --context 512
--device CPU --precision int8`.

**Why 24384 and not 24576**: measured. "the previous 8,192-column/core state corrupted a
subsequent FP16 health operation before the proposed CBUF change ran. The replacement head and
actual remainder pass 28 integer/canary checks and 29 cached FP16 checks" (`benchmark.md:2011`).

---

## 3. Optimisation methodology — numbered procedure

1. **Locate and pin.** "Start from the 21-model inventory … Identify GGUF/hash, existing working
   source tree, relevant custom kernels/patches, canonical recipe and runner. Record commits plus
   dirty-file hashes and effective flags. Do not reconstruct a release candidate for every model."
   (`plan.md:240-244`)
2. **Verify the path during preparation.** "Confirm the runner imports the selected backend and
   uses the intended optimized path. Record path evidence and label any fallback explicitly."
   (`plan.md:245-246`)
3. **Quick screen first** — exploratory, not a win: "Ordinary text, context 512, fixed 128-ID input
   and 64-token output, empty KV state, one measured request, and no separate warm-up … This is
   exploratory triage: first-call JIT/preparation may affect it, so it cannot establish a fair win
   against a warmed published RKLLM row … Report apparent leads as unqualified." (`plan.md:247-254`)
4. **Qualify contenders later.** "Only a quick-screen candidate that merits follow-up gets the
   standard one completed warm-up plus three unprofiled measured requests, with cold load and
   compilation separated. First-emitted-text TTFT is not isolated prefill; label prompt-tokens/TTFT
   as a proxy." (`plan.md:255-258`)
5. **Save and advance.** "Write unique durable results incrementally … Every model gets a summary
   row: `MEASURED`, `BLOCKED` with a specific cause, or `MISSING INPUT`, with an artifact link. Do
   not optimize a measured loser during the coverage pass." (`plan.md:259-262`)
6. **Failure budget.** "at most one targeted repair and one retry per model in the first pass; stop
   earlier after 30 minutes of active diagnosis … An unexplained repeat failure becomes `BLOCKED`;
   advance to the next model … Reopening a blocked row requires separate user authorization."
   (`plan.md:264-271`)
7. **Hardware discipline.** "user-run clock preparation, exclusive ownership, no overlapping
   tests/compiles/copies, and per-trial clock/temperature observations. Compare local pairs within
   the same finishing-clock bucket; an 800 MHz NPU finish alone is not a rejection reason."
   (`plan.md:273-278`)
8. **Pre-register the contract before running** (`plan.md:539-560`): pin and hash checkpoint,
   tokenizer, template, converter, runtime, both commits, dirty source, effective flags, runner and
   fixtures; "Predeclare process-level repeats, trial counts, order, exclusion rules and statistical
   summaries. Publish medians/spread, paired effects where genuinely paired, and uncertainty. A
   small positive median alone is insufficient for a robust win; preserve losses and inconclusive
   results." (`plan.md:552-554`)
9. **Quality is a gate, not a footnote.** "No independently quality-qualified model is currently
   recorded. Failed quality means no product-win claim, even if the timing is fast."
   (`plan.md:594-595`)
10. **Restore.** "Restore original options/settings/selection and obtain fresh replies from original
    models after a hardware lease." (`plan.md:1502`)

### Acceptance criteria, verbatim

| gate | verbatim text | source |
|---|---|---|
| decode-gain gate | "Acceptance: ≥0.5%decode gain, ≤3%TTFT regression, positive95%paired-block bootstrap lower bound." | `plan.md:1595` |
| prefill-gain gate | "Require39actualcalls/request,source-specificcounters,64full-vocabulary arrays/tokenresponses exact,≥1%pairedTTFTgainwithpositive95%lower bound,≤2%decode regression." | `plan.md:1611` |
| combined-recipe gate | "3%TTFT/2%decoderegressiongatesineachpair;stoponinvalidorfutility" | `plan.md:1620` |
| Gemma4 GELU gate (worked example) | "Median TTFT falls only1.11% (1163.01→1150.11ms), below the predeclared3%minimum in each pair … Stop the unstarted reverse pair for futility and reject standalone promotion." | `benchmark.md:923` |
| primitive gate | "below the predeclared10%primitive threshold for integration" | `benchmark.md:907` |
| decode regression limit | "decode regresses2.40%, exceeding the declared2%limit" → reject | `benchmark.md:1195` |
| soak protocol | "55 °C cold start, n=3 pairs, 5 trials, 64 tokens" | `progress.md:392`, `chatglm.md:6` |
| temperature gate | `--max-temp-gap` default **2 °C**; pairs whose arms start further apart are `rejected: temp_gap` | `scripts/run_soak_coldstart.py:289`, `:409-413`; `progress.md:987-988` |
| `timing_validity` scope | "Runner `timing_validity=PASS` covers completion, workload and path assertions, **not TTFT consistency**." | `plan.md:291-292` |
| id parity | "`generated_ids` identical to `[6181, 290, 267, 1603, 5611, 2116, 1580, 906, 3013, …]` on the ChatGLM path / `69d2b997…` on the MiniCPM3 path, plus `fused_swiglu_ffn_blocks == 28`" | `progress.md:980-983` |
| id parity, honest limit | "**"Bit-exact" means identical to our own `analysis/minicpm3/recorded_reference.json`** … reproducibility, not competitor parity. Compare *timing* against RKLLM and *quality* by coherence / logit distance." | `progress.md:866-870` |
| measurement floor | "paired-ratio sd 0.0027–0.0056 = **0.6–1.1 ms** of wall. Below that is **UNRESOLVED, not null** — stack small wins before soaking." | `progress.md:1003` |
| clocks validity test | "this is the validity test, and it is **`min_freq`**, not `max_freq`" | `progress.md:899` |
| NPU 800 MHz | "Measured cost is **−0.54 %** … **So do not reject a run for it; bucket trials by the clock they finished at and compare arms within the same bucket.**" | `progress.md:906-913` |
| pairing discipline | "Never quote cross-window medians on this box: RKLLM's own median moved 1447.3 → 1580.0 ms between windows … Only paired, same-window numbers count." | `minicpm3.md:30-32` |
| report wins as counts | "Report wins as pair counts, not medians" (9/10, 10/10) | `AGENTS.md` mirror of `metal_kernels` §3; applied at `benchmark.md:1189` |

### How a loss is recorded

Rejected candidates stay in the ledger with their numbers and are not deleted:
"Preserve both failed setup attempts and corrected raw cohorts. No default promotion."
(`plan.md:1340`); "Reject production promotion and retain all measured losses."
(`benchmark.md:919`); "The `+0.52%` "eighth win" was an artefact … Corrected anchor: `rms_all` =
**−0.05%**" (`progress.md:555-559`); "soak-v11 was reported as "the arm improved" when it went
down … I read the sign off the ratio direction rather than the arm" (`progress.md:561-562`).

### Two retractions that define the method

- **Isolated ≠ in-model.** "Never quote an isolated measurement as an in-model number."
  (`AGENTS.md`; the archive's own example: `plan.md:180-190` — the app's `cache_prompt=false` cold
  path returned 0 tokens on all four trials, so the CLI runner is the valid cold harness).
- **Attribute by ablation, never by multiplication.** `AGENTS.md`; the archive's failure was
  building a fused per-Linear kernel before ablation showed the scan was 85 % of prefill
  (`AGENTS.md`).

---

## 4. The trial-order / stabilisation rule, in full

### Why 1+3 is wrong

"**The TTFT spread is a deterministic trial-order artifact, not thermal noise.** Three independent
cold 1+3 cohorts (c/d/e) produced the same pattern: trial 1 ≈15.1–15.5 s, trial 2 ≈62.5–65.8 s,
trial 3 ≈1.52–1.54 s (83–84 prefill t/s). The third trial is the hottest (80–81 °C) and the
fastest; the sampled clocks are 1 GHz NPU / 2.256 GHz CPU6 / 2.112 GHz DDR before and after every
trial. **The median of 3 is therefore the first-trial artifact, not a stable prefill estimate.**
The clean prefill is the third trial; the 1+3 protocol needs a post-JIT stabilization rule (discard
the first two trials or require two consecutive stable trials) before any TTFT cell is promoted."
(`plan.md:191-197`)

Across nine retimed models the last measured trial is **6.7–15.7× faster** than the median of the
three measured trials (`plan.md:208-210`). Full table at `plan.md:212-223`; machine-readable at
`h5-cold-cohorts-20261117/trial_order_analysis.json`. Qwen3.5 4B specifically: median 9335.8 ms,
last 1210.6 ms, `last_vs_median` **7.711×**, `last_ratio` **1.709** (`trial_order_analysis.json:100-113`).

### What 3+3 measured

"The **3+3 rule was confirmed on MiniCPM4** on 2026-11-17: measured TTFT 103.39 / 104.01 / 115.94 ms
(median **104.01 ms**, **1.301×**), decode median **57.10 tok/s** (**1.259×**),
`timing_validity=PASS`." (`plan.md:232-234`)

Receipt detail (`stabilized-minicpm4-20261117/benchmark.json`): 3 warmups measured
**5218.45 / 1651.69 / 1955.46 ms**, then 3 measured **103.394 / 104.008 / 115.939 ms**; decode
57.744 / 56.345 / 57.100. So the warmups absorb the artifact and all three measured trials land
within 12 % of each other. Measured-trial temperatures 66.538–68.384 °C; ready 57.307 °C
(`receipt.json` `timing_summary`, `hardware_after_lock`).

### The exact rule for quoting a prefill number

1. **A 1+3 run cannot supply a prefill/TTFT cell.** "Do not promote any row without qualified
   evidence" (`plan.md:610`); "Do not rerun or replace the official TTFT cell without a predefined
   TTFT acceptance rule" (`stale-row-retime-execution-plan-20261116.md:111`).
2. **A 3+3 run can**, and the quoted value is the **median of the three measured trials**, not the
   last trial: MiniCPM4 is quoted as 104.01 ms median, not 115.94 ms last (`plan.md:233`,
   `SCOREBOARD.md:66`).
3. **The last-trial reading is a re-scoring, not a replacement.** "This does not promote any row by
   itself: the last-trial choice is post-hoc, so a pre-registered stabilization rule (3 warm-ups +
   3 measured, or discard the first two measured trials) must be confirmed on at least one model
   before re-scoring." (`SCOREBOARD.md:104-107`)
4. **Decode is stable and may be quoted from either.** "Decode is stable (paired medians
   ≈1.01–1.04), but the prefill cell cannot be replaced by a median over three trials when the
   trials span an order of magnitude. A fourth trial does not repair a bimodal distribution."
   (`plan.md:169-172`)
5. **The 4B's 3+3 has not been run.** The scoreboard annotates its 4B row "3+3 confirmed on
   MiniCPM4" (`SCOREBOARD.md:83`) — i.e. the rule was validated on a *different* model; the 4B
   itself has only the 1+3 last-trial reading.

---

## 5. Recorded failures and rejections touching Qwen3.5 or the GDN scan

| # | failure | evidence |
|---|---|---|
| 1 | **8,192-column/core head state corrupts the next cached FP16 op.** 4B head moved to 24,384-column tiles; replacement passes 28 integer/canary + 29 cached-FP16 checks | `benchmark.md:2011`; `rockchip_options.py:382-383` |
| 2 | **Coefficient-only GDN conv-weight preparation rejected**: "Only 2–4% component savings; not promoted." | `docs/benchmark-qwen-inner-heads-20261103.json` `rejected_coefficient_only_screen` |
| 3 | **errno-14 on a 20 MiB INT8 projection buffer (8,192×2,560)** with >5 GiB free; kernel logged an IOVA allocation failure; DMA ranges span ~4 GiB | `benchmark.md:2871-2878` |
| 4 | **2 GiB reserve check refused the unmodified 4B load before timing**; a later partial probe report failed finalization without timing | `benchmark.md:2869` |
| 5 | **Wait overlays launched from the wrong working directory** selected the stock package, so those attempts did not test their candidates; copied-pycache and guard failures retained | `benchmark.md:2164` |
| 6 | **Qwen3.5 0.8B warmed retime BLOCKED**: 52–55 °C gate timed out after 7,202.559 s, 1,439 samples 40.692–51.769 °C, **zero in-band**, zero warmups, zero trials | `plan.md:29-31`; `TIMING_INVENTORY.md:28` |
| 7 | **Do not repeat rejected K2560 CBUF tuning**; require a concrete new mechanism before another register variant | `plan.md:1342` |
| 8 | **CPU GDN head-parallel is real but insufficient**: parked pool 71.15/10.80 tok/s vs serial 63.92/10.00, both **below** RKLLM 164.75/13.59. The app worker does not enable the profile | `benchmark.md:4853-4861` |
| 9 | **GDN head-parallel C microbenchmark is not a model result**: 128-row scan 56.409 → 12.723 ms parallel, but "These are scan timings, not full-model speedups" | `benchmark.md:4867-4878` |
| 10 | **GDN direct decode / spin are default-OFF in the release candidate**: "`ONPU_GDN_DIRECT_DECODE` and `ONPU_GDN_DECODE_SPIN` default 0 … Keep out unless those models are in the approved support set" | `R0_SOURCE_HUNK_REVIEW.md:54` |
| 11 | **GDN preparation routes not selected** for the release candidate | `R0_SOURCE_HUNK_REVIEW.md:117` |
| 12 | **The recurrent scan is only ~1 % of prefill** on the 2B FP16 target: 0.063–0.067 s of a 6.0–7.3 s chunk | `benchmark.md:4558-4567` |
| 13 | **Qwen3.5 4B prefill is host-bound, not scan-bound**: decode is 89.91 % NPU matmul, 9.70 % CPU, 0.39 % copies — so "Target weight DMA, task/submission behavior and the dominant matrix shapes before another CPU microkernel attempt" | `benchmark.md:1392`; `plan.md:1342` |
| 14 | **Qwen2 alias / smaller-attention-tile candidates rejected** (adjacent Qwen work, same session): no copies eliminated, all four alias blocks slower, CPU tile 5.0 % slower | `plan.md:1340` |

---

## 6. What the archive says is still missing or unresolved for Qwen3.5 4B

1. **A TTFT acceptance rule.** "Retain decode evidence only; keep the full `STALE-Nov4` row until a
   TTFT acceptance rule exists." (`stale-scoreboard-evidence-matrix-20261116.md:57`). The 3+3 rule
   exists but has only been executed on MiniCPM4 (`plan.md:232`, `SCOREBOARD.md:83`).
2. **Independent output quality — 0/21 rows.** "Quality remains unassessed" (`plan.md:38`);
   `quality_qualified=false`, `maximum_context_qualified=false` in
   `docs/benchmark-qwen-inner-heads-20261103.json`. "Quality is not yet independently qualified for
   any of the 21 models." (`PUBLISH.md:89-90`)
3. **Model-maximum context.** "Model-maximum context remains open" (`benchmark.md:1337`);
   `maximum_context_qualified=false` (`benchmark.md:2885-2886`).
4. **Repeatability beyond the cohorts measured.** "Independent NLL and model-maximum context remain
   unqualified" (`benchmark.md:2149`); "independent-process repeatability … remain unqualified"
   (`benchmark.md:2060`).
5. **The 4B decode lead is narrow and not every-request.** "+1.18 %, 7/10 leading" on the Nov 2
   cohort (`benchmark.md:1755`); "Qwen3.5 4B decode on 7/10" (`plan.md:1385`). The Nov 3 cohort
   claims 20/20 (`benchmark.md:1189`) — the two cohorts disagree on consistency.
6. **Clock basis is not matched.** oNPU runs "stock adaptive governors and no sudo tuning; RKLLM
   publishes maximum CPU/NPU frequencies" (`benchmark.md:1169`). No pinned-clock 4B API cohort
   exists; only the Nov 17 retime is pinned (`receipt.json`).
7. **The 4B's own 3+3 has not been run**, and its 1+3 median TTFT (9335.78 ms) is *worse* than the
   published RKLLM 2068.88 ms — the win depends entirely on the post-hoc last-trial reading.
8. **Cross-scheme basis is permanent.** "every row in this project is **cross-scheme** (our Q8_0
   GGUF vs the rival's w8a8) … never present the 20 non-LFM rows as w8a8-vs-w8a8 parity."
   (`progress.md:880-883`). The 4B's rival artifact does not exist locally at all
   (`table.md:23` — "— none —" for RKLLM ckpt).
9. **The 4B is not in the release candidate.** No 4B row appears in the R1 allowlist scope; the
   approved candidate is LFM-only (`plan.md:799-802`).
10. **Upstream portability is a separate, newer question.** The current `~/pr/tinygrad` tree loads
    and runs one 4B forward pass (`arch: qwen35`, 426 tensors, 4,841,450,496 params, `status: PASS`,
    143.1 s) — `analysis/upstream-support-20261117/results.jsonl:18`. That is a load/forward check,
    not a timing or quality result.

---

### Cross-reference: the numbers this extract contradicts in our own tree

`AGENTS.md` (this repo) quotes oNPU 4B canonical values as TTFT **1263.09 ms**, prefill proxy
**101.34 tok/s**, decode **6.48 tok/s**, decode step 154.22 ms = 138.66 + 14.95 + 0.61, sourced to
`~/oNPU/benchmark.md:1179` and `:1392`. Those are the **Nov 3 API cohort** values (§1a) and the
**native profile** split (`benchmark.md:1392`), both correct as cited. They are *not* the Nov 17
retime values (§1b: 6.3555 tok/s, 9335.78 ms median / 1210.63 ms last). The clock basis differs:
our tree pins NPU 1 GHz / DDR 2112 MHz / CPU6 2256 MHz, while the 1263.09 ms cohort ran on
**adaptive governors** (`benchmark.md:1169`). Comparing our pinned run to that unpinned reference
understates our gap, not flatters it.
