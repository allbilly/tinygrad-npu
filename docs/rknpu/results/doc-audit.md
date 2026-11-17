# Adversarial audit — `docs/rknpu/npu-gemm-baseline.md`

Read-only audit, 2026-11-17. No file in the repo was modified except this one. The
NPU was not touched and no model was loaded; the only model access was parsing the
GGUF *metadata* header (tensor names/dims, no tensor data) to re-derive weight sizes.

Artifact status up front:

- `results/submits-4b/report.json` **does not exist** — the directory
  `results/submits-4b/` is empty. `results/submits-4b.log` has two lines: a load
  line and `trial 1/1 prefill 31.0s 323 submits`. The run died before the decode
  line, so **submits/token has never been measured**.
- `results/qwen35-4b/report.json` predates the `npu_weight_bytes` / `npu_submits`
  fields added to `bench_llm.py:154-170`, so it contains neither.

---

### high — "at parity" is 6.2% behind once the doc's own submit count and its own k=4096 fit are used

- location: `npu-gemm-baseline.md:91` (table row `NPU matmul per step | 138.66 ms | 131.5 ms floor | **at parity**`), `:97-98` (floor derivation), `:100` ("agree to within 5%"), `:104`
- values: doc floor `131.5 ms` = 4.2 GB / 31.93 GB/s, plus "~8.5 ms of submits" in prose = 140.0 ms. Re-derived from the artifacts: packed weights are **4.2048 GB** (249 linears, GGUF dims + `npu_linear.py:279` `round_up(n,32)*round_up(k,32)`), but **1.007 GB of that is in k=4096 tiles** (ffn_down k=9216 → two 4096 tiles per linear, plus attn_output/ssm_out at k=4096), which the doc's own fit prices at **27.85 GB/s** (`:61`). Weights = **136.3 ms**, not 131.5. Submits = **323** (`results/submits-4b.log`), not 250 → **11.0 ms**. Total floor **147.3 ms** vs 138.66 ms = **1.062x, i.e. 6.2% behind**. The doc's 1.01x is only reached by using 250 submits *and* applying the k=2560 slope to 100% of the weights.
- settles it: `results/submit-timing.json` (both fits, `fits[0].implied_gbps=31.9279`, `fits[1].implied_gbps=27.8508`), `results/submits-4b.log` (323), GGUF metadata for the k=4096 share
- action: replace the table row with `147 ms floor` / `6% behind`, or label the 131.5 as "weights only, k=2560 slope applied to all weights" and stop calling it parity. The word "parity" in bold is not supported.

### high — "no device headroom left to win" is contradicted by the doc's own two measured rows

- location: `npu-gemm-baseline.md:63-66` ("The device path has no headroom left… A faster kernel cannot be ported in because there is nothing to gain"), `:100-102` ("there is no device headroom left to win"), vs `:61` (k=4096 fit 82.4%) and `:77-80` (m=64 at 52.5%)
- values: 94.5% of ceiling holds only for **m=1, k=2560, n≥4096**. The same artifact measures **82.4%** at k=4096 (`submit-timing.json` `fits[1]`) and **52.5%** at m=64 (`rows[15]`, 17.73 GB/s). 1.007 GB of the 4.2 GB stream (24%) runs at the 82.4% point, and the prefill row of the same table measures 52.5%. "Nothing to gain" is false for a quarter of the decode stream and for every row above m=1.
- settles it: `results/submit-timing.json`
- action: replace. Scope the claim to "m=1, k=2560, n≥4096" and state the two measured exceptions, or delete the "no headroom" sentence and the "retires the port-the-fast-kernels direction" sentence at `:101-102`.

### high — "prefill is gated by the CPU SSM scan" is refuted by the section's own number

- location: `npu-gemm-baseline.md:174` (section title), `:180-181`
- values: the section cites **56.4 ms serial / 12.7 ms pooled per block**. 24 GDN blocks × 56.4 ms = **1.35 s**, or × 12.7 = 0.30 s, against a **178.7 s** prefill — **0.76% / 0.17%**. Even the Python loop's own cost is measured at **14.9 s = 8.3%** (`todo.md:53-59`). "Gated by" implies the dominant term; the cited evidence is two orders of magnitude too small. ~90% of prefill is unattributed in this document.
- settles it: `~/oNPU/benchmark.md:4878` (the 56.409 / 12.723 row), `todo.md:53-59` (the 14.9 s figure)
- action: replace the title and the claim. "The CPU SSM scan is 8% of prefill; 90% is unattributed" is what the evidence supports.

### high — "~250 submits / 8.5 ms" is contradicted by the only measured submit count

- location: `npu-gemm-baseline.md:70-71` ("The fixed cost is 0.034 ms per submit. At ~250 submits that is 8.5 ms per token"), repeated at `:97-98`
- values: doc **~250**. Measured **323** (`results/submits-4b.log`, 16-token prefill). Structural count from `npu_linear.py` + GGUF shapes = **323** as well: 312 block submits (24 GDN blocks × 10 + 8 attention blocks × 9) + 11 lm_head column tiles. lm_head is always m=1 — `model.py:803-804` runs `self.output` on `x[:, -1:]` only — so the count is identical in prefill and decode. 0.034 × 323 = **11.0 ms**, not 8.5. Caveat: the 323 is a *prefill* line; the decode line was never printed and `submits-4b/report.json` does not exist, so 323 for decode is derived (code + shapes), not measured.
- settles it: `results/submits-4b.log`; `extra/rockchip/npu_linear.py:280-283,306-311`; `tinygrad/llm/model.py:803-804`; `extra/rockchip/bench_llm.py:130-133`
- action: replace 250 → 323 and 8.5 → 11.0, or label the 8.5 as projected. Re-run the submits-4b job to completion so a decode count exists.

### high — "the 4B completed a full FP16 run" has no artifact and two logs say it failed

- location: `npu-gemm-baseline.md:161-163`
- values: "Peak became 12.4 GB instead of 19 GB, and the 4B completed a full FP16 run." No FP16 4B `report.json` exists anywhere under `results/` (`qwen35-4b/report.json` is the INT8 run: `load_seconds=131.94`, matching `todo.md:38` "131.9 s against 27 s for FP16"). `results/sweep-run3.log` and `sweep-run4.log` both carry the marker: *"Qwen3.5 4B: 8.41 GB of FP16 packed linears does not fit in host RAM on this 16 GB board. Four OOM kills"*. The 12.4 GB and 19 GB figures appear in no log; `4b-defer-test.log` shows `self_shmem=11GB avail=1GB`.
- settles it: absence of any FP16 4B report; `results/sweep-run3.log`, `results/sweep-run4.log`, `results/4b-defer-test.log`
- action: delete the "completed a full FP16 run" clause (the FP16 table is gone, so the claim has no purpose), or attach the report that proves it. Replace 12.4/19 with the logged 11 GB / 1 GB available.

### medium — the headline TTFT is not the median

- location: `npu-gemm-baseline.md:89` (`TTFT | 1263 ms | 178,600 ms | 141x`)
- values: `qwen35-4b/report.json` trials are `220574.06, 178730.72, 178602.51`; median **178730.72**. The doc's 178,600 matches **trial 3** (178,602.51), i.e. the lowest trial, not the median. The protocol the doc itself states at `:17-19` is "median taken". Difference is 0.07%, so no conclusion moves, but this is exactly the post-hoc trial pick the protocol forbids.
- settles it: `results/qwen35-4b/report.json:18-28`
- action: replace with 178,731 (or 178.7 s), and keep 141x (141.5x).

### medium — the parity table's floor, the prose floor, and the residual are three different numbers

- location: `npu-gemm-baseline.md:91` (131.5), `:97-98` (131.5 + 8.5 = 140.0), `:92` (~514 residual)
- values: 654.4 − 131.5 = **522.9**, but the table states **~514**, which is 654.4 − **140.0**. So the residual column is computed against a floor that includes submits while the floor column excludes them. The two cells of the same table use different floors.
- settles it: arithmetic on `report.json` (`1000/1.5281 = 654.4 ms`) and `:97-98`
- action: replace the floor cell with 140 (or 147, per the first finding) so the row is self-consistent.

### medium — "like-for-like" is overstated: the reference is stock-governor HTTP medians

- location: `npu-gemm-baseline.md:84-85` ("same 128/64/512 protocol, so this is a like-for-like comparison")
- values: the reference row is `~/oNPU/benchmark.md:1179`, whose surrounding text (`:1329`) says *"Local governors remain adaptive, without sudo tuning"* and that the numbers come from *"two fresh ordinary API workers… across 20 requests"*. Ours are clock-pinned (`report.json` `clocks_start`, NPU 1 GHz / DDR 2112 / CPU6 2256000) and measured in-process over 3 trials. Different clock policy and different measurement surface (HTTP serving vs in-process).
- settles it: `~/oNPU/benchmark.md:1329`, `results/qwen35-4b/report.json:30-41`
- action: label as "same token protocol, different clock policy and measurement surface". The reference is at a disadvantage on clocks, so the real gap is *understated*, not overstated — say which direction.

### medium — the prefill row-tile analysis uses a shape the harness never issues

- location: `npu-gemm-baseline.md:77-80` ("Prefill row tiles… `m=64` reaches only 17.73 GB/s (52.5%) at 9216x2560")
- values: `bench_submit.py:60-62` submits m=64 directly. But `NPULinear.__call__` sets `row_tile = max(1, min(64, 65536 // k_max))` (`npu_linear.py:299-300`), so the harness issues **m=25** at k=2560 and **m=16** at k=4096 — never m=64. A 128-row prefill is therefore 6 row tiles, each re-streaming the full weight tile.
- settles it: `extra/rockchip/npu_linear.py:299-300`, `extra/rockchip/bench_submit.py:60-62`
- action: re-measure at m=25/m=16, or label the m=64 row as a probe shape that the model path does not produce.

### medium — the m=64 explanation cannot account for the measured loss

- location: `npu-gemm-baseline.md:78-80` ("because the same weight stream is amortised over 64 rows but the activation and output traffic grows with `m`")
- values: at m=64, n=9216, k=2560 the traffic is weights 23.59 MB + activations 0.16 MB + output 2.36 MB = **26.12 MB**. At 33.79 GB/s that is 0.773 ms; measured 1.331 ms → **58%** of ceiling. The doc reports 52.5% (weight bytes only). Including act+out explains 5.5 points of a 47.5-point deficit; **~42% of the loss is unexplained** by the stated cause.
- settles it: `results/submit-timing.json` `rows[15]` (`ideal_ms` counts weight bytes only, `bench_submit.py:100,107`)
- action: label the cause as a hypothesis, or delete it and state the deficit as unexplained.

### medium — the 56.4/12.7 ms scan number is 2B geometry, cited for the 4B

- location: `npu-gemm-baseline.md:180-181`
- values: the source table is `~/oNPU/benchmark.md:4878` and the text at `:4876-4877` says *"The 16-head/key128/value128 shape matches the recorded 2B recurrent geometry"*. The 4B GGUF has `attention.key_length=256`, `value_length=256`, `head_count=16`. The 4B per-block scan is a different size, so 56.4 ms is not the 4B number.
- settles it: `~/oNPU/benchmark.md:4876-4878`; GGUF metadata
- action: label as 2B geometry, or re-measure at the 4B shape.

### medium — `rockchip_recurrent.c` (491 lines) does not exist in the archive

- location: `npu-gemm-baseline.md:178-179`
- values: `find /home/orangepi/oNPU -name "*.c"` returns 20+ files, none named `rockchip_recurrent.c`; `grep -rn "rockchip_recurrent"` finds only Python modules `rockchip_recurrent.py`. The *measurement* exists (`benchmark.md:4870-4878`, evidence `docs/benchmark-cpu-gdn-head-parallel-perf-v1.json`), the named source file does not.
- settles it: filesystem search of `~/oNPU`; "none exists" for the cited filename
- action: replace with the real path (`tinygrad/llm/kernels/rockchip_recurrent.py` plus the C microbenchmark) or drop the line count.

### medium — the bandwidth sweep's clock and temperature state is not in its artifact

- location: `npu-gemm-baseline.md:41-44` ("20 reps, median, clocks pinned: NPU 1 GHz, DDR 2112 MHz, CPU6 2256000, 50.8 °C")
- values: `submit-timing.json` records only `reps`, `ddr_ceiling_gbps`, `rows`, `fits`. No clocks, no temperature. `bench_submit.py` has no `clocks()` call (contrast `bench_llm.py:33-46`). The 50.8 °C and the pin state are transcript-only, and per the project's own notes the rknpu driver can drop to 800 MHz under thermal load regardless of `max_freq`.
- settles it: `results/submit-timing.json` (absent fields); `extra/rockchip/bench_submit.py:155-156`
- action: add `clocks_start`/`clocks_end` to `bench_submit.py` and re-run, or label the pin/°C as unrecorded.

### medium — the document is still framed as an FP16 withdrawal notice

- location: `npu-gemm-baseline.md:1` (title "NPU GEMM baseline — FP16, withdrawn as a comparison"), `:3-9` (blockquote), `:21-37` (whole section "Why the FP16 numbers were withdrawn")
- values: the file now leads with two INT8 sections and contains no FP16 result at all. `:3-9` and `:21-37` exist only to explain a removed table. `:31-34` ("FP16 cannot reach RKLLM's decode rate on any of the twelve models while INT8 can on all twelve") is a claim about a table that no longer exists in this file.
- settles it: the file itself
- action: delete `:21-37` and the blockquote; retitle to "NPU GEMM baseline — INT8, Qwen3.5 4B". If the FP16 reasoning must survive, one sentence in a changelog, not a section.

### medium — "INT8 can on all twelve" — only 9 rows were measured, 3 were blocked

- location: `npu-gemm-baseline.md:32-34`
- values: "on any of the twelve models while INT8 can on all twelve". `model-support.md` last line: *"The 12 PASS rows are covered in `npu-gemm-baseline.md`: 9 measured, 3 blocked."* `todo.md:5-6`: *"Qwen3.5 4B only. The 12-model sweep is dropped."* So 3 of the 12 were never measured on either path, and the doc no longer contains any of the 12 rows.
- settles it: `docs/rknpu/model-support.md` (final paragraph), `todo.md:5-7`
- action: replace with "9 of 12 measured; 3 blocked", or delete the sentence with the FP16 section.

### medium — the quick-mode 4B run disagrees with the headline decode rate by 3.5x and is never mentioned

- location: `npu-gemm-baseline.md:90` (decode 1.53 tok/s, 654 ms/step); no mention of quick mode
- values: `results/quick-4b/report.json` — same checkpoint SHA, same `npu_linears=249`, same clocks — reports **0.4319 tok/s = 2315 ms/step** against the full protocol's 1.5281 tok/s = 654 ms/step. Quick mode has `warmups=0`, so JIT capture and cache warm-up land inside the 8 measured tokens; that probably explains it, but the document does not say so and the 654 ms/step figure has no second independent confirmation.
- settles it: `results/quick-4b/report.json:21-25`, `extra/rockchip/bench_llm.py:59-65`
- action: add one sentence stating quick-mode decode is not comparable (no warm-up) and must not be quoted — the same warning already in `todo.md:120-122`.

### medium — the 4B run started at 63.8 °C, not the 52-55 °C cold start the protocol requires

- location: `npu-gemm-baseline.md:87-93` (parity table presented without qualification)
- values: `report.json` `clocks_start.temperature_c = 63.8`, `clocks_end = 61.0`. `todo.md:87-88`: *"Cold-start revalidation (52–55 °C) for any row published as final. None has it."* The doc presents the 4B as the published baseline without this caveat.
- settles it: `results/qwen35-4b/report.json:40`, `todo.md:87-88`
- action: label as projected/qualification-pending, or add the cold-start caveat to the table caption.

### medium — "the ioctl is 99% of the call" is false at the small shapes in the same table

- location: `npu-gemm-baseline.md:72-73`
- values: `internal/total` from `submit-timing.json`: n=32 **82.1%**, n=256 **91.9%**, n=512 **94.2%**, n=1024 **96.2%**, n=2048 **97.7%**, n=4096 **98.8%**, n=8192 **99.3%**, n=9216 **99.4%**. "99% at every shape" only holds from n≥4096 (98.8%+); it is 82-96% below that.
- settles it: `results/submit-timing.json`
- action: replace "at every shape" with "at n≥4096 (98.8-99.4%); 82-96% below that".

### medium — `tile_columns: 24384` is a correctness workaround, not a speed lever

- location: `npu-gemm-baseline.md:109-110` (listed among the levers that are "the blueprint" for closing the host gap)
- values: the reference's own comment at `~/oNPU/onpu/rockchip_options.py:382-383` reads *"Keep the K2560 vocabulary head below the 8192-column/core transition that corrupts the next cached FP16 operation in the primitive check."* It is a corruption avoidance, not a measured gain, and it applies to the vocab head only.
- settles it: `~/oNPU/onpu/rockchip_options.py:382-385`
- action: move it out of the lever list and note it as a correctness constraint.

### medium — "4 domain switches per forward against 183" is not derivable and is inconsistent with the code

- location: `npu-gemm-baseline.md:116-118`
- values: 4.2048 GB of packed weights against `DOMAIN_BUDGET = 3 << 30` (3 GiB, `npu_linear.py:32`) gives **2 domains**. `_Domains.place` (`npu_linear.py:46-54`) only ever appends a new domain, so the domain index is monotonically non-decreasing along the walk → **1 switch per forward**, not 4. No log records either 4 or 183.
- settles it: `extra/rockchip/npu_linear.py:32,46-54`; no artifact for the switch counts
- action: re-measure with an ioctl/domain counter, or delete both numbers. `todo.md:110-113` repeats them, so both need the same fix.

### medium — the RAM-wall table does not add up, and two of its numbers have no source

- location: `npu-gemm-baseline.md:155-159` (table), `:161-163`
- values: `14 − 11 = 3`, but the table's last row says **available 2 GB**. The table faithfully copies `results/4b-load-profile.log` (`avail= 2GB`), so the defect is in the measurement, not the transcription — but it is presented as a reconciled budget. Separately, "Peak became 12.4 GB instead of 19 GB" matches no log (`4b-defer-test.log` shows `self_shmem=11GB avail=1GB`), and "Five OOM kills" contradicts `results/oom-stack.log` (**4** `Out of memory: Killed process` lines) and `sweep-run3.log` (*"Four OOM kills"*).
- settles it: `results/4b-load-profile.log`, `results/4b-defer-test.log`, `results/oom-stack.log`, `results/sweep-run3.log`
- action: replace the available column with the logged values and add a note that they do not reconcile; delete 12.4/19 or cite a log; fix the kill count to 4 (or 6 if the two `rc=-9` entries in `sweep.log` are counted, and say so).

### low — `8.008 bits/weight` assumes 4096-wide rows; the 4B's rows are 2560

- location: `npu-gemm-baseline.md:36-37`
- values: one fp32 scale per output row gives `8 + 32/k`. For k=4096 that is 8.0078; for the 4B's k=2560 it is **8.0125**. The packed total is 4.2048 GB codes + 0.0046 GB scales, so the conclusion (≈8 bits, same as RKLLM) is unaffected.
- settles it: GGUF `embedding_length=2560`; `npu_linear.py:83-89`
- action: replace with "8.01 bits/weight at k=2560", or say "≈8.008-8.013 depending on row length".

### low — "the cores are not contending for the bus" reads backwards

- location: `npu-gemm-baseline.md:74-75`
- values: one core reaches 11.07 GB/s (32.8% of ceiling), three reach 30.56 GB/s (90.4%). The correct statement is that **one core cannot saturate the bus and three can** — three cores are sharing one 33.79 GB/s bus, which is why the speedup is 2.76x and not 3x. "Not contending" implies independent bandwidth.
- settles it: `results/submit-timing.json` `rows[6]` and `rows[8]`
- action: replace the clause.

### low — GB and GiB are mixed in the domain paragraph

- location: `npu-gemm-baseline.md:114-117`
- values: "4 GiB aperture… 3.76 GB fits, 5.23 GB does not… `DOMAIN_BUDGET = 3 GiB`". 3.76 GB = 3.50 GiB and 5.23 GB = 4.87 GiB, so the boundary claim is consistent — but the reader cannot verify it without doing the conversion, and 3 GiB (3.22 GB) sits 0.54 GB below the stated 3.76 GB fit point with no explanation of why the budget is conservative.
- settles it: `npu_linear.py:25-32` (the comment does explain: 3.906 GiB was tried and reverted)
- action: use GiB throughout and add the one clause explaining why 3 GiB when 3.76 GB fits.

### low — the k=4096 fit rests on 3 points and is not flagged

- location: `npu-gemm-baseline.md:58-61`
- values: `submit-timing.json` `fits[1].n_points = 3` (n=512, 1024, 2560 at k=4096), against 8 points for k=2560. A 3-point two-parameter fit has one degree of freedom; the 27.85 GB/s slope is much weaker evidence than the 31.93 one, yet the doc gives them equal standing and then uses only the stronger one for the floor.
- settles it: `results/submit-timing.json` `fits`
- action: label the k=4096 row as a 3-point fit, and add a k=4096 point at large n (4096/8192) before relying on it either way.

### low — the 0.034 ms intercept exceeds the measured minimum-task time

- location: `npu-gemm-baseline.md:70`
- values: the fit intercept is 0.0342 ms, but the smallest task in the same sweep (`m=1,n=32,k=512,cores=1`) has `total_ms = 0.0245`. A per-submit cost above the cost of the cheapest submit is a fit extrapolation, not a floor. It is defensible (different core count) but the doc states it as a measured constant.
- settles it: `results/submit-timing.json` `rows[0]`
- action: label as "fit intercept (extrapolated from n≥256, cores=3)".

### low — `docs/rknpu/README.md` contradicts the doc on scope and on clocks

- location: `docs/rknpu/README.md` ("9 of the 12 PASS models… measured prefill/decode, RKLLM comparison, and the 3 blocked rows"; "`report.json` records neither clocks nor temperature"; "The 9 measured rows started at 59.2–72.1 °C")
- values: the doc now covers **only** Qwen3.5 4B; `qwen35-4b/report.json` **does** record clocks and temperature (`clocks_start`/`clocks_end`); the 4B started at **63.8 °C**. The README describes a document that no longer exists.
- settles it: `docs/rknpu/npu-gemm-baseline.md`, `results/qwen35-4b/report.json`
- action: update the README index row and the two stale sentences (out of scope to audit further, but it will mislead the next reader).

---

## Summary

**27 findings: 5 high, 16 medium, 6 low.**

The one that would change a conclusion is the first: the document's headline — bold
**"at parity"**, "within 5%", "no device headroom left" — is produced by applying the
best measured slope (31.93 GB/s, k=2560, m=1, n≥4096) to 100% of a 4.2048 GB weight
stream, and by using an assumed 250 submits when the only measured count is 323.
Using the doc's own artifacts and its own second fit, the per-token device floor is
**147.3 ms against the reference's 138.66 ms — 6.2% behind, not level**. That does
not overturn the *direction* of the argument (the matmul is within ~6%, so the 4.2x
decode gap really is host-side), but it does overturn the specific claim that there
is nothing left to win on the device, and it removes the basis for retiring the
kernel-port direction outright. The 52.5% prefill row and the 82.4% k=4096 row in
the same document say the same thing independently.

The second-order conclusion — "the gap is host-side, port the host engine" — survives
all six checks. It is the numbers supporting it, not the conclusion, that need fixing.

**Sections I believe are correct as written:**

- The n-sweep table (`:46-54`) and the fit table (`:56-61`). Every cell reproduces
  exactly from `submit-timing.json`, including the percentages.
- The 4.2 GB and 131.5 ms arithmetic (`:95-98`). Re-derived independently from the
  GGUF metadata and the packing rule: 249 linears, 4.2048 GB, 131.7 ms at 31.93 GB/s.
  The weight figure is right; it is the bandwidth applied to it that is optimistic.
- The device-findings section (`:112-146`) — the IOMMU aperture and 3.76/5.23 GB
  boundary match `npu_linear.py:25-32`; the INT8 per-core plane layout matches
  `_dequantize_source` (`npu_linear.py:186-190`); the 512-multiple K rule matches
  `ops_rockchip.py:108-113`; the packed-args ABI claim matches
  `ClangRenderer.packed_args` and the two hand-written entry points; `_finite_program`
  exists at `ops_rockchip.py:33`.
- "24 of the 4B's 32 blocks are gated-delta-net" — verified from the GGUF tensor
  names (24 blocks carry `ssm_*` and `attn_qkv`; 8 carry `attn_q/k/v`).
- `model.py:741` — the citation is exact; that line is `for t in range(T_pad):`.
- `~/oNPU/benchmark.md:1179` and `:1392` — both citations land on the right rows with
  the right values (1263.09 / 6.48 and 138.66 / 14.95 / 0.61).
- The `DEV=CPU` section (`:183-198`) — cites logs that exist and states the
  root-cause gap honestly.
