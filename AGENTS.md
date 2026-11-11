# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.

## RKLLM benchmark status

**2026-11-10: ChatGLM3-6B decode beats local RKLLM, bit-exact. Of the three
tracked models, LFM2.5-VL and ChatGLM3-6B are qualified in every measured phase;
MiniCPM3-4B decode has reached PARITY and its prefill is the one remaining
deficit (structurally blocked host-side).**

- **LFM2.5-VL 450M — wins both phases** (oNPU 70.13 ms prefill / 67.59 tok/s
  decode vs local RKLLM 103.70 / 54.58). Do not re-convert or re-benchmark LFM
  unless its execution path changes.
- **ChatGLM3-6B — prefill wins (1261.5 vs 1372.5 ms, 1.09x) and decode WINS.**
  soak-v18 (55 °C cold start, n=3 pairs, 5 trials, 64 tokens, every arm started
  at exactly 55 °C, gate checker PASS): oNPU **5.1212 tok/s / 195.3 ms/token** vs
  local RKLLM **5.0630 / 197.5 ms** — paired median ratio **1.0115 (+1.15%)**,
  all 3 pairs ahead (1.0109–1.0129), sd 0.0010; an earlier soak-v17 read +1.72%.
  All 64 `generated_ids` are bit-identical on every trial. Do not re-litigate
  ChatGLM decode: the whole 4.7711 → 5.1212 tok/s path is in
  `~/oNPU/progress.md` §0.3–0.4f; the remaining default-OFF levers are margin.
- **MiniCPM3-4B: decode now WINS; prefill is the remaining blocker.**
  soak-minicpm3-hb-v1 (55 °C cold start, n=3 pairs × 3 trials, 64 tokens, every
  arm at 55 °C, ids bit-exact, gate PASS): oNPU **6.0195 tok/s / 166.1 ms** vs
  RKLLM **5.8260 / 171.6 ms**, paired median **1.0389 (+3.89%)**, decode gap
  −5.5 ms, 312 submits/token. Trajectory: recorded 5.3807 → 5.87 (parity) → 6.02
  (win). The flip lever: **head-blocked MLA decode kernels**
  (`ONPU_MLA_HEAD_BLOCK=4`) — the CPU qk/pv dots re-read their shared K block /
  latent cache once per head (40×); loading it once for 4 heads with 16/8
  explicit accumulators keeps each output's ascending-c/key order (bit-exact),
  cutting qk 8.09→7.46 and pv 8.07→4.11 ms/token (CPU total 29.67→25.12). **Prefill
  is +50.2 ms (ratio 0.9661) behind on the current pre-registered cohort**
  (soak-v5 2026-11-11: oNPU **1477.5 ms** vs local RKLLM **1427.3 ms**, median pair
  delta **+54.7**, sign **0/7**, CI [+36.1,+61.0] ⇒ decisive LOSE; soak-v4 0/9, so
  **16/16 pairs favour RKLLM** on prefill — a real ~3.5 % gap, not drift). The
  trajectory is 2407 recorded → 1850 → 1799 → 1706 → 1684 → **1477.5** ms, i.e.
  −206 ms of genuine progress, so the older "≈−13 %" framing and its
  1683.9/1685.1 figures are **superseded**; do not compare them against the
  recorded 1461.6 ms rival figure, which came from a different cohort (this
  board's rival median moved +16.5 ms between cohorts). Prefill is *host-bound on
  the current stack*: the native prefill NPU phase is only ~462 ms (empirically
  at a local optimum — a 128-row K-split variant was bit-exact but 150 ms
  slower), while CPU MLA (attention now fully fused at 553 ms, projection 255),
  other host (~430 ms) and copies dominate. **Three prefill levers, all
  bit-exact:** `ONPU_MLA_QK_SOFTMAX=1` fuses qk + the exact softmax (−59 ms; the
  padded 13 MB score buffer is never materialised), `ONPU_MLA_FUSED_ATTN=1` also
  fuses the weighted value (−86 ms more; the padded probability matrix is never
  materialised, and both paths call the shared noinline pv row helper),
  `ONPU_MLA_FUSED_QUERY=1` splits the q_b output into padded head-major q_nope
  plus rotated q_rope in one kernel (drops the transpose copy, the q_nope pad and
  the rope kernels). Gates: CPU uint32 equality on the fused outputs and 64/64
  generated ids identical to the recorded reference on hardware. Other
  default-OFF margin: `ONPU_MLA_PROJ_BLOCKED=1` (projection −35.5%) and
  `ONPU_MLA_HEAD_BLOCK=4` (decode kernels + prefill rows path, TTFT −3.4%).
  Root cause found (2026-11-10 evening, session 5; progress.md +
  `~/oNPU/analysis/minicpm3/prefill/HOST_GAP_STRATEGY.md`): direct measurement of the
  stock RKLLM runner (ioctl `hw_elapse_time` shim) shows vendor prefill = 1468 ms
  wall = ~780 ms NPU/core + ~690 ms host vs oNPU 462 NPU + 1222 host — the gap is
  host-side, and it is that RKLLM runs MLA attention QK+PV as FP16 CNA matmuls on the
  NPU (250 ms device, softmax on CPU) while oNPU runs exact FP32 attention on the CPU
  (553 ms). The "vendor tiles m=128" premise is refuted: the executed vendor prefill
  is 2x64-row A16/C4 M-chains (14+50), weights stream twice exactly like oNPU, and
  `ONPU_SDK_DEEPK` is a measured wash/regression.
  **The vendor-style FP16 NPU attention prefill lever was implemented, measured
  faster, and is now CLOSED as quality NO-GO (2026-11-10 evening, session 6).**
  Flag `ONPU_ATTN_FP16_NPU=1` (12-file patch set, default OFF, manifest
  `~/oNPU/analysis/minicpm3/prefill/attn-npu/wiring/merged/MANIFEST.md`, status
  now CLOSED/CUT): unabsorbed MLA with grouped FP16 NPU expansion/QK/PV +
  exact-order CPU softmax. Speed is a real win: prefill TTFT
  **1289.2/1305.4/1319.7 ms (median 1305.4) vs in-session RKLLM
  1440.1/1462.5/1579.1 (median 1462.5)**, oNPU baseline 1697.3 ms, decode
  unchanged 6.02 tok/s. But competitor parity fails: on a wikitext prompt
  (128-token greedy continuation) the FP16 build gives 4 distinct tokens/128,
  max run 123, a period-1 attractor, while local RKLLM W8A8 gives 36 distinct,
  max run 2, coherent English prose; prompt-independent (r2 reproduces it).
  Root cause is certified as inherent **systematic FP16-rounding accumulation**,
  not a wiring/plan bug: device plans/packs correct (QK/PV 1e-7..1e-8, softmax
  pack bit-identical, expansions 1e-7 vs checkpoint), operands fp16-class,
  rounding-point emulation predicts 2.7e-4 per block, flag-OFF is bit-identical
  to the FP32 reference, an iid N(0,1.9e-3) injection control keeps 64/64
  reference ids, and a CPU-emulated FP16 chain also degenerates. Decision: exact
  FP32 attention retained; no reference/soak minted; the speed result is kept as
  a documented CUT/provable-negative. Evidence:
  `~/oNPU/analysis/minicpm3/prefill/attn-npu/wiring/gate/debug/FINAL_VERDICT_task26.md`
  + `.../verify/QUALIFICATION.md` §10-§12. Bit-exact prefill margin still
  available: `ONPU_PREFILL_QUANT_REUSE` −11 ms,
  `ONPU_MLA_VALUE_ROWMAJOR`+`ONPU_INT8_ROW_FOLD` ~−35 ms, row-paired attention
  kernel −95 ms (`prefill/attn3/`).
  Next prefill targets (ranked in
  `~/oNPU/analysis/minicpm3/prefill/`): f16→f32 residual cast fold (~39 ms, MUST
  find why the layer output is half first), E_* layout reshapes (~55 ms),
  swiglu pack/quantize/dequantize fusions; SDK M128/A16-C4 tiling would still be
  needed for the NPU phase.
- Fair comparison is W8A8 only: no speculative draft model, no lower precision,
  no changed algorithm basis. "Bit-exact" means identical `generated_ids`, not
  merely close logits.
- CPU tests use `DEV=CPU ROCKCHIP_TEST=0`. Hardware benchmarks are single-owner:
  do not overlap them with tests, compiles, tree copies or other benchmarks.
  - **Pin clocks before any hardware benchmark** (needs root, manual step):
    `sudo ~/oNPU/scripts/prepare_clocks_rk3588.sh`; verify CPU4/6 == 2256000 and
    `/sys/class/devfreq/fdab0000.npu/min_freq` == 1000000000. Without it CPU6
    oscillates 408 kHz–2.256 GHz and invalidates every A/B.
  - The NPU still drops to 800 MHz under sustained load even with min==max: the
    rknpu driver bypasses devfreq and there is no npu cooling device. That costs
    ~0.3–0.5%, so do not discard a run for it — bucket trials by finishing clock
    and compare within a bucket.
  - **The box is I/O- and memory-fragile during soaks.** The checkpoint is a
    file-backed GGUF (~6.6 GB) and the benchmark holds ~14 GB RSS; any teammate
    pytest/compile (which also writes the ~13 GB tinygrad cache.db), tree copy or
    other memory-heavy job evicts the checkpoint and the generator stalls in
    D-state (`folio_wait_bit_common`) for minutes. Check `ls ~/oNPU/.hw-lock`
    before heavy work; the Lead owns hardware windows.
  - **Pre-warm the box before a cold-start soak.** If the first arm starts below
    the target temperature while its paired arm starts at it, the >2 °C
    start-gap gate rejects the pair (soak-v17 pair 1). Run one short warm-up
    benchmark first so every arm begins at the target.
- Pending changes live in worktrees, not the main tree. Merge only after CPU
  goldens pass. Session-3 state: the winning decode stack is committed on branch
  `codex/decode-parity-20261110` (worktree `/home/orangepi/tg-commit`, base
  `3a1b2e1c9`, 10 commits / 170 files, exported to
  `~/oNPU/patches/tg-main-tree-20261110-commits/` and replay-validated); the
  benchmark copy `~/tg-decode-all` carries the same files; the dirty main
  working tree is untouched. Map + evidence:
  `~/oNPU/analysis/session3/GIT-COMMITS.md`.
- Decode is a **weight-streaming** problem: ~5.98 GB of INT8 weights per token
  against a 33.79 GB/s DRAM ceiling. Do not chase NPU tiling. But note the
  session-3 correction: the remaining gap is **not only CPU+COPY host work** — a
  large part was the driver's per-submit IOMMU detach/attach, and the per-submit
  fixed cost is per **ioctl**, not per task.

### The session-2/3 levers on the winning ChatGLM stack (all bit-exact)

In order of decision impact (`~/oNPU/progress.md` §0.3–0.4f has the full table):

1. **Block-major weight placement — the decisive one (~2.4 ms).** Every blocking
   NPU submit whose `iommu_domain_id` differs from the previous submit's calls
   `rknpu_iommu_domain_get_and_switch` → `iommu_detach_device` +
   `iommu_attach_device` (+ smmu TLB work + a `pr_info` printk), synchronously
   inside the next submit. The shipped two-pass weight prepare (joint
   projections in one pass, standalone ones in a second) put a layer's four
   submits in alternating domains. Measured execution order per layer was
   `[0,1,0,0,1,1]` = **111 domain switches/token**; dmesg's ring buffer was
   entirely full of "switch iommu domain" lines. `ONPU_BLOCK_DOMAIN_ORDER=1`
   prepares each block's standalone linears right after its joints so the
   sequential domain cursor walks block-major: **111 → 4 switches/token**, NPU
   phase 192.37 → 189.94 ms. Packed weight bytes are identical — only which
   domain a tensor lives in changes. *Detection:* count domain changes in the
   recorded label list (`ROCKCHIP:` / `ROCKCHIP:1:` / `ROCKCHIP:2:` in execution
   order), or `dmesg | grep -c "switch iommu domain"` after a run.
2. **Submit merging (~1.5–2 ms).** The fixed cost (~34.6 µs) is paid per ioctl,
   not per task: merging two 3-task submits into one ioctl saves 23–36 µs with
   byte-identical output. `ONPU_MERGE_SUBMITS=2` patches the earlier plan's
   per-core terminating command tail into a PC jump to the later plan's body and
   copies descriptors core-major into one array: 171 → 113 submits/token.
3. **Prepared INT8 QKV dequant+partial-rotary+KV-cache kernel (~0.6 ms).** One
   kernel replaces three host launches. It also fixed a lazy-weight trap: under
   `--defer-weight-realization` the fused QKV bias `contiguous()` densified a
   lazy gather inside every token (label `CPU:E_36_2_32_2`, 1.042 ms/token) —
   realise it once in `RockchipPreparedQKV.__init__`.
4. **Fused decode FFN/down host chain (~0.9 ms).** The raw SwiGLU pack computes
   the activation in-register with the exact shipped arithmetic and writes the
   down grouped payload in place, on the 4-worker pool.
5. **Fast replay validation (~0.4 ms).** `ONPU_REPLAY_FAST_VALID=1` reuses a
   successful full guard verdict while an epoch is unchanged (the debug variant
   runs the full walk every token); `valid()` measured 0.425 → 0.0183 ms/token.

Shipped env for the winning stack (all bit-exact; the runner scripts set these):

```
ONPU_MERGE_SUBMITS=2 ONPU_WIDE_RAW_SWIGLU_DECODE=1 ONPU_DECODE_FUSED_QKV_ROPE=1 \
ONPU_REPLAY_FAST_VALID=1 ONPU_BLOCK_DOMAIN_ORDER=1 \
ONPU_WIDE_RAW_SWIGLU=1 ONPU_NPU_DOMAINS=3 ONPU_NPU_DOMAIN_BYTES=3221225472 \
ONPU_VECTOR_SIGMOID=1 ONPU_CPU_DECODE_POOL=1 ONPU_HOST_POOL_WFE=1 \
ONPU_EXP2_FAST_PATH=1 ONPU_GROUPED_DECODE_DOWN=1 ONPU_CPU_RMS_ALL_ROWS=1 \
ONPU_INT8_REUSE_ACTIVATION=1 ONPU_INT8_DECODE_NARROW_CBUF=1
```

`~/oNPU/scripts/run_soak_coldstart.py` sets all of them; use it as the reference
config. WFE is mandatory with the decode pool.

### What produced the wins (6 bit-exact levers, 4.7711 → 5.1212 tok/s, +7.3%)

- **Read the emitted C and remove redundant/scalar work** (session 1, 3 of 5
  wins): group=16 fused GQA (+2.43%, 16× KV re-read), vectorised Q8 embedding
  (+1.82%), vectorised gated activation (+1.20%).
- **Decompose a null result by label** (session 1): `ONPU_CPU_DECODE_POOL=1`
  measured +0.45% (noise) until per-label profiling showed +1.68 ms on attention
  and −3.48 ms elsewhere from spinning idle workers; adding WFE kept the win.
- **Prove exactness, then toggle the flag** (session 1): fast exp2 (+0.47%).
- **Profile the driver, not the kernels** (session 3): the IOMMU domain switch
  above was invisible in kernel labels and dominated the remaining gap.
- **Count ioctls, not tasks** (session 3): submit merging by dependency analysis.
- **Zero wins** came from threaded per-core dispatch (3×8192 tasks cost +5.8%
  NPU throughput), removing the pool's main-thread pin (−9.6%), or "overlap the
  CPU behind the NPU" (oNPU already overlaps more than RKLLM).

### Technique notes

- **Report a gap as TWO tables, competitor-in-both**: TABLE 1 host/CPU time (oNPU rows vs RKLLM rows
  in the same table) and TABLE 2 device/NPU time. Never merge them — a device win then hides a host
  loss, which is exactly how "we are 1.67x faster on the NPU phase" coexisted with losing the request.
  Rows that are zero on one engine are the *finding*: mark them `on CPU` / `absorbed to device` instead
  of leaving a blank. Reconcile `sum(labels)` against the clean wall (~1–2 %, here 1450.6 vs 1460.8/1465.4).
- **Fill every unknown by measurement or mark it `UNKNOWN`; never with an inference dressed as a
  measurement.** A rival figure from a different cohort is not a comparison — this board's rival median
  moved +16.5 ms between cohorts. Stating the derivation formula is part of the measurement (e.g.
  `gap = t0(i+1) − t1(i)` on the same `core_mask` stream for the rival's host interval).
- **Tag every forecast row and let the tag set the total**: `MEASURED` and `COUNTED` (bytes / passes /
  instructions / calls removed, computed from the captured emitted C) count; `HYPOTHESIS` and
  `WITHDRAWN` count **0**. "Phase X is at 33 % of its floor, so 70 % saves Y" is `WITHDRAWN` — dividing
  a floor by a target efficiency re-describes the problem. Keep it as a diagnosis line marked 0. If the
  honest sum does not reach a win, say the path is bounded at `+Y behind`; do not add a withdrawn row
  back to make the total close.
- **Check the issue budget before proposing register work.** Census the inner loop from the asm:
  12 loads / 32 FMA = 0.38, and against A76's 2 loads/cycle vs 2 FMLA/cycle that is ~38 % of the issue
  budget ⇒ loads are **not** the wall, so "reduce loads" is already refuted. The two wins that ever
  landed in the projection each *deleted* a real instruction (pointer addressing, then `fcvtl`), and
  `fcvtl` was the last one visible.
- **Compute an achieved rate before optimising a copy-shaped kernel.** `E_40_128_64_4` moves 10.49 MB
  in 0.431 ms = **24.3 GB/s** and `E_128_640_4` 2.62 MB in 0.094 ms = **28.0 GB/s**, against a measured
  single-thread ceiling of **23.9 GB/s** ⇒ 102–117 % of ceiling ⇒ **memory-bound and irreducible**;
  record it as a memory floor (the whole `E_*`/`r_toks` set is 102.94 ms). Measure that ceiling with
  **disjoint** chunks per thread: a multi-thread test where every thread memcpy's the *same* buffers
  measures contention, and its rate *falls* with more threads — that is the tell.
- **A per-worker scratch the size of an L1 can mislead.** `float batch_values[4][4096]` is 64 KB,
  exactly one A76 L1D, which looks cache-bound — but the kernel streamed raw at only **4.0 GB/s** against
  23.9 GB/s, so it was **issue-bound** on ~40 vector ops/element (`exp2` + a true `fdiv`); removing the
  scratch would have *added* a 0.208 ms re-read for no instruction saving. Check the streaming rate first.
- **A flag can be plumbed in and never run — check the whole conjunction, and treat a dead fast path as
  free money.** The batched pack needed four flags at once (`ONPU_SWIGLU_BATCHED_PACK`,
  `ONPU_SWIGLU_CHANNEL_SCRATCH`, `ONPU_EXP2_FAST_PATH`, `ONPU_EXP2_SCALE_BITS`); the shipped env set two,
  so it was False in every benchmark ever run. Prove liveness by *diffing the emitted source* (here
  27,681 B / 0 `exp2_batch4` calls → 29,877 B / 5 calls), then ABBA (**−5.91 ms**, sign 2/2, pack label
  76.30/76.65 → 71.55/72.27) and prove byte-equality (623,616-byte payload identical, scales identical).
- **A flag can silently disable another lever.** `ONPU_SWIGLU_PACK_STATIC_LAYOUT=1` measured
  **+218…230 ms** in both orders because it changed the pack's output layout, the residual fold's
  registration check then failed and the fused kernel vanished (`dequantize_residual_add` 32.15 → 0.00,
  unfused `E_28toks` 11.59 → 36.48). Diff the **label set** between arms, not just the total.
- **Verify the profiled label really maps to the kernel you are reading.** `rockchip_int8_swiglu_pack`
  and `rockchip_int8_ffn_pack` are different kernels; the second appears **zero times** in the profile.
  Three rounds of accounting on the wrong one had to be retracted.
- Dump the emitted C before guessing: `~/.cache/tinygrad/cache.db`, table
  `compile_clang_obj_arm64_native_24` (look up a profiled label).
- Prove exactness before timing, including fallback ranges: per-kernel
  `max|diff|=0` plus identical `generated_ids` across every trial.
- Never reject a run for the 800 MHz NPU drop; bucket by finishing clock.
- Do not trust the ioctl tracer's NPU timings (`hw_elapse_time_ns` cannot time a
  GEMM); its submit counts are faithful.
- Per-label profiling adds ~1.5–2 ms to the wall: use unprofiled runs for timing
  and profiled runs for attribution only.
- Verify profile normalisation: the arms runner once divided delta sums by the
  wrong replay count (1.3387× inflation) before it was caught.
