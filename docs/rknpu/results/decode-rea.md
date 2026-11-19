# Decode: the 311 ms per token, line by line, and what to change

*Citations to `split/FINDINGS.md`, `doc-audit.md` or `audit-refdoc.md` resolve via git history; those files were deleted in the docs cleanup because their conclusions were superseded.*

Subject: Qwen3.5-4B, 128-token prompt, 64 greedy tokens, `extra/rockchip/bench_llm.py --npu`
(host CPU with every `nn.Linear` offloaded to the NPU GEMM), clocks pinned. This document is
about **decode only**; prefill has its own decomposition in `compare-ours-vs-ref.md`.

The question it answers: the reference decodes at 6.52 tok/s and we decode at 3.22 tok/s, while
the NPU device time is 143.0 ms/token for us and 137.1 ms/token for them. The device is at
parity. **Every millisecond of the decode gap is host-side**, and this document locates those
milliseconds and ranks the fixes.

Everything below is quoted from an artifact, with the path and field. Where a number is a
difference of two measured quantities, or a projection, it says so.

---

## 0. The two measurements being compared

| quantity | ours | reference | artifact |
| --- | --: | --: | --- |
| decode rate | **3.2186 tok/s** (310.7 ms/token) | **6.5233 tok/s** (153.3 ms/token) | `threadpool-3plus3.log:43`; `ref-profile/report.json` `median_decode_tps` |
| protocol | 128 prompt, 64 generated, 3 warm-ups + 3 measured | 128 prompt; 7 timed decode steps carry the 1,421 submit count (203/step), the profile aggregates 46 | `threadpool-3plus3.log`; `ref-profile/report.json` `native_execution.decode.runs=46` |
| NPU submits/token | **307.0** | **203.0** | `threadpool-3plus3.log:27`; `ref-profile/report.json` `captured_calls.decode` = 151 + 52 |
| NPU tasks/token | **825.0** | not recorded | `threadpool-3plus3.log:27` |
| device (blocked in submit) | **143.0 ms/token** | **137.1 ms/token** | same two artifacts (`npu 143.0 ms/token`; `native_execution.decode.profile`) |
| host (rate − device) | **167.7 ms/token** | **16.2 ms/token** | difference of the two rows above |

Our decode has been measured at 2.74-3.22 tok/s (311-365 ms/token) across recent arms;
`attnpool-3plus3.log:47` is the low end (2.7362 tok/s, 365.5 ms/token, npu 144.8) and
`threadpool-3plus3.log` the high end (3.2186). The 3.2186 arm is used as the reference point
below because it is the best measured and because its profile exists (`prof-merged.log`).

The reference figure is stable across four independent runs on two trees and this host:

| artifact | decode tok/s | ms/token |
| --- | --: | --: |
| `ref-profile/report.json` trials 4-7 | 6.52-6.55 | 152.7-153.4 |
| `ref-phase2/report.json` trials 3-4 | 6.4665 / 6.4857 | 154.6 / 154.2 |
| `ref-kernels/report.json` trials 2-4 | 6.37 / 6.44 / 6.45 | 157.0 / 155.3 / 155.0 |
| `phase/report.json` (64-token protocol, this host) | 6.372 / 6.366 / 6.436 | 157.0 / 157.1 / 155.4 |

---

## 1. The reference's decode profile, per kernel

`ref-profile/report.json` → `native_execution.decode.profile`. There are **39 entries**, and the
`runs` field is 46, so each entry divides into a per-token figure. The profile is present, not
absent; it is a real per-kernel breakdown, and it reconciles with the call counters:
`captured_calls.decode` records 1,086 CPU programs and 203 ROCKCHIP stores, the profile records
1,086.00 CPU calls/token and 203.00 ROCKCHIP calls/token. **One kernel per submit, 1:1.**

| kernel                                | calls | calls/token |   ms | ms/token |
| ------------------------------------- | ----: | ----------: | ---: | -------: |
| ROCKCHIP:rockchip_int8_1_18432_2560   |  1472 |       32.00 | 2225.49 |   48.380 |
| ROCKCHIP:rockchip_int8_1_12352_2560   |  1104 |       24.00 | 1139.76 |   24.777 |
| ROCKCHIP:rockchip_int8_1_2560_4096    |  2990 |       65.00 | 1040.16 |   22.612 |
| ROCKCHIP:1:rockchip_int8_1_24384_2560 |   460 |       10.00 | 903.34 |   19.638 |
| ROCKCHIP:1:rockchip_int8_1_2560_4096  |  1426 |       31.00 | 521.87 |   11.345 |
| ROCKCHIP:rockchip_int8_1_10240_2560   |   368 |        8.00 | 311.62 |    6.774 |
| CPU:rockchip_cpu_gated_delta_scan     |  1104 |       24.00 | 191.55 |    4.164 |
| ROCKCHIP:rockchip_int8_1_2560_1024    |  1012 |       22.00 | 101.10 |    2.198 |
| CPU:E_2_16_32_4                       |  2208 |       48.00 |  97.30 |    2.115 |
| CPU:rockchip_int8_swiglu_pack         |  1472 |       32.00 |  81.47 |    1.771 |
| CPU:E_8_32_4_4                        |  1104 |       24.00 |  73.02 |    1.587 |
| ROCKCHIP:1:rockchip_int8_1_2560_1024  |   460 |       10.00 |  46.18 |    1.004 |
| CPU:E_640_4                           |  4508 |       98.00 |  30.99 |    0.674 |
| CPU:E_1024_4                          |  2208 |       48.00 |  29.07 |    0.632 |
| CPU:rockchip_dequantize               |  7866 |      171.00 |  25.34 |    0.551 |
| CPU:r_16_32_4                         |  2208 |       48.00 |  22.12 |    0.481 |
| CPU:rockchip_quantize                 |  4922 |      107.00 |  21.39 |    0.465 |
| CPU:rockchip_cpu_attention            |   368 |        8.00 |  20.45 |    0.445 |
| CPU:E_8192_4                          |  1104 |       24.00 |  18.86 |    0.410 |
| CPU:E_16_64_4                         |   368 |        8.00 |  17.83 |    0.388 |
| ROCKCHIP:1:rockchip_int8_1_4480_2560  |    46 |        1.00 |  17.38 |    0.378 |
| CPU:E_62080_4                         |    46 |        1.00 |  17.08 |    0.371 |
| COPY:ROCKCHIP->CPU                    |  6946 |      151.00 |  13.74 |    0.299 |
| COPY:ROCKCHIP:1->CPU                  |  2392 |       52.00 |   9.78 |    0.213 |
| CPU:r_640_4                           |  2990 |       65.00 |   8.98 |    0.195 |
| CPU:E_16_32_2_4                       |   368 |        8.00 |   6.40 |    0.139 |
| CPU:E_8_4                             |  2208 |       48.00 |   4.76 |    0.103 |
| CPU:E_8192_3                          |  1104 |       24.00 |   3.99 |    0.087 |
| CPU:gpt2_argmax                       |    46 |        1.00 |   3.92 |    0.085 |
| COPY:CPU->ROCKCHIP                    |  6946 |      151.00 |   3.68 |    0.080 |
| CPU:r_32_32_4                         |  1104 |       24.00 |   3.66 |    0.080 |
| CPU:E_4_32_2_4_2                      |   368 |        8.00 |   1.77 |    0.039 |
| CPU:r_16_64_4                         |   368 |        8.00 |   1.39 |    0.030 |
| COPY:CPU->ROCKCHIP:1                  |  2392 |       52.00 |   1.28 |    0.028 |
| CPU:E_1040_4                          |  2944 |       64.00 |   0.91 |    0.020 |
| CPU:E                                 |  7084 |      154.00 |   0.59 |    0.013 |
| CPU:r_4_64_4                          |   368 |        8.00 |   0.40 |    0.009 |
| CPU:E_272_4                           |  1472 |       32.00 |   0.17 |    0.004 |
| CPU:cpu_packed_q8_embedding           |    46 |        1.00 |   0.13 |    0.003 |
| TOTAL                                 | 77970 |     1695.00 | 7018.92 |  152.585 |

Rollup by class (same artifact):

| class | calls/token | ms/token | share of 152.6 |
| --- | ----------: | -------: | -------------: |
| NPU INT8 GEMM | 203.00 | **137.106** | 89.9% |
| elementwise `E_*` | 435.00 | 6.569 | 4.3% |
| GDN scan | 24.00 | 4.164 | 2.7% |
| swiglu pack | 32.00 | 1.771 | 1.2% |
| quant / dequant | 278.00 | 1.016 | 0.7% |
| reduce `r_*` | 153.00 | 0.794 | 0.5% |
| copies | 406.00 | 0.619 | 0.4% |
| attention | 8.00 | 0.445 | 0.3% |
| sampling (`gpt2_argmax`) | 1.00 | 0.085 | 0.1% |
| elementwise `E` | 154.00 | 0.013 | 0.0% |
| embedding | 1.00 | 0.003 | 0.0% |
| **total** | **1695.00** | **152.585** | |

**Fraction of the decode measurement explained: 152.585 / 153.29 = 99.5%** (153.29 = 1000/6.5233).
The remaining 0.8 ms covers 1,695 kernel launches, so the reference's per-launch cost is **at most
0.47 µs** — their replay does not re-enter a Python executor per program.

`ref-phase2/report.json` (25 runs) reproduces this within 0.6%: total 153.519 ms/token, NPU
138.022, CPU 14.860, copies 0.590.

Two independent samples of the same profile disagree on the *CPU* rows and agree on the NPU rows:

| artifact | decode tok/s | profile total | NPU | elementwise | reduce | scan | attention |
| --- | --: | --: | --: | --: | --: | --: | --: |
| `ref-profile` | 6.5233 | 152.585 | 137.106 | 6.569 | 0.794 | 4.164 | 0.445 |
| `ref-phase2` | 6.4761 | 153.519 | 138.022 | 6.652 | 0.794 | 4.127 | 0.440 |
| `ref-kernels` | 6.4396 | **200.483** | 144.028 | **24.946** | **14.537** | **10.692** | **2.711** |

`ref-kernels` is the same configuration (identical `placement`, same 435/153/24 calls per token)
yet its *per-call* CPU times are 2.6-18x larger, and its own decode rate did **not** move
(6.4396 tok/s = 155.3 ms/token, less than its own 200.5 ms profile total). So that run cannot be
read as a serial budget: at least 45 ms of what it profiles did not extend its critical path.
This is the same "the profile is not one replay" defect documented for prefill in
`compare-ours-vs-ref.md`, and it has the same consequence: **the reference's CPU numbers are a
bracket — 14.9 ms (dispatch wall, pool working) to 55.8 ms (sum of serial work) — not a single
value.** Both ends of that bracket are used in §3.

---

## 2. Our decode budget

Four numbers are measured, and three more are differences between them. No estimate is smuggled
in.

| phase | ms/token | basis | artifact |
| --- | --: | --- | --- |
| NPU device (blocked inside `RKNPU submit`) | **143.0** | measured | `threadpool-3plus3.log:27` `npu 143.0 ms/token` |
| GDN scan (`cpu_gated_delta_scan`, 24 calls) | **27** | measured minus dispatch | `prof-merged.log` `graph_rows` decode rows: 0.030 s for 24 calls, minus 24 x 0.112 ms |
| all other CPU kernels (653 programs) | **46** | remainder | `kernel_s 0.22 s` minus qz_row ~0.147 minus scan ~0.027 |
| `qz_row` non-submit (quantise + 2 cache syncs + dequant) | **~4** | measured minus measured | qz_row runtime ~0.147 (its 0.180 pm_exec wall minus 34 ms of dispatch) minus 0.143 submit |
| **executor dispatch** (`pm_exec.rewrite`, 984 programs) | **110** | measured | `prof-merged.log` `bookkeeping_s` = 0.11 s |
| instrumented total | **330** | measured | `prof-merged.log` `loop_s` = 0.33 s |
| uninstrumented total | **311** | measured | `threadpool-3plus3.log` 3.2186 tok/s |

The profiler inflates the step by 19 ms (330 vs 311, 6%), so the dispatch figure is an upper
bound and the per-phase rows sum to the instrumented total, not the uninstrumented one.

Supporting details, all from the same `graph_rows` decode rows (`n=255`, median shown):

| kernel | calls | wall s | us/call | note |
| --- | ----: | -----: | ------: | --- |
| `qz_row` | 307 | 0.180 | 586 | fused INT8 Linear: quantise, 2 cache syncs, 1 submit, dequantise |
| `cpu_gated_delta_scan` | 24 | 0.030 | 1250 | |
| `E_131072_4` | 72 | 0.020 | 278 | |
| `E_640_4` | 97 | 0.020 | 206 | |
| `E_2_16_32_4` | 48 | 0.010 | 208 | |
| `E_1024_4` | 72 | 0.010 | 139 | |
| `E_62080_4` | 1 | 0.010 | 10000 | **one call, 5-15 ms, against the reference's 0.371 ms** |
| `r_640_4` | 65 | 0.010 | 154 | |
| `r_16_32_4` | 48 | 0.010 | 208 | |
| `E_3072_3`, `E_8_4`, `E_8_32_4_4`, `E_8192_4`, `exp2_scalar` | 136 | ~0 | ~0 | below the 10 ms resolution |
| programs outside the top-14 list | 114 | — | — | `programs` = 984 |

Caveats that belong with these numbers:

- The profiler rounds each entry to 0.01 s, so **every row except `qz_row` carries ±5 ms**. That
  is why the group total in the budget table is a *remainder* rather than a sum of rows; the
  remainder (46 ms) is the trustworthy figure, the row values are only for ranking.
- `kernel_s` (0.22 s) is time inside the kernel callable; `bookkeeping_s` (0.11 s) is
  `pm_exec.rewrite` around it. An earlier FINDINGS correction notes that `pm_exec.rewrite`
  includes kernel execution for the *default* `exec_kernel` path, but
  `profile_graph_dispatch.py:118` subtracts the `rt(...)` time explicitly, so the split is valid
  here.
- The submit counter is separate and exact: `stats[0]` accumulates `clock_gettime` around the
  `ioctl` in the generated C (`npu_linear.py:472-478`), read by `bench_llm.py:154`. 143.0 ms is
  time blocked there, device execution included, host work excluded.

**Per-program economics.** 984 programs cost 110 ms of dispatch = **112 µs each**. The 653
non-scan, non-Linear programs cost 46 ms of work = **70 µs each**. So in our decode *the launch
costs more than the kernel it launches*. That single ratio is the whole story of this document.

**Per-submit economics.** 143.0 ms / 307 submits = **0.466 ms blocked per submit**. The whole
program wall is 0.586 ms (`qz_row`, 586 µs), and of the 0.120 ms difference about 0.11 ms is
executor dispatch — the same 112 µs per program as everywhere else — and only ~0.01 ms is the C
body: quantise 2560 values, two `sync_mem` calls, the submit syscall, the dequant of 4096 int32.
An isolated probe of that same submit path agrees: `submit-timing.json` `host_ms` is 4.4-5.1 µs
across 17 shapes, where `bench_submit.py:126` defines it as total minus median device time. So
the per-submit host charge is neither the wrapper nor the ioctl; it is one more trip through the
executor. For the reference: 137.1 / 203 = **0.675 ms blocked per submit**. Our submits are
*cheaper* on the device than theirs; there are merely more of them, and 307 x 0.466 = 143.0
against 203 x 0.675 = 137.1.

**Why 307 rather than 233.** `baseline-recheck/report.json` records `npu_linears: 233`, and the
per-token submit count decomposes exactly (arithmetic that reproduces the measured 307):

| component | submits | why |
| --- | --: | --- |
| one per `nn.Linear` | 233 | `baseline-recheck/report.json` `npu_linears` |
| `ffn_down`, k=9216, split into 3 K-tiles | +64 | `npu_linear.py:722` `K_TILE = 4096`; AGENTS.md: k>4096 is a hard wall for this recipe |
| `lm_head`, n=248320 at TILE=24576 | +10 | `TILE = 24576`, `npu_linear.py:720-721` |
| **total** | **307** | matches `threadpool-3plus3.log` `307.0 submits/token` |

How the reference reaches 203 for the same model is **not** established by this artifact, and
nothing below depends on it. Its profile contains no k=9216 submit either (its largest K is
4096), so the K-split is not what separates 307 from 203, and the two counts produce the same
device time anyway: 307 x 0.466 = 143.0 against 203 x 0.675 = 137.1. The submit count is a
consequence of `NPULinear`'s tiling (`npu_linear.py:720-773`), not a lever.

---

## 3. Where the difference is, class by class

Ours are wall times. The reference has two columns because §1 established its CPU per-call times
are either a pool dispatch wall or 4x serial work: **`ref wall`** is the pool-working profile
(`ref-profile`), **`ref serial`** is the same class in `ref-kernels`, which is an upper bound on
the work, not on the critical path.

| class | ref wall | ref serial | ours | delta (vs ref wall) | source of ours |
| --- | --: | --: | --: | --: | --- |
| NPU INT8 GEMM (203 vs 307 submits) | 137.10 | 144.03 | **143.0** | **+5.9** | `stats[0]`, exact |
| GDN scan (24 calls both) | 4.16 | 10.69 | **27** | **+22.8** | profile, `kernel_s` remainder |
| attention (8 calls both) | 0.44 | 2.71 | (inside the 46) | ~0 | 8 programs, below resolution |
| elementwise `E_*` + `E` + reduce `r_*` | 7.38 | 39.49 | **46** | **+38.6** | `kernel_s` remainder (see note) |
| quant + dequant + swiglu pack | 2.79 | 2.83 | **~4** | **+1.2** | `qz_row` runtime minus submit |
| copies | 0.62 | 0.64 | inside `qz_row` | ~0 | cached in the fused C |
| embedding + sampling | 0.09 | 0.09 | (inside the 46) | ~0 | profile |
| **executor dispatch** | **0.8** (1695 launches) | 0.8 | **110** (984 launches) | **+109** | `bookkeeping_s`, exact |
| **total** | **152.6** | **200.5** | **330** instrumented / **311** not | **+177 / +158** | |

Note on the "46 ms" row: it is a remainder, not a sum of rows. `kernel_s` = 0.22 s covers every
program that is not `qz_row` and not the scan, including attention, embedding, sampling and the
114 programs the 14-entry `top_kernels` list does not name. Its individual rows in §2 are rounded
to 10 ms and cannot be added reliably; the remainder is the number to trust.

What that says, in order of size:

1. **Dispatch, +109 ms.** We run *fewer* programs than the reference (984 against 1,289 =
   1,086 CPU + 203 NPU) and pay ~240x more per program: 112 µs against ≤0.47 µs. Program count is
   not the problem and reducing it is not the fix; the per-program cost is.
2. **The elementwise/reduce group, +38.6 ms.** Ours is 46 ms of wall for 653 programs; theirs is
   7.4 ms of dispatch wall that is hiding 39.5 ms of work. Their serial-work bound (39.5) is
   *below* our wall (46), so this class is not where we are uniquely bad — we are paying serially
   what they pay concurrently. That makes this class a consequence of two other facts, not a
   kernel-quality problem: tinygrad's CPU device serialises kernels, and their pool does not.
3. **The scan, +22.8 ms.** This one *is* a kernel-quality gap. At T=1 the work is
   32 heads x 128x128 state = 2 MB read + 2 MB write per layer, 96 MB/token, a 2.9 ms bus floor
   at 33.79 GB/s. The reference measures 4.16 ms (dispatch) / 10.69 ms (serial) — near the floor.
   We measure 27 ms, 9x the floor, for a reason visible in source (below).
4. **NPU, +5.9 ms.** At parity; leave it.

### Why our scan is 1.25 ms/call at T=1 and theirs is 0.17

`extra/rockchip/recurrent.c` is **byte-identical** to the reference tree's
`tinygrad/llm/kernels/rockchip_recurrent.c` (verified with `diff -q` against
`~/tg-qwen35-pack-signature-fix`). The difference is a compile-time flag:

| | our build | reference recipe |
| --- | --- | --- |
| `ONPU_GDN_DIRECT_DECODE` | `0` (`recurrent.c:17-19`, nothing defines it) | `1` (`extra/rockchip/ref_kernel_profile.py:49`, `run_ref.py:56`) |

With the flag at 0 the decoded kernel takes the general path (`recurrent.c:88-89`), which for
every one of the 32 heads transposes the whole 128x128 state into the `float columns[256*256]`
stack buffer before the single step. With it at 1, `direct_row` (`recurrent.c:27-57`) reads the
state in place with in-register `transpose4` and never touches `columns`. Qwen3.5's `ssm.kda` is
false (`model.py:925` sets it only for kimi/glm/bailingmoe), so `alpha_dim == 1` and the guard at
`recurrent.c:75-79` is satisfied at `rows==1, capacity==1`: the direct path would engage if it
were compiled in.

`ATTN_KERNEL` / `exp2_scalar` are not part of the gap: 8 calls per token in both trees.

---

## 4. Ranked improvements

Ordered by expected gain divided by effort. Every one of them needs a model run to confirm; the
confirmation column says exactly which field to read. Nothing here is proposed as measured.

| # | change | expected gain | effort | where |
| --- | --- | --: | --- | --- |
| 1 | compile the decode scan with `ONPU_GDN_DIRECT_DECODE=1` | **18-22** | one line + a parity check | `extra/rockchip/scan_native.py::_build` |
| 2 | fold the K-tile accumulation into the fused Linear's C | **7-11** | medium, touches generated C | `extra/rockchip/npu_linear.py::_fused_source` / `NPULinear.__call__` |
| 3 | take the decode graph out of the Python executor | **up to 110** | high, new mechanism | new module beside `extra/rockchip/npu_linear.py`; model hook at `tinygrad/llm/model.py:895` |
| 4 | explain and fix `E_62080_4` (1 call, 5-15 ms) | **up to 8** | low to measure, unknown to fix | identify the node first |
| 5 | overlap CPU kernels with the NPU wait | up to 74 | closed for this architecture (§5) | — |
| — | NPU device time | ~0 | — | at parity |

Projected path: 311 − (20 + 9 + 110) = **~172 ms/token = 5.8 tok/s** without #4, ~6.0 with it.
The floor is the device: 143 ms/token = 7.0 tok/s. The reference's 153.3 sits between the two,
which is what a solved host side looks like.

### 1. Scan: enable the direct decode row

- **File / function:** `extra/rockchip/scan_native.py::_build` — prepend
  `#define ONPU_GDN_DIRECT_DECODE 1` to `source` (before `recurrent.c` is spliced in at line 43).
  A per-shape parameter would keep prefill on the general path, but the direct path is already
  guarded to `rows==1 && capacity==1`, so a global define is safe and matches the reference.
- **Change:** nothing in the kernel; the code is already there and already exact ("Exact
  key-ordered recurrence ... without whole-state transposes", `recurrent.c:15-16`).
- **Why it should transfer:** it removes a 2 MB strided transpose per layer, 32 per call, 24 calls
  per token. That transpose is the only difference between our 1.25 ms/call and the reference's
  0.174 ms/call.
- **Confirm with:** (a) `extra/rockchip/scan_case.py 1 32 1 128 128` rel error against fp64 with
  the flag on — the geometry is already in its verified list (2.588e-07) but that was measured
  with the flag off, so re-verify; (b) `profile_graph_dispatch.py` decode rows:
  `cpu_gated_delta_scan` 0.030 s → ~0.005 s; (c) `threadpool-3plus3`-equivalent decode
  3.22 → ~3.45 tok/s.
- **Watch for:** the `scan_disjoint` guards at `recurrent.c:76-79` fail silently. Log the
  computed `direct` value once, or the run may look unchanged with no diagnostic.

### 2. Fold the K-tile accumulate into the fused C

- **File / function:** `extra/rockchip/npu_linear.py::_fused_source` (the `planes` loop, lines
  435-442) and `NPULinear.__call__` (the `acc = piece if acc is None else acc + piece` at line
  777).
- **Change:** give `_FusedKey` a `k_tile` / `k_last` pair; when a Linear has more than one K tile,
  have every tile but the last write its planes as now, and have the last tile read the earlier
  tiles' int32 planes and accumulate `(float)src * a * w_scale` into one fp32 output. The `+`
  programs disappear.
- **What it removes:** 32 `ffn_down` Linears x 2 float adds over 2560 elements, i.e. 64 of the
  984 programs. At 112 µs each that is **7.2 ms** of dispatch, plus ~2 ms of the adds themselves.
- **Confirm with:** `programs` in the decode rows of `profile_graph_dispatch.py` 984 → ~920,
  `bookkeeping_s` 0.11 → 0.10, and a bitwise comparison against the current path
  (`smoke_fused.py` / `parity_shapes.py` style, `max abs diff 0.0`), because the accumulation
  order changes from fp32-add-after-dequant to fp32-add-inside-dequant. If exactness is not
  achievable without changing the arithmetic, drop this item — it is the smallest of the three.

### 3. Take the decode graph out of the Python executor

This is the only item that addresses the 110 ms, and it is the whole gap.

- **Where the cost is:** `tinygrad/engine/realize.py` `pm_exec.rewrite` per CALL node, invoked
  once per program by the captured linear's `src` list. 984 x 112 µs. There is no cheaper flag.
- **What to build:** a decode-only chain program, generated as one C function that calls the 984
  recorded kernels in order with baked argument blocks, feeding only the inputs that change per
  token. That is what the reference does: `native_execution.decode.direct_inputs = 75` and
  `borrowed_outputs = 65` (`ref-profile/report.json`) for 1,695 kernel launches per step.
- **What not to reuse:** `tinygrad/llm/kernels/rockchip_replay.py::NativeReplay`. It is already
  ported and it already works on the ported device path, but `NPULinear`'s fused call issues its
  submit inside a CPU-side C function, so the captured graph has no ROCKCHIP programs and the
  replay refuses with `UnsupportedReplay('no NPU work')`. Forcing it to engage via
  `GRAPH_VISIBLE_INT8` measured 31.3 s of prefill against 22.4 s (`split/FINDINGS.md`,
  "NativeReplay port attempt"). The chain has to be built for the fused path, at the level of
  `_FusedInt8Plan.program` calls, not at the level of tinygrad UOps.
- **Effort:** high. It needs the program order and the address table at capture time, and it must
  invalidate whenever a plan is rebuilt or a buffer moves — the same lifecycle problem
  `NativeReplay.valid()` solves for the other path.
- **Confirm with:** `bookkeeping_s` in the decode rows of `profile_graph_dispatch.py` → ~0 and
  `loop_s` 0.33 → 0.22; decode rate 3.22 → ~4.9 tok/s.
- **A cheaper probe, if the chain is too large to start:** the per-program dispatch is flat, so
  any single-decode-step harness that replays a captured linear through `exec_linear` with a
  no-op `pm_exec` (or a pre-linked linear) would measure the achievable ceiling on one step
  without touching model code.

### 4. `E_62080_4`: one call, 5-15 ms, 27x the reference

- **Evidence:** our decode profile has `E_62080_4` at 0.010 s for **1 call**; the reference has
  the same-named kernel at 0.371 ms for 1 call. 62080 elements is microseconds of arithmetic, so
  either the name is misleading or the kernel is doing something quadratic.
- **What to do first:** identify the node. `profile_graph_dispatch.py` already dumps the top-8
  kernel sources for the prefill row (`SOURCES`), and the decode row's `top_kernels` names the
  same kernel; a `kernel_source` dump for the decode row is a code change to the profiler, not a
  model change.
- **Gain:** up to 8 ms if it is real; 0 if it is profiler noise. **Cost of finding out:** one
  instrumented run.

### 5. Overlap: do not start here

Hiding our 74 ms of CPU kernels behind the 143 ms NPU wait removes up to 74 ms, which is larger
than #1 and #2 combined. It is listed fifth because it is closed for this architecture: "tinygrad's
CPU device serialises kernel execution" (`split/FINDINGS.md`, "Parallelism is refuted for this
architecture") — four Python threads over one kernel measured 0.24x, and C-level pthreads inside
the kernel measured +100 ms. The reference gets the overlap because its pool and its submits live
outside the tinygrad device. Item 3 is the prerequisite: once the decode chain is one native
function, the NPU wait and the next layer's CPU kernels can be interleaved inside it. Do not
attempt overlap before the chain exists.

---

## 5. What not to do, with the measurement that rejects it

**Decode submit coalescing ("they run 9:1, we run 1:1").** This claim is in
`split/FINDINGS.md:2286-2311` and it is wrong. `ref-profile/report.json` `captured_calls.decode`
records 1,086 CPU programs, 151 `STORE:ROCKCHIP`, 52 `STORE:ROCKCHIP:1` = **203 NPU submits**,
and `native_execution.decode.profile` records **203.00 ROCKCHIP calls/token**. 1,421 = 203 x 7
(the run decodes 7 steps), not 1,421/64. `phase/report.json` — the reference configuration re-run
on this host at 64 tokens — records `decode_submits = 12,789 = 203 x 63` at 6.372/6.366/6.436
tok/s. So 203 submits/token *is* 6.4 tok/s on this machine. The "6.9 ms per submit" and "0.98 ms
per submit" in that table are 153.4/22.2 and 316/323, i.e. whole-token-time per submit, which says
nothing about per-submit cost; the real figures are 0.675 ms (theirs) and 0.466 ms (ours), so our
submits are already the cheaper ones.

**Packing more Linears into one submit.** `threadpool-3plus3.log` gives 825.0 tasks/token over
307.0 submits = 2.69 of 3 cores busy per submit, so a perfect 3-core pack would take
ceil(825/3) = 275 submits: **-10.4% submits, ~0 device time, ~1-2 ms of host time.** Not worth
the generated-C surgery. The 307 count is not the problem; §2's per-program split is.

**The m-tile merge.** We merged m-tiles for prefill (603 → 355 submits,
`merge-correct-3plus3`). Decode is m=1: `_fused_plan` sets `m_task=64, m_rows=m` only when
`m == 128` (`npu_linear.py:618-619`), and `NPULinear.__call__` iterates `range(0, rows, row_tile)`
with `rows == 1`. There are no m-tiles to merge in decode, and none of the prefill gain transfers.

**Raising `_int8_supported_depth` / `K_TILE` above 4096 to make `ffn_down` one submit.** This
would remove 64 of our 307 submits, but it is not a difference against the reference: its decode
profile contains no k=9216 kernel either. It is also forbidden, and not as a precaution —
measured, `m=1, n=2560, k=9216, cores=3` makes `DRM_IOCTL_RKNPU_SUBMIT` return
`TimeoutError [Errno 110]`, the driver logs `RKNPU: soft reset, num: 6`, and the board reboots
(AGENTS.md; the `banks` formula collapses to its `0x1b` floor at 18 groups). 4096 is a hard wall
for this register recipe.

**Fusing gate+up at the model level.** Measured and rejected (`split/FINDINGS.md`, "Fusion of qkv
and gate/up: exact, but slower"): 635 → 603 submits, prefill **2,771 → 3,179 ms (+14.7%)**, decode
**3.10 → 3.14 tok/s (+1.3%)**. The 19456-column fused weight is padded to the 24576 tile, so
prefill pays 24.5% more arithmetic; decode gains 32 submits and 4 ms. Needs task-level changes
inside the NPU path, not weight concatenation.

**`ONPU_CPU_DECODE_POOL` / thread-per-call.** The reference runs a persistent pool; copying that
shape naively is measured negative: `cpu_gated_delta_scan_pooled` with real
pthread_create/join, 4 head groups, exact to 4.209e-07 — prefill 2,614 → **2,714 ms (+100 ms)**,
decode 3.15 → 2.93 tok/s (`parscan-3plus3`, `split/FINDINGS.md` "Parallel scan via pthreads").
Four threads per call cost more than the parallelism saves on a 1-25 ms call. Python threads over
head slices are worse: 39.03 ms serial → 159.98 ms with 4 threads (0.24x), because the CPU device
serialises.

**Pooling the scan.** Already measured twice: 1.17x in-model, 1.36x standalone, and pooled still
slower than serial (33.29 vs 30.62 ms/call at T=128). The revert in `scan_native.py:59-62` stands.

**Re-quantising less.** `qz_row` runs 307 times per token and it is tempting to call that duplicated
work. It is not: each activation row is consumed by exactly one Linear, and the non-submit part of
`qz_row` — quantise 2560 values, two `sync_mem` calls, the dequant of 4096 int32 — totals **~4
ms/token** (§2). The 143 ms inside those 307 calls is the device.

**Chasing the NPU.** 143.0 against 137.1 is 4%, it is within the device's own
800 MHz/1 GHz DVFS wobble (measured cost of the drop: -0.54%), and it is bounded below by the GEMM
work both trees do. Fixing it cannot pay for itself.

---

## 6. What would change this document

- A decode `graph_rows` profile of our tree at 307 submits with the C chain (item 3) built: it
  would replace the 110 ms dispatch row with a measurement near zero, and would show whether the
  remaining 46 ms of CPU kernels stays serial.
- A `phase`-style probe of *our* decode (the existing `phase/windows.jsonl` is the reference
  configuration). It would give `cores_busy` and `off_cpu_ms` for our decode directly, rather than
  the current inference from `stats[0]` — and would settle whether any host work overlaps the
  NPU wait today. The code says none can (`_FusedInt8Plan.__call__` blocks inside one C
  function), but it has not been measured on this tree.
- Re-running `ref-kernels`-style kernel dumping with the pool disabled, to separate "4x because
  the pool was off" from "4x because the profiler attributes four workers to one call" in the
  reference's CPU rows. Until then the reference's CPU total is a bracket (14.9-55.8 ms), and §3
  uses both ends.
