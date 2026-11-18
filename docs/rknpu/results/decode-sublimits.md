# Decode: where 307 submits and 825 tasks per token come from, and what merging them is worth

Scope: Qwen3.5 4B under `extra/rockchip/bench_llm.py --npu`, decode phase, `m=1` per Linear.
Everything here is **read off the code path or off a recorded artifact**; nothing was run for this
document (another measurement owned the machine). Values that are arithmetic on measured constants are
labelled *derived*; values with no artifact are labelled *unmeasured*.

Headline: the 307 submits are **fully accounted for by the weight stream**, and the device is at
29.3 GB/s of a 33.79 GB/s bus. The fixed part of a decode submit is **0.034 ms** (measured fit,
`submit-timing.json`), so all 307 of them carry only **10.4 ms of fixed cost per token — 7% of the
143.3 ms device time and 3% of the step**. Merging submits is therefore *not* the lever for decode.
The ~170-220 ms of host work outside the submit ioctl is, and the largest *identified* item there is a
GDN-scan fast path that is compiled out of this tree.

---

## 1. The counts, derived from code and cross-checked against four artifacts

### 1.1 Model inventory: 233 packed Linears (249 before qkv fusion)

`bench_llm.py:106-109` calls `fuse_linears`, then `replace_linears`, then `prepare_linears`.
`fuse_linears` fuses only `attn_q`/`attn_k`/`attn_v`; gate/up sits behind `if False`
(`npu_linear.py:887`).

| fact | value | source |
| --- | --: | --- |
| blocks | 24 GDN + 8 attention | `doc-audit.md:241` (from GGUF tensor names), `npu-gemm-baseline.md:450` |
| Linears before fusion | 249 | `load-cost-plan.md` §1.4 total row; `FINDINGS.md:1219` |
| qkv fusions | 8 (one per attention block) | `fuse_linears`, `npu_linear.py:876-883` |
| Linears after fusion | **233** | `prof-merged/report.json: "npu_linears": 233` |

`prof-merged/report.json` is the tree that recorded 307/825:

```
"npu_linears": 233, "npu_weight_bytes": 4204789760,
"npu_submits": [{"prefill": 355, ..., "decode_per_token": 307.0,
                 "decode_tasks_per_token": 825.0, "decode_npu_ms_per_token": 143.3}]
```

### 1.2 Submits per Linear at m=1: `ceil(depth/4096) x ceil(columns/24576)`

`NPULinear.__call__` (`npu_linear.py:735-784`):

- `row_tile = (128 if k_max <= 2048 else 64)`, `step = merge or row_tile` (`:747`, `:751-753`).
  At `rows == 1` the merge predicate is false (`rows == 2*row_tile`), so `step = row_tile` and the
  `for r0 in range(0, rows, step)` loop (`:754`) runs **once**.
- `width = 24576 if part.shape[0] == 1 else 12288` (`:756`) — m=1 takes the wide path, so
  `for col in range(0, n, width)` (`:760`) runs `ceil(columns/24576)` times.
- `for pc, pk, weight, scales in self.packed` (`:763`) iterates the **K tiles** laid down by
  `prepare()` (`:722-729`, `K_TILE = 4096`), and each yields one `packed_matmul_int8_fused` call
  (`:771`) = one C function = **one submit** (`_fused_source:475`). The tiles are summed on the host at
  `:777` (`acc = piece if acc is None else acc + piece`).

So at m=1 **one Linear costs `ceil(depth/4096) x ceil(columns/24576)` submits**, one per call.

### 1.3 The per-token submit count, by shape

Shapes are from `load-cost-plan.md` §1.4 (our own `prepare()` inventory of the 4B): `ffn_gate`/`ffn_up`
9216x2560 (64 Linears), `ffn_down` 2560x9216 (32, K split 4096+4096+1024), `output` 248320x2560 (11
tiles), `attn_qkv` 8192x2560 (32), `attn_output`/`ssm_out` 2560x4096 (32), `attn_gate` 4096x2560 (24),
`ssm_alpha`+`ssm_beta` 32x2560 (48), `attn_k`/`attn_v` 1024x2560 (16). That inventory is the
**pre-fusion** 249-Linear tree: the 8 fused `qkv` Linears of 10240 columns (q 8192 + k 1024 + v 1024)
replace the 8 `q` + 8 `k` + 8 `v` entries.

That table's own total is **323 tiles**, which is exactly the pre-fusion decode submit count recorded in
`fz1.log:4` (`323.0 submits/token 873.0 tasks/token`). The 8 fusions remove 2 tiles each:
`323 - 16 = 307`, and each removed tile carried 3 tasks: `873 - 48 = 825`. Both deltas reproduce
exactly, which is the check that this derivation is right rather than merely plausible.

The same count as a formula: `24x10 + 8x7 + 11 = 240 + 56 + 11 = 307`, where a GDN block is 10 submits
(qkv, gate, alpha, beta, ssm_out, ffn_gate, ffn_up, ffn_down x3), an attention block is 7 (qkv fused,
attn_output, ffn_gate, ffn_up, ffn_down x3) and the head is `ceil(248320/24576) = 11`.

### 1.4 Tasks: 48 single-core + 259 three-core = 825

`int8_task_plan` (`ops_rockchip.py:190-202`) builds one body per entry of
`_matmul_parts(m, n, k, cores, int8=True)` and **one** cmd chain for all of them, so all of a Linear's
per-core parts are already inside one submit. `_matmul_parts` (`ops_rockchip.py:97-104`):

```python
width = round_up(ceildiv(n,cores),32)
parts = tuple((offset,min(width,n-offset)) for offset in range(0,n,width))
if len(parts) not in (1,3): parts = ((0,n),)
```

`n=32` (`ssm_alpha`, `ssm_beta`) gives `width=32`, one part, so those 48 submits run on **one core with
one task** (`_encode_task_chain(..., 1)` -> `core_mask=1`). Every other shape splits into 3.

| class | submits/token | tasks each | tasks |
| --- | --: | --: | --: |
| 3-part submits | 259 | 3 | 777 |
| 1-part submits (`ssm_alpha`, `ssm_beta`, 24 blocks) | 48 | 1 | 48 |
| **total** | **307** | | **825** |

---

## 2. Per-program table for one decode token

`KernelInfo` name is `rockchip_linear_{m}_{n}_{k}` (`npu_linear.py:499`); programs are memoised by
geometry in `_FUSED_PROGRAMS` (`:490-505`), so e.g. every `ffn_gate` and `ffn_up` share one compiled C
function and differ only in the parameter block. "Device ms" is *derived* from the measured fits in
`docs/rknpu/results/submit-timing.json` — `t = 0.0342 + 8.018e-5*n` at k=2560, `t = 0.0360 + 1.471e-4*n`
at k=4096 (8 and 3 points), with `n=32` from the measured minimum task (0.0245 ms), the k=1024 tile
scaled by bytes, and the 24576-wide head chunk as 3 cores x 8192 columns each.

| program | Linears it serves | count | tasks | share of 307 | device ms (derived) |
| --- | --- | --: | --: | --: | --: |
| `rockchip_linear_1_8192_2560` | 24 GDN `attn_qkv` | 24 | 72 | 7.8% | 16.6 |
| `rockchip_linear_1_4096_2560` | 24 GDN `attn_gate` | 24 | 72 | 7.8% | 8.7 |
| `rockchip_linear_1_32_2560` | 24 `ssm_alpha` + 24 `ssm_beta` (1 core each) | 48 | 48 | 15.6% | 1.2 |
| `rockchip_linear_1_2560_4096` | 24 `ssm_out` + 8 `attn_output` + `ffn_down` k=0 and k=4096 in 32 blocks | 96 | 288 | 31.3% | 39.6 |
| `rockchip_linear_1_9216_2560` | `ffn_gate` + `ffn_up` in all 32 blocks | 64 | 192 | 20.8% | 49.5 |
| `rockchip_linear_1_2560_1024` | `ffn_down` k=8192 tile, all 32 blocks | 32 | 96 | 10.4% | 3.7 |
| `rockchip_linear_1_10240_2560` | 8 attention `qkv` (fused) | 8 | 24 | 2.6% | 6.8 |
| `rockchip_linear_1_24576_2560` | 10 head chunks of 24576 columns | 10 | 30 | 3.3% | 20.7 |
| `rockchip_linear_1_2560_2560` | head tail, 2560 columns | 1 | 3 | 0.3% | 0.2 |
| **total** | 233 Linears | **307** | **825** | **100%** | **147.1** |

Derived device time sums to **147.1 ms against the measured 143.3 ms** (`decode_npu_ms_per_token`,
`prof-merged/report.json`) — +2.7%, inside the fits' accuracy at the extrapolated shapes. The
conclusion that matters: the 143.3 ms is the **weight stream**, `4.205 GB / 143.3 ms = 29.3 GB/s`
against a 33.79 GB/s DDR ceiling and the fit's own 31.93 GB/s at these shapes.

### 2.1 What each call does on the host (`_fused_source`, `npu_linear.py:411-484`)

```c
int err = 0;{quantize}                                    /* :469  qz_row, 1 row          */
if (!err) err = sync_mem(fd,1,params[5],{payload_size});   /* :470  clean payload          */
    struct Submit submit={5,6000,0,{tasks},0,0,...};      /* :474  tasks = 1 or 3         */
    err = ioctl(fd,{_SUBMIT_REQUEST},&submit);            /* :475  BLOCKING submit        */
if (!err) err = sync_mem(fd,2,params[6],{out_size});      /* :481  invalidate planes      */
...{planes}                                               /* :482  dequant int32 -> fp32  */
```

So per token: 307 quantises of `1 x depth_in` floats, **614 `RKNPU_MEM_SYNC` ioctls**, 307
`DRM_IOCTL_RKNPU_SUBMIT` ioctls, and 1.54 M fp32 outputs dequantised (6.2 MB).

### 2.2 The CPU side of the same token

`FINDINGS.md:2290-2298` records our decode graph at **1,032 programs of which 323 are NPU**, i.e. ~709
CPU programs, in the pre-qkv-fusion tree. That is the only recorded decode program count; no artifact in
`docs/rknpu/results/` profiles a decode phase per kernel for the 307-submit tree, so 709 is quoted as
the pre-fusion figure and is not re-derived here.

---

## 3. The fixed-cost model: what merging can and cannot buy

### 3.1 The per-submit constant at decode is 0.034 ms, not 1.3 ms

The 1.3 ms figure is **prefill**. `FINDINGS.md:1649-1663` derived `F = 0.920 ms` from two prefill runs
(323 and 603 submits, 0.360 / 0.790 s). Both issue the *same* `m=64` submit shapes — they differ in
submit *count*, not rows — so that fit is not a fixed-vs-rows decomposition and must not be carried into
decode.

The decode-shape measurement is the `n` sweep in `docs/rknpu/results/submit-timing.json`
(`bench_submit.py`; `m=1, k=2560, cores=3`, 8 points, `RKNPU_JOB_BLOCK`):

| quantity | value | note |
| --- | --: | --- |
| fixed per submit (fit intercept, k=2560) | **0.0342 ms** | `fits[0].intercept_ms` |
| fixed per submit (fit intercept, k=4096) | 0.0360 ms | `fits[1]`, 3 points only |
| host prep outside the ioctl | 0.0044 ms | `rows[*].host_ms`, 0.0044-0.0051 |
| smallest measured submit (`n=32,k=512,1 core`) | 0.0245 ms | `rows[0]` — the intercept exceeds a real submit, so it is an extrapolation |

Measured marginal cost of a removed submit at decode, from two runs: `fz1.log` 323 submits -> 143.5 ms;
`prof-merged` 307 -> 143.3 ms. Sixteen submits removed with **no change in streamed weight bytes** cost
0.2 ms, i.e. **<=0.013 ms per submit** — the same order as the 0.034 ms intercept.

### 3.2 Arithmetic for the ways of merging

| configuration | submits/token | saves | device ms saved @0.034 | arithmetic |
| --- | --: | --: | --: | --- |
| today | 307 | — | — | `24x10 + 8x7 + 11` |
| every pair of independent submits merged | **154** | 153 | 5.2 | `307 = 153x2 + 1` |
| every Linear's per-core submits merged into one call | **307** | **0** | 0 | already done: 259 submits carry 3 tasks in 1 ioctl (`_encode_task_chain`, `ops_rockchip.py:158-187`) |
| `ffn_down`'s 3 K tiles as one submit (3 sources, 9 tasks) | **243** | 64 | 2.2 | `32 x (3 -> 1)`; `int8_task_plan_multi` already loops `for core in range(3) for src in ...` |
| + `attn_qkv`+`attn_gate` concatenated (12288 cols) | **219** | 88 | 3.0 | `24 x (2 -> 1)` |
| + `ffn_gate`+`ffn_up` concatenated (18432 cols) | 187 | 120 | 4.1 | 32 more, but this is the change `FINDINGS.md:1300-1321` measured at **+14.7% prefill** |

The per-core row is the important one: **it is a no-op**, because each Linear's three cores are already
three tasks inside a single submit. The only remaining merge is *across* Linears.

**Bound.** Take the whole submit count to 1: 306 x 0.034 = **10.4 ms** of device, plus at most
306 x 0.06 ms of host (~18 ms) even if every submit's sync pair and C call cost 60 us. That is
**<=28 ms of a 316-365 ms step (+9%)**. Decode cannot reach 6.5 tok/s by merging submits, and the
measured device time already equals `bytes / bandwidth` to within 3%. The 2.1x decode gap is not in the
submit count — it is in the **~170-220 ms outside the submit ioctl**:

| | ours | reference | source |
| --- | --: | --: | --- |
| step | 311-365 ms (2.74-3.22 tok/s) | 153.4 ms (6.52 tok/s) | `prof-merged`; `FINDINGS.md:2265-2266` |
| in the NPU submit ioctl | **143.3 ms** | 138.66 ms | `prof-merged/report.json`; `~/oNPU/benchmark.md:1392` |
| outside it | **~170-222 ms** | **15.56 ms** (14.95 CPU + 0.61 copies) | same |
| submits/step | 307 | **203** (1:1 with NPU programs) | `audit-refdoc.md:14` |

The reference's `22.2 submits/token` (`FINDINGS.md:2267`, `:2296`) is `1,421/64`: it divides by the
64-token published protocol while that run decodes 7 steps. `1,421 = 203 x 7` exactly, so the true
figure is 203 per step at 1:1. **The "reference coalesces ~9 programs per submit, so coalesce to ~36 and
reach 6.4 tok/s" projection at `FINDINGS.md:2305-2307` is retracted** — it was built on 22.2. Our 307
vs their 203 is 1.51x, worth ~3.5 ms at the measured intercept.

---

## 4. Duplicated host work: per token, or amortised

| # | item | where | per token? | duplicated? | cost |
| --: | --- | --- | --- | --- | --- |
| 1 | `qz_row` activation quantise | `npu_linear.py:353-390` (NEON), called from `_fused_source:428-433` | **yes**, once per submit (307) | **yes**: every Linear reading the same `x` re-quantises it — GDN `attn_qkv`+`attn_gate`+`ssm_alpha`+`ssm_beta` share one row (4x), `ffn_gate`+`ffn_up` share one (2x) => **104 redundant** of 307 | ~2.5 us of arithmetic each (`npu-gemm-baseline.md:106`) => ~0.26 ms/token |
| 2 | int8 weight quantise / repack | `pack_weight_int8`, `npu_linear.py:86-92`, from `prepare()` `:714-733` | **no — amortised at load** (62.3 s one-time, `load-cost-plan.md` §1.2) | not per token | 0 per token |
| 3 | dequantise of **weights** per token | — | **does not exist**: weights stay int8 plus one fp32 scale per row (`quantize_rows` `:80-83`); the NPU reads them directly | — | 0 |
| 4 | dequantise of **output planes** | `_fused_source:435-442` | yes, `1 x n` per call | no | 1.54 M floats / 6.2 MB => ~1 ms |
| 5 | `x.reshape(-1, depth).contiguous()` | `npu_linear.py:739` | **yes, per Linear (233)** | **yes**: `contiguous()` returns self only for a buffer identity (`mixin/elementwise.py:62-68`), so a computed `x` gets an `Ops.STAGE` materialisation — once per reader of the same row | 2560 floats each => <1 ms, but adds up to 233 programs to the graph |
| 6 | K-tile reduction on the host | `npu_linear.py:777` (`acc + piece`) | **yes**: 2 adds x 32 `ffn_down` = 64 kernels/token | no (the tiles are disjoint in K) | 64 x 2560 floats => ~0.2 ms |
| 7 | head column-chunk concat | `:781` (`Tensor.cat(*pieces)`) | yes, 1 kernel over 248320 floats | no | ~1 ms |
| 8 | GDN scan state **transpose in and out** | `recurrent.c:88-89` (in), `:430` (back) — general path only | **yes, 24 calls/token** | no, but **avoidable**: `direct_row` (`:27-57`) is compiled out because `ONPU_GDN_DIRECT_DECODE` defaults to 0 (`:17-19`) and `scan_native._build` (`:41-45`) never defines it | 32 heads x 2 x 64 KB per block = **192 MB/token** of host traffic; **unmeasured** (§6) |
| 9 | `short_conv` window rebuild | `model.py:104-114` (`state.cat(x).pad_to(...).contiguous()`) | yes, 24x/token | no | 4 x 8192 floats per block = 3 MB/token |
| 10 | Gumbel-max sample over 248320 logits | `model.py:906-908` | yes, every token | no, but wasteful at `temperature=0` (greedy): `rand_like` + `log` + `neg` + `maximum` + `log` + div + sub + `argmax` | ~6-8 passes over 248320 elements plus a host RNG draw => ~5-10 ms (derived) |
| 11 | KV-cache store | `model.py:452` (`_update_cache`) | yes, 8x/token | no | 2 x 4 x 256 floats |
| 12 | scalar sampling in Python | — | **no**: the sample is a tensor expression inside the captured graph; the only Python scalar is `out.item()` at `bench_llm.py:146`, which is harness | — | — |

Items 1, 5 and 10 are genuine per-token duplication; items 2 and 3 are the two the hypothesis expects and
are **not** there. The largest identified per-token host item is 8, the scan transpose.

---

## 5. Three cheapest changes

Ordered by expected effect per line of risk. Each is a single-function change.

### A. Take the `direct_row` branch in the decode scan — one line, host work only

- **Where**: `extra/rockchip/scan_native.py:_build` (`:41-45`), which assembles the C source, or
  `extra/rockchip/recurrent.c:17-19`.
- **Change**: prepend `#define ONPU_GDN_DIRECT_DECODE 1` to the generated source. The macro already
  exists, is `#ifndef`-guarded, and the predicate at `recurrent.c:75-79` tests
  `rows==1 && capacity==1 && start>0 && !history && alpha_dim==1` — all true at our decode geometry
  (`start` is baked as 1, `scan_native.py:63`). Every disjointness test still falls back to the general
  path if it fails, so the macro cannot change correctness — only whether it helps.
- **Why it should matter**: the general path copies each head's 128x128 state into a transposed
  `columns[256*256]` stack buffer and back (`:88-89`, `:430`) — 192 MB/token across 24 blocks, all pure
  address shuffling. `direct_row`'s own comment: "Exact key-ordered recurrence on four value rows,
  **without whole-state transposes**."
- **Submits after**: **307, unchanged** (host work, not submits).
- **Confirmation**: (i) function-boundary parity — `extra/rockchip/scan_case.py` compares a fresh process
  per geometry against an fp64 replay; run it at `(1,32,1,128,128)` and require the same order as the
  recorded cases (2.588e-07 / 4.479e-07 / 4.209e-07, `scan_native.py:19-21`). **It cannot be bitwise
  against the general path**: `direct_row` sums in a different order by design, so state and out will
  differ at ~1e-7 and greedy ids must be re-baselined — the precedent for this kernel is already set at
  `FINDINGS.md:166-168`. (ii) a 3+3: `bench_llm.py --npu --warmups 3 --runs 3`, every trial and the
  median, decode.
- **Reference**: the reference runs with `ONPU_GDN_DIRECT_DECODE=1` (`run_ref.py:56`, `split/run.sh:29`),
  so this is also the configuration the 6.52 tok/s row was measured in.

### B. Merge `ffn_down`'s three K tiles into one submit — 64 fewer submits, bitwise

- **Where**: `extra/rockchip/npu_linear.py:_FusedInt8Plan.__init__` (`:527-534`) and the `planes` block
  of `_fused_source` (`:434-442`); `_fused_plan` (`:612-629`) to key on three sources.
- **Change**: for `depth > K_TILE`, call `int8_task_plan_multi` with the three tiles as three sources
  (`(m, n, k, payload, weight_dma+k_offset*k, out_plane+k)`), 9 bodies = 3 tasks per core, and have the
  C dequant read the three plane regions and compute `t0 + t1 + t2` in the same order the host uses today
  at `:777` (`t = (float)raw * a_scale * w_scale` per tile, then add). **Do not raise `K_TILE`** —
  `ops_rockchip.py:116-121` records k=9216 wedging the board.
- **Submits after**: **243** (`307 - 32x2`); tasks unchanged at 825.
- **Confirmation**: function-boundary bitwise — one call of the current path against one call of the
  merged path on the same weights and activation, `max_abs_diff == 0` and identical NaN pattern over the
  `(1,2560)` output (the harness pattern `extra/rockchip/parity_shapes.py` already uses at every 4B
  geometry); then a 3+3. `FINDINGS.md:2035-2050` shows this mechanism already reaches column-bitwise
  parity for two different activations (`0 of 336 words`).
- **Expected**: 64 x 0.034 = 2.2 ms device, plus 64 fewer sync pairs, C calls, and host `acc + piece`
  kernels; order 3-8 ms/token.

### C. Concatenate `attn_qkv` + `attn_gate` per GDN block — 24 fewer submits, bitwise

- **Where**: `extra/rockchip/npu_linear.py:fuse_linears` (`:861-895`) — a second pair next to the existing
  `attn_q`/`attn_k`/`attn_v` case — plus the split at `model.py:GatedDeltaNetBlock._attention` (`:768`,
  `:777-780`).
- **Change**: build one Linear of `8192+4096 = 12288` columns (`_cat_linears`, `:815-831`). Exact for the
  reason the qkv fusion is: `pack_weight_int8` gives one fp32 scale per **output row**
  (`npu_linear.py:846-851`) and 12288 < `TILE` (24576). It also removes one `qz_row` and one
  `x.contiguous()` per block.
- **Submits after**: **283**; combined with B, **219**.
- **Prefill side effect**: at m=64 the 12288 columns fit the 12288 slice exactly and split into 3 legal
  4096-wide parts, so prefill also loses 1 submit per GDN block (355 -> 331 in the current tree). This is
  the reverse of the gate/up case, which `FINDINGS.md:1300-1321` rejected because 18432 columns do not fit.
- **Confirmation**: function-boundary bitwise against two separate `packed_matmul_int8_fused` calls
  (column-for-column, `max_abs_diff == 0`), then a 3+3. The qkv half of this mechanism is already measured
  at -37 ms prefill and 3.10 -> 3.14 tok/s decode (`FINDINGS.md:1438-1456`).

**Deliberately not in the three**: enabling `ffn_gate`+`ffn_up` concatenation (32 more submits). It is the
one change with a measured regression (`+14.7%` prefill, `FINDINGS.md:1313-1321`), and it is already
switched off in the code at `npu_linear.py:887`.

---

## 6. What is *not* attributed, and the cheapest experiment that would

The 143.3 ms of device time is accounted for by the weight stream to within 3% (§2). The **~170-222 ms of
host time outside the submit ioctl has no artifact** in `docs/rknpu/results/`: every profile there is a
prefill row (`progcost.log` labels its rows `capture` / `instrumented-a` / `bare` / `instrumented-b`, all
with 355 prefill submits; `prof-merged.log` dumps prefill kernel sources; `threadsplit/` is the prefill
critical path).

The candidates, with their counts (§4): ~709 CPU programs (pre-fusion count), 614 `sync_mem` ioctls, 24
scan calls carrying 192 MB of transpose traffic, up to 233 `STAGE` copies, ~8 passes over 248320 logits.

Cheapest attribution, in order:

1. `extra/rockchip/inmodel_program_cost.py --gen 1` — it already splits every `CPUProgram.__call__` into
   pack / compiled kernel / bookkeeping and reports "everything that is not a CPU program". One decode
   step separates "709 CPU kernels cost ~150 ms" from "~150 ms is non-program time" in a single run.
2. `extra/rockchip/profile_graph_dispatch.py` with `--gen 1` — per-kernel names and times for the decode
   graph (it already separates prefill-sized graphs via `PREFILL_MIN_PROGRAMS`).
3. Ablation: return `core = 0` at `model.py:835` (the `if core is None` fall-through point) and diff the
   decode step. This is the only way to price item 8, and it follows the pattern `ATTN_STUB`
   (`model.py:180`) already provides.

Do not multiply 307 by any per-call figure to get a decode number: `npu-gemm-baseline.md:102`
(`323 submits x 1.5 ms = 485 ms`) is contradicted by the measured 316 ms step, and its 1.48-1.63 ms table
is the **pre-fusion six-launch** path from `profile_int8.py`, not the fused C call.

---

## 7. Already known not to work — do not retry

| attempt | measurement | where |
| --- | --- | --- |
| Fuse `ffn_gate`/`ffn_up` by weight concatenation | prefill 2,771 -> 3,179 ms (**+14.7%**), submits only 635 -> 603, decode 3.10 -> 3.14 tok/s; the 19456-wide tile splits back into 2 submits at prefill | `FINDINGS.md:1300-1321`; disabled at `npu_linear.py:887` |
| m-tile merge at prefill (two 64-row tiles in one submit) | landed, 603 -> 355 submits, **-89 ms**; recovery per merged submit was 0.36 ms, not the projected 0.920 | `FINDINGS.md:2313-2346` |
| "Reference coalesces ~9 NPU programs per submit, so coalesce to ~36" | the 22.2 figure is `1,421/64`; the reference is **203 per step at 1:1** — the projection is dead | `FINDINGS.md:2305-2307`, retracted by `audit-refdoc.md:14` |
| Raise `K_TILE` past 4096 to avoid the K-tile submits | k=9216 at m=1 times out (`DRM_IOCTL_RKNPU_SUBMIT` Errno 110), driver logs `RKNPU: soft reset, num: 6`, board can reboot; 4096 is a hard wall, not a cap | `ops_rockchip.py:116-121`; AGENTS.md "k=9216 wedges the NPU" |
| `int8_task_plan_multi` at N=1 passing the full `n` as body width/stride | produced a submit the NPU never completed (job timeout); each part must pass **its own** width | `FINDINGS.md:1982-2034` |
| Whole-graph `NativeReplay` | refuses with `UnsupportedReplay('no NPU work')` because the fused submit is invisible to the graph; the graph-visible INT8 path that makes it engage measured **31.3 s vs 22.4 s** prefill | `FINDINGS.md:274-281`, `:333-334` |
| Fuse CPU programs to cut dispatch | in-model dispatch is **~4 us per program**, 4 ms per prefill; the earlier "1.3 ms per CPU program" was an isolated `fast_sigmoid` call and is withdrawn | `FINDINGS.md:2696-2716` |
| Async dispatch / thread pool over the graph | list-scheduling the real graph: 1 worker 2,417.7 ms, 4 workers 1,973.2 ms (**1.23x**), critical path 1,742.3 ms — floors at ~1.9 s | `FINDINGS.md:2718-2732` |
| Split submit from dequant to overlap NPU with CPU | removing the trailing `sync_mem` (the dependency) is **-4 ms**; the split adds 249 launches and a ~300 MB int32 pass | `FINDINGS.md:2496-2521`, `:2522-2541` |
| Pool the scan across 4 threads | 1.21x ceiling on the kernel; pooled 33.29 ms vs 30.62 ms serial — pooled loses | `compare-ours-vs-ref.md:118`; reverted at `scan_native.py:59-62` |
| `_fused_plan` m_rows merge for decode | never fires at m=1: `m_task, m_rows = ((64, m) if (m == 128 and ...))`, so m=1 takes the single-source path | `npu_linear.py:618-619` |
