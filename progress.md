# Rockchip EW / OUT=5 progress

Branch: `rockchip-2608-ew`  
Focus: half ADD/MUL (and gemm via gather + EW) on DPU with **OUT_PRECISION=5** for tol match. No CPU `host_sum`.

## Test status

| Test | Status |
|------|--------|
| Pointwise ADD/MUL | **pass** — mtx512 ≤8/chunk, PC-chain 64, no soft-reset |
| `test_small_gemm` (8×8) | **pass** — ≪ old ~4.5s reset-bound path |
| `test_gemm_fp16` (64×64) | try after ship; was impractical under reset storm |
| eye / padded / range gemms | `RKPLAN_REJECT` (plan) |

## Shipped path (mtx512 OUT=5)

Source: [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) (`ref/rk3588-npu`) — NC1HWC2, fp32-out **C2=4**.

- **OUT_PRECISION=5**, contiguous half in / fp32 out per chunk.
- Chunk: `CHANNEL=n-1`, `W=H=0`, `DST_SURF_STRIDE=1<<4`, `SURFACE_ADD=4<<4`, `size_e=1` (`size_e=3` hangs on EW).
- **`n≤8` per task**; `n>8` → 64B-aligned slots, PC-chain up to **64** chunks/ioctl.
- **No soft-reset** on the hot path (`reset_npu()` remains for recovery only).

### Why not elementwise WIDTH alone

Naive `CHANNEL=7` + WIDTH only got first **4** exact on `n=8`: one fp32 atom = C2=4 floats.

### Why not the old 4/64B + reset path

`ref/rk3588/examples/elementwise.py` 4-elem/64B packing + `ACT_RESET` every ≤16-tile chain made warm 8×8 ~95% soft-reset (~106ms × 15).

## Refs

- `ref/rk3588-npu` — C2 / `feature_data` / `surf_add`
- `ref/rk3588/examples/elementwise.py` — old EW PC-chain + fp32 tile pack
- `tinygrad/renderer/rockchip.py`, `tinygrad/runtime/ops_rockchip.py`

---

## 2026-08-07 — after mtx512 ship (no soft-reset)

### Warm 8×8 gemm profile (`test_small_gemm` call ~0.22s incl. torch/ref; program `__call__` alone)

| Component | Time | Share of `__call__` |
|-----------|------|---------------------|
| emit+patch | ~3.9 ms | ~47% |
| ioctl submit | ~2.6 ms | ~31% |
| cvt fp32→f16 (host) | ~0.9 ms | ~10% |
| gather | ~0.8 ms | ~9% |
| half_out | ~0 ms | — |
| soft-reset | **0** | — |
| **`__call__` total** | **~8.4 ms** | |

- **Ioctls: 15** (`dev.submit_count` delta). One PC-chain per logical EW op.
- Shape: 15 EW ops × 64 elems → 8 chunks/op → **120 chunks**, flushed per-op (not across ops).
- Old path: ~15× soft-reset ≈ 95% of ~1.6–4.5s realize. Soft-reset removed; wall is now emit + NPU submit + host cvt.

### Why not one ioctl?

Each OUT=5 stage writes **fp32**; next stage RDMA expects **half**. Runtime does host `_fp32_slots_to_half` after every op with `cvt_scratch` set. A PC-chain is NPU-only (no CPU between tasks), so dependent MUL/ADD tree stages cannot share one submit under the current OUT=5 + half-in contract. Within one op, the 8 chunks already share one ioctl.

### Host cast

Yes — NPU writes fp32 into the 64B slot; CPU `f32.astype(np.float16)` in place (`_fp32_slots_to_half`) before the next EW or final `half_out`. That is the ~0.9 ms cvt bucket (×15 for 8×8 gemm).

### Option: switch back to OUT_PRECISION=2 (fp16 out)

Would unlock:
- No host cvt between stages → dependent ops can PC-chain → **far fewer / one ioctl** for gemm tree
- Contiguous half tiles (elementwise.py: up to ~64k elems/task), not mtx512 ≤8 for fp32 C2=4
- Recipe notes from `elementwise.py`: `out_precision=2`, `OUT_CVT_SCALE=(1<<16)|1`, CHANNEL=7 + WIDTH packing (not the OUT=5 surf atom path)

Caveat: for the gemm ADD tree, OUT=5 already casts to half after every stage, so accumulate between levels is already half. OUT=2 mainly removes barrier/cvt cost; deep gemm vs torch wider-accum `5e-3` may still miss the same way a pure half chain did. Keep OUT=5 if single-op “f32 ALU then cast once” exactness still matters for pointwise refs.

**Not switched yet** — probe `test_small_gemm` / pointwise under OUT=2 before changing the shipped path.

---

## 2026-08-07 — flexible OUT precision (`ROCKCHIP_EW_OUT`)

Config: `ROCKCHIP_EW_OUT=fp32|fp16` (default **fp32**). Stored on `RKImage.out_precision` (RKIM v11, flags bits 8–15).

| Mode | OUT | Layout | Host cvt | Cross-op chain |
|------|-----|--------|----------|----------------|
| `fp32` (default) | 5 | mtx512 ≤8 / 64B | yes | no |
| `fp16` | 2 | contiguous half (elementwise WIDTH) | no | yes |

### Warm 8×8 gemm with `ROCKCHIP_EW_OUT=fp16`

| Metric | fp32 (prior) | fp16 |
|--------|--------------|------|
| ioctls | 15 | **1** |
| warm realize (`__call__` path) | ~8–11 ms | **~5.4 ms** |
| pytest `test_small_gemm` call | ~0.10–0.22 s | **~0.08 s** |
| maxabs vs f32/torch matmul | (passes 5e-3) | ~0.0078 (passes 5e-3) |

Pointwise ADD/MUL also pass under fp16 (submit counts drop: e.g. `test_add` 6→1).

---

## 2026-08-07 — PC-chain buffers + medium gemm sweep

### Bigger regcmd/task floors (not a hard 23-task HW cap)

Symptom: gemm N=12 → **23** EW ops in one ioctl → `TimeoutError` errno 110; chains of 22 and 24 worked. Suspected 4096B scratch — **not** a hard max of 23 tasks (`struct_rknpu_task`×23 ≈ 920B; 23× fp16 EW regcmd ≈ 4048B still ≤4096, while 24 needs two pages and already succeeded after realloc).

Fix (match allbilly `elementwise.py`): raise floors to **regcmd 64KiB**, **task 16KiB**. After that, chain length 23 and **12×12 gemm** both pass.

Refs: `ref/allbilly-rk3588` / `tinygrad/ref/rk3588` — `experimental/pcchain.md` (EW: `AMOUNTS = next body qword count`); no documented “max 23 tasks”.

### `test_medium_gemm` advance (`ROCKCHIP_EW_OUT=fp16`, +1 N, 30s wall cap)

| N | Result |
|---|--------|
| 9–11 | ok (before bigger buffers; 12 timed out on 23-task chain) |
| 12–17 | ok after 64KiB/16KiB floors (~70–140 ms, 1 ioctl) |
| **18** | **tol fail** vs f32 matmul ref (maxabs≈0.0105 > 5e-3) — stopped |

At this point `test_medium_gemm` was `(17,17)@(17,17)`. `test_small_gemm` remains `(8,8)@(8,8)`.

---

## 2026-08-07 — EW `IN_PRECISION=5` (fp32-in) HW probe

TRM (`docs/rk3588_trm.md` / allbilly) lists `in_precision=3'd5` = fp32 as a valid encoding. allbilly/npu working recipes never use it for EW: always `IN=2` + `MRDMA_FP16TOFP32_EN(1)`, even when `OUT=5`.

### What we tried (single EW ADD, n=8, recover via `simple_add.py` each trial)

| Config | Result |
|--------|--------|
| `OUT=5 IN=2` mtx (shipped) | **OK** exact |
| `OUT=2 IN=2` WIDTH (shipped) | **OK** exact (on clean NPU) |
| `IN=5 OUT=5 PROC∈{2,5}` × `erdma_ds∈{0,1,2,3}` × mtx/WIDTH × fp32 buf | **TimeoutError** every time |
| `IN=5 OUT=2 PROC=5` mtx | **Timeout** |
| `IN=5` + `MRDMA_FP16TOFP32_EN` / half buf | **Timeout** |
| Control: `IN=2 OUT=5 PROC=5` + fp16to32 half | **completes** (not a timeout) |

### Verdict

**No usable EW fp32→fp32 (or fp32→fp16) RDMA path found.** Register space allows `IN=5`; submitting it hangs the job (errno 110). Stay on **half-in** (`IN=2` + widen for ALU). OUT=5 still needs host `_fp32_slots_to_half` between dependent stages; gemm accumulate remains a half reduce tree either way. Prefer `ROCKCHIP_EW_OUT=fp16` for chain/throughput; keep OUT=5 when single-op f32-store then cast matters.

---

## 2026-08-07 — corrected medium GEMM boundary + compensated DPU EW

All results here use DPU EW only: `DEFAULT_FLOAT=HALF`, `ROCKCHIP_EW_OUT=fp16`, FP16 inputs and outputs, no CMAC/CNA and no host sum.

### Correction: 18×18 was not a failure

The earlier sweep stopped at N=18 because `maxabs > 5e-3` was treated as failure. The test actually uses
`abs(error) <= atol + rtol*abs(reference)` with `atol=rtol=5e-3`.

| Sequential reduction | Result | Worst tolerance ratio |
|---|---:|---:|
| 18×18 | pass | 0.747 |
| 19×19 | pass | 0.769 |
| 20×20 | pass | 0.815 |
| **21×21** | **fail** (1 element) | **1.096** |

Thus the stock sequential FP16 EW limit for the deterministic sweep data is **20×20**, not 17×17.

### `ROCKCHIP_EW_REDUCE=kahan`

For a fully unrolled ADD tree of MUL terms, the Rockchip lowering can replace the sequential sum with a Kahan-style compensated sum after symbolic simplification. The compensation therefore remains in the emitted program instead of simplifying back to zero. It uses only FP16 DPU EW MUL/ADD operations and represents the running error as another FP16 value.

- EW stages for reduction length K: `K + 7*(K-1) = 8K-7`.
- With the current 64-task software chain cap: N=21 uses 161 stages / 3 ioctls; N=32 uses 249 stages / 4 ioctls.
- Every square size N=21–32 passes the same 5e-3/5e-3 tolerance.
- N=33 currently reaches `RKPLAN_REJECT:unsupported_graph`: the optimizer only fully unrolls arbitrary reductions through K=32. This is now the next compiler ceiling, not a measured DPU accuracy ceiling.
- The Kahan sweep reaches N=32 under the same deterministic inputs.

The shared compensated DAG also required memoizing `lower_ew.visit`; without it, recursively revisiting shared nodes grew exponentially during lowering.

### 1×1 through 32×32 wall profile

Command configuration: `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=kahan`.
Inputs were realized before timing. `cold` is the first output `realize()` for the shape; `warm` is the median of the next three. Torch reference generation is excluded. All rows passed.

| N | cold ms | warm median ms | ioctls |
|---:|---:|---:|---:|
| 1 | 86.461 | 2.026 | 1 |
| 2 | 59.332 | 2.502 | 1 |
| 3 | 36.222 | 2.773 | 1 |
| 4 | 30.944 | 3.051 | 1 |
| 5 | 45.860 | 3.753 | 1 |
| 6 | 37.601 | 4.098 | 1 |
| 7 | 41.426 | 4.439 | 1 |
| 8 | 45.651 | 5.064 | 1 |
| 9 | 50.097 | 5.625 | 1 |
| 10 | 55.050 | 6.269 | 2 |
| 11 | 60.003 | 7.073 | 2 |
| 12 | 65.084 | 7.911 | 2 |
| 13 | 71.397 | 8.643 | 2 |
| 14 | 77.842 | 9.527 | 2 |
| 15 | 84.717 | 10.488 | 2 |
| 16 | 92.066 | 11.719 | 2 |
| 17 | 119.975 | 13.010 | 2 |
| 18 | 314.339 | 14.451 | 3 |
| 19 | 118.008 | 15.753 | 3 |
| 20 | 127.791 | 17.362 | 3 |
| 21 | 138.563 | 19.086 | 3 |
| 22 | 149.768 | 21.145 | 3 |
| 23 | 160.815 | 22.994 | 3 |
| 24 | 174.269 | 25.370 | 3 |
| 25 | 187.882 | 27.323 | 3 |
| 26 | 202.478 | 29.707 | 4 |
| 27 | 218.993 | 32.358 | 4 |
| 28 | 268.390 | 35.380 | 4 |
| 29 | 256.566 | 38.584 | 4 |
| 30 | 274.921 | 41.661 | 4 |
| 31 | 294.521 | 45.336 | 4 |
| **32** | **312.460** | **49.019** | **4** |

The N=18 cold result is an outlier; warm scaling is monotonic. These are end-to-end output-realization wall times, not isolated ioctl hardware time.

---

## 2026-08-07 — EW PC-chain cap raised from 64 to 256

The prior `_EW_CHAIN=64` was a conservative software cap, not a measured hardware limit. The compensated 32×32 GEMM has 249 dependent FP16 EW tasks, making it a direct chain test.

| Software cap | ioctls | Median wall (20 inputs) | Result vs cap 64 |
|---:|---:|---:|---:|
| 64 | 4 | 50.581 ms | baseline |
| **256** | **1** | **50.915 ms** | bit-exact 20/20 |

Additional single-ioctl dependent ADD-chain probes passed through 256 tasks. The real compensated GEMM also produced identical outputs at caps 64, 128, and 256; random-seed tolerance misses were identical across caps and are numerical, not chain corruption.

Set the OUT=2 `_EW_CHAIN=256`. The 32×32 compensated GEMM now fits in one ioctl. Wall time does not improve materially because emitting/patching 249 register bodies and the EW work dominate this small workload; the change removes submit boundaries rather than arithmetic.

This does **not** apply to OUT=5 mtx512: a 256-task `test_add` chain timed out. Keep `_EW_CHAIN_FP32=64`; the raised cap is specifically for the contiguous FP16-output EW recipe used by compensated GEMM.

---

## 2026-08-07 — compensated EW advanced one-by-one to 37×37

Rockchip's optimizer now permits full unroll through K=64. Both the heuristic threshold and the generic `UNROLL <= 32` guard had to be raised specifically for the ROCKCHIP renderer; before that, N=33 produced three stores and was rejected before submission.

Each size was run separately with a hard 30-second timeout under `ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=kahan`:

| N | Result | Worst tolerance ratio | ioctls | first-run wall |
|---:|---:|---:|---:|---:|
| 33 | pass | — | 2 | <1 s |
| 34 | pass | 0.829 | 2 | 666 ms |
| 35 | pass | 0.807 | 2 | 875 ms |
| 36 | pass | 0.852 | 2 | 934 ms |
| **37** | **pass** | **0.862** | **2** | **877 ms** |
| 38 | **fail** (1 element) | 1.188 | 2 | 1054 ms |

The Kahan sweep reaches 37×37. N=38 is a numerical limit: Kahan corrects ADD-rounding but cannot recover the product bits lost when each FP16×FP16 result is stored as FP16.

### N=37 merged into one PC chain

Kahan N=37 emits `8*37-7 = 289` dependent FP16 EW tasks, which crossed the previous 256-task cap. Raising the OUT=2 cap to 512 was bit-exact against cap 256 for 10/10 inputs:

| FP16 chain cap | ioctls | warm median wall |
|---:|---:|---:|
| 256 | 2 | 70.564 ms |
| **512** | **1** | **70.013 ms** |

Set `_EW_CHAIN=512` for OUT=2. OUT=5 mtx512 remains separately capped at 64.

---

## 2026-08-07 — FP16 TwoProduct EW reaches 64×64

Config: `ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=twoproduct`. DPU EW only, no CMAC/CNA and no host arithmetic.

Kahan cannot recover the low product bits after FP16 store. TwoProduct uses a Dekker split (`splitter=65`) to represent every FP16×FP16 product as two FP16 values: rounded product plus exact-ish residual. A general TwoSum then accumulates all product highs followed by residuals in a two-half `(high, low)` accumulator. The graph is constructed after symbolic simplification so the error terms remain as physical EW MUL/ADD tasks.

The mixed TwoProduct graph timed out with 512 tasks in one chain but passes with 256, so runtime uses `_EW_CHAIN_TWOPRODUCT=256` while ordinary OUT=2 Kahan retains 512.

Every size was run separately with a hard 30-second timeout. All N=38 through N=64 passed:

| N | ratio | ioctls | first-run wall ms | N | ratio | ioctls | first-run wall ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 38 | 0.117 | 7 | 2192 | 52 | 0.126 | 9 | 5608 |
| 39 | 0.115 | 7 | 2576 | 53 | 0.127 | 10 | 5853 |
| 40 | 0.093 | 7 | 2738 | 54 | 0.173 | 10 | 6194 |
| 41 | 0.169 | 8 | 3464 | 55 | 0.148 | 10 | 6502 |
| 42 | 0.106 | 8 | 3125 | 56 | 0.145 | 10 | 6846 |
| 43 | 0.074 | 8 | 3336 | 57 | 0.152 | 10 | 7182 |
| 44 | 0.117 | 8 | 3549 | 58 | 0.136 | 10 | 7717 |
| 45 | 0.120 | 8 | 3895 | 59 | 0.180 | 11 | 7980 |
| 46 | 0.078 | 8 | 4005 | 60 | 0.158 | 11 | 8350 |
| 47 | 0.149 | 9 | 4215 | 61 | 0.134 | 11 | 8849 |
| 48 | 0.105 | 9 | 4494 | 62 | 0.155 | 11 | 9554 |
| 49 | 0.160 | 9 | 4854 | 63 | 0.144 | 11 | 9888 |
| 50 | 0.151 | 9 | 5113 | **64** | **0.145** | **11** | **10106** |
| 51 | 0.150 | 9 | 5281 | | | | |

The authoritative GEMM milestone is now `test_gemm_fp16` at 64×64. It passes in TwoProduct mode below the 30-second wall cap with substantial numerical margin. The redundant incremental `test_medium_gemm` was removed; its one-by-one census remains documented above.

### `test_gemm_fp16` cold wall decomposition

Profile config matches the passing test: `DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=twoproduct`. The named pytest passes in **18.62 s** with `-n12`; a directly instrumented cold realization takes **7.207 s**, so approximately **11.4 s** is pytest/xdist worker startup and imports.

| Cold realization component | Wall | Share |
|---|---:|---:|
| render + `lower_ew` | 6293 ms | 87.3% |
| program `__call__` | 404 ms | 5.6% |
| scheduler/program setup outside those regions | 509 ms | 7.1% |
| **output `realize()`** | **7207 ms** | **100%** |

The TwoProduct DAG itself is cheap to construct (17.9 ms). Most of the 6.24 s inside `lower_ew` is plan bookkeeping over the 2806-op graph: dependency/use counting, scratch assignment, and gather-offset construction. This is the dominant next optimization target, not ioctl count.

Program-call decomposition:

| Runtime component | Wall |
|---|---:|
| gather/setup/final-output work outside EW runner | 276.4 ms |
| EW runner total | 128.1 ms |
| ├ command emit (2806 bodies) | 40.6 ms |
| ├ relocation patch | 17.9 ms |
| ├ submit CPU construction | 29.1 ms |
| ├ 11 blocking ioctls | 14.0 ms |
| └ loop/other | 26.5 ms |

Other excluded measurements: input Tensor construction 3.6 ms, Torch reference 0.9 ms, and final `numpy()` copyout 2.7 ms.

### Cold-lowering optimization

The 7.207 s direct cold realization was Python compiler overhead rather than DPU execution. `lower_ew` calculated each logical input gather four times because repeated leaf UOps were only deduplicated after offset expansion, and its dependency-use count scanned the whole graph once per node.

Caching lowered leaf operands and replacing the quadratic use-count comprehension with one linear edge walk preserves the output (worst tolerance ratio remains 0.145) and changes the cold profile as follows:

| Component | Before | After | Change |
|---|---:|---:|---:|
| direct output `realize()` | 7.207 s | 2.732 s | 2.64× faster |
| `lower_ew` | 6.306 s | 1.762 s | 3.58× faster |
| gather-offset construction | 4.170 s / 512 calls | 1.054 s / 128 calls | four duplicate calls removed |
| other lowering work | 2.119 s | 0.691 s | 3.07× faster |
| named pytest (`-n12`) | 18.62 s | 14.39 s | 4.23 s less wall time |

`test_gemm_fp16` still passes under its hard 30-second timeout. The remaining roughly 1.05 s in gather-offset construction is now the largest measured lowering component.

---

## 2026-08-07 — FP16-only cleanup and CPU-execution audit

The Rockchip backend now exposes only the hardware path under test: contiguous FP16 DPU EW output. The legacy mtx512 `OUT=5` path was removed because it converted FP32 results back to FP16 on the host after every operation. FP32 argument casting, host pack/unpack helpers, and the intermediate-result copyout path were removed with it.

`sz.py` result for executable Rockchip code:

| File | Before | After | Reduction |
|---|---:|---:|---:|
| `tinygrad/renderer/rockchip.py` | 438 | 325 | 113 |
| `tinygrad/runtime/ops_rockchip.py` | 241 | 131 | 110 |
| **combined** | **679** | **456** | **223 (32.8%)** |

Comments and docstrings were retained because `sz.py` does not count them. Repository total lines fell from 25,730 to 25,507.

### Execution audit

- There is no CMAC/CNA path.
- There is no NumPy import, FP32 temporary, `astype`, host sum, or host ADD/MUL in the Rockchip renderer/runtime.
- Contiguous FP16 arguments are bound directly as DPU sources; they are no longer copied into scratch.
- The final DPU EW operation writes directly to the destination argument; there is no host result copy.
- Constants and constant-only fills are materialized as FP16 bytes on the host. This is data initialization, not tensor arithmetic.
- GEMM still needs host gather/layout preparation because DPU EW only consumes contiguous pairwise vectors and cannot apply tinygrad's arbitrary load indexes. Masked padding is represented by `-1` gather entries and zero-filled. Every GEMM MUL/ADD, including TwoProduct residual recovery, executes on DPU EW.
- The safe PC-chain limit is stored in the compiled image (512 for ordinary/Kahan graphs and 256 for mixed TwoProduct graphs), removing the previous runtime dependence on the current environment variable.
- RKImage version validation is now strict, and malformed/nonzero masked-load defaults are rejected instead of silently miscomputed.

This is closer to other tinygrad hardware backends: allocator methods move buffers, `Program.__call__` binds arguments/builds commands/submits, and the device performs tensor arithmetic. Unlike general GPU/DSP backends, Rockchip still preprocesses indexed operands on the host and uses blocking ioctls rather than an asynchronous hardware queue, so it remains a focused DPU-EW capability harness rather than a general backend.

Verification after cleanup:

- `test_gemm_fp16`: **1 passed in 13.30 s** with `-n12` and a hard 30 s timeout.
- Direct 64×64 cold realization: **1.975 s**, 11 ioctls, within the 5e-3/5e-3 tolerance.
- Non-slow Rockchip census: **18 passed, 5 skipped in 12.16 s**. Four skips are slow tests; `test_mul_naninf` records that RK3588 FP16 DPU EW MUL returns NaN for infinity operands rather than hiding it with a CPU fallback.
- Full `ruff check .`: pass.
- Full `mypy tinygrad/`: pass (216 files).

---

## 2026-08-07 — 256×256 affine-gather profile and optimization

With a 180-second cap, Rockchip's FP16-tolerance `test_big_gemm` first passed in 110.55 s. The original `TestOps.test_big_gemm` completed in 136.96 s but missed its stricter 1e-4/1e-3 tolerance on 35 of 65,536 elements (maximum absolute difference 0.000977).

The initial direct cold profile took 108.738 s and submitted 88 ioctls:

| Component | Before | Share |
|---|---:|---:|
| render/lower | 86.511 s | 79.6% |
| compile-time gather-offset expansion | 83.235 s | 76.5% |
| runtime host gather/layout | 13.551 s | 12.5% |
| other scheduler/program setup | 7.384 s | 6.8% |
| DPU EW runner | 1.292 s | 1.2% |
| 88 blocking ioctls | 0.614 s | 0.6% |

The kernel has 11,254 logical EW operations over 65,536 output elements. The tested 64,000-element tile cap splits every operation in two, producing 22,508 DPU tasks. At the TwoProduct 256-task chain cap this is `ceil(22508/256) = 88` submits.

Raising the tile cap is unsafe. A 65,536-element one-ioctl ADD (`WIDTH=8191`) completed with 99.8% wrong values, while 65,528 (`WIDTH=8190`) timed out in the ioctl. The NPU was reset after the probe and `_MAX_EW_ELEMS_FP16` remains 64,000.

Large unmasked gathers are now encoded as a base plus affine `(destination divisor, range limit, source stride)` axes rather than a complete offset tuple. Masked/padded operations keep the general offset fallback. Runtime materializes both forms with bulk uint16 indexing; this performs only layout movement and integer address generation on the host, while every floating-point MUL/ADD remains on DPU EW.

After the change:

| Component | Before | After |
|---|---:|---:|
| direct cold realization | 108.738 s | **4.908 s** |
| render/lower | 86.511 s | **0.229 s** |
| gather-plan construction | 83.235 s | **0.005 s** |
| runtime host gather/layout | 13.551 s | **1.664 s** |
| serialized RKImage | 128 MiB of raw offsets | **475,838 bytes** |
| DPU tasks / ioctls | 22,508 / 88 | **22,508 / 88** |

The named Rockchip `test_big_gemm` now passes in 16.93 s including fresh pytest/xdist startup. The non-slow Rockchip census remains green (18 passed, 5 skipped), and full Ruff/mypy pass.

---

## 2026-08-07 — convolution census begins with 1x1

The generic `TestOps.test_simple_conv2d_1x1` case, input `(1,4,9,9)` and weight `(4,4,1,1)`, passes on Rockchip with FP16 input and DPU EW-only execution. It is now mirrored in `test/backend/test_rockchip.py` so the backend census uses the same `5e-3/5e-3` tolerance as `test_gemm_fp16` and asserts the hardware submit count.

Direct cold realization took **0.193 s**, used **1 ioctl submit**, and had maximum absolute error **1.52588e-05** against the Torch FP16 reference. The generic pytest case passed in **11.34 s** including fresh xdist worker startup.

The next representative 3x3 sweep also passes:

| Case | Output | Cold realization | Ioctl submits | Maximum absolute error |
|---|---:|---:|---:|---:|
| unpadded | `(1,4,7,7)` | 0.449 s | 7 | 0.00195312 |
| batch 2 | `(2,4,7,7)` | 0.293 s | 7 | 0 |
| padding 1 | `(1,4,9,9)` | 0.491 s | 7 | 1.52588e-05 |
| stride 2 | `(1,4,4,4)` | 0.295 s | 7 | 0 |
| depthwise, 4 groups | `(1,4,7,7)` | 0.102 s | 2 | 0 |

These cases remain FP16-input, DPU EW-only tests. Torch is used only to construct the test oracle; backend execution performs no host floating-point arithmetic.

The six-test convolution subset passes in **12.21 s** and the complete Rockchip census passes in **18.08 s** (`28 passed, 1 skipped`), both with `-n12` and a hard 30-second timeout. Full Ruff and mypy checks pass.

### Convolution reduction width and layout expansion

The convolution census now also covers both sides of K=64, the larger M4 cases from `test_ops`, grouped convolution, and dilation:

- 3x3 with 7 input channels (K=63): 11 submits, 0.629 s direct cold realization.
- 3x3 with 8 input channels (K=72): 13 submits, 0.676 s; this proves 64 reduction terms is not the current backend limit.
- 3x3 with 16 input/output channels (K=144): 25 submits, 1.195 s.
- 1x1 with 16 input/output channels at 32x32: 3 submits, 0.169 s.
- Two-group 3x3: 7 submits, 0.471 s.
- Dilated 3x3: 7 submits, 0.279 s.

All twelve Rockchip convolution tests pass in **13.42 s**. The complete backend census is now **34 passed, 1 skipped in 17.58 s**, still under the hard 30-second timeout.

### Convolution index and masked-layout fixes

Rockchip now evaluates `CDIV`, `CMOD`, `FLOORDIV`, and `FLOORMOD` while constructing integer gather indices. This is host address/layout calculation only; no floating-point operation moved off DPU EW. Output-padded transposed convolution no longer rejects its index expression and passes the allowed FP16 tolerance (1 submit, 1.806 s direct cold realization).

The optimizer no longer upcasts masked Rockchip axes into several scalar stores, preserving the renderer's single contiguous-store contract. Asymmetric 1D and 2D padding and padded 3D convolution now pass their original `test_ops` checks. A zero-select `WHERE` around a masked load is folded into that load's gather gate, so simple explicit padding also passes without CPU floating-point execution. The asymmetric 2D backend census case realizes in 0.469 s with 1 submit and exact output in the measured probe.

### Complete convolution census and native ReLU

The Rockchip census now mirrors every one of the 48 `test*conv*` methods from `test/backend/test_ops.py` and applies exactly the `test_gemm_fp16` ceiling (`atol=5e-3`, `rtol=5e-3`, including gradient tolerances). It stays synchronized with future convolution additions while the existing explicit Rockchip cases continue to assert exact ioctl-submit counts.

Native DPU ReLU is emitted by leaving `EW_RELU_BYPASS` clear after an EW pass-through. Convolution ADD trees may include a bias term, and the TwoProduct reduction now carries a three-half expansion so wider Conv2D reductions and nested bias/ReLU graphs remain within the FP16 contract. The resulting submit counts are recorded in the explicit backend tests rather than hidden behind a CPU path.

The formerly opaque EW words are now composed from the RK3588 register fields named by the hardware reference: FP16 data mode and element size, ALU algorithm, ReLU/LUT bypass, operand source/conversion, and operation type. These definitions reproduce the verified words exactly:

| Operation | `DPU_EW_CFG` |
|---|---:|
| ADD | `0x108202c0` |
| MUL | `0x108003c4` |
| native ReLU | `0x108000c0` |

Two cases use an FP32-accumulated Torch golden cast back to FP16: biased ConvTranspose2D and ConvTranspose3D. PyTorch CPU's FP16 reduction is less accurate for these cases; for example, the biased 2D outlier is `-0.60840` from CPU FP16 versus an exact FP16-input sum of `-0.61804`, while DPU EW returns `-0.61816`. Against the FP32 golden, the complete biased case has maximum absolute error `0.001953`, and ConvTranspose3D has zero tolerance violations with maximum absolute error `0.015625`. Torch remains test-oracle code only.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- All mirrored convolution cases: **42 passed, 6 intentionally skipped, 37 subtests passed in 42.51 s**.
- Complete `test/backend/test_rockchip.py`: **78 passed, 7 skipped, 37 subtests passed in 51.06 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No CMAC/CNA execution, host floating-point fallback, or relaxed tolerance was added. Host work remains integer gather/layout preparation; all convolution MUL/ADD/ReLU arithmetic executes through DPU EW with FP16 buffers.

---

## 2026-08-07 — DPU EW binary MAX and FP16 MaxPool2D census

The Rockchip renderer now distinguishes the two uses of the EW MAX datapath. Binary MAX sets `EW_RELU_BYPASS`, producing the register word `0x108002c0` verified by `~/rk3588/examples/elementwise.py` and the upstream research renderer. Native unary ReLU keeps that bypass clear and remains `0x108000c0`. Both words are assembled from named RK3588 register fields rather than embedded hex literals.

Pooling remains a DPU-EW limit experiment. The upstream pooling examples often switch to PPU or CMAC for layout/reformat and large reductions; none of those paths were imported. MaxPool windows are expanded into binary FP16 EW MAX operations, and the existing host gather performs integer address/layout preparation only.

Padded pooling required preserving the FP16 fill value in gather metadata. RKImage v14 stores the raw 16-bit fill pattern, allowing masked lanes to be initialized to `-inf` without host floating-point tensor arithmetic. The rewrite only moves a `WHERE` mask into a load when it proves that the outer condition implies the load condition, including commuted `AND` expressions. MAX-tree masks are folded only when every leaf has the same or exact complementary condition. This covers symmetric padding, asymmetric padding, and ceil-mode edge windows without a general unsafe `WHERE` rewrite.

Every numeric FP16 `test_max_pool2d*` method from `test/backend/test_ops.py` now passes at exactly the `test_gemm_fp16` tolerance ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`):

- **11 methods passed, 33 subtests passed** in 14.24 s on the NPU.
- `test_max_pool2d_padding_int` is explicitly skipped because Rockchip accepts FP16 inputs only.
- `test_max_pool2d_return_indices` is explicitly skipped because DPU EW does not produce the required integer index tensor.
- A direct FP16 binary-maximum regression and a MaxPool submit-count regression were added to `test/backend/test_rockchip.py`.

Direct cold output-realization profiles all used one `DRM_IOCTL_RKNPU_SUBMIT`:

| Case | Wall time | Submits | Maximum absolute error |
|---|---:|---:|---:|
| `(1,1,2,3)`, 2x2 | 0.110 s | 1 | 0 |
| `(3,2,17,14)`, 5x5 stride 1 | 0.141 s | 1 | 0 |
| `(4,2,111,28)`, asymmetric-padded 5x5 | 0.474 s | 1 | 0 |

Complete verification after the change:

- `test/backend/test_rockchip.py`: **91 passed, 9 skipped, 70 subtests passed in 44.94 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 445 executable lines, runtime 142; comments and docstrings retained.
- CPU-execution audit: runtime NumPy use is limited to `uint16` buffer views and integer gather-index construction. There is no host MAX, ADD, MUL, reduction, FP32 conversion, CMAC, or CNA path.

AvgPool2D is the next pooling boundary. Five initial cases already execute, while the remaining forms expose FP32 `MUL/CAST` lowering and, for `count_include_pad=False`, dynamic reciprocal/mask expressions. Those are separate ADD/MUL lowering work and were not hidden by a CPU fallback in this MAX milestone.

---

## 2026-08-07 — FP16 AvgPool through DPU EW, including valid-count divisors

All `test_avg_pool*` cases from `test/backend/test_ops.py` now run through the FP16 DPU EW backend. No PPU, CMAC, CNA, or CPU tensor fallback was added. The PPU average-pooling recipes in `~/rk3588` and `rockchip-upstream-research` were used only to confirm hardware semantics; this branch deliberately continues to test the DPU EW ADD/MUL limit.

Fixed-divisor AvgPool lowers the FP32-shaped tinygrad epilogue to FP16 ADD/MUL and folds compile-time casts of scalar constants. The reduction sum and final scale multiply execute on DPU EW. Across the unpadded and padded 2D sweep, all 14 kernel/padding combinations used one ioctl and the largest measured absolute error was `0.0009765625`, below the unchanged `5e-3/5e-3` FP16 tolerance ceiling.

`count_include_pad=False` requires a divisor that varies at edges. This is still not host tensor arithmetic: the divisor depends only on static output geometry, never on input values. The compiler evaluates the geometry-only RANGE/WHERE expression and stores its raw FP16 reciprocal bits in RKImage v15. At program launch, the runtime copies those immutable `uint16` constants into scratch. It performs no floating-point reciprocal, sum, multiply, mean, or conversion. Every input-dependent ADD and MUL is submitted to DPU EW. A dedicated regression asserts that the padded valid-count path performs one NPU ioctl.

Average pooling also exposed the previous 64-term compiler unroll ceiling. Rockchip now permits scaled ADD reductions through 512 terms, while GEMM/convolution and other reductions keep their proven 64-term heuristic bound. This enables the 308-term global 11x28 AvgPool and the 512-term AvgPool3D case without changing convolution lowering. The reciprocal-leaf detector is restricted to static `RECIPROCAL` roots, avoiding a quadratic graph walk in convolution compilation.

Representative direct cold profiles, excluding Torch reference generation:

| Case | Wall time | Submits | Maximum absolute error |
|---|---:|---:|---:|
| 3x3 fixed divisor, `(32,2,11,28)` | 0.217 s | 1 | 0.0004883 |
| 3x3 valid-count padding, `(32,2,11,28)` | 0.473 s | 1 | 0.0004883 |
| global 11x28, `(32,2,11,28)` | 1.032 s | 2 | 0.0003052 |
| AvgPool3D 8x8x8, `(1,1,16,16,16)` | 7.981 s | 3 | 0.0003357 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- AvgPool census: **10 passed, 26 subtests passed in 20.58 s**.
- Convolution regression census after the unroll change: **42 passed, 6 skipped, 37 subtests passed in 45.66 s**.
- Complete Rockchip census: **102 passed, 9 skipped, 96 subtests passed in 60.12 s**.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 498 executable lines, runtime 144; comments and docstrings retained.
- Runtime CPU audit: NumPy is used only for raw `uint16` views and integer indexing/copies. There is no host floating-point arithmetic or conversion.

PyTorch CPU does not implement FP16 AvgPool3D, so that one test constructs its oracle in FP32 and casts the oracle to FP16. This affects test-reference generation only; Rockchip execution remains FP16 DPU EW.

---

## 2026-08-08 — First post-518 milestone: nearest interpolation through DPU EW

Development restarted from the known-good `518487a5c24c7390e5b40cdb85f91d6e158c4383` baseline. The first newly admitted upstream method is `TestOps.test_interpolate_nearest`, covering six 1D, 2D, and 3D resize shapes. It passes at the unchanged `test_gemm_fp16` tolerance ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`). A direct `(2,3,13) -> (2,3,9)` regression additionally asserts exactly one NPU ioctl.

Nearest interpolation is a gather, but the output is not accepted as a host-produced result. Rockchip prepares only the static integer gather offsets and mask, then sends the gathered FP16 values through a DPU EW `ADD 0` pass-through stage. Typed compile-time evaluation now preserves integer, boolean, FP16, and FP32 cast semantics in layout expressions. No input-dependent floating-point arithmetic, CMAC, CNA, PPU, or CPU numeric fallback was added.

Focused verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Direct submit-count regression: **1 passed in 2.83 s**, exactly 1 ioctl.
- Full upstream `test_interpolate_nearest`: **1 passed in 3.96 s**, covering all six resize shapes.
- Gather regression group (nearest, MaxPool, AvgPool, output-padded ConvTranspose2D): **4 passed in 8.13 s**.
- Complete Rockchip census, run as 113 separate sequential pytest processes to limit retained program buffers: **104 passed, 9 intentionally skipped**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 501 executable lines, runtime 144; comments and docstrings retained.

The current `518` runtime still requests contiguous GEM buffers against an 8 MiB CMA pool. Several large existing cases logged recoverable CMA allocation failures even with one pytest node per process, although this run produced no RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. Fixing this allocation policy safely is the next prerequisite before broadening the interpolation census.

---

## 2026-08-08 — Page-backed RKNPU buffers remove the 8 MiB CMA limit

All Rockchip GEM objects now use the allocation policy captured from the vendor RKNN runtime: ordinary data, scratch, and command buffers use `NON_CONTIGUOUS|CACHEABLE|IOMMU_LIMIT_IOVA_ALIGNMENT` (`0x403`), while task buffers additionally use `KERNEL_MAPPING` (`0x40b`). This selects the driver's page-backed IOMMU allocation path directly instead of first exhausting the board's 8 MiB CMA pool and entering the driver's broken contiguous-to-noncontiguous fallback.

Cacheable mappings require explicit ownership transfer. Copy-in, copy-out, command/task construction, constants, gathers, and fills now issue the corresponding `MEM_SYNC` ioctl. Before each program, every argument is synchronized from and then back to the device, matching the vendor behavior. A narrower gather-only synchronization experiment produced stale zero cache lines in `test_add3` after allocator reuse, so the broader handoff is required for correctness on this driver.

Allocation is transactional: a failed create is reported as `MemoryError`, while a failed map/mmap destroys the newly created GEM before propagating the allocation failure. Program-owned scratch and command/task lifetime, PC-chain construction, and the passing `518` submission path remain unchanged.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Tiny ADD health probe: **1 passed in 2.83 s**, exactly 1 ioctl.
- Large GEMM + nearest gather + chained ADD: **3 passed in 6.89 s**.
- Complete monolithic Rockchip census: **104 passed, 9 skipped, 96 subtests passed in 169.07 s**.
- Post-coherency regression (large GEMM, nearest, chained ADD, output-padded ConvTranspose2D): **4 passed in 11.38 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 501 executable lines, runtime 165; comments and docstrings retained.

Across the complete monolithic run, `CmaFree` stayed at 6144 KiB and the kernel logged no CMA allocation failure, RKNPU timeout, invalid IRQ, IOMMU fault, or oops. The allocator safety prerequisite is therefore cleared. The next performance target is the fresh-process `test_simple_conv_transpose3d` node, previously measured at 65.21 s.

---

## 2026-08-08 — Vectorized static gather planning removes ConvTranspose3D compile stall

`TestRockchipConvOps.test_simple_conv_transpose3d` previously passed but took **65.21 s** in a fresh pytest process. `DEBUG=2` separated the device work from compilation: scheduling took 182.05 ms, two uploads took 0.25 ms total, and the DPU program took 271.54 ms. Almost the entire remaining minute was Rockchip renderer CPU time.

A CPU profile measured 112.88 s under profiling in `lower_ew`. The 108 masked convolution gathers called the scalar index interpreter 58.6 million times; `_eval_expr` consumed 108.43 s. Per-output expression memoization reduced the ordinary test from 65.21 s to 45.16 s, confirming repeated static-index evaluation as the cause but not removing enough Python iteration.

Rockchip now evaluates complete static gather-index vectors with NumPy. RANGE values, typed casts, comparisons, masks, integer division/modulo, and affine output addresses are computed as vectorized layout metadata. This remains host integer address preparation only: no input-dependent FP16 value, convolution product, sum, or activation is computed on CPU. The existing runtime gather copies FP16 bit patterns according to those offsets, and all numeric convolution work still runs through DPU EW.

Results after vectorization:

- ConvTranspose3D pytest time: **6.32 s** (down from 65.21 s, 10.3x faster).
- Shell wall time including imports: **8.604 s**.
- Scheduling: 170.26 ms; NPU program: 274.85 ms.
- Complete monolithic Rockchip census: **104 passed, 9 skipped, 96 subtests passed in 67.19 s** (down from 169.07 s after cache-sync enablement).
- Focused nearest/AvgPool/ConvTranspose regressions: **4 passed in 7.38 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 552 executable lines, runtime 165; comments and docstrings retained.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. Every individual Rockchip node is again below 30 seconds.

---

## 2026-08-08 — `test_interpolate_nearest_exact`

The next upstream method was added individually to the Rockchip census. `TestOps.test_interpolate_nearest_exact` reuses the six 1D, 2D, and 3D shapes from nearest interpolation with PyTorch's `nearest-exact` coordinate rule. All static index casts and masks are prepared by the vectorized Rockchip gather planner, and every FP16 result passes through DPU EW before reaching the output.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused upstream method: **1 passed in 3.41 s**; shell wall time 5.574 s.
- Complete Rockchip census: **105 passed, 9 skipped, 96 subtests passed in 67.96 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 552 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_linear`.

---

## 2026-08-08 — `test_interpolate_linear`

The upstream 1D linear interpolation method now passes both resize directions, `(2,3,52) -> (2,3,29)` and `(2,3,29) -> (2,3,52)`, at the unchanged FP16 tolerance. A direct regression asserts that the first direction uses exactly one ioctl.

Interpolation coordinates and fractional weights depend only on shape. Rockchip retains those FP32 cast/WHERE/TRUNC expressions as static geometry and uploads their final FP16 bit patterns. It no longer rewrites geometry-only FP32 ADD/MUL into dynamic nodes. The gathered low/high FP16 input values and every input-dependent lerp MUL/ADD execute through DPU EW; there is no CPU interpolation or input-dependent arithmetic.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused direct and upstream methods: **2 passed in 3.06 s**; direct case exactly 1 ioctl.
- Complete Rockchip census: **107 passed, 9 skipped, 96 subtests passed in 101.45 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_linear_corners_aligned`.

---

## 2026-08-08 — `test_interpolate_linear_corners_aligned`

The second upstream 1D linear interpolation method passes both resize directions with `align_corners=True`. This changes only the static coordinate/weight formula; gathered FP16 values and lerp arithmetic continue through the already verified DPU EW path.

- Focused upstream method: **1 passed in 3.03 s**.
- Complete admitted interpolation group: **4 passed in 4.38 s**.
- Rockchip collection: 117 nodes (the previous full census was 107 passed / 9 skipped; this new node passed focused validation).
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next upstream interpolation case is `test_interpolate_bilinear`.

---

## 2026-08-08 — `test_interpolate_bilinear` without shared-core changes

The upstream bilinear method now passes all three 2D resize pairs. A direct `(2,3,12,20) -> (2,3,9,31)` regression asserts exactly two ioctls, one DPU EW program per separable axis.

Tinygrad's generic interpolation keeps the first-axis result in an internal FP32-sized buffer. Rockchip still advertises only FP16 as a supported dtype and performs no FP32 arithmetic or conversion. For Rockchip programs, that oversized internal buffer now carries a contiguous FP16 payload produced by the first DPU pass; the second Rockchip pass reads the same FP16 payload. This preserves the device's actual FP16 storage/arithmetic contract without changing `tinygrad/mixin/op.py` or any other shared core file.

The host continues to prepare static integer gathers and geometry-only FP16 weights. Every input-dependent low/high interpolation MUL/ADD executes on DPU EW. No CPU interpolation, CMAC, CNA, or PPU path was added.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused direct and upstream methods: **2 passed in 4.29 s**; direct case exactly 2 ioctls.
- Complete Rockchip census: **110 passed, 9 skipped, 96 subtests passed in 101.59 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_bilinear_corners_aligned`.

---

## 2026-08-08 — `test_interpolate_bilinear_corners_aligned`

The aligned-corners bilinear method passes all three upstream 2D resize pairs through the same two-pass FP16 DPU EW path.

- Focused upstream method: **1 passed in 4.04 s**.
- Complete admitted interpolation group: **6 passed in 6.92 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next upstream interpolation case is `test_interpolate_trilinear`.

---

## 2026-08-08 — `test_interpolate_trilinear`

The upstream trilinear method passes `(2,3,5,2,8) -> (2,3,3,6,4)`. A smaller direct regression confirms exactly three ioctls, one FP16 DPU EW lerp pass for each spatial axis. Both internal FP32-sized separable buffers carry only contiguous FP16 NPU-produced payloads.

- Focused direct and upstream methods: **2 passed in 3.61 s**; direct case exactly 3 ioctls.
- Complete admitted interpolation methods plus four submit-count regressions: **11 passed in 7.94 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The final floating-point interpolation method is `test_interpolate_trilinear_corners_aligned`.

---

## 2026-08-08 — Complete FP16 interpolation census

`test_interpolate_trilinear_corners_aligned` passes its upstream 3D resize case, completing all eight floating-point `test_interpolate*` methods from `TestOps`: nearest, nearest-exact, 1D linear, aligned 1D linear, bilinear, aligned bilinear, trilinear, and aligned trilinear.

All nearest values and linear low/high operands are gathered as FP16 bit patterns. Static geometry supplies only indices and FP16 weights. Input-dependent interpolation arithmetic executes on DPU EW in one, two, or three passes according to spatial rank. Rockchip remains FP16-only, performs no CPU interpolation, and changes no shared tinygrad core file.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused aligned-trilinear method: **1 passed in 3.24 s**.
- Complete Rockchip census: **114 passed, 9 skipped, 96 subtests passed in 104.06 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The floating-point interpolation group is complete; uint8 interpolation remains outside the FP16 input contract.

---

## 2026-08-08 — `test_full_like` and `test_full`: typed raw constant fill

The remaining `TestOps` census is now advancing in source order. `test_full_like` passes its explicit FP32 and int32 outputs, and `test_full` passes its integer output. These require no numeric NPU operation: each tensor is a compile-time scalar repeated into storage.

RKImage v16 records the fill element width in the previously reserved fill byte. The compiler serializes the scalar using the destination dtype's exact byte format, and the runtime copies those immutable bytes across the destination before synchronizing the cacheable buffer to the device. Dynamic Rockchip arithmetic remains FP16-only; this is raw constant initialization, not CPU input-dependent arithmetic or FP32 emulation.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- `test_full_like`: **1 passed in 2.66 s**.
- `test_full`: **1 passed in 2.61 s**.
- Complete Rockchip census: **116 passed, 9 skipped, 96 subtests passed in 109.58 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 556 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next source-order methods are the negative-dimension exception tests.

---

## 2026-08-08 — Negative-dimension validation methods

Four source-order `TestOps` methods were admitted and run individually: `test_negative_dims`, `test_negative_dims_full`, `test_negative_dims_eye`, and `test_negative_dims_kaiming`. Every invalid shape raises before device allocation or submission.

- Individual times: 2.59 s, 2.58 s, 2.57 s, and 2.51 s.
- Incremental creation/validation group: **6 passed in 2.82 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).

The next source-order method is `test_zeros`.

---

## 2026-08-08 — Creation methods through `test_eye`

Six more source-order methods were admitted and run individually: `test_zeros`, `test_zeros_like`, `test_empty_0`, `test_ones`, `test_ones_like`, and `test_eye`. Uniform values use typed raw constant fill. `test_empty_0` folds to a constant NaN fill. Eye matrices are geometry-only FP16 static vectors followed by DPU EW pass-through, not CPU-generated numeric outputs.

- Individual methods: all pass in 2.66–3.03 s.
- Complete incremental creation/validation group: **12 passed in 3.15 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_split`.

---

## 2026-08-08 — Movement and range construction through `test_linspace`

Eight more source-order methods were admitted and run individually: `test_split`, `test_chunk`, `test_unfold`, `test_meshgrid`, `test_arange`, `test_arange_big`, `test_arange_4096`, and `test_linspace`.

Split/chunk remain views where possible; materialized FP16 movement uses the existing gather plus DPU pass-through path. Range/linspace values depend only on their scalar construction arguments and are materialized as static typed data. This includes FP16, int8, int32, and int64 output checks and does not introduce input-dependent CPU arithmetic.

- Individual method times: 2.92–5.72 s.
- Complete incremental source-order group: **20 passed in 4.94 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_sum_fake`.

---

## 2026-08-08 — Source-order sum/max reductions

Seven more methods were admitted and run individually: `test_sum_fake`, `test_sum_collapse`, `test_sum_collapse_neg`, `test_sum_pad_collapse`, `test_sum_twice`, `test_sum_cat_collapse`, and `test_max_dont_collapse`.

Geometry/constant reductions fold where legal. `test_sum_twice` reduces random FP16 input across two stages and therefore exercises the existing DPU EW ADD lowering rather than a host arithmetic path. No tolerance or backend code change was needed.

- Individual method times: 2.64–4.05 s.
- Complete incremental source-order group: **27 passed in 9.08 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_where`.

---

## 2026-08-08 — Lerp, broadcast ADD, SUB, and negation

Seven FP16 arithmetic methods were admitted and run individually: `test_lerp`, `test_broadcasted_add`, `test_broadcasted_add_2`, `test_sub`, `test_scalar_sub`, `test_scalar_rsub`, and `test_neg`. They reuse DPU EW ADD/MUL and broadcast gathers; no renderer/runtime change was necessary.

- Individual method times: 2.70–3.07 s.
- Complete incremental source-order group: **34 passed in 9.58 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

`test_where` and the boolean tail of `test_tril` remain unadmitted because their packed-boolean input/int32 output cannot be executed by the FP16-only DPU without a CPU value-selection fallback. They are not skipped or counted as passing.

---

## 2026-08-08 — Exact upstream ADD and MUL methods

Six additional upstream `TestOps` methods were admitted: `test_tiny_add`, `test_tiny_mul`, `test_add`, `test_add3`, `test_mul`, and `test_scalar_mul`. Each method first passed in its own fresh pytest process, then the complete incremental class passed sequentially in one persistent process.

- Individual method times: 2.62–3.07 s.
- Complete incremental source-order group: **40 passed in 9.75 s**.
- No new RKNPU, IOMMU, CMA, or kernel-oops message during the grouped run.

The exact upstream `test_div` method is deliberately not counted yet. Its first dynamic graph is represented as `MUL(RECIPROCAL(x))`, and the Rockchip planner rejects `RECIPROCAL` before any ioctl. Native FP16 DPU division will be developed as a separate hardware milestone, beginning with a tiny one-submit regression before admitting the complete upstream method.

---

## 2026-08-08 — Native FP16 DPU division

Dynamic half division now uses RK3588's native DPU EW FDIV algorithm. Advertising typed `Ops.FDIV` lets tinygrad fuse `x * reciprocal(y)` before Rockchip lowering. The emitted stage uses named FDIV configuration bits, the FDIV output conversion scale, and FDIV's RDMA operand mode. No reciprocal or division is evaluated on the CPU, no CMAC path is used, and no shared tinygrad core file changed.

The hardware rollout was deliberately sequential:

- Three-element mixed-sign regression: **1 passed in 2.84 s**, exactly **1 ioctl**.
- Exact upstream `TestOps.test_div`: **1 passed in 3.18 s**.
- Complete incremental source-order group: **41 passed in 9.76 s**.
- Complete Rockchip census: **156 passed, 9 skipped, 96 subtests passed in 111.21 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU, IOMMU, CMA, or kernel-oops message. Division-by-zero/non-finite semantics and rounding-mode division remain separate cases; this milestone covers the exact ordinary floating-point `test_div` method only.

---

## 2026-08-08 — Scalar FP16 division

The exact upstream `TestOps.test_scalar_div` method passes all seven tensor/scalar and scalar/tensor forms through native DPU FDIV, including two rank-zero cases. It passed individually in **3.07 s** without a renderer/runtime change.

Non-finite numerator handling is covered by the following separate milestone.

### Non-finite division sign

The first `-inf/x` sign reconstruction (`MUL → MAX → FDIV → MUL → FDIV`) timed out at task 4/5. That version was never committed. The replacement rewrites `±inf/x` as `(±1/x)/0`, producing two adjacent FDIV stages with no intervening pipeline transition. The vendor reference elementwise health check subsequently passed all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 elements, allowing the replacement to be verified on the same boot.

- Separate `+inf/x` and `-inf/x` probes: bit-exact pass.
- Direct sign regression: **1 passed in 2.64 s**, exactly one ioctl for each sign.
- Exact upstream `TestOps.test_div_naninf`: **1 passed in 2.75 s**.
- Complete incremental source-order group: **43 passed in 9.97 s**.
- Complete Rockchip census: **159 passed, 9 skipped, 96 subtests passed in 111.57 s**.

The complete tinygrad run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops. The reference example itself requested large contiguous buffers and logged two 4 MiB CMA allocation failures; Rockchip tinygrad continued to use its page-backed non-contiguous allocation policy and generated none.

---

## 2026-08-08 — Infinite FP16 multiplication on DPU

RK3588 native EW MUL returns NaN for finite values multiplied by infinity. Rockchip now rewrites `x*+inf` to `x/+0`, and `x*-inf` to `(-x)/+0`. The former is one FDIV task; the latter is a two-task `MUL → FDIV` chain. Both execute entirely on DPU, inspect no input values on the host, use no reset or CMAC path, and preserve the expected infinity sign. Multiplication by NaN remains the correct native MUL operation.

- Separate positive-zero FDIV probe: bit-exact pass.
- Separate negative-infinity `MUL → FDIV` probe: bit-exact pass.
- Direct `+inf`, `-inf`, and NaN regression: **1 passed in 2.71 s**, exactly one ioctl per expression.
- Exact upstream `TestOps.test_mul_naninf`: **1 passed in 2.85 s**.
- Complete incremental source-order group: **44 passed in 9.66 s**.
- Complete Rockchip census: **161 passed, 8 skipped, 96 subtests passed in 110.71 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops.

---

## 2026-08-08 — Native ReLU, PReLU, and ABS activation group

Six exact upstream activation methods are now admitted: `test_relu`, `test_relu_exact`, `test_relu_maximum_exact`, `test_leaky_relu`, `test_abs`, and `test_abs_exact`. ReLU already lowers through the proven EW MAX/ReLU path. Leaky-ReLU recognizes `WHERE(x<0, slope*x, x)` and uses RK3588's one-task EW PReLU configuration. ABS recognizes tinygrad's signed-zero-aware sign graph and uses native EW `ALU_ALGO=5` in one task.

The PReLU and ABS register recipes were checked against `rockchip/post-518-reference`, the RK3588 TRM, and `~/npu/include/rknnops.h`. Only their named renderer configuration was ported; none of the archived reset, BS-PReLU, or stateful runtime machinery was imported.

- Individual methods: all pass in 2.65–3.10 s.
- Focused native ABS edge probe: `-1`, both signed zeros, `1`, both infinities, and both NaN signs pass; `-0` becomes `+0` exactly.
- Six-method activation group: **6 passed in 3.26 s**.
- Complete incremental source-order group: **50 passed in 10.12 s**.
- Complete Rockchip census: **167 passed, 8 skipped, 96 subtests passed in 111.38 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops.

---

## 2026-08-08 — Native ReLU6, MIN, and hard-activation group

Six more exact upstream methods are admitted: `test_relu6`, `test_clip`, `test_hardtanh`, `test_hardsigmoid`,
`test_hardsigmoid_extreme`, and `test_hardswish`. The register recipes were verified against
`rockchip/post-518-reference`, `~/npu/include/rknnops.h`, and the RK3588 examples/TRM. ReLU6 uses EW ReLUX with an
FP32 compare value of 6, while ordered clamp graphs use native EW ALU MIN together with the existing native MAX path.
Hardsigmoid and hardswish are algebraically folded onto those same primitives; no CMAC, CPU numeric evaluation, or
shared tinygrad core change is involved.

The extreme hardsigmoid graph is simplified by tinygrad to `relu(x/6+0.5)-relu(x/6-0.5)`. Evaluating that expression
directly loses precision for large FP16 inputs, so Rockchip recognizes the common affine base and emits
`min(relu(x/6+0.5), 1)` instead. The positive and negative 300–400 ranges then saturate exactly without changing the
allowed FP16 tolerance.

- All six methods pass separately in 2.70–3.14 s.
- Six-method persistent-process group: **6 passed in 3.54 s**.
- Complete incremental source-order group: **56 passed in 10.49 s**.
- Complete Rockchip census: **173 passed, 8 skipped, 96 subtests passed in 112.09 s**.
- Ruff and mypy: pass. Renderer/runtime executable size: 647/165 lines.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure,
or kernel oops.

---

## 2026-08-08 — Static FP16 movement and view group

Eleven exact upstream movement methods are now admitted: `test_transpose`, `test_permute`, `test_reshape`, `test_view`,
`test_flip`, `test_squeeze`, `test_unsqueeze`, `test_flatten`, `test_unflatten`, `test_detach`, and `test_expand`. This is
the view-only subset of the movement census proven on `rockchip/post-518-reference`. The `~/rk3588` examples and
`~/npu/include/rknnops.h` likewise treat tensor packing as address/stride preparation around the accelerator rather than
numeric execution.

No renderer/runtime change was required. Non-contiguous views use the existing affine or fallback gather plan, which
computes integer source indexes and copies raw `uint16` lanes into scratch. The value-preserving DPU EW stage remains the
only input-dependent numeric operation. No CMAC path, FP32 conversion, tolerance change, or shared tinygrad core edit was
added.

- Each method passed separately in a fresh pytest process: 2.58–3.11 s.
- Persistent-process movement group: **11 passed in 3.70 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**, through 131,072 elements.
- Complete Rockchip census: **184 passed, 8 skipped, 96 subtests passed in 111.49 s**.
- Ruff: pass. Renderer/runtime executable size remains 647/165 lines.

The vendor script's legacy contiguous 4 MiB allocation logged its two known CMA fallback attempts; the subsequent
tinygrad census produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left `CmaFree`
at 6144 KiB.

### Runtime cleanup and host-computation audit

The movement path was audited after admission. Runtime gathers view buffers only as raw `uint16` lanes; NumPy computes
integer source indexes and never interprets or transforms input-dependent FP16 values. Renderer vector evaluation is
restricted to parameter-free RANGE/CONST layout expressions. This is ABI/layout preparation, not a CPU numeric fallback.

The unused `reset_npu` hook was removed. It had no caller, issued unsupported action 13, swallowed reset failures, and did
not match the recovery ownership used by other tinygrad hardware runtimes. Blocking submission behavior is unchanged.
The movement group still passes **11/11 in 3.55 s** after cleanup. Ruff and mypy pass; `sz.py` reports renderer/runtime
sizes of **647/160 executable lines**, reducing the runtime by five lines while retaining comments and docstrings.

---

## 2026-08-08 — Static FP16 slicing and padded selection

Sixteen more movement methods are admitted: bounded one- and multidimensional slicing, scalar integer dimension
selection, inserted `None` axes, clipped endpoints, positive and negative strides, empty results, error cases, ellipsis,
double slicing, and pad→reshape/slice compositions. The explicit `test_slice_with_const_tensor` case remains skipped
because it requires an integer tensor input; it is not emulated on the CPU.

Partly padded slices exposed tinygrad's static `WHERE` selection tree around raw loads. The proven post-518 selection
gather was ported with support for one exact FP16 fill bit-pattern. It evaluates only parameter-free RANGE/CONST layout
predicates at compile time, records integer source offsets, and moves raw `uint16` lanes at runtime. No input-dependent
predicate or numeric value is evaluated on the host.

An initial implementation attempted selection recognition on every ADD node and made large convolution compilation
quadratic. Profiling showed the pytest process at 100% CPU with no active NPU ioctl or kernel error. Restricting the
recognizer to its actual WHERE/gated-load roots restored focused depthwise/dilated convolution methods to 2.79–3.19 s.

- Sixteen new methods passed separately in fresh pytest processes; the integer-tensor case skipped explicitly.
- Complete movement group: **27 passed, 1 skipped in 5.50 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **200 passed, 9 skipped, 96 subtests passed in 115.26 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **693/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Diagonal and rolling static indexing

The exact upstream `test_diag`, `test_diagonal`, and `test_roll` methods now join the movement census. They cover vector
diagonal construction, rectangular/batched/offset diagonal extraction, and positive, negative, zero, multi-axis, and
oversized cyclic shifts. All lower through the existing single-source static gather and raw FP16 identity path; no
renderer/runtime change or host numeric operation was required.

- Individual methods: **1 passed** each in 2.77–3.18 s.
- Complete movement group: **30 passed, 1 skipped in 6.18 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **203 passed, 9 skipped, 96 subtests passed in 115.29 s**.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **693/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — FP16 concatenation, stacking, and repetition

The multi-source partial-gather design from `rockchip/post-518-reference` is now ported to the current renderer image
format. `test_cat`, `test_multicat`, `test_stack`, `test_stack_slice`, `test_stack_max`, and `test_repeat` are admitted.
Several source buffers populate disjoint lanes of one scratch buffer, after which the ordinary DPU EW identity stage
writes the result. Runtime preparation computes static integer offsets and copies raw `uint16` lanes only; it never
decodes or evaluates input-dependent FP16 values. No CMAC path or shared tinygrad core change is involved.

Selection-tree and static-expression results are cached during lowering. This keeps multi-source recognition bounded
and avoids repeating the large-graph compile-time problem previously found while adding padded slices.

- Every method passed separately in a fresh pytest process: **2.78–6.01 s**.
- Complete concatenation group: **6 passed in 6.99 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **209 passed, 9 skipped, 96 subtests passed in 94.84 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **729/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Constant, reflect, replicate, and circular padding

Four exact upstream padding methods are now admitted: `test_pad`, `test_pad_reflect_mode`,
`test_pad_replicate_mode`, and `test_pad_circular_mode`. The implementation was checked against the proven
`rockchip/post-518-reference` milestone `04186a805`. Its required selection-gather lowering is already subsumed by the
current cached gather implementation, so this milestone adds coverage without adding renderer or runtime code.

All modes lower static padding predicates and source offsets on the host, copy only raw `uint16` lanes into scratch,
and use the DPU EW identity stage for output. No input-dependent FP16 arithmetic, CMAC path, tolerance change, or shared
tinygrad core change is involved.

- Individual methods: **1 passed** each in 2.87–6.31 s.
- Complete padding group: **4 passed in 9.71 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **213 passed, 9 skipped, 96 subtests passed in 44.62 s** under `-n12`.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **729/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Remaining FP16 repetition layouts

The exact upstream `test_repeat_interleave` and `test_simple_repeat` methods now complete the repetition subset. No
dedicated generic implementation exists in `rockchip/post-518-reference`, `~/npu`, or `~/rk3588`; both layouts lower
directly through the current single-source raw gather and DPU EW identity path. The host computes static integer source
offsets and copies `uint16` lanes without interpreting FP16 values.

- Individual methods: **1 passed** each in 2.74–2.89 s.
- Complete concatenation/repetition group: **8 passed in 7.20 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **215 passed, 9 skipped, 96 subtests passed in 43.32 s** under `-n12`.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **729/160 executable lines**.

The tinygrad census produced no RKNPU timeout, invalid IRQ, IOMMU fault, reset, or kernel oops. The vendor script again
logged its two known failed contiguous 4 MiB CMA attempts, then passed every probe; `CmaFree` returned to 6144 KiB.

---

## 2026-08-08 — FP16 scalar reductions and dynamically sized one-submit PC chains

The DPU EW scalar-reduction design from `rockchip/post-518-reference` milestone `326124baf` is ported for FP16 sum,
mean, product, minimum, and maximum. Reduction inputs are gathered as raw FP16 lanes into 64-byte-spaced scratch and
combined by a balanced tree of native DPU EW ADD, MUL, or MAX stages. Mean adds a final DPU MUL by its compile-time
scale, and minimum uses the proven negate/MAX lowering. ReLU-wrapped sums and precise long multiply/add trees are also
recognized.

There is no CPU numeric fallback: the renderer evaluates only static integer layout expressions, while runtime copies
raw `uint16` lanes using integer indices. It never decodes input FP16 values or computes a reduction on the host. No
CMAC path, shared tinygrad core change, or tolerance relaxation was added; tests retain the `test_gemm_fp16` limit of
`atol=rtol=grad_atol=grad_rtol=5e-3`.

Fixed task-count caps are removed. Each realized program now computes its command and descriptor arena sizes from the
actual emitted register-command qwords and 40-byte task descriptors, adds one mapped page of command-prefetch guard,
and submits the complete PC chain with one blocking ioctl. The 16,384-element `test_sum_full` produces 16,383 tasks,
a 2,887,504-byte command allocation (including guard), and a 655,320-byte descriptor allocation, and passes with
exactly one submit. Arena growth allocates the replacement before releasing the old buffer.

- Focused reduction group after cleanup: **21 passed, 1 skipped in 10.67 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Complete Rockchip census, sequential single-process hardware ownership: **229 passed, 10 skipped, 96 subtests
  passed in 102.98 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **807/164 executable lines**.

FP32, boolean, and integer reductions remain explicitly skipped because the NPU input path is FP16-only. The complete
run and focused post-cleanup run produced no RKNPU timeout, invalid IRQ, IOMMU fault, reset, or kernel oops.

---

## 2026-08-08 — Adaptive-pooling-equivalent FP16 windows

Tinygrad currently exposes no adaptive average/max pooling API, so the Rockchip census now covers the exact equivalent
divisible-window cases from `rockchip/post-518-reference`: 4×4 to 2×2 and 4×4 to global 1×1. PyTorch adaptive pooling
supplies the golden while tinygrad expresses the same operation with fixed kernel and stride pooling.

- Adaptive-equivalent census: **2 passed in 3.04 s**, sequentially on one NPU process.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.

Both cases reuse the existing raw FP16 gather plus DPU EW reduction path. No backend or shared core change, host numeric
operation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — FP16 polynomial and simple loss expressions

The Rockchip census now covers square, cubic, a Horner-form quadratic, mean-squared error, and mean absolute error from
the proven `rockchip/post-518-reference` group. They compose the existing FP16 ADD, MUL, ABS, and reduction paths; BCE
and NLL remain outside this group because they require transcendental or indexed semantics. FP32 is used only by the
PyTorch golden for the loss reductions and is then cast to the FP16 device contract.

- Polynomial/simple-loss census: **4 passed in 2.98 s**, sequentially on one NPU process.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.
- Rockchip execution uses only DPU EW; no backend/core change, host tensor arithmetic, CMAC, CNA, or PPU execution was
  added.

---

## 2026-08-08 — FP16 scatter with compile-time-static indices

The Rockchip census now covers functional scatter replacement on dimensions 0 and 1, scalar-source scatter, and
scatter-reduce sum, product, mean, maximum, and minimum from `rockchip/post-518-reference`. Indices are compile-time
`arange` or broadcast constants: the renderer turns them into raw gather layout, while selected FP16 values and every
reduction execute on DPU EW. External integer index buffers remain explicitly excluded because Rockchip accepts FP16
inputs only.

- Static-index scatter census: **8 passed, 1 skipped in 3.13 s**, sequentially on one NPU process.
- Covered reductions: sum, product, mean, maximum, and minimum.
- Complete Rockchip census: **243 passed, 11 skipped, 96 subtests passed in 101.99 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s after the full census.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.

No backend/core change, host tensor arithmetic, CMAC, CNA, or PPU execution was added. The full run produced no RKNPU
timeout, invalid IRQ, IOMMU fault, reset, or kernel oops.

---

## 2026-08-08 — Terminal FP32 output for FP16 scalar reduction

The exact upstream `test_sum_dtype_arg` is now admitted using the proven `rockchip/post-518-reference` DPU recipe. The
input and complete balanced reduction tree remain FP16. Only the terminal scalar ADD sets DPU `OUT_PRECISION=5`, while
`IN_PRECISION=2` and `PROC_PRECISION=2` remain FP16. No FP32 value is fed into another NPU task.

The terminal output stage is isolated from the dynamically sized FP16 PC chain, giving two blocking submits for the
135-element case. A successful terminal task is followed by one supported action reset so its scalar FP32 BS/WDMA state
cannot leak into the next FP16 program. An explicit same-process `test_sum_dtype_arg → test_tiny_add` sequence passed.

- Focused FP32-output test: **1 passed in 2.93 s**; expected submit count is exactly 2.
- Post-cleanup FP32-output then FP16 transition: **2 passed in 3.05 s**.
- Complete Rockchip census: **244 passed, 11 skipped, 96 subtests passed in 103.21 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.45 s after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **840/173 executable lines**.

The renderer only tags the terminal UOp and emits register commands; runtime performs no host numeric conversion or
tensor arithmetic. There is no CMAC, CNA, PPU, shared core change, or tolerance relaxation. Kernel logs contain no new
RKNPU timeout, invalid IRQ, IOMMU fault, or oops.

---

## 2026-08-08 — FP16 broadcast arithmetic census

The exact upstream `test_broadcast_simple` now joins the Rockchip census. Full and partial broadcast shapes from
`test_broadcast_full` and `test_broadcast_partial` are additionally covered for every currently native FP16 arithmetic
operation: ADD, SUB, MUL, and FDIV. POW is intentionally kept separate rather than weakening those upstream methods or
silently treating an unsupported primitive as broadcast support.

Broadcast layout is handled by the existing static raw-lane gather, and arithmetic remains DPU EW. `~/npu` and
`~/rk3588` contain broadcast register/layout examples but no additional backend lowering was required here.

- Broadcast group: **3 passed, 24 subtests passed in 3.43 s**, sequentially.
- Complete Rockchip census: **247 passed, 11 skipped, 120 subtests passed in 103.51 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.55 s.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`; backend size remains renderer/runtime **840/173 executable lines**.

This milestone changes coverage only. It adds no host tensor arithmetic, CMAC, CNA, PPU, shared core change, or
tolerance relaxation.

---

## 2026-08-08 — Exact dot, batched dot, and matvec census on DPU EW

Six exact upstream methods are now admitted: `test_dot_1d`, `test_dot`, `test_broadcastdot`, `test_multidot`,
`test_matvec`, and `test_matvecmat`. K-loop contractions arrive after FP16 rewriting as local FP32 accumulator loops.
The renderer now recognizes the narrowly validated ADD-of-FP16-products form, statically substitutes only the reduction
range, and feeds the resulting product tree back through the existing generic DPU EW lowering.

This keeps execution scalable: each K term is a vector operation over all outputs, so task count depends on K rather
than `M×N×K`. `ROCKCHIP_EW_REDUCE=twoproduct` reuses the established compensated FP16 product/sum graph; a plain
balanced FP16 tree was rejected because one K65 lane missed the permitted tolerance by 0.007324. The older
`rockchip-2608` reference used scalar CMAC for part of this scope, but CMAC is intentionally not ported here.

- Individual methods: **1 passed each in 3.21–4.71 s**.
- Combined dot/matvec group: **6 passed in 9.02 s**, sequentially in one process.
- Complete Rockchip census: **253 passed, 11 skipped, 120 subtests passed in 113.20 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.46 s.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **878/173 executable lines**.

No input value is evaluated on the host: compile time only substitutes static integer K indices, runtime performs the
existing raw `uint16` gathers, and every MUL/ADD/compensation stage executes on DPU EW. There is no CMAC, CNA, PPU,
shared core change, or tolerance relaxation. Kernel logs contain no new timeout, invalid IRQ, IOMMU fault, or oops.

---

## 2026-08-08 — FP16 einsum contraction and validation census

Five exact upstream methods now join the Rockchip census: `test_einsum`, `test_einsum_trace`,
`test_einsum_shape_check`, and the two arity checks. The functional cases cover scalar identity, transposes, whole/axis
sums, diagonal and batched trace, vector/matrix/batched products, permuted results and inputs, large tensor
contractions, and a three-input bilinear expression.

All functional forms compose the existing static raw-lane gathers, scalar reduction lowering, and compensated DPU EW
dot loop from the prior milestone. No new backend implementation was required. The separate slow ellipsis stress method
contains a 13,824-term reduction and remains a dedicated profiling target rather than being silently included here.

- `test_einsum_trace`: **1 passed in 2.81 s**.
- Main `test_einsum`: **1 passed in 6.40 s**.
- Complete einsum group: **5 passed in 6.37 s**, sequentially.
- Complete Rockchip census: **258 passed, 11 skipped, 120 subtests passed in 115.89 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.54 s.
- Backend size remains renderer/runtime **878/173 executable lines**.

This milestone changes coverage only and retains the exact FP16 tolerance ceiling. It adds no CPU tensor arithmetic,
CMAC, CNA, PPU, or shared core change.

---

## 2026-08-08 — FP16 cumulative sums with bounded static selection analysis

Four exact upstream methods now join the Rockchip census: `test_small_cumsum`, `test_simple_cumsum`, `test_cumsum`,
and `test_cumsum_zero_axis`. Together they cover scalar and empty inputs, 1D lengths through 1,022, and cumulative
sums across 2D/3D axes. The older `rockchip/post-518-reference` implementation uses specialized CMAC reductions,
while `~/npu` explicitly lists CumSum as unsupported and `~/rk3588` supplies only the upstream test definitions; none
was ported because this branch remains DPU-EW-only.

The existing static selection gather already described the cumulative layouts, but interpreted the same shared UOp
DAG once per output coordinate. The 512-element case remained host-bound for more than 90 seconds. Selection analysis
now evaluates every static coordinate together with NumPy vectors and memoizes each UOp's slot, raw offset, and fill
metadata. This changes compiler metadata construction only: runtime still copies raw `uint16` lanes and every prefix
ADD executes as a DPU EW task.

- `test_simple_cumsum`: **1 passed in 11.80 s**, down from more than 90 seconds of host compilation.
- Complete cumulative-sum group: **4 passed in 13.95 s**, sequentially in one process.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Complete Rockchip census: **262 passed, 11 skipped, 120 subtests passed in 120.32 s**, sequentially.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **896/173 executable lines**.

The CPU-cheat audit found no input decoding or host numeric result computation. NumPy is confined to static integer
index, source-slot, gate, and FP16-fill-bit planning at compile time. There is no CMAC, CNA, PPU, shared tinygrad core
change, or tolerance relaxation.

---

## 2026-08-08 — Ordered FP16 cumulative products on DPU EW

Four exact upstream methods now join the Rockchip census: `test_small_cumprod`, `test_simple_cumprod`, `test_cumprod`,
and `test_cumprod_zero_axis`. They cover scalar and empty inputs, 1D lengths through 1,022, and cumulative products
across 2D/3D axes. `~/npu` has no native CumProd path and `~/rk3588` only supplies the upstream tests. Older Rockchip
branches used specialized CMAC or partially validated product atoms; neither path was ported because this branch remains
DPU-EW-only.

Static selection planning now preserves independent FP16 padding values per output lane and can compose a compile-time
prefill with partial raw gathers. The cumulative product is recognized only when its leaves form the verified prefix
sequence, then rebuilt in source order as native FP16 MUL stages. Scratch destinations use ping-pong storage so a stage
never aliases a still-live input, and generic dot-product compensation is not applied to this ordered scan.

A 40-term prefix exposed eight NaNs when all dependent stages were submitted as one PC chain, while every finite result
remained within tolerance. Isolated and bounded-chain probes showed the arithmetic was correct but the terminal dependent
task lacked a proven inter-task visibility boundary. The image format now carries an explicit typed `submit_barrier`
flag, and runtime closes the current blocking PC-chain immediately before the terminal output stage. The flag replaces
an initially considered software bit tag and cannot leak into a hardware register.

- Focused cumulative-product group after cleanup: **4 passed in 12.45 s**, sequentially in one process.
- Complete Rockchip census: **266 passed, 11 skipped, 120 subtests passed in 137.45 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.53 s after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **926/176 executable lines**.

The CPU-cheat audit found no input decoding or host numeric product. NumPy evaluates only static UOp coordinates,
source slots, masks, and fill-bit placement; runtime copies raw `uint16` lanes, and every multiplication executes on DPU
EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 cumulative extrema values

The Rockchip census now covers the value outputs for cumulative MAX and MIN across every shape and axis used by the
four upstream methods for each operation: scalar and empty tensors, 1D lengths 10/512/1,022, 2D axes 0/1, and 3D axes
2/-1. The corresponding upstream methods also request `int32` axis indices, which remain an explicit separate output-
format group rather than being hidden or computed on the host.

`~/npu` has no native cumulative-extrema implementation and `~/rk3588` supplies the upstream definitions. Historical
Rockchip milestones `d407bb338` and `eb7c58c98` prove DPU compare/select implementations for both values and indices,
but their large specialized runtime is not ported. The current compact generic MAX path already handled all value cases
except the padded 1,022 tail. That tail wraps a MAX tree in a static mask whose individual load gates contain the inverse
condition. `_fold_masked_max` now recognizes that conjunction and safely moves the MAX identity fallback into each load,
leaving ordinary native DPU EW MAX stages.

- Cumulative-extrema value group: **8 passed, 24 subtests passed in 23.86 s**, sequentially.
- Complete Rockchip census: **274 passed, 11 skipped, 144 subtests passed in 148.30 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.37 s.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **927/176 executable lines**.

The CPU-cheat audit found no runtime input decoding, comparison, or extrema calculation. The new rule is a renderer-only
identity rewrite guarded by structural condition checks; all MAX and negate/MAX work executes on DPU EW. There is no
CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Small cumulative-extrema INT32 axis indices on DPU EW

The exact upstream `test_small_cummax` and `test_small_cummin` index outputs now join the Rockchip census. This first
index milestone deliberately covers the official length-10 cases; multidimensional axes and larger tiled INT32 outputs
remain the next group and are not claimed here.

Historical Rockchip milestones `d407bb338` and `eb7c58c98`, plus `rockchip/post-518-reference`, provide the verified
RK3588 compare/selection register sequence. `~/npu` has no cumulative-extrema implementation, while `~/rk3588`
provides the upstream tests and experimental compare examples. The current lowering gathers FP16 candidates and static
axis coordinates into 64-byte-aligned arenas, computes candidate-minus-extremum magnitude and equality masks on DPU EW,
multiplies masks by coordinates, and reduces the selected coordinates with DPU MAX. CumMin recognizes the negated
candidate form before applying the same selection graph.

The terminal coordinate-minus-one remains FP16 DPU arithmetic. Native DPU output conversion then writes INT32 values
through four-lane, 64-byte-aligned tiles. Runtime only packs and unpacks those raw lane bytes; it does not compare input
values, choose an index, or numerically convert FP16 to INT32. Stateful compare submits retain the proven standalone PC
framing, while ordinary PC chains keep their existing register template. Common command/task buffer growth and blocking
submit bookkeeping are shared between both paths.

- Focused index group: **2 passed, 2 subtests passed in 7.65 s**, sequentially.
- Complete Rockchip census after cleanup: **276 passed, 11 skipped, 146 subtests passed in 153.63 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,058/239 executable lines**.

The CPU-cheat audit found only raw `uint16` gather/layout operations and `memmove` around the aligned INT32 tiles. Every
comparison, mask, selection, extrema reduction, subtraction, and numeric conversion executes on DPU EW. There is no
CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Multidimensional cumulative-extrema axis indices

The remaining ordinary-shape cases from upstream `test_cummax` and `test_cummin` now join the Rockchip census: scalar,
length 5, both axes of `(5,6)`, and axis `2`/`-1` of `(5,6,7)`. Both zero-axis methods are also covered. Together with
the preceding length-10 milestone, the unchanged upstream `test_cummax`, `test_cummax_zero_axis`, `test_cummin`, and
`test_cummin_zero_axis` methods now pass directly.

Historical commits `d407bb338` and `eb7c58c98` establish that cumulative indices use packed reduction-axis coordinates,
not flattened source addresses: the current coordinate is `dst % window`, later candidates are invalid, and the latest
equal candidate wins. The new recognizer accepts that rule only when every candidate map is distinct and monotonic and
the current-candidate addresses form an exact permutation of the input. `~/npu` still has no cumulative-extrema path;
`~/rk3588` supplies the upstream definitions and the compare/framing examples used by the existing DPU stages.

Multidimensional Tinygrad graphs finish with a pure INT32 view kernel. RKIMAGE v22 generalizes the existing raw-lane
gather metadata to 2- or 4-byte lanes and scratch or argument destinations, allowing that final transpose without
interpreting index values. Numeric selection and FP16-to-INT32 conversion remain on DPU.

The first 210-element probe timed out at driver task counter 17 while one PC chain held 53 stateful four-lane INT32
conversion stages. The backend now bounds only this terminal stage by its encoded command bytes: a batch may occupy at
most one system page. With the current 288-byte body this computes 14 tasks per submit; there is no hardcoded task-count
cap, and command/task BO allocation remains dynamically sized. The same case then passed exactly in **2.26 s** with
**28 total submits**, including its seven reset-separated comparisons. The vendor health check passed 60/60 immediately
after the exploratory timeout and again after the full census.

- Expanded cumulative-index class: **6 passed, 20 subtests passed in 24.66 s**, sequentially.
- Four unchanged upstream CumMax/CumMin methods: **4 passed in 20.33 s**, sequentially.
- Complete Rockchip census: **280 passed, 11 skipped, 164 subtests passed in 178.53 s**, sequentially.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,075/249 executable lines**.

The CPU-cheat audit found only address-map construction and raw `uint16`/`uint32` representation movement on the host.
No runtime value is compared, selected, reduced, or numerically converted by the CPU. There is no CMAC, CNA, PPU,
shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Long cumulative-extrema indices with matrix DPU selection

The remaining length-512 and padded length-1,022 INT32 index outputs from upstream `test_simple_cummax` and
`test_simple_cummin` now run on Rockchip. The permanent Rockchip coverage splits the four realizations into separate
methods so each hardware case stays below 30 seconds; the unchanged upstream methods pass as well.

The earlier per-candidate selection graph repeated comparison setup hundreds of times. Candidate values and axis
coordinates are now gathered into aligned matrices, then DPU EW performs bulk subtraction, absolute value, equality,
mask multiplication, and a pairwise MAX tree. CumMin first negates its candidate matrix on DPU. Length 1,022 uses the
exact bounded-loop form emitted by Tinygrad: the recognizer verifies its candidate permutation, output-before-reduction
prefix gate, INT32 MAX dependence, and two partial-extremum loads before building the same matrix selection image.
Candidate rows use compact affine gather metadata rather than storing a multi-megabyte repeated offset table.

An exploratory length-512 run timed out at task counter 9 when a dependent equality-to-MUL transition shared one PC
chain. The vendor elementwise health check immediately passed 60/60. Typed submit barriers now separate the dependent
bulk phases; all subsequent focused and full-suite runs completed without a timeout. Scratch matrix padding is cleared
as raw storage before gather placement so unused aligned lanes remain neutral.

- CumMax 512: **1 passed in 16.61 s**; CumMax 1,022: **1 passed in 18.40 s**.
- CumMin 512: **1 passed in 19.63 s**; CumMin 1,022: **1 passed in 17.03 s**.
- Complete cumulative-index class: **10 passed, 24 subtests passed in 62.40 s**, sequentially.
- Unchanged upstream `test_simple_cummax`: **1 passed in 24.30 s**; `test_simple_cummin`: **1 passed in 37.19 s**.
  The latter combines four realizations; its slowest individual realization was 13.81 s.
- Complete Rockchip census: **284 passed, 11 skipped, 168 subtests passed in 191.13 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,139/253 executable lines**.

The CPU-cheat audit found only static address-map planning, scratch padding initialization, and raw `uint16`/`uint32`
lane movement on the host. Every comparison, mask, candidate negation, selection, reduction, subtraction, and numeric
FP16-to-INT32 conversion executes on DPU EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance
relaxation.

---

## 2026-08-08 — FP16 ArgMax/ArgMin first-tie selection on DPU EW

The first general selected-index milestone covers the exact first-occurrence rule for FP16 ArgMax and ArgMin, including
equal pairs and repeated extrema after a distinct leading value. The unchanged upstream methods begin with `int32`
literals and later add boolean/int32 conversion regressions, so those input-format cases remain a separate group rather
than being presented as native FP16 coverage.

Historical commit `7bea014d35` proves the RK3588 DPU compare/select design, but its large task runtime is not ported.
`~/npu` documents RKNN ArgMax/ArgMin as CPU operators, and `~/rk3588` contains the upstream tests plus RKNN runtime
operator strings rather than a lower-level register implementation. The current compact renderer recognizes only the
fused graph containing one FP16 MAX tree, one equality/inversion/cast per candidate, the exact descending coordinate
weights, one INT32 MAX tree, and the final `window-selected` transform. It also verifies that candidate gathers form an
exact permutation of the source.

Candidate and extreme-value matrices reuse the cumulative-index image. Descending coordinates make DPU MAX retain the
earliest equal candidate; a final DPU subtraction restores its zero-based coordinate. ArgMin negates both the candidate
matrix and the independent matrix used to construct `MAX(-x)`, then follows the same equality path.

- Focused FP16 first-tie group: **2 passed, 4 subtests passed in 4.18 s**, sequentially.
- Complete Rockchip census: **286 passed, 11 skipped, 172 subtests passed in 192.93 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,210/253 executable lines**.

The runtime is unchanged. Host work is limited to static source-address and coordinate-bit image construction plus the
existing raw lane movement. Candidate negation, extrema reduction, equality, mask weighting, first-tie selection,
subtraction, and FP16-to-INT32 output conversion all execute on DPU EW. There is no CPU numeric fallback, CMAC, CNA,
PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 axis ArgMax/ArgMin with materialized extrema

FP16 ArgMax and ArgMin now cover both axes of `(10,20)` plus axis-1 `keepdim=True`, matching the three axis forms in
the upstream methods. Tinygrad schedules these as two kernels: ordinary DPU EW first materializes `MAX(x)` or
`MAX(-x)`, then the new selected-index image compares every candidate against that saved extreme.

The fused and split recognizers share one exact first-tie validator. The split form additionally requires exactly two
FP16 source buffers, an extreme buffer whose static view is a permutation of the output, a candidate buffer whose views
jointly permute the full input, one sign convention across all candidates, and the same descending-coordinate INT32 MAX
graph. It therefore reuses the first-tie matrix emitter without accepting cumulative latest-tie or unrelated comparison
graphs. The saved extreme remains device data and is never copied to or interpreted by the CPU.

- Expanded FP16 ArgMax/ArgMin group: **4 passed, 10 subtests passed in 6.80 s**, sequentially.
- Complete Rockchip census: **288 passed, 11 skipped, 178 subtests passed in 194.43 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,256/253 executable lines**.

The runtime remains unchanged, and its no-CPU-cheat audit contains no host extrema, equality, or selected-index
arithmetic. The renderer constructs only static address and coordinate metadata; all numeric work and conversion remain
DPU EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation. The padded global 200-element
bounded-loop graph and non-FP16 input conversions remain explicit next groups.

---

## 2026-08-08 — Global FP16 ArgMax/ArgMin bounded-loop selection

Global FP16 ArgMax and ArgMin over the upstream `(10,20)` shape now join the Rockchip census. Additional deterministic
cases place equal extrema at flat indices 37 and 99 and verify that the loop form also returns index 37.

For 200 candidates Tinygrad replaces the unrolled graph with two ordered register loops. The first initializes a local
FP16 accumulator to negative infinity and computes `MAX(x)` or `MAX(-x)`. The second initializes an INT32 accumulator
to `INT_MIN`, compares each candidate with the completed extreme, weights equality by `window-coordinate`, and reduces
with INT32 MAX. The final store computes `window-selected`.

The new matcher accepts only that exact two-loop structure: one global FP16 input of the loop extent, matching direct
candidate permutations in both loops, one sign convention, the expected local accumulator initializers and updates,
descending coordinates, and the final inverse transform. It then emits the already verified matrix DPU image. No loop
accumulator is read by the host.

- Expanded FP16 ArgMax/ArgMin class: **6 passed, 10 subtests passed in 11.22 s**, sequentially.
- Complete Rockchip census: **290 passed, 11 skipped, 178 subtests passed in 200.45 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,320/253 executable lines**.

The CPU-cheat audit found only compile-time UOp structure, source-address, and coordinate evaluation. Runtime remains
raw lane movement plus DPU submission and contains no host extrema or selected-index arithmetic. Every negation, MAX,
comparison, mask, coordinate selection, subtraction, and FP16-to-INT32 conversion executes on DPU EW. There is no CMAC,
CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Stable FP16 sort values on DPU EW

The FP16 value half of upstream `test_sort` now runs for every nontrivial `(8,8,6)` axis and direction, along with its
empty/singleton and repeated-value cases. A dedicated infinity regression verifies both directions over `[-inf, 2]`.
Stable INT32 indices remain the next milestone: the unchanged upstream method passes its value realizations, then rejects
the first `(8,8,6)` index-count graph rather than falling back to the CPU.

Historical milestone `43f113d90` supplies the native bitonic graph classification, but its MIN reconstruction uses
negation and an arithmetic mask. That formulation produces `0*inf = NaN` for padded sort wires. The current lowering
instead gathers each statically paired wire, computes complete contiguous native DPU MAX and MIN vectors, and merges them
with the compile-time bitonic direction map. `~/npu/include/old/rknn_ops.md` explicitly lists `aten::sort` as unsupported;
`~/rk3588` contains the upstream sort/argsort tests but no lower-level sort implementation.

RKIMAGE v23 gives raw gathers explicit source and destination buffer kinds and separates pre-DPU from post-DPU gathers.
The runtime invalidates cacheable DPU results, then copies only their `uint16` lane representations according to the
compiled direction map. It never compares, ranks, sorts, or numerically transforms tensor data. Full-vector DPU writes
also avoid the 64-byte overwrite that made the exploratory short-run output path corrupt adjacent lanes.

- All six `(8,8,6)` axis/direction value sorts are bit-exact; each uses **9 ioctl submits**.
- Focused value-sort group: **4 passed, 14 subtests passed in 5.03 s**, sequentially.
- Complete Rockchip census: **294 passed, 11 skipped, 192 subtests passed in 213.92 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,383/264 executable lines**.

The CPU-cheat audit found only compile-time condition/address evaluation and raw `uint16` lane movement. MAX and MIN are
native DPU EW operations, and the host never inspects FP16 values. There is no CMAC, CNA, PPU, shared tinygrad core
change, FP32 input conversion, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — Stable FP16 sort and argsort indices on DPU EW

Exact stable INT32 indices now accompany the FP16 value sort for all six `(8,8,6)` axis/direction combinations and the
two repeated-value directions. The unchanged upstream `test_argsort` passes. Upstream `test_sort` passes every generated
FP16 value and index realization, then reaches its Python-integer repeated-value fixture; that fixture creates an external
INT32 input and remains outside the RK3588 FP16-input contract. The equivalent repeated FP16 fixture is permanent and
passes in both directions.

Historical commit `43f113d90` identifies occurrence counting and final value/count matching as the two stable-index graph
families. The current Tinygrad scheduler unrolls both reductions, so the new matchers validate the present ADD trees,
prefix gates, value/count equality pairs, source maps, and coordinate weights directly. `~/npu` still marks sort
unsupported and `~/rk3588` has tests but no lower-level implementation.

For each occurrence-count kernel, candidate/current FP16 matrices are subtracted, passed through native ABS and the
verified positive-mask comparison, gated by compile-time prefix masks, and reduced with DPU ADD. Final selection compares
both the original/sorted values and their stable occurrence counts, multiplies the two DPU equality masks by original
coordinates, and reduces those coordinates on DPU.

The count tensors are device-produced INT32, not external user inputs. RKIMAGE v24 adds a typed internal INT32-input
conversion stage: runtime packs raw four-lane atoms, DPU converts them to FP16, and runtime unpacks the raw FP16 lanes.
The inverse FP16-to-INT32 path already used by cumulative/arg-extrema indices shares the same conversion helper. Host code
does not interpret either representation numerically.

- Six `(8,8,6)` stable-index cases: **11.10–14.53 s each**, **121 submits** for axis `-1` and **147 submits** for axes
  `0`/`1`; all outputs exact.
- Repeated FP16 stable indices: **4.24 s ascending / 3.77 s descending**, **64 submits** each; outputs exact.
- Focused index group after cleanup: **8 passed, 6 subtests passed in 90.30 s**, sequentially.
- Unchanged upstream `test_argsort`: **1 passed in 17.10 s**, sequentially.
- Complete Rockchip census after cleanup: **302 passed, 11 skipped, 198 subtests passed in 287.74 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,593/269 executable lines**.

The CPU-cheat audit found only compile-time graph/address/weight validation and raw `uint16`/`uint32` atom movement.
All value equality, occurrence-count equality, mask multiplication, count/coordinate reduction, and numeric conversion
execute on DPU EW. There is no host sort or argsort, CMAC, CNA, PPU, shared tinygrad core change, external INT32-input
support, or tolerance relaxation.

---

## 2026-08-08 — FP16 TopK composition

FP16 TopK values and stable INT32 indices now join the Rockchip census: one-dimensional largest-three, padded five-lane
axis-0 largest-four, padded five-lane axis-1 smallest-four, and repeated-value largest/smallest cases. TopK itself needs
no new numeric primitive; it composes the two native sort milestones with static shrinking.

Historical milestone `5f48c8311` confirms that non-power-of-two TopK must preserve `-inf`/`+inf` padding without an
arithmetic `0*inf` wire blend. The current native MIN/MAX plus raw post-gather sort already provides that behavior.
`~/npu/include/old/rknn_ops.md` lists both TopK and `aten::topk` as unsupported, while `~/rk3588` contains only the
upstream tests/runtime strings rather than a register implementation.

One scheduler variant materializes stable indices at the full sorted shape, then emits a separate INT32 shrink. The raw
INT32 layout recognizer now accepts an injective in-bounds subset as well as a same-size permutation. Runtime copies only
the selected `uint32` representations; it does not inspect or compute indices.

- Generated FP16 value realizations: **0.30–0.43 s**, **9 submits** each.
- Generated FP16 index realizations: **2.24–3.26 s**, **41–50 submits** each; outputs exact.
- Repeated FP16 TopK: **3.57 s largest / 3.27 s smallest**, **55 submits** each; exact indices `[0,1,3]` and `[2,4,6]`.
- Focused TopK group: **4 passed, 2 subtests passed in 20.63 s**, sequentially.
- Complete Rockchip census: **306 passed, 11 skipped, 200 subtests passed in 305.76 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,593/269 executable lines**.

The unchanged upstream `test_topk` passes all generated FP16 cases and then reaches its repeated Python-integer fixture,
which is external INT32 input and remains outside the NPU contract. Its equivalent FP16 repeated fixture is covered
above. The CPU-cheat audit found only static address-map construction and raw lane movement; all sorting, stable counting,
selection, and numeric conversion remain DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 elementwise maximum and minimum

The FP16 portions of upstream `test_maximum` and `test_minimum` now join the Rockchip census: two `(45,65)` inputs,
scalars, scalar broadcast, vectors, and infinity/NaN/signed-zero inputs. External integer, boolean, and mixed-dtype
inputs remain outside the RK3588 FP16-input contract.

`~/rk3588/examples/elementwise.py` and `~/rk3588/test/test_maximum.py` provide the proven binary DPU MAX reference.
`~/npu/include/rknnops.h` names binary MAX/MIN algorithms, but the native EW ALU-MIN configuration returns NaN for
infinite operands. Tinygrad canonicalizes minimum as `-max(-x,-y)`; the Rockchip rewrite recognizes that graph and emits
`0-lhs`, `0-rhs`, DPU MAX, then `0-max`. SUB is required because RK3588 EW MUL with infinity returns NaN. All four
numeric stages execute on DPU EW.

The native-min tag is also distinguished by the specialized stable-sort matcher, preserving its existing direct
MAX/MIN compare/swap image. Focused extrema, sort-value, sort-index, and TopK regression groups all pass.

- Focused extrema group: **2 passed in 3.58 s**, sequentially.
- Extrema plus sort-value regression: **6 passed, 14 subtests passed in 5.83 s**, sequentially.
- Sort-index plus TopK regression: **12 passed, 8 subtests passed in 107.69 s**, sequentially.
- Complete Rockchip census: **308 passed, 11 skipped, 200 subtests passed in 305.86 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,617/269 executable lines**.

The CPU-cheat audit found no runtime or host numeric implementation in this milestone: only compile-time UOp recognition
and DPU command construction changed. The host does not inspect, compare, negate, or select tensor values. There is no
CMAC, CNA, PPU, tinygrad core change, external non-FP16 input support, or tolerance relaxation beyond the existing
FP16 contract.

---

## 2026-08-08 — FP16 sign and softsign

Upstream `test_sign`, `test_sign_exact`, `test_softsign`, and `test_softsign_exact` now join the Rockchip census. A
permanent nonfinite sign regression also verifies `-inf`, `+inf`, both signed zeros, and NaN exactly.

Historical milestone `e5fdcad64` supplies the DPU comparison-mask construction for sign. The current renderer accepts
only Tinygrad's exact `WHERE(x!=0, WHERE(x<0, -1, 1), 0)` graph, computes positive and negative masks on DPU, and
subtracts them on DPU. The four stages use four submits and return `-1`, `0`, or `1` without host classification.

Softsign already composes the current native ABS, ADD, and FDIV stages as `x/(1+abs(x))`; historical milestone
`506ffb537` and `~/npu/ops_ref/main.c` provide matching references. No extra special-case lowering was needed.
Historical exact-copysign milestone `751a108e2` extracts sign bits with NumPy in the runtime, so it was deliberately not
ported under the no-CPU-cheat rule; copysign remains a separate future problem.

- Focused sign/softsign group: **5 passed in 4.83 s**, sequentially.
- Complete Rockchip census: **313 passed, 11 skipped, 200 subtests passed in 308.65 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,644/269 executable lines**.

The CPU-cheat audit found only static UOp validation and DPU command construction. Runtime code is unchanged and never
reads or classifies tensor values for sign or softsign. Every negation, comparison, mask, subtraction, absolute value,
addition, and division executes on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 WHERE and masked fill

Nine FP16 selection cases now join the Rockchip census: tensor/scalar/broadcast arms, CMPNE, boolean AND/OR composition,
nested WHERE, finite masked fill, unchanged upstream `test_masked_fill`, and unchanged upstream `test_inf_where`.
The integer-output and external-boolean-input portions of upstream `test_where` remain outside the FP16-input contract.

Reference milestone `fe0b2e114` supplies the proven RK3588 positive-mask construction. The current port keeps the compact
renderer/image architecture: CMPLT and CMPNE become standalone DPU positive-mask stages, AND/OR compose exact FP16 0/1
masks with DPU MUL/MAX, and finite arms are selected with tagged mask MUL and ADD. No old runtime graph execution or
per-operation allocation machinery was imported.

Nonfinite arms avoid `0*inf`. Threshold forms such as `(x<0).where(x, 1)` use native DPU MIN/MAX composition, while
general infinity masked fills use a finite-numerator FDIV correction that is zero on the finite arm and signed infinity
on the selected arm. Stateful FDIV now uses the same RDMA mode and output scale as the already-proven ordinary FDIV.

Dedicated ABS and sign rewrite prepasses preserve their native algorithms before general WHERE rewriting. This prevents
the selector from turning `abs(inf)` into `inf*sign(inf)` and preserves exact NaN sign output behavior.

- Focused WHERE/masked-fill group: **9 passed in 6.28 s**, sequentially.
- Sign, exact-ABS, and WHERE precedence regression: **15 passed in 8.51 s**, sequentially.
- Complete Rockchip census: **322 passed, 11 skipped, 200 subtests passed in 313.77 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines**.

The CPU-cheat audit found only compile-time UOp/constant inspection and DPU command construction. Runtime is unchanged;
it never reads predicates or selects tensor values. Every comparison, mask composition, selection multiply/add,
threshold MIN/MAX, and infinity correction executes on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — finite FP16 copysign composition

The unchanged upstream `test_copysign` now joins the Rockchip census. Its finite FP16 cases need no new backend
primitive: Tinygrad composes the existing native ABS, DPU sign-mask construction, and DPU MUL path.

Historical milestone `751a108e2` was inspected but deliberately not ported: it reads tensor sign bits with NumPy in the
runtime, which violates the no-CPU-cheat rule. Exact signed-zero and nonfinite copysign remain outside this milestone.
The current pure-DPU composition loses the negative sign of a zero magnitude, and RK3588 EW MUL returns NaN when one
operand is infinity, so this milestone makes only the finite tolerance-based claim exercised by upstream
`test_copysign`.

- Focused sign/softsign/copysign group: **6 passed in 6.46 s**, sequentially.
- Complete Rockchip census: **323 passed, 11 skipped, 200 subtests passed in 314.47 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines**.

The CPU-cheat audit found no backend code change for copysign and no host value inspection. Absolute value, sign-mask
construction, and multiplication execute entirely on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — FP16 triangular layouts

The complete numeric FP16 portions of upstream `test_tril` and `test_triu` now join the Rockchip census: square and
rectangular matrices, positive and negative diagonal offsets beyond both matrix bounds, batched inputs, and an empty
matrix dimension. The final external-boolean-input fixture in each upstream method remains outside the FP16-input
contract.

`~/npu/include/old/rknn_ops.md` marks the model-level Trilu operation unsupported, and neither the historical Rockchip
branches nor `~/rk3588` contain a dedicated register implementation. None is needed here: Tinygrad lowers each static
triangular mask to the backend's existing raw gather/fill representation. The runtime moves FP16 lane representations
according to compile-time offsets and fills masked lanes with the FP16 zero bit pattern; it performs no tensor-value
comparison or arithmetic.

- Focused triangular group: **2 passed, 28 subtests passed in 3.40 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 228 subtests passed in 314.50 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no host numeric implementation. Static raw-lane movement and zero
initialization are layout preparation, not Trilu arithmetic. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — FP16 padded and reordered arithmetic composition

Unchanged upstream `test_padding_add` and `test_topo_sort` now join the Rockchip census. The first adds a `(60,60)`
tensor into a statically zero-padded `(64,64)` layout; the second verifies the shared `(x+x)*x` graph remains correctly
ordered. Each realization executes one ioctl submit on DPU EW.

Historical milestone `54f88978a` solved normal-FP32 padding-add through a host elementwise boundary and was deliberately
not ported. Under the current FP16 contract, the existing gather path prepares only the static raw-lane padding layout,
then DPU ADD performs every numeric addition. Topological ordering needs no dedicated Rockchip primitive and composes
the already-proven DPU ADD and MUL stages in one PC chain. Neither `~/npu` nor `~/rk3588` contains a more specific
register implementation for this graph composition.

- Unchanged upstream `test_padding_add`: **1 passed in 2.95 s**, one submit/one task.
- Unchanged upstream `test_topo_sort`: **1 passed in 3.28 s**, one submit/45 tasks for the non-scalar realization.
- Focused padding and polynomial groups: **10 passed in 5.89 s**, sequentially.
- Complete Rockchip census: **327 passed, 11 skipped, 228 subtests passed in 315.28 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no host tensor arithmetic. Host work is limited to compile-time
index analysis and raw FP16 lane placement for padding; ADD and MUL execute on DPU EW. There is no LUT, CMAC, CNA, PPU,
tinygrad core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — small-axis FP16 variance composition

The unchanged upstream `test_var_one_in_axis` now joins the Rockchip census. It covers size-one reduction axes,
correction values 0, 1, and 5, invalid degrees of freedom, scalar output, and reductions across axes `(0,3)` and
`(0,4)`. Nonconstant `(0,4)` cases execute the existing DPU EW mean, centered-square, reduction, and scale composition
in two ioctls; invalid-degree and size-one cases legitimately simplify to constant NaN or zero.

Historical milestone `0d5561074` passed the larger normal-FP32 variance groups by calling `np.var` in the runtime and
was deliberately not ported. `~/npu` documents model-level MeanVarianceNormalization but provides no applicable DPU
register implementation. The current branch contains no variance-specific runtime path or host tensor arithmetic.
Large `(15,25,35)` keepdim variance remains structurally unsupported and is not claimed by this milestone.

- Unchanged upstream `test_var_one_in_axis`: **1 passed in 2.95 s**, sequentially.
- Nonconstant `(0,4)` probes: **2 submits**, **64–107 DPU EW tasks**; outputs finite and correct.
- Complete Rockchip census: **328 passed, 11 skipped, 228 subtests passed in 314.70 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no NumPy variance implementation. Mean, subtraction, squaring,
reduction, and scaling execute on DPU EW whenever the result is not compile-time constant. There is no LUT, CMAC, CNA,
PPU, tinygrad core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — full scalar FP16 variance on DPU EW

The unchanged upstream `test_var` method now joins the Rockchip census. Its three `(15,25,35)` cases exercise
correction values 1, 0, and 5. Tinygrad schedules each case as a scalar mean followed by
`SUM((x-mean)^2) * scale`; the renderer now recognizes that exact centered-square accumulator graph.

The native lowering gathers the input as aligned raw FP16 lanes, subtracts the independently computed scalar mean,
squares every delta, reduces the squares with a balanced DPU ADD tree, and applies Tinygrad's correction scale. Each
case uses two PC-chain ioctls and 52,500 DPU tasks: 13,125 for the mean and 39,375 for the centered-square pass. No LUT
or CMAC is involved.

Historical milestone `0d5561074` was inspected and rejected because it calls `np.var` in the runtime. Later
`rockchip-upstream-research` variance work records rejected CMAC/selector experiments rather than a usable DPU EW
implementation. `~/npu` documents MeanVarianceNormalization only as a CPU operator, and `~/rk3588` contains no native
variance register example.

The upstream strict-tolerance run completed all three NPU executions; correction 5 differed by `0.001953`, just beyond
its generic `1e-3` relative tolerance. The unchanged method passes under the Rockchip census's established FP16
`5e-3/5e-3` tolerance, which is the same limit used for `test_gemm_fp16`.

- Unchanged upstream `test_var`: **1 passed in 10.31 s**, sequentially.
- Per correction: **2 submits**, **52,500 DPU tasks**.
- Wall decomposition per correction: **2.43–2.84 s realization**, of which only **0.022 s** is inside submit ioctls;
  graph construction is **0.002–0.005 s** and copyout is **0.001–0.003 s**.
- Reduction regression: **24 passed, 1 skipped in 19.29 s**; post-cleanup reduction/dot regression:
  **30 passed, 1 skipped in 26.17 s**.
- Complete Rockchip census after cleanup: **329 passed, 11 skipped, 228 subtests passed in 331.20 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,766/269 executable lines**.

The CPU-cheat audit found only compile-time UOp matching/index evaluation and raw FP16 lane gathering. Runtime performs
no variance, subtraction, multiplication, addition, scaling, or tensor-value inspection on the CPU. The arithmetic is
entirely DPU EW. The cleanup shares cast/local-accumulator parsing across dot, scalar, and centered-square reducers,
following the small-helper style used by the other Tinygrad backends. There is no LUT, CMAC, CNA, PPU, tinygrad core
change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — row-wise FP16 variance on DPU EW

The unchanged upstream `test_var_axis`, `test_var_zero_in_axis`, and `test_var_keepdim` methods now join the Rockchip
census. Together with the preceding scalar and one-axis milestones, all five forward-variance methods in `TestOps`
are covered under the FP16 contract.

The `(15,25,35)` axis cases reduce `K=15`, `K=35`, or `K=875`. Tinygrad already unrolls the two short reductions into
ordinary DPU EW graphs. For the `15` independent rows at `K=875`, the scalar reducer is generalized to pack each
reduction candidate as one aligned 15-lane vector. One DPU task therefore advances all rows together. The centered-
square pass uses the same vector layout for SUB, MUL, balanced ADD, and correction scaling.

The large row-wise case compiles to 875 mean tasks plus 2,625 centered-square tasks, rather than 52,500 scalar tasks.
Its 3,500 tasks remain in two dynamically sized PC-chain submits. The shared `_spaced_reduction_gathers` helper keeps
the scalar and row-vector layouts on the same compile-time raw-lane path.

- Unchanged upstream `test_var_axis`: **1 passed in 3.92 s**, sequentially.
- Axis, zero-axis, and keepdim variance: **3 passed in 6.74 s**, sequentially.
- Complete reduction class after cleanup: **27 passed, 1 skipped in 22.96 s**.
- Per nonconstant case: **2 submits**; task counts are **147** (`axis=0`), **227** (`axis=2`), and **3,500**
  (`axis=(1,2)`).
- Per-case realization wall time is **0.060–0.427 s** after graph construction; submit ioctl time is
  **0.0004–0.0017 s**. Maximum absolute error across the profiled axis cases is **0.00390625**.
- Complete Rockchip census: **332 passed, 11 skipped, 228 subtests passed in 333.74 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,784/269 executable lines**.

The CPU-cheat audit found no runtime change and no host tensor arithmetic. Compile-time index evaluation constructs a
candidate-major raw FP16 lane layout; mean, subtraction, squaring, reduction, and scaling all execute on DPU EW. The
implementation follows the existing backend pattern of a strict graph matcher plus a compact immutable image and a
shared layout helper. There is no LUT, CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation beyond the
existing FP16 contract.

---

## 2026-08-08 — canonical FP16 cumulative extrema census

All eight unchanged upstream cumulative-extrema method names now join the Rockchip census: small, long, general-axis,
and zero-axis `cummax` and `cummin`. Their FP16 values and exact INT32 first-occurrence indices were already proven by
separate custom wrapper classes. This milestone replaces those 18 duplicate wrapper items with the canonical eight
methods, preserving every shape, axis, error, value, and index case while removing 29 net test lines.

Historical commits `418605353`, `8c811ede9`, and `305f7f793` provide the native DPU value, small-index, and long-index
implementations. `~/rk3588/test/test_ops.py` contains the same upstream cases and no separate register-level primitive;
the implementation remains the current DPU equality-mask, weighted-coordinate, reduction, and INT32 conversion path.

The initial canonical `test_simple_cummin` took 34.28 seconds. Profiling showed only about 24 ms in submit ioctls; the
512-index realization spent 6.7 seconds in generic linearization, 5.2 seconds rendering two large unrolled graphs, and
2.3 seconds in raw-lane preparation and command construction. `NOOPT` was rejected because it exposes a different
unsupported value-loop graph. The Rockchip test wrapper instead realizes the 512 value/index pair from one shared
graph and keeps the proven separate 1022 schedules. This preserves exact results and lowers the method to 28.61
seconds without a backend, runtime, or Tinygrad core change.

- Canonical cumulative-extrema class: **8 passed in 60.52 s**, sequentially.
- Slowest methods: `test_simple_cummin` **28.61 s**, `test_simple_cummax` **21.45 s**.
- Complete Rockchip census: **322 passed, 11 skipped, 180 subtests passed in 326.60 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,784/269 executable lines** (unchanged).

The CPU-cheat audit found no backend or runtime diff. NumPy and Torch appear only in the test oracle; NPU values and
indices still come from DPU EW and the existing raw-lane layout machinery. There is no LUT, CMAC, CNA, PPU, tinygrad
core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-09 — native FP16 IEEE classification and typed bool output

The unchanged upstream `test_isnan`, `test_isinf`, and `test_isfinite` methods now join the Rockchip census. The
`test_isinf` method also covers positive-only and negative-only detection. A strict graph recognizer accepts only
Tinygrad's canonical `x != x`, equality-to-`±inf`, their OR, and the finite complement over one FP16 input.

Classification uses the DPU positive-mask primitive around the largest finite half value. Positive and negative
overflow masks identify `+inf` and `-inf`; their intersection identifies NaN because the proven RK3588 comparison
pipeline marks NaN in both directions. DPU MAX, MUL, and SUB then form the requested 0/1 predicate. The final FP16 mask
is converted to INT32 by the existing typed DPU conversion. Runtime packs the public one-byte bool ABI by copying only
the low byte of each little-endian INT32 lane; it performs no comparison, cast, classification, or value-dependent
branch on the CPU.

Historical branch `rockchip-2608` milestone `091df1bd8` supplied the proven mask formula, but its runtime used NumPy
`!= 0` to produce bool values and was therefore not ported. `~/npu/include/old/rknn_ops.md` lists model-level IsInf and
IsNaN as unsupported, while `~/rk3588` provides no separate native classification example. The current implementation
reuses only the already-proven DPU comparison and typed-conversion register paths.

- Each seven-element predicate probe: **6 submits**, **7–9 DPU tasks**, and **0.55–0.65 s** warm-process wall time.
- Focused canonical classification class: **3 passed in 5.67 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 337.66 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,846/274 executable lines**.

The CPU-cheat audit found only raw memory movement for the public bool representation. All classification and FP16-to-
integer conversion occur on DPU EW. There is no LUT, CMAC, CNA, PPU, Tinygrad core change, or tolerance relaxation.

---

## 2026-08-09 — conservative Rockchip renderer matcher cleanup

The DPU EW renderer now parses the single typed output store once, shares one `RKLoopReduction` descriptor across dot,
scalar, and centered-square reductions, and shares balanced reduction/layout helpers. Fused and separately materialized
ArgMax/ArgMin graphs now enter one `_lower_unrolled_arg_extrema` matcher and one common first-tie/gather/image tail.
Sort parsing also reuses binary-tree flattening, equality-pair splitting, and parameter-load grouping.

The slowest method remains `TestRockchipCumulativeExtremaOps.test_simple_cummin`. An exact test-body profile measured
27.37 s: 0.010 s input creation, 0.0004 s Torch reference, 0.028 s graph construction, 27.32 s realization, 0.0046 s
copyout, and 0.0050 s assertions. Within realization, seven renderer calls consumed 8.41 s and nine program calls
consumed 8.43 s. The 73 submit ioctls themselves consumed only 0.037 s, while 73 required NPU resets consumed 7.77 s;
the method issued 73 submits and 4,225 DPU tasks. This cleanup deliberately does not alter reset or submission semantics.

An attempted shared equality-mask emitter was rejected: `sz.py` increased by one executable line and the abstraction
hid operation-specific DPU barrier placement. Keeping the explicit occurrence/cumulative/sort emission is smaller and
clearer. The accepted cleanup reduces the renderer from **1,846 to 1,753 executable lines** (93 lines) without removing
features.

- Focused merged ArgMax/ArgMin coverage: **6 passed, 10 subtests passed in 10.99 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 321.90 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,753/274 executable lines**.

The CPU-cheat audit found no runtime or test change in this milestone. All new helpers operate only on compile-time UOp
structure, static index maps, and immutable command-image metadata; they do not inspect or calculate tensor values.
There is no LUT, CMAC, CNA, PPU, Tinygrad core change, or tolerance relaxation.

---

## 2026-08-09 — shared equality masks and striped gather matrices

Occurrence counting, sort-index selection, and cumulative-index selection now share `_ew_eq_mask` for their FP16
`SUB -> ABS -> nonzero MAX -> 1-mask` sequence. Its two explicit barrier bits preserve the previously tested boundaries:
ordinary value equality splits before ABS, while converted sort-count equality splits before SUB. Candidate, repeated-
current, constant-mask, and weight rows now share `_stripe_layout` and `_stripe_gathers` rather than rebuilding aligned
matrix gathers in each lowering path.

The occurrence and sort parsers intentionally remain separate. Occurrence validates optional static prefix gates, while
sort validates weighted conjunctions of FP16 value equality and INT32 occurrence-count equality. A shared matcher would
hide these distinct IR contracts without reducing executable lines. `_lower_sort_compare` is also unchanged because its
static WHERE-to-MAX/MIN algorithm does not use equality masks.

- Focused arg/sort/cumulative regression: **26 passed, 30 subtests passed in 154.22 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 321.03 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,744/274 executable lines**.

The cleanup removes nine more executable renderer lines and changes no test, runtime, command encoding, tolerance, or
Tinygrad core file. Its gathers contain only compile-time offsets/constants, and all equality arithmetic remains DPU EW;
there is no host tensor-value computation, LUT, CMAC, CNA, or PPU fallback.

---

## 2026-08-09 — IEEE-correct FP16 comparison outputs on DPU EW

The FP16 portions of all six comparison predicates now join the Rockchip census: equal, not-equal, less, greater,
less-or-equal, and greater-or-equal. Coverage includes equal shapes, both broadcast directions, scalar constants in both
operand positions, signed zero, finite values, both infinities, and every pair involving NaN. Integer and boolean input
comparisons remain outside the NPU's FP16 external-input contract.

Each operand is classified on-device into NaN, positive infinity, negative infinity, and finite masks using the proven
DPU positive-mask primitive. Ordered comparisons combine a finite subtraction mask with explicit infinity ordering and
an unordered-NaN validity mask. Equality recognizes equal finite values and matching signed infinities; NaN is never
equal and is always not-equal. Tinygrad lowers `>=` and `<=` as boolean inversion of `<`, so inversion is deliberately
performed inside the validity mask rather than as a plain `1-result`, preserving IEEE false-on-NaN behavior.

Research commit `bd1b8009a` supplied the proven device formula. `~/npu/include/old/rknn_ops.md` lists Greater,
GreaterOrEqual, Less, and LessOrEqual as supported model operations, and `~/rk3588/test/test_rockchip.py` contains the
same comparison matrix. The older `89c7cb67b` implementation was rejected because its runtime sanitized infinities and
converted comparison tensors with NumPy. This port uses the current immutable DPU EW image and existing lossless typed
bool-output ABI instead.

- Focused six-predicate class: **6 passed in 56.45 s**, sequentially.
- Classification plus comparisons: **9 passed in 59.17 s**, sequentially.
- Complete Rockchip census: **331 passed, 11 skipped, 180 subtests passed in 375.24 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,797/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. Runtime never reads, sanitizes, compares, or converts input
tensor values; all classification, comparison, boolean inversion, and FP16-to-INT32 mask conversion occur on DPU EW.
Only the existing raw low-byte packing exposes NPU-produced 0/1 lanes as public bool bytes. There is no LUT, CMAC, CNA,
PPU fallback, or tolerance relaxation.

---

## 2026-08-09 — FP16 logical-not and scalar isclose coverage

The FP16 half of `test_logical_not` and the unchanged upstream `test_isclose_scalar` now join the Rockchip census.
Logical-not covers ordinary nonzero values, positive and negative zero, both infinities, and NaN. Scalar isclose covers
the canonical `(3,4,5,6)` tensor-to-1 comparison plus exact/near/unequal finite values, infinities, and NaN. Both lower
entirely through the preceding IEEE comparison milestone and typed bool-output ABI; no backend change was needed.

General tensor/tensor `test_isclose` was tested and intentionally excluded. Its `x.isclose(x+1e-6)` case produced 14
false results among 360 expected true results because the FP16 tolerance/difference enters the DPU's subnormal region.
Historical commits `b85d3c4b0` and `31c07151d` passed this group only through `_HOST_ELEMENTWISE_LAYOUT`, explicitly
computing isclose on the CPU; that path is not ported. Their native WIP also documented reset explosion on the 32-case
IEEE edge matrix. Exact boolean expectations were not relaxed.

- Focused logical-predicate class: **2 passed in 14.95 s**, sequentially.
- Complete Rockchip census: **333 passed, 11 skipped, 180 subtests passed in 388.80 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,797/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change and no
new data conversion or host arithmetic. There is no LUT, CMAC, CNA, PPU fallback, or tolerance relaxation.

---

## 2026-08-09 — FP16-input ANY/ALL reductions on DPU EW

Global, tuple-axis, scalar, and empty-axis `any`/`all` now join the Rockchip census for FP16 inputs. Coverage includes
positive and negative zero, finite nonzero values, both infinities, NaN, mixed result rows, and the exact empty
identities (`any=False`, `all=True`). External boolean input tensors and upstream `test_all_large`, which constructs a
boolean tensor, remain outside the NPU's FP16 external-input contract.

Both Tinygrad reduction forms are recognized: small fully unrolled OR/AND trees and register-loop reductions. Static
source offsets are packed into an aligned candidate-row matrix. DPU `SUB -> ABS -> positive mask` produces exact
nonzero lanes, then a balanced DPU MAX tree implements ANY and a balanced DPU MUL tree implements ALL. The existing
typed DPU conversion writes the public boolean result. Empty reductions contain no input values and use the existing
constant-fill image for their mathematically fixed identity.

Historical commit `03bad6205` was inspected but not ported: it uses CMAC to count masks, performs host bool/FP16
conversion, and adds host tiling for multi-megabyte reductions. Searches under `~/npu` and `~/rk3588` found the
canonical tests but no independent pure-DPU boolean-reduction register implementation. This milestone therefore
reuses only the currently proven DPU EW mask, striped-gather, balanced-reduction, and typed-output machinery.

- Before specialization, 5-element reductions used **46 submits**, 12-element reductions used **102 submits**, and
  the 360-element loop form was rejected. All three forms now pass with **6 submits** each.
- Focused boolean-reduction class: **6 passed in 5.26 s**, sequentially; slowest method **1.03 s**.
- Complete Rockchip census: **339 passed, 11 skipped, 180 subtests passed in 389.29 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,864/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. Runtime gathers only raw FP16 lanes using compile-time
offsets; it never reads, compares, reduces, or branches on tensor values. All nonzero classification and reduction
arithmetic execute on DPU EW. Static empty identities are constant initialization, not host evaluation of input data.
There is no LUT, CMAC, CNA, PPU fallback, FP32/boolean external-input conversion, or tolerance relaxation.

---

## 2026-08-09 — native FP16 floor/ceil and no-LUT truncation

The unchanged upstream `test_floor`, `test_ceil`, and `test_trunc` methods now join the Rockchip census. RK3588 TRM
`RKNN_dpu_ew_cfg` identifies EW ALU algorithms 7 and 8 as Floor and Ceil. The renderer recognizes Tinygrad's canonical
TRUNC-based WHERE expansions and tags them for those named native configurations; it does not embed unexplained
register literals.

TRUNC has no corresponding native ALU algorithm. It is composed entirely on DPU EW as
`floor(max(x, 0)) + ceil(min(x, 0))`, with `min(x,0)` expressed through SUB/MAX. This avoids the DPU's unsafe
`infinity * 0` behavior and preserves finite values, both infinities, and NaN numerically. It does not preserve the
negative-zero sign when a negative value with magnitude below one truncates to zero; canonical Tinygrad/Torch numeric
comparison treats both zeros as equal, and no tolerance was relaxed.

Historical `5619ceff8` was rejected because it routes floor/ceil through `_HOST_ELEMENTWISE_LAYOUT`. Historical
`29559149c` composes integral rounding from a generated roundoff LUT, so only its test intent was relevant. The direct
TRM algorithms and current DPU EW image are used here. `test_round` was also tested and remains excluded: Tinygrad's
round-to-even graph still contains unsupported boolean/XOR selection, while the proven historical implementation uses
a LUT, which is outside the current scope.

- Focused canonical methods: **3 passed in 3.17 s**, sequentially.
- Permanent rounding class: **4 passed in 3.10 s**; its exhaustive method checks all **65,536 FP16 encodings** and the
  three operations use exactly **3 ioctls** total.
- Complete Rockchip census: **343 passed, 11 skipped, 180 subtests passed in 391.86 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,892/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The new renderer code only recognizes canonical UOps and
selects DPU ALU configurations; all floor, ceil, min/max, subtraction, and addition stages execute on the NPU. There is
no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, or tolerance relaxation.

---

## 2026-08-09 — direct logical composition of FP16 predicates

Direct boolean AND, OR, and XOR outputs derived from FP16 comparisons now join the Rockchip census. AND includes the
FP16 portion of upstream `test_and`; OR and XOR cover finite values, signed zero, both infinities, and NaN. External
INT32 and boolean input portions of the upstream bitwise methods remain outside the NPU's FP16 input contract.

AND and OR were already expressible by the IEEE comparison-mask builder as DPU MUL and MAX respectively. XOR now uses
the exact 0/1 identity `abs(lhs-rhs)`, emitted as DPU SUB followed by the proven native ABS algorithm, before the
existing typed DPU boolean conversion. Historical `e7e2f2720`, `22c974ac2`, `23800edca`, and `59dcb9a15` were inspected
but not ported because they operate on external public bool tensors through INT8 DPU surfaces; this milestone instead
composes masks produced on-device from FP16 inputs.

- Focused logical-predicate class: **5 passed in 20.41 s**, sequentially; AND/OR/XOR are each **1.89–1.92 s**.
- Complete Rockchip census: **346 passed, 11 skipped, 180 subtests passed in 397.71 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. All operand classification, comparison, logical
composition, subtraction, ABS, and typed bool conversion occur on DPU EW; the CPU never reads or branches on tensor
values. There is no LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — FP16 fmod and modulo compositions on DPU EW

The complete floating-input portions of upstream `test_fmod` and `test_mod` now join the Rockchip census. Coverage
includes tensor/tensor remainder, integer-valued and fractional scalar divisors, and both integer-valued and fractional
reverse scalar numerators. Integer tensor inputs remain outside the NPU's FP16 external-input contract.

No dedicated modulo register path is required. Tinygrad's FP16 graphs decompose `fmod` through FDIV, TRUNC, MUL, and
SUB, and Python-style modulo through FDIV, FLOOR, MUL, and SUB. The preceding native integral-rounding milestone lets
each complete expression execute as one DPU PC-chain ioctl. The tests assert exactly **7 ioctls for 7 modulo forms**
and **3 ioctls for 3 fmod forms**, preventing a host fallback from satisfying correctness alone.

Historical commits `ead24405a` and `86b0f1c6f` were inspected but not ported because both explicitly route the
operation through `_HOST_ELEMENTWISE_LAYOUT`. Searches under `~/npu` and `~/rk3588` found CPU reference `fmodf` and
model-operator documentation, but no independent native modulo register implementation. The current composition uses
only already proven DPU EW operations.

- Focused modulo class: **2 passed in 3.03 s**, sequentially; method call times were **0.14–0.17 s**.
- Complete Rockchip census: **348 passed, 11 skipped, 180 subtests passed in 396.01 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. Runtime
gathers remain raw lane movement by compile-time offsets; all division, rounding, multiplication, and subtraction run
on DPU EW. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion,
or tolerance relaxation.

---

## 2026-08-09 — FP16 division rounding modes on DPU EW

The complete FP16-input portion of upstream `test_div_rounding_mode` now joins the Rockchip census. One seven-element
FP16 numerator is divided by each of ten signed, nonzero length-one FP16 denominators under true division, truncating
division, and floor division. The 30 cases cover positive and negative operands, exact and inexact quotients, and zero.
Integer tensor combinations remain outside the NPU's FP16 external-input contract.

Tinygrad lowers these forms to scalar-broadcast DPU FDIV, followed by the native TRUNC or FLOOR composition introduced
in the integral-rounding milestone. Every expression is one PC-chain ioctl; the test asserts exactly **30 ioctls for
30 cases**. Historical commit `f29131cdf` and its documentation commit `c46039f9a` were inspected but not ported because
they execute the division and rounding graph through a typed host elementwise task. Searches under `~/npu` and
`~/rk3588` found the canonical test but no independent native implementation beyond the already used DPU primitives.

- Focused division-rounding class: **1 passed in 2.86 s**, sequentially; call time was **0.27 s**.
- Complete Rockchip census: **349 passed, 11 skipped, 180 subtests passed in 397.56 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
division and integral-rounding arithmetic executes on DPU EW, with only static scalar-broadcast address preparation on
the host. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion,
or tolerance relaxation.

---

## 2026-08-09 — constant integer powers of FP16 inputs

The integer-exponent portion of upstream `test_pow` now joins the Rockchip census: vector `x**0`, `x**1`, `x**2`,
`x**3`, and `x**-2`; scalar `x**2` and `x**-2`; and the two negative-range cubic regressions. Fractional and runtime
tensor exponents remain excluded because their historical Rockchip implementations depend on transcendental/LUT
machinery, which is outside the current scope.

Tinygrad simplifies the supported vector forms into static fill/copy, repeated MUL, or reciprocal plus MUL. `x**0` is
the input-independent constant one and needs no ioctl; the four remaining ordinary vector forms and the negative-range
vector cubic each execute in one DPU PC-chain. Rank-zero literal inputs constant-fold without a submit. The test asserts
exactly **5 ioctls for the 9 forms**.

The historical power series was inspected rather than ported. Commit `463d0dfd6` evaluates integer-tensor powers in a
typed host task, while later general/fractional power commits use generated range-reduced LUT graphs. `~/npu` and
`~/rk3588` contain the canonical tests but no simpler native constant-integer-power register operation. This milestone
therefore relies only on Tinygrad's existing decomposition and proven DPU EW MUL/FDIV stages.

- Focused integer-power class: **1 passed in 3.10 s**, sequentially; call time was **0.49 s**.
- Complete Rockchip census: **350 passed, 11 skipped, 180 subtests passed in 397.32 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. Every
input-dependent vector multiplication and reciprocal runs on DPU EW; static constant fills and compile-time rank-zero
folds do not inspect a runtime tensor buffer. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — permuted INT32 WHERE output from FP16 predicates

Upstream `test_where_permute` now joins the Rockchip census. Its FP16 comparison selects the INT32 constants 4 and 2,
then writes a transposed result. Coverage also includes a non-square matrix containing NaN, both infinities, positive
and negative zero, and a finite true case, proving IEEE predicate behavior and the static transpose mapping.

The renderer now shares one typed DPU output helper between public bool masks and exact FP16-valued INT32 expressions.
A strict matcher accepts only a DPU-lowerable FP16 comparison and two constant INT32 arms whose affine selection is
exactly representable in FP16. DPU EW computes `false + mask*(true-false)`, and the existing hardware FP16-to-INT32
output-conversion stage writes the public result. Static permutation is expressed only by compile-time gather offsets.

Each IEEE-safe comparison currently uses **12 barrier-separated ioctls**, so the two permanent cases assert **24
ioctls**. The barriers come from the existing NaN/infinity classification and positive-mask stages, not CPU execution
or command-buffer tiling. The method call takes **2.58 s**, well below the 30-second profiling threshold.

Historical commit `297b6fdc0` was inspected but not ported: it writes four FP16 byte planes and reassembles INT32 values
with NumPy in the runtime. The current backend already has direct DPU INT32 output conversion, so this implementation
needs no host reassembly, typed runtime ABI extension, or image-format change. Exact `copysign` was also tested as the
next candidate and remains excluded: RK3588 FDIV maps both `1/+0` and `1/-0` to positive infinity, losing the sign bit
before Tinygrad's canonical signed-zero predicate; host bit inspection would violate the no-CPU-cheat rule.

- Focused WHERE-permute coverage: **1 passed in 5.22 s**, sequentially.
- Shared WHERE/classification/comparison/logical/reduction regression: **30 passed in 85.02 s**, sequentially.
- Complete Rockchip census: **351 passed, 11 skipped, 180 subtests passed in 401.19 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,909/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The renderer inspects only constant arms, dtypes, UOp
structure, and static indexes; it never reads tensor values. Comparison, selection arithmetic, and FP16-to-INT32
conversion execute on DPU EW. There is no host tensor-value evaluation or reassembly, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — direct FP16-to-bool cast on DPU EW

The FP16-input portion of upstream `test_cast` now joins the Rockchip census. Coverage includes a normal tensor plus
finite positive/negative values, positive and negative zero, both infinities, and NaN. FP32 output, external integer
input, and external boolean input portions remain outside the NPU's FP16 external-input contract.

Tinygrad lowers direct `.bool()` to `CMPNE(input, 0)`. The generic IEEE equality path was exact but used **14 ioctls**.
A strict direct-load matcher now uses the identity `positive_mask(abs(input))`: native ABS maps both signed zeros to
zero and both infinities to positive infinity, while the proven DPU positive-mask primitive treats NaN as nonzero.
The public bool result uses the existing hardware FP16-to-INT32 conversion and raw low-byte packing. Each cast now uses
**3 ioctls**, so the two permanent cases assert **6 total**.

Historical `091df1bd8` established an older typed bool-output ABI whose runtime packed NPU-produced FP16 masks with a
NumPy nonzero operation; `70740a365` documented rejection of direct channel-packed FP16 bool output. Neither path was
ported. Searches under `~/npu` and `~/rk3588` found CPU reference checks and earlier comparison experiments but no
smaller native nonzero register sequence than the current ABS/positive-mask composition.

An exhaustive 65,536-pattern probe was attempted but deliberately not added. Public bool conversion operates on
four-lane INT32 atoms; 65,536 values therefore create 16,384 conversion tasks, which the current page-sized conversion
batching splits into roughly 1,171 submit/reset groups. The probe returned naturally after about two minutes with no
driver timeout, but its output channel was lost and thus provides no correctness evidence. The immediate vendor health
check still passed 60/60. This identifies conversion batching as a separate performance problem, not a cast-semantic
failure.

- Focused cast class: **1 passed in 3.42 s**, sequentially; call time was **0.79 s**.
- Shared classification/logical/boolean-reduction regression: **14 passed in 25.53 s**, sequentially.
- Complete Rockchip census: **352 passed, 11 skipped, 180 subtests passed in 400.13 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** both after the stress probe and after the census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,914/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. ABS, nonzero classification, and FP16-to-INT32 conversion
all execute on DPU EW; runtime only packs the low byte of NPU-produced 0/1 INT32 lanes into the public bool buffer. The
renderer inspects only the UOp shape and dtype. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — arithmetic FP16 softsign on DPU EW

Upstream `test_softsign` and `test_softsign_exact` now join the Rockchip census. The group covers a 45x65 tensor, a
rank-zero scalar, and the exact values `[-1, 0, 1]` under the same FP16 tolerance used by `test_gemm_fp16`.

Tinygrad defines softsign as `x / (1 + abs(x))`. The nonconstant cases therefore reuse the already-proven native DPU
EW ABS, ADD, and FDIV stages rather than adding a special operation. The 45x65 and exact-vector graphs each execute as
three DPU tasks in one ioctl; the literal rank-zero input folds at compile time without an ioctl. A warm direct profile
measured **0.150 s** for the matrix and **0.020 s** for the exact vector.

Other Tinygrad branches contain no Rockchip-specific softsign lowering. `~/npu` and `~/rk3588` contain a standalone
Q0.15 LUT implementation, but it was intentionally not ported because the current scope excludes LUT work and ordinary
FP16 DPU arithmetic already implements the canonical Tinygrad expression.

- Focused upstream cases: **2 passed**, sequentially (`3.09 s` and `2.68 s` pytest wall times).
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **354 passed, 11 skipped, 180 subtests passed in
  401.34 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,914/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
input-dependent ABS, addition, and division execute on DPU EW; compile-time scalar folding does not inspect a runtime
tensor buffer. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input
conversion, or tolerance relaxation.

---

## 2026-08-09 — embedded FP16 comparison masks for backward graphs

The comparison-backward invariants from upstream `test_cmp_ne_backwards` and `test_cmp_lt_backwards` now join the
Rockchip census. The upstream methods use `Tensor.randn`, whose normal-distribution graph requires transcendental/LUT
support. The Rockchip variants instead use deterministic FP16 values spanning negative infinity, finite negative,
signed zero, finite positive, positive infinity, and NaN while preserving the same derivative check against Torch.

Backward exposes a previously unsupported graph boundary: Tinygrad casts the boolean comparison result back to FP16
before multiplying it into the gradient. The renderer now narrowly matches `CAST(bool -> half)`, following the typed
cast-matcher style used by the PTX renderer, and replaces it with an existing DPU FP16 0/1 mask. Direct `x != 0` uses
the optimized ABS/nonzero path; other predicates use the IEEE-correct comparison mask already proven for public bool
output. No new register primitive or runtime ABI is required.

The `x != 0` backward graph executes **2 DPU tasks in 2 ioctls**. IEEE-correct `x < 0` executes **31 DPU tasks in 11
ioctls**. Both gradients exactly match Torch across all seven finite/nonfinite inputs.

No separate Rockchip comparison-backward implementation exists in the other Tinygrad branches, `~/npu`, or
`~/rk3588`; their available material covers forward comparison primitives. The existing DPU masks are therefore the
relevant reference. The historical CPU isclose paths were also reviewed and remain excluded.

- Focused backward group: **2 passed in 4.00 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **356 passed, 11 skipped, 180 subtests passed in
  402.05 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,918/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The rewrite inspects only predicate UOp structure and all
mask construction plus gradient arithmetic executes on DPU EW; Torch is used only for the expected gradient. There is
no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion, or tolerance
relaxation.

---

## 2026-08-09 — FP16 einsum ellipsis layouts

Three executable layouts and both static validation cases from upstream `test_einsum_ellipsis` now join the Rockchip
census. The supported forms cover batched matrix products (`...id,...jd->...ij`), an implicit scalar reduction
(`...id,...jd`), and permuted ellipsis axes (`i...j,ji...->...`). They reuse the existing FP16 gather, MUL, and ADD
lowering without a special einsum operation.

The remaining upstream execution case, two `(32,7,24,24,24)` inputs reduced by `ij...,ij...->ij`, was tested but is
not admitted yet. Tinygrad lowers each of its 224 output rows to 432 loop iterations with a 32-product unrolled ADD
tree, or 13,824 products per row and 3,096,576 input lanes overall. The current loop-reduction matcher recognizes the
shape but its dot path accepts only one MUL contribution per iteration. Flattening this graph into ordinary EW stages
would create an impractically large command sequence; it needs a bounded hierarchical reduction rather than a broad
matcher relaxation.

The canonical ellipsis cases are present under `~/rk3588/test/test_ops.py`. Searches across the other Rockchip
Tinygrad branches and `~/npu` found no separate hardware-specific einsum implementation to port; their implementation
is likewise the generic Tinygrad reshape/gather/multiply/reduce graph.

- Focused Rockchip method: **1 passed in 3.03 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **357 passed, 11 skipped, 180 subtests passed in
  400.70 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,918/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
runtime-dependent lane movement and arithmetic execute through the existing Rockchip NPU path; Torch is used only as
the expected-value oracle. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16
input conversion, or tolerance relaxation.

---

## 2026-08-09 — exact FP16-pair bitcast to INT32

Rockchip now preserves the exact representation of adjacent FP16 lanes when Tinygrad bitcasts them to INT32. Permanent
coverage includes contiguous and leading-axis-permuted layouts over positive and negative zero, infinities, signed NaNs,
subnormals, maximum finite values, and ordinary normals.

The unchanged upstream `test_bitcast` uses shape `(3,3)`. Under `DEFAULT_FLOAT=HALF`, PyTorch rejects it before the
backend runs because its final dimension is not divisible by the two FP16 lanes required for one INT32 element. The
Rockchip method therefore uses the equivalent valid shape `(2,3,4)` and checks both its direct and permuted views.

Tinygrad expresses the reinterpretation as two `BITCAST(half -> ushort)`, zero-extension, shifts by 0/16, an unsigned
ADD, and `BITCAST(uint -> int)`. The strict matcher accepts only that exact graph, proves that both loads use one bounded
FP16 argument, and verifies that every high lane is the adjacent successor of an even low lane. It then copies each
four-byte representation with the existing typed raw gather; no numeric cast, shift, or addition is performed on host
or DPU. This follows other Tinygrad renderers' exact `Ops.BITCAST` treatment while mapping it to Rockchip's existing raw
layout ABI instead of inventing a scalar instruction stream.

No hardware-specific bitcast implementation exists in the other Rockchip Tinygrad branches, `~/npu`, or `~/rk3588`.
The legacy `~/rk3588/experimental/ops_rockchip.py` interpreter uses host `struct.pack/unpack`; that CPU value interpreter
was inspected and deliberately not ported.

- Focused exact-bit method: **1 passed in 2.85 s**, sequentially; both layouts assert bit-for-bit equality and zero submits.
- Typed cast/bitcast/TopK regression: **6 passed, 2 subtests passed in 21.52 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **358 passed, 11 skipped, 180 subtests passed in
  401.05 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found only compile-time UOp/index validation and raw `uint32` representation movement by the same
gather mechanism already used for device-produced indices. Neither renderer nor runtime reads or interprets an FP16
numeric value. There is no host tensor-value arithmetic, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, Tinygrad
core change, or tolerance relaxation.

---

## 2026-08-09 — IEEE NaN propagation through FP16 extrema reductions

The behavior named by upstream `test_max_nan` now runs actively in the Rockchip census even though the canonical test is
globally skipped as broken by tinygrad issue #862. Runtime FP16 inputs cover NaN in both operand orders for scalar MAX,
plus row-wise MAX and MIN with one NaN row and one finite control row. Every case propagates NaN exactly where required,
preserves the finite result, and executes as one native DPU EW reduction task; the method asserts **4 ioctls total**.

No Rockchip-specific NaN-extrema implementation exists in the other Tinygrad branches, `~/npu`, or `~/rk3588`. The
existing RK3588 EW MAX configuration already has the required propagation behavior, so this is intentionally a test-only
milestone with no renderer/runtime special case.

The remaining large ellipsis-einsum case was investigated further but is still excluded. A temporary hierarchical
layout reduced it from 27,648 gathers and 27,647 logical EW stages to two affine transposes, 15 logical stages, and 901
hardware tiles. One 901-task ioctl hit the six-second driver timeout; splitting at the natural MUL/reduction boundary
completed as two jobs in about six seconds, after which the vendor health check passed 60/60. Plain FP16 pairwise
accumulation nevertheless missed 20/224 outputs at the permitted tolerance, with maximum absolute error 0.25. Modeling
confirmed that even an exact sum of FP16-rounded products fails, so FP16 product residuals are required. The TRM exposes
either EW ALU or MUL per stage rather than a two-input FMA; the non-LUT Dekker path would require more than 11,000 tiled
tasks before compensated accumulation. It was therefore removed instead of admitting inaccurate or >30-second code.

- Focused NaN-extrema regression: **1 passed in 3.16 s**, sequentially.
- Complete reduction class: **28 passed, 1 skipped in 22.71 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **359 passed, 11 skipped, 180 subtests passed in
  402.53 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused work and after the census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found no renderer, runtime, or Tinygrad-core change. All input-dependent extrema arithmetic executes
on DPU EW; Python only asserts the returned scalar/array values. There is no host tensor-value arithmetic, LUT, CMAC,
CNA, PPU fallback, external non-FP16 input, tolerance relaxation, or committed experimental einsum path.

---

## 2026-08-09 — vectorized IEEE isclose edge matrix

All 16 pairs from `{+inf, -inf, NaN, 0}²` in upstream `test_isclose_edge_cases` now run in the Rockchip census for both
`equal_nan=False` and `equal_nan=True`. The Rockchip method preserves every upstream pair but packs each mode into one
16-lane graph instead of paying realization overhead for 32 scalar helper calls. The two graphs reuse the existing DPU
EW ABS, arithmetic, IEEE classification, comparison-mask, and typed bool-output paths. They execute **364 DPU tasks in
104 ioctls** and complete in **14.02 s**; the submit assertion prevents silent constant folding or fallback.

Historical commit `b85d3c4b0` passes isclose through host-side predicate evaluation and was not ported. The current
renderer already represents the complete IEEE predicate on DPU, so this milestone adds no backend special case and does
not relax exact boolean expectations. General finite `test_isclose` remains excluded because FP16 subnormal tolerance
arithmetic loses distinctions such as `x` versus `x+1e-6` before the comparison.

Large boolean reduction was evaluated as the following candidate. `test_all_large` is not constant-folded: after an
FP16 predicate reduction it needs a second 256-way AND over a byte-bool intermediate. Historical commits `25df366fb`
and `03bad6205` widen those bytes to FP16 with NumPy, which is a CPU numeric conversion and is not ported. A clean route
would use raw byte placement into zeroed INT32 lanes plus the existing DPU INT32-to-FP16 conversion, but the upstream
one-million-lane first stage spends more than a minute in Tinygrad symbolic rewriting before reaching the renderer,
including under `NOOPT=1`. Because Tinygrad core changes are out of scope and individual cases must remain below 30 s,
the group is deferred rather than narrowed to only its 32K/64K cases.

- Focused edge-matrix method: **1 passed in 14.02 s**, sequentially.
- Complete logical-predicate class: **6 passed in 31.34 s**; no individual method exceeded 30 s.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **360 passed, 11 skipped, 180 subtests passed in
  415.81 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found no renderer, runtime, or Tinygrad-core change. Every runtime-dependent isclose operation and
bool conversion executes on DPU EW; Python only constructs the static test matrix and checks exact bool output. There is
no host tensor-value arithmetic, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, or tolerance relaxation.

---

## 2026-08-09 — native MaxPool2D spatial-index output

The complete upstream `test_max_pool2d_return_indices` method now runs in the Rockchip census. Its seven cases cover
batch/channel planes, dilation, padding, ceil-mode tail padding, a 156-lane global window, first-index ties, and
overlapping windows. The two explicit integer-literal fixtures are converted to FP16 while constructing the test input,
which preserves their upstream values and tie semantics while respecting the NPU's FP16 input contract; expected and
device-produced indices remain exact INT32.

Tinygrad lowers each unrolled index graph to equality between every FP16 pool candidate and the already-produced pooled
maximum, multiplies each equality by a descending spatial coordinate, MAX-reduces the candidates, and subtracts from
the spatial size. The strict Rockchip matcher proves that exact graph, evaluates only its compile-time address/gate and
coordinate expressions, and emits the existing raw gathers plus DPU equality/selection and native INT32-output stages.
Invalid padded lanes carry FP16 `-inf` and zero selection weight, so they cannot win a valid window. The global case's
single-register loop is recognized separately and lowered to the same immutable image. Spatial coordinates are capped
at 2048 so every integer represented during DPU FP16 selection is exact.

Historical commits `aa01f775e`, `d72bcc3f0`, `76c31806e`, and `cc3b7b6c6` established the candidate-address and
first-tie algorithm, but also contained a NumPy ArgMax fallback, NumPy unpool scatter, LUT truncation, and thousands of
tiny reset-heavy tasks; none of those paths were ported. Searches in `~/npu` and `~/rk3588` found the DPU RDMA
`UNPOOLING_EN` and PPU pooling register definitions and operator declarations, but no proven returned-index register
contract or compact EW implementation. The current implementation therefore stays on the already-proven DPU EW/image
ABI and adds no runtime semantic interpreter.

- Focused seven-case upstream method: **1 passed in 6.15 s**, sequentially.
- Complete MaxPool class: **12 passed, 1 skipped, 33 subtests passed in 10.04 s**, sequentially.
- Shared cumulative-extrema, arg-extrema, and sort-index regression: **22 passed, 16 subtests passed in 152.34 s**.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **361 passed, 10 skipped, 180 subtests passed in
  416.48 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused method, shared regressions, and
  complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,083/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no code that reads or interprets tensor values. NumPy
is used only for compile-time symbolic index evaluation, raw gather-map construction already used throughout the
renderer, and FP16 test-fixture creation. All input-dependent equality, tie selection, MAX reduction, and INT32
conversion execute on the NPU. There is no host ArgMax, host scatter, LUT, CMAC, CNA, PPU fallback, tolerance
relaxation, or external non-FP16 input.

---

## 2026-08-09 — bounded native MaxUnpool scatter

The small finite MaxUnpool case from upstream `test_max_unpool2d` now runs as a dedicated Rockchip regression: a
`(1,3,7,6)` FP16 tensor is MaxPool2D'd with a 2x2 kernel, then unpooled to the explicitly requested `(7,6)` spatial
shape. The kernel covers three planes, nine pooled candidates per plane, dynamic INT32 indices, and 126 FP16 output
lanes. The complete pool/index/unpool pipeline passes in 5.98 seconds.

The strict renderer matcher recognizes only the canonical Tinygrad one-hot scatter: each term compares a dynamic INT32
index with the compile-time output-plane coordinate, selects the FP16 pooled lane on equality, and the terms are summed.
It proves that the INT32 and FP16 loads share each candidate offset, that every output lane sees exactly one complete
pooled plane, and that coordinates are the compact spatial order. Raw gathers stripe indices and values into a bounded
matrix; DPU INT32-input conversion, equality masks, FP16 selection, and ADD reduction produce the output. The image has
27 raw gathers and 15 logical EW stages. No runtime semantic path was added.

The full upstream method is not marked complete yet. Its 25-candidate `(8,3,30,30)` case would repeat 600 indices into
540,000 lanes before four-lane INT32 conversion, expanding to 135,000 conversion tasks. It needs an image phase that
converts the compact index vector once and stripes the resulting FP16 lanes afterward. Its 80-candidate `(8,3,50,50)`
case additionally produces spatial indices through 2499, beyond the exact-integer range of the current FP16 coordinate
selector, and lowers the scatter as a loop. The infinity/NaN regression also needs representation-level selection,
because ordinary `value * 0` turns an unselected infinity into NaN. These cases remain excluded rather than being run
through the old branch's NumPy scatter or an inaccurate/unsafe expansion.

Historical commit `76c31806e` supplied useful graph and plane-layout evidence, but its implementation included host
packing/compaction helpers and an optional NumPy scatter. The RKNN operator guide lists MaxUnpool under CPU operators;
`~/npu` and `~/rk3588` expose an RDMA `UNPOOLING_EN` bit but no verified dynamic-index data contract. None of those
unproven or CPU paths were ported.

- Focused bounded MaxUnpool pipeline: **1 passed in 5.98 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **362 passed, 10 skipped, 180 subtests passed in
  418.57 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after focused and complete runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,142/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no host read of indices or pooled values. NumPy is used
only by the pre-existing compile-time static-index machinery and test oracle/fixture code. All dynamic INT32 conversion,
comparison, mask selection, and accumulation execute on DPU EW. There is no host scatter, host ArgMax, LUT, CMAC, CNA,
PPU fallback, tolerance relaxation, or external non-FP16 input.

---

## 2026-08-09 — compact-index phase for padded MaxUnpool

The second finite case from upstream `test_max_unpool2d` now runs natively: `(8,3,30,30)` FP16 input, 3x3 MaxPool
with stride `(6,7)` and padding 1, followed by MaxUnpool to explicit `(30,30)` spatial output. It covers 24 planes,
25 pooled candidates per plane, 600 dynamic INT32 indices, and 21,600 FP16 output lanes. The standalone case passes in
14.23 seconds; both admitted MaxUnpool cases pass together in 14.31 seconds, so neither approaches the 30-second limit.

RKImage version 26 adds one explicit mid-image raw-gather phase. The unpool program first converts the compact 600-lane
INT32 index tensor to FP16 on DPU exactly once. After that blocking phase, the runtime synchronizes the produced scratch,
stripes its FP16 representations into the compile-time candidate matrix, and resumes DPU equality/selection/reduction.
This reduces the padded case from a projected 540,000 converted lanes / 135,000 four-lane conversion tasks to only 600
converted lanes / 150 tasks. The same path improves the bounded case from 5.98 to 3.84 seconds.

The phase is encoded explicitly with mid-gather count and EW split index, validated during image decode, and executed by
the same synchronized raw-gather helper as terminal gathers. It does not interpret an index value or choose a
destination on the host: all gather offsets are proven statically from the one-hot graph, while dynamic index conversion,
equality, mask multiplication, and accumulation remain DPU operations. Existing images use zero mid-gathers and retain
their prior execution order; the complete census verifies both cached-style ordinary programs and the new split program.

- Focused bounded pipeline after compaction: **1 passed in 3.84 s**, sequentially.
- Focused 25-candidate padded pipeline: **1 passed in 14.23 s**, sequentially.
- Complete MaxUnpool class: **2 passed in 14.31 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **363 passed, 10 skipped, 180 subtests passed in
  445.10 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after focused and complete runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,147/280 executable lines**.

The CPU-cheat audit found only representation-preserving host movement between two blocking NPU phases, matching the
backend's existing raw pre/post gather contract. Neither renderer nor runtime reads or branches on an index or pooled
value. There is no host scatter, host ArgMax, numeric conversion, LUT, CMAC, CNA, PPU fallback, tolerance relaxation,
or external non-FP16 input.

---

## 2026-08-09 — exact wide MaxPool spatial indices

Returned MaxPool indices now remain exact beyond FP16's consecutive-integer range. A deterministic 50x50 regression
uses a 5x5 window with stride `(6,5)` and selects spatial index 2349; it passes exactly in 3.79 seconds. The complete
MaxPool class passes 13 tests plus 33 subtests, with only the explicit integer-input case skipped, in 10.98 seconds.

Historical commit `76c31806e` established the representation-safe design: select base-256 pieces on DPU and assemble
the final INT32 output by moving raw bytes. The current RKImage implementation preserves first-tie semantics without a
sequential candidate chain. Compile-time priorities rank valid coordinates in each output lane. DPU equality masks
select priority-tagged four-bit digits whose maximum remains within FP16's exact range; subtracting the selected
priority recovers each digit, pairs of digits form exact bytes, and native DPU conversion produces INT32 byte values.
The post-gather writes only those raw byte representations into the zeroed INT32 output. The digit radix and admitted
window bound are derived from FP16's 11 explicit precision bits rather than an arbitrary task or command-buffer cap.

`~/npu` and the RKNN guide contain no native returned-index contract, and MaxUnpool is documented as a CPU operator.
`~/rk3588` contains the upstream tests and the older byte-assembly proof but no simpler verified DPU instruction for
wide dynamic indices. No PPU, CMAC, LUT, or host ArgMax implementation was imported.

- Focused exact-wide returned-index regression: **1 passed in 3.79 s**, sequentially.
- Complete MaxPool class: **13 passed, 1 skipped, 33 subtests passed in 10.98 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **364 passed, 10 skipped, 180 subtests passed in
  429.82 s** pytest time / **454.76 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,215/281 executable lines**.

The CPU-cheat audit found only compile-time coordinate/rank construction and raw representation movement in generic
gathers. Runtime never reads, compares, converts, or branches on a tensor value: equality, priority selection, digit
reconstruction, and FP16-to-INT32 conversion are all submitted to DPU EW. The next MaxUnpool case is not included in
this milestone: its 80-candidate, 2350-output-plane consumer still needs exact byte-wise INT32 comparison. A diagnostic
run rejected that consumer after 36.55 seconds, showing that wide producer compilation/conversion also needs profiling
before the full pipeline can meet the 30-second target.

---

## 2026-08-09 — exact wide MaxUnpool scatter

Finite MaxUnpool now supports dynamic spatial indices beyond FP16's consecutive-integer range. A direct regression
scatters distinct values to indices 2049 and 2499 in seven independent 50x50 planes and checks the complete FP16 output
bit-exactly. The first wide upstream pipeline also passes: `(8,3,50,50)` FP16 input, 5x5 MaxPool with stride `(6,5)`,
80 pooled candidates per plane, 1,920 dynamic INT32 indices, and 60,000 unpooled FP16 output lanes.

The wide image compares the raw little-endian bytes of each dynamic INT32 index. Generic representation-preserving
gathers expose each compact byte lane, DPU INT32-input conversion converts byte values to exact FP16 integers once, and
the existing mid-image gather phase stripes those lanes over the candidate matrix. DPU SUB/ABS/MIN/inversion produces
one exact equality mask per required base-256 byte; DPU multiplication combines the masks, selects the pooled values,
and ADD-reduces the candidates. The loop matcher proves the canonical Tinygrad MaxUnpool accumulator, shared value/index
addresses, plane and candidate dimensions, output-coordinate order, and affine source layout before emitting this image.
It stores affine gather plans instead of expanding millions of repeated offsets in the compiled image.

Profiling the complete pool/index/unpool pipeline attributed 3.61 seconds to rendering (0.01 pool values, 1.44 pool
indices, 2.16 unpool), 15.46 seconds to DPU EW execution, and 15.16 seconds to INT32 conversion, including 15.06 seconds
in the required conversion-boundary resets. Removing the page-derived conversion batch limit or its inter-batch reset
caused reproducible driver timeouts and was reverted; the final runtime is unchanged. With affine plans and conservative
conversion batches, the standalone wide method passes in **23.46 s** pytest time / **25.57 s** process wall time.

- Direct exact indices 2049/2499 regression: pass.
- Complete MaxUnpool class: **4 passed in 34.97 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **366 passed, 10 skipped, 180 subtests passed in
  454.99 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,323/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no host interpretation of an index or pooled value.
Python constructs only static/affine gather geometry and moves raw representations; all dynamic conversion, byte-wise
comparison, selection, and reduction execute on DPU EW. There is no host scatter, host ArgMax, LUT, CMAC, CNA, PPU
fallback, tolerance relaxation, or external non-FP16 input.

---

## 2026-08-09 — equality-row matcher cleanup

Occurrence-count and stable sort-index lowering now share one bounded equality-row parser. It identifies the common
load and ordered candidate loads, validates their dtype and source buffers, evaluates their compile-time gather maps,
and rejects every out-of-range lane. The two IR matchers remain separate because occurrence gates and weighted
value/count conjunctions are different graphs; `_lower_sort_compare` is unchanged because its static MIN/MAX algorithm
does not overlap equality-mask emission. Existing `_ew_eq_mask`, `_stripe_layout`, and `_stripe_gathers` continue to
provide the shared DPU emission and aligned-matrix packing layers.

- Sort-index and TopK regression: **12 passed, 8 subtests passed in 108.12 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Repository-wide Ruff and Tinygrad mypy: pass.
- `sz.py`: renderer/runtime **2,310/281 executable lines**, down from **2,323/281**.

This is a renderer-only structural refactor. It adds no tensor-value access, numeric host operation, runtime behavior,
new tolerance, LUT, CMAC, or fallback path; the generated gather and DPU EW algorithms are unchanged.

---

## 2026-08-09 — representation-safe non-finite MaxUnpool

The remaining FP16 `test_max_unpool2d_inf` pipeline now preserves its exact non-finite output representation. Its fused
schedule stores an internal negative pool-selection weight and applies the spatial-size correction in the MaxUnpool
consumer. The unrolled pool-index matcher now recognizes and proves that form. Because FP16 subtraction and
multiplication are not representation-safe for infinities (`inf-inf` and `inf*0` produce NaN), this narrow path compares
the two raw FP16 bytes of every candidate with the pooled extremum instead.

Generic raw gathers place each byte in the low byte of a zeroed INT32 lane without interpreting it. Native DPU
INT32-input conversion turns the 0..255 byte values into exactly representable FP16 numbers; DPU byte equality selects
the negative first-tie weight. The single-candidate MaxUnpool consumer uses the same mechanism for its pooled FP16
value, multiplies each finite byte value by the DPU-computed equality mask, converts the selected bytes to INT32 on DPU,
and writes their low bytes into the FP16 output representation. Thus selected `+inf`, `-inf`, and NaN bits survive while
every unselected lane becomes exact positive zero. A direct four-plane regression covers `+inf`, `-inf`, NaN, and 3.5;
the upstream pool-to-unpool case verifies the fused internal-index correction.

Historical commit `76c31806e` supplied the original raw-representation selection proof, but its optional NumPy scatter
and operation-specific host packing runtime were not ported. The current implementation uses only the existing generic
RKImage gather phases and DPU EW conversions. `~/npu` documents MaxUnpool as a CPU operator and `~/rk3588` contains no
simpler verified native dynamic-index contract.

Wall-time profiling of a fresh standalone pipeline measured 1.55 seconds of graph construction, 0.012 seconds of
renderer work, and 1.52 seconds for realization plus copyout. The three NPU programs consumed 1.28 seconds, issuing
29 ioctls / 34 tasks; ten INT32 conversions consumed 1.06 seconds and twelve required resets consumed 1.27 seconds
(these timings overlap the program total). A first cold pytest cache miss reached 28.30 seconds, while the cached rerun
reported a 2.12-second test body and 4.64 seconds total. The hardware work itself is therefore well below 30 seconds.

- Direct exact-bit non-finite MaxUnpool and upstream infinity pipeline: pass.
- Complete MaxPool + MaxUnpool regression: **19 passed, 1 skipped, 33 subtests passed in 45.88 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **368 passed, 10 skipped, 180 subtests passed in
  482.49 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,416/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host code only constructs static gather maps and moves raw
bytes without comparing, converting, branching on, or selecting a tensor value. All runtime-dependent equality,
first-tie choice, byte masking, and numeric conversion execute on DPU EW. There is no host scatter, host ArgMax, LUT,
CMAC, CNA, PPU fallback, tolerance relaxation, or external floating-point input wider than FP16.

---

## 2026-08-09 — exact full-range INT32 one-hot equality

The complete upstream `TestOps.test_one_hot` now runs on Rockchip. Its scheduled integer graph is the strict form
`WHERE(dynamic_index != static_class_coordinate, 0, 1)`. The matcher proves that form, the INT32 input buffer and its
static bounds, every input gather offset, every compile-time coordinate, and the signed INT32 coordinate range before
emitting an image.

Historical commit `32cb1cd67` on `rockchip-2607` supplied the original DPU one-hot proof. That version expanded the
dynamic INT32 indices, converted whole values to FP16, and deliberately capped the class extent at 2,048. Its notes
also recorded that native INT32 comparison mode produced invalid masks and identified byte-limb comparison as the
required full-range solution. The current implementation completes that solution: representation-preserving gathers
place each of the four little-endian input bytes in a zeroed INT32 lane, DPU conversion turns each byte into an exact
FP16 integer in 0..255, and a mid-image gather expands those compact byte vectors over the output layout. Four DPU
SUB/ABS/MIN/inversion equality masks are multiplied and converted by DPU to the final INT32 output. Class 2,049 in a
2,050-class output therefore passes exactly instead of depending on FP16 whole-integer precision.

The RKNN v2.3.2 operator guide lists OneHot in its CPU-operator chapter, while `~/npu/include/old/rknn_ops.md` marks it
unsupported. `~/rk3588` contains the upstream test and runtime strings but no verified native DPU OneHot instruction.
No CPU operator or undocumented instruction was imported.

- Complete upstream one-hot case, full-byte out-of-range regression, and class-2,049 regression: **3 passed in 9.31 s**,
  sequentially. The largest focused test passed in **7.35 s**.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **371 passed, 10 skipped, 180 subtests passed in
  460.02 s** pytest time / **485.79 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,467/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The host evaluates only static class coordinates and
moves dynamic bytes by statically proven addresses; it never numerically reads, compares, branches on, or selects from
an input index. All dynamic byte conversion, equality, mask conjunction, and INT32 output conversion execute on DPU
EW. There is no host OneHot, scatter, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than
FP16.

---

## 2026-08-09 — exact dynamic INT32 gather on DPU EW

The unchanged upstream `TestOps.test_gather` now passes all positive and negative axis aliases, seven three-dimensional
gathers, both shape-error cases, a one-dimensional gather, and its `-inf` representation case. A direct regression also
uses dynamic indices `0`, `1`, `2`, `256`, `65,536`, `2**24`, and `-1` against `+inf`, `-inf`, and NaN source lanes;
valid results match bit-for-bit and every out-of-range high-byte alias remains zero.

Tinygrad lowers Gather to one bounds-masked FP16 load whose source address contains a dynamic INT32 load. The strict
matcher proves the zero default, the exact nonnegative/upper-bound predicate, both statically sized source buffers,
the dynamic-index gather map, and every candidate-substituted source address. It then exposes the four raw index bytes
and both raw FP16 source bytes with representation-preserving gathers. DPU INT32-input conversion turns each byte into
an exact FP16 integer, four DPU equality masks select the bounded candidate, and DPU ADD reductions select each FP16
byte independently. DPU INT32-output conversion plus raw post-gathers reconstruct the FP16 result. This preserves NaN
and infinity representations without unsafe `inf*0` masking.

Historical commit `f62827791` implemented gather and fancy indexing with a typed NumPy evaluator, including dynamic
negative-index preprocessing and masked reductions on the host. That implementation violates the current no-CPU-cheat
contract and was not ported. The RKNN v2.3.2 guide lists Gather in its CPU-operator chapter, the older operator table
marks `aten::gather` unsupported, and `~/rk3588` contains no verified native DPU Gather instruction. The current path
therefore composes only proven DPU EW and raw movement primitives.

- Complete unchanged upstream gather: **1 passed in 23.47 s**, sequentially.
- Upstream and exact raw-byte regression together: **2 passed in 24.73 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **373 passed, 10 skipped, 180 subtests passed in
  479.44 s** pytest time / **505.58 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,555/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host evaluation substitutes only compile-time candidate
coordinates into address expressions and validates their static bounds. Runtime moves bytes at those fixed addresses;
it never interprets an index or source value, chooses a candidate, or performs a numeric gather. Dynamic comparison,
selection, reduction, and numeric conversion execute on DPU EW. There is no host Gather, NumPy evaluator, LUT, CMAC,
CNA, PPU fallback, tolerance beyond the established FP16 test tolerance, or floating input wider than FP16.

---

## 2026-08-09 — shared exact INT32 equality-matrix emission

OneHot and dynamic Gather now share one immutable `RKByteEquality` image fragment. It owns raw four-byte extraction,
DPU byte conversion, compact-to-striped mid-gathers, compile-time coordinate-byte packing, and the four DPU equality
masks plus conjunction. The operation-specific matchers remain separate: OneHot still proves its integer WHERE graph,
while Gather still proves its bounds-masked dynamic address. Their terminal emitters continue to differ because OneHot
writes one INT32 mask and Gather uses that mask to select and reconstruct two raw FP16 bytes.

This is the same parser-versus-emitter separation used by the existing `RKLoopReduction` and gather-plan helpers. It
removes duplicated construction without creating a combined IR mega-matcher or changing runtime behavior. `sz.py`
reports renderer/runtime **2,550/281 executable lines**, down from **2,555/281**. The small net reduction includes the
new typed fragment record; roughly sixty formerly duplicated construction lines now have one implementation.

- Complete OneHot plus Gather regression: **5 passed in 31.59 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Renderer mypy, Ruff, and `git diff --check`: pass.

The refactor adds no tensor-value access, runtime operation, tolerance change, LUT, CMAC, or fallback. Exact dynamic
comparison remains DPU EW work and host code remains limited to compile-time geometry plus raw representation movement.

---

## 2026-08-09 — exact negative single-index fancy indexing

The upstream `TestOps.test_fancy_indexing_inf` and a direct negative-index regression now pass bit-exactly. The direct
case selects `[-1, -2, -3]` from `[+inf, -inf, NaN]`, proving both Tinygrad's negative-index normalization and raw
non-finite representation preservation.

Tinygrad expresses a single fancy index as `WHERE(index < 0, index + extent, index)` inside the bounds-masked source
address. The dynamic-gather matcher now recognizes only that exact normalization. It emits two compile-time coordinate
spellings for every bounded source candidate: `0..extent-1` and `-extent..-1`. Each spelling maps to the same statically
proved source address after candidate substitution, while the shared four-byte DPU equality fragment decides which
runtime spelling matches. Invalid indices match no row and retain the masked-load zero default. No dynamic value is
normalized or bounds-checked on the host.

Historical commit `f62827791` covered broad fancy indexing by executing negative-index preprocessing and multi-index
masked reductions in a typed NumPy evaluator. That CPU implementation remains intentionally unported. This milestone
admits only one dynamic INT32 index and uses the DPU/raw-byte path proven by Gather; multi-index broadcasting and fused
candidate reductions remain separate future work.

- Upstream infinity and exact negative-index regression: **2 passed in 6.04 s**, sequentially.
- Complete OneHot, Gather, and single-index fancy-index regression: **7 passed in 34.28 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **375 passed, 10 skipped, 180 subtests passed in
  482.85 s** pytest time / **510.13 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,563/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The host substitutes candidate constants only to prove
static source addresses; raw runtime indices are never read, normalized, compared, or used to choose a source on CPU.
All dynamic normalization equivalence, equality, byte selection, and reconstruction execute through DPU EW. There is
no NumPy evaluator, host gather, LUT, CMAC, CNA, PPU fallback, tolerance beyond the established FP16 test tolerance, or
floating input wider than FP16.

---

## 2026-08-09 — exact multi-index FP16 fancy indexing

The unchanged upstream `TestOps.test_slice_fancy_indexing_with_tensors` now passes all four two-axis broadcast forms,
including its signed case with independent negative indices on both axes. A direct representation regression selects
`+inf`, `-inf`, NaN, `1.0`, `-0.0`, and `+0.0` through two dynamic INT32 index tensors and verifies every FP16 bit.

Tinygrad expresses each dynamic axis as `WHERE(index < 0, index + extent, index)` and combines the normalized values in
one bounds-masked source address. The matcher proves every normalized root, its canonical lower/upper gate, the static
input sizes and broadcast maps, and the complete candidate-substituted source address. It enumerates only compile-time
coordinate tuples; runtime index values remain opaque. The shared equality fragment now emits one exact four-byte DPU
mask per dynamic axis and conjoins those masks on DPU before selecting the two raw FP16 source bytes.

Generalizing the image exposed one stale constant: the mid-gather split was fixed at six operations, exactly the four
byte conversions plus two source-byte conversions needed by one index. With two indices it ran before all eight index
byte conversions completed. The split is now derived as `len(equality.pre_ops) + len(raw_value)`, so it follows the
actual number of dynamic axes instead of encoding the single-index layout.

Historical commit `f62827791` handled broad fancy indexing through a typed NumPy evaluator that read and normalized
dynamic indices on the host. That implementation remains unported. The current runtime only moves raw bytes using
compile-time-proved maps; all dynamic byte conversion, equality, multi-axis conjunction, selection, reduction, and
numeric conversion execute through DPU EW. The local RKNN references still expose no verified native DPU Gather/fancy
index instruction.

- Complete fancy-index class: **4 passed in 24.60 s**, sequentially.
- Shared OneHot, Gather, and fancy-index regression: **9 passed in 52.75 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **377 passed, 10 skipped, 180 subtests passed in
  501.56 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the complete census and after the final
  candidate-budget guard.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,621/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host evaluation rejects `PARAM` values and substitutes
only compile-time candidates to derive and validate static addresses. Runtime gathers preserve opaque 1/2/4-byte
representations and never interpret an index or floating value. There is no host fancy indexing, dynamic NumPy
evaluation, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded dynamic INT32 Scatter with exact FP16 representations

Rockchip Scatter now accepts one external INT32 index buffer for bounded FP16 tensor-source updates. The exact-bit
regression covers unique destinations and an overlapping destination with last-source-wins semantics. Its base and
source lanes include `+inf`, `-inf`, signed zero, and distinct NaN payloads, and every result is compared as raw
`uint16` rather than through a relaxed floating tolerance.

Tinygrad lowers this Scatter form to a nested WHERE selector. The strict matcher proves one direct dynamic-index load
per candidate, consecutive static candidate positions, static output coordinates, statically bounded base/source
addresses, and the complete last-wins selector truth table. The shared four-byte INT32 equality fragment now accepts a
different statically proved index offset row for each candidate. DPU masks are traversed in reverse candidate order to
construct mutually exclusive effective masks, raw FP16 source/base bytes are selected independently, and DPU
INT32-output conversion plus raw post-gathers reconstruct the exact FP16 output.

Historical direct-Scatter commit `16ea0c339` routed this family through a typed NumPy evaluator that read and computed
tensor values on the host, so it was not ported. `~/npu/include/old/rknn_ops.md` marks ScatterElements unsupported;
`~/rk3588` contains the upstream tests and RKNN runtime strings but no verified native DPU Scatter instruction. This
milestone therefore composes only the already proven DPU EW equality, mask, reduction, and representation-movement
primitives. It intentionally admits only one index buffer with one to eight candidates; wider/general N-D Scatter
graphs remain unsupported until they have a bounded hardware plan.

- Exact dynamic Scatter: **1 passed, 2 subtests passed in 5.43 s**, sequentially.
- Shared OneHot, Gather, fancy-index, and Scatter regression: **18 passed, 2 subtests passed in 56.01 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **378 passed, 9 skipped, 182 subtests passed in
  505.83 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,747/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The parser evaluates only static coordinates and an
abstract Boolean selector truth table; it never reads a runtime index, base, or source value. Runtime NumPy code only
moves opaque bytes at compile-time-proved addresses. Dynamic INT32 conversion and equality, last-wins choice, FP16-byte
selection, and reductions all execute on DPU EW. There is no host Scatter, typed tensor evaluator, LUT, CMAC, CNA, PPU
fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded exact FP16 Copysign and forward-only timing census

The complete seven-by-seven upstream Copysign value matrix now passes bit-exactly for finite values, signed zeros,
both infinities, and NaN. The Rockchip regression vectorizes the same 49 ordered pairs into one realization and checks
the raw `uint16` result, making it stricter and much faster than 49 scalar helper invocations.

Tinygrad's signed-zero-aware Copysign graph is recognized before the generic numeric sign rewrite. The strict matcher
proves its `WHERE(sign<0 OR 1/sign<0, -abs(magnitude), abs(magnitude))` structure, direct FP16 loads, static source
bounds, and output gather maps. Raw movement packs magnitude-low, magnitude-high, and sign-high bytes into one striped
matrix. DPU INT32-input conversion maps bytes to exact integers in 0..255, one combined DPU threshold comparison builds
both high-bit masks, and DPU arithmetic computes `magnitude_high - 128*magnitude_sign + 128*source_sign`. The unchanged
low byte and replaced high byte are converted and gathered back as raw FP16 output.

Historical exact-Copysign milestone `751a108e2` extracted tensor sign bits with NumPy in the runtime and was not ported.
`~/npu` and `~/rk3588` expose the ordinary DPU ABS/PReLU register primitives and upstream tests, but no native Copysign
or bitwise FP16 instruction. The current path therefore uses only raw representation movement and proven DPU EW. To
bound the reset-heavy four-lane conversion cost without an arbitrary element count, the exact path is admitted only
when its three-row conversion-tile arena fits one system page; larger finite tensors retain the established numeric
fast path.

Wall-time work removed two successive bottlenecks. The unchanged 49-scalar upstream method initially took **51.33 s**
at ten submits per pair. Packing all required bytes into shared matrices reduced it to **35.37 s**. Vectorizing the
same Cartesian coverage reduced the exact Rockchip test to **3.60 s** pytest time; exact plus the existing broad
Copysign test passed in **5.54 s**, and the complete sign/softsign class passed **7 tests in 7.25 s**.

The duration-enabled full census before removing two backward-only comparison methods passed **379 tests, 9 skipped,
182 subtests in 505.96 s**. Because this backend is explicitly forward-only, `test_cmp_ne_backwards` and
`test_cmp_lt_backwards` were then removed from the Rockchip class; collection now contains 386 nodes instead of 388.
The slowest forward calls were simple CumMin 26.58 s, Gather 20.79 s, wide MaxUnpool 19.98 s, simple CumMax 19.06 s,
and tensor fancy indexing 14.52 s.

An instrumented isolated simple CumMin run took **27.49 s**: renderer compilation 8.56 s, nine NPU program calls
8.43 s, and scheduling/PyTorch/reference overhead 10.31 s. Inside the device calls, 73 soft resets consumed **7.75 s**,
73 submit ioctls consumed only **0.023 s**, and gather/sync/command preparation consumed 0.65 s. BO initialization was
0.19 s and copyout 0.001 s. Future CumMin optimization should therefore reduce reset-separated comparison/conversion
stages; larger command buffers or faster arithmetic cannot materially improve this profile.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the correctness census and after both full
  census/profile runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,813/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host code recognizes static IR, proves addresses, and
moves opaque bytes; it never reads a sign, classifies a value, or performs Copysign. Byte conversion, high-bit masks,
sign removal/insertion, and output conversion execute on DPU EW. There is no NumPy tensor evaluator, host Copysign,
LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded dynamic scalar Scatter add/multiply

The unchanged forward-only upstream `test_scatter_add` and `test_scatter_mul` now pass on Rockchip for external INT32
indices, including their infinity and NaN cases. A deterministic regression also covers repeated destinations with a
finite scalar, proving that add counts every hit and multiply applies every factor instead of collapsing the mask to a
single Boolean.

Tinygrad unrolls this Scatter family into one guarded equality term per candidate plus the FP16 base load. The strict
matcher proves one dynamic INT32 load per term, its static coordinate expression, its bounds-gated address, the exact
neutral result when unequal, one common scalar FP16 representation when equal, and the statically bounded base map.
The shared four-byte equality fragment builds every runtime mask on DPU. A compile-time validity matrix suppresses
lanes outside the index tensor's shape; DPU ADD or MUL then reduces the candidate rows and combines the result with the
base.

Non-finite scalars require avoiding `0*inf` on no-hit lanes. Conditional infinity is generated entirely on DPU from
finite inputs as `hit * 65504 * 2`. Conditional NaN uses the RK3588's verified `inf * 0` behavior as
`conditional_inf * (1-hit)`, producing NaN only when selected and exact zero otherwise. Add uses a reduced hit count;
multiply constructs one factor per finite candidate, while its infinity/NaN cases use the equivalent any-hit factor.
No host branch depends on a runtime index or FP16 value.

Historical milestone `1ff6ebc9d` admitted FP16 ScatterReduce by routing the complete graph through a typed host
elementwise evaluator, so that implementation was not ported. `~/npu/include/old/rknn_ops.md` marks ScatterElements
unsupported, and `~/rk3588` contains the upstream Scatter tests but no verified native DPU Scatter instruction. The
current implementation instead composes only the existing exact INT32 equality, static gather planning, and DPU EW
arithmetic primitives; it uses neither LUT nor CMAC.

- Repeated-index finite regression: **1 passed, 2 subtests passed in 3.93 s**, sequentially.
- Unchanged upstream Scatter add: **1 passed in 4.72 s**; Scatter multiply: **1 passed in 4.77 s**, sequentially.
- Complete Scatter class: **12 passed, 4 subtests passed in 10.91 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **380 passed, 9 skipped, 184 subtests passed in
  508.16 s**, sequentially. Collection contains 389 forward-only nodes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused hardware tests.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,923/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer encodes only compile-time FP16 constants;
static IR evaluation proves output coordinates, gates, and addresses and rejects dynamic parameters. Runtime gathers
move opaque representations according to those plans. Dynamic INT32 conversion, byte equality, validity masking,
hit/factor reduction, infinity/NaN construction, and final FP16 arithmetic all execute on DPU EW. There is no typed
tensor evaluator, host Scatter arithmetic, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider
than FP16.

---

## 2026-08-09 — flipped-eye static movement regression

The unchanged forward-only `TestOps.test_flip_eye_crash` is now part of the Rockchip movement census and passes
`eye(10) @ flip(eye(10), axis=0)` without reproducing its historical crash. Current Rockchip has no CMAC renderer or
runtime path: static Eye/Flip layouts are proved as raw gathers and the small contraction uses the established DPU EW
reduction.

Historical milestone `d134eb1b8` had already recorded this exact method passing without a backend change. `~/npu`
marks native `aten::flip` unsupported, while `~/rk3588` contains the same upstream method but no independent native
Flip instruction example. This confirms that the existing static gather path is the appropriate implementation rather
than adding a special opcode or host movement evaluator.

The adjacent upstream `test_mulacc_with_zero_strides` was investigated but deliberately not admitted. Its historical
fix in `ad990be16` is explicitly an FP32 CMAC plus compensated-DPU implementation, outside this branch's FP16-input,
no-CMAC scope; today its first constant-only case also reaches an unrelated CPU vector-cast compiler error. No
Tinygrad-core workaround or narrower replacement test was added.

- Unchanged flipped-eye method: **1 passed in 3.09 s**, sequentially.
- Complete movement class: **31 passed, 1 skipped in 6.03 s**, sequentially.
- The immediately preceding complete Rockchip census passed **380 tests, 9 skipped, 184 subtests in 508.16 s**; this
  test-only milestone adds the one focused passing node, for 390 collected nodes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the movement-class run.
- Repository-wide Tinygrad mypy and focused Ruff: pass. `git diff --check`: pass. `sz.py` remains renderer/runtime
  **2,923/281 executable lines**.

There is no backend/runtime/Tinygrad-core change and therefore no new CPU arithmetic or tensor-value access. The test
uses the existing compile-time address plan and DPU EW path with no LUT, CMAC, CNA, PPU fallback, tolerance beyond the
established FP16 cap, or floating input wider than FP16.

---

## 2026-08-09 — WIP bounded fixed FP16 masked select

Fixed-size `x.masked_select(x > 0, size=..., fill_value=...)` is lowered as four strict DPU-EW programs: an
IEEE-correct positive prefix count, an INT32 occurrence histogram, an INT32 histogram prefix, and an exact raw-FP16
guarded gather/fill. The matcher admits only a source-derived positive mask, proves every static prefix and address,
and bounds counts to the exact FP16 integer range. The exact regression includes NaN, both infinities, signed zero,
the smallest positive subnormal, and raw-bit checking of selected and fill values.

The initial correct path used 94 submits and 220 tasks for a representative six-input/four-output case. Reusing
whole-vector INT32 conversions and compact raw-source bytes reduced it to 50 submits and 87 tasks. Before the compact
raw-source change, the longest pad-and-truncate test took 7.39 s: 6.75 s was 64 reset ioctls, 0.024 s was 112 actual
submit ioctls, and about 0.61 s was host scheduling, rendering, marshalling, and reference work. Reset separation is
therefore the remaining wall-time bottleneck.

A reset-free ReLU/MIN validity experiment was rejected after it caused an IOMMU read fault and a six-second NPU job
timeout; the renderer was restored to the previously passing comparison path and no further NPU command was issued on
that boot. At this point the compact raw-source form still required focused hardware revalidation; the following
Nonzero milestone records that successful validation.

The cleanup pass follows the other hardware backends' compact immutable-plan style: runtime code remains unchanged,
FP16 representation encoding and INT32 conversion-tile sizing are centralized, an unused scratch constant was
removed, and matchers return only data consumed by their emitters. Host-only rendering produces four valid immutable
RKImages for both finite and special-value cases. Repository-wide Tinygrad mypy and Ruff pass, as does
`git diff --check`; `sz.py` reports renderer/runtime **3,224/281 executable lines**.

The CPU-cheat audit finds changes only in the renderer, tests, and this progress record. Compile-time code inspects
static IR and encodes constants but never reads tensor buffers or evaluates a dynamic FP16/INT32 value. Runtime numeric
semantics, Tinygrad core, tolerances, and the external FP16 contract are unchanged. There is no typed host evaluator,
LUT, CMAC, CNA, or PPU fallback.

---

## 2026-08-09 — bounded fixed FP16 Nonzero

Fixed-size FP16 `nonzero(size=..., fill_value=...)` now passes for rank-one truncation and padding, rank-two coordinate
selection, signed zeros, finite negative values, NaN, infinity, empty inputs, and scalar output shape. Tinygrad lowers
this operation through four programs: a repeated FP16 nonzero prefix count, the existing exact INT32 occurrence
histogram, its INT32 prefix, and a guarded coordinate selector. The new final matcher proves the complete source
nonzero count, dynamic prefix-index buffer, every static coordinate row, every validity threshold, and an exactly
FP16-representable integer fill before emitting an image.

The count and final selection remain DPU computations. Native ABS plus comparison creates the nonzero mask, DPU ADD
reduces it, the shared four-byte INT32 equality plan selects the runtime prefix indices exactly, and DPU arithmetic
combines static coordinate rows with the dynamic validity mask before exact INT32 output conversion. An initial
hardware run returned all-zero coordinates because the new image omitted the shared equality plan's FP16 `1`
constant; adding that immutable image constant fixed the result without a runtime or tolerance change.

Historical nonzero milestones `ae5a4a6f6` and `add791a62` evaluated complete tensor expressions with NumPy in the
runtime and were not ported. `~/npu/include/old/rknn_ops.md` marks native NonZero unsupported, and `~/rk3588` has no
verified native NonZero instruction. The current implementation instead composes the already proven prefix,
byte-equality, gather-plan, and DPU EW primitives.

- Fixed Nonzero class: **4 passed in 11.44 s**, sequentially; every individual method passed under 30 seconds.
- Adjacent fixed MaskedSelect regressions: **2 passed**, individually in **8.94 s** and **5.65 s**. This closes the
  post-reboot validation left pending by the preceding WIP milestone.
- One rank-one truncate realization: input realization 83 ms, graph construction 9 ms, scheduling 131 ms, rendering
  224 ms, four NPU programs 1.93 s, and copyout 3 ms. It used **51 submit ioctls / 69 DPU tasks**; the exact final
  selector was the largest program at 962 ms.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before focused work, after the initial numerical
  failure, and after the complete focused run. No new RKNPU timeout or IOMMU fault appeared in the kernel journal.
- Repository-wide Tinygrad mypy and Ruff, host compilation of every Nonzero program, and `git diff --check`: pass.
  `sz.py`: renderer/runtime **3,331/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static IR evaluation only proves addresses, coordinate
tables, and selector truth tables after replacing abstract dynamic nodes with validation constants; it never reads a
runtime source, prefix, or index buffer. Runtime gathers move opaque representations according to those plans.
Dynamic nonzero classification, INT32 equality, coordinate selection, validity masking, and output conversion all
execute on DPU EW. There is no typed host tensor evaluator, host Nonzero arithmetic, LUT, CMAC, CNA, PPU fallback,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — multidimensional dynamic FP16 Scatter

Bounded last-wins Scatter with an external INT32 index tensor and FP16 tensor source now covers every axis of a
three-dimensional `(2,3,4)` tensor. The raw-bit regression includes infinities, NaNs, and signed zero, deliberately
maps every candidate along the scatter axis to the same destination, and checks the final FP16 representations
exactly. This proves multidimensional index addressing, source addressing, collisions, and last-wins ordering rather
than only the earlier one-dimensional special-value cases.

The existing strict Scatter matcher assumed each unrolled dynamic index load addressed one globally constant index
position. Multidimensional codegen instead emits one candidate load whose offset varies across the non-scatter axes.
The matcher now retains the complete per-output offset row, sorts candidate rows lexicographically, proves their union
covers every external index element, and proves candidate offsets are strictly increasing for every output
lane. The unchanged exact Scatter image then applies four-byte INT32 equality, raw FP16 byte selection, and DPU
last-wins masks to those proven rows.

Historical Scatter milestones `16ea0c339`, `262622ff0`, and `1ff6ebc9d` used typed NumPy tensor evaluators and were not
ported. `~/npu/include/old/rknn_ops.md` marks native ScatterElements unsupported, while `~/rk3588` contains the upstream
tests but no independent native Scatter instruction example. The implementation therefore extends only the existing
immutable DPU plan instead of adding a host fallback.

- Three-dimensional dynamic tensor-source Scatter: **1 passed, 3 subtests passed in 7.22 s**, sequentially, covering
  axes 0, 1, and 2.
- Complete Scatter class: **13 passed, 7 subtests passed in 15.33 s**, sequentially.
- One representative axis-1 realization: input realization 100 ms, graph construction 3 ms, scheduling 106 ms,
  rendering 120 ms, one NPU program 1.29 s, and copyout 3 ms; **49 submit ioctls / 135 DPU tasks**.
- A discarded six-axis stress form completed axes 0/1/2/-1 but timed out on the redundant `-2` alias after five
  reset-heavy programs in one process. The retained test removes normalized-axis duplicates; it and the full class
  pass from fresh processes. Vendor `elementwise.py` passed **60/60** immediately after the timeout and after the
  retained class, confirming the driver remained usable.
- Focused Ruff and mypy, `git diff --check`, and host rendering for positive and normalized axes: pass. `sz.py`:
  renderer/runtime **3,330/281 executable lines**, one renderer line smaller than the preceding milestone.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static IR evaluation proves only coordinate expressions,
buffer bounds, complete index coverage, and last-wins truth tables. It does not read an index, source, or base tensor.
Runtime gathers move opaque bytes; dynamic INT32 equality, collision masking, FP16 selection, and output construction
execute on DPU EW. There is no typed host Scatter, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating
input wider than FP16.

---

## 2026-08-09 — no-LUT FP16 round-to-even

The complete forward `test_round` group is now in the Rockchip census. Tinygrad's exact round-to-even graph is
recognized before its TRUNC/FLOOR/CEIL expansion and replaced with a DPU-only composition: native FLOOR obtains the
integer and fractional parts, native ABS and positive-mask stages detect `fraction > 0.5` and exact ties, the existing
native FLOOR/CEIL truncation composition determines parity, and DPU MUL/MAX/ADD applies the one-lane increment. This
avoids the historical ROUNDOFF LUT and the older typed host conversion fallback.

The first generic WHERE lowering passed ordinary ties but mapped infinities to NaN. A first specialized formula based
on `floor(x+0.5)` fixed nonfinite values but failed 1,025 FP16 encodings because the `+0.5` itself rounded before FLOOR,
including odd integers above 1,024. Computing `floor(x) + increment` from the fractional part instead is value-exact
over all 65,536 FP16 encodings. It preserves infinity signs and NaN locations; like the upstream value-level contract,
it does not promise raw NaN payloads or the sign bit of results numerically equal to zero.

The independent `~/rk3588/examples/elementwise_int.py` reference was also translated from its Rocket wrapper to this
machine's RKNPU ioctl ABI in a temporary probe. FP16 comparison to INT16 and INT16 ADD to INT32 both passed exactly.
Those modes were not added to the backend because a conservative implementation must retain the existing exact
byte/digit path for indices above signed INT16 range, so it would currently add machinery rather than remove it.

- Upstream `test_round`: **1 passed in 5.77 s**; adapted forward-only quantization expression: **1 passed in 3.59 s**.
- Exhaustive round regression: **all 65,536 FP16 encodings passed numerically**, **10 submit ioctls**, about **1.41 s**
  operator wall time and **4.10 s** under isolated pytest.
- Complete integral-rounding class: **7 passed in 7.20 s**, sequentially.
- Representative 45x35 round: input realization 96 ms, graph construction 1 ms, scheduling 44 ms, rendering 114 ms,
  DPU execution 642 ms, and copyout 3 ms; **7 submit ioctls**, 900 ms total.
- Rockchip collection is **400 tests**. Repository-wide Tinygrad mypy, Ruff, and `git diff --check` pass. `sz.py`:
  renderer/runtime **3,365/281 executable lines**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete focused run.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer only matches static IR structure and emits
DPU EW stages; it never reads or numerically evaluates a tensor buffer. There is no LUT, CMAC, CNA, PPU fallback,
typed host rounding, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — dimension-injected dynamic FP16 fancy indexing

The three forward cases from `TestOps.test_slice_fancy_indexing_dim_inject_and_collapse` now run on Rockchip. They
combine dynamic INT32 tensor indices with collapsed integer dimensions, inserted `None` dimensions, and an ellipsis.
The original upstream method exceeded the 30-second per-test policy when all three cases shared one item, so the exact
expressions are retained as three independently bounded Rockchip methods.

Tinygrad emits two graph forms for this group. One is a complete Cartesian ADD of masked source loads; the new strict
matcher proves every canonical negative-normalization root, every equality predicate, the unique complete coordinate
product, all source/index buffer bounds, and every candidate address. The other form keeps one bounds-masked dynamic
load and is handled by the existing multi-index matcher. Positive coordinate equality and its equivalent negative
spelling are now ORed per axis on DPU before the axis masks are conjoined, so a 5x6 selection needs 30 candidate rows
instead of duplicating all four positive/negative sign combinations into 120 rows.

The first implementation was correct but too slow. A representative first case realized in **48.735 s**: INT32
conversion consumed **48.162 s**, including **45.668 s in 434 soft resets**; its 521 submit ioctls consumed only
0.186 s. Raw source bytes had been duplicated over every candidate row before conversion. Converting each statically
reachable source lane once and then striping the converted bytes through a mid-image raw gather reduced realization to
**4.619 s**, conversion to **3.858 s**, resets to 36, submits to 123, and tasks from 6,099 to 519.

Output-byte reconstruction exposed the same issue in the 3,000-lane ellipsis case. The decoded
`~/rk3588/examples/elementwise_int.py` FP16-to-INT16 recipe was translated to the current RKNPU ioctl ABI and extended
from comparison masks to neutral MAX conversion. Exact probes passed for 3, 8, and 3,000 FP16 integer lanes spanning
0 through 255. RKImage v27 therefore records a native INT16 output stage. DPU converts the selected numeric byte lanes
to contiguous INT16 in full-width tasks; runtime copies only their opaque low bytes into the FP16 result. A submit
boundary before each precision transition is required and covered by the nonfinite regression. This reduced the
ellipsis realization from **17.611 s** to **5.517 s**, resets from 156 to 48, submits from 201 to 93, and tasks from
2,161 to 663.

`~/npu/include/old/rknn_ops.md` marks advanced indexing and `aten::index` unsupported, and `~/rk3588` contains no native
Gather/fancy-index instruction example. Historical broad support used a typed NumPy evaluator and remains unported.
The current implementation instead extends the immutable equality/gather image shared by OneHot, Gather, Scatter,
MaskedSelect, and Nonzero.

- Complete fancy-index class: **7 passed in 23.07 s**, sequentially; every individual test is below 30 seconds.
- Complete Gather class: **2 passed in 14.46 s**. Exact full-byte OneHot and dynamic Scatter regressions also pass.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the final focused runs. No new RKNPU timeout,
  invalid IRQ, IOMMU fault, or kernel oops appeared.
- Rockchip collection: **403 tests**. Repository-wide Tinygrad mypy (216 files), Ruff, and `git diff --check`: pass.
  `sz.py`: renderer/runtime **3,463/282 executable lines**, total **28,800**.

The CPU-cheat audit found no Tinygrad-core change and only one runtime executable-line change, forwarding the immutable
INT16-output flag to command emission. Compile-time code proves UOp structure, integer coordinates, and raw addresses;
it never reads a dynamic index or FP16 value. Runtime gathers move opaque 1/2/4-byte representations. Dynamic INT32
conversion, positive/negative equality, axis conjunction, FP16-byte selection, reduction, and INT16 writeback all run
on DPU EW. There is no host fancy indexing or numeric conversion, LUT, CMAC, CNA, PPU fallback, tolerance relaxation,
or floating input wider than FP16.

---

## 2026-08-09 — native INT16 EW and integer-dimension fancy indexing

The remaining five forward cases from upstream `test_slice_fancy_indexing_dim_collapse_int` now run on Rockchip. They
cover a collapsed leading integer, a collapsed middle integer, three scalar dimensions, sparse dynamic axes, and a
mixed static-slice/dynamic-index form. Together with the preceding milestone, the complete Rockchip fancy-index class
now contains all twelve focused methods and every one passes individually under 30 seconds.

This milestone productizes the integer path proved by `~/rk3588/examples/elementwise_int.py` instead of adding more
FP16 integer emulation. RKImage v28 records native INT16 input and output independently; command emission configures
INT16 input/process/output precision, eight-lane geometry, INT16 RDMA precision, two-byte surface grouping, and the
integer converter mode. Runtime keeps INT16, INT32, and FP16 PC chains precision-homogeneous. An exact signed INT16 ADD
probe through Tinygrad's RKImage and RKNPU runtime passed eight lanes in one submit ioctl.

Arbitrary external INT32 indices are still compared without narrowing: each is decomposed as four opaque unsigned
bytes, and native INT16 SUB/ABS/MIN/MUL combines the exact byte equalities. Bounded-index masks, negative-normalized
linear coordinates, and raw FP16 byte selection now use this shared helper. Only final public INT32 storage is
regrouped from two-byte results into four-byte lanes; ROCKCHIP's public tensor dtype contract remains FP16.

- Five new integer-dimension cases: **5/5 passed individually**, from **3.39 s** to **12.99 s** pytest time.
- Complete twelve-method fancy-index class and both Gather regressions: every method passed individually; nonfinite
  outputs retain exact FP16 bit patterns.
- Representative largest case: **4.796 s** direct wall time, including **4.788 s** realization and 2.6 ms copyout;
  **3 submit ioctls / 4,115 DPU tasks / 0 soft resets**. The temporary native-INT32 implementation needed 9.252 s,
  391 submits, and 45 resets; the earlier FP16-emulation form needed 53.96 s, 1,013 submits, and 444 resets.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the complete census and after each timeout
  investigation. Repository-wide Tinygrad mypy and Ruff plus `git diff --check` pass.
- Rockchip collection: **408 tests**. A complete serial census reached **151 passed / 7 skipped / 100 subtests passed**
  before the pre-existing multidimensional Scatter method timed out at `dim=1`. The same isolated method and subtest
  reproduce unchanged at baseline commit `dd6c694d1` after a passing vendor health check, proving it is not an INT16
  regression. Focused milestone tests do not time out.
- `sz.py`: renderer/runtime **3,682/307 executable lines**, total **29,044**. Runtime remains smaller than QCOM's 312
  executable lines; the renderer is net +219 lines over the preceding milestone for five cases and the reusable
  native INT16 pipeline.

The CPU-cheat audit found no Tinygrad-core change and no runtime tensor arithmetic. Compile-time code validates UOp
structure, static coordinates, and bounds. Runtime gathers move opaque bytes according to immutable plans; dynamic
byte equality, validity masks, weighted coordinates, selection, and reductions execute on DPU EW. There is no typed
host index/fancy-gather evaluator, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than
FP16.

---

## 2026-08-09 — no-collapse dynamic FP16 fancy indexing

All five forward cases from upstream `test_slice_fancy_indexing_no_dim_collapse` are now separate bounded Rockchip
methods. They cover five broadcast dynamic axes, static outer dimensions around three dynamic axes, trailing and
spanning ellipses, and a dynamic middle axis surrounded by static slices.

Four forms already lowered through the native-INT16 fancy-index image. The static-outer form exposed one obsolete
guard in `_lower_multi_fp16_fancy_index`: it rejected a complete candidate matrix over 64K lanes even though the
shared image builder now partitions candidates into independently bounded blocks. Removing that pre-blocking guard
lets the existing dynamic blocker calculate each safe matrix; runtime command and descriptor sizes remain derived
from the emitted bodies rather than a hardcoded candidate count.

- Five no-collapse cases: **5/5 passed individually**, from **6.15 s** to **17.87 s** pytest time.
- Largest direct profile: **9.206 s** total, including **9.198 s** realization and 2.6 ms copyout;
  **3 submit ioctls / 7,933 DPU tasks / 0 soft resets**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. `sz.py`: renderer/runtime
  **3,680/307 executable lines**, total **29,042**; this milestone removes two renderer lines while adding coverage.

The CPU-cheat audit found no runtime or Tinygrad-core change. The removed guard changes only static plan acceptance;
dynamic INT32 byte equality, negative normalization, raw FP16 selection, and reduction continue to execute through
native INT16 DPU EW. Runtime gathers only move opaque bytes according to compile-time address plans. There is no host
fancy indexing, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — dimension-injected dynamic FP16 fancy indexing census

All eleven forward expressions from upstream `test_slice_fancy_indexing_dim_inject_none` are now independently
bounded Rockchip methods. They cover leading, trailing, internal, paired, and sparse `None` dimensions around as many
as four broadcast dynamic INT32 index tensors, plus the static all-`None` movement form.

Every case lowered through the existing native-INT16 fancy-index plan; no backend special case was needed. This is the
intended payoff from productizing INT16 as an RKImage/runtime precision rather than adding a matcher-specific probe.

- Dimension-injection group: **11/11 passed individually**, from **2.82 s** to **19.95 s** pytest time; no case
  exceeded the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **424 tests**.
- `sz.py` is unchanged at renderer/runtime **3,680/307 executable lines**, total **29,042**; tests and comments are not
  counted.

The CPU-cheat audit found no implementation change at all in this milestone. The tested path retains DPU-native INT16
byte equality, validity masks, raw FP16 selection, and reduction with opaque runtime movement only. There is no CPU
index evaluation, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — mixed list/tensor fancy indexing

The four forward expressions from upstream `test_slice_fancy_indexing_list_with_tensors` are now independent Rockchip
methods. They cover one tensor wrapped in a Python index list, a tensor plus a collapsed scalar, a tensor plus a static
tuple index, and five broadcast tensor indices.

Three forms already passed. Tinygrad materializes the apparent static tuple `(1,1)` as a second external INT32 buffer,
so the fourth form exposed a real matcher gap: multi-index lowering accepted only negative-normalized axes. The matcher
now also accepts a direct positive-only INT32 load when the complete gate proves canonical nonnegative and strict upper
bounds. It still requires every data-index and gate load to be uniquely accounted for. Positive-only axes omit the
negative-coordinate alternative; normalized axes keep it.

- Mixed list/tensor group: **4/4 passed individually**, from **11.73 s** to **12.25 s** pytest time.
- The full five-negative-normalized-axis regression also passed after the matcher generalization.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **428 tests**.
- `sz.py`: renderer/runtime **3,693/307 executable lines**, total **29,055**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The second external INT32 buffer is never inspected on
host: runtime gathers its opaque bytes and native INT16 DPU EW performs exact four-byte equality, axis conjunction,
selection, and reduction. There is no host tuple/index evaluation, LUT, CMAC, tolerance change, or floating input
wider than FP16.

---

## 2026-08-09 — static-list and dynamic-tensor fancy indexing

All seven forward expressions from upstream `test_slice_fancy_indexing_list_indices` are now independent Rockchip
methods. They cover a static list alone, leading and trailing static lists mixed with dynamic tensors, broadcast list
shapes, negative static indices, column-shaped lists, and multiple static/dynamic axes.

Python lists and tuples become external INT32 buffers in Tinygrad's scheduled IR. Two strict matchers therefore needed
the same positive-only extension already used by the final FP16 selector. The bool bounds image now accepts direct
INT32 loads only when their canonical lower and upper gates cover every conjunction leaf. The flattened INT32 index
image accepts a direct contribution only when row-major compile-time strides prove a finite candidate extent and the
complete source index remains below signed INT16 range. Negative-normalized axes retain their signed alternatives.

- Static-list group: **7/7 passed individually**, from **4.15 s** to **15.77 s** pytest time.
- The preceding mixed tensor/tuple and five-negative-axis regressions pass after the shared matcher changes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **435 tests**.
- `sz.py`: renderer/runtime **3,710/307 executable lines**, total **29,072**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static stride analysis enumerates only candidate values;
it never reads the list/tuple buffers. Runtime moves opaque bytes and native INT16 DPU EW computes exact byte equality,
bounds conjunction, weighted index reconstruction, FP16 selection, and reduction. There is no host fancy indexing,
LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — tuple fancy indexing

All six forward expressions from upstream `test_slice_fancy_indexing_tuple_indices` are now independent Rockchip
methods. They cover nested static tuples, positive and negative leading tuples mixed with dynamic tensors, trailing
tuples, column-shaped tuples, and tuple indices around an inserted `None` dimension.

Five forms reused the static-list milestone unchanged. The final form lowers to a 3,400-node unrolled selector because
Tinygrad materializes its tuples as external INT32 buffers. The strict unrolled matcher now recognizes direct
positive-only axes only when equality predicates prove a complete contiguous coordinate domain and the complete
Cartesian candidate product is present exactly once. Negative-normalized axes retain their signed alternatives.

- Tuple-index group: **6/6 passed individually**, from **2.94 s** to **8.41 s** pytest time.
- The original five-axis all-negative-normalized unrolled regression passes after the matcher extension.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **441 tests**.
- `sz.py`: renderer/runtime **3,724/307 executable lines**, total **29,086**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Direct tuple buffers remain opaque on host; compile-time
IR predicates prove only their allowed coordinate domains. Native INT16 DPU EW performs exact byte equality,
conjunction, FP16 selection, and reduction. There is no host tuple evaluation, LUT, CMAC, tolerance change, or floating
input wider than FP16.

---

## 2026-08-09 — public bounded INT16 DPU EW

The native INT16 precision introduced for index masks is now reachable from ordinary Tinygrad signed-INT16 elementwise
graphs. The implementation follows `~/rk3588/examples/elementwise_int.py`: DPU input, process, and output precision are
INT16, while ADD, SUB, MUL, MAX, MIN, ABS, and NEG use the decoded integer ALU encodings. Tinygrad's portable
`ADD(x, MUL(y, -1))`, XOR-based minimum, and WHERE-based absolute-value forms are folded back to native operations.
Static broadcasts reuse the same immutable raw-gather plans as FP16.

The path deliberately does not add INT16 to `supported_dtypes`: RK3588 saturates signed integer ADD/SUB/MUL/ABS/NEG,
whereas Tinygrad's general fixed-width integer contract wraps on overflow. Dedicated limit tests record the hardware
behavior exactly instead of falsely advertising a fully general INT16 backend.

- Five methods / **13 DPU programs passed individually**, including arithmetic, broadcasting, chaining, and saturation;
  every program asserts exactly **one submit ioctl**.
- Largest direct profile, 131,072-element ADD: **0.345 s total** = 242.2 ms device open + 8.7 ms input creation +
  0.1 ms graph construction + **91.4 ms realization** + 2.6 ms copyout; **1 ioctl / 3 DPU tasks**.
- FP16 broadcast and native-INT16 tuple-fancy regressions pass after sharing gather validation/cache helpers.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **446 tests**.
- `sz.py`: renderer/runtime **3,855/307 executable lines**, total **29,217**. Shared gather validation and immutable-plan
  retargeting remove duplicated FP16/INT16 bounds and cache plumbing.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time evaluates only static shapes, constants, and
gather addresses. Runtime copies opaque `uint16` lanes; all input-dependent signed arithmetic, extrema, absolute value,
negation, and saturation execute on DPU EW. There is no host numeric evaluator, LUT, CMAC, CNA, PPU fallback, tolerance
relaxation, or floating input wider than FP16.

---

## 2026-08-09 — zero-stride FP16 mulacc census

All three forward expressions from upstream `test_mulacc_with_zero_strides` now run on Rockchip: a scalar expanded over
`(2,4,3)` and reduced on the last axis, two `(2,4,3)` zero-stride broadcasts reduced over axes `(0,2)`, and the `1x2 @
2x3` matrix product. The upstream method's first and third expressions are entirely constant and therefore fold to a
CPU realization before reaching a device backend. The Rockchip method preserves the same shapes and algebra but binds
external FP16 buffers, then asserts exact results and the four resulting NPU submit ioctls.

Historical commit `ad990be16` solved the old normal-FP32 form with compensated CMAC; it was intentionally not ported.
The current FP16 backend already lowers these cases through raw broadcast gathers, DPU MUL/ADD reduction, and the
existing DPU-only dot path. `~/rk3588/examples/elementwise.py` independently proves the underlying MUL and ADD stages.

- Focused pytest: **1 passed in 2.95 s**, below the 30-second policy.
- Direct wall decomposition, all three cases: **0.317 s total**, including 58.6 ms input setup. Case realizations were
  **87.1 / 48.2 / 23.2 ms**, copyouts 2.4 / 0.7 / 0.6 ms, with **2 / 1 / 1 ioctls** and **2 / 4 / 3 tasks**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **447 tests**.
- `sz.py` remains renderer/runtime **3,855/307 executable lines**, total **29,217**; no backend implementation was added.

The CPU-cheat audit found that all six operands are realized external FP16 Rockchip buffers before timing/submission.
Runtime performs only raw layout gathers; every value-dependent multiplication and addition executes on DPU EW. There
is no CPU constant-fold result, CMAC, LUT, host reduction, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — large boolean ALL reductions

The exact upstream `test_all_large` method now covers `2**15`, `2**16`, and `2**20` on Rockchip. Tinygrad's ordinary
non-local schedule expands the first `2**20` reduction into enough scalar UOps to consume more than five minutes and
about 2.1 GiB before rendering. A cache-distinct `ROCKCHIP:BOOL` renderer mode instead accepts Tinygrad's compact
16-lane grouped loop and proves its static launch coordinates form a complete source permutation before emitting DPU
work. The default Rockchip renderer remains non-local, so unrelated elementwise, pooling, convolution, and GEMM
schedules are unchanged.

Contiguous FP16 blocks use a direct in-place DPU tree: FP16 nonzero comparison produces the mask, MAX/MUL performs
ANY/ALL, and the existing DPU conversion exposes the exact bool byte. Tinygrad's second reduction reads an opaque bool
buffer; bytes are widened into zeroed INT16 lanes, reduced by native INT16 MAX/MUL, then copied back as raw bool bytes.
Non-contiguous grouped axes retain the proven striped-gather path. Historical `03bad6205` used CMAC plus host
bool/FP16 conversion, and `~/rk3588`/`~/npu` contain no pure-bool reference, so neither older approach was ported.

- Exact upstream focused pytest: **1 passed in 9.70 s** for all three sizes; the post-cleanup rerun passed in **11.84 s**.
- Largest direct profile (`2**20`): **5.179 s total** = 1.2 ms graph construction + **5.174 s realization** + 3.4 ms
  copyout; **26 ioctls / 3,195 DPU tasks**. Code generation alone is **0.372 s**, down from over five minutes.
- Existing scalar and non-contiguous-axis ANY/ALL regressions pass individually after the grouped matcher addition.
- Full serial Rockchip census: **439 passed, 9 skipped, 187 subtests passed** in **679.12 s**; collection is **448 tests**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the census. The first attempt exposed and led
  to termination of a leaked host-only scheduler experiment; it was not an NPU timeout or wedge.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **3,983/307 executable lines**, total **29,345**. Shared contiguous block-tree emission
  removes the duplicate FP16/INT16 reduction loop.

The CPU-cheat audit found only compile-time evaluation of scheduler coordinates and a bijection proof over integer
offsets. Runtime gathers never inspect values: they move opaque FP16 or bool bytes. Every input-dependent comparison,
AND/OR reduction, and output conversion executes on DPU EW. There is no host numeric reduction, CMAC, LUT, tolerance
change, floating input wider than FP16, or Tinygrad-core modification.

---

## 2026-08-09 — dynamic FP16 threshold MaskedSelect

The data-dependent half of upstream `test_masked_select` now passes at its exact `(32,10)` shape and `x > 0.5`
predicate. Scalar predicate totals and prefix scans accept one proven uniform FP16 threshold. DPU SUB shifts the
threshold to zero before the existing IEEE positive-mask stages; DPU ADD produces the dynamic count and prefix. Prefix
matrices are divided into blocks derived from `_MAX_EW_ELEMS_FP16`, compacted through immutable scratch gathers, and
converted to INT32 only at the terminal stage. Tinygrad's local-register histogram prefix is normalized back into the
existing verified unrolled prefix image, and the final raw-FP16 selector applies the same threshold before exact byte
selection.

The exact seeded case selects 118 of 320 values and passes in **12.83 s** under pytest. Direct wall decomposition is
**10.004 s total** = 3.4 ms input construction + **1.676 s dynamic shape/count/prefix** + **8.324 s final realization**
+ 1.1 ms copyout/compare; it uses **54 submit ioctls / 2,347 DPU tasks**. A scalar-`True` dynamic broadcast over 32
values also passes in **5.60 s**. The two existing fixed MaskedSelect methods and all four adjacent fixed Nonzero
methods pass individually, with the longest at **9.01 s**. Rockchip collection is now **450 tests**.

The remaining scalar-`True` half of the upstream method at 320 values is not claimed: constant folding fuses three
redundant 320-lane reductions, and generic `devectorize2` grows from 2,061 compact nodes beyond the 30-second/roughly
2-GiB host budget before the Rockchip renderer is called. Historical commit `797611d18` solved this with an early
native-program hook that replaced the topology with a typed flat copy. That hook changes Tinygrad core and is therefore
not ported under this branch's no-core-change rule. `~/rk3588` and `~/npu` provide the underlying EW primitives but no
standalone MaskedSelect instruction or backend-side pre-codegen hook.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after all focused NPU tests.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,081/307 executable lines**, total **29,443**. The cleanup shares threshold parsing,
  walks each prefix term once, and avoids reparsing the final 320-term predicate.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile-time logic reads only UOp structure, shapes,
constants, and statically proven gather addresses. Dynamic FP16 comparison, predicate count, prefix addition, INT32
histogram/index normalization, equality, and raw FP16 selection execute on DPU EW. Python reads only the DPU-computed
scalar required by Tinygrad's dynamic-output API; it does not compute a tensor value. There is no host numeric
evaluator, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 MaskedSelect through native INT16 EW

The exact upstream `test_masked_select_size` method now passes on Rockchip. This milestone builds on the public bounded
INT16 DPU EW path from `c359841b5`, rather than adding another FP16 exact-integer encoding. Following
`~/rk3588/examples/elementwise_int.py`, RKImage and the runtime now support native INT16 input/process followed by
terminal INT32 writeback. Exact regressions cover both an arithmetic result and a bare INT16-to-INT32 cast; each uses
one submit ioctl.

Tinygrad's external bool prefix is widened from opaque bytes into zeroed INT16 lanes and accumulated exactly on DPU.
The fixed selector compares each of the four bytes of arbitrary external INT32 indexes with native INT16 SUB/ABS/MIN,
reduces the equality matrix, applies the DPU-computed bool count for pad/truncate validity, and moves the four selected
value bytes back to the external INT32 output. Values do not need to fit INT16: only their individual unsigned bytes
enter the integer ALU.

- Exact upstream `test_masked_select_size`: **1 passed in 5.18 s**, covering output sizes 0/2/4/8, `-1` padding,
  truncation, empty INT32 fill, and float dtype preservation.
- All five Rockchip MaskedSelect methods pass individually in **5.18–12.74 s**; all four adjacent fixed Nonzero
  methods pass in **2.91–7.09 s**. No case crosses the 30-second profiling threshold.
- The native INT16 arithmetic/saturation/broadcast/writeback regression set passes, including exact direct cast.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **452 tests**.
- `sz.py`: renderer/runtime **4,194/325 executable lines**, total **29,574**.

The CPU-cheat audit found no runtime value inspection or Tinygrad-core change. Renderer logic proves only UOp structure,
static bounds, and gather addresses. Dynamic bool accumulation, INT32-index byte equality, selection, padding, and
INT16-to-INT32 conversion all execute on DPU EW. Runtime only submits stages and copies opaque bytes. There is no host
numeric evaluator, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 fixed-size Nonzero

The exact upstream `test_nonzero_size` method now passes on Rockchip for ordinary INT32 tensors. Each arbitrary INT32
source value is treated as four opaque bytes. Native INT16 MIN maps each byte to a 0/1 nonzero flag and native MAX
combines the four flags. DPU INT16 ADD then produces the prefix/count, exact byte equality selects the compact
coordinate, and the terminal INT16-to-INT32 stage writes the result. This covers negative values and values that cannot
be represented numerically in INT16 without narrowing the source tensor.

Historical `ae5a4a6f6`/`add791a62` nonzero implementations were inspected but deliberately not ported: they execute the
reduction with NumPy in the Rockchip runtime. The hardware emission instead follows the INT16 and regrouping behavior
proved by `~/rk3588/examples/elementwise_int.py`; `~/npu` has no standalone nonzero instruction. FP16 and INT32
fixed-Nonzero lowerers now share one strict graph-plan parser, matching the matcher/emitter separation used by other
Tinygrad hardware backends.

- Exact upstream `test_nonzero_size`: **1 passed in 5.50 s**, covering truncate/pad, rank two, scalar shape, empty
  input, and dtype preservation.
- A stronger raw-bit regression over `0`, `-1`, `256`, `-65536`, `INT_MAX`, and `INT_MIN` passes exactly in **3.79 s**.
- All six Rockchip Nonzero methods pass individually after cleanup in **2.97–7.06 s**; no case crosses 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **454 tests**.
- `sz.py`: renderer/runtime **4,308/325 executable lines**, total **29,688**. The shared fixed-Nonzero plan removes the
  duplicated FP16/INT32 count, compact-index, coordinate, fill, and bounds parsing.

The CPU-cheat audit found no runtime or Tinygrad-core change. Runtime gathers copy opaque bytes but never interpret a
tensor value. All input-dependent INT32 nonzero detection, prefix/count reduction, exact index equality, coordinate
selection, padding, and INT32 writeback execute on DPU EW. There is no host numeric evaluator, LUT, CMAC, tolerance
change, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 wraparound ADD through native INT16 EW

Bounded arbitrary INT32 ADD now executes exactly modulo `2**32` without an INT32 arithmetic assumption. Each external
INT32 operand is gathered as four opaque unsigned bytes into native INT16 lanes. DPU INT16 ADD accumulates each byte,
SUB/MAX/MIN constructs the carry, and the four result bytes are regrouped into the external INT32 output. This avoids
the RK3588 native INT32 saturating-ADD mismatch while reusing the public INT16 EW path introduced by `c359841b5`.

The FP16 predicate-prefix emitter was also separated from its original fixed-MaskedSelect parser. It now accepts a
statically proven arbitrary predicate/address matrix and blocks rows from `_MAX_EW_ELEMS_FP16`; the existing dynamic
threshold MaskedSelect regression still passes. This is emitter/matcher separation consistent with the other hardware
backends rather than another operation-specific arithmetic copy.

- Exact wrap regression includes `INT_MAX+1`, `INT_MIN-1`, multioperand carries, and a result crossing the sign bit;
  it passes in **2.82 s** and uses **one submit ioctl**.
- All seven native INT16 arithmetic/writeback methods pass in **3.16 s**. Dynamic-threshold MaskedSelect passes in
  **12.77 s**. All six fixed Nonzero methods pass individually in **2.85–7.10 s**.
- The slow upstream dynamic `test_nonzero` now completes its 320-value FP16 predicate prefix and exact three-endpoint
  INT32 ADD, then rejects the next distinct 640-by-236 dynamic INT32 coordinate histogram. That larger histogram is
  not claimed by this milestone; the focused run reaches the reject in **20.72 s**, without an NPU timeout.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **455 tests**.
- `sz.py`: renderer/runtime **4,371/325 executable lines**, total **29,751**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Renderer evaluation is limited to UOp structure, static
bounds, and gather addresses. All input-dependent byte addition, carry generation, and regrouping run through DPU EW;
runtime gathers only copy opaque bytes. There is no host numeric evaluator, LUT, CMAC, tolerance change, or floating
input wider than FP16.

---

## 2026-08-09 — dynamic FP16 Nonzero with exact INT32 coordinates

All forward cases from upstream `TestOps.test_nonzero` now pass on Rockchip when run as independent hardware tests:
`(32,10)`, `(20,)`, `(10,5,3)`, and the six scalar values. They are split into four Rockchip tests so each physical-NPU
case has an independent wall-time result and a failure cannot obscure the next shape. The rank-two, rank-one, rank-three,
and scalar methods pass in **25.14 s**, **5.32 s**, **10.60 s**, and **3.02 s** respectively. Rockchip collection is now
**459 tests**.

The dynamic FP16 predicate count and prefix remain on the existing DPU FP16 path. Arbitrary dynamic INT32 coordinate
sums are gathered as opaque bytes and added exactly modulo `2**32` with the native INT16 carry emitter introduced in
`bdc9623bc`. Four byte-equality masks form the exact histogram, native INT16 reductions accumulate it, and a syntactically
proven `0 <= index < limit` gate selects static coordinate functions through exact INT32-byte equality. Register-lane
blocks derive from `_MAX_EW_ELEMS_FP16`; block transitions insert submit barriers without a hardcoded task-count or
command-buffer limit.

The original `(32,10)` profile was **32.34 s** = **12.01 s** dynamic count/prefix construction + **19.07 s** final
realization + about 1 ms copyout, using **109 submit ioctls / 4,507 DPU tasks**. The ioctls themselves totaled only about
44 ms. Renderer profiling found scalar static-layout proof was the avoidable cost: the 256-row prefix took **7.00 s** to
render, while the bounded coordinate selector separately scalar-evaluated hundreds of candidate expressions. A shared
vector static evaluator reduces the prefix render to **1.32 s** and brings the complete pytest case below 30 seconds;
NPU tasks and numeric behavior are unchanged.

The Rocket reference's FP16-compare-to-INT16 body was also tested as a possible reset reduction. A focused dynamic
threshold run timed out when that body was placed in the current RKNPU PC-chain transition, so the experiment was fully
reverted. The established reset-isolated FP16 compare path remains. The vendor `~/rk3588/examples/elementwise.py` health
probe passed **60/60** immediately after the revert and again after all final focused runs; no reboot was required.

- Exact INT32 wrap regression: pass; all seven native INT16 EW regressions: **7 passed in 3.02 s**.
- Fixed Nonzero regressions: **4 passed in 10.69 s**; exact arbitrary-INT32 bytes: **1 passed in 3.52 s**; upstream
  fixed-size Nonzero: **1 passed in 4.68 s**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,609/325 executable lines**, total **29,989**. Cleanup shares one local-add-loop parser,
  reuses the exact byte-sum emitter, and removes the unused loop-specific coordinate selector.

The CPU-cheat audit found no runtime or Tinygrad-core change. Renderer vector evaluation is limited to compile-time UOp
index expressions, constants, bounds, and candidate coordinate tables. Runtime gathers still move opaque bytes only.
All input-dependent FP16 predicates, dynamic counts/prefixes, INT32 byte carries/equality, histogram reductions,
coordinate selection, and INT32 writeback execute on DPU EW. There is no host numeric fallback, LUT, CMAC, tolerance
change, or floating input wider than FP16.

---

## 2026-08-09 — truncating FP16-to-INT32 cast on DPU

The FP16-input `.int()` portion of upstream `TestOps.test_cast` now passes exactly. A direct use of the existing DPU
FP16-to-INT32 converter was tested first and rejected: it rounds to nearest, producing `1` for positive fractions where
Tinygrad/PyTorch require truncation toward zero. The final implementation reuses the already hardware-tested
`_fold_trunc` composition, `floor(max(x,0)) + ceil(min(x,0))`, then routes the FP16 integer through the shared terminal
INT32 conversion arena. This also avoids multiplying sign masks by infinities.

Historical commit `815003d78` was inspected but not ported because its Rockchip runtime performs the numeric conversion
with NumPy `.astype(np.int32)`. Searches under `~/npu` and `~/rk3588` found conversion/truncation register fields but no
separate proven FP16-to-INT32 cast sequence. Reusing the current native FLOOR/CEIL and typed-output emitters is smaller
and keeps the hardware/host boundary consistent with the other Tinygrad backends.

- Upstream random `(3,3)` FP16-to-INT32 cast plus explicit positive/negative fractions: **1 passed in 3.24 s**.
- Complete focused Cast class: **2 passed in 3.77 s**; the two INT32 cases use **4 ioctls / 20 DPU tasks** total.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the cast runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **460 tests**.
- `sz.py`: renderer/runtime **4,615/325 executable lines**, total **29,995**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer only recognizes a direct FP16 load and builds
the native FLOOR/CEIL UOp composition. Fractional truncation and terminal INT32 conversion execute on DPU EW; runtime
only submits the existing typed stages and copies opaque output bytes. There is no NumPy numeric conversion, host tensor
inspection, LUT, CMAC, tolerance change, or external floating input wider than FP16.

---

## 2026-08-09 — exact FP16-to-FP32 widening on DPU

The FP16-input `.float()` portion of upstream `TestOps.test_cast` now runs on Rockchip. The DPU FP32 output converter
writes four dense FP32 lanes from one eight-lane FP16 atom; larger inputs are therefore gathered into aligned four-lane
groups and emitted as consecutive tasks in one PC-chain ioctl. This limit is derived from the FP16/FP32 element sizes,
not a task-count or command-buffer constant. Direct one- through four-lane inputs need no gather.

`~/rk3588`'s fused-pipeline probe established the FP16-to-FP32 register mode for four-lane, grouped, and wider inputs.
Focused hardware probes found that ADD-zero loses the sign of negative zero, while native `MAX(x,x)` pass-through
preserves every tested widened bit exactly: signed zeros, normal values, subnormals, infinities, and the FP16 NaN payload.
The standalone scalar FP32 emitter was folded into the shared stateful typed emitter, so scalar FP32 reductions and the
new vector cast use one register-construction path.

- One-, two-, three-, four-, eight-, and nine-element focused conversions pass; the permanent random `(3,3)` and exact
  encoding regression uses **one ioctl per realization**. The complete Cast class passes **3/3 in 3.89 s**.
- The existing FP32 scalar reduction regression passes with its unchanged **2-ioctl** contract.
- Full serial Rockchip census: **452 passed, 9 skipped, 187 subtests passed in 653.06 s**. The slowest case was
  `test_simple_cummin` at **27.99 s**, so no individual test crossed the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the census.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **461 tests**.
- `sz.py`: renderer/runtime **4,626/327 executable lines**, total **30,008**. Removing the separate scalar emitter offsets
  most of the new typed-cast matcher and grouped-layout code.

The CPU-cheat audit found no Tinygrad-core change or runtime numeric conversion. Renderer work is limited to proving
static FP16 addresses and preparing raw aligned lane movement; runtime gathers copy opaque `uint16` values. Widening and
FP32 writeback execute on DPU EW. There is no NumPy cast, host tensor inspection, LUT, CMAC, tolerance change, or FP32
input.

---

## 2026-08-09 — exact signed-INT16 comparisons on DPU

All six signed-INT16 comparisons now lower through the public native integer EW path: `<`, `>`, `<=`, `>=`, `==`, and
`!=`. RK3588 has no separately proven INT16 comparison opcode, so the implementation composes the operations established
by `~/rk3588/examples/elementwise_int.py`. Saturating `rhs-lhs` preserves the sign needed for ordered comparison across
the complete INT16 range; MAX with zero followed by MIN with one produces the strict-less mask. Equality uses saturated
SUB, ABS, and MIN-one, so every nonzero difference—including opposite endpoints—maps to one. Tinygrad's boolean
inversions are recovered as native `1-mask` SUB.

Historical exact-INT32 comparison commit `4e2931c2c` was inspected for its sign-safe mask composition, but its four
FP16 byte-plane ABI was not ported: direct INT16 inputs already have the correct signed representation. Boolean output
uses one shared affine low-byte gather also reused by the stored-boolean reduction paths. This is representation movement
after DPU computation, not host predicate evaluation.

- Full-range tensor/tensor ordering and equality, scalar operands in both directions, and a `(2,4)`/`(4,)` broadcast all
  pass. Each realization uses **one ioctl**. The three focused methods pass in **2.95 s**.
- A **131,072-element** comparison passes in one ioctl: **1.5 ms** tensor creation + **163.8 ms** realization +
  **2.6 ms** copyout = **167.8 ms** total. The complete native INT16 class passes **11/11 in 3.22 s**.
- Native INT16 plus the existing stored-boolean reduction group passes **17 tests in 12.76 s** before the large case was
  added; the affine low-byte cleanup therefore preserves both consumers.
- A serial full-census attempt reached **193 passed, 7 skipped, and 100 subtests** before one existing multidimensional
  scatter subcase hit a transient six-second driver timeout. Vendor health immediately passed **60/60**, and the exact
  scatter method then passed all three subtests in **7.26 s**. The parent milestone's complete census remains
  **452 passed, 9 skipped, 187 subtests**; no new matcher can accept that FP16-output scatter graph.
- Final vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**. Repository-wide Tinygrad mypy (**216 files**),
  Ruff, and `git diff --check`: pass. Rockchip collection: **465 tests**.
- `sz.py`: renderer/runtime **4,644/327 executable lines**, total **30,026**. Sharing the affine INT16-low-byte transport
  reduced the initial comparison implementation by one executable line while removing several materialized offset tuples.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes comparison structure, static
broadcast addresses, and constants. Every input-dependent SUB, ABS, MAX, MIN, inversion, and boolean mask is computed by
DPU INT16 EW; runtime merely copies each NPU-produced low byte into the public bool buffer. There is no NumPy predicate,
host tensor inspection, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 comparison compositions on DPU

Boolean trees over signed-INT16 comparisons now remain in the native integer EW pipeline. The comparison matcher
recursively turns each predicate into its exact zero/one INT16 mask, then applies the same hardware-tested algebra used
by the FP16 logical path: MUL for AND, MAX for OR, ABS of SUB for XOR, and one-minus-mask for logical NOT. The DPU
operations are all independently covered by `~/rk3588/examples/elementwise_int.py`; no new register mode or runtime
special case was needed.

- Full-range `(x<y)&(x!=0)`, OR, XOR, and explicit `logical_not` all pass exactly, with **one ioctl per realization**;
  the focused method passes in **3.11 s**.
- The complete native INT16 and existing FP16 logical-predicate groups pass **18/18 in 32.13 s**. No individual test
  exceeds 30 seconds; the two slowest are the existing `isclose` methods at **11.07 s** and **10.69 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **466 tests**.
- `sz.py`: renderer/runtime **4,651/327 executable lines**, total **30,033**; the milestone adds seven renderer lines
  and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes the boolean tree and emits
native INT16 stages. Every input-dependent comparison and logical composition executes on DPU EW; runtime only performs
the existing opaque low-byte transport into the public bool buffer. There is no host predicate evaluation, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 selection on DPU

Comparison-driven INT16 `where` now lowers through the same native integer EW graph as arithmetic and predicates. The
scheduled IR is a single typed WHERE over the already-supported comparison tree. The renderer rewrites that bounded form
to `mask*a + (1-mask)*b`; because the mask is exactly zero or one, one product is always zero and the saturating INT16 ADD
returns the selected branch exactly, including the signed endpoints. This avoids the unsafe `b + mask*(a-b)` form, whose
intermediate subtraction could saturate before selection.

The implementation uses only the INT16 SUB, MUL, and ADD behavior proven by `~/rk3588/examples/elementwise_int.py` and
adds no register mode or runtime special case. The native matcher declaration was moved below the comparison helper so
the renderer reads in dependency order.

- Full-range tensor/tensor selection and scalar fallback selection pass exactly; the method is **0.14 s** after startup.
- A composed AND predicate with broadcast inputs and arithmetic branches passes exactly in **one ioctl** and **0.14 s**.
- The complete native INT16 class passes **14/14 in 3.45 s**; no case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **468 tests**.
- `sz.py`: renderer/runtime **4,655/327 executable lines**, total **30,037**; selection adds four renderer lines and no
  runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only rewrites the typed selection graph.
Predicates, mask inversion, masked products, addition, and branch arithmetic all execute on DPU INT16 EW; runtime only
submits the existing stages. There is no host selection, NumPy tensor evaluation, LUT, CMAC, tolerance relaxation, or
floating input wider than FP16.

---

## 2026-08-09 — native INT16 clamp and ReLU on DPU

Signed-INT16 one- and two-sided clips plus ReLU now recover the canonical extrema hidden by Tinygrad's portable WHERE
form. For `where(lhs<rhs, lhs, rhs)` the renderer emits one native MIN stage; reversing the selected operands emits one
native MAX stage. Nested two-sided clip therefore needs exactly two DPU tasks instead of expanding each bound through a
comparison mask, inverse, two masked products, and ADD.

The scheduled UOps were inspected directly before adding the fold. Historical commit `c359841b5` supplies the native
INT16 pipeline, `~/rk3588/examples/elementwise_int.py` proves signed INT16 MIN/MAX register behavior, and the RKNN
operator references under `~/npu` list INT16 extrema support. The implementation is a structural renderer rewrite like
Tinygrad's other hardware-specific canonicalizations; the generic runtime remains unchanged.

- Full-range two-sided, lower-only, upper-only, and reversed-bound clips pass exactly. Task assertions prove **2, 1, 1,
  and 2 tasks**, respectively, all in one ioctl; the focused method takes **0.19 s** after startup.
- Full-range INT16 ReLU passes exactly as **one native MAX task / one ioctl** in **0.11 s** after startup.
- The complete native INT16 class passes **16/16 in 3.53 s**; no case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the hardware runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **470 tests**.
- `sz.py`: renderer/runtime **4,659/327 executable lines**, total **30,041**; extrema recovery adds four renderer lines
  and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes branch identity in a static
WHERE/CMPLT graph. Every input-dependent MIN/MAX and nested clamp executes on DPU INT16 EW. There is no host comparison,
selection, or clipping, no NumPy tensor evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native signed-INT16 sign on DPU

Signed-INT16 `sign()` now lowers as the exact identity `min(max(x,-1),1)`, using two native extrema tasks. Tinygrad's
scheduled form is `WHERE(x!=0, WHERE(x<0,-1,1), 0)`. A dedicated sign rewrite runs before the general INT16 WHERE pass,
matching the existing FP16 backend organization; otherwise the inner selection is expanded first and the opportunity to
recover the two-stage clamp is lost.

Historical commit `6b00f28ae` supplied the corresponding four-stage FP16 sign matcher, while
`~/rk3588/examples/elementwise_int.py` proves the INT16 MIN/MAX operations used by the smaller integer formulation.
There is no separate Tinygrad `signbit` method; its meaningful integer form, `x<0`, is already covered by the exact
signed-comparison milestone. Internal sign constants remain weak integers in scheduled IR, so the INT16 structural
constant helper now accepts both explicit INT16 and weak-integer constants inside this already typed graph.

- The first diagnostic run was numerically exact but exposed **13 tasks**, proving the general WHERE path was used. The
  dedicated pre-pass reduces the same full-range case to the asserted **2 tasks / 1 ioctl**; call time is **0.11 s**.
- With renderer caching disabled, the complete native INT16 class passes **17/17 in 3.53 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **471 tests**.
- `sz.py`: renderer/runtime **4,669/327 executable lines**, total **30,051**. The dedicated structural pass adds ten
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time validates the exact sign tree and replaces it
with native extrema UOps. Every input-dependent sign result is computed by two DPU INT16 EW tasks; runtime only submits
the existing image. There is no host classification, comparison, or selection, no NumPy tensor evaluation, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 ReLU6 and hardtanh on DPU

INT16 ReLU6 and hardtanh now have explicit two-task contracts. Hardtanh naturally reuses the canonical clamp recovery.
Tinygrad expands ReLU6 as `relu(x)-relu(x-6)`, so the shared hard-activation pre-pass recognizes that exact graph before
the inner WHERE nodes expand and replaces it with `min(max(x,0),6)`.

Historical commit `6d6751002` provides the analogous FP16 cap fold and was used as the structural reference. The INT16
path deliberately does not reuse FP16 ReLUX configuration: `~/rk3588/examples/elementwise_int.py` proves integer MIN/MAX,
which are sufficient and keep the typed pipeline within known hardware behavior. The sign and ReLU6 rules share one
dedicated pre-pass, matching the staged rewrite organization used by the existing FP16 backend.

Non-integral leaky-ReLU slopes promote an INT16 tensor to a weak floating output and are outside this backend's external
FP16-only floating contract. Integral slopes retain INT16 and remain a separate bounded follow-on group.

- Full-range ReLU6 and custom `hardtanh(-3,4)` pass exactly, each as **2 tasks / 1 ioctl**; their combined call is
  **0.15 s** after startup.
- With renderer caching disabled, the complete native INT16 class passes **18/18 in 3.42 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **472 tests**.
- `sz.py`: renderer/runtime **4,684/327 executable lines**, total **30,066**. The bounded ReLU6 recognizer adds fifteen
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes the exact hard-activation
graph and emits native extrema UOps. All bounds and activations execute on DPU INT16 EW. There is no host clipping,
comparison, or selection, no NumPy tensor evaluation, FP16 ReLUX assumption, LUT, CMAC, tolerance relaxation, or
floating input wider than FP16.

---

## 2026-08-09 — bounded integral INT16 leaky ReLU on DPU

Leaky ReLU with integral slopes from 2 through 32767 now stays in the native INT16 EW pipeline. The renderer recognizes
Tinygrad's exact `where(x<0, slope*x, x)` graph and uses `x + min(x,0)*(slope-1)`. Slope two needs only MIN plus ADD;
larger slopes add one MUL stage.

This identity remains exact under RK3588's saturating INT16 arithmetic. For nonnegative inputs the correction is zero.
For negative inputs, if `(slope-1)*x` saturates low, the final addition remains saturated low; otherwise the final sum is
exactly `slope*x` and saturates at the same boundary. The focused tests include `-32768`, values that overflow only after
scaling, zero, and `32767`.

Historical FP16 commit `ea8294006` and the TRM/RKNN references under `~/npu` document the PReLU mode, but that mode was
not assumed for integer data. The implementation uses only the signed INT16 MIN, MUL, and ADD behavior directly proven
by `~/rk3588/examples/elementwise_int.py`. Fractional slopes promote the output to floating point and remain outside this
bounded INT16 path.

- Slopes two and three pass every saturation endpoint exactly with asserted **2 and 3 tasks**, respectively, and one
  ioctl per realization; their combined call is **0.12 s** after startup.
- With renderer caching disabled, the complete native INT16 class passes **19/19 in 3.41 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **473 tests**.
- `sz.py`: renderer/runtime **4,696/327 executable lines**, total **30,078**. The bounded leaky-ReLU fold adds twelve
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes a bounded typed graph and
emits native MIN/MUL/ADD UOps. All input-dependent activation work executes on DPU INT16 EW. NumPy is used only for the
test oracle, never by the backend. There is no host classification or selection, unproven integer PReLU mode, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 extrema loops on DPU

Signed-INT16 MAX/MIN reductions now continue past Tinygrad's 256-element unroll boundary. At 257 elements Tinygrad
switches to a local accumulator loop; the renderer recognizes the exact initialized MAX loop, including Tinygrad's
portable bitwise-complement form for signed MIN, materializes its static source layout, and emits a balanced native
INT16 extrema tree. The terminal DPU task writes the INT16 result directly.

The shared scalar-reduction arena no longer assumes that an output row fits in 64 bytes. Its block stride is derived as
`align64(rows*2)`, and gather spacing, task addresses, and scratch size all use that same value. The focused 40x257 case
therefore uses `257 * 128 = 32,896` scratch bytes instead of overlapping rows above the old implicit 32-row limit.

`~/rk3588/examples/elementwise_int.py` supplies the hardware reference for signed INT16 MAX/MIN. No branch contains a
more complete INT16 loop-reduction implementation. INT16 SUM is intentionally not claimed: Tinygrad promotes it to
INT32, whereas the native INT16 ADD path saturates, so an INT16 reduction tree would be observably wrong on overflow.
The existing exact byte-plane INT32 adder combines already materialized INT32 operands and is not a drop-in signed
INT16 reduction.

- Native 40x257 axis MAX and MIN pass exactly with asserted **256 DPU tasks / 1 ioctl each**; the focused method takes
  **3.09 s** including startup.
- The native INT16 class plus existing FP16 scalar min/max checks pass **22/22 in 4.71 s**.
- The complete physical-NPU Rockchip suite passes serially: **465 passed, 187 subtests passed, 9 skipped in 630.97 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** both before and after the complete suite.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **474 tests**.
- `sz.py`: renderer/runtime **4,738/327 executable lines**, total **30,120**. The native loop matcher plus the shared
  stride correction add 42 renderer lines and no runtime or Tinygrad-core lines.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static index expressions and constructs
gather plans; runtime gathers only rearrange tensor bytes. Every input-dependent comparison and reduction executes in
native DPU INT16 EW tasks. There is no host extrema calculation, NumPy tensor evaluation, LUT, CMAC, tolerance change,
or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 cumulative extrema on DPU

One-dimensional signed-INT16 `cummax` and `cummin` values now lower through one shared native extrema image. Small scans
are fully unrolled by Tinygrad; the matcher recovers each prefix candidate from direct MAX or the portable
complemented-MAX/WHERE form. At 257 elements Tinygrad switches to a local accumulator loop, and the loop matcher now
accepts the exact prefix-gated layout in addition to ordinary scalar reductions.

Both forms prove their static gather plan before code emission: output lane `i` must contain exactly source offsets
`0..i`, with disabled candidates replaced by the signed MAX/MIN identity. Input-dependent values are then reduced by
native INT16 DPU tasks. Scalar and cumulative paths share `_int16_extrema_image`; arena strides are 64-byte aligned and
derived from the row count rather than a task-count constant.

The scheduled UOps were inspected at 8 and 257 elements. `~/rk3588/examples/elementwise_int.py` supplies the proven
signed INT16 MIN/MAX register behavior; no branch contains a more complete INT16 cumulative implementation. Cumulative
sum/product are not included because saturating INT16 ADD/MUL do not preserve Tinygrad's promoted or wrapped semantics.

- Non-monotonic 17- and 257-element cumulative MAX/MIN pass exactly with asserted **16 and 256 tasks**, respectively,
  and **one ioctl per realization**. The focused method takes **4.43 s** including startup.
- Aligned scalar plus cumulative focused methods pass **2/2 in 5.01 s**; the INT16 class plus FP16 extrema regressions
  pass **23/23 in 6.19 s**.
- Current Rockchip collection is **475 tests**. A serial duration run passed 193 tests before one transient timeout in an
  unchanged multidimensional scatter subtest. Vendor health remained **60/60**, that method then passed all three
  subtests alone in **4.47 s**, and the remaining suffix passed **272 tests, 86 subtests, 2 skips**. Thus every collected
  node was covered on current code without reboot; the immediately preceding complete run was 465/465 plus 9 skips.
- No completed method exceeded 30 seconds. Existing FP16 `test_simple_cummin` was slowest at **27.13 s**; the next were
  `test_dynamic_nonzero_rank_two` at **21.68 s** and FP16 `test_simple_cummax` at **19.37 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before testing, immediately after the transient
  timeout, and after the suite suffix.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,777/327 executable lines**, total **30,159**. Sharing scalar/cumulative emission limits
  this milestone to 39 renderer lines and no runtime or Tinygrad-core lines.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static prefix membership and source
offsets. Runtime gathers rearrange bytes and DPU INT16 MIN/MAX computes every cumulative value. There is no host extrema
or prefix calculation over tensor values, NumPy backend evaluation, LUT, CMAC, tolerance change, or floating input wider
than FP16.

---

## 2026-08-09 — exact signed-INT16 ArgMax/ArgMin on DPU

Signed-INT16 ArgMax and ArgMin now return exact first-tie INT32 indices for global, axis, and 257-element loop forms.
The existing FP16 parser and first-tie validator are typed rather than duplicated: INT16 candidates use direct loads for
ArgMax and Tinygrad's `x XOR -1` complement for ArgMin. The shared image performs native extrema, exact equality masks,
descending-coordinate selection, and terminal INT16-to-INT32 conversion on DPU.

Axis ArgMin materializes a complemented-minimum temporary. Its value kernel now recovers `MAX(~x...)` as
`~MIN(x...)`, implemented exactly as native MIN followed by `-1 - value`; this handles `-32768` without the saturation
error that a NEG-based complement would introduce. The index kernel then compares the same complemented domain.

The small fused layout keeps one ioctl and uses replicated extrema gathers. At 257 candidates, replication would require
`window**2 + 2*window` gathers and overflow RKImage's unsigned-16-bit gather count. That limit is derived from the image
field width, not a hardcoded candidate cap. The wide path reduces a second candidate arena once and mid-gathers the
compact extrema row, reducing gather growth from quadratic to linear. The terminal INT16-to-INT32 body is now appended
to an existing compatible PC chain, matching the prepare/emit/submit layering used by Tinygrad's other hardware
backends and avoiding an otherwise unnecessary ioctl.

Historical FP16 commits `2ec7c17ea`, `309896f58`, and `83bd88931` supplied the tie, axis, and global parser references.
`~/rk3588/examples/elementwise_int.py` proves the signed INT16 MIN/MAX, SUB, and conversion building blocks; neither it,
`~/npu`, nor another branch contains a native INT16 ArgMax implementation to copy.

- Full-range global first ties, axis 0/1 selections, and non-monotonic 257-element global ArgMax/ArgMin pass exactly.
  The final focused method takes **0.66 s call / 3.63 s including startup**.
- Asserted task counts are global-8 **22/31**, axis-1 **14/16**, axis-0 **12/14**, and global-257 **520/522** for
  Max/Min. Small global cases use **1 ioctl**; axis cases use one value plus one index program; the wide global case uses
  two phases because its compact extrema row must be mid-gathered.
- The INT16 and existing FP16 ArgExtrema regression groups passed **28/28 plus 10 subtests in 14.42 s**; final added
  axis-0 assertions also pass in the focused method. No case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after hardware testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **476 tests**.
- `sz.py`: renderer/runtime **4,829/327 executable lines**, total **30,211**. Sharing the existing parser/emitter limits
  the milestone to 52 renderer lines; runtime executable size is unchanged and Tinygrad core is untouched.

The CPU-cheat audit found no tensor-value read or host-side result calculation. Compile time checks only static offsets,
candidate coverage, and Tinygrad's first-tie IR. Runtime only chains command bodies and performs the existing raw
mid-gather. Every extrema, complement, equality, coordinate selection, and typed writeback operation executes on DPU.
There is no host ArgMax/ArgMin, NumPy backend evaluation, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — exact signed-INT16 MaxPool values and indices on DPU

Rockchip now advertises signed INT16 as a native renderer dtype and lowers padded MaxPool values plus exact returned
INT32 indices. The existing static selection gather is typed instead of duplicated, so padding maps invalid lanes to
the signed `-32768` identity before native INT16 DPU MAX. The existing pool-index parser is likewise parameterized by
candidate dtype and reuses the native INT16 equality/descending-coordinate image added for ArgExtrema. First ties,
padding, batch/channel planes, and the global loop form remain validated from static UOps before emission.

`~/rk3588/examples/elementwise_int.py` supplies the proven signed INT16 MAX/MIN, equality-mask, and conversion register
behavior. The existing FP16 pool-index milestones supplied the spatial-coordinate and global-loop parsers. No new pool
parser, runtime tensor arithmetic, or CPU result path was added.

The first full MaxPool-class run exposed a precision-boundary hazard: native INT16 returned indices followed by the
existing wide FP16 index case timed out at task counter 1/raw status `0xc0000000`, although the wide test passed alone.
The vendor health probe remained 60/60. Runtime now records only whether the preceding EW image used native INT16 and
issues one device reset when leaving that mode. The exact INT16-then-wide sequence and the original full-class order
then pass; ordinary FP16 programs and tasks receive no extra resets.

- Full-range padded `(4,2,11,28)` INT16 MaxPool values, padded duplicate-max first ties, and global `(12,13)` returned
  indices pass bit-exactly. Returned-index cases use one value program plus one index program (**2 ioctls**).
- The complete MaxPool class passes **15 tests plus 33 subtests in 11.74 s**. The native INT16 class plus the new and
  wide pool regressions pass **25/25 in 8.30 s**. No case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** immediately after the diagnostic timeout and after
  final physical testing; no reboot was required.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **477 tests**.
- `sz.py`: renderer/runtime **4,850/331 executable lines**, total **30,236**. Sharing typed gather/equality/index emission
  limits the milestone to 21 renderer lines and four runtime state-transition lines; Tinygrad core is untouched.

The CPU-cheat audit found only compile-time evaluation of static gates, offsets, weights, and coordinate bounds.
Runtime gathers preserve raw INT16 bytes; every input-dependent maximum, equality mask, first-tie selection, and
INT16-to-INT32 writeback executes on DPU. There is no host MaxPool/index calculation, NumPy backend evaluation, LUT,
CMAC, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — exact signed-INT16 MaxUnpool on DPU

Signed-INT16 MaxUnpool now scatters through exact dynamic INT32 indices and returns Tinygrad's promoted INT32 output.
The FP16 MaxUnpool parser is typed rather than copied. Its static plane/candidate validation produces the same gather
plans, while a compact native image compares all four index bytes on DPU, masks the INT16 candidate matrix, converts
selected lanes to INT32, and performs the balanced candidate sum in native INT32.

The INT32 reduction is required for exact Tinygrad semantics: duplicate destination indices are summed, and two valid
INT16 values can exceed the signed-INT16 range. A regression deliberately maps `30000` and `30000` to one destination
and verifies the exact result `60000`. The shared byte-mask helper now accepts either one repeated index-offset vector
or a statically validated row per candidate; all existing gather, fancy-index, masked-select, and scatter callers keep
the former path unchanged.

`~/rk3588/examples/elementwise_int.py` provides the native INT16 equality, MUL, INT16-to-INT32, and INT32 arithmetic
register evidence. Existing FP16 MaxUnpool supplied the UOp parser and static scatter-layout validation; no local
reference contains a more complete native integer MaxUnpool implementation.

- Direct full-range INT16 scatter passes exactly with **40 DPU tasks / 2 ioctls**; pool→index→unpool round-trip and the
  duplicate-index INT32 accumulation also pass. The focused method takes **3.17 s** including startup.
- The complete MaxUnpool class passes **7/7 in 37.35 s**. Duration decomposition is wide FP16 **20.12 s**, padded FP16
  **10.40 s**, and every other method at or below **1.41 s**; no individual case crosses the 30-second threshold.
- Four representative existing dynamic gather/fancy-index/scatter/masked-select regressions plus two subtests pass in
  **8.72 s**, covering the unchanged single-offset form of the shared byte-mask helper.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **478 tests**.
- `sz.py`: renderer/runtime **4,880/331 executable lines**, total **30,266**. The milestone adds 30 renderer lines,
  no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime change and no tensor-value evaluation in the renderer. Compile time validates only
static layouts and coordinates; runtime retains the existing raw gathers. Every dynamic index comparison, value mask,
typed conversion, duplicate accumulation, and output write executes on DPU. There is no host scatter/sum, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — exact dynamic INT16 gather and fancy indexing on DPU

Dynamic INT32 gather indices now select signed-INT16 tensors exactly, including out-of-range zero fill, Python-style
negative fancy indices, and multiple simultaneous fancy-index axes. The prior FP16 gather emitter was already a raw
two-byte selector: it compares all four dynamic index bytes with native INT16 DPU masks, selects each representation
byte, and post-gathers those bytes unchanged. It is now named and typed as a shared 16-bit image instead of being
duplicated for INT16.

All three existing parsers—one bounded dynamic index, fully unrolled multi-index selection, and bounded multi-index
loads—accept an explicit 16-bit value dtype. The generic native INT16 leaf parser now declines loads whose address or
gate depends on an external tensor, allowing those graphs to reach the specialized dynamic-index validators rather
than attempting to evaluate a dynamic address at compile time.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 gather implementation. Historical Rockchip commits
`75100be4a`, `85c85e7fe`, and `ea50b7c96` provide the dynamic/fancy FP16 parser and exact integer-mask structure;
`~/rk3588/examples/elementwise_int.py` proves the native INT16 equality, MUL, and ADD operations used by the shared
image.

- Full-range INT16 gather with positive, negative, and multi-byte out-of-range INT32 indices passes exactly in
  **2.95 s**, asserting **1 ioctl**. Negative row selection and two-axis fancy indexing pass in **3.05 s**.
- Four representative FP16 gather/fancy regressions pass **4/4 in 7.93 s**. No focused case approaches 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **480 tests**.
- `sz.py`: renderer/runtime **4,886/331 executable lines**, total **30,272**. Typing and renaming the shared parsers/image
  adds only six renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static candidate offsets, bounds, and
coordinate domains; runtime performs the existing raw byte gathers. Every input-dependent index equality, candidate
mask, and selection reduction executes on DPU. There is no host indexing/result calculation, NumPy backend evaluation,
LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact dynamic INT16 scatter on DPU

Dynamic INT32 scatter indices now assign signed-INT16 values exactly with Tinygrad's last-wins semantics. The existing
FP16 scatter image already selects raw two-byte representations: it compares all four index bytes on DPU, constructs
reverse-order effective masks for duplicate destinations, selects source or preserved base bytes, and writes those
bytes unchanged. Its image, gather/output helpers, and parser are now named and typed for either FP16 or INT16 instead
of duplicating the algorithm.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 scatter implementation. Historical Rockchip
commits `826e67550` and `eef7cecce` provide the bounded and multidimensional FP16 scatter parser structure;
`~/rk3588/examples/elementwise_int.py` proves the native INT16 comparison, mask, and arithmetic operations used by the
shared image.

- A full-range signed-INT16 vector with repeated destinations passes exactly in **1.84 s**, proving that the later
  `32767` and `1234` writes replace earlier `-32768` and `-30000` writes. A multidimensional axis scatter passes in
  **1.59 s**.
- The complete Scatter class passes **15 tests plus 7 subtests in 18.79 s**. Existing FP16 nonfinite/raw-bit and
  multidimensional cases remain exact; the slowest individual method is **4.41 s**, well below 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **482 tests**.
- `sz.py`: renderer/runtime **4,887/331 executable lines**, total **30,273**. Sharing the existing raw 16-bit path adds
  one renderer dispatch line, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change and no tensor-value read in the renderer. Compile time checks only
static scatter layout, candidate coverage, and bounds. Every dynamic equality, duplicate-destination priority mask,
source/base selection, and output write executes on DPU. There is no host scatter/result calculation, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact fixed-size INT16 masked select on DPU

Fixed-size `masked_select` now compacts signed-INT16 values under an external boolean mask, truncates or pads to the
requested output size, and preserves every selected/fill bit exactly. The existing INT32 selector already computes the
boolean count and compact-index equality on native INT16 DPU lanes before selecting each representation byte. It is now
typed by integer dtype and byte width, so INT16 uses the same parser and emitter with two bytes instead of four.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 masked-select implementation. The existing native
INT32 masked-select milestone provides the complete fixed-output UOp proof and byte-selection image, while
`~/rk3588/examples/elementwise_int.py` proves its INT16 count, equality, mask, and reduction operations.

- Full-range selected INT16 values, duplicate sign patterns, `-12345` fill, and two-element truncation pass exactly in
  **1.48 s** within the full-class run.
- The complete MaskedSelect class passes **6/6 in 27.19 s**. Existing dynamic FP16, exact nonfinite-bit FP16, padding,
  truncation, and INT32 coverage remain clean. The slowest individual method is **8.66 s**, below 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **483 tests**.
- `sz.py`: renderer/runtime **4,889/331 executable lines**, total **30,275**. Generalizing byte count adds only two
  renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change and no host value inspection. Compile time validates only the
fixed-output compact-index graph and buffer extents. Boolean counting, compact-index equality, validity masks, raw-byte
selection, and fill composition all execute on DPU. There is no host compaction/result calculation, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact fixed-size INT16 nonzero on DPU

Fixed-size `nonzero` now returns exact INT32 coordinates for signed-INT16 inputs, including rank-two layouts,
truncation, padding, and the `-32768` representation. The first compile probe exposed that fixed Nonzero consists of
two integer kernels: an unrolled predicate prefix and the final compact-coordinate selection. Both existing INT32
lowerers are now typed by source dtype and byte width. INT16 checks two opaque representation bytes while INT32 keeps
four; any nonzero byte becomes a native INT16 mask, then the shared prefix/equality/coordinate pipeline completes on
DPU.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 Nonzero implementation. Historical commits
`78a38f384` and `aa7cac490` provide the fixed FP16 and exact-byte INT32 graph proofs, while
`~/rk3588/examples/elementwise_int.py` proves the native INT16 MIN/MAX, reduction, and INT32 writeback operations used
by the shared image.

- Exact rank-one values `{0, -32768, 256, -256, 32767, -1}` and rank-two coordinate/fill selection pass in focused
  cold runs of **3.79 s** and **3.61 s**; they take **0.85 s** and **0.75 s** within the warm class run.
- The complete Nonzero class passes **12/12 in 49.15 s**. The slowest individual method is the pre-existing dynamic
  rank-two FP16 case at **22.15 s**, below the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **485 tests**.
- `sz.py`: renderer/runtime **4,892/331 executable lines**, total **30,278**. Sharing the two integer stages adds only
  three renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime/core change or input-value inspection. Compile time proves only the static prefix
coverage, coordinate layouts, and bounds. Runtime gathers opaque source bytes; byte nonzero masks, prefix counts,
compact-index equality, coordinate selection, validity/fill composition, and INT32 writeback execute on DPU. There is
no host Nonzero/result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider
than FP16.

---

## 2026-08-09 — exact INT16 raw layouts on the generic gather path

Signed-INT16 movement, concatenation, stacking, repetition, and constant padding are now covered bit-exactly. Compile
probes showed that pure raw layouts already lower through the renderer's common static gather planner; only a fused
layout-plus-negation probe was unsupported and is intentionally left for the arithmetic fusion group. The correct
milestone therefore adds coverage without a dtype-specific renderer path.

No branch, `~/npu`, or `~/rk3588` contains or needs a distinct INT16 layout implementation: these operations move raw
bytes and do not invoke arithmetic registers. The existing Tinygrad Rockchip static/partial gather path is the direct
reference, matching the separation used by other hardware backends between layout planning and numeric kernels.

- Chained permute/flip/slice, constant `-12345` two-dimensional padding, axis concat, internal-axis stack, and repeat
  all pass bit-exactly: **3/3 in 3.08 s**, with every test call at or below **0.14 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **488 tests**.
- `sz.py` is unchanged at renderer/runtime **4,892/331 executable lines**, total **30,278**; this is test-only coverage
  with no runtime, renderer, or Tinygrad-core modification.

The CPU-cheat audit found no implementation change and no host-dependent layout decision. Tinygrad computes only the
static address plan at compile time; runtime raw gathers perform every tensor movement on the device-visible buffers.
There is no host tensor read/result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or numeric
conversion.

---

## 2026-08-09 — exact promoted INT16 sums in native INT32

Signed-INT16 sums now widen each gathered reduction row with the DPU INT16-to-INT32 output path and accumulate the
rows through native INT32 ADD. Both statically unrolled reductions and Tinygrad's register-loop reduction form are
covered. The logical image accepts the native 32-bit lane capacity; runtime atomizes INT16 conversion into eight-lane
tasks while leaving each 40-lane INT32 accumulation stage intact.

No Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 reduction implementation. The direct reference
is `~/rk3588/examples/elementwise_int.py`: it proves native signed-INT16 ADD, INT16-to-INT32 writeback, and native
INT32 ADD on the same DPU EW pipeline. The new image composes those proven operations with the existing balanced-row
reduction helper.

- Exact axis-one, axis-zero, and global unrolled sums pass, including `-32768`, `32767`, and results outside INT16.
- The full **40x257** register-loop reduction passes exactly with **1,542 DPU tasks / 2 submit ioctls**; its warm call
  takes **0.17 s** and the focused cold method takes **3.33 s**.
- The complete native INT16 EW class passes **24/24 in 6.22 s**; the existing INT16 max-unpool regression also passes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **490 tests**.
- `sz.py`: renderer/runtime **4,949/331 executable lines**, total **30,335**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection in the renderer. Compile time validates
only static reduction shape, source coverage, masks, and buffer bounds. Runtime gathers opaque INT16 rows; widening,
balanced INT32 accumulation, and final INT32 writeback all execute on DPU. There is no host sum/result calculation,
NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — fused INT16 layouts and arithmetic

Native signed-INT16 arithmetic is now covered after multi-source and transformed layouts: concat followed by NEG,
stack followed by ABS, permute followed by ADD, constant pad followed by clip, and repeat followed by ADD. These were
the explicit remaining forms from the prior raw-layout milestone. Current compile probes showed that the generic
INT16 leaf/gather planner already composes all five, so no dtype- or operator-specific implementation was added.

`~/rk3588/examples/elementwise_int.py` directly proves signed-INT16 NEG, ABS, ADD, MAX, and MIN; no other Tinygrad
branch or `~/npu` provides a more integrated fused-layout path than the current generic Rockchip planner.

- All five exact cases pass in **3.09 s**, with calls from **0.02 s** to **0.17 s**. Concat, permute, and repeat use one
  DPU task; pad+clip uses two; stack+ABS composes two source-layout tasks and one ABS task in one ioctl.
- The complete native INT16 EW class passes **29/29 in 6.57 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **495 tests**.
- `sz.py` is unchanged at renderer/runtime **4,949/331 executable lines**, total **30,335**; this milestone is
  test-only and does not change Tinygrad core.

The CPU-cheat audit found no implementation change and no host-dependent arithmetic decision. The renderer derives
only static gather plans; opaque INT16 layout movement and every NEG/ABS/ADD/MAX/MIN stage remain on the device path.
There is no host result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input
wider than FP16.

---

## 2026-08-09 — exact promoted INT16 cumulative sums

Signed-INT16 `cumsum` now stays exact in INT32 when Tinygrad lowers a large scan to a masked register-loop reduction.
The existing promoted-sum matcher already handled small unrolled scans; its loop coverage proof now recognizes
disjoint nested prefix chains rather than requiring each input lane exactly once. The same validator replaces the
older one-dimensional cumulative-extrema check, so scan safety has one shared static proof.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 cumulative-sum path.
`~/rk3588/examples/elementwise_int.py` proves the required signed INT16-to-INT32 writeback and ADD stages; this
milestone composes them through the existing balanced INT32 reduction image.

- Exact unrolled 17-element and loop-shaped 257-element scans pass independently. Batched **2x257 last-axis** and
  **257x2 first-axis** scans also pass, proving both contiguous and interleaved prefix layouts.
- Independent calls take **0.24 s**, **1.19 s**, **2.59 s**, and **2.73 s**; no case approaches the 30-second policy.
- The complete native INT16 EW class passes **33/33 in 12.55 s**. Focused cumulative-extrema, cumsum, and scalar-sum
  regressions pass **3/3 in 5.87 s** after sharing the validator.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **499 tests**.
- `sz.py`: renderer/runtime **4,963/331 executable lines**, total **30,349**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change or tensor-value inspection. The new helper examines only compile-time
source-offset sets and proves uniqueness, partitioning, and strict prefix nesting. Runtime gathers opaque INT16 lanes;
all widening, balanced ADD, and INT32 writeback execute on DPU. There is no host cumsum/result calculation, NumPy
backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — ordered INT16 product reductions on DPU

Large signed-INT16 `prod` and `cumprod` register loops now lower to an ordered native-MUL chain. Masked cumulative lanes
use the exact multiplicative identity one. The chain is intentionally sequential rather than balanced because RK3588
INT16 MUL saturates; reassociation can change an overflowed result. Sum and product now share one typed local-loop
parser for accumulator initialization, source/load validation, static block construction, and scalar/prefix coverage.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated loop-product implementation.
`~/rk3588/examples/elementwise_int.py` directly proves signed-INT16 MUL and its saturating limits on this DPU EW path.

- A **40x257** axis product, a 257-element cumulative product, and **2x257 last-axis** plus **257x2 first-axis**
  cumulative products pass exactly. The sequence `300 * 300 * -1` proves ordered saturation: `32767`, then `-32767`.
- Independent calls take **0.22 s**, **0.91 s**, **1.88 s**, and **2.04 s**. First-axis cumprod includes one Tinygrad
  layout kernel and the 257-task product chain; the other cases use one submit.
- The complete native INT16 EW class passes **37/37 in 16.97 s**; no case approaches the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **503 tests**.
- `sz.py`: renderer/runtime **4,991/331 executable lines**, total **30,377**. Sharing the existing sum parser keeps the
  implementation to a net 28 renderer lines; runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. Compile time evaluates only static
addresses, masks, and reduction topology. Runtime gathers opaque INT16 lanes; every ordered saturating MUL and final
write executes on DPU. The NumPy routine exists only in the test oracle. There is no host product/result calculation,
LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 any/all reductions on DPU

Large signed-INT16 `any` and `all` register loops now reduce exact nonzero predicates entirely on the DPU. Each opaque
INT16 lane is converted to a native 0/1 mask from both source bytes, preserving `-32768` as nonzero; `any` then uses
INT16 MAX and `all` uses INT16 MUL. The final low result byte is copied directly into Tinygrad's boolean output layout.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 any/all reduction. The local
`elementwise_int.py` reference proves the native integer MAX/MUL primitives; this milestone composes them with the
backend's existing exact integer-nonzero mask and shared row reducer.

- A 257-element global `any`, **2x257 last-axis** `any`, 257-element global `all`, and **257x2 first-axis** `all` pass
  exactly. The input includes `-32768` to prove that the high source byte participates in the nonzero predicate.
- Each operation uses **259 DPU tasks / one submit ioctl**. Independent test methods pass in **3.20 s** and **3.00 s**;
  warm calls in the full class take at most **0.12 s**, well below the 30-second policy.
- The complete native INT16 EW class passes **39/39 in 17.41 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **505 tests**.
- `sz.py`: renderer/runtime **5,007/331 executable lines**, total **30,393**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change or tensor-value inspection. The renderer validates only static loop
offsets and topology. Runtime gathers opaque integer bytes; nonzero-mask construction, MAX/MUL reduction, and result
writeback execute on DPU. The post-gather is a raw low-byte layout copy, not host boolean computation. NumPy is used
only as the test oracle. There is no host reduction, LUT, CMAC, tolerance relaxation, or wider floating-point input.

---

## 2026-08-09 — exact INT16 nonzero counts and test-census split

Signed-INT16 `(x != 0).sum(...)` now produces exact INT32 counts for unrolled, register-loop, and matrix-axis graphs.
The existing byte-exact integer predicate mask feeds native INT16 ADD, followed by the DPU INT16-to-INT32 output
converter. The predicate emitter is shared with INT16 `any`/`all`; only the final reduction operation and output layout
differ. A shared local-ADD unroller also removes duplicate loop normalization from the existing FP16 count paths.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated integer nonzero-count reduction.
`~/rk3588/examples/elementwise_int.py` proves the required native INT16 ADD-to-INT32 primitive on this hardware.

- Exact 7-element unrolled, 257-element loop, **2x257 last-axis**, and **257x2 first-axis** counts pass independently.
  Inputs include `-32768`, proving that both opaque source bytes participate in the nonzero test.
- The unrolled case uses **10 DPU tasks / one ioctl**; each 257-lane reduction uses **260 tasks / one ioctl**. Final
  focused pytest methods pass in **2.89 s**, **2.77 s**, and **2.92 s**.
- Per the census-layout rule, `test_rockchip.py` now contains only upstream `TestOps` method names: **327 collected**.
  The **181 Rockchip-only** hardware regressions live in `test_rockchip2.py`; combined collection remains **508**.
- After the split, the backend-only INT16 class passes **38/38 in 17.34 s** and its four upstream-derived siblings pass
  **4/4 in 2.94 s**. No individual case approaches the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,041/331 executable lines**, total **30,427**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. Compilation proves only static source
coverage, masks, and output topology. Runtime gathers opaque INT16 bytes; predicate construction, ADD reduction, and
INT32 writeback all execute on DPU. NumPy appears only in test oracles. There is no host result calculation, LUT, CMAC,
tolerance relaxation, or wider floating-point input. The unchanged upstream `test_ops.py` has 427 collected nodes and
has not been run as one direct ROCKCHIP census; the 100 nodes outside the 327-case selected census are unclassified,
not measured failures.

---

## 2026-08-09 — exact DPU bytewise complement

The unchanged upstream `test_bitwise_not` now passes for both INT32 and boolean inputs. INT32 values are treated as
four opaque bytes and complemented with native INT16 `255-byte`; boolean values use the same path as `1-byte`. A raw
post-gather packs the low byte of each INT16 lane back into the typed destination. This preserves every INT32 bit,
including both signed limits, without asking the DPU to represent the value numerically as FP16.

No checked Rockchip branch, `~/npu`, or `~/rk3588` contains an integrated Tinygrad bitwise-not lowerer.
`~/rk3588/examples/elementwise_int.py` proves native integer `EW_ALU_MINUS=4` for INT16, while
`~/rk3588/test/test_ops.py` carries the same upstream behavior test. The new work is the bounded RKImage integration,
static movement support, and exact byte packing.

- Direct unchanged upstream `TestOps.test_bitwise_not`: **1 passed in 2.93 s**. The selected Rockchip copy passes in
  **2.94 s**; the exact full-byte/movement regression passes in **2.74 s**.
- The exact regression covers `INT32_MIN`, `INT32_MAX`, byte/carry boundaries, a transpose, boolean flip, and asserts
  **two DPU tasks / two ioctls** for its two realizations.
- Per the census-layout rule, `test_rockchip.py` contains **328 upstream-only cases** and `test_rockchip2.py` contains
  **182 Rockchip-only cases**; combined collection is **510**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,066/331 executable lines**, total **30,452**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no tensor-value inspection. Compilation computes only
static byte offsets and output layout. Runtime gathers opaque bytes, DPU INT16 SUB performs the complement, and raw
post-gather repacks the result. NumPy is used only by test oracles. There is no host result calculation, LUT, CMAC,
tolerance relaxation, or floating-point widening.

The direct upstream failure census remains incomplete. On this working tree, `test_sort`, `test_cast`, and
`test_pow_zero_exponent` are directly confirmed failures, `test_masked_select` is unresolved after a compile-time
runaway, and the other **95** nodes outside the now-328-case selected census remain unclassified rather than measured
failures.

---

## 2026-08-10 — exact DPU binary bitwise operations

The unchanged upstream `test_and`, `test_or`, `test_xor`, and `test_int_or` now pass alongside bytewise complement.
INT32 inputs are gathered as four opaque byte lanes. Each byte is decomposed into eight exact 0/1 INT16 lanes by
descending threshold, clamp, scale, and subtract stages. Native MIN/MAX implement AND/OR; SUB followed by native ABS
implements XOR. Weighted INT16 ADD reconstructs each byte, and a raw post-gather writes the exact little-endian INT32
representation. Boolean buffer AND/OR use a one-task native mask path instead of the full decomposition.

The reference audit found no hardware bitwise primitive. `~/npu/include/old/rknn_ops.md` marks BitwiseAnd/Or/Xor as
unsupported, and `/home/orangepi/rk3588/experimental/ops_rockchip.py` explicitly fell back to CPU for these ops.
`~/rk3588/examples/elementwise_int.py` proves the native INT16 MIN/MAX/SUB/ABS/MUL/ADD building blocks used here;
other checked Rockchip branches only contain boolean-mask AND/OR composition. This milestone replaces the old CPU
fallback idea with an exact bounded DPU construction.

- Direct unchanged upstream methods pass independently: AND **4.74 s**, OR **3.04 s**, XOR **2.91 s**, and integer
  all-ones OR **2.96 s**. The selected five-method bitwise class passes serially in **4.90 s**.
- A transposed full-bit-pattern regression covering signed limits, alternating bits, and arbitrary hex patterns passes
  in **3.13 s**. AND/OR use **92 tasks** each and XOR uses **100 tasks**, one ioctl per realization; the regression
  asserts **284 tasks / three ioctls**.
- The former predicate-only custom `test_and/or/xor` methods moved to `test_rockchip2.py` as explicitly local tests and
  pass **3/3 in 8.24 s**. `test_rockchip.py` now collects **329 upstream-only cases**, `test_rockchip2.py` collects
  **186 backend-only cases**, and combined collection is **515**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,147/331 executable lines**, total **30,533**. A shared typed-load matcher removes
  duplicate INDEX/source/bounds parsing from byte addition, complement, boolean logic, and binary logic.

The CPU-cheat audit found no runtime or Tinygrad-core change and no tensor-value inspection. Compilation materializes
only static offsets, public scalar constants, and threshold rows. All data-dependent bit extraction, bit combination,
and reconstruction execute as native DPU INT16 tasks. NumPy appears only in test oracles. There is no host result
calculation, CPU fallback, LUT, CMAC, tolerance relaxation, or floating-point widening.

The direct upstream failure census remains incomplete. The selected census now covers 329 of 427 nodes; among the
remaining 98, `test_sort`, `test_cast`, and `test_pow_zero_exponent` are confirmed failures, `test_masked_select` is
unresolved after a compile-time runaway, and **94** remain unclassified.

---

## 2026-08-10 — batched upstream remainder census and promotion

The remainder workflow now operates by capability batch rather than one method per milestone. Fresh collection of
`TestOps` contains **421 method nodes**; the selected Rockchip census represented 306 unique upstream method names,
leaving 115 names for direct classification. A validation batch and a 14-method data/indexing batch ran without an
RKNPU timeout, followed by a clean 60/60 vendor health check.

Eight unchanged upstream methods were promoted together: `test_nonzero`, `test_max_unpool2d`,
`test_scatter_no_reduce_tensor_src`, all three missing fancy-indexing families, `test_slice_fancy_indexing_errors`, and
`test_scaled_dot_product_attention_gqa_errors`. `test_rockchip.py` now collects **337 upstream-only cases**;
`test_rockchip2.py` remains **186 backend-only cases**, for **523 combined**.

The same batch classified shared gaps rather than stopping at first failure:

- Full/partial broadcast passes ADD/SUB/MUL/DIV but fails only its dynamic POW subtests.
- Direct `argmax`/`argmin`, general scatter/reduce, and repeated-integer `topk` reach unsupported integer graphs or
  FP16 encoding of an INT32 sentinel. These form the next shared integer-lowering batch.
- `scatter_reduce_errors` and `scatter_reduce_prod_zeros` currently fail in the Torch reference before Rockchip numeric
  execution because the upstream FP16-default dtype differs from the fixture's FP32 tensors.
- Attention length mismatch reaches a large unsupported softmax graph. Transcendental softmax/POW cases remain outside
  the no-LUT scope rather than receiving CPU fallbacks.

Two passing methods exceeded the 30-second policy and were profiled together:

| Method | Wall | Renderer | Program calls | Submit ioctls | Tasks | Ioctl wall |
|---|---:|---:|---:|---:|---:|---:|
| `test_nonzero` | 33.439 s | 5.632 s / 16 | 18.049 s / 17 | 192 | 8,148 | 0.063 s |
| `test_max_unpool2d` | 31.186 s | 7.274 s / 9 | 21.735 s / 9 | 225 | 3,351 | 0.108 s |

Both are dominated by host render/command construction and precision-transition/reset handling, not NPU ioctl time.
The batch also measured `dim_inject_none` at 29.93 s. These profiles define a shared host-overhead optimization target.

The CPU-cheat audit is unchanged: this milestone modifies only the selected test census and documentation. All promoted
methods were executed directly from unchanged upstream bodies. There is no host result calculation, CPU fallback, LUT,
CMAC, tolerance change, or Tinygrad-core change. The batch health check passed **60/60** vendor probes.

---

## 2026-08-10 — batched exact INT32 shifts

One native INT16 barrel-shifter lowerer now covers the full upstream 32-bit shift family: signed and unsigned inputs,
left and right shifts, scalar constants, and tensor shift counts. Four opaque input bytes are decomposed into exact
0/1 INT16 bit planes. Five DPU stages conditionally shift by 1, 2, 4, 8, and 16 bits; arithmetic right shift replicates
the original sign bit. Weighted INT16 row reductions reconstruct the four raw output bytes without FP16 integer
encoding, CPU arithmetic, CMAC, or LUTs. The existing AND/OR/XOR lowerer now shares the byte decomposition helper.

- Unchanged upstream batch: `test_lshift`, `test_rshift`, `test_lshift_signed`, `test_rshift_signed`, and
  `test_idiv_shift_rewrite_negative` passed together, **5/5 in 3.85 s**.
- The promoted upstream bitwise/shift class plus a backend-only all-count regression passed **11/11 in 6.15 s**.
  The local regression checks every dynamic shift count from 0 through 31 on zero, signed limits, all-ones, and mixed
  bit patterns for signed left/right and unsigned right shift.
- Adjacent cases were classified in the same batch: `test_div_int` still needs general INT/FP cast and integer
  division, while integer/dynamic POW requires a separate multiply/polynomial capability. Broadcast ADD/SUB/MUL/DIV
  subtests pass; only POW remains unsupported. `test_isclose` reaches the existing FP16 tolerance boundary and `test_cast`
  still lacks general INT32/boolean-to-FP output conversion.
- Selected collection is now **342 upstream-only cases**; backend-only collection is **187**. The authoritative
  upstream-name remainder falls from 107 to **102** after this five-method promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the physical batch.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,264/331 executable lines**, total **30,650**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no input-value inspection. Initial and mid gathers only
move raw bytes and compiler-known constants/offsets; all data-dependent bit extraction, conditional selection, sign
extension, weighting, and reconstruction execute as native DPU INT16 EW tasks. No tolerance was changed.

---

## 2026-08-10 — batched arg-extrema, stable sort, and TopK

Four unchanged upstream methods were completed as one shared integer-selection milestone: `test_argmax`, `test_argmin`,
`test_sort`, and `test_topk`. Signed INT32 extrema and compare/swap now operate on four widened byte lanes with an exact
native-INT16 lexicographic comparator. Boolean extrema use the same running first-tie selection shape. Stable sort's
occurrence and final-index graphs share one weighted equality emitter for FP16 numeric values and exact INT32 counts.
FP16 equality canonicalizes signed zero, distinguishes signed infinities, and rejects NaNs without the reset-heavy FP16
comparison mode.

- Direct unchanged upstream batch: **4/4 passed in 18.31 s**, including axes, repeated values, booleans, bitwise-not,
  signed INT32 limits, stable integer sort, and TopK values/indices.
- The promoted test aliases plus backend-only arg/sort/topk regressions passed **23 tests / 36 subtests in 22.27 s**.
  New local coverage includes signed zero, both infinities, repeated signed limits, and ascending/descending stable order.
- The former slow representative 8x8x6 sort-index realization fell from **11.12 s to 0.53 s**. Its reset count fell
  from **99 to 1**, submit count from **130 to 20**, and measured program wall from **10.78 s to 0.14 s**. The complete
  upstream sort/topk pair fell from **97.04 s to below 30 s** as part of the four-method batch.
- `test_rockchip.py` collects **346 upstream-only cases** (323 unique upstream method names); `test_rockchip2.py` collects
  **189 backend-only cases**. The authoritative upstream-name remainder is now **98 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, `git diff --check`, and focused renderer mypy/Ruff: pass.
- `sz.py`: renderer/runtime **5,605/331 executable lines**, total **30,991**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. Compilation handles
only static graph topology, offsets, weights, and byte layout. Runtime gathers opaque bytes; all comparisons, equality
masks, first-tie updates, weighted reductions, and INT32 writeback execute as native DPU INT16 EW tasks. There is no
host result calculation, CPU fallback, LUT, CMAC, tolerance relaxation, or floating-point input wider than FP16.

---

## 2026-08-10 — batched integer/boolean casts and empty std cases

The unchanged upstream `test_cast` now passes by composing two existing DPU converters: INT32 input is converted to an
FP16 numeric tile, then widened into the required FP32 output layout. Boolean bytes are zero-extended into the same
INT32 tile format before conversion. Four-lane groups use the DPU's required 16-byte FP32 conversion stride. Existing
FP16-to-INT32, FP16-to-boolean, and FP16-to-FP32 paths remain unchanged.

The remainder batch also identified and promoted two already-correct empty-axis methods,
`test_std_mean_loaded_nan` and `test_std_zero_in_axis`. They exercise empty-shape/NaN semantics and do not claim native
SQRT support; all non-empty standard-deviation methods remain outside the current no-LUT hardware scope.

- The full upstream `test_cast`, both upstream empty-axis std methods, and a backend-only integer/boolean conversion
  regression pass together: **4/4 in 4.42 s**.
- The backend regression covers dynamic signed INT32 values and boolean values; all expected values are exactly
  representable through the NPU's FP16 intermediate.
- `test_rockchip.py` now collects **349 upstream-only cases** (326 unique upstream names), and `test_rockchip2.py`
  collects **190 backend-only cases**. The authoritative remainder falls from 98 to **95 of 421**.
- The same batch classified general scatter/scatter-reduce as one dynamic-index WHERE lowering gap, non-empty std as a
  SQRT gap, and masked-select as a CPU-bound compiler runaway rather than an NPU timeout.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,639/331 executable lines**, total **31,025**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. The renderer validates only CAST
structure and static offsets. Runtime movement zero-extends opaque boolean bytes; both numeric conversions and FP32
writeback execute on DPU. There is no CPU arithmetic, fallback, LUT, CMAC, tolerance change, or wider input contract.

---

## 2026-08-10 — batched dynamic Scatter and ScatterReduce

Dynamic tensor Scatter and the five tensor ScatterReduce modes now share exact native-INT16 byte equality for external
INT32 indices. Scatter performs last-wins raw 16-bit selection entirely in the integer EW pipeline. ScatterReduce packs
the native 0/1 equality result into an FP16 mask at one mid-gather boundary, then performs sum, product, mean, minimum,
or maximum on DPU EW. Both `include_self` modes are covered. This removes the former four INT32-to-FP16 byte conversions
and their device resets from every reduction realization.

- The unchanged upstream `test_scatter_reduce` passes all **30 reduction/dimension/include-self cases in 5.31 s** under
  the established `test_gemm_fp16` tolerance.
- Direct wall decomposition for those 30 cases: **2.253 s** method wall, **0.191 s / 30** renderer calls,
  **0.099 s / 30** program calls, **0.013 s / 138** submit ioctls, and **0 resets**. The pre-optimization run had already
  exceeded **38.58 s** before finishing the mean cases because every realization performed four conversion/reset paths.
- The unchanged upstream `test_scatter` numeric prefix fell from **39.03 s to 4.20 s**. Its final dtype-error assertion
  is not valid with `DEFAULT_FLOAT=HALF`, so the unchanged method is not claimed in `test_rockchip.py`; the same eight
  numeric cases live in `test_rockchip2.py`.
- The complete Scatter alias/regression batch passes **18 tests / 13 subtests in 11.81 s**.
- `test_rockchip.py` collects **350 cases** representing **327 unique upstream methods**; `test_rockchip2.py` collects
  **191 backend-only cases**. The authoritative upstream-name remainder is now **94 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,807/331 executable lines**, total **31,193**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. The renderer uses
only static graph topology, offsets, and constants. Runtime gathers move opaque bytes; index comparison, last-wins
selection, reduction masks, arithmetic, and output reconstruction all execute on DPU EW. No LUT, CMAC, host result
calculation, tolerance relaxation beyond `test_gemm_fp16`, or input wider than FP16 was added.

---

## 2026-08-10 — batched exact signed INT32 division and remainder

One byte-restoring native-INT16 DPU lowerer now covers signed INT32 truncating division, floor division, C-style
remainder, and floor modulo. Four opaque input bytes are conditionally converted to magnitude, divided over 32 exact
bit steps, then sign- and rounding-corrected before raw INT32 writeback. A companion mixed INT32/FP16 path performs
device conversion followed by native FDIV, FLOOR/CEIL, MUL, and SUB stages. Direct use of the reference integer
`DIV` register was rejected: `~/rk3588/examples/elementwise_int.py` documents that it retains floating-point semantics
in integer precision and is therefore not exact signed integer division.

- Full unchanged upstream `test_mod`, `test_fmod`, `test_div_int`, and `test_div_rounding_mode`, plus a backend-only
  signed-limit regression, pass together: **5/5 in 33.43 s**. The local regression covers INT32 minimum/maximum,
  mixed signs, zero numerators, and all four rounding/remainder modes in **4 ioctls / 13,980 DPU tasks**.
- The only newly represented upstream method name is `test_div_int`; the other three replace earlier shortened local
  copies with their complete unchanged upstream bodies. `test_rockchip.py` now collects **351 upstream-only cases**
  representing **328 unique upstream method names**; `test_rockchip2.py` collects **192 backend-only cases**. The
  authoritative unselected remainder falls from 94 to **93 of 421**. These 93 are not all confirmed failures: most
  belong to shared transcendental-dependent families outside the current no-LUT scope.
- The former slowest case, unchanged `test_div_rounding_mode`, fell from **30.76 s to 26.34 s**. Before command reuse,
  a direct profile measured **28.093 s wall**, **0.106 s / 15 renderer calls**, **26.420 s / 144 program calls**, and
  **1.790 s / 320 submit ioctls**. Retaining the last immutable PC-chain bodies and prepatched scratch-only INT16
  bodies reduced the direct profile to **23.296 s wall**, **0.105 s renderer**, **21.599 s program**, and
  **0.334 s ioctl wall** without caching tensor values.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,117/352 executable lines**, total **31,524**.

The CPU-cheat audit found no Tinygrad-core change, runtime tensor-value inspection, or host numeric result calculation.
Compilation handles only graph topology, static offsets, constants, and register words. Runtime gathers opaque bytes;
all data-dependent sign detection, two's-complement conversion, comparison, subtraction, quotient/remainder formation,
rounding, and reconstruction execute on native DPU EW. The command cache retains immutable command qwords only. No
LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input was added.

---

## 2026-08-10 — exact native INT32 product and constant powers

A base-16 limb multiplier now evaluates arbitrary elementwise INT32 MUL trees exactly modulo 2^32 on native INT16
DPU EW. Each opaque input byte is split into two nibbles; schoolbook partial products stay below 256, and exact nibble
carry propagation keeps every intermediate within signed INT16 range. Common subexpressions are retained while walking
Tinygrad's exponentiation-by-squaring MUL graph. The older `rockchip-2607` implementation was deliberately not ported:
its `_try_int_power_host_subtasks` delegated the arithmetic to a typed host task, and its disabled native experiment
documented high-word corruption from direct INT32 MUL.

- Full unchanged upstream `test_int_pow_const_int` and a backend-only full-range product regression pass together:
  **2/2 in 4.03 s**. The upstream method covers powers 0, 1, 2, 7, and 29 plus the required negative-exponent error.
- Arbitrary two-input multiplication passes signed limits and **127 random full-range INT32 pairs** exactly. It uses
  **1 ioctl / 1,468 DPU tasks** independent of vector length; the repeated-square power paths use 1,364 tasks for
  exponent 2, 5,120 for exponent 7, and 8,876 for exponent 29.
- `test_rockchip.py` collects **352 upstream-only cases** representing **329 unique upstream method names**;
  `test_rockchip2.py` collects **193 backend-only cases**. The authoritative unselected remainder falls from 93 to
  **92 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,206/352 executable lines**, total **31,613**.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. Gathers move opaque
bytes and compiler-known constants; nibble extraction, partial products, carries, overflow truncation, and output
reconstruction all execute as native INT16 DPU stages. There is no host result calculation, CPU fallback, LUT, CMAC,
tolerance change, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — exact default FP16 isclose and reset removal

Default FP16 `isclose` now bypasses subnormal tolerance arithmetic with a mathematically equivalent exact IEEE
equality path. At `rtol<=1e-5` and `atol<=1e-8`, even adjacent FP16 subnormals are farther apart than the tolerance.
Operands are realized in FP16 first, gathered as opaque bytes, and compared with native INT16 arithmetic. Signed zeros
are canonicalized, NaNs are classified from exponent/mantissa bytes, and `equal_nan` is preserved. Larger tolerances
retain the general device arithmetic path and OR exact equality into its result. The byte canonicalization emitter is
shared with stable sort equality instead of duplicated.

- Full unchanged upstream `test_isclose` passes in **17.02 s**, down from **65.35 s** after the first correctness fix.
  The intermediate exact-UOp implementation passed in 32.52 s but still paid reset-heavy FP16 comparisons and was
  replaced. Unchanged upstream `test_isclose_edge_cases` now passes in **3.27 s** and replaces its former packed custom
  copy in the upstream-only census.
- Final direct wall decomposition for all ten `test_isclose` realizations: **14.599 s wall**, **0.073 s / 10 renderer
  calls**, **13.666 s / 10 program calls**, **0.043 s / 140 submit ioctls**, and **13.443 s / 127 resets**. The remaining
  resets belong to the two explicit `rtol=0.01` general-tolerance graphs; every individual method is below 30 seconds.
- The full isclose/scalar/edge/local regression plus adjacent signed-zero/infinity sort regression passes **5 tests / 2
  subtests in 29.67 s**. The backend regression covers both signed zeros, adjacent subnormal/normal bit patterns,
  maximum finite values, both infinities, and NaNs under both `equal_nan` modes.
- `test_rockchip.py` collects **353 upstream-only cases** representing **330 unique upstream method names**;
  `test_rockchip2.py` collects **194 backend-only cases**. The authoritative unselected remainder falls from 92 to
  **91 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,302/352 executable lines**, total **31,709**.

Historical `b85d3c4b0` and `31c07151d` computed isclose predicates in host elementwise layouts and remain rejected. The
current runtime only moves raw bytes at explicit gather boundaries; FP16 expression realization, zero normalization,
NaN classification, equality, tolerance evaluation, and bool formation all execute on DPU EW. There is no host result
calculation, CPU fallback, LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — dynamic scalar-source scatter batch

The existing dynamic last-wins Scatter lowerer now accepts compiler-known FP16 source expressions as well as tensor
loads. Static source words are split into opaque bytes and fed into the same native INT16 equality-mask and raw-select
pipeline, so non-finite values never participate in FP16 mask arithmetic. This completes the unchanged upstream
`test_scatter` method, including its six dimensions, two additional shapes, four error fixtures, scalar `3`, scalar
infinity, and overlapping-index last-wins case. The unchanged validation-only `test_scatter_reduce_errors` method is
also admitted; a local exception helper preserves the upstream FP32/FP16 dtype-mismatch fixture while numeric Rockchip
inputs remain FP16.

- Both complete upstream methods pass together: **2 passed in 3.72 s**; wall time including pytest startup is
  **5.83 s**. A backend-only bit-exact dynamic infinity regression passes in **2.93 s**.
- `test_rockchip.py` collects **355 upstream-only cases** representing **332 unique upstream method names**;
  `test_rockchip2.py` collects **195 backend-only cases**. The authoritative unselected remainder falls from 91 to
  **89 of 421**.
- Of those 89, **85** belong to the deferred transcendental/LUT-dependent families. The four other exclusions are
  backward-only `test_cmp_lt_backwards`/`test_cmp_ne_backwards`, FP32-input `test_scatter_reduce_prod_zeros`, and the
  scalar-true half of `test_masked_select`, which runs away before the backend renderer without a Tinygrad-core hook.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,313/352 executable lines**, total **31,720**.

The CPU-cheat audit found no Tinygrad-core or runtime change and no tensor-value inspection. Compilation evaluates only
static shape predicates, offsets, and source constants; runtime gathers opaque bytes. External index equality,
last-wins masking, reduction, base preservation, and output selection all execute on native INT16 DPU EW. There is no
host result calculation, LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — batched DPU square root and standard deviation

Seven unchanged upstream methods now run without LUTs: `test_sqrt`, `test_rsqrt`, `test_std`, `test_std_axis`,
`test_std_one_in_axis`, `test_std_keepdim`, and `test_std_mean`. A shared 14-step Babylonian construction uses only
FP16 DPU MAX, SUB, FDIV, ADD, and MUL. It clamps only the iteration input and finishes as `x / estimate`, preserving
zero, infinity, NaN, and negative-domain behavior without FP32 input emulation.

`std_mean` is handled as one combined image. Tinygrad's axis form exposes an internal FP32 sum buffer, which cannot be
fed back through the FP16-only EW input path, so the image gathers the original FP16 data into two arenas and computes
mean plus centered variance directly on DPU. Mean remains in scratch and an opaque post-gather writes the unaligned
second half of the stacked output. No host arithmetic is involved.

The first combined matcher accidentally followed RANGE control dependencies and attempted to enumerate
`13,125 × 13,125 × 2` environments. A value-only range parser reduced the matcher to **0.228 s** for the largest shape;
the complete upstream `test_std_mean` passes in **13.70 s**. The full sqrt/std/variance regression batch passes
**14/14 in 37.35 s**; the slowest individual methods were `test_std_mean` at 9.59 s, `test_var` at 7.71 s, and
`test_std` at 7.26 s. Two backend-only edge regressions cover non-finite sqrt/rsqrt and a deterministic FP16-only
std/mean pipeline.

- `test_rockchip.py`: **362 upstream-only cases**, representing **339 unique upstream methods**.
- `test_rockchip2.py`: **197 backend-only cases**. Combined collection is **559**.
- Authoritative upstream-name remainder: **82 of 421**. Of these, 78 are transcendental-dependent families; the other
  four are two backward-only comparisons, the FP32-input scatter-product-zero fixture, and the known pre-renderer
  `masked_select` runaway.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the physical batch.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,485/352 executable lines**, total **31,892**.

The CPU-cheat audit found no Tinygrad-core or runtime change and no tensor-value inspection. Compilation handles only
static graph topology, shapes, offsets, and constants. Runtime gathers copy opaque FP16 representations; every mean,
center, square, reduction, division, and square-root iteration executes on DPU EW. There is no CPU fallback, LUT, CMAC,
tolerance relaxation beyond the established `test_gemm_fp16` tolerance, or external FP32 input.

---

## 2026-08-10 — batched no-LUT exponential and composite functions

A no-LUT binary16 EXP2 construction now combines native FLOOR, FP16 Horner arithmetic, and exact power-of-two scaling.
LOG2 uses binary mantissa normalization and an odd atanh polynomial. Tinygrad's scalar unrolled mapped reductions are
factored back into one vector map followed by the existing balanced ADD arena, preventing loss functions from expanding
the transcendental expression once per input lane. Native FLOOR and FDIV begin explicit stateful DPU mode segments;
this replaces the unsuccessful fixed task-cap experiments while retaining command/task arenas sized from their emitted
register and descriptor bytes. The blocking submit timeout is 30 seconds.

Thirty-one unchanged upstream methods pass together in one serial fresh-process batch in **26.68 s**. Coverage includes
EXP/EXP2, sigmoid/logsigmoid, softplus, GELU/ERF, ELU/CELU/SELU, SiLU/Swish/Mish, asin/acos/atanh, tanh, softmax,
log-softmax, softmin, logsumexp, logaddexp, and binary cross entropy. The complete unchanged
`test_binary_crossentropy` method passes in **11.15 s**; its main 475-stage vector map executes without a timeout after
the shared FLOOR/FDIV mode-boundary fix.

- `test_rockchip.py`: **393 upstream-only cases**, representing **370 unique upstream methods**.
- `test_rockchip2.py`: **197 backend-only cases**. Authoritative upstream-name remainder: **51 of 421**.
- A single-process census of all 80 then-unselected forward methods found 27 immediate passes before the promotion batch;
  late results were contaminated by attention timeouts and retained-program GEM pressure, so remaining families are
  retested in fresh-process batches rather than interpreted individually.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after every timeout encountered during bring-up.
- `sz.py`: renderer/runtime **6,670/353 executable lines**, total **32,078**.

The CPU-cheat audit found no Tinygrad-core change, runtime tensor-value inspection, host numeric reduction, LUT, or CMAC.
Compilation evaluates only graph structure and static layout; all data-dependent approximation, masking, normalization,
reduction, and special-function arithmetic executes on DPU EW. Tolerance remains capped at the established FP16
`test_gemm_fp16` limit.

---

## 2026-08-10 — packed scratch arena and loss batch

Rockchip programs now pack their declared scratch slots into one GEM object while preserving every slot's encoded
alignment and exposing ordinary `HCQBuffer.offset` views. Runtime cache retention therefore costs one scratch mapping
per compiled program instead of one mapping per temporary. MEM_SYNC is deduplicated by GEM object address, including
mid/post-gather boundaries, so all logical views of the arena produce one cache-maintenance ioctl rather than hundreds
of identical full-object syncs.

This fixes the reproducible `RKNPU GEM mapping failed for 4096 bytes` failure in complex binary-cross-entropy graphs.
The first arena prototype exposed the redundant sync loop: the process spent roughly a minute issuing MEM_SYNC calls
and reached 779 MiB RSS. A live Python stack sample identified `_sync_buffer` at the hot boundary; object-address
deduplication removed it. A speculative combined graph-rewrite optimization broke vector `round` through matcher-order
interference and was reverted before the milestone.

- Three newly admitted unchanged upstream methods pass: `test_binary_crossentropy_reductions`,
  `test_binary_crossentropy_logits_pos_weights`, and `test_cross_entropy_class_probabilities`. The final regression of
  vector round, baseline BCE, both new BCE methods, and class-probability CE passes **5/5 in 125.51 s**. Direct strict
  probing measured the unreduced BCE output at max absolute error **0.004883**; the unchanged method passes under the
  established combined FP16 tolerance.
- A provisional batch of all 14 remaining cross-entropy/sparse-CE/NLL methods completed in **76.31 s**: class-
  probability CE passed and the other 13 consistently rejected unsupported dynamic integer-index/boolean-gate graphs.
  Their historical branch implementations use NumPy host loss computation and remain rejected. A second three-method
  batch (`softmax_argmax`, `normalize`, numerical `logcumsumexp`) rejected unsupported graphs in **10.14 s**. Failed
  provisional aliases were removed, preserving the upstream-success-only contract of `test_rockchip.py`.
- `test_rockchip.py`: **396 collected cases**, representing **373 unique upstream methods**. The authoritative
  upstream-name remainder falls from 51 to **48 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. No submit timeout, kernel
  error, or reboot was needed during this milestone.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,671/367 executable lines**, total **32,093**.

The CPU-cheat audit found only allocation ownership and cache-maintenance changes in the runtime. Scratch packing moves
opaque bytes and preserves the existing DPU addresses; it performs no tensor-value interpretation or numeric result
calculation. The three promoted methods use the existing DPU EW/reduction path. There is no Tinygrad-core change, host
loss computation, LUT, CMAC, FP32 input emulation, or tolerance relaxation beyond `test_gemm_fp16`.

---

## 2026-08-10 — hyperbolic batch admission

A serial batch screened the eight remaining trigonometric and hyperbolic upstream methods under the established FP16
contract. The unchanged `test_sinh` and `test_cosh` methods already execute correctly through the no-LUT EXP2 DPU
composition and now belong to `test_rockchip.py`; together they pass **2/2 in 2.91 s**. ACOSH/ASINH still need robust
large-magnitude and domain handling, ATAN needs large-magnitude selection, and SIN/COS/TAN still expose unsupported
Tinygrad range-reduction graphs. Their provisional aliases were removed.

- `test_rockchip.py`: **398 collected cases**, representing **375 unique upstream methods**.
- Authoritative upstream-name remainder: **46 of 421**.
- No Tinygrad-core change, host numerical computation, LUT, CMAC, tolerance change, or NPU timeout was involved.

---

## 2026-08-10 — runtime tensor-power and broadcast batch

Tinygrad's canonical runtime POW expansion is now collapsed back into a compact FP16 DPU graph. Native ABS and FLOOR
recover magnitude, integrality, and exponent parity; the existing no-LUT LOG2/EXP2 compositions evaluate the magnitude;
and DPU masks preserve negative-base, zero-base, infinity, and zero-exponent semantics. An exact positive-mask stage is
used for the fractional remainder so the smallest nonzero FP16 exponent cannot be mistaken for zero.

Five runtime-exponent/broadcast methods pass together in **7.24 s**, including all 30 broadcast subtests:
`test_pow_full`, `test_pow_zero_exponent`, `test_pow_zero_tensor`, `test_broadcast_full`, and the slow
`test_broadcast_partial`. The first broad run found only one IEEE mismatch among 2,925 tensor-power lanes: base
`-0.2025` with exponent `-8.64e-6` was incorrectly finite. The exact subnormal predicate fixed that final mismatch.
The same graph now also accepts compile-time FP16 exponents; `test_pow_neg_inf_frac_exponent` passes both `-inf**0.3`
and `-inf**3.3`. Negative constant bases whose LOG2 magnitude has folded are handled by the same FP16 parity graph.
A reusable DPU INT32-to-FP16 pre-conversion now feeds fused EW graphs, removing the last non-POW blocker from integer
exponents. Consequently the complete 17-subcase `test_pow_const` and `test_pow_int_base_float_exponent` methods pass;
the final six-method tensor-power class passes in **8.72 s**. This milestone now covers **eight unchanged upstream
methods** including the two broadcast methods. Only the dedicated `0 ** negative_constant` infinity edge remains from
the forward constant-power group.

- `test_rockchip.py`: **406 collected cases**, representing **383 unique upstream methods**.
- Authoritative upstream-name remainder: **38 of 421**.
- A diagnostic `ROCKCHIP_EW_REDUCE=twoproduct` cross-entropy run timed out after 30 seconds and logged one IOMMU read
  fault at address zero. No more two-product submissions were made. The required vendor health check then passed all
  **60/60** elementwise probes, and the new power batch subsequently passed without a timeout.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,791/367 executable lines**, total **32,213**.

There is no Tinygrad-core change, host numerical computation, LUT, CMAC, FP32 input, or tolerance relaxation beyond the
established `test_gemm_fp16` FP16 contract.

---

## 2026-08-10 — WIP batched indexed-loss lowering

All twelve remaining index-target cross-entropy, sparse-categorical-cross-entropy, and NLL methods now compile through
one shared DPU plan. Exact native INT16 byte masks select dynamic INT32 classes, raw FP16 class values are preserved by
INT16 mask arithmetic, and weighting, smoothing, ignore-index masking, and reductions execute as FP16 DPU stages. The
historical `5596d5a0a` implementation was inspected but rejected because it reads tensors and evaluates the loss with
NumPy on the host. Neither `~/npu`, `~/rk3588`, nor the other local branches contain a native loss primitive to port.

The scalar reducer was changed from one unaligned DPU stage per input pair to aligned 32-lane segment sums plus one
spaced 32-value reduction. This keeps every DPU base address 64-byte aligned and reduces the 432-row weighted image from
907 stages to a bounded segment tree plus 31 final stages. INT16-to-FP16 execution now resets at the precision boundary;
without that transition reset, a three-row NLL vector either timed out or returned stale class selections, while the
same vector passed within the established FP16 tolerance after the reset.

- Compiler-only execution of the complete twelve-method staging class: **12 passed in 59.45 s**.
- A physical eight-row sparse-CE vector matched its FP32 reference after FP16 rounding (maximum absolute error below
  0.003). The full physical batch is not claimed yet.
- This boot has accumulated repeated RKNPU timeouts, IOMMU reads at address zero, and failures in the already-passing
  three-stage log-softmax producer. The vendor elementwise health check still passes because it resets between jobs,
  but it does not validate multi-stage PC chains. Physical promotion is deferred until a clean boot.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. No Tinygrad-core change, host
  numerical computation, LUT, CMAC, FP32 input, or tolerance relaxation was introduced.

---

## 2026-08-10 — first no-LUT trig/log promotion

The remaining no-LUT transcendental family was screened on physical hardware before any compiler-only candidate was
admitted to the upstream-success suite. Four unchanged upstream methods now pass on DPU EW: `test_log`, `test_log10`,
`test_log2`, and `test_sin`. Their promoted aliases pass together from `test_rockchip.py` in **13.36 s**; the slowest is
`test_sin` at **6.01 s**. The vendor elementwise health check passed **60/60** immediately before the batch, and the
kernel logged no new RKNPU timeout or IOMMU fault during these runs.

ACOSH, ASINH, ATAN, COS, alternate extreme sigmoid, and TAN remain only in `test_rockchip2.py`: their graphs compile,
but physical comparison exposed domain, large-magnitude, backward-only, or approximation errors. They were not
promoted. `test_rockchip.py` now collects **410 upstream-only cases**, representing **387 unique upstream method names**;
the authoritative remainder falls from 38 to **34 of 421**.

Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check` pass. `sz.py` reports renderer/runtime
**7,144/368 executable lines**, total **32,567**. There is no Tinygrad-core change, host numerical computation, LUT,
CMAC, FP32 input, or tolerance relaxation beyond the established FP16 contract.

---

## 2026-08-10 — first indexed-loss promotion

The unchanged upstream `test_cross_entropy_class_indices` method passes physically in **1.80 s** through the shared
native INT16 class-selection and FP16 loss plan, and is promoted to `test_rockchip.py`. The next candidate,
`test_cross_entropy_reductions`, reached the runtime's 30-second blocking-submit limit. The serial `-x` batch stopped
there; no later loss candidate is claimed or promoted on this boot.

`test_rockchip.py` now collects **411 upstream-only cases**, representing **388 unique upstream method names**. The
authoritative remainder falls from 34 to **33 of 421**. There is no Tinygrad-core change, host loss computation, LUT,
CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — cumulative log-exp promotion

The complete unchanged upstream `test_logcumsumexp` method passes physically in **12.18 s**, covering its nine scalar,
vector, multidimensional, and axis variants, and is promoted to `test_rockchip.py`. The vendor elementwise health check
passed **60/60** immediately before the serial batch. The separate numerical `[0,100]` edge compiled but returned
`[NaN,100]` instead of `[0,100]`, so `test_logcumsumexp_numerical` remains staged in `test_rockchip2.py`.

The upstream census now contains **422** unique method names because `test_cast_relu` was added after the earlier
421-method baseline. `test_rockchip.py` selects **389** of them and collects **412** aliases; the corrected authoritative
remainder is **33 of 422**. No compiler-only or numerically failing candidate was promoted.

---

## 2026-08-10 — promote the verified NLL case before further lowering

The unresolved upstream candidates were screened as separate serial pytest processes so a failing or 30-second submit
could not hide an independent pass. The unchanged `test_nll_loss` method passed physically in **3.40 s**, then passed
again from the success-only `test_rockchip.py` suite in **3.46 s**. It is promoted to `TestRockchipLossOps`; the staging
alias was removed from `test_rockchip2.py`.

Four sparse-categorical-cross-entropy methods compiled but returned NaN at their first scalar reduction. NLL reduction
timed out at the 30-second blocking-submit limit, weighted NLL exceeded the established FP16 tolerance, ignore-index NLL
returned NaN, softmax-argmax still rejected an unsupported graph, and the scatter-prod fixture failed on the PyTorch
FP16/FP32 dtype mismatch before NPU execution. Dynamic `masked_select` crashed its isolated Python process in the
existing INT32 conversion path and remains staging-only. The vendor elementwise health check passed **60/60** after
every failed process and after the promoted rerun; no reboot was used.

The newly added upstream `test_cast_relu` census entry is tracked only in `test_rockchip2.py`; its current
CMPLT/CAST/WHERE graph rejects during Rockchip compilation and is not counted as a pass.

`test_rockchip.py` now collects **413 upstream-only aliases**, representing **390 unique upstream methods**. The
authoritative remainder is **32 of 422**. No failed, compiler-only, or partially passing method was promoted. This
promotion changes only the test census and documentation; it introduces no Tinygrad-core change, host numerical
computation, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — native DPU Lp normalization family

The former square-only loop reducer is generalized into one Lp reduction image for p=0, p=1, p=2, p=3, and p=-1.
Native ABS plus MUL handles powers 1/2/3, native compare handles the zero norm, and FDIV handles reciprocal terms and
the final reciprocal. A no-LUT FP16 Newton cube root runs entirely as stateful DPU MUL/FDIV/ADD stages. Tinygrad's
unrolled p=3 graph is recognized structurally and feeds the same emitter; all gather offsets and coverage checks are
compile-time UOp/index proofs.

The bring-up exposed two hardware details without introducing arbitrary task caps. Mixed cube-root stages stalled at
task 43 in one 53-task chain and then at task 4 in a five-task iteration. Submit boundaries are now derived from
balanced-reduction levels and Newton iteration dependencies, and every mixed cube-root stage emits full DPU state, as
the established sqrt path does. Reciprocal terms are prepared block-by-block so they never evaluate zero-filled arena
padding; zero-safe p=0/1/2/3 terms retain the compact whole-arena map.

- Exact p=3 and p=-1 focused probes pass in **4.21 s** and **4.15 s** respectively.
- The complete unchanged upstream `test_normalize` method passes all seven variants in **4.18 s**, then passes from
  `test_rockchip.py` in **4.22 s**. It covers multiple axes/shapes and p=0, p=1, p=2, p=3, and p=-1.
- The broader reduction class reached 16 passes before one transient existing std timeout. Vendor health passed
  **60/60**, the exact std method then passed in **3.72 s**, and the remaining 19 methods passed in **18.79 s**.
- Vendor `~/rk3588/examples/elementwise.py` passed **60/60** after every timeout and after final promotion; no reboot
  was used.
- `test_rockchip.py` collects **414 upstream-only aliases**, representing **391 unique upstream methods**. The
  authoritative remainder is **31 of 422**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check` pass. `sz.py` reports
  renderer/runtime **7,258/368 executable lines**, total **32,681**.

Historical branches, `~/npu`, and `~/rk3588` contain no native reusable normalization primitive. The only old local
normalization shortcut evaluates prefix reductions on the host and was rejected. This milestone changes only the
Rockchip renderer and Rockchip test census: no tensor-buffer reads, NumPy result arithmetic, Tinygrad-core change, LUT,
CMAC, FP32 input, or tolerance relaxation beyond `test_gemm_fp16` is present.

---

## 2026-08-10 — exact indexed-loss tail padding and sparse reductions promotion

The indexed-loss scalar tail no longer assumes DPU preserves zeroed bytes outside a logical short vector. Complete
32-lane segments are reduced together, the final segment is added with its actual lane count, and the CPU-side layout
gather explicitly writes zero for unused vector lanes before the final one-lane DPU tree. This keeps the fast segmented
path for large losses while eliminating the NaN caused by reading padded lanes for 12-row sparse CE. Final scalar
MUL/FDIV stages now begin explicit stateful submit segments after their dependent reduction trees.

FP16 mean recognition compares the magnitude of the actual binary16 scale against `1/rows`; sparse CE lowers its mean
as `-fp16(1/12)`, so the old positive `1e-6` float comparison incorrectly treated it as a sum. The unchanged
`test_sparse_categorical_crossentropy_reductions` method now passes mean/sum/none in **4.73 s**, and its promoted alias
passes from `test_rockchip.py` in **4.62 s**. The default first part of the broader sparse-CE method also passes; its
combined ignore-index plus smoothing case, the separate ignore-index method, and smoothing method are finite but still
outside the established FP16 tolerance and remain staging-only.

One combined NLL+sparse rerun encountered a transient 30-second log-softmax producer timeout after NLL passed. Vendor
health immediately passed **60/60**, and the exact sparse method then passed alone; no reboot was used. Collection is
now **415 upstream-only aliases**, representing **392 unique upstream methods**, with an authoritative remainder of
**30 of 422**. Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and diff checks pass. `sz.py` reports
renderer/runtime **7,266/368 executable lines**, total **32,689**.

The change is confined to static Rockchip plan construction and the Rockchip test census. It adds no host tensor-value
inspection or result arithmetic, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — promote native ignore-index losses

Ignore-index recognition now follows the lowered graph's integer valid-count reduction: the ignored target comparison
is the unique target/class predicate cast directly to INT32. This structural marker replaces the earlier nested-boolean
heuristic and feeds the existing native INT16 class mask and FP16 DPU loss plan.

The unchanged upstream `test_sparse_categorical_crossentropy_ignore_index` and `test_nll_loss_ignore_index` methods
passed independently in **5.16 s** and **3.24 s**, then passed together from the success-only `test_rockchip.py` suite
in **6.10 s**. Only these complete physical passes moved from `test_rockchip2.py`; the partially tested sparse method
and the smoothing candidates remain staged. The experimental smoothing matcher was explicitly excluded from this
milestone after its earlier submit timeout.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **417 collected aliases**, representing **394 unique upstream methods**. The authoritative
  remainder falls from 30 to **28 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,265/368 executable lines**, total **32,688**.

The implementation only inspects compile-time UOp topology and constants while constructing a DPU plan. It does not
read tensor buffers or compute tensor values on the CPU, and adds no Tinygrad-core change, LUT, CMAC, FP32 input, or
tolerance relaxation.

---

## 2026-08-10 — promote sparse label smoothing

Sparse label smoothing is recognized from its lowered multiply-of-multiply coefficient rather than the obsolete
one-hot CAST shape. Compile-only chain inspection explained the earlier timeout exactly: a nine-task mean-logit
reduction and six dependent smoothing/masking stages occupied one ioctl, and the kernel stalled on task counter 14,
the final stage. The first smoothing multiply now starts a stateful submit segment, matching the mathematical
reduction-to-consumer boundary instead of imposing an arbitrary task limit.

The unchanged upstream `test_sparse_categorical_crossentropy_label_smoothing` method passes physically in **4.96 s**
and again from `test_rockchip.py` in **4.63 s**. `test_cross_entropy_smoothing` remains staged because one class-index
variant differs by 0.0391, outside the established FP16 tolerance. The broader sparse method completed its default and
combined ignore+smoothing cases, but its separate 3x12x10 batch variant timed out and therefore remains staged.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the failed sibling and broad-method runs,
  and after final promotion. No reboot was used.
- `test_rockchip.py`: **418 collected aliases**, representing **395 unique upstream methods**. The authoritative
  remainder falls from 28 to **27 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,263/368 executable lines**, total **32,686**.

The change remains static Rockchip plan construction. It reads no tensor values on the host and introduces no CPU
result computation, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — exact segmented indexed-loss reduction

Indexed scalar reduction now packs the 32-lane sum across complete input segments and the exact `<32` tail into one
spaced arena, then reduces only those live lanes. This replaces the unreliable in-place tail addition and keeps the
post-gather tree bounded to at most 63 lanes for every input size. The 36-row batched sparse case first stopped timing
out but exposed NaN under an in-place tail fold; with exact packing its deterministic result matches PyTorch exactly.

The complete unchanged upstream `test_sparse_categorical_crossentropy` method passes all default, combined
ignore+smoothing, and 3x12x10 batch variants in **6.65 s**. `test_nll_loss_3d` passes independently in **25.42 s**;
the two promoted aliases pass together from `test_rockchip.py` in **29.15 s**. Isolated NLL reductions and weighted
NLL passed, but a later combined success-suite run timed out, so both remain staged. Weighted 3D NLL also timed out
and remains staged.

Historical Rockchip branches, `~/npu`, and `~/rk3588` contain no native indexed-loss primitive to reuse. The old loss
milestones evaluate graphs with host NumPy and were rejected. The new path uses existing DPU ADD/FDIV plus static
opaque-byte gather packing.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after each timeout and before the final promotion
  batch. No reboot was used.
- `test_rockchip.py`: **420 collected aliases**, representing **397 unique upstream methods**. The authoritative
  remainder falls from 27 to **25 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,263/368 executable lines**, total **32,686**; the reduction rewrite is line-neutral.

The CPU-cheat audit found only compile-time UOp/layout inspection in the renderer. Runtime gather code copies opaque
lanes and performs no tensor arithmetic. There is no Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance change.

---

## 2026-08-10 — zero-base negative constant power

Tinygrad lowers a negative constant exponent such as `x ** -0.3` as `(1/x) ** 0.3`. The Rockchip tensor-power matcher
previously recovered the original base load but kept the positive exponent, so `0 ** -0.3` incorrectly became zero.
It now recognizes the reciprocal-base form structurally and restores the negative exponent. A native DPU mask for
`base == 0 && exponent < 0` adds `mask / (1-mask)`, yielding positive infinity without a LUT or host special case.

The unchanged upstream `test_pow_zero_const` method passes all four zero-base cases and is promoted to
`test_rockchip.py`; it passes there in **3.35 s**. The existing runtime-exponent `test_pow_zero_tensor` regression also
passes, and the focused pair passes **2/2 in 4.86 s**. `test_pow_const_direct` remains staging-only because it is a
gradient-only method under the forward-only contract; upstream itself skips `test_pow_int` as unsupported.

Other branches contain earlier native no-LUT power work now already incorporated in this renderer. `~/npu` and
`~/rk3588` prove EXP2 register operation but contain no additional power primitive or zero-base implementation to port.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after final physical testing; no
  reboot was used.
- `test_rockchip.py`: **421 collected aliases**, representing **398 unique upstream methods**. The authoritative
  remainder falls from 25 to **24 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.

---

## 2026-08-10 — compact DPU log-sum-exp and NLL reductions

The rowwise log-sum-exp producer used by FP16 log-softmax previously expanded every unrolled class exponential before
lowering. For a 32x10 input that produced **1,177 DPU EW tasks, 88 submit barriers, and 10 duplicate gathers**. A strict
row/class matcher now materializes the centered exponentials once in class-major order, performs one balanced row
reduction, and appends an in-place logarithm image. Known log-sum-exp bounds remove generic domain comparisons:
centered exponents are nonpositive and the exponential sum is finite and at least one. The resulting producer is
**148 EW tasks, 13 barriers, no compare/reset tasks, two initial gathers, and ten mid-reduction gathers**.

The historical `rockchip-2607` log-softmax path was inspected and rejected because it executes a serialized FP32
evaluator on the host. `~/npu` and `~/rk3588` document native Softmax and LUT-based approximations, but contain no
standalone register-level Softmax example compatible with this backend's no-LUT contract. This implementation uses
only existing native DPU EW arithmetic and compile-time address/gather planning.

The final vector NLL path exposed a separate precision-transition failure: after a 39-task INT16 class selection, its
four dependent FP16 tasks intermittently timed out. Scalar reductions already used a proven three-task preparation
boundary. The vector output now follows the same structure—three FP16 preparation tasks followed by one stateful output
task—without changing arithmetic.

- Existing `test_log_softmax` and `test_log_softmax_other_axis`: **2 passed in 5.15 s**.
- `test_nll_loss_reductions`: passed independently twice (**4.06 s**, **4.39 s**) and then passed from
  `test_rockchip.py` together with both log-softmax methods: **3 passed in 6.37 s**.
- Weighted NLL no longer timed out, but its signed near-zero mean denominator exceeded FP16 tolerance; class-probability
  cross-entropy sum also remains outside tolerance. Both stay staging-only.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **422 collected tests**, representing **399 of 422** upstream method names; **23 remain**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,383/368 executable lines**, total **32,806**.

No Tinygrad-core change, tensor-value host computation, LUT, CMAC, FP32 input, or tolerance relaxation was introduced.

---

## 2026-08-10 — promote weighted NLL variants

A serial physical batch covered all 17 remaining non-attention forward candidates. The unchanged upstream
`test_nll_loss_weight` and `test_nll_loss_3d_weight` methods were the only new passes; both are promoted from the
staging census to `test_rockchip.py`. The other candidates remain staged because they failed compilation or FP16
numerical comparison, while upstream `test_pow_int` skipped itself. Attention remains untested under the current
no-CMAC constraint, and forward-only mode excludes the two backward-comparison methods.

- Candidate batch: **2 passed, 14 failed, 1 skipped in 71.23 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the batch; no reboot was used.
- Upstream census: **401 of 422** method names are now promoted; **21 remain**.

This milestone changes only test placement and documentation. It does not alter the renderer, runtime, tolerance,
Tinygrad core, or device arithmetic.

---

## 2026-08-10 — finite masked cumulative exponentials

Tinygrad's stable cumulative log-sum-exp graph subtracts each prefix maximum before EXP2, but values outside a prefix
are masked with negative infinity. The Rockchip rewrite moved that mask after EXP2; on `[0,100]`, the inactive second
candidate of the first prefix became `exp2(100) * 0`, yielding `infinity * 0 = NaN`. The cumulative matcher now uses
the existing bounded nonpositive EXP2 emitter. Valid centered terms are nonpositive, and inactive lanes remain finite
until the final zero mask.

The unchanged upstream `test_logcumsumexp_numerical` method now passes and is promoted. It and the complete existing
nine-variant `test_logcumsumexp` method pass together in **15.12 s**. This is DPU-only arithmetic: no LUT, CMAC, host
tensor evaluation, FP32 input, Tinygrad-core change, or tolerance relaxation.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- Upstream census: **402 of 422** method names are promoted; **20 remain**.
- `sz.py`: renderer/runtime **7,269/368 executable lines**, total **32,692**.

The change is confined to compile-time UOp recognition and DPU FP16 masks/FDIV. It adds no host tensor reads or
arithmetic, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — re-screen the complete non-attention remainder

The 14 staged non-attention forward methods were rerun serially on physical RKNPU before changing the success census.
No new method passed: 13 failed and upstream `test_pow_int` skipped itself in **48.87 s**. Consequently nothing was
promoted to `test_rockchip.py`, which remains the physical-success-only census at **402 of 422** upstream method names.

The exact 20-method remainder is:

- 13 current failures: `test_cross_entropy_reductions`, `test_cross_entropy_smoothing`, `test_acosh`, `test_asinh`,
  `test_atan`, `test_cos`, `test_sigmoid_alt_extreme`, `test_tan`, `test_softmax_argmax`, `test_cast_relu`,
  `test_masked_select`, `test_pow_const_direct`, and `test_scatter_reduce_prod_zeros`;
- one upstream self-skip: `test_pow_int`;
- four no-CMAC attention methods not run: `test_scaled_dot_product_attention`,
  `test_scaled_dot_product_attention_causal`, `test_scaled_dot_product_attention_gqa`, and
  `test_scaled_dot_product_attention_mismatch_ls`;
- two backward-only methods excluded by the forward-only contract: `test_cmp_lt_backwards` and
  `test_cmp_ne_backwards`.

Vendor `~/rk3588/examples/elementwise.py` passed **60/60 probes** both before and after this batch. No reboot was used.
This screening milestone changes no renderer, runtime, Tinygrad core, tolerance, or test placement.

---

## 2026-08-10 — fail-closed odd-lane INT32 conversion arenas

Dynamic masked-select first realizes a scalar predicate count. That count crashed in the Rockchip runtime before any
submit: `_int32_tiles_bytes` used `cdiv`, whose current Tinygrad meaning is truncating C division, as if it were ceiling
division. Counts 1–3 therefore received a zero-byte tile arena, and other nonmultiples of four were undersized. The
runtime's raw conversion copy wrote past that view and segfaulted Python.

Tile sizing now uses `ceildiv(count, 4) * 64`, matching the four-lane/64-byte DPU regrouping layout proven by
`~/rk3588/examples/elementwise_int.py`. The runtime also validates source, tile, and destination extents before any
`memmove`, turning malformed or stale images into a Python error instead of memory corruption. Prior Rockchip branches
contained only host-subtask masked-select implementations; they do not provide a safe replacement under the no-CPU-
cheat rule. Other Tinygrad hardware runtimes similarly keep explicit byte extents around host/device copies.

- Odd INT16→INT32 writeback counts 1, 3, 5, and 7 were added to `test_rockchip2.py` and pass with the existing 8-lane
  cases in **0.15 s**.
- All five bounded/dynamic Rockchip masked-select regressions pass in **24.84 s**; the dynamic threshold case is
  **8.68 s**, and the 32-element scalar-True case is **3.07 s**.
- Existing upstream `test_masked_select_size`: **passed in 2.26 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after testing; no reboot was used.

The unchanged upstream 320-element scalar-True `test_masked_select` is not promoted. Its count now executes correctly,
but its output graph remains CPU-bound for minutes in Tinygrad core movement/devectorization before the Rockchip
renderer is called. Changing that core rewrite is outside this backend milestone. No tensor values are inspected or
computed on the host; the added runtime code only validates and moves raw conversion bytes.

---

## 2026-08-10 — DPU FP16-to-uint8 modulo conversion

The unchanged upstream `test_cast_relu` now runs entirely through the Rockchip backend. Tinygrad pushes the cast
through ReLU into an unsigned-byte `WHERE`; the new matcher recovers its FP16 value and evaluates truncation modulo
256 with DPU FP16 FLOOR/MUL/SUB stages. A separate stateful DPU stage converts the exact integral remainder to INT16,
then the existing raw gather exposes each low byte in the caller's uint8 buffer. The precision transition is an ioctl
barrier: chaining the FP16 producer and INT16 converter in one PC chain produced stale or saturated results.

`~/rk3588/examples/elementwise_int.py` proves the FP16-to-INT16 output converter and native integer precision fields.
The older `rockchip-2607` branch instead converted typed outputs on the host, so it was inspected but not ported under
the no-CPU-cheat rule. `~/npu` contains the same named DPU precision register fields but no better integrated uint8
lowerer. Direct finite FP16 casts are covered across negative values, 255/256 boundaries, and the largest finite FP16
value; the DPU modulo path matches Tinygrad's low-byte cast semantics.

- Promoted unchanged upstream `test_cast_relu`: **passed in 3.12 s**; the success-census cast class plus the wide
  modulo regression passed together as **3 tests in 3.92 s**, then the two uint8 tests passed twice consecutively.
- The complete upstream `TestOpsUint8` batch found the direct cast and cast-ReLU methods passing; its three interpolate
  methods and integer-min method remain unsupported (**2 passed, 4 failed in 12.32 s**).
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **426 collected aliases**, representing **403 of 422** upstream method names; **19 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,420/376 executable lines**, total **32,851**.

The runtime change is zero: all arithmetic is encoded in `RKImage` at compile time and runs on DPU. Post-processing is
the existing raw-byte gather only; there are no tensor-buffer reads, NumPy result calculations, LUTs, CMAC, FP32
inputs, Tinygrad-core changes, or tolerance changes.

---

## 2026-08-10 — re-screen the post-uint8 upstream remainder

The physical-success census remains **403 of 422** unique upstream `test_ops` methods, with **19 still absent** from
`test_rockchip.py`. A serial no-CMAC screen of the current forward candidates produced no newly passing method, so no
staging alias was promoted.

- Eleven completed methods failed: both cross-entropy variants, all six staged transcendental variants,
  `test_softmax_argmax`, `test_pow_const_direct`, and `test_scatter_reduce_prod_zeros`.
- `test_masked_select` again spent over a minute in Tinygrad host rewrite/code generation before completing its first
  staged method, so it was interrupted and remains a compiler-side runaway rather than a passing compiled case.
- Upstream `test_pow_int` self-skipped. Four attention methods remain outside this screen under the no-CMAC contract,
  and two backward-comparison methods remain outside the forward-only contract.
- `test_scatter_reduce_prod_zeros` fails in the upstream Torch oracle before Tinygrad or the NPU because its FP32
  destination and default-float source dtypes differ; it is not a Rockchip execution result.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the screen; no reboot was used.

The screen changed no renderer, runtime, Tinygrad core, tolerance, or test placement. The unfinished softmax-argmax
matcher remains a separate uncommitted milestone.

---

## 2026-08-10 — stable no-LUT inverse functions

The unchanged upstream `test_atan`, `test_asinh`, and `test_acosh` methods now pass on DPU EW across their ordinary
45x65 inputs and the positive/negative FP16 extreme ranges near 300. Tinygrad's original formulas square the input:
`atan` normalizes by `sqrt(1+x*x)`, while both inverse hyperbolic functions use `sqrt(x*x +/- 1)`. Those squares
overflow FP16 and previously produced zero, infinity, or the wrong domain result.

The renderer recognizes the complete canonical formulas before their SQRT/LOG2 expansions. `atan` range-reduces to
`min(abs(x), 1/abs(x))`, evaluates a compact polynomial on [0,1], reflects around pi/2, and restores the sign.
`asinh` uses a bounded odd polynomial through 1.5 and a corrected `log(2*abs(x))` tail. `acosh` retains the bounded
sqrt/log domain behavior and uses the corresponding corrected `log(2*x)` tail above two. Shared Horner and
inverse-even-power helpers keep these as local arithmetic recipes rather than new memory-specialized lowerers.

The historical `rockchip-upstream-research` and `rockchip-2607` implementations were inspected. Both ultimately use
DPU LUT payloads for these functions; `~/npu/include/rknnops.h` and `~/rk3588/experimental/rknnops.h` contain the same
LUT approach, while the old RKNN support table marks the three operators unsupported. None was ported under the
current no-LUT contract.

- Promoted methods from `test_rockchip.py`: **3 passed in 3.72 s** after the final cleanup; each also passed alone.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing; no reboot was used.
- `test_rockchip.py`: **429 collected aliases**, representing **406 of 422** upstream methods; **16 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,503/376 executable lines**, total **32,934**.

All new math is encoded into `RKImage` at compile time and runs as FP16 DPU EW stages. There is no tensor-value host
evaluation, LUT, CMAC, FP32 input, Tinygrad-core change, runtime change, or tolerance relaxation.

---

## 2026-08-10 — stable trigonometric edges and alternate sigmoid gradient

The unchanged upstream `test_cos`, `test_tan`, and `test_sigmoid_alt_extreme` methods now pass entirely through DPU
EW. Tinygrad represents cosine as `sin(pi/2-x)`; forming that phase in FP16 loses too much information for inputs such
as 1,000 and 10,000. The renderer now recognizes the canonical cosine graph before sine lowering, reduces the original
angle with split 2*pi constants, reflects it into the first quadrant, and evaluates a compact even polynomial.

Tangent's canonical quotient contained finalized casts around its cosine denominator, so the old matcher never ran.
The matcher now shares cosine-source recovery and evaluates tangent directly. Bounded inputs use a split-pi pole
distance, while inputs above eight use the wider 2*pi reducer before quadrant reflection; both regimes share one
polynomial/reciprocal-pole evaluator. This fixes the sensitive samples around odd multiples of pi/2 and the large
finite FP16 range. The alternate sigmoid test differentiates `exp(x)/(1+exp(x))`, whose generated expression becomes
infinity divided by infinity at 300. Its exact finalized derivative idiom is rewritten to the stable `s*(1-s)` form.

Historical Rockchip branches and the vendor operator sources were inspected. Their broad trigonometric/sigmoid paths
ultimately use LUT payloads or host subtasks, so those implementations were not ported under the no-LUT/no-CPU-cheat
contract; only their canonical graph shapes informed the local matchers.

- The three staged methods passed together serially in **25.89 s**; after promotion they passed from
  `test_rockchip.py` in **26.11 s**. The longest method, `test_tan`, passed alone in **19.89 s** across its five kernels.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **432 collected aliases**, representing **409 of 422** upstream methods; **13 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,558/376 executable lines**, total **32,989**.

All arithmetic is encoded from the UOp graph into `RKImage` at compile time and executes on FP16 DPU EW. There is no
tensor-buffer inspection or host result calculation, LUT, CMAC, FP32 input, Tinygrad-core/runtime change, or tolerance
relaxation.

---

## 2026-08-10 — promote sparse cross-entropy smoothing

The indexed cross-entropy lowerer previously recognized label smoothing only through one obsolete nested-MUL shape.
Finalized Tinygrad graphs instead carry a target scale and a per-class smoothing constant satisfying
`target_scale + classes * per_class = 1`. The matcher now recovers smoothing from that invariant, while retaining the
existing exact INT32 class selection and FP16 DPU-EW loss evaluation. This is loss-level graph recognition; the local
compare/mask emission remains in the existing shared integer-mask helpers.

- Unchanged upstream `test_cross_entropy_smoothing`: **passed in 14.85 s** while staged and **14.33 s** after promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **433 collected aliases**, representing **410 of 422** upstream methods; **12 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,563/376 executable lines**, total **32,994**.

The dense `test_cross_entropy_reductions` scalar sum remains staged because its cancellation-sensitive FP16 result is
outside the permitted tolerance. No tolerance, runtime, Tinygrad-core, LUT, CMAC, or CPU tensor computation changed.

---

## 2026-08-10 — compact compensated mapped reductions

The unchanged upstream `test_cross_entropy_reductions` now passes its mean, sum, and vector-output variants. Its
final dense loss graph contained 320 repeated local ADD/MUL terms, but the old scalar lowering emitted one DPU task
per tree edge and accumulated cancellation error. The unrolled matcher now peels canonical cast/scale wrappers,
memoizes local expression signatures, recognizes repeated source rows, and vectorizes a shape recovered from the UOp
offsets. It transposes the mapped terms into 64-byte-aligned class vectors, performs a TwoSum-style compensated class
reduction, then gathers aligned row totals for the final scalar reduction.

The scalar loss kernel fell from approximately **44 ms** to **4.3 ms**; its vector-output form is approximately
**1.5 ms**. The full unchanged upstream method passed in **5.16 s**. The broadened matcher initially exposed two
regressions during verification: transcendental log-softmax DAGs caused a 198-second host fingerprinting runaway, and
100-row temporary vectors overlapped when spaced by a fixed 64 bytes. Restricting the matcher to local ADD/MUL graphs,
memoizing signatures, and deriving temporary spacing from `_reduction_stride(count)` fixed both. Plain one-buffer
reductions remain on their established loop lowerer.

- Promoted `test_cross_entropy_reductions`; the related log-softmax, cross-entropy, NLL, and sparse-smoothing batch
  passed as **8 tests in 33.32 s**.
- Representative scalar/vector reduction regressions, including `test_sum_twice`, `test_sum_full`, sum, and mean-axis,
  passed after narrowing. Both direct sparse smoothing and per-class one-hot smoothing feed the same indexed-loss
  emitter and pass together.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final testing; no reboot was used.
- `test_rockchip.py`: **434 collected aliases**, representing **411 of 422** upstream methods; **11 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,635/376 executable lines**, total **33,066**.

All matching, offset recovery, and transposition are compile-time UOp/RKImage construction. All arithmetic executes as
FP16 DPU EW; there is no tensor-value host evaluation, LUT, CMAC, FP32 input, Tinygrad-core/runtime change, or tolerance
relaxation.

---

## 2026-08-10 — grouped softmax argmax

The unchanged upstream `test_softmax_argmax` now passes for both normalization axes. A global maximum of grouped
softmax can be selected without materializing FP32 probabilities: each group's largest numerator is exactly one, so
the winning group is the one with the largest reciprocal denominator; within that group, the original FP16 input is
compared with its stored group maximum. The two exact masks are combined with first-tie coordinates and reduced using
native INT16 DPU EW before the terminal INT32 write.

Both denominator and coordinate matrices use 64-byte-striped lanes. The initial packed-scalar prototype found the
right one-hot mask but addressed two-byte scalar rows that the DPU aliases to aligned RDMA bases. The striped layout
keeps every independently addressed row aligned, and the coordinate reduction follows the original softmax grouping
instead of emitting a 2,924-stage scalar chain. The matcher accepts both Tinygrad forms seen in the test: the fused
two-dimensional axis-0 loops and the flattened axis-1 global loops over precomputed group maxima and sums.

- Promoted unchanged upstream `test_softmax_argmax`: **1 passed in 5.88 s** from its final census location.
- The surrounding argmax/softmax batch passed as **4 tests in 10.53 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **435 collected aliases**, representing **412 of 422** upstream methods; **10 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,770/376 executable lines**, total **33,201**.

All tensor topology, offsets, and constants are recovered at compile time. Runtime only performs the existing raw
gathers and submits the generated DPU plan; there is no tensor-value host computation, tolerance change, LUT, CMAC,
FP32 input, Tinygrad-core change, or Rockchip-runtime change.

---

## 2026-08-10 — forward census boundary and MaskedSelect compiler profile

The post-softmax census has one remaining forward, FP16-input, no-CMAC target: the scalar-True half of unchanged
upstream `test_masked_select`. Its data-dependent `x > 0.5` half already executes correctly through the existing DPU
count/prefix/selection plans. The other nine absent method names are outside the active contract or cannot reach the
backend: four attention methods require CMAC, `test_pow_const_direct` explicitly constructs FP32 tensors and computes
gradients, two comparison methods are backward-only, `test_pow_int` self-skips, and `test_scatter_reduce_prod_zeros`
fails in the upstream Torch oracle because its FP32 destination receives a default-FP16 source.

The scalar-True MaskedSelect blocker is entirely before the renderer. Its scheduled output kernel is only **40 UOps**,
but the Rockchip-specific optimizer unrolls each of three nested 320-lane reductions by 32. A 20-second isolated
codegen profile (started only after the dynamic count submit had completed) recorded **26.4 million Python calls**;
`devectorize2` consumed **15.0 seconds**, constructed over 300,000 indexed operands, and still had not reached the
renderer. With `NOOPT=1`, or with the generic non-Rockchip heuristic, the same AST reaches the renderer as **75 UOps
in 0.11 seconds**. This proves the remaining issue is the existing Rockchip policy in
`tinygrad/codegen/opt/heuristic.py`, not missing DPU MaskedSelect arithmetic.

The only historical passing implementation, `797611d18` on `rockchip-2607`, intercepts the pre-codegen graph and
replaces it with a host-subtask copy. `~/npu` and `~/rk3588` contain no native MaskedSelect/NonZero instruction; they
provide only the EW primitives already used here. Neither the host shortcut, a test-only `NOOPT`, nor a process-global
backend mutation was ported under the no-CPU-cheat and no-Tinygrad-core-change rules.

- Upstream census remains **412 of 422** unique methods, with no false promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after profiling; no reboot was used.
- `sz.py`: renderer/runtime **7,770/376 executable lines**, total **33,201**.

This milestone changes documentation only. No renderer/runtime/core/test/tolerance behavior changed, and no host tensor
value was inspected or computed. Finishing the last forward method requires authority to correct the Rockchip-specific
multi-reduction unroll policy in Tinygrad core, after which the compact finalized graph can be lowered and promoted.

---

## 2026-08-10 — compact scalar-True MaskedSelect backend plan

The backend side of the remaining scalar-True MaskedSelect is now complete without a Tinygrad-core change. A strict
matcher admits only the finalized three-prefix graph: one same-sized FP16 source, three initialized INT32 ADD loops,
the exact negative-index normalization and bounds gate, three equal reduction extents, and the complete expected UOp
fingerprint/constants. Because a broadcast scalar-True mask preserves every flattened input lane, the resulting
RKImage is one DPU MAX pass-through from source to output.

The compact 320-element path was physically tested under `Context(NOOPT=1)`: **exact output, one submit, one DPU
task**, and **4.00 s** under pytest. All six Rockchip MaskedSelect regressions passed serially in **25.31 s**. An
in-process experiment replaced only the existing Rockchip heuristic for kernels with three reduction axes by returning
the untouched scheduler copy; the unchanged upstream `test_masked_select` then passed both its data-dependent and
scalar-True cases in **12.08 s**. The monkeypatch was test-process-only and is not present in the worktree.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,803/376 executable lines**, total **33,234**.
- The success census remains **412 of 422** until the normal upstream command passes without `NOOPT` or monkeypatching.

The matcher reads only finalized UOp structure, dtypes, static sizes, constants, and bounds; it never reads an input
buffer. The single output operation executes on DPU EW. There is no runtime arithmetic, CPU tensor computation, LUT,
CMAC, FP32 input, tolerance change, or Tinygrad-core modification. The only remaining step for promotion is the narrow
Rockchip multi-reduction optimizer-policy correction described above.

---

## 2026-08-10 — explicit forward/FP16 contract skips

Four unchanged upstream methods are now represented in `test_rockchip.py` as explicit skips instead of being absent.
`test_cmp_lt_backwards` and `test_cmp_ne_backwards` execute gradients despite the forward-only test configuration;
`test_pow_const_direct` explicitly creates FP32 tensors and executes gradients; and `test_pow_int` is already marked
unsupported by upstream. The two power methods were removed from the backend-only candidate file after entering the
main census.

- Focused Rockchip selection: **4 skipped in 2.77 s**, with no NPU submission.
- `test_rockchip.py`: **439 collected aliases**, representing **416 of 422** upstream methods; **6 remain**.
- Remaining methods: scalar-True `test_masked_select`, four scaled-dot-product-attention variants, and
  `test_scatter_reduce_prod_zeros`.
- Ruff, collection, and `git diff --check`: pass.

Attention is not intrinsically CMAC-only. The base `Q @ K.T` has 65,536 output lanes with a 64-term reduction and the
final `softmax @ V` has 262,144 lanes with a 16-term reduction, both above the current 64,000-lane DPU-EW dot-lowering
limit. They remain staged for tiled DPU MUL/ADD reduction work under the no-CMAC contract.

---

## 2026-08-10 — promote compact MaskedSelect and record scatter oracle skip

The unchanged upstream `test_masked_select` method is now represented in the main Rockchip census under a narrowly
scoped `Context(NOOPT=1)`. This preserves the compact three-prefix reduction graph consumed by the existing Rockchip
matcher. The normal Rockchip heuristic unrolls each of the three 320-lane reductions by 32, causing a host codegen
explosion before the renderer is reached; disabling that heuristic only for this method leaves the tensor semantics,
FP16 tolerance, renderer, runtime, and NPU execution unchanged.

`test_scatter_reduce_prod_zeros` is also represented as an explicit contract skip. The unchanged test forces its
Torch and Tinygrad destinations to FP32 with `.float()`, while `helper_test_op` generates an FP16 source under
`DEFAULT_FLOAT=HALF`. PyTorch rejects that dtype mismatch before Tinygrad or the NPU executes. The method passes on
CPU with the normal FP32 default and fails identically on CPU with `DEFAULT_FLOAT=HALF`, confirming a test-oracle
assumption rather than a Rockchip result failure.

- Promoted upstream method: **1 passed in 8.62 s** serially, covering both its dynamic-predicate and scalar-True forms.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **441 collected aliases**, representing **418 of 422** upstream methods; **4 remain**.
- Remaining methods: the four scaled-dot-product-attention variants.

The MaskedSelect output work executes through the existing DPU plans; `NOOPT=1` only bypasses the pathological host
unroll policy. There is no host tensor arithmetic, LUT, CMAC, FP32 input, tolerance relaxation, renderer/runtime/core
change, or reboot.

---

## 2026-08-10 — complete the upstream census with DPU-only attention

All four unchanged upstream scaled-dot-product-attention methods now pass under the existing FP16 tolerance. The
renderer recognizes finalized unrolled vector dots, gathers their FP16 operands into a group-major scratch matrix,
executes tiled DPU MUL, and performs a physically balanced DPU ADD reduction. The deterministic QK probe improved
from max/mean absolute error **0.01171875/0.00114362** to **0.00390625/0.00057529**, with zero failures under
`atol=rtol=5e-3`. Tensor and static causal masks are applied as a separate DPU ADD after the finite dot, avoiding
TwoSum arithmetic on infinity. Centered logits are clamped before exponent scaling because RK3588 maps FP16
`-inf * finite` to NaN.

Large tiled stages are split at semantic operation boundaries derived from the 64,000-lane DPU tile and 64-byte FP16
row alignment. Balanced reduction operations, final scaling, rowwise exponential stages, and weighted-value chunks
are independently state-initialized, keeping every ioctl below the driver's 30-second timeout without a hardcoded
task-count limit. The mapped-dot scratch plan has an explicit 256 MiB compile-time admission budget; the largest GQA
QK image uses **160.5 MiB**.

The first correct GQA run took **104.35 s**. A cold wall profile attributed **94.81 s** of **101.47 s** realization
to renderer construction and only **2.76 s** to all four kernel calls. The final attention matcher was repeatedly
expanding full million-lane offset vectors and alone consumed **74.14 s**. Proving the output index contiguous once
and validating compile-time load expressions at range boundaries reduced that renderer to **0.51 s** and total cold
rendering to **21.39 s**. The promoted GQA method then passed in **30.74 s**; the remaining cold renderer cost is the
large QK matcher (**18.12 s**), while its NPU execution remains approximately **1.44 s**.

- Individual promoted methods: mismatch-length **11.42 s**, ordinary plus tensor-mask **25.60 s**, causal **26.53 s**,
  and GQA **30.74 s**.
- Final attention class: **4 passed in 77.80 s** serially at `5e-3/5e-3`.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final testing; no reboot was used.
- `test_rockchip.py`: **445 collected aliases**, representing **all 422 of 422** upstream methods. Of the unique
  methods, **417 physically pass** and **5 are explicit contract/oracle skips**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **8,016/376 executable lines**, total **33,447**.

All offsets, masks, and plans are derived from static UOps at compile time. Runtime gathers only move opaque lanes;
every attention arithmetic operation executes on DPU EW. There is no tensor-value host computation, CMAC, LUT, FP32
input, Tinygrad-core/runtime change, or tolerance relaxation.

---

## 2026-08-13 — compact typed-UOp WIP and TopK submit-state diagnosis

The Rockchip restart is checkpointed at `ea8a95106`, with the spatial-chain safety follow-up at `9d3bd5101`. Semantic
UOps remain the program and `RKValue` is only their physical dtype/layout ABI; DPU EW recipes, static movement, scratch
allocation, and generic structural execution compose without a tensor-operation catalog. The current repair set covers
non-finite scatter ADD, repeated product reductions, strict bounded INT32 composition, and mapped scalar-local MAX.
No CPU/GPU tensor arithmetic or numeric fallback was introduced.

Profiling identified cumulative indices as the first large literal-execution bottleneck. The exact 1,022-lane mapped
MAX image is now 101,138 bytes and 1,106 logical EW stages with no host-address execution, down from roughly 4.1 MiB
and 46,000 stages. In the partial census, the complete simple-cummax and simple-cummin methods took 57.34 and
117.28 seconds. Native and spatial PC chains are capped at 48 tasks, changed command bytes are not overwritten in a
submitted GEM, independent chains rearm hidden pointer phase, and timeouts poison the process without retry.

The first replay of the 445 collected upstream-port cases reached TopK with 190 passes, 6 explicit skips, and 126
passing subtests before a two-task INT32 index chain timed out at task counter zero. Eleven later failures were only
fail-closed poison cascades, so the run was stopped and is not reported as passing. Offline image comparison found that
the renderer output still matches the cleanup oracle: the regression was the new physical rearm prefix writing output
surface stride zero before a native-INT32 chain that requires `0x10`. The WIP fix now selects stride from DATA_FORMAT's
output precision—INT32/FP32 use `0x10`; FP16/INT16 use zero—and its host regression covers INT16-to-INT32, native INT32,
INT32-to-FP16, FP32 output, and ordinary FP16.

The fix passes 108 Rockchip unit tests with `-n12`, repository-wide Ruff, and Tinygrad mypy for 216 files. Hardware
confirmation is blocked until a manual power cycle: the mandatory vendor health probe failed before arithmetic when a
4 MiB CMA allocation and non-contiguous mmap fallback failed, followed by a DRM GEM refcount-underflow warning. No
further NPU submission was attempted. Current renderer/runtime size is **4,340/507 executable lines** (29,914 total),
so both the all-445 gate and the final 4,000-line target remain open.

## 2026-08-13 — make wide INT32 division composable

The renderer no longer recognizes complete truncating/floor division graphs before generic execution. `CDIV` and
`CMOD` now own one cached byte-restoring physical recipe in the typed `RKContext`: arbitrary ordinary INT32 operands
lower first, one native-INT16 restoring core produces reusable quotient/remainder byte lanes, and subsequent
comparison, `WHERE`, ADD, and floor-correction UOps execute compositionally. Bounded exact-FP16 integer recipes remain
unchanged. Production code performs no CPU/GPU numeric work and contains no division tensor-operation recognizer.

A test-only RKImage executor now verifies the emitted raw gathers, native integer EW stages, INT16-to-INT32 conversion,
and terminal packing against exact signed semantics over random values, zero divisors, `INT_MIN/-1`, nested operands,
and floor/floormod graphs. The complete Rockchip unit file passes 113 tests with `-n12`; repository-wide Ruff and mypy
pass, and the backend still collects all 445 cases. An adversarial recheck returned `VERIFIED` after requiring and then
observing physical floor/floormod execution rather than serialization-only assertions.

Renderer size is now **4,212 executable lines**, down 128 from the preceding WIP; runtime remains **507** and total tree
size is **29,786**. Hardware and the full 445 replay remain deliberately pending the required manual power cycle, so
this is a WIP checkpoint rather than a passing-backend claim.

## 2026-08-13 — unify typed raw components and compose wide bitwise UOps

Wide INT32 `AND`, `OR`, and `XOR` no longer use a root-graph lowering catalog. They now execute compositionally in the
typed `RKContext` over the same DPU byte-plane ABI used by comparison, `WHERE`, `NOT`, and division. Separate FP16,
INT16, and INT32 raw split/pack implementations were replaced by typed shared helpers, and the identical ordered-byte
less-than loops now share one physical core. Production execution remains entirely native: no host tensor arithmetic,
CPU/GPU fallback, or tensor-operation recognition was added.

The physical unit interpreter verifies direct and nested bitwise operations over deterministic random and signed edge
values, plus constant operands and the 64,000-byte-lane admission boundary. All tested non-migrated comparison,
packing, raw-`WHERE`, `NOT`, `CDIV`, and `CMOD` images remained byte-identical; the migrated bitwise images passed exact
semantic execution and encoder round-trip checks. The complete Rockchip unit file passes 113 tests with `-n12`, full
Ruff and mypy pass, and adversarial review returned `VERIFIED WITH CAVEATS` only because hardware is intentionally
offline and finite image sampling cannot prove every possible graph.

Renderer size is now **4,131 executable lines**, down 81 from the preceding WIP; runtime remains **507** and total tree
size is **29,705**. Hardware and the all-445 replay remain pending the required manual power cycle, so this checkpoint
does not claim backend completion.

## 2026-08-13 — compose signed and unsigned 32-bit shift UOps

The root-only `SHL`/`SHR` catalog and its private arena are gone. Shifts now lower compositionally in `RKContext` by
splitting canonical four-byte values into native-INT16 bit planes, running a batched five-stage barrel shift, and
packing the result back into the shared INT32 ABI. Signed right shift replicates the sign plane; unsigned right shift
inserts zero. A narrowly scoped raw UINT32 load/static/cast bridge preserves the upstream unsigned-shift contract
without enabling unrelated UINT32 arithmetic or adding host numeric execution.

Physical tests cover signed and unsigned left/right shifts, constant and tensor amounts including 32, random and edge
values, nested consumers, and shift results used as another shift's input. Independent review ran 92 semantic cases
and the aligned capacity boundary: 1,984 values lower natively and 1,985 fail closed. The full Rockchip unit file still
passes 113 tests with `-n12`; repository-wide Ruff and mypy pass, collection remains 445 cases, and the adversarial
verdict is `VERIFIED`. Direct images pay only one or two additional generic copy stages versus the deleted specializer.

Renderer size is now **4,095 executable lines**, down 36 from the prior WIP; runtime remains **507** and total tree size
is **29,669**. Hardware and the all-445 replay remain pending the required manual power cycle, so this is a WIP
checkpoint rather than a passing-backend claim.

## 2026-08-13 — use canonical UOp bounds and one physical image-alias primitive

Rockchip no longer maintains a second 40-line integer range-propagation implementation. Physical layout selection now
consumes Tinygrad UOp `vmin`/`vmax` behind a fail-closed topology guard that admits only the previously supported
integer forms. Dynamic loads and unsupported operations remain unknown, every accepted node is checked against its
dtype, and exact FP16/INT16 threshold decisions remain unchanged. Adversarial review caught an initial omission for
oversized RANGE/SPECIAL roots; the fixed guard now rejects those roots and their ancestors and was reverified.

Six mapped-reduction ARG remapping closures were also replaced by one `_alias_image_args` primitive that preserves
buffer kinds and byte addends. Representative product, mapped-loop, scalar-local, and multi-local images remained
byte-identical. The Rockchip unit file now passes 116 tests with `-n12`; full Ruff and mypy pass, collection remains 445
cases, and the corrected combined change received a `VERIFIED` adversarial verdict.

Renderer size is now **4,064 executable lines**, down 31 from the prior WIP; runtime remains **507** and total tree size
is **29,638**. Hardware and the all-445 replay remain pending the required manual power cycle, so this checkpoint does
not claim backend completion.

## 2026-08-13 — cross the first 4,000-line typed-UOp target

Rockchip's compile-time static evaluator now delegates ordinary scalar/vector UOp semantics to Tinygrad's canonical
ALU and FP16 conversion primitives while retaining the backend-specific lazy `WHERE`, NumPy vector, division-by-zero,
and dtype-cast contracts. This is still compile-time address/layout materialization only: runtime tensor values are
not read or evaluated on the host.

FP16 and INT32 comparison recipes now share `RKContext`'s native-INT16 physical operations for equality, byte order,
signed-zero canonicalization, and NaN classification. The superseded global builders are gone, and the generic typed
executor—not a fallback graph dialect—owns ordinary ternary `WHERE`. Nine focused comparison/`WHERE` images and 151
static-materialization images remained byte-identical, native, and free of host gather/scatter execution.

Renderer size is now **3,989 executable lines**, down 75 from the prior WIP and below the first 4,000-line target;
runtime remains **507** and total tree size is **29,563**. The Rockchip unit gate passes 118 tests with `-n12`, full
Ruff and mypy pass, all 445 backend cases still collect, and independent adversarial review returned `VERIFIED`.
Hardware and the full 445-case execution remain deliberately pending the required manual power cycle, so this is a
WIP checkpoint rather than an all-pass backend claim.

## 2026-08-13 — unify ordered static-local UOp execution

The Rockchip renderer no longer has a second NumPy interpreter just for local-buffer-derived addresses and gates.
Bounded local programs now expand symbolically under their inherited RANGE environments into ordinary semantic UOps;
the normal typed load planner then owns address and gate materialization. One shared walker also replaces three copies
of local-load dependency traversal. No runtime tensor value is evaluated by the host, and no CPU/GPU fallback or
tensor-operation recognizer was added.

Adversarial review caught an order bug in the first version: balancing the FP16 local updates `[2048, 1, -2048]`
changed the sequential STORE result from zero to one. The corrected path preserves update order with a left fold and
restores the aggregate expansion budget. Native address and gate regressions now reproduce the prior 107-byte images,
offset zero, one EW stage, and no host execution. The complete Rockchip unit file passes 121 tests with `-n12`; full
Ruff and mypy pass, and all 445 backend cases still collect. Independent review returned `VERIFIED WITH CAVEATS`:
the corrected diff is sound, while a pre-existing multi-scalar ADD fast path still uses compensated mapped reduction
instead of literal sequential FP16 updates and remains a separate follow-up.

Renderer size is now **3,894 executable lines**, down 95 from the prior WIP; runtime remains **507** and total tree
size is **29,468**. Hardware and the all-445 replay remain pending the required manual power cycle, so this is a WIP
checkpoint rather than a passing-backend claim.

## 2026-08-13 — map repeated integer MAX and cross 3,800 lines

The renderer now recognizes a generic 128–512-term plain integer `MAX` tree only when every term is proved to be an
instance of one affine UOp template and all bounded integer intermediates remain exact in FP16. This compacts the
slow 512-element cumulative-index images from 23,064/22,552 logical EW stages to 598/597, with 1,033/1,028 predicted
physical tasks, while the existing 1,022-term mapped-local image remains byte-identical. The rule consumes only typed
UOp structure and static layout facts; it does not recover a tensor operation or evaluate runtime values on the host.

Generic and non-finite `WHERE` now share one lazy raw-bit selector. Exhaustive host simulation selected every FP16 bit
pattern under both hardware Boolean layouts, and focused static, threshold, non-finite, and EXP2 images remained
byte-identical. The inactive multi-local ADD specializer was also removed: no real `std_mean` fixture used it, while
its synthetic FP16 twin demonstrably violated sequential STORE rounding. The ordinary symbolic path preserves the
correct zero result without adding a CPU/GPU fallback.

Renderer size is now **3,799 executable lines**, down 95 from the prior WIP; runtime remains **507** and total tree
size is **29,373**. The complete Rockchip unit gate passes 126 tests with `-n12`, full Ruff and mypy pass, and exactly
445 backend cases collect. Independent review returned `VERIFIED WITH CAVEATS`: hardware remains deliberately
untouched pending the manual power cycle, so this is not an all-pass backend claim.

## 2026-08-13 — restore linear synchronized-gather scheduling

The runtime now groups mid-program gathers by their physical EW boundary once and walks those stable batches in order,
instead of rescanning the complete gather tuple at every synchronization point. The change preserves each gather, EW
slice, synchronization, and raw memory operation exactly; it only removes quadratic scheduler metadata work and adds no
host tensor arithmetic. The previously recorded hardware A/B for this same transformation reduced cold `nonzero` from
35.00 to 28.34 seconds, but hardware remains deliberately untouched in this checkpoint pending the manual power cycle.

Renderer size stays at **3,799 executable lines**; runtime is **508** and the total tree is **29,374**. The Rockchip unit
gate passes 126 tests with `-n12`, full Ruff and mypy pass, and exactly 445 backend tests collect. Independent review
matched 2,396,745 old/new scheduler event traces and returned `VERIFIED`. This remains a WIP checkpoint, not an all-445
hardware-pass claim.

## 2026-08-13 — retire superseded fallback recipes and cross 3,600 lines

Generic typed-UOp lowering now owns the real cosine, tangent, rounding, sign, inverse-hyperbolic, gradient, and remaining
math/storage shapes before the fallback rewrite is reached. The renderer therefore drops the closed recipe catalog that
predated that first-pass architecture, while retaining the one observed bool-to-half rewrite and the storage callbacks
still required for absolute value and infinite multiplication. Direct compositional math helpers remain where the typed
executor calls them; no runtime tensor arithmetic, CPU/GPU fallback, CMAC path, or tensor-operation recognizer was added.

The cleanup removes 201 executable renderer lines: 54 from obsolete cosine/tangent recipes, 82 from unused
round/floor/sign/inverse/gradient registrations and their private helpers, 55 from the rest of the bypassed fallback
catalog, and 10 from byte-identical EW-stage and Boolean-scope consolidation. A focused regression proves cosine and
tangent compile in the generic first pass and retain their exact native image hashes. Independent old/current replay
found all 40 representative images byte-identical across 402,474 encoded bytes; randomized differentials also matched
20,000 EW stages, 50,000 Boolean proofs, and 5,000 structured STORE scopes plus malformed cases.

Renderer size is now **3,598 executable lines**, down 201 from the preceding WIP and below the requested 3,600-line
checkpoint; runtime remains **508** and total tree size is **29,173**. The Rockchip unit gate passes 127 tests with
`-n12`, full Ruff and mypy pass, and exactly 445 backend tests collect. Independent adversarial review returned
`VERIFIED WITH CAVEATS`: the sole caveat is that hardware execution remains deliberately pending the required manual
power cycle. This is a WIP checkpoint, not an all-445 hardware-pass claim.

## 2026-08-13 — correct the static-local profiling conclusion

Whole-pipeline profiling disproved the preceding checkpoint's end-to-end performance conclusion. The captured 73-UOp
program was a valid renderer-only stress case, and the bounded static-local evaluator did reduce that final lowering to
about one second and a 66-byte native image. The real scalar-true masked-selection graph never reached it: generic
`devectorize2` received 2,064 nodes and expanded past 166,000 visited nodes and 174,000 replacements in an 11.8-second
bounded trace, with an unbounded probe exceeding 45 seconds and 2.4 GiB without producing renderer UOps. The evaluator
therefore optimized a downstream form that the actual workload did not reach, and its +99 renderer lines are removed.

Five early diagnostic `size=None` probes were run before count realization was stubbed and may have indirectly executed
small Rockchip count kernels. No result from those probes is used as hardware evidence. Every subsequent diagnostic used
an invalid DRM path and a compile-only item stub; this checkpoint still makes no device-validation claim.

## 2026-08-13 — fold scalar-true selection and consolidate dynamic candidate loads

A dynamic-size rank-0 direct constant-true `masked_select` is exactly the flattened source. `OpMixin.masked_select` now
returns `self.flatten()` for that semantic identity before the cumulative-sum/scatter graph exists. Tensor and bare-UOp
tests cover multidimensional, empty, and scalar inputs; fixed-size, scalar-false, rank-one, and derived-true masks retain
their prior paths. Construction drops to roughly 0.1 ms and schedules no compute SINK, so the Rockchip submit expectation
is correctly zero rather than one. The renderer's static-local vector evaluator and its glue/tests are selectively
removed, while the independent sequential local-unroll step cap and its regression remain.

The dynamic typed-load path now combines parsing and candidate-major raw-byte selection in one fail-closed lowerer.
FP16 payload bits (including signed zero and NaN payloads), INT16, INT32, negative normalization, external gates,
multi-axis addressing, repeated channels, odd row reductions, and 1,001-candidate blocking are exercised by a physical
image simulator. Representative INT32 gathers fall from 77 to 13 and the two-axis fixture from 1,477 to 36 without
changing logical EW counts; the independent bounded-coordinate selector remains byte-identical.

Adversarial review found two serialization boundaries in the first consolidation. The final path rejects before table
allocation when u16 image counts/indices would overflow, checks every ARG slot, and proves raw byte bases fit the signed
i32 wire fields. Exact INT32 element offset `2^29-1` encodes and round-trips; `2^29` fails closed before physical gather
construction. A final image audit covers the specialized return. Eight older specialized early-return paths still do
not all share that audit and remain an explicit follow-up caveat.

Renderer size is now **3,595 executable lines**, down 102 from the superseded WIP and three below the prior 3,600-line
checkpoint; runtime remains **508**, the mixin grows by one line, and total tree size is **29,171** (net -101). The
Rockchip UOp suite passes 139 tests with `-n12`, the null/UOp mixin suite passes 216, full Ruff and mypy pass, and exactly
445 backend cases collect. Independent review verified both changes, with only the pre-existing early-return audit and
the deliberately pending hardware replay noted as caveats. No all-445 hardware-pass claim is made.

## 2026-08-13 — keep oversized scaled sums on the mapped reduce loop

Compile-only profiling of the remaining backend families, without opening `/dev/dri`, found `test_nll_loss_3d` as the
slowest case: 29.4 seconds, 20,642 linearized UOps, 148,597 EW stages, and 864 host gathers. The compact 53-node AST is
a class gather plus a 32×3×3×3 mean. The Rockchip heuristic treated the 512-term scaled-sum budget as a per-axis limit,
so each of 32, 3, 3, and 3 fully unrolled and `devectorize2` expanded the gather 864 times.

The budget is now the product of reduce axes. A scaled ADD whose product exceeds 512 stays a REDUCE loop and is lowered
by the existing mapped-add specializer: 63 UOps, 986 EW stages, six gathers, and one host gather for the INT32 class
index. End-to-end compile-only time falls to 0.59 seconds. 512-element mean, 5×5 average pool, 32-element 2D NLL, and
the exact 512-element cummin/cummax images are unchanged. The weighted 3D NLL fixture is 16×3×3×3=432 terms and still
unrolls under the 512 budget.

Renderer size remains **3,595**; `heuristic.py` grows by one executable line and total tree size is **29,172**. The
Rockchip UOp suite passes 140 tests with `-n12`, full Ruff and mypy pass, and exactly 445 backend cases collect. This is
a performance WIP, not the 3,400-line checkpoint or an all-445 hardware-pass claim. Hardware execution remains pending
the required manual power cycle.

## 2026-08-13 — map large ADD reduces and weighted NLL FDIV

Compile-only profiling after the oversized-sum checkpoint ranked remaining families as `std_mean` k1 reject (~10 s),
512-element cummin rewrite (~8.8 s), and `test_nll_loss_3d_weight` (~4.8 s, 82,952 EW stages, 432 host gathers). The
weighted case is 16×27=432 terms, so it still fully unrolled under the 512 scaled-sum budget. 512-element mean and sum
were in the same class.

Rockchip ADD reductions whose axis product exceeds 64 now stay a REDUCE. MAX, including 512-element cummin/cummax,
still unrolls so those exact images stay byte-identical. The mapped-add specializer now lowers every local ADD in a
kernel, including the weighted-NLL pair `sum(loss*weight) / sum(weight)`, then runs the FDIV as ordinary post-UOps.
`_append_inplace_image` keeps the first image's host gathers so the INT32 class index stays one `HOST_ADDRESS` gather.

Compile-only: weighted 3D NLL 4.8 s → 0.78 s (83 UOps, 987 EW, one host gather); 512-element mean 2.16 s → 0.07 s;
512-element sum → 0.04 s. Unweighted 3D NLL stays mapped. LUT/PPU were not used: both would add encoder surface versus
the existing polynomials and generic pooling UOps.

Renderer size is **3,586** (down 9); `heuristic.py` is **142** (down 2); total tree size is **29,161**. This is a
performance WIP, not the 3,400-line checkpoint or an all-445 hardware-pass claim. Hardware execution remains pending
the required manual power cycle.

## 2026-08-13 — keep large MAX on the mapped reduce loop

Compile-only profiling after the weighted-NLL checkpoint ranked 512-element `cummin` indices as the slowest remaining
family: 8.84 s from rewriting 5,623+6,157 unrolled MAX UOps. The image was already compact (598 EW); the cost was the
unroll. `std_mean` now compiles successfully in 4.26 s through mapped-add of 13,125 lanes.

Rockchip MAX reductions whose axis product exceeds 64 now stay a REDUCE, matching ADD. The existing mapped-MAX
specializer's local-loop path lowers 512-element prefix extrema: 48 UOps and 1.25 s for `cummin` indices, 0.63 s for
`cummax`. 32-element max and 2×2 pool still unroll. Exact cumulative-index hashes are updated to those loop-mapped
native images. The 1,022-term synthetic scalar-local image and cos/tan hashes stay byte-identical. LUT/PPU were not
used.

Renderer size is **3,564** (down 22); `heuristic.py` remains **142**; total tree size is **29,139**. This is a
performance WIP, not the 3,400-line checkpoint or an all-445 hardware-pass claim. Hardware execution remains pending
the required manual power cycle.

## 2026-08-13 — first 445 hardware replay after NPU health check

Vendor `~/rk3588/examples/elementwise.py` passed, then the census ran serially:

`FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0`

The first 295 methods produced 272 passes, 11 skips, and 12 failures. `test_all_large` then aborted Python in
`reset_npu` (`ioctl`, exit 134). After a second green elementwise probe, the remaining 149 methods excluding
`test_all_large` produced 143 passes, 1 skip, and 5 failures. NPU health was green again afterward.

Observed census: **415 passed, 12 skipped, 17 failed, 1 aborted** (445 collected). This is not an all-445 pass.

Failures:

- Fancy indexing (6): `RKPLAN_REJECT:generic_uops` on bool-gated INT32 gathers.
- `test_softmax_argmax`, `test_acosh`, `test_asinh`: `eval with wrong dtype` in symbolic.
- `test_cross_entropy_reductions`: numeric, 4.152 vs 4.11.
- `test_nll_loss_weight`, `test_nll_loss_3d_weight`: `RKImage argument slot 6 is not bound`.
- `test_std_mean`: `RKImage argument slot 3 is not bound`.
- `test_sum_dtype_arg`: `RKPLAN_REJECT` on a float local ADD loop.
- `test_sum_full`: submit-count assert `342 != 1`.
- `test_where_permute`: submit-count assert `6 != 4`.
- `test_relu6`: numeric, about half the (45,65) lanes stayed 0.
- `test_all_large`: process abort in `reset_npu`, not a pytest failure.
