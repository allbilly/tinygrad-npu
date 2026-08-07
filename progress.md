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

## 2026-08-07 — Rockchip backend style cleanup

The renderer and runtime were simplified after comparing their structure with the Metal, DSP, QCOM, NV, and AMD backends. The cleanup shares one compile-time expression evaluator between integer index maps and geometry-only FP16 constant generation, relies on the recursive lowering walk as the single supported-graph check, removes the renderer's pass-through constructor, and uses the common `round_up` helper. The runtime no longer shadows the command buffer's existing size or keeps an unused per-program submit counter, and gather source setup is shared between affine and explicit-offset paths.

`sz.py` decreased the Rockchip implementation from **642 to 618 executable lines**:

| File | Before | After | Change |
|---|---:|---:|---:|
| `tinygrad/renderer/rockchip.py` | 498 | 478 | -20 |
| `tinygrad/runtime/ops_rockchip.py` | 144 | 140 | -4 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Complete Rockchip census with `-n12`: **102 passed, 9 skipped, 96 subtests passed in 50.59 s**.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- DPU-only audit unchanged: runtime NumPy operations remain raw `uint16` views, integer index construction, and copies. There is no host floating-point tensor arithmetic, CMAC, CNA, or PPU execution path.

---

## 2026-08-07 — FP16 interpolation through gather + DPU EW

All eight `test_interpolate*` methods from `test/backend/test_ops.py` are now mirrored into the Rockchip census at exactly the `test_gemm_fp16` tolerance ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`). This covers 24 linear, bilinear, trilinear, nearest, nearest-exact, and corner-aligned shape/mode combinations.

Nearest interpolation is lowered as an integer host gather followed by an FP16 DPU EW add-zero pass-through, so output realization still submits NPU work rather than becoming a host output copy. The compile-time geometry evaluator now respects integer cast points and FP16/FP32 rounding. This fixed multidimensional nearest maps where delaying `float→int` until after applying a row stride selected the wrong source elements.

Linear interpolation keeps coordinate calculations as input-independent FP32 compiler geometry, materializes the resulting weights as immutable FP16 bit tables, and executes every input-dependent blend through DPU EW MUL/ADD. Each separable axis casts its result back to the input dtype, preventing bilinear and trilinear schedules from creating an FP32 intermediate that the Rockchip NPU cannot consume.

Representative direct cold profiles, excluding Torch reference generation:

| Case | Wall time | Submits | Maximum absolute error |
|---|---:|---:|---:|
| nearest 13→9, `(2,3,13)` | 0.211 s | 1 | 0 |
| linear 29→52, `(2,3,29)` | 0.681 s | 1 | 0.0009766 |
| bilinear `(12,20)→(9,31)` | 7.278 s | 2 | 0.0019531 |
| trilinear `(5,2,8)→(3,6,4)` | 2.742 s | 3 | 0.0009766 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Interpolation census: **8 passed in 33.19 s** with `-n12`.
- Explicit nearest/trilinear ioctl regressions: **2 passed**; nearest submits once and trilinear once per axis.
- Complete Rockchip census: **112 passed, 9 skipped, 96 subtests passed in 111.05 s** with `-n12`.
- Generic FP32 `test_ops` interpolation census: **8 passed**.
- `test/unit/test_dtype_weak.py`: **32 passed, 1 xfailed**.
- Full Ruff and `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 487 executable lines, runtime 140; comments and docstrings retained.
- No CPU floating-point tensor arithmetic, tolerance relaxation beyond `test_gemm_fp16`, CMAC, CNA, or PPU execution was added.

---

## 2026-08-07 — Native DPU EW activation and FP16 broadcast census

The activation census now mirrors ReLU, LeakyReLU, ReLU6, ABS, clip, maximum, and minimum from `test/backend/test_ops.py`. The Rockchip-specific maximum and minimum cases cover tensor-tensor and tensor-scalar inputs. A separate broadcast census exercises FP16 ADD, MUL, and SUB across scalar, row, column, channel, batch, and full-tensor operands, in addition to the existing generic broadcast, scalar subtraction, reverse subtraction, and negation cases.

The compact lowering follows the proven RK3588 recipes in `~/npu` and `~/rk3588`:

- ABS uses the proven BS PReLU recipe with a slope of -1, followed by a DPU identity MAX that restores neutral pipeline state.
- LeakyReLU selects EW multiply with `EW_MUL_PRELU`, so the FP16 slope is applied only to negative elements.
- ReLU6 keeps EW ReLU active, enables `EW_RELUX_EN`, and writes the FP32 representation of 6 to `EW_RELUX_CMP_VALUE`.
- Minimum selects EW ALU algorithm 1 after recognizing tinygrad's negated-MAX form.

All configuration words are assembled from named register fields. Direct hardware probes for ABS, LeakyReLU, and minimum were bit-exact against FP16 NumPy results. ReLU6, LeakyReLU, and minimum each lower to one DPU EW task. ABS is recognized before generic WHERE lowering and uses two DPU submissions: the BS operation and a neutral-state identity restore.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Activation and broadcast census: **19 passed in 13.26 s** with `-n12`.
- Complete Rockchip census: **131 passed, 9 skipped, 96 subtests passed in 126.79 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 542 executable lines, runtime 140; this milestone adds 55 renderer lines.
- Tolerances remain exactly the existing FP16 ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`).
- No CPU floating-point tensor arithmetic, CMAC, CNA, or PPU execution path was added.

---

## 2026-08-07 — Group 3: DPU WHERE and masked arithmetic

FP16 `WHERE` now lowers entirely to DPU work. CMPLT and CMPNE predicates use the proven BS/BN positive-mask recipe, boolean AND/OR compose exact 0/1 masks with EW MUL/MAX, and finite selection is reconstructed with EW MUL/ADD. Nested selection and broadcasted tensor/scalar arms use the same lowering.

Constant infinity arms avoid the invalid `0 * inf` form. The compiler emits an FDIV correction for general masked fills and uses native MIN/MAX threshold selection when one arm is the compared tensor. No host code inspects input values: the runtime still uses NumPy only as raw `uint16` storage plus integer gather indexing.

Comparison masks and BS-PReLU ABS change global DPU pipeline state. Rockchip submissions therefore use one cross-process device lock. Mask stages use the vendor-style standalone task with reset boundaries, while all following stages in a mask-containing program explicitly configure BS/BN and conversion state. ABS submits its BS task, resets, then submits a DPU identity MAX to restore neutral state. Ordinary GEMM/conv programs keep their compact PC-chain bodies and tested chain limits.

Direct warm profiles:

| Case | Wall time | NPU submits |
|---|---:|---:|
| scalar WHERE, 120 elements | 0.373 s | 3 |
| infinity masked fill, 320 elements | 0.234 s | 3 |
| exact ABS including `±inf` | 0.138 s | 2 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Group 3 WHERE census: **9 passed in 15.45 s** with `-n12`.
- Activation regression: **10 passed in 13.71 s** with `-n12`.
- Post-state GEMM/conv regression: **2 passed in 13.46 s** with `-n12`.
- Complete Rockchip census: **140 passed, 9 skipped, 96 subtests passed in 125.46 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 659 executable lines, runtime 180; comments and docstrings retained.
- Tolerances remain exactly the existing FP16 ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`).
- No CPU floating-point tensor arithmetic, CMAC, CNA, or PPU execution path was added.

---

## 2026-08-07 — Group 4: FP16 padding modes through gather + DPU EW

All constant, reflect, replicate, and circular padding cases mirrored from `test/backend/test_ops.py` now pass at the unchanged
`test_gemm_fp16` tolerance (`atol=rtol=grad_atol=grad_rtol=5e-3`). This covers positive and negative padding, finite and infinite
constant fills, 1D/2D/3D reflect and replicate layouts, circular wrapping, and the original invalid-argument checks.

Reflect and replicate edges are represented by tinygrad as chains of gated loads whose fallback is another load. The renderer now
proves these predicates and indexes from static output geometry and collapses each mutually exclusive load tree into one raw gather
map. Runtime preparation remains an integer address selection and `uint16` copy. A final DPU EW add-zero pass-through realizes the
output on the NPU; input values are never examined or arithmetically transformed by the host.

Direct realization of the focused `(5,5,5)` / `(0,2)` cases:

| Mode | Wall time | NPU submits | Maximum absolute error |
|---|---:|---:|---:|
| reflect | 0.227 s | 1 | 0 |
| replicate | 0.046 s | 1 | 0 |
| circular | 0.038 s | 1 | 0 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Padding census: **4 passed** in 14.74 s with `-n12`; the four methods execute 32 numeric helper cases plus their exception checks.
- Focused submit regression: **1 passed, 3 subtests passed**; each mode submits exactly once.
- Complete Rockchip census: **145 passed, 9 skipped, 99 subtests passed in 366.76 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 701 executable lines, runtime 180; comments and docstrings retained.
- No host floating-point tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution was added.

## 2026-08-07 — Group 5: FP16 movement and static indexing

The Rockchip census now covers slicing (including negative strides and empty results), transpose, permute, reshape/view, flip,
squeeze/unsqueeze, flatten/unflatten, expand, detach, diagonal extraction, roll, and pad-to-reshape/slice compositions from
`test/backend/test_ops.py`. The upstream negative-stride test explicitly creates FP32 input, so its Rockchip equivalent uses the same
indexes over FP16 input. The constant integer-tensor indexing case is explicitly skipped because the DPU accepts FP16 inputs only;
it is not emulated on the CPU.

The static selection gather introduced for padding now preserves one proven nonzero FP16 fill bit-pattern. This enables sliced padded
layouts such as a `3.456` constant border without a floating-point runtime operation. The compiler evaluates only output geometry;
the runtime continues to copy raw `uint16` lanes according to integer offsets before a DPU EW add-zero realization.

Representative direct realization profiles were bit-exact and used one ioctl each:

| Case | Wall time | NPU submits | Maximum absolute error |
|---|---:|---:|---:|
| permute `(3,4,5) -> (5,3,4)` | 0.146 s | 1 | 0 |
| reverse/stride slice | 0.050 s | 1 | 0 |
| expand `(2,3,1) -> (2,3,4)` | 0.024 s | 1 | 0 |
| nonzero pad then slice | 0.130 s | 1 | 0 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Movement census: **30 passed, 1 skipped in 15.78 s** with `-n12`.
- Focused submit regression: **1 passed, 4 subtests passed**; every materialized layout submits exactly once.
- Complete Rockchip census: **176 passed, 10 skipped, 103 subtests passed in 331.50 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 704 executable lines, runtime 180; comments and docstrings retained.
- No host floating-point tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-07 — Group 6: FP16 concatenation, stacking, and repetition

Cat, multicat, stack, stack-to-slice, stack-to-max, and repeat from `test/backend/test_ops.py` now pass on Rockchip. RKImage v16 adds
a partial raw gather: several FP16 source buffers may populate disjoint lanes of the same scratch buffer, with negative offsets skipped
instead of filled. The compiler proves that every destination lane selects exactly one source and that there are no constant-fill lanes
before emitting this form. Runtime work remains integer indexing and raw `uint16` copies; one DPU EW add-zero stage realizes each
scheduled output kernel.

The upstream scalar-stack assertion bypasses `helper_test_op` and compares FP16 `3.140625` to float64 `3.14` at zero tolerance. Its
Rockchip equivalent keeps the same operation but uses the unchanged `test_gemm_fp16` ceiling (`5e-3/5e-3`). No backend tolerance was
relaxed.

Representative direct profiles were bit-exact:

| Case | Wall time | NPU submits | Maximum absolute error |
|---|---:|---:|---:|
| cat axis 0 | 0.192 s | 1 | 0 |
| cat axis 1 | 0.060 s | 2 | 0 |
| stack axis 1 | 0.025 s | 2 | 0 |
| repeat | 0.041 s | 1 | 0 |

This milestone also removed two compile-time outliers exposed by full duration reporting. Selection/static-expression classification is
now memoized per kernel, and static gather index/gate UOps are compiled once into exact geometry evaluators with a cached output-lane
map. These evaluators inspect only immutable graph geometry and constants, never input values.

| Case | Before | After | Dominant cause removed |
|---|---:|---:|---|
| nested Conv2D compile | 63.15 s | 4.84 s | repeated selection/static subtree walks |
| nested Conv2D isolated test call | 61.61 s | 6.81 s | same |
| ConvTranspose3D compile | 64.93 s | 3.18 s | recursive evaluation of 108 gather maps |
| ConvTranspose3D isolated test call | 64.82 s | 5.11 s | same |
| complete Rockchip census | 331.50 s | 51.13 s | both compiler bottlenecks |

The ConvTranspose3D wall-time decomposition before the gather evaluator was: 64.93 s compile, 0.34 s schedule, 0.14 s raw gather
setup, 0.21 s command construction, 0.075 s across 27 blocking ioctls, and 0.89 s total execution. This confirms the optimization
target was host compilation, not NPU execution or ioctl waiting.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Concat census: **6 passed in 19.32 s** with `-n12`.
- Focused submit regression: **1 passed, 4 subtests passed**.
- Complete Rockchip census: **183 passed, 10 skipped, 107 subtests passed in 51.13 s** with `-n12` and `--durations=20`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 794 executable lines, runtime 180; comments and docstrings retained.
- No host floating-point tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — Group 7: FP16 reductions and measured EW chain boundary

The Rockchip census now covers the FP16 sum, mean, minimum, maximum, and product reduction cases selected from
`test/backend/test_ops.py`, including the `NOOPT=1` product register loop and the 16,384-element full sum. ReLU inside an unrolled
reduction is recovered from tinygrad's FP32-shaped `WHERE` as native half DPU `MAX(0,x)`. Non-FP16 boolean, integer, and explicit
FP32 reduction cases remain explicitly excluded because the NPU accepts FP16 tensor inputs only.

Direct dependent-ADD probes found that the previous 512-task PC-chain setting was not a safe hardware limit. A 383-task chain passes,
while 384 tasks in one chain time out; the driver reports task counter 510 for the earlier 512-task failure and resets the NPU. Ordinary
FP16 EW images are therefore capped at **383 tasks per ioctl**. Mixed TwoProduct images retain their independently proven 256-task cap.

The full sum originally spent more than 59 seconds in compiler expansion and never reached an ioctl. Its wall decomposition was
0.127 s import, 0.012 s graph construction, 0.053 s scheduling, then over 59 s in kernel optimization/code generation. The Rockchip
regression now uses tinygrad's existing `NOOPT=1` compact register-loop path so no shared codegen change is required. The Rockchip
renderer gathers raw FP16 lanes into 64-byte-aligned scratch slots and emits a balanced DPU EW tree. It also preserves fused scalar
post-scales and lowers minimum as `-max(-x)` entirely through DPU EW. Host work is integer address/layout construction and raw `uint16`
movement only; every arithmetic operation executes on DPU EW.

| 16,384-element sum | Before | After |
|---|---:|---:|
| Realization wall time | >59 s, timed out before submit | 1.26 s |
| DPU tasks | not reached | 16,383 |
| ioctl submits | 0 | 43 |
| Absolute error vs FP32 sum (magnitude 56.32) | not reached | 0.057 |

The complete census profile identified the slowest pooling case as host-compile-bound:

| `test_avg_pool3d` phase | Wall time |
|---|---:|
| Torch golden | 0.0007 s |
| Tensor construction | 0.007 s |
| Scheduling | 0.344 s |
| Compile/link before Rockchip gather caches | 10.045 s |
| Rockchip rendering within compile | 2.230 s |
| NPU execution | 0.080 s |
| Command construction + 3 blocking ioctls | 0.0073 s |

Bounded Rockchip-only index/selection caches reduced the controlled compile/link phase from **10.04 s to 8.15 s** and Rockchip
rendering from **2.23 s to 0.93 s**. The remaining approximately 7.2 s is shared optimizer/codegen work and was deliberately left
unchanged. `--dist=worksteal` varied from 52.53 s to 59.41 s across full cold runs, so it is not treated as a reliable optimization.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Reduction census: **21 passed, 1 skipped in 19.86 s** with `-n12` after restoring the shared optimizer unchanged.
- `test_sum_full` call: **1.20 s**, with an explicit **43-submit** regression.
- Complete Rockchip census: **204 passed, 11 skipped, 107 subtests passed in 59.41 s** with `-n12 --dist=worksteal`; cold-run
  scheduling/compile contention varies materially, while NPU execution and ioctl time remain sub-second in the profiled slow case.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 891 executable lines, runtime 181; comments and docstrings retained.
- No shared tinygrad core change, host floating-point tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — Group 8: adaptive-pooling-equivalent FP16 windows

Tinygrad currently exposes no adaptive average/max pooling API, so the Rockchip census covers the exact equivalent cases whose input
windows divide evenly: 4×4 to 2×2 and 4×4 to global 1×1. PyTorch adaptive pooling supplies the golden while tinygrad expresses the
same operation with fixed kernel/stride pooling. No core API or backend lowering change was needed.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Adaptive-equivalent census: **2 passed in 12.30 s** with `-n12`.
- Average and maximum cases both execute through the existing FP16 gather + DPU EW reduction path.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`; no CPU arithmetic, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — Group 9: FP16 polynomial and simple loss expressions

The Rockchip census now covers square, cubic, a Horner-form quadratic, mean-squared error, and mean absolute error. These expressions
compose the already proven FP16 ADD/MUL/ABS and reduction paths; BCE and NLL are intentionally outside this group because they require
transcendental or indexed semantics. FP32 is used only by the PyTorch golden for the loss reductions, then cast to the FP16 device
contract.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Polynomial/simple-loss census: **4 passed in 12.73 s** with `-n12`.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.
- Rockchip execution uses only DPU EW; no host tensor arithmetic, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — Group 10: FP16 scatter and indexed reductions

The Rockchip census now covers functional scatter replacement on dimensions 0 and 1, scalar-source scatter, and scatter-reduce sum,
product, mean, maximum, and minimum. Indices are compile-time-static `arange` or broadcast constants: the renderer turns them into raw
gather layout, while selected FP16 values and every reduction execute on DPU EW. External integer index buffers remain explicitly
excluded because Rockchip accepts FP16 inputs only.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Static-index scatter census: **8 passed, 1 skipped in 13.06 s** with `-n12`.
- Covered reductions: sum, product, mean, maximum, and minimum.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`; no core change, host tensor arithmetic, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — Ten-group completion audit

Final standard census command:

```sh
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct \
  python -m pytest test/backend/test_rockchip.py -q -n12 --durations=10
```

- **218 passed, 12 skipped, 107 subtests passed in 55.85 s**.
- The added Groups 8–10 contribute 14 passing cases while total wall time remains approximately the earlier 55-second census.
- Slowest calls were Conv1D 14.45 s, AvgPool3D 11.42 s, dilated ConvTranspose2D 10.57 s, and bilinear interpolation 10.21 s;
  the measured slow-path decomposition remains host-compile-bound rather than ioctl/device-bound.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: Rockchip renderer 891 executable lines, runtime 181; comments and docstrings retained.
- Changes since Group 6 are confined to `progress.md`, `test/backend/test_rockchip.py`, `tinygrad/renderer/rockchip.py`, and
  `tinygrad/runtime/ops_rockchip.py`; shared tinygrad core is unchanged.
- Audit found no host floating-point tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution in the Rockchip backend.

---

## 2026-08-08 — Rockchip backend style cleanup

The renderer/runtime split was reviewed against tinygrad's QCOM, DSP, AMD, and NV backends. Rockchip retains the same core structure:
`Renderer` lowers and encodes device work, `Program` relocates/submits it, `Allocator` only manages buffers, and `Compiled` connects the
pieces. The cleanup remains entirely Rockchip-local and makes the following conventions explicit:

- shared `getenv`, `ceildiv`, and `round_up` helpers replace local equivalents;
- one named `rk_cmd` encoder is shared by renderer and runtime instead of duplicating PC command packing;
- FP16/FP32 rounding and register-bit conversion use named helpers;
- command/task buffer growth is centralized in `RockchipProgram`;
- the FDIV ALU selector and NPU counter-clear action are named rather than embedded numeric literals.

Verification:

- Complete Rockchip census: **218 passed, 12 skipped, 107 subtests passed in 48.37 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer **891 → 890** executable lines, runtime **181 → 176**; comments and docstrings retained.
- No shared tinygrad core change, CPU tensor arithmetic, tolerance relaxation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — IEEE FP16 multiply by infinity on DPU EW

RK3588's native FP16 DPU EW `MUL` returns NaN for an infinity operand, including nonzero finite values. The Rockchip renderer now
recognizes a scalar `+inf` or `-inf` multiplier and synthesizes the IEEE result entirely on DPU EW:

1. DPU positive-mask stages compute `x > 0` and `-x > 0`.
2. Their difference forms `sign(x)` (or its inverse for `-inf`), followed by `+0` to normalize the DPU SUB stage's `-0`.
3. DPU FDIV divides that sign by zero, producing signed infinity for nonzero values and NaN for either signed zero or NaN.

The regression includes negative/positive infinity inputs, FP16 extrema, ordinary signs, both signed zeros, and NaN. A one-tile
infinite scalar multiply uses **4 blocking ioctl submits**: two stateful positive-mask submissions, one preparation chain, and one
final sign/FDIV chain. Multiplication by NaN retains the native one-submit EW path.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`:

- Focused infinity/NaN regression: **1 passed in 12.49 s** with `-n12`.
- Complete Rockchip census: **219 passed, 11 skipped, 107 subtests passed in 49.16 s** with `-n12`.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.
- Full Ruff and `mypy tinygrad/` pass; `sz.py` reports renderer **898** executable lines and runtime **176**.
- No shared tinygrad core change, host tensor arithmetic, CMAC, CNA, or PPU execution was added.
