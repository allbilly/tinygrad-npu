> **Historical.** Every "ours" figure below is the pre-INT8 generic-GEMM baseline
> (31.9 s prefill, 1.5 tok/s). Current: **1,198.87 ms** — see `../prefill-parallel-quantise.md`.
> The 9-model table and the device findings are still valid.

# NPU GEMM baseline — INT8, Qwen3.5 4B

Every `nn.Linear` replaced by `NPULinear` (`extra/rockchip/npu_linear.py`),
issuing one matmul per (row, column, K) tile. Attention, norms, softmax, rope,
embedding, sampling and the KV cache stay on the CPU host.

Protocol: 128-token prompt, 64 greedy decode tokens, 1 warm-up + 3 measured
passes, every trial reported, median taken. Prefill proxy is
`128000 / median_ttft_ms`.

## The headline

Decode is **1.5281 tok/s = 654.4 ms/step**. The device floor is **147 ms**, so
**507 ms of every step is host work**. That is the whole problem, and it is not a
kernel problem.

Provenance, because this was disputed: `results/qwen35-4b/report.json` is the
**INT8** run. `results/int8-4b.log` is stamped the same minute (22:43) and records
`1383.4s` against the report's `total_seconds = 1373.4`; the earlier FP16 run is
`results/rerun-4b.log` at `1212.4s`, whose report was overwritten. The report
predates the `npu_weight_mib` field, so it cannot be classified by its contents —
only by the log. Its three decode trials are 1.5273 / 1.5281 / 1.5352 tok/s and
its three TTFT trials 220574 / 178731 / 178603 ms; the median is quoted.

## The INT8 matmul, measured

`extra/rockchip/bench_submit.py`, 20 reps, median. Each submission is
`RKNPU_JOB_BLOCK`, so the call returns after the NPU finishes. Artifact:
`docs/rknpu/results/submit-timing.json`. Clock state was **not recorded in that
artifact**; it was read separately at the time as NPU 1 GHz, DDR 2112 MHz, CPU6
2256000, 50.8 °C.

| m | n | k | cores | total ms | GB/s | % of 33.79 |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 256 | 2560 | 3 | 0.056 | 11.73 | 34.7 |
| 1 | 512 | 2560 | 3 | 0.075 | 17.45 | 51.6 |
| 1 | 1024 | 2560 | 3 | 0.115 | 22.70 | 67.2 |
| 1 | 2048 | 2560 | 3 | 0.198 | 26.47 | 78.3 |
| 1 | 4096 | 2560 | 3 | 0.366 | 28.67 | 84.8 |
| 1 | 8192 | 2560 | 3 | 0.686 | 30.56 | 90.4 |
| 1 | 9216 | 2560 | 3 | 0.777 | 30.38 | 89.9 |

Least squares over the `n` sweep at fixed `k` separates the two costs:

| k | fixed per submit | effective bandwidth | % of ceiling | points |
| --: | --: | --: | --: | --: |
| 2560 | 0.034 ms | 31.93 GB/s | 94.5% | 8 |
| 4096 | 0.036 ms | 27.85 GB/s | 82.4% | **3** |

The k=4096 fit rests on three points and is much weaker evidence than the
eight-point k=2560 fit. The 0.034 ms intercept is a fit extrapolation from
`n >= 256, cores=3`, not a measured constant: the smallest task in the sweep
(`m=1, n=32, k=512, cores=1`) costs 0.0245 ms.

**The 94.5% figure holds only for `m=1, k=2560, n >= 4096`.** The same artifact
measures two exceptions, and both are real:

| region | bandwidth | % of ceiling |
| --- | --: | --: |
| m=1, k=2560, n >= 4096 | 31.93 GB/s | 94.5% |
| m=1, k=4096 | 27.85 GB/s | 82.4% |
| m=64, k=2560, n=9216 | 17.73 GB/s | 52.5% |

So the device path is close to the bus for the bulk of decode, but **24% of the
weight stream runs at k=4096 where there is 18% of headroom left**, and every
row above `m=1` is far worse. "No device headroom anywhere" is not supported.

Other measured facts:

- **323 submits per forward**, identical for prefill and decode. At 0.034 ms that
  is **11 ms per token**, so submit latency is not the decode problem.
- **Host prep is 5 µs.** The `total - ioctl` gap is 0.004-0.005 ms. The ioctl is
  98.8-99.4% of the call at `n >= 4096`, but only 82-96% below that.
- **Three cores give 2.76x** (n=8192: 1.894 ms on one core, 0.686 ms on three).
  One core reaches 11.07 GB/s and cannot saturate the bus; three reach
  30.56 GB/s and share it, which is why the speedup is 2.76x and not 3x.

The `m=64` row is a **probe shape the model path does not issue**: `NPULinear`
sets `row_tile = max(1, min(64, 65536 // k_max))`, so the harness actually
submits `m=25` at k=2560 and `m=16` at k=4096. A 128-row prefill is therefore 6
row tiles, each re-streaming the full weight tile. Including activation and
output traffic, m=64 reaches 58% of ceiling rather than 52.5%, so roughly 42% of
that deficit is unexplained by the traffic accounting.

## Decode: the host cost is fixed per call, not per element

`extra/rockchip/profile_int8.py` times each stage of one INT8 `NPULinear` call at
the 4B's real shapes. The result is the same at `m=1` and `m=64`, which means the
cost is **launch overhead, not data**:

| stage | ms | share |
| --- | --: | --: |
| quantize (`int8_payload`) | **0.53** | 36% |
| dequantize (`.float()` x 2 multiplies) | **0.40** | 27% |
| submit (NPU ioctl) | 0.30 | 20% |
| scatter (`Tensor.cat` of per-core planes) | 0.13-0.24 | 11% |
| alloc | 0.054 | 4% |
| pad | 0.056 | 4% |
| d2h | 0.028 | 2% |
| **total** | **1.48-1.63** | **81% host** |

323 submits x 1.5 ms = **485 ms**, which is the measured decode host cost.

Isolating the quantize stage: `Tensor.empty` 0.045 ms, `_quantize_program`
construction 0.071 ms, `custom_kernel` with a pre-built program 0.127 ms, full
`int8_payload` 0.473 ms. The kernel's actual arithmetic on 2560 floats is ~2.5 us
of that 473 — **99.5% is launch overhead**.

**The arithmetic bound:** the device is 149.6 ms/token and the reference's whole
step is 154.2 ms, so host must reach ~5 ms. At 323 submits the ioctl alone is
11 ms. Reaching the reference therefore requires both fewer submits (it runs 113
ioctls, `~/oNPU/benchmark.md:510`) and near-zero per-call host work — that is the
native-replay port, not a kernel tweak.

## The fused INT8 Linear: one C function per Linear

`packed_matmul_int8_fused` (`extra/rockchip/npu_linear.py`) replaces the six
launches above with **one CALL** whose body is a single C function. It

1. quantizes the fp32 activation straight into an NPU mapping (per-row
   `max/127`, 1e-8 floor, `__builtin_nearbyintf` codes, zero-fill past the real
   depth), then
2. issues `RKNPU_MEM_SYNC_TO_DEVICE` on the payload and `DRM_IOCTL_RKNPU_SUBMIT`
   itself, then
3. issues `RKNPU_MEM_SYNC_FROM_DEVICE` on the plane buffer and dequantizes the
   per-core int32 planes into fp32 `(m, n)` with `a_scale[row] * w_scale[col]`.

No host↔device copy survives: the quantize writes into the NPU mapping and the
dequantize reads out of it. The command chain is byte-identical to
`RockchipMatmul.prepare`'s, because both now call the same `_encode_task_chain`.

### What the ABI forces

| constraint | how it is met |
| --- | --- |
| CPU/CLANG ABI is one packed argument buffer | entry is `void rockchip_linear(void *args)`; it unpacks `out`, `x` and a parameter block itself |
| `_states` rejects `Ops.CONST`, so no scalar rides through the call | geometry, `domain`, `core_mask`, `subcores`, `tasks` and the ioctl request words are literals |
| entry must sit at offset 0 | helpers are `static __attribute__((always_inline))`; `nm` shows `rockchip_linear` as the only `T` symbol at 0 |
| must not contract | `#pragma clang fp contract(off)`; `objdump` shows 23 `fmul` and 0 `fmla` |
| rebuilding the PROGRAM costs 0.26 ms | memoised in `_FUSED_PROGRAMS`, keyed by geometry |

Buffer addresses and the drm fd are deliberately **not** literals: they travel in
a 9-slot u64 parameter block, so the generated source is a function of the
geometry alone. Baking them made every plan a unique source, i.e. 500 clang
invocations per model load instead of ~16, and defeated the on-disk compile cache
across processes. `ioctl` and `clock_gettime` are declared `extern` and resolved
by the JIT loader against libc, for the same reason.

Scratch is pooled per geometry, not owned per plan: a per-plan plane buffer costs
2.4 MB for every 64-row ffn plan, ~600 MB for the 4B. The payload and plane
buffers are shared per `(m, k)` and `(m, n)`; calls are serial and blocking, so no
two plans are ever in flight.

Two smaller costs fell out of the same measurement. `Tensor.custom_kernel`
rebuilds two placeholders (0.16 ms) that the `fxn` then ignores, so the CALL is
bound directly — 0.02 ms instead of 0.28 ms. And `Tensor.empty`'s dtype/device
canonicalisation is 0.03 ms, so the output is `UOp.alloc` + `Tensor`.

### A latent bug the gate found

The old tail read `w_scale[offset + j]`, i.e. the scales of the first `n` output
rows, so any call with `column_offset != 0` dequantized with the wrong per-row
scale. That is reachable at `m>1`, where a 24576-wide chunk is split into two
12288-wide slices; decode never hits it because `TILE == width == 24576` at
`m=1`. `_dequant_source` now takes the offset. Before the fix the old path and
the correct reference differed on both `column_offset != 0` cases in
`smoke_fused.py`; after it, they agree.

## The fused per-Linear call

`NPULinear` now issues **one C call per Linear** instead of seven tinygrad
launches: `rockchip_linear` quantizes the activation, builds the `struct Submit`
and calls `ioctl` directly, then dequantizes the int32 planes — all in one
function, with the buffer addresses travelling in a parameter block so the source
stays a function of geometry alone.

This is the archive's `NativeReplay` mechanism (`rockchip_replay.py:167-169`,
`:494-506`) applied at one-Linear granularity rather than whole-forward.

**Verified bit-exact on a real model** (`extra/rockchip/parity_token.py`, Qwen2
0.5B, same prompt, same process, fused vs the six-launch path):

```
fused  169 linears  ->  [27130, 315, 279, 18484]  (0.46 tok/s over 4 steps)
old    169 linears  ->  [27130, 315, 279, 18484]  (0.33 tok/s over 4 steps)
tokens match
```

Identical tokens, **1.39x faster** on a 4-step run. The 4-step figure is noisy and
is not a decode rate; the full-protocol 4B number is what decides.

## The reference, reproduced on this board

`docs/rknpu/results/ref-warmed-4b/benchmark.json`, `complete: true`, 1 warm-up + 3
measured, 128/64, context 512:

| trial | first token | decode |
| --: | --: | --: |
| warm-up | 16,863.06 ms | 2.6670 tok/s |
| 1 | 9,266.53 ms | 6.4176 tok/s |
| 2 | 20,037.00 ms | 6.1977 tok/s |
| 3 | **1,224.53 ms** | 6.1539 tok/s |

| metric | reproduced | archive published | delta |
| --- | --: | --: | --: |
| median TTFT | 9,266.53 ms | 9,335.78 ms | -0.7% |
| last-trial TTFT | **1,224.53 ms** | 1,210.63 ms | +1.1% |
| median decode | **6.1977 tok/s** | 6.3555 tok/s | -2.5% |

**The reference's numbers are real on this board.** The configuration that produces
them, pinned and verified:

```
tree    ~/tg-qwen35-pack-signature-fix @ 1b2a5db332e1198b3ace8c7559a19c7d73717dfb
        + 4-file dirty patch, diff sha256 7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77
app     ~/oNPU @ 5229660be83d3ee8e155e31109518fb90d641e80
runner  analysis/release-20261115/r2-lfm-20261116/run_candidate_warmed_retime.py --model qwen35-4b
protocol 1 warm-up + 3 measured, 128 prompt / 64 decode, context 512, 52-55 C start band
```

The 1+3 artifact is structural and reproduced: the same run yields **1.2 s** and
**20.0 s** for the same 128-token prefill depending on trial order. Only the last
trial is a prefill figure. Clocks are identical in all four trials (NPU 1 GHz,
DDR 2112, CPU4/6 2256000) and temperature does not correlate — trial 2 was the
hottest *and* slowest. The remaining candidate is `--file-backed-cpu-weights`
mmapping the 4.5 GB GGUF on a 15 GB board, but that is **not established**.

### How the trees relate — settled by the session history

- **`tg-qwen35-pack-signature-fix` is a git worktree of `tg-commit`**, created by
  `git -C ~/tg-commit worktree add -b codex/qwen35-pack-signature-fix … 1b2a5db`.
  Its `.git` is a **file**, not a directory, so any `[ -d "$t/.git" ]` test reports
  "no git" for a valid worktree. Do not use that test.
- **`tg-decode-all` never had a `.git`.** No `rm -rf .git` exists in 141k session
  rows; the 2026-11-17 plan was to `git init` it with `--exclude=.git` in the rsync.
  It also cannot be revision-pinned, so it cannot be a reproduce tree.
- **The 4B reference never went through `onpu.worker`.** It called
  `benchmark_inference` inside the pinned worktree, driven by
  `run_candidate_warmed_retime.py`. `retime_with_profile.py qwen35-4b` exists only
  as a `--dry-run`; no measured 4B run ever went through that harness.
- The recorded original run: **2026-11-17 03:52:22**, same runner, `--out
  …/candidate-retime-20261117-qwen35-4b`.

### The gap, both sides measured

| | ours | ref (reproduced) | gap |
| --- | --: | --: | --: |
| prefill, last trial | 23,000 ms | **1,224.53 ms** | **18.8x** |
| decode | 2.32 tok/s | **6.1977** | **2.67x** |
| decode device | 145.7 ms | 138.66 ms | 1.05x |

## Against the reference and RKLLM

Same **token protocol** (128 prompt / 64 decode / 512 context), but a different
clock policy and measurement surface: the reference row is an HTTP-serving median
over 20 requests with adaptive local governors and no sudo tuning
(`~/oNPU/benchmark.md:1329`), while ours is in-process with clocks pinned. The
reference is at a disadvantage on clocks, so the real gap is **understated** here.

| metric | ours | oNPU ref | RKLLM | vs ref | vs RKLLM |
| --- | --: | --: | --: | --: | --: |
| TTFT, 128 tokens | **31,933 ms** | 1,263 ms | 2,069 ms | **25×** | **15×** |
| prefill proxy | 4.01 tok/s | 101.34 | 61.87 | 25× | 15× |
| decode | 1.5088 tok/s | 6.48 | 6.22 | 4.3× | 4.1× |
| decode step | 662.8 ms | 154.2 ms | — | 4.3× | — |
| — device | **151 ms** | 138.66 ms | — | **1.09×** | — |
| — host | **512 ms** | 15.56 ms | — | **33×** | — |

All three TTFT trials and all three decode trials are in
`results/qwen35-4b-fixed/report.json`; medians are quoted. Prefill NPU is 442 ms
of 31,933 ms, so prefill is still 98.6% host. Decode device is 151 ms, which is
6.62 tok/s — within 2% of the reference's 6.48, so **decode host is the only
remaining decode cost**.

The rate and TTFT are 3-trial medians from `results/qwen35-4b`. The device
figure is **measured**, from `results/qwen35-4b-cold` trial 1
(`decode_npu_ms_per_token = 149.6`), which is the first run that recorded it; that
run was stopped after trial 1 to chase the prefill cost, so it has no median. The
device figure is stable across runs: 149.4 (warmup) and 149.6 (trial 1) on the
cold run, 155.6 on an earlier hot run.

**The device is within 8% of the reference. The host is 32x it.** Every ratio
that looks catastrophic is host.

Reference numbers: `~/oNPU/benchmark.md:1179` (table) and `:1392` (the native
profile that breaks 154 ms into 138.66 / 14.95 / 0.61).

**The 147.4 ms is derived, not measured**, and it is derived carefully: 4.2048 GB
of packed weights (`npu_weight_mib = 4010.0`, from
`results/submits-4b/report.json`), of which 1.007 GB is in k=4096 tiles priced at
27.85 GB/s and 3.198 GB at 31.93 GB/s gives 136.3 ms, plus 323 submits x
0.034 ms = 11.0 ms.

**The in-model measurement is still open.** `bench_llm.py` now records
`decode_npu_ms_per_token` from an accumulator inside the blocking submit. The
first attempt gave two passes that disagree by 4x:

| pass | prefill NPU | decode NPU | per submit |
| --- | --: | --: | --: |
| warmup (cooler) | 0.2 s | 155.6 ms/token | 0.48 ms |
| trial 1 (hotter) | 0.8 s | 693.6 ms/token | 2.15 ms |

The isolated sweep measures 0.24-0.78 ms per submit, so the warmup pass agrees
with it and the second does not. 693.6 ms/token also exceeds the full protocol's
entire 654.4 ms step, so it cannot be right. The board was at **78.5 °C** when
that run ended and **84 °C** shortly after, against a protocol that requires a
52-55 °C cold start, so the run is invalid and the cause of the 4x is not
established. It is not the NPU clock: `cur_freq` read 1 GHz throughout.

The one usable data point is the cooler pass: **155.6 ms/token, 5.6% above the
147.4 ms derived floor**, which is the agreement the derivation predicted.

**What survives either way:** the device is within ~12% of the reference's
138.66 ms even taking the cooler measurement, so the 4.2x decode gap is
host-side. The host CALL chain is ~129 ms (10 CALLs x 40 us x 323). The
remaining **~370 ms is unexplained** and is the thing to measure next, on a cold
board.

**Prefill is 99% host.** Measured on the same run: 32.7 s of prefill with
**0.2 s in the NPU** (warmup pass), 27.6 s with 0.8 s (hot pass). Even the
pessimistic pass leaves 96% of prefill in the host path. The SSM scan is 8.3% and
attention 1.1-1.5%, so the prefill problem is the same host path as decode, not
a missing kernel.

The 4B row started at **63.8 °C**, not the 52-55 °C cold start the protocol
requires, so it is qualification-pending rather than final.

Quick mode is **not comparable** to these numbers: `--quick` defaults to no
warm-up, so JIT capture lands inside the 8 measured tokens and reports 0.43 tok/s
against 1.53. Never quote a quick number.

## Device findings that still stand

**The driver gives each IOMMU domain a 4 GiB aperture** (`rknpu_iommu.c:
dma_limit = 1ULL << 32`). Measured single-domain boundary: 3.76 GB (3.50 GiB) of
packed weights fits, 5.23 GB (4.87 GiB) does not. `DOMAIN_BUDGET = 3 GiB` in
`npu_linear.py` is deliberately conservative below that fit point; 3.906 GiB was
tried and reverted for no measurable gain and one board reboot.

**Domain switches per forward is 1, not 4.** 4.2048 GB against a 3 GiB budget
gives two domains, and `_Domains.place` only ever appends, so the domain index is
monotonically non-decreasing along the walk. The earlier "4 against 183" figure
has no artifact and is withdrawn.

**The INT8 output is compact per-core planes, not a row-major `(m, n)` matrix.**
Each core writes a dense `(m, part_width)` block at `offset*m` int32 elements, so
the host must scatter the planes. The FP16 layout `(n/4, m, 4)` is channel-group
major and concatenates correctly under a column split, which is why only INT8
needs the scatter.

**INT8 K must be a 512 multiple** except for the natural depths 576/768/1152/2304,
so a ragged K tile is padded before packing.

**k=4096 is a hard wall for this recipe, not a conservative cap.** The reference
reports the INT8 CBUF wall at 11264, but raising our limit to accept k=9216
**hangs the device**: `DRM_IOCTL_RKNPU_SUBMIT` returns `TimeoutError [Errno 110]` at
`M=1, N=2560, K=9216, cores=3`. The cause is the `banks` formula collapsing to its
`0x1b` floor at 18 groups. So `ffn_down` (k=9216) stays split into three K tiles,
and **323 submits per forward is the floor** for this recipe — the reference's
11264 figure does not transfer.

**The CPU/CLANG kernel ABI is one packed argument buffer**, not the natural C
signature. `ClangRenderer.packed_args = True`, so `pack_args` rewrites a kernel's
params into a single `void *args` and the runtime calls `fxn(args)`. A
hand-written `Ops.PROGRAM` with a precompiled `Ops.BINARY` skips that rewrite, so
its entry point must take `void *args` and unpack the buffer addresses itself. A
natural `void f(unsigned char*, const float*, int, ...)` instead receives the
16-byte args buffer as its first pointer and garbage after it — it returns
without writing anything and without crashing, which reads as "the output buffer
was never touched". Scalar geometry cannot ride through the call either:
`_states` in `codegen/__init__.py` asserts every scheduled call arg resolves to a
buffer state, so shapes are baked into the source, the way `_matmul` bakes them
into its descriptor. `_finite_program` sidesteps all of this by calling
`program.fxn` directly instead of going through the scheduler.

**The runtime calls the loaded image at offset 0**, so a hand-written object must
contain exactly one function: helpers stay `static` and get inlined away at the
fixed `-O2`. A survivor at offset 0 shifts the entry point and the kernel
silently runs the wrong code.

**Async submits are available without fences, by polling the output.** I first
concluded they were impossible because `CONFIG_ROCKCHIP_RKNPU_FENCE is not set`
and `RKNPU_JOB_FENCE_OUT` returns `-EINVAL`. That is true and irrelevant: ORK
(`~/rkllama/oRKLLM_llama.cpp-rockchip/ggml/src/ggml-ork/ork-driver/src/npu.c:1271`)
uses `RKNPU_JOB_NONBLOCK` (`1 << 1`) plus a **sentinel poll on the output buffer** —
pre-fill with `0x7fffffff`, submit with `flags = PC|NONBLOCK|PINGPONG = 0x7`,
spin on the last word with a `DC CIVAC` invalidate, then verify the surface. The
async branch kzallocs its own args copy (`rknpu_job.c:157-165`) so the user struct
is never written back, which is why the output buffer is the only signal.

Two traps it documents: the poll must run on an **A76**, not an A55 (1.27x
end-to-end on M=1 decode otherwise), and a *blocking* submit miss hard-wedges the
NPU while a doorbell miss self-heals — so async is the safer path, not the riskier
one.

**k=4096 is a hard wall for this recipe, not a conservative cap.** Raising the INT8
limit to accept k=9216 **hangs the device**: `DRM_IOCTL_RKNPU_SUBMIT` returns
`TimeoutError [Errno 110]` at `M=1, N=2560, K=9216, cores=3`, because `banks`
collapses to its `0x1b` floor at 18 groups. So `ffn_down` stays split into three K
tiles and **323 submits per forward is the floor**. The reference's 11264 figure
does not transfer.

**A fused CPU kernel must disable FP contraction to stay bit-exact.** The CLANG
renderer compiles at `-O2` with clang's default `-ffp-contract=on`, so
`t*w + b` fuses to `fmla` and rounds once where the unfused chain rounds twice.
Add `#pragma clang fp contract(off)` to the source; verify with `objdump` that
the emitted arm64 has `fmul`/`fadd` and no `fmla`.

**Biases in these models are fp16, not fp32.** `Transformer.from_gguf` defaults
`HALF=1`, so a kernel that reads a bias as `float*` reads garbage silently.

**The activation pad is already redundant.** `_QUANTIZE_SOURCE` zero-fills
`[depth, padded)`, and `_int8_padded_depth` equals the packed K for every tile,
so the caller's `x.pad(...)` is a wasted CALL.

**GGUF is not the handicap.** Q8_0 is ~8.5 bits/weight on disk; re-packing to int8
plus one fp32 row scale gives **8.01 bits/weight at k=2560** (8 + 32/2560), the
same as RKLLM's 8.

## Qwen3.5 4B — the RAM wall, and its fix

The 4B could not load in FP16: **four** OOM kills, then a load-only probe refused
on IOVA. The cause was not the packed weights alone: `bench_llm.py` realized every
parameter before packing, costing **7 GB of shmem (4 GB -> 11 GB)** before a
single NPU buffer existed.

| stage | shmem | available |
| --- | ---: | ---: |
| start | 0 GB | 14 GB |
| after `from_gguf` | 4 GB | 10 GB |
| after `realize` params | **11 GB** | **2 GB** |

The logged values do not reconcile arithmetically (14 - 11 = 3, not 2); they are
reproduced as logged rather than reconciled.

Fixed by dropping that loop on the `--npu` path: each Linear dequantizes its own
slice inside `prepare()` and drops the source.

Two hypotheses were tested and refuted, recorded so they are not retried:

- **zram tuning.** Under zstd the run still died at 4.69 GB of shmem, with only
  0.81 GB stored at 3.33x. The kernel barely swapped, so the ratio was never the
  constraint.
- **Deferring CPU weight realization as the OOM cause.** `anon-rss` was 31 MB at
  the kill because the weights live in **shmem**, not anonymous memory. The
  profile above is what found it.

## Prefill: 155 s is in the GDN blocks, and 140 s of that is unexplained

24 of the 4B's 32 blocks are gated-delta-net layers. Measured on a real
`GatedDeltaNetBlock` at the 4B config (dim 2560, ffn 9216, ssm 32x128x128, conv
8192) with **the NPU matmul stubbed out**, so these are host-only numbers:

| | per block | x 24 blocks |
| --- | --: | --: |
| GDN block, T=128 | **6.45 s** | **155 s** |
| GDN block, T=1 | 13 ms | 0.31 s |
| scan alone (`model.py:741`, 4.47 ms/iter) | 572 ms | 13.7 s |
| conv + non-scan | 47 ms | 1.1 s |
| Linear host half (68 calls x 0.65 ms) | 44 ms | 1.1 s |
| **unattributed** | **~5.8 s** | **~140 s** |

The cause is **not established**: the profiling runs that would have split the
6.45 s were killed for memory. Suspects, in order: the
`@function(precompile=True, allow_implicit=True)` at `model.py:355`, which wraps a
`_run` closure defined *inside* `__call__`, so the graph is re-traced on every
block call; and cache pressure on the 2 MB recurrent state.

### A DEBUG=2 profile narrows it to one kernel

`DEBUG=2` on a 128-token prefill (`results/debug-prefill.log`) inflates wall time
to 307.8 s against ~196 s clean, because it prints 3500+ lines, so the absolute
numbers are not usable. The ranking is:

| item | time | share |
| --- | --: | --: |
| **139-dim kernel, 24 calls x 6.28 s** | **150.7 s** | **68.9% of kernel time** |
| `E_2_16_32_32_4_4`, 3096 calls | 9.0 s | 4.1% |
| `r_3072_3_640_4`, 128 calls x 69 ms | 8.9 s | 4.1% |
| `r_2_16_32_4_32_4`, 3120 calls | 8.1 s | 3.7% |
| `E_288_80_32_8_4`, 64 x 111 ms | 7.1 s | 3.3% |
| other kernels | 35.0 s | 16.0% |
| kernel total | 218.7 s | 100% |
| — of which CPU | 214.8 s | — |
| — of which NPU | 3.8 s | — |
| `@function` wrapper, 32 blocks | 15.65 s | — |
| schedule cache misses, 62 of them | 29.7 s | — |

**24 calls, one per GDN block.** The kernel is 68.9% of all kernel time and 98%
of the work runs on the CPU, so prefill is a host problem twice over.

**The decorator is exonerated**: 489 ms per block, 7.6% of a 6.45 s GDN block, and
`schedule_cache` already hits on `function.key` for identical bodies.

The single dominant item is a reshape lowered to a **139-dimensional** kernel:

```
r_2_2_2_16_128_2_2_2_2_2_128_128_..._128_32_4     arg 132   tm 6314.22ms
```

128 of those 139 dims are size 128, and `T_pad` is 128 — one dim per token, so
something accumulates a dim per scan step instead of collapsing into the batch
axis. The `16` and `32` are the GDN head counts (`num_k_heads`, `num_v_heads`),
which places it in the recurrent block, but **which op builds it is not
established**; it is not `stack` (a native `Ops.STACK`), not `repeat`, not
`rolling_window`, and not `_causal_mask` (which returns `None` at T != 1).

That is the next thing to find: 63 s of a 196 s prefill is one kernel doing
something structurally wrong.

Two things this rules out:

- **It is not the Linear path.** Prefill is one `(1,128,2560)` tensor with no
  per-token loop, and it issues **2139 NPU calls** (24 x 68 + 8 x 62 + 11) against
  323 per decode token — 19x *fewer* per token. The Linear host half is 44 ms of
  6.45 s.
- **It is not the SSM scan.** 572 ms of 6.45 s. The reference does not move the
  scan to the NPU either (it is a serial recurrence, not a matmul); it replaces
  the Python loop with `tinygrad/llm/kernels/rockchip_recurrent.c` (491 lines),
  compiled by clang and run on the **CPU**, one call per block per chunk,
  optionally across the worker pool. Its 56.4 ms serial / 12.7 ms pooled per block
  is **2B geometry** (`~/oNPU/benchmark.md:4876-4878`), not the 4B's, so it cannot
  be applied directly. Even the slow C version would be ~1.35 s for 24 blocks.

## Why `DEV=CPU`

The baseline is the CPU host plus NPU linears, and `DEV=CPU` selects that host.
It is not optional on this board: the default device is `CL`, and a run without
`DEV=CPU` dies at `parameter.realize()` with an LLVM IR renderer error on a
2-element vector index:

```
RuntimeError: src:4:42: error: '%reg_0' defined with type '[2 x i64]*' but expected 'i64*'
  %v0 = getelementptr inbounds i64, i64* %reg_0, i32 0
```

Retained in `results/dev-cl-llvm-error.log` and `results/token-parity.log:116`.
The model in those logs is MiniCPM4 0.5B, not Qwen3.5 4B, but the claim is
model-agnostic. The failure is not root-caused; only the correlation with
`DEV=CL` is observed.
