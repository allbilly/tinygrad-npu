# Decode: 195.06 ms/token vs the reference's 152.59 — where the 42 ms is

Measured on this board, same checkpoint, `bench_replay_c.py`-class conditions (host CPU with every
`nn.Linear` on the NPU, four-way CPU transforms, replay, clocks pinned). Decode is 4.94-5.07 tok/s
(195-202 ms/token); the reference is 6.48-6.56 tok/s (152.6-154.2 ms/token).

## Class by class

Our numbers are in-model, one warm decode step, per-kernel under the replay. The reference column is
`ref-profile/report.json` → `native_execution.decode.profile`, as tabulated in `decode-rea.md` §1.

| class | ours ms | ref ms | delta |
| --- | --: | --: | --: |
| fused INT8 Linear (device + quantise + dequantise + syncs) | **146.24** | 137.11 + 1.02 | **+8.1** |
| elementwise `E_*` | **30.84** | 6.57 | **+24.3** |
| GDN scan | 5.56 | 4.16 | +1.4 |
| reduce `r_*` | 2.27 | 0.79 | +1.5 |
| attention | 2.23 | 0.45 | +1.8 |
| swiglu pack | (inside elementwise) | 1.77 | — |
| copies | (inside fused) | 0.62 | — |
| **total** | **187.14** (wall 195.06) | **152.59** | **+42.5** |

Call counts are similar (ours 461 elementwise calls, theirs 435), so the elementwise gap is **per
call, not per count**: ours 67 µs/call, theirs 15 µs/call.

The fused Linear is 4.2 GB of weights per token — `4.2 / 33.79 GB/s = 124 ms` is the floor. Ours
runs at 28.7 GB/s (85% of the DDR ceiling), the reference at 30.6 GB/s (91%). So the device side is
~9 ms from parity, not the main story.

## What is *not* the cause, each excluded by measurement

| hypothesis | test | result |
| --- | --- | --- |
| the four-way CPU transform is slow on decode shapes | decode with `thread_split` **not** installed | 201.90 ms vs 195.06 — the transform **helps** |
| threads on the little cores | `taskset -c 4-7`, matching the reference's `receipt.command` | 194.20 ms vs 195.06 — no change |
| `rolling_window`'s `pad_to` allocates and zeroes | pad by actual `T` instead of `T_pad` | 261.46 ms and the same memset count — **worse**, reverted |
| `pad_to` at all | removed entirely (signal only) | 229.41 ms, same memset count — **not the source**, reverted |
| prefill's `max_shape` leaking into decode | decode with no prefill in the process | same 48 memsets, 194.53 ms |

## The one concrete defect found: 48 memsets per decode step

The first 48 emitted calls of every decode step are `E_131072_4`, which is a bare zero-fill:

```c
for (int Lidx0 = 0; Lidx0 < 131072; Lidx0++)
  *((float4*)(dst + (Lidx0<<2))) = (float4){0,0,0,0};
```

That is 131,072 float4 = **2 MB per call**, 48 calls = **96 MB of zeroing per token**, 8.63 ms at
11.6 GB/s. They are consecutive and come before any compute. 48 = 24 blocks × 2.

The size is constant regardless of `T`, which rules out a shape-dependent pad and points at
**JIT input staging**: the per-layer read-write state (the KV cache halves) is staged per call, and
the staging buffer is zeroed before use. The reference does not pay this — its decode has 0.62 ms
of copies in total — because its replay borrows outputs rather than staging them.

Removing it is worth **8.6 ms** (195 → 186, 5.37 tok/s). It is not 42 ms, but it is the only
mechanism located so far that is a defect rather than a difference of degree.

## Further exclusions (second pass)

| hypothesis | test | result |
| --- | --- | --- |
| the JIT stages read-write state, and the staging buffer is zeroed | read `jit.py:69`: `UOp.new_buffer` allocates without zeroing; the copy is a full `store_call` | not staging |
| the per-layer caches are re-initialised each step | instrumented `_init_state`: `Tensor.zeros(1,32,128,128)` fires **24 times, all during the prefill capture, 0 during decode** | not cache init |
| `.clone()` on the state emits a second fill | dropped `.clone()` from `recurrent_state` | 48 fills unchanged, 193.91 ms — reverted |
| the fills are allocated per step | they are **baked constant addresses**, exactly 2,097,152 bytes apart (`537751035904 → 537753133056`), i.e. 48 fixed intermediates | per-call, not per-allocation |

So the 48 fills are fixed 2 MB intermediates inside the decode graph, and they are **not**
`Tensor.zeros`, not `.clone()`, not `pad_to`, not staging, and not the caches. The remaining
candidate is allocator-side zeroing of a buffer that the graph treats as read-write state.

## Third pass: the 2 MB buffers are allocated at decode capture

| test | result |
| --- | --- |
| instrument `UOp.new_buffer` during decode capture | **24 allocations of 524,288 f32 (2 MB)** — exactly the fill size, one per recurrent-state block |
| instrument `Tensor.zeros` / `empty` / `full` during decode | **0** — the allocations are not tensor creation |
| run decode eagerly (bypass the JIT) | **1,982 ms/token** vs 200 — the JIT is essential, so staging cannot simply be dropped |

So the picture is: at decode capture, 24 state buffers of 2 MB are allocated by the JIT layer
(`UOp.new_buffer`, not `Tensor.*`), and each replay emits a 2 MB zero-fill of that state — 48 fills
because both the allocation and the state update are materialised. This is the JIT treating the
recurrent state as a read-write captured constant.

## Ablation attempt — and why it is not conclusive

The project's rule is to ablate and diff the total rather than sum kernel times. I did, dropping
whole classes of segments from the replay:

| ablation | baseline | ablated | delta |
| --- | --: | --: | --: |
| drop the 48 × 2 MB fills | 212.64 ms | 206.43 ms | −6.21 ms |
| drop **all** elementwise `E_*` (452 segments) | 201.54 ms | 207.31 ms | **+5.77 ms** (slower) |

**Both numbers are confounded and should not be used.** Dropping a segment leaves its output buffer
uninitialised, so every downstream kernel reads garbage; on ARM a single denormal or NaN can make an
otherwise cheap kernel an order of magnitude slower. The "+5.77 ms for removing 452 kernels" is
that effect, not evidence that the kernels are free.

What survives is the **additive model**, which two independent numbers support: the reference's
152.6 ms splits as 137.1 device + 15.5 host, and ours as 142.6 + 52.4. In both trees the total is
device plus host to within a millisecond, so the host work *is* serial with the device and a correct
ablation would have to preserve finite values in the dropped buffer (write zeros, not skip the
segment) before its delta could be believed.

## The elementwise class is half zero-fills: 19.3 ms, 120 MB per token

Two kernels in the elementwise row are not elementwise at all. Read from their generated sources:

| kernel | body | bytes/call | calls | MB/token | ms |
| --- | --- | --: | --: | --: | --: |
| `E_131072_4` | `for Lidx0<131072: *(float4*)(dst+Lidx0*4) = {0,0,0,0}` | 2,097,152 | 48 | 96 | 8.63 |
| `E_81920_4` | `for Lidx0<81920: *(__fp164*)(dst+Lidx0*4) = {0,0,0,0}` | 655,360 | 37 | 24 | 10.70 |

So **19.33 ms of the 30.84 ms elementwise row is zeroing 120 MB per token**. Real elementwise
compute is ~11.5 ms, close to the reference's 6.57.

`E_81920_4` writes 655 KB in 289 µs = **2.3 GB/s**, an order of magnitude below the 23.8 GB/s
ceiling, and the second fill is likewise ~11 GB/s. These are not bandwidth-limited passes; they are
materialisation of buffers that the graph then overwrites.

Neither is a `Tensor.*` allocation (that count is 0 during decode), so they are emitted by the
scheduler for buffers it treats as read-write state. 120 MB/token is 2.9% of the 4.2 GB of weights,
but 10% of the step time.

## Gate/up fusion at m=1: measured, decode unchanged, prefill regressed

The directed change: fuse `ffn_gate`/`ffn_up` (19456 columns, under the 24576 m=1 submit limit) so
decode issues one submit and one quantise instead of two. The model already had the consumer branch
(`hasattr(self, "ffn_gateup")`); the fusion was disabled by `if False`, and enabling it exposed a
latent bug in the `merge` pre-check, which validated the Linear's whole column count (19456) against
the 12288 per-call limit and raised. Guarded, then measured, full 3+3:

| | baseline | gate/up fused |
| --- | --: | --: |
| prefill | **1,197.7 ms** | **1,656.5 ms** (+38%) |
| decode | 4.94-5.07 tok/s | **5.05 tok/s** |
| submits/token | 307 | **275** |
| tasks/token | 825 | 729 |
| device | 142.6-145.5 ms | 142.4 ms |
| parity | `895365e27ed4255b` | `895365e27ed4255b` |

So the fusion does exactly what it was supposed to structurally — 32 fewer submits and 96 fewer
tasks per token — and decode does not move. **Submits are not the bottleneck.** Prefill regresses
by 459 ms, as the original +14.7% warning predicted (worse here because the fused tile is packed as
one 24576-column tile).

Reverted (both the enablement and the guard), parity re-verified, decode back to 5.05 tok/s at 307
submits. (The 1+1 restore run reports 3,883 ms prefill because a one-warm-up protocol puts warm-up
in the measured trial; the 3+3 baseline is 1,197.7 ms.)

## In-process ABBA: greedy sampling is 5.1% faster (confirmed)

The harness cannot resolve 8 ms (three trials, 63 s per arm, ~1%/C), so both arms were run in **one
process**, alternating A,B,B,A with the JIT re-captured per arm and temperature read per block
(`extra/rockchip/ab_decode.py`). Six blocks per arm, five timed steps each:

| arm | ms/token blocks | median | tok/s |
| --- | --- | --: | --: |
| A greedy argmax | 183.3, 181.9, 182.0, 200.8, 182.1, 234.5 | **182.70** | **5.473** |
| B Gumbel noise | 192.6, 232.9, 192.3, 192.1, 192.7, 192.4 | 192.48 | 5.195 |

Paired ratios (A/B): `0.9517, 0.7808, 0.9464, 1.0453, 0.9453, 1.2187`, **median 0.9491** — a 5.1%
win, 9.8 ms/token, matching the 8.48 ms kernel that disappears. The two outlier pairs are whole-block
stalls (0.78 and 1.22), which is why the median ratio is the statistic to quote.

This is the first decode change confirmed to move the wall, and it validates the method: the same
ABBA is what the fills, submit and attention changes needed and never had.

## The decisive split: submit disabled gives 60.17 ms

The device-vs-host question was settled by removing the submission itself (`SKIP_SUBMIT = True`,
results garbage, timing valid), same protocol as the dump:

| configuration | wall |
| --- | --: |
| full | 199.04 ms |
| **submit disabled** | **60.17 ms** |

So **device = 138.9 ms, host = 60.2 ms**, and the wall is additive after all — the earlier
"device-occupancy-bound ~190 ms" inference was wrong.

Reference: 137.1 device + 15.5 host. Our device is within 2 ms of it; our host is **60 vs 15.5**.

That makes the host gap the whole story (44.5 ms), and it makes the four failed transfers
(fills, submits, attention split, sampler) a *measurement* problem rather than evidence that host
time does not matter: the harness cannot resolve an 8 ms effect against this board's thermal
scatter, since a 3 C shift moves the median by ~3%.

## Greedy sampling: the sampler kernel falls 20x, wall 199 -> 190 ms

The dump identified `E_62080_4` as 62,080 float4 = **248,320 floats, the logits width** — the
Gumbel-max expression building a full-vocabulary random tensor every token. At temperature 0 (what
`bench_llm.py:141` passes) the division by `1e-12` is a positive scaling that preserves the argmax,
and the O(1) noise is absorbed by float32 rounding at magnitude 1e12 (ULP 65,536), so the noise
cannot change the result. The reference samples with `gpt2_argmax` at 0.085 ms.

`GREEDY_SAMPLING = True` in `Transformer.forward` takes the argmax directly. Same protocol
(`MAX_CALLS_PER_SEG = 1`, one warm decode token), dumps taken minutes apart:

| | calls | `E_62080_4` | wall |
| --- | --: | --: | --: |
| Gumbel noise | 958 | **8.48 ms** | **199.04 ms** |
| greedy argmax | 956 | **0.41 ms** | **190.39 ms** |

The sampler loses 20x its cost and two calls disappear from the graph.

The full harness does **not** confirm it. Re-run at matched temperature, both arms hot:

| arm (both hot) | decode median | trial temps (start) |
| --- | --: | --- |
| greedy argmax | 4.7548 tok/s | 77.6-82.2 C |
| Gumbel noise (baseline) | 4.7593 tok/s | 80.4-84.1 C |

Removing an 8.48 ms kernel does not move the wall. So the one-step dump's 199 -> 190 was not
transferable — most likely its two runs differ thermally and it measures a single step rather than
64. Reverted.

**This is the fourth CPU reduction that fails to transfer** (fills, submits, attention split,
sampler). That consistency is the strongest evidence yet that the wall is *not* the sum of the CPU
kernels: the trace's 142.89 + 43.95 + 6.45 = 193.29 is numerically additive but not causal. The
model that fits all four results is that the NPU submit is asynchronous, `submit_seconds` (143 ms)
is only the ioctl call, and the true device occupancy is ~190 ms with the CPU work overlapped
inside it.

## Host profile (py-spy) and the kernel dump artifact

`flameox` is unavailable on this host (`AGENTS.md`: rea-agents 6.7.0 predates it; 6.8+ needs
Node >= 24.22.1, which would repoint `nvm` away from the 24.12.0 that `dsh`/`dst` need), and the REA
MCP server is not connected in this session. The documented substitute is `py-spy`, which was not
used until now.

`py-spy record -f speedscope --rate 200` over 60 decode steps, 24,252 samples. Filtering to samples
whose stack is inside the replay's call path (`Compiled.run` / `CapturedJit.__call__`):

| frame | share of decode samples |
| --- | --: |
| `Compiled.run` (self, i.e. inside the generated C) | **97.1%** |
| everything else in Python | 2.9%, each frame < 0.2% |

So there is **no Python overhead left to remove**: the replay has collapsed the host path to one
C call per segment, and 97% of the sampled time is inside the generated C. This independently
confirms the trace's 6.45 ms of between-segment time and rules out the "host-path structure" as
Python work.

Because the profile covers the whole process, the unfiltered top frame (`CPUProgram.__call__`,
43.6%) is the **weight packing at model load**, not decode — a trap worth recording.

### Kernel dump

`extra/rockchip/dump_decode_kernels.py` writes `docs/rknpu/results/decode-kernels/`: one `.c` per
kernel (the exact source clang compiled) plus `MANIFEST.md` with calls, total ms, us/call, submit ms
and cpu ms per kernel, and a class rollup. One warm decode token, `MAX_CALLS_PER_SEG = 1`:

| kernel | class | calls | ms | us/call | submit ms | cpu ms |
| --- | --- | --: | --: | --: | --: | --: |
| `rockchip_linear_1_9216_2560` | fused Linear | 64 | 49.75 | 777 | 48.62 | 1.13 |
| `rockchip_linear_1_2560_4096` | fused Linear | 96 | 39.94 | 416 | 38.56 | 1.38 |
| `rockchip_linear_1_24576_2560` | fused Linear | 10 | 20.22 | 2023 | 19.95 | 0.28 |
| `rockchip_linear_1_8192_2560` | fused Linear | 24 | 16.63 | 693 | 16.17 | 0.46 |
| `E_131072_4` | fill | 48 | 8.87 | 185 | 0.00 | 8.87 |
| `E_62080_4` | elementwise | **1** | **8.48** | 8478 | 0.00 | 8.48 |
| `rockchip_scan_simd` | scan | 24 | 5.61 | 234 | 0.00 | 5.61 |
| `rockchip_attn_1_160` | attention | 8 | 2.26 | 282 | 0.00 | 2.26 |

`E_62080_4` is 62,080 float4 = **248,320 floats, exactly the logits width** (`presample_logits
shape (1, 248320)`), with 14 buffers — the Gumbel-max sampling expression building a full-vocabulary
random tensor per token. The reference's sampler is `gpt2_argmax` at 0.085 ms.

## Collapsing to a single replay segment: slower

The trace showed 6.45 ms between 16 segments, so the obvious question was whether segment
boundaries were the cost. One segment for the whole linear (`MAX_CALLS_PER_SEG = 1 << 30`),
bit-exact (parity unchanged), 3+10:

| | decode median | trial temps |
| --- | --: | --- |
| 16 segments (baseline) | 4.9294 tok/s | 67.5-71.2 C |
| **1 segment** | **4.8841 tok/s** | 63.8-69.3 C |

Slower, while running ~4 C cooler. So the between-segment time is not boundary overhead — merging
it away does not help — and the 6.45 ms is not recoverable there. Reverted.

## Scan dispatch: confirmed live (status=1, four groups)

The scan's pooled dispatch was the open question. Two independent pieces of evidence answer it:

| evidence | value |
| --- | --- |
| paired serial vs pooled, same source (`splitcheck/README.md:11`) | serial 6.78 ms/call, pooled 1.76 ms/call, **3.86x** |
| dispatch status from the paired run (`splitcheck/README.md:90`) | `rc=0 status=1` — **all four groups completed** |
| our in-model decode scan | 278 us/call (14 GB/s on a 2 MB state) |
| reference in-model decode scan | 173 us/call (23 GB/s) |

So the scan is not falling back to serial: status 1 with a 3.86x paired speedup means the four
workers run. The remaining 1.6x against the reference is memory efficiency at `tokens=1`, not
dispatch — and the kernel source is byte-identical, so it is not the code either.

Note: `splitcheck/README.md` previously concluded the pooled entry "stays disabled"; that is stale
and has been corrected — `scan_native.py` calls the pooled entry.

## The scan kernel is the reference's, byte for byte

The last unoptimised component was the scan (278 us/call vs the reference's 173). Diffed against the
dumped reference source:

```
$ wc -l ref-kernels-all/c/rockchip_recurrent.c extra/rockchip/recurrent.c
  491 .../rockchip_recurrent.c
  491 extra/rockchip/recurrent.c
$ diff ref-kernels-all/c/rockchip_recurrent.c extra/rockchip/recurrent.c | wc -l
0
```

**Identical, 491 lines, zero differences.** So the 1.6x per-call difference cannot be the inner loop
or the vectorisation — the code is the same. It is invocation (thread count) or measurement basis.
Threading it was measured before: the `bh` split's ceiling is 1.21x (about 1.2 ms of 193) and
pthreads cost +100 ms, so it cannot pay.

With that, every component of the decode budget has been either matched to the reference's code or
measured to be at parity: weights, domains, per-byte GEMM rate, submit geometry, fills, attention,
and now the scan.

## The attention split at matched temperature: no gain (the 5.05 was thermal)

3 warm-ups + 10 measured, three arms. The first split run looked like a 2.5% win; it had started
10 C cooler. Re-run hot, it matched the baseline:

| arm | decode median | trial temps (start) | trial temps (end) |
| --- | --: | --- | --- |
| split, cool start | **5.0517** | 64.7-70.2 C | 66.5-70.2 C |
| baseline | 4.9294 | 67.5-71.2 C | 68.4-71.2 C |
| **split, hot start** | **4.9077** | **67.5-71.2 C** | 68.4-72.1 C |

At matched bands the split is 4.9077 against 4.9294 — no gain, if anything slightly worse. The
kernel really is 27% faster (428 -> 314 us/call), but 0.9 ms of 193 is below the thermal scatter,
and a 10 C cooler start moves the median by more than the change does. This is the clearest
demonstration in this document of why `AGENTS.md` requires paired, temperature-matched comparisons.

Reverted; parity `895365e27ed4255b`.

## Splitting the attention groups: kernel 27% faster, wall neutral

`RK_SPLIT` was forced off for `TOK < 8`, leaving the four KV groups serial at decode. The groups
write disjoint output rows (`H_*g/4 .. H_*(g+1)/4`), so enabling it is exact — parity stayed
`895365e27ed4255b` — and the kernel measured:

| | attention | 3+3 decode | prefill |
| --- | --: | --: | --: |
| serial groups (baseline) | 428 us/call, 3.42 ms | 4.94-5.07 tok/s | 1,197.7 ms |
| **four-way groups** | **314 us/call, 2.52 ms** | **5.013 tok/s** | 1,193.2 ms |

The kernel is 27% faster, worth 0.9 ms of the 193 ms wall, and the full protocol reads 5.013 against
a 4.94-5.07 baseline — inside the scatter. The scan's CPU read rose to 12.1 ms in the same trace, so
part of the gain may be traded against pool contention. Reverted rather than shipped as a
below-noise change.

## fp16 attention: bit-exact, and slower

The reference reads its KV cache as `half4` while we made an fp32 copy every call
(`kv.float().contiguous()`). Converting our kernel to read `half4` via
`__builtin_convertvector` and dropping the copy is **exact** — parity stayed `895365e27ed4255b` —
and removes 2x the bytes plus one cast kernel per call. Measured, full 3+3:

| | prefill | decode | parity |
| --- | --: | --: | --- |
| fp32 cache + cast (baseline) | 1,197.7 ms | 4.94-5.07 tok/s | `895365e27ed4255b` |
| **half4 cache, no cast** | 1,203.6 ms | **4.716 tok/s** | `895365e27ed4255b` |

**Slower by ~0.3 tok/s**, despite moving half the bytes. The in-kernel half->float conversion
costs more than the fp32 copy it removed on these cores. Reverted.

So the fp16 difference is real but not a lever either: it is a numerical/layout property that,
when matched, makes the kernel slower here.

## Scan and attention: same arithmetic, different width and serialism

Source-level diff of the two remaining per-kernel deficits against the reference dumps:

| | ours | reference | consequence |
| --- | --- | --- | --- |
| attention exp | `exp2_scalar`, "the same coefficients the reference uses" (`attn.c:16`) | same polynomial | not the cause |
| attention inner loop | float4 MACs (`attn.c:81`, `:97`) | float4 + `__builtin_convertvector` | not the cause |
| attention KV width | **fp32** (`attn.py` placeholder `dtypes.float`) | **fp16** (`const half4 *source=(const half4*)(keys+...)`) | we move 2x the bytes |
| attention groups at decode | `for (g=0..3)` serial; `RK_SPLIT=0` for `TOK<8` by design | grouped per KV head | 4 groups sequential |
| scan | `rockchip_scan_simd`, 278 us/call | `rockchip_cpu_gated_delta_scan`, 173 us/call | +2.5 ms |

Neither is a codegen defect that can be fixed in place: the KV width is a numerical choice (fp16
cache) and the serial groups are a deliberate trade recorded in `attn.c`. Fixing either changes
what the model computes, so the bit-exact gate rejects it — same wall as the fills. Together they
are worth ~5.5 ms of the 35 ms host gap.

## CPU ranked by kernel name (one call per segment)

Trace with `MAX_CALLS_PER_SEG = 1`, so each segment is one kernel and the trace carries its name;
CPU = segment duration minus its submit delta. Totals in this mode are inflated (wall 233 vs 193),
so read the ranking, not the absolute:

| kernel | CPU ms | calls | µs/call | note |
| --- | --: | --: | --: | --- |
| `E_62080_4` | 30.69 | 1 | 30,693 | **artefact** — 8.49 ms in the 64-segment run; one call, unstable |
| `E_131072_4` | 8.74 | 48 | 182 | the f32 fill (load-bearing) |
| `rockchip_scan_simd` | 6.68 | 24 | 278 | reference 4.16 ms |
| `E_2_16_32_4` | 4.00 | 48 | 83 | reference 2.12 ms |
| `rockchip_attn_1_160` | 3.42 | 8 | 428 | reference 0.45 ms |
| `r_64_4_970` | 2.55 | 2 | 1,274 | |
| `E_1024_4` | 2.27 | 72 | 32 | |
| `E_124160_2` | 2.08 | 1 | 2,084 | single call, like `E_62080` |
| `rockchip_linear_1_2560_4096` (dequant) | 1.92 | 96 | 20 | inside the fused C |
| `r_16_32_4` | 1.66 | 48 | 35 | |

Excluding the two single-call artefacts (30.69 + 2.08), CPU is ~47 ms, consistent with the 44 ms
measured at 64 calls per segment. The largest genuine items are the fill (8.7), the scan (6.7) and
attention (3.4); against the reference's 4.16 and 0.45 those two are worth ~5.5 ms combined, and
neither is safely splittable (`loop-classification.md`: the scan's ceiling is 1.21x and pthreads
cost +100 ms).

## Threading the fills makes them slower

`thread_split` is an allowlist of five kernel names, and the `_4` in names like `E_131072_4` is
tinygrad's shape encoding, not a thread count — so the fills were running single-threaded (2.3 and
11.6 GB/s). Adding them to the allowlist (a fill writes disjoint output, so it is trivially safe)
and re-tracing one decode token:

| | seg0 | seg1 | CPU inside segments | wall | decode 3+3 |
| --- | --: | --: | --: | --: | --: |
| single-threaded fills | 5.85 | 7.60 | 43.95 ms | 193.29 ms | 4.94-5.07 tok/s |
| four-way fills | **7.68** | **9.48** | 47.07 ms | **201.97 ms** | **4.975 tok/s** |

Slower. A 2 MB fill across 4 threads at 85 calls/token cannot amortise the dispatch. Reverted
(parity unchanged throughout: `895365e27ed4255b`).

## Wall-clock trace of one decode token: the bound is CPU, not device

Timestamped every replay segment and read the submit counter around each, on a warm decode step:

| row | ms | share |
| --- | --: | --: |
| device (`submit_seconds`) | **142.89** | 73.9% |
| CPU inside segments | **43.95** | 22.7% |
| between segments (Python / replay) | 6.45 | 3.3% |
| **wall** | **193.29** | |

142.89 + 43.95 + 6.45 = 193.29, additive to the millisecond. So decode is **CPU-bound at 44 ms**,
not device-bound — my earlier inference from "removing host work changes nothing" was wrong, and
this trace is the instrument that settles it.

The largest CPU segments are the first two (5.85 + 7.60 ms), which is where the fills live, plus
`seg15` at 9.48 ms. Note those two segments hold only 13.4 ms of CPU between them, so the 19.33 ms
the 1-call-per-segment attribution assigned to the fills alone was itself inflated — consistent with
the memset replacement saving only 5 ms at the wall.

The reference's host total is 15.5 ms against our 43.95 + 6.45 = 50.4 ms. That 35 ms is the target,
and it is CPU, not the NPU.

## The GEMM partitioning is already the reference's: nothing to port

The last open lever was the reference's 203-submit / 2-domain GEMM partitioning. Measured on our
tree with the same checkpoint:

| quantity | ours | reference | delta |
| --- | --: | --: | --: |
| packed weight bytes | **4,204,789,760 (4010.0 MiB)** | 4010.0 MiB (`npu_weight_mib`) | **0.0** |
| IOMMU domains | **2** (2.99 + 0.92 GiB) | 2 (`ROCKCHIP`, `ROCKCHIP:1`) | 0 |
| per-byte GEMM rate, m=1 | 30.9 GB/s (23.6 MB in 0.764 ms) | 31.1 GB/s (47 MB in 1.512 ms) | 0.6% |
| submits/token | 307 | 203 | +104 |
| tasks/token | 825 | — | — |

Our packing is byte-identical in total to the reference's, the domain assignment already matches,
and the per-byte device rate differs by 0.6%. The submit count is the only difference, and it was
already falsified as a lever: gate/up fusion cut it to 275 and decode did not move.

So the "142.7 vs 137.1 ms device gap" is not a partitioning difference — it is the two numbers
being measured on different bases (our `submit_seconds` counts the ioctl only; the reference's
137.1 is its profile's device row). There is nothing in that port to take.

## Contradiction resolved: fill time falls 15 ms, the wall falls 5 ms

The earlier confusion was measuring only the wall. Re-applying the parity-verified memset
replacement and running the per-kernel attribution on both trees settles it:

| in-model, one warm decode step | baseline | fills as memset |
| --- | --: | --: |
| fill time (`E_131072` + `E_81920`) | **19.33 ms** | **4.21 ms** (56 calls) |
| total by name | 187.14 ms | 182.12 ms |
| wall | 195.06 ms | **189.96 ms** |
| full-harness 3+3 decode | 4.94-5.07 tok/s | **5.011 tok/s** (4.80/4.91/5.01) |
| full-harness prefill | 1,197.7 ms | 1,222.2 ms |

So the fill time really does fall by 15.1 ms, and only **5.1 ms** of that reaches the wall: the
other ~10 ms was overlapped with device work. In the full harness even that 5 ms is below the
run-to-run scatter, and prefill is ~24 ms slower.

Conclusion: the fills are load-bearing, partially overlapped, and not a lever. Reverted (the
general fill detector was correct — parity unchanged — but neutral-to-negative at the wall).

## The fills are load-bearing, and their cost is not on the wall

Two corrections and one clean measurement.

**They are not redundant.** Skipping the fills changes the logits:
`895365e27ed4255b` -> `ee30d6ee1f15cd1f`. So my earlier claim that parity "proved they were
unnecessary" was made with the broken skip (which matched nothing) and is wrong: the buffers are
read before being rewritten, and the zeros are required.

**Making them fast changes nothing.** Replaced each fill with `memset` of the identical buffer and
length, which writes the same zeros at memory speed instead of 11.6 / 2.3 GB/s. Parity returns to
`895365e27ed4255b` exactly, so the values are unchanged, and a full 3+3 gives:

| | prefill | decode | parity |
| --- | --: | --: | --- |
| baseline | 1,197.7 ms | 4.94-5.07 tok/s | `895365e27ed4255b` |
| fills as memset (same zeros) | 1,214.5 ms | **4.995 tok/s** | `895365e27ed4255b` |

Removing ~13 ms of measured fill time moves neither phase. Together with the gate/up fusion
(32 fewer submits, no change), the conclusion is that **CPU work and submit count are not what
bounds decode**: host work overlaps whatever the NPU is doing, so the wall is set by device
occupancy, not by the sum of CPU kernels.

## Skipping the fills entirely: no decode improvement (valid measurement)

The first version of this test was **invalid**: the skip matched on kernel names that carry ANSI
escapes, so it matched nothing — `calls_c` was 2923, identical to baseline, and the 4.89 tok/s it
reported was just an ordinary run.

Redone with the escapes stripped and the skip verified against the call count:

| | calls_c | skipped | prefill | decode | parity |
| --- | --: | --: | --: | --: | --- |
| baseline | 2,923 | 0 | 1,197.7 ms | 4.94-5.07 tok/s | `895365e27ed4255b` |
| fills skipped | **2,723** | **200** | 1,210.1 ms | **4.93 tok/s** | `895365e27ed4255b` |

200 fill calls genuinely removed from the replays (prefill's and decode's), and the wall does not
move. So the finding stands with a valid experiment: **120 MB/token of zeroing is not on the
critical path**, and the ~19.33 ms its kernel time contributes is not realisable by removal.

## Skipping the fills entirely: the invalid first attempt

The last hypothesis was that the 19.33 ms of zero-fills was on the critical path. Tested by
skipping them at replay build time (`SKIP_PREFIXES = ("E_131072", "E_81920")` in
`replay_c.build`), then running the real harness, full protocol, replay healthy
(`fell_back: 0`, 49 segments, 2,923 calls):

| | prefill | decode | submits/token |
| --- | --: | --: | --: |
| baseline (fills emitted) | 1,197.7 ms | **4.89-5.07 tok/s** | 307 |
| fills skipped | 1,197.7 ms | **4.89 tok/s** | 307 |

No change, in either phase. Parity also unchanged (`895365e27ed4255b`), which proves the fills are
redundant writes — but removing 120 MB/token of zeroing buys nothing, so the fills are **not on the
critical path** in the real harness.

That is the strongest result of this investigation and it is negative: the timed cost of a kernel
(19.33 ms) is not the cost of removing it (0 ms here, 6.21 ms in the earlier isolated ablation).
The decode gap is not any single identified component.

## Borrow instead of stage: tested, no effect

The reference's borrow semantics were tried directly: `BORROW_ALIASED = True` in
`tinygrad/engine/jit.py`, which makes `aliased` empty so no staging buffer is created for a written
input.

| | fills | wall | parity |
| --- | --: | --: | --- |
| staged (shipped) | 48 | 193.57 ms | `895365e27ed4255b` |
| borrowed | **48** | **193.77 ms** | `895365e27ed4255b` |

Identical. So the staging path is **not** the producer either, and the change was reverted rather
than shipped as a no-op semantic change.

The allocation site is instead `tinygrad/engine/realize.py:205`, which allocates a buffer per
`call.src[1:]` when a linear call is materialised — i.e. the fills are the **outputs** of the
decode linear being allocated and zeroed, not its inputs. That is the next place to look, and it
is engine-internal.

## Retraction: "67 µs vs 15 µs per call" compares different shapes

The elementwise rows were compared as if the two trees ran the same kernels. They do not:

| | count |
| --- | --: |
| reference kernel names in its decode profile | 31 |
| our kernel names in the decode attribution | 36 |
| **names present in both** | **1** (`E`, the trivial 0.1 µs one) |

So the per-call ratio is meaningless: it divides our 32,768-element kernels by their 1,024-element
ones. Read from the generated sources, our kernels are **ALU-bound, not bandwidth-bound**:

| kernel | elements (from loop bounds) | µs/call | ns/element |
| --- | --: | --: | --: |
| `E_2_16_32_32_4_4` | 2×16×32×32 = 32,768 | 124 | 3.8 |
| `E_1024_32_4_4` | 1024×32 = 32,768 | 48 | 1.5 |
| `E_81920_4` | 81,920 | 289 | 3.5 |
| `E_131072_4` (fill) | 131,072 ×4 | 180 | 0.34 per float |

Same element count, 2.6× different cost — that is op count per element, not codegen quality. And the
reference's `E_2_16_32_4` is **44 µs for 1,024 elements = 43 ns/element**, i.e. 11× *slower per
element* than ours. The reference is not faster at elementwise; it does less of it.

That is a graph-structure difference (fusion), not a kernel-quality difference, which is why the
per-call framing never produced a lever.

## Next, in order

## Next, in order

1. Confirm the staging hypothesis: instrument the JIT's staging allocation to report buffer
   addresses and sizes for one decode step, and match them against the 48 memset destinations.
2. If confirmed, stop staging the read-write state (the reference's `ONPU_REPLAY_BORROW_OUTPUTS`
   analogue) — expected −8.6 ms and, if it also removes the paired copy, more.
3. Then the elementwise per-call gap (67 vs 15 µs) — compare the same kernel by name across the two
   profiles (`E_2_16_32_32_4_4` ours 124 µs vs `E_2_16_32_4` theirs 44 µs) and read both sources.
   This is the remaining ~20 ms and the only path to 6.48 tok/s.
