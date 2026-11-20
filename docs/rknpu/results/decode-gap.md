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

## q/dq row closed: the code has no toggle for it

`npu_linear` exposes toggles only for `PARALLEL_QUANT`, `PARALLEL_DEQUANT`, `SKIP_SYNCS` and
`SKIP_SUBMIT`. Quantise and dequantise themselves are **inside the fused C call** and cannot be
switched off without removing the call's purpose; the only alternative that separates them is the
unfused path, already measured **1.43-1.80x slower**. So the +3.7 ms row is not "unoptimised" -- it
is the price of the design that wins by 43-80%, and no ABBA arm exists to close it.

Gaps re-derived per protocol: **8.95 ms** on the 1-call-per-segment dump, **6.45 ms** on the shipped
64-call path (the trace). Both are carried in the table because they are different instruments.

## Elementwise row: the transform arm was a placebo

The elementwise ABBA toggled `thread_split.THREAD_SPLIT` between its full table and empty. It read
4-way 197.01 vs serial 210.81 ms (ratio 0.8864), which looks like a 13 ms win -- but the arm check
said `transform_hits=0` in **both** arms, and the reason is decisive:

```
THREAD_SPLIT entries: ['E_2_16_32_32_4_4', 'E_1024_32_4_4', 'E_32_32_128_4',
                       'r_32_16_4_32_4', 'E_393216_3']
decode kernels covered by the transform table: NONE   (of 35 in the dump)
```

The table holds **prefill shapes only**. Both arms therefore ran identical code, the ratios scattered
0.68-1.64 as noise, and the 0.8864 is not a result. The elementwise row (+8.2) stays **open**, and its
lever is not "turn the transforms on" -- they never apply in decode -- but "transform decode shapes",
which the fills experiment already showed is not free (adding `E_131072_4` made the fills slower).

## The 182 vs 152 table

Our column is the shipped configuration; the reference column is
`ref-profile/report.json` (46 runs, its own 36-flag recipe). Classes are matched by hand because the
trees share **one kernel name out of 31 and 36**, so a per-kernel join is impossible.

| class | ours, dump (ms) | ours, greedy (ms) | reference (ms) | delta vs ref |
| --- | --: | --: | --: | --: |
| NPU GEMM (device submit) | 142.89 | 142.89 | 137.11 | +5.8 |
| quantise / dequantise / syncs (inside the fused call) | 4.70 | 4.70 | 1.02 | +3.7 — **closed: no toggle exists** |
| elementwise (incl. sampler) | 22.81 | 14.74 | 6.57 | **+8.2 (measured: 4-way split null 1.0031; word-load rewrite -2.5 ms CPU / -3.1 ms wall)** |
| — of which the sampler | 8.48 | 0.41 | 0.09 | +0.3 |
| state fills | 9.52 | 9.52 | ~0 | +9.5 |
| GDN scan | 5.23 | 5.23 | 4.16 | +1.1 |
| reduce | 2.68 | 2.68 | 0.79 | +1.9 |
| attention | 2.26 | 2.26 | 0.45 | +1.8 |
| Python / replay gaps | 8.95 (dump) / **6.45 (shipped)** | 8.95 | ~0 | **open; two protocols, named** |
| **total** | **199.04** | **190.97** | **152.59 (archived) / 157.34 (measured tree)** | **+38.4 / +33.6** |

**Reference basis, corrected:** the archived `ref-profile` gives 6.48-6.56 tok/s (152.59 ms), but the
measured tree's own retime receipt reports `median_decode_tps: 6.355455` = **157.34 ms** over 1+3 at
128/64. Against our ABBA headline of ~200 ms the gap is therefore **~+43 ms against the receipt**
and +30.11 against the archive; both are quoted, and the receipt is the like-for-like one.

Two reconciliations, both stated because they are different protocols:

- the greedy column is the dump column with the sampler at its measured 0.41 instead of 8.48, giving
  190.97; the measured greedy dump is **190.39**, a 0.6 ms agreement.
- the ABBA headline of **~200** sits ~9 ms above the dump's 190.97 because the dump runs one call
  per segment (958 ctypes calls) to attribute per kernel, and the ABBA runs the shipped 64-call
  segments. So: class shares come from the dump; the ABBA total (~200) comes from the ABBA; neither
  number is quoted in place of the other.

Reference columns sum to 152.585 as published. Our classes sum to the dump wall with a 8.95 ms gap
row; that gap is the replay's between-segment time, which the reference's native replay does not pay
(it issues one C call per linear).

## ABBA: the attention split is 9.4% SLOWER at the wall

The split made the attention kernel 27% faster in isolation (428 -> 314 us/call), which the harness
could never resolve. In-process ABBA with the arms verified by their generated `#if RK_SPLIT` line:

| arm | ms/token blocks | median | ratio |
| --- | --- | --: | --: |
| A split 4 (`#if 1`) | 199.6, 201.9, 184.4, 203.2, 222.3, 201.1 | 201.46 | — |
| B serial (`#if 0`) | 202.2, 183.3, 183.8, 183.5, 203.3, 183.7 | **183.77** | **median 1.0940** |

**Serial is 9.4% faster.** The four pooled workers compete with the scan and the elementwise kernels
for the same four big cores, and that costs more than the kernel saves. The shipped configuration was
already serial, so nothing changes -- but the "27% faster kernel" is now correctly recorded as a
decode regression, not a pending win.

## ABBA status: greedy confirmed, fills measured and rejected

The first fills run reported `memset_fills=0` in every arm. That was **my measurement bug**: the
counter increments at *capture*, and the harness read it after the warm-up steps. Verified properly
with a multi-step run — `built=1 calls_c=900 memset_fills=56` — so the replacement does fire.

Re-run with the counter captured before warm-up, arms verified per block:

| arm | ms/token blocks | median |
| --- | --- | --: |
| A memset fills (`memset_fills=56`) | 198.4, 241.2, 244.5, 198.7, 210.4, 178.1 | 204.54 |
| B kernel fills (`memset_fills=0`) | 207.6, 183.4, 201.8, 183.3, 207.1, 201.5 | 201.62 |

Paired ratios `0.9554, 1.3152, 1.2117, 1.0838, 1.0159, 0.884`, **median 1.0498**. The spread is
wide (0.88-1.32), so the honest reading is **no gain** rather than 5% slower — the 15 ms of fill
time removed in isolation does not reach the wall. Toggle ships off.

The greedy arm is confirmed (below) and kept.

## In-process ABBA: greedy sampling is NOT faster (retracted)

**Retracted.** The first run used 3 rounds (6 pairs) and gave median 0.9491, which I reported as
"confirmed" despite a sign test of p = 0.344. Re-run with 8 rounds (16 pairs):

| arm | median |
| --- | --: |
| A greedy argmax | **202.83 ms** |
| B Gumbel noise | **199.26 ms** |
| paired ratio | median **1.0065**, **8 of 16** pairs favour greedy |

No gain. The sampler kernel genuinely falls from 8.48 to 0.41 ms and two calls leave the graph, but
the wall does not move -- the same pattern as the fills, the attention split and the submit
reduction: removing CPU work here does not reach the wall.

Three independent ABBAs, with their bands (the third re-run retained the per-block temperatures that
the second discarded):

| run | band | pairs | median ratio | A | B |
| --- | --- | --: | --: | --: | --: |
| 1 (6 pairs) | 70-72 C | 6 | 0.9491 | 182.70 | 192.48 |
| 2 (16 pairs) | not recorded | 16 | 1.0065 | 202.83 | 199.26 |
| **3 (16 pairs, temps kept)** | **64.7-70.2 C** | 16 | **0.9991** | **192.09** | **192.24** |
| 3, restricted to the reference's band 69.3-76.7 C | 69.3-70.2 C | 5 | 0.9789 | 192.8 | 196.0 |

Verdict: **the sampler has no effect** (0.9491, 1.0065, 0.9991 -- no consistent direction, and the
one band-matched subset is n=5). GREEDY_SAMPLING stays reverted.

The third run is also the most stable absolute measurement of the session: 14 of 16 blocks are
191.9-192.3 ms, so **our decode is ~192 ms at 64.7-70.2 C**. Against the reference's 157.34 ms at
69.3-76.7 C that is a gap of **~+35 ms**, and unlike the earlier numbers this one is stable enough
to compare.

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


## The host rows, measured with arms that actually fire

Every in-process ABBA in this tree reset the JIT between arms but not `to_program_cache`
(`tinygrad/codegen/__init__.py:539`), which memoises the PROGRAM uop **including the compiled
binary** and is keyed on a structurally identical ast. So an arm that changed only the emitted
source ran the *other* arm's binary: the comparison was an arm against itself, and it could only
ever return ~1.0. Both harnesses now clear it, and `ab_elem` raises if arm A transforms nothing.

### 1. Fused-call CPU split: corrected, and it is not the 19 us/call I said

Timers around both `sync_mem` ioctls in the emitted C, over one decode step, 307 fused calls:

| component | ms/token | us/call |
| --- | --: | --: |
| submit (device) | 142.65 | 464.7 |
| quantise | 0.83 | 2.7 |
| both syncs | 0.87 | 2.8 |
| dequantise | 1.20 | 3.9 |
| **CPU inside the fused call** | **2.90** | **9.4** |

So the fused call's host half is **2.9 ms/token**, not the ~7.7 ms implied by a "19 us/call
remainder": the two cache-maintenance ioctls cost 2.8 us/call combined, not ~14. Nothing in this
row is worth attacking -- the largest item is 3.9 us/call of dequantise.

These timers were temporary and are not shipped: all seven `fused_stats` slots are live counters
(0 submit, 1 calls, 2 tasks, 3 quantise successes, 4 quantise ns, 5 dequantise successes,
6 dequantise ns), so the sync measurement needs an eighth slot, and writing past seven would run
into the adjacent buffer on a device path. Re-measuring means adding that slot, not reusing one.

### 2. Elementwise four-way split: null (valid measurement)

| arm | blocks (ms/token) | median |
| --- | --- | --: |
| A 4-way (7 transforms fired) | 206.7, 210.2, 193.1, 194.0, 208.8, 193.5, 193.4, 194.6, 193.0, 220.7, 209.5, 193.9 | **194.31** |
| B serial (0 transforms) | 197.6, 222.9, 192.6, 193.4, 192.4, 203.6, 192.6, 211.2, 192.8, 218.8, 241.4, 192.5 | 195.49 |

Paired ratio median **1.0031**, 8 of 12 pairs favouring serial. The decode kernels on the
allowlist (`E_2_16_32_4` 90.5 us/call, `E_62080_4` 2634.6, `E_8192_4` 70.7, `E_3072_3` 49.6,
`E_8_32_4_4` 60.5, `E_16_64_4` 68.7) totalled 12.3 ms of CPU time, and splitting them four ways
moved nothing: at 20-90 us per call the four-way dispatch costs about what the work costs. The
allowlist additions are reverted. Log: `results/threadsplit/decode-split-verified.log`.

### 3. Attention group split: unresolved, not shipped

The arm was rejected earlier at 1.0940, but `attn_native._PROGRAM` is keyed `(tokens, capacity)`
and the flag was not in the key, and `to_program_cache` was not cleared -- that arm could not fire.
Re-measured with both fixed:

| arm | median | pairs |
| --- | --: | --- |
| A split 4 | **192.57** | 9 of 12 favouring split |
| B serial | 194.08 | |

Paired ratio median **0.9918**; sign test p = 0.073. Directionally positive at ~0.8% (1.5 ms) and
not resolvable at 12 pairs, so the shipped configuration stays serial. Log:
`results/threadsplit/attn-split-verified.log`.

**Scope of the defect:** `ab_decode` and `ab_fills` change the graph, so their keys differ and
their results stand. `ab_elem` and `ab_attn` changed only emitted source and are re-measured above.
`ab_scan` has the same shape as `ab_attn` and its conclusion should be treated as unmeasured until
re-run.


## The elementwise row is on the wall, and one rewrite moves it

### First: does the row reach the wall?

Multiplying `us/call` by `calls` has been wrong here before (the fills took 15 ms of kernel and
5 ms of wall), so the class was ablated -- every elementwise kernel body replaced with a no-op,
same ABI, `ab_noop.py`, A/B/B/A in one process, `to_program_cache` cleared per arm:

| arm | blocks (ms/token) | median |
| --- | --- | --: |
| A real | 225.7, 193.2, 192.7, 192.4, 192.6, 192.9, 321.7, 194.1, 211.1, 193.1, 193.4, 192.8 | **193.12** |
| B whole class no-op | 172.5, 179.1, 172.4, 172.1, 194.9, 180.7, 172.0, 195.4, 200.2, 172.2, 196.7, 178.0 | **178.57** |
| **wall saved** | | **+14.55 ms** |

So the elementwise class costs ~14.5 ms of wall -- it is the first row measured to transfer almost
fully, and the largest addressable item left. Log: `results/threadsplit/noop-elementwise.log`.

### The rewrite

The emitted code rebuilds floats from byte-addressed buffers:

```c
(__builtin_bit_cast(float, (unsigned int)(((unsigned int)(b0))+(((unsigned int)(b1))<<8u)+
                                        (((unsigned int)(b2))<<16u)+(((unsigned int)(b3))<<24u))))
```

where `b0..b3` are four `unsigned char` loads from four consecutive addresses. On a little-endian
host that is exactly one 4-byte load, so `thread_split.word_loads` rewrites it to a single
`__builtin_memcpy` load. 72 sites in seven decode kernels. The rewrite is refused unless the four
offsets are consecutive from a common base, and the result is discarded if it leaves any bracket
unbalanced.

**Parity:** `presample_logits_sha256_16 895365e27ed4255b`, unchanged.

**Per-kernel effect** (`decode_kernels_by_name.py`, 5 warm decode steps, CPU self time):

| kernel | byte-load ms | word-load ms | delta |
| --- | --: | --: | --: |
| `E_2_16_32_4` | 3.79 | **2.26** | **-40%** |
| `E_1024_4` | 2.35 | **1.72** | -27% |
| `E_640_4` | 1.81 | **1.38** | -24% |
| `r_16_32_4` | 1.57 | 1.44 | -8% |
| `E_8_32_4_4` | 1.25 | 1.25 | 0 |
| **all CPU kernels** | **197.06** | **194.53** | **-2.53** |

**Wall effect**, two independent ABBAs (temperature per block, arms verified by rewrite counter):

| run | A byte-load | B word-load | ratio | pairs |
| --- | --: | --: | --: | --- |
| 1 | 193.39 | **189.90** | 1.0155 | 10/12 |
| 2 | 192.74 | **189.74** | 1.0161 | 15/16 |
| combined | | | **1.016** | **25/28, sign test p = 1.4e-05** |

Logs: `results/threadsplit/wordload.log`, `wordload-replicate.log`.

### The trap this change walked into

Applied to **every** kernel, the same rewrite regressed the 128-token prefill by **37%**:

| arm | prefill blocks (ms) | median |
| --- | --- | --: |
| A byte-load | 1209.1, 1211.7, 1213.8, 1212.1, 1215.5, 1214.3, 1221.8, 1232.4 | **1214.04** |
| B word-load (all kernels) | 1657.0, 1662.3, 1678.0, 1683.9, 1657.0, 1656.2, 1666.0, 1664.2 | **1663.24** |
| ratio | | **0.7314, 0/8 pairs** |

The damage is in the 16-site prefill kernels (`E_2_16_32_32_4_4`, `r_32_16_4_32_4`). The rewrite is
therefore scoped by name to the seven decode kernels, exactly like `THREAD_SPLIT`; prefill is then
neutral (1211.95 vs 1210.50 on the 4 residual sites it still matches). A change that reads as a
pure win in one phase is a 37% regression in the other -- the phase has to be measured separately.

### Three figures for one class, and they are not interchangeable

| quantity | value | protocol |
| --- | --: | --: |
| elementwise class, previous `MANIFEST.md` | 12.61 ms | one call per segment, **with the greedy sampler** |
| elementwise class, regenerated `MANIFEST.md` | **20.04 ms** | one call per segment, shipped config |
| the seven rewritten kernels, itemiser | 10.77 -> 8.24 ms | 5 warm decode steps, `CPUProgram` self time |
| elementwise class, ablation | **14.55 ms** | wall difference, 12 pairs |

**Reconciled.** The 7.4 ms between the two manifests is one kernel: `E_62080_4`, the 248,320-element
vocabulary op, **8.50 ms** in the shipped configuration against 2.63 ms when the manifest was
generated while `GREEDY_SAMPLING` was on. The old manifest was a snapshot of a configuration that is
no longer shipped, so 12.61 was never the shipped cost.

It also explains why 20.04 ms of kernel time is consistent with 14.55 ms of wall: `E_62080_4` does
not transfer. Removing the greedy sampler -- which deletes the gumbel noise over the same 248,320
elements -- measured ratio 0.9991 over 16 pairs, i.e. no wall change from ~8 ms of CPU.
`20.04 - 8.50 = 11.5` transfers, against 14.55 measured for the class.

The regenerated table also shows the rewrite per kernel: `E_2_16_32_4` **1.93 ms** (3.47 before),
`E_1024_4` **1.24** (1.85), `E_640_4` **0.64** (1.81 in the itemiser's byte-load arm), `r_16_32_4`
1.13.

The itemiser's total CPU (197.06 ms/token) also counts the fused Linear calls including their
blocking submit, so it is not a class total and must not be used as one.

**Fixed:** `dump_decode_kernels.py` could not reproduce per-kernel rows because `MAX_CALLS_PER_SEG`
was a local inside `build()`. It is now module-level, the dump pins it to 1 and refuses to overwrite
a manifest whose row count it does not match; the regenerated table above is the shipped state.

### Position

| | ms/token |
| --- | --: |
| ours, word-load (two ABBAs) | **189.74 - 189.90** |
| reference, receipt | 157.34 |
| **gap** | **+32.4** |
| gap before this change | +35.4 |

**Caveat on scope:** the rewrite lives in `thread_split`, which `bench_llm.py` does not install, so
the gain applies to the instrument that defines the decode comparison (the `ab_*` harnesses and the
parity gate) and not yet to `bench_llm`. Landing it there is a one-line install, but it changes that
script's historic numbers and was not done here.


## The fills row is closed: a pad that is never read was being materialised every token

`E_131072_4` is a pure 2 MB zero fill (`E_131072_4.c`: `*(float4*)... = {0,0,0,0}` over 131072
float4). It was emitted **48 times per token** -- 96 MB of zeros per token, 9.43 ms of kernel time --
for the `where(0, recurrent_state)` at `model.py:802`, whose zero branch is statically **False** at
decode because `initial` is `pos == 0`. The branch can never be selected and was still materialised.

This is the local form of the reference's `ONPU_ZERO_PAD_VIEWS` (`rockchip_padding.py:13,37`): stop
materialising a pad that is never read. Their code reaches it with a view over a capacity-zeroed
buffer; ours skips the branch.

```python
_first = _first_step(pos)                      # False when the position is statically non-zero
state = (self.recurrent_state.float() if (SKIP_ZERO_PAD_DECODE and _first is False)
         else initial.where(0, self.recurrent_state.float()))
```

**Prefill keeps the original expression.** That is not caution for its own sake: the earlier attempt
at this line changed both branches at once and moved the parity hash to `c693475255158a69`. With the
change confined to decode, parity is **`895365e27ed4255b`, unchanged**.

### Measured

Two independent ABBAs, temperature per block, `to_program_cache` cleared per arm:

| run | A no-pad | B zero-pad | ratio | pairs |
| --- | --: | --: | --: | --- |
| 1 (12 pairs) | **181.26** | 189.82 | 0.9549 | 10/12 |
| 2 (16 pairs) | **181.45** | 191.00 | 0.9546 | 11/16 |
| combined | | | **0.955** | **21/28, sign test p = 6.3e-03** |

Logs: `results/threadsplit/nopad.log`, `nopad-replicate.log`.

Per-kernel, from the regenerated dump (`MAX_CALLS_PER_SEG = 1`, one warm token):

| | before | after |
| --- | --: | --: |
| fill (kernel) class | 9.43 ms | **0.68** |
| calls per token | 958 | **910** |
| wall, dump protocol | 196.18 | **187.58** |
| elementwise | 20.04 | 19.98 |
| `E_131072_4` | 48 calls, 8.76 ms | **absent** |

The 8.56 ms of wall removed against 8.76 ms of kernel removed is a transfer factor of ~1.0: these
kernels were pure writes, so nothing of them was hidden behind the device.

`E_131072_4.c` remains in `decode-kernels/` for the record but is **no longer emitted**; the dump
wrote 24 sources and 35 kernels this time. Deleting it needs a decision, so it stays and is flagged
here rather than left to look current.

### Position

| | ms/token |
| --- | --: |
| ours, word-load + no-pad | **181.26 - 181.45** (5.51 tok/s) |
| reference, receipt | 157.34 |
| **gap** | **+24.1** |
| gap before the fills fix | +32.4 |


## Greedy at temperature 0: 8.06 ms of CPU removed, and why the old "null" was wrong

The sampler at temperature 0 computed, per vocabulary element:

```python
(logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log())
```

Two `log`s and one `rand` over **248,320** elements. `E_62080_4` was that expression: **8.49 ms/token**.
At temperature 0 the Gumbel term is dead (`logits/1e-12` dominates a [-1,1] noise), and the reference
runs a bare `gpt2_argmax` for this benchmark -- its own profile shows `CPU:gpt2_argmax` 0.086 ms and
`E_62080_4` 0.375 ms.

Replacing it with `logits.argmax(-1, keepdim=True)` at temperature 0:

| | before | after |
| --- | --: | --: |
| `E_62080_4` | 8.49 ms | **0.43 ms** |
| total CPU, itemiser | 186.07 ms | **175.11 ms** |
| parity (pre-sampling logits) | `895365e2` | **`895365e2`** |

**Why the earlier ABBA said this was null.** `to_program_cache` is keyed such that the greedy graph's
elementwise kernel keeps the *same name*, `E_62080_4`. With the harness resetting only the JIT, the
second arm reused the first arm's compiled binary -- the noisy body -- and measured 0.9991. The
structural check above (same name, 8.49 -> 0.43 ms, in a fresh process) is what exposed it. This is
M83's defect surviving in a harness that *did* change the graph, because the key was not graph-deep.

**Wall measurement, with its problem stated.** The arm is bimodal: both arms stall for ~70 ms in
roughly 20% of blocks (NPU 1 GHz, DDR 2112, CPU6 2256000 -- the clocks are pinned, so this is not
DVFS). Medians are therefore corrupted and not quoted. The fast modes are tight and consistent
across two runs:

| run | A greedy fast mode | B gumbel fast mode | delta |
| --- | --: | --: | --: |
| 1 (12 pairs) | 171.0 (8/12) | 180.9 (11/12) | **-9.9 ms** |
| 2 (12 pairs) | 171.2 (7/12) | 181.4 (9/12) | **-10.2 ms** |

Two independent methods -- CPU self time and fast-mode wall -- agree at **-10.0 to -11.0 ms**, which
is the strongest agreement between a CPU delta and a wall delta yet measured here.

### The next kernel: our argmax is 78x slower than theirs

| | ours | theirs |
| --- | --: | --: |
| argmax over 248,320, isolated, 20 runs | **6.6 - 6.8 ms** (stable) | `gpt2_argmax` **0.086 ms** |

Our reduce is 148 MB/s; theirs is 11 GB/s. That is **6.6 ms still in the path** after the greedy
change, and it is a hand-written C kernel in their tree (`rockchip_greedy.py:10 _argmax`), not an
architecture difference.

## Next, in order

## Next, in order

1. Confirm the staging hypothesis: instrument the JIT's staging allocation to report buffer
   addresses and sizes for one decode step, and match them against the 48 memset destinations.
2. If confirmed, stop staging the read-write state (the reference's `ONPU_REPLAY_BORROW_OUTPUTS`
   analogue) — expected −8.6 ms and, if it also removes the paired copy, more.
3. Then the elementwise per-call gap (67 vs 15 µs) — compare the same kernel by name across the two
   profiles (`E_2_16_32_32_4_4` ours 124 µs vs `E_2_16_32_4` theirs 44 µs) and read both sources.
   This is the remaining ~20 ms and the only path to 6.48 tok/s.
