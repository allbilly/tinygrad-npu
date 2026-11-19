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
