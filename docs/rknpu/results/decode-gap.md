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

## Next, in order

1. Confirm the staging hypothesis: instrument the JIT's staging allocation to report buffer
   addresses and sizes for one decode step, and match them against the 48 memset destinations.
2. If confirmed, stop staging the read-write state (the reference's `ONPU_REPLAY_BORROW_OUTPUTS`
   analogue) — expected −8.6 ms and, if it also removes the paired copy, more.
3. Then the elementwise per-call gap (67 vs 15 µs) — compare the same kernel by name across the two
   profiles (`E_2_16_32_32_4_4` ours 124 µs vs `E_2_16_32_4` theirs 44 µs) and read both sources.
   This is the remaining ~20 ms and the only path to 6.48 tok/s.
