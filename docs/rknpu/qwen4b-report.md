# Qwen3.5 4B on the RK3588 NPU — where we are, what the reference does, and the method

Written 2026-11-18 after reading `~/oNPU` and `~/tinygrad` properly. It supersedes
every earlier summary in this directory. Numbers carry their conditions, per the
rule in `AGENTS.md`.

## 1. The targets, corrected

The reference's own latest 4B row is **not** the 1,263 ms / 6.48 tok/s from the
Nov 3 production API cohort. It is the source-pinned 1+3 retime
(`~/oNPU/plan.md:35`):

| metric | oNPU ref | RKLLM | source |
| --- | --: | --: | --- |
| decode | **6.3555 tok/s** | 6.22 tok/s | `plan.md:35` |
| TTFT, median of 3 | 9,335.78 ms | 2,068.88 ms | `plan.md:221` |
| TTFT, **last trial** | **1,210.6 ms** | — | `plan.md:221` |
| TTFT spread | 15.804x | — | `plan.md:35` |

**Only the last trial is a prefill number.** `plan.md:206-230`:

> *"The 1+3 retimes' TTFT spreads are not random thermal noise. Across nine retimed
> models and three independent H5 cold cohorts, the last measured trial is
> 6.7-15.7x faster than the median of the three measured trials ... The median of 3
> is the first-trial artifact, not a stable prefill estimate."*

The reference **confirmed the 3+3 rule** on MiniCPM4 on 2026-11-17: measured TTFT
103.39 / 104.01 / 115.94 ms, median 104.01 ms, decode median 57.10 tok/s,
`timing_validity=PASS` (`plan.md:230`).

## 2. Where we are

| metric | ours | ref | gap |
| --- | --: | --: | --: |
| TTFT, stabilized | 31,749 ms | 1,210.6 ms | **26x** |
| decode | 1.5088 tok/s | 6.3555 tok/s | **4.2x** |
| decode device | 151 ms | 138.66 ms | **1.09x** |
| decode host | 512 ms | 15.56 ms | **33x** |

Source: `docs/rknpu/results/qwen35-4b-fixed/report.json`, 3/3 trials, clocks pinned.

**The device is at parity.** Our NPU does the matmul in 151 ms against the
reference's 138.66 ms, and 151 ms is 6.62 tok/s — faster than the reference's whole
6.3555 step. Both gaps are host.

## 3. Speed of light, and what it says

Per the `metal_kernels` method, compute the ceiling from measured bandwidth before
touching anything:

| phase | bytes | at 33.79 GB/s | we measure | % of SOL |
| --- | --: | --: | --: | --: |
| decode, 1 token | 4.2 GB | **124 ms** = 8.04 tok/s | 662.8 ms | **19%** |
| prefill, 128 tokens, 2 row tiles | 8.4 GB | **249 ms** | 31,749 ms | **0.8%** |

Decode at 19% of SOL and prefill at 0.8% says, before any kernel work, that the
matmul is not the problem. It cannot be: the device is already at 94.5% of the DDR
ceiling when measured in isolation (`bench_submit.py`).

## 4. Where our time actually goes

Measured by **ablation** — stubbing a component and diffing the total — not by
multiplying a per-call cost by a call count. That multiplication is what produced
two wrong answers earlier in this project.

### Prefill, 31.9 s

| component | time | share |
| --- | --: | --: |
| **24 SSM branches (the scan)** | **27.2 s** | **85.1%** |
| Linear path | 2.5 s | 7.9% |
| 8 attention blocks | 2.0 s | 6.1% |
| residual | 0.3 s | — |

### Decode, 662.8 ms

| component | time |
| --- | --: |
| device (NPU matmul) | 151 ms |
| host | **512 ms** |

The host is 323 submits at 1.55 ms, and the cost is **fixed per call, not per
element** — identical at m=1 and m=64.

## 5. What the reference does that we do not

Its kernels are **the same as ours**. `rockchip_int8.py:280-295` is quantize →
matmul → dequantize, three `custom_kernel` calls, structurally identical to our
`packed_matmul_int8`. Copying them would change nothing.

The difference is five levers, all in `~/oNPU/onpu/rockchip_options.py:379-401`
(the `large = True` branch):

| lever | what it removes | our state |
| --- | --- | --- |
| `ONPU_REPLAY_NATIVE_FROM_LARGE` + `BORROW_OUTPUTS` / `BORROW_BATCH` / `SCALAR_COPY` / `DEDUP_GUARDS` | the whole forward becomes **one C function** that issues the ioctls itself (`rockchip_replay.py:606-626`) | absent; we dispatch 13,910 kernels per prefill |
| `scalar_decode_loop` | no tinygrad graph per decode token | graph per token |
| `cpu_recurrent_scan` | the scan becomes a C kernel, 24 calls | 9,216 tinygrad launches |
| `ONPU_FFN_PACKED_UPLOAD` | the FFN down projection reads a packed payload from NPU memory, no round trip | round trip |
| `share_projection_weights` + `RockchipProjections` | q/k/v and gate/up fused into one submit → 113 ioctls | 323 submits |

**Why this matters mechanically.** tinygrad's JIT captures a linear and replays it
with a Python loop over kernels (`engine/realize.py:281`). That is fine on a GPU,
where launches are async and the CPU runs ahead. Our NPU submits are **blocking**
(`RKNPU_JOB_BLOCK`), so every dispatch's cost is exposed. On a blocking device the
only way to hide dispatch is to eliminate it — which is exactly what the
megakernel does.

## 6. The method

`https://github.com/allbilly/metal_kernels` is the working method for this class of
problem, and this project broke every rule in it for a day:

1. **Profile the reference engines first**, before any change.
2. **Compute SOL from measured bandwidth and quote the percentage.**
3. **Paired A/B in ABBA rounds** — *"the ratio cancels thermal/DVFS drift"*.
4. **Report wins as pair counts** (9/10), not medians.
5. **Ablate to find the cause, then fix the route, not the kernel.** Its own
   rejection: *"0.581 TFLOP/s vs 1.640 for mx.matmul (0.35x), so the fix was the
   route, not a new GEMM."*
6. **Retract invalid measurements explicitly.**

The oNPU archive adds the acceptance gates: paired ratios against RKLLM, `ids`
bit-exact on every trial, `timing_validity=PASS`, clocks pinned at 1 GHz NPU /
2.256 GHz CPU6 / 2.112 GHz DDR before and after every trial, and a temperature
bound before the run starts.

## 7. The plan

| # | action | expected | evidence |
| --: | --- | --- | --- |
| 1 | **C scan kernel** (`cpu_recurrent_scan`) | up to 24.5 s of prefill | ablation: 85% of prefill; reference has the kernel |
| 2 | **Fused per-Linear C call** (done, in flight) | decode host 512 → ~100 ms | 9/9 bit-exact at 4B geometry, 2.4-3.8x per call, Qwen2 3.64 → 7.95 tok/s |
| 3 | **Native replay** (whole forward as one C function) | the rest of the decode host | `rockchip_replay.py` |
| 4 | **Projection fusion** | 323 → 113 submits | `RockchipProjections` |
| 5 | **Async sentinel poll** | overlap device with host | `RKNPU_JOB_NONBLOCK` + `0x7fffffff` sentinel; pin to A76 |

## 8. What was wasted, and why

| work | outcome |
| --- | --- |
| 139-dim kernel hunt | real bug, ~14 s — but found by measurement, not by reading the reference |
| quantize memoisation | **zero** — the JIT never paid it |
| k=9216 test | wedged the NPU, caused a reboot |
| fused Linear kernel | **genuinely good** — but the reference already had it |
| four wrong claims about where the time was | hours of re-measurement |

The root cause is one thinking error, repeated: **optimising locally when a
known-good endpoint was on disk.** With a working reference at 6.3555 tok/s, the
correct method is *diff against it*, not *search for wins*. Every hour spent
discovering that the scan was 85% was an hour the diff would have given directly.

The two specific measurement errors, now rules in `AGENTS.md`:

- **"the scan is 0.88 s"** — an isolated `_attention` A/B applied to the model. The
  in-model figure is 24.5 s.
- **"~15 ms per Linear launch"** — the eager path applied to the JIT-captured path.
  The captured path pays 0.79 ms.
