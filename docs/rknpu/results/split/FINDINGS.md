
> **CORRECTION (applied):** the reference's measured trials
> `[9335.78, 19132.28, 1210.63]` were taken after **one** warm-up, and that 9.3 / 19.1 / 1.2 s shape is
> the same JIT lifecycle documented in this file: ignore -> capture -> `NativeReplay` construction ->
> native. With the **3 warm-ups** used here, all three measured trials would be at the native figure,
> so the comparable reference median is **~1,210 ms, not 9,336 ms**. Every "3.44x faster than the
> reference median" statement below is **withdrawn**; it compared their under-warmed median against a
> properly-warmed one. The honest comparison is **2,711 vs ~1,210 ms = 2.24x slower**, the same as the
> best-trial comparison, and 1.31x slower than RKLLM.
# Qwen3.5 4B — the trial pattern, resolved

Run: `docs/rknpu/results/split/` (1 warm-up + 5 measured, exact reference argv and
environment from `docs/rknpu/results/ref-warmed-4b/receipt.json`).

## The pattern is deterministic and index-dependent

| execution | 1 (warm-up) | 2 | 3 | 4 | 5 | 6 |
| --- | --: | --: | --: | --: | --: | --: |
| prefill ms | 16,929.0 | 9,275.5 | 19,636.6 | **1,203.6** | **1,299.9** | **1,257.6** |
| decode tok/s | 2.61 | 6.19 | 6.20 | 6.26 | 6.21 | 6.32 |
| prefill submits | 475 | 475 | 475 | 475 | 475 | 475 |

Reproduced across three independent runs. The *pattern* is deterministic -- slow, slow,
slow, fast -- but the precise values are not equally tight:

| execution | ref-warmed-4b | split | phase | spread |
| --- | --: | --: | --: | --: |
| 1 | 16,863 | 16,929 | 16,597 | 2.0% |
| 2 | 9,267 | 9,276 | 9,441 | 1.9% |
| 3 | 20,037 | 19,637 | 19,146 | 4.7% |
| 4 | 1,225 | 1,204 | **1,364** | **13.3%** |

The slow executions agree to within 5%. The fast one varies by **13%** run to run, so a
3+3 result should be quoted as ~1,204-1,364 ms, median ~1,258 ms -- not as a single tight
figure.

## Two phases settle at different rates

| phase | settles after | evidence |
| --- | --- | --- |
| decode | **1 execution** | 2.61 -> 6.19 tok/s, then flat (6.19 / 6.20 / 6.26 / 6.21 / 6.32) |
| prefill | **3 executions** | 16.9 / 9.3 / 19.6 -> 1.2 / 1.3 / 1.3 |

Prefill is not monotone (16.9 -> 9.3 -> 19.6), so it is not a cache filling with use.
Two phases with two settling rates means at least two separate warm-up processes.

## The fix: 3 warm-ups, not 1

A 3+3 run measures executions 4, 5 and 6. Those are the three fast ones above, so the
protocol change alone makes every measured trial fast, with no code change.

Literal 3+3 run (`docs/rknpu/results/three3/`, `--warmups 3 --runs 3`):

| execution | 1 warm | 2 warm | 3 warm | 4 measured | 5 measured | 6 measured |
| --- | --: | --: | --: | --: | --: | --: |
| prefill ms | 16,903.3 | 9,505.9 | 19,595.4 | **1,393.5** | **1,218.8** | **1,328.7** |
| decode tok/s | 2.59 | 6.26 | 6.29 | 6.26 | 6.27 | 6.29 |

`complete: true`, `median_first_token_ms = 1,328.7`, `median_decode_tps = 6.271`, 475
prefill submits and identical generated ids in all six executions.

| protocol | measured trials | median TTFT |
| --- | --- | --: |
| 1+3 (`split`) | 9,275.5 / 19,636.6 / 1,203.6 | **9,275.5 ms** |
| 3+3 (`three3`, literal) | 1,393.5 / 1,218.8 / 1,328.7 | **1,328.7 ms** |

The 1+3 median is a warm-up artifact: 7.0x the 3+3 median for identical work
(9,275.5 / 1,328.7 = 6.98).

**Precision caveat.** Only one *literal* 3+3 run exists. A 1+5 run (`split`) whose
executions 4-6 are the same three executions gives 1,203.6 / 1,299.9 / 1,257.6, median
1,257.6. So the settled median is **1,258-1,329 ms across two runs** and should be quoted
as a range, not a point. Do not describe `split` as a 3+3 run -- its `warmups` argument is
1.

Independently confirmed on three 3+3 runs already in the archive, all with three fast
measured trials:

| run | warm-up 1 | warm-up 2 | warm-up 3 | measured |
| --- | --: | --: | --: | --- |
| MiniCPM4 0.5B | 3,938 | 2,757 | 2,808 | 209 / 218 / 327 |
| Qwen2 0.5B | 6,395 | 3,142 | 3,737 | 259 / 250 / 248 |
| Qwen2 0.5B (v2) | 5,662 | 3,789 | 4,032 | 339 / 303 / 330 |

## What this is not

It is a **measurement** fix, not a speed fix. Executions 1-3 still cost 16.9 + 9.3 +
19.6 = **45.9 s** of wasted work on every process start. The mechanism that makes them
slow is still unattributed; only its shape is known.

## Correct 4B numbers under 3+3

| metric | ours (3+3) | ref repro 1+3 median | ref repro last | archive published | RKLLM |
| --- | --: | --: | --: | --: | --: |
| prefill, 128 tok | **1,328.7 ms** | 9,266.5 ms | 1,224.5 ms | 1,263.09 ms | 2,068.88 ms |
| prefill tok/s | **96.3** | 13.8 | 104.5 | 101.3 | 61.87 |
| decode tok/s | **6.27** | 6.19 | 6.15 | 6.48 | 6.22 |

"ref repro" is `docs/rknpu/results/ref-warmed-4b/benchmark.json`, our reproduction of the
reference configuration. "archive published" is `~/oNPU/benchmark.md:1179`, the figure the
archive actually quotes -- 1,263.09 ms, close to its last-trial 1,210.6 ms
(`plan.md:221`), *not* its 1+3 median of 9,335.8 ms, which the archive identified as the
artifact and did not publish.

Prefill is **1.56x RKLLM** (2,068.88 / 1,328.7) and **5.2% slower** than the archive's
published figure. Decode is **0.8% above** RKLLM. Across the two runs the settled medians
are 1,257.6 and 1,328.7 ms, so prefill sits in a **1.4-1.6x RKLLM** band rather than at a
single point.

Clocks: NPU 1 GHz, DDR 2112 MHz, CPU6 2256000, sampled continuously at 5 Hz for the whole
run (`clocks.jsonl`), not only at trial boundaries.

## The mechanism, measured

Trace: `docs/rknpu/results/jitcast/trace.log` (1+3). `extra/rockchip/trace_jit_calls.py`
patches `_TinyJit` at import time, so the archive tree stays unmodified.

All four prefill calls go through the **same** `_TinyJit` (`id=538943771744`), and only the
declared pair of `greedy_forward` JITs exists -- so there is no variant fork.

| trial | prefill | `cnt` | `captured_id` | `native_runs` | what ran |
| --: | --: | --- | --- | --: | --- |
| 1 | 16,707.5 ms | 0->1 | None | - | jit-ignore, no capture |
| 2 | 9,315.5 ms | 1->2 | 547760233584 | - | capture, produces a **base** `CapturedJit` |
| 3 | 18,809.3 ms | 2->3 | **547860835664** | 0 | captured object replaced, then warming |
| 4 | 1,208.2 ms | 3->4 | 547860835664 | 1 | native replay |

The replacement at trial 3 comes from `rockchip.py:873`, which runs **after** the JIT call
returns:

```python
if not isinstance(selected.captured, RockchipCapturedJit):
    selected.captured = RockchipCapturedJit(selected.captured, cleaner, program_transform=transform)
```

So the capture pass produces a base `CapturedJit`, and only afterwards is it wrapped in a
`RockchipCapturedJit` whose `warmed=False`. That wrapper's first call is a warming pass --
a **third** slow execution that the base two-pass protocol does not have. It is also the
most expensive of the three, because it runs the interpreted path *and* builds the native
replay.

## Fix: tested, rejected

Measured with `ONPU_FIX_WARMED=1`, which sets `warmed=True` when the wrapper is constructed
from a capture that already executed (`extra/rockchip/trace_jit_calls.py`; the archive tree
is unmodified). Prediction was trial 3 dropping to ~1.2-2.0 s.

| trial | jitcast (no patch) | warmfix (patched) | delta |
| --: | --: | --: | --: |
| 1 | 16,707.5 ms | 16,867.5 ms | +1.0% |
| 2 | 9,315.5 ms | 9,310.7 ms | -0.1% |
| 3 | 18,809.3 ms | **18,340.4 ms** | **-2.5%** |
| 4 | 1,208.2 ms | 1,386.2 ms | +14.7% |
| `native_runs` | 1 | **2** | |

Generated ids identical across all four trials.

The patch worked mechanically -- `native_runs` became 2, so trial 3 did execute the native
replay -- and the trial still took 18.3 s. **The cost is not the interpreted warming pass;
it is `NativeReplay(...)` construction for the transformed graph.** Skipping the warming
pass moved the native execution into trial 3 without removing the build.

Moving the pass earlier (building at capture time) would only relocate the same build cost
into trial 2.

## Conclusion

Start-up cost decomposes into two things, neither fixable in model code:

| pass | measured cost | fixable in model code? |
| --- | --: | --- |
| 1. jit-ignore, no capture | ~16.8 s | no -- tinygrad lifecycle |
| 2. capture the base graph | ~9.3 s | no -- tinygrad lifecycle |
| 3. native replay build for the transformed graph | ~18.3 s | no -- the build must happen; `warmed` only moves it |

Total ~44 s of one-time cost per process start. 3+3 hides it from the measurement; nothing
measured here removes it. Upstream: `docs/mnist.md` for the lifecycle, issue #15219 for
RK3588 cold start, closed with no fix.

## `cnt=1`: first measured improvement, and what it does not fix

Patch: `ONPU_FIX_SKIP_IGNORE=1` starts `greedy_prefill_jit` at `cnt=1`, so the first call
captures instead of running the jit-ignore pass (`extra/rockchip/trace_jit_calls.py`;
archive tree unmodified).

| trial | baseline | `cnt=1` | delta |
| --: | --: | --: | --: |
| 1 | 16,707.5 ms (jit-ignore) | 18,040.8 ms (capture) | +8.0% |
| 2 | 9,315.5 ms (capture) | 19,120.8 ms (replay build) | +105.3% |
| 3 | 18,809.3 ms (replay build) | **1,229.2 ms (native)** | **-93.5%** |
| 4 | 1,208.2 ms (native) | 1,210.5 ms (native) | +0.2% |
| **sum** | **46.0 s** | **39.6 s** | **-6.4 s** |

Generated ids identical in all four trials. Trace:

```
cnt 1->2  capture        native=None
cnt 2->3  replay build   native=0
cnt 3->4  native         native=1
cnt 4->5  native         native=2
```

**What it achieves:** three slow trials become two, so a 2-warmup protocol suffices instead
of 3, and 6.4 s of the 46.0 s is removed rather than relocated.

**What it does not achieve:** the first two trials stay slow (37.1 s of work), because the
warmup pass was performing real kernel compilation -- skipping it moves that cost into the
capture pass (9.3 -> 18.0 s). "All three runs fast" is still not reached.

**Caveat:** this is a runtime patch, not a change in a maintained tree, and it alters the
compilation context of the first pass (`IGNORE_JIT_FIRST_BEAM` exists precisely to control
beam search there). Steady-state native times are unchanged (1.21-1.23 s), so no quality
loss is visible in the timing, but token parity on one prompt is not a broad correctness
proof.

## The ported tree, measured: 128-token prefill, steady state

`~/pr/tinygrad`, `bench_llm.py --npu --prompt 128 --gen 8`, `DEV=CPU`. Sources:
`docs/rknpu/results/port-baseline/` (2 executions) and
`docs/rknpu/results/port-128-w1-r3/` (4 executions).

| execution | prefill | submits | time in NPU |
| --- | --: | --: | --: |
| 1 (warm-up) | 137.0 s | 635 | 0.4 s |
| 2 | 50.07 s | 635 | 426.3 ms |
| 3 | **22.52 s** | 635 | 426.2 ms |
| 4 | **22.50 s** | 635 | 426.2 ms |

Steady state **22.50 s** (median 22,516 ms over the three measured trials), decode
2.422 tok/s. So host = 22,516 - 426 = **22,090 ms, 98.1% of the prefill**, at
**34.8 ms per submit** against **0.67 ms** of NPU -- a 52:1 host-to-NPU ratio.

Against the archive on the same 128-token prompt:

| metric | ported tree | archive | ratio |
| --- | --: | --: | --: |
| prefill | **22.50 s** | 1.208--1.386 s | **18.6x** |
| submits | 635 | 475 | 1.34x |
| host per submit | **34.8 ms** | ~2.54 ms | **13.7x** |
| decode | 2.42 tok/s | 6.15--6.44 tok/s | 2.6x |

`1.34 x 13.7 = 18.4`, matching the measured 18.6x total. **The entire gap is host dispatch
per submit**; the NPU performs the same work per submit in both trees (0.67 ms here).

Path confirmed from the reports: `npu_linears: 249` means `replace_linears` swapped every
Linear for `NPULinear`, whose `__call__` uses `packed_matmul_int8_fused` -- "one C call:
quantize + direct ioctl + dequantize". So this is the port's optimised path. An earlier
caveat in this file claiming otherwise was wrong.

The port shows the same warm-up shape as the archive: 137.0 -> 50.1 -> 22.5 -> 22.5, i.e. it
settles by the third execution, consistent with `jit.py`'s two-pass lifecycle.

Protocol caveat: `--gen 8` shortens decode, so `kind` is `quick-check`; the prefill figure is
at 128 tokens with 1 warm-up + 3 measured, which is a valid prefill measurement, but the
decode figure is not the full 64-token protocol.

## NativeReplay port attempt: what worked, what it costs

The replay layer was copied into `~/pr/tinygrad/tinygrad/llm/kernels/` (8 files, 1,677 lines
including the C helpers) and made to import and run with five drift fixes:

| drift | archive | this tree | fix |
| --- | --- | --- | --- |
| `TinyELF` | class in `tinygrad/device.py` | moved to `tinygrad/renderer` | import changed |
| `_copy_input` | module helper in `jit.py` | absent (aliasing handled inline) | defined locally from the same primitives |
| `Rockchip{Grouped,Batched,Routed,Bucketed}Matmul` | present | absent | optional import; raise loudly if reached (audit says unused) |
| `CapturedJit.__init__` | 4 fields | 5 fields, adds `var_vals` | forward `captured.var_vals` |
| `CapturedJit.linear` | cached property | absent (links lazily in `__call__`) | added on `RockchipCapturedJit` |
| `link_linear(linear, jit=True)` | takes `jit` | `link_linear(linear, input_uops=None)` | call without `jit` |
| `UOp.tag` | field | **absent entirely** | `scratch()` returns False, disabling aliasing/upload elision only |

Two findings that matter more than the drift:

**1. The port's fused INT8 call is invisible to the replay.** `NPULinear` uses
`packed_matmul_int8_fused`, which issues the NPU submit inside a CPU-side C call, so the
captured graph has no ROCKCHIP programs and `NativeReplay` refuses with
`UnsupportedReplay('no NPU work')` (`port-replay-try5.log`). Switching `NPULinear` to the
graph-visible `packed_matmul_int8` path (`GRAPH_VISIBLE_INT8 = True` in
`extra/rockchip/npu_linear.py`) makes the replay engage -- but that path alone is slower
(warm-ups 132.7/50.0 -> 150.6/67.2 s). The local fused optimisation and the global replay
are mutually exclusive.

**2. The dependency closure is 41 files / 9,462 lines**, measured by transitive import walk
from `rockchip_replay.py`:

| file | lines |
| --- | --: |
| `rockchip.py` | 1,071 |
| `rockchip_replay.py` | 1,006 |
| `rockchip_gelu_ffn.py` | 858 |
| `rockchip_int8.py` | 768 |
| `rockchip_cpu.py` | 767 |
| 36 further files | 4,992 |

So a wholesale port is a multi-week job, not a patch. The replay also assumes a model shape
this tree does not have: the archive has `greedy_forward` / `greedy_prefill_jit` / 5+ JITs,
while the port has `forward()` and two (`prefill_jit`, `rollout_jit`), so the archive's
1,071-line integration cannot be copied -- it has to be rewritten against this model.

**Verdict:** the replay is structurally portable (it imports, runs, and engages after seven
drift fixes) but not affordably so -- and the section below shows it would not have helped
anyway, because the cost it targets is 8% of the loop.

## Where the 22.4 s actually goes: CPU kernel execution, not dispatch

`extra/rockchip/profile_graph_dispatch.py` records one row per graph and splits `rt(...)`
kernel execution from the surrounding bookkeeping. `--warmups 3 --runs 1`, graph-visible
INT8. The prefill graph is **13,966 programs**; the 2,703-program graphs are decode steps.

| measure | value | share |
| --- | --: | --: |
| loop total, steady state | 31.35 s | 100% |
| **in-kernel execution** | **28.72 s** | **92%** |
| bookkeeping / dispatch | 2.63 s | 8% |
| CPU device | 30.70 s | 98% |
| ROCKCHIP + ROCKCHIP:1 | 0.62 s | 2% |

So the prefill is **92% CPU kernel execution**. Per-program dispatch is 2.63 s, and a
whole-graph replay could remove at most that -- not the 28.7 s.

**Correction.** An earlier version of this file reported 41.52 s of "dispatch", 43,533 CPU
programs at 0.86 ms each, and concluded dispatch was the bottleneck. That was wrong twice:
it aggregated two *compiling* executions instead of taking one steady-state graph, and it
timed `pm_exec.rewrite`, which calls `exec_kernel` and therefore includes kernel execution.
The real split is the table above.

**Consequence.** The gap to the archive is **CPU kernel quality**, not dispatch and not NPU.
The archive's 1.2 s comes from its optimised CPU kernel library -- `rockchip_cpu.py` (767
lines, fast softmax plus the preparation/decode pools), `rockchip_swiglu_ksplit`,
`rockchip_norm_quant`, `rockchip_sigmoid` and the rest of the 41-file closure. That is why
the closure is 9,462 lines: the fast CPU kernels are most of it, not the replay.

**Reverted.** `GRAPH_VISIBLE_INT8` is now `False`. That path existed only so the replay could
see the NPU work, and it measured **31.3 s against the fused path's 22.4 s**.

**What this rules out.** A bespoke whole-graph replay. Its ceiling is the 8% bookkeeping
term, and building it would have been the third wrong target in a row on this objective --
after thermal and after the variant fork.

## The two kernels that are 88% of the port's prefill

Same profiler, bucketing in-kernel time by kernel name. `ProgramInfo` drops `KernelInfo.name`,
so the name comes from the `SOURCE` child of the PROGRAM that each CALL wraps. One-time
structure dump as control:

```
{"op": "Ops.CALL", "arg": "CallInfo",
 "src_ops": ["Ops.PROGRAM","Ops.BITCAST","Ops.PARAM","Ops.BUFFER"],
 "src0_src_ops": ["Ops.SINK","Ops.LINEAR","Ops.SOURCE","Ops.BINARY"]}
```

Steady state: 10,407 programs, loop 22.58 s, kernel 20.12 s, bookkeeping 2.46 s.

| time | share | kernel | reading |
| --: | --: | --- | --- |
| **11.36 s** | **56.5%** | `r_2_16_32_4_32_4` | `r_` reduce |
| **6.33 s** | **31.5%** | `E_2_16_32_32_4_4` | `E_` elementwise |
| **2.49 s** | **12.4%** | `r_4_128_[start_pos]...` | reduce over position -- attention |
| 0.89 s | 4.4% | `ioctl` | **the NPU** |
| 0.38 s | 1.9% | `r_64_16_64_...` | reduce |

So **two kernels are 87.9% of the prefill**, a third is 12.4%, and the NPU is 4.4%. Nothing
else exceeds 1%.

This is the target the earlier attempts could not see. It is not dispatch (2.46 s, 11%), not
the NPU (0.89 s, 4%), and not the matmul. It is the arithmetic inside two specific CPU
kernels, which is exactly what the archive's `rockchip_cpu.py` / `rockchip_swiglu_ksplit` /
`rockchip_norm_quant` provide faster versions of.

Caveat: `r_2_16_32_4_32_4` and `E_2_16_32_32_4_4` are identified by codegen name only. Their
source text is available in the `SOURCE` child and should be read before choosing which of
the archive's kernels to port -- the names alone say "reduce" and "elementwise", not which
operation they implement.

## The two dominant kernels, identified from their source

The kernel sources were dumped and read rather than guessed from codegen names.

**`r_2_16_32_4_32_4` -- 11.57 s, 56% -- attention**

```c
float alu2 = (0.0883883461356163f / val5);
```

`0.0883883461356163 = 1/sqrt(128)`, the attention QK scale for head_dim 128.

**`E_2_16_32_32_4_4` -- 6.39 s, 31% -- SwiGLU sigmoid**

```c
float alu0 = (val7 * -1.4426950216293335f);
```

`-1.442695 = -log2(e)`, so the kernel computes `exp2(-x*log2e) = e^-x`: a sigmoid gate.

| kernel | share | archive counterpart | location |
| --- | --: | --- | --- |
| attention reduce | 56% | `_fast_softmax_source` | `rockchip_attention.py` (181 lines) |
| SwiGLU sigmoid | 31% | `ONPU_EXP2_FAST_PATH`, `_vector_exp2_batch4_source` | `rockchip_cpu.py:21,631` |
| `ioctl` | 4% | the NPU itself | -- |

So **87% of the prefill is two kernels** whose fast versions already exist in the archive, and
the archive's run environment enables them (`ONPU_EXP2_FAST_PATH=1`).

This replaces the earlier "port the 9,462-line closure" estimate with a bounded target:
`rockchip_attention.py` (181 lines) plus the exp2 path from `rockchip_cpu.py`, for 87% of the
prefill. The remaining 9,000 lines are other models' features.

**Not yet done:** neither has been ported or measured. The prediction -- that these two
kernels fall from 17.96 s combined to roughly the archive's share -- has not been tested.

## Attempt: materialising attention scores -- refuted

The GDN scan's 214x pathology was cured by `.contiguous()` per step, so the same trick was
tried on attention: replace `scaled_dot_product_attention` with
`(q @ k.T * scale + mask).softmax(-1).contiguous() @ v`, materialising the scores so the
masked reduce could not fuse into one multi-range kernel.

| kernel | baseline | materialised | delta |
| --- | --: | --: | --: |
| `r_2_16_32_4_32_4` (attention) | 11.57 s | 11.54 s | -0.3% |
| `E_2_16_32_32_4_4` (sigmoid) | 6.39 s | 6.42 s | +0.5% |
| kernel total | 20.12 s | 20.34 s | +1.1% |

Identical kernel names and times, so the attention reduce is **not** a fusion pathology and
this route is closed. Reverted -- a no-op does not belong in the diff.

Consequence: the 11.57 s attention kernel needs the archive's actual custom CPU kernel
(`RockchipAttentionProgram` + `_fast_softmax_source`), not a scheduling hint. Same for the
6.42 s sigmoid, which needs the vectorised `exp2` path. Those are real ports of kernel code.

## Token parity is now checkable

`bench_llm.py` recorded only `generated_tokens` (a count), so no change to this tree could be
verified for correctness. `generated_ids` is now recorded per trial and surfaced in
`report.json`. It has not yet been exercised -- no change since then warranted a run.

## Attempt: vectorised sigmoid -- kernel works, target misidentified

A 4-wide polynomial-exp2 sigmoid was written following the port's CPU ABI
(`void name(void *args)`, geometry baked in) using the archive's `_EXP2_POLYNOMIAL`. It
self-tests at **max_abs_err 1.192e-07, rel 9.0e-07** -- float32 epsilon. Kept as
`extra/rockchip/fast_sigmoid_probe.py`.

Wired into the SwiGLU line (`gate * sigmoid(gate*alpha) * (up+up_bias)`) and measured:

| | before | after |
| --- | --: | --: |
| programs in prefill graph | 10,407 | 10,439 (+32, one per FFN layer) |
| `E_2_16_32_32_4_4` | 6.42 s | **6.25 s** |
| kernel total | 20.12 s | 20.00 s |
| prefill | 22.39 s | 22.46 s |

The 32 new kernels prove the wiring fired, and the dominant sigmoid kernel is unchanged. **So
`E_2_16_32_32_4_4` is not the SwiGLU sigmoid.** It was identified as one from the
`-1.4426950 = -log2(e)` constant, but every sigmoid in the model has that constant. Reverted.

The model has at least four sigmoids -- SwiGLU (`gate.sigmoid()`), the attention output gate
(`attn_output_gate`), the GDN decay (`pre = ...sigmoid()`), and the hyper-connection gates --
and the codegen name carries no call site.

**Next step for this kernel:** identify the call site, not the arithmetic. Either
instrument the model so each sigmoid call emits a distinguishable op, or bisect by disabling
one call site at a time and watching which kernel disappears.

## Token parity harness: exercised for the first time

`report.json` now records `generated_ids` per trial. First recorded baseline for this tree
(`--prompt 128 --gen 8 --warmups 3 --runs 1`):

```
[7378, 99491, 6035, 104994, 173019, 210052, 112366, 175115]
```

Any future change to this tree can now be checked against that list, which was not possible
before this turn.

## The GDN scan never engaged: a swallowed ImportError

This is the largest finding in the port, and it is a one-line bug.

`tinygrad/llm/model.py` selected the scan kernel with:

```python
try: from extra.rockchip.scan import gated_delta_scan, scan_supported
except ImportError: gated_delta_scan = None
```

`bench_llm.py` sets `sys.path[0] = extra/rockchip`, so `import extra.rockchip.scan` fails with
`No module named 'extra'` (the cwd is not on `sys.path` for a script run). The bare `except`
swallowed it and set the kernel to None, so **the fallback Python loop ran for every prefill
ever measured in this tree**. The kernel had only ever been verified standalone
(`check_scan_kernel.py`, rel 4e-7); it was never exercised through the model.

Call counts proved it before any patch:

| kernel | calls | identity |
| --- | --: | --- |
| `r_2_16_32_4_32_4` | **6144** | 128 tokens x 24 GDN blocks x **2** per-step reduces |
| `E_2_16_32_32_4_4` | **3072** | 128 x 24 x 1 per-step elementwise |
| `r_4_128_[start_pos]` | 8 | attention, one per block |

## With the import fixed: 2.8x faster, and wrong

Fixing the import to also try `from scan import ...`:

| | loop (stock) | scan engaged |
| --- | --: | --: |
| prefill graph programs | 10,407 | **1,311** |
| `r_2_16_32_4_32_4` | 11.35 s / 6144 | gone |
| `E_2_16_32_32_4_4` | 6.34 s / 3072 | 0.29 s / 48 |
| `scan_head` | absent | 3.01 s / 24 |
| **prefill** | **22.62 s** | **8.07 s** |

**2.8x -- the first real speedup measured in this tree.** But the ids differ:

```
stock: [175541, 188694, 18670, 188165, 244823, 134424, 246769, 12367]
scan : [ 56193,  95621, 157545,   6962, 230731,   1556,  94726, 53726]
```

Since the kernel is exact in isolation, the fault is in the wiring. The documented risk was
the `state_store` / `conv_writes` ordering, and that is where to look.

Gated off via `GDN_SCAN = False` with the numbers in the comment. The import fix is kept -- it
is a genuine bug fix and makes the intended path reachable once correctness is resolved.

## Corrected baseline ids

The baseline recorded earlier in this file was taken from the `port-fastkernels` run, which
had `FAST_SIGMOID` enabled -- it was contaminated and should not be used. The stock ids for
`--prompt 128 --gen 8 --warmups 3 --runs 1` are:

```
[175541, 188694, 18670, 188165, 244823, 134424, 246769, 12367]
```

## The scan kernel is verified correct; the id difference is unexplained

After the 2.8x measurement, the kernel was tested against the loop at every operand regime the
model actually uses:

| test | result |
| --- | --- |
| `check_scan_kernel.py`, random state | out rel **3.995e-07**, state rel 2.086e-07 |
| state = **all zeros** (the model's `pos==0` case) | out rel **3.995e-07** -- MATCH |
| alpha in [1e-3, 1e-1] | rel 4.859e-07 -- MATCH |
| alpha in [1e-6, 1e-4] | rel 5.707e-07 -- MATCH |
| alpha in [0, 1e-6] (extreme decay) | rel 4.892e-07 -- MATCH |
| real model geometry | `alpha_dim=1, q=(1,32,128,128), k=(1,32,128,128), v=(1,32,128,128), beta=(1,32,128), state=(1,32,128,128), T_pad=128` -- identical to the test |

So the kernel matches the loop to ~4e-7 relative in every regime, and the geometry is exactly
the tested one. `4e-7` is **not** bit-exactness, so a greedy argmax on a near-tie can flip on
it alone -- which would make the id difference benign. That is the leading explanation.

**It is not confirmed.** The decisive test is to compare pre-sampling logits between the two
paths. That was attempted with a `logits()` accessor added to `Transformer`, and the accessor
returned **all zeros** (`scale 0` over 248,320 logits) -- so the comparison was inconclusive
and the accessor was removed. The port's `Transformer` predates the archive's `logits()`.

**State:** `GDN_SCAN = False`. The tree is at 22.6 s prefill with the stock ids
`[175541, 188694, 18670, 188165, 244823, 134424, 246769, 12367]`. The import fix is kept; the
2.8x path is one correctness confirmation away.

## Logits comparison: not obtained, and the readout is at fault

The decisive test for whether the 2.8x is numerically equivalent -- compare pre-sampling
logits between the two GDN paths -- was attempted twice and failed both times, returning **all
zeros** over 248,320 logits:

| attempt | method | result |
| --- | --- | --- |
| 1 | new `Transformer.logits()` copying `forward`'s block loop | `scale 0` |
| 2 | patch `Transformer.forward` itself before model construction, so `TinyJit` binds it, returning `logits` | `AssertionError: degenerate logits (all zero)` |

Since the model demonstrably produces non-zero token ids, the model is fine and the readout is
what is broken. Both attempts were removed; no diagnostic code remains in the tree.

A third approach -- printing `core`'s values from inside the model -- failed differently:
`core.abs().max().item()` forces a realize during graph construction and raises
`usage of device CPU disallowed`. That probe was removed too.

**Status of the 2.8x:** measured (22.62 s -> 8.07 s) and the kernel is verified to ~4e-7 in
every regime, but equivalence to the loop is **unproven**. `GDN_SCAN = False`; tree at 22.6 s
with stock ids. The next attempt should validate the readout on a *known* non-zero tensor
before touching the model -- printing a single `forward` output for the loop path and checking
it is non-zero would have saved both failed runs.

## The scan path is wrong: 17% logit difference

The equivalence question is now **answered: no.**

Readout that works (two earlier ones returned all zeros):

```python
# bypass TinyJit entirely -- calling model.forward directly avoids capture bookkeeping
out = model.forward(prompt, temperature, v.bind(0)).realize().numpy()
```

Validated on **Qwen3.5-0.8B** first (18 GDN blocks): unpatched forward gives token 519, and
the readout gives logits with `scale 13.93`, so the recipe is sound. Applying it to 4B:

| | loop (`GDN_SCAN=False`) | scan (`GDN_SCAN=True`) |
| --- | --: | --: |
| logits scale | 8.3e+03 | ~1e+04 |
| **max abs diff** | -- | **1.4351e+04** |
| **relative** | -- | **1.738e-01 (17.4%)** |

17% is three to five orders above the ~4e-7 the kernel shows against the loop in isolation, so
this is a wiring fault, not floating-point drift. The earlier "benign argmax sensitivity"
hypothesis is **refuted**.

**Why the earlier readouts failed:** patching `Transformer.forward` to return logits and
letting `TinyJit` capture it produced all zeros on 4B, twice. Bypassing the JIT fixed it. The
0.8B validation also showed that the patch must be followed by rebuilding `prefill_jit` /
`rollout_jit`, since they bind `forward` at construction.

**State:** `GDN_SCAN = False`, tree at 22.6 s with stock ids. The kernel is exonerated; the
fault is in the wiring at `model.py:741-755`. To localize it, the operands at the call site
(`q, k, v, beta, alpha, state`) need comparing against what the loop uses -- they are computed
before the branch, so a difference there would mean the branch changes them, which it does not
read; that points at the `state_store` / `conv_writes` ordering or the returned tensor's
layout.

## Localized: the scan is bit-exact at T=1 and diverges through the carried state

Isolating a single `GatedDeltaNetBlock` from Qwen3.5-0.8B and calling it twice -- once per path,
on identical inputs -- reproduces the divergence outside the full model:

| T | rel diff (block output) |
| --: | --: |
| **1** | **0.0000e+00 -- bit-identical** |
| 2 | 5.3949e-06 |
| 4 | 1.0824e-04 |
| 8 | 2.0931e-04 |
| 32 | 9.4792e-05 |

And at T=32, comparing the two things separately:

| quantity | rel diff |
| --- | --: |
| block output | 9.479e-05 |
| **carried recurrent state** | **2.077e-07 -- 1-2 ulps of fp32** |

So: the output formula is right (T=1 is bit-exact), the state recurrence is right to 1-2 ulps,
and the divergence is that **1-ulp state difference amplified by the recurrence** as it carries
across steps. It is not a wiring bug of the kind previously assumed (`state_store`/`conv_writes`
ordering would show at T=1).

**What this means for the decision.** The scan path is not demonstrably wrong -- it is a
slightly different numerical trajectory (1 ulp in the state) that this model amplifies to 17%
in the final logits over 18 GDN blocks and 128 tokens. Under greedy decoding that changes every
token. Whether that is acceptable is a product decision, not a measurement:

| option | consequence |
| --- | --- |
| keep the loop | stock ids, 22.6 s, bit-stable |
| enable the scan | 8.07 s, ids differ, 1-ulp-per-step divergence amplified |
| require bit-equality | not achievable -- the kernel sums in a different order by design |

The honest statement: **the 2.8x is real and the implementation is numerically sound per step,
but it is not bit-reproducible against the loop, and this model amplifies the difference.**

## Where the divergence starts: exactly one ulp, at step 2

Per-T comparison of one `GatedDeltaNetBlock` (Qwen3.5-0.8B, flag flipped between calls, state
reset by `_init_state` each time):

| T | out max abs | out bit-equal | state max abs | state bit-equal |
| --: | --: | :---: | --: | :---: |
| **1** | **0.000e+00** | **yes** | **0.000e+00** | **yes** |
| 2 | 7.644e-06 | no | 2.980e-08 | no |
| 3 | 1.013e-06 | no | 1.490e-08 | no |
| 4 | 1.226e-04 | no | 2.980e-08 | no |
| 8 | 2.442e-04 | no | 5.960e-08 | no |

Reading it:

- **The first step is bit-identical in both output and state.** So the formulas, the operand
  layouts and the first state update all agree exactly.
- From step 2 the state differs by **2.98e-08 absolute on a state scale of 0.574**, and one
  ulp of 0.574 in fp32 is ~2.98e-08. The divergence is therefore **exactly one ulp** -- not a
  formula error, not a layout error.
- The output error (1e-6 to 2.4e-4) is that 1-ulp state difference propagated through
  `(state * q).sum(-1)` and amplified by the recurrence.

**The leading hypothesis is FMA contraction.** The state update is

```
s1    = state * alpha
delta = (v - (s1 * k).sum(-1, keepdim=True)) * beta
state = s1 + delta * k
```

Clang defaults to `-ffp-contract=on`, so `s1 + delta * k` compiles to a single fmla with **one**
rounding, while tinygrad's separated `add` of a separated `mul` rounds **twice**. One ulp is
exactly the expected difference.

This port has already hit this precise bug: `_dequant_source` in `npu_linear.py` carries

```c
#pragma clang fp contract(off)
```

with the comment *"clang defaults to -ffp-contract=on, which fuses `t*w` into an fmla and
rounds once where the unfused chain rounds twice. The unfused path this replaces rounds twice,
so the fused kernel must too."*

The next step is to apply the same pragma to the scan kernel's source and re-run the per-T
comparison: if the state becomes bit-equal at every T, the 2.8x is bit-stable and can be
enabled with no fidelity loss.

## The kernel's bit-exactness claim does not hold

`extra/rockchip/scan.py` documents bit-exactness as a design requirement:

> "Bit-exactness. The replacement must produce the same fp32 bits as the loop, so the
> reduction order is not free. tinygrad's CPU scheduler upcasts the reduce axis by 4
> (`SPLIT axis=3 arg=(4, UPCAST)` for a 128-wide fp32 reduce), which generates
> `acc = acc + (p0*k0 + p1*k1 + p2*k2 + p3*k3)` ... `#pragma clang fp contract(off)` is
> required for the same reason"

Both mechanisms are present in the emitted source (`#pragma clang fp contract(off)` at
`_scan_source`, the 4-wide grouping in `_SCAN_SOURCE`), and `alpha_dim == 1` correctly
substitutes `at[0]`.

**Yet the state is not bit-exact from step 2.** At T=2:

| observation | value |
| --- | --- |
| differing state elements | **61,039 of 262,144 (23.3%)** |
| distribution | spread over all `k` and all `v`, no positional pattern |
| magnitude | 1-2 ulps, e.g. `-2.3059998e-07` vs `-2.3060011e-07` |

A broad last-bit difference across every index is a **reduction-order mismatch**, not a
localised defect -- so either the documented grouping is not the order tinygrad actually emits
for this shape, or the STORE/accumulate structure differs.

**Consequence for the goal.** Bit-stability cannot be reached by tuning the pragma (already
correct) or by fixing an index (none is wrong). It requires mirroring the *actual* generated
reduce that tinygrad emits for `(s1 * k).sum(-1)` in the GDN state update. The concrete next
step is to dump that generated kernel -- the same way `r_2_16_32_4_32_4` was dumped earlier --
and match its loop nest exactly.

## The loop is not ground truth: the scan kernel is 2x more accurate

Every earlier section of this file treated the CPU loop (`GDN_SCAN=False`) as the reference the
scan had to match. That was an assumption, never a measurement -- and it is wrong.

Computing the same recurrence in **float64** with numpy and comparing both fp32
implementations against it (random inputs, real geometry, `extra/rockchip/check_scan_kernel.py`
layouts):

| arm | out rel vs fp64 | state rel vs fp64 |
| --- | --: | --: |
| loop (fp32) | 4.209e-07 | 2.228e-07 |
| **scan kernel** | **2.003e-07** | **1.599e-07** |

**The scan kernel is roughly 2x closer to the exact answer than the loop.** So:

- The ids differing does **not** mean the scan is broken. It means both arms are fp32
  approximations, they differ from each other by ~4e-7, and the model amplifies that.
- The loop, which produced the "stock" ids `[175541, ...]`, is the **less accurate** of the two.
- "Matches the loop" was the wrong acceptance criterion. "Closer to fp64" is the right one, and
  on that criterion the scan passes.

Not measured: comparison against the **reference repo's** logits. Token-for-token agreement
across independent fp32 implementations is not expected for greedy decoding, so the meaningful
comparison is numerical accuracy, not ids.

**Revised verdict on the 2.8x.** It is a speedup *and* an accuracy improvement over the loop.
The earlier framing -- "numerically sound but not bit-reproducible, so the loop wins by
default" -- is withdrawn.

## Against the reference repo: both arms are equally distant

The comparison that was missing all along. Same prompt (32 ids), same `start_pos`, Qwen3.5-0.8B,
three implementations: this tree's loop, this tree's scan kernel, and the **archive's own
`Transformer.logits()`** (`~/tg-qwen35-pack-signature-fix`, run with `~/tinygrad/.venv`).

| comparison | max abs diff | relative |
| --- | --: | --: |
| our loop vs **REF REPO** | 3.4804e-03 | **2.498e-04** |
| our scan vs **REF REPO** | 3.6969e-03 | **2.653e-04** |
| our loop vs our scan | 3.6588e-03 | 2.626e-04 |
| argmax, all three | -- | **519, top-2 margin 0.389** |

- The two arms differ from each other (2.63e-4) by the **same amount** they differ from the
  reference (2.50-2.65e-4). The scan is not "further from correct" than the loop; both are
  equally distant from a third fp32 implementation.
- The loop is 6% closer to the reference than the scan -- negligible against the 250x that both
  are away from it.
- All three pick the **same argmax** on this prompt, because the margin is large (0.389). The
  4B id divergence happens where margins are small, not universally.

**Supersedes** the earlier fp64-based claim that "the scan is 2x more accurate than the loop".
That held for one block-level sample and does not hold at model level against the reference.
Both statements are true; neither decides the question.

**What this means for the decision.** Enabling the scan does not make the model more or less
faithful to the reference than the loop does -- it is a numerically equivalent fp32 path, 2.8x
faster, that will pick a different token only where the reference itself is within ~2.6e-4 of
flipping. The fidelity argument against it does not survive this measurement.

## The benchmark checkpoint is unsloth's Q8_0; rkllama ships ggml-org's

Both local `Qwen3.5-0.8B-Q8_0.gguf` files are legitimate Q8_0, from **different converters**:

| source | size | sha256 | location |
| --- | --: | --- | --- |
| **unsloth** (ships `imatrix_unsloth.gguf_file`) | 811,843,840 | `0ad885ff...b7a6c` | `~/vlm-bench/baseline-20261021/checkpoints/` -- **every reference row** |
| **ggml-org** | 833,592,096 | `37ae482d...7814f` | `~/rkllama/models/` -- rkllamacpp |

Identified by matching file sizes and sha256 against the Hugging Face tree APIs for
`unsloth/Qwen3.5-0.8B-GGUF` and `ggml-org/Qwen3.5-0.8B-GGUF`.

Consequences:

- The reference rows are reproducible from their own pinned checkpoint sha256
  (`10cc391b...` for 4B), so nothing in this file is invalidated.
- A "Q8_0 baseline" from another source is **not the same weights**: these differ by 2.6% in
  size, so the quantization decisions differ.
- Any cross-engine logits comparison must load the **same** file on both sides. rkllamacpp as
  installed would use ggml-org's; the benchmark uses unsloth's.

## Scan accepted: the loop is no more faithful to the reference at 4B

4B, reference prompt (128 ids from `qwen35-4b.txt`), checkpoint `10cc391b`, 64 greedy tokens,
compared against the archive's own `ref-warmed-4b` ids (stable across all its trials):

| arm | tokens agreeing with REF | first 8 |
| --- | --: | --- |
| our **loop** (`GDN_SCAN=False`) | **0/64** | 200263, 102772, 131208, 141521, ... |
| our **scan** (`GDN_SCAN=True`) | **0/64** | 37606, 105672, 236349, 8472, ... |
| REF (archive) | 64/64 | 10771, 303, 279, 220, ... |

**The stock loop is also 0/64.** So the loop's ids are not a fidelity standard: our tree does
not reproduce the reference's 4B tokens with either arm. The argument "keep the loop because it
matches" has no basis at 4B.

Ruled out as causes for the 4B divergence: `embedding_scale` is 1.0 for `qwen35` (so our
forward's omission is harmless), and `rope.dimension_sections = [11, 11, 10, 0]` is identical
in 0.8B and 4B and referenced by **neither** tree. At 0.8B our logits agree with the reference
to 2.5e-4, so the 4B token divergence is consistent with amplification of that scale over 64
greedy steps rather than a structural 4B defect -- but it is not proven, because the archive's
4B logits could not be obtained (its CPU-only path did not finish in 25 minutes).

**Separate pre-existing gap:** our tree has never reproduced the reference's 4B tokens. That is
independent of the GDN choice and predates this work.

**Verdict: the scan is accepted.** It is numerically equivalent to the loop at 0.8B (2.653e-4
vs 2.498e-4 against the reference) and identical to it in divergence at 4B (0/64 both). It is
2.8x faster. The condition is that the ids be re-baselined -- `[175541, ...]` describes one
approximation's trajectory, not a standard.

## Reference per-kernel profile, 4B (the measurement the method requires)

The `metal_kernels` method's first step is to profile the reference before changing anything.
That had never been done for 4B. `extra/rockchip/benchmark_prefill.py` is SmolVLM-specific and
cannot be pointed at this model, so the profile came from the archive's own
`benchmark_inference.py` with `ONPU_REPLAY_PROFILE=1` — run
`docs/rknpu/results/ref-profile/`, 2 warm-ups + 1 run, `--tokens 8`.

That run reproduced the known shape (17.2 / 9.2 / 26.3 then 1.200 / 1.201 / 1.201 / 1.203 s), so
its steady state matches the published 1.263 s. `native_execution.prefill.runs = 4`,
`reason = null`.

| ms per run | calls | kernel |
| --: | --: | --- |
| 765.9 | 448 | `ROCKCHIP:rockchip_int8_64_12288_2560` |
| **704.6** | **96** | **`CPU:rockchip_cpu_gated_delta_scan`** |
| 396.9 | 192 | `CPU:E_2_16_32_32_4_4` |
| 348.3 | 520 | `ROCKCHIP:rockchip_int8_64_2560_4096` |
| 229.5 | 256 | `ROCKCHIP:rockchip_int8_64_6144_2560` |
| 216.2 | 128 | `CPU:rockchip_int8_swiglu_pack` |
| 204.5 | 96 | `CPU:E_32_32_32_4_4` |
| 203.9 | 64 | `CPU:E_64_16_64_4` |
| 188.0 | 96 | `CPU:E_32_32_128_4` |
| 182.1 | 192 | `CPU:r_32_16_4_32_4` |

**Inflation.** The listed kernels sum to 3,440 ms against a 1,200 ms wall, so
`ONPU_REPLAY_PROFILE` inflates ~2.9x here. Use these numbers for **ranking only**, never as
timings.

### What this says to port

| component | ours | archive (divided by 2.9) | ratio |
| --- | --: | --: | --: |
| GDN scan | `scan_head` 3.01 s, 24 calls | `gated_delta_scan` ~246 ms, 96 calls | **~12x** |
| NPU matmul | `ioctl` 0.88 s | 3 `rockchip_int8_*` ~450 ms | ~2x |
| SwiGLU packing | inside generic elementwise | `swiglu_pack` ~75 ms | -- |

Priority is therefore **the archive's `rockchip_cpu_gated_delta_scan`**, not the NPU matmuls:
the scan differs by ~12x per block against ~2x for the matmul, and our whole scan is 3.01 s
versus the archive's ~0.25 s.

### Two corrections this profile settles

- `E_2_16_32_32_4_4` is present **in the archive too** (396.9 ms, 192 calls). It was never a
  port-specific kernel, and identifying it as the SwiGLU sigmoid was wrong for both trees.
- The archive's advantage is **dedicated CPU kernels** (`gated_delta_scan`, `swiglu_pack`) where
  we emit generic tinygrad elementwise ops -- not the NPU path.

## 3+3 with the scan enabled: 7.95 s, and stable

`docs/rknpu/results/scan-3plus3/`, `kind: npu-baseline-timing`, `complete: true`:

| trial | prefill | decode |
| --: | --: | --: |
| 1 | 7,950 ms | 2.95 tok/s |
| 2 | 7,964 ms | 2.97 tok/s |
| 3 | 7,912 ms | 2.96 tok/s |
| **median** | **7,950 ms** | **2.96 tok/s** |

Trial spread **0.7%** -- the scan path shows no warm-up artefact, unlike the loop.

## Reference kernel port: the ABI was the blocker (and it is now fixed)

The SIMD GDN scan from `rockchip_recurrent.c` is ported and **executes**. What blocked it for
six attempts was the argument convention, and it is worth writing down because every future
kernel port will hit it.

`CPUProgram` calls the loaded object as `ctypes.CFUNCTYPE(None, c_void_p)`, i.e. **one packed
argument buffer** (`ClangRenderer.packed_args = True`). The convention that works is the one
`scan.py` and `_dequant_source` already use:

- exactly **one non-static function** in the object -- that exported symbol is what gets called;
- its signature is `void entry(void *args)`;
- it unpacks each buffer itself: `float *out = *(float**)args;  const float *q = *((const float**)args + 1); ...`
- every helper is `static`, and `static` functions are not the entry.

My port declared seven typed parameters (`float *out, const float *q, ...`). The caller passes a
single pointer, so the callee read garbage, wrote nothing, and -- in one variant -- segfaulted.
Six "fixes" chased the wrong thing (entry offset, `#include` order, `outs`, argument count,
`start`/`rows` placeholders, `extern memset`). Reading `ops_cpu.py:46` and comparing the *number
of non-static functions* in each source is what found it.

### Result

| | ours `scan_head` | ported SIMD | speedup |
| --- | --: | --: | --: |
| per block, min of 5 | **128.36 ms** | **29.32 ms** | **4.38x** |
| projected 24-block prefill scan | 3.08 s | **0.70 s** | -- |

**Correctness so far:** `state` matches the existing kernel to **2.136e-07**. `out` differs on
**8.06% of elements** (42,238 of 524,288, threshold 1e-4 relative), spread across 126 of 128
positions and all 32 heads -- so it is a *semantic* difference in how `out` is derived, not a
layout or indexing error. One sample: ours 0.2172 vs port 0.1839.

That is the remaining work before this can be wired in, and it is a convention question (very
likely whether `out` is taken from the pre- or post-update state), not a performance one.

**Projection, not a measurement:** with `out` matched, the prefill scan falls 3.08 -> 0.70 s,
which would take the 4B prefill from **7.95 s to about 5.6 s**. Still 2.7x behind RKLLM's
2.069 s, so this is a step, not the finish.

## The ported scan: 4.38x faster, wrong, and geometry-dependent

fp64 comparison at the real geometry (1,32,128,128,128):

| arm | out vs fp64 | state vs fp64 |
| --- | --: | --: |
| ours `scan_head` | **2.003e-07** | 1.599e-07 |
| ported SIMD | **1.132e-01** | 2.118e-07 |

That looked like "recurrence right, output wrong". It is not. Isolating to one head and one token:

| geometry | ours out | port out | port state |
| --- | --: | --: | --: |
| (1,1,1,128,128) | 3.199e-07 | **3.533e-01** | **2.947e-01** |
| (1,1,2,128,128) | 4.006e-07 | 1.198e-01 | 5.270e-02 |
| (1,1,4,128,128) | 3.020e-07 | 1.558e-01 | 6.412e-02 |
| (1,1,1,64,128) | 2.307e-07 | 4.562e-01 | 3.174e-01 |
| (1,1,1,16,128) | 1.515e-07 | 4.715e-01 | 2.680e-01 |

Our kernel is correct at every geometry. The port is wrong on **both** out and state, and its
state error grows as the geometry shrinks: 2.1e-07 at H=32, **2.9e-01 at H=1**. So the recurrence
is not correct either, and the H=32 state agreement was not evidence that it was.

Consequence: this is a **geometry-dependent defect in the port**, not an output convention. Fixing
it is a debugging task in the ported C (the `value_dim%64` / `%16` tile selection, the `group`/
`groups` loop bounds, or the `capacity` stride), not a reshape.

**Verdict: not wired in.** It is 4.38x faster per block (128.36 -> 29.32 ms) and numerically
wrong at every geometry tested, so there is no projection to make from it.

### Single-step probe: the port is wrong before any recurrence

At B=1, H=1, T=1, K=V=128 the initial state is zero and `start=0`, so the whole step collapses to

    s = (v * beta) * k          (since s1 = 0 and predicted = 0)

Both implementations must produce that identically. They do not:

| | state[0,0,0,:6] |
| --- | --- |
| ours | 0.070503, 0.011687, 0.045804, -0.036798, -0.007884, -0.012312 |
| port | 0.074578, 0.038130, 0.028446, -0.052618, 0.016882, -0.012478 |

Not a permutation: `rel(s2, s1)` = 2.947e-01, `rel(s2, s1.transpose)` = 1.490e+00,
`rel(s2, s1[..., ::-1])` = 1.573e+00.

So the defect is not in the recurrence accumulation, the tile selection or the output layout -- it is
present in the very first step, where `columns` is zero and `predicted` must be zero. That points at
the operand addressing (`kt`/`vt`/`at` offsets) or at the `predicted` accumulators not starting from
zero, and it explains why the H=32 state agreement was misleading: a longer sequence happens to mask
a first-step error.

**Status: unwired, defective, and not a quick fix.** It is 4.38x faster per block and cannot be used.

## The SIMD scan is wired in: 7,950 -> 5,680 ms

The defect was **one argument**. The reference kernel zeroes the state when `start==0`
(`columns[c*V+r] = start ? s[r*K+c] : 0`), and the wrapper had `start` baked to `0`, so the port
silently **ignored the supplied initial state**. With `start=1` it matches an fp64 replay of the
recurrence at every geometry tested:

| geometry | ours out / state | port out / state |
| --- | --- | --- |
| (1,1,1,128,128) | 3.199e-07 / 1.084e-07 | **2.588e-07 / 1.084e-07** |
| (1,1,4,128,128) | 3.020e-07 / 1.451e-07 | **4.479e-07 / 2.052e-07** |
| (1,32,128,128,128) | 2.003e-07 / 1.599e-07 | **4.209e-07 / 2.118e-07** |

Each case was run in a **fresh process** (`extra/rockchip/scan_case.py`), which is what removed the
contradiction: the earlier readings reused one interpreter and therefore one cached program.

### Measured end to end

`docs/rknpu/results/scan-simd-3plus3/`, `kind: npu-baseline-timing`, `complete: true`:

| trial | prefill | decode |
| --: | --: | --: |
| 1 | 5,678 ms | 3.00 tok/s |
| 2 | 5,680 ms | 2.99 tok/s |
| 3 | 5,714 ms | 2.99 tok/s |
| **median** | **5,680 ms** | **2.99 tok/s** |

| metric | now (SIMD scan) | before (scan_head) | archive | RKLLM | vs RKLLM |
| --- | --: | --: | --: | --: | --: |
| prefill, 128 tok | **5,680 ms** | 7,950 ms | 1,263 ms | 2,069 ms | **2.75x slower** |
| prefill tok/s | **22.5** | 16.1 | 101.3 | 61.87 | 2.75x |
| decode tok/s | **2.99** | 2.96 | 6.48 | 6.22 | 2.08x slower |

The 5.6 s projection is therefore *measured*, not projected: 5,680 ms against a predicted 5,600.

**Caveat on the A/B discipline:** this is a before/after from two separate 3+3 runs, not an ABBA
pair sequence. The internal spread is 0.6-0.7% per run and the effect is 28.5%, so the direction is
not in doubt, but it is not the pair-count form the method prescribes.

**Token ids changed** (`180130, 6608, 179997, ...` against the previous `176841, 161398, ...`).
Expected: this is a different fp32 path at 2-4e-07, exactly as scan_head differed from the loop. The
ids must be re-baselined per configuration.

## The attention port is not applicable to Qwen3.5 4B

`RockchipAttentionProgram` gates on the model geometry at construction:

```python
if (magic!=ATTENTION_MAGIC or version!=1
    or (self.heads, self.kv_heads, self.dim) not in ((16,8,128),(32,4,64))
    or self.context not in (64,128,256,512) ...):
  raise ValueError('Unsupported NPU prefill attention descriptor')
```

Qwen3.5 4B from its GGUF metadata:

| field | value |
| --- | ---: |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | 4 |
| `attention.key_length` | 256 |
| `attention.value_length` | 256 |

That is **(16, 4, 256)** -- not in `((16,8,128), (32,4,64))`, so the descriptor is rejected before
anything runs. Porting it would have been a dead end, and the reference's own 4B profile agrees: it
contains no attention-specific kernel, only generic ones (`E_32_32_128_4` 188 ms, `r_32_16_4_32_4`
182 ms, `E_64_16_64_4` 204 ms profiled).

**So the attention gap is not a missing specialised kernel -- it is kernel *quality*.** Ours is
`r_4_128_[start_pos]` at 2.48 s; the archive's generic equivalents rescale to roughly 0.15-0.2 s. A
~16x gap between two compositions of generic CPU kernels.

The next step is therefore a comparison of the two *generated* attention kernels for the same
geometry, not a port. Ours carries a symbolic `start_pos` extent and an added causal mask; the
archive's are plain `E_`/`r_` kernels over fixed shapes. Whether that symbolic extent is what costs
16x is the question to answer before writing any kernel.

## Attention: 456 ms in-model, not 2.48 s (isolated timing did not transfer)

The earlier claim that attention was ~2.48 s of prefill was built from an isolated per-block
measurement (252 ms x ~8). Ablation refutes it. Same tree, same protocol, only the attention
sublayer changed:

| configuration | prefill | delta |
| --- | --: | --: |
| generated `scaled_dot_product_attention` | 5,680 ms | -- |
| hand-written fp32 kernel (`attn.c`) | 5,669 ms | **-11 ms** |
| attention **removed entirely** | **5,213 ms** | **-456 ms** |

So the whole attention sublayer is **456 ms** -- an **upper bound**, because that ablation ran on the
capture path (1 warm-up), so it removes attention's trace cost as well as its execution cost. The
knee-jerk "8 blocks x 252 ms" was an isolation artefact -- the isolated probe ran ~5x slower than the same work in-model, the opposite
of the scan case. The mask/GQA test (`mask costs 7 ms of 252`, `GQA costs nothing`) is still valid
for what it measured, but it was measuring the wrong thing: the block cost never applied in-model.

The hand-written kernel is kept because it is verified (1.888e-07 vs fp64 at T=128, 4.165e-07 for a
decode step at pos=137) and 14.6x faster in isolation, but its **end-to-end contribution is 11 ms**
and must not be reported as more than that.

### Why it could not be wired the obvious way

`RockchipAttentionProgram` rejects this model outright -- it admits only `(16,8,128)` and
`(32,4,64)`, and 4B is `(16,4,256)`. And the KV side arrives as a slice with a **symbolic extent**
`(1,4,start_pos+128,256)`, which tinygrad cannot turn into a placeholder
(`no placeholder-like on symbolic shape`). The kernel therefore takes the **whole concrete cache
buffer** `(B,2,KVH,CAP,D)` plus `start_pos` as a float scalar. Float, not int32: an int32
placeholder is the failure mode `scan_native.py` records as a kernel that silently never executes,
and reading a float buffer as int crashed the process until it was fixed.

## The extreme pattern in the trials is per-trial graph construction

`bench_llm.py` called `model(prompt_tensor, (v_start.bind(0),), temperature)` **inside** the trial
loop. `bind` allocates a fresh UOp each call and `TinyJit` keys capture on that UOp's identity, so
**every trial re-traced the whole 1,359-program graph** -- about 2.7 s of Python on top of the work.
That is the entire explanation for three slow trials with a tight spread.

Binding once, outside the loop:

| trial | before | after |
| --- | --: | --: |
| warmup 1 | 9.1 s | 9.1 s (capture + compile) |
| warmup 2 | 5.7 s | 5.7 s |
| warmup 3 | 5.7 s | **3.0 s** |
| measured 1 | 5,678 ms | **2,972 ms** |
| measured 2 | 5,680 ms | **2,971 ms** |
| measured 3 | 5,714 ms | **2,969 ms** |

`docs/rknpu/results/bind-3plus3/`, `kind: npu-baseline-timing`, `complete: true`, 3 warm-ups + 3
measured, 128-token prompt, 64 greedy tokens.

| metric | now | RKLLM | archive | vs RKLLM | vs archive |
| --- | --: | --: | --: | --: | --: |
| prefill, 128 tok | **2,971 ms** | 2,069 ms | 1,263 ms | **1.44x slower** | 2.35x slower |
| prefill tok/s | **43.1** | 61.87 | 101.3 | 1.44x | 2.35x |
| decode | **3.10 tok/s** | 6.22 | 6.48 | **2.01x slower** | 2.09x slower |

**What did and did not get faster.** The 5,680 -> 2,971 ms improvement is *entirely* the harness
fix: the model's own execution was always ~2.97 s and the old number carried ~2.7 s of redundant
tracing. Nothing about the model got faster this turn. For the record, the earlier numbers under the
same defective harness were 22.45 s (scan never ran) -> 7.95 s (scan enabled) -> 5.68 s (SIMD scan),
so the native equivalents are roughly 19.7 -> 5.25 -> **2.97 s**, i.e. the SIMD port was worth ~2.28 s
of real execution and remains the only genuine model-level speedup of the session.

This is a genuine fix, not relocating work into warm-ups: an application creates its position
variable once, so the per-trial 2.7 s was never work a real caller pays. What the warm-ups absorb is
one-time JIT capture and compilation (call 1 ignores, call 2 captures, call 3 replaces, call 4+
replays), which is inherent to the runtime and which RKLLM's own warmed numbers also exclude.

### Native per-kernel profile of the 2.97 s prefill

`profile_graph_dispatch.py` at **3 warm-ups + 2 runs**, so the reported prefill rows are native
replays. The prefill rows come out at **2.96 s loop / 2.75 s kernel / 0.20 s bookkeeping**, which
matches the 2,971 ms trial -- the earlier 3.34 s figure was a capturing traversal and is superseded.

`docs/rknpu/results/native-prof/`, prefill row:

| kernel | calls | time | us/call | share |
| --- | --: | --: | --: | --: |
| `ioctl` (NPU linear submit) | 635 | **1.02 s** | 1,606 | 34% |
| `cpu_gated_delta_scan` (our port) | 24 | **0.62 s** | 25,833 | 21% |
| `E_2_16_32_32_4_4` (SwiGLU sigmoid) | 48 | 0.35 s | 7,292 | 12% |
| `E_589824_2` | 64 | 0.21 s | 3,281 | 7% |
| `E_2048_32_2_4` (attention) | 24 | 0.17 s | 7,083 | 6% |
| `r_32_16_4_32_4` (attention) | 48 | 0.17 s | 3,542 | 6% |
| `E_32_32_128_4` (attention) | 24 | 0.16 s | 6,667 | 5% |
| `exp2_scalar` (our attention kernel) | 8 | 0.12 s | 15,000 | 4% |
| `E_163840_2` | 32 | 0.10 s | 3,125 | 3% |
| `E_131072_4` | 96 | 0.06 s | 625 | 2% |
| `E_64_640_2_4`, `E_268288_4`, `r_64_2_640_4`, `E_32_640_4_4` | 120 | 0.12 s | 1,000 | 4% |

Two facts that constrain what is reachable:

- **The whole attention sublayer is 0.56 s of kernels** (0.17 + 0.17 + 0.16 + 0.12), consistent with
  the 456 ms ablation. Removing it *entirely* would still leave 2.4 s, so attention cannot close the
  gap to RKLLM on its own.
- **`ioctl` is 1.02 s, but `bench_llm` attributes only 0.4 s of it to NPU submit time.** The remaining
  ~0.62 s is host-side work inside the generated `rockchip_linear` (the INT8 activation quantization
  loop, the `sync_mem` call, and the completion wait). That is the largest single item and the only
  one large enough to matter: halving it is worth ~0.5 s.

### Ablation: the NPU offload is worth 110x and cannot be removed

Same tree, same prompt, only `--npu` dropped so every `nn.Linear` runs on the CPU host:

| arm | prefill | linears offloaded |
| --- | --: | --: |
| `--npu` (current) | **2,971 ms** | 249 |
| CPU-only | **327.3 s** | 0 |

So `ioctl` is not overhead to be eliminated -- it is the only reason this model runs at all on this
board. The question is strictly how much of its 1.02 s is avoidable, and 0.62 s of that is host-side
work inside the generated `rockchip_linear` (INT8 activation quantization, `sync_mem`, completion
wait) rather than NPU compute (0.4 s measured as submit time).

### Bound: what closing the prefill gap would actually require

Prefill is 2,971 ms; RKLLM is 2,069 ms. The 0.90 s that must go, against the native profile:

| item | time | can it yield 0.90 s? |
| --- | --: | --- |
| `ioctl` | 1.02 s | only if roughly halved, and 0.4 s of it is real NPU compute |
| `cpu_gated_delta_scan` | 0.62 s | only if roughly halved again; it is already the reference kernel at 4.38x our scalar one |
| SwiGLU + elementwise | 0.68 s | no single item is large enough |
| attention (all four kernels) | 0.56 s | no -- removing it entirely leaves 2.4 s |
| bookkeeping | 0.20 s | no |

**No single kernel can close it.** Even deleting the entire attention sublayer and the whole SwiGLU
path would land at ~2.07 s, i.e. exactly RKLLM, with nothing left over. The reachable route is
submit-count reduction (fuse `attn_q`/`attn_k`/`attn_v` and `ffn_gate`/`ffn_up`, each of which
currently pays its own submit *and* its own quantization of the same input): 249 linears produce 635
submits, and fusing those two groups removes ~3 submits per layer, ~17% of `ioctl`.

## NEON INT8 activation quantization: 2,971 -> 2,773 ms, bit-identical

The fused Linear quantizes its activation in C, and that loop is scalar: two passes over `depth_in`
per row (max, then codes). Its arithmetic matched the measurement exactly -- ~6 linears x 128x2560
plus `down` at 128x9728 is ~3.2 M elements/layer, x2 passes x32 layers = ~205 M scalar element-ops,
and a standalone timing gave 0.831 ms (depth 2560) + 3.247 ms (depth 9728) per layer = 8.2 ms, i.e.
~0.30 s of the 0.62 s that `rockchip_linear` spends outside the NPU submit.

Replaced with a NEON version (`qz_row` in `extra/rockchip/npu_linear.py`, with a scalar `#else` for
non-ARM builds). Measured standalone:

| shape | scalar | NEON | speedup |
| --- | --: | --: | --: |
| m=128, depth=2560 | 0.831 ms | **0.275 ms** | 3.02x |
| m=128, depth=9728 | 3.247 ms | **1.063 ms** | 3.06x |

In-model, `--prompt 128 --gen 64 --warmups 3 --runs 3`:

| trial | before (`bind-3plus3`) | after (`neonqz2-3plus3`) |
| --- | --: | --: |
| 1 | 2,972 ms | 2,773 ms |
| 2 | 2,971 ms | 2,772 ms |
| 3 | 2,969 ms | 2,776 ms |
| **median** | **2,971** | **2,773** |
| decode | 3.10 tok/s | 3.10 tok/s |

**-198 ms, -6.7%.** Two runs per arm, `--prompt 128 --gen 64 --warmups 3 --runs 3`:

| arm | run 1 | run 2 |
| --- | --: | --: |
| baseline (scalar quantize) | 2,971 ms | 3,003 ms |
| NEON quantize | 2,773 ms | 2,770 ms |

Quoted prefill is the mean of the two valid NEON arms: **2,771 ms (46.2 tok/s)**, decode 3.10 tok/s.
`docs/rknpu/results/neonqz-3plus3/` (2,680 ms) is the broken NaN-collapse run, marked `INVALID.md`,
and must not be quoted -- its faster time is a degenerate model, not a better kernel.

**Pairs where NEON is faster: 4 of 4.** The worst NEON run beats the best baseline run by 198 ms, so
the arms do not overlap (baseline spread 32 ms, effect 198-233 ms). This is two runs per arm from
adjacent sessions, not a four-run ABBA sequence; the arms are separated by more than their own spread,
which is the property the pair count is there to establish.

**Parity is proven at the function boundary, not end to end.** `qz_row` was compared against the
exact scalar loop it replaced, using the *emitted* source, over 120 cases: depths
2560/9728/4608/5120/1280/640/2559/9727/33/31/8/4/3/2/1, uniform values, exact half-integer ties,
and 30 cases with NaN lanes. **0 codes differing, 0 scales differing.** Identical codes and scales
mean the NPU receives identical input, so the change cannot alter the model's outputs.

### The NaN trap that made the first attempt useless

The first version used `vmaxq_f32` directly and the model collapsed to a repeated token
(`248320` x 8). Cause: `vmaxq_f32` **propagates** NaN, while the scalar `if (a > largest)` ignores
it. The activations do contain NaN lanes, so one NaN turned the row's scale into NaN, zeroed every
code, and produced zero output everywhere.

`a < a` is **not** a NaN test -- every IEEE comparison against NaN is false, so `vcltq_f32(a, a)`
is false for NaN and the guard did nothing. The correct test is `a != a`:
`vmvnq_u32(vceqq_f32(a, a))`. Substituting 0 for NaN lanes is exact because `largest` starts at 0 and
every term is `|x| >= 0`.

### Correction: `generated_ids` is not a parity oracle

Greedy decode in this tree is **not reproducible**, even between trials of one run:

| config | trial 1 | trial 2 | trial 3 |
| --- | --- | --- | --- |
| `bind-3plus3` | 101852, 52866 | 88701, 29775 | 7638, 147157 |
| `neonqz2-3plus3` | 157783, 211219 | 248162, 246833 | 180658, 147830 |
| `scan-simd-3plus3` | 180130, 6608 | 223440, 9027 | 124825, 205571 |

Same binary, same arguments, three different outputs. So `generated_ids` cannot verify a kernel
change, and the earlier statements in this document that treated a change of ids as "expected for a
different fp32 path" were unfounded. Whether this is state carried between trials or genuine
non-determinism is **not yet established** and is now an open defect in the harness: a benchmark
whose output is not reproducible cannot support any numerics claim.

## Fusion of qkv and gate/up: exact, but slower -- rejected

Goal was fewer NPU submits. 40 groups were fused (`attn_q`/`attn_k`/`attn_v` for the 8 attention
blocks, `ffn_gate`/`ffn_up` for all 32). Both fusions are **exact**: the fused weight is bit-identical
to the concatenation, the output is bit-identical to the separate Linears (`max abs diff 0.0` on the
float path), and INT8 weights carry **one fp32 scale per output row** with `TILE=24576` larger than
both fused widths, so every column keeps its scale through a fused GEMM.

Measured, `--prompt 128 --gen 64 --warmups 3 --runs 3`:

| arm | submits | prefill | decode |
| --- | --: | --: | --: |
| before fusion | 635 | **2,771 ms** | 3.10 tok/s |
| fused, 40 groups | **603** | **3,179 ms** (+14.7%) | 3.14 tok/s (+1.3%) |

**Rejected and reverted.** Two reasons the submit saving never materialised:

1. The fused `gate`/`up` weight is 19456 columns, above the 12288 three-core limit for `m>1`, so the
   wrapper splits it back into two submits -- the same count as before the fusion. Only `m==1`
   (decode) gets a single submit, which is why decode moved and prefill did not.
2. `prepare` packs by `TILE=24576`, so the fused 19456-column weight becomes one 24576-column tile
   and the NPU does the same arithmetic over a **24.5%-wider** tile. That is the prefill regression.

Reverted tree re-verified: 635 submits, **2,874 ms**, 3.11 tok/s.

**What this rules out:** the reference's `rockchip_int8_64_12288_2560` cannot be reproduced by fusing
Linears at the model level, because a fused width above 12288 splits and a width below it is padded
up to the tile. Reaching that shape needs task-level changes inside the NPU path (building one
submit with 3 cores x 4096 columns from several Linears' packed tiles), not weight concatenation.

## Per-submit cost is FLAT at ~1.1 ms -- the submit count is the lever

Same code (post-NEON), two prompt sizes, native rows (8/13 of each profile):

| prompt | prefill submits | NPU program time | per call | prefill loop |
| --- | --: | --: | --: | --: |
| 64 | 323 | 0.360 s | **1,115 us** | 1.45 s |
| 128 | 635 | 0.700 s | **1,102 us** | 2.87 s |

Per-submit cost is **independent of the work per submit** (m=64 vs m=128 gives the same 1.1 ms), so
it is fixed overhead -- the two `sync_mem` ioctls, the submit ioctl, and completion latency -- not
arithmetic. This corrects the statement in the previous section that the cost "scales with work":
it does not. Total prefill NPU time is therefore very nearly `submits x 1.1 ms`.

### Where the 635 comes from, and what halving it is worth

`npu_linear.py:688`:

    row_tile = (128 if k_max <= 2048 else 64) if self.int8 else max(1, min(64, 65536 // k_max))

K_TILE is 4096, so `row_tile` is 64 and a 128-token prefill splits **every** Linear into two submits
(635 ~ 2 x 313, plus the extra K tiles for `ffn_down`). Two consequences:

- **Halving the submit count is worth ~0.35 s (12.5% of prefill)** -- the single largest identified
  win remaining, and it needs no new kernel.
- It requires `m <= 128` at `k = 4096`, i.e. raising `row_tile` to 128. The recipe comment records
  the verified envelope as "m<=128 at k<=2048 and m<=64 above that, exact at (64, 2560, 4096) in
  smoke_int8.py", so **m=128 at k=2560/4096 is outside the verified envelope**.

**Blocked deliberately.** AGENTS.md is explicit that reboots on this board come from bad NPU submits
rather than heat (`k=9216` wedged the driver, logged `RKNPU: soft reset, num: 6`) and that geometry
must be verified at a single shape before a new one is run. Testing m=128 at k=4096 is exactly that
class of change, so it should be done deliberately -- one shape in `smoke_int8.py`, with the board in
a state where a reboot costs nothing -- not appended to a long session.

Reducing K_TILE to 2048 instead does **not** help: K tiles are separate submits, so k=2560 becomes
2 K tiles x 1 submit (same 2) and only `ffn_down` improves (6 -> 5 submits per layer, ~35 ms total).

## The m=128 prefill path is closed: the recipe's guard encodes a known wedge

`npu_linear.py:688` splits a 128-token prefill into two 64-row submits because of
`row_tile = 128 if k_max <= 2048 else 64`. Since per-submit cost is a flat ~1.1 ms, halving that
split was worth ~0.35 s. It was probed at a single shape in `smoke_int8.py` -- `(128, 2560, 2560, 3)`
-- and **rejected in software before any hardware was touched**:

    ValueError: Unsupported Rockchip INT8 matmul geometry (128, 864, 2560) stride=864

The guard is `1<=m<=(128 if k<=2048 else 64)` in `_int8_matmul_body`, and the bank formula next to it
says why it exists:

    groups = ceildiv(k,512)                                            # k=2560 -> 5
    banks  = max(0x1b, 177-15*(groups-1)-15*groups*(ceildiv(m,64)-1))
                                                                        # m=128 -> 42 -> 0x1b

At m=128, k=2560 the bank count collapses to the **0x1b floor**, which is exactly the condition the
same file documents for the `k=9216` submit timeout and reboot (`RKNPU: soft reset, num: 6`). So this
is not "unverified territory that might be fine": it is the known-failure geometry by construction.
`row_tile` stays 64, and **2 submits per Linear at k>2048 is the floor for this recipe**.

That retires the last cheap lever. The remaining route is the one the reference's own profile shows:
its `rockchip_int8_64_12288_2560` is **m=64, n=12288** -- the same 64-row limit we have, but with
three Linears' packed tiles merged into **one** submit across 3 cores x 4096 columns. Our wrapper
already emits `n=12288` as its maximum width (`width = 24576 if rows==1 else 12288`), so this is a
task-assembly change, not a geometry change -- and it does not need `m>64`.

## Determinism bisection: the non-determinism predates the hand-written kernels

Two identical runs (same binary, same `--prompt 128 --gen 8 --warmups 1 --runs 1`) differ from the
very first token in every configuration tested:

| configuration | run A first tokens | run B first tokens | identical |
| --- | --- | --- | --- |
| both kernels on (current) | 149904, 120597 | 70483, 131434 | **no** |
| `ATTN_KERNEL=False` (generated sdpa) | 10286, 73044 | 150676, 7076 | **no** |
| both off (sdpa + scalar scan) | 47066, 127164 | 168879, 155489 | **no** |

With both hand-written kernels disabled the two runs **still** differ, so the non-determinism is in
the base NPU/tinygrad path and **predates this session's kernel work**. It is not caused by
`attn.c` or the SIMD scan, and the earlier id-based claims about "different fp32 paths" were never
measuring what they claimed. Still unlocated; it blocks end-to-end numerics verification, which is why
every kernel change this session was verified at the function boundary instead.

### Correction: the attention kernel is worth ~3.0 s, not 11 ms

The bisection run produced numbers that contradict two statements in this document.

Same 1-warmup protocol, only `ATTN_KERNEL` changed:

| attention path | prefill (1 warmup) |
| --- | --: |
| generated `scaled_dot_product_attention` | **8,580 ms** / 8,554 ms |
| hand-written `attn.c` | **5,529 ms** / 5,535 ms |

So sdpa costs **~3.05 s more** than the kernel, and the earlier entries are both wrong: "-11 ms"
came from comparing a 1-warmup run against a 3-warmup run (invalid), and "the whole attention
sublayer is 456 ms" came from an ablation that measured *our kernel's* cost against zero, not the
sublayer's cost against sdpa. The correct decomposition is: sdpa ~3.5 s, our kernel ~0.46 s, saving
~3.0 s. The original `r_4_128_[start_pos]` 2.48 s attribution was therefore approximately right, and
its retraction was the error.

## Where the tree stands after this turn

| state | value |
| --- | --- |
| prefill (3+3, best) | **2,771 ms** |
| prefill (restored tree, 1 trial, 3 warmups) | 2,852 ms |
| decode | 3.10 tok/s |
| prefill submits | 635 |
| row_tile at k>2048 | 64 (unchanged -- m=128 is the known-wedge geometry) |

## qkv-only fusion: kept, -37 ms. The ~200-submit target is structurally unreachable

Model-level fusion was split into its two halves and measured separately. Only the qkv half fits the
prefill width, and it pays:

| arm | submits | prefill trials | median | decode |
| --- | --: | --- | --: | --: |
| NEON baseline | 635 | 2,773 / 2,770 | **2,771 ms** | 3.10 tok/s |
| **qkv fused (8 groups)** | **603** | 2,735 / 2,734 / 2,741 | **2,735 ms** | 3.12 tok/s |

-37 ms (-1.3%), submits exactly 32 lower (8 attention layers x 4: three Linears at two 64-row
submits each becomes one Linear at two). The arms do not overlap (worst fused 2,741 < best baseline
2,770), so **2 of 2 pairs**, though not a four-run ABBA sequence and smaller than the board's +-4%
run-to-run drift. The weight and output parity of the fusion was verified bit-identical earlier.

`ffn_gate`/`ffn_up` remain **not** fused: measured +14.7% prefill (n=19456 exceeds the 12288 prefill
width so it still splits, and `prepare` packs it as a single 24576-column tile).

### Why 635 -> ~200 submits cannot be done this way

`int8_task_plan` builds every part of a submit with a **shared `lhs_dma`**:

    bodies = tuple(_int8_matmul_body(m, n, k, lhs_dma, rhs_dma + offset*k, out_dma + offset*m*4, n)
                   for offset, n in parts)

So tasks inside one submit may differ in their weight tile and output columns, but **not in their
input**. Merging several Linears into one 12288-column submit is therefore only possible for Linears
that read the *same* activation -- and there are exactly two such groups in a transformer block,
qkv and gate/up. Both have now been measured: qkv pays 37 ms, gate/up regresses 407 ms. There is no
untested configuration left, and the reference's `rockchip_int8_64_12288_2560` cannot be reproduced by
task merging across different inputs without giving each task its own lhs pointer, which is a change
to the command-buffer encoding, not to tiling.

**Submit floor with this recipe: 603 per 128-token prefill.**

## Emitted-C audit of the four largest CPU kernels: two are genuinely scalar

`profile_graph_dispatch.py` fixed (true prefill row via max `loop_s`, top-8 sources in full) and all
four sources read from `prof-src2.log`:

| kernel | time | float4 loads | divides | exp2 | scalar? |
| --- | --: | --: | --: | --: | --- |
| `E_2_16_32_32_4_4` | 0.34 s | 7 | 16 | 16 | no -- already 4-wide |
| `E_589824_2` | 0.21 s | **0** | 2 | 2 | **YES -- 1 element/iteration** |
| `E_2048_32_2_4` | 0.17 s | 4 | 8 | 8 | no |
| `r_32_16_4_32_4` | 0.16 s | 7 | 16 | 16 | no |
| `E_32_32_128_4` | 0.15 s | **0** | 4 | 4 | **YES -- 4 unrolled scalar lanes** |

`E_589824_2` is the clearest target: a flat `for (i = 0; i < 589824; i++)` computing two scalar
sigmoids, `out[i] = 1/(1+exp2poly(-x))` and `out[i+589824] = 1/(1+exp2poly(-y))`, no vector types
anywhere. 179 M sigmoids/s. A float4/NEON version should reach the 3-4x that `qz_row` achieved from
the same starting point, i.e. ~0.15 s saved.

`E_32_32_128_4` is the same story for SiLU: four unrolled scalar lanes writing
`*(((float*)(val0))+(alu2+128)) = (alu3/(1.0f+alu38))`, no float4.

Combined scalar headroom: **0.36 s, ~0.25 s recoverable.**

The other two (`E_2_16_32_32_4_4`, `E_2048_32_2_4`, `r_32_16_4_32_4`) are already 4-wide, so their
cost is the arithmetic itself -- 16 divides and 16 polynomial exps per iteration -- not codegen.
For those the only lever is a numerics-changing approximation, which cannot be validated while the
run-to-run non-determinism is unlocated.

### `E_589824_2` is now identified: the SwiGLU gate sigmoid

Instrumenting `Tensor.sigmoid` through a real 128-token prefill gives the call sites:

| calls | shape | elements | site |
| --: | --- | --: | --- |
| **32x** | **(1, 128, 9216)** | **1,179,648** | **`gated_activation`, `model.py:47`** |
| 24x | (1, 128, 8192) | 1,048,576 | attention output gate, `model.py:453` |
| 24x | (1, 128, 32, 128) | 524,288 | GDN decay |
| 24x | (1, 128, 32) | 4,096 | GDN beta |
| 8x | (1, 128, 4096) | 524,288 | -- |

So `E_589824_2` is the `(gate * alpha).sigmoid()` in `gated_activation`, 1,179,648 elements per layer
x 32 layers, issued as two 589,824-element halves per kernel call -- exactly the kernel's two-input,
two-output shape. Note it costs 0.21 s *in addition to* `E_2_16_32_32_4_4` (0.34 s), which is the
same activation path fused with dequantisation: **SwiGLU is 0.55 s of the 2.73 s prefill in total.**

**Blocker for acting on it:** the kernel is *generated* by the renderer, not hand-written here, so
vectorising it needs either a renderer-level change to elementwise emission or routing
`gated_activation` to a custom NEON kernel the way `attn_native.py` handles attention -- then a full
implement / bit-identical-verify / A-B / re-profile cycle. Not attempted: no budget remained to verify
it, and an unverified numerics change is exactly what `M37` records.

## Vectorised sigmoid: bit-identical, 1.57x faster isolated, and 2.7% SLOWER in-model

The identified scalar kernel was attacked directly. `sigmoid.c` replicates tinygrad's emitted
arithmetic lane-by-lane with float4 -- same NaN/+-inf scrubbing, same `ci>>1` two-factor exponent
split, same association order -- so it is bit-identical **by construction**, and that was proved:

    20 cases: shapes (128,9216) (1,9216) (32,256) (4,2560) (64,128) x
    (uniform, exact half-integer ties, NaN+-inf sprinkled, |x| up to 200)
    -> ALL BIT-IDENTICAL, every element

Isolated: 7.76 ms -> 4.95 ms at 1,179,648 elements, **1.57x faster** (152 -> 238 M elem/s).

In-model, `--prompt 128 --gen 64 --warmups 3 --runs 3`:

| arm | trials | median | decode |
| --- | --- | --: | --: |
| qkv fusion only | 2,735 / 2,734 / 2,741 | **2,735 ms** | 3.12 tok/s |
| + vectorised sigmoid | 2,804 / 2,811 / 2,810 | **2,810 ms** | 3.08 tok/s |

**+75 ms, +2.7%: rejected and reverted.** The isolated win does not transfer because tinygrad
*fuses* the sigmoid into the surrounding `gate * sigmoid * (up + up_bias)` chain -- that fused form is
what `E_2_16_32_32_4_4` (0.34 s) is. Replacing it with a standalone kernel adds a separate read+write
pass over 9.4 MB per layer, which costs more than the faster arithmetic saves.

This is the second time this session that an isolated speedup failed to transfer (after the attention
"252 ms/block"), and the same lesson: **fuse boundaries, not arithmetic.** A kernel can only beat the
generated code if it removes a pass, not merely if it computes the same values faster.

`sigmoid_native.py` stays in the tree marked MEASURED AND REJECTED; the model calls `.sigmoid()`
again exactly as before.

## Removing `.contiguous()` from `gated_activation`: -24 ms, at 1 ULP

The vectorised-sigmoid attempt failed because it *added* a pass. The pass existed only because of a
`.contiguous()` carrying the comment "TODO: remove the need for this contiguous" -- so the pass can
be removed instead of replaced.

| arm | trials | median | decode |
| --- | --- | --: | --: |
| with `.contiguous()` | 2,734 / 2,735 / 2,741 | **2,735 ms** | 3.12 tok/s |
| **without** | 2,710 / 2,711 / 2,711 | **2,711 ms** | **3.15 tok/s** |

-24 ms (-0.9%), decode +1.0%; arms do not overlap (worst 2,711 < best 2,734), so **1 adjacent pair,
separated**, and the within-arm spread is 0.04%.

**It is not bit-identical, and that is stated rather than hidden.** Fusion reassociates, so the last
bit changes in ~22% of elements:

    split : -0.12474042     fused : -0.12474043
    max relative difference over 20 cases (5 shapes x uniform/ties/NaN+-inf/huge): ~1e-7, i.e. 1 ULP

Accepted deliberately: the tree is already not bit-reproducible run to run (identical runs differ in
the first token), so an unverifiable 1-ULP change in one activation is immaterial, whereas the 24 ms
is measured. A bit-identical version of this change does not exist -- any fusion that removes the pass
reassociates.

Running total on prefill: 22.45 s -> 7.95 s (ImportError) -> 5.68 s (SIMD scan) -> 2.97 s (bind fix)
-> 2.77 s (NEON quantize) -> 2.735 s (qkv fusion) -> **2.711 s** (no contiguous).

## Post-change profile of the 2,711 ms tree: no materialisation boundary left

Re-profiled with the fixed profiler (`--prompt 128 --gen 64 --warmups 3`), native prefill row:

| kernel | calls | time | us/call |
| --- | --: | --: | --: |
| `qz_row` (NPU submits) | 603 | 0.790 s | 1,310 |
| `cpu_gated_delta_scan` | 24 | 0.610 s | 25,417 |
| `E_2_16_32_32_4_4` (SwiGLU, fused) | 48 | 0.350 s | 7,292 |
| `E_2048_32_2_4` | 24 | 0.170 s | 7,083 |
| `r_32_16_4_32_4` | 48 | 0.170 s | 3,542 |
| `E_32_32_128_4` | 24 | 0.150 s | 6,250 |
| `E_589824_2` | **32** | **0.150 s** | 4,688 |
| `exp2_scalar` (our attention) | 8 | 0.120 s | 15,000 |
| `E_163840_2` | 32 | 0.100 s | 3,125 |
| `E_131072_4` | 96 | 0.070 s | 729 |
| rest (4 kernels) | 120 | 0.120 s | -- |
| **sum** | | **2.80 s** | |

Two facts that close the proposed next step:

1. **`E_589824_2` fell from 0.210 s / 64 calls to 0.150 s / 32 calls** -- the `.contiguous()` removal did
   halve it and cut 60 ms of kernel time, consistent with the 24 ms measured end to end. There is no
   second copy of that pass to remove.
2. **There is no `kv.float()` conversion kernel in the profile at all**, and no `pad_to` kernel in the
   top 14. Both proposed materialisation candidates were already cheap enough not to appear; the
   attention kernel's whole-cache `.float()` is under 20 ms and invisible at this granularity.

So the CPU side is now ~1.77 s (2.56 s kernel minus 0.79 s NPU) plus bookkeeping, with **no single
removable pass left**. The only items large enough to matter are the NPU submit count (0.79 s, blocked
by the shared-`lhs` task encoding) and the scan (0.61 s, already the reference SIMD kernel).

One anomaly worth noting for a future session: `E_163840_2` runs at **52 M elements/s** (3,125 us for
163,840 elements), an order below its neighbours -- but it is only 0.100 s, so fixing it alone cannot
move the objective.

## Poison-fill test: the KV cache is not the source of the non-determinism

`extra/rockchip/det_poison.py` (previously broken -- `cache_kv` lives on the blocks, not on
`Transformer`). 8 of 32 blocks own one, shape `(2, 1, 4, 160, 256)`; all 8 are refilled with a known
value and three prefill+8-token sequences are run in one process:

| phase | cache fill | ids |
| --- | --: | --- |
| A1 | 0.0 | 248046 x8 |
| B  | 0.5 | 248046 x8 |
| A2 | 0.0 | 248046 x8 |

**The fill has no effect whatsoever.** Writing 0.5 into every cache byte, including the region beyond
`pos+T` that the model never writes, changes nothing -- which is what a kernel that reads only
`j <= pos+i` should do, and it rules the KV cache out as the source.

**Caveat, stated because it limits the conclusion:** the probe's output *collapsed* to a single
repeated token (all-ones prompt, id 1 x 128), so it has limited discriminating power -- a degenerate
output can be insensitive to inputs that a real prompt would expose. The result is "no evidence the
cache matters", not a proof. A conclusive version needs a real tokenized prompt and would compare
logits rather than the sampled id, which `Transformer.forward` does not expose (it returns the
Gumbel-max sample).

Combined with the earlier bisection (`ATTN_KERNEL=False` still non-deterministic, both hand-written
kernels off still non-deterministic), the remaining suspects are the NPU path's own state (submitted
buffers, `sync_mem` completion, domain/core masking) rather than anything in the KV cache or in our
two hand-written kernels.

## 70% of every NPU submit is fixed overhead -- submit merging is worth ~0.28 s

Two measured points at equal code, native profile rows:

| prompt | submits | NPU program time | per submit |
| --- | --: | --: | --: |
| 64 | 323 | 0.360 s | 1.115 ms |
| 128 | 603 | 0.790 s | 1.310 ms |

Solving `cost(m) = F + k*m` from those two points:

    F = 0.920 ms fixed per submit
    k = 0.195 ms per 64 rows of compute
    fixed share at m=128 = 70%

So of the **0.79 s** NPU time in a 128-token prefill, **0.55 s is fixed per-submit overhead** (two
`sync_mem` ioctls, the submit ioctl, completion latency) and only ~0.24 s is arithmetic.

Consequences:

- **Merging submissions is the single largest remaining lever: halving 603 -> ~302 saves
  ~0.28 s (10% of prefill, 2,711 -> ~2,430 ms, i.e. 1.17x behind RKLLM).**
- The count cannot be halved by tiling: `m=128` is refused by the recipe guard (banks -> 0x1b, the
  known-wedge geometry) and `k=2560` needs two K-tiles at m=128 anyway.
- It therefore requires **per-task `lhs`** in `int8_task_plan` / `_encode_task_chain`, which today build
  every part as `_int8_matmul_body(m, n, k, lhs_dma, rhs_dma + offset*k, out_dma + offset*m*4, n)`
  -- one shared `lhs_dma`. That is a command-buffer encoding change in the register/task path, the
  same path that wedged this board at k=9216, so it needs a clean tree and a board that can be lost.

**Status: sized and justified, not implemented.** The design is: add a per-part lhs offset to the part
table, thread it through `_encode_task_chain`, then merge `attn_output` (k=2560, n=2560) with a
same-k Linear reading a different activation into one 12288-column, 3-core submit, and compare the
merged output column-for-column against the current 603-submit path.

## The encoding already supports per-task lhs -- the planned register change is unnecessary

`_encode_task_chain(dev, bodies, active_cores)` (ops_rockchip.py:158) does not assume anything about
where a body reads from:

    tasks = len(bodies)
    if tasks % active_cores: raise ValueError('Unbalanced Rockchip matrix task chains')
    for i, body in enumerate(bodies):
      next_task = i+1 if (i+1)%per_core else None
      ...   # links body i to body next_task, per core

and each body comes from `_int8_matmul_body(m,n,k,lhs,rhs,dst,stride)`, so **it already embeds its own
lhs, rhs and dst addresses**. The single `lhs_dma` is a limitation of `int8_task_plan`'s *signature*:

    bodies = tuple(_int8_matmul_body(m,n,k,lhs_dma, rhs_dma+offset*k, out_dma+offset*m*4, n)
                   for offset,n in parts)

Consequences for the planned merge:

- **No change to `_encode_task_chain` and no change to any register value is required.** The risky part
  of the plan -- the path that wedged the board at k=9216 -- does not have to be touched at all.
- Merging two 3-core Linears yields `3 + 3 = 6` bodies, `6 % 3 == 0` -> 2 tasks per core, which is
  exactly the balanced chain `_encode_task_chain` requires.
- The work reduces to: a new `int8_task_plan_multi(dev, sources)` that concatenates
  `_int8_matmul_body(...)` for each source's parts and calls `_encode_task_chain(dev, bodies, 3)`,
  plus a caller that decides which same-`k` Linears to pair and gives them one output buffer with
  disjoint column ranges.

Expected gain is unchanged: 0.920 ms fixed per submit, 603 -> ~302 saves ~0.28 s.

This supersedes the earlier framing that per-task lhs "needs a command-buffer encoding change"; it is
a plan-construction change, which is materially safer.

## `int8_task_plan_multi` added; premise verified without touching hardware

Structural verification of the merge design, pure Python, no device:

| check | result |
| --- | --- |
| one 3-core Linear -> parts | `((0,864),(864,864),(1728,832))` = 3 |
| two Linears -> merged bodies | **6**, `6 % 3 == 0` -> balanced chain |
| each body carries its own lhs | True for both `A` and `B` |
| bodies for different lhs | differ in **1 word of 108** (only the address) |

So `_int8_matmul_body` bakes lhs/rhs/dst into the register program, and concatenating bodies from
different-input Linears is well-formed with no register change. `int8_task_plan_multi(dev, sources)`
now does exactly that and calls the unchanged `_encode_task_chain(dev, bodies, 3)`.

**Still to do (needs the device):** allocate one output buffer, merge two real same-k Linears, and
compare the merged output column-for-column against the 603-submit path; then A/B against 2,711 ms.
Expected ~0.28 s from halving 603 -> ~302 submits at the measured 0.920 ms fixed cost.

## The 1,263 ms target is the reference's BEST TRIAL, not its median

Read from the reference's own receipt for the 4B retime run --
`~/oNPU/analysis/release-20261115/r2-lfm-20261116/candidate-retime-20261117-qwen35-4b/receipt.json`,
`execution_status: COMPLETE`, `timing_validity: PASS`, protocol `{"warmups":1,"measured_runs":3,
"prompt_tokens":128,"output_tokens":64,"context":512,"device":"CPU+ROCKCHIP"}`, measured at
69.3-76.7 C:

| | reference | ours (`nocontig-3plus3`) |
| --- | --- | --- |
| ttft_ms trials | **9,335.78 / 19,132.28 / 1,210.63** | 2,710 / 2,711 / 2,711 |
| median | **9,335.78 ms** | **2,711 ms** |
| max/min ratio | **15.80x** | **0.04%** |
| decode tps | 6.342 / 6.376 / 6.355 (median 6.355) | 3.15 / 3.15 / 3.15 |
| native_prefill_replay_runs | 1 | -- |

**The published 1,263 ms is the last-trial / stabilised figure, not a median.** The reference's median
prefill is **9,335.8 ms** with a **15.8x** trial-to-trial spread -- the same shape this project's notes
record for its own earlier runs (67,246 / 31,933 / 31,749 ms).

So the fair statements are:

| comparison | result |
| --- | --- |
| our median vs reference median | **2.24x slower** |
| our last trial vs reference last trial | 2,710 vs 1,210.6 ms -> **2.24x slower** |
| our median vs RKLLM 2,069 ms | 1.31x slower |
| our median vs RKLLM last-trial-class | ~1.31x (RKLLM publishes one number) |

Every earlier row in this document that compares our 2,7xx ms against "the archive 1,263 ms" is
therefore comparing a **stable median against someone else's best trial**, and overstates the gap.
The correct framing: our prefill is **3.4x more stable and 3.4x faster at the median**, and **2.24x
slower at the best trial**. Closing the best-trial gap is the real remaining work.

## `int8_task_plan_multi` builds a valid merged submit on real device buffers

Executed against `Device["ROCKCHIP"]` with genuine allocations -- two activations (`lhs_a`, `lhs_b`),
two weight tiles, one output buffer with disjoint column ranges (`out`, `out + n*4`):

    single : tasks=3 core_mask=0x7 subcores=((0,0),(0,0),(0,1),(1,1),(2,1))
             parts=((0,864),(864,864),(1728,832))
    merged : tasks=6 core_mask=0x7 subcores=((0,0),(0,0),(0,2),(2,2),(4,2))
             parts per source: identical for both sources
    cmd bytes=8192  task bytes=4096

The subcore table is the evidence that the merge is laid out as designed: with 6 bodies the chain
links task i -> i+2 instead of i -> i+1, i.e. **2 tasks per core, one from each Linear**, over the same
3 cores (`core_mask=0x7`).

**What is now proven:** two same-`k` Linears that read *different* activations can be expressed as one
6-task, 3-core submit with no register or encoding change, using `out_dma` offsets to keep their
output columns disjoint.

**What is NOT done:** the submit has never been *dispatched*, so column-for-column parity against the
603-submit path is unverified, and no speed is measured. Those are the remaining steps, and dispatch is
the part that carries wedge risk -- `plan` construction does not touch the NPU, submission does.

## Device regression check after adding int8_task_plan_multi: 7/7 exact

`smoke_int8.py` on the real device, after the `ops_rockchip.py` edit:

    m=1   n=2560 k=2560 cores=1  max_abs_err=0.0000  ok
    m=1   n=2560 k=2560 cores=3  max_abs_err=0.0000  ok
    m=1   n=9216 k=2560 cores=3  max_abs_err=0.0000  ok
    m=64  n=2560 k=2560 cores=3  max_abs_err=0.0000  ok
    m=64  n=9216 k=2560 cores=3  max_abs_err=0.0000  ok
    m=64  n=2560 k=4096 cores=3  max_abs_err=0.0000  ok
    m=128 n=2560 k=2048 cores=3  max_abs_err=0.0000  ok
    (128,2560,2560) -> ValueError, as documented

So the added function does not disturb the single-submit path, and every supported geometry is still exact.

**New: `m=128` at `k=2048` is exact on this device.** `_int8_matmul_body` allows `m<=128` when
`k<=2048`, and the smoke test confirms it numerically. `m=128` is therefore not closed in general --
only at `k>2048`, where `banks` collapses to `0x1b`.

That compounds with the merge. Submits per 128-token prefill:

| configuration | submits |
| --- | --: |
| today: m=64, k-tile 4096 | **603** |
| merge two Linears per submit, m=64 | ~302 |
| merge + m=128 at k-tile 2048 | **~150** |

At 0.920 ms fixed cost per submit, ~150 submits is **~0.42 s**, not the 0.28 s projected for the merge
alone: prefill would land near **2.29 s**, ~1.11x from RKLLM's 2,069 ms.

Caveat: tiling K at 2048 costs an extra K tile for the `k=2560` linears (2048+512), so the two effects
partly cancel -- the ~150 figure assumes the merge is applied on top, and must be measured, not
inferred.

## The merged-submit address recipe (read off `_FusedInt8Plan`, not guessed)

`npu_linear.py:519` builds the single-Linear plan as:

    header = round_up(m * 4, 64)
    int8_task_plan(dev, m, n, k, cores, self.out.dma, self.payload.dma + header, weight_dma + column_offset*k)

so `lhs_dma` is the payload **past its fp32 scale header**, `rhs_dma` already includes the column
offset, and `out_dma` is the int32 plane buffer. The merged form follows directly:

    int8_task_plan_multi(dev, (
        (m, n, k, payload_a.dma + header, weight_a.dma, out.dma),
        (m, n, k, payload_b.dma + header, weight_b.dma, out.dma + n*4)))

with one `(m, 2n)` plane buffer, two input `sync_mem` calls (payload_a, payload_b), one submit and one
output `sync_mem`.

What remains before dispatch, and nothing else:
1. a submit-only C function (the ~25-line sync/submit/sync body of `_fused_source`, without the
   quantize and dequant stages), since activations are pre-quantised via `int8_payload` and weights via
   `pack_weight_int8`;
2. dequantise the int32 planes as `out * a_scale[m] * w_scale[n]` in numpy;
3. assert `merged[:, :n] == separate_a` and `merged[:, n:] == separate_b` bitwise;
4. A/B against 2,711 ms and re-profile for `qz_row` 0.790 s -> ~0.5 s.

Steps 1-2 were NOT written: the first dispatch of a 6-task heterogeneous chain is the operation that has
already wedged this board, and there was no remaining budget to debug a failed submit or a parity
mismatch. The tree is clean at 47217d41f, so this can be attempted without risk to the session's work.

## Bug caught before dispatch: `int8_task_plan_multi` had the bodies in the wrong order

`_encode_task_chain` links `next_task = i+1 if (i+1)%per_core else None`, so with `per_core = tasks/3`
the chain runs task i -> i+1. That means bodies must be grouped **by core**, not by source.

The first version appended per source -- `[A0,A1,A2,B0,B1,B2]` -- which under `per_core=2` chains
`A0 -> A1`: two *column ranges of the same core*, i.e. a completely different task sequence from the
intended "core 0 runs Linear A then Linear B". The subcores table looks identical either way
(`((0,0),(0,0),(0,2),(2,2),(4,2))`), because it depends on the core count and mask rather than on body
order, so this would not have been visible in the plan listing -- it would have produced wrong output
at best, or a bad register sequence at worst.

Fixed by building `per_source[core]` and interleaving:

    bodies = [per_source[src][core] for core in range(3) for src in range(len(per_source))]

giving `[A0,B0,A1,B1,A2,B2]`: core 0 chains A0 -> B0, core 1 A1 -> B1, core 2 A2 -> B2. Plan still
verifies at 6 tasks / `core_mask=0x7` / `per_core=2`.

**Note for whoever dispatches:** `per_core=2` has never executed in this tree before. Every existing
path uses `tasks==cores` (`per_core=1`), so `per_core = 1` makes `next_task` always `None` -- the
chaining tail in `_encode_task_chain` is *dead code* today. The merged submit is the first thing to
exercise it, which is a second, independent reason to dispatch it carefully.

## `per_core=2` chaining verified by decoding the command buffer (no dispatch)

The merged chain was built at the safest first geometry -- `m=1, n=2560, k=2048` (smoke-exact, and
`banks` is far from its floor at groups=4) -- and the command buffer decoded rather than submitted:

    tasks 6  per_core 2  core_mask 0x7
    PC_BASE_ADDRESS writes: 3
    chain targets (register holds addr>>4): 0x5f7380, 0x5f7a80, 0x5f8180
    expected body starts: 0xff5f7000 0xff5f7380 0xff5f7700 0xff5f7a80 0xff5f7e00 0xff5f8180

The three chain writes land on bodies **1, 3 and 5**, i.e. `A0->B0`, `A1->B1`, `A2->B2`, and only three
of the six tasks carry a link (the other three terminate). That is exactly the intended per-core
layout, and it is the **first time `per_core = 2` chaining has been exercised at all** -- every existing
path has `tasks == cores`, which makes `next_task` always `None`, so the chaining tail of
`_encode_task_chain` was dead code until now.

So before any dispatch: the body ordering is right (fixed in d167e2a5c), and the chaining that the
ordering depends on is encoded correctly. Those were the two things that could go silently wrong and
produce wrong output rather than an error.

**Still not dispatched.** The submit-only C program, the two input syncs, the `out*a_scale*w_scale`
dequant, and the column-for-column parity assertion are unwritten.

## The exact patch left to write (multi-source descriptor)

Read off the working path, not inferred:

1. **`npu_linear.py:179` `_matmul_int8`** -- today emits
   `struct.pack("<8I", m, n, k, cores, column_offset*k, round_up(m*4,64), 0, 1)` with `ins=(1,2)`.
   A multi variant appends `N` sources of `<5I` = `(n, k, lhs_offset, rhs_offset, out_offset)` and
   sets `ins=tuple(range(1, 2+2N))`, with `lhs`/`rhs` device tensors as arguments 1..2N.
2. **`ops_rockchip.py:268` `RockchipMatmul.__init__`** -- parses `obj.lib[:16]` as `<4I` (m,n,k,cores)
   then `rhs/lhs/split_k/int8` from 16..32. Add: if `len(obj.lib) > 32`, parse
   `self.sources = tuple(struct.unpack('<5I', obj.lib[32+20*i:52+20*i]) for i in range((len(obj.lib)-32)//20))`
   and skip `_matmul_parts` for the single-source case.
3. **`ops_rockchip.py:295` `prepare(self, out, lhs, rhs)`** -- builds the bodies **inline** (it does
   not call `int8_task_plan`):
   `bodies = tuple(_int8_matmul_body(self.m, n, self.k, lhs.dma+self.lhs_offset,
    rhs.dma+self.rhs_offset+offset*self.k*self.itemsize, out.dma+offset*self.m*4, n)
    for offset, n in parts)`
   Change the signature to `prepare(self, out, *ins)`; when `self.sources` is set, iterate
   `zip(self.sources, pairs(ins))` and extend `bodies` per source, using `out.dma+out_offset`.
   The two buffer-size checks must be applied per source.
4. **`ops_rockchip.py:318` `__call__(self, *args, ...)`** -- currently `out, lhs, rhs = args` and later
   uses `lhs`/`rhs` for `update_inputs` and the weight-version check. Change to `out, *ins = args`,
   keep the weight check on the last rhs (weights are immutable after packing), and compare
   `(out,)+tuple(ins)` against `self.bufs`.

`_encode_task_chain` needs no change -- it takes arbitrary bodies, which is why the plan-level merge
already works. Dispatch target: `m=1, n=2560, k=2048` (smoke-exact; `banks` far from `0x1b`).
Assert `merged[:, :n] == separate_a` and `merged[:, n:] == separate_b` bitwise against
`packed_matmul_int8` before any timing.

**Not written.** Three probe-level attempts failed on `AssertionError` at `device.py:139` because a
submit cannot be injected outside this descriptor path; the change has to be made here, in four places,
and then dispatched once -- which is the step that can reboot the board and needs an untruncated
session.

## The merged submit now DISPATCHES (parity still wrong)

The four-site multi-source descriptor is applied in `ops_rockchip.py` (additive: the branch is inert
unless a descriptor longer than 32 bytes is present), and the single-source path is unaffected --
`smoke_int8.py` is still 7/7 exact after the patch.

First successful dispatch of a merged chain, `m=1, n=2560, k=2048`, two Linears with different
activations, one 6-task 3-core submit writing to one (m, 2n) int32 plane buffer:

    refs computed 0.20156772434711456 -0.3938932716846466
    PARITY A: False  maxdiff 0.8058053255081177
    PARITY B: False  maxdiff 0.703873872756958

**No AssertionError, no timeout, no reboot** -- the mechanism works and the board accepted a 6-task
chain with `per_core=2` chaining, which had never executed before. What is wrong is a layout detail,
and `maxdiff ~0.8` on values of order ~0.4 says the planes are being read from the wrong offsets
rather than the arithmetic being wrong.

Most likely cause, for the next attempt: the plane regions of the two sources. For one source, the
three cores write `(m, part)` planes at byte offset `offset*m*4` inside an `m*n` int32 buffer, and
`_dequantize(..., parts, column_offset)` scatters them. For two sources sharing a `(m, 2n)` buffer the
second source's plane base must be source a's *whole* region, i.e. `n*4` bytes only if a's planes are
packed as `(m, n)`; if they are `(3, m, part)` stacked, the base is `3*m*4`. That distinction has not
been verified and is the first thing to check, by dumping the raw int32 planes for a single source and
confirming their layout in memory before assembling the merged buffer.

## Merged submit: output IS written and deterministic; parity fault is in the inputs

The sentinel pre-fill is not available here: `ROCKCHIP` has no renderer, so a fill can only arrive via
`.to(DEV)`, and copy-then-`custom_kernel` fails `UOp` verification
(`Ops.AFTER` over `COPY`+`CALL`). Replaced with a stronger test -- run the merged submit twice in one
process:

    RUN TWICE identical: True | differing ints: 0 of 5120
    r1[:6] [-31492, 125348, 166376, 348943, -436117, -98846]
    PARITY A: False maxdiff 0.8058053255081177
    PARITY B: False maxdiff 0.703873872756958

**The buffer is written, deterministically.** So the "output never written / stale empty" hypothesis is
refuted, and the fault is in what the merged tasks compute or where they read it -- not in whether the
submit happened.

**Bisection to run next (one line):** build the *same* multi descriptor with `N=1` -- a single source
through the new multi path -- and compare against `packed_matmul_int8`. If N=1 also fails, the bug is in
the multi branch's own addressing (`lhs_off`/`rhs_off` handling in `prepare`); if N=1 passes, the bug is
specific to the second source's `out_offset` or to which `(lhs, rhs)` pair the second task group binds.

## The N=1 bisection WEDGED the NPU -- the multi branch can emit an invalid submit

Running the multi descriptor with a **single** source at `m=1, n=2560, k=2048` did not return wrong
data; it hung the job:

    RKNPU: failed to wait job, task counter: 0, flags: 0x5, elapsed time: 6104735us
    RKNPU: job timeout, flags: 0x0  (core 0/1/2 irq status 0x0, task counter 0x0)
    RKNPU: soft reset, num: 6

The driver recovered (lock free, no orphaned processes, `smoke_int8.py` 7/7 exact afterwards), but this
is the documented wedge signature and it came from my own probe.

**What the asymmetry tells us:** the *same* branch with **N=2** ran to completion, deterministically, twice
(`0 of 5,120 ints` differing); with **N=1** it produces a submit the NPU never completes. So the multi
path is not merely computing the wrong thing -- it can encode an *invalid task program*, and the fault
depends on the source count. That is a register/task-encoding problem in the new branch, not a dequant
or offset problem, and it means **the merged path must not be dispatched again until the emitted task
program for N=1 is decoded and compared against the single-source path**, the way `per_core=2` chaining
was verified by decoding `PC_BASE_ADDRESS` writes.

Do not run this probe again on this board without that offline check first.

## Root cause of the merged-submit failure: the part width, found by word-diffing programs

Decoding both emitted command buffers offline (no submits), with **shared** buffers so only encoding
differences remain, showed 18 differing words for N=1. They are readable:

    reg 0x1030   0x1b   vs 0x50
    reg 0x1038   0x361  vs 0xa01      0x361 = 865,  0xa01 = 2561
    reg 0x3018   0x35f  vs 0x9ff      0x35f = 863,  0x9ff = 2559
    reg 0x4038   0xd7   vs 0x27f

The multi branch encoded **n = 2560 (full width)** where the single-source path encodes the **part
width** (864/832). The single path gets this by shadowing:

    bodies = tuple(_int8_matmul_body(self.m, n, self.k, ..., n) for offset, n in parts)

where the loop rebinds `n` to the part width, and `_int8_matmul_body` requires `stride == n`. The multi
branch passed `n_i` (the full width) in both positions.

**That one error explains both symptoms at once:** at N=2 it produced wrong values (wrong channel
counts per task), and at N=1 it produced a submit the NPU never completed -- job timeout,
`soft reset, num: 6` -- because the encoded task was invalid rather than merely wrong.

**Fix applied** -- pass the part width `w` as both the width and the stride. Verified offline, no
dispatch:

    single vs multi1 identical: True | differing 0 of 336

The N=1 multi-source descriptor now emits a command buffer **byte-identical** to the working
single-source path. Parity and timing remain unmeasured; the next step is the N=2 column check, which
was previously producing `maxdiff 0.8` and should now be exact.

## MERGED SUBMIT: column parity HOLDS bitwise

After the part-width fix (`a4da75812`), the N=2 merged chain at `m=1, n=2560, k=2048` -- two Linears
reading different activations, one `(m, 2n)` plane buffer, one 6-task 3-core submit:

    RUN TWICE identical: True | differing ints: 0 of 5120
    PARITY A: True maxdiff 0.0
    PARITY B: True maxdiff 0.0

So `merged[:, :n] == separate_a` and `merged[:, n:] == separate_b` **exactly**, against
`packed_matmul_int8`. Combined with the earlier offline check that the N=1 program is byte-identical to
the single-source program (0 of 336 words), the merged submit is now correct by both routes: the
encoding matches for one source, and the output matches for two.

Established so far, in order: plan construction, body ordering by core, `per_core=2` chaining
(three `PC_BASE_ADDRESS` writes onto bodies 1/3/5), dispatch (no fault, board healthy), determinism
(0 of 5,120 ints across runs), and now bitwise column parity.

**Not yet done:** the lever is correct but not *used*. The model still issues 603 single-source submits;
nothing yet pairs two of its Linears into a merged submit. So there is no 3+3 A/B against 2,711 ms and
`qz_row` has not been shown to fall from 0.790 s. The next step is integration, not verification:
choose a pair of same-`k` Linears per block whose activations are both live at the same point
(e.g. `attn_output` and a second `k=2560` projection), emit the multi descriptor from
`packed_matmul_int8`'s caller, and then measure with `--warmups 3 --runs 3` on both arms.

## Corrected projection for the merge: ~60 ms for the clean pair, not 0.42 s

The 0.42 s figure assumed halving all 603 submits. That requires merging across **different**
activations, which reorders the graph. The merge that is clean -- same activation, same k, both live at
the same point, which is `ffn_gate` + `ffn_up` -- is bounded much lower, because at `m=128` each Linear
is already split into two submits by `row_tile=64`:

| configuration | submits per layer | total submits | at 0.920 ms |
| --- | --: | --: | --: |
| today | 4 (gate 2 + up 2) | 603 | 0.79 s |
| merged gate+up | 2 | ~539 | -59 ms |

So the clean merge is worth **~59 ms plus ~9 ms** of shared activation quantisation, i.e. **~60-70 ms**,
about 2.5% of prefill. Reaching 0.42 s needs pairing Linears with *different* activations, which the
descriptor supports (each source carries its own `lhs`) but which changes block evaluation order.

Integration is also not a caller tweak: the model's Linears are `NPULinear` on the fused C path
(`_FusedInt8Plan`), not `packed_matmul_int8`, so a merged path needs a new module (an `NPULinearPair`)
plus model wiring and a fallback -- roughly 60-80 lines -- before any A/B is possible.

**Status: correct and unused.** No integration has been written, so no A/B exists and `qz_row` has not
been shown to fall from 0.790 s.

## Integration finding: the model's Linears are on the OTHER submit path

The verified multi-source descriptor lives in `RockchipMatmul` (the descriptor path that
`packed_matmul_int8` uses). The model does not use that path -- its Linears are `NPULinear`, which
submits through `_FusedInt8Plan`: one generated C function per geometry that quantises the activation,
issues both ioctls itself, and dequantises in the same call.

So an `NPULinearPair` cannot simply reuse `int8_task_plan_multi`'s verified plan; it needs the fused-C
analogue:

1. a `_fused_source_pair(key_a, key_b)` that quantises **the shared activation once** (the reason the
   gate+up pair also saves ~9 ms) and emits one submit carrying both sources' task groups;
2. `int8_task_plan_multi` supplies the chain (already proven: 6 tasks, `core_mask 0x7`,
   `per_core=2`, bitwise column parity `maxdiff 0.0`);
3. two dequant stages writing `gate` and `up` planes, `out_off` of the second at `n*4`;
4. model wiring in `FFNBlock.__call__` behind `hasattr(self,'ffn_gateup_pair')` with the existing
   `gated_activation` untouched, plus the `replace_linears`/`prepare_linears` post-pass that builds the
   pair from two already-wrapped `NPULinear`s.

That is the remaining work: roughly 100 lines of generated C plus Python wiring, and then adjacent 3+3
arms to resolve a ~60 ms (2.2%) effect against the board's +-4% run-to-run drift.

**Not written.** The lever is correct and unused; the reason it is still unused is this path
mismatch, not the descriptor.

## The best merge is not gate+up -- it is the two m-tiles of every Linear

The ~60 ms gate+up projection was based on pairing two *different* Linears. That is the wrong pair.

At `m=128`, `row_tile = (128 if k_max <= 2048 else 64)` splits **every** Linear into two submits --
rows 0..63 and rows 64..127 of the *same* activation. Those two halves are already:

- reading the same tensor (different `row_offset`), so nothing new is live;
- writing disjoint output rows, so `out_dma` differs only by `64*n`;
- sharing one weight tile -- so no packing change, no reordering of the graph, no new pairing rules.

Merging them as two sources of one submit is the same mechanism that already reaches bitwise parity for
two different activations, and it halves the submit count **globally**:

| configuration | submits | at 0.920 ms fixed cost |
| --- | --: | --: |
| today (2 per Linear) | 603 | 0.79 s |
| both m-tiles in one submit | **~302** | -0.28 s |

That is the 0.28 s that the cross-activation merge was supposed to buy, obtained without reordering any
block and without a new module pairing rule -- just `out_off = 64*n` and `lhs_off = 64*row_stride` on the
second source, inside the existing fused-C plan.

It also explains why the fused-C pair plan is still required: the model's `NPULinear` submits through
`_FusedInt8Plan`, so this is a change to the generated C (one sync, one submit with 6 tasks, one output
sync, two dequant row ranges) plus `int8_task_plan_multi` for the chain -- the same ~100-line shape as the
gate+up pair but with a ~5x larger prize.

## The m-tile merge needs one new key field: m for quantise/dequant vs m per task

Working through `_FusedInt8Key` for the merged form, the blocking detail is that `m` means two
different things and the current key has only one:

- the **quantise pass** covers all rows it will submit, so for a merged plan it is 128 (both tiles);
- each **task group** is still `m=64` rows, because `_int8_matmul_body` refuses `m>64` at `k>2048`
  (banks -> 0x1b, the known-wedge geometry).

`_FusedKey.m` currently serves both. So the merged key needs `m_rows` (quantise/dequant extent, 128)
alongside `m` (per-task, 64), and `_fused_source` must emit:

    quantise over m_rows                                  (payload header = round_up(m_rows*4, 64))
    one sync_mem(payload), one submit with tasks = 6       (subcores from int8_task_plan_multi)
    planes loop over (row_group, column_part)              (two row ranges: 0 and 64*row_stride)
    one sync_mem(out)

with source offsets `lhs_off = 64*row_stride` and `out_off = 64*n` for the second group. That is the
whole change; everything else (chain construction, parity, ordering, part-width) is already proven.

Recorded because it is the one non-obvious design constraint left: without splitting `m` into
`m_rows` and per-task `m`, the merged C would either exceed the m=64 limit or quantise only half the
rows.

## packed_matmul_int8_multi added and verified: the merge needs no fused-C surgery

The descriptor path carries the merge, so the planned _FusedInt8Plan C rewrite is not needed to
produce one. npu_linear.packed_matmul_int8_multi(xa, ca, sa, xb, cb, sb, columns, device) emits the
2-source descriptor and returns both dequantised outputs.

Bitwise vs two packed_matmul_int8 calls:

    m=1   k=2560  A True maxdiff 0 | B True maxdiff 0
    m=64  k=2560  A True maxdiff 0 | B True maxdiff 0

k=9216 raises Unsupported Rockchip INT8 matmul geometry -- that depth must be K-split to <=4096, as
ffn_down already is; documented wall, not a defect.

Layout trap: the second source plane region starts at m*n*4 bytes, not n*4. Each source occupies m*n
int32 (three stacked (m, part) planes) and the two coincide only at m==1, so an m=1-only check passes
while m=64 fails with maxdiff ~1.2. Splitting must be by plane region (flat[:m*n], flat[m*n:]), not by
column.

Open trade-off: the fused C exists because the descriptor path costs ~1.55 ms of host time over six
launches per Linear versus one call. The merge removes ~half the submits but may add host overhead; the
net needs a 3+3 A/B of one block paired Linears through both paths.

## The descriptor route for the merge is REFUTED by measurement

Isolated timing, same path both arms, median of 20:

    m=64 k=2560: two packed_matmul_int8 132.003 ms | one packed_matmul_int8_multi 135.660 ms | +3.657 ms (+2.8%)
    m=1  k=2560: two packed_matmul_int8 138.318 ms | one packed_matmul_int8_multi 132.351 ms | -5.967 ms (-4.3%)

Two conclusions, the first decisive:

1. The descriptor path costs ~66 ms per Linear in isolation against ~1.3 ms for the fused path in-model
   (0.79 s / 603 submits) -- a ~50x gap. That is precisely why _FusedInt8Plan exists (one CALL instead
   of six tinygrad launches). So packed_matmul_int8_multi is bitwise correct but CANNOT be wired into
   the model: it would be a large regression.
2. The +2.8% / -4.3% split is not trustworthy: both arms are dominated by the .numpy() sync in the
   isolated loop, which the model never pays per op. This is the same isolation error made twice before
   in this session (attention 252 ms/block; vectorised sigmoid +1.57x isolated -> +75 ms in-model).

Therefore the m-tile merge must be implemented in the fused C after all -- m_rows (quantise extent 128)
separate from per-task m=64, one sync, one 6-task submit via int8_task_plan_multi, one output sync,
two dequant row ranges. packed_matmul_int8_multi stays as a correctness reference for the chain, not as
a shipping path.

## Scan chunking REFUTED: the reference's 96 calls/prefill is not an available speedup

The reference invokes `cpu_gated_delta_scan` 96x per prefill against our 24x, which suggested a 4x
finer chunking might be faster. Measured (B=1,H=32,T=128,K=V=128, median of 5, standalone):

| chunking | time | vs one call | reproduces one call |
| --- | --: | --: | --- |
| 1 x T=128 | **42.04 ms** | -- | reference |
| 2 x T=64 | 56.13 ms | **+33% slower** | no, maxdiff 1.10 |
| 4 x T=32 | 74.76 ms | **+78% slower** | no, maxdiff 0.78 |

So chunking is both slower *and* numerically different -- the state carried across calls does not
reproduce the single-call result, so it is not even an equivalent configuration. The reference's call
count must reflect a different decomposition (or per-layer geometry), not a faster one, and it does not
transfer.

Incidentally the standalone call is 42 ms against 25.4 ms in-model, i.e. in-model is *faster* here --
another case where isolation is not representative.

**Consequence:** the 0.61 s scan item has no cheap lever left. Beating it needs a better algorithm than
the reference kernel we already ported, not a different invocation.

## MEASURED: the reference's native prefill is 1,201.4 ms (not an inference)

`docs/rknpu/results/ref-profile/report.json` is a reference run with **`warmups: 2`, 7 trials**. Its
`first_token_ms` per trial:

    17219.1  9236.6  26274.4  1200.4  1201.4  1201.2  1203.1

After three calls the reference sits at **1200-1203 ms for four consecutive trials** -- 0.2% spread.
So:

| | value |
| --- | --- |
| reference native prefill median | **1,201.4 ms** |
| min / max (native) | 1200.4 / 1203.1 ms |
| ours (baseline-recheck, 3+3) | **2,703 ms** |
| ratio | **2.25x slower** |
| reference native decode | 6.52-6.55 tok/s (trials 3-7) |

This **supersedes** the earlier caveat that "the ~1,210 ms like-for-like median is an inference from a
single native trial": the target is measured, tight, and reproducible. The 1.2 s goal is therefore not
a best-case artefact -- it is what the reference does on every native call.

**Consequence for the plan:** the 1,502 ms gap must come from the items already measured --
0.79 s NPU submits (0.55 s fixed cost), 0.61 s scan, 0.61 s attention-adjacent, 0.35 s SwiGLU, 0.35 s
elementwise, 0.22 s bookkeeping. The m-tile merge addresses 0.28 s of it; the remaining ~1.2 s needs
the CPU-side work that has not been started.

## Decomposition of the 1,200 ms target (metal_kernels style): two different problems

`ref-profile/report.json` native trial carries `prefill_submits`, `decode_submits`, `prefill_tps`,
`decode_tps` -- the fields needed for a proper decomposition. Ours come from `baseline-recheck`.

| metric | reference | ours | ratio |
| --- | --: | --: | --: |
| prefill | **1,200.4 ms** | 2,703 ms | 2.25x |
| prefill tok/s | 106.6 | 47.3 | 2.25x |
| **prefill submits** | **475** | **603** | **1.27x** |
| prefill per-submit (incl. CPU) | **2.53 ms** | 4.48 ms | 1.77x |
| decode | 6.52 tok/s | 3.16 tok/s | 2.06x |
| decode ms/token | 153.4 | 316 | 2.06x |
| **decode submits/token** | **22.2** | **323** | **14.6x** |
| decode per-submit | **6.9 ms** | 0.98 ms | 7.0x |

**The reference does not win prefill by fusing submits.** 475 vs 603 is 1.27x -- precisely the ~0.28 s
the m-tile merge is worth. It wins because each submit carries far more work: 2.53 ms/submit against our
4.48 ms, and in decode 6.9 ms/submit against our 0.98 ms.

**So the two gaps are different problems:**

- **prefill** is work-bound: comparable submit counts, but each of ours costs 1.77x more wall time. The
  merge addresses the 1.27x; the remaining 1.77x is the CPU-side kernel time inside those submits.
- **decode** is dispatch-bound: **14.6x more submits per token**, each only 0.98 ms, i.e. tiny. Our NPU
  time is already 143.5 ms/token against the reference's 138.66 (device at parity), so the decode gap is
  almost entirely 323 small launches per token versus their 22 large ones.

Consequence for the goal: the m-tile merge is worth ~0.28 s of the 1,502 ms prefill gap. Closing the rest
requires making each prefill submit do more useful work per millisecond (CPU-side kernels), and decode
requires ~15x submit reduction -- a different, larger change than any prefill lever.

## Full decomposition: the reference coalesces decode programs ~9:1, we do 1:1

`ref-profile/report.json` also carries `captured_calls`, per-device program counts for both phases:

| | reference | ours (prof-2711) |
| --- | --: | --: |
| prefill programs | **2,072** (1,597 CPU + 475 NPU) | 1,359 |
| prefill NPU submits | **475** (1:1 with NPU programs) | **603** |
| prefill ms/submit | 2.53 | 4.48 |
| decode programs | **1,289** (1,086 CPU + 203 NPU) | 1,032 |
| decode NPU submits/token | **22.2** | **323** |
| decode programs per submit | **~9:1** | **1:1** |
| decode ms/submit | 6.9 | 0.98 |

**The reference runs MORE programs than we do in prefill (2,072 vs 1,359) and is still 2.25x faster.**
So program count is not the discriminator, and neither is prefill submit count (1.27x). What differs is:

1. **prefill**: each of our submits costs 4.48 ms against their 2.53 ms -- work-bound, i.e. CPU-side kernel
   time inside the submit, not launch count.
2. **decode**: they coalesce ~9 NPU programs per submit; we issue 1 per program, 14.6x more submits per
   token. At our measured ~0.56 ms/submit, 323 submits ~= 181 ms of the 316 ms/token. Coalescing 9:1
   would give ~36 submits -> ~156 ms/token -> **~6.4 tok/s, i.e. parity with the reference's 6.52**.

**So the submit-coalescing machinery belongs in decode, not prefill.** `int8_task_plan_multi` -- already
verified (bitwise `maxdiff 0.0`, N=1 byte-identical, `per_core=2` chaining) -- is worth ~0.28 s in prefill
but roughly a **doubling of decode throughput**, which is the bigger prize and the same mechanism.

## m-tile merge LANDED: 603 -> 355 submits, 2,703 -> 2,614 ms, bitwise identical

Four bugs had to be fixed in sequence, each found by a different failure:

1. `_fused_plan` mapped `m==128` at `k>2048` to `m_task=64, m_rows=128`, but `m_rows` was **not passed
   to `_FusedKey`**, so the generated C quantised only 64 rows and emitted one dequant region -> rows
   64-127 came out **zero** (`rel(merged,ref) 9.6e-01` vs `rel(sep,ref) 5.4e-03`).
2. `int8_task_plan_multi` passed the **full n** as the body width/stride instead of each part's width,
   which made the fused qkv (n=8192) illegal: `n<=4096 when m>1`.
3. The merge must be limited to shapes whose `_matmul_parts` really returns 3 parts: narrow `n`
   (e.g. `ssm_beta`, n=32) collapses to one part and indexed out of range.
4. `m_rows` had to be passed by keyword to `_FusedInt8Plan`, or the positional order silently dropped it.

Verified:

    plans: [(64, m_rows=128, tasks=6), (64, 0, 3)]
    rel(merged,ref) 5.448e-03  |  rel(sep,ref) 5.448e-03
    BITWISE merged == sep: True

Measured (`docs/rknpu/results/merge-correct-3plus3/`, 3+3, complete:true):

| | baseline | merged |
| --- | --: | --: |
| submits | 603 | **355** |
| trials | 2703 / 2701 / 2710 | **2614 / 2612 / 2614** |
| median | 2,703 ms | **2,614 ms** |
| decode | 3.16 tok/s | 3.15 tok/s |

**-89 ms (-3.3%)**, bitwise identical. Note the 2,399 ms seen mid-debug was with the zero-row bug and is
**discarded** -- it was faster and wrong.

The projected -0.28 s assumed the full 0.920 ms fixed cost would be recovered per merged submit; the
measured recovery is ~0.36 ms (89 ms / 248 fewer submits), because a merged submit still performs both
tiles' work and carries both planes through one sync.

## Post-merge profile (2,614 ms tree) and how far it still is

`docs/rknpu/results/prof-merged`: prefill row loop 2.89 s, kernel 2.51 s, bookkeeping 0.38 s, 1,031 programs.

| kernel | calls | time | note |
| --- | --: | --: | --- |
| `qz_row` (NPU) | 355 | **0.720 s** | 2.03 ms/submit -- merged submits carry more work |
| `cpu_gated_delta_scan` | 24 | **0.610 s** | the reference's own SIMD kernel |
| `E_2_16_32_32_4_4` (SwiGLU) | 48 | 0.360 s | already 4-wide |
| `E_1024_32_4_4` | 24 | 0.180 s | already 4-wide (9 float4 loads, 16 div/exp) |
| `E_32_32_128_4` | 24 | 0.160 s | |
| `r_32_16_4_32_4` | 48 | 0.160 s | already 4-wide |
| **`E_393216_3`** | 32 | **0.140 s** | **SCALAR: flat 393,216-loop, 0 float4 loads, ~90 M elem/s** |
| `exp2_scalar` | 8 | 0.120 s | our attention kernel |

`E_393216_3` is three sigmoids over `H*T*32` per GDN block (32 calls), the clearest remaining
vectorisation target at 0.14 s.

**But the arithmetic has to be faced:** the kernel set sums to **2.51 s** and the target is 1.20 s, so
~1.4 s must go -- roughly a halving of *every* kernel. The reference does it with **more** programs than
us (2,072 vs 1,031), i.e. its kernels are individually faster, not fewer. Single 0.1 s wins do not
close 1.4 s; only broad per-kernel efficiency does.

## float4 SwiGLU: bit-identical, and +52 ms in-model -- rejected

`E_393216_3` (0.14 s, 32 calls) turned out to be **elementwise**, not strided: the 3-lane form is an
unrolling artifact and each index computes `out[j] = (x/(1+exp(-x))) * y` = `silu(gate)*up`, i.e. all of
`gated_activation` when alpha==1 and up_bias==0.

So a flat float4 kernel is possible, and it was written and proven **bit-identical** to tinygrad's own
output on uniform and NaN/inf data (`maxdiff 0`, `ALL IDENTICAL`), replacing the WHOLE expression rather
than a sub-expression -- which was the fix for the earlier sigmoid failure (+75 ms, added a pass).

Measured (`docs/rknpu/results/swiglu-3plus3`, 3+3, complete:true):

| arm | trials | median | decode |
| --- | --- | --: | --: |
| merged baseline | 2614 / 2612 / 2614 | **2,614 ms** | 3.15 tok/s |
| + float4 SwiGLU | 2665 / 2666 / 2670 | **2,666 ms** | 3.13 tok/s |

**+52 ms (+2.0%): rejected and reverted.** So the "fuse the boundary" rule was necessary but not
sufficient: even a like-for-like replacement of an entire fused expression with verified-identical
arithmetic loses, because the `custom_kernel` launch + fresh output buffer costs more than the ~0.09 s
the faster arithmetic could save on a 0.14 s kernel.

**This is now the third custom-kernel rejection (sigmoid +75 ms, attention +/-0 net, SwiGLU +52 ms)
against two successes (SIMD scan 4.38x, NEON qz_row 3.02x).** The pattern: a custom kernel wins only
when the generated code is *much* further from the hardware (scalar with a long loop body), and loses
when the generated kernel is already short-bodied or small, because the launch/buffer overhead dominates.

## MEASURED reference CPU/NPU decomposition -- and we are serial where they are 4.26 cores busy

`docs/rknpu/results/ref-phase2` (phase probe wrapping their own config; native trials reproduced at
**1,200.09 / 1,198.62 ms**, median 1,200.09, decode 6.486 tok/s):

| phase | idx | wall_ms | main CPU ms | total CPU ms | cores busy | off-CPU ms | D% | S% |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| prefill | 3 (native) | **1,206** | **720** | 5,140 | **4.26** | 486 | 18% | 0% |
| decode | 3 (native) | 1,090 | 180 | 4,590 | 4.21 | 910 | 36% | 0% |

Wall wait channels: `rknpu_submit_ioctl` -- the blocking ioctl, consistent with 486 ms off-CPU.

**The mechanism, in our own copy of their kernel** (`recurrent.c`, last function):

    int cpu_gated_delta_scan_pooled(..., scan_parallel_fn parallel, int *dispatch_status) {
      if(!parallel||batch<1||heads<1||(size_t)batch*heads<4)
        return cpu_gated_delta_scan(...);                  /* serial fallback */
      ... int error=parallel(scan_pool_part,args,0);       /* 4 head-groups */

Our `scan_native` calls `cpu_gated_delta_scan(..., NULL)` -- **the parallel hook is NULL, so every scan
runs on one core.** Same for the other CPU kernels: their env sets `ONPU_CPU_POOL=1`,
`ONPU_CPU_PREFILL_POOL=1`, `cpu_prefill_pool_requested: true`, which we do not have.

**So the prefill gap is parallelism, not codegen quality.** Ours: ~1,790 ms of CPU work on one core +
~720 ms blocked = 2,614 ms. Theirs: 720 ms of main-thread CPU across 4.26 cores + 486 ms blocked =
1,206 ms. Spreading our CPU kernels over ~4 cores projects ~450 ms + 720 ms = **~1,170 ms, i.e. 1.2 s**.

This supersedes the earlier "all kernels are ~2.5x slower" conclusion, which was inferred from a
**rescaled** archive profile rather than measured.

## Parallel scan via pthreads: correct, and +100 ms -- rejected

Implemented `cpu_gated_delta_scan_pooled` with a real `scan_parallel_fn` (pthread_create/join, 4 head
groups) -- the hook our wrapper was passing NULL for. It compiles (pthread declared extern, since the
`-nostdlib` build cannot see system headers) and stays exact (`4.209e-07` vs fp64, unchanged).

Measured (`docs/rknpu/results/parscan-3plus3`, 3+3, complete:true):

| arm | trials | median | decode |
| --- | --- | --: | --: |
| serial (baseline) | 2614 / 2612 / 2614 | **2,614 ms** | 3.15 tok/s |
| parallel scan | 2714 / 2724 / 2701 | **2,714 ms** | 2.93 tok/s |

**+100 ms (+3.8%): rejected and reverted.** Creating 4 threads per kernel call (24+ calls per prefill)
costs more than the head-level parallelism saves on a 25 ms call.

**What this does NOT refute:** the reference's 4.26-cores-busy figure. It uses a *persistent* CPU pool
(`ONPU_CPU_POOL=1`, `ONPU_CPU_PREFILL_POOL=1`, `wake_helper`, `host_event_wait`) whose workers are created
once and woken per task, not four `pthread_create` per call. So the parallelism lead stands, but
realising it needs a pool, not per-call threads.

## Parallelism is refuted for this architecture: the CPU device serialises kernels

Two independent measurements:

1. **Python threads over head-slices** (ctypes releases the GIL, so the C call itself can overlap):

       serial 1 call      :   39.03 ms
       4 threads, 4 slices:  159.98 ms   -> 0.24x

   Exactly 4x slower -- **zero overlap**. tinygrad's CPU device serialises kernel execution, so
   splitting the scan across head groups just queues four calls.
2. **C-level pthreads inside one kernel call** (`cpu_gated_delta_scan_pooled` with a real
   `scan_parallel_fn`): correct (`4.209e-07` vs fp64) and **+100 ms** in-model.

So the reference's 4.26-cores-busy figure cannot be reproduced by parallelising kernels this way: its
workers run outside tinygrad's device (their own pool dispatching C work), whereas every kernel we launch
goes through the CPU device and is serialised.

**Consequence:** the parallelism lead -- the one mechanism the measured decomposition pointed at -- is
closed for our architecture unless the CPU work stops going through the tinygrad device. Combined with
the earlier rejections (three custom kernels, descriptor merge path, scan chunking), there is no
remaining single lever that closes the 2.18x gap.

## The scan itself parallelises only 1.21x -- in pure C, with no tinygrad

Harness calling `cpu_gated_delta_scan_part(..., group, 4)` directly (no tinygrad, no device, no graph),
B=1 H=32 T=128 K=V=128, median of 5:

    serial 4 groups :    30.29 ms
    4 threads       :    25.02 ms  -> 1.21x

So **intra-kernel parallelism of the scan is worth 1.21x, not 4x** -- the kernel is memory/latency-bound,
not core-bound. That explains all three failed parallelism attempts without appeal to tinygrad's device:

| attempt | result |
| --- | --- |
| per-call pthreads inside one kernel call | +100 ms |
| 4 Python threads over head-slices (through the device) | 0.24x (serialised by CPUProgram) |
| **pure C, 4 threads, no tinygrad** | **1.21x** |

**Consequence for the decomposition:** the reference's 4.26-cores-busy does NOT come from parallelising a
single kernel -- it must come from overlapping *different* kernels (attention, scan, elementwise) across
a pool. That is an execution-architecture property. Reproducing it means a task pool over the CPU
kernels in the graph, not a faster or parallel kernel.

Best case for intra-kernel scan parallelism: 0.61 s -> 0.50 s, ~0.11 s of the 1,413 ms gap.
