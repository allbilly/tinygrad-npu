# Prefill: 1,198.87 ms via parallel quantise inside the fused Linear

Parity-verified prefill for Qwen3.5-4B on this board, 3 warm-ups + 3 measured, one process:

| trial | TTFT ms | start °C | end °C |
| --: | --: | --: | --: |
| 1 | 1,198.87 | 76.7 | 77.6 |
| 2 | 1,200.56 | 77.6 | 78.5 |
| 3 | 1,197.96 | 78.5 | 78.5 |
| **median** | **1,198.87** | | |

Spread 2.60 ms (0.22%). `complete: true`, 355 prefill submits, clocks pinned throughout.

Replicated on the lint-clean tree (the code that would be committed), a separate process:

| trial | TTFT ms | start °C | end °C |
| --: | --: | --: | --: |
| 1 | 1,193.13 | 71.2 | 72.1 |
| 2 | 1,193.30 | 72.1 | 74.8 |
| 3 | 1,240.95 | 73.9 | 74.8 |
| **median** | **1,193.30** | | |

So across two independent 3+3 runs the median is **1,193.30 and 1,198.87 ms**, and 5 of 6 trials are
at or below 1,200.56 ms. The sixth (1,240.95) is a 4% outlier and is the reason the margin is
described as thin rather than comfortable.

The target is 1,200 ms, so this meets it **on the median by 1.13 ms**, and one of the three trials
(1,200.56) is 0.56 ms over. It is at the line, not comfortably under it.

## Comparison (AGENTS canonical values)

| metric | ours | oNPU ref | RKLLM | vs ref | vs RKLLM |
| --- | --: | --: | --: | --: | --: |
| TTFT, 128 tokens | **1,198.87 ms** | 1,263.09 ms | 2,068.88 ms | −64.22 ms (1.054x) | −870.01 ms (1.726x) |
| prefill proxy | **106.79 tok/s** | 101.34 tok/s | 61.87 tok/s | 1.054x | 1.726x |
| decode | 4.57-5.07 tok/s (varies run to run) | 6.48 tok/s | 6.22 tok/s | **0.71-0.78x** | **0.73-0.82x** |

Decode is still behind both. Prefill is ahead of the reference and of RKLLM.

Clock basis, because it decides which way a gap moves:

| | clocks | temperature |
| --- | --- | --- |
| ours | NPU 1 GHz, DDR 2112 MHz, CPU6 2256000, pinned | 76.7 → 78.5 °C |
| oNPU ref | adaptive governors, no sudo tuning (`benchmark.md:1329`) | 52-55 °C start gate |
| RKLLM | published at maximum clocks | — |

Against the reference's archived `ref-profile` run (last trial 1,210.6 ms) this is 11.73 ms faster.
Against today's re-run of that same tree, which produced a best trial of 2,033.81 ms and a median of
7,968.69 ms, it is far ahead — but that run's spread (2.0-21.4 s) is 300x the difference, so today's
re-run cannot arbitrate a 12 ms question. The archived figure is the one being beaten.

## What the lever was

`PARALLEL_QUANT` in `extra/rockchip/npu_linear.py`: four-way quantise through the `rkpool` dispatch,
inside the fused C, writing straight into the NPU-mapped payload so no copy is added.

| | serial | 4-way |
| --- | --: | --: |
| quantise, in-model C counter | 105.56 ms | **41.37 / 42.62 / 47.49 ms** (3 runs) |
| wall, same probe | 1,282.71 ms | **1,220.68 / 1,233.50 ms** |
| pool hits | — | **344/355 calls, 0 refused** |

It had been recorded as "inert" for the whole project because nothing counted a pool hit: the emitted
C falls back to the serial loop when the dispatch returns non-zero, and both branches produce correct
output. Adding `stats[3] += 1` on success made it observable (`mistake.md` M68).

## What was refuted

Parallel dequantise, the same treatment on the other host loop, measured **worse**:

| | serial | 4-way |
| --- | --: | --: |
| dequantise | 62.46 ms | 67.85 ms |
| wall | 1,220.68 ms | 1,233.23 ms |

The pool ran (248/355 dispatched, 0 refused), so the loop is the cause, not the dispatch: it writes
`m*n` floats and is bandwidth-bound at 22.3 GB/s, where threads only add handoff. Disabled with the
verdict in the code; `PARALLEL_DEQUANT = False`.

## Provenance

| | |
| --- | --- |
| report | `docs/rknpu/results/par-quant/report.json` |
| log | `docs/rknpu/results/par-quant2.log` |
| driver | `extra/rockchip/bench_replay_c.py` (asserts thread_split + rkpool + replay + `--npu`) |
| command | `PARALLEL=0 DEV=CPU:CLANG .venv/bin/python extra/rockchip/bench_replay_c.py --ckpt .../Qwen3.5-4B-Q8_0.gguf --name pq-3x3 --out docs/rknpu/results/par-quant --warmups 3 --runs 3 --npu` |
| parity | pre-sampling logits `895365e27ed4255b`, identical to the stock tree |
| config check | log carries `[thread_split] transforms applied: {...}`, the same set as the 1,259.09 baseline |

A 3+3 from this driver is only comparable to the baseline when that transforms line is present.
Its absence produced a 1,858.49 ms "regression" that was entirely the missing transform set
(`mistake.md` M72), and launching without `--npu` produced 0 submits (`mistake.md` M71).
