# Reference prefill decomposed per kernel, and our same-named kernels against it

Source: `docs/rknpu/results/ref-phase2/report.json`, `native_execution.prefill.profile` — the
reference's own per-kernel wall clock for one native prefill (trial 3, 1,205.70 ms). Our column:
`docs/rknpu/results/progcost/trace.jsonl` (1,006 programs, 2,417.7 ms of kernel time, one core).

- CPU programs: **736.6 ms** over 1597 calls
- NPU programs: **428.0 ms** over 475 calls, on 7 domains
- device→host copies: **35.4 ms** over 758 copies
- prefill wall for that trial: **1205.70 ms**

## Kernels present under the same name in both trees

| kernel | ref calls | ref ms | ref ms/call | our calls | our ms | our ms/call | our/ref |
| --- | --: | --: | --: | --: | --: | --: | --: |
| `E_2_16_32_32_4_4` | 48 | 99.3 | 2.07 | 48 | 310.0 | 6.46 | **3.12x** |
| `E_32_32_128_4` | 24 | 47.4 | 1.98 | 24 | 139.9 | 5.83 | **2.95x** |
| `r_32_16_4_32_4` | 48 | 45.6 | 0.95 | 48 | 140.8 | 2.93 | **3.09x** |
| `E_8192_3` | 24 | 0.7 | 0.03 | 24 | 0.9 | 0.04 | **1.25x** |
| `E` | 121 | 0.0 | 0.00 | 9 | 0.2 | 0.03 | **183.27x** |

## Every reference CPU kernel, by time

| kernel | calls | ms | ms/call | ours (same name) |
| --- | --: | --: | --: | --- |
| `rockchip_cpu_gated_delta_scan` | 24 | 177.4 | 7.39 | — |
| `E_2_16_32_32_4_4` | 48 | 99.3 | 2.07 | 48 calls, 310.0 ms |
| `rockchip_int8_swiglu_pack` | 32 | 53.7 | 1.68 | — |
| `E_32_32_32_4_4` | 24 | 51.3 | 2.14 | — |
| `E_64_16_64_4` | 16 | 51.0 | 3.19 | — |
| `E_32_32_128_4` | 24 | 47.4 | 1.98 | 24 calls, 139.9 ms |
| `r_32_16_4_32_4` | 48 | 45.6 | 0.95 | 48 calls, 140.8 ms |
| `rockchip_dequantize` | 243 | 28.9 | 0.12 | — |
| `E_131_2048_4` | 24 | 26.7 | 1.11 | — |
| `E_toks_3088_4` | 24 | 20.5 | 0.86 | — |
| `E_28toks2A256029` | 64 | 19.4 | 0.30 | — |
| `rockchip_cpu_attention` | 8 | 19.2 | 2.40 | — |
| `r_128_32_32_4` | 24 | 16.7 | 0.70 | — |
| `rockchip_quantize` | 251 | 16.5 | 0.07 | — |
| `E_64_1024_4` | 48 | 9.5 | 0.20 | — |
| `E_32_32_4` | 48 | 9.3 | 0.19 | — |
| `E_128_640_4` | 64 | 8.3 | 0.13 | — |
| `E_16_128_32_2_4` | 8 | 5.9 | 0.73 | — |
| `r_16_toks_64_4` | 8 | 5.2 | 0.65 | — |
| `rockchip_cpu_rmsnorm` | 64 | 5.1 | 0.08 | — |

## What the numbers say

Three kernels exist under identical names and identical shapes in both trees:
`E_2_16_32_32_4_4`, `r_32_16_4_32_4`, `E_32_32_128_4`. Their per-call times differ by
**3.12x, 3.09x and 2.95x** — uniform, not kernel-specific. The total CPU-program wall differs by
**3.28x** (736.6 ms against our 2,417.7 ms). A uniform ~3.1x on the same
generated code is the signature of one mechanism applied to all of them: four threads inside one
call, not a better kernel. It also matches the only clean measurement in this tree (3.70x on
`E_393216_3`, `docs/rknpu/results/threadsplit/README.md`).

The 5,140 ms of process CPU in the phase probe is therefore **thread-seconds, not work**: 736.6 ms
of wall at four threads is ~2,950 ms, and the pool's spin-wait supplies the rest. The two figures
are consistent; neither implies more than the four cores in the `taskset -c 4-7` mask.

## Port target, in order of measured gap

| kernel | gap (our ms - ref ms) | note |
| --- | --: | --- |
| `E_2_16_32_32_4_4` | 210.7 | ref 2.07 vs our 6.46 ms/call |
| `r_32_16_4_32_4` | 95.2 | ref 0.95 vs our 2.93 ms/call |
| `E_32_32_128_4` | 92.5 | ref 1.98 vs our 5.83 ms/call |
| `E` | 0.2 | ref 0.00 vs our 0.03 ms/call |
| `E_8192_3` | 0.2 | ref 0.03 vs our 0.04 ms/call |

`rockchip_cpu_gated_delta_scan` (ref 177.4 ms / 24 = 7.39 ms/call) has no same-named kernel here;
ours is `rockchip_scan_simd_1_32_128_128_128_1` at 597.4 ms / 24 = 24.90 ms/call, a 3.4x gap that
needs their scan implementation, not just threading (our port measured a 1.21x parallel ceiling).
