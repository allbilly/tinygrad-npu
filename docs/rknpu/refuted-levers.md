# Refuted and landed levers — one line each

The reason a lever was rejected is worth keeping forever; the derivation that produced it is not.
This table replaces the long analyses (`compare-ours-vs-ref.md`, `split/FINDINGS.md`,
`int8-name-reconciliation.md` §4+, `doc-audit.md`) that argued each case at length and then argued
the opposite further down. Those files are in git history if a derivation is ever needed.

Every row was measured on this board. "Refuted" means measured worse or neutral, not argued against.

## Landed (in the 1,198.87 ms tree)

| lever | measured | note |
| --- | --- | --- |
| fused INT8 Linear (one C call: quantise, submit, dequantise) | 1.43–1.80x faster than the unfused path | bitwise identical, `ab_split.py` |
| compiled-C replay (`replay_c.py`) | **−153.80 ms** | 1,412.89 → 1,259.09; dispatch was 136.24 ms over 1006 programs |
| four-way CPU transforms (`thread_split`) | **−599 ms** if missing | 1,858.49 vs 1,259.09: the run that forgot to install it (M72) |
| four-way quantise inside the fused C | **−60.22 ms** | 105.56 → 41.37 ms; 344/355 calls, 0 refused |
| m-tile merge, 603 → 355 submits | −89 ms | 0.36 ms per submit saved |
| scan buffer reuse | −82 ms | |
| qkv fusion | −37 ms | |
| INT8 rather than FP16 weights | prefill 2,532 → 1,403 ms | the four-way split commit `de1f5576d` |

## Refuted (measured worse or neutral)

| lever | measured | why |
| --- | --- | --- |
| split the fused call into quantise / submit / dequantise kernels | **1.43–1.80x slower** | adds the host↔device copies the fused call exists to remove |
| parallel dequantise (same pool, same split) | 62.46 → 67.85 ms | bandwidth-bound: writes `m*n` floats |
| quantise sharing across same-row consumers | **+6.61 ms worse** | duplicate quantise did not reach the clock |
| NEON dequantise | −0.3% | already at 94% of the 23.8 GB/s ceiling |
| gate/up fusion | +14.7% prefill | |
| float4 SwiGLU | +52 ms | |
| vector sigmoid standalone | +75 ms | |
| pthread scan | +100 ms | |
| Python threads instead of the C pool | 0.24x | |
| 8-way pool | 2,734 ms | |
| direct dispatch | neutral | |
| position cache | 318 hits / 371,626 misses | |
| second IOMMU domain | 0.30x | |
| K-split | +53% | |
| narrow tiling | worse | |
| FFN pack kernels (three forms) | 22–31% slower | |

## Unsafe — never retry

| lever | result |
| --- | --- |
| `k=9216` in one call | **wedges the NPU**: `DRM_IOCTL_RKNPU_SUBMIT` times out, driver logs `soft reset, num: 6`, board may reboot. The `banks` formula collapses to its `0x1b` floor at 18 groups |
| non-blocking submit | same soft reset, four times |
| `SKIP_SYNCS` / `SKIP_SUBMIT` ablations | produce wrong results by construction; ablation-only |

## Numbers that were wrong, and what replaced them

| quoted | reality | cause |
| --- | --- | --- |
| 83.13 ms "outside every counter" | **~8 ms** | overlapping decomposition rows: device/quantise/dequantise/syncs are *inside* the fused-Linear row (M69) |
| quantise 59.32 ms, dequantise 31.86 ms | **105.56 / 62.62 ms** | per-geometry × calls instead of a counter over one forward (M70) |
| 154.2 ms "outside the kernels" | ~8 ms | same overlap as the 83.13 |
| 4-way quantise "neutral, 1.0006x" | **−60.22 ms** | the dispatch was silently falling back; no hit counter (M68) |
| 1,402.60 ms baseline | 1,259.09 after replay, 1,198.87 after quantise | superseded |
| 2,532 ms / 3.22 tok/s | 1,198.87 ms / 4.57–5.07 tok/s | pre-INT8, pre-replay |

## What the 1,198.87 ms is made of

| row | ms | share |
| --- | --: | --: |
| NPU submit (device) | 424.57 | 33% |
| fused-Linear host: quantise 41.37 + dequantise 62.62 | ~104 | 8% |
| non-fused CPU kernels (scan 177.7, attention 31.0, elementwise/reduce ~440) | 651.52 | 51% |
| outside every counter | 8.31 | 0.6% |

Current result and protocol: `prefill-parallel-quantise.md`.
