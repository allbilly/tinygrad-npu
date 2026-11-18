# Audit — `docs/rknpu/reference-qwen35-4b.md` and the FINDINGS correction

Condensed from an adversarial read-only audit; every id is preserved so fixes can be tracked
against it. Verdict: kernel quotations and the critical-path table reproduce exactly; four
headline claims do not.

## High severity — fix before this document is used for a build decision

| id | location | defect | fix |
| --- | --- | --- | --- |
| C2/B2 | ref doc §4 | Measured "4.26 cores", but the reference's **own** prefill profile totals **736.6 ms of CPU-program wall over 1,597 calls**. Even at 4 threads on every program that is 2,944 ms = 2.44 cores, against 5,140 ms reported. 2,194 ms is outside the profiled programs. | Both numbers are consistent once threads are accounted for: 736.6 ms of *wall* at 4 threads is ~2,950 ms of *thread* CPU, and the pool's spin (`ONPU_HOST_POOL_WFE`) supplies the rest. State the two quantities separately and never derive "threads needed" from the 5,140 figure. |
| B1 | ref doc §4 | 5,140 / 1,206 = 4.26 cores on a `taskset -c 4-7` **4-CPU mask**; sampler period 0.1 s over a 1.2 s window gives ±0.33 cores. | "saturated at ≈4 (±0.3), i.e. the mask's four cores" — never "more than four". |
| C2b | ref doc §4 | The attribution "4.26 cores is only possible if different kernels overlap" is wrong and was already contradicted by FINDINGS. | Replace with the intra-kernel mechanism, evidenced by the 3.0-3.1x per-call ratios in `ref-kernels/compare.md`. |
| C3/T1/B4 | ref doc §4 | "1,421 decode submits (22.2/token)" and "decode submits/token 307 vs 22.2". The run decodes **7** steps from 8 tokens and the decode graph is **203** NPU programs; `1,421 = 203 × 7` exactly → **1:1, 203 per step**. `22.2` is `1,421/64`, the 64-token published protocol. | "1,421 (203/step, 1:1)". The decode gap is therefore ~1.5x, not 14.6x. |
| U4 | ref doc §6.7, §8 | "reference invokes the scan 96x per prefill, ours 24x". `ref-phase2` (one replay) has **24**; `ref-profile` (four replays) has 96 = 4 × 24. Swiglu 128 vs 32 shows the same ratio. | **24x, identical to ours.** The chunking refutation then refutes a configuration the reference does not run — keep the refutation as a standalone result, not as a reference comparison. |
| U3 | ref doc §2 | "1,263.09 ms is best-of-three, not a median". | It is a **median over 20 requests** (`~/oNPU/docs/benchmark-qwen-latest-audit-20261103.json`). Keep only "1,201.4 replaces it" and the protocol difference. |
| U1/U2 | ref doc §2, §9 | Cites `ref-phase2/windows.jsonl` (does not exist) and attributes 1,200.09 / 1,198.62 to `ref-phase2`. | The split is `ref-phase2/phases.jsonl`; 1,200.09 / 1,198.62 are `ref-phase/report.json` trials 3-4, and `ref-phase2`'s clean native prefill is **1,205.70**. |
| S1/B3/C6 | ref doc §4 | "while 1,006 programs cost us ~1,790 ms". That was CPU *work* on one core, not main-thread cost, and it predates the scan-buffer change. | "our 1,006 programs measured 2,391.3 ms inside the kernels and 154.2 ms outside them". |
| R1/R2/R3 | ref doc §3 | The quoted reproduce command is **not** verbatim from `ref-profile/run.sh`: 14 env vars dropped (including `ONPU_INT8_NATURAL2304`, `ONPU_FFN_PACKED_UPLOAD`, `ONPU_GDN_*`, `ONPU_SWIGLU_PACK_STATIC_LAYOUT`, `ONPU_REPLAY_SCALAR_COPY`), `--input`/`--fuse-projections`/`--cpu-attention` missing, `taskset` missing, and `ONPU_CPU_POOL`/`ONPU_CPU_PREFILL_POOL` invented (they are CLI flags). Error caught: `extra/rockchip/ref_kernel_profile.py` now carries the full env and the full argv. | Replace §3 with the list from `ref_kernel_profile.py:ENV` + `:ARGS`, which is the executed configuration. |

## Medium

| id | location | defect |
| --- | --- | --- |
| C4 | §6.4 | "standalone float4 sigmoid … lost 52 ms" vs §8 "vectorised sigmoid +75 ms" and "float4 SwiGLU +52 ms" — two experiments conflated. |
| C5 | §5.1 | "Every fused CPU kernel takes the pool address" — false: rmsnorm and swiglu_pack call `pthread_create` directly on every launch; only `norm_quantize` and the scan use the pool. |
| S3 | FINDINGS | The withdrawn "280 ms fusion bound" is not the same failure as the per-call error; the 0.28 s submit-merge projection was measured at −89 ms and stands. |
| S4 | §8 | "descriptor route ~66 ms/Linear vs 1.3 ms fused" reuses the withdrawn 1.3 ms. Use the measured 2.04 ms/call (`rockchip_linear_64_2560_4096`, 196 ms / 96). |
| P1 | FINDINGS | "Threading the elementwise group 4x **alone** is worth ~390 ms" sums three treatments with two assumed scale factors. Relabel as a projection with each factor named. |
| U5 | FINDINGS | "5.5 ns/element" is not reproducible from any artifact; measured 3.45 ns/element on-chain (528.4 ms / 153.0 M), 4.27 across all `E_*`. |
| T2 | §5.2 | "128 x 9216 = 1,179,648" — the swiglu_pack call handles **4 rows** (`ONPU_SWIGLU_PREFILL_ROWS=4`), 4 x 9216 = 36,864, only 1.13x the 32,768 threshold. The 4-thread conclusion holds; the arithmetic is 32x off and hides that the threshold is marginal. |
| M9 | FINDINGS | "Ours has none: every kernel is single-threaded" — `extra/rockchip/thread_split.py` now exists; say "no kernel on our default path is threaded; the transformer exists but is unwired". |
| M7 | §6 | The `norm_quantize` quote has `{rows}/{depth}/{padded}/{address}` substituted by hand; label it "template parameters substituted for the 4B shapes". |

## Verified correct, no action

1,201.4 / 1,201.41 ms; the seven trial values; 720 / 5,140 / 486 (values, not attribution);
1,597 / 475 / 2,072; 2,391.3 / 154.2 / 7.7; 1,742.3 / 2,417.7 / 1.23x; the whole critical-path
table; "~390 ms" (396.3 actual); "~1,140 ms of chain"; the `diff -U0` = 2-hunk claim on
`rockchip_cpu.py`.

## Unresolved

Whether `1b2a5db` + the 4-file patch (`7f87f221…`) is the tree behind the published 1,263.09 row
(the 20-request cohort's trials are not in this checkout).
