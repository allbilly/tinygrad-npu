# Handoff — Qwen3.5-4B decode on RK3588 NPU

## Goal

Two halves, from the task statement:

1. **Improve decode to match the reference.** Reference measured on this board, in-session:
   **153.2 ms/token** (`6.525 tok/s`, `run_ref.py`, `ref-oracle.log`).
2. **Dump kernels and give a detailed timing decomposition** — per-kernel `.c` sources with
   calls / ms / µs-per-call / submit / cpu, a class rollup, a per-token two-sided table, a per-shape
   device table, and a wall reconciliation.

Half 2 is delivered and current. Half 1 is **not met**: our best measured decode is
**170.76 ms/token**, a **+17.6 ms** gap.

---

## State in one table

| | ms/token | source |
| --- | --: | --- |
| **ours** | **170.76** | `reconcile_wall.py`, steady-state steps |
| — device wait | 142.34 | driver `npu.submit_seconds` |
| — CPU (all) | 31.11 | `thread_time_ns` |
| — clock overlap | −2.69 | residual of the two |
| **reference** | **153.2** | `run_ref.py`, 1 warm-up + 3 measured |
| **gap** | **+17.6** | |

Per token we stream **4,205 MB of weights** (batch-1 decode reads every weight every token). At this
board's 33.79 GB/s LPDDR ceiling that is a **124.4 ms floor**; our device time is 87% of peak, the
reference's ~91–97%.

Banked this session, by timing: **189.8 → 170.76 ms (−10%)**.

---

## READ THIS BEFORE USING ANY NUMBER ABOVE: numerics are broken

**The fused INT8 stack produces zero logits.** Measured through the real `Transformer.forward`:

| build | sampled token | logits abs-sum | nonzero |
| --- | --: | --: | --: |
| stock `nn.Linear` (no `npu_linear`) | **830** | 334,391 | 100% |
| fused (`fuse_linears` + `replace_linears`) | **0** | **0.000** | **0%** |

Corroboration: `bench_llm` reports in `docs/rknpu/results/threadsplit/*/report.json` contain
`generated_ids: [0, 0, 0, 0, ...]`.

**And the parity gate never detected it**, because the gate itself hashed zeros for the whole session:

```
sha256(zeros(1, 248320, f32))[:16] = 895365e27ed4255b
the session's "presample_logits"    = 895365e27ed4255b     <- identical
```

Consequences, stated plainly:

- **No change landed in this session is numerically verified** — word-load rewrite, zero-pad skip,
  greedy sampler, SIMD argmax, `ssm_alpha`/`ssm_beta` fusion. They were all certified against a
  constant.
- The **timing** evidence still stands (wall/device/host reconciliation, per-kernel counts, structural
  signals such as "the kernel disappeared"), because none of it depends on the gate. But it measures
  the *work* of a numerically dead pipeline.
- Every earlier claim of the form "parity `895365e2` held" in session notes should be read as
  "prefill was run and hashed a constant".

### Where the fused path diverges (updated: cumulative inside one block)

Measured through the real forward, prompt 32 tokens, `start_pos 0` (which re-initialises the caches):

| build | logits abs-sum | finite |
| --- | --: | --- |
| stock `nn.Linear` | **3.3439e5** | yes |
| + `attn_gate` fused only | 5.4198e5 | yes |
| + `attn_qkv` fused only | 3.2733e5 | yes |
| + `ssm_out` fused only | 6.7801e5 | yes |
| + `ffn_gate` fused only | 3.3441e5 | yes |
| + `ssm_beta` fused only | 3.3423e5 | yes |
| **all Linears fused** | **0.0000** (block 0 alone is 1.112e27, max 6.6e23) | at the output layer |

So **no single fused Linear destroys the model** (worst shift is `+ssm_out`, ~2x), and a single fused
GEMM agrees with the stock matmul on random input to **2.6% max relative error** -- expected for INT8
activation quantisation. The blow-up is **cumulative within a single block**: block 0 with every Linear
fused reaches 1e27, and the final quantiser then maps the NaN to zero codes, which is why the logits
are exactly 0 rather than NaN.

An earlier block-0 bisect that replaced one Linear at a time *in place* reported `attn_gate` at 3.5e6 --
**that result is invalid**: each run wrote the block's conv/scan state at the same position, so runs
differed by accumulated state, not by the fused layer. The table above supersedes it.

### Ruled out as the cause (each measured, not argued)

| candidate | test | result |
| --- | --- | --- |
| `WORD_LOAD` transform | gate with `WORD_LOAD=0` vs `1` | both zero |
| `ssm_alpha`/`ssm_beta` fusion | `FUSE_ALPHA_BETA=False` vs `True` | both zero |
| activation payload reuse | `PAYLOAD_REUSE=False` vs `True` (always quantise) | both zero |
| fused output layer | `model.output(rand(1,2560))` in isolation | correct: abs-sum 73,258, 99.99% nonzero |

What remains is the interaction of several fused GEMMs inside a GDN block -- the recurrence amplifies
INT8 activation-quantisation error far beyond the ~2.6% a single layer shows. Whether that is our
quantiser being coarser than the reference's (their path keeps state in fp16 and packs activations for
a packed down-GEMM), or a genuine defect in one fused path, is not yet decided.

## Instruments (all in `extra/rockchip/`)

| tool | what it gives | status |
| --- | --- | --- |
| `reconcile_wall.py` | wall = device (`submit_seconds`) + CPU (`thread_time`) + overlap; argv[2]=1 passes the embedding row | **works** |
| `decode_kernels_by_name.py` | per-kernel calls/ms/µs for one decode token; argv[2]=1 passes the embedding row | **works** |
| `shape_device_ab.py` | one fused GEMM in isolation, JIT-captured, device time from `submit_seconds` | **works** |
| `prefill_ab.py` | prefill A/B in the frozen target's configuration (ABBA, per-process arm) | **works** |
| `dump_decode_kernels.py` | regenerates `docs/rknpu/results/decode-kernels/` (MANIFEST + `.c`) | works; refuses to overwrite on shape mismatch |
| `logits_gate.py` | prefill logits hash | **now exits 2 on all-zero logits** |
| `decode_gate.py` | T=1 logits hash + no-JIT ground truth | **invalid**: returns the prefill value for any token/position, and its `body` is a copy of the forward that also yields zeros |
| `ab_decode.py`, `ab_nopad.py`, `ab_elem.py`, `ab_attn.py`, `ab_noop.py`, `ab_wordload.py` | in-process ABBA harnesses | work; **all their "parity" checks were vacuous** |

Ledger: `mistake.md`, **M1–M98**. Read M90–M98 before trusting any comparison; M98 is the zero-hash
gate.

---

## Reference facts established (useful, independent of our bug)

- **Submit count**: ours 283/token vs theirs 203. Gate/up fused to `18432` and qkv to `12352` in
  theirs; we issue `9216 x2` and `8192+4096`.
- **Per-GEMM, `m=1, n=2560, k=4096`** (the one shape both trees run at the same count): ours
  418 µs/GEMM, theirs 345 µs. Nine hypotheses refuted: cores (3 is optimal), typed ABI (−150 ns),
  CBUF value (0x b1 both), task encoding (identical source), K padding (4096 both), pipelining
  (their 1-token loop is still 345 µs), IOMMU domains (419.8 vs 417.8), call form (raw is 84 ms
  worse), model-context penalty (retracted — it was summing parallel tasks).
- **Their label semantics**: `rockchip_replay.py:460` wraps each call as
  `begin=now(); …; stats[i+2]=now()-begin` with `CLOCK_MONOTONIC` — wall per call, one slot per call
  site. Their components sum to **163.52** against a **153.2** wall because the windows **overlap**
  (1.65× from their 4-worker `SHARED_HOST_POOL`). That makes their **host wall 16.0 ms** against our
  serial **31.11**.
- **Fusion is a prefill trade, measured**: qkv+gate at 12288 gives −0.71 ms decode for **+15–22%
  prefill** (1,180 → 1,361–1,458 ms, 4 ABBA runs). `FUSE_QKV_GATE = False` in `npu_linear.py`.
- **Width helps at m=1 only**: at k=4096, n=2560 → 25.9 GB/s; n=5120 → 31.1; n=10240 → 31.8. At
  m=128 a 12288-wide call is **14% slower** than two calls inside it (3,461 vs 3,038 µs) — the width
  limit flips the sign.

---

## Sized but unlanded levers

| lever | ms | state |
| --- | --: | --- |
| embedding gather (`extra/rockchip/embed_native.py`) | **1.59** | kernel verified bit-exact at 0.02 ms vs 1.59; `forward`/`__call__` accept an `embd` row and the harness arm shows **167.54 → 166.01**; **not wired into any shipped path** |
| swiglu → quantise fusion | 0.83 | not built; our `gated_activation` is already one fused kernel, so only the down-GEMM quantise is absorbable |
| norm+quantise fusion | 0.66 | not built |

Sum of all three: **3.08 ms = 18% of the gap**. Instrument resolution: ABBA ~1.5% (2.6 ms), itemiser
run-to-run 4.6% (7.7 ms).

---

## Next steps, in order

1. **Fix the fused-path numerics — this comes before any optimisation.**
   Bisect **within block 0** by fusing Linears in pairs/subsets (single fusions are safe, all-fused is
   1e27), and split by role: the Linears feeding the GDN scan (`attn_qkv`, `ssm_alpha/beta`) versus the
   FFN (`ffn_gate/up/down`). If the scan-side subset explodes and the FFN-side does not, the fault is
   the recurrent state under INT8 activation quantisation, and the fix is at the quantiser (fp16
   state / per-block scales like the reference's `_swiglu_pack`), not in a GEMM.
2. **Make a real gate.** `decode_gate.py` must execute a T=1 step and change its hash when the token
   or position changes; both gates must keep the all-zero refusal.
3. **Re-verify every landed change** (word-load, zero-pad, greedy, argmax, alpha/beta) against that
   gate. Keep what is numerically exact; revert what is not.
4. **Then** wire the embedding row into the measured decode path (1.59 ms) and re-measure.
5. Only after 1–3, resume optimisation: the remaining 17.6 ms is 83% device (uniform per-GEMM
   efficiency) and ~15 ms host parallelism, both of which need the driver counters (root) or a
   same-run profile on the reference tree.

## Traps that cost time here (do not repeat)

- `use the recorded protocol`: 3 warm-ups + 3 measured; the first measured trial is still warming.
- A constant hash looks exactly like a stable hash. Assert **sensitivity** (different input →
  different digest) and **non-degeneracy** (not all zeros) in any gate.
- The MANIFEST `calls` column counts **tasks**, not GEMMs (96 vs 32 for `2560x4096`); the three tasks
  run in parallel, so one task's duration *is* the GEMM wall. Never `sum(parts)`.
- `npu_linear`'s fused Linear binds a CALL uop and `replay_c` calls the compiled entry by pointer, so
  `CPUProgram.__call__` hooks measure **nothing** for it. Use `thread_time_ns` or `submit_seconds`.
- A custom kernel at the **graph entry** defeats JIT capture (640–782 ms/token). `argmax_native` works
  because it is the last op.
- `USE_ATOMICS` is a `ContextVar`, not an env var.
- k=9216 wedges this board (soft reset, reboot). Do not test that shape.
