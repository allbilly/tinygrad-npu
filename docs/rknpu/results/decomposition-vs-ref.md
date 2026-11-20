# Timing decomposition, ours vs the reference

Every number carries its protocol, because three different instruments are in play and they do not
produce interchangeable figures. Rows marked **comparable** were measured the same way on both sides;
rows marked *indicative* were not, and are labelled so rather than averaged.

## 1. Wall

| | ours | theirs | delta |
| --- | --: | --: | --: |
| decode, ms/token | **170.76** | **153.20** | **+17.56** |
| tok/s | 5.51 | 6.525 | |

- ours: `reconcile_wall.py`, steady-state steps 4-8 of 8 (170.4, 170.4, 170.5, 170.7, 170.8)
- theirs: `run_ref.py` in-session, `ref-oracle.log`, 1 warm-up + 3 measured, median of 6.348 / 6.549 / 6.525 tok/s

## 2. Ours, by component (reconciled, one instrument, same steps)

| component | ms | share | source |
| --- | --: | --: | --- |
| device wait | **142.34** | 83.4% | driver `npu.submit_seconds` delta |
| CPU, all | **31.11** | 18.2% | `thread_time_ns` delta |
| — kernels outside the fused call | 24.55 | 14.4% | MANIFEST class sum (elementwise 12.94 + scan 5.91 + reduce 2.71 + attention 2.30 + fill 0.69) |
| — in-call quantise/sync/dequantise | 4.85 | 2.8% | fused class 147.51 − its submit column 142.66 |
| — python / executor | 1.71 | 1.0% | thread CPU 31.11 − kernels 24.55 − in-call 4.85 |
| clock overlap | −2.69 | −1.6% | residual of the three measurements |
| **wall** | **170.76** | 100% | |

Device is 83.4% of the token. That single ratio decides where any further work must come from.

## 3. Ours, by kernel class (dump protocol, one warm token)

Wall 182.28 ms, 881 calls, 33 kernels. Slower than the 170.76 headline because the dump instruments
every call; use it for **shares**, not absolutes.

| class | ms | share |
| --- | --: | --: |
| fused INT8 Linear | 147.51 | 85.7% |
| elementwise | 12.94 | 7.5% |
| GDN scan | 5.91 | 3.4% |
| reduce | 2.71 | 1.6% |
| attention | 2.30 | 1.3% |
| fill | 0.69 | 0.4% |
| sum of kernels | 172.07 | |
| outside kernels | 10.22 | |

## 4. Theirs, by component (profile, per token)

Median of the nine 1700-call decode blocks in `ref-calls.jsonl`:

| component | ms |
| --- | --: |
| device (m=1 GEMM labels) | 136.62 |
| CPU (all labels) | 26.32 |
| copies | 0.59 |
| **sum** | **163.52** |
| their own unprofiled wall | 153.20 |
| **profile inflation** | **+10.32** |

**The inflation is why component-by-component comparison is only indicative.** Their profiled labels
sum to 10.32 ms *more* than their harness wall for the same work, from a different run. Any absolute
delta against their CPU figure therefore carries ±10 ms of uncertainty.

## 5. The comparison, with what is and is not comparable

| row | ours | theirs | delta | comparable? |
| --- | --: | --: | --: | --- |
| device | 142.34 | 136.62 | **+5.7** | yes (both device-side accounting) |
| host CPU | 31.11 | 26.32 | +4.8 | **no** -- ours is a thread clock, theirs a profiled run inflated by up to 10.3 ms |
| copies | (inside in-call sync) | 0.59 | -- | no |
| wall | 170.76 | 153.20 | **+17.56** | yes (both harness walls) |

Of the +17.56 ms wall gap, **+5.7 ms is measured device** and the rest is device we cannot separate
from the profile inflation plus the host term that the inflated baseline makes unstable. That is why
the earlier "+8 ms unattributed" kept moving: it was being computed against a baseline with a 10 ms
unknown in it.

## 6. Per-kernel, where the same kernel exists in both trees

Per-call, each side measured with its own itemiser on its own run. Ratios are meaningful; absolutes
carry the inflation caveat above.

| kernel | calls ours/theirs | ours us/call | theirs us/call | verdict |
| --- | --- | --: | --: | --- |
| `E_62080_4` (vocab op) | 1/1 | **434** | 8524 | ours **20x faster** (dead Gumbel term removed) |
| argmax | 1/1 | 97 | 86 | parity |
| `E_640_4` | 97/98 | 6.8 | 6.9 | **parity** |
| `E_2_16_32_4` | 48/48 | 46 | 51 | ours faster |
| `r_640_4` | 65/65 | 6.6 | 3.0 | theirs 2.2x faster |
| `E_8_4` | 48/48 | 6.0 | 2.2 | theirs 2.7x faster |
| attention | 8/8 | 287 | 34 | theirs 8x faster (attribution differs; see note) |
| `n=2560,k=4096` GEMM | 32/32 GEMMs | **418** | **345** | **+73 us/GEMM = +6.0 ms** |
| `n=9216,k=2560` GEMM | 64 | 779 | 756 | +23 us |

Note on attention: our `attention` class (2.30 ms, 8 calls) includes the softmax reduce that their
tree splits into separate `r_*` labels, so the 8x per-call ratio is not a like-for-like row.

## 7. Where the +17.56 ms sits

| item | ms | status |
| --- | --: | --- |
| `n=2560,k=4096` GEMM (418 vs 345 us/GEMM) | **+6.0** | nine hypotheses refuted; needs driver counters (root) |
| GEMMs we still split (gate/up 64->32, qkv 48->24) | **+3.7** | fusion measured at **+15-22% prefill** |
| device, unattributed within the inflation band | ~+5.7 | bounded by their +10.32 profile inflation |
| host | ~+2 | unstable; their baseline is inflated |
| **total** | **+17.6** | |

## 8. Banked this session

189.8 -> 170.76 ms (-10.0%), parity `895365e27ed4255b` unchanged:

| change | gain | test |
| --- | --: | --- |
| word-load rewrite | -3.1 | 25/28 pairs |
| skip never-selected zero pad | -8.6 | 21/28 pairs |
| greedy at temperature 0 | -10.0 | CPU 186.07 -> 175.11, structural |
| SIMD argmax | -0.1 | parity with their kernel |
| `ssm_alpha`+`ssm_beta` fusion | -0.35 | 48 -> 24 submits |
| `cpu_profile` bypass | (CPU only) | 174.20 -> 167.96 CPU, no wall |
| args-buffer reuse | (CPU only) | 175.01 -> 173.92 CPU |

## The last unexplained millisecond: E_80_32_64_4, 1.60 ms for a single call

It is the only entry in the MANIFEST with no counterpart in the reference profile, so it was traced
rather than assumed.

**What the emitted C says** (`identify_kernel.py E_80_32_64_4`, 39 lines):

```c
for (Lidx1_0 = 0; Lidx1_0 < 80; Lidx1_0++)          // 80 Q8_0 blocks = 2560 values
  alu0 = val1*2720 + Lidx1_0*34;                    // val1 = a RUNTIME index, 34-byte blocks
  for (Lidx1_1 = 0; Lidx1_1 < 32; Lidx1_1++)
    for (Lidx2 = 0; Lidx2 < 64; Lidx2++) {
      alu1 = Lidx2*3880;                            // 3880 = 248320/64 = vocab/64
      ... write *(val3 + alu0 + Lidx1_1 + 2) if val1 falls in that tile, else 0 byte
```

- source `val3`: Q8_0 rows of 80 blocks (2-byte scale skipped by the `+2`), i.e. 2560 values
- index `val1`: a **runtime scalar** read from a buffer
- destination: 80 x 32 x 64 = 163,840 bytes across 64 tiles of 3,880 -- **248,320 = the vocabulary**
- the comparisons `val1 < alu1 + 970/1940/2910/3880` write a byte only in the tile containing `val1`
  and **zero everywhere else**

So it is a **masked one-hot scatter over the vocabulary, sourced from a Q8_0 matrix** -- the shape of
an embedding/product implemented as a matmul rather than a gather. One call per token, 1.60 ms.

**Not yet named.** The compile-time stack only reaches `tinygrad/llm/model.py:967` (`Transformer.__call__`),
because the kernel is compiled during realise. Naming the op needs the sink uop (`DEBUG=4`) for that
kernel; guessing it from the arithmetic would be the fourth guess in this stretch, so it is left as the
one open item with a measured 1.60 ms attached.

For scale: 1.60 ms is 0.9% of the token -- ten times the norm+quantise fusion's realistic capture, which
is why this was traced before that port was started.
