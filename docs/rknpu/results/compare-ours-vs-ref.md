# Our 2.53 s against the reference's 1.20 s — valid per-kernel decomposition

Ours: the last **valid** baseline, 3+3 median **2,531.67 ms** (`results/scanreuse-3plus3/report.json`,
`complete: true`), decomposed with `results/progcost` (unthreaded, wall 2,549.7, kernel 2,391.3,
outside 154.2). Reference: `results/ref-phase2/report.json`, `native_execution.prefill.profile`
(wall 1,205.70; CPU 736.6, NPU 428.0, copies 35.4).

The threaded figures (1,848.56 / 1,929.52 / 1,933.34 / 1,854.9) are **void**: the transform
executed a helper instead of the entry, so each kernel ran one quarter of its loop. See
`results/threadsplit/README.md`. Nothing in this document uses them.

## The gap, row by row

| item | ours ms | ref ms | delta | share of 1,344 |
| --- | --: | --: | --: | --: |
| **elementwise `E_*`** | **910.2** | 366.8 | **+543.4** | **40%** |
| **GDN scan** | 597.4 | 177.4 | **+420.0** | **31%** |
| NPU programs | 610.8 | 428.0 | +182.8 | 14% |
| reduce `r_*` | 179.1 | 68.7 | +110.4 | 8% |
| attention | 120.3 | 19.2 | +101.1 | 8% |
| other CPU (norm/quant/pack) | 0 | 104.5 | -104.5 | -8% |
| outside every kernel / copies | 154.2 | 35.4 | +118.8 | 9% |
| **total** | **2,549.7** | **1,205.7** | **+1,344** | |

Our kernel sum is 2,417.7 ms over 1,006 programs; theirs is 1,164.6 over 2,072. So we run **half
as many programs, each ~3.7x more expensive**:

| unit | ours | ref |
| --- | --: | --: |
| CPU programs | 1,006 | 1,597 |
| CPU ms per program | **1.77** | **0.46** |
| NPU submits | 355 | 475 |
| NPU ms per submit | 1.72 | 0.90 |

## What this ranks

1. **Elementwise, 543 ms over 461 calls at 1.97 ms each.** Largest single gap, and the only one
   with a proven mechanism: the same kernel measures 6.78 ms/call serial and 1.76 ms/call through
   a four-way split in a standalone binary, bitwise equal (`results/splitcheck/`). In-model the
   split does not yet execute -- that is the one open blocker.
2. **Scan, 420 ms over 24 calls at 24.9 ms.** Kernel is byte-identical to theirs; measured to
   scale at most 2.24x (plain threads) and 2.83x overlap after fixing worker placement.
3. **NPU, 183 ms over 355 submits.** Real per-submit cost (1.72 vs 0.90) but hidden rather than
   exposed: off-CPU is 405 ms here against the reference's 486.
4. **Bookkeeping, 119 ms.** Outside every kernel; no kernel work affects it.

The single change that addresses the top row is the four-way split, and it is validated
standalone. Its in-model failure is the only thing standing between 2.53 s and ~1.99 s
(2,531.67 - 543), not 1.2 s.

## Next: the scan, justified by idle cores rather than kernel throughput

At the 2,531.67 baseline the scan is 597.4 ms over 24 calls, unthreaded, and the per-program
measurement gives it `thread/wall = 1.00` -- one core busy for ~600 ms of wall while the other
three idle. Its cost to the prefill is therefore the wall time, not the CPU time.

That reverses an earlier verdict. A standalone harness measured 30.62 ms serial against
33.29 ms pooled, and I concluded threading the scan loses. That compared kernel *throughput*
using a pool that predated the entry-order and memory-ordering fixes. For a serial segment in a
four-core prefill the relevant number is different:

| measurement | value |
| --- | --: |
| scan wall per prefill | 597 ms |
| parallel scaling with plain pinned threads | 2.24x |
| scaling with the reference's own pool | 1.50x |
| wall removed at 2.24x | **~330 ms** |

So the scan is worth porting on the idle-core argument alone, and the per-call kernel question
("does the pooled call beat the serial call in isolation") is not the deciding one.

**Gate:** the in-model parity check (`results/parity/{split-a,split-b,nosplit}`) must pass before
any new timing is taken, and the scan port itself gets a function-boundary bitwise check against
`cpu_gated_delta_scan` before it is enabled. The decomposition above must also be re-measured at
the new baseline; the rows here describe the unthreaded tree.

## At 1,402.60 ms: what is left, and how much of it is ours

Best parity-verified prefill: **1,402.60 ms** (`results/threadsplit/scanpool-3plus3`, `complete:
true`), 1.168x from the reference's 1,201.4. The gap is 201 ms. Decomposition at that baseline
(`results/threadsplit/attrib25`) and the device's own ioctl timer:

| row | ours | ref | recoverable? |
| --- | --: | --: | --- |
| NPU: blocked in `rknpu_submit_ioctl` | **420.0** | -- | no; it is the device's work |
| NPU: host (quantise, dequantise, copies) | **184.4** | -- | yes, CPU-side |
| outside every kernel (executor, copies) | **155.5** | ~35 | yes |
| GDN scan | **178.2** | 177.4 | at parity |
| elementwise `E_2_16` | 91.8 | 99.3 | ahead |
| reduce `r_*` | 42.4 | 45.6 | ahead |
| attention | ~31 | 19.2 | +12 |

The NPU row is 603.2 ms over 355 submits = 1.70 ms each, of which **1.18 ms is the blocking
ioctl** and 0.52 ms is host. So the row that looked like a 180 ms gap is mostly device time, and
only ~184 ms of it is addressable from our side.

### Consequence

The two rows above the reference that we control are **host work in the NPU path (184 ms)** and
**bookkeeping (155 ms)** -- 339 ms together, against a 201 ms gap. Neither is kernel arithmetic:
every arithmetic row is now at or ahead of the reference. The next measurement is the host half
broken down per submit (quantise vs copies), because 0.52 ms x 355 is the only place in that row
where code changes can land.
