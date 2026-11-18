# Can the guarded scalar `exp2` be parallelised 4-way, and what is the reference's guard restructuring worth?

Pure C, no Python, no tinygrad, **no NPU access**. Two sizes, the four required
SiLU variants, a sigmoid pair, a copy floor, and seven ablations. Every timed
number below is measured on this board (Orange Pi 5, RK3588) with the exact
command in [Build](#build).

## Answer

| question | measured answer |
| --- | --- |
| 4-thread scaling of the emitted guarded scalar code (V1 -> V2) | **3.98x / 4.00x** (N=131072 / 1179648) |
| 4-thread scaling of the reference vector code (V3 -> V4) | **3.99x / 4.00x** |
| reference vector vs emitted scalar, 1 thread (V3 / V1) | **1.321x**, identical at both sizes |
| vector + 4 threads vs emitted scalar 1 thread (V4 / V1) | **5.26x** |
| per-call thread spawn vs persistent pool | **6.6-11 ms vs 0.58 us** — spawn is disqualified at this kernel size |
| is the result bit-identical? | **yes, 0 ULP** over [-20,20] and [-1000,1000], all guard probes, all specials |

Threading the existing scalar code captures **92% of the combined gain**. The
guard restructuring on top of it is worth the remaining 8% (1.32x).

## Variants

All compute `y = x*sigmoid(x)` over N contiguous floats, reading `a` and writing
`b`, with `t = x * -1.4426950216293335f` and the degree-6 polynomial from
the emitted kernel.

| variant | what it is |
| --- | --- |
| **V1** | the emitted C of `E_393216_3`, transcribed per element. **Faithful including its 3-lanes-per-loop-body structure**: the emitted kernel handles 3 lanes per iteration, so the compiler already has 3 independent polynomial chains to interleave. The trailing `*gate` multiply and the 2-stream input are dropped (this measures SiLU, not the gated SwiGLU product). |
| **V2** | V1 split into 4 equal ranges over pre-created pthreads, one range per thread. |
| **V3** | the reference's form: `float4` via `ext_vector_type`, `rockchip_exp2_unchecked` polynomial verbatim, **one magnitude branch per 4 lanes**, `\|t\| < 125` fast path (`0x42fa0000`), checked scalar path otherwise. |
| **V4** | V3 split 4 ways over pre-created pthreads. |
| V1s | V1 arithmetic, stride-1 loop. Extra row: prices the emitted 3-wide body. |
| S1 / S3 | `sigmoid(x)` scalar / vector. Extra rows. |
| V3e | V3's exp2 with the divide removed. Ablation: prices the divide. |
| V3u2 | V3 with two independent 4-lane chains and one guard per 8 lanes. Ablation: prices the serial Horner chain. **Not part of the reference.** |
| COPY | `y = x`. Memory floor. |
| DISP | the 4-thread dispatch with no work at all. Prices the barrier. |
| V2p / V4p | V2/V3 with 3 pthreads created and joined **per call**. |
| TCREATE | 3 pthreads created and joined per call, doing nothing. |

Thread affinity: thread 0 (the caller) on **cpu4**, workers on **cpu5,6,7** —
the four Cortex-A76 cores, set with `sched_setaffinity` (no call reported an
error), so no thread ran on cpu0-3 (A55). Times are one
kernel call per repetition, **20 repetitions, median**, taken from run-0.

## Conditions (stated, not assumed)

- **The clock was not pinned by me** — `sudo` is unavailable on this host. The
  exact readings are recorded before and after every run in each `run-*.log`:

  ```
  policy0  gov=performance min=1800000 max=1800000 cur=1800000   (cpu0-3, A55)  min==max
  policy4  gov=performance min=408000  max=2256000 cur=2256000   (cpu4-5, A76)  NOT pinned
  policy6  gov=performance min=408000  max=2256000 cur=2256000   (cpu6-7, A76)  NOT pinned
  ddr      cur=2112000000 min=2112000000                                        min==max
  npu      cur=1000000000
  ```

  The two A76 domains are **not** frequency-pinned: `min_freq` is 408 MHz and only
  the governor holds them at 2256 MHz. Every sample taken before and after every
  run read 2256000, but that is the governor's choice, not an enforced floor.

  Because I could not *enforce* that state, the protocol does not depend on it:
  **every variant is interleaved within one round**, so all ratios are taken
  between adjacent runs of the same round, never across time. Single-thread and
  parallel phases are separate so that a spinning worker never shares a core
  with a single-threaded variant; V1 and V3 are measured in **both** phases, and
  the two agree to within 0.2% (V1 0.4340 vs 0.4346 ms, V3 0.3284 vs 0.3290 ms),
  so the phase split is not a confound.

- **Validity gate, fixed before the timed runs.** A run is reportable iff the
  dispatch-only variant DISP has median < 5 us and no phase-1/2 variant has more
  than one 3.33 ms tick spike. Four runs were taken; **all four passed**
  (DISP = 0.58 us, zero spikes) and agree within 0.7% on every variant. Two earlier
  runs failed the gate and are reported in [Variance](#variance-and-rejected-runs).

- **Timing input keeps every lane on the fast path** (`|x| <= 20`, so
  `|t| <= 28.9 < 125`). The slow path is therefore not timed; it is the same
  scalar arithmetic as V1 behind one branch. In-model, `|x| >= 86.7` is already
  numerically saturated, so this is representative — but it is an assumption,
  not a measurement.

## Build

Exact command (`build_bench.sh`, which is `build.sh`'s content; see the note
about this shared directory under [Files](#files)):

```
clang -O2 -fPIC -ffreestanding -fno-math-errno -march=armv8.2-a -mcpu=native \
      -Wall -Wextra -Wno-unused-parameter \
      bench.c -o bench -lpthread -lm
```

- `-nostdlib` was **dropped**: it prevents linking a normal executable. Flags
  are otherwise the project's own.
- `-lpthread -lm` added for this harness only.
- clang 14.0.0; `-mcpu=native` resolves to `-target-cpu cortex-a76`, and cc1
  receives `-ffp-contract=on` (so both V1 and V3 contract to `fmla`, which is
  what the zero-ULP result confirms).

## Results

### Summary

| variant | N | median ms | ns/element | GB/s moved | speedup vs V1 1 thread |
| --- | --: | --: | --: | --: | --: |
| V1 | 131072 | 0.4340 | 3.311 | 2.42 | 1.00x |
| V1 | 1179648 | 3.9117 | 3.316 | 2.41 | 1.00x |
| V1s | 131072 | 0.5550 | 4.235 | 1.89 | 0.78x |
| V1s | 1179648 | 5.0063 | 4.244 | 1.89 | 0.78x |
| V3 | 131072 | 0.3284 | 2.506 | 3.19 | 1.32x |
| V3 | 1179648 | 2.9614 | 2.510 | 3.19 | 1.32x |
| V3u2 | 131072 | 0.3135 | 2.392 | 3.34 | 1.38x |
| V3u2 | 1179648 | 2.8277 | 2.397 | 3.34 | 1.38x |
| V3e | 131072 | 0.2780 | 2.121 | 3.77 | 1.56x |
| V3e | 1179648 | 2.5047 | 2.123 | 3.77 | 1.56x |
| S1 | 131072 | 0.5670 | 4.326 | 1.85 | 0.77x |
| S1 | 1179648 | 5.1090 | 4.331 | 1.85 | 0.77x |
| S3 | 131072 | 0.3252 | 2.481 | 3.22 | 1.33x |
| S3 | 1179648 | 2.9342 | 2.487 | 3.22 | 1.33x |
| COPY | 131072 | 0.0292 | 0.223 | 35.95 | 14.88x |
| COPY | 1179648 | 0.3770 | 0.320 | 25.03 | 10.38x |
| V2 | 131072 | 0.1091 | 0.832 | 9.61 | 3.98x |
| V2 | 1179648 | 0.9813 | 0.832 | 9.62 | 3.99x |
| V4 | 131072 | 0.0825 | 0.630 | 12.70 | 5.26x |
| V4 | 1179648 | 0.7435 | 0.630 | 12.69 | 5.26x |
| DISP | 131072 | 0.0006 | — | — | — |
| DISP | 1179648 | 0.0006 | — | — | — |
| TCREATE | 131072 | 0.0458 | — | — | — |
| TCREATE | 1179648 | 0.0626 | — | — | — |
| V2p | 131072 | 0.2632 | 2.008 | 3.98 | 1.65x |
| V2p | 1179648 | 2.0180 | 1.711 | 4.68 | 1.94x |
| V4p | 131072 | 0.2109 | 1.609 | 4.97 | 2.06x |
| V4p | 1179648 | 1.5574 | 1.320 | 6.06 | 2.51x |

### 4-thread scaling, measured in phase 2 (adjacent runs)

| comparison | N=131072 | N=1179648 |
| --- | --: | --: |
| V2 / V1 (scalar, 4 threads) | 3.984x | 3.999x |
| V4 / V3 (vector, 4 threads) | 3.986x | 3.995x |
| V4 / V1 (vector + 4 threads) | 5.258x | 5.261x |
| V3 / V1 (vector, 1 thread) | 1.321x | 1.321x |
| V3u2 / V1 (2 chains, 1 thread) | 1.384x | 1.383x |
| V3e / V1 (divide removed, 1 thread) | 1.561x | 1.562x |
| V4 / COPY (how far above the copy floor) | 2.83x | 1.97x |
| dispatch overhead (DISP) | 0.58 us | 0.58 us |


## Correctness: bit-identical, not merely within 1 ULP

`compare()` compares the ordered integer representation of every float, so a
"0" below is bit equality, and `max_ulp=0` means no lane anywhere in the buffer
differed at all.

```
# cmp V3 vs V1  x in [-20,20]      n=131072 differing=0 max_ulp=0
# cmp V4 vs V1  x in [-20,20]      n=131072 differing=0 max_ulp=0
# cmp V2 vs V1  x in [-20,20]      n=131072 differing=0 max_ulp=0
# cmp V1s vs V1 x in [-20,20]      n=131072 differing=0 max_ulp=0
# cmp S3 vs S1  x in [-20,20]      n=131072 differing=0 max_ulp=0
# cmp V3 vs V1  x in [-1000,1000]  n=131072 differing=0 max_ulp=0
# cmp V4 vs V1  x in [-1000,1000]  n=131072 differing=0 max_ulp=0
```

and the same seven lines repeat with `n=1179648`. That is the required 1-ULP
check satisfied at 0 ULP, over both a dense `[-20,20]` sweep and a dense
`[-1000,1000]` sweep that crosses the fast-path guard.

Why they are expected to match:

- `q`: both forms compute `(int)(t + (t<0 ? -0.5f : 0.5f))`, i.e. truncation of
  `t` offset by half, so the same integer with the same rounding rule.
- `s = t - q`: the same operation on the same two operands in both forms, so the
  bit pattern is identical by construction. (It is also exact for `|t| < 128`,
  which is why the polynomial is evaluated on the same `s` either way.)
- the polynomial: the same six coefficients in the same order, and
  `-ffp-contract=on` contracts both to the same `fmla` chain.
- the exponent: the emitted code multiplies by `2^(q>>1)` then by
  `2^(q-(q>>1))`, while the reference adds `q<<23` to the exponent field. Both
  are **exact** scalings by a power of two, so they agree bit for bit rather than
  approximately — in the region the guard admits, where nothing overflows or
  goes subnormal.

The guard boundary is exercised, and it discriminates: at `x = -90` the
argument is `t = 129.8 > 125`, so V3 must take the checked path and flush to
`-0` exactly as V1 does. Had the guard not fired, `rockchip_exp2_unchecked`
would have returned a garbage small number instead. All 23 probes agree:

```
# guard probe N=131072 (V1 scalar vs V3 vector):
#   x=-1000    V1=-0             V3=-0             same
#   x=-90      V1=-0             V3=-0             same
#   x=-86.7    V1=-1.92614209e-36 V3=-1.92614209e-36 same
#   x=-86.6    V1=-2.12626419e-36 V3=-2.12626419e-36 same
#   x=0        V1=0              V3=0              same
#   x=1        V1=0.731058598    V3=0.731058598    same
#   x=86.6     V1=86.5999985     V3=86.5999985     same
#   x=86.7     V1=86.6999969     V3=86.6999969     same
#   x=125      V1=125            V3=125            same
#   x=1000     V1=1000           V3=1000           same
#   -> 0/23 differ at the guard probes
# specials (V1 vs V3):
#   x=nan      V1=nan    V3=nan    same
#   x=inf      V1=inf    V3=inf    same
#   x=-inf     V1=nan    V3=nan    same
#   x=-0       V1=-0     V3=-0     same
#   x=1e-30    V1=5.00000002e-31 V3=5.00000002e-31 same
#   x=-1e-30   V1=-5.00000002e-31 V3=-5.00000002e-31 same
```

One design note that the NaN case forced: the fast-path test is an **integer**
max over `|bits|` (`bic`+`umaxv`+`cmp`), not a float `fmax`. ARM's `FMAX`/`FMAXV`
return the non-NaN operand, so a float reduction could let a NaN lane through
onto the fast path, where `convertvector` of an out-of-range value is undefined.
With the integer compare every NaN encoding exceeds `0x42fa0000` and takes the
checked path, which reproduces V1's NaN exit. This is one `bic` + one `umaxv` +
one `cmp` + one branch per 4 lanes:

```
    3d20:  ldr   q6, [x25]
    3d24:  fmul  v0.4s, v6.4s, v4.4s     ; t = x * -log2e
    3d28:  mov   v1.16b, v0.16b
    3d2c:  bic   v1.4s, #0x80, lsl #24   ; |t| bits
    3d30:  umaxv s1, v1.4s               ; max over the 4 lanes
    3d34:  fmov  w8, s1
    3d38:  cmp   w8, w27
    3d3c:  b.hi  <slow path>
```

## What the ablations rule out

The vector form wins only 1.32x, not 4x. The two obvious explanations were
tested and **both are refuted**:

| ablation | result | what it rules out |
| --- | --- | --- |
| V3e: V3 with the divide removed | 2.121 vs 2.506 ns/elem -> divide is 0.385 ns/elem, **15% of V3** | the divide is not what limits V3, and `1/(1+e)` is not worth replacing with a reciprocal estimate |
| V3u2: two independent chains, one guard per 8 lanes | 2.392 vs 2.506 ns/elem -> **1.05x** | the serial 6-`fmla` Horner chain is not the limiter either, so Estrin/unrolling is not the missing 4x |
| V1s: same arithmetic, stride 1 | 4.235 vs 3.311 ns/elem | the emitted 3-wide body is itself worth **1.28x** — the scalar code is not as naive as it looks |

So the residual is per-4-lane instruction throughput in the vector loop (the
emitted loop is ~30 instructions per 4 elements once the coefficient reloads,
conversions, select and cross-lane `umaxv` are counted), not a single
identifiable bottleneck. That is an inference from the disassembly, not a
measurement. The practical consequence is measured and does not depend on the
explanation: **further arithmetic work on this kernel has ~1.3-1.4x left in it,
while threading has 4x.**

Also notable: `S1` (scalar sigmoid) is *slower* than `V1` (scalar SiLU) at
4.33 vs 3.31 ns/elem despite one fewer multiply. The reason is structural, not
arithmetic: `S1` is a stride-1 loop like `V1s`, and like-for-like the two are
indistinguishable — `V1s/S1 = 0.98`, and in vector form `V3/S3 = 1.01`. So
removing the `*x` is worth nothing measurable in either form, and `S1` only looks
slow because it lacks the emitted body's 3-way ILP. (S1 -> S3 is 1.74x, but that
number mixes the vector win with the 3-wide-structure win and should not be
quoted as a vectorization factor.) This is one more reason not to reason about
this kernel from op counts.

## Per-call thread creation: two orders of magnitude, and the reason is `HZ=300`

`TCREATE` creates and joins the 3 threads with **no work at all**, so it is a
pure price for per-call threading. It is strongly **bimodal**, and the two modes
differ by ~150x, so both are reported rather than averaged:

| condition | N | TCREATE median | V2 pre-created | V2p per call | V4p per call |
| --- | --: | --: | --: | --: | --: |
| cores still warm from the previous phase (full sweep, run-0) | 131072 | 0.0458 ms | 0.1091 ms | 0.2632 ms | 0.2109 ms |
| cores still warm from the previous phase (full sweep, run-0) | 1179648 | 0.0626 ms | 0.9813 ms | 2.0180 ms | 1.5574 ms |
| **isolated sweep, nothing else running** (`percall-1`) | 131072 | **6.6061 ms** | — | 3.4428 ms | 9.9565 ms |
| **isolated sweep, nothing else running** (`percall-1`) | 1179648 | **7.5890 ms** | — | 4.3423 ms | 7.5721 ms |
| isolated sweep (`percall-2`) | 131072 / 1179648 | 6.5886 / 9.2455 ms | — | 10.0921 / 7.6455 ms | 8.1633 / 7.5723 ms |
| isolated sweep (`percall-3`) | 131072 / 1179648 | 11.1145 / 6.7130 ms | — | 6.7608 / 9.3065 ms | 6.5651 / 6.4820 ms |

Within the isolated sweeps, **20 of 20** repetitions of `V2p`/`V4p` at
N=1179648 were above 1 ms, and the observed range across all repetitions was
**0.048 ms to 27.65 ms**. This is a >20% run-to-run difference of the same
variant by two orders of magnitude; it is reported, not averaged. Both modes are
real: the fast mode is what you get while other work keeps the cores busy, the
slow mode is what you get on an otherwise idle board — which is the in-model
case, because prefill runs its CPU kernels on one core and leaves the other
seven idle.

The slow mode is not noise and not `pthread_create`. Isolated, with timestamps
taken inside the child:

```
rep  0  create   0.119  join   9.904  | child start lag: 6.705 10.002 6.685 ms
rep  1  create   0.060  join  19.915  | child start lag: 19.955 12.473 6.608 ms
rep  5  create   0.054  join  19.945  | child start lag: 19.981 6.636 6.631 ms
```

`pthread_create` returns in 0.05 ms; the child then does not *start* for 2-6
ticks. This kernel is `CONFIG_HZ=300` (`/boot/config-6.1.99-rockchip-rk3588`)
with `CONFIG_NO_HZ_IDLE=y`, and `cpu-sleep` is disabled with only `WFI` idle
available, so a blocked join plus a wake onto an idle big core costs integer
multiples of 3.33 ms. Consequences:

- **A per-call spawn costs 6.6-11 ms against a 0.434 ms kernel: 15-25x the work
  it would parallelise.** It is not a trade-off to tune, it is disqualified.
- **A persistent pool that polls (what V2/V4 use) costs 0.58 us** — the `DISP`
  variant — i.e. 0.5% of V2 at N=131072 and 0.06% at N=1179648. That is 4 orders
  of magnitude cheaper than spawning, and it is the only viable shape.

## Variance and rejected runs

Four runs were taken under the pre-registered gate; all four passed and agree
within **0.7%** (V1 0.4340/0.4358/0.4360/0.4366 ms; V3 0.3284/0.3296/0.3302/0.3305;
V2 0.1091/0.1094/0.1094/0.1094; V4 0.0825/0.0828/0.0831/0.0831). Reps within a
run vary by ~1% (V1 min 0.4334, max 0.4375).

Eight sweeps were run in total (two early ones, `run-0..3`, `run-4/5`). **One was
rejected and is reported rather than quietly dropped.** The cause was a competing
job on this host (`ref_kernel_profile.py`, a sibling task, ~190% CPU migrating
across cpu4-7). In that sweep, `DISP` — which does no work — read a median of
**14.99 ms**, and V2/V4 read 15.99/16.15 ms, i.e. it measured the scheduler, not
the kernel. A `COPY` median moved 0.0316 -> 0.1149 ms (+263%) in the same pair.
Any run in which `DISP` exceeds a few microseconds is void by construction, which
is why the gate is on `DISP` and not on the kernels under test. The early
sweep's raw log was overwritten by later runs; its medians are preserved here
because they were captured when it ran.

A fifth and sixth sweep (`run-4`, `run-5`) were taken after a rebuild that only
added the phase-3-only switch. They reproduce every compute-bound variant to
within **4%** (V1 +1.28%/+1.80%, V3 +1.11%/+2.14%, V2 +1.07%, V4 +1.06%) but move
**`COPY` by up to +91%** — and *within* those runs `COPY` is bimodal, some reps
at the clean value and others 2-4x it:

```
run-0 COPY N=1179648: 0.3704 0.3815 0.3704 0.3952 0.3789 0.3704 0.3762 ...  uniform, 7% spread
run-4 COPY N=1179648: 0.7032 1.5490 0.5168 0.5090 1.0762 0.6950 0.4355 ...  bimodal
```

That is external **memory-bandwidth** contention, and it exposes the gate's blind
spot: it inflates `COPY` from 0.03 to 0.10 ms, a factor of 2-4 but far *below* one
3.33 ms tick, so neither the spike rule nor `DISP` catches it. The headline
ratios are unaffected (the compute-bound variants moved 1-4%), but **the memory
floor is the fragile measurement here**, so the floor used below is run-0's,
whose 20 `COPY` reps are uniform within 7%. This is also a direct demonstration
of the project's own rule: the memory-bound variant is the one that cannot be
trusted across time, while the compute-bound variants are stable under the same
interference.

Two artefacts in *this harness* were found and fixed before the reported runs,
and are noted so the numbers can be trusted:

1. `exp2_vec4` initially did not inline, so V3 paid an AAPCS64 call and a
   4-float stack round-trip every four lanes. V3 measured 0.3972 ms before and
   0.3284 ms after forcing the fast path inline and moving the rare checked path
   out of line, which moved the reported V3/V1 ratio from 1.09x to **1.32x**. The
   uninlined form understated the vector win by 21%.
2. An early per-call measurement left the persistent workers spinning while new
   threads were created on the same cores, which produced a fabricated
   "13 ms per thread create". That was CFS timeslice, not thread creation; the
   phases are now separated.


## What this implies in-model, and the honest caveat

Measured factors, all from the tables above: threading alone **3.98x** (V2/V1),
the reference's guard restructuring alone **1.32x** (V3/V1), both together
**5.26x** (V4/V1). Applying them to the two elementwise groups on the serial
critical path, **assuming for the projection that 100% of each group is this
kernel family and that it threads as it does in a standalone loop**:

| group | as measured | threading only (3.98x) | vector only (1.32x) | threading + vector (5.26x) |
| --- | --: | --: | --: | --: |
| elementwise group A | 310 ms | **78 ms** (-232) | 235 ms (-75) | **59 ms** (-251) |
| elementwise group B | 529 ms | **133 ms** (-396) | 400 ms (-129) | **101 ms** (-428) |
| both | 839 ms | 211 ms (-628) | 635 ms (-204) | 160 ms (-679) |

If both groups really are this kernel family end to end, prefill goes
2,532 -> ~1,904 ms (threading alone) or ~1,853 ms (threading + vector), i.e.
**1.33x / 1.37x end to end** — and **threading produces 92% of it**, with the
reference's restructuring worth ~19 ms on group A and ~32 ms on group B. The
ordering of the work is therefore not close: thread the emitted scalar kernels
first, and treat the guard restructuring as a 1.32x multiplier on the arithmetic
that remains.

**The caveat, and it is the whole risk.** These kernels are compute-bound here
by a wide margin (V1 runs 14.9x above the copy floor at N=131072, V3 11.3x), and that is
exactly why 4 threads scale at 3.98x. At 4 threads the picture has already
changed: V4 moves 12.7 GB/s while `COPY` at the same size reaches 25.0 GB/s on
the clean run, so V4 sits only **1.97x above the memory floor** at N=1179648 and
2.83x at N=131072 — and when a competing workload inflated `COPY` to 0.72 ms
(run-4) the 4-thread kernel was **at** that floor. Two consequences:

1. The measured 3.98x is close to the best case. The in-model kernel
   `E_393216_3` moves **12 bytes per element** (two inputs read, one written)
   against the 8 bytes/elem measured here, so its memory floor is ~50% higher
   and the achievable threading gain is correspondingly lower — it is the same
   4 cores fighting for the same 33.79 GB/s. Any part of group A or B that is
   already bandwidth-limited will gain far less than 3.98x, and the vector
   restructuring will gain nothing there.
2. Because the 4-thread form is already within 2x of the copy floor, the
   remaining arithmetic headroom after threading is small. That is consistent
   with the ablations: removing the divide entirely (V3e) buys 1.18x and
   breaking the Horner chain (V3u2) buys 1.05x. There is no 4x sitting in the
   arithmetic; the 4x was always in the threads.

For scale, the copy floor itself is consistent with the DRAM ceiling: at
N=1179648 `COPY` reaches 25.03 GB/s, 74% of the 33.79 GB/s implied by 2112 MHz
x 8 bytes x 2; at N=131072 it reaches 35.95 GB/s, *above* the DRAM ceiling,
which confirms that the 1 MB working set is cache-resident and that the
N=131072 row is not measuring DRAM.

## Not measured, and so not claimed

- The **slow path is not timed**. Every timed lane took the `|t| < 125` fast
  path. A model input of `|x| >= 86.7` would fall back to 4 scalar chains.
- **No end-to-end model run.** The projection above is arithmetic on measured
  ratios, labelled as a projection; nothing here was verified against prefill.
- **The rejected sweep's raw log no longer exists** (it was overwritten by the
  later runs); its medians are preserved in the [Variance](#variance-and-rejected-runs)
  section because they were captured at the time.
- **A55s are excluded.** All threads were pinned to cpu4-7. Whether 8 threads
  (adding the 4 A55s) scales better, worse, or not at all was not measured.
- **tinygrad's dispatch is not modelled.** The 0.58 us here is a pthread
  flag-poll, not the compiler/executor path that would have to issue the
  threaded kernel.
- **`-mcpu=native` selects `cortex-a76`**, so these are A76-tuned code
  sequences; A55 codegen for the same source was not examined.

## Files

**This directory is shared with a sibling task** that works on
`extra/rockchip/thread_split.py`. Its artifacts are `orig.c`, `split.c`,
`E_393216_3_split.c`, `parity_driver.c`, `parity`, `README.md`,
`loop-classification.md` and `inmodel/`; they are not part of this report and
were not modified. Everything listed below is this report's.

| file | what |
| --- | --- |
| `bench.c` | the harness: 4 SiLU variants, sigmoid pair, copy floor, 7 ablations, interleaved sweeps, 3 phases, correctness |
| `build_bench.sh` | the exact compile command above (also kept as `build.sh`) |
| `run.sh` | runs the sweep N times, recording clock/temperature/load around each |
| `gen_tables.py` | regenerates every table in this report from a run log |
| `analyze.py` | statistics, tick-spike accounting, in-phase ratios |
| `run-0.log` .. `run-5.log` | raw output, ns integers, clock state, correctness; run-0 is the primary, run-4/5 are the contention evidence |
| `percall-1.log` .. `percall-3.log` | phase-3-only sweeps: the cold-core per-call mode |
| `tables.md` | the generated tables (identical content to those pasted below) |

Raw numbers per repetition follow. One kernel call per repetition; each value is
`CLOCK_MONOTONIC_RAW` around exactly one call, so these are the numbers the
medians were taken from.

### Raw values per repetition (ms, one kernel call per rep)


1-single, N=131072, 20 reps

| rep | V1 | V1s | V3 | V3u2 | S1 | S3 | V3e | COPY |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | 0.4343 | 0.5550 | 0.3258 | 0.3141 | 0.5699 | 0.3255 | 0.2780 | 0.0297 |
| 1 | 0.4340 | 0.5547 | 0.3287 | 0.3132 | 0.5740 | 0.3249 | 0.2785 | 0.0300 |
| 2 | 0.4340 | 0.5547 | 0.3255 | 0.3135 | 0.5670 | 0.3252 | 0.2806 | 0.0286 |
| 3 | 0.4337 | 0.5550 | 0.3255 | 0.3135 | 0.5670 | 0.3252 | 0.2780 | 0.0286 |
| 4 | 0.4358 | 0.5553 | 0.3252 | 0.3138 | 0.5670 | 0.3252 | 0.2777 | 0.0289 |
| 5 | 0.4337 | 0.5574 | 0.3293 | 0.3141 | 0.5670 | 0.3252 | 0.2782 | 0.0286 |
| 6 | 0.4334 | 0.5550 | 0.3258 | 0.3156 | 0.5670 | 0.3252 | 0.2780 | 0.0286 |
| 7 | 0.4337 | 0.5550 | 0.3281 | 0.3135 | 0.5685 | 0.3255 | 0.2771 | 0.0271 |
| 8 | 0.4337 | 0.5550 | 0.3287 | 0.3135 | 0.5667 | 0.3272 | 0.2780 | 0.0265 |
| 9 | 0.4337 | 0.5547 | 0.3281 | 0.3132 | 0.5673 | 0.3252 | 0.2777 | 0.0300 |
| 10 | 0.4337 | 0.5550 | 0.3287 | 0.3135 | 0.5670 | 0.3255 | 0.2780 | 0.0274 |
| 11 | 0.4340 | 0.5568 | 0.3284 | 0.3132 | 0.5673 | 0.3249 | 0.2777 | 0.0280 |
| 12 | 0.4340 | 0.5585 | 0.3287 | 0.3141 | 0.5676 | 0.3252 | 0.2777 | 0.0312 |
| 13 | 0.4343 | 0.5553 | 0.3284 | 0.3176 | 0.5667 | 0.3255 | 0.2780 | 0.0300 |
| 14 | 0.4343 | 0.5553 | 0.3284 | 0.3135 | 0.5702 | 0.3252 | 0.2780 | 0.0306 |
| 15 | 0.4343 | 0.5550 | 0.3258 | 0.3135 | 0.5667 | 0.3281 | 0.2780 | 0.0295 |
| 16 | 0.4340 | 0.5553 | 0.3287 | 0.3135 | 0.5670 | 0.3255 | 0.2774 | 0.0295 |
| 17 | 0.4375 | 0.5550 | 0.3287 | 0.3138 | 0.5670 | 0.3252 | 0.2780 | 0.0300 |
| 18 | 0.4340 | 0.5571 | 0.3284 | 0.3141 | 0.5673 | 0.3255 | 0.2774 | 0.0289 |
| 19 | 0.4340 | 0.5553 | 0.3307 | 0.3135 | 0.5667 | 0.3252 | 0.2777 | 0.0306 |

2-parallel, N=131072, 20 reps

| rep | V1 | V2 | V3 | V4 | DISP |
| --: | --: | --: | --: | --: | --: |
| 0 | 0.4346 | 0.1088 | 0.3290 | 0.0828 | 0.000292 |
| 1 | 0.4343 | 0.1091 | 0.3351 | 0.0825 | 0.000583 |
| 2 | 0.4349 | 0.1091 | 0.3284 | 0.0825 | 0.000292 |
| 3 | 0.4346 | 0.1091 | 0.3287 | 0.0825 | 0.000583 |
| 4 | 0.4346 | 0.1088 | 0.3287 | 0.0828 | 0.000583 |
| 5 | 0.4381 | 0.1094 | 0.3284 | 0.0825 | 0.000584 |
| 6 | 0.4343 | 0.1088 | 0.3290 | 0.0828 | 0.000583 |
| 7 | 0.4343 | 0.1088 | 0.3290 | 0.0828 | 0.000583 |
| 8 | 0.4349 | 0.1091 | 0.3322 | 0.0825 | 0.000584 |
| 9 | 0.4346 | 0.1091 | 0.3287 | 0.0825 | 0.000584 |
| 10 | 0.4349 | 0.1088 | 0.3287 | 0.0825 | 0.000584 |
| 11 | 0.4349 | 0.1088 | 0.3290 | 0.0828 | 0.000583 |
| 12 | 0.4375 | 0.1094 | 0.3287 | 0.0828 | 0.000292 |
| 13 | 0.4346 | 0.1088 | 0.3290 | 0.0825 | 0.000292 |
| 14 | 0.4346 | 0.1091 | 0.3290 | 0.0825 | 0.000291 |
| 15 | 0.4346 | 0.1091 | 0.3310 | 0.0825 | 0.000291 |
| 16 | 0.4349 | 0.1091 | 0.3287 | 0.0825 | 0.000292 |
| 17 | 0.4346 | 0.1088 | 0.3293 | 0.0825 | 0.000291 |
| 18 | 0.4352 | 0.1094 | 0.3290 | 0.0828 | 0.000583 |
| 19 | 0.4378 | 0.1094 | 0.3287 | 0.0828 | 0.000583 |

3-percall, N=131072, 20 reps

| rep | TCREATE | V2p | V4p |
| --: | --: | --: | --: |
| 0 | 0.0467 | 0.2648 | 0.2106 |
| 1 | 0.0458 | 0.2640 | 0.2112 |
| 2 | 0.0449 | 0.2631 | 0.2129 |
| 3 | 0.0452 | 0.2628 | 0.2106 |
| 4 | 0.0458 | 0.2672 | 0.2117 |
| 5 | 0.0455 | 0.2634 | 0.2109 |
| 6 | 0.0481 | 0.2695 | 0.2115 |
| 7 | 0.0455 | 0.2622 | 0.2091 |
| 8 | 0.0545 | 0.2622 | 0.2106 |
| 9 | 0.0426 | 0.2631 | 0.2190 |
| 10 | 0.0452 | 0.2736 | 0.2161 |
| 11 | 0.0461 | 0.2707 | 0.2091 |
| 12 | 0.0452 | 0.2645 | 0.2109 |
| 13 | 0.0516 | 0.2637 | 0.2088 |
| 14 | 0.0458 | 0.2619 | 0.2109 |
| 15 | 0.0545 | 0.2628 | 0.2077 |
| 16 | 0.0455 | 0.2616 | 0.2225 |
| 17 | 0.0440 | 0.2622 | 0.2103 |
| 18 | 0.0458 | 0.2692 | 0.2126 |
| 19 | 0.0528 | 0.2631 | 0.2106 |

1-single, N=1179648, 20 reps

| rep | V1 | V1s | V3 | V3u2 | S1 | S3 | V3e | COPY |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | 3.9139 | 5.0065 | 2.9429 | 2.8271 | 5.1053 | 2.9333 | 2.5019 | 0.3704 |
| 1 | 3.9115 | 5.0059 | 2.9590 | 2.8286 | 5.1088 | 2.9342 | 2.5002 | 0.3815 |
| 2 | 3.9115 | 5.0050 | 2.9613 | 2.8303 | 5.1071 | 2.9327 | 2.5037 | 0.3704 |
| 3 | 3.9101 | 5.0070 | 2.9607 | 2.8277 | 5.1062 | 2.9342 | 2.5080 | 0.3952 |
| 4 | 3.9133 | 5.0059 | 2.9607 | 2.8312 | 5.1085 | 2.9336 | 2.5045 | 0.3789 |
| 5 | 3.9115 | 5.0050 | 2.9636 | 2.8274 | 5.1068 | 2.9327 | 2.5048 | 0.3704 |
| 6 | 3.9086 | 5.0111 | 2.9645 | 2.8303 | 5.1062 | 2.9339 | 2.5078 | 0.3762 |
| 7 | 3.9104 | 5.0068 | 2.9625 | 2.8248 | 5.1100 | 2.9342 | 2.5107 | 0.3739 |
| 8 | 3.9098 | 5.0041 | 2.9625 | 2.8274 | 5.1103 | 2.9345 | 2.4987 | 0.3789 |
| 9 | 3.9159 | 5.0056 | 2.9633 | 2.8265 | 5.1065 | 2.9371 | 2.5045 | 0.3789 |
| 10 | 3.9107 | 5.0079 | 2.9560 | 2.8254 | 5.1094 | 2.9339 | 2.5060 | 0.3765 |
| 11 | 3.9112 | 5.0053 | 2.9604 | 2.8265 | 5.1094 | 2.9336 | 2.5019 | 0.3760 |
| 12 | 3.9130 | 5.0047 | 2.9625 | 2.8333 | 5.1074 | 2.9348 | 2.5098 | 0.3768 |
| 13 | 3.9110 | 5.0079 | 2.9616 | 2.8277 | 5.1106 | 2.9342 | 2.5107 | 0.3865 |
| 14 | 3.9145 | 5.0088 | 2.9601 | 2.8300 | 5.1094 | 2.9345 | 2.5031 | 0.3777 |
| 15 | 3.9145 | 5.0059 | 2.9630 | 2.8295 | 5.1097 | 2.9333 | 2.5075 | 0.3771 |
| 16 | 3.9142 | 5.0103 | 2.9607 | 2.8373 | 5.1080 | 2.9345 | 2.5063 | 0.3695 |
| 17 | 3.9118 | 5.0079 | 2.9598 | 2.8268 | 5.1123 | 2.9350 | 2.5022 | 0.3786 |
| 18 | 3.9130 | 5.0062 | 2.9642 | 2.8335 | 5.1094 | 2.9330 | 2.5092 | 0.3815 |
| 19 | 3.9250 | 5.0065 | 2.9625 | 2.8271 | 5.1091 | 2.9345 | 2.5040 | 0.3751 |

2-parallel, N=1179648, 20 reps

| rep | V1 | V2 | V3 | V4 | DISP |
| --: | --: | --: | --: | --: | --: |
| 0 | 3.9174 | 0.9806 | 2.9668 | 0.7426 | 0.000584 |
| 1 | 3.9223 | 0.9823 | 2.9668 | 0.7429 | 0.000292 |
| 2 | 3.9217 | 0.9797 | 2.9680 | 0.7423 | 0.000583 |
| 3 | 3.9235 | 0.9800 | 2.9692 | 0.7423 | 0.000583 |
| 4 | 3.9229 | 0.9972 | 2.9665 | 0.7458 | 0.000584 |
| 5 | 3.9191 | 0.9803 | 2.9686 | 0.7426 | 0.000583 |
| 6 | 3.9212 | 0.9841 | 2.9697 | 0.7429 | 0.000583 |
| 7 | 3.9244 | 0.9812 | 2.9689 | 0.7426 | 0.000583 |
| 8 | 3.9241 | 0.9815 | 2.9706 | 0.7432 | 0.000583 |
| 9 | 3.9223 | 0.9812 | 2.9697 | 0.7467 | 0.000583 |
| 10 | 3.9255 | 0.9809 | 2.9709 | 0.7435 | 0.000584 |
| 11 | 3.9238 | 0.9835 | 2.9703 | 0.7435 | 0.000584 |
| 12 | 3.9258 | 0.9812 | 2.9703 | 0.7435 | 0.000292 |
| 13 | 3.9258 | 0.9809 | 2.9744 | 0.7432 | 0.000584 |
| 14 | 3.9282 | 0.9815 | 2.9496 | 0.7487 | 0.000292 |
| 15 | 3.9247 | 0.9812 | 2.9718 | 0.7440 | 0.000583 |
| 16 | 3.9276 | 0.9832 | 2.9692 | 0.7461 | 0.000292 |
| 17 | 3.9250 | 0.9815 | 2.9735 | 0.7438 | 0.000291 |
| 18 | 3.9261 | 0.9844 | 2.9724 | 0.7443 | 0.000291 |
| 19 | 3.9255 | 0.9820 | 2.9738 | 0.7440 | 0.000584 |

3-percall, N=1179648, 20 reps

| rep | TCREATE | V2p | V4p |
| --: | --: | --: | --: |
| 0 | 12.6198 | 4.9414 | 3.2798 |
| 1 | 0.2217 | 2.2963 | 1.5730 |
| 2 | 0.0642 | 2.0163 | 1.5382 |
| 3 | 0.0502 | 2.0145 | 1.5555 |
| 4 | 0.0624 | 2.0151 | 1.5382 |
| 5 | 0.0528 | 2.9756 | 2.9695 |
| 6 | 1.7185 | 23.3071 | 5.8826 |
| 7 | 0.0633 | 2.0183 | 1.5348 |
| 8 | 0.0583 | 2.0081 | 1.5327 |
| 9 | 0.1245 | 2.0143 | 21.1298 |
| 10 | 2.5748 | 11.0442 | 8.8588 |
| 11 | 0.0683 | 2.0134 | 1.5520 |
| 12 | 0.0557 | 2.0177 | 2.3678 |
| 13 | 0.0534 | 2.0271 | 1.5248 |
| 14 | 0.0545 | 2.0189 | 1.5382 |
| 15 | 0.0496 | 2.0280 | 1.7045 |
| 16 | 0.0627 | 2.0128 | 1.5365 |
| 17 | 0.0464 | 2.0151 | 5.7035 |
| 18 | 0.0703 | 2.0256 | 1.5593 |
| 19 | 0.0508 | 2.0175 | 1.5368 |
