# Loop classification of the eight largest CPU kernels: which loop is splittable

**What this document is.** For each of the eight largest CPU kernels in the Qwen3.5-4B prefill,
the emitted C source is read and classified: which loop indices are reductions (an accumulator
that carries across iterations) and which only select disjoint output elements. That decides
which loop can be partitioned across four threads without changing a byte of output.

**What it is not.** It contains no timing claim of its own. The mechanism is proven
(`README.md`: bitwise parity on `E_393216_3`) but the timings in this directory are not yet
valid, and the README's rule stands: do not quote a speedup from here until the quiet-machine
re-run exists. Everything below is either read off the emitted source, or read off the two
measured profiles named in the next section.

This is the source that `extra/rockchip/thread_split.py:THREAD_SPLIT` is populated from.

## Sources, and which question each answers

| source | what it gives | used for |
| --- | --- | --- |
| `docs/rknpu/results/prof-merged.log` | one `{"event":"kernel_source","name":…,"head":"<C>"}` per kernel, full source | the loop nests, the accumulators, the store index algebra |
| `docs/rknpu/results/progcost/report.json` (trial `instrumented-b`) | calls, `fxn_ms` per CPU kernel | "call count", "total in-model ms" |
| `docs/rknpu/results/progcost/trace.jsonl` | per-program `name`, `dur`, `ins`, `outs` | "ms on the critical path" |
| `docs/rknpu/results/split/FINDINGS.md` | published chain table (line 2744) and the parallel-scan A/B (line 2428) | cross-check, and the thread-overhead constant |

`report.json` and `trace.jsonl` are separate replays, so their per-kernel totals differ by
<1.2% (`rockchip_scan_simd_1_32_128_128_128_1`: 594.6 in the report, 597.4 in the trace). The
chain column below is from `trace.jsonl`; no number in this document mixes the two runs.

**Identification note.** `prof-merged.log` names two kernels differently from `report.json`:
`cpu_gated_delta_scan` and `exp2_scalar`. They are the same kernels as the report's
`rockchip_scan_simd_1_32_128_128_128_1` (24 calls, 594.9 ms) and `rockchip_attn_128_145`
(8 calls, 120.3 ms) — matched on call count, on total time, and on the source itself
(`exp2_scalar` is the `static inline` helper inside `rockchip_attn`; `cpu_gated_delta_scan` is
the C scan). The identification is from those three facts, not from a shared label.

## Table 1 — measured cost and position on the critical path

`ms on chain` is the sum of a kernel's `dur` over the programs that lie on the makespan path of
`trace.jsonl`, computed with the same `build_deps`/`critical_path` rule as
`extra/rockchip/trace_deps.py`. It reproduces the published decomposition exactly (see
"Verification" below), so the per-kernel split of the grouped rows is the same arithmetic.

| kernel | calls | total in-model ms | ms on chain | per call ms | source of the cost |
| --- | --: | --: | --: | --: | --- |
| `qz_row` (in `rockchip_linear`, m=1) | 355 | 650 | not separable | 1.83 | `prof-merged.log` |
| `cpu_gated_delta_scan` (= `rockchip_scan_simd_*`) | 24 | 594.6 | **597.4** | 24.8 | report / trace |
| `E_2_16_32_32_4_4` | 48 | 310.1 | **158.5** (24 of 48 calls) | 6.46 | report / trace |
| `E_32_32_128_4` | 24 | 142.5 | **0 — not on chain** | 5.94 | report / trace |
| `r_32_16_4_32_4` | 48 | 140.4 | **71.7** (24 of 48 calls) | 2.93 | report / trace |
| `E_1024_32_4_4` | 24 | 136.0 | **136.7** | 5.67 | report / trace |
| `E_393216_3` | 32 | 132.1 | **134.1** | 4.13 | report / trace |
| `exp2_scalar` (= `rockchip_attn_128_145`) | 8 | 120.3 | **120.3** | 15.04 | report / trace |

`qz_row` has no separate chain entry because it is not its own program: it is the entry point of
the `rockchip_linear_*` program, so its cost is already inside FINDINGS' "NPU Linear 336 ms"
row (`rockchip_linear_*` = 336.2 ms of chain, derived the same way). It cannot be given a
per-kernel chain number without double counting.

`E_32_32_128_4` is the one surprise in this table: 142.5 ms of work, and **not one of its 24
calls is on the makespan path**. Today that does not make it worthless — the CPU device runs
kernels one at a time (`FINDINGS` line 2449), so with one worker every kernel's full duration is
on the wall and 142.5 ms of work is 142.5 ms of prefill. Being off-chain only means it stops
paying the moment other work can be overlapped with it, i.e. it is the first row to drop if a
pool ever exists.

## Table 2 — loop classification and recommendation

Occurrence (`occ`) is the 0-based index into the `for (int X = 0; X < N; X++)` statements in
source order inside the kernel body — the same index `thread_split.split_source` takes. Source
order puts the outermost loop first.

| kernel | loop nest | reduction index | indices that only select disjoint output | output write index | recommendation | `THREAD_SPLIT` |
| --- | --- | --- | --- | --- | --- | --- |
| `qz_row` | `for row(1) { qz_row: for i(depth) twice } ; 3x for j(2752/2752/2688)` | `i` in pass 1 (`largest`) | `row`; `j` | `o[i]`, `dst[j]`, `scales+row` | **do not split** — >99% of the 1.83 ms/call is `ioctl`, the CPU loops are ~13 k element ops | n/a (not reachable) |
| `cpu_gated_delta_scan` | `for bh(bh_count, strided by groups) { for t(rows) { for r(value_dim) { for c(key_dim) } } }` | `c` (`predicted*`, `result*`); `t` (state) | `bh` only | `out[step*value_dim + r]`, `s[r*key_dim+c]` | **do not split** — `bh` is already the `group/groups` split, measured ceiling 1.21x, per-call pthreads +100 ms | n/a (not reachable) |
| `E_2_16_32_32_4_4` | `for Lidx0_0(2) { for Lidx0_1(16) { for Lidx1(32) { for Lidx2(32) } } }` | **none** | all four | `alu196 + {0,128,256,384}` | **split Lidx1** (occ 2, 32 iters) — outermost is only 2 iters and the transform rejects it (`MIN_ITERS`); occ 3 (Lidx2, 32) is equally safe | `2` |
| `E_32_32_128_4` | `for Lidx0(32) { for Lidx1(32) { for Lidx2(128) } }` | **none** | all three | `alu2 + {0,128,256,384}` | **split outermost** (occ 0, 32 iters) — fully safe, but last in the queue: only 5.94 ms/call and 0 of 24 calls is on the critical path | `0` |
| `r_32_16_4_32_4` | `for Lidx1(32) { for Lidx2(16) { for Lidx0(32) } }` | **`Lidx0`** (`buf0[0..3]`) | `Lidx1`, `Lidx2` | `alu219 + {0,16,32,48}`, from `sqrt(buf0)` | **split Lidx1** (occ 0, `split outermost`) — correct only because the transform hoists `buf0` into `_part`, giving each thread its own accumulator | `0` |
| `E_1024_32_4_4` | `for Lidx0(1024) { for Lidx2(32) }` | **none** | both | `alu1 + {0,128,256,384}` | **split outermost** (occ 0) — 1024 x 32 = 32,768 independent stores | `0` |
| `E_393216_3` | `for Lidx0(393216)` | **none** | `Lidx0` | `alu0 + {0,1,2}` | **split outermost** (occ 0) — already in the allowlist, bitwise parity proven in `README.md` | `0` (present) |
| `exp2_scalar` | `for h(16) { for i(TOK=1) { attn_row: 3x for j(last), 2x for d(256) } }` | **`j`** (`mx`, `sum`, and the `oi` accumulation) | `h`, `i`, `d` | `oi[d]` (accumulated over `j`) | **do not split** as written — `split h` is the only safe index and it is a hand edit of `attn.c`, not the compiler-boundary transform | n/a (not reachable) |

## Table 3 — is splitting worth it? (modelled, not measured)

The transform spawns and joins threads **per kernel call**, so its overhead scales with the
call count, not with the work. The only measured figure for that overhead is the parallel-scan
A/B (FINDINGS line 2428): 4-way pthreads cost **+100 ms over 24 calls = ~4.2 ms per call**,
which includes create/join and imbalance. That constant is from a different kernel; it is used
here as an order-of-magnitude bound, not as a per-kernel measurement.

| kernel | per call ms | ideal 4x saving (0.75 x total) | overhead at 4.2 ms/call | net, modelled |
| --- | --: | --: | --: | --: |
| `cpu_gated_delta_scan` | 24.8 | 446 | 101 | **+345** (but measured ceiling is 1.21x → +3) |
| `exp2_scalar` | 15.0 | 90 | 34 | **+56** |
| `E_2_16_32_32_4_4` | 6.46 | 233 | 202 | **+31** |
| `E_32_32_128_4` | 5.94 | 107 | 101 | **+6**, and off-chain |
| `E_1024_32_4_4` | 5.67 | 102 | 101 | **+1** |
| `E_393216_3` | 4.13 | 99 | 134 | **−35** |
| `r_32_16_4_32_4` | 2.93 | 105 | 202 | **−96** |

Break-even per call at 4x is ~5.6 ms. Only the scan and the attention kernel clear it, and the
scan does not actually reach 4x. So **the ordering is set by per-call size, not by how
parallel the loop is** — and the one entry currently in the allowlist (`E_393216_3`, 4.13 ms per
call) is modelled to be a net loss, which is consistent with no in-model win having been
reported for it.

The reference applies the same kind of rule, but on total size rather than per call: it gates
its pthread spawns at `rows*width >= 131072` for `rockchip_cpu_rmsnorm` and
`rows*depth >= 32768` for `rockchip_swiglu_pack` (FINDINGS line 2736), and it runs them on a
persistent pool (`wake_helper`), so its spawn cost is paid once. Checked against the computed
shapes here, `r_32_16_4_32_4` is 2,048 rows x 128 wide = 262,144 elements — *above* that
131,072 threshold, i.e. squarely in the class the reference does parallelise. So for that kernel
the blocker is not the loop and not the size; it is that we pay create/join per call (4.2 ms
against a 2.93 ms call). That is the one thing a persistent pool would fix, and it is the
argument for building one rather than for adding entries to `THREAD_SPLIT`.

Summed over the four reachable `E_*`/`r_*` kernels (`E_2_16_32_32_4_4`, `E_1024_32_4_4`,
`r_32_16_4_32_4`, `E_393216_3`), Table 3's rows give 232.6 + 102.0 + 105.3 + 99.1 = **539 ms**
saved against 152 calls x 4.2 ms = **638 ms** spent, i.e. **−99 ms** on the same basis as the
table. The critical-path view is stricter, not looser: only 501.0 ms of that 718.6 ms of work is
on the chain, so even a scheduler that overlapped everything else could realise at most 375.8 ms
of the saving — 21.6% of the 1,742.3 ms chain, and still short of 638 ms. Break-even for the set
is ~3.5 ms of overhead per call (539/152), against a measured constant of ~4.2 ms.

---

## Per-kernel notes

### 1. `qz_row` — inside `rockchip_linear`, m = 1

Not emitted C from tinygrad: this is the hand-written `rockchip_linear` wrapper, and the source
in the log is the m=1 specialisation (`for (int row = 0; row < 1; row++)`).

Loop nest, in source order:

```
for (int row = 0; row < 1; row++)  qz_row(xr, 2560, 2560, o, scales + row)
   qz_row: for i+4<=depth (NEON max-abs)  ->  for i<depth        (scalar tail)
           for i+4<=depth (NEON quantise) ->  for i<depth
           for j = depth..k (zero pad)
for (int g...) for (int row...) for (int j = 0; j < 2752; j++) dst[j] = (float)src[j] * a * w_scale[j];
for (int g...) for (int row...) for (int j = 0; j < 2752; j++) ...
for (int g...) for (int row...) for (int j = 0; j < 2688; j++) ...
```

Reduction: `i` in the first pass of `qz_row` — the accumulator is `largest`, and the write
follows it:

```
  for (; i + 4 <= depth; i += 4) {
    float32x4_t a = vabsq_f32(vld1q_f32(xr + i));
    ...
    vmax = vmaxq_f32(vmax, a);
  }
  float largest = vmaxvq_f32(vmax);
  ...
  *scale_out = scale;
```

`*scale_out` is written at index `row`, which does not depend on `i`: splitting `i` across
threads would race on `largest`/`scale` and produce four different quantisation scales.

Disjoint: the three `j` dequant loops write `dst[j]` at indices that depend only on `j`, and
`dst` is `out + (g*1+row)*8192 + {0,2752,5504}` — disjoint per block.

**Recommendation: do not split.** The CPU work is 2,560 max-abs + 2,560 quantise + 8,192
dequant = ~13,300 element operations. Even at the pessimistic 5.5 ns/element this tree measures
for elementwise code (FINDINGS line 2747), that is ~73 us against a 1.83 ms call — under 4%.
The call is two `sync_mem` ioctls plus a submit; there is nothing to win.
`thread_split.py` rejects it anyway (entry is `void rockchip_linear(void *args)`, not the
`unsigned char* restrict` ABI, and the only literal-bound loop is `row < 1`).

### 2. `cpu_gated_delta_scan` — the GDN scan

The only dimension with independent iterations is `bh` (batch x heads, 32 here), and it is
already expressed as a split:

```
  float columns[256*256] __attribute__((aligned(16)));
  for(size_t bh=(size_t)group;bh<bh_count;bh+=(size_t)groups) {
```

Reductions, all inside the per-`bh` body:

```
            predicted0=predicted0+old0*key;
```
```
            F4 old0=*(const F4*)(p+0);
            F4 updated0=old0*a+delta0*key;
            *(F4*)(p+0)=updated0;
            result0=result0+updated0*query;
```

`predicted0..15` and `result0..15` carry across `c`, and `columns` (`p`) is read-modify-written
across `c`; `columns` and `state` carry across `t`. So `c` and `t` are both reductions, and
splitting `c` would also change the summation order of `result*` (not bitwise).

Output write indices: `out + step*value_dim + r` with `step = bh*capacity + t` — depends on
`bh`, so disjoint across `bh`. `s` is `state + bh*stride` — also per-`bh`.

**Race that would appear if `bh` were split textually inside one invocation:** `columns` is a
single 256 KB stack array per *call*. `cpu_gated_delta_scan_pooled` is safe only because each
group is a separate invocation of `cpu_gated_delta_scan_part`, so each gets its own frame and
its own `columns`. A naive split of the `bh` loop in place would put all four threads on one
`columns` buffer.

**Recommendation: do not split.** It is already implemented (`cpu_gated_delta_scan_pooled` with
a real `scan_parallel_fn`) and measured at +100 ms (FINDINGS line 2428), because in pure C the
kernel only reaches 1.21x (line 2472) — it is memory/latency bound, not core bound. It is also
unreachable from `thread_split.py`: the loop is
`for(size_t bh=(size_t)group;bh<bh_count;bh+=(size_t)groups)`, not the literal-bound
`for (int X = 0; X < N; X++)` pattern.

### 3. `E_2_16_32_32_4_4` — SwiGLU, 4 lanes per iteration

```
for (int Lidx0_0 = 0; Lidx0_0 < 2; Lidx0_0++) {          /* occ 0, 2 iters */
  for (int Lidx0_1 = 0; Lidx0_1 < 16; Lidx0_1++) {      /* occ 1 */
    for (int Lidx1 = 0; Lidx1 < 32; Lidx1++) {          /* occ 2 */
      for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {        /* occ 3 */
```

**No reduction.** There is no accumulator in the source at all, and the output buffer is written
exactly once per iteration:

```
          int alu196 = ((Lidx0_0<<18)+(Lidx0_1<<14)+((Lidx1<<9)+alu2));   /* alu2 = Lidx2<<2 */
          *((float4*)((((float*)(val0))+(alu196+128)))) = (float4){...};
```

All four indices occupy disjoint bit ranges of `alu196` — `Lidx0_0` → 2^18, `Lidx0_1` → 2^14,
`Lidx1` → 2^9, `Lidx2` → 2^2, with the four store offsets 0/128/256/384 filling 512 — so any
index can be split, and the write index always depends on the split loop. **No race on any
split.**

**Recommendation: split `Lidx1` (occ 2).** `split outermost` is *not* available here: `Lidx0_0`
has only 2 iterations, and `split_source` correctly rejects it (`MIN_ITERS = 8`). `Lidx2` (occ 3)
is equally safe and equally sized, so this is a locality choice, not a correctness one: the store
stride of `Lidx1` is 512 floats (2 KB) while `Lidx2` only spans 128 floats, so splitting `Lidx1`
gives each thread one contiguous 2 KB block per iteration instead of interleaving four threads
across the same 2 KB at 512-byte granularity. Both are clean at the 64-byte cache-line level.

Half of its calls (24 of 48) are on the chain, worth 158.5 ms. Modelled net at 4.2 ms/call:
+31 ms. Marginal.

### 4. `E_32_32_128_4` — SiLU, four unrolled scalar lanes

```
for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {    /* occ 0 */
  for (int Lidx1 = 0; Lidx1 < 32; Lidx1++) {  /* occ 1 */
    for (int Lidx2 = 0; Lidx2 < 128; Lidx2++) { /* occ 2 */
```

**No reduction**, and no vector types — this is the scalar-lane kernel the earlier emitted-C
audit flagged (`E_32_32_128_4` is "4 unrolled scalar lanes", FINDINGS line 1484).

```
        int alu2 = ((Lidx1<<9)+Lidx2+(Lidx0<<14));
        *(((float*)(val0))+(alu2+128)) = (alu3/(1.0f+alu38));
```

Strides: `Lidx2` → 1 (range 128), `Lidx1` → 512, `Lidx0` → 16384, offsets 0/128/256/384 fill
the 512. Disjoint, and the write index depends on every index. No race on any split.

**Recommendation: split outermost (occ 0)** — the loop is safe and the transform accepts it. But
it is last in the queue for two reasons. First, economy: 5.94 ms per call against ~4.2 ms of
thread overhead leaves a modelled net of **+6 ms**. Second, it is the only kernel in this set
with **zero** calls on the critical path, so it is the first one whose win disappears if the CPU
kernels are ever overlapped. It is not worthless today — the CPU device is serial, so its
142.5 ms of work is 142.5 ms of prefill — but it is the weakest of the five splittable `E_*`/`r_*` kernels.

### 5. `r_32_16_4_32_4` — the RMSNorm sum-of-squares reduce

This is the one kernel where **the output write index depends on a reduction result**, and the
only one whose accumulator is a named buffer rather than an SSA value.

```
void r_32_16_4_32_4(unsigned char* restrict args_24) {
  float buf0[4];
  ...
  for (int Lidx1 = 0; Lidx1 < 32; Lidx1++) {      /* occ 0 */
    for (int Lidx2 = 0; Lidx2 < 16; Lidx2++) {    /* occ 1 */
      *(buf0+0) = 0.0f;
      ...
      for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {  /* occ 2, the reduction */
```

The accumulator, verbatim:

```
        *(buf0+0) = ((*(buf0+0))+(alu150*alu150)+(alu151*alu151)+(alu152*alu152)+(alu153*alu153));
        *(buf0+1) = ((*(buf0+1))+(alu170*alu170)+(alu171*alu171)+(alu172*alu172)+(alu173*alu173));
        *(buf0+2) = ((*(buf0+2))+(alu190*alu190)+(alu191*alu191)+(alu192*alu192)+(alu193*alu193));
        *(buf0+3) = ((*(buf0+3))+(alu210*alu210)+(alu211*alu211)+(alu212*alu212)+(alu213*alu213));
```

and the store that consumes it, after the `Lidx0` loop closes:

```
      int alu219 = ((Lidx1<<6)+Lidx2);
      *(((float*)(val0))+(alu219+16)) = __builtin_sqrtf((((*(buf0+1))*0.0078125f)+7.812499980275334e-09f));
```

Classification: `Lidx0` is a reduction. `Lidx1` and `Lidx2` only select disjoint output
elements — `alu219 = (Lidx1<<6) + Lidx2` with `Lidx2 < 16` and offsets 0/16/32/48 filling the 64,
so each `(Lidx1, Lidx2)` owns four distinct outputs.

**Race to flag:** `buf0` is written at indices `+0..+3` that do **not** depend on `Lidx1`. Split
`Lidx1` in place and all four threads zero and accumulate the same four floats. The split is
therefore legal **only if the accumulator is private per iteration**, which is what the existing
transform happens to do: `float buf0[4]` sits before the split loop, and `split_source` moves
everything before the loop into `_part`, giving each thread its own copy. Verified on the real
source:

```
static void r_32_16_4_32_4_part(unsigned char *restrict args_24, int begin, int end) {
  float buf0[4];
  ...
  for (int Lidx1 = begin; Lidx1 < end; Lidx1++) {
```

**Recommendation: split Lidx1 (occ 0, `split outermost`)** — correct as long as the transform,
not a hand edit, performs it. Worth it? 2.93 ms per call against ~4.2 ms of thread overhead:
modelled net **−96 ms**. Not now.

### 6. `E_1024_32_4_4` — dequant x gate epilogue, `__fp16` out

```
for (int Lidx0 = 0; Lidx0 < 1024; Lidx0++) {   /* occ 0 */
  float4 val5 = (*((float4*)((((float*)(val2))+(Lidx0<<2)))));   /* per-row scale */
  for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {   /* occ 1 */
```

**No reduction.** No `buf` in the source; four fresh stores per iteration:

```
      int alu1 = ((Lidx0<<9)+(Lidx2<<2));
      int alu2 = (alu1+128);
      ...
      *((__fp164*)((((__fp16*)(val0))+alu1))) = (__fp164){((__fp16)(((val25[0]/val5[0])*cast0*(val29[0]/(1.0f+alu147))))), ...};
```

`Lidx2` → 4 (range 128), `Lidx0` → 512, offsets 0/128/256/384 fill the 512. Disjoint, and the
write index depends on both indices. No race on either split.

**Recommendation: split outermost (occ 0).** The outer loop is 1024 iterations of 32, i.e. 32,768
independent 4-element stores, and `val5` sits inside `Lidx0` so it is reloaded correctly for each
split range. On chain 136.7 ms; modelled net +1 ms — a wash at the current per-call overhead.

### 7. `E_393216_3` — flattened three-lane SwiGLU

```
for (int Lidx0 = 0; Lidx0 < 393216; Lidx0++) {   /* occ 0, the only loop */
  int alu0 = (Lidx0*3);
  ...
    *(((float*)(val0))+alu1) = ((val3/(1.0f+alu27))*val6);
    *(((float*)(val0))+alu2) = ((val4/(1.0f+alu31))*val7);
    *(((float*)(val0))+alu0) = ((val5/(1.0f+alu35))*val8);
```

**No reduction.** One loop, `alu0 = Lidx0*3`, three consecutive outputs and three consecutive
inputs per iteration: index 3i, 3i+1, 3i+2. Each `Lidx0` owns exactly three elements, so the
partition is disjoint and the write index depends on the split loop. No race.

**Recommendation: split outermost (occ 0)** — already the allowlist entry, and the README's
bitwise parity (`differing=0`) was established on this kernel. Worth it? 4.13 ms per call is
below the ~5.6 ms break-even: modelled net **−35 ms**, which is the most likely reason no
in-model win has been reported for it.

### 8. `exp2_scalar` — the CPU attention kernel (`rockchip_attn`)

```
void rockchip_attn(void *args) {
  ...
  for (int h = 0; h < H_; h++) {            /* occ 0, 16 heads */
    for (int i = 0; i < TOK; i++) attn_row(out, q, k, v, i, pos, bh_q, bh_kv);
```

Inside `attn_row`, all four `j` loops are reductions:

```
  float mx = scores[0];
  for (int j = 1; j <= last; j++) if (scores[j] > mx) mx = scores[j];
  ...
  for (int j = 0; j <= last; j++) { scores[j] = exp2_scalar((scores[j] - mx) * log2e); sum += scores[j]; }
  ...
  for (int d = 0; d < D_; d += 4) *(F4 *)(oi + d) = *(F4 *)(oi + d) + pv * *(const F4 *)(vj + d);
```

`mx` is a max reduction (exactly reassociable, but still a carried accumulator), `sum` is a
float sum (not bitwise reassociable), and `oi[d]` is accumulated over `j` — so the output write
index depends on the reduction result.

**Race to flag:** `oi` is written at `out + (bh_q*TOK + i)*D_ + d`, which does **not** depend on
`j`. Splitting `j` would both race on `oi` and change the summation order. Splitting `h` is
safe: `scores[CAP]` is a stack array inside `attn_row`, so each thread's rows get their own
scratch, and `bh_q` is per-head so `oi` is disjoint.

**Recommendation: do not split** as this tree stands. `split h` is the only correct index and it
is a hand edit of `attn.c` (the entry is `void rockchip_attn(void *args)` and the bound is the
macro `H_`, so `split_source` rejects both the ABI and the bound). At 15.0 ms/call and 8 calls
it is the second-best candidate economically (+56 ms modelled), behind only the scan — if the
threads were amortised across calls rather than created per call.

---

## Kernels not in the log — skipped, not classified

`prof-merged.log` carries exactly eight `kernel_source` records. Ranked by total time in
`trace.jsonl` (NPU programs excluded), the eight largest CPU **programs** are the scan,
`E_2_16_32_32_4_4`, `r_32_16_4_32_4`, `E_32_32_128_4`, `E_1024_32_4_4`, `E_393216_3`,
`rockchip_attn_128_145` and `E_128_16_64_4` at 51.5 ms. The log covers the first seven plus
`qz_row`, which is not a separate program (it is the CPU pack inside `rockchip_linear_*`, 650 ms
attributed by the profiler). **So the eighth-largest CPU program is the one omission**, and it is
recorded below rather than guessed at.

Two entries that *look* large in `progcost/report.json` are not CPU kernels and have no source
in the log:

| entry | calls | ms | why skipped |
| --- | --: | --: | --- |
| `rockchip_linear_64_9216_2560` | 64 | 219.8 | the NPU submit + dequant wrapper (it contains `qz_row`); a device dispatch, not a compute loop |
| `rockchip_linear_64_2560_4096` | 96 | 195.7 | the NPU submit + dequant wrapper; a device dispatch, not a compute loop |

`E_128_16_64_4` — 8 calls, 51.5 ms total, 51.5 ms of chain, the 8th largest CPU program — is
**not classified here** because its source is not in the log. It would be the next one to read
if a ninth source is dumped. Note it is fully on the chain and 6.4 ms per call, i.e. it would
rank between `E_2_16_32_32_4_4` and `E_1024_32_4_4` in Table 3's economics.

## Verification of the chain numbers

Summing my per-kernel chain attribution into the groups FINDINGS publishes reproduces its table
to 0.1 ms, including the total:

| FINDINGS row | published | recomputed from `trace.jsonl` |
| --- | --: | --: |
| GDN scan (`rockchip_scan_simd_*`) | 597.4 | 597.4 |
| elementwise `E_*` | 529 | 528.2 |
| NPU Linear | 336 | 336.2 |
| attention (`rockchip_attn_*`, `r_*`) | 229 | 229.0 |
| `E_128_16_64_4` | 51.5 | 51.5 |
| **critical path** | **1,742.3** | **1,742.3** |

So the per-kernel split of those rows used in Table 1 is the same arithmetic that produced the
published totals, not a different method.

## What this document does not claim

- No speedup. The only timings in `threadsplit/` are the two contradictory isolated runs in
  `README.md` (1.87x vs 3.70x on identical commands), which the README itself rules out of use.
- No in-model result for any split. `THREAD_SPLIT` currently holds `E_393216_3: 0`, and
  Table 3 models that entry as a net loss at the measured per-call thread cost.
- The 4.2 ms/call overhead constant is measured on the scan, not on these kernels. It decides
  the sign of every row in Table 3, so any decision to enable an entry must come with an A/B
  that measures it, and with bitwise-identical prefill output between `THREAD_SPLIT = {}` and
  the enabled set (the README's parity requirement).
