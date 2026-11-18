# Audit: `extra/rockchip/thread_split.py` as a code change

Read-only audit. Nothing was compiled, loaded or run for this document; **no new measurement is
reported anywhere in it**. Every number is quoted from an existing artifact and cited to it.
`thread_split.py`, `rkpool.c`, `bench_threaded.py` and `probe_dump.py` are untracked working-tree
files (`git status`), so there is no commit to diff against.

The change is a text transformer on the input side of `ClangCompiler.compile_cached`
(`tinygrad/runtime/support/compiler_cpu.py:5`, `tinygrad/renderer/__init__.py:15-20`). It rewrites
one loop of a named CPU kernel into a four-way partition and emits a new entry point that
dispatches the four groups through a pool in `extra/rockchip/librkpool.so`, built from
`extra/rockchip/rkpool.c`.

It has already produced one wrong result and one invalid measurement. Each of those gets a named,
checkable invariant below rather than a paragraph of memory.

---

## 0. The three history items, re-verified against the current source

| # | claim | current state | evidence |
| --- | --- | --- | --- |
| 1 | an earlier version emitted the helper before the entry, so `CPUProgram` ran one quarter of the loop once | **fixed, but not enforced** — it holds by accident of renderer output order | `thread_split.py:93` is the first *definition*; `thread_split.py:91-92` are declarations; `thread_split.py:140-143` places `preamble` in front of them. Mechanically confirmed on the checked-in object: `splitcheck/transformed.o` puts the entry at `.text+0x0`, helpers at `+0x68` and `+0x74` |
| 2 | file-scope `static` in a jit-loaded kernel segfaults, so the pool lives in `librkpool.so` | **held** | all pool state is `.so` state (`rkpool.c:25-50`); the emitted wrapper declares only locals (`thread_split.py:94-102`); the positive control that produced the segfault is `splitcheck/README.md:37` |
| 3 | a call by materialised constant address returns nonzero and falls back; a pointer loaded from a volatile local works | **the volatile form is present as claimed** | `thread_split.py:99-102` builds `target[2]` as `volatile unsigned long`, reassembles the address from two loads (100) and calls through `fp` (103). The fallback covers all groups (`thread_split.py:104`). The *reason* this form is required is not established — see §7 |

---

## 1. Correctness invariants

Each invariant states how the current code enforces it, and what a violation looks like in numbers.

### I1 — the entry point is the first function in the object

**Mechanism.** `CPUProgram` executes the image base: `self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.buf.buf)`
(`tinygrad/runtime/ops_cpu.py:46`). The image is built by appending the PROGBITS sections in
section-header order, `.text` first (`tinygrad/runtime/support/elf.py:33-41`), and clang emits
functions into `.text` in source order. So "first definition in the source" == "first instruction
executed". Confirmed on a real object: `objdump -t splitcheck/transformed.o` gives
`transformed_kernel` at `0x0`, `_group` at `0x68`, `_part` at `0x74`.

**Enforcement.** None. It holds because `_wrapper` orders its own output correctly
(`thread_split.py:91` declaration, `93` entry, `106` `_group`, and `_part` appended afterwards by
`142`) *and* because the text placed before it is harmless: `split_source` returns
`preamble + _PRELUDE + wrapper + ...` (`thread_split.py:140-143`), and for `ClangRenderer`
`preamble` contains typedefs, `#define`s and at most `extern` declarations
(`tinygrad/renderer/cstyle.py:279`). Nothing checks it.

**Two ways it silently breaks:**

- any *definition* lands in `preamble`. `CStyleLanguage.render` emits
  `static inline ... void name(...) { ... }` **before** the kernel whenever `inline_prefix` is set
  (`cstyle.py:280-281`, joined ahead of the kernel at `cstyle.py:172`). `ClangRenderer` inherits
  `inline_prefix = None` (`tinygrad/renderer/__init__.py:109`), so `enter_calls` is false
  (`cstyle.py:275`) and only `extern` declarations are emitted. The invariant therefore rests on an
  attribute defined two files away, changed for other devices at `cstyle.py:514`.
- `_PRELUDE`/`_wrapper` order is edited at `thread_split.py:141-143`.

**Numeric signature of a violation.** The loader calls the first definition with the single
packed-args pointer. If that is `_group`, it reads `ctx[0] = *(void**)args` — the *output buffer*
pointer — and `group` from the uninitialised third argument register, then calls
`_part(out_ptr, group)`, which reads the output buffer's first 8 bytes as its args pointer. The
measured signature of exactly this class is in `splitcheck/README.md:21-25`: **1.89 ms/call against
a serial 6.46 ms in-model and 6.78 ms in the standalone driver** — i.e. one quarter of the work
(`6.78/4 = 1.70`, `6.46/4 = 1.62`, so the reading sits at 1.1-1.2x the quarter-work figure), which
is numerically indistinguishable from a 3.41x win. Note that the source line that quotes it as
`1.89 = 6.46/4` (`threadsplit/README.md:6`) does not compute; the conclusion survives, the
arithmetic in that sentence does not. With a poisoned output buffer it is a SIGSEGV instead (§5).

### I2 — the four groups partition `[0,N)` exactly once

**Mechanism.** `thread_split.py:137-138` emits
`for (int v = N*g/T; v < N*(g+1)/T; v++)`. With floor division, group `g` covers
`[floor(Ng/T), floor(N(g+1)/T))`: adjacent endpoints coincide, groups are disjoint, and the union
is `[0,N)` for every `N` and `T >= 1`. No divisibility requirement, so `N % T != 0` is correct and
must not be "fixed".

**Also required, and provided:** the text before the selected loop is duplicated into `_part`
(`thread_split.py:137`, `body[:loop.start()]`). That is what makes a per-iteration accumulator
legal — `r_32_16_4_32_4`'s `float buf0[4]` is declared before the split loop and zeroed per
`Lidx1` iteration (`docs/rknpu/results/threadsplit/loop-classification.md:285-330`).

**Enforcement.** Nothing checks `N > 0`, `N*T <= INT_MAX`, or `T <= RK_MAX` (the pool's limit,
`rkpool.c:22`).

**Numeric signature of violations:**

- a gap or an overlap in the formula leaves output elements unwritten (stale values) or written
  twice (idempotent here, therefore invisible) — see §5 for how to make it visible;
- `N*g` is emitted as `int` arithmetic (`thread_split.py:138`), so `N*3` overflows for
  `N > 715,827,882`. Signed overflow is UB and clang may compute a negative bound, giving that
  group zero iterations: one quarter of the output never written, no error. `MIN_ITERS = 8`
  (`thread_split.py:53`) is a floor only; there is no ceiling.

### I3 — the split loop carries no state between iterations

**Mechanism.** Only the hand-written allowlist `THREAD_SPLIT` (`thread_split.py:42-50`) and the
comment at `21-25`.

**Enforcement.** None, and this is the weakest point in the file. `split_source` accepts a kernel
on a regex hit (`_FOR`, `thread_split.py:56`) plus a literal integer bound
(`thread_split.py:133-134`). It cannot distinguish a reduction from a scatter, and it locates the
loop by **ordinal position** (`occurrence`), not by identity. Any change in the emitted source
that inserts or removes a `for (int X = 0; X < N; X++)` ahead of the intended loop retargets the
split silently — to a different loop, possibly a reduction, possibly with a different body.

**Numeric signature.** Splitting a reduction either reorders an fp32 sum (last-bit differences,
invisible in a token stream) or, if a shared accumulator is zeroed per group, drops 3/4 of the
terms (large, but still invisible to `generated_ids`, which `threadsplit/README.md:136-137`
records as *not* a valid oracle). It is also invisible in timing: a wrong kernel is not slower.

### I4 — the fallback runs all groups exactly once

**Mechanism.** `thread_split.py:103-104`: `if (fp(...) != 0) for (g = 0; g < T; g++) _part(arg, g);`.
Covers `0..T-1`, once each.

**Enforcement.** This is correct only because `rk_run` returns nonzero *before* it publishes
anything — `rkpool.c:83` (bad group count) and `rkpool.c:89` (pool failed to start) both precede
the publish loop at `91-97`. If a later edit adds a failure return after `91`, the fallback
re-runs groups that already ran.

**Numeric signature.** For these kernels, re-running a disjoint iteration writes the same value to
the same address, so a partial-fallback bug is invisible numerically and appears only as a timing
loss. The invariant therefore depends on the *stricter* property "no group has started when
nonzero is returned" (`rkpool.c:9-11`), which is stated in the pool's header comment but not at
the call site. An accumulating kernel allowlisted later would be biased by exactly the fraction
that ran twice.

### I5 — dispatch is all-or-nothing

**Mechanism.** `rk_run` waits on every worker it published (`rkpool.c:103-104`) before returning
0, so the entry cannot return before the workers finish. What it does *not* verify is that the
workers ran the task it published — see R1/R3/R4.

### I6 — the ABI is unchanged

**Mechanism.** The entry is `void NAME(unsigned char *restrict a)` (`thread_split.py:93`),
identical to the renderer's (`docs/rknpu/results/threadsplit/orig.c:2`); `ctx[0] = (void*)a` (95)
and `_group` casts back (108); `_part` reads the same packed-args offsets. `_PRELUDE` adds only
typedefs (`58-61`), so no header is needed and `-nostdlib` still applies
(`compiler_cpu.py:21-22`).

**Signature of a violation.** None — a broken ABI is a crash or garbage, never a subtle delay.
That is why an ABI regression is not what this audit is worried about.

### I7 — the pool address is `rk_run` in *this* process and *this* build

`pool_address()` (`thread_split.py:64-76`) resolves it once per compile and formats it as a
decimal `ULL` into the source (`99-102`, fed from `141`). Neither the process nor the build
identity is part of the disk-cache key — §3.

---

## 2. Race conditions in `rkpool.c`

### R1 — payload fields are read without an acquire *(wrong result)*

`rkpool.c:26-30` makes `seq` and `done` volatile and leaves `fn`, `args` and `group` ordinary.
The worker spins on `rk_slot[idx].seq` (`rkpool.c:68`) — a plain volatile load, which is not an
acquire — then reads `.group`, `.fn`, `.args` at `rkpool.c:72-74`. The publisher uses a release
store (`rkpool.c:96`), which orders its own prior writes, but there is no matching acquire on the
consumer side. The spin body is only `wfe` (an asm with no `"memory"` clobber, `rkpool.c:69`), so
nothing stops a compiler from hoisting the three payload loads above the spin, and nothing stops
the core from reordering them on AArch64.

Consequence: a worker can call the previous dispatch's `fn` with the new `args`, or the right `fn`
with the previous `group`. On the very first dispatch that is a call through `fn == NULL`
(`.bss` is zero, `rkpool.c:25-31`). This is a wrong result or a crash, not a delay. It is not
known to be live — clang currently reloads after the indirect call — but it is the kind of hazard
that reappears with a compiler upgrade, which is the worst kind.

Fix: `while (__atomic_load_n(&rk_slot[idx].seq, __ATOMIC_ACQUIRE) == seen) ...`, and read the
payload through `__atomic_load_n(..., RELAXED)` or make those fields volatile as well.

### R2 — the group index indexes a 4-element array unchecked

`rkpool.c:72` writes `rk_cpu[rk_slot[idx].group]`. Values come only from `rkpool.c:95`
(`group = i`, `i < workers <= 3`), so it is safe today; a future dispatch path that publishes a raw
group from the kernel turns this into an out-of-bounds write into `.bss`.

### R3 — re-entrancy: a pooled kernel calling another pooled kernel *(wrong result or hang)*

There is no depth counter, no owner field and no recursion guard anywhere in `rk_run`
(`rkpool.c:80-106`). The path is reachable: the renderer emits `extern void other();` for a linked
CALL body (`cstyle.py:279`) and `CPUProgram` links those symbols (`ops_cpu.py:19-20`), so a
allowlisted kernel can contain a call to another kernel — and `attn.c:92` and
`thread_split.py:100` call the *same* `rk_run`.

If a `_part` body calls a second pooled kernel, the inner `rk_run`:

- publishes slots `0..T-2` while the outer dispatch's workers are still inside those same slots;
- overwrites `rk_slot[i].fn/args/group` under a running worker (R1 makes that read torn);
- resets `rk_slot[i].done = 0` (`rkpool.c:92`) while the **outer** caller is waiting on that same
  counter (`rkpool.c:103-104`).

Two outcomes, both reachable: the inner call is satisfied by a `done` increment that belongs to
the *outer* task and returns before one of its groups ran (a quarter of that kernel's output never
written — a wrong result with no diagnostic), or its wait lands on a slot whose worker is itself
blocked in the inner `rk_run` (deadlock — visible, at least).

Live today? No allowlisted kernel contains a CALL, and `attn.c`'s pooled kernel is invoked as its
own program. It becomes live the moment a fused or linked kernel is added to `THREAD_SPLIT`.

### R4 — two threads calling `rk_run` concurrently *(wrong result)*

`rk_started` and `rk_failed` are plain `int`s (`rkpool.c:44-45`); `rk_slot[]` is one global array;
`done` is one counter per slot, reset by whoever publishes (`rkpool.c:92`). Two concurrent
dispatches interleave those resets and waits, and each can return on the other's completion.

Is it reachable? `CPUProgram.__call__` invokes through `ctypes.CFUNCTYPE` (`ops_cpu.py:46`), which
releases the GIL for the duration of the call, so the device is *not* serialized by the
interpreter. Nothing in this tree issues CPU kernels from two threads today (`run_linear` walks
calls on one thread, `tinygrad/engine/realize.py:281`), so it is latent. It stops being latent the
first time a kernel is issued from a second thread.

### R5 — a worker mid-task when the next dispatch arrives

Cannot happen from a single caller: `rk_run` waits on `done` (`rkpool.c:103-104`) before returning,
so the next dispatch is a later call. If it did happen (R3/R4), the publish would clear `done`
under the running worker and that worker's increment (`rkpool.c:76`) would be counted as the new
task's completion — the R3 failure, not a separate one.

### R6 — dispatches with different `groups` share the slot array

Slots at or above `workers` are neither reset nor published (`rkpool.c:91-97`), so their `done`
stays at 1 from an earlier dispatch. Nothing waits on those slots in the smaller dispatch, so this
is harmless *today*. It is harmless only because `done` is reset inside the same call, before the
wait (`rkpool.c:92` vs `103`). Any code that samples `done` outside `rk_run` reads stale state.

### R7 — the interval arrays are single-buffered *(metadata is wrong, never a result)*

`rk_cpu`, `rk_start` and `rk_end` are indexed by group and overwritten by every dispatch
(`rkpool.c:72-75`, `99-102`). `rk_t0` is reset at the *top* of `rk_run` (`rkpool.c:82`), **before**
the range check at `83`, so a refused call also invalidates the previous dispatch's intervals —
after a refusal every `rk_group_start(i)` comes out negative. Any read after the fact describes
the last dispatch, not the one you think; `docs/rknpu/results/scanprobe/overlap.c:23` reads them
that way. The `double` stores cross a thread boundary with no synchronisation (atomic in practice
on AArch64 for aligned 8-byte accesses).

### R8 — the accounting counters are broken *(diagnostics are wrong)*

`rk_calls_total` is declared (`rkpool.c:47`) and returned (`49`) and **never incremented anywhere
in the file**. The shipped library proves it: `objdump -d librkpool.so` shows `rk_calls` as
`mov x0, xzr; ret` at `0x8c4`, and `rk_calls_total` does not exist as a symbol at all. Every
`[rkpool] dispatch calls N` line ever printed reads 0, whatever the pool did.

`rk_refused_total` counts only the failed-start path (`rkpool.c:89`). The other refusal —
`groups > RK_MAX` at `rkpool.c:83` — returns -1 without counting either. A `THREAD_SPLIT_THREADS = 5`
run is therefore silently serial and leaves no trace.

This matters twice over. `splitcheck/README.md:20` and `threadsplit/README.md:5` both cite
`dispatch calls 0` as the evidence that the pool was never reached; that line cannot distinguish
pooled from un-pooled. And `inmodel_program_cost.py:151-153` prints it as a health signal.

### R9 — the caller is unpinned, the workers are pinned *(slower, never wrong)*

Workers are pinned one per core at `rkpool.c:58-61`. The caller is said to keep CPUs 4-7 to itself
(`rkpool.c:54-57`) but never calls `sched_setaffinity`, so it can land on cpu4 and share that core
with worker 0. The loss is up to 2x of the gain, visible only in `rk_group_cpu()`.

### R10 — `groups == 0` means 4

`rkpool.c:81` normalises 0 to `RK_MAX` before the bounds check (`83`). A caller that passed 0 while
its own partition divided by `threads` would divide by zero in the *emitted source*
(`thread_split.py:138`) and fail at compile time — loud. `groups > 4` is the silent one.

**Summary: R1, R3 and R4 can produce a wrong result. R2 is a latent out-of-bounds write. R5–R7
and R9 cost time or corrupt metadata. R8 invalidates the diagnostic the project used to convict
mistake #1.**

---

## 3. Cache-key hazards

### Mechanism, traced end to end

- Cache table name: `compile_clang_obj_arm64_native_<VERSION>` —
  `Compiler.__init__(cachekey)` (`tinygrad/renderer/__init__.py:13`) fed by
  `ClangCompiler.__init__` (`compiler_cpu.py:16`); `VERSION = 25` (`tinygrad/helpers.py:408`).
- Cache key: **the source string and nothing else** — `diskcache_get(self.cachekey, src)`
  (`renderer/__init__.py:16`) wraps it as `{'key': src}` (`helpers.py:426-435`). Table lives at
  `CACHEDB = ~/.cache/tinygrad/cache.db` (`helpers.py:405-406`). Caching is on by default:
  `CCACHE` defaults to 1 (`helpers.py:294`) and `CACHELEVEL` to 2 (`helpers.py:258`).
- `install()` transforms *inside* the patched call (`thread_split.py:190-191`), so the key is the
  **transformed** source, i.e. it contains the decimal address of `rk_run` in this process
  (`thread_split.py:99-102`, from `141`). `attn_native.py:54` bakes the same address into every
  attention source.

### Can a cache hit from a previous process execute a stale address?

A hit requires byte-identical source, and the source contains the address, so a hit requires the
previous process to have used the **same numeric address**. Then:

- if `librkpool.so` is the same file and happens to be mapped at the same base again, the call is
  correct (same base + same offset);
- if `librkpool.so` has been **rebuilt** since that object was cached, the same base no longer
  implies the same `rk_run` offset, and the kernel jumps to some other address in the library.
  Nothing in the key identifies the build: not a hash, not a size, not the symbol offset. Only
  `Path.exists()` is checked (`thread_split.py:73-74`, `179-180`).

**Consequence of that case**, in order of likelihood: SIGSEGV inside the kernel (the process dies
mid-trial; `bench_llm` has already rewritten `report.json` after the previous trial,
`bench_llm.py:227`); a jump into a function that returns nonzero (the fallback runs, so the result
is right and only a suspiciously slow number appears); or a jump into a function that returns 0
without doing the work — **the output is never written and the run reports a fast, wrong result**.
The third is the one this design cannot detect.

### The larger, opposite consequence

Because the address varies per process, the key varies per process, so the cache essentially never
hits across processes: every run clang-compiles every allowlisted kernel (and every attention
program) from scratch. That cost is outside the timed region for a 1+3 or 3+3 protocol, but it is
*inside* the first trial for `--quick` (`bench_llm.py:64-68`, warm-ups default to 0). It also means
`splitcheck/README.md:20`'s "fresh cache each time" was a property of the design, not a controlled
variable.

**Poisoning is impossible.** A worker process that compiles the untransformed source writes the
untransformed key (`worker.py:41`, `realize.py:239`); the parent's transformed key is a different
string. The disk cache can never return an untransformed object for a transformed key. The failure
mode is therefore "this run was never transformed" (§4), not "this run used the wrong object".

### Recommendation

1. **Do not bake the address.** The cache-safe form is a normal `extern int rk_run(...)` call
   resolved by `jit_loader` (`elf.py:9-14`, `ops_cpu.py:19-20`); the pool is already registered
   into `DLL._loaded_` for exactly that purpose (`thread_split.py:183-185`). That is what the
   checked-in `splitcheck/transformed.c` calls, what `attn.c:81-82` declares, and what
   `install()`'s own docstring claims is implemented (`thread_split.py:172-175`) — while the code
   at `99-102` does the opposite. The docstring and the code disagree; at least one of them is
   wrong in a way a reader cannot resolve from the file.
2. If the address must stay baked: mix a hash of `librkpool.so`'s *contents* into the emitted
   source, so a rebuild changes the key; and prefer passing the address as a runtime scalar
   through the packed args, which makes the source process-independent and the key stable.
3. Export and verify a version symbol in the pool. The kernel types the callee itself
   (`thread_split.py:59-60`) while the definition is `rkpool.c:80`; nothing detects a mismatch.

---

## 4. Measurement validity: every way a run is silently un-transformed

The transform exists only in the process that called `install()`, because it is a class-attribute
patch (`thread_split.py:186-194`).

| # | path | what happens | what is visible afterwards |
| --- | --- | --- | --- |
| U1 | `PARALLEL != 0` with more than one kernel to compile | `lower_and_compile` hands compiles to spawn workers (`realize.py:239`); the pool is `None` only when `PARALLEL == 0` (`worker.py:41`). Workers have no patch: the untransformed object is compiled, cached under the untransformed key, and **executed by this run**. `threadsplit/README.md:64-70` records 7 of ~30 kernels affected | `[thread_split] transforms applied: NONE` or a short list, printed by an `atexit` hook (`thread_split.py:196-201`) *after* `report.json` was written (`bench_llm.py:221`, `227`) |
| U2 | `install()` never called (running `bench_llm.py` rather than `bench_threaded.py`) | no patch at all | same line prints `NONE`; the run is a baseline carrying the arm's name |
| U3 | `install()` called after anything was compiled | `to_program_cache` is a process-global keyed on AST+renderer+config (`tinygrad/codegen/__init__.py:536-541`) and `lower_and_compile` skips any key already in it (`realize.py:235`). Those kernels never reach `compile_cached` again and stay untransformed for the life of the process | nothing: `CALLS`/`HITS` simply never mention them |
| U4 | a non-CLANG CPU renderer (`DEV=CPU:LLVM`) | `select_renderer` picks `CPULLVMRenderer` (`device.py:327-332`, `ops_cpu.py:61`), whose compiler is not `ClangCompiler` | `NONE` — and the arm is not comparable anyway, since `Device.DEFAULT` string and renderer both changed |
| U5 | a kernel on the allowlist whose source does not match | `_ENTRY` requires the exact packed ABI (`thread_split.py:55`); `_FOR` requires `for (int X = 0; X < N; X++)` with a literal bound (`56`, `134`) | a **partial** `HITS`: the run looks transformed because other names fired |
| U6 | the transform fires but the entry is not the first definition (history #1) | the loader runs the helper | nothing at all; needs §5 |
| U7 | a second `install()` returns early | `_rk_split_installed` (`thread_split.py:187`). The patch is already present, so it is harmless today — but it means a `THREAD_SPLIT` edited after the first install is live (read at compile time, `159`) while `HITS` already contains counts from the previous arm | nothing |
| U8 | `THREAD_SPLIT_THREADS > 4` | `rk_run` refuses (`rkpool.c:83`) and every call serialises (`thread_split.py:104`). Output correct, parallelism absent | nothing: `rk_refused` does not count this path (R8) and `rk_calls` is a constant 0 |
| U9 | a name fires that the run never executes (shape not reached at this prompt length) | `HITS` counts compile-time transform invocations (`thread_split.py:162-164`), not executions | nothing |
| U10 | a source containing two allowlisted kernels | `transform` returns on the first hit (`thread_split.py:159-164`) | the second name is absent from `HITS` |

### The one assertion that should be required after every run

```python
assert EXPECTED <= set(thread_split.HITS), f'untransformed: {EXPECTED - set(thread_split.HITS)}'
```

Four details decide whether it is worth having:

- **`EXPECTED` is a literal list of the kernels this run must execute pooled**, not
  `set(THREAD_SPLIT)`. The allowlist is a superset of what a given model and prompt length reach
  (U9); asserting the superset would fail spuriously and train the reader to ignore it.
- **Subset, not count.** `HITS[name]` counts transform *invocations*, and the transform runs before
  the cache lookup (`thread_split.py:190-191`), so one kernel can be counted more than once. The
  claim worth asserting is "every required name fired at least once".
- **Before `write_report`, as a hard failure.** Today the report is written at `bench_llm.py:227`
  and the transform count is printed by `atexit` (`thread_split.py:201`). A `report.json` that
  looks like a valid arm therefore exists even when the run prints `transforms applied: NONE`.
- **It does not cover U6 or U8.** U6 needs the §5 test; U8 needs a working pool counter
  (R8/D1).

Second, cheap, and worth doing in the same place: write
`{'thread_split': sorted(HITS), 'threads': THREAD_SPLIT_THREADS, 'rkpool_sha256': sha256(librkpool.so)}`
into `report.json`. The current field list (`bench_llm.py:182-220`) has no record that the arm was
an arm — the only provenance is a stdout line that a later reader of the artifact never sees.

---

## 5. Minimal acceptance test

Design rules, each of which exists because of a recorded failure:

- values are read back through the tensor returned by `custom_kernel`
  (`tinygrad/tensor.py:214-220`), never through the buffer that was passed in — reading the input
  buffer is what produced the retracted conclusions in `threadsplit/README.md:11-14`;
- the output is **poisoned**, not `Tensor.empty`, because `Tensor.empty` reuses freed storage and
  an unwritten location is then indistinguishable from a correct one (`splitcheck/README.md:39-41`);
- the structural check runs before the kernel, so an entry-order regression fails with a message
  instead of a segfault;
- the oracle is proved by a serial control in the same script.

```python
#!/usr/bin/env python3
'''thread_split acceptance test: one transform, two calls, one element-wise compare.

Catches: entry-not-first (one quarter of the loop, once), a group that never runs from a lost
done, a gap or overlap at a group boundary, a fallback that runs nothing, a kernel compiled
untransformed, and a stale buffer read as a result.
'''
import re, platform
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
import thread_split

N = 4096                                  # N % 4 == 0; > 4*MIN_ITERS; one call is microseconds
NAME = 'split_probe'
EXPECT = [i * 3 + 7 for i in range(1, N + 1)]           # input is 1..N

SRC = f'''void {NAME}(unsigned char *args) {{
  int *out = *(int **)args;
  const int *in = *((const int **)args + 1);
  for (int i = 0; i < {N}; i++) out[i] = in[i] * 3 + 7;
}}
'''

def first_definition(src):
  # no backslashes to escape: [ ]+ stands in for the whitespace class, [A-Za-z_] for a word char
  m = re.search(r'void[ ]+[A-Za-z_][A-Za-z0-9_]*[ ]*\([^)]*\)[ ]*\{', src)
  return m.group(0) if m else '<none>'

def build(src):
  arch = {'aarch64': 'arm64', 'amd64': 'x86_64'}[platform.machine().lower()]
  r = ClangRenderer(Target('CPU', arch=arch + ',native', renderer='CLANG'))
  out = Tensor.full((N,), -1, device='CPU', dtype=dtypes.int32)      # poison, do not use empty
  inp = Tensor.arange(N, device='CPU', dtype=dtypes.int32) + 1
  def fxn(o, a):
    sink = UOp.sink(o, a, arg=KernelInfo(NAME))
    info = replace(ProgramInfo.from_sink(sink, r.target), outs=(0,), ins=(1,))
    return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(o, a)),
              UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=r.compiler.compile_cached(src))))
  return out.custom_kernel(inp, fxn=fxn)[0]        # read the RETURNED tensor, not out/inp

thread_split.install()
thread_split.THREAD_SPLIT.pop(NAME, None)
try:
  # 0) negative control: the expected values are the serial semantics, with no transform
  assert build(SRC).tolist() == EXPECT, 'serial control failed: the oracle is wrong'

  # 1) structural: the entry is the first definition in what will be compiled
  t = thread_split.split_source(SRC, NAME, 0, thread_split.THREAD_SPLIT_THREADS)
  assert t is not None, 'split_source refused the test kernel'
  assert NAME in first_definition(t), f'entry is not first: {first_definition(t)!r}'

  # 2) behavioural: transform via the patch, every element written exactly once, twice
  thread_split.THREAD_SPLIT[NAME] = 0
  for call in (1, 2):                     # the second call catches cross-call state
    got = build(SRC).tolist()
    bad = next((i for i, (g, e) in enumerate(zip(got, EXPECT)) if g != e), None)
    assert bad is None, f'call {call}: first bad index {bad} got {got[bad]} want {EXPECT[bad]}'

  # 3) the transform actually fired end to end (compile-time observable, not execution)
  assert thread_split.HITS.get(NAME, 0) >= 1, 'no transform: the kernel was compiled untransformed'
finally:
  thread_split.THREAD_SPLIT.pop(NAME, None)
print('thread_split acceptance: PASS')
```

What each failure produces:

| failure | outcome |
| --- | --- |
| helper emitted before the entry (mistake #1) | check 1 fails by name. Without check 1, check 2 fails: the loader calls `_group(args)`, which takes `ctx[0]` (the *output* pointer) as the args array and dereferences the poison `0xFFFFFFFFFFFFFFFF` — SIGSEGV, or an all-poison result |
| one group never runs (R3/R4) | a quarter of `got` is still `-1`; the reported index is a multiple of 1024 and names the boundary |
| a gap or overlap in `N*g/4` | same shape of failure; the index says which boundary |
| fallback runs nothing | every element is `-1` |
| kernel compiled untransformed (U1-U5) | check 3 |
| stale buffer mistaken for a result | the poison makes an unwritten slot a failure rather than a pass |
| a group runs twice | **not detected**, by construction: the writes are idempotent. This is exactly why I4 needs the pool counter fixed (D1) rather than another end-to-end test |

Cost: one clang invocation of ten lines of C plus two kernel calls. No model, no GGUF, no NPU. It
runs in well under the ten-second budget, which is the point — a test that costs a minute is a
test that gets skipped.

---

## 6. Defect register, by severity

| # | defect | location | wrong result? | fix |
| --- | --- | --- | --- | --- |
| D1 | `rk_calls()` is a compile-time zero, so the evidence that convicted mistake #1 is void | `rkpool.c:47,49`; `mov x0,xzr` in `librkpool.so` @ `0x8c4` | diagnostics only | increment `rk_calls_total` on the success path before `rkpool.c:91`; add a `rk_groups_done` counter |
| D2 | `fn/args/group` read without an acquire | `rkpool.c:68-74` vs `96` | **yes**, stale task | acquire load for `seq`, atomic loads for the payload |
| D3 | re-entrant `rk_run` resets `done` under a live dispatch | `rkpool.c:92`, `103-104`, no guard | **yes**, lost group; or hang | refuse a nested call, or use per-depth slots |
| D4 | concurrent `rk_run` from two threads | `rkpool.c:44-45`, `84-88`, `92` | **yes** | atomic `rk_started`, or serialise with a CAS |
| D5 | `rk_run`'s address baked into the source | `thread_split.py:99-102`, `141`; `attn_native.py:54` | **yes** if the .so is rebuilt and the base repeats | call by symbol, or hash the .so into the key |
| D6 | `install()` docstring describes a design the code does not implement | `thread_split.py:172-175` vs `100-102` | misleading, costs a reader an hour | rewrite the docstring or implement it |
| D7 | nothing asserts the entry is first | `thread_split.py:140-143` | **yes** (history #1) | assert `preamble` has no `{`; assert the first definition is the entry |
| D8 | the loop is selected by ordinal, not identity | `thread_split.py:56`, `131-132` | **yes**, silent retarget | pin `(name, occurrence, var, bound)` and fail on a mismatch |
| D9 | `N*g` can overflow `int` in the emitted source | `thread_split.py:138` | **yes** for `N >= 715,827,883` at `T = 4` (`N*(T-1) > INT_MAX`) | emit `(long)N*group/threads`, or reject `N > 5e8` |
| D10 | `THREAD_SPLIT_THREADS > 4` is silently serial | `thread_split.py:52`, `103`; `rkpool.c:83` | no, slow | assert `1 <= THREAD_SPLIT_THREADS <= 4` at import |
| D11 | an un-transformed run still writes a valid-looking `report.json` | `bench_llm.py:182-227` vs `thread_split.py:196-201` | **invalid measurement** | the §4 assertion before the report, plus provenance fields |
| D12 | 3 of the 5 allowlisted kernels have no parity artifact | `THREAD_SPLIT` `45-49`. Parity exists for `E_393216_3` (`threadsplit/README.md:28-31`) and for `E_2_16_32_32_4_4` (`splitcheck/README.md:5-14`); nothing for `E_1024_32_4_4`, `E_32_32_128_4`, `r_32_16_4_32_4` | **yes** for `r_32_16_4_32_4`: its correctness rests on an accumulator argument (`loop-classification.md:285-330`), not on a measurement | parity each allowlisted kernel at the model's real geometry before enabling it |
| D13 | `DUMP_DIR: pathlib.Path` with no `import pathlib` | `thread_split.py:148` vs `37` | no | add the import; it works today only because of `from __future__ import annotations` (`35`) |
| D14 | `librkpool.so` identity is unchecked | `thread_split.py:73-74`, `179-180` | **yes** if stale | export and verify a version symbol |

D1 first because it is the only defect that has already caused a wrong *conclusion* to be
published, and it will cause another one the next time someone reads `dispatch calls 0`.

---

## 7. Claims that must not be quoted until they are re-established

- **'The pool is never reached through `jit_loader`.'** (`splitcheck/README.md:16-25`,
  `threadsplit/README.md:5`.) Its stated evidence is `dispatch calls 0`, and that counter is a
  compile-time zero (R8/D1). What remains is a timing: 1.89 ms/call against 6.46 ms serial in-model
  and 6.78 ms in the standalone driver — one quarter of the work, which is *also* the signature of
  the helper-before-entry bug, and the emission of that era was helper-first and file-scope-static.
  The checked-in `threadsplit/split.c` is that emission: the worker is defined above the entry and
  the pool state is `static`. The negative result is confounded and does not establish that a
  symbol call cannot work.
- **'The address must be baked.'** (`thread_split.py:96-98`.) Same source, same confound. It may be
  true. No artifact in the tree demonstrates it.
- **`dispatch calls N` / `refused M`** in any form, as a health signal.
- **Any `thread_ms/wall` ratio derived from `rk_group_*`.** Single-buffered per group (R7),
  invalidated by a refused dispatch (`rkpool.c:82-83`), and the caller's own interval starts after
  the workers were already woken (`rkpool.c:98-100`).

---

## 8. Adjacent instance of the same failure mode (out of scope, flagged)

`extra/rockchip/attn.c` defines `exp2_scalar` (`27`), `attn_row` (`48`) and
`rockchip_attn_group` (`84`) **before** its entry `rockchip_attn` (`88`), while its header comment
claims the entry is 'the only non-static function' (`attn.c:19`). Offset 0 is not the only
non-static function; it is the *first* function. Applying the mechanism confirmed on
`splitcheck/transformed.o` (entry at `.text+0`, helpers after) to that source, the loaded entry
would be `attn_row` or `rockchip_attn_group`, not `rockchip_attn`.

This is an **inference from source order plus the symbol layout of a different object, not a
measurement**; settling it needs `objdump -t` on the object this program actually compiles, which
is not in the tree. It is flagged because `ATTN_KERNEL = True` (`tinygrad/llm/model.py:181`), so if
the inference is right, every prefill figure produced with the attention kernel is suspect. I did
not run, compile or time anything here.

Also stale: `loop-classification.md:452` says `THREAD_SPLIT` holds only `E_393216_3: 0`; the code
holds five entries (`thread_split.py:45-49`).
