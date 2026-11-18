> ## History: four invalid measurements and a broken diagnostic
>
> The figures `1,848.56 / 1,929.52 / 1,933.34 / 1,854.9 ms` are **not quotable**. They were taken
> while the emitted helper sat before the entry point, so kernels ran one quarter of their loop.
> Later runs fixed that but had a second defect: `attn.c` also had a helper defined before its
> entry, so the attention kernel was executing the wrong function.
>
> The diagnostic that "proved" the pool was never reached was itself broken: `rk_calls_total` was
> declared, returned, and **never incremented**, so `[rkpool] dispatch calls 0` could only ever
> print 0. With the counter fixed, the same code reports **`dispatch calls 736, refused 0`** --
> the pool is entered four times per threaded call and never refuses. `thr/wall = 1.00` is also
> expected, not suspicious: `rk_run` waits by spinning on `wfe`, so the waiter's own CPU time
> equals the wall.
>
> Both entry orders are now fixed (`E_*` verified by source order, `attn.c` verified by definition
> order: `rockchip_attn, _group, attn_row, _part`), and `rkpool.c` uses acquire loads with memory
> clobbers. The baseline from that configuration is `results/threadsplit/pooled-3plus3/`.
>

> ## 1,402.60 ms — scan port landed, parity intact
>
> `results/threadsplit/scanpool-3plus3/report.json`, `complete: true`:
>
> | field | value |
> | --- | --- |
> | ttft | 1,404.81 / 1,402.60 / 1,398.06 ms |
> | median | **1,402.60 ms** |
> | decode | 3.4624 / 3.4603 / 3.4616 tok/s |
> | temperature | 61.9 -> 64.7 C |
>
> Evidence carried with it: five elementwise transforms applied and bitwise-equal (0 differing
> bytes over 256 MB each), the scan's pooled entry bitwise-equal to serial on finite input
> (0 differing over 256 MB, 3.37x), `logits finite: True` on every trial, and
> `[rkpool] dispatch calls 832, refused/serial 0`.
>
> Cumulative: 2,531.67 -> 1,402.60 ms, **-1,129 ms (-44.6%)**. The scan is now **178.2 ms against
> the reference's 177.4** -- that row is at parity.

> ## The first run with both entry orders fixed: 1,820.74 ms
>
> `results/threadsplit/pooled-3plus3/report.json`, `complete: true`, three trials within 0.2%:
>
> | field | value |
> | --- | --- |
> | ttft | 1,820.46 / 1,824.16 / 1,820.74 ms |
> | median | **1,820.74 ms** |
> | decode | 3.3605 / 3.3794 / 3.3727 tok/s |
> | temperature | 61.0 -> 61.9 C |
> | transforms applied | all five |
>
> That is -711 ms (-28.1%) against the 2,531.67 ms baseline, and **the pool is proven to run**
> (`736 dispatches, 0 refused`). **Parity is established at the function boundary**: all five
> transformed kernels are bitwise identical to their untransformed originals over 256 MB of
> compared memory each (`results/splitcheck/parity4.c`), at 3.59-3.86x. The whole-prefill hash
> could not be used for this -- `split-a != split-b` for the *same* configuration, so the model is
> non-deterministic across runs. **This figure is quotable.**
>
> **Superseded baseline: 2,531.67 ms** (`results/scanreuse-3plus3/report.json`), now replaced by
> the 1,820.74 ms above.

# Thread-split CPU kernels — pooled wrapper, 3.7-3.9x standalone, bitwise equal

## What is proven

`extra/rockchip/thread_split.py` rewrites a generated CPU kernel at the compiler boundary
(`ClangCompiler.compile_cached`) into a four-way **pooled** wrapper over one chosen loop, with
no change to the kernel's ABI (`void name(unsigned char *args)`) and no change to its
arithmetic. It compiles with the project's exact flags:

    clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident \
          --target=arm64-none-unknown-elf -ffixed-x18 -mcpu=native

Bitwise parity on the real `E_393216_3` kernel (1,179,648 elements, random data in [-30,30]
for the gate input and [-4,4] for the multiplier, exercising the exp2 guards):

    n=393216 elements=1179648 differing=0 verdict=BITWISE-EQUAL

Artifacts in this directory: `orig.c` (dumped emitted kernel), `split.c` (same kernel after
the transform), `parity_driver.c` (runs both on identical buffers and memcmps the output;
`./parity <n> <reps>` also times them interleaved).

## Three blockers, all measured, in the order they bit

### 1. File-scope data in a jit_loader'd kernel is at address 0

Measured with a two-line kernel through `custom_kernel`:

    static int g_x = 0;
    void probe(unsigned char *args) { g_x = 42; out[0] = g_x; out[1] = (int)(long)&g_x; }

    store->load: 0   address: 0x0   reread: 0

So **any `static` in a CPU kernel is inert** in this tree: the store goes nowhere, the load
returns 0, and a pool that parks workers in a static array can never be woken. A first wrapper
that did exactly that reported `pthread_create == 0` (success) while the worker never ran, and
the kernel silently fell back to serial — measured 0.98-1.00x on five kernels.

The reference does not hit this because its partition index is a **function argument**, not
state, and its pool lives in a separately loaded `.so`, whose statics are real. Its generated
kernel is:

    static void E_2_16_32_32_4_4_part(float* data0, float* data1, unsigned char* data2,
                                      float* data3, int group) {
      for (int Lidx0_1 = 16*group/4; Lidx0_1 < 16*(group+1)/4; Lidx0_1++) {

and the dispatcher address is **baked into the emitted source as an integer**, the same way
`rockchip_norm_quantize` embeds `{address}`.

### 2. Kernels compile in a worker pool, so an in-process patch sees almost nothing

`lower_and_compile` uses `get_worker_pool()` for multi-kernel batches, and `worker.py` returns
`None` only when `PARALLEL == 0`. Patching `ClangCompiler.compile_cached` in the parent process
therefore affected **7 of the model's ~30 kernels** — the ones compiled in single-task batches.
`transforms applied: NONE` was the tell. `PARALLEL=0` fixes it (it is also the reference's own
setting), and it only changes compile-time parallelism, not the timed path.

### 3. A thread started on an idle A76 does not run for 2-6 ticks

`CONFIG_HZ=300` with `NO_HZ_IDLE`: per-call `pthread_create` + `join` measured **6.6-11 ms per
call** against a 0.434 ms kernel. A parked worker woken by a sequence counter costs **0.58 us**.

## The pool that works: `extra/rockchip/rkpool.c` + `librkpool.so`

    clang -shared -fPIC -O2 -o librkpool.so rkpool.c -lpthread

Three workers created once, pinned to CPUs 4-7, woken by a per-slot sequence counter with
`sev`/`wfe` (bounded spin then `wfe`, so they idle cheaply while the NPU ioctl blocks). The
caller runs the last group. `rk_run` returns nonzero only before dispatching, so a failed pool
falls back to running all groups serially.

Measured standalone, 4 equal work items, `taskset -c 4-7`, three runs:

| serial 4 groups | pooled | speedup |
| --: | --: | --: |
| 64.58 ms | 16.30 ms | **3.96x** |
| 64.59 ms | 16.30 ms | **3.96x** |
| 64.61 ms | 16.30 ms | **3.96x** |

Cost of the pool call is invisible at this granularity (16.298 ms wall for 16.1 ms of work).

## In-model attribution (valid)

The transform now emits `_part(..., group)` and a driver that calls `rk_run` through the baked
address. Same probe, same protocol, `PARALLEL=0`:

| kernel | before | after | speedup |
| --- | --: | --: | --: |
| `E_2_16_32_32_4_4` | 6.46 | 1.89 | **3.41x** |
| `r_32_16_4_32_4` | 2.93 | 0.88 | **3.33x** |
| `E_32_32_128_4` | 5.94 | 1.56 | **3.81x** |
| `E_393216_3` | 4.13 | 1.07 | **3.86x** |
| `E_1024_32_4_4` | 5.67 | 1.71 | **3.31x** |
| kernel total | 2,391.3 ms | **1,772.2 ms** | -619 ms |
| traced wall | 2,549.7 ms | **1,920.3 ms** | -629 ms |

## Re-run (machine must be idle — no subagents, no other benchmarks)

```bash
cd docs/rknpu/results/threadsplit
taskset -c 4-7 ./parity 393216 20      # repeat 3x, expect <1% spread
taskset -c 4-7 ./parity 1179648 10
```

Then the in-model A/B, which is the number that matters:

```bash
cd /home/orangepi/pr/tinygrad/extra/rockchip
PARALLEL=0 DEV=CPU:CLANG ../../.venv/bin/python bench_threaded.py -- \
  --ckpt <gguf> --name qwen35-4b-rkpool --out ../../docs/rknpu/results/threadsplit/rkpool \
  --npu --warmups 3 --runs 3
```

Compare against the current baseline `docs/rknpu/results/scanreuse-3plus3/report.json`
(2,528.67 / 2,531.67 / 2,534.62 ms, `complete: true`).

`PARALLEL=0` is required: without it the kernels compile in a worker process where the
transform does not exist (blocker 2 above). It changes compile-time parallelism only.

Parity note: the split partitions an iteration space, so it is exact by construction, and the
earlier begin/end version was verified bitwise on `E_393216_3` at 1,179,648 and 3,538,944
elements. The group-indexed version uses the same partition. `generated_ids` is **not** a valid
oracle in this tree — the three baseline trials differ from each other.

## attrib22 — the fixed emission still does not pool (measured, invalid)

After fixing the entry order (verified in the object: entry at offset 0, `_group` 0x90, `_part`
0xa0) and switching the dispatch to a pointer loaded from a `volatile` local, the probe reports:

    wall 1848.0  thread_cpu 1280.8  ratio 0.69
    E_2_16_32_32_4_4   wall 91.2  thr 91.5  thr/wall 1.00

`E_2_16` is 1.90 ms/call = 6.46/4 again, and the caller owns every cycle of it. So the in-model
path still executes **one quarter of the loop, once, with no pool**, and 1,848.0 ms is not a
prefill measurement.

What this rules out, in order:

| eliminated | evidence |
| --- | --- |
| helper before entry | `readelf -sW` on the emitted object: entry at 0x0 |
| program-cache aliasing | two same-shape kernels with different sources produce different binaries |
| loader image overflow | image (112-128 B) is smaller than the object (1144 B) |
| symbol resolution | `link_sym('rk_run') == base + 0x8d8 == ctypes address` |
| constant vs loaded call | both fail in-model; the loaded form works in a standalone binary |
| the dispatch counter | reads 0 even when `rc=0` and workers write output, so it proves nothing |

This is now a single open question, and it is the one the loader-analysis subagent is answering:
why an indirect call inside a `jit_loader` image does not reach its target.
