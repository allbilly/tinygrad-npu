# The four-way split: validated standalone, broken through jit_loader

## What works (measured)

`transformed.c` is the real `E_2_16_32_32_4_4`, transformed with `thread_split.split_source`
(entry emitted first, dispatch through `rk_run`), linked **by symbol** against `librkpool.so`
rather than through a baked address. `driver.c` feeds both kernels identically-shaped buffers and
compares every output byte:

    differing=0 verdict=BITWISE-EQUAL
    reps=10  serial 6.78 ms/call  pooled 1.76 ms/call  speedup 3.86x

Serial 6.78 ms matches the in-model baseline of 6.46 ms, so the driver is reproducing the real
kernel, and the 3.86x is consistent with the pure-C pool benchmark (3.96x).

## What does not (measured)

The same transform through `CPUProgram`/`jit_loader` never reaches the pool:

    [rkpool] dispatch calls 0  refused/serial 0      (four separate runs, fresh cache each time)

with the pool registered in tinygrad's `DLL._loaded_` so the symbol resolves (no relocation error
is raised). `E_2_16` then measures **1.89 ms/call**, and note that this is indistinguishable from
the pooled figure above: **6.78/4 = 1.70 ms**. Time alone cannot tell "pooled" from "one quarter of
the work", which is what made the earlier invalid numbers look convincing.

## The rules this established (with a working positive control)

Control file: `/tmp/opencode/control.py`. A kernel that writes `[100,101,102,103]` must read it
back; it does.

| test | result | rule |
| --- | --- | --- |
| single function | `[100,101,102,103]` | harness correct |
| entry first, helper after | `[100,101,102,103]` | multi-function objects work |
| helper first, entry after | kernel never ran (stale buffer) | **the entry must be the first definition** |
| `static int` incremented across calls | **segfault** | file-scope state cannot live in a jit-loaded kernel |

`Tensor.empty` reuses freed storage, which is why test 3 appeared to return the previous test's
values: reading an unwritten buffer is indistinguishable from a kernel that did not run. That is
the same trap that produced the earlier "address 0" and "worker never ran" conclusions.

## Unknown, and the next thing to measure

Why the call site does not reach `rk_run` through `jit_loader`. The ELF loads and the symbol
resolves; the call still does nothing. Candidates: the relocation is applied to the PLT/GOT slot
but the call binds elsewhere, or the loaded image's call targets a stub. The bounded check is to
disassemble the loaded text and see what address the call instruction actually holds, or to make
`rk_run` trap and observe whether the trap fires.

Until that is answered, **no in-model number from this mechanism is valid**, and the last valid
prefill measurement in this tree remains **2,531.67 ms** (`results/scanreuse-3plus3/report.json`).

## All five transformed kernels: bitwise parity at the function boundary

`parity4.c` gives both kernels eight identical 32 MB buffers, filled from a fixed LCG stream, and
compares **all 256 MB** afterwards -- no knowledge of each kernel's argument sizes is needed, and
any out-of-range write also shows up. Split sources are linked against `librkpool.so` by symbol
(the path proven here), and the binaries are built `-no-pie` so the symbol addresses are absolute.

| kernel | serial ms | split ms | speedup | differing bytes |
| --- | --: | --: | --: | --: |
| `E_393216_3` | 3.88 | 1.05 | 3.68x | **0** |
| `E_32_32_128_4` | 6.08 | 1.61 | 3.77x | **0** |
| `E_1024_32_4_4` | 7.20 | 1.87 | 3.85x | **0** |
| `r_32_16_4_32_4` | 3.15 | 0.88 | 3.59x | **0** |
| `E_2_16_32_32_4_4` | 6.60 | 1.71 | 3.86x | **0** |

Serials match the in-model per-call times (6.46-6.60 for `E_2_16`), so the harness reproduces the
real kernels. This is the parity check that makes the in-model timing quotable.

### Why the whole-prefill hash could not serve this purpose

Three probe runs were compared on `logits_sha256_16`:

    split-a  dd45754bda6007a4   wall 2689.0
    split-b  9a05014b91a89468   wall 1813.9
    nosplit  16d26a8ed3b2029d   wall 2422.7

`split-a != split-b`, i.e. **two identical configurations disagree**, so the model is
non-deterministic across runs and a whole-prefill hash is not an oracle. That is a property of the
benchmark, not of the transform, and it is why parity is asserted per kernel instead.

## The scan split: two harnesses disagree, so it stays disabled

`parityplan.c` compares only `out` and `state`, with small finite inputs, and tests three paths:
serial, the four `_part` calls run **sequentially** (no threads), and the pooled path.

    parts-sequential out differing=0  state differing=0
    pooled           out differing=0  state differing=0   rc=0 status=1

So the head partition itself is correct, and the pooled dispatch reproduces it on finite data.

`parity4.c` -- the generic harness that passed for all five elementwise kernels -- reports
**559,637 differing bytes, first at offset 871** for the same pair. The two harnesses differ in
two ways: `parity4` compares all 256 MB (including regions no kernel writes) and fills buffers
from a byte LCG, so its floats include **NaN, Inf and denormals**; `parityplan` uses finite values.

That points at a **data-dependent** divergence, which is plausible for this kernel (the general
path branches on `scan_disjoint` and on `direct`, and NaN changes comparisons) and unacceptable to
ignore, because a model can produce extreme values.

**Resolution (superseded):** this originally concluded that the pooled entry stays disabled. `scan_native.py` now calls `cpu_gated_delta_scan_pooled`, and the paired run above records `status=1` (all four groups completed) at 3.86x over serial, so the dispatch is live.
The next test is `parity4` with finite inputs; if it passes, the divergence is confined to
non-finite inputs and the decision becomes whether the model can produce them; if it fails, the
partition is wrong in a way `parityplan` does not model.

The wall argument for splitting the scan (593 ms at `thread/wall = 1.00`, ~330 ms available) does
not outrank a correctness question this specific.
