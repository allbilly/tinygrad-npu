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

**NAMED: it is the token-embedding lookup.** `tinygrad/nn/__init__.py:360-362`:

```python
def _embedding_fwd(weight, idx):
  arange = Tensor.arange(weight.shape[0])                       # 248,320
  return (arange == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2, dtype=weight.dtype)
```

That is a mask over the **entire vocabulary** plus a masked select of the Q8_0 weight row -- exactly
the emitted kernel (compare a runtime index against vocab positions, write the weight byte or zero,
64 tiles of 3880 = 248,320). So one token fetch of 2,560 values costs **1.60 ms**, 0.94% of the wall,
and the reference profile has no counterpart because their lookup is not O(vocab).

`USE_ATOMICS` does not help: it is a ContextVar (`helpers.py:300`), so the env var I first tried did
nothing, and `_embedding_fwd_fxn` calls the same `_embedding_fwd` anyway.

**The obvious fix is measured and does not work:** `weight[idx]` (advanced indexing) is correct but
costs **5.37 ms** per call on a 1,000-row table in isolation -- it lowers to the same scan. This tree
has no gather op (`Ops.GATHER` does not exist). So the fix is a native Q8_0 row-dequant kernel, in the
shape of `argmax_native.py`: read the token id, dequantise `depth/32` blocks (34 bytes each) into the
output row. Expected **1.60 -> ~0.01 ms**.

The compile-time stack only reaches `tinygrad/llm/model.py:967` (`Transformer.__call__`), because the
kernel is compiled during realise. Naming the op needs the sink uop (`DEBUG=4`) for that
kernel; guessing it from the arithmetic would be the fourth guess in this stretch, so it is left as the
one open item with a measured 1.60 ms attached.

For scale: 1.60 ms is 0.9% of the token -- ten times the norm+quantise fusion's realistic capture, which
is why this was traced before that port was started.

### E_80_32_64_4: real, reproducible, and still unnamed after six approaches

| check | result |
| --- | --- |
| itemiser, two runs | **1.59 / 1.60 ms**, 1 call/token |
| fresh schedule (`SCACHE=0`) | **still 1.60 ms** -- not a stale cache artefact |
| parity with `SCACHE=0` | `895365e27ed4255b` unchanged |
| ablation: `ARGMAX_KERNEL=1` vs `0` | present in both -- not argmax |
| `_cache_prefix` multi-batch mask | needs `len(start_pos) > 1`; decode passes one -- not it |
| `_one_hot_along_dim` in core | does not exist outside `nn/onnx.py` -- so the "one-hot" reading is also wrong |
| `to_program` / `do_render` hooks | **never fire for it**, with or without `SCACHE=0` |
| `CPUProgram.__init__` hook | fires, stack = `realize.get_runtime` at **decode**, i.e. the PROGRAM was built earlier |

**Why tracing failed, mechanically:** the program is built during the JIT capture (symbolic shapes) and
only *loaded* at decode, so the concrete source with `Lidx1_0 < 80` and `Lidx2 < 64` never passes through
`do_render` in a decode run -- and in a capture run the loop bounds are still symbolic. Every hook I tried
was on the wrong side of that boundary.

**What is still true about it:** reads Q8_0 rows of 80 blocks (2560 values) at a runtime scalar index,
writes a byte per (row, block) and zeros the rest across 64 tiles of 3880 (= vocab/64). One call, 1.60 ms.

**The way through is a source bisect, not a trace:** tag candidate emit sites with a unique
`KernelInfo` name and see which one renames the kernel. That is an edit to the model, and it is the next
step rather than another instrument.

### Status of the two open items

| item | ms | state |
| --- | --: | --- |
| `E_80_32_64_4` | **1.60** | **named**: the embedding lookup is O(vocab); fix = native row gather |
| their components-vs-wall discrepancy | 10.32 | unexplained; not used in any claim |

Neither is claimed as a lever. The decomposition otherwise closes at
`142.34 device + 31.11 CPU - 2.69 overlap = 170.76`.

### The embedding gather: kernel built and verified, wiring rejected (tree at baseline)

`extra/rockchip/embed_native.py` is a one-row gather for the fp16 `(248320, 2560)` token table:
read the token id, convert 2,560 `__fp16` values to f32 with `half4` loads. Verified bit-exact against
`weight[idx]` for tokens 0, 5 and 999, and it costs **0.02 ms** against the mask path's **1.59** -- 80x.

**Both wiring forms fail, and the tree is back at 167.25 ms CPU / 170.76 wall with parity intact.**

| wiring | result |
| --- | --- |
| `Tensor.custom_kernel(...)` at the graph root | **640-782 ms/token** -- the forward is rebuilt and recompiled every token, because the fxn sits at the entry of the captured graph. `argmax_native` gets away with `custom_kernel` because it is the *last* op, not the first |
| direct `prog.call(out, weight, idx)` (the `_FusedInt8Plan` form) | works for 2 decode steps, **errors at 5** |

So the 1.60 ms is real and the replacement is 80x faster in isolation, but the embedding sits at the
graph's entry, where a custom kernel defeats capture. Landing it needs a capture-safe gather -- either
`Ops.GATHER` in the backend (absent) or hoisting the lookup out of the forward into the caller, which
is a model-interface change rather than a kernel.

**Net effect of the attempt: zero, deliberately.** Nothing is shipped; `EMBED_KERNEL` is not in the
tree and the measured baseline is unchanged.

### `weight[idx]` in-model: same kernel, no gain (and my isolated 5.37 ms was wrong)

Advanced indexing was the obvious capture-safe replacement, since it is a plain op rather than a
custom kernel. Tested in-model, parity held:

| lookup form | total CPU | `E_80_32_64_4` |
| --- | --: | ---: |
| `nn.Embedding(idx)` (mask) | 167.25 ms | **1.59 ms** |
| `weight[idx]` (advanced index) | 167.46 ms | **1.66 ms** |

So advanced indexing lowers to the *same* masked scan, and the 5.37 ms I quoted for it earlier was an
isolated non-JIT call -- the M90 error again, this time caught before it was used.

### Why the 1.60 ms is not worth what it costs to land

With no `Ops.GATHER` in the backend, the lookup is O(vocabulary) by construction. The only route left
is hoisting it out of `Transformer.forward` into the decode caller, which changes the model interface
and the JIT's input set, for **1.59 ms = 0.9% of the wall**.

Measured against that: two itemiser runs of the *same* code today gave **167.25** and **174.92 ms CPU**
-- a 4.6% spread, five times the gain. Landing a 0.9% change behind an interface change, measured by
an instrument that varies 4.6% between runs, is not a defensible trade.

## What their per-label timings measure: wall per call, and the windows overlap

`rockchip_replay.py:460` wraps every labelled call as

```python
'{U64 begin=now();' ... 'stats[{index+2}]=now()-begin;}'     # assignment, not +=
```

and `now()` is `clock_gettime(CLOCK_MONOTONIC)` (line 470). So each label records the **wall time of
one call**, on one slot per call site -- not summed worker CPU, not accumulated.

Given that, their components exceeding their measured wall is not an accounting artefact: the windows
themselves overlap. With device submits blocking (136.62) and copies (0.59), the host wall is

    153.2 - 136.62 - 0.59 = **16.0 ms**

against **26.32 ms** of summed per-call host wall -- **1.65x concurrency**, supplied by their
`ONPU_SHARED_HOST_POOL=1` / `ONPU_HOST_POOL_WFE=1` worker pool. Run-to-run variation is ruled out:
their profiled run reported 6.500 tok/s against 6.525 unprofiled, 0.65 ms apart.

**So the host lead is real: 16.0 ms wall against our 31.11 ms serial.**

### But it is not recoverable by threading our current kernels

Our pool (`librkpool.so`, `rk_run(fn, args, groups)`, spin-then-`wfe`) is the same interface, and the
four-way split of our elementwise kernels was measured **null** (ratio 1.0031 over 12 pairs): those
kernels are 8-97 us gathers where four workers contend rather than scale. Their pool parallelism pays
on **fused C functions** that do real per-element work -- norm+quantise, swiglu pack -- whose total in
our tree is the 1.91 ms ceiling already measured.

So the 15 ms host gap is real, is theirs by design, and the only way to reach it is to port the fused
host kernels that give each worker enough work to be worth a thread. That is the same port named
earlier (`register_norm`, `_swiglu_pack`), and its capture on our side is ~2 ms, not 15.
