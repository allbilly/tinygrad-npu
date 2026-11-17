# Reference archive extract: Qwen3.5 4B + optimisation methodology

Source archive: `/home/orangepi/tinygrad` (read-only). All citations are `file:line`
in that archive unless stated otherwise.

Note on the session logs: `codex-session-faster-prefil-and-decode.md` is a **strict
byte prefix** of `codex-session-01a14a74-51aa-7872-94b0-1728db28547c_fastonpu.md`
(`cmp` reports EOF on the former at byte 112,502,454 / line 1,559,693). Only the
`fastonpu` file needs to be read.

Every number below is labelled **[measured]** or **[inferred]**. Anything derived by
arithmetic from a config rather than read from a report is **[inferred]**.

---

## 1. The GDN scan kernel (`tinygrad/llm/kernels/rockchip_recurrent.c`)

### 1.1 What it computes

One FP32 Gated Delta Net (delta-rule linear attention) scan over `rows` valid time
steps, for `batch*heads` independent (B,H) lanes, updating a recurrent state in place
and writing per-step outputs.

Per step, per (b,h), with state `s` of shape `[V,K]`:

```
s1      = s * alpha            # decay, alpha broadcast over K if alpha_dim==1
pred    = sum_k s1[k] * k[k]   # scalar per value row
delta   = (v - pred) * beta    # scalar per value row
s       = s1 + delta * k       # rank-1 update, outer product over K
out     = sum_k s[k] * q[k]    # scalar per value row
```

The file header states the intent: `/* CPU GDN scan with serial key reduction and SIMD
across value rows. */` (`rockchip_recurrent.c:1`). The staged intent record gives the
same math: `"math": "Per step decay s1=s*alpha, delta=(v-sum(s1*k))*beta,
newstate=s1+delta*k, output=sum(newstate*q). State layout[B,H,V,K], reset at
position0."` (`codex-session-...fastonpu.md:370722`).

The reference Python recurrence this replaces is `tinygrad/llm/model.py:618-630`:
`state = initial.where(0, state.float())`, then a Python `for t in range(T_pad)` loop
doing exactly the four lines above.

### 1.2 Signature and argument contract

Two exported entry points and one internal worker.

```c
int cpu_gated_delta_scan(float *out, const float *q, const float *k, const float *v,
    const float *beta, const float *alpha, float *state, int start, int rows,
    int batch, int heads, int capacity, int key_dim, int value_dim,
    int alpha_dim, float *history);                                    // :454-459
```
```c
int cpu_gated_delta_scan_pooled(float *out, const float *q, const float *k,
    const float *v, const float *beta, const float *alpha, float *state,
    int start, int rows, int batch, int heads, int capacity, int key_dim,
    int value_dim, int alpha_dim, float *history,
    scan_parallel_fn parallel, int *dispatch_status);                  // :463-466
```
```c
static int cpu_gated_delta_scan_part(float *out, const float *q, const float *k,
    const float *v, const float *beta, const float *alpha, float *state,
    int start, int rows, int batch, int heads, int capacity, int key_dim,
    int value_dim, int alpha_dim, float *history, int group, int groups); // :64-67
```

The single-threaded entry is just the general worker with `group=0, groups=1`
(`:457-458`). `groups` shards the **head** axis only: `for(size_t bh=(size_t)group;
bh<bh_count;bh+=(size_t)groups)` (`:81`). Each head keeps its serial recurrence, so
head-parallelism is exact — this is what makes the pool safe.

Argument validation (`:68-70`) is a hard gate, and it defines the whole supported
envelope:

| constraint | value | line |
|---|---|---|
| `capacity` | `1..256` | `:68` |
| `key_dim` | `1..256` | `:69` |
| `value_dim` | `1..256` | `:69` |
| `alpha_dim` | exactly `1` or `key_dim` | `:70` |
| `rows` | `1..capacity` | `:68` |
| `batch`, `heads` | `>=1` | `:69` |
| `start` | `>=0` | `:68` |

Note `capacity>256` is rejected, which is why the Python wrapper caps at 256
(`rockchip_recurrent.py:94`).

### 1.3 Memory layout

All operands are FP32, contiguous, and **step-major over `bh`**:

```
q       [batch, heads, capacity, key_dim]      stride per step = key_dim
k       [batch, heads, capacity, key_dim]      stride per step = key_dim
v       [batch, heads, capacity, value_dim]    stride per step = value_dim
beta    [batch, heads, capacity]               stride per step = 1
alpha   [batch, heads, capacity, alpha_dim]    stride per step = alpha_dim
state   [batch, heads, value_dim, key_dim]     stride per bh   = value_dim*key_dim
out     [batch, heads, capacity, value_dim]    stride per step = value_dim
history [capacity, batch, heads, value_dim, key_dim]   (optional)
```

The flat step index is `step = bh*capacity + t` (`:91`), i.e. the (B,H) axis is the
outer axis and `t` is inner. `stride` and `bh_count` are computed at `:71`; the byte
sizes used for the aliasing test at `:72-73`.

`state` is **V-major**: `s[r*key_dim + c]` is value row `r`, key column `c`
(`:29`, `:430`). The kernel does not use that layout directly for the inner loops — it
transposes into a **K-major scratch buffer** `columns[]` at entry and transposes back
at exit:

```c
float columns[256*256] __attribute__((aligned(16)));          // :80
for(int c=0;c<key_dim;c++)for(int r=0;r<value_dim;r++)
  columns[(size_t)c*value_dim+r] = start ? s[(size_t)r*key_dim+c] : 0;   // :88-89
...
for(int r=0;r<value_dim;r++)for(int c=0;c<key_dim;c++)
  s[(size_t)r*key_dim+c] = columns[(size_t)c*value_dim+r];    // :430
```

So `columns[c*value_dim + r]` is key `c`, value row `r`, and the inner loops walk
`c` (serial, in order) with `r` in SIMD lanes. 256 KiB of stack scratch; the
`256*256` bound is exactly the `key_dim<=256 && value_dim<=256` gate.

`history`, when non-null, records the **post-update state of every valid step** in
`[t, bh]` order: `history[((size_t)t*bh_count+bh)*stride + r*key_dim+c]`
(`:427-428`). Padding rows are never written to history.

The tail of `out` beyond `rows` is zeroed every call:
`memset(out+(bh*capacity+rows)*value_dim, 0, (capacity-rows)*value_dim*sizeof(float))`
(`:431`). This is what makes padded steps exact no-ops for the downstream graph.

### 1.4 Loop structure — three paths

`F4` is `float __attribute__((vector_size(16), aligned(1)))` (`:4`), i.e. 4 FP32
lanes, **unaligned-tolerant**. `load_rows`/`store_rows` (`:6-13`) handle a partial
final vector by zero-filling unused lanes and scalar-copying on store.

There are three code paths selected per `value_dim`:

**(a) 64-lane path, `value_dim % 64 == 0`** (`:95-322`). The hot path. Per 64-value
block it keeps **sixteen** `F4` accumulators (`predicted0..predicted15`, `:97-112`),
so 64 independent value lanes are in flight. Two passes over `c`:

```c
// pass 1: predicted = sum_c (s[c,r] * decay) * k[c]
for(int c=0;c<key_dim;c++) {
  const float *p = columns + (size_t)c*value_dim + r;
  F4 key = (F4){kt[c],kt[c],kt[c],kt[c]};
  F4 old0 = *(const F4*)(p+0); ... F4 old15 = *(const F4*)(p+60);
  if(alpha_dim!=1) { F4 a=(F4){at[c],...}; old0=old0*a; ... old15=old15*a; }
  predicted0 = predicted0 + old0*key; ... predicted15 = predicted15 + old15*key;
}
// delta, factored exactly as the reference does it
F4 decay = (F4){at[0],at[0],at[0],at[0]};
delta0 = (value0 - predicted0*decay)*b;      // alpha_dim==1
delta0 = (value0 - predicted0)*b;            // alpha_dim==key_dim
// pass 2: s = s*alpha + delta*k, and out += s*q
for(int c=0;c<key_dim;c++) {
  float av = at[alpha_dim==1?0:c];
  F4 a=(F4){av,...}, key=(F4){kt[c],...}, query=(F4){qt[c],...};
  F4 old0 = *(const F4*)(p+0);
  F4 updated0 = old0*a + delta0*key;
  *(F4*)(p+0) = updated0;
  result0 = result0 + updated0*query;
  ... x16
}
```
(`:113-167` pass 1, `:168-219` delta, `:236-305` pass 2, `:306-321` store.)

**(b) 16-lane path, `value_dim % 16 == 0`** (`:323-394`). Same structure with four
`F4` accumulators (16 lanes).

**(c) partial path, everything else** (`:395-426`). `r += 4` with `count =
min(4, value_dim-r)` and `load_rows`/`store_rows`; one accumulator. This is the
fallback that regressed by ~1 µs in the measurement (§1.7).

Critical arithmetic detail: **the key reduction is serial in `c` and each lane keeps
its own accumulation order.** The 64-lane path shares the `k[c]` and `q[c]` broadcast
loads across 16 vectors but does not reassociate any lane's sum. This is the property
that produced bitwise equality (§1.6).

### 1.5 The direct-decode path (opt-in, default off)

`ONPU_GDN_DIRECT_DECODE` (`:17-19`, default `0`) selects `direct_row` (`:27-57`) when
**all** of these hold (`:75-79`):

- `rows==1 && capacity==1 && start>0`
- `!history`
- `alpha_dim==1`
- `key_dim%4==0 && value_dim%4==0`
- `state` and `out` ranges are each disjoint from every other operand, checked by
  `scan_disjoint` (`:59-62`) over 12 pairs

Rationale in the source comment: *"The general path snapshots state. Preserve its
behavior for aliased inputs."* (`:74`). `direct_row` avoids materialising the full
`columns[]` transpose: it works on 4×4 state tiles with `transpose4` SIMD shuffles
(`:21-26`), accumulating keys in original order, then writes the same rows back
(`:31-54`). Documented at `extra/rockchip/recurrent_direct_decode.md:3-5`.

### 1.6 How it is called

Python side: `tinygrad/llm/kernels/rockchip_recurrent.py`.

`_scan(...)` (`:19-42`) reads the `.c` file, rewrites the two exported names to
`static` so only the wrapper is visible (`:22-23`), builds the argument list, and
emits a `rockchip_cpu_gated_delta_scan` wrapper:

```python
arguments = f'out,q,k,v,beta,alpha,state,*start,*rows,{batch},{heads},{capacity},{key_dim},{value_dim},{alpha_dim},' + \
  ('history' if history is not None else 'NULL')                       # :24-25
call = f'if(cpu_gated_delta_scan_pooled({arguments},(scan_parallel_fn){pool}ULL,NULL))__builtin_trap();' if pool is not None else \
  f'cpu_gated_delta_scan({arguments});'                                # :28-29
source = f'#define ONPU_GDN_DIRECT_DECODE {int(direct_decode)}\n' + body + f'''
void rockchip_cpu_gated_delta_scan(float *out, const float *q, ..., const int *start, const int *rows
    {', float *history' if history is not None else ''}) {{ {call} }}'''   # :30-36
```

`start` and `rows` arrive as **device pointers to one int each**, not scalars
(`:105-106`: `Tensor(start_pos,...).reshape(1).clone()`), so the same captured graph
serves every position and row count. Outputs are declared `outs=(0,6,9)` or `(0,6)`
(`:40`) — i.e. `out`, `state`, and optionally `history` are written in place.

`gated_delta_scan(...)` (`:83-115`) is the public hook. It validates device (CPU only),
dtype (FP32 only), ndim, exact shapes, `1<=key_dim<=256`, `1<=value_dim<=256`,
`1<=capacity<=256`, and that `state` is a whole storage (`_whole_storage`, `:45-48`,
which unwraps `RESHAPE`/`AFTER` and requires `BUFFER`/`PARAM` with matching numel).
It then calls `out.custom_kernel(...)` with the operands made contiguous.

`configure(model, enabled, pool=...)` (`:51-80`) installs the hook on every
`GatedDeltaNetBlock` and **refuses** any block whose `attn_qkv.weight` is not CPU, or
whose `head_k_dim`/`head_v_dim` is outside `1..256`, or `num_v_heads<1` (`:69-71`).
Installation is idempotent (`:73-78`). It also forces
`Tensor.realize(block.conv_state, block.recurrent_state)` (`:79`) so the state buffers
exist before capture.

Model integration: `model.py:611-613` — the hook replaces the Python loop when
present; otherwise an AMD fused kernel is tried (`:614-616`), and the generic
per-token loop is the final fallback (`:617-630`). The hook also gates bulk prefill:
`prefill_chunk_size` returns `min(requested,256)` only if **every** `GatedDeltaNetBlock`
has a `gated_delta_scan` (`model.py:1040-1042`), else `1`.

Pool dispatch (`:463-491`) has an explicit safety contract. `dispatch_status` is
documented as `0 serial, 1 pooled, 2 serial after pre-dispatch error, 3
partial-dispatch error, 4 completed with a dispatcher error` (`:461-462`). If fewer
than 4 workers completed, it **never replays serially**, because *"Recurrent state may
already have changed: never replay serially here"* (`:488-490`). The Python side turns
any nonzero return into `__builtin_trap()` (`:28`) rather than consuming a partial
result.

### 1.7 Measured effect

**Standalone, CPU-only, Clang production settings, 3 warmups + 40 alternating
ABBA/BAAB blocks (80 samples per variant)** — `extra/rockchip/cpu_gdn_unroll.md:11-21`:

| Fixture | Control ms | Candidate ms | Median paired speedup |
|---|---:|---:|---:|
| 128-token prefill (16 heads, K=128, V=128) | 50.260 | 18.364 | **2.735×** |
| Single-token decode | 0.945 | 0.691 | **1.369×** |
| Per-key decay, K=17/V=32, 7 tokens | 0.060 | 0.046 | 1.314× |
| Scalar decay, K=17/V=16, 7 tokens | 0.041 | 0.034 | 1.203× |
| Fallback K=17/V=9, 7 tokens | 0.039 | 0.040 | **0.974×** (regression ~1 µs) |

All 40 paired blocks favoured the candidate on each optimized fixture
(`cpu_gdn_unroll.md:23`). Output, final state and history matched **bit for bit** for
reset and nonzero positions (`cpu_gdn_unroll.md:28-29`).

**In-model cost, Qwen3.5-4B prefill [measured]:** `604.72 ms across 24 scan calls`
(`cpu_gdn_unroll.md:3-4`). 24 calls is consistent with 32 layers at
`full_attention_interval=4` → 24 GDN blocks [inferred from the config in §3].

**Head-parallel pool, operator level [measured]:** the 16-head / key128 / value128 scan
fell from **56.41 → 12.74 ms at 128 rows**, output and state bits preserved
(`wmma_fast.md:432-436`).

**Head-parallel pool, whole model [measured]:** Qwen3.5-2B went from 63.92/10.00 to
**71.15/10.80** prefill/decode tok/s with pooling on in both arms
(`wmma_fast.md:438-441`). The note is explicit that the whole-model gain is far smaller
than the isolated one.

**Direct-decode path [measured]:** paired median decode **6.235 → 6.424 tok/s** across
twelve alternating requests, all 32×248,320 logits preserved
(`extra/rockchip/recurrent_direct_decode.md:7`). The pre-final-guard CPU screen was
**0.387674 ms → 0.154605 ms** (`recurrent_direct_decode.md:7`).

### 1.8 The numerical bug that cost a full-model run

This is the single most transferable lesson in the archive.

First implementation used four-lane partial sums and factored the decay differently
from the tinygrad-generated CPU reference. Arithmetic probe, synthetic, CPU only
(`codex-session-...fastonpu.md:381300-381330`):

| case | output max_abs | output norm-RMSE | state bitwise |
|---|---:|---:|---|
| h=16, k=128, v=128, start=0 | 1.117e-08 | 2.30e-07 | equal |
| h=16, k=128, v=128, start=5 | **0.0291** | **0.2299** | not equal |

The `start=5` case was not rounding — 0.0291 absolute on values of order 0.1 is a
different answer. Cause, identified by reading the generated CPU program
(`:381400-381470`): the reference **factors scalar decay outside the dot product**
(`(v - (s*a·k))*b`), while the kernel applied decay inside the accumulation
(`(v - ((s*a)·k))*b`), and the reference accumulates keys sequentially.

The fix was to match the reference order exactly — one `F4` accumulator per value row,
serial `c`, scalar decay factored out (`:381900`). Result, same probe
(`:381770-381790`):

| case | output bitwise | state bitwise |
|---|---|---|
| h=16, k=128, v=128, start=0 | **equal** | **equal** |
| h=16, k=128, v=128, start=5 | **equal** | **equal** |

The h=2/k=8/v=8 case still differs by 2.3e-10 (1 element), which is a genuine
reduction-order difference the reference tolerates.

**But it did not fix the model.** Both v1 and v2 full-model runs diverged from the
control at **token index 13** (the 14th token), and the two candidates produced
*identical* 64-token sequences to each other
(`codex-session-...fastonpu.md:382410-382430`). So the residual divergence was **not**
the scan reduction order. See §6.

---

## 2. Native replay (`tinygrad/llm/kernels/rockchip_replay.py`)

### 2.1 What it does

A captured `CapturedJit` linear graph (a flat list of `Ops.CALL`s, mixing CPU
programs, NPU matmuls and `Ops.STORE` copies) is **transcribed into one C function**
at runtime, compiled with the Clang renderer, and called directly — replacing the
Python/UOp executor loop for every subsequent call.

Module docstring: `"""Native replay of prepared CPU/Rockchip inference calls, with
guarded fallback."""` (`rockchip_replay.py:1`).

The generated function signature is fixed (`:205-215`):

```c
int replay(void *input0, ..., void *inputN, long long value0, ..., unsigned long long *stats)
```

i.e. one `void*` per captured input buffer, one `long long` per symbolic variable, and
a `stats` array where `stats[0]` counts submits and `stats[1]` counts tasks. The
preamble defines `U32`/`U64` and packed `struct Sync` / `struct Submit` mirroring the
DRM ABI (`:211-215`), plus `_FINITE_HALF_SOURCE` for the activation guards.

### 2.2 Construction, step by step (`__init__`, `:133-220`)

1. **Roots.** Every input becomes a `Buffer` via `ensure_allocated()`; all must be CPU
   or it raises `UnsupportedReplay('non-CPU inputs')` (`:135-137`).
2. **Calls.** `self.calls = [call.without_after for call in linear.src]` (`:147`).
3. **Work check.** At least one call must be an NPU `Ops.CALL` or an attention call, or
   `UnsupportedReplay('no NPU work')` (`:148-149`). `_attention_call` sniffs a 32-byte
   binary starting `b'RNPA'` (`:43-45`).
4. **Function pointers.** `ioctl`, `memmove`, `clock_gettime` are resolved to raw
   addresses and emitted as literal `ULL` casts so the generated C needs no headers
   (`:166-170`, `:207-209`).
5. **Alias planning.** `copy_aliases()` (`:288-371`) runs *before* code generation.
6. **Per-call emission.** For each call: `Ops.STORE` → `copy()`, `Ops.PROGRAM` →
   `kernel()`, anything else → `UnsupportedReplay('unsupported call body')`
   (`:187-189`).
7. **Post-checks.** `if not self.submits: raise UnsupportedReplay('no prepared NPU
   work')` (`:200`); `if self.mutated.intersection(self.weight_guards): raise
   UnsupportedReplay('mutable matrix weights')` (`:201`).
8. **Compile.** `ClangRenderer` on the CPU target, `TinyELF(...,'replay',...)`, then
   `cpu.runtime(...)`; `program.fxn.restype = ctypes.c_int` (`:216-219`).

### 2.3 What it inlines

| call kind | emitted as | line |
|---|---|---|
| `Ops.STORE` (copy) | optional `SYNC_FROM_DEVICE`, then `memmove(dst,src,n)` unless aliased, then optional `SYNC_TO_DEVICE` | `:373-387` |
| CPU `Ops.PROGRAM` | direct call through the resolved `runtime.fxn` address, cast to the exact param list | `:424-429` |
| NPU `RockchipProgram` | finite-activation guard, optional INT8 scale-header loop, then a literal `struct Submit` + one `ioctl` | `:451-507` |
| NPU attention (`RockchipAttentionProgram`) | `runtime.emit(...)`, counted as 2 submits | `:416-423` |
| GPU FFN (`RockchipGPUProgram`) | `gpu_plan.emit(...)` plus three `SYNC_FROM_DEVICE` | `:430-450` |
| routed MoE | per-group weight-address patch into the command buffer, then `SYNC_TO_DEVICE` | `:474-482` |
| bucketed MoE | call into `plan.updater.fxn` with the dynamic count state | `:483-490` |

Scalars are inlined as constants when bound, else passed as `valueN` (`:389-397`).
Only integer/bool scalars are allowed: `UnsupportedReplay('noninteger scalar')`
(`:390`).

### 2.4 Buffer handling and aliasing

`buffer(node)` (`:222-237`) returns `(Buffer, raw, c_expression)`:

- symbolic views are rejected outright (`:224`);
- device must be `CPU`, `ROCKCHIP`, `PYTHON` or `NPY` (`:226-227`);
- if the buffer shares a base with a captured input root, the expression is
  `((char*)inputN + (buf.offset - root.offset))` — i.e. **input-relative addressing**,
  so the same code works for any input pointer (`:234-235`);
- otherwise the absolute address is baked in and the storage is recorded in
  `self.guards` so `valid()` can detect reallocation (`:236-237`).

`copy_aliases()` (`:288-371`) is the interesting part. It **redirects private CPU pack
outputs and adjacent read-only consumers into NPU mappings**, eliminating a copy. For
each `STORE` it requires all of:

- source is CPU, destination is a `RockchipBuffer`, sizes equal (`:326-327`);
- the source buffer is tagged `scratch` (`:327`, `scratch()` at `:309`);
- there is a prior writer of that exact range that is a CPU kernel writing
  `half`/`uint8` (`:328-332`);
- `_full_write(...)` proves the writer fills the whole view (`:333`);
- no other event touches the range in that call (`:334`);
- `dead_after(key, i)` proves no future read before a proven full overwrite
  (`:335`, `:310-323`);
- no aliasing NPU write overlaps the destination (`:336-337`).

`_full_write` (`:48-129`) is the proof engine: it accepts `Ops.STORE` slot 0
unconditionally, and for CPU programs requires the slot to be in `outs` and not `ins`,
a single non-ALU param whose byte size equals the view, no `IF`/`BACKEDGE`, then
proves the index expression is a **contiguous affine interval** covering
`param.max_numel()` (`:113-129`). A whitelist of known-complete packers
(`rockchip_swiglu_pack`, `rockchip_quantize`, `rockchip_int8_ffn_pack`, and — only
under `ONPU_REPLAY_EXTRA_PACK_ALIAS` — `rockchip_int8_swiglu_pack`,
`rockchip_quantize_sdk`, `rockchip_int8_erf_gelu_pack`, `rockchip_int8_gelu_pack`)
short-circuits the proof (`:63-70`).

**Borrowed outputs** (`ONPU_REPLAY_BORROW_OUTPUTS`, default off, `:172`, `:341-370`)
redirect an NPU output straight into the CPU consumer's scratch instead of copying
down. Conditions are stricter: destination and source both `scratch`, neither is an
input root, the NPU buffer is `CACHEABLE` and 64-aligned (`:345-347`), the consumer is
the immediately following non-`STORE` CPU program (`:348-351`), no intervening event
touches the range (`:359-365`), the consumer must read exactly that key and not write
it (`:363-364`), and the *next* call after the consumer must be a `STORE` (`:366-368`).

### 2.5 Cache maintenance

`sync(raw, flag, index)` (`:239-286`) chooses between three strategies:

1. **Native invalidate** (FROM_DEVICE only): only when the previous call's NPU output
   is exactly this `(meta, addr, size)` and no CPU kernel intervened, and
   `raw.size <= 4 MiB` (or 16 MiB with `ONPU_REPLAY_NATIVE_FROM_LARGE`)
   (`:259-268`). Emits `if(invalidator(addr,size)) return 2000000+index;`.
2. **Native clean** (TO_DEVICE only): user-space `dc cvac`/`dsb sy` helper, requires
   `raw.size <= 1 MiB` and 64/4096 alignment (`:271-276`).
3. **Driver ioctl** fallback: a literal `struct Sync` plus `ioctl` (`:283-286`).

The extra-pack alias (`ONPU_REPLAY_EXTRA_PACK_ALIAS`, default 1, `:28`) admits a
*view* inside a larger arena to native cleaning, but only on an exact recorded upload
match — `uploaded == (id(raw.meta), raw.addr, raw.size)` — explicitly because *"A
shared pack arena can be far larger than the view an upload actually wrote"*
(`:25-27`, `:243-255`).

### 2.6 Validity guards and fallback

`valid(inputs)` (`:534-543`) delegates to `_valid_deduplicated` when
`ONPU_REPLAY_DEDUP_GUARDS` (default 1). It checks: input count and byte sizes, CPU
device, `buf._storage is storage` for every guarded buffer (catches reallocation),
weight `meta.version` unchanged, attention programs still owned and open, and
`plan.bufs is buffers` — *identity*, not equality, because *"Preparing again replaces
this tuple even if the buffer values compare equal; the old command addresses must
then be rebuilt"* (`:526-529`). Output metas must satisfy
`cpu_clean_version == version` (`:530-531`).

`RockchipCapturedJit.__call__` (`:606-626`) is the fallback ladder:

```python
if DEBUG>=2 or PROFILE or self.reason is not None: return super().__call__(...)   # :607
warming = not self.warmed
if warming: result = super().__call__(...); self.warmed = True                     # :608-611
concrete = [_copy_scalar_input(u) if u in self._jit_bufs else u for u in input_uops]
if self.native is not None and not self.native.valid(concrete): self.native = None
if self.native is None:
  try: self.native = NativeReplay(...)
  except (UnsupportedReplay,KeyError,AssertionError) as exc:
    self.reason = str(exc)                       # sticky: never retried
    return result if warming else super().__call__(...)
if warming: return result                        # preparation does not execute
self.native(concrete,var_vals); self.native_runs += 1
```

Two deliberate properties: a failed preparation sets `self.reason` and the JIT never
tries again (`:616-618`), and preparation happens during the warmup so the update is
not applied twice (`:619-621`, documented at `wmma_fast.md:495-500`).

Error codes are partitioned by magnitude so a failure identifies its stage
(`:555-565`): `7000000+` attention dispatch, `6000000+` attention range,
`5000000+` non-finite dynamic weights, `4000000+` GPU FFN, `3000000+` expert routing,
`2000000+` submission, `1000000+` non-finite activations. Submission and GPU failures
set `self.dev.poisoned = True`.

### 2.7 Measured effect

**Qwen3-0.6B, FP16, context 512, 128-token prompt, 32 generated tokens, clocks pinned
(NPU 1 GHz, DDR 2112 MHz), 4 warmups + 3 measured** — `codex-session-...fastonpu.md`:

| arm | TTFT ms | decode tok/s | prefill submits | decode submits | IDs |
|---|---:|---:|---:|---:|---|
| parallel attention, no replay (`:43139`) | 1054.72 | 5.96 | 427 | 3689 | match |
| native replay (`:43285`, `:43270`) | **703.48** | **16.36** | 427 | 3689 | match |

**2.74× decode, 1.50× TTFT, identical submission counts and identical token IDs.**
The gain is entirely host-side dispatch removal.

Host-overhead refinements, paired worker diagnostics
(`extra/rockchip/native_host_overhead.md`):

| change | control | candidate | note |
|---|---:|---:|---|
| scalar snapshot (`:9`) | 334.53 / 15.966 | 337.20 / 16.144 | 62 scalar copies/request; not a proven decode gain |
| dedup guards (`:19`) | 338.142 / 16.061 | 346.543 / 16.085 | ~0.15% decode; explicitly not claimed |
| scalar snapshot, CPU micro (`:11`) | 496.4 µs | 49.9 µs | isolated copy only |
| dedup guards, CPU micro (`:17`) | 211.1 µs | 57.5 µs | 526 storage guards, 175 plans |

Borrowed outputs: implemented behind `ONPU_REPLAY_BORROW_OUTPUTS=1`, **default off**,
and on the measured Qwen3 graph it found **zero eligible eliminated output copies** —
no demonstrated gain (`wmma_fast.md:623-627`).

---

## 3. Qwen3.5 4B specific configuration

### 3.1 The checkpoint

`Qwen3.5-4B-Q8_0.gguf`, 4,482,403,488 bytes,
sha256 `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1`
(`codex-session-...fastonpu.md:255224`, `:234073`).

GGUF metadata (`:254526`):

| key | value |
|---|---|
| `block_count` | 32 |
| `embedding_length` | 2560 |
| `feed_forward_length` | **9216** |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | 4 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `ssm.conv_kernel` | 4 |
| `ssm.state_size` | **128** |
| `ssm.group_count` | 16 |
| `ssm.time_step_rank` | 32 |
| `ssm.inner_size` | 4096 |
| `full_attention_interval` | **4** |
| `rope.freq_base` | 1e7 |
| `rope.dimension_count` | 64 |
| `attention.layer_norm_rms_epsilon` | 1e-6 |

Derived [inferred] from `model.py:537-540`:
`head_k_dim = state_size = 128`, `num_k_heads = group_count = 16`,
`num_v_heads = time_step_rank = 32`,
`head_v_dim = inner_size // time_step_rank = 128`,
`conv_channels = 4096 + 2*16*128 = 8192`, `q_dim = 128*16 = 2048`.
32 layers at interval 4 → 8 full-attention + **24 GDN blocks**, which matches the
measured "24 scan calls in Qwen3.5-4B prefill" (`cpu_gdn_unroll.md:3-4`).

### 3.2 Code gates that fire on this shape

| gate | condition | line | effect on 4B |
|---|---|---|---|
| INT8 K-split selection | `precision=='int8' and value.weight.shape[1]>4096` → `RockchipInt8SplitLinear` | `rockchip.py:558-560` | `ffn_down` has k=9216 → split into 4096+4096+1024 |
| split-depth guard | `4096 < depth <= 32768` | `rockchip_int8.py:507` | 9216 passes |
| weight-depth admission | `weight.shape[1] > (32768 if int8 else 4096)` → skip | `rockchip.py:556` | 9216 admitted only in INT8 |
| FP16 split-down | `split_down and down.depth%96==0 and down.columns<=4096` | `rockchip.py:305` | 9216%96==0, n=2560 → **enabled** |
| packed FFN rejection | `len(joint.packed)==1` required | `rockchip_int8.py:680-681` | joint.columns=18432 / tile 4096 → 5 tiles → **packed path off**, float path used |
| float FFN path | `16<=x.max_shape[1]<=128 and 4<=down.depth<=32768` | `rockchip_int8.py:683-686` | 9216 in range → sigmoid-gate path |
| `ffn_pack` depth bound | `512<=depth<=4096` | `rockchip_int8.py:656` | 9216 **rejected** — confirms packed path cannot serve 4B |
| recurrent scan geometry | `1<=head_k_dim<=256`, `1<=head_v_dim<=256` | `rockchip_recurrent.py:69-70` | 128/128 pass |
| scan capacity | `capacity<=256` | `rockchip_recurrent.c:68` | 128-row chunk fine |
| pool eligibility | `num_v_heads>=4` | `rockchip.py:479-480` | 32 → pool eligible |
| attention pool | `n_kv_heads>=4` or (`>=2` and `max_context>=4096`) | `rockchip.py:481-483` | 4 KV heads → eligible |
| fused recurrent proj. | `not hasattr(block,'ssm_g_a')` (non-KDA) | `rockchip.py:643-644` | 4B is non-KDA → eligible |
| `prefill_chunk_size` | all GDN blocks hooked → `min(req,256)` | `model.py:1040-1042` | 256 max |

There is **no** `large`/4B string gate anywhere in the kernels. Selection is entirely
by geometry and by `configure()` flags; the only model-specific pinning lives outside
this tree in the app recipe (`oNPU/onpu/rockchip_options.py`, cited at
`recurrent_direct_decode.md:9`).

### 3.3 Measured Qwen3.5 4B numbers

All at context 512, 128 input tokens, 64 output tokens, 3 warmups + 3 measured.

| configuration | TTFT ms | prefill proxy tok/s | decode tok/s | source |
|---|---:|---:|---:|---|
| GPU one-token fallback (initial) | 480,831.24 | 0.27 | 0.27 | `wmma_fast.md:588` |
| GPU fused recurrent scan + bulk prefill | 23,436.57 | 5.46 | 0.60 | `wmma_fast.md:588` |
| CPU/NPU INT8, scan + gated attention | — | 32.20 → 41.74 | 4.84 → 5.23 | `wmma_fast.md:322` |
| CPU/NPU INT8, complete candidate | — | 47.80 | 5.29 | `wmma_fast.md:294-296` |
| CPU/NPU, staged decode treatment | — | — | 6.34 | `recurrent_preparation.md:45-46` |
| CPU/NPU, final API cohort | — | 82.52 | 6.32 | `recurrent_preparation.md:58-59` |
| CPU/NPU, four-worker head partition (API) | 1263.09 | **101.34** | **6.48** | `recurrent_preparation.md:79` |
| RKLLM reference | 2068.88 | 61.87 | 6.22 | `codex-session-...fastonpu.md:210013` |

The 4B CPU/NPU profile that explains the remaining gap [measured]
(`wmma_fast.md:305-311`): **INT8 NPU submissions are 437.01 of 3,942.81 ms (11.08%) of
native prefill call time**; CPU scan and generic attention/reduction dominate. In
decode the balance inverts: NPU INT8 is **76.27%** of profiled native time.

---

## 4. Optimisation methodology stated in the session docs

### 4.1 Profile the reference first, and compute speed-of-light before touching a kernel

`wmma_fast.md:520-546` gives the cost model and the discipline:

```
token time = matrix arithmetic + weight/activation/KV traffic
           + attention or recurrent state work + normalization/activation
           + layout conversion and device/cache synchronization
           + submission, host scheduling and sampling
```

with the explicit caveat *"Some work can overlap, so this is an accounting guide
rather than a universal timing identity. Improving one term can expose another
bottleneck."* (`wmma_fast.md:531-532`). The worked illustration: if matmul is half the
token, doubling only matmul gives `1/(0.5+0.5/2) = 1.33x` — *"This is an illustration,
not a measured breakdown."* (`wmma_fast.md:544-546`).

The 4B profile is the model case: 11.08% of prefill is NPU matmul, so **no amount of
matmul work can fix 4B prefill** (`wmma_fast.md:305-311`).

### 4.2 Measure in-model, attribute by ablation, never by multiplication

The archive states this as a rule learned from two concrete failures
(`AGENTS.md`, project rules): an isolated `_attention` A/B said 0.88 s while the
in-model cost was 24.5 s (85% of prefill) — *"the isolated figure was not merely
imprecise, it inverted the priority"*; and "~15 ms per Linear launch" was the eager
path while the JIT-captured path paid 0.79 ms.

`wmma_fast.md:787-788` restates it: *"Reduce actual weight/KV traffic and boundary
costs identified in whole-model profiles. A fast operator microbenchmark is
insufficient evidence."*

The pool result is the clean demonstration: isolated scan 56.41 → 12.74 ms (4.4×) but
whole model 63.92 → 71.15 tok/s (1.11×) (`wmma_fast.md:432-446`).

### 4.3 Paired A/B, ABBA, report pair counts not medians

`cpu_gdn_unroll.md:11-13`: *"identical inputs and the production Clang compiler
settings, three warmups, then 40 alternating ABBA/BAAB blocks. Each variant has 80
timing samples."* Result reported as *"All 40 paired blocks favored the candidate for
each optimized fixture"* (`:23`).

`wmma_fast.md` uses the same shape throughout: *"three warmups and seven rotating
measurement rounds"* (`:683-684`), *"twelve alternating requests"*
(`recurrent_direct_decode.md:7`).

### 4.4 Bitwise equality is the acceptance bar for arithmetic changes

Every accepted change in this archive carries a bitwise claim:

- scan: *"Output, final state and available history matched the original C kernel bit
  for bit for reset and nonzero positions"* (`cpu_gdn_unroll.md:28-29`), and a
  dedicated ARM-only test `test_arm_clang_decode_matches_tensor_output_and_state_bitwise`
  across three seeds (`codex-session-...fastonpu.md:381910-381940`).
- head pool: *"preserving output and state bits"* (`wmma_fast.md:435`).
- direct decode: *"preserves all 32×248,320 logit values"* (`recurrent_direct_decode.md:7`).
- native replay: *"matched every token ID in all seven requests"* (`:43285`).
- packed upload: *"all raw integer outputs exact"* (`:1088251`).

Where bitwise is not achievable, the gate is explicit and numeric: INT8 placement
requires `normalized_rmse <= 0.03` (`rockchip.py:604-607`); FP16 requires
`|a-b| <= 2e-3 + 3e-3*|b|` (`rockchip.py:602`).

### 4.5 Reject on the full-model gate even when the operator wins

Repeatedly, a faster operator is rejected because the model fails:

- three-product residual INT8: 10.64/3.98 → 16.20/4.41 tok/s, but relative logit RMS
  0.159280, mean TV 0.063834, perplexity ratio 1.028491 — *"rejected for preferred-row
  promotion"* (`wmma_fast.md:38-44`).
- grouped INT8 MLA: 19.84/4.81 vs 10.62/3.97 control, fails logit RMS and
  probability-distribution limits → rejected (`wmma_fast.md:26-32`).
- NPU attention for Qwen2: *"passed its operator oracle but was slower for this small
  head layout. Its data preparation and submission costs outweighed its compute
  saving; the prototype was removed"* (`:1088048`).
- smaller fetch grain, buffer shrinking, SDK down tiling, temporary padding,
  approximate exp, O3: *"did not establish the required complete-model gain"*
  (`:1088049`).
- main-CPU wake helper: *"only ~2% paired prefill gain; not ported"* (`:1088251`).
- futex sleeping: decode fell to 10.17 tok/s; rejected (`recurrent_preparation.md:41-42`).
- native-clean extension: numerically exact but TTFT 102.26 → 119.53 ms; rejected
  (`extra_pack_upload_policy.md:21-23`).
- `legacy_division` alternative: *"preserved generated IDs but changed full logits and
  was rejected"* (`recurrent_preparation.md:15`).

### 4.6 Never let a faster path change the arithmetic silently

The scan bug (§1.8) is the canonical case. The archive's own summary of the rule:
*"A 2.38e-7 projection difference later crossed an INT8 vocabulary quantization
boundary"* (`wmma_fast.md:804-805`). And on the scan specifically, the intent record
pre-registered the risk before any code ran: *"Four-lane partial sums change reduction
ordering and require numerical and full-model qualification"*
(`codex-session-...fastonpu.md:370722`).

### 4.7 Pre-register the gate, then keep the failed run

The CPU-scan hardware experiment pre-declared `expected_generated_ids` = the exact
prior 64-token control sequence, and the validator fails the case if any measured
sequence differs (`codex-session-...fastonpu.md:382410`). When v1 failed at token 13,
the record was written as a diagnostic with `'qualified_speedup': False, 'reason':
'Candidate fails exact prior INT8 token-ID gate; stable own output does not establish
parity.'` (`:381057`) rather than being discarded.

### 4.8 Timing protocol

3 warmups + 3 measured, not 1+3; quote the stabilized number and say so; report every
trial and the median (`AGENTS.md` project rules, and `cpu_gdn_unroll.md:11`). The
archive records the failure mode this prevents: an 11.65 s first-measured TTFT that was
a setup outlier, fixed by moving replay compilation into warmup
(`wmma_fast.md:495-500`).

### 4.9 Clocks

Every benchmark log line carries `clocks_before`/`clocks_after` for NPU, GPU, DDR and
three CPU policies (e.g. `codex-session-...fastonpu.md:43139`). The replay comparison
ran at NPU 1 GHz, DDR 2112 MHz, CPU6 1800000.

---

## 5. Every measured effect: scan, replay, projection fusion, FFN packed upload

| # | change | model / fixture | conditions | effect | source |
|---|---|---|---|---|---|
| 1 | scan, 64 value lanes | synthetic 16h/K128/V128, 128 rows | CPU only, 3 warm + 40 ABBA blocks | **2.735×** (50.260 → 18.364 ms) | `cpu_gdn_unroll.md:17` |
| 2 | scan, 64 value lanes | same, 1 row | same | **1.369×** (0.945 → 0.691 ms) | `cpu_gdn_unroll.md:18` |
| 3 | scan, per-key decay K=17/V=32 | 7 rows | same | 1.314× | `cpu_gdn_unroll.md:19` |
| 4 | scan, scalar decay K=17/V=16 | 7 rows | same | 1.203× | `cpu_gdn_unroll.md:20` |
| 5 | scan, fallback K=17/V=9 | 7 rows | same | **0.974×** (regression) | `cpu_gdn_unroll.md:21` |
| 6 | head-parallel pool | 16h/K128/V128, 128 rows | CPU operator | **56.41 → 12.74 ms** | `wmma_fast.md:434` |
| 7 | head-parallel pool | Qwen3.5-2B | 128/64, ctx512, INT8 | 63.92/10.00 → **71.15/10.80** tok/s | `wmma_fast.md:440` |
| 8 | direct decode | Qwen3.5-4B | 12 alternating requests | decode **6.235 → 6.424** tok/s | `recurrent_direct_decode.md:7` |
| 9 | direct decode | CPU micro | pre-final guard | 0.387674 → 0.154605 ms | `recurrent_direct_decode.md:7` |
| 10 | scan, in-model cost | Qwen3.5-4B prefill | 24 calls | **604.72 ms total** | `cpu_gdn_unroll.md:3` |
| 11 | native replay | Qwen3-0.6B FP16 | ctx512, 128/32, pinned clocks | **5.96 → 16.36** decode tok/s; TTFT 1065 → 703 ms; same 427/3689 submits; same IDs | `:43139`, `:43285` |
| 12 | replay scalar snapshot | paired worker | 62 copies/request | 334.53/15.966 → 337.20/16.144; not claimed | `native_host_overhead.md:9` |
| 13 | replay dedup guards | paired worker | 526 guards, 175 plans | 338.142/16.061 → 346.543/16.085; ~0.15%, not claimed | `native_host_overhead.md:19` |
| 14 | replay borrowed outputs | Qwen3 graph | default off | **zero eligible copies** — no gain | `wmma_fast.md:623-627` |
| 15 | projection fusion (QKV + gate/up) | Qwen3-0.6B FP16 | ctx512 | part of the 5.96 → 16.36 path; not isolated | `wmma_fast.md:558` |
| 16 | FFN packed upload | Qwen2-0.5B SDK gate/up | 20 paired blocks | raw gate/up **~1.4×**; whole-model API 901.11 vs 877.31 published prefill, 43.98 vs 41.58 decode | `:1088045`, `:1088251` |
| 17 | FFN pack kernel | Qwen3-0.6B, prefill | profiled, 28 calls | 68.439 ms → **0.527 ms** per 28 calls (decode) | `:47453`, `:47475` |
| 18 | extra pack alias / native clean | LFM2.5-VL 450M | ctx512 API | alias-only: TTFT **122.70 → 112.48 ms**; native-clean extension: 102.26 → 119.53 ms (rejected) | `extra_pack_upload_policy.md:21-24` |
| 19 | pooled sigmoid | Qwen3.5-2B | same pooled scan both arms | prefill **71.26 → 79.87** tok/s; decode 10.69/10.73 (no gain) | `wmma_fast.md:456-459` |
| 20 | four-worker head partition | Qwen3.5-4B | 4 ABBA prototype processes | CPU component 8.1 → 2.05 ms; TTFT −20.42% / −19.93%; decode unchanged | `recurrent_preparation.md:77` |

Item 15 needs a caveat: projection fusion is implemented and active in every measured
Qwen3.5/CPU-NPU cohort (`rockchip.py:639-654`), but **no document in this archive
isolates its contribution**. The only fusion evidence is the aggregate
`wmma_fast.md:592-593`: *"These gains combine the recurrent scan, bulk prefill and
prepared projections: they do not isolate the benefit of one kernel."*

Item 17 needs a caveat too: the 68.439 → 0.527 ms pair is the *decode* profile row for
`rockchip_swiglu_pack`; the prefill row in the same profile is still 68.439 ms /
28 calls (`:47453`), so the 130× figure is a decode-path number, not a prefill one.

---

## 6. What is still unsolved or rejected for Qwen3.5 / GDN

### 6.1 The CPU scan still fails full-model token parity — unsolved

Two full-model paired runs, Qwen3.5-0.8B, INT8, ctx512, 128 in / 64 out, shared packed
weights, CPU scan on in the candidate only:

| arm | chunk | scan | blocks | median TTFT ms | prefill tok/s | decode tok/s | IDs |
|---|---:|---|---:|---:|---:|---:|---|
| control v1 | 1 | off | 0 | 16519.67 | 7.75 | 8.05 | reference |
| candidate v1 | 128 | on | 18 | **1718.33** | **74.49** | **19.92** | **fail @ token 13** |
| control v2 | 1 | off | 0 | 16462.37 | 7.78 | 8.06 | reference |
| candidate v2 | 128 | on | 18 | **1551.13** | **82.52** | **17.32** | **fail @ token 13** |

(`codex-session-...fastonpu.md:381040-381057` for v1; `:382410-382430` for v2.)

The diagnostic record states it plainly: `'qualified_speedup': False, 'reason':
'Candidate fails exact prior INT8 token-ID gate; stable own output does not establish
parity.'` (`:381057`).

Two things were ruled out:

1. **Not the scan reduction order.** The reference-order kernel is bitwise-exact
   against the CPU reference at h=16/k=128/v=128 for start=0 and start=5 (§1.8), and
   v1 and v2 produced **identical** 64-token sequences to each other
   (`:382420-382430`).
2. **Not RMSNorm batching.** Serial vs bulk Tensor normalization matched **bit for
   bit** (`different_bits: 0`) for both FP32 and FP16 weights
   (`:383000-383030`). The fast native `rms_norm` did differ — 108,545 differing bits,
   max_abs 3.34e-6, and 111 differing scale bits — though the quantized codes matched
   in that sample (`:383010-383040`).

The next planned step was recorded as: *"isolate CPU scan reduction arithmetic from
batched prefill, compare teacher-forced logits/state, and retain the existing ID gate"*
(`:381057`), staged as `cpu-gdn-isolation-v1` with two arms — `qwen35-08b-serial-scan`
(chunk 1, scan on, cpu_norm true) and `qwen35-08b-bulk-scan-no-fast-norm` (chunk 128,
scan on, cpu_norm false) (`:383140-383150`). **No result for that isolation run appears
in the archive.** The isolation is therefore open.

### 6.2 The 4B prefill gap is not a matmul problem — open

11.08% of native prefill time is NPU INT8 submissions; CPU scan and
attention/reduction dominate (`wmma_fast.md:305-311`). The archive's conclusion is
structural, not a missing kernel: *"Hybrid models additionally update
recurrent/convolution state; speeding only their projections leaves that state path
serial"* (`wmma_fast.md:538`).

### 6.3 Decode remains dominated by NPU submission — open

In decode, NPU INT8 is 76.27% of profiled native time, *"including driver/submission
work"* (`wmma_fast.md:448-451`). The 4B decode rate (6.48 tok/s) is only marginally
above RKLLM's 6.22 (`recurrent_preparation.md:79`, `:210013`).

### 6.4 Rejected for Qwen3.5 / GDN specifically

- **Packed recurrent projections.** *"The earlier packed Qwen3.5-0.8B experiment
  achieved 56.79 prefill tok/s at 2910.21 MiB steady allocation. The current recurrent
  row-major path gives 46.79 tok/s at 2169.96 MiB, saving about 740 MiB while decode
  stays about 4.54 tok/s."* (`wmma_fast.md:599-602`) — i.e. packing bought 21% prefill
  for 740 MiB, and was traded away. The 4B run disables global weight packing entirely
  (`wmma_fast.md:602-603`).
- **`ffn_pack` for the 4B FFN.** `512<=depth<=4096` (`rockchip_int8.py:656`) excludes
  depth 9216, and `len(joint.packed)==1` (`:680-681`) excludes the 5-tile 18432-column
  joint. The 4B FFN therefore runs the float sigmoid-gate path, not the packed INT8
  path.
- **Speculative decoding on Qwen3.5-2B/DFlash2.** Every cohort loses: v2 22.88 s
  ordinary vs 86.02 s speculative (`wmma_fast.md:634`); v3 10.01 vs 22.12 (`:651-653`);
  v4 10.10 vs 21.11 (`:660-661`); v5 33.36 vs 106.79 (`:673-674`); v6 33.49 vs 115.09
  (`:691-692`); v7 33.38 vs 70.90 (`:706-707`); v8 33.38 vs 65.11 (`:724-726`);
  v9 33.51 vs 63.18 (`:739`); v10 32.53 vs 57.21 (`:750`). *"All pairs still lose to
  ordinary"* (`:757`).
- **NPU draft head.** 5.57× micro (350.43 ms GPU vs 62.90 ms CL/NPU/CL) but costs an
  extra 970 MiB packed head and is not a trained-model win (`wmma_fast.md:714-720`).
- **`ONPU_GDN_DIRECT_DECODE` default.** Off by default (`rockchip_recurrent.c:17-19`);
  only enabled for the pinned 4B checkpoint at ctx512 (`recurrent_direct_decode.md:7`).
- **CPU prefill pool generally.** *"an opt-in CPU prefill pool has not demonstrated an
  overall improvement and is not an automatic serving optimization"*
  (`wmma_fast.md:626-627`).

### 6.5 Quality is unqualified everywhere

Every Qwen3.5 row in the archive is marked unqualified. `wmma_fast.md:299-300`: *"All
three remain below the recorded RKLLM rates, with comparable quality unqualified."*
The 4B API cohort result carries the same caveat: *"This does not qualify independent
original/RKLLM output quality or model-maximum context"* (`recurrent_preparation.md:79`).
The broader campaign statement is `all_published_models_faster_at_comparable_quality:
false, objective_complete: false` (`.team-notes/04-rockchip-npu.md:275-278`).

---

## Appendix: reimplementation checklist for the scan

To reproduce `cpu_gated_delta_scan` from this report alone:

1. FP32 only. `F4 = float __attribute__((vector_size(16), aligned(1)))`.
2. Gate: `capacity<=256`, `key_dim<=256`, `value_dim<=256`, `alpha_dim in {1,key_dim}`,
   `1<=rows<=capacity`, `start>=0`. Return 1 on violation.
3. Allocate `float columns[256*256]` aligned 16.
4. For each `bh` in `[group, bh_count)` step `groups`:
   a. If `start` is 0, treat the state as zero; else load
      `columns[c*value_dim+r] = s[r*key_dim+c]`.
   b. For each `t` in `[0,rows)`:
      - `step = bh*capacity + t`; `qt=q+step*key_dim`, `kt=k+step*key_dim`,
        `at=alpha+step*alpha_dim`, `b=beta[step]`.
      - Choose lane width: 64 if `value_dim%64==0`, else 16 if `value_dim%16==0`,
        else 4-with-partial-tail.
      - Pass 1 over `c` in `[0,key_dim)`: `pred[r] += (columns[c*value_dim+r] *
        (alpha_dim==1 ? 1 : at[c])) * kt[c]`, per lane, serial in `c`.
      - `delta[r] = (v[step*value_dim+r] - pred[r]*(alpha_dim==1 ? at[0] : 1)) * b`.
      - Pass 2 over `c` in `[0,key_dim)`: `av = at[alpha_dim==1?0:c]`;
        `upd = columns[c*value_dim+r]*av + delta[r]*kt[c]`; store back;
        `result[r] += upd * qt[c]`.
      - Write `result` to `out+step*value_dim`.
      - If `history`, write the post-update state for this `t` in
        `[t, bh, r, c]` order.
   c. Store `columns` back to `s` in V-major order.
   d. `memset` `out[(bh*capacity+rows)*value_dim .. +capacity*value_dim]` to zero.
5. Return 0.

Two invariants must not be relaxed: the key reduction is **serial in `c`** per lane,
and scalar decay is **factored outside** the dot product
(`(v - (s·a)*k)*b`, not `(v - (s·a·k))*b`). Both are required for bitwise agreement
with the tinygrad-generated CPU reference (`codex-session-...fastonpu.md:381400-381470`,
`:381770-381790`).
