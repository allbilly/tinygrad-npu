# Why 157.34 is not matched: the kernels are ported, the mechanisms are not

The reference's 157.34 ms is not produced by its kernels. It is produced by a prepared stack selected
by **25 opt-in mechanisms**, and the receipt records exactly which ones were on. We ported the
kernels; we did not port the ownership and dispatch layer, and that layer is where our measured gap
lives.

## The recipe, from the receipt

`~/oNPU/analysis/release-20261115/r2-lfm-20261116/candidate-retime-20261117-qwen35-4b/receipt.json`,
`effective_environment`, the run that reported `median_decode_tps = 6.355455`:

```
ONPU_REPLAY_BORROW_BATCH=1        ONPU_REPLAY_BORROW_OUTPUTS=1
ONPU_SHARED_HOST_POOL=1           ONPU_HOST_POOL_WFE=1            ONPU_CPU_DECODE_POOL=0
ONPU_ZERO_PAD_VIEWS=1             ONPU_REPLAY_NATIVE_FROM=1       ONPU_REPLAY_NATIVE_FROM_LARGE=1
ONPU_REPLAY_SCALAR_COPY=1         ONPU_REPLAY_DEDUP_GUARDS=1
ONPU_GDN_DIRECT_DECODE=1          ONPU_GDN_DECODE_PREP_POOL=1     ONPU_GDN_DECODE_SPIN=1
ONPU_GDN_PREP_POOL=1              ONPU_INT8_DECODE_NARROW_CBUF=1  ONPU_INT8_NATURAL2304=1
ONPU_NPU_DOMAINS=3                ONPU_NPU_DOMAIN_BYTES=3221225472
ONPU_SWIGLU_PACK_STATIC_LAYOUT=1  ONPU_EXP2_FAST_PATH=1           ONPU_CPU_TOKEN_INPUTS=1
ONPU_FFN_PACKED_UPLOAD=1          ONPU_CPU_PREFILL_LAYOUT=1       ONPU_PREFILL_CPU_TUNING=1
```

Two of these are the whole answer, because their cost is what our class table shows as our largest
non-device rows.

## Mechanism by mechanism, against what we have

| reference mechanism | what it removes | our tree | our measured row | ref row | delta |
| --- | --- | --- | ---: | ---: | ---: |
| `BORROW_BATCH` + `BORROW_OUTPUTS` (`rockchip_replay.py:391,405`, both **default 0**) | per-token copies of state/output buffers | **closed locally**: the decode-only zero-pad skip removed 48 x 2 MB fills (parity unchanged, 21/28 pairs, -8.56 ms) | fills **0.68 ms** | ~0 | **+0.7** |
| `SHARED_HOST_POOL` + `HOST_POOL_WFE` (`rockchip_shared_pool.py`) | per-kernel dispatch of host chains; kernels run in one shared pool | **absent at decode** -- `thread_split` is harness-level and prefill-scoped | elementwise **20.04 ms** | 6.57 | **+13.5** |
| `GDN_DIRECT_DECODE` + `GDN_DECODE_PREP_POOL` + `GDN_DECODE_SPIN` | Python-side GDN bookkeeping per step | `scan_native` (kernel only, no prep pool) | scan **5.22 ms** | 4.16 | +1.1 |
| `REPLAY_NATIVE_FROM` + `_LARGE` | per-call `sync_mem` ioctls | `sync_mem` ioctls intact | 2.8 us/call = 0.87 ms | ~0 | +0.9 |
| `ZERO_PAD_VIEWS` (`rockchip_padding.py:13 mark_zero_tail`, `:37 capacity_input`) | materialising zero padding -- pad becomes a **view over a pre-zeroed full buffer**, so the scan writes the full buffer in place | **absent** -- `model.py:819 where(0, state)` materialises a 2 MB copy per block per token | inside the fills row (9.43) | -- | -- |
| `CPU_TOKEN_INPUTS` + `gpt2_decode:DecodeLoop` | one Python->graph entry per token | we re-enter `model(...)` per token | outside-kernel **9.32 ms** | ~0 | +9.3 |
| `INT8_DECODE_NARROW_CBUF`, `INT8_NATURAL2304` | CBUF/geometry slack in the device recipe | ported where it applies | device **142.65 ms** | 137.11 | +5.5 |
| `SWIGLU_PACK_STATIC_LAYOUT`, `EXP2_FAST_PATH`, `FFN_PACKED_UPLOAD`, `SCALAR_COPY`, `DEDUP_GUARDS` | assorted copies and dispatch | partial / absent | not separately measured | -- | -- |
| `NPU_DOMAINS=3` | IOMMU domain count | we run **2** | not separately measured | -- | -- |

Sum of the measured deltas before the fills fix: 9.4 + 13.5 + 1.1 + 0.9 + 9.3 + 5.5 = 39.7 ms against a
gap of 32.4. With the fills row closed (+0.7) the remaining attribution is ~31 ms against a 24.1 ms gap. The classes overlap somewhat (the outside-kernel time and the fills are both buffer
plumbing), so this is an attribution, not an exact budget -- but it names where the 32 ms sits.

## The single most important line in their source

`rockchip_gpt2_decode.py:24-40` generates and compiles this:

```c
for(int i=0;i<count;i++) {
  U64 current[N]={0};
  int error=((int(*)(long long,long long,U64*))REPLAY)(position+i,token,current);
  ...
  token=*((volatile int*)OUTPUT);      // next token read inside C
}
```

The token loop, the next-token read and the replay entry are one C call for `count` tokens. We call
the model once per token from Python and let TinyJit replay it. That is the `outside 9.32 ms`, and it
is also why our per-token submit count is 307-355 against their 203: their loop reuses one prepared
replay with borrowed buffers, so fewer programs exist to submit.

## What this means for the objective

The gap is **not** a kernel we have not written. Every kernel row we measured has a reference
counterpart at or near parity -- the device is 142.65 against 137.11. The gap is that their decode
runs a *prepared* stack: buffers are borrowed rather than copied (`BORROW_*`, `ZERO_PAD_VIEWS`), host
work runs in a shared pool instead of per-kernel dispatch (`SHARED_HOST_POOL`), and the token loop
lives in C (`CPU_TOKEN_INPUTS`, `DecodeLoop`).

Porting any one kernel cannot close it. Closing it means adopting those three properties, in that
order of measured size: elementwise dispatch (13.5) > fills/borrow (9.4) = C token loop (9.3).

`ZERO_PAD_VIEWS` is the one that can be tested without adopting their replay: their pad is a view
over a buffer that is zeroed **once at capacity**, so nothing is materialised per token. Our
`where(0, state)` materialises per token, and removing the copy outright breaks parity
(`c693475255158a69`) because the scan then writes the persistent buffer. A capacity-sized zero buffer
that the scan writes in place is a third option between those two, and it is what their code does.

## Today's measurements that feed this

- **`E_62080_4` ablation: +5.31 ms** (ratio 1.0473, 6 rounds). One kernel, 8.50 ms of CPU, 62% of it
  on the wall. It is the 248,320-element vocabulary op and it carries the byte-assembly pattern
  **not at all**, so the word-load rewrite does not touch it. Log:
  `results/threadsplit/noop-E_62080_4.log`.
- This **contradicts** the earlier inference that the vocabulary-wide op does not transfer. That
  inference came from the `GREEDY_SAMPLING` ABBA (ratio 0.9991 over 16 pairs), which changed the
  sampling *path* rather than removing this kernel. Both measurements are real; the earlier
  inference from one to the other was not. The kernel is unidentified: it appears in both sampling
  paths, costs 8.50 ms, and is the largest single addressable CPU kernel we have.
- **Two configurations exist for "ours"**: `bench_llm.py --npu` reports **3.42 tok/s (292 ms)** and
  4252 ms TTFT because it never installs `replay_c`, `attn_native` or `scan_native`; the `ab_*`
  harnesses and the kernel dump run the replay path at ~190 ms. Every number in the class table is
  the replay path. Installing the word-load hook in `bench_llm.py` was reverted: it changes a script
  whose numbers are not the objective's basis, and a single 1+1 run cannot validate a 1.6% effect.

## The oracle: their stack measured on this board, in this session

Everything above was attribution from their source and receipt. `extra/rockchip/run_ref.py` already
carries the receipt's environment and drives their worker, so the reference number can be *measured*
rather than quoted. Run with the receipt's protocol (1 warm-up + 3 measured, 64 tokens):

| | ours | reference (oracle) |
| --- | --: | --: |
| decode | **181.26 - 181.45 ms** (5.51 tok/s) | **153.2 ms** (6.525 tok/s) |
| temperature | 71-73 C | 79.5 - 84.1 C |
| trials | 12 + 16 paired | 6.348, 6.549, 6.525 tok/s |
| TTFT | 1.2 s (frozen) | 2120 - 2247 ms |

Trial log: `results/threadsplit/ref-oracle.log`. So the 157.34 from the receipt reproduces here as
153.2 ms, at a *higher* temperature than our run, which means the ~28 ms gap is real and is not a
thermal or archive artefact.

## The one comparison from their profile that is unambiguous

Their own per-call profile (`results/ref-calls.jsonl`, `--profile` run) is recorded per **batch**
(m=32 segments), not per token, so a per-token class split from it is not yet trustworthy and is not
claimed. One entry is directly comparable, because it is a single call of a kernel both trees emit
under the same name and shape (62080 float4 = 248,320 elements, the vocabulary op):

| kernel | theirs | ours | ratio |
| --- | --: | --: | --: |
| `E_62080_4` | **0.375 ms** | **8.497 ms** | **22x** |

That one kernel is **8.1 ms of the ~28 ms gap**, and it is a *kernel-quality* difference, not an
architecture difference -- the same op, the same shape, the same tinygrad emitter. It also means our
remaining gap is not purely architectural, which is what the earlier mechanism table implied.

Their profile also shows multi-millisecond host kernels (`E_2_16_8_32_4_4` 96.7 ms, `r_32_16_32_4`
50.4 ms, `rockchip_cpu_gated_delta_scan` 47.2 ms) inside a 32-token segment, so "their host work is
small" is false: it is *batched* over 32 tokens, and m=32 GEMMs appear in the same segment. That is
the `ONPU_CPU_DECODE_POOL`/`DecodeLoop(count=32)` shape, and it is the next thing to quantify before
porting anything.

**Not claimed:** any per-token class decomposition from `ref-calls.jsonl` until the segment split is
shown to isolate a single token. My first attempt at that split produced a 1886-call / 563 ms
"token" containing m=32 GEMMs, which is a batch, not a token.

## Two-sided per-token decomposition, each side from its own instrumentation

Their `ref-calls.jsonl` resolves into two block shapes: prefill blocks (1886-2836 calls, 11 m=1
GEMMs, 363-740 ms) and **decode blocks of exactly 1700 calls with 203 m=1 GEMMs, device 136.4-137.0 ms,
host 26.2-26.7 ms, copies 0.59 ms**. One block is one token -- not the 4-token batch I first
normalised by, which had made the count look like 487 and briefly contradicted the archive's 203. The
archive was right.

| per token | ours | reference | delta |
| --- | --: | --: | --: |
| device submits | **307** | **203** | **+104** |
| device ms | 142.66 | 136.6 | **+6.1** |
| us per submit | 465 | 673 | ours are 31% cheaper each |
| host CPU ms | 32.35 | 26.4 | **+6.0** |
| host calls | 910 | **1700** | ours are *fewer* |
| us per host call | 35.5 | 15.5 | **ours are 2.3x dearer** |
| copies ms | -- | 0.59 | -- |

Sources: ours = `decode-kernels/MANIFEST.md` (device, 307 submits) plus the itemiser's 175.01 ms total
(host = total - device); theirs = `ref-calls.jsonl`, median of the eight 1700-call blocks. Their
profile carries per-call `clock_gettime`, so their 163.5 ms block total sits ~10 ms above the 153.2 ms
their own harness reports -- the same block, measured with and without instrumentation.

**What this refutes.** The "they batch 32 tokens so they make far fewer host calls" story is wrong:
they make **1700 host calls per token against our 910**. The advantage is not fewer calls, it is
**cost per call** -- 15.5 us against our 35.5. Likewise on the device they make *fewer but dearer*
submits (673 us against our 465).

**What it implies.** Both remaining rows are efficiency-per-call, which is exactly what kernel-level
porting addresses, and it is what the four landed changes did (word-load, zero-pad, greedy, argmax).
The structural-batching project is not indicated by this data and should not be started on the
strength of it.

## Matched-kernel comparison: the host gap is dispatch, not arithmetic

The decode graphs are the same. Their 1700-call decode block and our 910-call token contain the
**same kernel names with the same call counts**: `E_2_16_32_4` 48, `E_8_32_4_4` 24, `E_1024_4`
48/72, `E_640_4` 98/97, `r_16_32_4` 48, `r_640_4` 65, `E_8192_4` 24, `E_16_64_4` 8, `E_8_4` 48,
`E_62080_4` 1. So this is not two different models -- it is the same graph, compiled differently.

| kernel | calls ours/theirs | theirs µs/call | ours µs/call | ratio |
| --- | --- | --: | --: | --: |
| `E_8_4` (8 elements!) | 48/48 | **2.2** | **8.7** | **4.0x** |
| `r_640_4` | 65/65 | 3.0 | 13.6 | **4.5x** |
| `E_640_4` | 97/98 | 6.9 | 14.2 | 2.1x |
| `E_1024_4` | 72/48 | 16.3 | 23.9 | 1.5x |
| `r_16_32_4` | 48/48 | 16.1 | 23.6 | 1.5x |
| `E_8192_4` | 24/24 | 25.2 | 34.6 | 1.4x |
| `E_2_16_32_4` | 48/48 | 50.6 | 46.8 | 0.9x |
| `E_124160_2` | 1/1 | 2129.7 | 2104 | 1.0x |
| attention | 8/8 | **34.4** | **290** | **8.4x** |
| argmax | 1/1 | 86.3 | 96.6 | 1.1x |
| `E_62080_4` | 1/1 | 8523.8 | 433.9 | **0.05x (ours wins)** |

An 8-element kernel costing 8.7 us against 2.2 us cannot be arithmetic. It is the fixed dispatch
cost, and the reference's source shows why:

```python
# theirs (ops_cpu.py, CPUProgram.__call__)
self.fxn(*[ctypes.c_uint64(x) for x in args])
# ours (same method, before this change)
kernargs = ctypes.create_string_buffer(hdr + len(args) * 8)   # malloc + zero, every launch
self.packer.pack_into(kernargs, hdr, *args)
self.fxn(kernargs)
```

Every CPU kernel in this tree takes `void *args` and reads its pointers back out of that buffer; the
reference passes typed ctypes arguments. At ~900 launches per token that is a malloc per launch.

**Fixed so far:** the buffer is now allocated once per `CPUProgram` and reused. Total CPU
**175.01 -> 173.92 ms/token**, parity `895365e2` unchanged. Worth 1.09 ms of the 6.07 ms host gap;
too small for the ABBA to resolve (0.6%), so the itemiser is the instrument and the wall claim is
deliberately not made.

**Still open:** the rest of the dispatch difference is the ABI itself. Porting the typed-argument
signature means changing `ClangRenderer` and every custom kernel in `extra/rockchip` (`attn.c`,
`scan`, `argmax_native` all use `void *args`), for an estimated 4-5 us/call over ~900 calls. That is
a project, and it is the largest remaining *identified* host item after attention.

## The per-call dispatch overhead: found, removed, and it does not reach the wall

The matched-kernel table pointed at dispatch: `E_8_4` is 8 elements at 8.7 us against their 2.2.
Two candidates were tested by direct measurement.

**Refuted: the argument ABI.** Their `CPUProgram` passes typed ctypes arguments, ours packs a buffer.
A trivial kernel called 200,000 times through both styles:

| dispatch style | ns/call |
| --- | --: |
| packed args buffer (ours) | **694** |
| typed ctypes args (theirs) | 844 |

Typed arguments are **150 ns slower** -- `c_uint64` construction costs more than `pack_into` plus one
ctypes call. The ABI port I had queued as "the largest remaining identified host item" is worth
**-0.135 ms** if done, i.e. nothing. Dropped.

**Confirmed: `cpu_profile` per call.** `CPUProgram.__call__` entered a generator context manager and
built a `ProfileRangeEvent` on *every* launch, then discarded it when `PROFILE` is off. Bypassed when
profiling is disabled:

| | before | after |
| --- | --: | --: |
| total CPU, itemiser | 174.20 ms | **167.96** |
| `E_640_4` | 13.5 us/call | **6.8** (theirs 6.9) |
| `r_640_4` | 13.1 | **6.6** (theirs 3.0) |
| `E_8_4` | 12.4 | **6.0** (theirs 2.2) |
| parity | `895365e2` | **`895365e2`** |

**-6.24 ms of CPU, and no wall movement.** The paired A/B (arm A re-enables `PROFILE`, i.e. the old
path; arm B is the new one):

| arm | median | fast mode |
| --- | --: | --: |
| A profiled (old) | 192.62 | 171.5 |
| B bare (new) | 189.08 | 171.4 |

7 of 12 pairs, both arms bimodal across 171-245 ms, fast modes within 0.1 ms. So the 6.24 ms of
host work is **not on the critical path** -- the third time a CPU-side saving has failed to transfer
here (fills, sampler, this). The per-kernel numbers now match the reference and the wall does not,
which means the wall's 171 ms is set by something the itemiser does not attribute: the device path,
not the host.

**Kept anyway**: it is strictly less work per call, parity is unchanged, and the per-kernel table is
now comparable with theirs. But it is not a step toward 153.2 and the gap stays +17.8.

## Device, per GEMM shape: where the 104 extra submits are

Both per-token tables, joined on `(n, k)`:

| shape (n,k) | ours calls / ms | theirs calls / ms | delta |
| --- | ---: | ---: | ---: |
| 9216 x 2560 | **64** / 49.78 | **0** | +64 calls |
| 18432 x 2560 | 0 | **32** / 48.29 | -32 calls |
| 2560 x 4096 | 96 / **40.01** | 96 / **33.70** | **+6.31 ms, same calls** |
| 24576 x 2560 | 10 / 20.19 | (24384) 10 / 19.58 | +0.61 |
| 8192 x 2560 | **24** / 16.62 | 0 | +24 calls |
| 4096 x 2560 | **24** / 8.63 | 0 | +24 calls |
| 12352 x 2560 | 0 | **24** / 24.52 | -24 calls |
| 10240 x 2560 | 8 / 6.84 | 8 / 6.76 | +0.08 |
| 2560 x 1024 | 32 / 3.61 | 32 / 3.22 | +0.39 |
| 32 x 2560 | **48** / 1.44 | 0 | +48 calls |
| **total** | **307 / 147.36** | **203 / 136.44** | **+104 calls, +10.92 ms** |

Three findings:

1. **`+6.31 ms` is a same-shape inefficiency.** 96 calls in both trees, `n=2560, k=4096`, ours
   417 us/call against their 351. At 10.49 MB of weights per call that is **25.2 GB/s against
   29.9 GB/s**. This is the single largest device item and it is not a submit-count problem.
2. **`64 -> 32` calls is gate/up fusion.** Their `n=18432` is exactly `2 x 9216`. We already have
   the code: `npu_linear.fuse_linears` fuses gate/up and its own docstring records the blocker --
   *"MEASURED AND REJECTED -- do not call this. With 40 groups fused, prefill went 2,771 -> 3,179 ms
   (+14.7%)"*, because `n=19456` exceeds the 12288 prefill width so the call still splits, and the
   fused weight is packed as one 24576-column tile. At `m == 1` the limit is 24576, so decode would
   get one submit instead of two: **32 fewer submits, ~1.5 ms**. The switch is `if False and ...`.
3. **`8192 + 4096 -> 12352` is the qkv fusion** (12288 padded to 12352): 24 fewer submits, ~0.73 ms,
   same function, same blocker.

Plus `48 x n=32` submits costing 1.44 ms for 48 tiny GEMMs.

**Discarded measurement:** a cores sweep for `n=2560,k=4096` (1/2/3) via `packed_matmul_int8` in a
loop reported 96 ms/call, i.e. the JIT never captured the call and every iteration re-scheduled.
That is M90 again and it says nothing about the device. A cores sweep has to be in-model.

**Blocked on a decision, not on work:**

- enabling gate/up + qkv fusion removes **56 of the 104** extra submits (~2.2 ms) but its own
  recorded cost is **+14.7% prefill**, against a frozen 1,198.87 ms target;
- the `+6.31 ms` same-shape gap points at CBUF/tiling width (`ONPU_INT8_DECODE_NARROW_CBUF` in their
  recipe), and the only CBUF experiment this tree has run (k=9216) wedged the board and rebooted it.

### Attempt on the +6.31 ms same-shape gap: the core split is not it

`n=2560, k=4096`, 96 calls in both trees, ours 417.8 us against their 351. The fused forward
hard-codes `cores=3`, so the first hypothesis was that 2560 is being split badly. A/B **in-model**
(JIT captured, itemiser over 5 warm decode steps, override on `npu_linear.CORES_OVERRIDE[2560]`):

| cores for n=2560 | per call | total |
| --- | --: | --: |
| **3 (shipped)** | **417.8 us** | **40.11 ms** |
| 1 | 965.2 us | 92.65 ms |

Three cores are worth 2.3x and are already in use. The hypothesis is refuted, so the 19% is not the
core split. What is left is the task plan itself (`int8_task_plan`: tasks per submit, CBUF banks,
subcore assignment) -- the recipe area that wedged this board at k=9216, and which I have not
touched.

`npu_linear.CORES_OVERRIDE` is kept: default empty (so every shape uses 3, the shipped value) and it
is the hook an in-model A/B needs, since the fused forward does not expose the argument.

## Reconciling the unattributed 7.8 ms: it is a comparison artefact, not work

| | device | host | copies | sum | wall | outside |
| --- | --: | --: | --: | --: | --: | --: |
| ours | 142.66 | 24.85 | -- | **167.51** | **171.0** | +3.49 |
| theirs | 136.44 | 26.28 | 0.59 | **163.31** | **153.2** | **-10.11** |

Their profiled per-label sum **exceeds their own unprofiled wall by 10.11 ms**. The two numbers come
from different runs of their tree (`--runs 3` unprofiled for 153.2, `--runs 1 --profile` for the
labels), so the 10.11 ms is either instrumentation (5.9 us/call over 1700 calls) or run-to-run
variation -- and either way it means **summing profiled labels from one run and comparing against an
unprofiled wall from another cannot reconcile by construction**.

So the "+7.8 ms unattributed" was my own arithmetic mixing two runs. It is not work. What remains
defensible:

- the wall-to-wall gap is **+17.8 ms** (both sides measured properly);
- the per-shape device table gives **+10.92 ms** (same-instrument deltas, one run per side);
- the host comparison is **at or slightly better than theirs, with an unknown bias**: our 24.85 ms
  has no instrumentation, their 26.28 ms is a profiled run. The honest statement is that the host is
  no longer the gap, not that it is proven equal.

## No software tiling knob is left for n=2560

| knob | value | effect at columns=2560 |
| --- | --- | --- |
| `TILE` | 24576 | 1 piece (2560 < 24576) |
| `width` (m==1 rule) | 24576 | 1 submit (no width split) |
| `cores` | 3 | `_matmul_parts` = (0,864),(864,864),(1728,832) |

`n=2560, k=4096` already runs as **one submit, three tasks, three cores**, with a 864/864/832 split.
The A/B the last turn asked for (TILE / the 24576 width choice) therefore has **nothing to vary**:
any setting reproduces the same single call. The remaining 19% against their 351 us/call is inside
`int8_task_plan` -- tasks, banks, subcore assignment -- not in the Python tiling.

## Landed: ssm_alpha + ssm_beta fused (48 submits -> 24), and why balancing the 2560 split cannot help

`ssm_alpha` and `ssm_beta` are both `Linear(dim, num_v_heads=32)` on the same `x`, 2 submits per
GDN layer. Fusing them to one `n=64` GEMM (24 GDN blocks, so 48 -> 24 submits/token):

| | before | after |
| --- | --: | --: |
| `rockchip_linear_1_32_2560` | 48 calls, **1.47 ms** | **gone** |
| `rockchip_linear_1_64_2560` | -- | 24 calls, **0.98 ms** |
| total CPU, itemiser | 167.54 ms | **167.19** |
| parity | `895365e2` | **`895365e2`** |
| prefill spot-check (`bench_llm` TTFT, same config) | 4251.8 ms | **4243.8** |

Unlike gate/up this cannot push prefill over a tiling boundary: `n=64` against a 12288 prefill width.

**The bug that made the first attempt silent:** the model's projections are
`tinygrad.llm.kernels.amd.Linear`, a **subclass** of `nn.Linear`, so `type(x) is nn.Linear` matched
nothing and `fuse_linears` fused 0 of 24 blocks while reporting success. `isinstance` fixed it.
Measured the hard way: the itemiser still showed 48 x `n=32` after the rule was added.

**Balance 864/864/832 -> 864/864/864: not tested, because it cannot help.** `_matmul_parts(1, 2560,
4096, 3)` returns `(0,864), (864,864), (1728,832)`. Per-call time is set by the **largest** part,
which is already 864; padding `n` to 2592 raises the smallest part to 864 and adds 1.25% of work
without touching the critical part. Reporting it as an untested hypothesis with that arithmetic is
more useful than a run that can only come back neutral.

### attn_qkv + attn_gate fused to 12288: decode wins, prefill spot-check loses, reverted

`attn_qkv` (4096) + `attn_gate` (8192) = **12288**, exactly the prefill width limit -- the case the
`fuse_linears` docstring's objection does *not* cover (gate/up is 19456, over the limit). Their tree
runs the same pair as one 12352-wide GEMM.

| | before | after |
| --- | --: | --: |
| `8192x2560` | 24 calls, 16.62 ms | -- |
| `4096x2560` | 24 calls, 8.87 ms | -- |
| `12288x2560` | -- | 24 calls, **24.86 ms** |
| submits/token | 283 | **259** |
| total CPU | 167.19 ms | **166.48** |
| parity | `895365e2` | **`895365e2`** |
| prefill TTFT spot-check | 4243.8 ms | **4336 ms (+2.2%)** |

The decode side is exactly what the per-shape table predicted (-0.71 ms, 24 fewer submits, 1.6%
off their 24.52 ms for the same shape), and parity held.

**Settled by measurement, and rejected.** `prefill_ab.py` runs prefill in the *native* configuration
-- the one whose 1,198.87 ms target is frozen -- with the flag flipped per process, in ABBA order:

| arm | stabilized trials (ms) | median |
| --- | --- | --: |
| off | 1180.4, 1179.8 | **1180.4** |
| on | 1361.3, 1342.1 | **1361.3** |
| on | 1456.2, 1457.9 | 1457.9 |
| off | 1186.9, 1189.5 | **1189.5** |

**+15% to +22% prefill** against -0.71 ms of decode, consistent across both orders and one thermal
band (73-75 C). `fused_groups` goes 32 -> 56, so the arm demonstrably ran. That is the same verdict
the gate/up docstring recorded at 19456 (+14.7%): a wider GEMM costs prefill more than the submit it
saves, even at 12288, which is exactly the width limit.

**The measurement error worth keeping:** my earlier spot-check was a `bench_llm --warmups 1 --runs 1`
run and reported +2.2%, understating a +15-22% regression by an order of magnitude -- because that
script's prefill (4.2 s) is a different, CPU-dominated configuration from the frozen one (1.18 s).
Measuring the right thing in the wrong configuration gave a number that was both precise and
irrelevant. `FUSE_QKV_GATE` stays False and the rule stays in the tree behind it.

**Also at the resolution limit:** the shipped alpha/beta fusion measured 167.54 -> 167.19 when it
landed and 167.64 on a later run of the same code, so its *total* effect (~0.35 ms) is inside the
itemiser's run-to-run spread (~0.45 ms). Its structural effect is solid -- 48 x `n=32` (1.47 ms) is
now 24 x `n=64` (0.96 ms) and 24 submits are gone.

## The CBUF experiment is withdrawn: the register program is already theirs

The +6.31 ms same-shape gap on `m=1, n=2560, k=4096` pointed at `int8_task_plan` (tasks, banks,
subcores), and testing it needed authorisation because that recipe family wedged the board at
k=9216. Before asking, the comparison was done from source -- no board access, no register change.

**CBUF_CON0 for this shape is `0xb1` in both trees.** Their `tinygrad/runtime/support/rockchip_int8.py:135-146`
is the same formula as our `tinygrad/runtime/ops_rockchip.py:136-146`, including the same override:

```python
groups = ceildiv(k,512)                                   # 8 at k=4096
banks  = max(0x1b, 177-15*(groups-1)-15*groups*(ceildiv(m,64)-1))
if m==1 and k==4096 and (n==512 or n>=1024): banks = 0xb1  # both trees
```

Their extra clause (`n in NARROW_DECODE_WIDTHS and ONPU_INT8_DECODE_NARROW_CBUF`) is unreachable
for `n=2560` because `n>=1024` already matches. Then the whole emitted register set was parsed and
diffed:

| | registers parsed | identical | differing |
| --- | --: | --: | --: |
| ours vs theirs, int8 body | 31 | **31** | **0** |

Byte-identical across CNA, DPU and CORE targets. So:

1. **The CBUF/task-plan experiment has no parameter to change for this shape.** It is withdrawn, and
   the board-wedge risk it carried buys nothing. That authorisation is not needed.
2. **The +6.31 ms is not in the register recipe.** It has to be in what surrounds the registers: the
   task/core mapping (`_encode_task_chain` builds **3 tasks with 3 bodies**, one per `_matmul_parts`
   slice `(0,864),(864,864),(1728,832)`, while one task with three subcores would set up CNA/CBUF
   once), or in the buffer stride/padding, or in how the per-call CPU is attributed.

The next comparison is therefore **task granularity, not registers** -- same math, one CNA setup
instead of three. That is a task-encoding change, which is a different risk class from the register
program that wedged the board.

## Task encoding and padding are also identical: what is left of the +6.31 ms

The 1-task-with-3-subcores hypothesis is refuted as well. Their encoder
(`runtime/ops_rockchip.py:222`) is the same expression as ours, character for character:

```python
self.subcores = ((0,self.tasks),) if active_cores==1 else (
  (0,0),(0,0),*((i*(self.tasks//3),self.tasks//3) for i in range(3)))
```

So they also emit N tasks with one subcore each -- not one task with three subcores. And the K
padding matches: their `padded_depth(4096)` returns 4096, ours `_int8_padded_depth(4096)` returns
4096, so neither tree pads this depth.

Everything structural that can be read has now been compared for `m=1, n=2560, k=4096`:

| aspect | ours vs theirs |
| --- | --- |
| emitted int8 registers (CNA/DPU/CORE) | **21 of 31 identical, 0 differing** |
| `CBUF_CON0` | 0xb1 both |
| task encoding / subcore expression | identical source |
| K padding at 4096 | 4096 both |
| core split | 3 parts (864/864/832) both use `_matmul_parts` |

**The remaining candidate, and it is a hypothesis:** their per-token cost may be lower because their
decode loop runs four tokens inside one C function, so the next token's weight DMA can overlap the
current token's compute. 10.5 MB streams from DRAM per call; four calls in a loop can pipeline the
stream, four calls issued from Python cannot, because each one waits for its own submit. That would
explain an 18% per-call advantage with byte-identical registers -- and it is the same
`CPU_TOKEN_INPUTS` / `DecodeLoop` mechanism already listed as a structural gap, not a kernel.

It is labelled a hypothesis because it cannot be tested without building the token loop, and every
cheaper explanation has now been measured and eliminated.
