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
| `BORROW_BATCH` + `BORROW_OUTPUTS` (`rockchip_replay.py:391,405`, both **default 0**) | per-token copies of state/output buffers | **absent** -- we copy, and the copy emits fills | fills **9.43 ms** | ~0 | **+9.4** |
| `SHARED_HOST_POOL` + `HOST_POOL_WFE` (`rockchip_shared_pool.py`) | per-kernel dispatch of host chains; kernels run in one shared pool | **absent at decode** -- `thread_split` is harness-level and prefill-scoped | elementwise **20.04 ms** | 6.57 | **+13.5** |
| `GDN_DIRECT_DECODE` + `GDN_DECODE_PREP_POOL` + `GDN_DECODE_SPIN` | Python-side GDN bookkeeping per step | `scan_native` (kernel only, no prep pool) | scan **5.22 ms** | 4.16 | +1.1 |
| `REPLAY_NATIVE_FROM` + `_LARGE` | per-call `sync_mem` ioctls | `sync_mem` ioctls intact | 2.8 us/call = 0.87 ms | ~0 | +0.9 |
| `ZERO_PAD_VIEWS` (`rockchip_padding.py:13 mark_zero_tail`, `:37 capacity_input`) | materialising zero padding -- pad becomes a **view over a pre-zeroed full buffer**, so the scan writes the full buffer in place | **absent** -- `model.py:819 where(0, state)` materialises a 2 MB copy per block per token | inside the fills row (9.43) | -- | -- |
| `CPU_TOKEN_INPUTS` + `gpt2_decode:DecodeLoop` | one Python->graph entry per token | we re-enter `model(...)` per token | outside-kernel **9.32 ms** | ~0 | +9.3 |
| `INT8_DECODE_NARROW_CBUF`, `INT8_NATURAL2304` | CBUF/geometry slack in the device recipe | ported where it applies | device **142.65 ms** | 137.11 | +5.5 |
| `SWIGLU_PACK_STATIC_LAYOUT`, `EXP2_FAST_PATH`, `FFN_PACKED_UPLOAD`, `SCALAR_COPY`, `DEDUP_GUARDS` | assorted copies and dispatch | partial / absent | not separately measured | -- | -- |
| `NPU_DOMAINS=3` | IOMMU domain count | we run **2** | not separately measured | -- | -- |

Sum of the measured deltas: **9.4 + 13.5 + 1.1 + 0.9 + 9.3 + 5.5 = 39.7 ms**, against a measured
gap of 32.4. The classes overlap somewhat (the outside-kernel time and the fills are both buffer
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
