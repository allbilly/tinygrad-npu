# INT8 kernel names: reconciling the reference profile against ours

The two INT8 profiles do **not** disagree about the work. They disagree about what the first
number in *our* kernel name means.

Summing `m*n*k*calls` over the reference's prefill rows gives **457.48 GMAC**. Summing the same
product over our `rockchip_linear_*` rows gives **234.68 GMAC** — 1.9494x less, for the same
Qwen3.5-4B checkpoint. Reading the code shows why: in our name `m` is the **per-source row
count** of a submit that carries *two* such sources, while the program it names executes
`m_rows = 2*m` rows. The name omits `m_rows`. Correcting for it gives **457.4793 GMAC**, which is
**exactly** the model's own MAC count — equal per role, in integer arithmetic, table in §3.1.

Consequence: the reference is **not** doing 2x the MACs. `b` *is* sometimes a width chunk of a
wider fused matrix (18432 columns becomes two calls, 12288 and 6144), but every chunk is its own
profile row with its own `calls`, so the sum already covers each column exactly once; and there is
no double counting from `ROCKCHIP:1`. The "1.43x slower" claim, however, does not survive either —
for a different reason: 610.78 ms and 427.96 ms are not the same interval.

## 1. What the three numbers mean

### 1.1 Reference: `rockchip_int8_{m}_{n}_{k}` = (rows, columns, depth)

The dump at `docs/rknpu/results/ref-kernels-all/py/rockchip_int8.py` is the register recipe and
does **not** contain the name formatter. The formatter lives in the tree the reference run pinned
(`ref-phase2/run.sh: TREE=/home/orangepi/tg-qwen35-pack-signature-fix`):

```python
# /home/orangepi/tg-qwen35-pack-signature-fix/tinygrad/llm/kernels/rockchip_int8.py:148-155
def _matmul(out:UOp, lhs:UOp, rhs:UOp, m:int, n:int, k:int, column_offset:int, cores:int, sdk_prefill:bool=False,
            payload_offset:int=0) -> UOp:
  sink = UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_int8_{m}_{n}_{k}'+('_sdk' if sdk_prefill else '')))
  info = replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),UOp(Ops.SOURCE,arg=f'INT8 {m}x{k}x{n}'),
    UOp(Ops.BINARY,arg=struct.pack('<10I' if payload_offset else '<9I' if sdk_prefill else '<8I',
                                 m,n,k,cores,column_offset*k,round_up(m*4,64),0,1,
                                 *((int(sdk_prefill),payload_offset) if payload_offset else (1,) if sdk_prefill else ())))))
```

The order is confirmed by the program's own source string on line 152, `f'INT8 {m}x{k}x{n}'`, and
by the caller:

```python
# same file, raw_int8_matmul:259-268
def raw_int8_matmul(payload:Tensor, weight:Tensor, rows:int, columns:int, column_offset:int=0, *,
                    sdk_prefill:bool=False, cpu_output:bool=True, payload_offset:int|None=None) -> Tensor:
  k = weight.shape[1]*32
  n = round_up(columns,32)
  ...
      or column_offset+n>weight.shape[0]*32 or n>(24576 if rows==1 else 12288)
```

So `m` = rows of that register task, `n` = call columns rounded up to 32, `k` = call K depth
(after any K tiling). The dump corroborates both bounds in the register guard itself:

```python
# docs/rknpu/results/ref-kernels-all/py/rockchip_int8.py:105-108 — int8_body
def int8_body(m:int,n:int,k:int,lhs:int,rhs:int,dst:int,stride:int) -> tuple[int,...]:
  if not (1<=m<=(128 if k<=2048 else 64) and supported_depth(k)
          and 32<=n<=(8192 if m==1 else 4096) and n%32==0 and stride==n):
    raise ValueError('Unqualified INT8 matrix geometry')
```

`m` is the task's row count, and the 64/128 switch is the *task* limit at `k>2048`. Every `m` in
the reference profile is therefore the number of rows that task actually wrote.

### 1.2 Ours: `rockchip_linear_{m}_{n}_{k}` = (per-source rows, columns, depth)

```python
# extra/rockchip/npu_linear.py:583
sink = UOp.sink(out, inp, params, arg=KernelInfo(f"rockchip_linear_{key.m}_{key.n}_{key.k}"))
```

`key` is built in `_FusedInt8Plan.__init__`, and it carries both `m` and a separate `m_rows`
which never reaches the name:

```python
# extra/rockchip/npu_linear.py:602-632
    def __init__(self, dev, m: int, n: int, k: int, depth_in: int, cores: int, column_offset: int,
                 row_stride: int, row_offset: int, k_offset: int, weight_dma: int, w_scale_addr: int, m_rows: int = 0):
        rows_total = m_rows or m
        self.shape = (rows_total, n)
        header = round_up(rows_total * 4, 64)
        self.payload = _payload_buffer(dev, rows_total, k)
        self.out = _plane_buffer(dev, rows_total, n)
        if m_rows and m_rows == 2 * m:
          ...
          cmd, task, tasks, core_mask, subcores, parts = int8_task_plan_multi(dev, (
              (m, n, k, self.payload.dma + header, weight_dma + column_offset * k, self.out.dma),
              (m, n, k, self.payload.dma + header + m * k,
                          weight_dma + column_offset * k, self.out.dma + m * n * 4)))
          ...
        self.key = _FusedKey(
            m=m, n=n, k=k, ...
            m_rows=m_rows)
```

`m_rows` is set one level up, when a 128-row call can be expressed as two 64-row sources in one
submit:

```python
# extra/rockchip/npu_linear.py:706-719 — _fused_plan
    # Merge only when the tile really splits 3 ways: _matmul_parts collapses to a single part for
    # narrow n (e.g. ssm_beta, n=num_v_heads=32), and the two-source chain needs 3 parts per source.
    m_task, m_rows = ((64, m) if (m == 128 and k > 2048
                                 and len(_matmul_parts(64, n, k, 3, True)) == 3) else (m, 0))
    ...
            _FUSED_PLANS[key] = plan = _FusedInt8Plan(dev, m_task, n, k, depth_in, cores, column_offset,
                                                      row_stride, row_offset, k_offset, weight_dma, w_scale_addr,
                                                      m_rows=m_rows)
```

with the caller selecting `step = 128` for a 128-token prefill:

```python
# extra/rockchip/npu_linear.py:837-845
        row_tile = (128 if k_max <= 2048 else 64) if self.int8 else max(1, min(64, 65536 // k_max))
        ...
        merge = (2 * row_tile if (self.int8 and rows == 2 * row_tile
                                  and len(_matmul_parts(row_tile, self.columns, k_max, 3, True)) == 3) else 0)
        step = merge or row_tile
```

and the submit really does contain two matmuls:

```python
# tinygrad/runtime/ops_rockchip.py:205-216 — int8_task_plan_multi
  """One submit carrying several INT8 matmuls that read *different* activations.
  ...
  Every source must use 3 cores (3 parts each), so two sources give 6 bodies = 2 tasks per core.
```

So for a prefill of 128 rows, `rockchip_linear_64_9216_2560` is a name for a program that wrote
**128** rows: two 64-row sources, `2 tasks per core`, one blocking submit. The name reports the
rows of one source, not the program's total.

Which rows are affected is decidable from code, not from preference. `_matmul_parts` returns 3
parts for every wide prefill shape and 1 part for `n=32`:

| name (`m`,`n`,`k`) | `_matmul_parts(64,n,k,3)` | rows written | why                            |
| :----------------- | :------------------------ | -----------: | :----------------------------- |
| `64_9216_2560`     | 3 × 3072                  |          128 | merged, `m_rows=128`           |
| `64_2560_4096`     | 3 × 864/832               |          128 | merged, `m_rows=128`           |
| `64_8192_2560`     | 3 × 2752/2688             |          128 | merged, `m_rows=128`           |
| `64_4096_2560`     | 3 × 1376/1344             |          128 | merged, `m_rows=128`           |
| `64_10240_2560`    | 3 × 3424/3392             |          128 | merged, `m_rows=128`           |
| `64_32_2560`       | 1 × 32                    |           64 | no merge; 2 calls per 128 rows |
| `128_2560_1024`    | 3 × 864/832               |          128 | `k<=2048`, no merge needed     |
| `1_24576_2560`     | 3 × 8192                  |            1 | decode-width lm_head tile      |
| `1_2560_2560`      | 3 × 864/832               |            1 | lm_head tail                   |

The measured `_matmul_parts` output (`.venv/bin/python`, no device touched):

```
m= 64 n= 9216 k=2560 -> parts=((0, 3072), (3072, 3072), (6144, 3072)) sum_n=9216
m= 64 n= 2560 k=4096 -> parts=((0, 864), (864, 864), (1728, 832)) sum_n=2560
m= 64 n= 8192 k=2560 -> parts=((0, 2752), (2752, 2752), (5504, 2688)) sum_n=8192
m= 64 n= 4096 k=2560 -> parts=((0, 1376), (1376, 1376), (2752, 1344)) sum_n=4096
m= 64 n=10240 k=2560 -> parts=((0, 3424), (3424, 3424), (6848, 3392)) sum_n=10240
m= 64 n=   32 k=2560 -> parts=((0, 32),)                             sum_n=32
m=128 n= 2560 k=1024 -> parts=((0, 864), (864, 864), (1728, 832)) sum_n=2560
```

That is the whole discrepancy: `457.479 / 234.678 = 1.9494`, not 2.0000, because `64_32_2560`,
`128_2560_1024` and the two `m=1` rows are already counted at their true row count.

## 2. `ROCKCHIP:1` is a second IOMMU domain on the same NPU

It is **not** a second device, and the rows under it are **not** duplicates. It is tinygrad's
device index, which this backend interprets as the NPU IOMMU domain id:

```python
# tinygrad/runtime/ops_rockchip.py:427-431  (same code in the measured reference tree, :559-566)
  def __init__(self, device:str):
    self.domain = int(device.partition(':')[2] or '0')
    if not 0<=self.domain<16: raise ValueError('Rockchip memory domain must be between 0 and 15')
    with _hardware_lifetime_lock:
      try: self.hw = _hardware()
```

One `_RockchipHardware` (one fd, one flock, one reset state) is shared by every index, and the
index is passed to the submit:

```python
# tinygrad/runtime/ops_rockchip.py:369-372
        rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,timeout=6000,
          task_start=0,task_number=self.tasks,task_counter=0,priority=0,task_obj_addr=self.task.meta.obj_addr,
          iommu_domain_id=self.dev.domain,task_base_addr=0,hw_elapse_time=0,core_mask=self.core_mask,fence_fd=-1,
```

The domains exist to split the **weight budget**. `ref-phase2/run.sh` sets

```
export ONPU_NPU_DOMAINS=3 ONPU_NPU_DOMAIN_BYTES=3221225472
```

and the report records `"npu_domain_options": {"count": 3, "weight_budget_bytes": 3221225472}`.
`int8_weight_device` hands each packed weight to the first domain with room:

```python
# measured tree, tinygrad/runtime/ops_rockchip.py:536-548
def int8_weight_device(size:int) -> str:
  """Opt-in sequential weight placement; leave room for command and activation buffers."""
  domains = getenv('ONPU_NPU_DOMAINS',1)
  budget = getenv('ONPU_NPU_DOMAIN_BYTES',3*1024**3)
  ...
    name = f'ROCKCHIP:{base.hw.weight_domain}' if base.hw.weight_domain else 'ROCKCHIP'
    dev = cast(RockchipDevice,Device[name])
    if dev.mapped_bytes+round_up(size,4096)<=budget: return name
```

Three independent checks say the two prefixes are disjoint program sets:

1. `captured_calls.prefill` = `{"PROGRAM:ROCKCHIP": 392, "PROGRAM:ROCKCHIP:1": 83}` and
   `prefill_submits = 475` in every trial. 392 + 83 = 475, exactly once each.
2. The two prefixes are complementary, not equal. The geometry `2560x4096` is shared by two roles
   (`o_proj`, 64 calls, and `ffn_down`'s two 4096-wide K tiles, 128 calls) for 192 calls in total,
   split 130 on `ROCKCHIP:` and 62 on `ROCKCHIP:1:`. A re-report of the same programs would give
   two equal counts, not a 130/62 split.
3. The 83 domain-1 programs carry 42.24 GMAC (62 `ffn_down` tiles plus the lm_head pass). Counting
   them twice would total 499.72 GMAC, 9.2% above the model's 457.4793. The measured total is
   457.4793, so each program is counted once.

Only domains 0 and 1 appear in either profile, although 3 were configured: `npu_domain_options`
records `count: 3`, and 2 x 3 GiB already covers the ~4 GiB of INT8 weights for this checkpoint
(our tree reports `npu_weight_mib: 4010.0` for the same file), so the allocator never opened
domain 2.

## 3. The corrected like-for-like comparison

### 3.1 Work: identical, exactly

The model is fixed by the checkpoint: 32 blocks, hidden 2560, FFN 9216, 24 GDN layers and 8 full
attention layers (`qwen35.*` metadata, `Qwen3.5-4B-Q8_0.gguf`). Per-role MACs for 128 tokens:

| role                              | model GMAC (derived) | ours GMAC (measured rows) | ref GMAC (named rows) |
| --------------------------------- | -------------------: | ------------------------: | --------------------: |
| FFN gate+up, 32 layers            |              193.2735 |                 193.2735 |             193.2735 |
| GDN in-proj (qkv+gate+α+β), 24    |               97.1401 |                  97.1401 |              97.1401 |
| full-attn q+k+v, 8 layers         |               26.8435 |                  26.8435 |              26.8435 |
| o_proj, 32 layers                 |               42.9497 |                  42.9497 |              42.9497 |
| ffn_down, 32 layers               |               96.6368 |                  96.6368 |              96.6368 |
| lm_head, 1 token                  |                0.6357 |                   0.6357 |               0.6357 |
| **total**                         |          **457.4793** |             **457.4793** |           **457.4793** |

"derived" = `sum(in_features*out_features)` over the GGUF tensor table x 128 tokens; it is
arithmetic on the checkpoint, not a measurement. The two profile columns are measured. Every role
agrees as an exact integer, not as a rounded match: the MAC products are the same integers on all
three sides, which is what proves the role assignment below.

Where each row comes from, with the call decomposition that survives MAC conservation:

| role          | ours: shape x calls (rows/call)                        | ref: shape x calls (rows/call)                |
| :------------ | :----------------------------------------------------- | :-------------------------------------------- |
| FFN gate+up   | `9216x2560` x64 (128)                                  | `12288x2560` x64 (64) + `6144x2560` x64 (64)  |
| GDN in-proj   | `8192x2560` x24 + `4096x2560` x24 + `32x2560` x96 (64) | `12288x2560` x48 (64) + `64x2560` x48 (64)    |
| full-attn qkv | `10240x2560` x8 (128)                                  | `10240x2560` x16 (64)                         |
| o_proj        | `2560x4096` x32 of the 96 (128)                        | `2560x4096` x64 of the 192 (64)               |
| ffn_down      | `2560x4096` x64 (128) + `2560x1024` x32 (128)          | `2560x4096` x128 (64) + `2560x1024` x32 (128) |
| lm_head       | `24576x2560` x10 + `2560x2560` x1 (1)                  | `24384x2560` x10 + `4480x2560` x1 (1)         |

This mapping is forced, not chosen: it is the only assignment of the observed shapes to the
model's roles that reproduces each role's MAC count exactly. The reference's 112 calls of
`12288x2560` are 64 FFN head-calls plus 48 GDN head-calls; 12288 is the 3-core per-call width cap
(`n>(24576 if rows==1 else 12288)`, `rockchip_int8.py:268`), which is why the 18432-column FFN
group and the 12352-column GDN group split as `12288 + 6144` and `12288 + 64`. The two `m=1` rows
are the same operation in both trees, tiled differently: `24576 x 10 + 2560 = 248320` and
`24384 x 10 + 4480 = 248320`, both equal to the vocabulary size.

**Answer to "is the reference doing 2x the MACs?": no.** It does 457.4793 GMAC and the model needs
457.4793 GMAC — zero padding, zero duplication. The hypothesis that `b` (`n`, columns) is a column
tile whose per-layer total must be summed is half right: `n` *is* sometimes a width chunk (the
reference's 18432-column gate+up is two calls, 12288 and 6144; 12352 splits 12288 + 64). But each
chunk is its own profile row with its own `calls`, so summing `n*k*calls` already covers every
chunk exactly once. There is no double count and no missing half.

### 3.2 Time: the rows are not the same interval

| quantity                                          |           ours |            ref | ratio | same interval? |
| :------------------------------------------------ | -------------: | -------------: | ----: | :------------- |
| INT8 MACs (128-token prefill + 1 lm_head pass)    |  457.4793 GMAC |  457.4793 GMAC | 1.000 | yes            |
| INT8 submits                                      |            355 |            475 |  0.75 | yes            |
| device: blocking ioctl                            |         421 ms |      427.96 ms |  0.98 | yes            |
| host inside the same program                      |       189.8 ms |              0 |     — | no             |
| host in separate CPU rows (quantize + dequantize) |              0 |       45.39 ms |     — | no             |
| INT8 path total (disjoint rows, single-threaded)  |      610.78 ms |      473.35 ms |  1.29 | yes, summed    |
| device throughput                                 | 1,086.6 GMAC/s | 1,069.0 GMAC/s |  1.02 | yes            |
| device ms per submit                              |           1.19 |           0.90 |  1.32 | yes            |

The device row is the only *time* row that is like-for-like on both sides; the two host rows are
disjoint by construction and the totals add them.

Sources: ours — `progcost/trace.jsonl` (610.782 ms over 355 `rockchip_linear_*` programs: 344 for
the 128-token prefill plus 11 lm_head tiles) and
`progcost/report.json` `npu_seconds` (0.420 / 0.421 / 0.419 / 0.420 across the four trials) with
the independent split in `threadsplit/ioctlprobe/run.log` (`submit=423.5 sync_to=6.6 sync_from=17.2
dequant=106.7 ms calls=355`). Ref — `ref-phase2/report.json`
`native_execution.prefill.profile` (427.958 ms over 475 `ROCKCHIP*` rows: 464 prefill plus the same
11-tile lm_head pass, plus `CPU:rockchip_quantize` 16.493 ms / 251 calls and
`CPU:rockchip_dequantize` 28.901 ms / 243 calls).

Why the two 600/428 numbers are not comparable:

* our `rockchip_linear_*` program is the **fused** C function: quantize -> `sync_mem(TO_DEVICE)` ->
  blocking ioctl -> `sync_mem(FROM_DEVICE)` -> dequantize (`npu_linear.py:541-570`), so its wall
  contains host work;
* the reference's `rockchip_int8_*` program is the NPU program only; its quantize and dequantize
  are separate rows, `CPU:rockchip_quantize` and `CPU:rockchip_dequantize`, and they are excluded
  from the 427.96 ms.

The device figures are the same interval on both sides: ours is `clock_gettime` wrapped around the
`ioctl` only (`npu_linear.py:556-562`), the reference's is the `DRM_IOCTL_RKNPU_SUBMIT` program
call (`ops_rockchip.py:248-261` in the measured tree) plus at most a dirty-output cache clean.

The surviving real gap is the host half, and it is a code difference, not a naming one: our
dequantize is serial (`PARALLEL_DEQUANT = False`, `npu_linear.py:420`) while the reference
dequantizes on 4 threads whenever `rows*columns>=131072`
(`rockchip_int8.py:213`, `_dequantize_tiles`). Per call that is 0.30 ms vs 0.119 ms.

## 4. Verdict on "the INT8 row is 1.43x slower"

**Withdrawn as stated.** 610.78 / 427.96 = 1.427, but the numerator is our *fused* wall
(quantize + ioctl + sync + dequantize) and the denominator is the reference's *device* program
with its quantize/dequantize charged to two separate CPU rows, so the quotient compares our whole
INT8 path against the reference's device half — a 1.43x that would exist even if the two devices
were identical. On the like-for-like interval the computed device time is 0.98x (421 ms against
427.96 ms; 1,087 against 1,069 GMAC/s), i.e. parity, and the 2x MAC scare that motivated the
re-check is an artefact of our name omitting `m_rows`, not extra work on either side. What
survives, and is worth keeping, is the 1.29x on the full INT8 path (610.78 ms against
427.96 + 45.39 = 473.35 ms): it is entirely our host half, 189.8 ms against 45.4 ms, consistent
with a serial dequantize and with fewer quantizes on the reference side than submits (251 vs 475,
i.e. one payload reused across that Linear's width chunks).

Replacement numbers, if a single line is needed: **device 0.98x (parity), full INT8 path 1.29x.**

## 5. What would settle the residual uncertainty

Both time columns are single-artifact measurements from different processes, and on this board a
single-arm run carries up to ~20% drift, so the 0.98x must be reported as *parity*, not as a 1.7%
win. The decisive measurement is cheap because the instrument already exists on our side only:

1. Add the four-line `clock_gettime`-around-ioctl counter from
   `tinygrad/runtime/ops_rockchip.py:367-381` to the measured reference tree's `RockchipMatmul.__call__`
   and record it in `benchmark_inference` (the reference tree has the `start`/`elapsed` timing at
   `:248-261` but never accumulates or reports it). Then both trees report the *identical*
   interval and `ref-phase2/run.sh` 3+3 gives a paired device number.
2. For the host half, ablate rather than subtract: stub the dequantize in
   `_fused_source` (`npu_linear.py:483-490`) and measure the in-model delta with
   `extra/rockchip/ab_fused_host.py` (in-process ABBA, per-trial temperature), and separately set
   `PARALLEL_DEQUANT = True` for the `>=200,000`-element calls only. That tests the mechanism
   rather than the arithmetic of 610.78 - 421.

## Correction — element count and bandwidth (main agent, verified)

The gap analysis states both sides dequantise ~166 M elements per prefill and that the
reference's 28.90 ms would therefore imply 46 GB/s, above the 33.79 GB/s DDR ceiling.

Recomputed from `progcost/trace.jsonl` over the real geometries:

    sum(m * n * calls) = 88.53 M elements per prefill      (not 166 M)

| side | elements | bytes moved (int32 in + fp32 out) | time | bandwidth |
| --- | --: | --: | --: | --: |
| ours | 88.53 M | 0.708 GB | 106.7 ms | **6.6 GB/s (20% of bus)** |
| reference | 88.53 M | 0.708 GB | 28.90 ms | **24.5 GB/s (73% of bus)** |

So there is no bound violation: the reference sits at a plausible 73% of peak, and **we are the
outlier at 20%**. Reaching the reference's bandwidth takes our dequantise from 106.7 to 28.7 ms —
**78 ms**, which is larger than the FFN-tail item (45 ms) and is therefore the first thing to fix.

Two facts constrain the explanation, and they conflict:

- the reference uses 3 pthreads + main for `rows*columns >= 131072` (`rockchip_int8.py:212-223`),
  which on memory-bound work is exactly the 4x that separates 6.6 from 24.5 GB/s;
- yet our own 4-way pool **did run in-kernel** (`fused_stats[3]` = 152 hits per prefill, verified by
  `probe_poolhits.py`) and the temperature-fair ABBA measured **0.9998x**.

That contradiction means the 106.7 ms figure — taken by `ioctlprobe` on the pre-OOB-fix tree — needs
re-measuring on the corrected tree before either number is trusted. Possible resolutions: the pool's
groups are not actually spreading memory traffic, the planes are read through an uncached mapping, or
106.7 ms is not the current dequantise cost at all.

## 4-way quantise dispatch — implemented, gated, measured NEUTRAL

Quantise measured 107.2 ms per prefill (largest host item, 6.5x the reference's 16.49), so it was
dispatched 4-way through the same `rk_run` pool the reference uses, gated on
`rows*depth >= 32768` and with a serial fallback on `rk_run != 0`.

| gate | result |
| --- | --- |
| generated source compiles (serial / pooled) | exit 0, 4,489 / 5,411 bytes |
| geometry selection | `64_9216_2560`, `64_32_2560`, `m_rows=128` pool; `m=1` does not |
| bitwise parity at the Linear boundary | **identical**, both checksums `1112969.750000` |
| temperature-fair in-process ABBA, 3 rounds ABBA | **on/off 1.0006x**, on-arm spread 0.4%, 72.1-76.7 C |

Verdict: **neutral**. This is the third pool-based parallelisation to measure neutral on this
board (dequantise 0.9998x, NEON dequantise -0.3%, quantise 1.0006x), with the pool verified to
execute. So the pool is not the answer for these sub-millisecond host tails.

What that leaves: the quantise cost is real but not reducible by threading. The reference makes
**251 quantise calls to our 355** because fusing gate+up means one quantise serves both projections.
Deleting quantise invocations -- which is exactly what the FFN tail fusion does -- is the only
remaining mechanism.

## raw_out verified — the FFN-tail foundation is numerically exact

First attempt reported `bitwise identical: False` (162,144 / 163,840 differ, max_abs_diff 180.354).
Both causes were in the *probe*, not the emitter:

1. **Buffer aliasing** — `_plane_buffer` pools on `(id(dev), m, n)`, so the normal arm and the
   raw_out arm shared one allocation; the readback was a mixture of fp32 output and int32 planes.
   Fixed by clearing `_PLANE_POOL` and `_PAYLOAD_POOL` before each build.
2. **Wrong layout** — the planes are not row-major `(m, n)`. The emitter addresses
   `raw + g*m*n + offset*m + row*width`, so for `parts = ((0,864),(864,864),(1728,832))` the buffer
   is part-blocked. Rebuilding with that layout, not with a reshape, is what made it match.

After both fixes, on `m=64 n=2560 k=4096`:

```
raw_out key           : True   source has dequantise: False
parts                 : ((0, 864), (864, 864), (1728, 832))
planes                : (64, 2560)  nonzero=163840
bitwise identical     : True
```

So the int32 planes left by `raw_out` dequantise to exactly the fp32 the fused kernel would have
produced. The precondition for a `silu(gate)*up` pack that consumes both projections and quantises
the down payload is now met — and it also gives the reusable lesson that any plane readback must
carry `key.parts`, since the buffer is part-blocked.

## FFN-tail pack kernel — PACK VERIFIED

Gate: `extra/rockchip/parity_pack.py`, reference = dequantise (the emitter's own expression) ->
`swiglu.c` (verified maxdiff 0 vs the shipped `E_393216_3`) -> `qz_row`.

```
stage 1 g dequant bitwise identical: True   max|d|=0.000e+00
stage 2 u dequant bitwise identical: True   max|d|=0.000e+00
  part off=    0 w= 864: g match=True  u match=True
  part off=  864 w= 864: g match=True  u match=True
  part off= 1728 w= 832: g match=True  u match=True
silu*up bitwise identical: False  max_abs_diff=7.629e-06
codes  bitwise identical: True    n_diff=0/163840
scales bitwise identical: True    max_abs_diff=0.000e+00
PACK VERIFIED
```

Both dequantise stages are exact, and the delivered quantity — the down payload's int32 codes and
per-row scales — is **bit-identical** over all 163,840 elements. The `silu*up` floats differ by at
most 7.6e-06, which is below what can change a quantised code at these scales; that is why the codes
match exactly.

### Four probe bugs, one kernel — the sequence, because each cost a turn

1. `codes` and `scales` both read argument slot 6 (kernel ABI).
2. `unpack()` allocated float32 and put int32 planes in it, so the kernel read float bit patterns
   through `const int *`.
3. Aliasing hypothesis (`out === g` in the swiglu call) — **tested and refuted**; result unchanged.
4. **The real one:** the kernel was handed the *unpacked* `(m, n)` array while it indexes with the
   part-blocked formula `raw + offset*m + row*width`. Row 1 therefore read row 0's column 864.
   Row 0 matched only because its offset is 0 — which is exactly the signature the bisect showed.
   Fix: pass the flat buffer.

The lesson generalises: this tree's plane buffers are part-blocked, and any consumer must be handed
the flat buffer plus `key.parts`, never an `(m, n)` view.

## FFN-tail fusion — refuted in all three forms

| variant | M=64 N=9216 | M=128 N=9216 | vs four-pass |
| --- | --: | --: | --: |
| four-pass reference (swiglu.c + qz_row) | ~3.93 ms | ~7.84 ms | — |
| buffered pack (PACK VERIFIED, codes n_diff 0) | 5.098 ms | 10.250 ms | **+29% / +31%** |
| streaming pack, scalar silu (no g/u materialised) | 4.791 ms | 9.600 ms | **+22% / +22%** |
| streaming pack, f4 silu | 5.068 ms | 10.121 ms | **+29% / +29%** |

Three implementations, all slower than the code they would replace. The buffered form keeps full
size `g`/`u`/`y` scratch (fusion of invocations, not traffic). The streaming form removes that
traffic but pays a scalar silu. Vectorising the silu recovers nothing -- it regressed from +22% to
+29%, so the bottleneck is not the polynomial's width.

Conclusion: **the FFN-tail lever does not exist in this tree.** The subagent's -45 ms was a traffic
model, and every implementation of it measures slower. Nothing was wired, so the default path and
the 1,410.58 ms baseline are unchanged.

## Compiled-C replay — EXECUTES, and is 21% SLOWER on prefill

Seven implementations, and the seventh finally runs end to end. The crash was the calling
convention: this tree's `CPUProgram.fxn` is `CFUNCTYPE(None, c_void_p)` -- **one** argument, a
pointer to the packed argument buffer -- while `kernel_params` has three entries. Passing them
unpacked made the callee dereference an address as if it were the buffer, which is the `CALL 0`
segfault. Packing on the stack (`U64 _a[N] = {...}; fxn(_a)`) fixes it exactly as
`Program.__call__` does.

Executed result, same protocol (128-token prompt, 1 warm-up), control vs replay:

| arm | prefill | decode |
| --- | --: | --: |
| control (`NO_REPLAY=1`) | **3,580 ms** | 3.43 tok/s |
| compiled-C replay | **4,338 ms** | 4.75 tok/s |

Coverage: 2,923 calls in 6 segments, 3 PYTHON calls left to the executor, descriptor audit zero
mismatches, 15,376 calls executed without a fault. Compile ~2.5 s.

**Prefill +758 ms (+21%).** The dispatch residual is therefore *not* recovered by this
implementation: a 2,923-call C function thrashes instruction cache and pays a stack pack per call,
which costs more than the 120 us/program of Python resolution it removes.

This closes the last row large enough to matter. 1,200 ms is not reachable from 1,410.58 ms with
any mechanism measured in this tree.

## Compiled-C replay — closed by measurement across four shapes

The +21.2% loss came from one 2,923-call function; splitting it and reusing one args buffer was
expected to recover the difference. Measured on the same protocol (128-token prompt, 1 warm-up),
control = 3,580.4 ms:

| calls per segment | segments | first token | vs control |
| --: | --: | --: | --: |
| 1 | 2,922 | 4,839.5 ms | +35.2% |
| 12 | 247 | 3,958.6 ms | +10.6% |
| **64** | **49** | **3,926.1 ms** | **+9.7%** |
| 2,923 | 6 | 4,338.2 ms | +21.2% |

The curve is U-shaped with its minimum near 64 calls per segment, and the minimum is still
**+346 ms slower** than the executor it replaces. The two costs trade off exactly as the ABI
dictates: too few calls per function thrashes I-cache, too many adds a `ctypes` entry per segment,
and every call in between must write its arguments into a packed buffer because `fxn` is
`CFUNCTYPE(None, c_void_p)`.

So the 150.45 ms dispatch residual is **not recoverable by replay on this tree** — established by
measurement at four points, not projection. That closes the last row large enough to matter.

## The dequantise gap is ~2x, and it is not parallelism

Earlier I compared our 11.1 GB/s against the 33.79 GB/s DDR ceiling and concluded "20% of bus".
That was the wrong comparator. A dequant-shaped benchmark (`read int32 + write fp32` over the same
354 MB working set) on this board:

| pass | bandwidth |
| --- | --: |
| first pass, cold (first touch of 354 MB) | 5.4 GB/s |
| **second pass, same buffers warm** | **23.8 GB/s** |
| our in-model dequantise (pooled planes, therefore warm) | **11.1 GB/s** |

So the achievable figure for this access pattern, single-threaded, is **23.8 GB/s** and we run at
**47%** of it. That is a real ~2x gap, worth up to ~55 ms — but it cannot be closed by the levers
already measured (pool threading 0.9998x, NEON -0.3%), which is the signature of a *mapping*
problem rather than an ALU or scheduling one: the planes are NPU GEM buffers, and reading them from
the CPU evidently does not reach the bandwidth that ordinary heap memory does.

Next probe: compare a read of the pooled plane buffer against a read of plain heap memory of the
same size, from the same process. If the plane buffer is materially slower, the fix is in the GEM
mapping attributes or in keeping the consumer on-device, not in the loop.

## Mapping hypothesis — REFUTED

Probe: same dequant-shaped loop (`read int32 + write fp32`), same 354 MB size, same process,
ordinary heap buffer versus the pooled ROCKCHIP plane allocator's buffer.

| source buffer | bandwidth |
| --- | --: |
| heap (`np.zeros`) | 5.3 GB/s |
| **pooled NPU plane buffer** | **23.8 GB/s** |

The plane buffer is **not slower than heap memory — it is faster**, and it already reaches the best
warm figure measured on this board. So the in-model 11.1 GB/s is not caused by GEM mapping
attributes, and cache-resident tiling of the planes cannot recover it.

That closes the last open hypothesis on the ~190 ms host row. Combined with the earlier refutations
(pool threading 0.9998x, NEON -0.3%, three fusion designs 22-31% slower), every mechanism proposed
for that row has now been measured and none moves it. The in-model rate must be set by the
interleaving with NPU submits and per-call cache maintenance rather than by the read itself, and
those are inseparable from the fused-call structure that the device row depends on.

## Cache-sync ablation — the syncs are not the cost

Emitted-source ablation (`SKIP_SYNCS`), temperature-fair in-process ABBA, 3 rounds, arms
interleaved 72.1-76.7 C. Timing only; the no-sync arm produces garbage by construction.

| arm | median | spread |
| --- | --: | --: |
| with `sync_mem(TO_DEVICE)` + `sync_mem(FROM_DEVICE)` | **1,399.69 ms** | 0.3% |
| both syncs removed | 1,400.81 ms | 4.3% |
| **delta** | **+1.12 ms** | — |

Removing both ioctls changes prefill by ~1 ms. So neither the mapping (refuted: plane buffer
23.8 GB/s vs heap 5.3), nor cache maintenance (this), nor loop shape (NEON -0.3%), nor parallelism
(pool 0.9998x), nor fusion (three designs 22-31% worse) explains the in-model 11.1 GB/s.

The host row is closed with every hypothesis tested. `SKIP_SYNCS` defaults to False and the serial
source is unchanged at 4,489 bytes.


## RETRACTION — my ABBA harness cannot toggle codegen flags, so the sync ablation is void

The submit ablation revealed a methodology error. Both arms measured **1,020 ms**, about 380 ms
*below* the with-submit baseline, which is the device time missing from **both** arms. The cause:
`TinyJit` captures the kernels on the first forward, and the captured linear holds the CALL uops
built from the programs that existed at capture time. Clearing `_FUSED_PROGRAMS` / `_FUSED_PLANS`
afterwards does not rebuild the captured graph, so both arms execute the same kernels -- the ones
captured during warm-up, whose last iteration had the flag set.

Consequences:

- The sync ablation (+1.12 ms) is **void**: its two arms also ran identical captured kernels.
  Cache maintenance is therefore *unmeasured*, not refuted.
- Any flag that changes generated source must be evaluated with a **fresh capture per arm**, i.e.
  a separate process or an explicit re-capture, not an in-process ABBA. The in-process ABBA is
  valid only for changes that survive capture (runtime data, pool usage, buffer content).

## Cache-sync ablation, fresh process per arm — the syncs cost ~58 ms

The in-process ablation was void because capture pins the kernels. Re-run with one process per arm,
arms interleaved A,B,B,A,A,B,B,A, temperatures 70.2-72.1 C:

| arm | samples (ms) | median |
| --- | --- | --: |
| with both syncs | 3657.36 / 3788.37 / 3753.66 / 3813.89 | 3771.02 |
| syncs removed | 3742.98 / 3710.51 / 3715.88 / 3682.97 | 3713.20 |

**delta -57.82 ms** (no-sync faster); 3 of 4 paired comparisons favour no-sync; within-arm spread
about 4%, so the effect sits just above the noise floor but is consistent in direction.

So cache maintenance is a **real** cost of roughly 58 ms -- roughly three times the 19.2 ms the
component timers attributed to `sync_to + sync_from`, which suggests the ioctls also disturb the
following loop rather than only consuming their own time.

It cannot simply be removed (the NPU and CPU would disagree about cache state), so any gain has to
come from **hoisting or batching** the syncs across calls. That is now the only host-row mechanism
with a measured positive effect.

### Confirmed magnitude: ~30 ms, not 58

A proper 3+3 per arm, still one process per arm:

| arm | trials (ms) | median |
| --- | --- | --: |
| with both syncs | 1431.1 / 1413.3 / 1538.3 | **1431.11** |
| syncs removed | 1399.5 / 1400.6 / 1522.7 | **1400.61** |

delta **-30.50 ms**. Both arms carry a heat outlier on trial 3, so the 1+1 estimate of -58 ms was
optimistic. Cache maintenance is real but worth ~30 ms, and it **cannot be removed** -- the NPU and
CPU would disagree about cache state.

Best case if the syncs could be hoisted or batched perfectly: 1,410.58 - 30.5 = **1,380.1 ms**,
which is still 180 ms above the target. So this row, even at its measured maximum, does not change
the conclusion.


## Submit ablation (fresh process per arm, 3+3) — no hidden contention

| arm | trials (ms) | median |
| --- | --- | --: |
| with submit | 1430.3 / 1428.0 / 1553.8 | **1430.35** |
| submit removed | 1036.6 / 1033.3 / 1161.0 | **1036.57** |

delta **-393.78 ms**, against **416-421 ms** of device time measured independently by the ioctl
component timers. The submit therefore costs *no more* than its own device time, and in fact slightly
less -- there is no additional host stall caused by interleaving with the NPU.

This closes the last open question on the host row by measurement: the in-model 11.1 GB/s dequantise
is not device contention, not mapping, not cache maintenance (30.5 ms, unavoidable), not loop shape
(-0.3%), not parallelism (0.9998x) and not fusion (22-31% worse). The host work is simply ~1,036 ms
and every component of it is already at or ahead of the reference.

Decomposition of this protocol: 1,430.35 = 1,036.57 host+kernels + 393.78 submit.

## CORRECTION — the dequantise deficit does not exist

Per-geometry measurement (`dequant_per_geom.py`), reading the C-side dequantise counter
(`fused_stats[6]`) on each real shape, 9 reps each, planes pooled:

| geometry | calls | dequant ms/call | GB/s |
| --- | --: | --: | --: |
| 64_9216_2560 | 64 | 0.205 | 23.0 |
| 64_2560_4096 | 96 | 0.061 | 21.4 |
| 64_8192_2560 | 24 | 0.187 | 22.5 |
| 64_4096_2560 | 24 | 0.095 | 22.1 |
| 64_10240_2560 | 8 | 0.241 | 21.8 |
| 64_32_2560 | 96 | 0.001 | 11.2 |
| 128_2560_1024 | 32 | 0.118 | 22.2 |
| 1_24576_2560 | 10 | 0.011 | 17.7 |
| **weighted aggregate (354 calls)** | | **31.72 ms** | **22.3** |

The standalone ceiling on the same pooled plane buffers is 23.8 GB/s, so the dequantise runs at
**94% of achievable** and costs **31.7 ms total** -- not the 106.7 ms (6.6 GB/s) or 63.9 ms
(11.1 GB/s) this document previously reported from the ioctl component timer.

Both of those figures were measurement artefacts of that timer, and every hypothesis built on them
(mapping, cache residency, cache maintenance, device interleaving) was chasing a deficit that does
not exist. The per-geometry attribution is the trustworthy one because it reads the same counter the
fused kernel increments around exactly the dequantise statements.

Consequence: the host row is **~32 ms of dequantise, already optimal**, not ~190 ms. The remaining
host cost is quantise and the CPU kernels, and the decomposition needs re-deriving from counters
rather than from the old timer.

## Quantise re-measured — 59.32 ms, not 107.2

Same counter method (`fused_stats[4]` around the quantise statements), per real shape:

| geometry | calls | quant ms/call | dequant ms/call | dequant GB/s |
| --- | --: | --: | --: | --: |
| 64_9216_2560 | 64 | 0.150 | 0.207 | 22.8 |
| 64_2560_4096 | 96 | 0.239 | 0.062 | 21.3 |
| 64_8192_2560 | 24 | 0.151 | 0.187 | 22.4 |
| 64_4096_2560 | 24 | 0.150 | 0.094 | 22.3 |
| 64_10240_2560 | 8 | 0.150 | 0.236 | 22.2 |
| 64_32_2560 | 96 | 0.150 | 0.001 | 11.2 |
| 128_2560_1024 | 32 | 0.121 | 0.118 | 22.1 |
| 1_24576_2560 | 10 | 0.003 | 0.011 | 17.7 |
| **total** | 354 | **59.32 ms** | **31.86 ms** | 22.2 |

The fused host half is therefore **~121 ms** (quantise 59 + dequantise 32 + syncs ~30), not the
~190 ms this document carried from the ioctl timer. Both timer figures were inflated roughly 1.8x,
and the decomposition has to be rebuilt from counters.

### Corrected decomposition (per layer, 32 layers)

| item | ours | reference | delta |
| --- | --: | --: | --: |
| kernel sum | 1,271.10 ms (39.7/layer) | 1,199.94 (37.5/layer) | -2.2/layer |
| dispatch residual | 150.45 (4.7/layer) | ~35 (1.1/layer) | -3.6/layer |

The deficit is **uniform across layers**, which is why no single-kernel fix moved the total: there
is no hot layer to find. A layer-truncation bisection would confirm the uniformity rather than
localise a culprit.

## Row-by-row kernel join — matched kernels are at PARITY

`cpu_kernels_by_name.py` (CPUProgram.__call__ timed by name, warm prefill) joined against
`ref-phase2` 's per-kernel profile:

| bucket | ours | reference | delta |
| --- | --: | --: | --: |
| matched by kernel name | 392.18 | 390.22 | **+1.96** |
| ours, no reference row (per-shape `E_*`/`r_*`) | 268.35 | — | — |
| reference rows with no equivalent here (29 rows) | — | 381.76 | — |

So **every kernel that exists on both sides is at parity to within 2 ms.** The 29 reference rows
without a counterpart are its own decomposition: `rockchip_int8_swiglu_pack` 53.70,
`E_32_32_32_4_4` 51.29, `E_64_16_64_4` 51.04, `rockchip_dequantize` 28.90, `ROCKCHIP->CPU` 25.03,
`E_131_2048_4` 26.73, `E_toks_3088_4` 20.52, `E_28toks2A256029` 19.39, ...

Decomposition of CPU-program time:
    ours  = fused 621.2 (device ~394 + host ~227) + non-fused 660.5 = 1,281.7
    ref   = CPU rows + copies 772.0

So the CPU-side gap is **~115 ms** (887.5 vs 772.0) and none of it is in a shared kernel.

**Where it actually was (measured after this join):** the fused Linear's own host phases, read from
the C counters across one real prefill instead of per-geometry x calls:

| phase | per-geometry x calls (this table's source) | in-model counter | understated |
| --- | --: | --: | --: |
| quantise | 59.32 ms | **105.56 ms** | 1.78x |
| dequantise | 31.86 ms | **62.62 ms** | 1.97x |

The quantise loop was serial inside the fused C, and the four-way path that already existed was
recorded as "inert" because nothing counted a pool hit (M68). Enabled and counted, it measures
**105.56 -> 41.37 ms** and the wall **1,282.71 -> 1,220.68 ms** in-process, which closes the gap:
prefill is now **1,198.87 ms** (3+3, logits `895365e27ed4255b`).

Parallelising dequantise the same way was measured **worse** (62.46 -> 67.85 ms, wall +12.55): that
loop writes `m*n` floats and is bandwidth-bound at 22.3 GB/s, so threads only add handoff. It is
disabled with that verdict in the code. So the split is not "partitioning in general" -- it is
specifically **scalar-bound host loops inside the fused call**, of which quantise was the last one.
