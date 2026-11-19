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


## What superseded everything below this point

The sections that used to follow were a chronological lab log in which a lever's verdict and its
retraction both survived, so a reader could cite whichever they reached first. They are in git
history. Their conclusions, replaced:

| claim in those sections | measured reality |
| --- | --- |
| "4-way quantise — implemented, gated, measured NEUTRAL (1.0006x)" | **−60.22 ms**: quantise 105.56 -> 41.37 ms. The dispatch had been silently falling back and nothing counted it (M68) |
| "quantise re-measured — 59.32 ms, not 107.2" | **105.56 ms in-model**: 59.32 came from per-geometry x calls (M70) |
| "dequantise 31.86 ms" | **62.62 ms in-model**; parallelising it made it worse (67.85) |
| "the dispatch residual of 150.45 ms is not recoverable by replay" | **replay recovered it**: −153.80 ms |
| "1,200 ms is not reachable from 1,410.58 ms with any mechanism measured in this tree" | reached: **1,198.87 ms** |
| "the CPU-side gap is how work is partitioned into per-shape kernels" | it was **one serial loop** inside the fused call |

Result and protocol: `prefill-parallel-quantise.md`. Levers: `../refuted-levers.md`.
