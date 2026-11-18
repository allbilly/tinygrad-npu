# Decode port list — the reference's decode-only machinery, mapped onto our tree

Scope: read-only reverse engineering of `~/tg-qwen35-pack-signature-fix/tinygrad` (the tree the
6.52 tok/s row came from) and `~/oNPU/onpu/` (its recipe). Nothing was executed; no NPU work.

## 0. The budget the list is ordered against

| quantity                          | ours                       | reference                    | source |
| --------------------------------- | -------------------------: | ---------------------------: | --- |
| decode rate                       | 2.74 / 2.80 / 2.56 tok/s   | **6.52** tok/s               | `results/attnpool-3plus3.log:31,38,45`; `reference-qwen35-4b.md:203` |
| ms per token                      | 357–391 (median 365)       | **153.4**                    | derived from the two rates |
| NPU device time, `submit_seconds` | 144.8–146.7 ms             | **138.66 ms**                | `results/attnpool-3plus3.log`; `benchmark.md:1392` |
| host-visible remainder            | **211–244 ms**             | **14.7 ms**                  | difference |
| NPU submits / token               | **307**                    | **203**                      | `results/attnpool-3plus3.log`; `reference-qwen35-4b.md:207` |
| NPU tasks / token                 | 825                        | not reported                 | same |
| all programs / token              | **1,031**                  | 1,289 (1,086 CPU + 203 NPU)  | `results/native-prof.log` `graph_rows`; `reference-qwen35-4b.md:188` |

Our own probe splits the 1,031-program decode token three ways (`results/native-prof.log`,
`graph_rows` event, 323-submit vintage):

| component                          | ms/token | note |
| ---------------------------------- | -------: | --- |
| `kernel_s`                         |      220 | wall inside CPU/NPU programs (0.22–0.25 s) |
| — of which `ioctl` (323 calls)     |  180–200 | the blocking submit, incl. wrapper |
| — pure CPU kernel arithmetic       |      ~40 | 220 − 180 |
| `bookkeeping_s`                    |  **110** | `loop_s − kernel_s`: everything not inside a kernel (0.11–0.19 s) |
| `loop_s`                           |      330 | 0.33–0.44 s; ≈ 1 / 2.99 tok/s |

The device is already at parity (145 vs 138.66). **The whole gap is host.** Two facts decide the
whole ordering below:

1. The reference executes **more programs than we do** (1,289 vs 1,031) in **roughly 1/15th the host time**,
   so the difference is cost per program, not program count.
2. Our `bookkeeping_s` is 110 ms/token — 106 µs per program. That is the Python/UOp executor, and
   it is what the reference's replay layer deletes.

---

## 1. Inventory of decode-only mechanisms

Status: **(a)** already in our tree · **(b)** in our tree but not enabled · **(c)** absent.

| # | mechanism | reference file:line | what it does | ours |
| --: | --- | --- | --- | --- |
| 1 | `ScalarDecoder` | `llm/kernels/rockchip_decode.py:17`, `:40-67` | Wraps a `TinyJit` step; calls the C loop when the capture qualifies, else one Python step per token. | **(c)** |
| 2 | `install_text_generation` | `rockchip_decode.py:70-119` | Keeps the first sampled token, then replays batches of `min(width, budget)` tokens (stream 4, non-stream 32), publishing the completed prefix before yielding (`:106`). | **(c)** |
| 3 | `DecodeLoop` — generated C token loop | `rockchip_gpt2_decode.py:17-67`, loop at `:31-39` | Compiles `for(i=0;i<count;i++) replay(position+i,token,current); token=*out;` — up to 32 tokens inside **one** Python call, no graph per token, stop tokens handled in C (`:37`). | **(c)** |
| 4 | Native replay (`RockchipCapturedJit`) | `rockchip_replay.py:956`, `:911-955`; wired at `rockchip.py:855-875` | Transcribes the captured linear into one C function that calls each CPU/attention kernel by address and issues each NPU `ioctl` itself. | **(b)** file ported, never installed for decode (see §3.1) |
| 5 | `ONPU_REPLAY_SCALAR_COPY` | `rockchip_replay.py:265-273` | Snapshots a 1-element local token buffer into a replay-owned buffer instead of compiling a copy graph. | **(b)** present in the ported file, not reached |
| 6 | Decode attention: grouped GQA | `rockchip_cpu.py:70-105` (`_decode_heads`), emitted at `:307-308` for `max_rows==1` | One pass per **KV head** handling `heads//kv_heads`=4 query heads, sharing the key/value loads. | **(c)** ours is per-query-head (`attn.c:48-70`) |
| 7 | Decode attention: head-slice pool split | `rockchip_cpu.py:109-180`, dispatch `:256-266`, switch `ONPU_CPU_HEAD_SLICE` `:20` | Splits the 4 heads of one KV group across pool workers; bit-exact because heads are independent. | **(c)** |
| 8 | `ONPU_CPU_DECODE_POOL` | `rockchip_cpu.py:14`; used `rockchip_cpu.py:427-428` | Routes decode attention through `pool.address`. **Off in the 4 B recipe** (`rockchip_options.py:395`: `"ONPU_CPU_DECODE_POOL": "0"`). | **(c)** — and not needed, see §4.1 |
| 9 | `ONPU_GDN_DIRECT_DECODE` | `rockchip_recurrent.py:15`, `:108-110`; C: `rockchip_recurrent.c:27-57` (`direct_row`), guard `:75-79`, branch `:83-87` | Single-row recurrence in one pass over the state, no 256 KB stack transpose. Requires `rows==1 && capacity==1 && start>0 && !history && alpha_dim==1`, all disjoint from `state`/`out`. | **(b)** the C is byte-identical (`diff -q` clean vs `extra/rockchip/recurrent.c`), the `#define` is never emitted |
| 10 | `ONPU_GDN_DECODE_SPIN` | `rockchip_recurrent.py:14`, `:108-109` | Selects the low-latency pool entry (`pool.address` vs `pool.prefill_address`) when `capacity==1`. Needs an active pool. | **(c)** no pool in our decode |
| 11 | `ONPU_GDN_DECODE_PREP_POOL` | `rockchip_prepare_pool.py:17`, `:125-171`; gate `rockchip.py:863` | Rewrites the 5 decode preparation programs (conv/SiLU/norm) into pooled 4-way bodies. | **(b)** file ported, no caller — `rewrite_preparation_programs` is only invoked from the reference's `in_phase` |
| 12 | `decode_preparation_programs: 5` | `onpu/rockchip_options.py:388`; asserted at `onpu/worker.py:871` | The recipe declares decode must own ≥5 preparation programs; the harness fails the run otherwise. | **(c)** we have no such declaration |
| 13 | `ONPU_NORM_QUANT_DECODE` | `rockchip_cpu.py:30`; `rockchip_norm_quant.py:97-141`; gate `:148-162` | Fuses one-token RMSNorm + INT8 quantise into one kernel from the exact source text of both. **Not in the 4 B recipe** — absent from the whole `QWEN35_FAST_CHECKPOINTS` env block (`rockchip_options.py:389-401`). | **(c)** |
| 14 | Dedicated decode norm (`exact=rows<16`) | `rockchip_cpu.py:580`; `register_norm` call `:582`; route `rockchip.py:318-322` | One `rockchip_cpu_rmsnorm` call per norm instead of tinygrad's reduce+scale pair. **Not reachable for 4 B**: `decode_width_ok` is `width==4096 or ONPU_CPU_RMS_ALL_WIDTHS`, and the recipe sets neither. | **(c)** — and the reference does not use it at 4 B either |
| 15 | `ONPU_HOST_REALIZE_NORM_WEIGHTS` | `rockchip_cpu.py:36`; `rockchip.py:305-310` | Materialises the deferred norm weight once so its cast does not re-run per token (0.155 ms/token, `rockchip_cpu.py:34-35`). Only bites with `--defer-weight-realization` / `--file-backed-cpu-weights`, which the reference passes and we do not. | **(c)** |
| 16 | `ONPU_INT8_REUSE_ACTIVATION` | `llm/kernels/rockchip_int8.py:494-504` | At `count==1`, quantises once and reuses the payload across all column tiles. | **(c)** |
| 17 | Dual decode weight packing | `rockchip_int8.py:369-370`, `_phase_tiles` `:429-432`, phase select `:478-479` | Separate weight layout for decode (`decode_tile_columns`). `--decode-tile-columns` defaults to `None` (`benchmark_inference.py:55`), so **the 6.52 row does not use it**. | **(c)** — do not port |
| 18 | `ONPU_INT8_DECODE_NARROW_CBUF` | `runtime/support/rockchip_int8.py:22`, `:145-146` | Forces `CBUF_CON0=0xb1` for `m==1,k==4096` narrow widths so a wider CBUF keeps the weight resident. **Device time, not host.** | **(a)** for the widths that occur: `ops_rockchip.py:139` already sets `0xb1` at `n==512 or n>=1024`, and the narrow set is 128–960 |
| 19 | `_sigmoid_source` (scalar) | `rockchip_cpu.py:56-67`, selected `:724` | Builds the sigmoid from tinygrad's own `to_program` legalisation so a non-tiled (i.e. decode) SwiGLU pack stays bit-identical. | **(c)** — but we have no fused SwiGLU pack at all (`swiglu_native.py:1-10` is a standalone elementwise pass) |
| 20 | Prepared-QKV decode reuse | `rockchip.py:23-26`, `:305-354`; installed `:889-922` | One host kernel for INT8 dequant + rotary + KV store, reused by the one-token path. | **(c)** ours emits dequant, rotary and the cache store as separate programs |
| 21 | `--scalar-decode-loop` harness flag | `patches/tg-decode-all-20261110-session2/extra/rockchip/benchmark_inference.py:68`, gate `:172`, `:287`, `:352` | The switch that installs #1–#3. It is on in the 6.52 run (`--scalar-decode-loop`, `docs/rknpu/results/jitcast/run.sh:50`). | **(c)** |

---

## 2. Reference decode-only mechanisms that are **off** in the 6.52 row

Worth stating explicitly, because they are the ones most likely to be ported by mistake. The
QWEN35 env block is `onpu/rockchip_options.py:389-401` and the verbatim command is
`docs/rknpu/results/jitcast/run.sh:27-51`.

| mechanism | why it is not in the row |
| --- | --- |
| `ONPU_CPU_DECODE_POOL` | explicitly `"0"` (`rockchip_options.py:395`) |
| `ONPU_NORM_QUANT_DECODE` | key absent from the block; its only recorded use is the MiniCPM3 soak (`oNPU/scripts/run_soak_coldstart.py:123`) |
| `ONPU_CPU_RMS_ALL_WIDTHS` / `_ALL_ROWS` | absent, so the 4 B decode norm runs tinygrad's generic decomposition (`rockchip.py:318-322`) |
| `decode_tile_columns` | `--decode-tile-columns` not passed (`benchmark_inference.py:55`) |
| `ONPU_DECODE_HOST_WFE` | absent (Gemma2 sets it, `rockchip_options.py:377`); 4 B uses `ONPU_HOST_POOL_WFE=1` only |
| `ONPU_HOST_REALIZE_NORM_WEIGHTS` | absent, so the reference pays that 0.155 ms/token itself |

So the 4 B decode advantage is **four** mechanisms: native replay (#4), the scalar decode loop
(#1–#3), GDN direct decode (#9), and the fused/prepared projections that get the submit count to
203. Everything else in this document is a lever the reference measured elsewhere or left off.

---

## 3. Port list, ordered by expected gain per unit of work

All effects are **estimates** unless marked measured. The order is for *replay landing first*,
because that is the item that changes the value of everything else.

### 3.1 Native replay for the decode graph — the whole gap

| | |
| --- | --- |
| status | **(b)** file ported (`tinygrad/llm/kernels/rockchip_replay.py`, 1,040 lines), never installed for decode |
| replaces in our 307-submit path | the executor loop over all 1,031 programs: `TinyJit.__call__` → `CapturedJit` → per-program buffer lookup, `ensure_allocated`, kernel dispatch, `dev.lock`, `submit_count` bookkeeping. Collapses it to one C function call. |
| where it goes | `tinygrad/llm/model.py:915` (`__call__` dispatch to `rollout_jit`) via a hook like `extra/rockchip/try_native_replay.py:36-62`, plus removal of the three blockers below |
| estimate | **−140 to −150 ms/token → ~4.6–4.8 tok/s** |
| reasoning | `bookkeeping_s` 110 ms disappears entirely; the `ioctl 180–200` minus `submit_seconds` 145 leaves ~40 ms of wrapper/lock that the replay also removes. 365 − 150 ≈ 215 ms. Cross-check: the archive measured native replay alone at **5.96 → 16.36 tok/s decode** (2.75×) on Qwen3-0.6B, same submits, same IDs (`docs/rknpu/results/ref-tinygrad-extract.md:731`) |
| verify | function-boundary bitwise: same logits/token ids for 64 tokens, **plus** a full 3+3 for the rate |

Three named blockers, all observed, none structural:

1. `UnsupportedReplay('no NPU work')` — `rockchip_replay.py:402` requires at least one NPU
   `Ops.CALL` or an `RNPA` attention binary. Our NPU work is a **CPU** program that issues the
   ioctl itself (`extra/rockchip/npu_linear.py:493-506`, `_FusedInt8Plan.__call__` `:552`), so the
   gate rejects a graph that is entirely replayable. Observed: `results/port-replay-try5.log`
   (`"reason": "no NPU work"`, `native_runs: 0`). This is the blocking one.
2. `ModuleNotFoundError: tinygrad.runtime.support.rockchip_attention` —
   `rockchip_replay.py:711` imports a device-side attention program we do not have. Observed:
   `results/port-replay-try7.log`. Remove the branch; our attention is a CPU program.
3. **No decoder loop is installed**, so even a working replay pays one Python call per token.
   Covered by 3.2.

### 3.2 `DecodeLoop` + `ScalarDecoder` — amortise the per-token Python

| | |
| --- | --- |
| status | **(c)** |
| replaces | `extra/rockchip/bench_llm.py:144-146`: the `for step in range(gen)` loop, the per-token `model(...)`, `.realize()` and `int(out.item())` sync. 32 tokens per call instead of 1. |
| where it goes | new `tinygrad/llm/kernels/rockchip_decode.py` (port of `rockchip_decode.py:17-119`) + `rockchip_gpt2_decode.py:17-67`; driver replaces `extra/rockchip/bench_llm.py:144-146` and `tinygrad/llm/model.py:1176` |
| estimate | **−8 to −25 ms/token**; becomes the largest remaining host item once 3.1 lands |
| reasoning | per token we still pay `TinyJit.__call__`, `UOp.variable(...).bind(...)` (`tinygrad/llm/model.py:914`), output allocation, and a device→host sync for `.item()`. `.item()` alone serialises the blocking ioctl path. The reference pays these once per 4–32 tokens (`rockchip_decode.py:87,101`) |
| verify | full 3+3 only — it changes Python-side dispatch, so no kernel boundary exists to compare |

Depends on 3.1: `DecodeLoop.__init__` asserts `self.native is not None`
(`rockchip_gpt2_decode.py:24-25`).

### 3.3 `ONPU_GDN_DIRECT_DECODE` — cheapest real win in the list

| | |
| --- | --- |
| status | **(b)** — the C file is byte-identical, the `#define` is never emitted |
| replaces | 24 `cpu_gated_delta_scan` calls/token. Our entry (`extra/rockchip/scan_native.py:61-65`) only ever calls the general path, which transposes the whole state through `float columns[256*256]` (`recurrent.c:80,88-89`) before doing 128×128 MACs. At `capacity==1` that transpose is pure overhead. |
| where it goes | `extra/rockchip/scan_native.py:43-46` — prepend `#define ONPU_GDN_DIRECT_DECODE 1` to the body; the existing baked `start=1` (`:64`) already satisfies `start>0` |
| estimate | **−14 to −22 ms/token**, or **−5 ms** on the reference's own measurement |
| reasoning | our per-call cost from the trace is 0.025–0.035 s per 24 calls = **1.0–1.5 ms/call** (`results/native-prof.log` `graph_rows`, 2-decimal rounding). `direct_row` removes the transpose and both re-reads. The reference's ratio is 0.154605 / 0.387674 = 0.40 (`ref-tinygrad-extract.md:729`), so 60 % of the call is removable → 0.6–0.9 ms × 24. The same archive's **end-to-end A/B on this exact model** measured 6.235 → 6.424 tok/s = **4.7 ms/token** (`ref-tinygrad-extract.md:728`). Plan with 5 ms, hope for 20 |
| verify | function-boundary bitwise: `direct_row` is documented "exact key-ordered recurrence" (`recurrent.c:15-16`) so the scan output buffer must match byte-for-byte; then 3+3 for the rate |

Guard check against our geometry: `rows=1`, `capacity=1` (baked `tokens=1`), `alpha_dim=1`,
`history=NULL`, `key_dim=value_dim=128` (`recurrent.c:75-79`) — all satisfied at decode.

### 3.4 Gate/up fusion at `m==1` (submit count 307 → ~283)

| | |
| --- | --- |
| status | **(b)** implemented, deliberately disabled |
| replaces | 48 submits/token for `ffn_gate` + `ffn_up` with 24, by the code's own count |
| where it goes | `extra/rockchip/npu_linear.py:887` — the `if False and all(hasattr(node, "ffn_gate"), ...)` branch |
| estimate | **−4 to −9 ms/token** |
| reasoning | the device still does the same MACs, so what disappears is 24 submits' worth of dispatch and ioctl wrapper — 24 × ~0.2 ms ≈ 5 ms — plus one quantise per fused pair, minus whatever the 24.5 %-wider tile costs on the device. The docstring says the fused width 19456 "still splits into two submits" above the 12288 **prefill** limit, but "at m==1 the limit is 24576 so decode gets one submit instead of two" (`npu_linear.py:855-857`). It was disabled because of a **prefill** regression (+14.7 %, `:837-841`), which is not a decode argument |
| verify | full 3+3 — submit count changes, arithmetic is bit-identical by construction (`npu_linear.py:846-853`), so also check identical ids |

### 3.5 Pre-prepared QKV for the one-token path

| | |
| --- | --- |
| status | **(c)** |
| replaces | per layer, the separate dequantise, query-rotary and k/v cache-store programs. Our fused Linear stops at dequantise (`npu_linear.py:552-585`); rotary and the store are separate `E_*` programs |
| where it goes | `tinygrad/llm/model.py:450-478` (attention block, after `assigned_kv`) and `:656` (GDN block) |
| estimate | **−3 to −6 ms/token** now; **−1 to −2 ms** after 3.1 |
| reasoning | the reference's own note calls it "one host kernel for INT8 dequantize + rotary + KV-cache store ... instead of a separate dequantize, query-rotary and key/value-store kernel per layer" (`rockchip.py:23-26`). 8 attention layers × 2–3 programs = 16–24 programs; at 106 µs/program that is 1.7–2.5 ms of bookkeeping plus the merge of three small memory passes |
| verify | function-boundary bitwise on the q/k/v buffers, then 3+3 |

### 3.6 Fuse the pre-projection RMSNorm into the INT8 Linear

| | |
| --- | --- |
| status | **(c)** — the reference's version of this is `ONPU_NORM_QUANT_DECODE` (#13), which is *off* at 4 B |
| replaces | our decode norm is a reduce + a scale program (`r_640_4` ×65 and `E_640_4` ×97 in `results/native-prof.log` `graph_rows`; 640×4 = 2560 = hidden width, so the pair is inferred, not confirmed) followed by the fused Linear, which **re-quantises the row inside its own C function** (`npu_linear.py:552-568`). The norm tail and the quantise can be one program. |
| where it goes | `extra/rockchip/npu_linear.py:411-492` (`_fused_source`) — add an optional norm+quant prelude with the gamma pointer and eps baked |
| estimate | **−12 to −16 ms/token** before 3.1; **−1 to −2 ms after** |
| reasoning | 65 `r_640_4` + 97 `E_640_4` = 162 programs are width-2560-shaped; if the norm accounts for two per layer plus the final one, ~130 are ours to remove. At 106 µs of bookkeeping each that is ~14 ms; the arithmetic merge itself is small (~1 ms). **This value is almost entirely dispatch, so it evaporates the moment 3.1 lands.** Do not do this before 3.1. |
| verify | function-boundary bitwise on the uint8 payload + per-row scale (the quantise contract is `npu_linear.py:126-155`) |

### 3.7 Fused dequantise → SiLU → fp16 pack

| | |
| --- | --- |
| status | **(c)** |
| replaces | the gate/up dequantise, the separate `silu(gate)*up` pass (`extra/rockchip/swiglu.c`, wired as a standalone elementwise kernel) and the re-quantise inside the down Linear |
| where it goes | new kernel alongside `extra/rockchip/swiglu_native.py`; the reference's shape is `llm/kernels/rockchip_int8.py:588-671` (`_ffn_pack` / `ffn_pack`), with `_sigmoid_source` (`rockchip_cpu.py:56-67`) providing the bit-exact scalar sigmoid |
| estimate | **−3 to −6 ms/token** before 3.1; **−1 to −2 ms after** |
| reasoning | 24 GDN layers × 2 programs removed, and one fewer fp32 round trip of 4608 floats. The reference measures its pack at 0.726 ms/token for 28 calls on MiniCPM3 (`oNPU/progress.md:361`) |
| verify | function-boundary bitwise on the fp16 packed output — requires the scalar `_sigmoid_source` path, exactly as `rockchip_cpu.py:724` selects it for non-tiled (decode) shapes |

### 3.8 `ONPU_INT8_REUSE_ACTIVATION` — likely negative for us

| | |
| --- | --- |
| status | **(c)** |
| replaces | nothing; it *splits* our one fused call into a quantise program plus N raw matmul calls |
| estimate | **−0.3 to +1 ms** (i.e. probably a regression) |
| reasoning | our fused call quantises inside the same C function, so the only repeat is the multi-tile lm head (248320 columns / `TILE=24576` = 11 tiles, `npu_linear.py:25`). Reusing the payload would save 10 quantises of 2560 floats — microseconds — and add 10 program launches. Only worth it *after* replay makes a program cost ~2 µs |
| verify | bitwise on the payload bytes (trivially identical) — a 3+3 would not resolve 0.3 ms |

---

## 4. Traps

### 4.1 Decode attention is not worth porting at all

The reference has two decode-attention mechanisms (#6 grouped GQA, #7 head-slice pool) and uses
neither pool at 4 B. The arithmetic says skip both:

- our kernel gets `capacity = prompt + gen + 16 = 208` (`extra/rockchip/bench_llm.py:93`).
- work per token: 16 heads × 208 keys × 256 dim × 2 (k and v) × 2 flop = 3.4 MFLOP → **~0.34 ms**
  at the 10 GFLOP/s a single A76 reaches here.
- the grouped form saves the 4× re-read of the KV cache: 4 kv heads × 208 × 256 × 2 B × 2 = 0.85 MB,
  so 3.4 MB → 0.85 MB, i.e. **~0.08 ms** at 33.79 GB/s.

Estimated ceiling **0.1–0.2 ms/token**, and `ONPU_CPU_DECODE_POOL=0` in the reference's own row
(`rockchip_options.py:395`). Do not port `_decode_head_slice`.

### 4.2 Anything whose benefit depends on the pool

`ONPU_CPU_DECODE_POOL` (#8), `ONPU_GDN_DECODE_SPIN` (#10) and `ONPU_GDN_DECODE_PREP_POOL` (#11) all
need a live worker pool inside the timed path, and all of that benefit is dispatch, which 3.1
already removes. 3.1's C function cannot use a pool anyway (it calls kernels by address, once).

Two concrete hazards if they are attempted anyway:

- **File-scope data is at address 0 in a jit_loader'd kernel** — a `static` written by one worker
  reads back 0 in the next (`results/threadsplit/README.md`, blocker 1). The reference never relies
  on it: its pool body takes the partition index as an argument
  (`rockchip_norm_quant.py:58` `static void part(void **args,int unused,int worker)`) and bakes the
  dispatcher address in as an integer (`:84-87`). Any port that parks state in a `static` will
  silently run serial — measured 0.98–1.00× on five kernels.
- **Kernels compile in a worker pool unless `PARALLEL=0`** (blocker 2, same file). A transform
  applied at `ClangCompiler.compile_cached` reached **7 of ~30** kernels without it. This bites any
  plan to rewrite *generated* `E_*`/`r_*` programs, e.g. a norm-flattening transform. Our own
  hand-built kernels are safe (they call `compile_cached` directly,
  `extra/rockchip/scan_native.py:98`), because they are single-task compiles.

`ONPU_GDN_DECODE_SPIN` additionally has a correctness argument against it: pooling a **stateful**
scan means a partial completion can never be replayed serially — the reference makes the same point
and fails the worker instead (`rockchip_recurrent.py:26-29`). Our `scan_native.py:59-63` records that
pooling this kernel measured 1.17× on the kernel but 57 % worse end to end.

### 4.3 `ONPU_GDN_DIRECT_DECODE` and the pool interact

Enabling 3.3 makes the scan DRAM-bound rather than transpose-bound (state is 32 heads × 128 × 128 ×
4 B = 2 MB read+written per call ≈ 0.12 ms at the bus limit, matching the reference's 0.155 ms).
After that there is almost nothing left for a pool to win — another reason 4.2 is a trap.

### 4.4 `ONPU_NORM_QUANT_DECODE` is not portable as written

Its gate needs the norm result to be *recoverable* from a `WeakKeyDictionary` of registered norms
(`rockchip_norm_quant.py:18,27-34`) and it refuses to run without the shared host pool's `address`
(`:169-174`, which is why a `rows==1` fused kernel had to be written at all — `:100-103`). Our norm
is a tinygrad decomposition with no registration hook, and our quantise lives inside the fused
Linear. The transferable idea is 3.6, not this file.

### 4.5 `ONPU_INT8_DECODE_NARROW_CBUF` is a device lever, and we already have the part that applies

`runtime/support/rockchip_int8.py:145` extends `banks = 0xb1` to widths 128–960. Our
`ops_rockchip.py:139` already sets `0xb1` for `n==512 or n>=1024`, and 4 B's decode widths are
1024 / 2560 / 4096 / 4608 / 248320 — none in the narrow set. Porting it would change no register in
this model, and it is device time besides, where we are already at parity (145 vs 138.66).

### 4.6 `decode_tile_columns` is not in the reference's row

The dual decode packing is implemented and reachable, but `--decode-tile-columns` defaults to
`None` (`benchmark_inference.py:55`), and the recorded command does not pass it
(`results/jitcast/run.sh:27-51`). Treating it as part of the 6.52 configuration would be a
fabricated requirement, and it doubles packed weight memory (`rockchip_int8.py:326-328`).

### 4.7 `ONPU_HOST_REALIZE_NORM_WEIGHTS` is conditional on a flag we do not set

It is worth 0.155 ms/token (`rockchip_cpu.py:34-35`) **only** under
`--defer-weight-realization` / `--file-backed-cpu-weights`, which the reference passes
(`results/jitcast/run.sh:51`) and our `extra/rockchip/bench_llm.py:48-60` declares no such option. Porting it
alone would be a no-op.

---

## 5. Ordering, at a glance

| rank | port | effort | est. ms/token if it lands first | est. ms/token after 3.1 | verify |
| --: | --- | --- | --: | --: | --- |
| 1 | 3.1 native replay for decode | high — 3 named blockers, 1 is a one-line gate | **140–150** | — | bitwise ids + 3+3 |
| 2 | 3.2 `DecodeLoop` + `ScalarDecoder` | low — 231 lines to port, both files simple | 8–25 | 8–25 | 3+3 only |
| 3 | 3.3 `ONPU_GDN_DIRECT_DECODE` | **very low** — one `#define` line | 14–22 | 14–22 | bitwise scan output + 3+3 |
| 4 | 3.4 gate/up fusion at `m==1` | very low — flip one `if False` | 4–9 | 4–9 | 3+3 + identical ids |
| 5 | 3.5 prepared QKV one-token path | medium — new kernel | 3–6 | 1–2 | bitwise q/k/v + 3+3 |
| 6 | 3.6 norm fused into the Linear | medium — new kernel + gamma/eps plumbing | 12–16 | 1–2 | bitwise payload |
| 7 | 3.7 fused dequant→SiLU→pack | medium-high — new kernel + exact sigmoid | 3–6 | 1–2 | bitwise fp16 output |
| 8 | 3.8 reuse activation | low | −0.3 to +1 | 0 | bitwise payload |

Order by gain per unit of work (effort-weighted, so a one-line change outranks a large one):
**3.3 → 3.4 → 3.1 → 3.2 → 3.5 → 3.6 → 3.7 → 3.8**.

If effort is not the constraint, 3.1 comes first: it is worth more than every other item combined.
3.3 and 3.4 are a few lines each and are worth ~20–30 ms; 3.1 is the large one but has the highest
risk; 3.6 must be deliberately *deferred* past 3.1, or it will be measured against the wrong
baseline and reverted as "no gain" when its value was only dispatch.

Sum of the parts, if all of 3.1–3.7 land, compounding rather than adding: 365 − 150 (replay)
− 10 (decode loop) − 18 (direct scan) − 5 (gate/up) − 4 (prepared QKV) − 2 − 2 ≈ **175 ms/token ≈
5.7 tok/s (estimate)**. The reference is at 153.4, so ~20 ms would remain: per-program cost the
replay does not remove, plus CPU kernel work above theirs. No algorithmic change to the matmul, the
scan or the attention is required for any of this.

The least certain term is 3.3's −18. The trace's "0.03 s / 24 calls" is rounded to two decimals, and
the reference's own end-to-end A/B on this model was −4.7 ms. At −5 instead of −18 the total is
~188 ms ≈ **5.3 tok/s**, still 1.9x our current 2.74.

## 6. What is *not* on this list, despite looking like a decode item

| candidate | why not |
| --- | --- |
| `_decode_heads` grouped GQA | 0.1–0.2 ms ceiling, §4.1 |
| `ONPU_CPU_HEAD_SLICE` | only reachable through the pool; 0.1 ms |
| `ONPU_NORM_QUANT_DECODE` | off in the reference's row; not portable as written, §4.4 |
| `ONPU_INT8_DECODE_NARROW_CBUF` | device lever, no narrow widths in this model, §4.5 |
| `decode_tile_columns` | not in the reference's row, §4.6 |
| `_sigmoid_source` standalone | it is a *support* mechanism for 3.7, worth nothing alone |
| `ONPU_REPLAY_SCALAR_COPY` (#5) | part of 3.1; the archive's own paired measurement found no gain (item 12, `ref-tinygrad-extract.md:733`) |
