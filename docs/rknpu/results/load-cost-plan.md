# Load-cost plan — INT8 weight quantization in `NPULinear.prepare()`

Status: **queued, not started.** Nothing in this document has been implemented.
All numbers below are measured on this board with `DEV=CPU`; the NPU was never
opened. Projections are labelled as projections.

Owner: unassigned. Pick this up when the machine is idle (no NPU measurement in
flight) — the microbenchmarks here are CPU-only and can run any time, but the
end-to-end gate at the bottom needs the NPU.

---

## 0. The premise this plan was written against, and its correction

The task said "FP16 packed weights 26.7 s, INT8 packed weights 131.9 s, 5x
slower". **That comparison is not real.** Recorded evidence:

| artifact | `npu_weight_bytes` | load | which path |
| --- | ---: | ---: | --- |
| `docs/rknpu/results/qwen35-4b/report.json` | absent | 131.94 s | **FP16** |
| `docs/rknpu/results/submits-4b/report.json` | 4,204,789,760 (4010 MiB) | 70.90 s | **INT8** |
| `docs/rknpu/results/quick-4b/report.json` | absent | 62.6 s | FP16, quick protocol |

Why `qwen35-4b` is FP16 and not INT8:

- `git show HEAD:extra/rockchip/npu_linear.py` contains **zero** occurrences of
  `int8`, `quantize_rows` or `pack_weight_int8`. The entire INT8 weight path is
  an **uncommitted working-tree diff**; no commit ever added it
  (`git log --all --oneline -- extra/rockchip/npu_linear.py` returns only
  `933d2580f` and `1fc3f4b3e`, both FP16-only).
- `git show HEAD:extra/rockchip/bench_llm.py` has no `npu_weight_bytes` field, so
  a report without that field was produced by the HEAD script, i.e. the FP16
  path. `qwen35-4b` has no such field; `submits-4b` does.
- `docs/rknpu/npu-gemm-baseline.md:175` describes the same run as "the 4B
  completed a full FP16 run".

So the INT8 load is **70.9 s**, not 131.9 s, and FP16 is the *slower* of the two
(8.4 GB packed vs 3.9 GB, and it sits at the 12.4 GB RAM wall — the 131.94 s
figure is memory-pressure-bound, not compute-bound). No "5x INT8 regression"
existed.

The real problem is smaller but still worth fixing: **62.3 s of the 70.9 s INT8
load is `prepare()`, and 86% of that is the `quantize_rows` op chain.**

---

## 1. Measured stage split

Harness: `_Domains.place` monkeypatched to return `"CPU"`, so the final
`.to(device)` is a no-op and `/dev/dri/card1` is never opened. Everything else is
the production `NPULinear.prepare()` (`extra/rockchip/npu_linear.py:203-222`).

### 1.1 Whole-model total

| quantity | value | reps |
| --- | ---: | --- |
| Linears packed | 249 | 1 run |
| weight tiles | 323 | 1 run |
| elements quantized | 4,204.8 M | — |
| packed bytes | 3.92 GiB | — |
| `from_gguf` | 7.5–8.6 s | 2 runs |
| **`prepare()` INT8** | **62.27 s** | 1 run |
| `prepare()` FP16 (reference) | 50.21 s | 1 run |

### 1.2 Stage split, whole model

Measured by timing each stage inside the per-tile loop (1 run). The "dequant"
column here is the tile slice + `.contiguous()` on already-fp16 data; the real
GGUF dequant is measured separately in 1.3.

| stage | seconds | share | kernels per tile |
| --- | ---: | ---: | ---: |
| GGUF dequant + slice `.contiguous()` | 2.08 | 3% | 1 |
| `quantize_rows` chain | 58.38 | 86% | 2 |
| CNA transpose `.reshape().permute().contiguous()` | 7.29 | 11% | 1 |
| sum of stages | 67.76 | 100% | 4 |

The 67.76 s sum is ~9% above the 62.27 s real `prepare()` because the itemising
harness materialises a full fp16 copy per Linear. Use the **shares**, not the
absolute seconds: applied to 62.27 s → quantize ≈ 53.7 s, transpose ≈ 6.7 s,
dequant ≈ 1.9 s.

### 1.3 Per-shape, real GGUF data

`min` of 2 reps per stage. Input is the actual Q8_0 tensor dequantized to fp16.

| tensor | shape | M elems | dequant | `quantize_rows` | transpose | INT8 tile | FP16 tile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate` | 9216x2560 | 23.6 | 0.022 | 0.258 | 0.012 | 0.308 | 0.034 |
| `ffn_down` (k tile) | 2560x4096 | 10.5 | 0.016 | 0.079 | 0.009 | 0.115 | 0.023 |
| `attn_qkv` | 8192x2560 | 21.0 | 0.019 | 0.148 | 0.011 | 0.189 | 0.029 |
| `attn_gate` | 4096x2560 | 10.5 | 0.014 | 0.082 | 0.008 | 0.106 | 0.019 |
| `attn_k` | 1024x2560 | 2.6 | 0.009 | 0.029 | 0.006 | 0.045 | 0.011 |
| `ssm_beta` | 32x2560 | 0.1 | 0.007 | 0.022 | 0.005 | 0.021 | 0.009 |

Synthetic fp16, `min` of 2 reps, for the two shapes that only appear as full
tiles:

| shape | M elems | dequant proxy | `quantize_rows` | transpose | INT8 pack | FP16 pack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 24576x2560 | 62.9 | 0.023 | 0.667 | 0.023 | 0.863 | 0.032 |
| 24576x4096 | 100.7 | 0.043 | 1.067 | 0.041 | 1.278 | 0.050 |

### 1.4 Tile count (answers "is tiling costing anything?")

| columns | depth | Linears | tiles | why >1 tile |
| ---: | ---: | ---: | ---: | --- |
| 9216 | 2560 | 64 | 64 | 1 tile (`ffn_gate`/`ffn_up`) |
| 2560 | 9216 | 32 | 96 | K split 4096+4096+1024 (`ffn_down`) |
| 248320 | 2560 | 1 | 11 | column split, 10x24576 + 2048 (`output`) |
| 8192 | 2560 | 32 | 32 | 1 tile |
| 2560 | 4096 | 32 | 32 | 1 tile |
| 4096 | 2560 | 24 | 24 | 1 tile |
| 32 | 2560 | 48 | 48 | 1 tile |
| 1024 | 2560 | 16 | 16 | 1 tile |
| **total** | | **249** | **323** | |

Tiling costs almost nothing in bytes — only 74 extra tiles out of 323, all from
`ffn_down` (K) and `output` (columns). But each tile pays a **fixed ~10 ms of
tinygrad Python lowering per `realize()`** (measured: the 48 `32x2560` tiles cost
1.43 s for 3.9 M elements, i.e. 30 ms/tile against ~1 ms of actual kernel work).
323 tiles x 3 realizes x ~10 ms ≈ **10 s of the 62 s is Python, not arithmetic.**
A fused kernel removes 2 of the 3 realizes per tile, so part of the win below is
scheduling, not bandwidth.

`_int8_padded_depth` is a **no-op for every 4B tile**: 2560, 4096 and 1024 are
all 512 multiples, so `padded == k` and the zero-fill branch is never exercised
by this model. It must still be implemented — other checkpoints hit it.

---

## 2. The fix: one fused `custom_kernel` per tile

### 2.1 What to build

Replace the `quantize_rows` call in `pack_weight_int8`
(`extra/rockchip/npu_linear.py:83-89`) with a single hand-written kernel that
reads the fp16 tile and writes int8 codes plus fp32 per-row scales, in the
`_QUANTIZE_SOURCE` / `_quantize_program` style already in the file
(`extra/rockchip/npu_linear.py:92-142`).

The existing `_QUANTIZE_SOURCE` takes `const float *input` because it serves the
*activation* path (`int8_payload`, `npu_linear.py:145-154`, fed `x.float()`).
The weight path has fp16 input, so it needs its own source string (or a
`__fp16`/`float` template) — do not reuse `part()` unchanged.

Sketch (measured as variant `C_neon` below; `depth`/`padded`/`header` baked into
the source, one non-static function, entry at offset 0):

```c
#include <arm_neon.h>
static void wq(unsigned char *raw, const __fp16 *in, int rows, int depth, int padded, int header) {
  float *scales = (float*)raw;
  signed char *out = (signed char*)(raw + header);
  for (int r = 0; r < rows; r++) {
    const __fp16 *x = in + (long)r*depth;
    float32x4_t m0 = vdupq_n_f32(0.0f), m1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= depth; i += 8) {
      m0 = vmaxq_f32(m0, vabsq_f32(vcvt_f32_f16(vld1_f16(x+i))));
      m1 = vmaxq_f32(m1, vabsq_f32(vcvt_f32_f16(vld1_f16(x+i+4))));
    }
    float m = vmaxvq_f32(vmaxq_f32(m0, m1));
    for (; i < depth; i++) m = __builtin_fmaxf(m, __builtin_fabsf((float)x[i]));
    float s = m * (1.0f/127.0f); if (s < 1e-8f) s = 1e-8f;
    scales[r] = s;
    float inv = 127.0f / m;
    float32x4_t vi = vdupq_n_f32(inv);
    signed char *o = out + (long)r*depth;
    i = 0;
    for (; i + 8 <= depth; i += 8) {
      int32x4_t a = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvt_f32_f16(vld1_f16(x+i)), vi)));
      int32x4_t b = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvt_f32_f16(vld1_f16(x+i+4)), vi)));
      a = vmaxq_s32(vminq_s32(a, vdupq_n_s32(127)), vdupq_n_s32(-127));
      b = vmaxq_s32(vminq_s32(b, vdupq_n_s32(127)), vdupq_n_s32(-127));
      vst1_s8(o+i, vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))));
    }
    for (; i < depth; i++) { float q = __builtin_nearbyintf((float)x[i]*inv);
      if (q > 127.0f) q = 127.0f; if (q < -127.0f) q = -127.0f; o[i] = (signed char)q; }
    for (int i = depth; i < padded; i++) o[i] = 0;
  }
}
```

Entry point, per the packed-args ABI (`docs/rknpu/npu-gemm-baseline.md:99-111`):

```c
void rockchip_wq(unsigned char *args) {
  wq(*(unsigned char**)args, *((const __fp16**)args + 1), ROWS, DEPTH, PADDED, HEADER);
}
```

Call it exactly like `int8_payload` does: `Tensor.empty(header + rows*padded,
device="CPU", dtype=dtypes.uint8)` then `.custom_kernel(x, fxn=lambda o, a:
_program(o, a, ...))`, with `header = round_up(rows*4, 64)` so the scales stay
64-aligned. Keep the existing `.reshape(n//32,32,k//32,32).permute(0,2,1,3)
.contiguous()` transpose as a **separate** kernel (see 2.4).

### 2.2 Measured kernel variants

`min` of 3 reps, synthetic fp16 input, same output layout (row-major int8 +
fp32 scales). All three are drop-in for `quantize_rows`; they differ only in how
the inner loops are written.

| variant | 24576x2560 (62.9 M) | 9216x2560 (23.6 M) | 2560x4096 (10.5 M) | M elem/s |
| --- | ---: | ---: | ---: | ---: |
| `A_branchy` — what `_QUANTIZE_SOURCE` does today | 0.170 | 0.066 | 0.032 | 371 |
| `B_fmax` — `__builtin_fmaxf`/`__builtin_fabsf`, no intrinsics | 0.111 | 0.044 | 0.023 | 565 |
| `C_neon` — explicit NEON, 8 lanes | **0.072** | **0.029** | **0.015** | **877** |
| current `quantize_rows` chain | 0.664 | 0.260 | 0.078 | 95 |

So `C_neon` is **9.2x** the current chain on the dominant tile shape, and
`B_fmax` is 6.0x if intrinsics are unwanted (worth having as the portable
fallback: it is the same source with `fmaxf`→`__builtin_fmaxf`).

For reference, single-threaded `numpy` memcpy on this board is **11.45 GB/s**
(22.89 GB/s read+write). `C_neon` moves 3 bytes/element at 2.63 GB/s, i.e. it is
ALU-bound, not bandwidth-bound — the row-max pass plus the fp16→fp32 convert,
round and narrow cost more than the loads. Do not expect memcpy speed here.

### 2.3 Projected saving (projection, not a measurement)

| stage | now | projected | basis |
| --- | ---: | ---: | --- |
| `quantize_rows` | 53.7 s | 4.8 s | 4.2 G elem at the measured 877 M elem/s |
| transpose | 6.7 s | 6.7 s | unchanged |
| dequant | 1.9 s | 1.9 s | unchanged |
| **`prepare()`** | **62.3 s** | **~13 s** | sum of the above |
| **INT8 load** | **70.9 s** | **~22 s** | + 8.6 s `from_gguf` |

Saving ≈ **49 s of the 62.3 s prepare**. A further ~7 s is available by folding
the transpose into the same kernel (writing the CNA layout directly), which would
take `prepare()` to ~6 s and the load to ~15 s — but see 2.4: the naive version
of that is *slower*, so it needs a NEON rewrite before it is worth doing.

These are projections from per-shape rates at 2–3 reps; the end-to-end number
must be re-measured with `bench_llm.py --quick` before anyone quotes it.

### 2.4 Ranked options

1. **Fused fp16→int8 kernel, transpose kept separate** (recommended).
   Smallest change that gets the win: one new source string, one new
   `_program`-style helper, one call site (`npu_linear.py:88`). Measured 9.2x on
   the quantize stage. Projected load 70.9 → ~22 s.
2. **Fused kernel that writes the CNA layout directly.** Removes one kernel and
   one `realize()` per tile. Measured *slower* when written naively: 0.237 s vs
   0.195 s for `A` + separate transpose on 62.9 M, because the 32x32 blocked
   store is scalar byte stores. Only worth it with a NEON 32x32 block transpose
   inside the kernel; would take `prepare()` to ~6 s.
3. **Quantize from the raw Q8_0 blocks, skipping the fp16 dequant entirely.**
   Fastest in principle: `x = d*q`, so `rowmax = max_b(d_b*max|q_b|)` and
   `code = round(q*d*127/rowmax)` need no fp16 intermediate (3.1 bytes/element
   instead of 7). Measured prototype: 0.0xx s — **not measured**, this variant
   was written but the run was stopped before it reported. Rejected for now
   because it is Q8_0-specific and needs the raw GGUF entry threaded into
   `NPULinear`, which means touching `tinygrad/llm/model.py`. Revisit only if the
   load must go below ~10 s.
4. **Cheaper pure-tinygrad chain** (drop `round()` for
   `(y + y.sign()*0.5).trunc()`, avoid the fp32 upcast). Smallest diff of all,
   but the chain is ALU-bound on the divide and the `where`, so expect ~1.4x, not
   9x. Not worth doing on its own.

---

## 3. Traps already paid for — do not rediscover

1. **`round()` is not a primitive.** `Tensor.round` is
   `tinygrad/mixin/elementwise.py:893` and decomposes to
   `trunc`/`ceil`/`floor`/`where`. The generated kernel contains **four
   `__builtin_truncf` plus one true `fdiv` per element**. `abs()` likewise lowers
   to three ops (`(x<0)?-1:1`, `(x==0)?0:…`, multiply), and
   `.maximum(-127).minimum(127)` lowers to min/negate/min/negate (four ops).
   That is why the chain runs at 95 M elem/s while a memcpy runs at 11 GB/s.
2. **The transpose is faster as a separate kernel.** `A` + transpose = 0.173 +
   0.022 = 0.195 s; the fused CNA-writing kernel = 0.237 s on the same 62.9 M.
   tinygrad's `.permute().contiguous()` is well vectorised; a hand-written
   32-byte-strided store is not. Do not "optimise" the transpose away without a
   NEON implementation and a measurement.
3. **`_int8_padded_depth` is a no-op for every Qwen3.5 4B tile** (2560, 4096,
   1024 are all 512 multiples). The zero-fill path is untested by this model —
   write it, and test it on a shape that needs it (natural depths 576/768/1152/
   2304, or a ragged K tile).
4. **`fmaxf`/`fabsf` are undeclared under `-ffreestanding -nostdlib`**
   (`tinygrad/runtime/support/compiler_cpu.py:23` compiles with
   `-ffreestanding -nostdlib`). C99 implicit declaration then produces a call to
   nothing and **silently wrong results** (measured: 10.1 M of 10.5 M codes
   wrong, max diff 127). Use `__builtin_fmaxf` / `__builtin_fabsf`.
5. **CPU/CLANG ABI is one packed `void *args`.** `ClangRenderer.packed_args =
   True`; the entry must take `void *args` and unpack buffer addresses itself,
   and all scalar geometry must be baked into the source. A natural C signature
   receives the 16-byte args buffer as its first pointer and returns without
   writing anything, without crashing. See
   `docs/rknpu/npu-gemm-baseline.md:99-111`.
6. **The entry point must be at offset 0.** The runtime calls the loaded image at
   offset 0, so exactly one function may be non-`static`; helpers stay `static`
   and get inlined at `-O2`. A surviving helper shifts the entry and the kernel
   silently runs the wrong code.
7. **`HCQ_RUNTIME_DEV=CPU:CLANG` is required for any `DEV=CL` run.** The default
   `CPU:LLVM` host renderer fails to compile CL kernels:
   `'%reg_0' defined with type '[2 x i64]*' but expected 'i64*'`. Even `x+1` on
   `CL` fails. The baseline is `DEV=CPU` anyway
   (`docs/rknpu/README.md:20-24`), where this does not bite.
8. **Keep the per-tile `.realize()`.** `npu_linear.py:220` says why: a lazy graph
   holds every tile's temporaries at once. The 4B already sits at the RAM wall
   (12.4 GB peak FP16); do not batch tiles to save scheduling overhead.
9. **Parity checks: `Tensor.empty(..., dtype=uint8)` gives a uint8 numpy view.**
   `codes.numpy()` is uint8, so `-5` reads as `251`. Use `.view(np.int8)` or the
   comparison is nonsense (this cost one debugging cycle).
10. **`x/s` and `x*(1/s)` are not identical.** The chain divides; a kernel that
    precomputes `inv` differs by up to 1 LSB on a small, unrecorded fraction of
    codes. See the gate below.

---

## 4. Correctness gate (must pass before the fused kernel replaces anything)

**Gate A — per-shape exactness, CPU only.**

For every distinct `(columns, depth)` group in the model, and for every tile
including the ragged ones (`ffn_down` K tiles 4096/4096/1024 and `output`'s last
column tile 2048x2560), on the real GGUF data:

1. Compute the reference `codes, scales = quantize_rows(tile_fp16)` with the
   current code.
2. Compute the fused `codes2, scales2`.
3. Require `np.array_equal(scales, scales2)` (bit-exact fp32) and
   `np.array_equal(codes.view(np.int8), codes2.view(np.int8))` — **exact**, zero
   mismatches.
4. Print `max|diff|` and the mismatch count per shape even when it passes, so a
   later regression is visible.

To make step 3 achievable the kernel must use a **true division** `q1(x[i]/s)`,
not `x[i]*inv`; `__builtin_nearbyintf` is round-half-to-even, matching
`Tensor.round` (`tinygrad/mixin/elementwise.py:893`), so exact parity is
reachable. If exact parity turns out to cost too much, the fallback is
`x[i]*inv` with a documented tolerance — but then the gate becomes "max|diff| <=
1 and the mismatch count is recorded", and that must be an explicit decision in
the commit message, not a silent pass. The measured `inv` variants above had
`max|diff| = 1` and **zero** elements differing by more than 1; the count of
1-LSB differences was not recorded.

**Gate B — no-op on the packed bytes.** Pack a full Linear both ways and compare
the packed int8 buffer with `np.array_equal`. This catches a layout mistake that
Gate A (which compares pre-transpose codes) cannot see.

**Gate C — end-to-end token parity, needs the NPU (run when the board is free).**
`bench_llm.py --quick` on Qwen3.5 4B twice, chain vs fused, same prompt; the
generated token ids must be identical. Then one full 128/64 run to confirm
`load_seconds` moved and decode did not regress.

---

## 5. Where the code changes

| file | change |
| --- | --- |
| `extra/rockchip/npu_linear.py` | add `_WEIGHT_QUANTIZE_SOURCE` + `_weight_quantize_source(rows, depth, padded, header)` + `_weight_quantize_program(...)`; change `pack_weight_int8` (`:83-89`) to call it instead of `quantize_rows` |
| `extra/rockchip/npu_linear.py` | keep `quantize_rows` (`:77-80`) — it is the reference for Gate A and still used by nothing else in the weight path |
| `docs/rknpu/npu-gemm-baseline.md` | record the new `load_seconds` and the stage split once Gate C passes |

No change to `tinygrad/llm/model.py`, no change to the matmul, no change to the
activation path (`int8_payload`).

---

## 6. Reproducing the measurements

Scripts live in `/tmp/opencode/load/` (scratch, **not** durable — re-create from
this document if they are gone):

| script | what it measures |
| --- | --- |
| `bench_model.py` | whole-model `prepare()` with `place()` stubbed to `"CPU"` |
| `bench_model_fp16.py` | same, FP16 (`int8=False`) |
| `bench_real.py` | per-shape stages on real GGUF tiles |
| `bench_itemise.py` | per-stage split over the whole model + kernel counts |
| `bench_fused2.py` | the three fused kernel variants + parity |
| `bench_stages.py` | chain vs fused vs fp16 pack on synthetic shapes |

All of them set `DEV=CPU` and stub the upload, so none of them open
`/dev/dri/card1`. Run one at a time; each holds all cores for 1–2 minutes and
heats the board.
