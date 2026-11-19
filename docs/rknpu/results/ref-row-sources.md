# Where each row of the 182-vs-152 table lives in the reference tree

Sources are the REA-authenticated dump `docs/rknpu/results/ref-kernels-all/` (root
`9c3beb78e586ea4c`, 79 entries, C/JSON/Python). Line numbers are from that dump.

| row | ours | ref | reference source | mechanism, and why it is faster | verified |
| --- | --: | --: | --- | --- | --- |
| NPU GEMM | 142.89 | 137.11 | `py/rockchip_bucketed_int8.py`, `py/rockchip_grouped_int8.py` | Its decode issues **203 submits** at widths 18432 / 12352 / 24384; ours issues **307** at ≤ 9216. Same bytes, fewer and larger submits. | yes — kernel names in its profile |
| quantise / dequantise | 4.70 | 1.02 | **`py/rockchip_norm_quant.py:38` `_program`, `register_norm`** | It fuses **RMSNorm and INT8 quantise into one kernel**. **Caveat:** the module docstring scopes it to *"prepared 128-row, 896-wide CPU inputs"*, i.e. a prefill shape; whether its decode path uses this fusion is unverified. Our quantise runs inside every fused Linear (355 calls). | partial — fusion exists; decode applicability unverified |
| elementwise | 14.74 | 6.57 | **`py/rockchip_cpu.py:641` `_swiglu_pack`** | One kernel reads both int32 plane sets, dequantises, applies `silu(gate)*up` and writes the next payload — replacing four passes (dequant gate, dequant up, elementwise, quantise). Profile: 32 calls, 1.771 ms. | yes |
| state fills | 9.52 | ~0 | **not found in the dump** | Its profile contains no fill kernel at all. Whether that is persistent state, a fused zero, or a different GEP is **unverified** — this is the one row I cannot source. | **no** |
| GDN scan | 5.23 | 4.16 | `py/rockchip_recurrent.py` + `c/rockchip_recurrent.c` | **Byte-identical source** (`diff` = 0 over 491 lines) and both call the pooled entry. The 1.26x is invocation/thermal, not code. | yes |
| reduce | 2.68 | 0.79 | its graph shape | It emits **153** `r_*` calls per token where we emit **497**; same data, fewer reductions. | yes — call counts |
| attention | 2.26 | 0.45 | **`py/rockchip_cpu.py:286` `rockchip_cpu_attention(..., const __fp16 *cache)`**, driven by `:371 _cpu_attention(..., decode_only)` | fp16 cache reads and a decode-specialised entry. Matching the fp16 read in our kernel was bit-exact and measured **slower** (4.716 tok/s), so the win here is the decode specialisation, not the width. | yes |
| Python / replay gaps | 6.45 (shipped) | ~0 | `py/rockchip_replay.py` | Native replay: one C call per linear, no Python between kernels. Ours has 16 segments and 6.45 ms between them. | yes |

## What the mapping says

The reference is not faster because of one kernel. It is faster because of **three fusions** and a
**coarser submit geometry**:

1. **norm + quantise** (`rockchip_norm_quant.py`) — removes the per-projection quantise we pay.
2. **swiglu pack** (`rockchip_cpu.py:641`) — four FFN-tail passes become one.
3. **decode-specialised attention** (`_cpu_attention(decode_only=True)`) — a different entry for T=1,
   not a faster loop.

Plus `rockchip_bucketed_int8`/`grouped_int8` issuing 203 wide submits instead of our 307, and its
native replay removing the 6.45 ms of Python between segments.

Three of those four are fusions we do not have; the fourth (fp16 cache width) was tested and is not
the mechanism.

## What is worth porting, in table order

| row | port | expected | risk |
| --- | --- | --- | --- |
| q/dq (+3.7) | fuse quantise into the norm (`register_norm` equivalent) | closes the row | our norm is a tinygrad kernel; needs a custom kernel and a scale layout change |
| elementwise (+8.2) | `_swiglu_pack` | closes most of the row | the three FFN-pack forms tried here measured 22-31% slower, so it must be measured, not assumed |
| gaps (+6.45) | native replay | closes the row | we already have a replay; the 6.45 ms is between its segments |
| fills (+9.5) | unknown | unknown | **no source found**; every implementation tried here made it worse |
