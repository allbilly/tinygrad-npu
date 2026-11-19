# Where each row of the 182-vs-152 table lives in the reference tree

Sources are the **measured tree** `~/tg-qwen35-pack-signature-fix` (the one pinned by the receipt,
dirty diff `7f87f221...`), verified by sha256 against the dump. **Four of six files differ from the
dump** -- `rockchip_cpu.py` (`d0952f646368` live vs `8d5026c857d6` dump), `rockchip_int8.py`
(`1eb8aa766a4a` vs `a1c6510c168f`), `rockchip_norm_quant.py` (`dc1b46365a0c` vs `a49c60c46a0e`),
`rockchip_replay.py` (`29a5a701c29b` vs `e3d6b0f12b8a`); only `rockchip_greedy.py` and
`rockchip_recurrent.py` match. Line numbers below are from the **live** files, and one conclusion
changed because of it (q/dq).

| row | ours | ref | reference source | mechanism, and why it is faster | verified |
| --- | --: | --: | --- | --- | --- |
| NPU GEMM | 142.89 | 137.11 | `rockchip_int8.py:148-150 _matmul` (`KernelInfo(f'rockchip_int8_{m}_{n}_{k}')`) | Its decode issues **203 submits** at widths 18432 / 12352 / 24384; ours issues **307** at ≤ 9216. Same bytes, fewer and larger submits. | yes — kernel names in its profile |
| quantise / dequantise | 4.70 | 1.02 | **`rockchip_norm_quant.py:23-30 register_norm`** | RMSNorm **and** quantise fused in one kernel, and in the measured tree the gate is **generalised**: `(ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE)` with `depth%4==0 and 4<=depth<=8192` and `rows>0`, so it **does** apply at decode. Our quantise runs inside every fused Linear (355 calls). | **yes — and applicable at m=1** |
| elementwise | 14.74 | 6.57 | **`rockchip_cpu.py:677` `_swiglu_pack`** (dump said 641) | **Corrected:** it takes `const float *src` (fp32), not int32 planes. It fuses `silu(gate)*up` + cast to `__fp16` + **the transpose into the next GEMM's packed layout** (`out[start*depth + (c/8*m+row-start)*8 + c%8]`), 4-way pthreads above 32768 elements. Profile: 32 calls, 1.771 ms. | yes — but see the layout note below |
| state fills | 9.52 | ~0 | **not found in the dump** | Its decode profile has 1,086 CPU calls/token and **none is a fill**. The one candidate mechanism in the dump is the replay's borrow mode: `py/rockchip_replay.py:158` `self.borrow_batch = bool(getenv('ONPU_REPLAY_BORROW_BATCH',0))` with `ONPU_REPLAY_NATIVE_FROM` — write into the caller's buffers instead of owning and zeroing fresh ones. Consistent but **unproven**: no dumped file shows state initialised once. | **no** |
| GDN scan | 5.23 | 4.16 | `py/rockchip_recurrent.py` + `c/rockchip_recurrent.c` | **Byte-identical source** (`diff` = 0 over 491 lines) and both call the pooled entry. The 1.26x is invocation/thermal, not code. | yes |
| reduce | 2.68 | 0.79 | its graph shape | It emits **153** `r_*` calls per token where we emit **497**; same data, fewer reductions. | yes — call counts |
| attention | 2.26 | 0.45 | **`rockchip_cpu.py:70` `_decode_heads(...)`**, fp16 cache at **`:93 const __fp16 *keys=cache+kv*{context*dim}`** | fp16 cache reads and a decode-specialised entry. Matching the fp16 read in our kernel was bit-exact and measured **slower** (4.716 tok/s), so the win here is the decode specialisation, not the width. | yes |
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

## The norm+quantise fusion DOES apply at decode (corrected)

An earlier version of this file said the fusion was gated to 128x896 and therefore unusable at m=1.
That was read from the **dump**, whose `rockchip_norm_quant.py` (`a49c60c46a0e`) is not the measured
file. The live measured tree (`dc1b46365a0c`) generalises it:

```python
if ((ONPU_NORM_QUANT or ONPU_NORM_QUANT_DECODE) and ... and isinstance((depth:=source.shape[-1]),int)
    and depth%4==0 and 4<=depth<=8192 and all(isinstance(n,int) for n in source.shape)
    and (rows:=math.prod(source.shape[:-1]))>0 ...
```

with a dedicated decode flag. So the q/dq row (**+3.7 ms**) is a **live lever at m=1**, not an
inapplicable one -- it is the largest closable row left and the next thing to port.

## Correction and consequence: the pack is a layout contract, not just a fusion

`_swiglu_pack` consumes **fp32** gate/up and emits a **packed `__fp16` payload** in the transposed
layout that `rockchip_bucketed_int8` expects. Its speed comes from fusing the elementwise with the
**layout conversion**, which only exists because the reference's GEMM reads packed fp16 payloads.

Our fused Linear does the opposite: it takes a dense fp32 activation and quantises internally. So:

- there is no packed-payload consumer on our side for the pack to feed;
- a pack writing that layout would need a matching GEMM input format, i.e. the bucketed-INT8
  contract, not just one kernel;
- this is why the three FFN-pack forms tried here measured **22-31% slower** — each paid for a
  layout conversion whose consumer did not exist.

Porting `_swiglu_pack` in isolation therefore cannot close the elementwise row. It is only viable
together with `rockchip_bucketed_int8`'s packed-input GEMM.

## The orchestration row: its decode loop is generated C

The row I had been calling "replay gaps (6.45 ms)" is not a tuning difference. The reference does not
run a Python loop over tokens at all:

```c
// rockchip_gpt2_decode.py:11  DecodeLoop -> generated source
int gpt2_decode_loop(int *out,int token,int position,int count,U64 *stats,U64 *done) {
  for(int i=0;i<count;i++) {
    int error=((int(*)(long long,long long,U64*)){replay}ULL)(position+i,token,current);
    for(int j=0;j<{stat_count};j++) stats[j]+=current[j];
    if(error) return error;
    token=*((volatile int*){address}ULL);     // sampled token read from the output buffer
    out[i]=token; (*done)++;
  }
  return 0;
}
```

`replay` is `RockchipCapturedJit` (`rockchip_replay.py:956`), the native replay of the captured
graph. So **one C call drives up to 32 tokens** (`self.tokens` accumulates per call;
`rockchip_gpt2_decode.py:45`): no Python between tokens, no Python between kernels, and the sampled
token is read from a fixed buffer address rather than returned to the interpreter.

**Verified against the report**, not inferred from the code:

```json
// benchmark.json for the measured 4B run
scalar_decode_loop: {"batch": 4, "tokens": 249}
native_execution.decode: runs 249, borrow_outputs_enabled True, borrowed_outputs 65
native_execution.prefill: runs 1, borrow_outputs_enabled True, borrowed_outputs 233
```

`borrowed_outputs` is `len(native.output_copies)` -- **65 output copies identified** in decode, which
the borrow path then handles in place (`rockchip_replay.py:608 if self.borrow_outputs:`).

**65 is not a discriminating number.** At least two state models give it: `2x24 GDN + 2x8 KV + 1`
and `24 recurrent + 24 conv + 16 KV + 1`. Because `2x24 == 24+24`, the arithmetic cannot tell them
apart, so it is not evidence that the borrows are the state buffers.

What our side actually says (allocation counter, decode capture):
`{('full',24), ('zeros',24)}` at 524,288 elements -- i.e. **24 state initialisations, each emitting
two fill kernels** (24 x 2 = 48 fills). `conv_state` is `(1,3,channels)` and cannot be one of them.
So the honest reading is: our 48 fills are 24 recurrent-state initialisations doubled by codegen,
and the reference's 65 output copies are of unknown composition. The borrow hypothesis stays a
hypothesis.

That single fact explains two rows at once:

| row | reference | ours |
| --- | --- | --- |
| Python / replay gaps | ~0 -- the loop is C | **6.45 ms/token** -- 16 replay segments plus a Python `model()` call per token |
| sampler | 0.085 ms -- the loop reads the token directly from the buffer | 0.41 ms after our greedy fix |

It also explains why its 1,086 CPU calls/token contain no fill: the captured program's buffers are
created once and the loop reuses them, which is the same persistent-state property that our
`model.py:819 initial.where(0, state)` copy-on-write defeats.

## The fills: emitter found, and it is load-bearing (row closed)

The emitter is one line, found by counting `Tensor.where` results by size during the decode capture
(`{('where', 524288): 24}`):

```python
# tinygrad/llm/model.py:819  (GatedDeltaNetBlock.__call__)
state = initial.where(0, self.recurrent_state.float())
```

At decode the condition (`pos == 0`) is statically false, so `where` allocates a **fresh 2 MB
buffer** and copies the state into it — 24 blocks, twice per token counting the f16 path, 56 fill
kernels in the replay.

Replacing it with a Python-level select when the position is known:

| | fills | calls | parity |
| --- | --: | --: | --- |
| baseline | **56** | 956 | `895365e27ed4255b` |
| select at Python level | **8** | 908 | **`c693475255158a69`** |

So the fills disappear — and the result changes. **The copy is semantically load-bearing**: the scan
stores into the buffer `where` produced, so `where` is the copy-on-write step that leaves the
persistent state untouched. Removing it makes the scan write the persistent state in place and the
arithmetic diverges.

**Row closed:** the fills are not waste to be optimised away; they are the price of the state
ownership model. The reference does not pay them because its state is persistent and borrowed
(`ONPU_REPLAY_BORROW_*`), which is the same architectural change the other rows pointed at. Removing
them here requires that change, not a code edit at this line.

## The fills are not emitted at the allocation site

The last directed hypothesis was that the 48 x 2 MB fills come from the buffer allocation, so a
borrow could be installed there. Reading that path eliminates it:

```python
# tinygrad/uop/ops.py:820
def new_buffer(device, size, dtype, num=None):
    buf = MultiBuffer(...) if isinstance(device, tuple) else Buffer(device, size * dtype.itemsize)
    return UOp._buffer(size, dtype, slot, device=device, buffer=buf)

# tinygrad/device.py:157
def allocate(self, opaque=None, external_ptr=None) -> Buffer:
```

`Buffer(...)` allocates; there is **no initialising store and no poison** anywhere in this tree
(`grep -r poison tinygrad/` -> nothing). So nothing at the allocation site can emit a zero-fill, and
a borrow installed there would be a no-op by construction -- the experiment was not run.

That leaves the fills as **graph nodes**: `Tensor.zeros(1,32,128,128)` fires 24 times during capture,
so its store is captured and replayed. Cutting that producer chain was attempted three ways
(realise-at-creation 193.49 ms, realise-once-after-first-forward 193.57, eager-init-before-capture
254.02) and every one left 48 fills. The row stays open with its mechanism unidentified, and it is
the only row where I have no verified lever.

## Decision: the packed-payload contract cannot pay at decode

The contract's only lever on our side is the quantise the pack would pre-empt. Measured per call at
the decode geometry (`m=1, n=9216, k=2560`), from the fused C's own counters and the per-kernel dump:

| phase | per call | 64 calls/token |
| --- | --: | --: |
| whole fused call | 789 us | 50.5 ms |
| — of which device submit | 763 us | 48.9 ms |
| — of which CPU (quantise + dequantise + syncs) | 25 us | 1.6 ms |
| — of which **quantise alone** | **2 us** | **0.13 ms** |

So a packed input that removes quantise entirely is worth **0.13 ms/token**; even granting it the
whole CPU share (quantise *and* dequantise *and* syncs, which it cannot) it is **1.6 ms/token**.
Adding the entire device gap against the reference (+5.8 ms, already at per-byte parity) gives a
best-conceivable total of **~7.4 ms against a 30.1 ms gap**.

At m=1 there is simply nothing for the pack to save: the activation is one row, so quantising it is
2 us. The reference's pack pays because at **m=128** a quantise pass is material and its GEMM
consumes the packed layout directly — a prefill optimisation.

**Decision: do not port the contract for decode.** Building the GEMM entry first, as planned, would
have measured a difference bounded at 0.13 ms. This is the table doing its job one level down: the
elementwise row (+8.2) looked like the pack's target, but the pack's *mechanism* is layout+quantise,
and our quantise is already 2 us.

## What is worth porting, in table order

| row | port | expected | risk |
| --- | --- | --- | --- |
| q/dq (+3.7) | fuse quantise into the norm (`register_norm` equivalent) | closes the row | our norm is a tinygrad kernel; needs a custom kernel and a scale layout change |
| elementwise (+8.2) | `_swiglu_pack` **plus** a packed-input GEMM | unknown without the GEMM half | it is a layout contract: the pack emits fp16 in the next layer's transposed layout, which our fp32-input fused Linear does not consume |
| gaps (+6.45) | native replay | closes the row | we already have a replay; the 6.45 ms is between its segments |
| fills (+9.5) | unknown | unknown | **no source found**; every implementation tried here made it worse |
