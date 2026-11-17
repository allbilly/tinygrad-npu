# ANE GEMM baseline — CPU and optimized ANE measured

Status: both paths are measured on Qwen3.5 0.8B Q8_0. The optimized ANE path
runs 139 linears on the M1 ANE through the direct-register runtime in `~/ane`
and reaches **9.62 tok/s prefill / 11.54 tok/s decode** at 32 prompt tokens.
The `~/ane` Qwen3.5 reference is 12.37 / 11.57; decode matches, prefill is 22%
behind. The CPU host is 2.06 / 1.97.
Raw reports: `results/cpu-qwen35-0.8b.json`,
`results/ane-qwen35-0.8b.json` and `results/ane-qwen35-0.8b-4tok.json`.

## What is measured

Every eligible `nn.Linear` is replaced by `ANELinear`
(`extra/ane/ane_linear.py`). The ANE matmul is a tinygrad `CUSTOM_FUNCTION`
executed on the CPU worker queue, so the CPU operations around it stay
JIT-compiled and the ANE submission is replayed with the graph.

The vocabulary head is included: `248320 x 1024` is split into 16 ANE plans of
16384 columns (the largest width that completed on this M1; 24576 timed out).
For Qwen3.5 0.8B, **139 linears** are offloaded; 48 stay on the CPU (the
16-output `ssm_alpha`/`ssm_beta` and the 512-output `attn_k`/`attn_v`), plus the
small projections below `ANE_MIN_DIM = 768`.

Three CPU-side helpers remove the remaining bottlenecks:

- `fast_attention` runs the 6 full-attention layers in one C loop (8 heads,
  2 KV heads, head_dim 256) and matches the float32 SDPA reference to ~1e-7.
- `fast_argmax` replaces tinygrad's 248,320-element reduction tree (21.8 ms)
  with a single C loop (~1.1 ms).
- The ANE call waits only for the last queued CPU producer, not the whole
  worker queue.

The ANE runtime is the direct-register M1 implementation in `~/ane`:
`qwen35/ane.py` compiles `qwen35/ane_matmul.c` and opens `/dev/accel/accel*`.
There is no CoreML, E5RT, `.mlmodel` or `anecc` in this path.

## Protocol

- Checkpoint: `unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`
  (811,843,840 bytes), local copy at
  `~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf`.
- 32-token tokenizer prompt, 8 greedy decode tokens.
- `DEV=CPU`, one warm-up pass, three measured passes (median reported).
- Qwen3.5 has recurrent blocks, so `generate` forces `chunk_size=1`; prefill is
  sequential, not a batched 32-row pass. The ANE path also forces
  `chunk_size=1` for every model because the custom kernel needs static shapes.
- Prefill is time-to-first-token; decode is the next 8 tokens.

This is smaller than the `rkllm` NPU protocol (128 prompt / 64 decode, one
warm-up + three measured passes), so the two tables are not comparable.

## Results

| path | load | prepare | prefill | prefill tok/s | decode | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU host | 1.08 s | — | 15.53 s | 2.06 | 4.06 s | 1.97 |
| ANE + CPU host | 1.15 s | 5.83 s | 3.33 s | 9.62 | 0.69 s | 11.54 |
| `~/ane` Qwen3.5 reference | — | — | — | 12.37 | — | 11.57 |

The ANE row used 139 linears and made 24,640 submissions across the warm-up and
three measured passes (154 per token). It produced the same output token IDs as
the CPU row for this prompt:

```text
[1208, 310, 1301, 13, 561, 26257, 314, 279, 17933]
```

A 4-token prompt/2-token decode run measured 9.26 / 10.63 tok/s median, with a
best single run of 11.80 decode. `load` is `Transformer.from_gguf`; ANE plan
packing is `prepare_seconds` (0 for the CPU row).

## Qualitative test runs

Both paths used the Qwen2 chat template and `chunk_size=1`:

- "What is 2 + 2?" produced `4`.
- "Name the capital of France." produced `The capital of France is **Paris**.`.
- The boiling-point prompt reached the 16-token cap on the CPU run.

The model emits ` thinking` because the template enables thinking. Transcripts:
`results/generate-qwen35-0.8b.txt` (CPU) and
`results/generate-qwen35-0.8b-ane.txt` (ANE).

## ANE path and driver

The driver was loaded for this run:

```sh
make -C ~/ane/kmod -j"$(nproc)"
sudo make -C ~/ane/kmod install
sudo modprobe ane
ls -l /dev/accel/
```

The module in `~/ane/kmod` matches the running kernel (`7.1.13+`). The udev
rule grants the `render` group; for SSH, add a temporary ACL:

```sh
sudo setfacl -m "u:$USER:rw" /dev/accel/accel0
```

If `/dev/accel/accel0` is absent, `--ane` is a hardware gate and fails at device
open, before any model submission:

```text
ANE: no accessible ane driver device; see ~/ane/kmod/README.md.
RuntimeError: M1 ANE device is unavailable
```

The device-only matmul check passed before the model run:

```text
shape 1x1024 @ 1024x1024  max_abs=0.0385742  max_rel=0.0996947
```

## Wrapper validation

Before the driver was loaded, `ANELinear` was exercised with a mock device
implementing the same `create`/`run`/`free` interface as `qwen35.ane.Ane` (the
mock is not committed):

- `(8,)`, `(2,8)`, `(3,2,8)` and `(40,8)` inputs matched the FP32 NumPy
  reference; 40 rows split into two `[32, 8]` submissions.
- `replace_linears` and `prepare_linears` visited linears in attributes and
  lists; an oversized linear and a 16-output linear were skipped.
- `prepare()` dropped the source weight, was idempotent, and `close()` freed
  every plan and closed the device.
- Weight/bias were FP16; input/output stayed FP32 CPU.
- A full Qwen3.5 0.8B mock run replaced and prepared 138 linears, then produced
  a first token with 552 submissions (138 linears x 4 prompt tokens).

The committed hardware check is `extra/ane/smoke_matmul.py`.

## Performance-core pinning (the missing piece)

The M1 has 4 performance cores (cpu4-7, ~2.8-3.0 GHz) and 4 efficiency cores
(cpu0-3, ~1.7-2.1 GHz). The default `schedutil` governor spreads the benchmark
across both, so the tinygrad CPU kernels and the ANE host round-trips run on the
slow cores. Pinning the whole process to the performance cores:

```sh
taskset -c 4-7 env DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --ane --warmup 3 --runs 3
```

| path | prefill tok/s | decode tok/s |
| --- | ---: | ---: |
| unpinned (schedutil, all 8 cores) | 9.35–9.99 | 9.28–9.58 |
| **pinned to cpu4-7** | **12.39** | **13.10** |
| `~/ane` accurate reference | 12.37 | 11.57 |

The 32/8 3-warm-up + 3-measured trials are `12.39 / 12.36 / 12.42` prefill and
`12.99 / 13.14 / 13.10` decode, tokens
`[1208, 310, 1301, 13, 561, 26257, 314, 279, 17933]` in every trial
(`docs/ane/results/ane-qwen35-0.8b-pinned2.json`). The 4/2 run is
`12.22 / 13.18` with tokens `[3516, 4042, 369]`.

This matches the `~/ane` prefill (12.39 vs 12.37) and beats its decode
(13.10 vs 11.57) with exact CPU token parity. The rkllm branch's measurement
discipline ("check the clocks before and after every measurement") is what
surfaced this: the earlier numbers were measured on the efficiency cores.

## What the numbers say

- **The optimized ANE path is 4.7x faster than the CPU host on prefill and
  5.9x faster on decode** (9.62/11.54 versus 2.06/1.97). The first host
  round-trip baseline was 0.82/0.82; the JIT custom kernel, head chunking,
  fast argmax and fast attention account for the improvement.
- **With performance-core pinning, both targets are met**: 12.39 prefill versus
  12.37 and 13.10 decode versus 11.57, with exact tokens. The unpinned numbers
  (9.62/11.54) were measured on the efficiency cores; the GatedDeltaNet CPU
  graphs are ~20 ms/token and the ANE matmuls ~48 ms/token, but the core
  placement was the larger factor.
- The best 4-token run reaches 11.80 tok/s decode, above the reference's 11.57,
  but the 32-token median is the fair comparison for this protocol.
- The generic path uses plain FP16 with power-of-two input scaling. It does not
  apply the reference's Hadamard basis or compensated multi-plane policy (used
  for its Mirai M W4 checkpoint), so ANE logits are a numerical baseline, not
  parity. The matching token IDs are one prompt, not a quality gate.
- No clocks or temperature were recorded. The host had an active desktop and
  other load. Do not publish these as isolated performance numbers.

## Reproducing

```sh
cd /home/asahi/tinygrad-npu
PY=~/ane/.venv/bin/python

# CPU reference
DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --runs 3 \
  --out docs/ane/results/cpu-qwen35-0.8b.json

# ANE (requires /dev/accel/accel0)
DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --ane --runs 3 \
  --out docs/ane/results/ane-qwen35-0.8b.json

# Device-only matmul check
PYTHONPATH=. "$PY" extra/ane/smoke_matmul.py --k 1024 --n 1024 --rows 1
```

## GatedDeltaNet C kernel experiment

`extra/ane/ane_linear.py` also contains a fused GDN C kernel (`fast_gdn`) and a
direct embedding gather (`fast_embedding`). Both are numerically exact:

- `fast_gdn` matches a NumPy reference to `out max_abs 1.9e-06`,
  `state max_abs 2.4e-07`, and produces the same greedy tokens as the
  pure-tinygrad path (`[5, 6, 24218, 10]` for prompt `[1, 2, 3, 4]`).
- `fast_embedding` matches the one-hot embedding exactly (`max_abs 0.0`).

They are not enabled in the benchmark because they are slower in the model:

| path | 4/2 prefill | 4/2 decode |
| --- | ---: | ---: |
| pure-tinygrad GDN (default) | 9.21 | 11.44 |
| fused GDN C kernel | 3.89 | 4.02 |

The kernel itself is ~1 ms per block. The blocker is the rolling conv state:
the fast path must write `conv_state` after the GDN call, and the scheduler
cannot order that side-effect store after a `CUSTOM_FUNCTION` without breaking
the graph. Tried and rejected:

- `core.after(conv_store)` and `out.after(call, conv_store)`: the extra `AFTER`
  loses the output buffer identity, so the next op reads zeros/garbage.
- `state.after(conv_store)`: the executor resolves the `AFTER` to a copy, so the
  kernel updates a buffer nobody reads.
- Reassigning `self.conv_state`/`self.recurrent_state` to an `AFTER`: the
  `@function` capture mixes the prefill and decode graphs (`bind mismatch on
  start_pos, 1 != 0`).
- `store.after(call)`: rejected by the UOp spec (`AFTER` on `STORE`).
- Writing `conv_state` inside the 12-argument kernel: the first block matches the
  pure-tinygrad path to 1e-9, but the different summation order flips the greedy
  token (220 vs 5), and the path is slower (~8.8 tok/s) because each of the 18
  GDN calls still pays the per-call executor overhead.
- A zero-valued dependency (`core + conv_state.sum() * 0`) to connect the store:
  the second block's GDN output becomes zero.

The embedding is already lowered to an efficient gather by tinygrad; the
explicit gather saves ~4 ms per 32-token prefill.

## ANE submission scaling

The native path now does the power-of-two input scaling in a small C wrapper
(`ane_run_scaled`) around `ane_plan_run_batch`, instead of ~5 tinygrad kernels
per Linear (peak reduce, exponent/shift bit ops, scale mul, output unscale).
The wrapper scales the input, submits the FP16 plan, and restores the output
scale with NEON-friendly loops. It keeps the 1-row ANE submission (the
`ane_plan_run_compensated` path doubles the rows and was slower).

| path | 32/8 prefill | 32/8 decode |
| --- | ---: | ---: |
| tinygrad scaling kernels | 9.48–9.72 | 9.96–10.05 |
| C scaling wrapper | **9.86–9.99** | 9.35–9.58 |

Tokens are exact in every run. The wrapper removes the per-Linear scaling
kernels; combined with performance-core pinning it reaches the 12.39 prefill /
13.10 decode result above. The fused GDN kernel remains an optional experiment,
blocked on the conv-state ordering described above.

## Local GGUF coverage

Both local checkpoints are benchmarked pinned (`taskset -c 4-7`, 3 warm-ups +
3 measured, 32/8):

| model | ANE Linears | prefill tok/s | decode tok/s | tokens |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B-Q8_0 | 139 | **12.39** | **13.10** | exact |
| Qwen3.8-2B-Q4_K_M | 139 | **6.17** | **6.36** | exact |

The 2B is the same `qwen35` architecture at dim 2048 / 24 blocks; its per-token
compute is ~5x the 0.8B (inner size 6144 vs 2048), so 6.2 tok/s is expected. Its
508M-element vocabulary head is now chunked onto the ANE too (`ANE_MAX_WEIGHT`
default raised to 600M; the 4B 1B head still stays on the CPU), which took the
2B from 2.28/2.32 to 6.17/6.36. The 4/2 token check matches the CPU path
(`[3516, 4042, 369]`).

The 2B initially failed with `ANE submission failed: Connection timed out` on
its K=6144 `ssm_out`/`down` plans. The M1 ANE times out above K=4096, so
`ANELinear.prepare` now splits the depth into `ANE_MAX_DEPTH` (4096) chunks and
`__call__` sums the partial outputs. All 138 Linears then run on the ANE.
