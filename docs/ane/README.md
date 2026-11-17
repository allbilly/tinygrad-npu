# Apple ANE work — docs index

This directory records the Apple Neural Engine (ANE) work on this checkout.
Code lives in `extra/ane/`; the runtime is the direct-register M1 ANE
implementation in the reference tree `~/ane` (override with `ANE_REF`). No
CoreML, no E5RT, no `.mlmodel`, no `anecc`.

## What is here

| file | contents |
| --- | --- |
| `model-support.md` | Small-model selection from the `rkllm` matrix, with the ANE reference status and the recommended first run |
| `matmul-baseline.md` | The generic ANE GEMM baseline: implementation, protocol, measured CPU and ANE results, and caveats |

## Baseline definition

The baseline is **the CPU host with every eligible `nn.Linear` offloaded to the
ANE GEMM**. Everything else — attention, norms, softmax, rope, embedding
lookup, sampling, KV cache — stays on the CPU. The vocabulary head is offloaded
too: `248320 x 1024` is split into 16 ANE plans of 16384 columns. Projections
smaller than `ANE_MIN_DIM` (768) stay on the CPU because the ANE command
template is built around that dimension.

`ANELinear` packs each weight once into an ANE plan. The matmul is a tinygrad
`CUSTOM_FUNCTION` executed on the CPU worker queue, so the CPU operations
around it stay JIT-compiled and the ANE submission is replayed with the graph.
Three CPU helpers remove the remaining hot spots: `fast_attention` for the 6
full-attention layers, `fast_argmax` for the 248,320-element vocabulary
reduction, and a producer wait that does not drain the whole worker queue.

The ANE path is the analogue of `extra/rockchip/npu_linear.py` on the `rkllm`
branch. The walk (`replace_linears` / `prepare_linears`) is the same; only the
matmul target changes.

## Hardware and driver

The reference runtime targets **base M1 (T8103)** on Asahi Linux. The KMD is a
loadable module with a runtime device-tree overlay; the stock Asahi device tree
omits the ANE, DART and power-controller nodes. The module in `~/ane/kmod` is
built for the running kernel in this checkout (`7.1.13+`).

```sh
make -C ~/ane/kmod -j"$(nproc)"
sudo make -C ~/ane/kmod install
sudo modprobe ane
ls -l /dev/accel/
```

`modprobe ane` loads `apple-dart`, `ane_dt.ko` (the overlay) and `ane.ko`. The
installed udev rule grants the `render` group access; for SSH sessions either
join `render` and start a new login, or grant a temporary ACL:

```sh
sudo setfacl -m "u:$USER:rw" /dev/accel/accel0
```

The reference load test checks ADD, MUL, MAX, MIN, SUMSQ, RELU, CONV, GEMM,
CONCAT and SIGMOID, then unloads and reloads the KMD:

```sh
sudo ~/ane/kmod/load-test.sh
```

If `/dev/accel/accel0` is absent, `Ane()` raises
`RuntimeError: M1 ANE device is unavailable`; the helper prints the same
`~/ane/kmod/README.md` pointer. Do not expect a CPU fallback from `--ane` — the
flag is a hardware gate.

## Reproducing

The benchmark takes a local GGUF or a name from `tinygrad.llm.cli.models`.
On this host the tested interpreter is `~/ane/.venv/bin/python`; use any
environment with NumPy, and `PYTHONPATH=.` supplies tinygrad from this checkout.

```sh
PY=~/ane/.venv/bin/python
cd /home/asahi/tinygrad-npu

# CPU host reference
DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --runs 3 \
  --out docs/ane/results/cpu-qwen35-0.8b.json

# Every eligible Linear on the ANE
DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --ane --runs 3 \
  --out docs/ane/results/ane-qwen35-0.8b.json
```

The script prints JSON and can also write it with `--out`. The ANE path keeps
`TinyJit`: the custom kernel is part of the captured graph and is replayed with
the CPU operations. It forces `chunk_size=1` so every shape reaching
`ANELinear` is static.

## Scope and limits

- **One operation.** Only `nn.Linear` runs on the ANE. The fast attention and
  argmax helpers are CPU C loops, not ANE kernels; there is no ANE conv,
  embedding or sampling.
- **JIT.** The ANE matmul is a graph `CUSTOM_FUNCTION`, so `TinyJit` captures
  and replays it with the CPU operations. On Qwen3.5 0.8B the measured decode
  is 11.54 tok/s versus 1.97 on the CPU host at 32 prompt tokens; decode
  matches the `~/ane` 11.57 reference.
- **FP16 weights, FP32 activations.** The ANE plan is FP16; the activation
  crossing the ANE boundary is FP32 with power-of-two input scaling. The
  reference's Hadamard and compensated multi-plane policy (used for its Mirai M
  W4 checkpoint) is not applied here; treat the ANE logits as a numerical
  baseline, not parity.
- **Single owner.** The ANE device is opened once per process. Run one model at
  a time.
- **Memory.** The plans hold a packed FP16 copy of every offloaded weight,
  including the 508 MB vocabulary head. On an 8 GB M1 (7.3 GiB usable), the
  0.8B model fits; a 2B model's head stays on the CPU by the
  `ANE_MAX_WEIGHT` cutoff.
