# ANE matmul offload

Minimal Apple ANE GEMM baseline: every eligible `nn.Linear` runs on the ANE,
everything else stays on the CPU host. The runtime is the direct-register M1
implementation in `~/ane` (set `ANE_REF` to override).

```sh
cd /home/asahi/tinygrad-npu
PY=~/ane/.venv/bin/python

# CPU reference
DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8

# ANE (requires /dev/accel/accel0). Pin to the M1 performance cores: the
# default schedutil governor puts the CPU kernels on the efficiency cores and
# costs ~25% (9.4 vs 12.4 tok/s prefill).
taskset -c 4-7 env DEV=CPU PYTHONPATH=. "$PY" extra/ane/bench_llm.py \
  --model ~/.cache/ane-models/Qwen3.5-0.8B-Q8_0.gguf \
  --prompt-tokens 32 --decode-tokens 8 --ane --warmup 3 --runs 3

# Device-only matmul check
PYTHONPATH=. "$PY" extra/ane/smoke_matmul.py --k 1024 --n 1024 --rows 1
```

Files:

- `ane_linear.py` — `ANELinear`, `replace_linears`, `prepare_linears`,
  `disable_jit`.
- `bench_llm.py` — prefill/decode timing for one GGUF.
- `smoke_matmul.py` — NumPy comparison before loading a model.

The docs, driver setup and measured baseline are in `docs/ane/`.
