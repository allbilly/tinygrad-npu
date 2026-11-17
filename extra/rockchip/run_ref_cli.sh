#!/bin/bash
# Run the reference tinygrad on a model directly, with the Qwen3.5 4B recipe's env.
#
# oNPU is a harness plus the recipe; the kernels and the replay live in ~/tinygrad.
# The worker adds a JSON protocol and a `model` field we do not need, so drive the
# CLI's own benchmark instead: `--benchmark PREFILL,DECODE` prints per-step tok/s.
#
# Usage: bash extra/rockchip/run_ref_cli.sh [prefill] [decode]
set -euo pipefail
BACKEND=/home/orangepi/tinygrad
MODEL=/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
PP=${1:-128}
TG=${2:-64}

cd "$BACKEND"
export DEV=CPU HALF=1 REALIZE=1 PYTHONUNBUFFERED=1 BEAM=0 DEBUG=0
export ONPU_PROFILE=ROCKCHIP ONPU_NPU_PREFILL=off ONPU_NPU_DECODE=on ONPU_VISION_GPU=1
# The 4B recipe, verbatim from ~/oNPU/onpu/rockchip_options.py:389-401.
export ONPU_SWIGLU_PREFILL_ROWS=4 ONPU_SWIGLU_PACK_STATIC_LAYOUT=1
export ONPU_CPU_TOKEN_INPUTS=1 ONPU_ZERO_PAD_VIEWS=1 ONPU_SHARED_HOST_POOL=1
export ONPU_PREFILL_CPU_TUNING=1 ONPU_CPU_PREFILL_LAYOUT=1
export ONPU_REPLAY_BORROW_OUTPUTS=1 ONPU_REPLAY_BORROW_BATCH=1
export ONPU_REPLAY_NATIVE_FROM=1 ONPU_REPLAY_NATIVE_FROM_LARGE=1
export ONPU_GDN_DECODE_SPIN=1 ONPU_GDN_PREP_POOL=1 ONPU_FFN_PACKED_UPLOAD=1
export ONPU_EXP2_FAST_PATH=1 ONPU_CPU_DECODE_POOL=0 ONPU_HOST_POOL_WFE=1
export ONPU_GDN_DECODE_PREP_POOL=1 ONPU_GDN_DIRECT_DECODE=1
export ONPU_CPU_GQA_PREFILL=0 ONPU_INT8_SDK_PREFILL=0 ONPU_INT8_EXPERT_DEPTHS=0
export ONPU_INT8_DECODE_NARROW_CBUF=1 ONPU_REPLAY_SCALAR_COPY=1
export ONPU_REPLAY_DEDUP_GUARDS=1 ONPU_INT8_NATURAL2304=1
export ONPU_NPU_DOMAINS=3 ONPU_NPU_DOMAIN_BYTES=3221225472

exec .venv/bin/python -u -m tinygrad.llm.cli --model "$MODEL" --max_context 512 --benchmark "$PP,$TG"
