#!/bin/bash
# 4B paired A/B for the fused INT8 Linear, run detached so a server restart cannot kill it.
#
# Fused first, then the six-launch path, back to back on the same boot so both rows share a
# clock and thermal state. bench_llm records clocks_start/clocks_end in each report.
set -u
cd /home/orangepi/pr/tinygrad/extra/rockchip || exit 1
OUT=/home/orangepi/pr/tinygrad/docs/rknpu/results
CKPT=/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf

run() {
  local tag=$1 script=$2
  echo "=== $tag start $(date -Is) temp=$(cat /sys/class/thermal/thermal_zone0/temp) ==="
  DEV=CPU ../../.venv/bin/python -u "$script" --ckpt "$CKPT" --name "Qwen3.5 4B $tag" \
    --out "$OUT/fused4b-$tag" --npu --quick --warmups 1
  echo "=== $tag rc=$? $(date -Is) ==="
}

run fused  bench_llm.py
run legacy bench_legacy_int8.py
echo "=== all done $(date -Is) ==="
