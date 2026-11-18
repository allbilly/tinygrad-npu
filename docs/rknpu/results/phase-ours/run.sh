#!/usr/bin/env bash
# Split one prefill into main-thread CPU / total CPU / off-CPU, the same instrument the
# reference reports as 720 / 5,140 / 486 ms.
#
# Why this exists: the kernel budget and the wall disagree. Threading the five elementwise
# kernels cut the summed kernel time from 2,391 to 1,772 ms and then to ~1,700 with attention,
# yet the 3+3 wall moved only 2,531.67 -> 1,929.52 ms, and 150.6 ms is still outside every
# kernel. Before choosing between the NPU merge and the scan rewrite, this measures where the
# main thread actually is: running CPU work, waiting on the NPU ioctl, or blocked elsewhere.
#
# bench_llm emits `{"event": "first_token"}` and `{"event": "run"}` for exactly this probe.
#
# Usage: bash docs/rknpu/results/phase-ours/run.sh [runs]
set -euo pipefail

RUNS="${1:-3}"
OUT="$(cd "$(dirname "$0")" && pwd)"
REPO=/home/orangepi/pr/tinygrad

# PARALLEL=0 is required: without it kernels compile in a worker process where thread_split's
# patch does not exist (mistake M49, blocker 2). It affects compile-time parallelism only.
export PARALLEL=0 DEV=CPU:CLANG

cd "$REPO"
exec "$REPO/.venv/bin/python" "$REPO/extra/rockchip/trial_phase_probe.py" \
  --report "$OUT/report.json" \
  --out    "$OUT/phases.jsonl" \
  -- "$REPO/.venv/bin/python" -u "$REPO/extra/rockchip/bench_threaded.py" -- \
      --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
      --name qwen35-4b-phase --out "$OUT" --npu \
      --warmups 3 --runs "$RUNS"
