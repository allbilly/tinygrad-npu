#!/usr/bin/env python3
"""Run the oNPU reference on a model once and report its own timings.

The archive is read-only for design, but running it once is the oracle: it says what
the target actually is on this board, with its own recipe, rather than us inferring
it from a table.

Drives `python -m onpu.worker` over its JSON protocol: wait for `ready`, send one
`generate`, read events until one carries `ttft_ms`.

Usage: python extra/rockchip/run_ref.py --model <gguf> [--context 512] [--tokens 64]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

BACKEND = pathlib.Path.home() / "tinygrad"
ONPU = pathlib.Path.home() / "oNPU"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The history of the printing press begins with movable type in East Asia.")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--profile", action="store_true",
                    help="enable ONPU_REPLAY_PROFILE and dump per-call samples to --calls-out")
    ap.add_argument("--calls-out", default="/home/orangepi/pr/tinygrad/docs/rknpu/results/ref-calls.jsonl")
    args = ap.parse_args()

    env = dict(os.environ)
    env.update(DEV="CPU", PYTHONPATH=os.pathsep.join([str(BACKEND), str(ONPU)]), DEBUG="0",
               REALIZE="1", HALF="1", PYTHONUNBUFFERED="1", BEAM="0",
               ONPU_PROFILE="ROCKCHIP", ONPU_NPU_PREFILL="off", ONPU_NPU_DECODE="on",
               ONPU_VISION_GPU="1")
    # The Qwen3.5 4B recipe's own environment (`rockchip_options.py:389-401`, the
    # `large = True` branch). Without these the worker tries a single >3 GiB NPU
    # allocation and the driver returns EFAULT, which is what blocked the first two
    # attempts: the archive splits weights across 3 IOMMU domains for exactly this.
    env.update({
        "ONPU_SWIGLU_PREFILL_ROWS": "4", "ONPU_SWIGLU_PACK_STATIC_LAYOUT": "1",
        "ONPU_CPU_TOKEN_INPUTS": "1", "ONPU_ZERO_PAD_VIEWS": "1", "ONPU_SHARED_HOST_POOL": "1",
        "ONPU_PREFILL_CPU_TUNING": "1", "ONPU_CPU_PREFILL_LAYOUT": "1",
        "ONPU_REPLAY_BORROW_OUTPUTS": "1", "ONPU_REPLAY_BORROW_BATCH": "1",
        "ONPU_REPLAY_NATIVE_FROM": "1", "ONPU_REPLAY_NATIVE_FROM_LARGE": "1",
        "ONPU_GDN_DECODE_SPIN": "1", "ONPU_GDN_PREP_POOL": "1", "ONPU_FFN_PACKED_UPLOAD": "1",
        "ONPU_CPU_DECODE_POOL": "0", "ONPU_HOST_POOL_WFE": "1",
        "ONPU_GDN_DECODE_PREP_POOL": "1", "ONPU_GDN_DIRECT_DECODE": "1",
        "ONPU_CPU_GQA_PREFILL": "0", "ONPU_REPLAY_SCALAR_COPY": "1",
        "ONPU_REPLAY_DEDUP_GUARDS": "1", "ONPU_NPU_DOMAINS": "3",
        "ONPU_NPU_DOMAIN_BYTES": str(3 * 1024 ** 3),
        "ONPU_EXP2_FAST_PATH": "1", "ONPU_INT8_DECODE_NARROW_CBUF": "1",
        "ONPU_INT8_NATURAL2304": "1", "ONPU_INT8_SDK_PREFILL": "0",
        "ONPU_INT8_EXPERT_DEPTHS": "0",
    })
    if args.profile:
        # The replay collects per-label times into `profile_ns` but the text path never
        # prints them; ONPU_REPLAY_CALLS_OUT is the only channel that survives the
        # subprocess boundary. Profiling inflates the timings, so never quote a rate
        # from a --profile run.
        env["ONPU_REPLAY_PROFILE"] = "1"
        env["ONPU_REPLAY_CALLS_OUT"] = args.calls_out
        pathlib.Path(args.calls_out).unlink(missing_ok=True)
    python = BACKEND / ".venv/bin/python"
    cmd = [str(python), "-m", "onpu.worker", "--model", args.model, "--context", str(args.context),
           "--realize", "1", "--speculation", "off", "--draft-path", ""]
    print(f"launching: {' '.join(cmd[:6])} ...", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(BACKEND), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    started = time.perf_counter()
    sent = False
    try:
        for line in proc.stdout:
            if time.perf_counter() - started > args.timeout:
                print("timeout", flush=True)
                return 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "ready" and not sent:
                print(f"ready: {event.get('name')} context={event.get('context')} "
                      f"device={event.get('device')} after {time.perf_counter()-started:.1f}s", flush=True)
                body = {"model": pathlib.Path(args.model).stem,
                        "messages": [{"role": "user", "content": args.prompt}],
                        "max_tokens": args.tokens, "temperature": 0.0}
                proc.stdin.write(json.dumps({"op": "generate", "id": "t1", "body": body}) + "\n")
                proc.stdin.flush()
                sent = True
            elif kind == "error":
                print(f"error: {event.get('message')}", flush=True)
                return 1
            elif "ttft_ms" in event:
                print(json.dumps({k: event.get(k) for k in
                                  ("ttft_ms", "total_seconds", "decode_tps", "prefill_tps",
                                   "completion_tokens", "prompt_tokens", "device")}, indent=2), flush=True)
                return 0
    finally:
        proc.kill()
        # The worker reports failures on stderr, so never discard it: a silent exit is
        # what made the first reference attempt unattributable.
        err = proc.stderr.read() if proc.stderr else ""
        if err:
            print("--- worker stderr (tail) ---", flush=True)
            print("\n".join(err.strip().splitlines()[-25:]), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
