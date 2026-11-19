#!/usr/bin/env python3
"""Baseline prefill/decode timing for one GGUF on the native tinygrad path.

Protocol: 128-token prompt, 64 greedy decode steps, 1 warm-up pass, 3 measured
passes. Reports every trial plus the median, so a warm-up artifact stays visible.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, sys, time

PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def thermal() -> dict:
    """Per-trial temperature and live clocks.

    This exists because `clocks_start`/`clocks_end` cannot detect a thermal ramp *inside* a
    run, and a ramp charged to whichever code path ran later is exactly how two A/B verdicts
    were got wrong (+27.7 ms and +11.57 ms, both neutral when measured fairly with an
    in-process ABBA). Recording it per trial makes the fairness of a comparison checkable
    after the fact instead of assumed.
    """
    def rd(path: str) -> float | None:
        try:
            return float(pathlib.Path(path).read_text().strip())
        except OSError:
            return None
    temp = rd("/sys/class/thermal/thermal_zone0/temp")
    return {
        "temp_c": round(temp / 1000.0, 1) if temp is not None else None,
        "npu_cur_hz": rd("/sys/class/devfreq/fdab0000.npu/cur_freq"),
        "ddr_cur_hz": rd("/sys/class/devfreq/dmc/cur_freq"),
        "cpu6_cur_hz": rd("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"),
        "cpu6_governor": (pathlib.Path("/sys/devices/system/cpu/cpufreq/policy6/scaling_governor").read_text().strip()
                          if pathlib.Path("/sys/devices/system/cpu/cpufreq/policy6/scaling_governor").exists() else None),
    }

def clocks() -> dict:
    """Device state at call time, recorded so a row can be checked after the fact."""
    return {
        "npu_min_hz": _read("/sys/class/devfreq/fdab0000.npu/min_freq"),
        "npu_max_hz": _read("/sys/class/devfreq/fdab0000.npu/max_freq"),
        "npu_cur_hz": _read("/sys/class/devfreq/fdab0000.npu/cur_freq"),
        "npu_governor": _read("/sys/class/devfreq/fdab0000.npu/governor"),
        "ddr_cur_hz": _read("/sys/class/devfreq/dmc/cur_freq"),
        "cpu4_cur_hz": _read("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq"),
        "cpu4_governor": _read("/sys/devices/system/cpu/cpufreq/policy4/scaling_governor"),
        "cpu6_cur_hz": _read("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"),
        "cpu6_governor": _read("/sys/devices/system/cpu/cpufreq/policy6/scaling_governor"),
        "temperature_c": round(int(_read("/sys/class/thermal/thermal_zone0/temp") or 0) / 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--npu", action="store_true", help="replace nn.Linear with the generic NPU GEMM")
    parser.add_argument("--quick", action="store_true",
                        help="iteration mode: 16-token prompt, 8 decode tokens, no warm-up, 1 pass. "
                             "Surfaces crashes and gross regressions in ~1/16 the time of the full protocol. "
                             "Never quote a --quick number as a measurement.")
    args = parser.parse_args()
    if args.quick:
        args.prompt, args.gen, args.runs = 16, 8, 1
        # A warm-up is what removes JIT capture from the decode figure, so keep it
        # available in quick mode even though the default is off.
        args.warmups = 0 if args.warmups is None else args.warmups
    elif args.warmups is None:
        args.warmups = 1
    # Label by protocol, not by flag: a short run with a warm-up is still not a
    # measurement, and a full-length run without --quick still is.
    full_protocol = args.prompt >= 128 and args.gen >= 64 and args.warmups >= 1 and args.runs >= 3
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256(args.ckpt)

    from tinygrad import Device, Tensor
    from tinygrad.llm.gguf import gguf_parse
    from tinygrad.llm.model import Transformer
    from tinygrad.llm.tokenizer import SimpleTokenizer

    kv, _ = gguf_parse(str(args.ckpt))
    try:
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)
        ids = tokenizer.encode(PROMPT)[:args.prompt]
        prompt_source = "tokenizer"
    except Exception:
        vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else kv[f"{kv['general.architecture']}.vocab_size"]
        ids = [1] + [(i * 37 + 11) % (vocab - 1) + 1 for i in range(args.prompt - 1)]
        prompt_source = "synthetic"
    if len(ids) < args.prompt:
        ids = (ids * (args.prompt // max(1, len(ids)) + 1))[:args.prompt]
    max_context = args.prompt + args.gen + 16

    started = time.time()
    model, _ = Transformer.from_gguf(str(args.ckpt), max_context=max_context)
    # Do not realize every parameter here. The weights stay lazy and each Linear
    # dequantizes its own slice inside prepare(), which drops the source after
    # packing. Realizing up front costs 7 GB of shmem on Qwen3.5 4B (4 GB -> 11 GB)
    # before a single NPU buffer exists, which is what pushed it into OOM.
    npu_linears = 0
    npu_bytes = 0
    if args.npu:
        from npu_linear import packed_weight_bytes, prepare_linears, replace_linears
        from npu_linear import fuse_linears
        fused = fuse_linears(model)
        print(f"fused {fused} qkv groups (gate/up deliberately not fused)", flush=True)
        npu_linears = replace_linears(model)
        prepare_linears(model)
        npu_bytes = packed_weight_bytes(model)
    else:
        from tinygrad.nn.state import get_parameters
        for parameter in get_parameters(model):
            parameter.realize()
    load_seconds = time.time() - started

    temperature = Tensor([0.0])
    prompt_tensor = Tensor(ids, dtype="int32").reshape(1, args.prompt)
    from tinygrad.uop.ops import UOp
    v_start = UOp.variable("start_pos", 0, max_context - 1)
    # Bind ONCE. `v_start.bind(0)` allocates a fresh UOp, and TinyJit keys capture on that UOp's
    # identity, so binding inside the trial makes every trial re-trace the graph (~2.4 s of Python
    # per prefill on top of the 3.34 s of work). A real server creates its variable once.
    pos_zero = v_start.bind(0)

    trial_index = 0

    def trial(tag: str = "") -> dict:
        nonlocal trial_index
        npu_dev = Device["ROCKCHIP"] if args.npu else None
        th_start = thermal()
        begin = time.perf_counter()
        s0 = npu_dev.submit_count if npu_dev else 0
        t0 = npu_dev.task_count if npu_dev else 0
        w0 = npu_dev.submit_seconds if npu_dev else 0.0
        out = model(prompt_tensor, (pos_zero,), temperature).realize()
        ttft_ms = (time.perf_counter() - begin) * 1000
        prefill_submits = (npu_dev.submit_count - s0) if npu_dev else 0
        prefill_tasks = (npu_dev.task_count - t0) if npu_dev else 0
        prefill_npu_ms = (npu_dev.submit_seconds - w0) * 1000 if npu_dev else 0.0
        print(f"  {tag} prefill {ttft_ms/1000:.1f}s {prefill_submits} submits "
              f"{prefill_npu_ms/1000:.1f}s in npu", flush=True)
        # Machine-readable prefill boundary. `trial_phase_probe.py` splits each window into
        # main-thread CPU / total CPU / off-CPU by watching for exactly these events, so the
        # same instrument can be applied to this tree as to the reference (720/5,140/486 ms).
        print(json.dumps({"event": "first_token", "index": trial_index, "first_token_ms": ttft_ms}), flush=True)
        begin = time.perf_counter()
        s0 = npu_dev.submit_count if npu_dev else 0
        t0 = npu_dev.task_count if npu_dev else 0
        w0 = npu_dev.submit_seconds if npu_dev else 0.0
        generated = []
        for step in range(args.gen):
            out = model(out, (v_start.bind(args.prompt + step),), temperature).realize()
            generated.append(int(out.item()))
            if (step + 1) % 16 == 0:
                rate = (step + 1) / (time.perf_counter() - begin)
                print(f"  {tag} decode {step+1}/{args.gen} {rate:.2f} tok/s", flush=True)
        decode_seconds = time.perf_counter() - begin
        rate = args.gen / decode_seconds
        decode_submits = (npu_dev.submit_count - s0) if npu_dev else 0
        decode_tasks = (npu_dev.task_count - t0) if npu_dev else 0
        decode_npu_ms = (npu_dev.submit_seconds - w0) * 1000 if npu_dev else 0.0
        print(f"  {tag} done decode {rate:.2f} tok/s ({decode_seconds:.1f}s) "
              f"{decode_submits/args.gen:.1f} submits/token {decode_tasks/args.gen:.1f} tasks/token "
              f"npu {decode_npu_ms/args.gen:.1f} ms/token", flush=True)
        print(json.dumps({"event": "run", "index": trial_index, "first_token_ms": ttft_ms,
                          "decode_tps": rate, "prefill_submits": prefill_submits,
                          "decode_submits": decode_submits}), flush=True)
        trial_index += 1
        th_end = thermal()
        print(f"  {tag} thermal start {th_start['temp_c']} C -> end {th_end['temp_c']} C"
              f"  npu {th_end['npu_cur_hz']} cpu6 {th_end['cpu6_cur_hz']} ddr {th_end['ddr_cur_hz']}", flush=True)
        return {"ttft_ms": ttft_ms, "decode": rate, "generated_ids": generated, "prefill_submits": prefill_submits,
                "temp_start_c": th_start["temp_c"], "temp_end_c": th_end["temp_c"],
                "clocks_at_trial": th_end,
                "prefill_tasks": prefill_tasks, "decode_submits_per_token": decode_submits / args.gen,
                "decode_tasks_per_token": decode_tasks / args.gen,
                "decode_npu_ms_per_token": decode_npu_ms / args.gen,
                "prefill_npu_ms": prefill_npu_ms}

    print(f"loaded in {load_seconds:.1f}s, {npu_linears} linears, prompt {prompt_source}", flush=True)
    for i in range(args.warmups):
        trial(f"warmup {i+1}/{args.warmups}")
    clocks_start = clocks()

    def write_report(trials, clocks_end):
        """Rewrite after every trial: a crash must not cost the passes already run."""
        report = {
            "kind": ("npu-baseline-timing" if args.npu else "native-baseline-timing")
                    if full_protocol else "quick-check",
            "name": args.name,
            "checkpoint": str(args.ckpt),
            # Hashed once at startup: re-reading 4.4 GB per trial is what lost a
            # completed run before its report was written.
            "checkpoint_sha256": checkpoint_sha256,
            "arch": kv.get("general.architecture"),
            # The reference reports "CPU+ROCKCHIP" when parameters are on the host and
            # layers are placed on the NPU, which is exactly what --npu does. Reporting
            # bare "CPU" hides that every Linear runs on the device.
            "device": f"{Device.DEFAULT}+ROCKCHIP" if args.npu else Device.DEFAULT,
            "npu": args.npu,
            "npu_linears": npu_linears,
            "npu_weight_bytes": npu_bytes,
            "npu_weight_mib": round(npu_bytes / 2**20, 1),
            "prompt_tokens": args.prompt,
            "prompt_source": prompt_source,
            "generated_tokens": args.gen,
            "warmups": args.warmups,
            "runs": args.runs,
            "trials_completed": len(trials),
            "complete": len(trials) == args.runs,
            "load_seconds": round(load_seconds, 2),
            "ttft_ms": [round(t["ttft_ms"], 2) for t in trials],
            "generated_ids": [t["generated_ids"] for t in trials],
            "decode_tokens_per_second": [round(t["decode"], 4) for t in trials],
            "trial_temperature_c": {"start": [t.get("temp_start_c") for t in trials],
                                    "end": [t.get("temp_end_c") for t in trials]},
            "trial_clock_drift_hz": {
                key: (max(v) - min(v)) if (v := [t.get("clocks_at_trial", {}).get(k) for t in trials
                                                 if t.get("clocks_at_trial", {}).get(k) is not None]) else None
                for key, k in (("npu", "npu_cur_hz"), ("ddr", "ddr_cur_hz"), ("cpu6", "cpu6_cur_hz"))},
            "trial_temperature_drift_c": (
                round(max(v) - min(v), 1) if (v := [x for t in trials
                      for x in (t.get("temp_start_c"), t.get("temp_end_c")) if x is not None]) else None),
            "median_ttft_ms": round(statistics.median(t["ttft_ms"] for t in trials), 2),
            "median_decode_tokens_per_second": round(statistics.median(t["decode"] for t in trials), 4),
            "npu_submits": [{"prefill": t["prefill_submits"], "prefill_tasks": t["prefill_tasks"],
                             "prefill_npu_ms": round(t["prefill_npu_ms"], 1),
                             "decode_per_token": round(t["decode_submits_per_token"], 2),
                             "decode_tasks_per_token": round(t["decode_tasks_per_token"], 2),
                             "decode_npu_ms_per_token": round(t["decode_npu_ms_per_token"], 1)} for t in trials],
            "clocks_start": clocks_start,
            "clocks_end": clocks_end,
            "total_seconds": round(time.time() - started, 1),
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    trials: list[dict] = []
    for _ in range(args.runs):
        trials.append(trial(f"trial {len(trials)+1}/{args.runs}"))
        report = write_report(trials, clocks())
        print(f"  wrote report: {len(trials)}/{args.runs} trials, "
              f"median ttft {report['median_ttft_ms']:.0f} ms decode {report['median_decode_tokens_per_second']:.2f} tok/s", flush=True)
    print(json.dumps({k: report[k] for k in ("name", "median_ttft_ms", "median_decode_tokens_per_second")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
