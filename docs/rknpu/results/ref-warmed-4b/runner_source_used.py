"""Source-pinned, warmed 1+3 retimes for Qwen3.5 2B/4B and Spark-X2.5."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

APP = Path("/home/orangepi/oNPU")
BACKEND = Path("/home/orangepi/tg-qwen35-pack-signature-fix")
BACKEND_REVISION = "1b2a5db332e1198b3ace8c7559a19c7d73717dfb"
BACKEND_BRANCH = "codex/qwen35-pack-signature-fix"
BACKEND_DIFF_SHA256 = "7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77"
BACKEND_DIRTY_PATHS = {
    "extra/rockchip/benchmark_inference.py",
    "test/models/test_cpu_gdn_benchmark.py",
    "test/models/test_rockchip_gelu_ffn.py",
    "tinygrad/llm/kernels/rockchip_gelu_ffn.py",
}
APP_REVISION = "5229660be83d3ee8e155e31109518fb90d641e80"
PROFILE = APP / "onpu/rockchip_options.py"
PROFILE_SHA256 = "3394f6dffb81529bb567e2e5982c8ca306ae611b0004e5d703e8b641acbd85b6"
INPUT_ROOT = Path(
    "/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/"
    "staged/w8a8-sweep-v1/inputs"
)
CHECKPOINT_ROOT = Path("/home/orangepi/vlm-bench/baseline-20261021/checkpoints")
OUTPUT_ROOT = APP / "analysis/release-20261115/r2-lfm-20261116"
PYTHON = Path("/home/orangepi/tinygrad/.venv/bin/python")
HW_LOCK = APP / ".hw-lock"
MIN_START_MILLIC, MAX_START_MILLIC = 52_000, 55_000
REQUIRED_AVAILABLE_MEMORY = 8 * 1024**3
WARMUPS, RUNS = 1, 3
READY_GATE_TIMEOUT_SECONDS = 7200
CHILD = Path(__file__).with_name("candidate_timing_child.py")
CHILD_SHA256 = "b2e556b46764e3a06808ec566818e04338a7005bd4f0a1bfb4fe42e3cf66b88b"
WARM_MODEL_KEYS = {"qwen35-2b", "qwen35-4b", "spark25"}

MODELS = {
    "qwen35-2b": {
        "name": "Qwen3.5 2B",
        "checkpoint": CHECKPOINT_ROOT / "Qwen3.5-2B/Qwen3.5-2B-Q8_0.gguf",
        "checkpoint_sha256": "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
        "input": INPUT_ROOT / "qwen35-2b.txt",
        "input_sha256": "921df69467179a5ca2fc57e0a94544e29c774882ff1ced16eeb18c2ff10fd201",
        "profile_name": "qwen35-2b-context512",
        "projection_options": {
            "recurrent_scan": True, "swiglu": True, "recurrent_fast": True,
        },
    },
    "qwen35-4b": {
        "name": "Qwen3.5 4B",
        "checkpoint": CHECKPOINT_ROOT / "Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf",
        "checkpoint_sha256": "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1",
        "input": INPUT_ROOT / "qwen35-4b.txt",
        "input_sha256": "921df69467179a5ca2fc57e0a94544e29c774882ff1ced16eeb18c2ff10fd201",
        "profile_name": "qwen35-4b-context512",
        "projection_options": {
            "recurrent_scan": True, "swiglu": True, "recurrent_fast": True,
        },
    },
    "spark25": {
        "name": "Spark-X2.5 1.7B",
        "checkpoint": CHECKPOINT_ROOT / "Spark-X2.5-1.7B/Spark-X2.5-1.7B-Q8_0.gguf",
        "checkpoint_sha256": "cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442",
        "input": INPUT_ROOT / "spark-x25-17b.txt",
        "input_sha256": "81f2e439c9c82df5b7a136069d168693b7796db56c6a88cc26caf0604b0a5bd1",
        "profile_name": "spark-x2.5-1.7b-context512-shared",
        "projection_options": {"erf_gelu": True},
    },
}

BASE_FLAGS = (
    "--fuse-projections",
    "--cpu-attention",
    "--native-replay",
    "--fused-ffn",
    "--split-down",
    "--fused-qkv",
    "--cpu-norm",
    "--cpu-fast-softmax",
    "--cache-cleaning",
    "--wake-helper",
    "--cpu-pool",
)
OPTION_FLAGS = {
    "cpu_static_gemm": "--cpu-static-gemm",
    "cpu_gelu": "--cpu-gelu",
    "cpu_softcap": "--cpu-softcap",
    "fused_gelu_ffn": "--fused-gelu-ffn",
    "cpu_prefill_pool": "--cpu-prefill-pool",
    "share_projection_weights": "--share-projection-weights",
    "cpu_recurrent_scan": "--cpu-recurrent-scan",
    "fused_recurrent_projections": "--fused-recurrent-projections",
    "fused_swiglu_ffn": "--fused-swiglu-ffn",
    "fused_swiglu_decode": "--fused-swiglu-decode",
    "fused_qkv_without_norm": "--fused-qkv-without-norm",
    "cpu_erf_gelu": "--cpu-erf-gelu",
    "fused_erf_gelu_ffn": "--fused-erf-gelu-ffn",
    "fused_erf_gelu_decode": "--fused-erf-gelu-decode",
    "cpu_head_gate": "--cpu-head-gate",
    "cpu_exact_rows": "--cpu-exact-rows",
    "cpu_sparse_lookup": "--cpu-sparse-lookup",
    "fused_gelu_decode": "--fused-gelu-decode",
    "fused_gelu_sdk_prefill": "--fused-gelu-sdk-prefill",
}


class Blocked(RuntimeError):
    """A prerequisite failed; no inference may start."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )
    return result.stdout if binary else result.stdout.decode().strip()


def backend_identity() -> dict:
    status = git(BACKEND, "status", "--porcelain=v1", "--untracked-files=all", "-z", binary=True)
    diff = git(BACKEND, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(status, bytes) and isinstance(diff, bytes)
    paths = sorted(entry[3:].decode(errors="replace") for entry in status.split(b"\0") if entry)
    identity = {
        "path": str(BACKEND),
        "branch": str(git(BACKEND, "branch", "--show-current")),
        "revision": str(git(BACKEND, "rev-parse", "HEAD")),
        "dirty_paths": paths,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    if identity["branch"] != BACKEND_BRANCH or identity["revision"] != BACKEND_REVISION:
        raise Blocked(f"backend identity changed: {identity['branch']}@{identity['revision']}")
    if set(paths) != BACKEND_DIRTY_PATHS or identity["dirty_diff_sha256"] != BACKEND_DIFF_SHA256:
        raise Blocked(f"backend dirty patch changed: {paths}, {identity['dirty_diff_sha256']}")
    return identity


def load_profile(model: dict, checkpoint_sha256: str) -> tuple[dict, dict]:
    if sha256(PROFILE) != PROFILE_SHA256:
        raise Blocked("app profile source hash changed")
    if str(git(APP, "rev-parse", "HEAD")) != APP_REVISION:
        raise Blocked("oNPU app revision changed")
    sys.path.insert(0, str(APP))
    options_module = importlib.import_module("onpu.rockchip_options")
    if Path(options_module.__file__).resolve() != PROFILE.resolve():
        raise Blocked(f"profile imported from an unexpected path: {options_module.__file__}")
    profile = options_module.text_execution_profile(
        checkpoint_sha256, 512, "ROCKCHIP", "on", decode="on"
    )
    if not profile or profile.get("prefill_profile") != model["profile_name"]:
        raise Blocked(f"no exact source-pinned profile for {model['name']}")
    if profile.get("prefill") != "on" or profile.get("mapped_cpu") is not True:
        raise Blocked(f"{model['name']} profile is not the expected mapped CPU/NPU recipe")
    configure = options_module.text_projection_options(
        lambda **kwargs: kwargs,
        cpu_npu=True,
        precision="int8",
        text_checkpoint=True,
        prefill="on",
        decode="on",
        share_projection_weights=profile.get("share_projection_weights"),
        **model["projection_options"],
    )
    return profile, configure


def build_command(model: dict, profile: dict, configure: dict, out: Path) -> tuple[list[str], dict]:
    flags = list(BASE_FLAGS)
    for option, enabled in configure.items():
        if option not in OPTION_FLAGS:
            if enabled:
                raise Blocked(f"no CLI mapping for active backend option {option}")
            continue
        if enabled:
            flags.append(OPTION_FLAGS[option])

    if profile.get("cpu_q8_embedding"):
        flags.append("--cpu-q8-embedding")
    if profile.get("cpu_q8_side_embedding"):
        flags.append("--cpu-q8-side-embedding")
    if profile.get("scalar_decode_loop"):
        flags.append("--scalar-decode-loop")
    if profile.get("trim_preparation_cache"):
        flags.append("--trim-preparation-cache")
    if profile.get("mapped_cpu"):
        flags.extend(("--file-backed-cpu-weights", "--defer-weight-realization"))

    command = [
        shutil.which("taskset") or "taskset",
        "-c", "4-7",
        str(PYTHON), str(CHILD),
        "--checkpoint", str(model["checkpoint"]),
        "--input", str(model["input"]),
        "--device", "CPU",
        "--precision", "int8",
        "--prefill", "on",
        "--decode", "on",
        "--context", "512",
        "--tokens", "64",
        "--warmups", str(WARMUPS),
        "--runs", str(RUNS),
        "--chunk-size", str(profile.get("fixed_prefill_chunk", profile.get("prefill_chunk", 128))),
        "--tile-columns", str(profile["tile_columns"]),
        *flags,
        "--report", str(out / "benchmark.json"),
    ]
    return command, {
        "profile": profile,
        "configure_options": configure,
        "flags": flags,
    }


def load_config(key: str, out: Path) -> dict:
    if sys.version_info < (3, 11) or not PYTHON.is_file():
        raise Blocked(f"run with the project interpreter {PYTHON}")
    backend = backend_identity()
    if sha256(PROFILE) != PROFILE_SHA256:
        raise Blocked("app profile source hash changed")

    model = MODELS[key]
    if key not in WARM_MODEL_KEYS:
        raise Blocked(f"{key} is not an approved stale-row warmed retime")
    for path_key, hash_key, label in (
        ("checkpoint", "checkpoint_sha256", "GGUF"),
        ("input", "input_sha256", "fixed 128-ID fixture"),
    ):
        path = model[path_key]
        if not path.is_file() or sha256(path) != model[hash_key]:
            raise Blocked(f"{model['name']} {label} missing or hash mismatch")
    if not CHILD.is_file() or sha256(CHILD) != CHILD_SHA256:
        raise Blocked("the shared timing child source is missing or differs from its pinned hash")
    ids = [int(value) for value in model["input"].read_text().split()]
    if len(ids) != 128 or any(value < 0 for value in ids):
        raise Blocked(f"{model['name']} fixed input is not exactly 128 nonnegative token IDs")

    profile, configure = load_profile(model, model["checkpoint_sha256"])
    benchmark_path = BACKEND / "extra/rockchip/benchmark_inference.py"
    sys.path.insert(0, str(BACKEND))
    benchmark = importlib.import_module("extra.rockchip.benchmark_inference")
    if Path(benchmark.__file__).resolve() != benchmark_path.resolve():
        raise Blocked(f"benchmark imported from an unexpected path: {benchmark.__file__}")
    sources = benchmark.source_snapshot(BACKEND)
    if not sources or sources.get("extra/rockchip/benchmark_inference.py") != sha256(benchmark_path):
        raise Blocked("backend source snapshot does not include the selected benchmark source")
    help_result = subprocess.run(
        [str(PYTHON), "-m", "extra.rockchip.benchmark_inference", "--help"],
        cwd=BACKEND,
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
             "PYTHONPATH": str(BACKEND), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if help_result.returncode:
        raise Blocked(f"selected benchmark help failed: {help_result.stderr.strip()}")

    command, selected_recipe = build_command(model, profile, configure, out)
    declared = set(re.findall(r"--[a-z][a-z0-9-]*", help_result.stdout))
    missing = sorted({arg for arg in command if arg.startswith("--")} - declared)
    if missing:
        raise Blocked(f"selected backend CLI is missing profile options: {missing}")
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "XDG_RUNTIME_DIR")
        if key in os.environ
    }
    environment.update({str(key): str(value) for key, value in profile["environment"].items()})
    environment.update({
        "ONPU_BENCHMARK_CHECKPOINT_SHA256": model["checkpoint_sha256"],
        "DEV": "CPU", "HALF": "1", "JIT": "1", "CAPTURING": "1", "PARALLEL": "0",
        "DEBUG": "0", "PROFILE": "0", "VIZ": "0", "PYTHONPATH": str(BACKEND),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if environment.get("ONPU_CPU_DECODE_POOL") == "1" and environment.get("ONPU_HOST_POOL_WFE") != "1":
        raise Blocked("ONPU_HOST_POOL_WFE=1 is mandatory with ONPU_CPU_DECODE_POOL=1")

    return {
        "model_key": key,
        "model": model["name"],
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "backend": backend,
        "backend_source_map": sources,
        "backend_source_count": len(sources),
        "backend_benchmark": str(benchmark_path),
        "backend_benchmark_sha256": sha256(benchmark_path),
        "app_revision": APP_REVISION,
        "app_profile": {"path": str(PROFILE), "sha256": PROFILE_SHA256},
        "checkpoint": {"path": str(model["checkpoint"]), "sha256": model["checkpoint_sha256"],
                       "size_bytes": model["checkpoint"].stat().st_size},
        "input": {"path": str(model["input"]), "sha256": model["input_sha256"],
                  "token_count": len(ids), "token_ids": ids},
        "selected_recipe": selected_recipe,
        "command": command,
        "environment": environment,
        "recorded_environment": {
            key: value for key, value in environment.items()
            if key.startswith("ONPU_") or key in {
                "DEV", "HALF", "JIT", "CAPTURING", "PARALLEL", "DEBUG", "PROFILE", "VIZ",
                "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            }
        },
    }


def read_int(path: str) -> int:
    return int(Path(path).read_text().strip())


def hardware_snapshot() -> dict:
    values = {
        "temperature_millic": read_int("/sys/class/thermal/thermal_zone0/temp"),
        "npu_min_hz": read_int("/sys/class/devfreq/fdab0000.npu/min_freq"),
        "npu_cur_hz": read_int("/sys/class/devfreq/fdab0000.npu/cur_freq"),
        "ddr_cur_hz": read_int("/sys/class/devfreq/dmc/cur_freq"),
    }
    for policy in ("policy4", "policy6"):
        root = Path(f"/sys/devices/system/cpu/cpufreq/{policy}")
        values[f"{policy}_governor"] = (root / "scaling_governor").read_text().strip()
        values[f"{policy}_cur_khz"] = read_int(str(root / "scaling_cur_freq"))
    meminfo = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", meminfo, re.MULTILINE)
    values["mem_available_bytes"] = int(match.group(1)) * 1024 if match else None
    values["hw_lock_present"] = HW_LOCK.exists()
    fuser = shutil.which("fuser")
    if not fuser:
        raise Blocked("fuser is required to verify exclusive DRM ownership")
    owners = subprocess.run(
        [fuser, "/dev/dri/renderD128", "/dev/dri/renderD129"],
        capture_output=True,
        text=True,
        check=False,
    )
    values["drm_owner_output"] = (owners.stdout + owners.stderr).strip()
    values["drm_owner_present"] = owners.returncode == 0
    return values


def validate_start(snapshot: dict, *, allow_own_lock: bool = False) -> None:
    if not MIN_START_MILLIC <= snapshot["temperature_millic"] <= MAX_START_MILLIC:
        raise Blocked(
            f"start temperature {snapshot['temperature_millic']} m°C is outside "
            f"{MIN_START_MILLIC}–{MAX_START_MILLIC} m°C"
        )
    requirements = {
        "npu_min_hz": 1_000_000_000,
        "npu_cur_hz": 1_000_000_000,
        "ddr_cur_hz": 2_112_000_000,
        "policy4_governor": "performance",
        "policy6_governor": "performance",
        "policy4_cur_khz": 2_256_000,
        "policy6_cur_khz": 2_256_000,
    }
    failures = [f"{key}={snapshot.get(key)!r}, expected {value!r}"
                for key, value in requirements.items() if snapshot.get(key) != value]
    if failures:
        raise Blocked("clock gate failed: " + "; ".join(failures))
    if snapshot["hw_lock_present"] and not allow_own_lock:
        raise Blocked("hardware lock already exists")
    if snapshot["drm_owner_present"]:
        raise Blocked(f"DRM device already has an owner: {snapshot['drm_owner_output']}")
    if not snapshot["mem_available_bytes"] or snapshot["mem_available_bytes"] < REQUIRED_AVAILABLE_MEMORY:
        raise Blocked("less than 8 GiB available memory")


def append_sample(path: Path, event: str, trial_index: int, started_ns: int) -> None:
    sample = {
        "event": event,
        "trial_index": trial_index,
        "timestamp_ns": time.time_ns(),
        "elapsed_s": round((time.monotonic_ns() - started_ns) / 1e9, 3),
        "temperature_millic": read_int("/sys/class/thermal/thermal_zone0/temp"),
    }
    with path.open("a") as output:
        output.write(json.dumps(sample, sort_keys=True) + "\n")
        output.flush()


def validate_ready_path(key: str, event: dict, profile: dict) -> dict:
    placement = event.get("placement") or {}
    common = {
        "shared_host_pool": True,
        "cpu_pool": True,
        "cpu_prefill_pool": True,
        "cpu_fast_softmax": True,
        "cache_cleaning": True,
        "wake_helper": True,
        "native_replay": True,
        "share_projection_weights": True,
        "precision": "int8",
        "tile_columns": profile["tile_columns"],
    }
    for name, expected in common.items():
        if placement.get(name) != expected:
            raise ValueError(f"ready placement {name}={placement.get(name)!r}, expected {expected!r}")
    if placement.get("cpu_attention_blocks", 0) <= 0 or placement.get("shared_projection_layers", 0) <= 0:
        raise ValueError("ready placement lacks CPU attention or shared projection layers")
    if key in {"qwen35-2b", "qwen35-4b"}:
        attention = placement.get("cpu_attention_blocks", 0)
        recurrent = placement.get("cpu_recurrent_blocks", 0)
        blocks = attention + recurrent
        if (
            recurrent <= 0
            or placement.get("cpu_recurrent_scan") is not True
            or placement.get("cpu_gdn_direct_decode") is not True
            or placement.get("cpu_recurrent_pool") is not True
            or placement.get("fused_recurrent_projection_blocks") != recurrent
            or placement.get("fused_swiglu_ffn_blocks") != blocks
            or placement.get("fused_swiglu_decode_blocks") != blocks
        ):
            raise ValueError("ready Qwen3.5 placement lacks the complete recurrent/SwiGLU path")
    elif key == "spark25":
        erf_blocks = placement.get("cpu_erf_gelu_blocks", 0)
        if (
            erf_blocks <= 0
            or placement.get("cpu_head_gate_blocks") != erf_blocks
            or placement.get("fused_erf_gelu_ffn_blocks") != erf_blocks
            or placement.get("fused_erf_gelu_decode_blocks") != erf_blocks
        ):
            raise ValueError("ready Spark-X2.5 placement lacks the complete erf-GELU/head-gate path")
    else:
        raise ValueError(f"no ready-path contract exists for {key}")
    return {"passed": True, "placement": placement, "validated_before_warmup": True}


def wait_for_ready_temperature(out: Path) -> dict:
    """Wait at the ready FIFO until the first request begins within the accepted band."""
    path = out / "ready_temperature_gate.jsonl"
    started = time.monotonic()
    last_sample: dict = {}
    sample_count = 0
    while True:
        snapshot = hardware_snapshot()
        temperature = snapshot["temperature_millic"]
        sample = {
            "timestamp_unix": time.time(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "temperature_millic": temperature,
            "npu_cur_hz": snapshot["npu_cur_hz"],
            "ddr_cur_hz": snapshot["ddr_cur_hz"],
            "policy4_cur_khz": snapshot["policy4_cur_khz"],
            "policy6_cur_khz": snapshot["policy6_cur_khz"],
        }
        last_sample = sample
        sample_count += 1
        append_event(path, sample)
        if MIN_START_MILLIC <= temperature <= MAX_START_MILLIC:
            requirements = {
                "npu_min_hz": 1_000_000_000,
                "npu_cur_hz": 1_000_000_000,
                "ddr_cur_hz": 2_112_000_000,
                "policy4_governor": "performance",
                "policy6_governor": "performance",
                "policy4_cur_khz": 2_256_000,
                "policy6_cur_khz": 2_256_000,
            }
            failures = [
                f"{name}={snapshot.get(name)!r}, expected {expected!r}"
                for name, expected in requirements.items()
                if snapshot.get(name) != expected
            ]
            if failures:
                return {
                    "passed": False,
                    "error": "ready clock gate failed: " + "; ".join(failures),
                    "accepted_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
                    "wait_seconds": round(time.monotonic() - started, 3),
                    "sample_count": sample_count,
                    "last_sample": last_sample,
                }
            return {
                "passed": True,
                "accepted_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
                "wait_seconds": round(time.monotonic() - started, 3),
                "sample_count": sample_count,
                "last_sample": last_sample,
            }
        if time.monotonic() - started >= READY_GATE_TIMEOUT_SECONDS:
            return {
                "passed": False,
                "error": (
                    f"post-ready temperature did not enter {MIN_START_MILLIC}–"
                    f"{MAX_START_MILLIC} m°C within {READY_GATE_TIMEOUT_SECONDS} seconds"
                ),
                "accepted_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
                "wait_seconds": round(time.monotonic() - started, 3),
                "sample_count": sample_count,
                "last_sample": last_sample,
            }
        time.sleep(5)


def validate_report(config: dict, report_path: Path, temperatures_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    trials = report.get("trials", [])
    warmups = [trial for trial in trials if trial.get("warmup") is True]
    measured = [trial for trial in trials if trial.get("warmup") is False]
    if (
        report.get("complete") is not True
        or report.get("warmups") != WARMUPS
        or report.get("runs") != RUNS
        or len(warmups) != WARMUPS
        or len(measured) != RUNS
        or len(trials) != WARMUPS + RUNS
    ):
        raise Blocked(f"benchmark report did not complete exactly {WARMUPS}+{RUNS} requests")
    if report.get("checkpoint_sha256") != config["checkpoint"]["sha256"]:
        raise Blocked("benchmark report checkpoint hash differs from the pinned GGUF")
    if report.get("source_sha256") != config["backend_source_map"]:
        raise Blocked("benchmark report source map differs from the source-pinned backend")
    if report.get("context") != 512 or report.get("input_ids") != config["input"]["token_ids"]:
        raise Blocked("benchmark report context or fixed prompt IDs differ from the pinned workload")
    for trial in trials:
        if trial.get("prompt_tokens") != 128 or len(trial.get("generated_ids", [])) != 64:
            raise Blocked(f"trial {trial.get('index')} is not exactly 128/64")
        if not trial.get("prefill_submits") or not trial.get("decode_submits"):
            raise Blocked(f"trial {trial.get('index')} lacks prefill/decode submits")
    if len({tuple(trial["generated_ids"]) for trial in measured}) != 1:
        raise Blocked("measured trials generated different token IDs")
    for option, expected in config["selected_recipe"]["configure_options"].items():
        placement = report.get("placement") or {}
        if option == "cpu_prefill_pool":
            observed = placement.get(option)
        elif option == "fused_gelu_sdk_prefill":
            observed = placement.get("fused_gelu_sdk_prefill_blocks", 0) > 0
        else:
            observed = report.get(option)
        if observed != expected:
            raise Blocked(f"benchmark report {option}={observed!r}, expected {expected!r}")
    placement = report.get("placement") or {}
    for option in ("fused_qkv", "cpu_norm", "split_down"):
        if report.get(option) is not True:
            raise Blocked(f"benchmark report did not confirm required CLI option {option}")
    for option in ("native_replay", "cpu_fast_softmax", "cache_cleaning", "wake_helper", "cpu_prefill_pool"):
        if placement.get(option) is not True:
            raise Blocked(f"benchmark placement did not confirm required feature {option}")
    if not placement.get("cpu_attention_blocks") or not placement.get("shared_projection_layers"):
        raise Blocked("benchmark placement did not confirm CPU attention/shared projections")
    precision = report.get("precision")
    if report.get("device") != "CPU" or not isinstance(precision, str) or not precision.lower().startswith("int8"):
        raise Blocked("benchmark report does not match the requested CPU/INT8 execution")
    if report.get("prefill") != "on" or report.get("decode") != "on":
        raise Blocked("benchmark report did not enable both phases")

    native = report.get("native_execution") or {}
    prefill_native = native.get("prefill") or {}
    decode_native = native.get("decode") or {}
    for phase, evidence in (("prefill", prefill_native), ("decode", decode_native)):
        if evidence.get("runs", 0) < 1:
            raise Blocked(f"native {phase} replay did not execute")
    calls = report.get("captured_calls") or {}
    npu_calls = {
        phase: sum(
            count
            for label, count in (calls.get(phase) or {}).items()
            if label.startswith("PROGRAM:ROCKCHIP")
        )
        for phase in ("prefill", "decode")
    }
    cpu_calls = {
        phase: sum(
            count
            for label, count in (calls.get(phase) or {}).items()
            if label.startswith("PROGRAM:CPU")
        )
        for phase in ("prefill", "decode")
    }
    resident = report.get("npu_resident_bytes_after_prepare") or {}
    if not all(npu_calls.values()) or not all(cpu_calls.values()) or sum(resident.values()) <= 0:
        raise Blocked("report does not prove CPU+ROCKCHIP execution in both native phases")

    samples = [json.loads(line) for line in temperatures_path.read_text().splitlines() if line]
    sample_map: dict[int, dict[str, dict]] = {}
    for sample in samples:
        sample_map.setdefault(sample["trial_index"], {})[sample["event"]] = sample
    expected_indices = {trial["index"] for trial in trials}
    if set(sample_map) != expected_indices or any(
        set(events) != {"begin", "end"} for events in sample_map.values()
    ):
        raise Blocked("per-trial temperature observer lacks begin/end for every request")
    if any(
        "temperature_millic" not in sample
        for events in sample_map.values()
        for sample in events.values()
    ):
        raise Blocked("per-trial temperature observer has unavailable samples")

    ttft = [float(trial["first_token_ms"]) for trial in measured]
    decode = [float(trial["decode_tps"]) for trial in measured]
    if any(value <= 0 for value in (*ttft, *decode)):
        raise Blocked("benchmark report has non-positive measured timings")
    npu_path = "/sys/class/devfreq/fdab0000.npu/cur_freq"
    ddr_path = "/sys/class/devfreq/dmc/cur_freq"
    cpu4_path = "/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq"
    cpu6_path = "/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"
    for trial in measured:
        clocks = trial.get("clocks_after", {})
        if clocks.get(ddr_path) != "2112000000":
            raise Blocked(f"DDR clock differs on measured trial {trial.get('index')}")
        if clocks.get(cpu4_path) != "2256000" or clocks.get(cpu6_path) != "2256000":
            raise Blocked(f"CPU4/6 clocks differ on measured trial {trial.get('index')}")
        if clocks.get(npu_path) is None:
            raise Blocked(f"NPU finish clock missing on measured trial {trial.get('index')}")
    by_npu_finish: dict[str, list[dict]] = {}
    for trial in measured:
        by_npu_finish.setdefault(trial["clocks_after"][npu_path], []).append(trial)
    clock_groups = {
        clock: {
            "trial_indices": [trial["index"] for trial in bucket],
            "ttft_ms": [float(trial["first_token_ms"]) for trial in bucket],
            "decode_tps": [float(trial["decode_tps"]) for trial in bucket],
        }
        for clock, bucket in sorted(by_npu_finish.items())
    }
    measured_indices = {trial["index"] for trial in measured}
    measured_temperatures = [
        sample["temperature_millic"] / 1000
        for index, events in sample_map.items()
        if index in measured_indices
        for sample in events.values()
    ]
    return {
        "report": report,
        "warmup": warmups[0],
        "measured_trials": measured,
        "metrics": {
            "median_ttft_ms": statistics.median(ttft),
            "ttft_ms": ttft,
            "ttft_max_min_ratio": max(ttft) / min(ttft),
            "median_decode_tps": statistics.median(decode),
            "decode_tps": decode,
            "decode_interval_ms": [63_000 / value for value in decode],
            "prompt_tokens": 128,
            "generated_ids": 64,
            "measured_ids_identical": True,
        },
        "native_prefill_replay_runs": prefill_native.get("runs", 0),
        "native_prefill_replay_reason": prefill_native.get("reason"),
        "native_decode_replay_runs": decode_native.get("runs", 0),
        "npu_finish_clock_groups": clock_groups,
        "measured_trial_temperature_min_c": min(measured_temperatures),
        "measured_trial_temperature_max_c": max(measured_temperatures),
        "temperature_samples": samples,
        "path_evidence": {
            key: report.get(key)
            for key in (
                "device", "precision", "prefill", "decode", "share_projection_weights",
                "cpu_recurrent_scan", "fused_recurrent_projections", "fused_swiglu_ffn",
                "fused_swiglu_decode", "cpu_exact_rows", "cpu_erf_gelu", "cpu_head_gate",
                "cpu_q8_embedding", "cpu_q8_side_embedding", "fused_gelu_sdk_prefill",
                "scalar_decode_loop", "tile_columns", "chunk_size", "placement",
            )
        },
        "timing_validity": "PASS",
    }


def append_event(path: Path, event: dict) -> None:
    event["timestamp_unix"] = time.time()
    with path.open("a") as output:
        output.write(json.dumps(event, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def run_one(
    key: str,
    out: Path,
    start_wait_log: Path | None,
    max_wait_seconds: float,
    start_interval_seconds: float,
) -> int:
    if out.exists():
        raise Blocked(f"output directory already exists: {out}")
    source_validation_started = time.monotonic()
    config = load_config(key, out)
    source_validation_elapsed = time.monotonic() - source_validation_started
    gate_log = start_wait_log or out.with_name(f"{out.name}.start_gate.jsonl")
    gate_log.parent.mkdir(parents=True, exist_ok=True)
    gate_started = time.monotonic()
    gate_deadline = gate_started + max_wait_seconds
    lock_fd = -1
    owns_lock = False
    process = None
    ready_gate: Path | None = None
    run_started_ns: int | None = None
    warmup_allowed = False
    receipt: dict = {}
    try:
        preflight: dict | None = None
        post_lock: dict | None = None
        gate_passed_at: float | None = None
        while time.monotonic() < gate_deadline:
            snapshot = hardware_snapshot()
            temp = snapshot["temperature_millic"]
            append_event(gate_log, {
                "event": "preflight_sample",
                "temperature_millic": temp,
                "hw_lock_present": snapshot["hw_lock_present"],
                "drm_owner_present": snapshot["drm_owner_present"],
                "npu_min_hz": snapshot["npu_min_hz"],
                "npu_cur_hz": snapshot["npu_cur_hz"],
                "ddr_cur_hz": snapshot["ddr_cur_hz"],
            })
            if not MIN_START_MILLIC <= temp <= MAX_START_MILLIC:
                time.sleep(start_interval_seconds)
                continue
            validate_start(snapshot)
            preflight = snapshot
            lock_fd = os.open(HW_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            owns_lock = True
            os.write(lock_fd, json.dumps({"pid": os.getpid(), "model": key}).encode())
            os.fsync(lock_fd)
            os.close(lock_fd)
            lock_fd = -1
            post_lock = hardware_snapshot()
            if not MIN_START_MILLIC <= post_lock["temperature_millic"] <= MAX_START_MILLIC:
                append_event(gate_log, {
                    "event": "post_lock_temperature_race",
                    "temperature_millic": post_lock["temperature_millic"],
                })
                HW_LOCK.unlink()
                owns_lock = False
                time.sleep(start_interval_seconds)
                continue
            validate_start(post_lock, allow_own_lock=True)
            gate_passed_at = time.monotonic()
            break
        else:
            raise Blocked(f"no valid 52–55 °C start gate within {max_wait_seconds:g} seconds")

        assert preflight is not None and post_lock is not None and gate_passed_at is not None
        receipt = {
            "kind": "source-pinned warmed 1+3 timing retime",
            "model": config["model"],
            "execution_status": "STARTING",
            "timing_validity": "PENDING",
            "protocol": {
                "warmups": WARMUPS,
                "measured_runs": RUNS,
                "prompt_tokens": 128,
                "output_tokens": 64,
                "context": 512,
                "device": "CPU+ROCKCHIP",
                "affinity": "4-7",
            },
            "ready_temperature_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
            "start_wait_log": str(gate_log),
            "source_validation_elapsed_seconds": round(source_validation_elapsed, 3),
            "start_gate_wait_seconds": round(gate_passed_at - gate_started, 3),
            "preflight_hardware": preflight,
            "hardware_after_lock": post_lock,
            "runner": config["runner"],
            "backend": config["backend"],
            "backend_source_count": config["backend_source_count"],
            "backend_source_map": config["backend_source_map"],
            "app_revision": config["app_revision"],
            "app_profile": config["app_profile"],
            "checkpoint": config["checkpoint"],
            "input": {key: value for key, value in config["input"].items() if key != "token_ids"},
            "selected_recipe": config["selected_recipe"],
            "command": config["command"],
            "effective_environment": config["recorded_environment"],
        }
        out.mkdir(parents=True)
        archived_runner = out / "runner_source_used.py"
        shutil.copyfile(Path(__file__).resolve(), archived_runner)
        ready_gate = out / "ready_gate.fifo"
        os.mkfifo(ready_gate, 0o600)
        environment = dict(config["environment"])
        temperature_path = out / "trial_temperatures.jsonl"
        environment["ONPU_TIMING_TEMPERATURES_JSONL"] = str(temperature_path)
        environment["ONPU_TIMING_READY_GATE_FIFO"] = str(ready_gate)
        receipt.update(
            runner_source_copy_sha256=sha256(archived_runner),
            command=config["command"],
            effective_environment={
                key: value
                for key, value in environment.items()
                if key.startswith("ONPU_")
                or key in {
                    "DEV", "HALF", "JIT", "CAPTURING", "PARALLEL", "DEBUG",
                    "PROFILE", "VIZ", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                }
            },
            temperature_observer={
                "path": str(temperature_path),
                "source": str(CHILD),
                "source_sha256": CHILD_SHA256,
                "samples": "benchmark trial begin/end callback; includes warm-up and each measured trial",
            },
            ready_gate={
                "path": str(ready_gate),
                "timeout_seconds": READY_GATE_TIMEOUT_SECONDS,
                "accepted_temperature_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
                "position": "after ready path, immediately before warm-up",
            },
        )
        receipt["execution_status"] = "RUNNING"
        write_json(out / "receipt.json", receipt)
        run_started_ns = time.monotonic_ns()
        log = out / "benchmark.log"
        ready_event = None
        ready_error = None
        with log.open("x", encoding="utf-8") as output:
            process = subprocess.Popen(
                config["command"],
                cwd=BACKEND,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "ready":
                    continue
                ready_event = event
                try:
                    receipt["ready_path_validation"] = validate_ready_path(
                        key, event, config["selected_recipe"]["profile"]
                    )
                    receipt["ready_temperature_gate"] = wait_for_ready_temperature(out)
                    if receipt["ready_temperature_gate"]["passed"]:
                        gate_value = b"1"
                        warmup_allowed = True
                    else:
                        ready_error = Blocked(receipt["ready_temperature_gate"]["error"])
                        gate_value = b"0"
                except (Blocked, ValueError) as error:
                    ready_error = error
                    receipt["ready_path_validation"] = {
                        "passed": False,
                        "placement": event.get("placement") or {},
                        "error": str(error),
                        "validated_before_warmup": True,
                    }
                    gate_value = b"0"
                if "ready_temperature_gate" in receipt:
                    receipt["ready_temperature_millic"] = receipt["ready_temperature_gate"].get(
                        "last_sample", {}
                    ).get("temperature_millic")
                write_json(out / "receipt.json", receipt)
                with ready_gate.open("wb") as gate:
                    gate.write(gate_value)
                    gate.flush()
                if ready_error is not None:
                    break
            if ready_error is not None:
                for line in process.stdout:
                    output.write(line)
            returncode = process.wait()
            output.flush()
            os.fsync(output.fileno())
        receipt["benchmark_returncode"] = returncode
        receipt["benchmark_log_sha256"] = sha256(log)
        if ready_error is not None:
            raise Blocked(f"ready gate rejected before warm-up: {ready_error}")
        if ready_event is None:
            raise Blocked("benchmark exited without a ready event; no warm-up was allowed")
        if returncode != 0:
            receipt["execution_status"] = "FAILED"
            receipt["error"] = f"benchmark exited {returncode}"
            return 1
        result = validate_report(config, out / "benchmark.json", temperature_path)
        receipt["benchmark_sha256"] = sha256(out / "benchmark.json")
        backend_after = backend_identity()
        from extra.rockchip import benchmark_inference
        sources_after = benchmark_inference.source_snapshot(BACKEND)
        source_identity_after_run = {
            "backend_identity_unchanged": backend_after == config["backend"],
            "backend_source_map_unchanged": sources_after == config["backend_source_map"],
            "app_revision_unchanged": str(git(APP, "rev-parse", "HEAD")) == APP_REVISION,
            "app_profile_hash_unchanged": sha256(PROFILE) == PROFILE_SHA256,
            "runner_hash_unchanged": sha256(Path(__file__).resolve()) == config["runner"]["sha256"],
            "child_hash_unchanged": sha256(CHILD) == CHILD_SHA256,
        }
        receipt["source_identity_after_run"] = source_identity_after_run
        if not all(source_identity_after_run.values()):
            raise Blocked(f"source identity changed during timing: {source_identity_after_run}")
        receipt["execution_status"] = "COMPLETE"
        receipt["timing_validity"] = result["timing_validity"]
        receipt["metrics"] = result["metrics"]
        receipt["path_evidence"] = result["path_evidence"]
        receipt["native_prefill_replay_runs"] = result["native_prefill_replay_runs"]
        receipt["native_prefill_replay_reason"] = result["native_prefill_replay_reason"]
        receipt["native_decode_replay_runs"] = result["native_decode_replay_runs"]
        receipt["npu_finish_clock_groups"] = result["npu_finish_clock_groups"]
        receipt["measured_trial_temperature_min_c"] = result["measured_trial_temperature_min_c"]
        receipt["measured_trial_temperature_max_c"] = result["measured_trial_temperature_max_c"]
        write_json(out / "timing_review.json", {
            "model": config["model"],
            "metrics": result["metrics"],
            "timing_validity": result["timing_validity"],
            "native_prefill_replay_runs": result["native_prefill_replay_runs"],
            "native_decode_replay_runs": result["native_decode_replay_runs"],
            "npu_finish_clock_groups": result["npu_finish_clock_groups"],
            "measured_trial_temperature_min_c": result["measured_trial_temperature_min_c"],
            "measured_trial_temperature_max_c": result["measured_trial_temperature_max_c"],
            "derivation": {
                "ttft_ms": "benchmark first-emitted-token time; a prefill proxy, not isolated prefill",
                "decode_tps": "backend benchmark's 63 generated-token intervals / decode interval",
                "decode_interval_ms": "63,000 / decode_tps",
                "ttft_max_min_ratio": "max(measured first-token ms) / min(measured first-token ms)",
            },
            "quality_assessed": False,
            "caveat": "first-token latency is a proxy, not isolated prefill; no TTFT stability threshold is defined",
        })
        return 0
    except (Blocked, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        receipt["execution_status"] = "FAILED" if warmup_allowed else "BLOCKED"
        receipt["timing_validity"] = "FAIL" if warmup_allowed else "NOT_RUN"
        receipt["error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if lock_fd >= 0:
            os.close(lock_fd)
        if owns_lock and HW_LOCK.exists():
            HW_LOCK.unlink()
            owns_lock = False
        if ready_gate is not None:
            ready_gate.unlink(missing_ok=True)
        try:
            receipt["cleanup_hardware"] = hardware_snapshot()
            receipt["hw_lock_present_after_cleanup"] = HW_LOCK.exists()
        except (Blocked, OSError, ValueError) as error:
            receipt["cleanup_snapshot_error"] = f"{type(error).__name__}: {error}"
        if run_started_ns is not None:
            receipt["benchmark_outer_elapsed_seconds"] = round((time.monotonic_ns() - run_started_ns) / 1e9, 3)
        if out.is_dir():
            write_json(out / "receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--start-wait-log", type=Path)
    parser.add_argument("--max-wait-seconds", type=float, default=7200)
    parser.add_argument("--start-interval-seconds", type=float, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    out = args.out or OUTPUT_ROOT / f"candidate-retime-20261117-{args.model}"
    try:
        if args.validate_only:
            config = load_config(args.model, out)
            print(json.dumps({
                "validation": "PASS",
                "model": config["model"],
                "profile": config["selected_recipe"]["profile"]["prefill_profile"],
                "checkpoint_sha256": config["checkpoint"]["sha256"],
                "input_sha256": config["input"]["sha256"],
                "runner": config["runner"],
                "warmups": WARMUPS,
                "runs": RUNS,
                "ready_temperature_band_millic": [MIN_START_MILLIC, MAX_START_MILLIC],
                "requires_native_prefill_replay": True,
                "command": config["command"],
                "effective_environment": config["recorded_environment"],
                "backend_revision": config["backend"]["revision"],
                "backend_dirty_diff_sha256": config["backend"]["dirty_diff_sha256"],
                "backend_source_count": config["backend_source_count"],
                "hardware_started": False,
                "hw_lock_touched": False,
            }, indent=2))
            return 0
        return run_one(
            args.model, out, args.start_wait_log, args.max_wait_seconds, args.start_interval_seconds
        )
    except (Blocked, OSError, ValueError, KeyError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
