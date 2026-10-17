"""Run the blog's upstream forward checks serially; no raw evidence files are written.

Use --slow for the full POW/SQRT/SIN/EXP2/LOG2 methods and --examples for the
standalone register examples. These checks cover selected methods, not all dtypes
or the complete test_ops.py suite. The blog documents unsupported variants.
"""
import argparse, fcntl, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAST = ("add", "tiny_mul", "mul", "sub", "neg", "div", "maximum", "minimum", "cmp_eq", "cmp_lt", "where",
        "lshift", "lshift_signed", "rshift", "rshift_signed", "trunc", "floor", "ceil", "round",
        "div_int", "div_rounding_mode", "mod", "fmod")
SLOW = ("exp2", "log2", "pow", "sqrt", "sin")


def run(args:list[str], env:dict[str, str]) -> None:
  print("RUN", " ".join(args), flush=True)
  result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True)
  print(result.stdout, end="", flush=True)
  print(result.stderr, end="", file=sys.stderr, flush=True)
  # Some original register examples print FAIL without a nonzero exit code.
  if result.returncode or re.search(r"\bFAIL(?:ED)?\b", result.stdout): raise SystemExit(result.returncode or 1)


def pytest(names:tuple[str, ...], env:dict[str, str]) -> None:
  # This repository's conftest.py installs a 120s exit watchdog. No test cases or
  # assertions are omitted by disabling that hook for slow hardware execution.
  run(["-m", "pytest", "--noconftest", "-n0", "-q", *[f"test/backend/test_ops.py::TestOps::test_{name}" for name in names]], env)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--slow", action="store_true")
  parser.add_argument("--examples", action="store_true")
  args = parser.parse_args()
  env = dict(os.environ, DEV="ROCKCHIP", FORWARD_ONLY="1", DEFAULT_FLOAT="HALF", NOOPT="1",
             PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH", ""))
  env.pop("SUM_DTYPE", None)
  with open("/tmp/rk3588_npu_submit.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    pytest(FAST, env)
    if args.slow:
      for name in SLOW: pytest((name,), env)
    # The post uses half accumulation for this one optional MUL+ADD example.
    pytest(("mulacc_with_zero_strides",), env | {"SUM_DTYPE":"HALF"})
    pytest(("bitcast",), env | {"DEFAULT_FLOAT":"FLOAT"})
    pytest(("small_gemm", "small_gemm_padded", "gemm"), env | {"NOOPT":"0"})
    run(["extra/rockchip/probe_lut.py", "sigmoid", "SiLU", "sigmoid", "SiLU", "sigmoid", "SiLU"], env)
    if args.examples:
      for example in (("simple_add.py",), ("elementwise.py", "ADD"), ("elementwise.py", "CAST_BOOL_HALF"),
                      ("elementwise.py", "CAST_HALF_BOOL"), ("pooling.py", "all"), ("conv_simple.py", "--test-int8"),
                      ("conv_tiles.py", "--count", "1")):
        run(["extra/rockchip/examples/"+example[0], *example[1:]], env)
