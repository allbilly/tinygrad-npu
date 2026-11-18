#!/usr/bin/env python3
"""Dump every CPU kernel this tree compiles, one C file per kernel name.

Both this checkout and the reference compile CPU kernels through the same ClangCompiler
with the same flags, and both hand clang a `void <name>(void *args)` entry point:

  clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident \
        --target=arm64-none-unknown-elf -ffixed-x18 -mcpu=native - -o -

That common ABI is what makes a kernel-level comparison mechanical: dump ours here, read
theirs at the cited `file:line` in docs/rknpu/reference-qwen35-4b.md section 6, and diff
the arithmetic rather than the scaffolding.

The hook wraps `ClangCompiler.compile`, so it sees exactly the source text that clang
sees, including hand-written `custom_kernel` sources (the fused INT8 Linear, the scan)
and generated elementwise kernels.

Usage, from the repo root:
  .venv/bin/python extra/rockchip/dump_ckernels.py --out docs/rknpu/results/cdump -- --quick
  .venv/bin/python extra/rockchip/dump_ckernels.py --out docs/rknpu/results/cdump \
      -- --ckpt <gguf> --name qwen35-4b --out /tmp/ignore --npu --warmups 0 --runs 1

Any arguments after `--` go to bench_llm.py verbatim; `--quick` is a 16-token prompt with
no warm-up, which is enough to enumerate the kernels without paying for a full prefill.
"""
from __future__ import annotations

import argparse, os, pathlib, re, runpy, sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, required=True)
  p.add_argument("--bench-args", nargs=argparse.REMAINDER, default=[])
  args = p.parse_args()
  bench_args = args.bench_args[1:] if args.bench_args[:1] == ["--"] else args.bench_args
  args.out.mkdir(parents=True, exist_ok=True)

  from tinygrad.renderer.cstyle import ClangCompiler
  original = ClangCompiler.compile
  seen: dict[str, int] = {}

  def compile_and_dump(self, src: str) -> bytes:
    m = re.search(r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", src)
    name = m.group(1) if m else "unnamed"
    index = seen.get(name, 0)
    seen[name] = index + 1
    (args.out / f"{name}{'' if index == 0 else f'_{index}'}.c").write_text(src)
    return original(self, src)

  ClangCompiler.compile = compile_and_dump
  os.environ.setdefault("DEV", "CPU:CLANG")

  sys.argv = [str(HERE / "bench_llm.py"), *bench_args]
  try:
    runpy.run_path(str(HERE / "bench_llm.py"), run_name="__main__")
  except SystemExit as exc:
    if exc.code not in (0, None): return int(exc.code)

  total = sum(seen.values())
  print(f"\ndumped {total} kernel compiles, {len(seen)} distinct names, to {args.out}")
  for name, count in sorted(seen.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {count:4d}  {name}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
