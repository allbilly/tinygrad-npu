#!/usr/bin/env python3
"""Split selected tinygrad CPU kernels across four threads at the compiler boundary.

Why here: every CPU kernel in this tree is compiled by `ClangCompiler.compile_cached`, and
the compiled object is what `CPUProgram` loads. Rewriting the source there means no graph
change, no `model.py` change, and no new op -- the kernel keeps its exact ABI
(`void name(unsigned char *args)`) and its exact arithmetic.

What it does, for a kernel on the allowlist whose chosen loop has a literal bound:

    void NAME(unsigned char* restrict a) {          static void NAME_part(unsigned char* restrict a, int begin, int end) {
      <decls>                                          <decls>
      for (int L = 0; L < N; L++) {        -->        for (int L = begin; L < end; L++) {
        <body>                                           <body>
      }                                                }
    }                                                }

                                                     // plus a 4-way pthread wrapper that
                                                     // calls NAME_part on N*i/4 .. N*(i+1)/4

Correctness argument: the split is a partition of the loop's iteration space, so it is exact
provided the loop carries no state between iterations. `THREAD_SPLIT` therefore lists the
loop **index** to split per kernel, and only indices that select disjoint output elements are
listed -- reductions (`buf0` accumulators) are never split. See
`docs/rknpu/results/threadsplit/loop-classification.md`.

Caveat on caching: `compile_cached` keys its disk cache on the source string, so the transform
runs on the *input* side of that call. Toggling an entry therefore changes the cache key
automatically; toggling `THREAD_SPLIT_THREADS` does too, because the thread count is emitted
into the source.

Usage from a benchmark script:
    import thread_split; thread_split.install()
"""
from __future__ import annotations

import re

# kernel name -> occurrence index of the `for (int X = 0; X < N; X++)` loop to split
# (0 = outermost). Populated only with loops whose iterations write disjoint output, from
# `docs/rknpu/results/threadsplit/loop-classification.md`.
THREAD_SPLIT: dict[str, int] = {
  # occurrence of the `for (int X = 0; X < N; X++)` loop to split (0 = first in source order).
  # Verified against the emitted source in docs/rknpu/results/prof-merged.log:
  "E_2_16_32_32_4_4": 2,   # nest Lidx0_0(2), Lidx0_1(16), Lidx1(32) <- split, Lidx2(32)
  "E_1024_32_4_4": 0,      # Lidx0(1024) selects disjoint outputs
  "E_32_32_128_4": 0,      # Lidx0(32), writes indexed by Lidx0<<14
  "r_32_16_4_32_4": 0,     # Lidx1(32); buf0[4] accumulator hoists into <name>_part, so no race
  "E_393216_3": 0,         # 393216 iterations, disjoint 3-element groups; bitwise A/B passed
}

THREAD_SPLIT_THREADS = 4
MIN_ITERS = 8

_ENTRY = re.compile(r"void\s+(?P<name>[A-Za-z_]\w*)\s*\(\s*unsigned char\s*\*\s*restrict\s+(?P<arg>[A-Za-z_]\w*)\s*\)\s*\{")
_FOR = re.compile(r"for\s*\(\s*int\s+(?P<v>[A-Za-z_]\w*)\s*=\s*0\s*;\s*(?P=v)\s*<\s*(?P<bound>[^;]+?)\s*;\s*(?P=v)\s*\+\+\s*\)")

_PRELUDE = """
typedef void rk_fn_t(void **, int, int);
typedef int rk_dispatch_t(rk_fn_t *, void **, int);
"""


def pool_address() -> int:
  """Address of rk_run inside librkpool.so.

  The address is baked into the emitted source as an integer, exactly as the reference embeds
  its dispatcher into `rockchip_norm_quantize`. It must therefore be resolved before compile,
  and it changes per process, so the source (and hence the disk cache key) changes too.
  """
  import ctypes, pathlib as _p
  so = _p.Path(__file__).resolve().parent / "librkpool.so"
  if not so.exists():
    raise RuntimeError(f"{so} missing: build it with `clang -shared -fPIC -O2 -o librkpool.so rkpool.c -lpthread`")
  lib = ctypes.CDLL(str(so))
  return ctypes.cast(lib.rk_run, ctypes.c_void_p).value


def _wrapper(name: str, arg: str, var: str, n: int, threads: int, address: int) -> str:
  """Entry point: unpack nothing, dispatch four groups through the pool.

  `group` is an argument, not file-scope state: data at file scope in a jit_loader'd kernel
  resolves to address 0 on this tree (measured), which is why the reference's `_part(...,
  group)` signature works and a static pool cannot.
  """
  return f"""
/* --- {threads}-way split over {var} of {n} iterations, dispatched through librkpool --- */
/* CPUProgram jumps to offset 0, so the ENTRY must be the first definition emitted.
   Putting _group first makes every call execute _group with a garbage group index: it runs one
   quarter of the work and never reaches the pool. That is measured, not hypothetical. */
static void {name}_part(unsigned char *restrict {arg}, int group);
static void {name}_group(void **ctx, int unused, int group);
void {name}(unsigned char *restrict {arg}) {{
  void *ctx[1];
  ctx[0] = (void *){arg};
  /* The target must be loaded from memory. A materialised constant call never reaches the
     pool through jit_loader (measured: returns nonzero, fallback runs, no speedup), while this
     volatile round-trip does (measured: rc=0 and the workers write the output). */
  volatile unsigned long target[2];
  target[0] = {address}ULL >> 32; target[1] = {address}ULL & 0xFFFFFFFFUL;
  unsigned long addr = (target[0] << 32) | target[1];
  rk_dispatch_t *fp = (rk_dispatch_t *)addr;
  if (fp({name}_group, ctx, {threads}) != 0)
    for (int g = 0; g < {threads}; g++) {name}_part({arg}, g);
}}
static void {name}_group(void **ctx, int unused, int group) {{
  (void)unused;
  {name}_part((unsigned char *)ctx[0], group);
}}
"""

def _match_brace(src: str, open_idx: int) -> int:
  depth = 0
  for i in range(open_idx, len(src)):
    if src[i] == "{": depth += 1
    elif src[i] == "}":
      depth -= 1
      if depth == 0: return i
  raise ValueError("unbalanced braces")


def split_source(src: str, name: str, occurrence: int, threads: int) -> str | None:
  """Return the transformed source, or None when this source is not splittable."""
  m = _ENTRY.search(src)
  if m is None or m.group("name") != name: return None
  body_open = m.end() - 1
  body_close = _match_brace(src, body_open)
  body = src[body_open + 1:body_close]

  loops = list(_FOR.finditer(body))
  if len(loops) <= occurrence: return None
  loop = loops[occurrence]
  bound = loop.group("bound").strip()
  if not bound.isdigit() or int(bound) < MIN_ITERS: return None
  n, var = int(bound), loop.group("v")

  part_body = body[:loop.start()] + \
    f"for (int {var} = {n} * group / {threads}; {var} < {n} * (group + 1) / {threads}; {var}++)" + body[loop.end():]
  arg = m.group("arg")
  preamble, rest = src[:m.start()], src[body_close + 1:]
  wrapper = _wrapper(name, arg, var, n, threads, pool_address()) + \
    f"static void {name}_part(unsigned char *restrict {arg}, int group) {{{part_body}\n}}\n"
  return preamble + _PRELUDE + wrapper + rest


HITS: dict[str, int] = {}
CALLS: dict[str, int] = {}
DUMP_DIR: pathlib.Path | None = None


def transform(src: str) -> str:
  m = _ENTRY.search(src)
  key = m.group("name") if m else (re.search(r"void\s+(\w+)\s*\(", src).group(1)
                                   if re.search(r"void\s+(\w+)\s*\(", src) else "<none>")
  CALLS[key] = CALLS.get(key, 0) + 1
  if DUMP_DIR is not None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    (DUMP_DIR / f"{key}.c").write_text(src)
  for name, occurrence in THREAD_SPLIT.items():
    if name not in src: continue
    out = split_source(src, name, occurrence, THREAD_SPLIT_THREADS)
    if out is not None:
      HITS[name] = HITS.get(name, 0) + 1
      return out
  return src


def install() -> None:
  """Patch ClangCompiler.compile_cached so the transform is part of the cache key.

  The emitted code calls `rk_run` as an ordinary undefined symbol. `jit_loader` resolves those
  against the libraries already loaded in the process, so the pool must be in the process with
  RTLD_GLOBAL *before* the kernel is loaded. That removes the baked address entirely: a baked
  address makes the source (and therefore the disk-cache key) process-specific, and a cache hit
  from another process would jump to an address that means something else here.
  """
  import ctypes, pathlib as _p
  so = _p.Path(__file__).resolve().parent / "librkpool.so"
  if not so.exists():
    raise RuntimeError(f"{so} missing: clang -shared -fPIC -O2 -o librkpool.so rkpool.c -lpthread")
  # jit_loader resolves undefined symbols through tinygrad's own DLL registry, so the pool must
  # be registered there -- a plain ctypes.CDLL with RTLD_GLOBAL is invisible to it.
  from tinygrad.runtime.support.c import DLL
  if "rkpool" not in DLL._loaded_:
    DLL._loaded_["rkpool"] = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
  from tinygrad.runtime.support.compiler_cpu import ClangCompiler
  if getattr(ClangCompiler, "_rk_split_installed", False): return
  original = ClangCompiler.compile_cached

  def compile_cached(self, src: str) -> bytes:
    return original(self, transform(src))

  ClangCompiler.compile_cached = compile_cached
  ClangCompiler._rk_split_installed = True

  import atexit
  def report() -> None:
    top = sorted(CALLS.items(), key=lambda kv: -kv[1])[:12]
    print(f"\n[thread_split] compile_cached calls {sum(CALLS.values())}, names {top}", flush=True)
    print(f"[thread_split] transforms applied: {HITS or 'NONE'}", flush=True)
  atexit.register(report)


if __name__ == "__main__":
  import json, pathlib, sys
  log = pathlib.Path(__file__).resolve().parents[2] / "docs/rknpu/results/prof-merged.log"
  want = sys.argv[1] if len(sys.argv) > 1 else "E_393216_3"
  for line in log.read_text().splitlines():
    if not line.strip() or not line.lstrip().startswith("{"): continue
    try: ev = json.loads(line)
    except json.JSONDecodeError: continue
    if ev.get("name") == want:
      out = split_source(ev["head"], want, THREAD_SPLIT.get(want, 0), THREAD_SPLIT_THREADS)
      print(out if out else "not splittable")
      break
