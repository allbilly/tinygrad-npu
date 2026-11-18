"""SIMD CPU GDN scan, ported from the reference `rockchip_recurrent.c`. IN USE.

Our `scan_head` (extra/rockchip/scan.py) is scalar over the value rows and measured 3.01 s for the
4B prefill; this port is 4.38x faster per block (128.36 -> 29.32 ms) and is what `tinygrad/llm/model.py`
calls when `GDN_SCAN_SIMD` is set. Wired 2026-11-18; the end-to-end effect was 7,950 -> 5,680 ms.

`start` is baked as **1**, not 0. The reference kernel zeroes the state when `start==0`
(`columns[c*V+r] = start ? s[r*K+c] : 0`), so `start=0` silently ignores the supplied initial state.
Getting that wrong made the port look numerically broken while it was actually correct.

Builder layout follows `_build_scan_program` in scan.py: one output buffer, six inputs, `outs=(0,)`,
`ins=(1..6)`. `start` and `rows` stay baked constants -- an int32 placeholder is a kernel that
silently never executes.

Layouts: q, k (B,H,T,K)  v (B,H,T,V)  beta (B,H,T)  alpha (B,H,T,1|K)  state (B,H,V,K)
Returns (out (B,T,H,V), new_state (B,H,V,K)). The kernel reads and writes `state` in place, so it is
given a private copy and that copy is returned.

Verified against an fp64 replay in a fresh process per geometry (`extra/rockchip/scan_case.py`):
(1,1,1,128,128) 2.588e-07, (1,1,4,128,128) 4.479e-07, (1,32,128,128,128) 4.209e-07.
"""
from __future__ import annotations

import platform
from dataclasses import replace
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

_HERE = Path(__file__).resolve().parent


def _renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _build(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> UOp:
  from thread_split import pool_address
  body = (_HERE / "recurrent.c").read_text() \
    .replace("int cpu_gated_delta_scan(", "static int cpu_gated_delta_scan(") \
    .replace("int cpu_gated_delta_scan_pooled(", "static int cpu_gated_delta_scan_pooled(")
  # CPUProgram jumps to offset 0, so the entry must be the first function; the body starts with
  # static helpers, hence the forward declaration and the entry ahead of it.
  _unused_prototype = ("static int cpu_gated_delta_scan(float*,const float*,const float*,const float*,"
               "const float*,const float*,float*,int,int,int,int,int,int,int,int,float*);\n")
  entry = f'''
void rockchip_cpu_gated_delta_scan(void *args) {{
  float *out = *(float**)args;
  const float *q = *((const float**)args + 1);
  const float *k = *((const float**)args + 2);
  const float *v = *((const float**)args + 3);
  const float *beta = *((const float**)args + 4);
  const float *alpha = *((const float**)args + 5);
  float *state = *((const float**)args + 6);
  /* Four independent head groups through the shared pool. Parity: 0 differing bytes over 256 MB
     against the serial call on finite inputs, at 3.37x (docs/rknpu/results/splitcheck). The
     divergence seen with random-byte inputs is confined to non-finite data, so `scan_native`
     asserts once that the real inputs are finite before this path is trusted. */
  int status = 0;
  cpu_gated_delta_scan_pooled(out,q,k,v,beta,alpha,state,1,{tokens},{batch},{heads},{tokens},{key_dim},
                       {value_dim},{alpha_dim},NULL,(scan_parallel_fn){pool_address()},&status);
}}
'''
  # CPUProgram jumps to offset 0, so the entry must be the FIRST function definition in the
  # object. Every helper -- including local_memset -- is therefore forward-declared and defined
  # after the entry; putting any of them first makes the call silently do nothing.
  decls = ("static void *local_memset(void*,int,size_t);\n"
           "static int cpu_gated_delta_scan(float*,const float*,const float*,const float*,"
           "const float*,const float*,float*,int,int,int,int,int,int,int,int,float*);\n"
           "typedef int (*scan_parallel_fn)(void(*)(void**,int,int),void**,int);\n"
           "static int cpu_gated_delta_scan_pooled(float*,const float*,const float*,const float*,"
           "const float*,const float*,float*,int,int,int,int,int,int,int,int,float*,"
           "scan_parallel_fn,int*);\n")
  memset_body = ("static void *local_memset(void *d, int c, size_t n) {\n"
                 "  unsigned char *p = (unsigned char *)d;\n"
                 "  while (n--) *p++ = (unsigned char)c;\n"
                 "  return d;\n}\n")
  body = body.replace("extern void *memset(void *, int, size_t);", "")
  body = body.replace("memset(", "local_memset(")
  # Decode-only: rows==1, capacity==1, start>0 and no history take `direct_row`, which keeps the
  # state in its natural (row, key) layout instead of transposing each head's 128x128 state in
  # and out through `columns`. The reference runs this enabled; our build never defined it, so
  # every decode block paid the transpose. Gated to tokens==1 so prefill is byte-identical.
  define = "#define ONPU_GDN_DIRECT_DECODE 1\n" if tokens == 1 else ""
  source = "#include <stddef.h>\n" + define + decls + entry + body + memset_body
  out = UOp.placeholder((batch * tokens * heads * value_dim,), dtypes.float, 0)
  q = UOp.placeholder((batch * heads * tokens * key_dim,), dtypes.float, 1)
  k = UOp.placeholder((batch * heads * tokens * key_dim,), dtypes.float, 2)
  v = UOp.placeholder((batch * heads * tokens * value_dim,), dtypes.float, 3)
  beta = UOp.placeholder((batch * heads * tokens,), dtypes.float, 4)
  alpha = UOp.placeholder((batch * heads * tokens * alpha_dim,), dtypes.float, 5)
  state = UOp.placeholder((batch * heads * value_dim * key_dim,), dtypes.float, 6)
  renderer = _renderer()
  sink = UOp.sink(out, q, k, v, beta, alpha, state,
                  arg=KernelInfo(f"rockchip_scan_simd_{batch}_{heads}_{tokens}_{value_dim}_{key_dim}_{alpha_dim}"))
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2, 3, 4, 5, 6))
  return UOp(Ops.PROGRAM, arg=info,
             src=(sink, UOp(Ops.LINEAR, src=(out, q, k, v, beta, alpha, state)),
                  UOp(Ops.SOURCE, arg=source),
                  UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))


_OUT_POOL: dict = {}
_FINITE_CHECKED = False
_PROGRAMS: dict[tuple[int, int, int, int, int, int], UOp] = {}


def _program(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> UOp:
  key = (batch, heads, tokens, value_dim, key_dim, alpha_dim)
  if (prog := _PROGRAMS.get(key)) is None:
    _PROGRAMS[key] = prog = _build(*key)
  return prog


def supported(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> bool:
  return (1 <= batch and 1 <= heads and 1 <= tokens <= 256 and 1 <= key_dim <= 256
          and 1 <= value_dim <= 256 and alpha_dim in (1, key_dim))


def gated_delta_scan_simd(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor,
                          state: Tensor) -> tuple[Tensor, Tensor]:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  alpha_dim = 1 if alpha.ndim == 3 else alpha.shape[-1]
  if not supported(batch, heads, tokens, value_dim, key_dim, alpha_dim):
    raise ValueError(f"unsupported scan geometry {(batch, heads, tokens, value_dim, key_dim, alpha_dim)}")
  # No clone: the kernel writes the state in place, and the model stores the result straight back into
  # the same recurrent-state buffer, so a private copy is pure overhead (2 MB per call, 24 calls).
  st = state.contiguous()
  # The pooled entry is bitwise-identical to the serial call on finite inputs and diverges only on
  # non-finite ones (results/splitcheck). The finiteness of real data is checked at the model
  # boundary by the probe, because `t.numpy()` cannot run inside the captured prefill graph.
  # Reuse the output buffer per shape instead of allocating 2 MB on every call.
  _key = (batch, heads, tokens, value_dim)
  out = _OUT_POOL.get(_key)
  if out is None:
    out = _OUT_POOL[_key] = Tensor.empty(batch * heads * tokens * value_dim, device="CPU", dtype=dtypes.float)
  res = out.custom_kernel(q.contiguous().reshape(-1), k.contiguous().reshape(-1),
                          v.contiguous().reshape(-1), beta.contiguous().reshape(-1),
                          alpha.contiguous().reshape(-1), st.reshape(-1),
                          fxn=lambda o, a, b, c, d, e, s: _program(
                            batch, heads, tokens, value_dim, key_dim, alpha_dim))[0]
  return res.reshape(batch, heads, tokens, value_dim).transpose(1, 2), st
