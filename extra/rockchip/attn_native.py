"""fp32 causal GQA attention for the CPU host, replacing `scaled_dot_product_attention`.

The generated kernel costs 252 ms per block at (B=1,T=128,H=16,KVH=4,D=256) -- 1.06 GFLOP/s.
`attn.c` does the same work in 17.24 ms (7.85 GFLOP/s of causal work, half the matrix being
masked), a 14.6x improvement, and matches an fp64 reference to 2.1e-05.

Two interface facts make this work, both learned the hard way:

1. The KV side is the *whole concrete cache buffer* `(B,2,KVH,CAP,D)`, not the symbolic slice
   `cache[..., :start_pos+T, :]`. tinygrad cannot build a placeholder for a symbolic extent
   ("no placeholder-like on symbolic shape"), and the cache buffer is concrete.
2. `start_pos` arrives as a **float** scalar buffer, converted to `int` inside the kernel. An
   int32 placeholder is the failure mode `scan_native.py` records as a kernel that silently never
   executes; float32 is exact for any position below 2^24.

Verified geometry only: H=16, KVH=4, D=256. Anything else falls back to tinygrad's
`scaled_dot_product_attention`, because the kernel bakes the shape into the source.

CPU kernel ABI (same as the scan): one packed argument buffer, entry the only non-static function,
`#include <stddef.h>` hoisted above it.
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
H_, KVH_, D_ = 16, 4, 256

_PROGRAM: dict[tuple[int, int], UOp] = {}


def _renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def supported(tokens: int, heads: int, kv_heads: int, dim: int, capacity: int) -> bool:
  return (heads, kv_heads, dim) == (H_, KVH_, D_) and isinstance(tokens, int) and isinstance(capacity, int) \
    and 0 < tokens <= capacity <= 16384


def _source(tokens: int, capacity: int) -> str:
  body = (_HERE / "attn.c").read_text()
  if "#ifdef PROBE" in body:
    body = body[:body.index("#ifdef PROBE")]
  body = body.replace("#define TOK 128", f"#define TOK {tokens}")
  body = body.replace("#define CAP 512", f"#define CAP {capacity}")
  return body


def _build(tokens: int, capacity: int) -> UOp:
  body = _source(tokens, capacity)
  out = UOp.placeholder(((H_ * tokens * D_),), dtypes.float, 0)
  q = UOp.placeholder(((H_ * tokens * D_),), dtypes.float, 1)
  kv = UOp.placeholder(((2 * KVH_ * capacity * D_),), dtypes.float, 2)
  pos = UOp.placeholder((1,), dtypes.float, 3)
  renderer = _renderer()
  sink = UOp.sink(out, q, kv, pos, arg=KernelInfo(f"rockchip_attn_{tokens}_{capacity}"))
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2, 3))
  return UOp(Ops.PROGRAM, arg=info,
             src=(sink, UOp(Ops.LINEAR, src=(out, q, kv, pos)),
                  UOp(Ops.SOURCE, arg=body),
                  UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(body))))


def _program(tokens: int, capacity: int) -> UOp:
  if (prog := _PROGRAM.get((tokens, capacity))) is None:
    _PROGRAM[(tokens, capacity)] = prog = _build(tokens, capacity)
  return prog


def attention(q: Tensor, kv: Tensor, pos: int | UOp) -> Tensor:
  """q (1,H,T,D); kv (1,2,KVH,capacity,D) the full cache buffer; pos the absolute position of q[...,0,:].

  Returns (1,H,T,D)."""
  tokens, capacity = q.shape[2], kv.shape[3]
  if q.shape[0] != 1 or kv.shape[0] != 1:
    raise ValueError(f"attention kernel is built for B=1, got q={q.shape} kv={kv.shape}")
  if not supported(tokens, q.shape[1], kv.shape[2], q.shape[3], capacity):
    raise ValueError(f"unsupported attention geometry q={q.shape} kv={kv.shape}")
  # float, not int32: see the module docstring. An int constant goes straight to a buffer; a
  # symbolic start_pos stays in the graph as a 1-element cast.
  pos_buf = (Tensor([float(pos)], dtype=dtypes.float) if isinstance(pos, int)
             else Tensor.stack(Tensor(pos)).cast(dtypes.float))
  out = Tensor.empty(H_ * tokens * D_, device="CPU", dtype=dtypes.float)
  res = out.custom_kernel(q.contiguous().reshape(-1), kv.float().contiguous().reshape(-1), pos_buf,
                          fxn=lambda o, a, b, c: _program(tokens, capacity))[0]
  return res.reshape(1, H_, tokens, D_)
