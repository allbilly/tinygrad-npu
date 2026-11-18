"""Vectorised sigmoid, bit-identical to tinygrad's generated kernel.

**MEASURED AND REJECTED -- not called.** In-model it made prefill *slower*: 2,735 -> 2,810 ms
(+2.7%), even though it is 1.57x faster in isolation (7.76 -> 4.95 ms at 1,179,648 elements).
Reason: tinygrad fuses the sigmoid into the surrounding `gate * sigmoid * (up + up_bias)` chain (that
is what `E_2_16_32_32_4_4` is -- the same activation fused with dequantisation), so replacing it with
a standalone kernel adds a full extra read+write pass over 9.4 MB per layer instead of saving work.
Kept because the bit-identical C is correct and would pay if the op were ever *not* fused.

`gated_activation` (`tinygrad/llm/model.py:47`) calls `.sigmoid()` on `(1, 128, 9216)`, which tinygrad
emits as the scalar kernel `E_589824_2` (flat 589,824-element loop, no float4) at 152 M elem/s --
0.21 s of a 2.57 s CPU-side prefill, 64 calls.

`sigmoid.c` performs the *same* arithmetic, in the same association order, per lane of a float4, so
results are bit-identical by construction. Proved over 20 cases -- NaN, +-inf, |x| up to 200, exact
half-integer ties, shapes (128,9216)/(1,9216)/(32,256)/(4,2560)/(64,128) -- every element identical
to the last bit. Measured 1.57x faster (7.76 -> 4.95 ms at 1,179,648 elements).

Only used when the last dim is 9216 (the FFN width) and the dtype is float; anything else returns
None so the caller falls back to `.sigmoid()`.
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
_PROGRAM: dict[int, UOp] = {}


def _renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _build(n: int) -> UOp:
  body = (_HERE / "sigmoid.c").read_text().replace("N_ELEM", str(n))
  out = UOp.placeholder((n,), dtypes.float, 0)
  inp = UOp.placeholder((n,), dtypes.float, 1)
  renderer = _renderer()
  sink = UOp.sink(out, inp, arg=KernelInfo(f"rk_sigmoid_{n}"))
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1,))
  return UOp(Ops.PROGRAM, arg=info,
             src=(sink, UOp(Ops.LINEAR, src=(out, inp)),
                  UOp(Ops.SOURCE, arg=body),
                  UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(body))))


def _program(n: int) -> UOp:
  if (prog := _PROGRAM.get(n)) is None:
    _PROGRAM[n] = prog = _build(n)
  return prog


def sigmoid(x: Tensor) -> Tensor | None:
  """Bit-identical vectorised sigmoid, or None when this geometry is not the FFN width."""
  if x.dtype != dtypes.float or x.shape[-1] != 9216:
    return None
  dims = x.shape
  if not all(isinstance(d, int) for d in dims):
    return None
  n = 1
  for d in dims:
    n *= d
  if n % 4:
    return None
  out = Tensor.empty(n, device="CPU", dtype=dtypes.float)
  res = out.custom_kernel(x.contiguous().reshape(-1), fxn=lambda o, a: _program(n))[0]
  return res.reshape(dims)
