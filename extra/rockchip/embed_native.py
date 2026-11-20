"""One-row gather for the token embedding, replacing a full-vocabulary mask.

`tinygrad/nn/__init__.py:360` implements a lookup as

    arange = Tensor.arange(weight.shape[0])                       # 248,320
    return (arange == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2)

which the CPU backend emits as `E_80_32_64_4`: compare a runtime index against every vocabulary
position, select the weight value or zero, then reduce. Measured **1.60 ms/token** to fetch one row of
2,560 values -- 0.94% of the decode wall, and absent from the reference profile because their lookup
is not O(vocab).

`weight[idx]` does not help (measured 5.37 ms, same scan class) and this tree has no `Ops.GATHER`.
So the row is copied directly in C.

Exactness: the mask path sums one non-zero fp16 value with zeros, in fp16, which is the value itself.
This kernel reads that same fp16 row and upcasts to f32, which the model does immediately afterwards
(`.float()`), so the values are bit-identical.

ABI is this tree's packed-args form: `void name(void *args)` with (out, weight, index) at slots 0-2.
"""
from __future__ import annotations

import platform
from dataclasses import replace

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

MAX_DEPTH = 1 << 16
_PROGRAM: dict[int, UOp] = {}


def _renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _program(out: UOp, weight: UOp, idx: UOp, depth: int) -> UOp:
  """Build once per depth. A fresh program per token recompiles: that cost 640 ms/token."""
  if (prog := _PROGRAM.get(depth)) is None: _PROGRAM[depth] = prog = _build(out, weight, idx, depth)
  return prog


def _build(out: UOp, weight: UOp, idx: UOp, depth: int) -> UOp:
  """Build from the uops custom_kernel passes; fixed placeholders segfaulted for argmax_native."""
  body = f'''typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef __fp16 half4 __attribute__((aligned(8),ext_vector_type(4)));
  void rockchip_embed_{depth}(void *args) {{
    float *out = *(float **)args;
    const __fp16 *weight = *((const __fp16 **)args + 1);
    const int *idx = *((const int **)args + 2);
    const __fp16 *src = weight + (long)(idx[0]) * {depth};
    int i = 0;
    for (; i + 3 < {depth}; i += 4)
      *(float4 *)(out + i) = __builtin_convertvector(*(const half4 *)(src + i), float4);
    for (; i < {depth}; i++) out[i] = (float)src[i];
  }}'''
  renderer = _renderer()
  sink = UOp.sink(out, weight, idx, arg=KernelInfo(f"rockchip_embed_{depth}"))
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2))
  return UOp(Ops.PROGRAM, arg=info,
             src=(sink, UOp(Ops.LINEAR, src=(out, weight, idx)),
                  UOp(Ops.SOURCE, arg=body),
                  UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(body))))


def supported(weight: Tensor, idx: Tensor) -> bool:
  return (weight.device == "CPU" and weight.dtype is dtypes.half and weight.ndim == 2
          and idx.dtype in (dtypes.int, dtypes.int32) and idx.numel() == 1
          and isinstance(weight.shape[0], int) and isinstance(depth := weight.shape[1], int)
          and depth % 4 == 0 and depth <= MAX_DEPTH)


def _placeholder_program(vocab: int, depth: int) -> UOp:
  if (prog := _PROGRAM.get(depth)) is None:
    out = UOp.placeholder((depth,), dtypes.float, 0)
    weight = UOp.placeholder((vocab * depth,), dtypes.half, 1)
    idx = UOp.placeholder((1,), dtypes.int, 2)
    _PROGRAM[depth] = prog = _build(out, weight, idx, depth)
  return prog


def embed(weight: Tensor, idx: Tensor) -> Tensor:
  """(numel(idx), depth) f32 rows. Binds a CALL directly: `custom_kernel` here rebuilds the whole
  forward every token (measured 640-780 ms/token) because it sits at the graph root."""
  depth = weight.shape[1]
  prog = _placeholder_program(weight.shape[0], depth)
  out = Tensor(UOp.alloc((idx.numel() * depth,), dtypes.float, device="CPU", bind_on_realize=True))
  res = Tensor(out.uop.after(prog.call(out.uop, weight.contiguous().reshape(-1).uop,
                                       idx.contiguous().reshape(-1).uop)))
  return res.reshape(idx.numel(), depth)
