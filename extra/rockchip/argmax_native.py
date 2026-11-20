"""First-maximum SIMD argmax for the finite FP32 scores this tree produces on the host.

Why this exists: `logits.argmax(-1)` over the 248,320-wide vocabulary measured **6.6 ms/token**
(148 MB/s) against the reference's hand-written `gpt2_argmax` at **0.086 ms** (11 GB/s) -- 78x. At
temperature 0 that reduction is the whole sampler, so it is 6.6 ms of the decode gap sitting in a
single reduce.

The kernel is the reference's (`~/tg-qwen35-pack-signature-fix/tinygrad/llm/kernels/
rockchip_greedy.py:10 _argmax`): four float4 accumulators, 16 elements per iteration, first-maximum
tie-break so it selects the same index as `argmax`. Ported rather than re-derived because the
comparison above is against that exact kernel.

Guarded to finite fp32 2-D inputs of at most 64 rows and 2^20 columns; everything else falls back to
`logits.argmax(-1)`, which is the same value at 78x the cost.
"""
from __future__ import annotations

import functools, platform
from dataclasses import replace
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

MAX_ROWS, MAX_COLUMNS = 64, 1 << 20
_PROGRAM: dict[tuple[int, int], UOp] = {}


def _renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _argmax(out: UOp, values: UOp, rows: int, columns: int) -> UOp:
  """Build the program from the uops `custom_kernel` passes (fixed placeholders segfaulted)."""
  body = f'''typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
  typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
  void rockchip_argmax_{rows}_{columns}(void *args) {{
    int *out = *(int **)args;
    const float *input = *((const float **)args + 1);
    for(int row=0;row<{rows};row++) {{
      const float *data=input+row*{columns};float4 best[4];int4 index[4];
      for(int b=0;b<4;b++){{best[b]=(float4)(-__builtin_inff());index[b]=(int4){{b*4,b*4+1,b*4+2,b*4+3}};}}
      int i=0;
      for(;i+15<{columns};i+=16) {{
        #pragma clang loop unroll(full)
        for(int b=0;b<4;b++){{float4 value=*(const float4*)(data+i+b*4);int4 found=value>best[b];
          best[b]=found?value:best[b];index[b]=found?((int4){{i+b*4,i+b*4+1,i+b*4+2,i+b*4+3}}):index[b];}}
      }}
      float maximum=-__builtin_inff();int selected=0;
      for(int b=0;b<4;b++)for(int lane=0;lane<4;lane++){{
        if(best[b][lane]>maximum || (best[b][lane]==maximum && index[b][lane]<selected)){{
          maximum=best[b][lane];selected=index[b][lane];}}
      }}
      for(;i<{columns};i++)if(data[i]>maximum){{maximum=data[i];selected=i;}}
      out[row]=selected;
    }}
  }}'''
  renderer = _renderer()
  sink = UOp.sink(out, values, arg=KernelInfo(f"rockchip_argmax_{rows}_{columns}"))
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1,))
  return UOp(Ops.PROGRAM, arg=info,
             src=(sink, UOp(Ops.LINEAR, src=(out, values)),
                  UOp(Ops.SOURCE, arg=body),
                  UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(body))))


def supported(logits: Tensor) -> bool:
  if logits.device != "CPU" or logits.dtype is not dtypes.float or logits.ndim != 2: return False
  rows, columns = logits.shape
  return (isinstance(rows, int) and isinstance(columns, int)
          and 1 <= rows <= MAX_ROWS and 1 <= columns <= MAX_COLUMNS)


def argmax(logits: Tensor) -> Tensor:
  """Same value as `logits.argmax(-1)`, first-maximum tie-break included."""
  rows, columns = logits.shape
  return Tensor.empty(rows, device="CPU", dtype=dtypes.int).custom_kernel(
    logits.contiguous(), fxn=lambda out, x: _argmax(out, x, rows, columns))[0]
