"""GDN recurrent scan as one CPU kernel. NOT WIRED IN: it is not bit-exact.

Status: the kernel is structurally correct (bit-exact against numpy at
B=1,H=1,T=2,V=4,K=4, and within 1.6e-04 of numpy at the real 128x128 geometry)
but it does NOT reproduce the tinygrad loop bit-exactly. Measured against a live
run of `model.py:741-750` at the real 4B geometry: max abs delta 6.01 at T=1 and
18.94 at T=2, with the recurrent state diverging too. A 6.0 delta on a single
step is far beyond fp32 reduction-order noise (~1e-5), so the loop is not
computing the naive formula replayed here; the cause was not identified.
`model.py` still runs the Python loop. Do not wire this in until it is exact.

Original intent follows.


The scan is a serial recurrence over tokens, so the NPU cannot take it: the
ablation measured the Python loop at 27.2 s of a 31.9 s prefill (85%), i.e.
1.13 s per block across 24 blocks. Every token costs 3 tinygrad launches
(multiply, reduce, update), so a 128-token prefill pays 9,216 launches per
model, and each one is a full kernel with its own loop nest.

This kernel keeps the recurrent state in a column-major (K, V) scratch buffer so
the K reduction reads contiguously and the V rows vectorise, which is the same
layout choice the reference `rockchip_recurrent.c` makes.

Bit-exactness. The replacement must produce the same fp32 bits as the loop, so
the reduction order is not free. tinygrad's CPU scheduler upcasts the reduce
axis by 4 (`SPLIT axis=3 arg=(4, UPCAST)` for a 128-wide fp32 reduce), which
generates

    acc = acc + (p0*k0 + p1*k1 + p2*k2 + p3*k3)

over groups of four consecutive K, left-to-right inside the group. A plain
serial sum is a different rounding and does not match. `#pragma clang
fp contract(off)` is required for the same reason: clang defaults to
`-ffp-contract=on`, which fuses `t*w` into an fmla and rounds once where the
unfused chain the loop emits rounds twice.
"""
from __future__ import annotations

import platform
from dataclasses import replace

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

_SCAN_SOURCE = """
static void scan_head(float *out, const float *q, const float *k, const float *v,
                      const float *beta, const float *alpha, const float *state,
                      float *new_state, int b, int h) {
  const int bh = b*@H@ + h;
  const long sbase = (long)bh*@V@*@K@;
  /* The recurrence runs in place in the caller's state output, so no scratch is needed. */
  for (long i = 0; i < (long)@V@*@K@; i++) new_state[sbase + i] = state[sbase + i];
  float *st = new_state + sbase;
  for (int t = 0; t < @T@; t++) {
    const long step = (long)bh*@T@ + t;
    const float *qt = q + step*@K@, *kt = k + step*@K@, *vt = v + step*@V@;
    const float *at = alpha + step*@ALPHA_DIM@;
    const float bt = beta[step];
    for (int r = 0; r < @V@; r += 4) {
      float *p0 = st + (long)(r+0)*@K@, *p1 = st + (long)(r+1)*@K@;
      float *p2 = st + (long)(r+2)*@K@, *p3 = st + (long)(r+3)*@K@;
      float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
      for (int c = 0; c < @K@; c += 4) {
        float a0 = @DECAY0@, a1 = @DECAY1@, a2 = @DECAY2@, a3 = @DECAY3@;
        acc0 = acc0 + (p0[c]*a0*kt[c] + p0[c+1]*a0*kt[c+1] + p0[c+2]*a0*kt[c+2] + p0[c+3]*a0*kt[c+3]);
        acc1 = acc1 + (p1[c]*a1*kt[c] + p1[c+1]*a1*kt[c+1] + p1[c+2]*a1*kt[c+2] + p1[c+3]*a1*kt[c+3]);
        acc2 = acc2 + (p2[c]*a2*kt[c] + p2[c+1]*a2*kt[c+1] + p2[c+2]*a2*kt[c+2] + p2[c+3]*a2*kt[c+3]);
        acc3 = acc3 + (p3[c]*a3*kt[c] + p3[c+1]*a3*kt[c+1] + p3[c+2]*a3*kt[c+2] + p3[c+3]*a3*kt[c+3]);
      }
      float d0 = (vt[r+0] - acc0)*bt, d1 = (vt[r+1] - acc1)*bt;
      float d2 = (vt[r+2] - acc2)*bt, d3 = (vt[r+3] - acc3)*bt;
      float res0 = 0.0f, res1 = 0.0f, res2 = 0.0f, res3 = 0.0f;
      for (int c = 0; c < @K@; c += 4) {
        float a0 = @DECAY0@, a1 = @DECAY1@, a2 = @DECAY2@, a3 = @DECAY3@;
        float g0[4], g1[4], g2[4], g3[4];
        for (int i = 0; i < 4; i++) {
          g0[i] = p0[c+i]*a0 + d0*kt[c+i];
          g1[i] = p1[c+i]*a1 + d1*kt[c+i];
          g2[i] = p2[c+i]*a2 + d2*kt[c+i];
          g3[i] = p3[c+i]*a3 + d3*kt[c+i];
          p0[c+i] = g0[i]; p1[c+i] = g1[i]; p2[c+i] = g2[i]; p3[c+i] = g3[i];
        }
        res0 = res0 + (g0[0]*qt[c] + g0[1]*qt[c+1] + g0[2]*qt[c+2] + g0[3]*qt[c+3]);
        res1 = res1 + (g1[0]*qt[c] + g1[1]*qt[c+1] + g1[2]*qt[c+2] + g1[3]*qt[c+3]);
        res2 = res2 + (g2[0]*qt[c] + g2[1]*qt[c+1] + g2[2]*qt[c+2] + g2[3]*qt[c+3]);
        res3 = res3 + (g3[0]*qt[c] + g3[1]*qt[c+1] + g3[2]*qt[c+2] + g3[3]*qt[c+3]);
      }
      out[(long)t*@H@*@V@ + (long)h*@V@ + r + 0] = res0;
      out[(long)t*@H@*@V@ + (long)h*@V@ + r + 1] = res1;
      out[(long)t*@H@*@V@ + (long)h*@V@ + r + 2] = res2;
      out[(long)t*@H@*@V@ + (long)h*@V@ + r + 3] = res3;
    }
  }
}
"""


def _cpu_renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _scan_source(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> str:
  decays = ["at[0]"]*4 if alpha_dim == 1 else [f"at[c+{i}]" for i in range(4)]
  body = _SCAN_SOURCE
  for token, value in (("H", heads), ("V", value_dim), ("K", key_dim), ("T", tokens), ("ALPHA_DIM", alpha_dim),
                       ("DECAY0", decays[0]), ("DECAY1", decays[1]), ("DECAY2", decays[2]), ("DECAY3", decays[3])):
    body = body.replace(f"@{token}@", str(value))
  return f"""#pragma clang fp contract(off)
{body}
void rockchip_scan(void *args) {{
  float *out = *(float**)args;
  const float *q = *((const float**)args + 1);
  const float *k = *((const float**)args + 2);
  const float *v = *((const float**)args + 3);
  const float *beta = *((const float**)args + 4);
  const float *alpha = *((const float**)args + 5);
  const float *state = *((const float**)args + 6);
  float *new_state = out + {batch*tokens*heads*value_dim};
  for (int b = 0; b < {batch}; b++)
    for (int h = 0; h < {heads}; h++)
      scan_head(out + (long)b*{tokens}*{heads}*{value_dim}, q, k, v, beta, alpha, state, new_state, b, h);
}}
"""


def _build_scan_program(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> UOp:
  """One kernel for the whole scan: every token, every head, in one call.

  A single output buffer holds the per-token outputs followed by the new recurrent
  state. Two output slots do not bind: the runtime allocates an output separately
  from the tensor passed in, so a second `outs` entry writes a buffer nobody reads.
  """
  source = _scan_source(batch, heads, tokens, value_dim, key_dim, alpha_dim)
  out_len, state_len = batch * tokens * heads * value_dim, batch * heads * value_dim * key_dim
  buf = UOp.placeholder((out_len + state_len,), dtypes.float, 0)
  q = UOp.placeholder((batch * heads * tokens * key_dim,), dtypes.float, 1)
  k = UOp.placeholder((batch * heads * tokens * key_dim,), dtypes.float, 2)
  v = UOp.placeholder((batch * heads * tokens * value_dim,), dtypes.float, 3)
  beta = UOp.placeholder((batch * heads * tokens,), dtypes.float, 4)
  alpha = UOp.placeholder((batch * heads * tokens * alpha_dim,), dtypes.float, 5)
  state = UOp.placeholder((batch * heads * value_dim * key_dim,), dtypes.float, 6)
  name = f"rockchip_scan_{batch}_{heads}_{tokens}_{value_dim}_{key_dim}_{alpha_dim}"
  sink = UOp.sink(buf, q, k, v, beta, alpha, state, arg=KernelInfo(name))
  renderer = _cpu_renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2, 3, 4, 5, 6))
  return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(buf, q, k, v, beta, alpha, state)),
             UOp(Ops.SOURCE, arg=source),
             UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))


# One PROGRAM per geometry, reused for every call: 24 blocks share one shape, and
# rebuilding it costs more than the kernel itself at decode sizes.
_SCAN_PROGRAMS: dict[tuple[int, int, int, int, int, int], UOp] = {}


def _scan_program(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> UOp:
  key = (batch, heads, tokens, value_dim, key_dim, alpha_dim)
  if (prog := _SCAN_PROGRAMS.get(key)) is None:
    _SCAN_PROGRAMS[key] = prog = _build_scan_program(*key)
  return prog


def scan_supported(batch: int, heads: int, tokens: int, value_dim: int, key_dim: int, alpha_dim: int) -> bool:
  """Geometry the kernel is compiled for: K a multiple of 4 (the reduce grouping)."""
  return key_dim % 4 == 0 and value_dim % 4 == 0 and 0 < tokens and 0 < heads and 0 < batch and alpha_dim in (1, key_dim)


def gated_delta_scan(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
  """One call for the whole scan. Returns (out (B,T,H,V), new_state (B,H,V,K)).

  Operands are the loop's own layouts before its per-step indexing: q,k (B,H,T,K),
  v (B,H,T,V), beta (B,H,T), alpha (B,H,T,1|K), state (B,H,V,K), all fp32.
  """
  batch, heads, tokens, key_dim = q.shape
  value_dim, alpha_dim = v.shape[-1], alpha.shape[-1]
  if not scan_supported(batch, heads, tokens, value_dim, key_dim, alpha_dim):
    raise ValueError(f"unsupported scan geometry {(batch, heads, tokens, value_dim, key_dim, alpha_dim)}")
  out_len = batch * tokens * heads * value_dim
  state_len = batch * heads * value_dim * key_dim
  buf = Tensor.empty(out_len + state_len, device=q.device, dtype=dtypes.float)
  res = buf.custom_kernel(q.contiguous().reshape(-1), k.contiguous().reshape(-1),
                          v.contiguous().reshape(-1), beta.contiguous().reshape(-1),
                          alpha.contiguous().reshape(-1), state.contiguous().reshape(-1),
                          fxn=lambda o, a, b, c, d, e, f: _scan_program(batch, heads, tokens, value_dim, key_dim, alpha_dim))[0]
  return res[:out_len].reshape(batch, tokens, heads, value_dim), res[out_len:].reshape(batch, heads, value_dim, key_dim)
