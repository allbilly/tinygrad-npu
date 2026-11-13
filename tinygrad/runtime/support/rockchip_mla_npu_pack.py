"""Host pack/unpack kernels for the vendor-style FP16 NPU MLA prefill attention.

Geometry is the MiniCPM3-4B MLA shape: 40 heads, qk_nope 64 + rope 32 = 96, v 64,
latent 256.  The kernels produce/consume exactly the layouts the row-major grouped FP16
plans need (row-major LHS/P, N16K32 blocked RHS/weights) plus the token-major PV unpack.
All arithmetic that must match the reference attention is compiled with FP contraction off.
CPU only; every function returns a CPU tensor and the caller uploads it (`.to('ROCKCHIP')`).
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

# Debug-only boundedness check. ABI 10 (dynamic row-major without replay scans) trusts these
# pack outputs to be finite: non-finite values would be silently propagated to the NPU instead
# of raising a replay error. Set ONPU_MLA_NPU_PACK_CHECK=1 for a capture/validation run to make
# every pack kernel assert its output is finite (small CPU cost, off by default).
ONPU_MLA_NPU_PACK_CHECK = ContextVar('ONPU_MLA_NPU_PACK_CHECK',0)

HEADS, NOPE, ROPE, QKD, VDIM, LATENT = 40, 64, 32, 64 + 32, 64, 256
_PREFIX = '#pragma STDC FP_CONTRACT OFF\n' + Path(__file__).with_suffix('.c').read_text() + r'''
static float mla_exp2_slow(float x) {
  /* Mirrors rockchip_exp2_vector's checked path (rockchip_cpu.py) exactly. */
  union { float f; uint32_t u; } v;
  int nan_ = (x != x);
  float safe = ((x == __builtin_inff()) | (x == -__builtin_inff()) | nan_) ? 0.0f : x;
  int q = (int)(safe + (safe < 0 ? -0.5f : 0.5f)), q1 = q >> 1;
  float s = safe - (float)q;
  float y = ((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
       0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
  union { int i; float f; } a, b; a.i = (q1 + 127) << 23; b.i = (q - q1 + 127) << 23;
  y = y * a.f * b.f;
  y = x >= 128 ? __builtin_inff() : y;
  y = x < -150 ? 0.0f : y;
  y = nan_ ? __builtin_nanf("") : y;
  return y;
}
static float mla_exp2v(float x) {
  /* The fast exponent-adjust path is only valid for |x| < 125 (q in [-125,125]); beyond
     that the integer exponent add overflows and yields NaN -- take the checked path. */
  union { float f; uint32_t u; } v; v.f = x;
  if ((v.u & 0x7fffffffu) >= 0x42fa0000u) return mla_exp2_slow(x);
  int q = (int)(x + (x < 0 ? -0.5f : 0.5f));
  float s = x - (float)q;
  float y = ((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
       0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
  v.f = y; v.u += (uint32_t)q * 0x800000u; return v.f;
}
'''


def _program(*inputs: UOp, source: str, name: str) -> UOp:
  sink = UOp.sink(*inputs, arg=KernelInfo(name))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=tuple(range(1, len(inputs))))
  return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=inputs), UOp(Ops.SOURCE, arg=source),
                                         UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))


def _check_finite(out: Tensor, inputs: tuple[Tensor, ...], name: str) -> Tensor:
  if not ONPU_MLA_NPU_PACK_CHECK: return out
  import numpy as np
  values = out.numpy()
  if not np.isfinite(values).all():
    bad = np.argwhere(~np.isfinite(values))[:8].tolist()
    ranges = [(float(x.numpy().min()), float(x.numpy().max())) if hasattr(x,'numpy') else None for x in inputs]
    raise ValueError(f'{name}: non-finite pack output at {bad}; source value ranges {ranges}; '
                     'FP16 cast overflow or non-finite model input')
  return out


def _run(out: Tensor, inputs: tuple[Tensor, ...], source: str, name: str) -> Tensor:
  return _check_finite(out.custom_kernel(*inputs, fxn=lambda o, *a: _program(o, *a, source=source, name=name))[0],
                       inputs, name)


def _f32(x: Tensor) -> Tensor:
  if x.device != 'CPU' or x.dtype is not dtypes.float: raise ValueError('MLA NPU pack inputs must be CPU fp32')
  return x.contiguous()


def _squeeze(x: Tensor, dims: tuple[int, ...]) -> Tensor:
  """Drop leading singleton dims so the model's (1,heads,...) views match the 3-D kernel contract."""
  shape = tuple(x.shape)
  while shape and shape[0] == 1 and shape != dims: shape = shape[1:]
  if shape != dims: raise ValueError(f'MLA NPU pack shape {tuple(x.shape)} is not {dims} (leading 1s allowed)')
  return x.reshape(dims)


def _half(x: Tensor) -> Tensor:
  if x.device != 'CPU' or x.dtype is not dtypes.half: raise ValueError('MLA NPU pack weights must be CPU fp16')
  return x.contiguous()


def _shape(x: Tensor) -> tuple[int, ...]:
  if not all(isinstance(n, int) for n in x.shape): raise ValueError('MLA NPU pack needs static shapes')
  return tuple(x.shape)  # type: ignore[return-value]


def mla_npu_pack_query(q_nope: Tensor, q_rope: Tensor, rows: int, capacity: int) -> Tensor:
  """Row-major QK LHS: (HEADS, capacity, 96) fp16."""
  q_nope = _squeeze(_f32(q_nope), (HEADS, capacity, NOPE))
  q_rope = _squeeze(_f32(q_rope), (HEADS, capacity, ROPE))
  source = _PREFIX + f'''
void mla_npu_pack_query(__fp16 *out,const float *q_nope,const float *q_rope) {{
  for(int h=0;h<{HEADS};h++)for(int r=0;r<{capacity};r++){{
    const float *g=q_nope+((size_t)h*{capacity}+r)*{NOPE}, *u=q_rope+((size_t)h*{capacity}+r)*{ROPE};
    __fp16 *d=out+(((size_t)h*{capacity}+r)*{QKD});
    for(int c=0;c<{NOPE};c++)d[c]=(__fp16)g[c];
    for(int c=0;c<{ROPE};c++)d[{NOPE}+c]=(__fp16)u[c];
  }}
}}'''
  out = Tensor.empty(HEADS * capacity * QKD, device='CPU', dtype=dtypes.half)
  return _run(out, (q_nope, q_rope), source, 'mla_npu_pack_query').reshape(HEADS, capacity, QKD)


def mla_npu_pack_k(k_nope: Tensor, k_rope: Tensor, context: int) -> Tensor:
  """QK RHS N16K32: (HEADS, keys/16, 96/32, 16, 32) fp16 from head-major k_nope and shared k_rope."""
  k_nope = _f32(k_nope)
  k_rope = _f32(k_rope)
  capacity = tuple(k_rope.shape)[-2] if k_rope.ndim > 1 else tuple(k_rope.shape)[0]
  k_rope = _squeeze(k_rope, (capacity, ROPE))
  k_nope = _squeeze(k_nope, (HEADS, capacity, NOPE))
  keys = context
  if keys % 16 or capacity < keys: raise ValueError('Unexpected MLA K pack geometry')
  source = _PREFIX + f'''
void mla_npu_pack_k(__fp16 *out,const float *k_nope,const float *k_rope) {{
  for(int h=0;h<{HEADS};h++){{
    __fp16 *dst=out+(size_t)h*{keys}*{QKD};
    for(int nblk=0;nblk<{keys//16};nblk++)for(int kblk=0;kblk<{QKD//32};kblk++){{
      __fp16 *blk=dst+((size_t)nblk*({QKD//32})+kblk)*512;
      for(int ni=0;ni<16;ni++){{
        int key=nblk*16+ni;
        for(int ki=0;ki<32;ki++){{
          int dim=kblk*32+ki; float val;
          if(dim<{NOPE}) val=k_nope[((size_t)h*{capacity}+key)*{NOPE}+dim];
          else val=k_rope[(size_t)key*{ROPE}+(dim-{NOPE})];
          blk[ni*32+ki]=(__fp16)val;
        }}
      }}
    }}
  }}
}}'''
  out = Tensor.empty(HEADS * keys * QKD, device='CPU', dtype=dtypes.half)
  return _run(out, (k_nope, k_rope), source, 'mla_npu_pack_k').reshape(HEADS, keys//16, QKD//32, 16, 32)


def mla_npu_pack_v(v: Tensor, context: int) -> Tensor:
  """PV RHS N16K32: (HEADS, 64/16, keys/32, 16, 32) fp16 from head-major v."""
  v = _f32(v)
  capacity = tuple(v.shape)[-2]
  v = _squeeze(v, (HEADS, capacity, VDIM))
  keys = context
  if keys % 32 or capacity < keys: raise ValueError('Unexpected MLA V pack geometry')
  source = _PREFIX + f'''
void mla_npu_pack_v(__fp16 *out,const float *v) {{
  for(int h=0;h<{HEADS};h++){{
    __fp16 *dst=out+(size_t)h*{VDIM}*{keys};
    for(int nblk=0;nblk<{VDIM//16};nblk++)for(int kblk=0;kblk<{keys//32};kblk++){{
      __fp16 *blk=dst+((size_t)nblk*({keys//32})+kblk)*512;
      for(int ni=0;ni<16;ni++){{
        int vd=nblk*16+ni;
        for(int ki=0;ki<32;ki++){{
          int key=kblk*32+ki;
          blk[ni*32+ki]=(__fp16)v[((size_t)h*{capacity}+key)*{VDIM}+vd];
        }}
      }}
    }}
  }}
}}'''
  out = Tensor.empty(HEADS * keys * VDIM, device='CPU', dtype=dtypes.half)
  return _run(out, (v,), source, 'mla_npu_pack_v').reshape(HEADS, VDIM // 16, keys // 32, 16, 32)


def mla_npu_pack_prob(scores: Tensor, start: int, rows: int, pool_address: int = 0) -> Tensor:
  """Exact-order causal softmax of the QK scores (HEADS, capacity, keys) f32 -> fp16 P row-major.

  `pool_address` (default 0 = current single-threaded behavior) is the parked attention-pool
  prefill dispatch address. When non-zero the compiled kernel calls it exactly like
  `rockchip_mla_softmax` does and splits the 40 independent heads 10/10/10/10 across the four
  parked workers; every row's sequential max/sum order and arithmetic are unchanged, so the
  output is byte-identical to the single-threaded path (per-head independence).
  """
  scores = _f32(scores)
  capacity, keys = tuple(scores.shape)[-2], tuple(scores.shape)[-1]
  scores = _squeeze(scores, (HEADS, capacity, keys))
  if not 0 <= start or start + rows > keys or rows > capacity: raise ValueError('Unexpected MLA softmax geometry')
  if type(pool_address) is not int or pool_address < 0: raise ValueError('MLA softmax pool address must be a non-negative int')
  source = _PREFIX + f'''
typedef struct {{ __fp16 *out; const float *scores; }} MlaProbWork;
static void mla_prob_head_range(__fp16 *out,const float *scores,int h0,int h1) {{
  const float scale=0.10206207261596575f;  // 1/sqrt(96)
  for(int h=h0;h<h1;h++)for(int r=0;r<{rows};r++){{
    int end={start}+r+1;
    const float *sc=scores+((size_t)h*{capacity}+r)*{keys};
    __fp16 *dst=out+((size_t)h*{capacity}+r)*{keys};
    float scaled[{keys}]; float e[{keys}];
    for(int t=0;t<end;t++)scaled[t]=sc[t]*scale;
    float maximum=-3.402823466e+38f;
    for(int t=0;t<end;t++)if(maximum<scaled[t])maximum=scaled[t];
    float sum=0.0f; int t=0;
    for(;t+4<=end;t+=4){{
      e[t+0]=mla_exp2v((scaled[t+0]-maximum)*1.4426950216293335f);
      e[t+1]=mla_exp2v((scaled[t+1]-maximum)*1.4426950216293335f);
      e[t+2]=mla_exp2v((scaled[t+2]-maximum)*1.4426950216293335f);
      e[t+3]=mla_exp2v((scaled[t+3]-maximum)*1.4426950216293335f);
      sum+=e[t+0];sum+=e[t+1];sum+=e[t+2];sum+=e[t+3];
    }}
    for(;t<end;t++){{e[t]=mla_exp2v((scaled[t]-maximum)*1.4426950216293335f);sum+=e[t];}}
    for(t=0;t<end;t++)dst[t]=(__fp16)(e[t]/sum);
    for(;t<{keys};t++)dst[t]=(__fp16)0;
  }}
  for(int h=h0;h<h1;h++)for(int r={rows};r<{capacity};r++){{
    __fp16 *row=out+((size_t)h*{capacity}+r)*{keys};
    for(int t=0;t<{keys};t++)row[t]=(__fp16)0;
  }}
}}
static void mla_prob_pool_callback(void **opaque,int total,int group) {{
  (void)total;
  MlaProbWork *w=(MlaProbWork*)opaque[0];
  mla_prob_head_range(w->out,w->scores,10*group,10*(group+1));
}}
void mla_npu_softmax_pack(__fp16 *out,const float *scores) {{
  if({pool_address}ULL) {{
    MlaProbWork work={{out,scores}};
    void *args[4]={{&work,0,0,0}};
    if(!((int(*)(void(*)(void**,int,int),void**,int)){pool_address}ULL)(mla_prob_pool_callback,args,0))return;
  }}
  mla_prob_head_range(out,scores,0,{HEADS});
}}'''
  out = Tensor.empty(HEADS * capacity * keys, device='CPU', dtype=dtypes.half)
  return _run(out, (scores,), source, 'mla_npu_softmax_pack').reshape(HEADS, capacity, keys)


def mla_npu_pack_expand_lhs(c_kv: Tensor, rows: int, capacity: int) -> Tensor:
  """Expansion LHS: replicate token-major c_kv (capacity, 256) f32 -> (HEADS, capacity, 256) fp16."""
  c_kv = _f32(c_kv)
  c_kv = _squeeze(c_kv, (capacity, LATENT))
  if not 1 <= rows <= capacity: raise ValueError('Unexpected MLA expansion rows')
  source = _PREFIX + f'''
void mla_npu_expand_lhs(__fp16 *out,const float *c_kv) {{
  for(int h=0;h<{HEADS};h++)for(int r=0;r<{capacity};r++){{
    const float *src=c_kv+(size_t)r*{LATENT};
    __fp16 *dst=out+((size_t)h*{capacity}+r)*{LATENT};
    for(int c=0;c<{LATENT};c++)dst[c]=(__fp16)src[c];
  }}
}}'''
  out = Tensor.empty(HEADS * capacity * LATENT, device='CPU', dtype=dtypes.half)
  return _run(out, (c_kv,), source, 'mla_npu_expand_lhs').reshape(HEADS, capacity, LATENT)


def mla_npu_unpack_context(result: Tensor, rows: int) -> Tensor:
  """PV result (HEADS, capacity, 64) f32 -> token-major (capacity, HEADS*64) f32."""
  result = _f32(result)
  capacity = tuple(result.shape)[-2]
  result = _squeeze(result, (HEADS, capacity, VDIM))
  if not 1 <= rows <= capacity: raise ValueError('Unexpected MLA PV rows')
  source = _PREFIX + f'''
void mla_npu_unpack_context(float *out,const float *result) {{
  for(int r=0;r<{rows};r++)for(int h=0;h<{HEADS};h++){{
    const float *src=result+((size_t)h*{capacity}+r)*{VDIM};
    float *dst=out+(size_t)r*{HEADS * VDIM}+(size_t)h*{VDIM};
    for(int c=0;c<{VDIM};c++)dst[c]=src[c];
  }}
  for(int r={rows};r<{capacity};r++)for(int h=0;h<{HEADS};h++){{
    float *dst=out+(size_t)r*{HEADS * VDIM}+(size_t)h*{VDIM};
    for(int c=0;c<{VDIM};c++)dst[c]=0.0f;
  }}
}}'''
  out = Tensor.empty(capacity * HEADS * VDIM, device='CPU', dtype=dtypes.float)
  return _run(out, (result,), source, 'mla_npu_unpack_context').reshape(capacity, HEADS * VDIM)


def pack_expand_k(kb: Tensor) -> Tensor:
  """Expansion K weight: attn_k_b (HEADS, 256, 64) fp16 -> N16K32 (HEADS, 4, 8, 16, 32)."""
  kb = _half(kb)
  if _shape(kb) != (HEADS, LATENT, NOPE): raise ValueError('Unexpected attn_k_b shape')
  source = _PREFIX + f'''
void mla_npu_pack_wkb(__fp16 *out,const __fp16 *kb) {{
  for(int h=0;h<{HEADS};h++){{
    __fp16 *dst=out+(size_t)h*{NOPE}*{LATENT};
    for(int nblk=0;nblk<{NOPE//16};nblk++)for(int kblk=0;kblk<{LATENT//32};kblk++){{
      __fp16 *blk=dst+((size_t)nblk*({LATENT//32})+kblk)*512;
      for(int ni=0;ni<16;ni++)for(int ki=0;ki<32;ki++){{
        int n=nblk*16+ni, k=kblk*32+ki;
        blk[ni*32+ki]=kb[((size_t)h*{LATENT}+k)*{NOPE}+n];
      }}
    }}
  }}
}}'''
  out = Tensor.empty(HEADS * NOPE * LATENT, device='CPU', dtype=dtypes.half)
  return _run(out, (kb,), source, 'mla_npu_pack_wkb').reshape(HEADS, NOPE // 16, LATENT // 32, 16, 32)


def pack_expand_v(vb: Tensor) -> Tensor:
  """Expansion V weight: attn_v_b (HEADS, 64, 256) fp16 -> N16K32 (HEADS, 4, 8, 16, 32)."""
  vb = _half(vb)
  if _shape(vb) != (HEADS, VDIM, LATENT): raise ValueError('Unexpected attn_v_b shape')
  source = _PREFIX + f'''
void mla_npu_pack_wvb(__fp16 *out,const __fp16 *vb) {{
  for(int h=0;h<{HEADS};h++){{
    __fp16 *dst=out+(size_t)h*{VDIM}*{LATENT};
    for(int nblk=0;nblk<{VDIM//16};nblk++)for(int kblk=0;kblk<{LATENT//32};kblk++){{
      __fp16 *blk=dst+((size_t)nblk*({LATENT//32})+kblk)*512;
      for(int ni=0;ni<16;ni++)for(int ki=0;ki<32;ki++){{
        int n=nblk*16+ni, k=kblk*32+ki;
        blk[ni*32+ki]=vb[((size_t)h*{VDIM}+n)*{LATENT}+k];
      }}
    }}
  }}
}}'''
  out = Tensor.empty(HEADS * VDIM * LATENT, device='CPU', dtype=dtypes.half)
  return _run(out, (vb,), source, 'mla_npu_pack_wvb').reshape(HEADS, VDIM // 16, LATENT // 32, 16, 32)


# Short aliases used by unit tests; the integration names above are the public API.
pack_query, pack_k, pack_v = mla_npu_pack_query, mla_npu_pack_k, mla_npu_pack_v
softmax_pack, expand_lhs, unpack_context = mla_npu_pack_prob, mla_npu_pack_expand_lhs, mla_npu_unpack_context
