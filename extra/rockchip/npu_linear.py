"""Minimal RK3588 GEMM offload: one NPU matmul per Linear, no fused kernels.

This is the baseline path: every nn.Linear is replaced by the same generic FP16
GEMM, everything else stays on the CPU device. Layouts match ops_rockchip:
lhs (m,k) -> (k/8,m,8), weight (n,k) -> (n/16,k/32,16,32), out (n/4,m,4).

Packed weights are spread across IOMMU domains when they do not fit in one, so a
model larger than the ~4 GB per-domain IOVA window can still be loaded.
"""
from __future__ import annotations

import ctypes
import platform
import struct
from dataclasses import dataclass, replace
from typing import Any

from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.helpers import Context, Target, round_up
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _int8_padded_depth, _matmul_parts, int8_task_plan, int8_task_plan_multi
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

TILE = 24576  # max packed columns per weight tile
K_TILE = 4096  # max K per matmul call (CBUF limit)

# Bytes of packed weights per IOMMU domain. The driver gives each domain an
# aperture of exactly 4 GiB (rknpu_iommu.c: dma_limit = 1ULL << 32), but this
# budget counts weights only: command buffers, activation buffers and
# allocator-retained frees are not in the sum. 3.906 GiB was tried and reverted —
# it bought nothing measurable (Qwen3.5 2B 2.09/3.05 vs 2.05/3.04 tok/s) and one
# Gemma2 2B run ended in a board reboot. Measured: 3.76 GB of packed weights fits
# in one domain and 5.23 GB does not. 3 GiB matches the reference's default.
DOMAIN_BUDGET = 3 << 30


class _Domains:
    """Assign each Linear's packed weights to an IOMMU domain, in walk order.

    Walk order is block order, so a block's weights share a domain unless that
    block straddles a budget boundary, and the NPU switches domain a few times
    per token instead of once per Linear.
    """

    def __init__(self) -> None:
        self.used: list[int] = [0]

    def place(self, nbytes: int) -> str:
        if nbytes > DOMAIN_BUDGET:
            raise MemoryError(f"Linear of {nbytes} bytes exceeds the {DOMAIN_BUDGET}-byte domain budget")
        if self.used[-1] + nbytes > DOMAIN_BUDGET:
            if len(self.used) >= 16:
                raise MemoryError("16 NPU IOMMU domains exhausted")
            self.used.append(0)
        self.used[-1] += nbytes
        return "ROCKCHIP" if len(self.used) == 1 else f"ROCKCHIP:{len(self.used) - 1}"


# One allocator per process: the IOVA window belongs to the driver domain, so a
# second model must not start booking domain 0 from zero again.
_DOMAINS = _Domains()


def quantize_rows(x: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-row int8 quantization: x ~ codes * scales[:, None], codes in [-127, 127]."""
    scale = (x.float().abs().max(axis=-1, keepdim=True) / 127.0).maximum(1e-8)
    return (x.float() / scale).round().maximum(-127).minimum(127).cast(dtypes.int8), scale.reshape(-1)


def pack_weight_int8(weight: Tensor, device: str) -> tuple[Tensor, Tensor]:
    """INT8 weights in the CNA layout (n/32, k/32, 32, 32) plus one fp32 scale per output row."""
    n, k = weight.shape
    n, k = round_up(n, 32), round_up(k, 32)
    padded = weight.pad(((0, n - weight.shape[0]), (0, k - weight.shape[1])))
    codes, scales = quantize_rows(padded)
    return codes.reshape(n // 32, 32, k // 32, 32).permute(0, 2, 1, 3).contiguous().to(device), scales


_QUANTIZE_SOURCE = '''
static void part(unsigned char *raw, const float *input, int begin, int end,
                 int depth, int padded, int header) {
  float *scales = (float*)raw;
  signed char *output = (signed char*)(raw + header);
  for (int row = begin; row < end; row++) {
    const float *x = input + (long)row*depth;
    float largest = 0.0f;
    for (int i = 0; i < depth; i++) { float a = x[i] < 0.0f ? -x[i] : x[i]; if (a > largest) largest = a; }
    float scale = largest / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    scales[row] = scale;
    float inv = 1.0f / scale;
    signed char *o = output + (long)row*padded;
    for (int i = 0; i < depth; i++) {
      float q = __builtin_nearbyintf(x[i]*inv);
      if (q > 127.0f) q = 127.0f;
      if (q < -127.0f) q = -127.0f;
      o[i] = (signed char)q;
    }
    for (int i = depth; i < padded; i++) o[i] = 0;
  }
}
'''


def _cpu_renderer() -> ClangRenderer:
  arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
  return ClangRenderer(Target("CPU", arch=arch + ",native", renderer="CLANG"))


def _quantize_source(rows: int, depth: int, padded: int, header: int) -> str:
  """`part` plus the entry point. The CPU kernel ABI is one packed argument buffer
  (ClangRenderer.packed_args), so the entry takes `void *args` and holds only the two buffer
  addresses; the geometry is baked in, the way the INT8 matmul bakes it into its descriptor."""
  return _QUANTIZE_SOURCE + f'''
void rockchip_quantize(unsigned char *args) {{
  part(*(unsigned char**)args, *((const float**)args + 1), 0, {rows}, {depth}, {padded}, {header});
}}
'''


def _build_quantize_program(rows: int, depth: int, padded: int, header: int) -> UOp:
  """One kernel for the whole activation payload: per-row max, scale into the header, int8 codes."""
  source = _quantize_source(rows, depth, padded, header)
  out = UOp.placeholder((header + rows * padded,), dtypes.uint8, 0)
  inp = UOp.placeholder((rows, depth), dtypes.float, 1)
  sink = UOp.sink(out, inp, arg=KernelInfo(f"rockchip_quantize_{rows}_{depth}_{padded}"))
  renderer = _cpu_renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1,))
  return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, inp)),
             UOp(Ops.SOURCE, arg=source),
             UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))


# One PROGRAM per geometry, reused for every call. Placeholders rather than buffers make it
# call-agnostic; this is what to_program_cache does for generated kernels. Rebuilding it cost
# 0.071 ms per call and this runs 323 times per decode token.
_QUANTIZE_PROGRAMS: dict[tuple[int, int, int, int], UOp] = {}


def _quantize_program(rows: int, depth: int, padded: int, header: int) -> UOp:
  if (prog := _QUANTIZE_PROGRAMS.get(key := (rows, depth, padded, header))) is None:
    _QUANTIZE_PROGRAMS[key] = prog = _build_quantize_program(rows, depth, padded, header)
  return prog


def int8_payload(x: Tensor, device: str) -> tuple[Tensor, Tensor]:
    """Activation payload in one kernel: fp32 per-row scales in a 64-aligned header, then int8 codes."""
    rows, depth = x.shape
    padded = _int8_padded_depth(depth)
    header = round_up(rows * 4, 64)
    buf = Tensor.empty(header + rows * padded, device="CPU", dtype=dtypes.uint8)
    buf = buf.custom_kernel(x.float().contiguous(),
                            fxn=lambda o, a: _quantize_program(rows, depth, padded, header))[0]
    scales = buf[:rows * 4].bitcast(dtypes.float)
    return buf.to(device), scales


def _matmul_int8(out: UOp, lhs: UOp, rhs: UOp, m: int, n: int, k: int, cores: int, column_offset: int) -> UOp:
    sink = UOp.sink(out, lhs, rhs, arg=KernelInfo(f"rockchip_int8_{m}_{n}_{k}"))
    info = replace(ProgramInfo.from_sink(sink, Target("ROCKCHIP")), outs=(0,), ins=(1, 2))
    return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, lhs, rhs)),
               UOp(Ops.SOURCE, arg=f"Rockchip INT8 {m}x{k}x{n}"),
               UOp(Ops.BINARY, arg=struct.pack("<8I", m, n, k, cores, column_offset * k,
                                               round_up(m * 4, 64), 0, 1))))


def _dequant_source(m: int, n: int, parts: tuple[tuple[int, int], ...], scale_offset: int) -> str:
  """The whole tail as one kernel: plane scatter, int32 -> fp32, both multiplies.

  `parts` is baked in as straight-line code. The CPU kernel ABI is one packed argument
  buffer (ClangRenderer.packed_args), so the entry takes `void *args` and unpacks the four
  buffer addresses itself; the geometry cannot ride through the call because `_states`
  rejects Ops.CONST args. The entry is the only function in the object, so it sits at
  offset 0, which is where CPUProgram jumps.

  `scale_offset` is the output-column offset of this call inside the packed weight tile. It
  was missing: the tail read `w_scale[offset + j]`, i.e. the scales of the first `n` output
  rows, so any call with `column_offset != 0` dequantized with the wrong per-row scale. That
  is reachable at m>1, where the 24576-wide chunk is split into two 12288-wide slices.

  `#pragma clang fp contract(off)` is required: clang defaults to -ffp-contract=on, which
  fuses `t*w` into an fmla and rounds once where the unfused chain rounds twice. The
  unfused path this replaces rounds twice, so the fused kernel must too.
  """
  planes = "".join(f'''
  for (int row = 0; row < {m}; row++) {{
    const int *src = raw + {offset * m} + (long)row*{width};
    float *dst = out + (long)row*{n} + {offset};
    float a = a_scale[row];
    for (int j = 0; j < {width}; j++) dst[j] = (float)src[j] * a * w_scale[{scale_offset + offset} + j];
  }}''' for offset, width in parts)
  return f'''#pragma clang fp contract(off)
void rockchip_dequantize(void *args) {{
  float *out = *(float**)args;
  const int *raw = *((const int**)args + 1);
  const float *a_scale = *((const float**)args + 2);
  const float *w_scale = *((const float**)args + 3);{planes}
}}
'''


def _build_dequant_program(m: int, n: int, parts: tuple[tuple[int, int], ...], scale_offset: int) -> UOp:
  """One kernel for the whole tail, so the scatter never materializes a second buffer."""
  source = _dequant_source(m, n, parts, scale_offset)
  name = "_".join(f"{offset}-{width}" for offset, width in parts)
  out = UOp.placeholder((m * n,), dtypes.float, 0)
  raw = UOp.placeholder((m * n,), dtypes.int32, 1)
  a_scale = UOp.placeholder((m,), dtypes.float, 2)
  w_scale = UOp.placeholder((n,), dtypes.float, 3)
  sink = UOp.sink(out, raw, a_scale, w_scale, arg=KernelInfo(f"rockchip_dequant_{m}_{n}_{name}_{scale_offset}"))
  renderer = _cpu_renderer()
  info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2, 3))
  return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, raw, a_scale, w_scale)),
             UOp(Ops.SOURCE, arg=source),
             UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))


# One PROGRAM per geometry, reused for every call. The UOp is immutable and its params are
# bound to the call's buffers at schedule time, so this is what to_program_cache does for
# generated kernels. Rebuilding it costs 0.26 ms per call (four UOp.placeholder calls
# dominate) and this tail runs 323 times per decode token.
_DEQUANT_PROGRAMS: dict[tuple[int, int, tuple[tuple[int, int], ...], int], UOp] = {}


def _dequant_program(m: int, n: int, parts: tuple[tuple[int, int], ...], scale_offset: int) -> UOp:
  if (prog := _DEQUANT_PROGRAMS.get(key := (m, n, parts, scale_offset))) is None:
    _DEQUANT_PROGRAMS[key] = prog = _build_dequant_program(m, n, parts, scale_offset)
  return prog


def _dequantize(host: Tensor, a_scale: Tensor, w_scale: Tensor, m: int, n: int,
                parts: tuple[tuple[int, int], ...], scale_offset: int = 0) -> Tensor:
  """Scatter the per-core int32 planes into (m, n) fp32 and apply both scales, in one kernel.

  Bit-exact with the unfused `Tensor.cat` + `.float()` + two multiplies: same values, same
  left-to-right order, no contraction. Measured in profile_int8.py over the 4B's four
  projection shapes at m=1 and m=64: the tail was 0.571-0.719 ms unfused (0.146-0.284 for
  the cat, 0.405-0.435 for the two multiplies) and is 0.318-0.328 ms fused, which takes
  the whole call from 1.638-1.886 ms to 1.278-1.301 ms. The tail is fixed per call, not
  per element, so the saving does not shrink with m.
  """
  if w_scale.shape[0] < scale_offset + n:
    raise ValueError(f"w_scale has {w_scale.shape[0]} entries but the output is {n} wide at offset {scale_offset}")
  out = Tensor.empty(m * n, device=host.device, dtype=dtypes.float)
  return out.custom_kernel(host, a_scale, w_scale, fxn=lambda o, r, a, w: _dequant_program(m, n, parts, scale_offset))[0] \
      .reshape(m, n)


def packed_matmul_int8(x: Tensor, weight: Tensor, w_scale: Tensor, columns: int, device: str,
                       cores: int = 3, column_offset: int = 0) -> Tensor:
    """One INT8 matmul. Each core writes a dense (m, part) plane; one CPU kernel scatters them."""
    m, original_k = x.shape
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    # The pad is a no-op at the Tensor level when K already matches, but the call itself
    # costs 0.056 ms and this runs 323 times per token. _int8_padded_depth zero-fills the
    # tail inside the quantize kernel, so the only real padding needed is a ragged K tile.
    payload, a_scale = int8_payload(x if original_k == k else x.pad(((0, 0), (0, k - original_k))), device)
    out = Tensor.empty(m * n, device=device, dtype=dtypes.int32)
    out = out.custom_kernel(payload, weight, fxn=lambda o, a, w: _matmul_int8(o, a, w, m, n, k, cores, column_offset))[0]
    # The NPU device has no renderer, so the tail runs on the host, not on ROCKCHIP.
    return _dequantize(out.to(x.device), a_scale, w_scale, m, n,
                       _matmul_parts(m, n, k, cores, int8=True), column_offset)


# ---- fused INT8 Linear: quantize, submit and dequantize in one C function ---------------
#
# A decode Linear costs 1.55 ms of host time spread over six tinygrad launches (pad,
# quantize, h2d, alloc, submit, d2h, dequant). The arithmetic is ~2.5 us of quantize and
# 13-22 us of dequantize; the rest is launch overhead. The function below replaces all six
# with one CALL whose body writes the int8 payload straight into an NPU mapping, issues both
# ioctls itself, and reads the int32 planes straight back out of the other mapping, so no
# host<->device copy survives either.

# The source is compiled freestanding (-nostdlib), so libc is declared extern and resolved by
# the JIT loader against the libraries the process already has. Baking the address instead, as
# the reference replay does, would make the source depend on the ASLR slide and cost a clang
# invocation per model load.


def _drm_request(nr: int, size: int) -> int:
    """_IOWR('d', nr, struct) with the payload size in the argument-size field."""
    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


_SUBMIT_REQUEST = _drm_request(0x41, ctypes.sizeof(rk.struct_rknpu_submit))
_SYNC_REQUEST = _drm_request(0x45, ctypes.sizeof(rk.struct_rknpu_mem_sync))

# u64 slots in the per-plan parameter block: payload, int32 planes, w_scale, status word,
# task array, payload obj_addr, planes obj_addr, drm fd, device counter block.
_PARAMS = 10   # + skip_quantise: the four same-row consumers share one payload buffer


@dataclass(frozen=True)
class _FusedKey:
    """Everything the fused C function bakes in: geometry and submit fields, no addresses.

    `_states` rejects Ops.CONST arguments, so no scalar can ride through the call; domain, core
    mask, subcores, request words and the whole geometry are literals. Buffer addresses and the
    drm fd deliberately are *not*: they arrive in a parameter block, so the generated source is
    identical for every Linear of the same shape and the on-disk compile cache hits across
    processes. Baking them would make every plan a unique source, i.e. 500 clang invocations
    per model load instead of ~16.
    """
    m: int
    n: int
    k: int
    depth_in: int
    header: int
    row_stride: int
    row_offset: int
    k_offset: int
    column_offset: int
    parts: tuple[tuple[int, int], ...]
    payload_size: int
    out_size: int
    domain: int
    core_mask: int
    subcores: tuple[int, ...]
    tasks: int
    m_rows: int = 0
    # FFN tail: stop after the int32 planes so one pack kernel can consume gate and up
    # together. Default False keeps every existing published number on the same path.
    raw_out: bool = False


# Vectorised INT8 row quantisation for the fused Linear. The scalar loop it replaces measured
# 0.831 ms for m=128 x depth 2560 and 3.247 ms for depth 9728, and it is paid once per Linear per
# prefill -- roughly 8.2 ms/layer, ~0.3 s over the model, which is half of the 0.62 s that the
# generated `rockchip_linear` spends outside the NPU submit.
#
# The codes must be BIT-IDENTICAL to the scalar path or every published number changes, so the
# rounding has to be round-half-to-even: `vcvtnq_s32_f32` is FRINTN, the same rounding
# `__builtin_nearbyintf` uses. Verified over 42 cases (depth 2560/9728/2559/31/4/1/130, half of
# them constructed to land exactly on .5/127 ties): 0 codes differing, 0 scales differing.
# Measured 3.02x (depth 2560) and 3.06x (depth 9728).
_QZ_HELPER = r"""
#if defined(__aarch64__)
#include <arm_neon.h>
static void qz_row(const float *xr, int depth, int k, signed char *o, float *scale_out) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  const float32x4_t zero = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= depth; i += 4) {
    float32x4_t a = vabsq_f32(vld1q_f32(xr + i));
    /* The scalar loop is `if (a > largest)`, which IGNORES NaN; vmaxq_f32 propagates it, and one
       NaN lane then makes the row's scale NaN, which zeroes every code and collapses the model.
       NaN test is `a != a` (vceqq is false for NaN, so negate it) -- NOT `a < a`, which is also
       false for NaN. Substituting 0 is exact because `largest` starts at 0 and every term is
       |x| >= 0, so a dropped NaN leaves the result unchanged, exactly as the scalar compare does. */
    a = vbslq_f32(vmvnq_u32(vceqq_f32(a, a)), zero, a);
    vmax = vmaxq_f32(vmax, a);
  }
  float largest = vmaxvq_f32(vmax);
  for (; i < depth; i++) { float a = xr[i] < 0.0f ? -xr[i] : xr[i]; if (a > largest) largest = a; }
  float scale = largest / 127.0f;
  if (scale < 1e-8f) scale = 1e-8f;
  *scale_out = scale;
  const float32x4_t vinv = vdupq_n_f32(1.0f / scale);
  const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
  i = 0;
  for (; i + 4 <= depth; i += 4) {
    int32x4_t q = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xr + i), vinv));
    q = vminq_s32(vmaxq_s32(q, lo), hi);
    int16x4_t n = vmovn_s32(q);
    int8x8_t b = vmovn_s16(vcombine_s16(n, n));
    int32_t packed = vget_lane_s32(vreinterpret_s32_s8(b), 0);
    __builtin_memcpy(o + i, &packed, 4);
  }
  for (; i < depth; i++) {
    float q = __builtin_nearbyintf(xr[i] * (1.0f / scale));
    if (q > 127.0f) q = 127.0f;
    if (q < -127.0f) q = -127.0f;
    o[i] = (signed char)q;
  }
  for (int j = depth; j < k; j++) o[j] = 0;
}
#else
static void qz_row(const float *xr, int depth, int k, signed char *o, float *scale_out) {
  float largest = 0.0f;
  for (int i = 0; i < depth; i++) { float a = xr[i] < 0.0f ? -xr[i] : xr[i]; if (a > largest) largest = a; }
  float scale = largest / 127.0f;
  if (scale < 1e-8f) scale = 1e-8f;
  *scale_out = scale;
  float inv = 1.0f / scale;
  for (int i = 0; i < depth; i++) {
    float q = __builtin_nearbyintf(xr[i] * inv);
    if (q > 127.0f) q = 127.0f;
    if (q < -127.0f) q = -127.0f;
    o[i] = (signed char)q;
  }
  for (int i = depth; i < k; i++) o[i] = 0;
}
#endif
"""


# Quantise was the largest host item in the fused call and the only one still running on one
# core. The reference dispatches it 4-way at `rows*depth >= 32768` (ref rockchip_int8.py:76-87);
# this is the same pool with the same serial fallback.
#
# Enabled, measured in-model over one warm prefill (C counters, not per-geometry x calls):
#   quantise 105.56 -> 41.37 ms, wall 1,282.71 -> 1,220.68 ms, 344/355 calls dispatched,
#   3200 pool dispatches, 0 refused. Parity: logits 895365e27ed4255b unchanged.
# The dispatch is only trustworthy because stats[3] counts successes; before that a refusal was
# indistinguishable from a success and this path was recorded as "inert" (mistake.md M68).
PARALLEL_QUANT = True
# Ablation only: skip the two cache-maintenance ioctls. Produces garbage results (the NPU and
# the CPU then disagree about cache state) and exists solely to bound their timing cost.
SKIP_SYNCS = False
# Ablation only: do not submit to the NPU. The planes hold stale data, so results are garbage;
# this isolates how much of the host half is device contention rather than the loop.
SKIP_SUBMIT = False
QZ_THREADS = 4
QZ_MIN_ELEMS = 32768
# DISABLED by measurement: unlike quantise, this loop is bandwidth-bound (it writes m*n floats),
# so four threads only add handoff. In-model, one warm prefill, same process and probe:
#   serial 62.46 ms / 1,220.68 ms wall  ->  parallel 67.85 ms / 1,233.23 ms wall (+12.55 ms)
# The pool did run (248/355 calls dispatched, 0 refused), so this is the loop, not the dispatch.
# Kept so the experiment and its verdict are not lost, the same way _row_key is kept.
PARALLEL_DEQUANT = False
DQ_MIN_ELEMS = 32768


def _pool_addr():
  """Address of `rk_run`, or None when the pool is not registered in this process."""
  try:
    from tinygrad.runtime.support.c import DLL
    if "rkpool" not in DLL._loaded_:
      return None
    import thread_split
    return thread_split.pool_address()
  except Exception:
    return None


def _fused_source(key: _FusedKey) -> str:
    """The whole Linear as one C function: quantize, two ioctls, dequantize.

    `#pragma clang fp contract(off)` is required: clang defaults to -ffp-contract=on, which
    fuses `t*w` into an fmla and rounds once where the unfused chain rounds twice. The path
    this replaces rounds twice, so the fused kernel must too.

    The quantize is the same arithmetic as `_QUANTIZE_SOURCE`, including `1.0f/scale` and
    `__builtin_nearbyintf`, so the int8 codes and the fp32 scales are bit-identical. A ragged
    K tile is handled by `depth_in < k`: columns [depth_in, k) are zero, which is what the
    explicit pad used to produce, and zeros change neither the row max nor the codes.

    `args` is the packed argument buffer the CPU kernel ABI passes, so the entry takes
    `void *args` and unpacks it itself. Slot 2 is the parameter block, filled once per plan:
    payload, int32 planes, w_scale, status word, task array, and the two obj_addrs the cache
    syncs need.
    """
    quantize = f"""
  for (int row = 0; row < {key.m_rows or key.m}; row++) {{
    const float *xr = x + (long)({key.row_offset}+row)*{key.row_stride} + {key.k_offset};
    signed char *o = codes + (long)row*{key.k};
    qz_row(xr, {key.depth_in}, {key.k}, o, scales + row);
  }}"""
    qrows = key.m_rows or key.m
    POOL_TYPES = """
typedef void rkq_fn_t(void **,int,int);
typedef int rkq_dispatch_t(rkq_fn_t *,void **,int);"""
    qpool = _pool_addr() if (PARALLEL_QUANT and qrows * key.depth_in >= QZ_MIN_ELEMS) else None
    if qpool is None:
      qz_prelude = qz_group = ""
      qz_body = quantize
    else:
      nq = QZ_THREADS
      qz_prelude = """
static void rockchip_linear_qz(void **ctx,int unused,int group);"""
      qz_body = f"""
  {{
    void *qctx[3];
    qctx[0] = (void *)x; qctx[1] = (void *)codes; qctx[2] = (void *)scales;
    volatile unsigned long qt[2];
    qt[0] = {qpool}ULL >> 32; qt[1] = {qpool}ULL & 0xFFFFFFFFUL;
    unsigned long qaddr = (qt[0] << 32) | qt[1];
    rkq_dispatch_t *qfp = (rkq_dispatch_t *)qaddr;
    if (qfp(rockchip_linear_qz,qctx,{nq}) != 0) {{{quantize}}} else stats[3] += 1;
  }}"""
      qz_group = f"""
static void rockchip_linear_qz(void **ctx,int unused,int group) {{
  (void)unused;
  const float *qx = (const float *)ctx[0];
  signed char *qc = (signed char *)ctx[1];
  float *qs = (float *)ctx[2];
  int lo = {qrows}*group/{nq}, hi = {qrows}*(group+1)/{nq};
  for (int row = lo; row < hi; row++) {{
    const float *xr = qx + (long)({key.row_offset}+row)*{key.row_stride} + {key.k_offset};
    qz_row(xr, {key.depth_in}, {key.k}, qc + (long)row*{key.k}, qs + row);
  }}
}}"""
    regions = 2 if key.m_rows else 1
    planes_serial = "".join(f"""
  for (int g = 0; g < {regions}; g++) {{
  for (int row = 0; row < {key.m}; row++) {{
    const int *src = raw + g*{key.m * key.n} + {offset * key.m} + (long)row*{width};
    float *dst = out + (long)(g*{key.m}+row)*{key.n} + {offset};
    float a = scales[g*{key.m}+row];
    for (int j = 0; j < {width}; j++) dst[j] = (float)src[j] * a * w_scale[{key.column_offset + offset} + j];
  }}}}""" for offset, width in key.parts)
    dpool = _pool_addr() if (PARALLEL_DEQUANT and not key.raw_out
                             and regions * key.m * key.n >= DQ_MIN_ELEMS) else None
    if dpool is None:
      dq_prelude = dq_group = ""
      planes = "" if key.raw_out else planes_serial
    else:
      dq_rows = "".join(f"""
    {{{{
    const int *src = draw + g*{key.m * key.n} + {offset * key.m} + (long)row*{width};
    float *dst = dout + (long)t*{key.n} + {offset};
    for (int j = 0; j < {width}; j++) dst[j] = (float)src[j] * a * dw[{key.column_offset + offset} + j];
    }}}}""" for offset, width in key.parts)
      dq_prelude = """
static void rockchip_linear_dq(void **ctx,int unused,int group);"""
      dq_group = f"""
static void rockchip_linear_dq(void **ctx,int unused,int group) {{
  (void)unused;
  const int *draw = (const int *)ctx[0];
  float *dout = (float *)ctx[1];
  const float *dscales = (const float *)ctx[2];
  const float *dw = (const float *)ctx[3];
  int total = {regions * key.m};
  int lo = total*group/{QZ_THREADS}, hi = total*(group+1)/{QZ_THREADS};
  for (int t = lo; t < hi; t++) {{
    int g = t / {key.m}, row = t % {key.m};
    float a = dscales[g*{key.m}+row];{dq_rows}
  }}
}}"""
      planes = f"""
  {{
    void *dctx[4];
    dctx[0] = (void *)raw; dctx[1] = (void *)out;
    dctx[2] = (void *)scales; dctx[3] = (void *)w_scale;
    volatile unsigned long dt[2];
    dt[0] = {dpool}ULL >> 32; dt[1] = {dpool}ULL & 0xFFFFFFFFUL;
    unsigned long daddr = (dt[0] << 32) | dt[1];
    rkq_dispatch_t *dfp = (rkq_dispatch_t *)daddr;
    if (dfp(rockchip_linear_dq,dctx,{QZ_THREADS}) != 0) {{{planes_serial}}} else stats[5] += 1;
  }}"""
    sync_to = "" if SKIP_SYNCS else f"  if (!err) err = sync_mem(fd,1,params[5],{key.payload_size}ULL);"
    sync_from = "" if SKIP_SYNCS else f"  if (!err) err = sync_mem(fd,2,params[6],{key.out_size}ULL);"
    subcores = ",".join(str(value) for value in key.subcores)
    submit_block = "" if SKIP_SUBMIT else f"""
  if (!err) {{
    long t0[2], t1[2];
    clock_gettime(1,t0);
    struct Submit submit={{5,6000,0,{key.tasks},0,0,params[4],{key.domain},0,0,{key.core_mask},-1,{{{subcores}}}}};
    err = ioctl(fd,{_SUBMIT_REQUEST}UL,&submit);
    clock_gettime(1,t1);
    stats[0] += (U64)(t1[0]-t0[0])*1000000000ULL + (U64)(t1[1]-t0[1]);
    stats[1] += 1;
    stats[2] += {key.tasks};
  }}"""
    return f"""#pragma clang fp contract(off)
{_QZ_HELPER}
typedef unsigned int U32;
typedef unsigned long long U64;
extern int ioctl(int,unsigned long,...);
extern int clock_gettime(int,void*);{(POOL_TYPES if (qpool or dpool) else "")}{qz_prelude}{dq_prelude}
struct Sync {{U32 flags,reserved;U64 object,offset,size;}} __attribute__((packed));
struct Submit {{U32 flags,timeout,start,number,counter;int priority;U64 object,regcfg,base,user;
               U32 mask;int fence;U32 subcores[10];}} __attribute__((packed));
static __attribute__((always_inline)) inline int sync_mem(int fd,U32 flags,U64 object,U64 size) {{
  struct Sync sync={{flags,0,object,0,size}};
  return ioctl(fd,{_SYNC_REQUEST}UL,&sync);
}}
void rockchip_linear(void *args) {{
  float *out = *(float**)args;
  const float *x = *((const float**)args + 1);
  const U64 *params = *((const U64**)args + 2);
  int fd = (int)params[7];
  U64 *stats = (U64*)params[8];
  unsigned char *payload = (unsigned char*)params[0];
  const int *raw = (const int*)params[1];
  const float *w_scale = (const float*)params[2];
  volatile int *status = (volatile int*)params[3];
  float *scales = (float*)payload;
  signed char *codes = (signed char*)(payload + {key.header});
  int err = 0;
  if ((int)params[9] == 0) {{ long q0[2], q1[2];
  clock_gettime(1,q0);{qz_body}
  clock_gettime(1,q1);
  stats[4] += (U64)(q1[0]-q0[0])*1000000000ULL + (U64)(q1[1]-q0[1]); }}
{sync_to}
{submit_block}
{sync_from}
  if (err) {{ *status = err; return; }}
  {{ long d0[2], d1[2];
  clock_gettime(1,d0);{planes}
  clock_gettime(1,d1);
  stats[6] += (U64)(d1[0]-d0[0])*1000000000ULL + (U64)(d1[1]-d0[1]); }}
}}{qz_group}{dq_group}
"""


# One PROGRAM per geometry. Rebuilding it costs 0.26 ms and this runs 323 times per decode
# token; the key holds no addresses, so every Linear of the same shape shares one PROGRAM and
# one on-disk compile.
_FUSED_PROGRAMS: dict[_FusedKey, UOp] = {}


def _fused_program(key: _FusedKey) -> UOp:
    if (prog := _FUSED_PROGRAMS.get(key)) is None:
        source = _fused_source(key)
        out = UOp.placeholder((key.m * key.n,), dtypes.float, 0)
        inp = UOp.placeholder((key.row_stride * (key.row_offset + key.m),), dtypes.float, 1)
        params = UOp.placeholder((_PARAMS,), dtypes.uint64, 2)
        sink = UOp.sink(out, inp, params, arg=KernelInfo(f"rockchip_linear_{key.m}_{key.n}_{key.k}"))
        renderer = _cpu_renderer()
        info = replace(ProgramInfo.from_sink(sink, renderer.target), outs=(0,), ins=(1, 2))
        _FUSED_PROGRAMS[key] = prog = UOp(Ops.PROGRAM, arg=info, src=(
            sink, UOp(Ops.LINEAR, src=(out, inp, params)),
            UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source))))
    return prog


class _FusedInt8Plan:
    """One compiled C function plus its command chain for one fused INT8 Linear.

    The C function writes the quantized activation into an NPU payload mapping and reads the
    int32 planes back out of another, so the plan owns the addresses that make those work.
    They travel in a parameter block rather than in the source, so the source stays a function
    of the geometry alone. The two scratch buffers come from the geometry pools below, so 249
    ffn plans share one plane buffer.
    """

    def __init__(self, dev, m: int, n: int, k: int, depth_in: int, cores: int, column_offset: int,
                 row_stride: int, row_offset: int, k_offset: int, weight_dma: int, w_scale_addr: int,
                 m_rows: int = 0, raw_out: bool = False):
        self.status = ctypes.c_int(0)
        rows_total = m_rows or m
        self.shape = (rows_total, n)
        header = round_up(rows_total * 4, 64)
        self.payload = _payload_buffer(dev, rows_total, k)
        self.out = _plane_buffer(dev, rows_total, n)
        if m_rows and m_rows == 2 * m:
          # The payload stores quantised rows k-wide (`codes + row*k`), NOT at the source
          # stride. `m * row_stride` walked past the allocation whenever row_stride != k, which
          # is exactly the ffn_down prefill case (row_stride 9216, k tile 4096): measured 61,952
          # bytes out of bounds, and every one of the 163,840 values in rows 64-127 differed
          # from a plain m=64 plan (median relative error 1.42). `m * k` is row m's start.
          cmd, task, tasks, core_mask, subcores, parts = int8_task_plan_multi(dev, (
              (m, n, k, self.payload.dma + header, weight_dma + column_offset * k, self.out.dma),
              (m, n, k, self.payload.dma + header + m * k,
                          weight_dma + column_offset * k, self.out.dma + m * n * 4)))
          parts = parts[0]
        else:
          cmd, task, tasks, core_mask, subcores, parts = int8_task_plan(
              dev, m, n, k, cores, self.out.dma, self.payload.dma + header, weight_dma + column_offset * k)
        self.owned = (cmd, task)
        self.key = _FusedKey(
            m=m, n=n, k=k, depth_in=depth_in, header=header, row_stride=row_stride, row_offset=row_offset,
            k_offset=k_offset, column_offset=column_offset, parts=parts,
            payload_size=self.payload.size, out_size=self.out.size, domain=dev.domain,
            core_mask=core_mask, subcores=tuple(value for pair in subcores for value in pair), tasks=tasks,
            m_rows=m_rows, raw_out=raw_out)
        self.program = _fused_program(self.key)
        # payload, planes, w_scale, status, task array, payload obj_addr, planes obj_addr, fd,
        # device counter block. A raw BUFFER, not Tensor.empty().realize(): the model's forward
        # runs inside a @function capture, where forcing a realize would execute eagerly.
        self.params = UOp.new_buffer("CPU", _PARAMS, dtypes.uint64)
        ctypes.memmove(self.params.buffer._buf, struct.pack("<10Q", self.payload.addr, self.out.addr,
                       w_scale_addr, ctypes.addressof(self.status), task.meta.obj_addr,
                       self.payload.meta.obj_addr, self.out.meta.obj_addr, dev.fd.fd,
                       dev.fused_stats_addr, 0), _PARAMS * 8)

    def _row_key(self, x: Tensor):
      """DISABLED by measurement: sharing saved no wall time.

      Skipping a duplicate qz_row leaves the sync_mem and the submit on that payload, so the
      14.4 ms of duplicated quantise did not reach the clock: measured 1,265.70 ms with sharing
      vs 1,259.09 without (+6.61 ms, i.e. noise on the wrong side). Kept, disabled, so the
      experiment and its verdict are not lost.
      """
      return None

    def _row_key_impl(self, x: Tensor):
      """(payload, input buffer + version, row_offset, k_offset, depth) or None if unknowable.

      Same-(m,k) plans share one payload buffer, so four same-row consumers per layer each
      re-quantise an identical row (96 calls x 0.150 ms). The version is part of the key so a
      rewritten activation can never produce a false hit.
      """
      try:
        b = x.uop.buffer
        return (self.payload.addr, id(b), int(b.meta.version), self.key.row_offset,
                self.key.k_offset, self.key.depth_in)
      except Exception:
        return None

    def __call__(self, x: Tensor) -> Tensor:
        # The ioctl runs at realize time, so a failure is only visible here, on the next call.
        # Failing loud on the next Linear is better than silently returning zeroes.
        if (err := self.status.value) != 0:
            self.status.value = 0
            raise RuntimeError(f"fused Rockchip Linear ioctl failed with {err}")
        # UOp.alloc + Tensor rather than Tensor.empty: the wrapper's dtype/device canonicalisation
        # is 0.03 ms and this runs 323 times per decode token. 2-D so no reshape is needed.
        skip = 0
        k = self._row_key(x)
        if k is not None:
          skip = 1 if _QUANTISED.get(k) is not None else 0
          _QUANTISED[k] = True
          ctypes.memmove(ctypes.addressof(ctypes.c_uint64.from_buffer(self.params.buffer._buf)) + 9*8,
                         ctypes.byref(ctypes.c_uint64(skip)), 8)
        out = Tensor(UOp.alloc(self.shape, dtypes.float, device=x.device, bind_on_realize=True))
        # custom_kernel would rebuild the placeholders (0.16 ms for two buffers) and throw them
        # away, because the fxn ignores them and returns the memoised PROGRAM. Bind the CALL
        # directly: 0.02 ms instead of 0.28 ms, 323 times per decode token.
        return Tensor(out.uop.after(self.program.call(out.uop, x.uop, self.params)))


# Plans outlive the model: the command chain is a few KB and the NPU device frees every
# allocation it owns at finalize. Keyed by id(dev) so a stale plan can never be hit by a new
# device that reused an address.
_FUSED_PLANS: dict[tuple, _FusedInt8Plan] = {}
# which activation rows already have their codes in a given payload (see _row_key)
_QUANTISED: dict[tuple, bool] = {}

# Scratch is shared per geometry rather than per Linear: a per-plan plane buffer would cost
# 2.4 MB for every 64-row ffn plan, i.e. ~600 MB for the 4B. Pooled, the whole model needs one
# payload and one plane buffer per (m, k) and (m, n). Calls are serial and blocking, so no two
# plans are ever in flight together.
_PAYLOAD_POOL: dict[tuple[int, int, int], Any] = {}
_PLANE_POOL: dict[tuple[int, int, int], Any] = {}

# Device access is forbidden inside a @function context (tinygrad/function.py sets
# ALLOW_DEVICE_USAGE=0) and the model's forward runs in one. The device is already open -- the
# packed weight lives on it -- so this is a dict lookup cached per name; tinygrad/llm/kernels/
# amd.py uses the same escape hatch.
_DEVICES: dict[str, Any] = {}


def _device(name: str):
    if (dev := _DEVICES.get(name)) is None:
        with Context(ALLOW_DEVICE_USAGE=1):
            _DEVICES[name] = dev = Device[name]
    return dev


def _payload_buffer(dev, m: int, k: int):
    """fp32 per-row scales in a 64-aligned header, then the int8 codes, in one NPU mapping."""
    key = (id(dev), m, k)
    if (buf := _PAYLOAD_POOL.get(key)) is None:
        _PAYLOAD_POOL[key] = buf = dev.allocate(round_up(m * 4, 64) + m * k)
    return buf


def _plane_buffer(dev, m: int, n: int):
    """Per-core int32 planes, read straight back by the fused dequantize."""
    key = (id(dev), m, n)
    if (buf := _PLANE_POOL.get(key)) is None:
        _PLANE_POOL[key] = buf = dev.allocate(m * n * 4)
        # New cacheable GEM pages can carry dirty zeroes: clean before the NPU writes, or the
        # following invalidation overwrites the result with zeroes. Later calls only invalidate.
        dev.sync(buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
    return buf


def _fused_plan(dev, m: int, n: int, k: int, depth_in: int, cores: int, column_offset: int,
                row_stride: int, row_offset: int, k_offset: int, weight: Tensor, w_scale: Tensor) -> _FusedInt8Plan:
    weight_dma = weight.uop.buffer._buf.dma
    w_scale_addr = w_scale.uop.buffer._buf
    # Merge only when the tile really splits 3 ways: _matmul_parts collapses to a single part for
    # narrow n (e.g. ssm_beta, n=num_v_heads=32), and the two-source chain needs 3 parts per source.
    m_task, m_rows = ((64, m) if (m == 128 and k > 2048
                                 and len(_matmul_parts(64, n, k, 3, True)) == 3) else (m, 0))
    key = (id(dev), m, n, k, depth_in, cores, column_offset, row_stride, row_offset, k_offset,
           weight_dma, w_scale_addr)
    if (plan := _FUSED_PLANS.get(key)) is None:
        # Building a plan allocates, so it needs the device; the first call happens inside the
        # model's @function capture, which forbids that. Once per shape, then cached.
        with Context(ALLOW_DEVICE_USAGE=1):
            _FUSED_PLANS[key] = plan = _FusedInt8Plan(dev, m_task, n, k, depth_in, cores, column_offset,
                                                      row_stride, row_offset, k_offset, weight_dma, w_scale_addr,
                                                      m_rows=m_rows)
    return plan


# Per-output-width core split, for in-model A/B only (cores=3 is the shipped value everywhere).
# The fused forward hard-codes the default, so this is the single knob a harness can flip without
# touching the CBUF register recipe (which wedged the board at k=9216).
CORES_OVERRIDE: dict[int, int] = {}
# A/B switch: False routes NPULinear through the raw `packed_matmul_int8` (separate quantise,
# submit, dequantise) instead of the fused C wrapper, so the k-proportional excess on
# 2560x4096 can be attributed to the wrapper or to the device submit.
FUSED_CALL = True
# A/B switch for the attn_qkv+attn_gate fusion; decided by extra/rockchip/prefill_ab.py.
FUSE_QKV_GATE = False


def packed_matmul_int8_fused(dense: Tensor, weight: Tensor, w_scale: Tensor, columns: int, device: str,
                             m: int, row_offset: int = 0, k_offset: int = 0, column_offset: int = 0,
                             cores: int = 3) -> Tensor:
    """One INT8 matmul as a single C call: quantize, NPU submit, dequantize.

    `dense` is the whole (rows, depth) activation and `row_offset`/`k_offset`/`m` select the
    tile, so a ragged K tile never has to be materialised as a strided host copy. Bit-exact
    with `packed_matmul_int8`: same per-row max and 1e-8 floor, same `__builtin_nearbyintf`
    codes, same unfused `(float)src * a * w_scale` chain.
    """
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    depth_in = max(0, min(k, dense.shape[1] - k_offset))
    if w_scale.shape[0] < column_offset + n:
        raise ValueError(f"w_scale has {w_scale.shape[0]} entries but the call needs {column_offset + n}")
    plan = _fused_plan(_device(device), m, n, k, depth_in, cores, column_offset, dense.shape[1],
                       row_offset, k_offset, weight, w_scale)
    return plan(dense)


class NPULinear:
    """Drop-in nn.Linear replacement; weights packed once, rows, columns and K tiled."""

    def __init__(self, linear: nn.Linear):
        self.weight, self.bias = linear.weight, linear.bias
        self.columns, self.depth = self.weight.shape
        self.packed: list[tuple[int, int, Tensor, Tensor]] = []
        self.device = "ROCKCHIP"

    def prepare(self, domains: _Domains) -> None:
        if self.weight is None:
            return
        self.packed.clear()
        self.device = domains.place(round_up(self.columns, 32) * round_up(self.depth, 32))
        for c0 in range(0, self.columns, TILE):
            n = min(TILE, self.columns - c0)
            for k0 in range(0, self.depth, K_TILE):
                k = min(K_TILE, self.depth - k0)
                tile = self.weight[c0:c0 + n, k0:k0 + k].contiguous()
                # INT8 only accepts K on a 512 multiple, so a ragged K tile is padded.
                padded = _int8_padded_depth(k)
                codes, scales = pack_weight_int8(tile.pad(((0, 0), (0, padded - k))), self.device)
                self.packed.append((c0, k0, codes.realize(), scales.realize()))
        self.weight = None  # packed tiles carry everything; the source is the largest live copy

    def _row_key(self, x: Tensor):
      """DISABLED by measurement: sharing saved no wall time.

      Skipping a duplicate qz_row leaves the sync_mem and the submit on that payload, so the
      14.4 ms of duplicated quantise did not reach the clock: measured 1,265.70 ms with sharing
      vs 1,259.09 without (+6.61 ms, i.e. noise on the wrong side). Kept, disabled, so the
      experiment and its verdict are not lost.
      """
      return None

    def _row_key_impl(self, x: Tensor):
      """(payload, input buffer + version, row_offset, k_offset, depth) or None if unknowable.

      Same-(m,k) plans share one payload buffer, so four same-row consumers per layer each
      re-quantise an identical row (96 calls x 0.150 ms). The version is part of the key so a
      rewritten activation can never produce a false hit.
      """
      try:
        b = x.uop.buffer
        return (self.payload.addr, id(b), int(b.meta.version), self.key.row_offset,
                self.key.k_offset, self.key.depth_in)
      except Exception:
        return None

    def __call__(self, x: Tensor) -> Tensor:
        shape = x.shape
        # One contiguous (rows, depth) view for the whole call: the fused INT8 path bakes the
        # row and K offsets into the kernel instead of materialising a strided tile.
        dense = x.reshape(-1, shape[-1]).contiguous()
        rows = dense.shape[0]
        k_max = round_up(min(K_TILE, self.depth), 32)
        # 65536 is the FP16 CBUF bound (banks = ceil(m*k*2/32768) <= 4). The INT8
        # recipe has no such limit: _int8_matmul_body accepts m<=128 at k<=2048 and
        # m<=64 above that, exact at (64, 2560, 4096) in smoke_int8.py. Applying the
        # FP16 bound to INT8 made a 128-token prefill take 8 weight passes instead
        # of 2, which is why our prefill NPU time exceeds the reference's whole run.
        row_tile = 128 if k_max <= 2048 else 64
        out_rows = []
        # Only merge when the tile splits 3 ways: narrow n (e.g. ssm_beta, n=32) yields a single
        # part, and a 128-row task is illegal at k>2048 anyway.
        merge = (2 * row_tile if (rows == 2 * row_tile
                                  and len(_matmul_parts(row_tile, self.columns, k_max, 3, True)) == 3) else 0)
        step = merge or row_tile
        for r0 in range(0, rows, step):
            part = dense[r0:r0 + step]
            width = 24576 if part.shape[0] == 1 else 12288
            pieces = []
            for c0 in range(0, self.columns, TILE):
                n = min(TILE, self.columns - c0)
                for col in range(0, n, width):
                    w2 = min(width, n - col)
                    acc = None
                    for pc, pk, weight, scales in self.packed:
                        if pc != c0:
                            continue
                        # One C call: quantize + direct ioctl + dequantize. Measured
                        # 1.45-1.60 -> 0.150-0.208 ms of lazy construction per call (8.3-9.7x)
                        # and Qwen2 0.5B paired decode 3.64 -> 7.95 tok/s. Bit-exact at every
                        # 4B geometry (parity_shapes.py, 9/9).
                        if FUSED_CALL:
                          piece = packed_matmul_int8_fused(dense, weight, scales, w2, self.device,
                                                           part.shape[0], row_offset=r0, k_offset=pk,
                                                           column_offset=col,
                                                           cores=CORES_OVERRIDE.get(w2, 3))
                        else:
                          piece = packed_matmul_int8(part, weight, scales, w2, self.device,
                                                     cores=CORES_OVERRIDE.get(w2, 3), column_offset=col)
                        acc = piece if acc is None else acc + piece
                    if acc is None:
                        raise RuntimeError("NPULinear.prepare() was not called")
                    pieces.append(acc)
            out_rows.append(Tensor.cat(*pieces, dim=1) if len(pieces) > 1 else pieces[0])
        out = Tensor.cat(*out_rows, dim=0) if len(out_rows) > 1 else out_rows[0]
        out = out.reshape(*shape[:-1], self.columns)
        return out if self.bias is None else out + self.bias


def _walk(obj, visit) -> int:
    """Call visit on every Linear reachable from `obj`; returns the count."""
    count = 0

    def walk(node) -> None:
        nonlocal count
        for name, value in list(vars(node).items()):
            if isinstance(value, (nn.Linear, NPULinear)):
                new = visit(value)
                if new is not value:
                    setattr(node, name, new)
                count += 1
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, (nn.Linear, NPULinear)):
                        new = visit(item)
                        if new is not item:
                            value[index] = new
                        count += 1
                    elif hasattr(item, "__dict__"):
                        walk(item)
            elif hasattr(value, "__dict__") and not isinstance(value, (Tensor, dict, str, int, float)):
                walk(value)

    walk(obj)
    return count


def _cat_linears(parts: list) -> nn.Linear:
    """One Linear whose rows are the parts' rows, so `cat(parts_out) == fused(x)`.

    Built with `object.__new__` on purpose: `nn.Linear.__init__` would allocate and fill a random
    (sum(rows) x in) matrix first, which for the qkv group is 10240x2560 of pointless work and
    memory per layer.
    """
    weight = Tensor.cat(*[p.weight for p in parts], dim=0).realize()
    if all(p.bias is not None for p in parts):
        bias = Tensor.cat(*[p.bias for p in parts], dim=0).realize()
    elif any(p.bias is not None for p in parts):
        raise ValueError("fused Linears must agree on whether they have a bias")
    else:
        bias = None
    fused = object.__new__(nn.Linear)
    fused.weight, fused.bias = weight, bias
    return fused


def fuse_linears(model) -> int:
    """Fuse `attn_q`/`attn_k`/`attn_v` and `ffn_gate`/`ffn_up` into single Linears.

    **MEASURED AND REJECTED -- do not call this.** With 40 groups fused, prefill went
    2,771 -> 3,179 ms (+14.7%) while submits only fell 635 -> 603 and decode improved 3.10 -> 3.14
    tok/s. The fused gate/up weight is 19456 columns, so `prepare` packs a single 24576-column tile
    and the NPU does the same arithmetic over a 24.5%-wider tile; above the 12288 prefill limit the
    call still splits into two submits, so the submit saving never materialises. The float-path
    fusion itself is exact (weights and outputs bit-identical, verified), so this is a tiling
    result, not a correctness one. Kept because the qkv half alone may pay at m==1; re-measure with
    qkv fusion only before using it.

    Why this is exact rather than approximate: `pack_weight_int8` gives **one fp32 scale per output
    row** (`quantize_rows` on the weight), and TILE is 24576 columns, larger than the fused qkv
    width of 10240. So every output row keeps its own scale through the concatenation and the fused
    GEMM produces, column for column, the same codes, scales and K-reduction order as the three
    separate ones. The activation is also quantised **once instead of three times** (same x for all
    three projections), which is exact because the per-row scale depends only on that row of x.

    `ffn_gate`/`ffn_up` are fused too. Their combined width is 19456: above the 12288 three-core
    limit, so prefill splits it back into two submits (same count, but one activation quantisation
    instead of two), and at m==1 the limit is 24576 so decode gets **one** submit instead of two --
    32 fewer submits per token across the 32 blocks.

    Call after the GGUF weights are loaded and before `replace_linears`, so the fused Linear is
    wrapped (and packed) like any other.
    """
    count = 0

    def fuse(node) -> None:
        nonlocal count
        for value in list(vars(node).values()):
            if isinstance(value, (nn.Linear, NPULinear, Tensor, dict, str, int, float)):
                continue
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "__dict__") and not isinstance(item, (nn.Linear, NPULinear, Tensor)):
                        fuse(item)
            elif hasattr(value, "__dict__"):
                fuse(value)
        # `not hasattr(node, "attn_qkv")` protects blocks that already own a native fused qkv
        # (GatedDeltaNetBlock builds one unless it is KDA, and overwriting it would be a bug).
        if (all(hasattr(node, name) for name in ("attn_q", "attn_k", "attn_v"))
                and not hasattr(node, "attn_qkv")):
            parts = [node.attn_q, node.attn_k, node.attn_v]
            node._qkv_widths = tuple(p.weight.shape[0] for p in parts)
            node.attn_qkv = _cat_linears(parts)
            for name in ("attn_q", "attn_k", "attn_v"):
                delattr(node, name)
            count += 1
        # attn_qkv + attn_gate: 4096 + 8192 = 12288, exactly the prefill width limit, so unlike
        # gate/up (19456) it fits one prefill submit. Decode measured -0.71 ms and 283->259 submits.
        # Gated by FUSE_QKV_GATE because a one-sample prefill spot-check moved the wrong way; the
        # A/B that decides it is extra/rockchip/prefill_ab.py.
        if (FUSE_QKV_GATE and all(hasattr(node, name) for name in ("attn_qkv", "attn_gate"))
                and not hasattr(node, "attn_qkv_gate")):
            qkv, gate = node.attn_qkv, node.attn_gate
            if isinstance(qkv, nn.Linear) and isinstance(gate, nn.Linear) and qkv.weight.shape[1] == gate.weight.shape[1]:
                node._qkv_gate_width = qkv.weight.shape[0]
                node.attn_qkv_gate = _cat_linears([qkv, gate])
                delattr(node, "attn_qkv"); delattr(node, "attn_gate")
                count += 1

        # ssm_alpha + ssm_beta: both are Linear(dim, num_v_heads=32) on the same input, 2 submits
        # per GDN layer = 48 per token at 30 us each. The fused width is 64, far below the 12288
        # prefill width, so unlike gate/up this cannot push prefill over a tiling boundary. Exact
        # for the same reason as qkv: one fp32 scale per output row survives the concatenation.
        if (all(hasattr(node, name) for name in ("ssm_alpha", "ssm_beta"))
                and not hasattr(node, "ssm_alpha_beta")):
            alpha, beta = node.ssm_alpha, node.ssm_beta
            # isinstance, not `type(...) is nn.Linear`: this model's projections are
            # `tinygrad.llm.kernels.amd.Linear`, a subclass, so an exact-type check silently
            # matched nothing (measured: 0 of 24 blocks fused).
            if isinstance(alpha, nn.Linear) and isinstance(beta, nn.Linear) and alpha.weight.shape == beta.weight.shape:
                node._alpha_beta_width = alpha.weight.shape[0]
                node.ssm_alpha_beta = _cat_linears([alpha, beta])
                delattr(node, "ssm_alpha"); delattr(node, "ssm_beta")
                count += 1

        # gate/up fusion DISABLED: measured +14.7% prefill (n=19456 exceeds the 12288 prefill width,
        # so it still splits, and prepare packs it as one 24576-column tile). qkv alone is measured
        # below because it fits (n=10240).
        if False and all(hasattr(node, name) for name in ("ffn_gate", "ffn_up")):
            parts = [node.ffn_gate, node.ffn_up]
            node._gateup_width = parts[0].weight.shape[0]
            node.ffn_gateup = _cat_linears(parts)
            for name in ("ffn_gate", "ffn_up"):
                delattr(node, name)
            count += 1

    fuse(model)
    return count


def replace_linears(model) -> int:
    """Swap every nn.Linear reachable from `model` for NPULinear; returns the count."""
    return _walk(model, lambda linear: linear if isinstance(linear, NPULinear) else NPULinear(linear))


def prepare_linears(model) -> int:
    """Pack weights for every NPULinear reachable from `model`; returns the count."""
    def prepare(linear):
        if isinstance(linear, NPULinear):
            linear.prepare(_DOMAINS)
        return linear
    return _walk(model, prepare)


def packed_weight_bytes(model) -> int:
    """Total packed weight bytes reachable from `model`, for the bandwidth budget."""
    total = 0

    def measure(linear):
        nonlocal total
        if isinstance(linear, NPULinear):
            total += sum(w.numel() * w.dtype.itemsize for _, _, w, _ in linear.packed)
        return linear

    _walk(model, measure)
    return total
