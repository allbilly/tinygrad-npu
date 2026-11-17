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
from tinygrad.runtime.ops_rockchip import _int8_padded_depth, _matmul_parts, int8_task_plan
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


def _matmul(out: UOp, lhs: UOp, rhs: UOp, m: int, n: int, k: int, cores: int, column_offset: int) -> UOp:
    sink = UOp.sink(out, lhs, rhs, arg=KernelInfo(f"rockchip_matmul_{m}_{n}_{k}"))
    info = replace(ProgramInfo.from_sink(sink, Target("ROCKCHIP")), outs=(0,), ins=(1, 2))
    return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, lhs, rhs)),
               UOp(Ops.SOURCE, arg=f"Rockchip FP16 {m}x{k}x{n}"),
               UOp(Ops.BINARY, arg=struct.pack("<7I", m, n, k, cores, column_offset * k * 2, 0, 0))))


def pack_weight(weight: Tensor, device: str) -> Tensor:
    n, k = weight.shape
    n, k = round_up(n, 32), round_up(k, 32)
    return weight.half().pad(((0, n - weight.shape[0]), (0, k - weight.shape[1]))).reshape(n // 16, 16, k // 32, 32) \
        .permute(0, 2, 1, 3).contiguous().to(device)


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
  addresses; the geometry is baked in, the way _matmul bakes it into its descriptor."""
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
_PARAMS = 9


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
  for (int row = 0; row < {key.m}; row++) {{
    const float *xr = x + (long)({key.row_offset}+row)*{key.row_stride} + {key.k_offset};
    float largest = 0.0f;
    for (int i = 0; i < {key.depth_in}; i++) {{ float a = xr[i] < 0.0f ? -xr[i] : xr[i]; if (a > largest) largest = a; }}
    float scale = largest / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    scales[row] = scale;
    float inv = 1.0f / scale;
    signed char *o = codes + (long)row*{key.k};
    for (int i = 0; i < {key.depth_in}; i++) {{
      float q = __builtin_nearbyintf(xr[i]*inv);
      if (q > 127.0f) q = 127.0f;
      if (q < -127.0f) q = -127.0f;
      o[i] = (signed char)q;
    }}
    for (int i = {key.depth_in}; i < {key.k}; i++) o[i] = 0;
  }}"""
    planes = "".join(f"""
  for (int row = 0; row < {key.m}; row++) {{
    const int *src = raw + {offset * key.m} + (long)row*{width};
    float *dst = out + (long)row*{key.n} + {offset};
    float a = scales[row];
    for (int j = 0; j < {width}; j++) dst[j] = (float)src[j] * a * w_scale[{key.column_offset + offset} + j];
  }}""" for offset, width in key.parts)
    subcores = ",".join(str(value) for value in key.subcores)
    return f"""#pragma clang fp contract(off)
typedef unsigned int U32;
typedef unsigned long long U64;
extern int ioctl(int,unsigned long,...);
extern int clock_gettime(int,void*);
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
  int err = 0;{quantize}
  if (!err) err = sync_mem(fd,1,params[5],{key.payload_size}ULL);
  if (!err) {{
    long t0[2], t1[2];
    clock_gettime(1,t0);
    struct Submit submit={{5,6000,0,{key.tasks},0,0,params[4],{key.domain},0,0,{key.core_mask},-1,{{{subcores}}}}};
    err = ioctl(fd,{_SUBMIT_REQUEST}UL,&submit);
    clock_gettime(1,t1);
    stats[0] += (U64)(t1[0]-t0[0])*1000000000ULL + (U64)(t1[1]-t0[1]);
    stats[1] += 1;
    stats[2] += {key.tasks};
  }}
  if (!err) err = sync_mem(fd,2,params[6],{key.out_size}ULL);
  if (err) {{ *status = err; return; }}{planes}
}}
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
                 row_stride: int, row_offset: int, k_offset: int, weight_dma: int, w_scale_addr: int):
        self.status = ctypes.c_int(0)
        self.shape = (m, n)
        header = round_up(m * 4, 64)
        self.payload = _payload_buffer(dev, m, k)
        self.out = _plane_buffer(dev, m, n)
        cmd, task, tasks, core_mask, subcores, parts = int8_task_plan(
            dev, m, n, k, cores, self.out.dma, self.payload.dma + header, weight_dma + column_offset * k)
        self.owned = (cmd, task)
        self.key = _FusedKey(
            m=m, n=n, k=k, depth_in=depth_in, header=header, row_stride=row_stride, row_offset=row_offset,
            k_offset=k_offset, column_offset=column_offset, parts=parts,
            payload_size=self.payload.size, out_size=self.out.size, domain=dev.domain,
            core_mask=core_mask, subcores=tuple(value for pair in subcores for value in pair), tasks=tasks)
        self.program = _fused_program(self.key)
        # payload, planes, w_scale, status, task array, payload obj_addr, planes obj_addr, fd,
        # device counter block. A raw BUFFER, not Tensor.empty().realize(): the model's forward
        # runs inside a @function capture, where forcing a realize would execute eagerly.
        self.params = UOp.new_buffer("CPU", _PARAMS, dtypes.uint64)
        ctypes.memmove(self.params.buffer._buf, struct.pack("<9Q", self.payload.addr, self.out.addr,
                       w_scale_addr, ctypes.addressof(self.status), task.meta.obj_addr,
                       self.payload.meta.obj_addr, self.out.meta.obj_addr, dev.fd.fd,
                       dev.fused_stats_addr), _PARAMS * 8)

    def __call__(self, x: Tensor) -> Tensor:
        # The ioctl runs at realize time, so a failure is only visible here, on the next call.
        # Failing loud on the next Linear is better than silently returning zeroes.
        if (err := self.status.value) != 0:
            self.status.value = 0
            raise RuntimeError(f"fused Rockchip Linear ioctl failed with {err}")
        # UOp.alloc + Tensor rather than Tensor.empty: the wrapper's dtype/device canonicalisation
        # is 0.03 ms and this runs 323 times per decode token. 2-D so no reshape is needed.
        out = Tensor(UOp.alloc(self.shape, dtypes.float, device=x.device, bind_on_realize=True))
        # custom_kernel would rebuild the placeholders (0.16 ms for two buffers) and throw them
        # away, because the fxn ignores them and returns the memoised PROGRAM. Bind the CALL
        # directly: 0.02 ms instead of 0.28 ms, 323 times per decode token.
        return Tensor(out.uop.after(self.program.call(out.uop, x.uop, self.params)))


# Plans outlive the model: the command chain is a few KB and the NPU device frees every
# allocation it owns at finalize. Keyed by id(dev) so a stale plan can never be hit by a new
# device that reused an address.
_FUSED_PLANS: dict[tuple, _FusedInt8Plan] = {}

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
    key = (id(dev), m, n, k, depth_in, cores, column_offset, row_stride, row_offset, k_offset,
           weight_dma, w_scale_addr)
    if (plan := _FUSED_PLANS.get(key)) is None:
        # Building a plan allocates, so it needs the device; the first call happens inside the
        # model's @function capture, which forbids that. Once per shape, then cached.
        with Context(ALLOW_DEVICE_USAGE=1):
            _FUSED_PLANS[key] = plan = _FusedInt8Plan(dev, m, n, k, depth_in, cores, column_offset,
                                                      row_stride, row_offset, k_offset, weight_dma, w_scale_addr)
    return plan


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


def packed_matmul(x: Tensor, weight: Tensor, columns: int, device: str, cores: int = 3, column_offset: int = 0) -> Tensor:
    m, original_k = x.shape
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    lhs = x.half().pad(((0, 0), (0, k - original_k))).reshape(m, k // 8, 8).permute(1, 0, 2).contiguous().to(device)
    out = Tensor.empty(n // 4, m, 4, device=device, dtype=dtypes.float)
    out = out.custom_kernel(lhs, weight, fxn=lambda o, a, w: _matmul(o, a, w, m, n, k, cores, column_offset))[0]
    return out.to(x.device).permute(1, 0, 2).reshape(m, n)[:, :columns]


class NPULinear:
    """Drop-in nn.Linear replacement; weights packed once, rows, columns and K tiled."""

    def __init__(self, linear: nn.Linear, int8: bool = True):
        self.weight, self.bias = linear.weight, linear.bias
        self.columns, self.depth = self.weight.shape
        self.packed: list[tuple[int, int, Tensor, Tensor | None]] = []
        self.device = "ROCKCHIP"
        self.int8 = int8

    def prepare(self, domains: _Domains) -> None:
        if self.weight is None:
            return
        self.packed.clear()
        itemsize = 1 if self.int8 else 2
        self.device = domains.place(round_up(self.columns, 32) * round_up(self.depth, 32) * itemsize)
        for c0 in range(0, self.columns, TILE):
            n = min(TILE, self.columns - c0)
            for k0 in range(0, self.depth, K_TILE):
                k = min(K_TILE, self.depth - k0)
                tile = self.weight[c0:c0 + n, k0:k0 + k].contiguous()
                if self.int8:
                    # INT8 only accepts K on a 512 multiple, so a ragged K tile is padded.
                    padded = _int8_padded_depth(k)
                    codes, scales = pack_weight_int8(tile.pad(((0, 0), (0, padded - k))), self.device)
                    self.packed.append((c0, k0, codes.realize(), scales.realize()))
                else:
                    # realize per tile: a lazy graph would hold every tile's temporaries at once
                    self.packed.append((c0, k0, pack_weight(tile, self.device).realize(), None))
        self.weight = None  # packed tiles carry everything; the source is the largest live copy

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
        row_tile = (128 if k_max <= 2048 else 64) if self.int8 else max(1, min(64, 65536 // k_max))
        out_rows = []
        for r0 in range(0, rows, row_tile):
            part = dense[r0:r0 + row_tile]
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
                        if self.int8:
                            # One C call: quantize + direct ioctl + dequantize. Measured
                            # 1.45-1.60 -> 0.150-0.208 ms of lazy construction per call
                            # (8.3-9.7x) and Qwen2 0.5B paired decode 3.64 -> 7.95 tok/s.
                            # Bit-exact at every 4B geometry (parity_shapes.py, 9/9).
                            piece = packed_matmul_int8_fused(dense, weight, scales, w2, self.device,
                                                             part.shape[0], row_offset=r0, k_offset=pk,
                                                             column_offset=col)
                        else:
                            k = weight.shape[1] * 32
                            piece = packed_matmul(part[:, pk:pk + k], weight, w2, self.device, column_offset=col)
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
