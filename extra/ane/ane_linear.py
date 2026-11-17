"""Minimal Apple ANE GEMM offload: one ANE matmul per Linear, no fused kernels.

The runtime is the direct-register M1 ANE implementation from the reference
tree (``~/ane`` by default): ``qwen35/ane.py`` compiles ``qwen35/ane_matmul.c``
and opens ``/dev/accel/accel*``. No CoreML, no E5RT, no .mlmodel.

This is the ANE analogue of ``extra/rockchip/npu_linear.py``: every eligible
``nn.Linear`` is replaced by the same generic FP16 GEMM, and everything else --
attention, norms, softmax, rope, embedding, sampling, KV cache -- stays on the
CPU host.

The matmul is a tinygrad ``CUSTOM_FUNCTION`` that the scheduler executes as a
normal graph call. The ANE plan pointer and the native entry point travel in
the program's binary payload, so the CPU operations around the matmul stay
JIT-compiled and the ANE submission is replayed with the graph.
"""
from __future__ import annotations

import ctypes
import math
import os
import pathlib
import struct
import sys
from dataclasses import replace

import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Target, getenv
from tinygrad.uop.ops import KernelInfo, Ops, PatternMatcher, ProgramInfo, UOp, UPat

ANE_REF = pathlib.Path(os.path.expanduser(getenv("ANE_REF", "~/ane")))
ANE_BATCH = 32  # ane_matmul.c BATCH
# The command template is built around a 768-element dimension; smaller
# projections (Qwen3.5's 16-output ssm_alpha/ssm_beta and 512-output
# attn_k/attn_v) stay on the CPU.
ANE_MIN_DIM = int(getenv("ANE_MIN_DIM", 768))
# ane_matmul.c rejects inputs above 32736. Outputs above ANE_MAX_COLS are split
# into chunks; 16384 is the largest width that completed on this M1 (24576
# timed out).
ANE_MAX_DIM = 32736
ANE_MAX_COLS = int(getenv("ANE_MAX_COLS", 16384))
# The M1 ANE times out on K above 4096 (the 2B model's ssm_out/down are K=6144).
ANE_MAX_DEPTH = int(getenv("ANE_MAX_DEPTH", 4096))
# Include the 248320 x 1024 vocabulary head (254M elements) so it can be
# chunked onto the ANE; the 2B model's 508M-element head stays on the CPU.
ANE_MAX_WEIGHT = int(getenv("ANE_MAX_WEIGHT", 600 << 20))  # the 2B 508M head fits; the 4B 1B head does not

_ANE = None
_ARGMAX_PROGRAM = None
_ATTENTION_PROGRAM = None
_GDN_LIB = None
_GDN_FN = None
_GDN_PROGRAM = None
_GATHER_PROGRAM = None
_GATHER_LIB = None
_GATHER_FN = None
_ANE_SCALED_FN = None
_ANE_SCALED_LIB = None
_ANE_ENTRY = None
_ANE_SCALED_ENTRY = None
_GDN_ENTRY = None
_GATHER_ENTRY = None
_ATTENTION_ENTRY = None
_CPU_RENDERER = None
_PLANS: list[tuple[object, object]] = []
_PATCHED = False

_GDN_SRC = """
#pragma clang fp contract(off)
extern float sqrtf(float);
extern float exp2f(float);
extern float log2f(float);
void gdn(float *out, float *state_out, const float *state_in,
         const float *qkv, const float *z, const float *beta, const float *alpha,
         const float *ssm_a, const float *dt_bias, const float *norm,
         const float *qkv_raw, __fp16 *conv_state) {
  for (int h = 0; h < 16; ++h) {
    float q[128], k[128], rs[128], o[128], qs = 0, ks = 0;
    for (int j = 0; j < 128; j += 4) {
      q[j] = qkv[h * 128 + j]; q[j + 1] = qkv[h * 128 + j + 1];
      q[j + 2] = qkv[h * 128 + j + 2]; q[j + 3] = qkv[h * 128 + j + 3];
      k[j] = qkv[2048 + h * 128 + j]; k[j + 1] = qkv[2048 + h * 128 + j + 1];
      k[j + 2] = qkv[2048 + h * 128 + j + 2]; k[j + 3] = qkv[2048 + h * 128 + j + 3];
      qs += q[j] * q[j] + q[j + 1] * q[j + 1] + q[j + 2] * q[j + 2] + q[j + 3] * q[j + 3];
      ks += k[j] * k[j] + k[j + 1] * k[j + 1] + k[j + 2] * k[j + 2] + k[j + 3] * k[j + 3];
    }
    float qd = sqrtf(qs); if (qd < 1e-12f) qd = 1e-12f;
    float kd = sqrtf(ks); if (kd < 1e-12f) kd = 1e-12f;
    for (int j = 0; j < 128; ++j) { q[j] = q[j] / qd; q[j] = q[j] * 0.08838834764831845f; k[j] = k[j] / kd; }
    float bh = 1.f / (1.f + exp2f(-beta[h] * 1.4426950408889634f));
    float a = alpha[h] + dt_bias[h];
    float m = a > 0.f ? a : 0.f;
    float sp = log2f(exp2f((a - m) * 1.4426950408889634f) + exp2f(-m * 1.4426950408889634f)) * 0.6931471805599453f + m;
    float decay = exp2f(ssm_a[h] * sp * 1.4426950408889634f);
    for (int i = 0; i < 128; ++i) {
      const float *si = state_in + ((unsigned long)h * 128 + i) * 128;
      float *so = state_out + ((unsigned long)h * 128 + i) * 128;
      float sk = 0;
      for (int j = 0; j < 128; j += 4) {
        rs[j] = si[j] * decay; rs[j + 1] = si[j + 1] * decay;
        rs[j + 2] = si[j + 2] * decay; rs[j + 3] = si[j + 3] * decay;
        sk += rs[j] * k[j] + rs[j + 1] * k[j + 1] + rs[j + 2] * k[j + 2] + rs[j + 3] * k[j + 3];
      }
      float delta = bh * (qkv[4096 + h * 128 + i] - sk);
      for (int j = 0; j < 128; ++j) { rs[j] = rs[j] + delta * k[j]; so[j] = rs[j]; }
      float sq = 0;
      for (int j = 0; j < 128; j += 4)
        sq += rs[j] * q[j] + rs[j + 1] * q[j + 1] + rs[j + 2] * q[j + 2] + rs[j + 3] * q[j + 3];
      o[i] = sq;
    }
    float sumsq = 0;
    for (int i = 0; i < 128; i += 4)
      sumsq += o[i] * o[i] + o[i + 1] * o[i + 1] + o[i + 2] * o[i + 2] + o[i + 3] * o[i + 3];
    float inv = 1.f / sqrtf(sumsq / 128.f + 1e-6f);
    for (int i = 0; i < 128; ++i) {
      float zz = z[h * 128 + i];
      float gate = zz * (1.f / (1.f + exp2f(-zz * 1.4426950408889634f)));
      out[h * 128 + i] = (o[i] * inv) * norm[i] * gate;
    }
  }
  for (int c = 0; c < 6144; ++c) {
    conv_state[c] = conv_state[6144 + c];
    conv_state[6144 + c] = conv_state[12288 + c];
    conv_state[12288 + c] = (__fp16)qkv_raw[c];
  }
}
float gdn_call(float *out, float *state_out, const float *state_in,
               const float *qkv, const float *z, const float *beta, const float *alpha,
               const float *ssm_a, const float *dt_bias, const float *norm,
               const float *qkv_raw, __fp16 *conv_state) {
  gdn(out, state_out, state_in, qkv, z, beta, alpha, ssm_a, dt_bias, norm, qkv_raw, conv_state);
  return out[0];
}
"""


def _gdn_fn():
    global _GDN_LIB, _GDN_FN
    if _GDN_FN is None:
        import subprocess
        cache = pathlib.Path(os.environ.get("ANE_CACHE_DIR", "~/.cache/ane-gdn")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / "gdn.so"
        if not lib.exists():
            temporary = cache / f"gdn.{os.getpid()}.so"
            subprocess.run(["cc", "-O3", "-ffp-contract=off", "-march=armv8.2-a+fp16", "-fPIC", "-shared",
                        "-x", "c", "-", "-lm", "-o", str(temporary)], input=_GDN_SRC.encode(), check=True)
            temporary.replace(lib)
        _GDN_LIB = ctypes.CDLL(str(lib))
        _GDN_FN = ctypes.cast(_GDN_LIB.gdn_call, ctypes.c_void_p).value
    return _GDN_FN


def _gdn_program():
    global _GDN_PROGRAM
    if _GDN_PROGRAM is None:
        import platform, subprocess
        from tinygrad import Device
        from tinygrad.device import TinyELF
        from tinygrad.helpers import getenv
        from tinygrad.runtime.ops_cpu import CPUProgram
        from tinygrad.runtime.support.compiler_cpu import ClangCompiler
        from tinygrad.runtime.support.elf import jit_loader

        class O3Clang(ClangCompiler):
            def compile_to_obj(self, src:str) -> bytes:
                return subprocess.check_output([getenv("CC", "clang"), "-c", "-x", "c", "-O3", "-fPIC", "-ffreestanding",
                                                "-fno-math-errno", "-nostdlib", "-fno-ident", f"--target={self.arch}-none-unknown-elf",
                                                *self.args, "-", "-o", "-"], input=src.encode("utf-8"))

        arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        lib = jit_loader(O3Clang([arch, "native"]).compile_to_obj(_GDN_SRC), link_libs=["m"])
        signature = tuple((None, i, dtypes.uint64, ()) for i in range(12))
        _GDN_PROGRAM = CPUProgram(Device["CPU"], TinyELF(lib, "gdn_call", Target("CPU"), signature))
    return _GDN_PROGRAM


def _exec_gdn(ctx, call, ast):
    from tinygrad.engine.realize import _resolve, track_stats

    def resolve(b):
        while b.op is Ops.AFTER: b = b.src[0]
        return _resolve(b, ctx.input_uops)
    bufs = [resolve(b).buffer.ensure_allocated() for b in call.src[1:13]]
    global _GDN_ENTRY
    if _GDN_ENTRY is None:
        _GDN_ENTRY = ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * 12)(_gdn_fn())
    with track_stats(ctx, call, bufs[0].device, bufs, ctx.var_vals):
        _GDN_ENTRY(*[ctypes.c_void_p(b._buf.va_addr) for b in bufs])
    return None


def fast_gdn(state: Tensor, qkv: Tensor, z: Tensor, beta: Tensor, alpha: Tensor,
             a_log: Tensor, dt_bias: Tensor, norm: Tensor, qkv_raw: Tensor, conv_state: Tensor):
    """Qwen3.5 GatedDeltaNet recurrence in one C loop; updates state in place and returns the output."""
    out = Tensor.empty(2048, device=state.device, dtype=dtypes.float32)
    qkv, z, beta, alpha, a_log, dt_bias, norm = (t.float().contiguous() for t in (qkv, z, beta, alpha, a_log, dt_bias, norm))
    call = UOp(Ops.CUSTOM_FUNCTION, src=(), arg="gdn").call(
        out.uop, state.uop, state.uop, qkv.uop, z.uop, beta.uop, alpha.uop, a_log.uop, dt_bias.uop, norm.uop,
        qkv_raw.uop, conv_state.uop)
    return Tensor(out.uop.after(call)).reshape(1, 1, 2048)

_ATTENTION_SRC = """
extern float expf(float);
extern float fmaxf(float, float);
static float bfloat(float x) {
  union { float f; unsigned int u; } v; v.f = x;
  v.u = (v.u + 0x7fff + ((v.u >> 16) & 1)) & 0xffff0000;
  return v.f;
}
void attention(float *out, const float *q, const float *keys, const float *values,
               const float *gate, const int *length_ptr) {
  int length = *length_ptr;
  for (int h = 0; h < 8; ++h) {
    float query[256], acc[256] = {0}, maximum = -__builtin_inff(), total = 0;
    for (int j = 0; j < 256; ++j) query[j] = q[h * 256 + j] / 16.f;
    for (int t = 0; t < length; ++t) {
      const float *k = keys + (unsigned long)t * 512 + (h / 4) * 256;
      const float *v = values + (unsigned long)t * 512 + (h / 4) * 256;
      float score = 0;
      for (int j = 0; j < 256; ++j) score += query[j] * k[j];
      float next = fmaxf(maximum, score), factor = expf(maximum - next), weight = expf(score - next);
      total = total * factor + weight;
      for (int j = 0; j < 256; ++j) acc[j] = acc[j] * factor + weight * v[j];
      maximum = next;
    }
    for (int j = 0; j < 256; ++j) {
      float value = acc[j] / total;
      float sigmoid = 1.f / (1.f + expf(-gate[h * 256 + j]));
      out[h * 256 + j] = value * sigmoid;
    }
  }
}
"""


def _attention_program():
    global _ATTENTION_PROGRAM
    if _ATTENTION_PROGRAM is None:
        import platform
        from tinygrad import Device, dtypes
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_cpu import CPUProgram
        from tinygrad.runtime.support.compiler_cpu import ClangCompiler

        from tinygrad.runtime.support.elf import jit_loader

        arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        lib = jit_loader(ClangCompiler([arch, "native"]).compile_to_obj(_ATTENTION_SRC), link_libs=["m"])
        signature = ((None, 0, dtypes.uint64, ()), (None, 1, dtypes.uint64, ()), (None, 2, dtypes.uint64, ()),
                     (None, 3, dtypes.uint64, ()), (None, 4, dtypes.uint64, ()), (None, 5, dtypes.uint64, ()))
        _ATTENTION_PROGRAM = CPUProgram(Device["CPU"], TinyELF(lib, "attention", Target("CPU"), signature))
    return _ATTENTION_PROGRAM


def _exec_attention(ctx, call, ast):
    from tinygrad.engine.realize import _resolve, track_stats

    bufs = [_resolve(b, ctx.input_uops).buffer.ensure_allocated() for b in call.src[1:7]]
    with track_stats(ctx, call, bufs[0].device, bufs, ctx.var_vals):
        _attention_program()(bufs[0]._buf, bufs[1]._buf, bufs[2]._buf, bufs[3]._buf, bufs[4]._buf, bufs[5]._buf)
    return None


def fast_attention(q: Tensor, keys: Tensor, values: Tensor, gate: Tensor) -> Tensor:
    """Qwen3.5 0.8B attention (8 heads, 2 KV heads, head_dim 256) in one C loop."""
    q, gate = q.float().contiguous(), gate.float().contiguous()
    length = keys.shape[2]  # token count, before the K/V transpose
    # The C kernel reads K/V as [token, kv_head, head_dim].
    keys, values = keys.float().transpose(1, 2).contiguous(), values.float().transpose(1, 2).contiguous()
    length = Tensor.empty(1, device=q.device, dtype=dtypes.int32).assign(
        Tensor(length).cast(dtypes.int32).reshape(1))
    out = Tensor.empty(q.shape[0], q.shape[2], q.shape[1] * q.shape[3], device=q.device, dtype=dtypes.float32)
    call = UOp(Ops.CUSTOM_FUNCTION, src=(), arg="attention").call(
        out.uop, q.uop, keys.uop, values.uop, gate.uop, length.uop)
    return Tensor(out.uop.after(call))

_ARGMAX_SRC = """
void argmax(const float *logits, int *out, int n) {
  int best = 0;
  float bv = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  *out = best;
}
"""


def _argmax_program():
    global _ARGMAX_PROGRAM
    if _ARGMAX_PROGRAM is None:
        import platform
        from tinygrad import Device, dtypes
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_cpu import CPUProgram
        from tinygrad.runtime.support.compiler_cpu import ClangCompiler

        arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        lib = ClangCompiler([arch, "native"]).compile(_ARGMAX_SRC)
        signature = ((None, 0, dtypes.uint64, ()), (None, 1, dtypes.uint64, ()), (None, 2, dtypes.int32, ()))
        _ARGMAX_PROGRAM = CPUProgram(Device["CPU"], TinyELF(lib, "argmax", Target("CPU"), signature))
    return _ARGMAX_PROGRAM


def _exec_argmax(ctx, call, ast):
    from tinygrad.engine.realize import resolve_params, track_stats

    bufs = [b.buffer.ensure_allocated() for b in resolve_params(call, ctx.input_uops)]
    with track_stats(ctx, call, bufs[0].device, bufs, ctx.var_vals):
        _argmax_program()(bufs[1]._buf, bufs[0]._buf, vals=(bufs[1].nbytes // 4,))
    return None


def fast_argmax(x: Tensor) -> Tensor:
    """Greedy argmax over the vocabulary without tinygrad's reduction tree."""
    x = x.float().contiguous()
    out = Tensor.empty(1, 1, device=x.device, dtype=dtypes.int32)
    call = UOp(Ops.CUSTOM_FUNCTION, src=(), arg="argmax").call(out.uop, x.uop)
    return Tensor(out.uop.after(call))


_GATHER_SRC = """
void gather(unsigned long out, unsigned long weight, unsigned long idx, int n, int dim) {
  const __fp16 *w = (const __fp16 *)weight;
  __fp16 *o = (__fp16 *)out;
  const int *ix = (const int *)idx;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < dim; ++j)
      o[(unsigned long)i * dim + j] = w[(unsigned long)ix[i] * dim + j];
}
"""


def _gather_fn():
    global _GATHER_LIB, _GATHER_FN
    if _GATHER_FN is None:
        import subprocess
        cache = pathlib.Path(os.environ.get("ANE_CACHE_DIR", "~/.cache/ane-gdn")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / "gather.so"
        if not lib.exists():
            temporary = cache / f"gather.{os.getpid()}.so"
            subprocess.run(["cc", "-O3", "-fPIC", "-shared", "-x", "c", "-", "-o", str(temporary)],
                           input=_GATHER_SRC.encode(), check=True)
            temporary.replace(lib)
        _GATHER_LIB = ctypes.CDLL(str(lib))
        _GATHER_FN = ctypes.cast(_GATHER_LIB.gather, ctypes.c_void_p).value
    return _GATHER_FN


def _gather_program():
    global _GATHER_PROGRAM
    if _GATHER_PROGRAM is None:
        import platform
        from tinygrad import Device
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_cpu import CPUProgram
        from tinygrad.runtime.support.compiler_cpu import ClangCompiler
        from tinygrad.runtime.support.elf import jit_loader

        arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
        lib = jit_loader(ClangCompiler([arch, "native"]).compile_to_obj(_GATHER_SRC))
        signature = ((None, 0, dtypes.uint64, ()), (None, 1, dtypes.uint64, ()), (None, 2, dtypes.uint64, ()),
                     (None, 3, dtypes.int32, ()), (None, 4, dtypes.int32, ()))
        _GATHER_PROGRAM = CPUProgram(Device["CPU"], TinyELF(lib, "gather", Target("CPU"), signature))
    return _GATHER_PROGRAM


def _exec_gather(ctx, call, ast):
    from tinygrad.engine.realize import _resolve, track_stats

    bufs = [_resolve(b, ctx.input_uops).buffer.ensure_allocated() for b in call.src[1:4]]
    n = bufs[2].nbytes // 4
    dim = (bufs[0].nbytes // call.src[1].dtype.itemsize) // n
    global _GATHER_ENTRY
    if _GATHER_ENTRY is None:
        _GATHER_ENTRY = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int)(_gather_fn())
    with track_stats(ctx, call, bufs[0].device, bufs, ctx.var_vals):
        _GATHER_ENTRY(ctypes.c_void_p(bufs[0]._buf.va_addr), ctypes.c_void_p(bufs[1]._buf.va_addr),
                      ctypes.c_void_p(bufs[2]._buf.va_addr), n, dim)
    return None


def fast_embedding(weight: Tensor, tokens: Tensor) -> Tensor:
    """Row gather from a materialized embedding table; avoids the one-hot vocab sum."""
    out = Tensor.empty(int(np.prod(tokens.shape)), weight.shape[1], device=weight.device, dtype=weight.dtype)
    call = UOp(Ops.CUSTOM_FUNCTION, src=(), arg="gather").call(out.uop, weight.uop, tokens.uop)
    return Tensor(out.uop.after(call)).reshape(*tokens.shape, weight.shape[1])


def ane_device():
    """Open the shared ANE device through the reference runtime, once."""
    global _ANE
    if _ANE is None:
        source = ANE_REF / "qwen35" / "ane.py"
        if not source.exists():
            raise RuntimeError(f"ANE reference tree not found at {ANE_REF}; set ANE_REF")
        sys.path.insert(0, str(ANE_REF))
        from qwen35.ane import Ane

        _ANE = Ane()
    return _ANE


_ANE_SCALED_SRC = """
#include <math.h>
#include <stddef.h>
typedef int (*run_fn)(void *, const float *, float *, int);
static float scaled_buf[32 * 32768];
static int shifts_buf[32];
int ane_run_scaled(void *plan, void *run_batch, const float *input, float *output,
                   int rows, int in_cols, int out_cols, float limit) {
  run_fn run = (run_fn)run_batch;
  if (rows < 1 || rows > 32 || in_cols < 1 || in_cols > 32768 || out_cols < 1) return 0;
  for (int r = 0; r < rows; ++r) {
    const float *x = input + (size_t)r * in_cols;
    float *s = scaled_buf + (size_t)r * in_cols;
    float peak = 0;
    for (int k = 0; k < in_cols; ++k) { float a = fabsf(x[k]); if (a > peak) peak = a; }
    int shift = 0;
    if (peak > 0) {
      shift = (int)floorf(log2f(limit) - log2f(peak));
      if (shift > 126) shift = 126;
      if (shift < -126) shift = -126;
    }
    shifts_buf[r] = shift;
    float scale = ldexpf(1.f, shift);
    for (int k = 0; k < in_cols; ++k) s[k] = x[k] * scale;
  }
  if (!run(plan, scaled_buf, output, rows)) return 0;
  for (int r = 0; r < rows; ++r) {
    float inv = ldexpf(1.f, -shifts_buf[r]);
    float *o = output + (size_t)r * out_cols;
    for (int k = 0; k < out_cols; ++k) o[k] *= inv;
  }
  return 1;
}
"""


def _ane_scaled_fn():
    global _ANE_SCALED_FN
    if _ANE_SCALED_FN is None:
        import subprocess
        cache = pathlib.Path(os.environ.get("ANE_CACHE_DIR", "~/.cache/ane-gdn")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        lib = cache / "ane_scaled.so"
        if not lib.exists():
            temporary = cache / f"ane_scaled.{os.getpid()}.so"
            subprocess.run(["cc", "-O3", "-ffp-contract=off", "-march=armv8.2-a+fp16", "-fPIC", "-shared",
                            "-x", "c", "-", "-lm", "-o", str(temporary)], input=_ANE_SCALED_SRC.encode(), check=True)
            temporary.replace(lib)
        global _ANE_SCALED_LIB
        _ANE_SCALED_LIB = ctypes.CDLL(str(lib))
        _ANE_SCALED_FN = ctypes.cast(_ANE_SCALED_LIB.ane_run_scaled, ctypes.c_void_p).value
    return _ANE_SCALED_FN


def _exec_ane(ctx, call, ast):
    """Execute one ANE plan from the scheduler; buffers are CPU host memory."""
    from tinygrad.engine.realize import resolve_params, track_stats

    bufs = [b.buffer.ensure_allocated() for b in resolve_params(call, ctx.input_uops)]
    plan, fn, rows, cols, depth, limit, mode = struct.unpack("<QQIIIfB", ast.src[3].arg)
    if not any(int(p) == plan for _, p in _PLANS):
        raise RuntimeError(f"unknown ANE plan {plan:#x}")

    def submit():
        inp, out = ctypes.c_void_p(bufs[1]._buf.va_addr), ctypes.c_void_p(bufs[0]._buf.va_addr)
        global _ANE_ENTRY, _ANE_SCALED_ENTRY
        if mode == 2:
            if _ANE_SCALED_ENTRY is None:
                _ANE_SCALED_ENTRY = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                     ctypes.c_float)(_ane_scaled_fn())
            return _ANE_SCALED_ENTRY(ctypes.c_void_p(plan), ctypes.c_void_p(fn), inp, out, rows, depth, cols, limit)
        if _ANE_ENTRY is None:
            _ANE_ENTRY = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)(fn)
        return _ANE_ENTRY(ctypes.c_void_p(plan), inp, out, rows)

    with track_stats(ctx, call, bufs[0].device, bufs, ctx.var_vals):
        if getenv("ANE_SYNC", 0):
            # Diagnostic path: run on the main thread and report submission errors.
            bufs[1].allocator.dev.synchronize()
            ok = submit()
            if not ok:
                values = np.asarray(bufs[1].numpy(), dtype=np.float32)
                raise RuntimeError(f"ANE submission failed: plan={plan:#x} rows={rows} cols={cols} depth={depth} "
                                   f"input_max={float(np.nanmax(np.abs(values))):.6g} "
                                   f"finite={bool(np.isfinite(values).all())}")
        else:
            # Wait only for the last queued CPU producer, not the whole queue.
            # The worker is native code, so this spin does not block it.
            dev = bufs[1].allocator.dev
            target = dev.timeline_value - 1
            while target >= 0 and dev.timeline_signal.value < target:
                pass
            if not submit():
                raise RuntimeError(f"ANE submission failed: plan={plan:#x} rows={rows} cols={cols} depth={depth}")
    return None


def _patch_scheduler() -> None:
    global _PATCHED, _CPU_RENDERER
    if _PATCHED:
        return
    import tinygrad.engine.realize as realize
    from tinygrad import Device

    _CPU_RENDERER = Device["CPU"].renderer

    pattern = PatternMatcher([
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="ane", src=(UPat(Ops.PROGRAM, name="ast"),)),),
              name="call", allow_any_len=True), _exec_ane),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="argmax", name="ast"),),
              name="call", allow_any_len=True), _exec_argmax),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="attention", name="ast"),),
              name="call", allow_any_len=True), _exec_attention),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="gdn", name="ast"),),
              name="call", allow_any_len=True), _exec_gdn),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="gather", name="ast"),),
              name="call", allow_any_len=True), _exec_gather),
        ])
    realize.pm_exec = pattern + realize.pm_exec

    original_name = realize.get_call_name
    def get_call_name(call, bufs, var_vals=None):
        if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "ane":
            return "ANE matmul"
        if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "argmax":
            return "argmax"
        if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "attention":
            return "attention"
        if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "gdn":
            return "gdn"
        if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "gather":
            return "gather"
        return original_name(call, bufs, var_vals)
    realize.get_call_name = get_call_name
    _PATCHED = True


def close() -> None:
    """Free every ANE plan and close the device."""
    global _ANE
    devices = {id(ane): ane for ane, _ in _PLANS}
    for ane, plan in _PLANS:
        ane.free(plan)
    _PLANS.clear()
    if _ANE is not None:
        devices[id(_ANE)] = _ANE
    for ane in devices.values():
        ane.close()
    _ANE = None


class ANELinear:
    """Drop-in ``nn.Linear`` replacement backed by one resident ANE plan."""

    def __init__(self, linear: nn.Linear, ane=None):
        self.weight, self.bias = linear.weight, linear.bias
        self.columns, self.depth = self.weight.shape
        self.ane = ane
        self.plan = None
        self.fn = None
        self.chunks: list[tuple[object, int, int, int, float]] = []
        self.scaled = False
        self.log2_limit = 6  # floor(log2(64)), the reference Ane default

    def prepare(self) -> "ANELinear":
        if self.plan is None:
            if self.weight is None:
                raise RuntimeError("ANELinear has no weight to prepare")
            self.ane = self.ane or ane_device()
            weight = self.weight.numpy().astype(np.float16, copy=False)
            native = hasattr(self.ane, "lib") and hasattr(self.ane.lib, "ane_plan_run_batch")
            chunk_cols = ANE_MAX_COLS if native else self.columns
            chunk_depth = ANE_MAX_DEPTH if native else self.depth
            self.chunks.clear()
            for c0 in range(0, self.columns, chunk_cols):
                cols = min(chunk_cols, self.columns - c0)
                for k0 in range(0, self.depth, chunk_depth):
                    kd = min(chunk_depth, self.depth - k0)
                    plan = self.ane.create(weight[c0:c0 + cols, k0:k0 + kd])
                    if plan is None:
                        raise RuntimeError(f"ANE could not prepare a {kd} x {cols} matrix")
                    _PLANS.append((self.ane, plan))
                    scaled = native and _ane_scaled_fn() is not None
                    fn = ctypes.cast(self.ane.lib.ane_plan_run_batch, ctypes.c_void_p).value if native else 0
                    # One extra bit of headroom below the reference limit: the
                    # generic path has no compensated output planes, so keep the
                    # worst-case FP16 partial sum below 32768.
                    limit = math.floor(math.log2(self.ane.input_limits.get(plan, 64.0))) - 1 if native else 6
                    self.chunks.append((c0, cols, k0, kd, plan, fn, limit,
                                        float(self.ane.input_limits.get(plan, 64.0)) if scaled else 0.0))
            self.scaled = native and _ane_scaled_fn() is not None
            self.plan, self.fn = self.chunks[0][4], (self.chunks[0][5] if native else None)
            self.log2_limit = min(chunk[6] for chunk in self.chunks)
            if native:
                _patch_scheduler()
            self.weight = None  # the plans carry the packed copy
        return self

    def _kernel(self, out: UOp, lhs: UOp, rows: int, plan, fn: int, cols: int, depth: int, limit: float = 0.0,
                mode: int = 0) -> UOp:
        sink = UOp.sink(out, lhs, arg=KernelInfo(f"ane_matmul_{rows}_{cols}_{depth}"))
        info = replace(ProgramInfo.from_sink(sink, Target("CPU")), outs=(0,), ins=(1,))
        payload = struct.pack("<QQIIIfB", int(plan), fn, rows, cols, depth, limit, mode)
        program = UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR, src=(out, lhs)),
                      UOp(Ops.SOURCE, arg="ANE FP16 matmul"), UOp(Ops.BINARY, arg=payload)))
        return UOp(Ops.CUSTOM_FUNCTION, src=(program,), arg="ane")

    def __call__(self, x: Tensor) -> Tensor:
        if self.plan is None:
            raise RuntimeError("ANELinear.prepare() was not called")
        shape = x.shape
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        flat = x.reshape(rows, shape[-1]).float().contiguous()
        if self.fn is None:
            # Fallback for mock devices without a native entry point.
            arr = flat.numpy()
            out = np.empty((rows, self.columns), dtype=np.float32)
            for r0 in range(0, rows, ANE_BATCH):
                part = arr[r0:r0 + ANE_BATCH]
                out[r0:r0 + len(part)] = self.ane.run(self.plan, part, self.columns)
            y = Tensor(out.reshape(*shape[:-1], self.columns), device=x.device, dtype=dtypes.float32)
        else:
            # Same power-of-two input scaling as Ane.run, but take the shift
            # from the FP32 exponent bits so it is exact and cheap.
            peak = flat.abs().max(axis=1, keepdim=True).maximum(1e-30)
            exponent = peak.bitcast(dtypes.uint32).rshift(23).bitwise_and(0xFF).cast(dtypes.int32)
            shift = (127 + self.log2_limit - exponent).maximum(-126).minimum(126)
            scale = ((shift + 127).cast(dtypes.uint32) << 23).bitcast(dtypes.float32)
            inv_scale = ((-shift + 127).cast(dtypes.uint32) << 23).bitcast(dtypes.float32)
            native_scaled = self.scaled
            scaled = flat if native_scaled else flat * scale
            pieces = []
            for r0 in range(0, rows, ANE_BATCH):
                part = scaled[r0:r0 + ANE_BATCH]
                chunk_outs = []
                for c0 in range(0, self.columns, ANE_MAX_COLS if self.fn is not None else self.columns):
                    cols = min(ANE_MAX_COLS if self.fn is not None else self.columns, self.columns - c0)
                    acc = None
                    for cc0, ccols, k0, kd, plan, fn, _, flimit in self.chunks:
                        if cc0 != c0: continue
                        out = Tensor.empty(part.shape[0], cols, device=x.device, dtype=dtypes.float32)
                        o = out.custom_kernel(
                            part[:, k0:k0 + kd], fxn=lambda o, a, n=part.shape[0], p=plan, f=fn, c=cols, d=kd,
                            l=flimit, m=2 if native_scaled else 0:
                            self._kernel(o, a, n, p, f, c, d, l, m))[0]
                        acc = o if acc is None else acc + o
                    chunk_outs.append(acc)
                pieces.append(Tensor.cat(*chunk_outs, dim=1) if len(chunk_outs) > 1 else chunk_outs[0])
            out = Tensor.cat(*pieces, dim=0) if len(pieces) > 1 else pieces[0]
            y = out.reshape(*shape[:-1], self.columns) if native_scaled else (out * inv_scale).reshape(*shape[:-1], self.columns)
        return y if self.bias is None else y + self.bias


def _walk(obj, visit) -> int:
    """Call visit on every Linear/ANELinear reachable from ``obj``; returns visited."""
    count = 0

    def walk(node) -> None:
        nonlocal count
        for name, value in list(vars(node).items()):
            if isinstance(value, (nn.Linear, ANELinear)):
                count += 1
                new = visit(value)
                if new is not value:
                    setattr(node, name, new)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, (nn.Linear, ANELinear)):
                        count += 1
                        new = visit(item)
                        if new is not item:
                            value[index] = new
                    elif hasattr(item, "__dict__"):
                        walk(item)
            elif hasattr(value, "__dict__") and not isinstance(value, (Tensor, dict, str, int, float)):
                walk(value)

    walk(obj)
    return count


def replace_linears(model, ane=None, max_weight: int | None = None) -> int:
    """Swap every eligible ``nn.Linear`` for ``ANELinear``; returns replaced."""
    limit = ANE_MAX_WEIGHT if max_weight is None else max_weight
    replaced = 0

    def visit(linear):
        nonlocal replaced
        if isinstance(linear, ANELinear) or linear.weight is None:
            return linear
        if (linear.weight.numel() > limit or min(linear.weight.shape) < ANE_MIN_DIM
                or linear.weight.shape[1] > ANE_MAX_DIM):
            return linear
        replaced += 1
        return ANELinear(linear, ane)

    _walk(model, visit)
    return replaced


def prepare_linears(model) -> int:
    """Pack weights for every ``ANELinear`` reachable from ``model``."""
    prepared = 0

    def visit(linear):
        nonlocal prepared
        if isinstance(linear, ANELinear):
            linear.prepare()
            prepared += 1
        return linear

    _walk(model, visit)
    return prepared


def disable_jit(model) -> None:
    """Run the model eagerly: the ANE call is a host round-trip, not a graph op."""
    import tinygrad.llm.model as llm_model

    model.prefill_jit = model.forward
    model.rollout_jit = model.forward

    # Each block wraps its body in @function, which sets ALLOW_DEVICE_USAGE=0
    # and blocks host-side realization. The ANE custom kernel does not need
    # that bypass, but keeping the identity decorator makes the eager path
    # identical to the pre-kernel baseline.
    def eager_function(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fxn: fxn

    llm_model.function = eager_function
