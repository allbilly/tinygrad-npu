#!/usr/bin/env python3
"""Run the fused FFN-tail pack on real raw_out planes and compare against the four-pass path.

Reference (the shipped pipeline, in order):
    1. dequantise gate planes -> fp32   (the fused Linear's own expression)
    2. dequantise up   planes -> fp32
    3. silu(gate) * up                  (tinygrad's own approximation, the E_393216_3 form)
    4. qz_row                           (max-abs scale, round-half-even, +-127 clamp)

Candidate: one call to `rockchip_ffn_pack`, which does all four from the same planes.

Gate: codes and scales must match bitwise, element for element.
"""
from __future__ import annotations

import ctypes
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tinygrad import Device, Tensor  # noqa: E402
import npu_linear as NL  # noqa: E402
import pack_ffn_tail as PACK  # noqa: E402

M, N, K = 64, 2560, 4096


def build(dev, weight, w_scale, raw_out):
  NL._FUSED_PROGRAMS.clear(); NL._PLANE_POOL.clear(); NL._PAYLOAD_POOL.clear()
  return NL._FusedInt8Plan(dev, m=M, n=N, k=K, depth_in=K, cores=3, column_offset=0,
                           row_stride=K, row_offset=0, k_offset=0, weight_dma=weight.dma,
                           w_scale_addr=w_scale.addr, raw_out=raw_out)


def unpack(flat, parts, m, n, dtype=np.int32):
  """Rebuild the part-blocked plane buffer. Keep int32: these arrays are handed to the
  kernel as `const int *`, and a float32 copy would be reinterpreted as bit patterns."""
  out = np.empty((m, n), dtype)
  for off, w in parts:
    out[:, off:off + w] = flat[off * m: off * m + m * w].reshape(m, w)
  return out


def qz_row(x: np.ndarray):
  """The shipped quantiser, in numpy: max-abs per row, /127 with 1e-8 floor, half-even, clamp."""
  largest = np.abs(x).max(axis=1)
  scale = np.maximum(largest / 127.0, 1e-8).astype(np.float32)
  codes = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
  return codes, scale


def main() -> int:
  dev = Device["ROCKCHIP:0"]
  rng = np.random.default_rng(0)

  def mk(n):
    codes = rng.integers(-127, 128, size=n * K, dtype=np.int16).astype(np.int8)
    b = dev.allocate(n * K); dev.allocator._copyin(b, memoryview(bytearray(codes.tobytes())))
    sc = (np.abs(rng.standard_normal(n)).astype(np.float32) / 127.0)
    s = dev.allocate(n * 4); dev.allocator._copyin(s, memoryview(bytearray(sc.tobytes())))
    return b, s, sc

  gw, gs, gsc = mk(N)
  uw, us, usc = mk(N)
  x = Tensor(((np.arange(M * K, dtype=np.float32) % 97) / 97.0 - 0.5).reshape(M, K),
             device="CPU").contiguous().realize()

  pg = build(dev, gw, gs, True); pg(x).realize()
  pu = build(dev, uw, us, True); pu(x).realize()

  def read_i32(buf):
    o = np.empty(M * N, np.int32); dev.allocator._copyout(memoryview(o).cast("B"), buf); return o
  def read_f32(buf, k):
    o = np.empty(k, np.float32); dev.allocator._copyout(memoryview(o).cast("B"), buf); return o

  g_flat, u_flat = read_i32(pg.out), read_i32(pu.out)
  g_scales, u_scales = read_f32(pg.payload, M), read_f32(pu.payload, M)
  g_planes = unpack(g_flat, pg.key.parts, M, N)      # int32, passed to the kernel as-is
  u_planes = unpack(u_flat, pu.key.parts, M, N)

  # reference: dequantise exactly as the emitter does, then tinygrad's silu*up, then qz_row
  g_fp32 = (g_planes.astype(np.float32) * g_scales[:, None]) * gsc[None, :]
  u_fp32 = (u_planes.astype(np.float32) * u_scales[:, None]) * usc[None, :]
  assert g_planes.dtype == np.int32 and u_planes.dtype == np.int32, "planes must stay int32"
  # Reference must be the SAME silu form the model runs. `Tensor.silu()` alone is not
  # guaranteed to compile to the fused E_393216_3 kernel, and comparing against it produced
  # ulp-level divergence while agreeing to six decimals. swiglu.c is verified maxdiff 0
  # against E_393216_3, so it is the trusted reference here.
  sw_src = (HERE / "swiglu.c").read_text().replace("N_ELEM", str(M * N))
  d0 = pathlib.Path(tempfile.mkdtemp()); (d0 / "s.c").write_text(sw_src)
  sw_so = d0 / "libswiglu.so"
  r0 = subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-o", str(sw_so), str(d0 / "s.c")],
                      capture_output=True, text=True)
  if r0.returncode:
    print("swiglu build failed:", r0.stderr[:600]); return 1
  sw = ctypes.CDLL(str(sw_so))
  y_buf = np.zeros(M * N, np.float32)
  sw_args = (ctypes.c_void_p * 3)(y_buf.ctypes.data, g_fp32.ctypes.data, u_fp32.ctypes.data)
  sw.rk_swiglu(sw_args)
  y_ref = y_buf.reshape(M, N)
  codes_ref, scales_ref = qz_row(y_ref)

  # candidate: the fused pack
  src = PACK.emit(M, N)
  d = pathlib.Path(tempfile.mkdtemp()); (d / "p.c").write_text(src)
  so = d / "libpack.so"
  r = subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-o", str(so), str(d / "p.c")],
                     capture_output=True, text=True)
  if r.returncode:
    print("build failed:", r.stderr[:800]); return 1
  lib = ctypes.CDLL(str(so))
  codes_out = np.zeros((M, N), np.int8)
  scales_out = np.zeros(M, np.float32)
  scratch = np.zeros(3 * M * N, np.float32)   # g | u | y
  # The kernel indexes the plane buffer with the emitter's part-blocked formula
  # (`raw + offset*m + row*width`), so it must receive the FLAT buffer, not the
  # unpacked (m, n) array. Passing the unpacked array made row 1 read row 0's
  # column 864 — row 0 matched only because its offset is 0.
  args = (ctypes.c_void_p * 8)(
      g_flat.ctypes.data, g_scales.ctypes.data, gsc.ctypes.data,
      u_flat.ctypes.data, u_scales.ctypes.data, usc.ctypes.data,
      codes_out.ctypes.data, scales_out.ctypes.data)
  # scratch is slot 8; the array must be 9 wide
  args = (ctypes.c_void_p * 9)(*[args[i] for i in range(8)], scratch.ctypes.data)
  lib.rockchip_ffn_pack(args)

  # bisect: scratch is g | u | y, so the dequantise and the silu can be checked separately
  g_pack = scratch[0:M*N].reshape(M, N)
  u_pack = scratch[M*N:2*M*N].reshape(M, N)
  print(f"stage 1 g dequant bitwise identical: {np.array_equal(g_pack, g_fp32)}"
        f"  max|d|={np.abs(g_pack.astype(np.float64)-g_fp32).max():.3e}")
  print(f"stage 2 u dequant bitwise identical: {np.array_equal(u_pack, u_fp32)}"
        f"  max|d|={np.abs(u_pack.astype(np.float64)-u_fp32).max():.3e}")
  # which scale did the kernel use for row 1?  g = plane * a * w  =>  a = g / (plane*w)
  a_pack = g_pack[1, 0] / (g_planes[1, 0] * gsc[0])
  print(f"  scales seen by reference: g_scales[:4]={g_scales[:4]}")
  print(f"  implied scale used by pack for row 1: {a_pack:.8f}"
        f"   (ref row1={g_scales[1]:.8f}, ref row0={g_scales[0]:.8f})")
  print(f"  ratio pack/ref row1 = {a_pack/g_scales[1]:.6f}")
  print(f"  u_scales[:2]={u_scales[:2]}  gsc[:2]={gsc[:2]}  usc[:2]={usc[:2]}")
  print(f"  does implied == u_scales[1]? {abs(a_pack-u_scales[1])<1e-9}"
        f"  == gsc[1]? {abs(a_pack-gsc[1])<1e-9}")
  for off, w in pg.key.parts:
    blk_g = np.array_equal(g_pack[:, off:off+w], g_fp32[:, off:off+w])
    blk_u = np.array_equal(u_pack[:, off:off+w], u_fp32[:, off:off+w])
    print(f"  part off={off:5} w={w:4}: g match={blk_g}  u match={blk_u}")
  print(f"  g_pack[0,:3]={g_pack[0,:3]}  g_fp32[0,:3]={g_fp32[0,:3]}")
  print(f"  g_pack[1,:3]={g_pack[1,:3]}  g_fp32[1,:3]={g_fp32[1,:3]}")
  # the pack leaves silu*up in scratch[0:M*N]; comparing it localises the fault
  y_pack = scratch[2*M*N:3*M*N].reshape(M, N)
  print(f"silu*up bitwise identical: {np.array_equal(y_pack, y_ref)}  "
        f"max_abs_diff={np.abs(y_pack.astype(np.float64)-y_ref.astype(np.float64)).max():.3e}")
  print(f"  y_ref [0,:4] = {y_ref[0,:4]}")
  print(f"  y_pack[0,:4] = {y_pack[0,:4]}")
  print(f"  g_ref [0,:4] = {g_fp32[0,:4]}  u_ref [0,:4] = {u_fp32[0,:4]}")
  bad = np.argwhere(y_pack != y_ref)
  if len(bad):
    r0, c0 = bad[0]
    print(f"first y divergence at row {r0} col {c0} of {bad[:,1].min()}..{bad[:,1].max()} cols")
    print(f"  g={g_fp32[r0,c0]:.6f} u={u_fp32[r0,c0]:.6f} ref={y_ref[r0,c0]:.6f} pack={y_pack[r0,c0]:.6f}")
    print(f"  differing columns histogram (first 12): {sorted(set(bad[:,1]))[:12]}")
    print(f"  n differing rows: {len(set(bad[:,0]))}")
  c_ok = np.array_equal(codes_out, codes_ref)
  s_ok = np.array_equal(scales_out, scales_ref)
  print(f"codes  bitwise identical: {c_ok}  n_diff={(codes_out != codes_ref).sum()}/{M*N}")
  print(f"scales bitwise identical: {s_ok}  max_abs_diff={np.abs(scales_out-scales_ref).max():.3e}")
  if not c_ok:
    bad = np.argwhere(codes_out != codes_ref)[:3]
    for row, col in bad:
      print(f"  row {row} col {col}: pack={codes_out[row,col]} ref={codes_ref[row,col]} "
            f"y_ref={y_ref[row,col]:.6f} scale={scales_ref[row]:.8f}")
  print("PACK VERIFIED" if (c_ok and s_ok) else "PACK MISMATCH")
  return 0 if (c_ok and s_ok) else 1


if __name__ == "__main__":
  raise SystemExit(main())
