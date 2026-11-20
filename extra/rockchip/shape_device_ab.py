#!/usr/bin/env python3
"""Per-call device time for ONE fused INT8 GEMM, from the driver's own accounting.

Why this exists: `CPUProgram.__call__` hooks see nothing for the fused path (the plan binds its CALL
uop and `replay_c` calls the compiled entry by pointer), so per-shape device time could only be
inferred from a whole-token dump produced by a different protocol. This isolates the shape instead:
one `TinyJit`-captured call per step, and `npu.submit_seconds` / `submit_count` deltas divided by the
number of calls. No root, no codegen labels, no other work in the loop.

  python extra/rockchip/shape_device_ab.py [reps]

Prints us/call and GB/s per shape against this board's 33.79 GB/s LPDDR ceiling, so the k-dependence
of the excess (+12 us at k=1024, +73 us at k=4096 versus the reference) can be checked in isolation.
"""
import sys, pathlib, statistics as st, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tinygrad import Device, Tensor
from tinygrad.engine.jit import TinyJit
import npu_linear as NL

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CEIL = 33.79
dev = Device["ROCKCHIP"]
# width sweep at k=4096 decides whether fusing two n=2560 Linears into one n=5120 pays:
# if per-byte efficiency rises with n, the per-call cost is fixed and fusion is the fix.
SHAPES = [(2560, 4096), (5120, 4096), (7680, 4096), (10240, 4096)]

print(f"{'shape (n,k)':>16}{'MB':>8}{'submits/call':>14}{'us/call':>9}{'GB/s':>7}{'above floor':>12}")
for n, k in SHAPES:
    w = Tensor.rand(n, k).realize()
    # _fused_plan reads `weight.uop.buffer._buf.dma`, so the packed weight must be a realized
    # BUFFER on ROCKCHIP, not a pending COPY (AssertionError: must be a realized BUFFER).
    packed, wscale = (t.realize() for t in NL.pack_weight_int8(w, "ROCKCHIP"))
    x = Tensor.rand(1, k).realize()
    f = TinyJit(lambda xx: NL.packed_matmul_int8_fused(xx, packed, wscale, n, "ROCKCHIP", 1))
    f(x).realize()                      # capture
    ts, subs, devs = [], [], []
    for _ in range(REPS):
        s0, w0 = dev.submit_count, dev.submit_seconds
        a = time.perf_counter(); f(x).realize(); wall = (time.perf_counter()-a)*1e6
        ts.append(wall); subs.append(dev.submit_count-s0); devs.append((dev.submit_seconds-w0)*1e6)
    us = st.median(devs); mb = n*k/1e6; floor = mb/CEIL*1e3
    print(f"{f'{n}x{k}':>16}{mb:>8.2f}{st.median(subs):>14.0f}{us:>9.0f}{mb/us*1e3:>7.1f}"
          f"{100*(us-floor)/floor:>11.0f}%", flush=True)
    if n == 2560 and k == 4096:
        print(f"    (wall {st.median(ts):.0f} us, device share {100*us/st.median(ts):.0f}%)")
