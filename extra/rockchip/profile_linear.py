#!/usr/bin/env python3
"""Break one NPULinear call into its stages, so the port targets the real cost.

Raw submit is 0.044 ms, but a Linear costs 0.77-1.90 ms inside a captured
forward. The difference is packing, copies, allocation and unpacking. This
times each stage separately at the decode shape (m=1) and a prefill shape.

Run with DEV=CPU. One NPU owner at a time.
"""
from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, ".")
from tinygrad import Tensor, dtypes
from npu_linear import K_TILE, TILE, _matmul, pack_weight
from tinygrad.helpers import round_up

REPS = 20


def stages(x: Tensor, weight: Tensor, columns: int, device: str, cores: int = 3):
    """Mirror packed_matmul, timing each stage."""
    m, original_k = x.shape
    k = weight.shape[1] * 32
    n = round_up(columns, 32)
    t = {}

    def timed(name, fn):
        begin = time.perf_counter()
        out = fn()
        t[name] = (time.perf_counter() - begin) * 1000
        return out

    lhs_cpu = timed("cpu_pack", lambda: x.half().pad(((0, 0), (0, k - original_k)))
                    .reshape(m, k // 8, 8).permute(1, 0, 2).contiguous())
    lhs = timed("h2d", lambda: lhs_cpu.to(device))
    out = timed("alloc", lambda: Tensor.empty(n // 4, m, 4, device=device, dtype=dtypes.float))
    out = timed("submit", lambda: out.custom_kernel(
        lhs, weight, fxn=lambda o, a, w: _matmul(o, a, w, m, n, k, cores, 0))[0])
    host = timed("d2h", lambda: out.to(x.device))
    timed("unpack", lambda: host.permute(1, 0, 2).reshape(m, n)[:, :columns])
    return t


def main() -> int:
    # Qwen2 0.5B attn_q: 896 in, 896 out, the shape the baseline spends most of its time in.
    n, k = 896, 896
    weight = Tensor.rand(n, k).half()
    packed = pack_weight(weight, "ROCKCHIP").realize()
    print(f"linear {n}x{k}  reps={REPS}  (ms, median)\n")
    header = f"{'m':>4} " + " ".join(f"{s:>9}" for s in
                                     ("cpu_pack", "h2d", "alloc", "submit", "d2h", "unpack", "total"))
    print(header)
    for m in (1, 4, 64, 128):
        x = Tensor.rand(m, k).half()
        rows = []
        for _ in range(REPS):
            rows.append(stages(x, packed, n, "ROCKCHIP"))
        med = {name: statistics.median(r[name] for r in rows) for name in rows[0]}
        total = sum(med.values())
        print(f"{m:>4} " + " ".join(f"{med[s]:>9.3f}" for s in
                                    ("cpu_pack", "h2d", "alloc", "submit", "d2h", "unpack")) + f" {total:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
