#!/usr/bin/env python3
"""Fused Linear vs the existing unfused path, both JIT-captured, both under the replay.

The split was rejected when one program cost 149.6 us of Python dispatch. The replay removed that
(measured: stock dispatch 136.24 ms / 1006 programs, removed by the replay's -153.80 ms), so the
premise is gone and the lever has to be re-measured.

Isolated probe: it decides only whether to spend a 3+3 on an in-model A/B. It is NOT an in-model
number (AGENTS.md: never quote an isolated measurement as an in-model one).

The weight comes from the tree's own pack_weight_int8, because both paths expect the CNA layout
(n/32, k/32, 32, 32) -- a flat int8 buffer is what made the first version of this probe fail with
IndexError instead of measuring anything.
"""
import pathlib, statistics as st, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
if "noreplay" not in sys.argv:
    from tinygrad.llm.kernels import replay_c as RC
    RC.install()
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
import npu_linear as NL

GEOMS = [(64, 9216, 2560, "ffn_up   m64 n9216 k2560"),
         (64, 2560, 4096, "ffn_down m64 n2560 k4096")]
rng = np.random.default_rng(0)
mode = "stock" if "noreplay" in sys.argv else "replay"
print(f"mode {mode}")
for m, n, k, label in GEOMS:
    w = Tensor(((rng.standard_normal((n, k)) * 0.02).astype(np.float16))).contiguous().realize()
    codes, scales = NL.pack_weight_int8(w, "CPU")
    codes, scales = codes.realize(), scales.realize()
    # The fused plan dereferences weight.uop.buffer._buf.dma, so the codes must live on the NPU;
    # the scales are read by the CPU, which is why they stay host-side.
    codes_npu = codes.to("ROCKCHIP").realize()
    x = Tensor(((rng.standard_normal((m, k)) * 0.5).astype(np.float32))).contiguous().realize()
    arms = {
        "fused":   TinyJit(lambda: NL.packed_matmul_int8_fused(x, codes_npu, scales, n, "ROCKCHIP", m)),
        "unfused": TinyJit(lambda: NL.packed_matmul_int8(x, codes_npu, scales, n, "ROCKCHIP")),
    }
    out = {}
    for name, j in arms.items():
        out[name] = j().realize().numpy()
    same = np.array_equal(out["fused"], out["unfused"])
    mad = float(np.abs(out["fused"] - out["unfused"]).max())
    times = {k2: [] for k2 in arms}
    for _ in range(2):
        for name, j in arms.items(): j().realize()
    for _ in range(5):                      # ABBA rounds: thermal drift cancels in the ratio
        for name, j in arms.items():
            t0 = time.perf_counter(); j().realize(); times[name].append((time.perf_counter() - t0) * 1e3)
        for name, j in reversed(list(arms.items())):
            t0 = time.perf_counter(); j().realize(); times[name].append((time.perf_counter() - t0) * 1e3)
    f, u = st.median(times["fused"]), st.median(times["unfused"])
    print(f"{label}  bitwise={'YES' if same else 'NO'} maxdiff={mad:.3e}")
    print(f"   fused   {f:8.3f} ms  (spread {min(times['fused']):.2f}-{max(times['fused']):.2f})")
    print(f"   unfused {u:8.3f} ms  (spread {min(times['unfused']):.2f}-{max(times['unfused']):.2f})")
    print(f"   unfused/fused {u/f:.4f}x")
if "noreplay" not in sys.argv:
    s = RC.stats
    print(f"replay: built={s['built']} segments={s['segments']} c_calls={s['calls_c']} py={s['calls_py']}")
