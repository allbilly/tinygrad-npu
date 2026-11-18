"""One case per process. Usage: scan_case.py B H T V K <label>

No program-cache reuse between cases: each invocation is a fresh interpreter, so
`scan_native._PROGRAMS` and `scan._SCAN_PROGRAMS` are empty on entry. That is the point --
the contradictory readings came from reusing one process across geometries.

Compares both `gated_delta_scan` (ours) and `gated_delta_scan_simd` (ported) against:
  - the closed form (v*beta) x k, valid when T==1 and the state starts at zero
  - an fp64 replay of the recurrence, valid for any T
"""
import sys

sys.path.insert(0, "/home/orangepi/pr/tinygrad/extra/rockchip")
import numpy as np
from tinygrad import Tensor
from scan import gated_delta_scan
from scan_native import gated_delta_scan_simd

B, H, T, V, K = (int(x) for x in sys.argv[1:6])
label = sys.argv[6] if len(sys.argv) > 6 else ""


def fp64(q, k, v, beta, alpha, state, T):
    s = state.astype(np.float64).copy()
    outs = []
    for t in range(T):
        a = alpha[:, :, t].astype(np.float64)
        kt = k[:, :, t].astype(np.float64)
        vt = v[:, :, t].astype(np.float64)
        s1 = s * a[:, :, None]
        delta = (vt[..., None] - (s1 * kt[:, :, None, :]).sum(-1, keepdims=True)) \
            * beta[:, :, t].astype(np.float64)[:, :, None, None]
        s = s1 + delta * kt[:, :, None, :]
        outs.append((s * q[:, :, t].astype(np.float64)[:, :, None, :]).sum(-1))
    return np.stack(outs, axis=1), s


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


Tensor.manual_seed(0)
q = (Tensor.rand(B, H, T, K).float() - 0.5).realize()
k = (Tensor.rand(B, H, T, K).float() - 0.5).realize()
v = (Tensor.rand(B, H, T, V).float() - 0.5).realize()
beta = Tensor.rand(B, H, T).float().realize()
alpha = (Tensor.rand(B, H, T, 1).float() * 0.9 + 0.05).realize()
state = ((Tensor.rand(B, H, V, K).float() - 0.5) * 0.1).realize()

ref, refs = fp64(q.numpy(), k.numpy(), v.numpy(), beta.numpy(), alpha.numpy(), state.numpy(), T)

o1, s1 = gated_delta_scan(q, k, v, beta, alpha, state)
o1, s1 = o1.realize().numpy(), s1.realize().numpy()
o2, s2 = gated_delta_scan_simd(q, k, v, beta, alpha, state.clone())
o2, s2 = o2.realize().numpy(), s2.realize().numpy()

print(f"== {label}  B={B} H={H} T={T} V={V} K={K}")
print(f"   ours  out vs fp64 {rel(o1, ref):.3e}   state vs fp64 {rel(s1, refs):.3e}")
print(f"   port  out vs fp64 {rel(o2, ref):.3e}   state vs fp64 {rel(s2, refs):.3e}")
if T == 1:
    want = np.einsum("br,bc->brc", (v.numpy()[:, 0, 0] * beta.numpy()[:, 0])[:, None, :].squeeze(1),
                     k.numpy()[:, 0, 0])
    imp = np.zeros((B, H, V, K))
    for b in range(B):
        for h in range(H):
            imp[b, h] = (v.numpy()[b, h, 0] * beta.numpy()[b, h, 0])[:, None] * k.numpy()[b, h, 0][None, :]
    print(f"   ours  state vs closed form (v*beta)x k  {rel(s1, imp):.3e}")
    print(f"   port  state vs closed form (v*beta)x k  {rel(s2, imp):.3e}")
