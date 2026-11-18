"""One attention case per process: attn_case.py T CAP POS

Compares `attn_native.attention` against an fp64 numpy reference. Fresh process per case so no
compiled program is reused across geometries -- the contradiction that cost a day on the scan.
"""
import sys

sys.path.insert(0, "/home/orangepi/pr/tinygrad/extra/rockchip")
import numpy as np
from tinygrad import Tensor
from attn_native import attention

T, CAP, POS = (int(x) for x in sys.argv[1:4])


def ref_fp64(q, kv, pos):
  k, v = kv[0, 0], kv[0, 1]                                    # (KVH, CAP, D)
  out = np.zeros_like(q, dtype=np.float64)
  H, KVH, D = q.shape[1], k.shape[0], q.shape[3]
  for h in range(H):
    kh = h // (H // KVH)
    for i in range(T):
      last = pos + i
      s = (q[0, h, i].astype(np.float64) @ k[kh, :last + 1].astype(np.float64).T) / np.sqrt(D)
      s = np.exp(s - s.max())
      s /= s.sum()
      out[0, h, i] = s @ v[kh, :last + 1].astype(np.float64)
  return out


def rel(a, b):
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


Tensor.manual_seed(0)
q = (Tensor.rand(1, 16, T, 256).float() - 0.5).realize()
kv = (Tensor.rand(1, 2, 4, CAP, 256).float() - 0.5).realize()

res = attention(q, kv, POS).realize().numpy()
ref = ref_fp64(q.numpy(), kv.numpy(), POS)
print(f"T={T} CAP={CAP} pos={POS}  rel err vs fp64 = {rel(res, ref):.3e}  "
      f"out[0,0,0,:3]={np.round(res[0, 0, 0, :3], 6)} ref={np.round(ref[0, 0, 0, :3], 6)}")
