#!/usr/bin/env python3
"""CPU host-cost microbenchmark for the MLA NPU-attention pack kernels (production geometry).

Single-threaded CPU kernels as wired through the graph (transfers/ioctls excluded).
Target from npu-fuse's 4-worker C bench: 3.23 ms/layer host; 1-worker bound 5.23 ms.
Run: DEV=CPU PYTHONPATH=<tree> python test/runtime/bench_rockchip_mla_npu_pack.py
"""
import statistics, time
import numpy as np
from tinygrad import Tensor
from tinygrad.runtime.support import rockchip_mla_npu_pack as P

CAP = KEYS = ROWS = 128


def bench(fn, n=25, warm=8):
  for _ in range(warm): fn()
  samples = []
  for _ in range(n):
    start = time.perf_counter(); fn(); samples.append((time.perf_counter() - start) * 1000)
  return statistics.median(samples)


def main():
  rng = np.random.default_rng(42)
  q_n = Tensor(rng.normal(0, .2, (40, CAP, 64)).astype(np.float32), device='CPU').realize()
  q_r = Tensor(rng.normal(0, .2, (40, CAP, 32)).astype(np.float32), device='CPU').realize()
  c_kv = Tensor(rng.normal(0, .2, (CAP, 256)).astype(np.float32), device='CPU').realize()
  k_n = Tensor(rng.normal(0, .2, (40, CAP, 64)).astype(np.float32), device='CPU').realize()
  k_r = Tensor(rng.normal(0, .2, (CAP, 32)).astype(np.float32), device='CPU').realize()
  v = Tensor(rng.normal(0, .2, (40, CAP, 64)).astype(np.float32), device='CPU').realize()
  scores = Tensor(rng.normal(0, 1.0, (40, CAP, KEYS)).astype(np.float32), device='CPU').realize()
  result = Tensor(rng.normal(0, .2, (40, CAP, 64)).astype(np.float32), device='CPU').realize()
  ops = {
    'query (row-major LHS)': lambda: P.mla_npu_pack_query(q_n, q_r, ROWS, CAP),
    'expand_lhs (40x replicate)': lambda: P.mla_npu_pack_expand_lhs(c_kv, ROWS, CAP),
    'k (QK RHS N16K32)': lambda: P.mla_npu_pack_k(k_n, k_r, KEYS),
    'v (PV RHS N16K32)': lambda: P.mla_npu_pack_v(v, KEYS),
    'prob (exact softmax)': lambda: P.mla_npu_pack_prob(scores, 0, ROWS),
    'pv_unpack (token-major)': lambda: P.mla_npu_unpack_context(result, ROWS),
  }
  total = 0.0
  print(f"{'kernel':30s} {'ms/call':>9s}")
  for name, fn in ops.items():
    ms = bench(fn)
    total += ms
    print(f'{name:30s} {ms:9.3f}')
  print(f"{'host total / layer':30s} {total:9.3f}   (x62 layers = {total*62:7.1f} ms)")
  print('note: single-threaded CPU kernels; npu-fuse 4-worker C bench reference = 3.23 ms/layer, 1-worker 5.23')
  return total


if __name__ == '__main__':
  main()
