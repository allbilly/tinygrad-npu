"""CPU tests for the MLA NPU-attention host pack/unpack kernels and their plans.

Every packed layout is compared byte-exactly (fp16 bit patterns) against an independent
numpy reference. The four attention plan descriptors are validated against the plan_dump
record (`attn-npu/plan_dump.json`) and the runtime constraints. No NPU device is needed.
"""
import struct
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.runtime.ops_rockchip import RockchipGroupedMatmul
from tinygrad.runtime.support import rockchip_mla_npu_pack as P

CAP = ROWS = KEYS = 128
HEADS, NOPE, ROPE, QKD, VDIM, LATENT = 40, 64, 32, 96, 64, 256


def _np_bits(x):
  return np.asarray(x).view(np.uint16) if np.asarray(x).dtype == np.float16 else np.asarray(x)


def _n16k32(a, n, k):
  return a.reshape(HEADS, n // 16, 16, k // 32, 32).transpose(0, 1, 3, 2, 4).reshape(HEADS, n // 16, k // 32, 16, 32)


def _check(name, got, want):
  got, want = np.asarray(got), np.asarray(want)
  assert got.shape == want.shape, f'{name}: shape {got.shape} != {want.shape}'
  bad = int(np.count_nonzero(_np_bits(got) != _np_bits(want)))
  assert bad == 0, f'{name}: {bad}/{got.size} elements differ'


def _exp2v_scalar(x: np.float32) -> np.float32:
  x = np.float32(x)
  q = int(np.trunc(x + np.float32(-0.5 if x < 0 else 0.5)))
  s = np.float32(x - np.float32(q))
  poly = [np.float32(v) for v in (0.00015359208919107914, 0.0013392627006396651, 0.009618384763598442,
                                  0.055503472685813904, 0.24022644758224487, 0.6931471824645996, 1.0)]
  y = poly[0]
  for c in poly[1:]: y = np.float32(y * s + c)
  bits = np.float32(y).view(np.int32) + np.int32(q * 0x800000)
  return bits.view(np.float32)


def _expected_softmax(scores):
  scale = np.float32(1.0 / np.sqrt(96.0))
  out = np.zeros((HEADS, CAP, KEYS), dtype=np.float16)
  for h in range(HEADS):
    for r in range(CAP):
      end = r + 1
      sc = (scores[h, r, :end].astype(np.float32) * scale).astype(np.float32)
      maximum = np.float32(-3.402823466e+38)
      for t in range(end): maximum = maximum if maximum >= sc[t] else sc[t]
      e = np.empty(end, dtype=np.float32); total = np.float32(0.0); t = 0
      while t + 4 <= end:
        for j in range(4):
          e[t + j] = _exp2v_scalar(np.float32((sc[t + j] - maximum) * np.float32(1.4426950216293335)))
          total = np.float32(total + e[t + j])
        t += 4
      while t < end:
        e[t] = _exp2v_scalar(np.float32((sc[t] - maximum) * np.float32(1.4426950216293335)))
        total = np.float32(total + e[t]); t += 1
      out[h, r, :end] = np.asarray(e / total, dtype=np.float16)
  return out


def test_query_pack_byte_exact():
  rng = np.random.default_rng(101)
  q_n = Tensor(rng.normal(0, .2, (HEADS, CAP, NOPE)).astype(np.float32), device='CPU').realize()
  q_r = Tensor(rng.normal(0, .2, (HEADS, CAP, ROPE)).astype(np.float32), device='CPU').realize()
  got = P.mla_npu_pack_query(q_n, q_r, CAP, CAP).numpy()
  want = np.concatenate([q_n.numpy().astype(np.float16), q_r.numpy().astype(np.float16)], axis=-1)
  _check('qa', got, want)


def test_k_pack_byte_exact():
  rng = np.random.default_rng(102)
  kn = rng.normal(0, .2, (HEADS, CAP, NOPE)).astype(np.float32)
  kr = rng.normal(0, .2, (CAP, ROPE)).astype(np.float32)
  got = P.mla_npu_pack_k(Tensor(kn, device='CPU').realize(), Tensor(kr, device='CPU').realize(), KEYS).numpy()
  want = np.zeros((HEADS, CAP, QKD), dtype=np.float16)
  for h in range(HEADS):
    want[h, :, :NOPE] = kn[h].astype(np.float16)
    want[h, :, NOPE:] = kr.astype(np.float16)
  _check('ka', got, _n16k32(want, CAP, QKD))


def test_v_pack_byte_exact():
  rng = np.random.default_rng(103)
  v = rng.normal(0, .2, (HEADS, CAP, VDIM)).astype(np.float32)
  got = P.mla_npu_pack_v(Tensor(v, device='CPU').realize(), KEYS).numpy()
  want = np.zeros((HEADS, VDIM, CAP), dtype=np.float16)
  for h in range(HEADS): want[h] = v[h].astype(np.float16).T
  _check('va', got, _n16k32(want, VDIM, CAP))


def test_softmax_pack_byte_exact():
  rng = np.random.default_rng(104)
  scores = (rng.normal(0, .5, (HEADS, CAP, KEYS)) * 3.0).astype(np.float32)
  got = P.mla_npu_pack_prob(Tensor(scores, device='CPU').realize(), 0, ROWS).numpy()
  _check('prob', got, _expected_softmax(scores))


def test_expand_and_unpack_byte_exact():
  rng = np.random.default_rng(105)
  c_kv = rng.normal(0, .2, (CAP, LATENT)).astype(np.float32)
  got = P.mla_npu_pack_expand_lhs(Tensor(c_kv, device='CPU').realize(), ROWS, CAP).numpy()
  _check('exp_payload', got, np.broadcast_to(c_kv.astype(np.float16), (HEADS, CAP, LATENT)))
  pv = rng.normal(0, .2, (HEADS, CAP, VDIM)).astype(np.float32)
  got = P.mla_npu_unpack_context(Tensor(pv, device='CPU').realize(), ROWS).numpy()
  want = np.zeros((CAP, HEADS * VDIM), dtype=np.float32)
  for r in range(ROWS):
    for h in range(HEADS): want[r, h * VDIM:(h + 1) * VDIM] = pv[h, r]
  _check('pv_unpack', got, want)


def test_expansion_weight_pack_byte_exact():
  rng = np.random.default_rng(106)
  kb = rng.normal(0, .1, (HEADS, LATENT, NOPE)).astype(np.float16)
  got = P.pack_expand_k(Tensor(kb, device='CPU').realize()).numpy()
  want = np.stack([kb[h].T.astype(np.float16) for h in range(HEADS)])
  _check('wkb', got, _n16k32(want, NOPE, LATENT))
  vb = rng.normal(0, .1, (HEADS, VDIM, LATENT)).astype(np.float16)
  got = P.pack_expand_v(Tensor(vb, device='CPU').realize()).numpy()
  want = np.stack([vb[h] for h in range(HEADS)])
  _check('wvb', got, _n16k32(want, VDIM, LATENT))


# descriptor_lib from the delivered plan_dump.json (schema mla-npu-attention-plan-dump-v1)
PLANS = {
  'expand_k_nope': '8000000040000000000100000300000028000000800000004d4d524708000000',
  'expand_v': '8000000040000000000100000300000028000000800000004d4d524708000000',
  'qk': '8000000080000000600000000300000028000000800000004d4d524708000000',
  'pv': '8000000040000000800000000300000028000000800000004d4d524708000000',
}
EXPECTED = {
  'expand_k_nope': (128, 64, 256), 'expand_v': (128, 64, 256), 'qk': (128, 128, 96), 'pv': (128, 64, 128),
}


@pytest.mark.parametrize('name', sorted(PLANS))
def test_attention_plans_match_plan_dump(name):
  lib = bytes.fromhex(PLANS[name])
  m, n, k, cores, groups, tile, marker, version = struct.unpack('<8I', lib)
  assert (m, cores, groups, tile, marker, version) == (128, 3, 40, 128, 0x47524d4d, 8)
  assert (n, k) == EXPECTED[name][1:]
  mat = RockchipGroupedMatmul(SimpleNamespace(programs=set()), TinyELF(lib, 'grouped', Target('ROCKCHIP'), ()))
  assert mat.row_major and mat.dynamic_weights and not mat.skip_finite_checks
  assert mat.m in (128,) and mat.groups in (40,)
  # Static (ABI 9) and no-scan dynamic (ABI 10) share the same descriptor fields.
  for abi, static, skip in ((9, True, False), (10, False, True)):
    alt = struct.pack('<8I', m, n, k, cores, groups, tile, marker, abi)
    plan = RockchipGroupedMatmul(SimpleNamespace(programs=set()), TinyELF(alt, 'grouped', Target('ROCKCHIP'), ()))
    assert plan.row_major and plan.dynamic_weights is (not static) and plan.skip_finite_checks is skip
  # Group-strided per-core chains: 14/13/13 tasks, 40 tasks per submit, 108-word bodies.
  tasks_per_core = [len(range(core, groups, 3)) for core in range(3)]
  assert tasks_per_core == [14, 13, 13] and sum(tasks_per_core) == 40
  from tinygrad.runtime.ops_rockchip import _matmul_body
  body = _matmul_body(m, n, k, 0, 0, 0, True)
  assert len(body) == 108
  cmd_bytes = sum(struct.calcsize('<Q') * (108 + 4) for _ in range(40))
  assert cmd_bytes == 35840


def test_accepts_leading_singleton_dims_and_zeroes_prob_tail():
  rng = np.random.default_rng(107)
  q_n = Tensor(rng.normal(0, .2, (1, HEADS, CAP, NOPE)).astype(np.float32), device='CPU').realize()
  q_r = Tensor(rng.normal(0, .2, (1, HEADS, CAP, ROPE)).astype(np.float32), device='CPU').realize()
  got = P.mla_npu_pack_query(q_n, q_r, ROWS, CAP).numpy()
  want = np.concatenate([q_n.numpy()[0].astype(np.float16), q_r.numpy()[0].astype(np.float16)], axis=-1)
  _check('qa_singleton', got, want)
  # softmax: rows beyond the requested row count and all columns past the causal end are +0
  scores = Tensor((rng.normal(0, .5, (HEADS, CAP, KEYS)) * 3.0).astype(np.float32), device='CPU').realize()
  prob = P.mla_npu_pack_prob(scores, 0, 96).numpy()
  assert np.count_nonzero(prob[:, 96:, :]) == 0
  assert np.count_nonzero(prob[:, 95, 96:]) == 0
