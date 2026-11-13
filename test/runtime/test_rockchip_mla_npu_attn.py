"""CPU-only tests for the opt-in FP16 NPU MLA prefill attention wiring (ONPU_ATTN_FP16_NPU).

No NPU is opened: these tests cover the flag/installation guard logic, the captured start-position
resolution, and the register-level plan descriptors against the validated plan dump. The end-to-end
graph path and the pack-layout byte-exactness are covered by `verify_attn_pack_layout.py`
(stream-diag's pack module) and the hardware A/B (Lead).
"""
from __future__ import annotations
import json, struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from tinygrad.llm.kernels.rockchip_mla_npu_attn import (
  MLANPUAttention, ONPU_ATTN_FP16_NPU, _static_start)

PLAN_DUMP = Path('/home/orangepi/oNPU/analysis/minicpm3/prefill/attn-npu/plan_dump.json')


def test_flag_default_off():
  assert not bool(ONPU_ATTN_FP16_NPU)


def test_static_start_resolution():
  v = UOp.variable('start_pos', 0, 511)
  assert _static_start(0) == 0
  assert _static_start(7) == 7
  assert _static_start(v.bind(0)) == 0
  assert _static_start(v.bind(128)) == 128
  assert _static_start(v) is None


def test_configure_off_is_noop():
  from tinygrad.llm.kernels.rockchip_mla_attention import configure
  model = SimpleNamespace(blk=[])
  assert configure(model, enabled=False) == 0
  assert not hasattr(model, 'blk') or all(not hasattr(b, 'cpu_mla_attention') for b in model.blk)


def test_configure_on_without_hook_raises(monkeypatch):
  import tinygrad.llm.kernels.rockchip_mla_npu_attn as attn
  from tinygrad.llm.kernels.rockchip_mla_attention import configure
  monkeypatch.setattr(attn, 'ONPU_ATTN_FP16_NPU', 1)
  with pytest.raises(RuntimeError):
    configure(SimpleNamespace(blk=[]), enabled=False)


def test_hook_rejects_decode_rows():
  """Without constructing weights (would open the NPU), check the guard needs an instance: use __new__."""
  hook = MLANPUAttention.__new__(MLANPUAttention)
  hook.heads, hook.capacity, hook.context = 40, 128, 512
  hook.nope, hook.rope, hook.v_dim, hook.latent = 64, 32, 64, 256
  def cpu(*shape): return Tensor.empty(*shape, device='CPU', dtype=dtypes.float)
  # decode (rows != capacity) and non-zero start must fall back
  assert hook(cpu(1, 40, 128, 64), cpu(1, 40, 128, 32), cpu(1, 128, 256), cpu(1, 1, 128, 32), 0, 1) is None
  assert hook(cpu(1, 40, 128, 64), cpu(1, 40, 128, 32), cpu(1, 128, 256), cpu(1, 1, 128, 32), 7, 128) is None
  # wrong geometry dtypes must fall back
  assert hook(cpu(1, 40, 128, 64).half(), cpu(1, 40, 128, 32), cpu(1, 128, 256), cpu(1, 1, 128, 32), 0, 128) is None


def test_plan_descriptors_match_dump():
  """The four grouped plans must match the register-level plan dump validated in the GO report."""
  if not PLAN_DUMP.exists():
    pytest.skip('plan_dump.json not available')
  dump = json.loads(PLAN_DUMP.read_text())
  from tinygrad.llm.kernels.rockchip import _grouped
  plans = {'expand_k_nope': (128, 64, 256), 'expand_v': (128, 64, 256), 'qk': (128, 128, 96), 'pv': (128, 64, 128)}
  for name, (m, n, k) in plans.items():
    program = _grouped(Tensor.empty(40, m, n, device='CPU', dtype=dtypes.float).uop,
                       Tensor.empty(40, m, k, device='CPU', dtype=dtypes.half).uop,
                       Tensor.empty(40, n // 16, k // 32, 16, 32, device='CPU', dtype=dtypes.half).uop,
                       40, m, n, k, 128, 3, True, True)
    binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
    got = dict(zip(('m', 'n', 'k', 'cores', 'groups', 'tile', 'marker', 'version'), struct.unpack('<8I', binary)))
    want = dump['new_plans_per_layer'][name]
    assert (got['m'], got['n'], got['k'], got['cores'], got['groups'], got['tile'], got['version']) == \
           (want['m'], want['n'], want['k'], want['cores'], want['groups'], want['tile'], want['version']), name
    assert got['marker'] == 0x47524d4d


def test_pack_outputs_all_finite():
  """ABI-10 (validate_finite=False) trusts the operands: a non-finite pack output would propagate
  silently instead of raising the replay finite-check error. One-time smoke test on finite synthetic
  data at production shapes: every pack kernel output must be all-finite."""
  import numpy as np
  from tinygrad.runtime.support import rockchip_mla_npu_pack as pack
  rng = np.random.default_rng(20261110)
  heads, cap, latent, rope, nope, vdim = 40, 128, 256, 32, 64, 64
  c_kv = Tensor(rng.normal(0, 1.0, (1, cap, latent)).astype(np.float32), device='CPU', dtype=dtypes.float)
  q_nope = Tensor(rng.normal(0, 1.0, (1, heads, cap, nope)).astype(np.float32), device='CPU', dtype=dtypes.float)
  q_rope = Tensor(rng.normal(0, 1.0, (1, heads, cap, rope)).astype(np.float32), device='CPU', dtype=dtypes.float)
  k_rope = Tensor(rng.normal(0, 1.0, (1, 1, cap, rope)).astype(np.float32), device='CPU', dtype=dtypes.float)
  k_nope = Tensor(rng.normal(0, 1.0, (heads, cap, nope)).astype(np.float32), device='CPU', dtype=dtypes.float)
  v_exp = Tensor(rng.normal(0, 1.0, (heads, cap, vdim)).astype(np.float32), device='CPU', dtype=dtypes.float)
  scores = Tensor(rng.normal(0, 3.0, (heads, cap, cap)).astype(np.float32), device='CPU', dtype=dtypes.float)
  for name, out in (
      ('query', pack.mla_npu_pack_query(q_nope, q_rope, cap, cap)),
      ('expand_lhs', pack.mla_npu_pack_expand_lhs(c_kv, cap, cap)),
      ('k', pack.mla_npu_pack_k(k_nope, k_rope, cap)),
      ('v', pack.mla_npu_pack_v(v_exp, cap)),
      ('prob', pack.mla_npu_pack_prob(scores, 0, cap))):
    assert np.isfinite(out.numpy().astype(np.float32)).all(), f'non-finite {name} pack output'
  ctx = Tensor(rng.normal(0, 1.0, (heads, cap, vdim)).astype(np.float32), device='CPU', dtype=dtypes.float)
  assert np.isfinite(pack.mla_npu_unpack_context(ctx, cap).numpy()).all(), 'non-finite pv unpack output'


def test_prob_pool_dispatch_byte_exact():
  """Pool-dispatched softmax pack (heads 10/10/10/10) must be byte-identical to the serial path."""
  import numpy as np
  from tinygrad.runtime.support.rockchip_mla_npu_pack import mla_npu_pack_prob
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  rng = np.random.default_rng(20261110)
  scores = Tensor(np.asarray(rng.normal(0, 3.0, (40, 128, 128)), dtype=np.float32), device='CPU', dtype=dtypes.float)
  serial = mla_npu_pack_prob(scores, 0, 128).numpy()
  try: pool = AttentionPool()
  except OSError as exc: pytest.skip(f'attention pool unavailable: {exc}')
  try:
    with pool:
      pooled = mla_npu_pack_prob(scores, 0, 128, pool.prefill_address).numpy()
  finally:
    pool.close()
  assert serial.dtype == pooled.dtype and serial.shape == pooled.shape
  assert np.array_equal(serial.view(np.uint16), pooled.view(np.uint16))


def test_pool_address_helper():
  hook = MLANPUAttention.__new__(MLANPUAttention)
  hook.pool = None
  assert hook._pool_address() == 0
  from types import SimpleNamespace
  hook.pool = SimpleNamespace(active=False, prefill_address=0x1234)
  assert hook._pool_address() == 0
  hook.pool = SimpleNamespace(active=True, prefill_address=0x1234)
  assert hook._pool_address() == 0x1234

