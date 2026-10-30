from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip, rockchip_pool
from tinygrad.llm.kernels.rockchip import RockchipLinear, configure
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
from tinygrad.llm.kernels.rockchip_shared_pool import model_shared_host_pool
from tinygrad.nn.state import get_state_dict
from tinygrad.llm.model import Transformer, TransformerConfig


class _FakePool:
  address = 0x1234

  def __init__(self):
    self.active = False
    self.entries = 0

  def __enter__(self):
    assert not self.active
    self.active = True
    self.entries += 1
    return self

  def __exit__(self, *args):
    self.active = False


class _FakeSharedState:
  def __init__(self, pool):
    self.pool = pool
    self.active = False
    self.scope_calls = 0

  @contextmanager
  def scope(self):
    with self.pool:
      self.active = True
      self.scope_calls += 1
      try:
        yield
      finally:
        self.active = False


class _ProbeJit:
  def __init__(self, observe, *, fail=False):
    self.observe = observe
    self.fail = fail
    self.captured = None

  def __call__(self, *args, **kwargs):
    self.observe()
    if self.fail:
      raise RuntimeError("probe failure")
    return "ok"


def _model():
  # The minimal one-block shape that admits prepared QKV + NPU FP16 attention.
  with Context(DEV="CPU"):
    return Transformer(TransformerConfig(1, 2048, 3072, 16, 8, 1e-6, 16, 128, 10000.0, 128, 128, max_context=64, qk_norm=128))


def _mock_dense_placement(monkeypatch, pool):
  monkeypatch.setattr(rockchip_pool, "attention_pool", lambda: pool)

  def prepared(self):
    if not self.packed:
      self.packed = [Tensor.empty(1, device="CPU", dtype=dtypes.int8)]

  monkeypatch.setattr(RockchipLinear, "prepare", prepared)
  monkeypatch.setattr(RockchipInt8Linear, "prepare", prepared)


def _configure(model, **overrides):
  args = {
    "prefill": "on",
    "decode": "on",
    "fuse_projections": True,
    "fused_qkv": True,
    "fused_ffn": True,
    "cpu_attention": True,
    "npu_prefill_attention": True,
    "cpu_norm": True,
    "precision": "int8",
    "cpu_pool": True,
  }
  args.update(overrides)
  with Context(DEV="CPU"):
    return configure(model, **args)


def test_shared_host_pool_flag_off_leaves_original_placement(monkeypatch):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  calls = []
  monkeypatch.setattr(rockchip, "shared_host_pool", lambda pool: calls.append(pool) or _FakeSharedState(pool))
  model = _model()
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=0):
    result = _configure(model)
  assert result["shared_host_pool_requested"] is False
  assert result["shared_host_pool"] is False and model_shared_host_pool(model) is None
  assert calls == []


def test_supported_dense_int8_prefill_is_admitted(monkeypatch):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  states = []

  def create(value):
    state = _FakeSharedState(value)
    states.append(state)
    return state

  monkeypatch.setattr(rockchip, "shared_host_pool", create)
  model = _model()
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    result = _configure(model)
  assert result["shared_host_pool_requested"] is True and result["shared_host_pool"] is True
  assert model_shared_host_pool(model) is states[0] and states[0].pool is pool
  assert isinstance(model._shared_host_pool, weakref.ReferenceType)
  assert result["cpu_attention_blocks"] == result["npu_prefill_attention_blocks"] == 1
  assert result["prepared_qkv_blocks"] == result["fused_ffn_blocks"] == 1
  assert result["cpu_norms"] >= 2 * result["cpu_attention_blocks"]


@pytest.mark.parametrize(
  "overrides",
  [
    {"npu_prefill_attention": False},
    {"fused_ffn": False},
    {"cpu_norm": False},
    {"precision": "fp16"},
  ],
)
def test_incomplete_dense_placement_is_not_admitted(monkeypatch, overrides):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  calls = []
  monkeypatch.setattr(rockchip, "shared_host_pool", lambda value: calls.append(value) or _FakeSharedState(value))
  model = _model()
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    result = _configure(model, **overrides)
  assert result["shared_host_pool_requested"] is True
  assert result["shared_host_pool"] is False and model_shared_host_pool(model) is None
  assert calls == []


def test_missing_attention_pool_rejects_requested_feature_truthfully(monkeypatch):
  _mock_dense_placement(monkeypatch, None)
  monkeypatch.setattr(rockchip_pool, "attention_pool", lambda: (_ for _ in ()).throw(OSError("pool unavailable")))
  calls = []
  monkeypatch.setattr(rockchip, "shared_host_pool", lambda value: calls.append(value) or _FakeSharedState(value))
  model = _model()
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    result = _configure(model)
  assert result["shared_host_pool_requested"] is True
  assert result["cpu_pool"] is False
  assert result["shared_host_pool"] is False and model_shared_host_pool(model) is None
  assert calls == []


def test_phase_scope_is_prefill_only_and_exception_cleans_up(monkeypatch):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  states = []

  def create(value):
    state = _FakeSharedState(value)
    states.append(state)
    return state

  monkeypatch.setattr(rockchip, "shared_host_pool", create)
  model = _model()
  observed = []
  model.prefill_jit = _ProbeJit(lambda: observed.append(("prefill", states[0].active, pool.active)))
  model.rollout_jit = _ProbeJit(lambda: observed.append(("decode", states[0].active, pool.active)))
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    result = _configure(model)
  assert result["shared_host_pool"] is True
  assert model.prefill_jit() == "ok" and model.rollout_jit() == "ok"
  assert observed == [("prefill", True, True), ("decode", False, True)]
  assert states[0].scope_calls == 1 and not states[0].active and not pool.active

  failing = _model()
  failing.prefill_jit = _ProbeJit(lambda: None, fail=True)
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    _configure(failing)
  failure_state = states[-1]
  with pytest.raises(RuntimeError, match="probe failure"):
    failing.prefill_jit()
  assert failure_state.scope_calls == 1 and not failure_state.active and not pool.active


def test_reconfiguration_reuses_base_jit_without_nesting_phase_wrappers(monkeypatch):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  states = []

  def create(value):
    state = _FakeSharedState(value)
    states.append(state)
    return state

  monkeypatch.setattr(rockchip, "shared_host_pool", create)
  observed = []
  base = _ProbeJit(lambda: observed.append("base"))
  first = _model()
  first.prefill_jit = base
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    _configure(first)
  first_wrapper = first.prefill_jit
  assert isinstance(first_wrapper._rockchip_base_jit, weakref.ReferenceType)
  assert first_wrapper._rockchip_base_jit() is base

  # A reconfigured model can carry the old wrapper; configure must unwrap its marker.
  second = _model()
  second.prefill_jit = first_wrapper
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    _configure(second)
  second_wrapper = second.prefill_jit
  assert second_wrapper is not first_wrapper and isinstance(second_wrapper._rockchip_base_jit, weakref.ReferenceType)
  assert second_wrapper._rockchip_base_jit() is base
  assert second_wrapper() == "ok"
  assert observed == ["base"] and states[0].scope_calls == 0 and states[1].scope_calls == 1
  assert not states[1].active and not pool.active


def test_warmed_tinyjit_marker_does_not_become_model_parameter_state():
  with Context(DEV="CPU"):
    value=Tensor([1.0],device="CPU").realize()
    jit=TinyJit(lambda x:(x+1).realize())
    jit(value)
    model=SimpleNamespace(weight=Tensor([2.0],device="CPU").realize(),prefill_jit=jit,blk=[])
    configure(model,prefill="off",decode="off")
    state=get_state_dict(model)
  assert list(state)==["weight"]


def test_admitted_cyclic_runtime_state_is_not_traversed_as_parameters(monkeypatch):
  pool=_FakePool()
  _mock_dense_placement(monkeypatch,pool)
  model=_model()
  sentinel=Tensor([99.0],device="CPU").realize()
  states=[]
  def create(value):
    state=_FakeSharedState(value)
    state.runtime={"cycle":state,"not_a_parameter":sentinel}
    states.append(state)
    return state
  monkeypatch.setattr(rockchip,"shared_host_pool",create)
  with Context(DEV="CPU",ONPU_SHARED_HOST_POOL=1):result=_configure(model)
  state=model_shared_host_pool(model)
  assert result["shared_host_pool"] is True and state is states[0]
  assert isinstance(model._shared_host_pool,weakref.ReferenceType)
  discovered=get_state_dict(model)
  assert all(value is not sentinel for value in discovered.values())
  assert not any("_shared_host_pool" in key or "not_a_parameter" in key for key in discovered)


def test_cpu_attention_with_prepared_qkv_and_raw_swiglu_is_fully_pooled(monkeypatch):
  pool = _FakePool()
  _mock_dense_placement(monkeypatch, pool)
  def prepared(self):
    self.packed = [Tensor.empty(1, device="CPU", dtype=dtypes.int8)]
    self.scales = [Tensor.ones(self.columns, device="CPU", dtype=dtypes.float)]
  monkeypatch.setattr(RockchipInt8Linear, "prepare", prepared)
  states = []
  def create(value):
    state = _FakeSharedState(value)
    states.append(state)
    return state
  monkeypatch.setattr(rockchip, "shared_host_pool", create)
  with Context(DEV="CPU", ONPU_SHARED_HOST_POOL=1):
    model = Transformer(TransformerConfig(1, 2048, 5632, 32, 4, 1e-5, 16, 64, 10000.0, 64, 64, max_context=64))
    result = _configure(model, npu_prefill_attention=False, cpu_prefill_pool=True, split_down=True,
                        fused_qkv_without_norm=True, fused_swiglu_ffn=True, fused_swiglu_decode=True)
  assert result["shared_host_pool"] is True
  assert result["npu_prefill_attention_blocks"] == 0
  assert result["prepared_qkv_blocks"] == result["fused_swiglu_ffn_blocks"] == 1
  assert model_shared_host_pool(model) is states[0]
