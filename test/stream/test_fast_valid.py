#!/usr/bin/env python3
"""CPU-only tests for the opt-in ONPU_REPLAY_FAST_VALID epoch-gated validity path.

Requires the pending patch (scripts/pending_fast_valid.patch) applied to
tinygrad/llm/kernels/rockchip_replay.py. No NPU/DRM use: only Buffer("CPU") objects,
SimpleNamespace stand-ins and the real plan/allocator hook wrappers.

Proves:
  (a) flag OFF: valid() is the original code path and matches a verbatim reference of
      today's expression for every mutation state (byte-identical behaviour);
  (b) flag ON: the authoritative full check runs once and is then skipped;
  (c) each hooked invalidation (plan close/prepare, allocator _copyin, Buffer
      allocate/deallocate, attention close, new replay) moves the epoch and forces the
      authoritative check again, and the verdict matches the full path;
  (d) the always-authoritative input shape check cannot be skipped by the epoch;
  (e) DEBUG mode still runs the full check on every token and raises on a hidden
      (unhooked) invalidation;
  (f) _output_guards is built on the FIRST valid() call of a replay (cached afterwards),
      so its O(plans) build cost is a per-replay one-off, not part of the steady cost.

Run (after applying the patch):
  DEV=CPU ROCKCHIP_TEST=0 /home/orangepi/tinygrad/.venv/bin/python -m pytest \
    test/stream/test_fast_valid.py -x -q
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tinygrad.device import Buffer
from tinygrad.llm.kernels import rockchip_replay as rr
from tinygrad.runtime.ops_rockchip import (RockchipAllocator, RockchipMatmul, RockchipBatchedMatmul,
                                          RockchipGroupedMatmul, RockchipRoutedMatmul, RockchipBucketedMatmul)
from tinygrad.runtime.support.rockchip_attention import RockchipAttentionProgram


@pytest.fixture(autouse=True)
def _hooks() -> None:
  assert rr._install_fast_valid_hooks(), "hook installer must succeed"
  yield


def cpu_inputs(*sizes: int) -> list[SimpleNamespace]:
  return [SimpleNamespace(buffer=Buffer("CPU", int(size))) for size in sizes]


def make_replay(sizes: tuple[int, ...] = (8, 16, 32), register: bool = True) -> rr.NativeReplay:
  """NativeReplay.__new__ with exactly the attributes the validity path touches."""
  inst = object.__new__(rr.NativeReplay)
  inst.roots = [None] * len(sizes)
  inst.input_sizes = list(sizes)
  inst.variables = []
  guarded = [Buffer("CPU", 64), Buffer("CPU", 64)]
  # The guard records the storage identity, so the fixture must allocate: an
  # unallocated Buffer has _storage None and Buffer.deallocate() asserts that the
  # buffer is allocated. (A test that only allocates inside its own body leaves
  # every other guard test validating against a None storage, which silently
  # weakens them.)
  for buf in guarded: buf.allocate()
  inst.guards = {buf: buf._storage for buf in guarded}
  weight_meta = SimpleNamespace(version=1)
  inst.weight_guards = {id(weight_meta): (weight_meta, 1)}  # real init keys are id(meta)
  inst.attention_programs = [SimpleNamespace(closed=False, owned=[1])]
  out_meta = SimpleNamespace(version=1, cpu_clean_version=1)
  out_buf = SimpleNamespace(meta=out_meta)
  lhs, rhs = SimpleNamespace(), SimpleNamespace()
  plan = SimpleNamespace(owned=[SimpleNamespace()], bufs=(out_buf, lhs, rhs),
                         dev=SimpleNamespace(free=lambda *_: None))
  inst.plans = [(plan, plan.bufs)]
  inst._fast_valid = False
  inst._fast_valid_debug = False
  inst._fast_valid_epoch = None
  inst._fast_valid_full_runs = 0
  inst._fast_valid_skips = 0
  inst._fast_valid_mismatch = 0
  if register: rr._register_fast_valid_watch(inst)
  return inst


def set_fast(inst: rr.NativeReplay, debug: bool = False) -> rr.NativeReplay:
  inst._fast_valid, inst._fast_valid_debug = True, debug
  return inst


# Verbatim copy of today's non-deduplicated valid() expression: the reference for (a).
def original_full_valid(inst: rr.NativeReplay, inputs: list) -> bool:
  return (len(inputs) == len(inst.roots) and all(isinstance(u.buffer, Buffer) and u.buffer.device == 'CPU'
    and u.buffer.nbytes == size for u, size in zip(inputs, inst.input_sizes))
    and all(buf._storage is storage for buf, storage in inst.guards.items())
    and all(meta.version == version for meta, version in inst.weight_guards.values())
    and all(not program.closed and program.owned for program in inst.attention_programs)
    and all(all(buf.meta.cpu_clean_version == buf.meta.version for buf in
      ((plan.bufs[0], plan.bufs[4], plan.bufs[5]) if len(plan.bufs) == 6 else (plan.bufs[0],))) for plan, _ in inst.plans if plan.bufs)
    and all(plan.owned and plan.bufs == buffers for plan, buffers in inst.plans))


def _mutated_inputs(inst: rr.NativeReplay, inputs: list, size_delta: int = 0) -> list:
  return [SimpleNamespace(buffer=Buffer("CPU", buf.nbytes + size_delta)) for buf in (u.buffer for u in inputs)]


# {state name: mutate(inst, inputs) -> replacement inputs or None} - each induces one real
# fallback condition that the authoritative check must report.
MUTATIONS: dict[str, object] = {
  "input_count": lambda inst, inputs: inputs[:2],
  "input_size": lambda inst, inputs: _mutated_inputs(inst, inputs, 1),
  "guard_storage": lambda inst, inputs: setattr(next(iter(inst.guards)), "_storage", object()),
  "weight_version": lambda inst, inputs: setattr(next(iter(inst.weight_guards.values()))[0], "version", 99),
  "attention_closed": lambda inst, inputs: setattr(inst.attention_programs[0], "closed", True),
  "plan_owned": lambda inst, inputs: inst.plans[0][0].owned.clear(),
  "plan_bufs": lambda inst, inputs: setattr(inst.plans[0][0], "bufs", ()),
  "output_dirty": lambda inst, inputs: setattr(inst.plans[0][0].bufs[0].meta, "cpu_clean_version", -1),
}


def test_flag_off_is_byte_identical_to_today() -> None:
  cases = [("healthy", None), *MUTATIONS.items()]
  for name, mutate in cases:
    # reference (deduplicated) path, used by default
    inst, inputs = make_replay(), cpu_inputs(8, 16, 32)
    replaced = None
    if mutate is not None:
      candidate = mutate(inst, inputs)
      if isinstance(candidate, list): replaced = candidate
    check_inputs = replaced if replaced is not None else inputs
    expected = inst._valid_deduplicated(check_inputs)
    assert inst.valid(check_inputs) is expected, name
    # reference (non-deduplicated) path: ONPU_REPLAY_DEDUP_GUARDS=0 selects it
    inst2, inputs2 = make_replay(), cpu_inputs(8, 16, 32)
    replaced2 = None
    if mutate is not None:
      candidate2 = mutate(inst2, inputs2)
      if isinstance(candidate2, list): replaced2 = candidate2
    check_inputs2 = replaced2 if replaced2 is not None else inputs2
    with patch.object(rr, 'getenv', lambda key, default=0: 0 if key == 'ONPU_REPLAY_DEDUP_GUARDS' else default):
      expected2 = original_full_valid(inst2, check_inputs2)
      assert inst2.valid(check_inputs2) is expected2, name
    # the fast path must be completely inert when off
    assert inst._fast_valid_full_runs == inst._fast_valid_skips == inst._fast_valid_mismatch == 0


def test_fast_path_runs_authoritative_once_then_skips() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  assert (inst._fast_valid_full_runs, inst._fast_valid_skips) == (1, 0)
  for _ in range(5):
    assert inst.valid(inputs) is True
  assert (inst._fast_valid_full_runs, inst._fast_valid_skips) == (1, 5)


def test_fast_path_never_skips_the_input_shape_check() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  assert inst.valid(cpu_inputs(8, 16, 32)[:2]) is False         # count changed
  assert inst.valid([SimpleNamespace(buffer=Buffer("CPU", 7)), *inputs[1:]]) is False  # size changed
  assert inst._fast_valid_full_runs == 1  # rejected cheaply, epoch cache untouched


def test_plan_close_hook_forces_full_recheck() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  plan = inst.plans[0][0]
  epoch = rr._FAST_VALID_EPOCH
  RockchipMatmul.close(plan)  # hooked plan mutator, exactly what a re-prepare does
  assert rr._FAST_VALID_EPOCH > epoch
  assert not plan.owned and plan.bufs == ()
  assert inst.valid(inputs) is False
  assert inst._fast_valid_full_runs == 2
  # a failed authoritative run is never cached as a verdict: repair + epoch bump recovers
  plan.owned = [SimpleNamespace()]
  out_meta = SimpleNamespace(version=1, cpu_clean_version=1)
  plan.bufs = (SimpleNamespace(meta=out_meta), SimpleNamespace(), SimpleNamespace())
  inst.plans[0] = (plan, plan.bufs)
  RockchipMatmul.close(SimpleNamespace(owned=[], bufs=(), dev=SimpleNamespace()))  # hooked mutator: epoch moves
  assert inst.valid(inputs) is True


def test_allocator_copyin_hook_and_weight_version() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  weight_meta = next(iter(inst.weight_guards.values()))[0]
  epoch = rr._FAST_VALID_EPOCH
  # the real wrapper bumps the epoch only for a watched NPU meta, then runs the body
  with pytest.raises(AttributeError):
    RockchipAllocator._copyin(object(), SimpleNamespace(meta=weight_meta), None)
  assert rr._FAST_VALID_EPOCH > epoch
  weight_meta.version += 1  # what a weight upload does after the hook
  assert inst.valid(inputs) is False


def test_unrelated_buffer_allocations_do_not_bump_the_epoch() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  epoch = rr._FAST_VALID_EPOCH
  for _ in range(5):
    b=Buffer("CPU", 8); b.allocate(); b.deallocate()  # per-token churn on buffers the replay does not bake in
  assert rr._FAST_VALID_EPOCH == epoch
  for _ in range(3):
    assert inst.valid(inputs) is True
  assert (inst._fast_valid_full_runs, inst._fast_valid_skips) == (1, 3)


def test_watched_guard_storage_replacement_forces_recheck() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  guarded = next(iter(inst.guards))
  epoch = rr._FAST_VALID_EPOCH
  guarded.deallocate()
  assert rr._FAST_VALID_EPOCH > epoch  # watched buffer mutation bumps
  guarded.allocate()
  assert inst.valid(inputs) is False  # baked address no longer matches its recorded storage
  assert inst._fast_valid_full_runs == 2


def test_watch_registration_covers_guards_and_metas() -> None:
  inst = make_replay()
  assert all(buf in rr._FAST_VALID_WATCHED_BUFFERS for buf in inst.guards)
  assert all(meta_id in rr._FAST_VALID_WATCHED_META_IDS for meta_id in inst.weight_guards)
  assert all(id(meta) in rr._FAST_VALID_WATCHED_META_IDS for meta in inst._output_guards)


def test_attention_close_hook() -> None:
  inst, inputs = set_fast(make_replay()), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  program = RockchipAttentionProgram.__new__(RockchipAttentionProgram)
  program.closed = True  # close() returns early after the hook bumps the epoch
  RockchipAttentionProgram.close(program)  # hooked liveness mutator (no-op for a bare instance)
  assert inst.valid(inputs) is True  # its own state did not change, but the check reran
  assert inst._fast_valid_full_runs == 2


def test_debug_mode_raises_on_unhooked_invalidation() -> None:
  inst, inputs = set_fast(make_replay(), debug=True), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  inst.plans[0][0].bufs = ()  # mutate WITHOUT any hook/epoch bump: debug must catch it
  with pytest.raises(AssertionError, match='invalidation path is unhooked'):
    inst.valid(inputs)
  assert inst._fast_valid_mismatch == 1


def test_debug_mode_clean_sequence_never_mismatches() -> None:
  inst, inputs = set_fast(make_replay(), debug=True), cpu_inputs(8, 16, 32)
  assert inst.valid(inputs) is True
  RockchipMatmul.close(inst.plans[0][0])
  assert inst.valid(inputs) is False
  assert inst._fast_valid_mismatch == 0


def test_hook_wiring_covers_all_invalidation_sources() -> None:
  expected = {
    RockchipMatmul: ('prepare', 'close'), RockchipBatchedMatmul: ('prepare',),
    RockchipGroupedMatmul: ('prepare',), RockchipRoutedMatmul: ('prepare',),
    RockchipBucketedMatmul: ('prepare',), RockchipAttentionProgram: ('close',),
  }
  for cls, methods in expected.items():
    for method in methods:
      assert getattr(cls.__dict__[method], '_onpu_fast_valid_hook', None), f'{cls.__name__}.{method}'
  assert getattr(RockchipAllocator.__dict__['_copyin'], '_onpu_fast_valid_hook', None)
  for method in ('allocate', 'deallocate'):
    assert getattr(Buffer.__dict__[method], '_onpu_fast_valid_hook', None), f'Buffer.{method}'
  assert rr._FAST_VALID_HOOKS_INSTALLED


def test_output_guards_is_a_first_valid_one_off() -> None:
  inst, inputs = make_replay(register=False), cpu_inputs(8, 16, 32)  # flag OFF: full check every call
  assert '_output_guards' not in vars(inst)
  assert inst.valid(inputs) is True
  built = vars(inst)['_output_guards']
  assert built and inst.valid(inputs) is True
  assert vars(inst)['_output_guards'] is built  # cached: build cost is per-replay, not per-token
