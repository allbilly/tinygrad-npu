"""Fast-path guard epoch: skip the O(#guards+#plans+#outputs) identity/version walks
only when nothing that can invalidate them has happened since the last validation.

`ONPU_REPLAY_FAST_VALID` is opt-in (default off). The fast path keeps the cheap
complete checks (input buffers, plan ownership, attention programs) on every call
and gates only the expensive walks behind an epoch made of counters tinygrad
already maintains:

  * GlobalCounters.kernel_count - bumped by every executed kernel, including a
    different NativeReplay, the JIT fallback path, tracked buffer writes and NPU
    syncs (those run as kernels or inside a replay);
  (A bare Buffer allocate/free does not need to invalidate: the guarded buffers are
  strongly referenced by `self.kept` and the plans, so their storage cannot be
  reallocated while the replay lives.)

These tests use counting stand-ins for each walk component so a skipped walk is
observable, and they assert that every invalidation class still reaches (and is
rejected by) the full walk. A mutation that bypasses both counters AND direct
kernel/allocation activity is not visible to the fast path; such a mutation is not
produced by any tracked path in this runtime, and the flag is default-off.
"""
from types import SimpleNamespace
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.llm.kernels.rockchip_replay import NativeReplay


class StorageProbe:
  device = 'CPU'

  def __init__(self, storage):
    self._value, self.reads = storage, 0

  @property
  def _storage(self):
    self.reads += 1
    return self._value


class VersionProbe:
  def __init__(self, value):
    self._value, self.reads = value, 0

  @property
  def version(self):
    self.reads += 1
    return self._value

  @version.setter
  def version(self, value):
    self._value = value


class CleanProbe:
  def __init__(self):
    self.version, self.clean, self.reads = 3, 3, 0

  @property
  def cpu_clean_version(self):
    self.reads += 1
    return self.clean


class PlanProbe:
  def __init__(self, buffers):
    self.owned, self.buffers, self.reads = [object()], buffers, 0

  @property
  def bufs(self):
    self.reads += 1
    return self.buffers


class Fixture:
  def __init__(self, plans: int = 175):
    token = Tensor([7], device='CPU', dtype=dtypes.int32).realize().uop.base
    replay = NativeReplay.__new__(NativeReplay)
    replay.roots = [token.buffer]
    replay.input_sizes = [4]
    self.storage = object()
    self.guard = StorageProbe(self.storage)
    replay.guards = {self.guard: self.storage}
    self.weight = VersionProbe(8)
    replay.weight_guards = {1: (self.weight, 8)}
    self.attention = SimpleNamespace(closed=False, owned=[object()])
    replay.attention_programs = [self.attention]
    self.metas = [CleanProbe() for _ in range(2)]
    self.plans = []
    for _ in range(plans):
      buffers = tuple(SimpleNamespace(meta=self.metas[0]) for _ in range(3))
      self.plans.append((PlanProbe(buffers), buffers))
    replay.plans = self.plans
    replay._valid_epoch = None
    self.replay, self.token = replay, token

  def walks(self) -> int:
    return (self.guard.reads + self.weight.reads + sum(m.reads for m in self.metas)
            + sum(plan.reads for plan, _ in self.plans))


@pytest.fixture(autouse=True)
def restore_counters():
  kernel_count = GlobalCounters.kernel_count
  try:
    yield
  finally:
    GlobalCounters.kernel_count = kernel_count


def test_fast_path_skips_walks_when_epoch_unchanged():
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay.valid([ctx.token])
    walked = ctx.walks()
    assert walked > 0                                   # full walk ran
    assert ctx.replay.valid([ctx.token])                # fast path
    assert ctx.walks() == walked                        # zero guard/version/plan-buf reads
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() == walked


def test_kernel_execution_forces_revalidation():
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay.valid([ctx.token])
    walked = ctx.walks()
    GlobalCounters.kernel_count += 1                    # any kernel: possible tracked write
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() > walked                         # revalidated
    revalidated = ctx.walks()
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() == revalidated                   # stamp refreshed -> fast again


def test_allocation_churn_alone_does_not_invalidate():
  """Guarded buffers are kept alive, so a bare alloc/free cannot change a guard.

  This is the safety half of using kernel_count (not mem_used) as the epoch: the
  per-token scalar input copies allocate/free every token and must NOT force the
  expensive revalidation."""
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay.valid([ctx.token])
    walked = ctx.walks()
    GlobalCounters.mem_used += 4096
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() == walked


def test_no_fast_path_before_first_successful_validation():
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay._valid_epoch is None
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() > 0                              # full walk, not the fast path


@pytest.mark.parametrize('mutation', ('input_count', 'input_size', 'plan_released',
                                      'attention_closed', 'attention_released'))
def test_cheap_checks_reject_on_the_fast_path(mutation):
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay.valid([ctx.token])
    inputs = [ctx.token]
    if mutation == 'input_count':
      inputs = []
    elif mutation == 'input_size':
      inputs = [Tensor([7, 8], device='CPU', dtype=dtypes.int32).realize().uop.base]
    elif mutation == 'plan_released':
      ctx.plans[0][0].owned = []
    elif mutation == 'attention_closed':
      ctx.attention.closed = True
    else:
      ctx.attention.owned = []
    assert not ctx.replay.valid(inputs)


@pytest.mark.parametrize('mutation', ('guard_storage', 'weight_version', 'plan_bufs', 'output_dirty'))
def test_epoch_gated_invalidations_reach_the_full_walk(mutation):
  """These are only detectable by the expensive walks; the kernel/allocation epoch
  bump that accompanies a real tracked change must force the walk and reject."""
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=1):
    assert ctx.replay.valid([ctx.token])
    GlobalCounters.kernel_count += 1                    # as a tracked write / re-prepare would
    if mutation == 'guard_storage':
      ctx.guard._value = object()
    elif mutation == 'weight_version':
      ctx.weight.version = 9
    elif mutation == 'plan_bufs':
      ctx.plans[0][0].buffers = tuple(list(ctx.plans[0][1]))
    else:
      ctx.metas[0].clean -= 1
    assert not ctx.replay.valid([ctx.token])
    assert ctx.replay._valid_epoch is None


def test_flag_off_always_runs_the_full_walk():
  ctx = Fixture()
  with Context(ONPU_REPLAY_FAST_VALID=0):
    assert ctx.replay.valid([ctx.token])
    walked = ctx.walks()
    assert walked > 0
    assert ctx.replay.valid([ctx.token])
    assert ctx.walks() > walked                         # default behaviour unchanged


def test_fast_path_matches_full_walk_when_state_is_stable():
  with Context(ONPU_REPLAY_FAST_VALID=0):
    control = Fixture()
    expected = control.replay.valid([control.token])
  with Context(ONPU_REPLAY_FAST_VALID=1):
    candidate = Fixture()
    assert candidate.replay.valid([candidate.token]) is expected
    assert candidate.replay.valid([candidate.token]) is expected
