"""CPU prepared static Gemma3n projections preserve model and cache behavior."""
import itertools
import numpy as np
import pytest
from tinygrad import Tensor, Device, dtypes, nn
import test.models.test_gemma3n as gemma3n_reference
from tinygrad.llm.kernels.rockchip_static import configure


def _count_hooks(model):
  calls = {}
  for name in ('altup_input_projections', 'altup_output_projections'):
    hooks = getattr(model, name)
    group = 'input' if name == 'altup_input_projections' else 'output'
    for index, hook in enumerate(hooks):
      key = (group, index)
      calls[key] = 0
      def counted(x, *, _hook=hook, _key=key):
        calls[_key] += 1
        return _hook(x)
      hooks[index] = counted
  return calls


def _assert_logits_match(model, fixture, weights, ids, start, prefix):
  actual = model.logits(Tensor([ids], device='CPU'), start).numpy().ravel()
  expected = fixture.reference(weights, prefix).astype(np.float32)
  np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=3e-4)
  return actual


def test_static_projection_configure_preserves_full_gemma3n_paths_and_state():
  fixture = gemma3n_reference.TestGemma3n()
  generic, prepared = fixture.model(), fixture.model()
  weights = fixture.weights(generic)
  state_before = nn.state.get_state_dict(prepared)
  state_keys = tuple(state_before)
  state_identities = {key: id(value) for key, value in state_before.items()}

  assert configure(prepared) == 6
  state_after = nn.state.get_state_dict(prepared)
  assert tuple(state_after) == state_keys
  assert {key: id(value) for key, value in state_after.items()} == state_identities
  assert all(hasattr(prepared, key) for key in ('altup_input_projections', 'altup_output_projections'))
  calls = _count_hooks(prepared)

  sequences = (([3, 5, 12, 39], 0, [3, 5, 12, 39]),
               ([9], 4, [3, 5, 12, 39, 9]),
               ([6, 2, 18], 0, [6, 2, 18]))
  for ids, start, prefix in sequences:
    ordinary = _assert_logits_match(generic, fixture, weights, ids, start, prefix)
    accelerated = _assert_logits_match(prepared, fixture, weights, ids, start, prefix)
    np.testing.assert_allclose(accelerated, ordinary, rtol=4e-4, atol=3e-4)

  assert set(calls) == {('input', i) for i in range(3)} | {('output', i) for i in range(3)}
  assert all(count == len(sequences) for count in calls.values())

  prompt = [3, 5, 12, 39]
  expected_ids = []
  for _ in range(2): expected_ids.append(int(fixture.reference(weights, prompt + expected_ids).argmax()))
  ordinary_ids = list(itertools.islice(generic.generate(prompt.copy(), chunk_size=2), 2))
  prepared_ids = list(itertools.islice(prepared.generate(prompt.copy(), chunk_size=2), 2))
  assert ordinary_ids == prepared_ids == expected_ids

  assert configure(prepared, enabled=False) == 0
  assert not hasattr(prepared, 'altup_input_projections')
  assert not hasattr(prepared, 'altup_output_projections')
  _assert_logits_match(prepared, fixture, weights, [4, 1], 0, [4, 1])


def test_static_projection_cache_is_a_weight_snapshot_and_reconfigure_refreshes_it():
  fixture = gemma3n_reference.TestGemma3n()
  generic, prepared = fixture.model(), fixture.model()
  assert configure(prepared) == 6

  for model in (generic, prepared):
    for name in ('altup_proj', 'altup_unembd_proj'):
      table = getattr(model, name)
      old = table['weight']
      table['weight'] = (old + .25).cast(dtypes.half).realize()
  changed_weights = fixture.weights(generic)
  ids = [3, 5, 12, 39]
  expected = fixture.reference(changed_weights, ids).astype(np.float32)
  generic_output = generic.logits(Tensor([ids], device='CPU'), 0).numpy().ravel()
  stale_snapshot_output = prepared.logits(Tensor([ids], device='CPU'), 0).numpy().ravel()
  np.testing.assert_allclose(generic_output, expected, rtol=4e-4, atol=3e-4)
  assert not np.allclose(stale_snapshot_output, expected, rtol=1e-5, atol=1e-5)

  assert configure(prepared) == 6
  refreshed = prepared.logits(Tensor([ids], device='CPU'), 0).numpy().ravel()
  np.testing.assert_allclose(refreshed, expected, rtol=4e-4, atol=3e-4)


@pytest.mark.parametrize('unsupported', ['dtype', 'shape', 'table'])
def test_static_projection_configure_leaves_unsupported_weights_on_generic_path(unsupported):
  model = gemma3n_reference.TestGemma3n().model()
  assert configure(model) == 6
  if unsupported == 'dtype':
    model.altup_proj['weight'] = model.altup_proj['weight'].float()
  elif unsupported == 'shape':
    model.altup_unembd_proj['weight'] = model.altup_unembd_proj['weight'][0]
  else:
    model.altup_proj = []
  assert configure(model) == 0
  assert not hasattr(model, 'altup_input_projections')
  assert not hasattr(model, 'altup_output_projections')


def test_static_projection_uses_existing_workers_inside_and_outside_pool():
  from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  from tinygrad.llm.kernels.rockchip_static import prepare
  _needs_pool()
  with _pool_test_lock():
    pool=AttentionPool()
    try:
      rng=np.random.default_rng(129257)
      weight=Tensor(rng.normal(0,.03,(129,257)).astype(np.float16),device='CPU').realize()
      ordinary,pooled=prepare(weight),prepare(weight,pool=pool)
      for rows in (1,17):
        x=Tensor(rng.normal(0,.3,(rows,257)).astype(np.float32),device='CPU').realize()
        with pool:
          expected=ordinary(x).numpy().copy()
          actual=pooled(x).numpy().copy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        np.testing.assert_array_equal(pooled(x).numpy().view(np.uint32),expected.view(np.uint32))
      assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)
    finally:pool.close()
