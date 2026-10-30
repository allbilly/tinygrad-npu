"""Gemma CPU GELU hooks preserve the decoder's independent equations."""
import itertools
import numpy as np
import pytest
from dataclasses import replace
from tinygrad import Tensor, nn
import test.models.test_gemma3n as gemma3n_reference
import test.models.test_gemma4 as gemma4_reference
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu


def _install_counted_gelu(model):
  calls = {index: 0 for index in range(len(model.blk))}
  for index, block in enumerate(model.blk):
    def counted(gate, up, *, _index=index):
      calls[_index] += 1
      return gated_gelu(gate, up)
    block.cpu_gelu_activation = counted
  return calls


def _assert_logits(model, fixture, weights, ids, start, prefix, *, rtol, atol):
  actual = model.logits(Tensor([ids], device='CPU'), start).numpy().ravel()
  expected = fixture.reference(weights, prefix).astype(np.float32)
  np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
  return actual


def _model_hook_paths(fixture, *, static_projection=False, rtol, atol):
  generic, optimized = fixture.model(), fixture.model()
  weights = fixture.weights(generic)
  state_before = nn.state.get_state_dict(optimized)
  state_keys = tuple(state_before)
  state_ids = {key: id(value) for key, value in state_before.items()}

  if static_projection:
    from tinygrad.llm.kernels.rockchip_static import configure as configure_static
    assert configure_static(optimized) == 6
  calls = _install_counted_gelu(optimized)
  state_after = nn.state.get_state_dict(optimized)
  assert tuple(state_after) == state_keys
  assert {key: id(value) for key, value in state_after.items()} == state_ids

  sequences = (([3, 5, 12, 39] if fixture.__class__ is gemma3n_reference.TestGemma3n else [3, 5, 12, 7],
                0,
                [3, 5, 12, 39] if fixture.__class__ is gemma3n_reference.TestGemma3n else [3, 5, 12, 7]),
               ([9], 4,
                ([3, 5, 12, 39, 9] if fixture.__class__ is gemma3n_reference.TestGemma3n else [3, 5, 12, 7, 9])),
               ([6, 2, 18], 0, [6, 2, 18]))
  for ids, start, prefix in sequences:
    reference = _assert_logits(generic, fixture, weights, ids, start, prefix, rtol=rtol, atol=atol)
    actual = _assert_logits(optimized, fixture, weights, ids, start, prefix, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol)

  # Every block has a gated FFN and a gated per-layer side projection.
  assert calls == {index: 2 * len(sequences) for index in range(len(optimized.blk))}

  prompt = sequences[0][0]
  generated = []
  for _ in range(2): generated.append(int(fixture.reference(weights, prompt + generated).argmax()))
  generic_ids = list(itertools.islice(generic.generate(prompt.copy(), chunk_size=2), 2))
  optimized_ids = list(itertools.islice(optimized.generate(prompt.copy(), chunk_size=2), 2))
  assert generic_ids == optimized_ids == generated

  return optimized


def test_gemma4_cpu_gelu_ffn_and_side_projection_match_numpy_cached_decode_and_reset():
  fixture = gemma4_reference.TestGemma4()
  optimized = _model_hook_paths(fixture, rtol=2e-4, atol=1e-4)
  for block in optimized.blk:
    assert block.cpu_gelu_activation is not None


def test_gemma3n_cpu_gelu_sparse_ffn_side_projection_and_static_projection_interact():
  fixture = gemma3n_reference.TestGemma3n()
  _model_hook_paths(fixture, static_projection=True, rtol=4e-4, atol=3e-4)


def test_rockchip_configure_installs_and_clears_cpu_gelu_hooks_without_npu():
  from tinygrad.llm.kernels.rockchip import configure

  model = gemma4_reference.TestGemma4().model()
  installed = configure(model, cpu_gelu=True, prefill='off', decode='off')
  assert installed['cpu_gelu_blocks'] == len(model.blk)
  assert all(block.cpu_gelu_activation is gated_gelu for block in model.blk)

  cleared = configure(model, cpu_gelu=False, prefill='off', decode='off')
  assert cleared['cpu_gelu_blocks'] == 0
  assert all(not hasattr(block, 'cpu_gelu_activation') for block in model.blk)


@pytest.mark.parametrize('field,value', [('ffn_activation', 'silu'), ('swiglu_limit', 1.0),
                                         ('swiglu_up_bias', 0.25), ('num_experts', 2)])
def test_rockchip_configure_excludes_unsupported_gelu_configurations(field, value):
  from tinygrad.llm.kernels.rockchip import configure

  model = gemma3n_reference.TestGemma3n().model()
  model.blk[0].config = replace(model.blk[0].config, **{field: value})
  result = configure(model, cpu_gelu=True, prefill='off', decode='off')
  assert result['cpu_gelu_blocks'] == len(model.blk) - 1
  assert not hasattr(model.blk[0], 'cpu_gelu_activation')
  assert all(block.cpu_gelu_activation is gated_gelu for block in model.blk[1:])
