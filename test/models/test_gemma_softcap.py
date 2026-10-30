"""Bounded CPU final-logit softcapping preserves Gemma logits and ties."""
import itertools
import numpy as np
import pytest
from dataclasses import replace
from tinygrad import Tensor
import test.models.test_gemma3n as gemma3n_reference
import test.models.test_gemma4 as gemma4_reference
from tinygrad.llm.kernels.rockchip_softcap import softcap


_VOCAB = 1031


def _model_with_large_tied_head(fixture):
  model = fixture.model()
  # Equal, nonzero rows make all vocabulary logits tie while extending beyond
  # the CPU kernel's 1024-element vector boundary and keeping generated IDs valid.
  row = np.random.default_rng(1031).normal(0, .04, model.config.dim).astype(np.float16)
  weight = np.broadcast_to(row, (_VOCAB, model.config.dim)).copy()
  model.output.weight = Tensor(weight, device='CPU').realize()
  model.config = replace(model.config, vocab_size=_VOCAB)
  return model


def _attach_counted_softcap(model):
  calls = []
  def counted(logits, cap):
    calls.append((logits.shape, cap))
    return softcap(logits, cap)
  model.cpu_logit_softcap = counted
  return calls


def _assert_model_paths(fixture):
  generic, optimized = _model_with_large_tied_head(fixture), _model_with_large_tied_head(fixture)
  weights = fixture.weights(generic)
  calls = _attach_counted_softcap(optimized)
  prompt = [3, 5, 12, 39] if fixture.__class__ is gemma3n_reference.TestGemma3n else [3, 5, 12, 7]
  sequences = ((prompt, 0, prompt), ([9], len(prompt), prompt + [9]), ([6, 2, 18], 0, [6, 2, 18]))
  for ids, start, prefix in sequences:
    expected = fixture.reference(weights, prefix).astype(np.float32)
    ordinary = generic.logits(Tensor([ids], device='CPU'), start).numpy().ravel()
    actual = optimized.logits(Tensor([ids], device='CPU'), start).numpy().ravel()
    np.testing.assert_allclose(ordinary, expected, rtol=3e-4, atol=2e-4)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)
    np.testing.assert_allclose(actual, ordinary, rtol=3e-4, atol=2e-4)
    assert actual.shape == (_VOCAB,)
    # The deliberately duplicated nonzero rows exercise both the large output
    # tail and the stable first-index result when softcapping preserves ties.
    assert np.all(actual == actual[0])
    assert int(actual.argmax()) == 0
  assert calls == [((1, _VOCAB), optimized.config.final_logit_softcap)] * len(sequences)

  expected_ids = []
  for _ in range(2): expected_ids.append(int(fixture.reference(weights, prompt + expected_ids).argmax()))
  generic_ids = list(itertools.islice(generic.generate(prompt.copy(), chunk_size=2), 2))
  optimized_ids = list(itertools.islice(optimized.generate(prompt.copy(), chunk_size=2), 2))
  assert generic_ids == optimized_ids == expected_ids
  assert len(calls) > len(sequences)


def test_gemma4_cpu_softcap_matches_reference_prefill_decode_reset_and_greedy_ties():
  _assert_model_paths(gemma4_reference.TestGemma4())


def test_gemma3n_cpu_softcap_matches_reference_prefill_decode_reset_and_greedy_ties():
  _assert_model_paths(gemma3n_reference.TestGemma3n())


def test_cpu_softcap_kernel_matches_tanh_arithmetic_and_preserves_saturated_ties():
  cap = 3.0
  values = np.linspace(-8, 8, _VOCAB, dtype=np.float32)
  values[1000], values[1001] = 1000., 1001.
  logits = Tensor(values, device='CPU').realize()
  actual = softcap(logits, cap).numpy()
  expected = np.tanh(values / cap) * cap
  np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
  assert actual[1000] == actual[1001] == cap
  assert int(actual.argmax()) == 1000


def test_rockchip_configure_installs_and_clears_bounded_cpu_softcap_without_npu():
  from tinygrad.llm.kernels.rockchip import configure

  model = gemma3n_reference.TestGemma3n().model()
  installed = configure(model, cpu_softcap=True, prefill='off', decode='off')
  assert installed['cpu_softcap'] is True
  assert model.cpu_logit_softcap is softcap

  cleared = configure(model, cpu_softcap=False, prefill='off', decode='off')
  assert cleared['cpu_softcap'] is False
  assert not hasattr(model, 'cpu_logit_softcap')


@pytest.mark.parametrize('cap', [0.0, 129.0])
def test_rockchip_configure_leaves_zero_or_unbounded_softcap_on_generic_path(cap):
  from tinygrad.llm.kernels.rockchip import configure

  model = gemma3n_reference.TestGemma3n().model()
  model.config = replace(model.config, final_logit_softcap=cap)
  result = configure(model, cpu_softcap=True, prefill='off', decode='off')
  assert result['cpu_softcap'] is False
  assert not hasattr(model, 'cpu_logit_softcap')

  logits = Tensor(np.linspace(-5, 5, _VOCAB, dtype=np.float32), device='CPU').realize()
  actual = model._softcap_logits(logits).numpy()
  expected = logits.numpy() if cap == 0 else np.tanh(logits.numpy() / cap) * cap
  if cap == 0:
    np.testing.assert_array_equal(actual, expected)
  else:
    generic = ((logits / cap).tanh() * cap).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), generic.view(np.uint32))
    # Generic tanh uses an FP32 sigmoid expression; its cancellation error is
    # multiplied by cap. Independently bound that error as well as exact fallback.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=4*np.finfo(np.float32).eps*cap)
