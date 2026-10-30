"""Gemma's unit-scale attention can use the real bounded CPU kernels."""
import functools
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.gemma4 import Gemma4Block
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention
import test.models.test_gemma4 as gemma4_reference
import test.models.test_gemma3n as gemma3n_reference


def _install_cpu_attention(model):
  calls = []
  for index, block in enumerate(model.blk):
    window = block.config.sliding_window
    for phase, attribute, kernel in (('prefill', 'prefill_attention', prefill_attention),
                                     ('decode', 'decode_attention', decode_attention)):
      def counted(query, cache, length, *, fast_softmax=False, pool=None, score_scale=None, window=0,
                  _index=index, _phase=phase, _kernel=kernel):
        result = _kernel(query, cache, length, fast_softmax=fast_softmax, pool=pool,
                         score_scale=score_scale, window=window)
        # Keep tensor handles for comparison after the full forward realizes them.
        source_index = (next((i for i, earlier in enumerate(model.blk[:_index])
                              if bool(earlier.config.sliding_window) == bool(window)), None)
                        if model.blk[_index].shared else None)
        calls.append({'block': _index, 'phase': _phase, 'cache': cache,
                      'source_cache': model.blk[source_index].cache_kv if source_index is not None else None,
                      'source_index': source_index, 'window': window, 'score_scale': score_scale})
        return result
      setattr(block, attribute, functools.partial(counted, fast_softmax=False, score_scale=1.0, window=window))
  return calls


def _shared_cache_calls_match(calls):
  for row in calls:
    if row['source_index'] is None:
      continue
    np.testing.assert_array_equal(row['cache'].numpy(), row['source_cache'].numpy())


def _assert_full_model_paths(fixture, make_model, prompt, continuation, reset_tokens):
  generic = make_model()
  hooked = make_model()
  weights = fixture.weights(generic)
  calls = _install_cpu_attention(hooked)
  sequences = ((prompt, 0, prompt), (continuation, len(prompt), prompt + continuation),
               (reset_tokens, 0, reset_tokens))
  outputs = {'generic': [], 'hooked': []}
  for label, model in (('generic', generic), ('hooked', hooked)):
    for ids, start, full_prefix in sequences:
      call_start = len(calls) if label == 'hooked' else 0
      actual = model.logits(Tensor([ids], device='CPU'), start).numpy().ravel()
      if label == 'hooked':
        _shared_cache_calls_match(calls[call_start:])
      expected = fixture.reference(weights, full_prefix).astype(np.float32)
      np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)
      outputs[label].append(actual.copy())
  for ordinary, accelerated in zip(outputs['generic'], outputs['hooked']):
    np.testing.assert_allclose(accelerated, ordinary, rtol=3e-4, atol=2e-4)
  assert len(calls) == len(hooked.blk) * len(sequences)
  assert sum(row['phase'] == 'prefill' for row in calls) == 2 * len(hooked.blk)
  assert sum(row['phase'] == 'decode' for row in calls) == len(hooked.blk)
  assert all(row['score_scale'] == 1.0 for row in calls)
  assert all(row['window'] == hooked.blk[row['block']].config.sliding_window for row in calls)
  # Both generic fixture families deliberately include a short sliding layer;
  # the continuation crosses its window boundary and must retain causal masking.
  assert len(prompt) > hooked.blk[0].config.sliding_window
  assert continuation and continuation[0] == 9


def test_gemma4_cpu_attention_hook_matches_reference_cached_decode_and_reset():
  fixture = gemma4_reference.TestGemma4()
  _assert_full_model_paths(fixture, fixture.model, [3, 5, 12, 7], [9], [6, 2, 18])
  model = fixture.model()
  for block in model.blk:
    assert isinstance(block, Gemma4Block)
    assert block.attention_score_scale == 1.0
    assert block.v_norm.weight is None


def test_gemma3n_inherits_gemma_cpu_attention_hook_and_matches_reference():
  fixture = gemma3n_reference.TestGemma3n()
  _assert_full_model_paths(fixture, fixture.model, [3, 5, 12, 39], [9], [6, 2, 18])
  model = fixture.model()
  for block in model.blk:
    assert isinstance(block, Gemma4Block)
    assert block.attention_score_scale == 1.0
    assert block.v_norm.weight is None


@pytest.mark.parametrize('fixture_type', [gemma4_reference.TestGemma4, gemma3n_reference.TestGemma3n],
                         ids=['gemma4', 'gemma3n'])
def test_configure_binds_unit_scale_window_and_excludes_gemma_prepared_qkv(monkeypatch, fixture_type):
  """Placement setup uses the CPU hooks without opening or preparing NPU devices."""
  from tinygrad.llm.kernels.rockchip import RockchipLinear, RockchipProjections, configure

  model = fixture_type().model()
  # Placement selection can build the ordinary wrappers and fused-QKV group;
  # fake only the packed hardware preparation in this configuration test.
  monkeypatch.setattr(RockchipLinear, 'prepare', lambda self: None)
  monkeypatch.setattr(RockchipLinear, 'npu', lambda self, x: self.source(x))
  result = configure(model, prefill='on', decode='on', cpu_attention=True,
                     fuse_projections=True, fused_qkv=True)

  assert result['cpu_attention_blocks'] == len(model.blk)
  assert result['prepared_qkv_blocks'] == 0
  for block in model.blk:
    assert isinstance(block, Gemma4Block)
    assert block.prepared_attention_compatible is False
    assert not hasattr(block, 'prepared_qkv')
    if not block.shared: assert isinstance(block.attn_qkv, RockchipProjections)
    for hook_name in ('prefill_attention', 'decode_attention'):
      hook = getattr(block, hook_name)
      assert isinstance(hook, functools.partial)
      assert hook.keywords['score_scale'] == 1.0
      assert hook.keywords['window'] == block.config.sliding_window
      assert hook.keywords['fast_softmax'] is False
