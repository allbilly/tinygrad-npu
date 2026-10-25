"""A failed stateful decoder call must not make its old prefix reusable."""
import itertools, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig


def models():
  config = TransformerConfig(3,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=32)
  config = replace(config,ssm=SSMConfig(3,8,2,4,32),ssm_layers=(True,False,True))
  actual,reference = Transformer(config),Transformer(config)
  rng = np.random.default_rng(2110)
  for name,p in nn.state.get_state_dict(actual).items():
    values = rng.normal(0,.12,p.shape).astype(np.float32)
    if 'norm.weight' in name:values += 1
    if name.endswith('ssm_a'):values = -rng.uniform(.5,2,p.shape).astype(np.float32)
    p.replace(Tensor(values,device='CPU').realize())
  nn.state.load_state_dict(reference,{k:v.clone().realize() for k,v in nn.state.get_state_dict(actual).items()},verbose=False)
  return actual,reference


def first_tokens(model, prompt, count=3):
  generated = model.generate(prompt.copy())
  try:return list(itertools.islice(generated,count))
  finally:generated.close()


class TestGenerationRecovery(unittest.TestCase):
  def assert_matches_fresh(self, actual, reference, prompt):
    self.assertEqual(first_tokens(actual,prompt),first_tokens(reference,prompt))
    self.assertEqual(actual._cached_tokens,reference._cached_tokens)
    for a,b in zip(actual.blk,reference.blk):
      for name in ('conv_state','recurrent_state'):
        if hasattr(a,name):np.testing.assert_array_equal(getattr(a,name).numpy(),getattr(b,name).numpy())

  def test_failure_after_recurrent_prefill_execution_resets_prefix_and_retry(self):
    for failure in (InterruptedError,KeyboardInterrupt):
      with self.subTest(failure=failure.__name__):
        actual,reference = models()
        first_tokens(actual,[2,5,11,7],1)
        original,calls = actual.greedy_rollout_jit,0

        def interrupted(*args):
          nonlocal calls
          result = original(*args).realize()
          calls += 1
          if calls==2:raise failure('Execution completed before prefill failed')
          return result

        actual.greedy_rollout_jit = interrupted
        # A different prompt restarts at zero; the second call has a valid
        # one-token prefix, but its recurrent state already includes token two.
        with self.assertRaises(failure):first_tokens(actual,[8,3,19,9],1)
        self.assertEqual(actual._cached_tokens,[])
        self.assertFalse(actual._greedy_generation)
        actual.greedy_rollout_jit = original
        self.assert_matches_fresh(actual,reference,[8,3,19,9])

  def test_failure_after_cached_decode_execution_resets_prefix_and_retry(self):
    for failure in (InterruptedError,KeyboardInterrupt):
      with self.subTest(failure=failure.__name__):
        actual,reference = models()
        prompt = [2,5,11,7]
        first_tokens(actual,prompt,1)
        self.assertEqual(actual._cached_tokens,prompt)
        original = actual.greedy_rollout_jit

        def interrupted(*args):
          original(*args).realize()
          raise failure('Execution completed before decode failed')

        actual.greedy_rollout_jit = interrupted
        extended = prompt+[3,19,9,1]
        with self.assertRaises(failure):first_tokens(actual,extended,1)
        self.assertEqual(actual._cached_tokens,[])
        self.assertFalse(actual._greedy_generation)
        actual.greedy_rollout_jit = original
        self.assert_matches_fresh(actual,reference,extended)

  def test_clean_generator_close_preserves_completed_recurrent_prefix(self):
    actual,reference = models()
    prompt = [2,5,11,7]
    generated = actual.generate(prompt.copy())
    token = next(generated)
    before = [(block.conv_state.numpy().copy(),block.recurrent_state.numpy().copy())
              for block in actual.blk if hasattr(block,'recurrent_state')]
    generated.close()
    self.assertEqual(actual._cached_tokens,prompt)
    for block,(conv,state) in zip((b for b in actual.blk if hasattr(b,'recurrent_state')),before):
      np.testing.assert_array_equal(block.conv_state.numpy(),conv)
      np.testing.assert_array_equal(block.recurrent_state.numpy(),state)
    self.assertEqual(actual.get_start_pos(prompt+[token]),len(prompt))
    self.assert_matches_fresh(actual,reference,prompt+[token])

  def test_failure_during_live_decode_does_not_reuse_advanced_state(self):
    actual,reference = models()
    prompt = [2,5,11,7]
    generated = actual.generate(prompt.copy())
    token = next(generated)
    original = actual.greedy_rollout_jit

    def interrupted(*args):
      original(*args).realize()
      raise InterruptedError('Execution completed before live decode failed')

    actual.greedy_rollout_jit = interrupted
    with self.assertRaises(InterruptedError):next(generated)
    self.assertEqual(actual._cached_tokens,[])
    self.assertFalse(actual._greedy_generation)
    actual.greedy_rollout_jit = original
    self.assert_matches_fresh(actual,reference,prompt+[token])


if __name__=='__main__':unittest.main()
