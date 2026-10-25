import itertools, unittest
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestInterruptedGenerationCache(unittest.TestCase):
  def test_diverging_prefill_retains_only_completed_prefix_and_retry_matches_fresh(self):
    for interrupt_at in (1,3):
      with self.subTest(interrupt_at=interrupt_at):
        Tensor.manual_seed(79)
        config = TransformerConfig(1,16,32,2,1,1e-5,32,8,10000.,8,8,max_context=96)
        model,reference = Transformer(config),Transformer(config)
        nn.state.load_state_dict(reference,nn.state.get_state_dict(model),verbose=False)
        previous = model.generate([1]*48,chunk_size=8)
        next(previous)
        previous.close()
        tokens = [1]*8+[2]*32
        original,calls = model.greedy_prefill_jit,0

        def interrupted(*args):
          nonlocal calls
          calls += 1
          if calls==interrupt_at: raise InterruptedError('Stop between complete chunks')
          return original(*args)

        model.greedy_prefill_jit = interrupted
        with self.assertRaises(InterruptedError):next(model.generate(tokens.copy(),chunk_size=8))
        self.assertEqual(model._cached_tokens,tokens[:8+(interrupt_at-1)*8])
        model.greedy_prefill_jit = original
        actual = list(itertools.islice(model.generate(tokens.copy(),chunk_size=8),3))
        expected = list(itertools.islice(reference.generate(tokens.copy(),chunk_size=8),3))
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
