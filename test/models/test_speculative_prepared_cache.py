"""Actual tiny-model eviction, returned-view and recurrent-state checks."""
import os, unittest, weakref
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.speculative import _PreparedTargetForward
from test.models.test_speculative import real_pair
from test.models.test_speculative_prepared import assert_states, symbolic_eager_reference


def compare_outputs(test, actual, expected):
  test.assertEqual(actual.logits.shape,expected.logits.shape)
  for a,b in zip((actual.logits,actual.hidden,*actual.features),(expected.logits,expected.hidden,*expected.features)):
    np.testing.assert_array_equal(a.numpy().view(np.uint32),b.numpy().view(np.uint32))


class TestPreparedTargetCache(unittest.TestCase):
  device = 'CPU'

  def models(self, hybrid):
    raw,target = real_pair(hybrid)
    if self.device!='CPU':
      from tinygrad.llm.kernels.mali import configure
      for model in (raw,target):
        for parameter in nn.state.get_parameters(model):parameter.to_(self.device)
        configure(model,recurrent_scan=hybrid,pack_weights=False)
    return raw,target

  def test_rotating_prompt_remainders_keep_hot_full_chunk_and_two_prefill_graphs(self):
    raw,target = self.models(False)
    raw.target_forward = symbolic_eager_reference(raw)
    helper = _PreparedTargetForward(target,(0,1),dynamic_rows=False)
    full = None
    for rows in (4,4,1,4,2,4,3,4,1):
      ids = [(j*3+1)%13 for j in range(rows)]
      expected = raw.target_forward(Tensor([ids],device='CPU',dtype='int32'),0,capture_layers=(0,1),last_only=True)
      actual = helper(ids,0,'prefill',4,True)
      compare_outputs(self,actual,expected)
      assert_states(self,target,raw,rows)
      self.assertLessEqual(sum(key[0]=='prefill' for key in helper.calls),2)
      self.assertLessEqual(len(helper.calls),16)
      if full is None:full = helper.calls[('prefill',4,True)]
      self.assertIs(helper.calls[('prefill',4,True)],full)
    self.assertGreaterEqual(helper.evictions,3)
    self.assertIsNotNone(full)
    if full is not None:self.assertIsNotNone(full.jit.captured)

  def test_global_lru_preserves_live_views_releases_cycles_and_recaptures_hybrid_state_exactly(self):
    raw,target = self.models(True)
    raw.target_forward = symbolic_eager_reference(raw)
    helper = _PreparedTargetForward(target,(0,1),dynamic_rows=False,max_graphs=2,max_prefill_graphs=2)
    def run(rows, position, phase):
      ids = [(position+j*3+1)%13 for j in range(rows)]
      expected = raw.target_forward(Tensor([ids],device='CPU',dtype='int32'),position,capture_layers=(0,1))
      actual = helper(ids,position,phase,4,False)
      compare_outputs(self,actual,expected)
      assert_states(self,target,raw,position+rows)
      self.assertLessEqual(len(helper.calls),2)
      return actual
    for position in (0,2,4):run(2,position,'verify')
    self.assertIsNotNone(helper.calls[('verify',2,False)].jit.captured)
    run(1,6,'replay')
    replay = weakref.ref(helper.calls[('replay',1,False)])
    held = run(2,7,'verify')  # A cache hit must promote this graph to most recent.
    held_values = tuple(value.numpy().copy() for value in (held.logits,held.hidden,*held.features))
    old = weakref.ref(helper.calls[('verify',2,False)])
    run(3,0,'prefill')
    self.assertIsNone(replay())  # LRU eviction, not insertion order.
    self.assertIn(('verify',2,False),helper.calls)
    run(2,3,'replay')
    self.assertIsNone(old())  # No cyclic-GC call is needed to drop the graph.
    self.assertNotIn(('verify',2,False),helper.calls)
    for value,expected in zip((held.logits,held.hidden,*held.features),held_values):
      np.testing.assert_array_equal(value.numpy().view(np.uint32),expected.view(np.uint32))
    assert_states(self,target,raw,5)  # Reading evicted outputs must not rerun state writes.
    run(2,5,'verify')
    self.assertEqual(helper.calls[('verify',2,False)].jit.cnt,1)
    for position in (7,9):run(2,position,'verify')
    self.assertIsNotNone(helper.calls[('verify',2,False)].jit.captured)
    for value,expected in zip((held.logits,held.hidden,*held.features),held_values):
      np.testing.assert_array_equal(value.numpy().view(np.uint32),expected.view(np.uint32))
    assert_states(self,target,raw,11)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestPreparedTargetCacheMali(TestPreparedTargetCache):
  device = 'CL'


if __name__=='__main__':unittest.main()
