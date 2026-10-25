"""Bitwise native Mali prefix-history qualification; no floating-point tolerance."""
import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.mali_recurrent import gated_delta_scan
from test.models.test_mali_recurrent import operands


def bits_equal(actual,expected):
  np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliRecurrentTrace(unittest.TestCase):
  def test_history_keeps_core_final_state_bits_and_every_prefix_at_identical_capacity(self):
    rng = np.random.default_rng(3521)
    capacity = 8
    for batch,heads,key_dim,value_dim in ((1,2,8,5),(2,3,32,13),(1,16,128,128)):
      for position,rows in ((0,8),(7,8),(7,5),(0,2),(7,1)):
        with self.subTest(position=position,rows=rows):
          arrays = operands(rng,batch,heads,capacity,key_dim,value_dim,1)
          values = tuple(Tensor(value,device='CL').realize() for value in arrays)
          initial = rng.normal(0,.03,(batch,heads,value_dim,key_dim)).astype(np.float32)
          if position==0:initial[:]=np.nan
          baseline = Tensor(initial.copy(),device='CL').realize()
          traced = Tensor(initial.copy(),device='CL').realize()
          history = Tensor.full((8,batch,heads,value_dim,key_dim),float('nan'),device='CL').realize()
          pos = UOp.variable('history_test_pos',0,64).bind(position)
          count = UOp.variable('history_test_rows',1,8).bind(rows)
          expected = gated_delta_scan(*values,baseline,pos,count).numpy()
          actual = gated_delta_scan(*values,traced,pos,count,history=history).numpy()
          bits_equal(actual,expected)
          bits_equal(traced.numpy(),baseline.numpy())
          saved = history.numpy().copy()
          for prefix in range(1,rows+1):
            state = Tensor(initial.copy(),device='CL').realize()
            _ = gated_delta_scan(*values,state,pos,UOp.variable('history_prefix_rows',1,8).bind(prefix)).numpy()
            bits_equal(saved[prefix-1],state.numpy())
          self.assertTrue(np.isnan(saved[rows:]).all())

  def test_explicit_jit_workspace_mixed_rows_reset_and_scratch_release_preserve_bits(self):
    rng = np.random.default_rng(3522)
    state = Tensor.zeros(1,2,5,8,device='CL').realize()
    baseline = Tensor.zeros(1,2,5,8,device='CL').realize()
    history = Tensor.empty(8,1,2,5,8,device='CL').realize()
    identity = history.uop.buf_uop
    run = TinyJit(lambda q,k,v,beta,alpha,p,r,h:gated_delta_scan(q,k,v,beta,alpha,state,p,r,history=h).realize())
    for index,(rows,position) in enumerate(((8,0),(7,8),(5,15),(1,20),(2,21),(3,0),(8,3))):
      arrays = operands(rng,1,2,8,8,5,1)
      for value in arrays:value[:,:,rows:]=np.nan
      values = tuple(Tensor(value,device='CL').realize() for value in arrays)
      pos = UOp.variable('history_jit_pos',0,64).bind(position)
      count = UOp.variable('history_jit_rows',1,8).bind(rows)
      expected = gated_delta_scan(*values,baseline,pos,count).numpy()
      actual = run(*values,pos,count,history).numpy()
      bits_equal(actual,expected)
      bits_equal(state.numpy(),baseline.numpy())
      bits_equal(history.numpy()[rows-1],state.numpy())
      self.assertIs(history.uop.buf_uop,identity)
      if index>=2:run.captured.free_intermediates()
      # History is externally held, not memory-planned scratch.
      self.assertTrue(history.uop.buf_uop.is_realized)
    self.assertEqual(len(run.captured.expected_input_info),6)

  def test_invalid_history_capacity_dtype_or_view_is_rejected_before_state_write(self):
    arrays = operands(np.random.default_rng(3523),1,2,8,8,5,1)
    values = tuple(Tensor(value,device='CL').realize() for value in arrays)
    state = Tensor.zeros(1,2,5,8,device='CL').realize()
    histories = (Tensor.empty(7,1,2,5,8,device='CL').realize(),
                 Tensor.empty(8,1,2,5,8,device='CL',dtype='float16').realize(),
                 Tensor.empty(8,1,2,8,5,device='CL').realize().permute(0,1,2,4,3))
    for history in histories:
      with self.assertRaisesRegex(ValueError,'prefix state history'):gated_delta_scan(*values,state,0,8,history=history)
    self.assertFalse(state.numpy().any())


if __name__=='__main__':unittest.main()
