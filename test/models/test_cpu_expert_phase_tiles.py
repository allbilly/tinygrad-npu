import unittest
import numpy as np
from types import SimpleNamespace
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.model import ExpertWeights
from tinygrad.llm.kernels.rockchip_experts import CPUExpertWeights, configure


def make_experts(experts,columns,depth,seed=20261108):
  rng=np.random.default_rng(seed)
  original=ExpertWeights(experts,depth,columns)
  original.weight=Tensor(rng.normal(0,.04,(experts,columns,depth)).astype(np.float16),device='CPU').realize()
  return original


def bits(tensor): return tensor.numpy().view(np.uint32)


class TestCPUExpertPhaseTiles(unittest.TestCase):
  def test_prefill_rows_six_matches_four_with_symbolic_tails(self):
    opened=set(Device._opened_devices)
    original=make_experts(8,128,64)
    baseline=CPUExpertWeights(original,prefill_rows=4)
    baseline.prepare({})
    tiled=CPUExpertWeights(original,prefill_rows=6)
    tiled.prepare({})
    rowvar=UOp.variable('cpu_expert_prefill_rows',1,17)
    rng=np.random.default_rng(20261110)
    cases=((1,(0,1)),(7,(2,2)),(17,(3,0)),(5,(1,7)),(3,(4,4)),(11,(0,6)),(3,(4,5)),(4,(2,3)))
    for broadcast in (True,False):
      width=1 if broadcast else 2
      expected_run=TinyJit(lambda s,x,n:baseline(s[:,:n],x[:,:n]).pad_to((1,17,2,128)).contiguous().realize())
      actual_run=TinyJit(lambda s,x,n:tiled(s[:,:n],x[:,:n]).pad_to((1,17,2,128)).contiguous().realize())
      for step,(rows,ids) in enumerate(cases):
        selected_data=np.zeros((1,17,2),dtype=np.int32)
        selected_data[0,:rows]=np.asarray(ids,dtype=np.int32).reshape(1,2)
        selected=Tensor(selected_data,device='CPU').cast(dtypes.int).contiguous().realize()
        values=rng.normal(0,.2,(1,17,width,64)).astype(np.float32)
        x=Tensor(values,device='CPU').realize()
        row_count=rowvar.bind(rows)
        expected=expected_run(selected,x,row_count)
        actual=actual_run(selected,x,row_count)
        np.testing.assert_array_equal(bits(actual),bits(expected),err_msg=f'broadcast={broadcast},rows={rows},step={step}')
        self.assertTrue((actual.numpy()[:,rows:]==0).all())
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))

  def test_configure_rejects_invalid_phase_tile_options_before_mutating(self):
    original=make_experts(4,32,64)
    block=SimpleNamespace(ffn_gate_exps=original)
    model=SimpleNamespace(blk=[block])
    for kwargs in ({'prefill_rows':False},{'prefill_rows':6.0},{'prefill_rows':3}):
      with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
        configure(model,enabled=True,**kwargs)
      self.assertIs(block.ffn_gate_exps,original)
    self.assertEqual(configure(model,enabled=True,prefill_rows=6),1)
    self.assertIsInstance(block.ffn_gate_exps,CPUExpertWeights)


if __name__=='__main__': unittest.main()
