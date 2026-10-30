"""Exclusive three-core numerical/replay checks for narrow decode buffering."""
import os,unittest
import numpy as np
from tinygrad import Tensor,TinyJit,dtypes
from tinygrad.helpers import getenv


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
class TestNarrowCbufHardware(unittest.TestCase):
  def test_changed_inputs_three_core_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    self.assertEqual(getenv('ONPU_INT8_DECODE_NARROW_CBUF',0),1)
    rng=np.random.default_rng(20261028)
    for columns in (1024,2048,2560):
      depth=4096
      values=rng.integers(-128,128,(columns,depth),dtype=np.int8)
      weight=Tensor(values.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy(),device='ROCKCHIP').realize()
      run=TinyJit(lambda payload:raw_int8_matmul(payload,weight,1,columns).realize())
      for step in range(7):
        data=rng.integers(-128,128,(1,depth),dtype=np.int8)
        if step==5:data.fill(127)
        if step==6:data[:,::2],data[:,1::2]=-128,127
        payload=np.zeros(64+depth,np.uint8)
        payload[:4]=np.asarray([1.0],np.float32).view(np.uint8)
        payload[64:]=data.ravel().view(np.uint8)
        actual=run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize()).numpy()
        expected=(data.astype(np.int32)@values.astype(np.int32).T).ravel()
        np.testing.assert_array_equal(actual,expected)
        if step==2:run.captured=RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreaterEqual(run.captured.native_runs,2)


if __name__=='__main__':unittest.main()
