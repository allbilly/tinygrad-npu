import os, struct, unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.runtime.ops_rockchip import RockchipBatchedMatmul
from tinygrad.llm.kernels.rockchip import batched_matmul, pack_weight
from tinygrad.llm.kernels.rockchip_cpu import unpack_tiles
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit

class TestBatchedMatmul(unittest.TestCase):
  def test_reject_invalid_descriptors_without_device(self):
    for rows,k,tile in [(0,768,64),(1025,768,64),(1024,0,64),(1024,4096,64),(1024,768,65)]:
      descriptor=SimpleNamespace(lib=struct.pack('<10I',rows,768,k,3,0,0,0,0,tile,0x42544d4d))
      with self.assertRaises(ValueError): RockchipBatchedMatmul(None,descriptor)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_chains_rebind_native_and_reject_nonfinite(self):
    rng=np.random.default_rng(45)
    for rows,k,n in [(17,768,64),(1024,768,2304),(1024,768,3072),(1024,3072,768)]:
      values=rng.normal(0,.1,(n,k)).astype(np.float16)
      weight=pack_weight(Tensor(values,device='CPU')).realize()
      run=TinyJit(lambda x:unpack_tiles(batched_matmul(x,weight,n)).realize())
      for index in range(6):
        data=rng.normal(0,.1,(rows,k)).astype(np.float16)
        expected=data.astype(np.float32) @ values.astype(np.float32).T
        before=Device['ROCKCHIP'].submit_count
        actual=run(Tensor(data,device='CPU').realize()).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
        self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
        if index==1:run.captured=RockchipCapturedJit(run.captured)
      self.assertGreater(run.captured.native_runs,0)
      data[-1,-1]=float('nan')
      before=Device['ROCKCHIP'].submit_count
      with self.assertRaises(ValueError):run(Tensor(data,device='CPU').realize())
      self.assertEqual(Device['ROCKCHIP'].submit_count,before)
      self.assertFalse(Device['ROCKCHIP'].poisoned)

if __name__=='__main__':unittest.main()
