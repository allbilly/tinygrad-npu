import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.rockchip_vision_mlp import gelu_pack

class TestVisionGeluPack(unittest.TestCase):
  def test_bias_gelu_and_irregular_row_tiles(self):
    rng = np.random.default_rng(42)
    for rows in (17,128,1024):
      data = rng.normal(0,3,(rows,3072)).astype(np.float32)
      bias = rng.normal(0,.2,3072).astype(np.float16)
      parts = [Tensor(part.reshape(len(part),768,4).transpose(1,0,2).copy().ravel(),device='CPU')
               for first in range(0,rows,64) if len(part:=data[first:first+64])]
      actual = gelu_pack(parts,Tensor(bias,device='CPU'),3072,21).numpy()
      unpacked = []
      for first in range(0,rows,21):
        count = min(21,rows-first)
        unpacked.append(actual[first*3072:(first+count)*3072].reshape(384,count,8).transpose(1,0,2).reshape(count,3072))
      expected = (Tensor(data,device='CPU')+Tensor(bias,device='CPU')).gelu().half().numpy()
      np.testing.assert_allclose(np.concatenate(unpacked),expected,atol=2e-3,rtol=2e-3)

if __name__ == '__main__': unittest.main()
