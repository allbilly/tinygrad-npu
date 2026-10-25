import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_vision import vision_attention

class TestVisionAttention(unittest.TestCase):
  def test_bidirectional_fp32(self):
    rng = np.random.default_rng(7)
    for heads,rows,dim in [(1,8,8),(4,24,32),(12,128,64),(12,1024,64)]:
      arrays = [rng.normal(size=(1,heads,rows,dim)).astype(np.float32) for _ in range(3)]
      scores = arrays[0] @ arrays[1].swapaxes(-1,-2) / np.sqrt(dim)
      probs = np.exp(scores-scores.max(axis=-1,keepdims=True))
      probs /= probs.sum(axis=-1,keepdims=True)
      expected = probs @ arrays[2]
      q,k,v = [Tensor(a.transpose(0,1,3,2),device='CPU').transpose(-1,-2) for a in arrays]
      actual = vision_attention(q,k,v).numpy()
      np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)

  def test_reject_invalid_contracts(self):
    for shape in [(2,4,16,64),(1,4,17,64),(1,4,16,10),(1,4,1032,64)]:
      source = Tensor.zeros(*shape,device='CPU')
      with self.assertRaises(ValueError): vision_attention(source,source,source)
    q = Tensor.zeros(1,4,16,64,device='CPU')
    with self.assertRaises(ValueError): vision_attention(q.half(),q.half(),q.half())
    with self.assertRaises(ValueError): vision_attention(q,q,Tensor.zeros(1,4,16,32,device='CPU',dtype=dtypes.float))

if __name__ == '__main__': unittest.main()
