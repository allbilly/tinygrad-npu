"""Diagnostic feature exports must respect bound rows and realized state."""
import unittest
import numpy as np
from tinygrad import Tensor, UOp
from extra.rockchip.profile_speculative_prefill import feature_array


class TestSpeculativePrefillProfile(unittest.TestCase):
  def test_bound_feature_readback_excludes_padding_without_replaying_state(self):
    values=np.arange(24,dtype=np.float32).reshape(1,6,4)
    values[:,3:]=np.nan
    inputs=Tensor(values,device='CPU').realize()
    weights=Tensor([[[1.,2.,3.,4.]]],device='CPU').realize()
    state=Tensor([3.],device='CPU').realize()
    rows=UOp.variable('profile_readback_rows',1,6).bind(3)
    features=(inputs[:,:rows]*weights+state.assign(state+1).reshape(1,1,1)).realize()
    self.assertEqual(features.max_shape,(1,6,4))
    self.assertNotIsInstance(features.shape[1],int)
    np.testing.assert_array_equal(state.numpy(),[4.])
    weight_uop=weights.uop
    expected=values[:,:3]*np.array([[[1.,2.,3.,4.]]],np.float32)+4
    for _ in range(3):
      actual=feature_array(features,3)
      self.assertEqual(actual.shape,(1,3,4))
      self.assertTrue(np.isfinite(actual).all())
      np.testing.assert_array_equal(actual,expected)
      np.testing.assert_array_equal(state.numpy(),[4.])
      np.testing.assert_array_equal(weights.numpy(),[[[1.,2.,3.,4.]]])
      self.assertIs(weights.uop,weight_uop)
    for invalid in (0,7,True):
      with self.assertRaises(ValueError):feature_array(features,invalid)
    np.testing.assert_array_equal(state.numpy(),[4.])


if __name__=='__main__':unittest.main()
