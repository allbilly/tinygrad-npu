import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu


def _reference(gate, up):
  g,u = gate.astype(np.float64),up.astype(np.float64)
  exponent = -2.302208185195923*(g+0.044714998453855515*g*g*g)
  with np.errstate(over='ignore',invalid='ignore',under='ignore'):
    return (g/(1.0+np.exp2(exponent))*u).astype(np.float32)


class TestRockchipGelu(unittest.TestCase):
  def test_finite_values_match_generic_bitwise_and_float64_reference(self):
    rng = np.random.default_rng(20261026)
    for rows,depth,seed_scale in ((1,128,1.0),(17,512,2.0),(128,4096,1.0)):
      gate = rng.normal(0,seed_scale,(rows,depth)).astype(np.float32)
      up = rng.normal(0,1,(rows,depth)).astype(np.float32)
      # Exercise signed zeros, the tanh transition, and mixed finite magnitudes.
      gate.flat[:8] = [-0.0,0.0,-3.0,-1.0,0.0,1.0,3.0,12.0]
      up.flat[:8] = [-0.0,1.0,2.0,-2.0,-1.0,0.0,1.0,-1.0]
      actual = gated_gelu(Tensor(gate,device='CPU').realize(),Tensor(up,device='CPU').realize()).numpy()
      generic = (Tensor(gate,device='CPU').gelu(approximate='tanh')*Tensor(up,device='CPU')).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),generic.view(np.uint32))
      np.testing.assert_allclose(actual,_reference(gate,up),atol=2e-6,rtol=2e-6)

  def test_nan_infinity_and_large_finite_propagation(self):
    gate = np.array([[-np.inf,-1000.0,-100.0,-20.0,-1.0,0.0,1.0,20.0,100.0,1000.0,np.inf,np.nan]],np.float32)
    up = np.array([[1.0,1.0,-0.0,2.0,-3.0,1.0,2.0,-1.0,0.0,1.0,1.0,1.0]],np.float32)
    gate = np.tile(gate,(17,43))[:,:512].copy()
    up = np.tile(up,(17,43))[:,:512].copy()
    actual = gated_gelu(Tensor(gate,device='CPU').realize(),Tensor(up,device='CPU').realize()).numpy()
    generic = (Tensor(gate,device='CPU').gelu(approximate='tanh')*Tensor(up,device='CPU')).numpy()
    np.testing.assert_allclose(actual,generic,atol=1e-6,rtol=1e-6,equal_nan=True)
    expected = _reference(gate,up)
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6,equal_nan=True)

  def test_dynamic_rows_rebind_and_zero_tail(self):
    rng = np.random.default_rng(128)
    rows,depth = 128,512
    gate = rng.normal(0,1,(rows,depth)).astype(np.float32)
    up = rng.normal(0,1,(rows,depth)).astype(np.float32)
    g,u = Tensor(gate,device='CPU').realize(),Tensor(up,device='CPU').realize()
    count = UOp.variable('rows',1,rows)
    run = TinyJit(lambda a,b,n:gated_gelu(a[:n],b[:n]).realize())
    for n in (1,17,128,3,17):
      actual = run(g,u,count.bind(n)).pad_to((rows,depth)).numpy()
      expected = np.zeros((rows,depth),np.float32)
      expected[:n] = _reference(gate[:n],up[:n])
      np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
      np.testing.assert_array_equal(actual[n:].view(np.uint32),np.zeros_like(actual[n:]).view(np.uint32))

  def test_joint_projection_noncontiguous_slices(self):
    rng = np.random.default_rng(41)
    rows,depth = 17,512
    joint = Tensor(rng.normal(0,1,(1,rows,2*depth)).astype(np.float32),device='CPU').realize()
    gate,up = joint[:,:,:depth],joint[:,:,depth:]
    self.assertIsNone(gate.uop.contiguous_view())
    self.assertIsNone(up.uop.contiguous_view())
    actual = gated_gelu(gate,up).numpy()
    expected = (gate.gelu(approximate='tanh')*up).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_fallback_for_half_odd_width_and_broadcast_shapes(self):
    gate = Tensor(np.arange(15,dtype=np.float16).reshape(3,5),device='CPU').realize()
    up = Tensor(np.linspace(-1,1,5,dtype=np.float16),device='CPU').realize()
    actual = gated_gelu(gate,up)
    expected = gate.gelu(approximate='tanh')*up
    self.assertEqual(actual.dtype,dtypes.half)
    np.testing.assert_array_equal(actual.numpy().view(np.uint16),expected.numpy().view(np.uint16))

    odd_gate = Tensor(np.arange(15,dtype=np.float32).reshape(3,5),device='CPU').realize()
    odd_up = Tensor(np.linspace(-1,1,15,dtype=np.float32).reshape(3,5),device='CPU').realize()
    odd = gated_gelu(odd_gate,odd_up)
    odd_ref = odd_gate.gelu(approximate='tanh')*odd_up
    np.testing.assert_array_equal(odd.numpy().view(np.uint32),odd_ref.numpy().view(np.uint32))

    scalar = Tensor(-1.25,device='CPU').realize()
    vector = Tensor(np.arange(5,dtype=np.float32),device='CPU').realize()
    got = gated_gelu(scalar,vector)
    ref = scalar.gelu(approximate='tanh')*vector
    np.testing.assert_array_equal(got.numpy().view(np.uint32),ref.numpy().view(np.uint32))

    broadcast_gate = Tensor(np.array([[0.5]],np.float32),device='CPU').realize()
    broadcast_up = Tensor(np.arange(6,dtype=np.float32).reshape(2,3),device='CPU').realize()
    got = gated_gelu(broadcast_gate,broadcast_up)
    ref = broadcast_gate.gelu(approximate='tanh')*broadcast_up
    np.testing.assert_array_equal(got.numpy().view(np.uint32),ref.numpy().view(np.uint32))


if __name__=='__main__': unittest.main()
