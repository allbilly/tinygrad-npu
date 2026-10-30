import unittest
import gc
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.nn import state
from tinygrad.llm.kernels.rockchip_static import prepare


def _reference(x, weight):
  return (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)


class TestRockchipStatic(unittest.TestCase):
  def _check_case(self, columns, depth, row_counts, seed):
    rng = np.random.default_rng(seed)
    weight_np = rng.normal(0,.1,(columns,depth)).astype(np.float16)
    weight = Tensor(weight_np,device='CPU').realize()
    apply = prepare(weight)
    for rows in row_counts:
      x_np = rng.normal(0,.2,(rows,depth)).astype(np.float32)
      x = Tensor(x_np,device='CPU').realize()
      expected = _reference(x_np,weight_np)
      actual = apply(x).numpy()
      generic = (x @ weight.T).numpy()
      np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
      np.testing.assert_allclose(actual,generic,atol=2e-5,rtol=2e-5)

  def test_odd_depth_and_column_tails_with_multithread_output(self):
    for columns,depth,seed in ((5,7,57),(11,3,113),(131,7,713)):
      with self.subTest(columns=columns,depth=depth):
        self._check_case(columns,depth,(1,3,4,7,17),seed)

  def test_common_projection_sizes_and_2048_decode(self):
    self._check_case(256,256,(1,17),256)
    self._check_case(2048,2048,(1,),2048)

  def test_dynamic_tinyjit_rows_rebind_and_zero_padding(self):
    rng = np.random.default_rng(177)
    columns,depth,capacity = 19,13,17
    weight_np = rng.normal(0,.12,(columns,depth)).astype(np.float16)
    weight = Tensor(weight_np,device='CPU').realize()
    apply = prepare(weight)
    data = rng.normal(0,.2,(capacity,depth)).astype(np.float32)
    x = Tensor(data,device='CPU').realize()
    rows = UOp.variable('rows',1,capacity)
    run = TinyJit(lambda a,n:apply(a[:n]).realize())
    for count in (17,1,4,3,7,17):
      actual = run(x,rows.bind(count))
      padded = actual.pad_to((capacity,columns)).numpy()
      expected = np.zeros((capacity,columns),dtype=np.float32)
      expected[:count] = _reference(data[:count],weight_np)
      np.testing.assert_allclose(padded,expected,atol=2e-5,rtol=2e-5)
      np.testing.assert_array_equal(padded[count:].view(np.uint32),np.zeros_like(padded[count:]).view(np.uint32))

  def test_weight_snapshot_requires_reprepare_after_mutation(self):
    rng = np.random.default_rng(219)
    old_np = rng.normal(0,.1,(15,9)).astype(np.float16)
    weight = Tensor(old_np,device='CPU').realize()
    old_apply = prepare(weight)
    x_np = rng.normal(0,.2,(3,9)).astype(np.float32)
    x = Tensor(x_np,device='CPU').realize()
    before = old_apply(x).numpy()
    changed_np = (old_np.astype(np.float32)+.25).astype(np.float16)
    weight.assign(Tensor(changed_np,device='CPU')).realize()
    # Prepared packed weights are an immutable snapshot; source mutation is visible after prepare().
    np.testing.assert_array_equal(old_apply(x).numpy().view(np.uint32),before.view(np.uint32))
    new_apply = prepare(weight)
    np.testing.assert_allclose(new_apply(x).numpy(),_reference(x_np,changed_np),atol=2e-5,rtol=2e-5)

  def test_source_lifetime_state_dict_and_fallback(self):
    class Projection:
      def __init__(self,weight,apply): self.weight,self.apply = weight,apply
    weight = Tensor(np.arange(35,dtype=np.float16).reshape(5,7),device='CPU').realize()
    apply = prepare(weight)
    model = Projection(weight,apply)
    self.assertEqual(list(state.get_state_dict(model)),['weight'])
    params = state.get_parameters(model)
    self.assertEqual(len(params),1)
    self.assertIs(params[0],weight)
    x = Tensor(np.ones((2,7),dtype=np.float32),device='CPU').realize()
    expected = _reference(x.numpy(),weight.numpy())
    del weight
    np.testing.assert_allclose(model.apply(x).numpy(),expected,atol=1e-6,rtol=1e-6)

    half_x = Tensor(np.arange(14,dtype=np.float16).reshape(2,7),device='CPU').realize()
    fallback = model.apply(half_x)
    generic = half_x @ model.weight.T
    self.assertEqual(fallback.dtype,dtypes.half)
    np.testing.assert_array_equal(fallback.numpy().view(np.uint16),generic.numpy().view(np.uint16))
    del model.weight
    gc.collect()
    np.testing.assert_allclose(model.apply(x).numpy(),expected,atol=1e-6,rtol=1e-6)

  def test_rejects_unsupported_weight_dtype_and_rank(self):
    class UnopenedDeviceWeight:
      device,dtype,ndim,shape = 'GPU',dtypes.half,2,(5,7)
    with self.assertRaises(ValueError): prepare(UnopenedDeviceWeight())
    with self.assertRaises(ValueError): prepare(Tensor.zeros(5,7,device='CPU'))
    with self.assertRaises(ValueError): prepare(Tensor.zeros(5,7,device='CPU').half().reshape(35))
    with self.assertRaises(ValueError): prepare(Tensor.zeros(2,3,4,device='CPU').half())


if __name__=='__main__': unittest.main()
