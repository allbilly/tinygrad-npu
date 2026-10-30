"""Unpadded depth2304 removes zero channels while retaining integer/scale semantics."""
import os,unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor,TinyJit
from tinygrad.helpers import getenv,round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize,packed_int8_matmul
from tinygrad.runtime.support.rockchip_int8 import padded_depth,supported_depth,int8_body


class TestNatural2304(unittest.TestCase):
  def test_planner_option_and_geometry_limits(self):
    self.assertTrue(supported_depth(2304))
    try:
      with patch.dict(os.environ):
        os.environ.pop('ONPU_INT8_NATURAL2304',None)
        getenv.cache_clear()
        self.assertEqual(padded_depth(2304),2304)
      for enabled,expected in (('0',2560),('1',2304)):
        with patch.dict(os.environ,{'ONPU_INT8_NATURAL2304':enabled}):
          getenv.cache_clear()
          self.assertEqual(padded_depth(2304),expected)
          self.assertEqual(padded_depth(2305),2560)
          self.assertEqual(padded_depth(1152),1152)
    finally:getenv.cache_clear()
    for m in (1,17,64):self.assertTrue(int8_body(m,768,2304,0,0,0,768))
    with self.assertRaises(ValueError):int8_body(128,768,2304,0,0,0,768)

  def test_zero_padding_removal_keeps_quantized_bytes_scales_and_headers(self):
    rng=np.random.default_rng(2304)
    for rows in (1,17,64):
      x=rng.normal(0,.2,(rows,2304)).astype(np.float32)
      x[0]=0
      if rows>1:x[1,:8]=[0.,127.,-127.,.5,-.5,1.5,-1.5,126.5]
      t=Tensor(x,device='CPU').realize()
      old,sa=quantize(t,2560)
      new,sb=quantize(t,2304)
      a,b=old.numpy(),new.numpy()
      header=round_up(rows*4,64)
      np.testing.assert_array_equal(a[:header],b[:header])
      np.testing.assert_array_equal(sa.numpy().view(np.uint32),sb.numpy().view(np.uint32))
      old_codes=a[header:].reshape(rows,2560)
      np.testing.assert_array_equal(old_codes[:,:2304],b[header:].reshape(rows,2304))
      self.assertFalse(old_codes[:,2304:].any())

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_vocabulary_tail_and_partial_rows_match_reference_through_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(230417)
    for rows,columns in ((1,10240),(17,128),(64,9216)):
      codes=rng.integers(-127,128,(columns,2304),dtype=np.int8)
      sw=rng.uniform(.001,.02,columns).astype(np.float32)
      packed=codes.reshape(columns//32,32,72,32).transpose(0,2,1,3).copy()
      weight=Tensor(packed,device='ROCKCHIP').realize()
      scales=Tensor(sw,device='CPU').realize()
      run=TinyJit(lambda x:packed_int8_matmul(x,weight,scales,columns).realize())
      for step in range(7):
        x=rng.normal(0,.2,(rows,2304)).astype(np.float32)
        if step==6:x[:,::2],x[:,1::2]=-128,127
        sa=np.maximum(np.max(np.abs(x),axis=1)/np.float32(127),np.float32(1e-8))
        qa=np.rint(x*(np.float32(1)/sa[:,None])).clip(-127,127).astype(np.int32)
        expected=((qa@codes.astype(np.int32).T).astype(np.float32)*sa[:,None])*sw[None,:]
        actual=run(Tensor(x,device='CPU').realize()).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        if step==2:run.captured=RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreaterEqual(run.captured.native_runs,2)
