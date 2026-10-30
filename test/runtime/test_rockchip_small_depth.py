"""Optional small physical depths remove zero channels without changing quantization."""
import os,unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor,TinyJit,dtypes
from tinygrad.helpers import getenv,round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize,packed_int8_matmul,raw_int8_matmul
from tinygrad.runtime.support.rockchip_int8 import padded_depth,supported_depth,int8_body


class TestSmallDepth(unittest.TestCase):
  def test_planner_is_optional_and_direct_geometry_is_bounded(self):
    try:
      for enabled in ('0','1'):
        with patch.dict(os.environ,ONPU_INT8_SMALL_DEPTH=enabled):
          getenv.cache_clear()
          for logical,physical in ((1,64),(32,64),(64,64),(65,128),(96,128),(128,128),(129,256),(192,256),(256,256)):
            self.assertEqual(padded_depth(logical),physical if enabled=='1' else 512)
          self.assertEqual(padded_depth(257),512)
          self.assertEqual(padded_depth(576),576)
          for depth in (64,128,256):
            self.assertTrue(supported_depth(depth))
            for rows in (1,17,128):self.assertTrue(int8_body(rows,704,depth,0,0,0,704))
      for depth in (0,32,63,65,96,192,257,480,4097):
        self.assertFalse(supported_depth(depth))
        with self.assertRaises(ValueError):int8_body(1,704,depth,0,0,0,704)
      with self.assertRaises(ValueError):int8_body(129,704,64,0,0,0,704)
    finally:getenv.cache_clear()

  def test_payload_headers_scales_and_valid_codes_match_512_padding(self):
    rng=np.random.default_rng(640128)
    for rows,depth,padded in ((1,32,64),(17,64,64),(128,96,128),(17,128,128),(128,192,256),(128,256,256)):
      values=rng.normal(0,.2,(rows,depth)).astype(np.float32)
      values[0]=0
      if rows>1:values[1,:8]=[0.,127.,-127.,.5,-.5,1.5,-1.5,126.5]
      x=Tensor(values,device='CPU').realize()
      old,sa=quantize(x,512)
      new,sb=quantize(x,padded)
      a,b=old.numpy(),new.numpy()
      header=round_up(rows*4,64)
      np.testing.assert_array_equal(a[:header],b[:header])
      np.testing.assert_array_equal(sa.numpy().view(np.uint32),sb.numpy().view(np.uint32))
      np.testing.assert_array_equal(a[header:].reshape(rows,512)[:,:padded],b[header:].reshape(rows,padded))
      self.assertFalse(a[header:].reshape(rows,512)[:,padded:].any())

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_integer_and_fp32_projection_with_changed_inputs_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(640129)
    # N=2048 splits into 704/704/640; N=1536 splits into equal 512-column parts.
    for depth,rows,columns in ((64,1,2048),(64,17,1536),(64,128,2048),
                               (128,1,1536),(128,17,2048),(128,128,2048),
                               (256,1,2048),(256,17,1536),(256,128,2048)):
      codes=rng.integers(-128,128,(columns,depth),dtype=np.int8)
      packed=codes.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
      weight=Tensor(packed,device='ROCKCHIP').realize()
      sw=rng.uniform(.001,.02,columns).astype(np.float32)
      scales=Tensor(sw,device='CPU').realize()
      raw_run=TinyJit(lambda p:raw_int8_matmul(p,weight,rows,columns).realize())
      projection=TinyJit(lambda x:packed_int8_matmul(x,weight,scales,columns).realize())
      width=round_up((columns+2)//3,32)
      parts=[(first,min(width,columns-first)) for first in range(0,columns,width)]
      for step in range(7):
        source=rng.integers(-128,128,(rows,depth),dtype=np.int8)
        if step==6:source[:,::2],source[:,1::2]=-128,127
        header=round_up(rows*4,64)
        payload=np.zeros(header+rows*depth,np.uint8)
        payload[:rows*4]=np.ones(rows,np.float32).view(np.uint8)
        payload[header:]=source.ravel().view(np.uint8)
        actual=raw_run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize()).numpy()
        expected=source.astype(np.int32)@codes.astype(np.int32).T
        np.testing.assert_array_equal(actual,np.concatenate([expected[:,first:first+n].ravel() for first,n in parts]))
        x=source.astype(np.float32)*np.float32(.03125)
        sa=np.maximum(np.max(np.abs(x),axis=1)/np.float32(127),np.float32(1e-8))
        qa=np.rint(x*(np.float32(1)/sa[:,None])).clip(-127,127).astype(np.int32)
        target=((qa@codes.astype(np.int32).T).astype(np.float32)*sa[:,None])*sw[None,:]
        result=projection(Tensor(x,device='CPU').realize()).numpy()
        np.testing.assert_array_equal(result.view(np.uint32),target.view(np.uint32))
        if step==2:
          raw_run.captured=RockchipCapturedJit(raw_run.captured)
          projection.captured=RockchipCapturedJit(projection.captured)
      for run in (raw_run,projection):
        self.assertIsNone(run.captured.reason)
        self.assertGreaterEqual(run.captured.native_runs,2)
