import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, nn
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize, RockchipInt8Linear, ffn_pack, RockchipInt8FeedForward
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit


def reference_quantize(values):
  scales = np.maximum(np.max(np.abs(values),axis=1)/np.float32(127),np.float32(1e-8))
  return np.rint(values*(np.float32(1)/scales[:,None])).clip(-127,127).astype(np.int8),scales


class TestRockchipInt8(unittest.TestCase):
  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_wide_projection_matches_independent_block_quantization(self):
    from types import SimpleNamespace
    from tinygrad.llm.kernels.rockchip import configure
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8SplitLinear
    rng = np.random.default_rng(104)
    for rows,depth,columns in ((1,4864,97),(17,8960,64),(65,8192,32),(1,4097,17)):
      weights = rng.normal(0,.05,(columns,depth)).astype(np.float16)
      bias = rng.normal(0,.01,columns).astype(np.float32)
      source = nn.Linear(depth,columns,bias=True)
      source.weight,source.bias = Tensor(weights,device='CPU'),Tensor(bias,device='CPU')
      model = SimpleNamespace(projection=source)
      placement = configure(model,prefill='on',decode='on',tile_columns=24576,precision='int8')
      layer = model.projection
      self.assertIsInstance(layer,RockchipInt8SplitLinear)
      self.assertEqual((placement['layers'],placement['prefill_layers'],placement['decode_layers']),(1,1,1))
      run = TinyJit(lambda x:layer(x).realize())
      for _ in range(3):
        values = rng.normal(0,.15,(rows,depth)).astype(np.float32)
        expected = np.zeros((rows,columns),np.float32)
        for start in range(0,depth,4096):
          qw,sw = reference_quantize(weights[:,start:start+4096].astype(np.float32))
          qx,sx = reference_quantize(values[:,start:start+4096])
          expected += ((qx.astype(np.int32)@qw.astype(np.int32).T).astype(np.float32)*sx[:,None])*sw[None,:]
        expected += bias
        actual = run(Tensor(values,device='CPU').realize()).numpy()
        np.testing.assert_allclose(actual,expected,atol=4e-5,rtol=3e-5)
        dense = values@weights.astype(np.float32).T+bias
        self.assertLess(np.linalg.norm(actual-dense)/np.linalg.norm(dense),.03)
      bad = values.copy()
      bad[0,-1] = np.nan
      with self.assertRaises(ValueError):run(Tensor(bad,device='CPU').realize())
      self.assertTrue(np.isfinite(run(Tensor(values,device='CPU').realize()).numpy()).all())
      layer.release_packed()
      self.assertFalse(layer.parts or layer.packed or layer.scales)
      layer.prepare()
      np.testing.assert_allclose(layer(Tensor(values,device='CPU').realize()).numpy(),expected,atol=4e-5,rtol=3e-5)

  def test_raw_qkv_matches_dequantized_norm_rope_and_cache(self):
    from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv, prepare_qkv_int8
    rng = np.random.default_rng(55)
    heads,kv,dim = 8,4,64
    columns = (heads+kv*2)*dim
    for rows in (17,128,256):
      values = rng.integers(-100000,100000,(rows,columns),dtype=np.int32)
      sa = rng.uniform(.001,.003,rows).astype(np.float32)
      sw = rng.uniform(.001,.003,columns).astype(np.float32)
      raw,scales,counts = [],[],[]
      width = round_up((columns+2)//3,32)
      for start in range(0,rows,128):
        part = values[start:start+128]
        counts.append(len(part))
        raw.append(Tensor(np.concatenate([part[:,c:c+width].ravel() for c in range(0,columns,width)]),device='CPU'))
        scales.append(Tensor(sa[start:start+128],device='CPU'))
      projected = Tensor((values.astype(np.float32)*sa[:,None]*sw[None,:])[None],device='CPU')
      data = rng.normal(0,.2,(2,1,kv,512,dim)).astype(np.float16)
      cache = [Tensor(data.copy(),device='CPU').contiguous().realize() for _ in range(2)]
      qn = Tensor(rng.normal(1,.2,dim).astype(np.float16),device='CPU')
      kn = Tensor(rng.normal(1,.2,dim).astype(np.float32),device='CPU')
      freqs = Tensor(rng.normal(0,.5,(512,dim)).astype(np.float32),device='CPU')
      expected = prepare_qkv(projected,cache[0],qn,kn,freqs,0,heads,1e-6)
      actual = prepare_qkv_int8(raw,scales,Tensor(sw,device='CPU'),counts,cache[1],qn,kn,freqs,0,heads,1e-6,rows)
      np.testing.assert_array_equal(actual[0].numpy(),expected[0].numpy())
      np.testing.assert_array_equal(actual[1].numpy(),expected[1].numpy())

  def test_fused_ffn_payload_matches_regular_activation_quantization(self):
    rng = np.random.default_rng(34)
    for rows,depth in [(1,512),(17,512),(64,3072),(65,512),(128,3072)]:
      data = rng.integers(-100000,100000,(rows,depth*2),dtype=np.int32)
      sa = rng.uniform(.001,.003,rows).astype(np.float32)
      sw = rng.uniform(.001,.003,depth*2).astype(np.float32)
      width = round_up((depth*2+2)//3,32)
      packed = np.concatenate([data[:,start:start+width].ravel() for start in range(0,depth*2,width)])
      projected = Tensor(data.astype(np.float32)*sa[:,None]*sw[None,:],device='CPU')
      gate,up = projected[:,:depth],projected[:,depth:]
      activated = (gate*gate.sigmoid())*up
      expected_parts = [quantize(activated[start:start+64],depth) for start in range(0,rows,64)]
      expected = Tensor.cat(*(part[0] for part in expected_parts))
      expected_scales = Tensor.cat(*(part[1] for part in expected_parts))
      actual,actual_scales = ffn_pack(Tensor(packed,device='CPU'),Tensor(sa,device='CPU'),Tensor(sw,device='CPU'),rows,depth)
      np.testing.assert_array_equal(actual_scales.numpy(),expected_scales.numpy())
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())

  def test_quantization_ties_zero_tail_and_scale_header(self):
    rng = np.random.default_rng(32)
    from tinygrad.runtime.support.rockchip_int8 import padded_depth
    for rows,depth in [(1,9),(17,65),(64,1024),(1,576),(64,768),(32,1152)]:
      values = rng.normal(0,.1,(rows,depth)).astype(np.float32)
      values[0] = 0
      if rows>1: values[1,:9] = [0,127,-127,.5,-.5,1.5,-1.5,126.5,-126.5]
      expected,scales = reference_quantize(values)
      padded = padded_depth(depth)
      raw,actual_scales = quantize(Tensor(values,device='CPU'),padded)
      data = raw.numpy()
      header = round_up(rows*4,64)
      np.testing.assert_array_equal(actual_scales.numpy(),scales)
      codes = data[header:].view(np.int8).reshape(rows,padded)
      np.testing.assert_array_equal(codes[:,:depth],expected)
      self.assertFalse(codes[:,depth:].any())
      self.assertFalse(data[rows*4:header].any())

  def test_nonfinite_activations_mark_invalid_scales(self):
    for value in (float('nan'),float('inf'),float('-inf')):
      for position in (0,3,8):
        data = np.ones((2,9),dtype=np.float32)
        data[1,position] = value
        _,scales = quantize(Tensor(data,device='CPU'),512)
        self.assertTrue(np.isnan(scales.numpy()[1]))
        self.assertTrue(np.isfinite(scales.numpy()[0]))

  def test_nonfinite_weights_rejected_before_npu_upload(self):
    layer = nn.Linear(32,32,bias=False)
    data = np.ones((32,32),dtype=np.float32)
    data[17,5] = float('nan')
    layer.weight = Tensor(data,device='CPU')
    with self.assertRaises(ValueError): RockchipInt8Linear(layer,'on','on',24576).prepare()

  def test_invalid_channel_scales_rejected_before_npu_upload(self):
    source = nn.Linear(64,32,bias=False)
    for invalid in (0.,-1.,float('inf'),float('nan')):
      layer = RockchipInt8Linear(source,'on','on')
      data = np.ones(64,np.float32)
      data[7] = invalid
      layer.input_scale = Tensor(data,device='CPU')
      with self.assertRaises(ValueError): layer.prepare()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_full_outputs_rebinding_native_and_nonfinite_recovery(self):
    rng = np.random.default_rng(33)
    for rows,depth,columns in [(1,65,97),(17,1024,3072),(32,3072,1024),(128,1024,96),(256,1024,96),(129,3072,96),
                               (1,576,97),(64,576,576),(128,768,3072),(32,1152,576)]:
      weights = rng.normal(0,.1,(columns,depth)).astype(np.float32)
      source = nn.Linear(depth,columns,bias=False)
      source.weight = Tensor(weights,device='CPU')
      layer = RockchipInt8Linear(source,'on','on',24576)
      channel_scale = np.ones(depth,np.float32)
      if rows==1 and depth==576:
        channel_scale = np.resize(np.array([.25,.5,1.,2.,4.,8.,16.,32.],np.float32),depth)
        layer.input_scale = Tensor(channel_scale,device='CPU')
      layer.prepare()
      qw,sw = reference_quantize(weights*channel_scale)
      run = TinyJit(lambda x:layer(x).realize())
      for index in range(6):
        data = rng.normal(0,.1,(rows,depth)).astype(np.float32)
        qa,sa = reference_quantize(data/channel_scale)
        expected = (qa.astype(np.int32)@qw.astype(np.int32).T).astype(np.float32)*sa[:,None]*sw[None,:]
        actual = run(Tensor(data,device='CPU').realize()).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-4)
        if index==1: run.captured = RockchipCapturedJit(run.captured)
      self.assertGreater(run.captured.native_runs,0)
      self.assertGreater(len(run.captured.native.alias_copies),0)
      for value in (float('nan'),float('inf')):
        data[0,depth//2] = value
        before = Device['ROCKCHIP'].submit_count
        with self.assertRaises(ValueError): run(Tensor(data,device='CPU').realize())
        self.assertEqual(Device['ROCKCHIP'].submit_count,before)
      data = np.ones((rows,depth),dtype=np.float32)
      qa,sa = reference_quantize(data/channel_scale)
      expected = (qa.astype(np.int32)@qw.astype(np.int32).T).astype(np.float32)*sa[:,None]*sw[None,:]
      np.testing.assert_allclose(run(Tensor(data,device='CPU').realize()).numpy(),expected,atol=2e-5,rtol=3e-4)
    layers = []
    for depth,columns in [(64,512),(64,512),(512,96)]:
      source = nn.Linear(depth,columns,bias=False)
      source.weight = Tensor(rng.normal(0,.1,(columns,depth)).astype(np.float32),device='CPU')
      layer = RockchipInt8Linear(source,'on','on',24576)
      layer.prepare()
      layers.append(layer)
    projections = RockchipProjections(tuple(layers[:2]),24576)
    fused = RockchipInt8FeedForward(projections,layers[2])
    for rows in (1,17,128):
      x = Tensor(rng.normal(0,.1,(1,rows,64)).astype(np.float32),device='CPU')
      gate,up = projections(x)
      expected = layers[2]((gate*gate.sigmoid())*up).numpy()
      self.assertTrue(fused.enabled(x))
      np.testing.assert_allclose(fused(x).numpy(),expected,atol=2e-5,rtol=3e-4)

    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    from tinygrad.llm.kernels.rockchip_cpu import decode_attention
    import threading
    pool = attention_pool()
    for heads,kv_heads in [(16,8),(12,4)]:
      query = Tensor(rng.normal(0,.3,(1,heads,1,128)).astype(np.float32),device='CPU')
      cache = Tensor(rng.normal(0,.3,(2,1,kv_heads,512,128)).astype(np.float16),device='CPU')
      for length in (17,128,256):
        expected = decode_attention(query,cache,length).numpy()
        with pool:
          seen = []
          thread = threading.Thread(target=lambda:seen.append(pool.active))
          thread.start()
          thread.join()
          self.assertEqual(seen,[False])
          actual = decode_attention(query,cache,length,pool=pool).numpy()
          np.testing.assert_array_equal(actual,expected)
          fast = decode_attention(query,cache,length,pool=pool,fast_softmax=True).numpy()
          np.testing.assert_allclose(fast,expected,atol=2e-6,rtol=2e-5)
        self.assertFalse(pool.active)


if __name__=='__main__': unittest.main()
