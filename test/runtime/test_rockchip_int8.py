import os, unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, Device, nn, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_int8 import quantize, RockchipInt8Linear, ffn_pack, RockchipInt8FeedForward, RockchipInt8SplitLinear
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit


def reference_quantize(values):
  scales = np.maximum(np.max(np.abs(values),axis=1)/np.float32(127),np.float32(1e-8))
  return np.rint(values*(np.float32(1)/scales[:,None])).clip(-127,127).astype(np.int8),scales


class TestRockchipInt8(unittest.TestCase):
  def test_decode_output_tiles_reuse_activation_quantization_exactly(self):
    from tinygrad.helpers import Context
    from tinygrad.llm.kernels import rockchip_int8 as r8
    rng=np.random.default_rng(20261107)
    source=nn.Linear(64,250,bias=False)
    source.weight=Tensor(rng.normal(0,.1,(250,64)).astype(np.float32),device='CPU').realize()
    layer=RockchipInt8Linear(source,'on','on',128)
    layer.input_scale=Tensor(rng.uniform(.5,2.,64).astype(np.float32),device='CPU').realize()
    with patch('tinygrad.runtime.ops_rockchip.int8_weight_device',return_value='CPU'):layer.prepare()
    inputs=[rng.normal(0,.2,(1,1,64)).astype(np.float32),np.zeros((1,1,64),np.float32),
            np.resize(np.array([-127.,-1.,0.,1.,127.],np.float32),(1,1,64))]
    original_quantize=r8.quantize
    calls=[]
    def counted_quantize(*args,**kwargs):
      calls.append(1)
      return original_quantize(*args,**kwargs)
    def raw_cpu(payload,weight,rows,columns,column_offset=0,**kwargs):
      k=weight.shape[1]*32
      width=round_up(columns,32)
      data=payload.numpy()
      qa=data[round_up(rows*4,64):].view(np.int8).reshape(rows,k).astype(np.int32)
      qw=weight.numpy().transpose(0,2,1,3).reshape(-1,k).astype(np.int32)
      raw=qa@qw[column_offset:column_offset+width].T
      return Tensor(raw.reshape(-1),device='CPU',dtype=dtypes.int32)
    def packed_cpu(values,weight,scales,columns,column_offset=0):
      payload,activation_scales=counted_quantize(values,weight.shape[1]*32)
      raw=raw_cpu(payload,weight,values.shape[0],columns,column_offset)
      return r8.dequantize_tiles([raw],[activation_scales],scales[column_offset:column_offset+columns],
                                 [values.shape[0]],columns)
    with patch.object(r8,'quantize',side_effect=counted_quantize), patch.object(r8,'raw_int8_matmul',side_effect=raw_cpu), \
         patch.object(r8,'packed_int8_matmul',side_effect=packed_cpu), Context(ONPU_INT8_REUSE_ACTIVATION=0):
      for data in inputs:
        x=Tensor(data,device='CPU').realize()
        calls.clear()
        control=layer.npu(x).numpy()
        control_quantizations=len(calls)
        calls.clear()
        with Context(ONPU_INT8_REUSE_ACTIVATION=1):candidate=layer.npu(x).numpy()
        candidate_quantizations=len(calls)
        np.testing.assert_array_equal(candidate,control)
        self.assertEqual(control_quantizations,2)
        self.assertEqual(candidate_quantizations,1)

  def test_grouped_decode_flag_prepares_grouped_split_weights(self):
    from tinygrad.helpers import Context
    rng=np.random.default_rng(79)
    source=nn.Linear(13696,32,bias=False)
    source.weight=Tensor(rng.normal(0,.02,(32,13696)).astype(np.float32),device='CPU').realize()
    with patch('tinygrad.runtime.ops_rockchip.int8_weight_device',return_value='CPU'):
      layer=RockchipInt8SplitLinear(source,'on','on',24576)
      with Context(ONPU_GROUPED_DECODE_DOWN=0):layer.prepare()
      self.assertEqual(layer.grouped_count,0)
      self.assertIsNone(layer.grouped_weight)
      layer.release_packed()
      with Context(ONPU_GROUPED_DECODE_DOWN=1):layer.prepare()
      self.assertEqual(layer.grouped_count,3)
      self.assertEqual(tuple(layer.grouped_weight.shape),(3,1,128,32,32))
      self.assertEqual(tuple(layer.grouped_scale.shape),(3,32))

  def test_grouped_split_decode_branch_is_gated_by_the_new_flag(self):
    from tinygrad.helpers import Context
    parts=[]
    for depth in (4096,4096,4096,1408):
      parts.append(SimpleNamespace(depth=depth,columns=4096,npu=lambda x: Tensor.zeros(1,1,4096,device='CPU',dtype=dtypes.float).realize()))
    down=RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
    down.depth,down.columns,down.bias,down.shared_projection=13696,4096,None,None
    down.parts=parts
    down.grouped_weight=Tensor.zeros(3,128,128,32,32,device='CPU',dtype=dtypes.int8).realize()
    down.grouped_scale=Tensor.ones(3,4096,device='CPU',dtype=dtypes.float).realize()
    down.grouped_count=3
    seen=[]
    def grouped_matmul(x,weight,scales,columns,cores=3,*,residual=False,materialize=False):
      seen.append((tuple(x.shape),columns,materialize))
      return Tensor.zeros(3,1,4096,device='CPU',dtype=dtypes.float).realize()
    x=Tensor.ones(1,1,13696,device='CPU',dtype=dtypes.float).realize()
    with patch('tinygrad.llm.kernels.rockchip_grouped_int8.matmul',grouped_matmul),Context(ONPU_GROUPED_DECODE_DOWN=0):
      control=down.npu(x).numpy()
    self.assertEqual(seen,[])
    with patch('tinygrad.llm.kernels.rockchip_grouped_int8.matmul',grouped_matmul),Context(ONPU_GROUPED_DECODE_DOWN=1):
      candidate=down.npu(x).numpy()
    self.assertEqual(seen,[((3,1,4096),4096,True)])
    np.testing.assert_array_equal(control,candidate)

  def test_decode_bank_extension_is_limited_to_qualified_per_core_width(self):
    from tinygrad.runtime.support.rockchip_int8 import int8_body
    def banks(m,n,k):
      return {(word>>16)&0xffffffff for word in int8_body(m,n,k,0,0,0,n) if word>>48==0x201 and word&0xffff==0x1040}
    # RKLLM target captures use 0xb1 per-core at k=2560/768, but oNPU submits the full
    # N with cores=3; copying 0xb1 measured slower on MiniCPM3 (4.46 vs 5.34 tok/s),
    # so the extension stays limited to the qualified k=4096 widths.
    self.assertEqual(banks(1,512,4096),{0xb1})
    self.assertEqual(banks(1,704,4096),{0x48})
    self.assertEqual(banks(1,1024,4096),{0xb1})
    self.assertEqual(banks(64,512,4096),{0x48})
    self.assertEqual(banks(1,512,2048),{0x84})

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_narrow_output_wide_decode_survives_native_replay(self):
    from tinygrad import dtypes
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    rng = np.random.default_rng(20261027)
    columns,depth = 1536,4096
    values = rng.integers(-128,128,(columns,depth),dtype=np.int8)
    packed = values.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
    weight = Tensor(packed,device='ROCKCHIP').realize()
    run = TinyJit(lambda payload:raw_int8_matmul(payload,weight,1,columns).realize())
    for step in range(7):
      source = rng.integers(-128,128,(1,depth),dtype=np.int8)
      if step==6:source[:,::2],source[:,1::2]=-128,127
      payload = np.zeros(64+depth,dtype=np.uint8)
      payload[:4] = np.array([1],dtype=np.float32).view(np.uint8)
      payload[64:] = source.ravel().view(np.uint8)
      actual = run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize()).numpy()
      np.testing.assert_array_equal(actual,(source.astype(np.int32)@values.astype(np.int32).T).ravel())
      if step==2:run.captured=RockchipCapturedJit(run.captured)
    self.assertIsNone(run.captured.reason)
    self.assertGreaterEqual(run.captured.native_runs,2)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_domains_share_ownership_and_native_replay(self):
    from tinygrad import dtypes
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    from tinygrad.runtime.autogen import rockchip as rk
    rng = np.random.default_rng(71025)
    dev0,dev1 = Device['ROCKCHIP'],Device['ROCKCHIP:1']
    self.assertIs(dev0.hw,dev1.hw)
    self.assertIs(dev0.lock,dev1.lock)
    arrays = [rng.integers(-128,128,(96,512),dtype=np.int8) for _ in range(2)]
    weights = [Tensor(w.reshape(3,32,16,32).transpose(0,2,1,3).copy(),device=d).realize()
               for w,d in zip(arrays,('ROCKCHIP','ROCKCHIP:1'))]
    def calculation(payload):
      outputs = tuple(raw_int8_matmul(payload,w,1,96) for w in weights)
      Tensor.realize(*outputs)
      return outputs
    run = TinyJit(calculation)
    try:
      for index in range(7):
        data = rng.integers(-128,128,(1,512),dtype=np.int8)
        payload = np.zeros(64+512,dtype=np.uint8)
        payload[:4] = np.array([1],dtype=np.float32).view(np.uint8)
        payload[64:] = data.ravel().view(np.uint8)
        outputs = run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize())
        for out,w in zip(outputs,arrays):np.testing.assert_array_equal(out.numpy(),(data.astype(np.int32)@w.astype(np.int32).T).ravel())
        if index==1:run.captured = RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreater(run.captured.native_runs,0)
      self.assertEqual(dev0.submit_count,dev1.submit_count)
    finally:
      rk.DRM_IOCTL_RKNPU_ACTION(dev0.fd,flags=rk.RKNPU_SET_IOMMU_DOMAIN_ID,value=0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_wide_decode_integer_outputs_survive_replay(self):
    from tinygrad import dtypes
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    rng = np.random.default_rng(4025)
    depth = 4096
    for columns in (1536,3072,4096,6144,12288,24576):
      values = rng.integers(-127,128,(columns,depth),dtype=np.int8)
      packed = values.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
      weight = Tensor(packed,device='ROCKCHIP').realize()
      run = TinyJit(lambda payload:raw_int8_matmul(payload,weight,1,columns).realize())
      for step in range(4):
        source = rng.integers(-128,128,(1,depth),dtype=np.int8)
        if step==3: source[:,::2],source[:,1::2] = -128,127
        payload = np.zeros(64+depth,dtype=np.uint8)
        payload[:4] = np.array([1],dtype=np.float32).view(np.uint8)
        payload[64:] = source.ravel().view(np.uint8)
        actual = run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize()).numpy()
        expected = source.astype(np.int32)@values.astype(np.int32).T
        np.testing.assert_array_equal(actual,expected.ravel())

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_rkllm_captured_decode_geometries_integer_goldens(self):
    from tinygrad import dtypes
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    rng = np.random.default_rng(20261106)
    for columns,depth in ((96,2560),(256,2560),(1280,768),(1536,4096)):
      values = rng.integers(-127,128,(columns,depth),dtype=np.int8)
      packed = values.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
      weight = Tensor(packed,device='ROCKCHIP').realize()
      run = TinyJit(lambda payload:raw_int8_matmul(payload,weight,1,columns).realize())
      for step in range(3):
        source = rng.integers(-128,128,(1,depth),dtype=np.int8)
        if step==2: source[:,::2],source[:,1::2] = -128,127
        payload = np.zeros(64+depth,dtype=np.uint8)
        payload[:4] = np.array([1],dtype=np.float32).view(np.uint8)
        payload[64:] = source.ravel().view(np.uint8)
        actual = run(Tensor(payload,device='CPU',dtype=dtypes.uint8).realize()).numpy()
        expected = source.astype(np.int32)@values.astype(np.int32).T
        np.testing.assert_array_equal(actual,expected.ravel())

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

  def test_fused_ffn_static_layout_matches_fragment_boundaries_and_partial_tiles(self):
    from tinygrad import dtypes
    from tinygrad.llm.kernels.rockchip_int8 import _ffn_pack
    rng = np.random.default_rng(20261028)
    for depth in range(512,4097,512):
      rows,tile = (17,7) if depth%1024 else (65,64)
      data = rng.integers(-100000,100000,(rows,depth*2),dtype=np.int32)
      data[0]=0
      sa = rng.uniform(.001,.003,rows).astype(np.float32)
      sw = rng.uniform(.001,.003,depth*2).astype(np.float32)
      sa[-3:] = np.nan,np.inf,-np.inf
      width = round_up((depth*2+2)//3,32)
      packed = Tensor(np.concatenate([data[:,start:start+width].ravel() for start in range(0,depth*2,width)]),device='CPU')
      size = sum(round_up(min(tile,rows-first)*4,64)+min(tile,rows-first)*depth for first in range(0,rows,tile))
      outputs=[]
      for static in (False,True):
        result = Tensor.full(size,0xa5,device='CPU',dtype=dtypes.uint8).contiguous().custom_kernel(
          packed,Tensor(sa,device='CPU'),Tensor(sw,device='CPU'),Tensor.empty(rows,device='CPU',dtype=dtypes.float),
          fxn=lambda o,r,a,w,s:_ffn_pack(o,r,a,w,s,rows,depth,tile,static))
        Tensor.realize(result[0],result[4])
        outputs.append((result[0].numpy(),result[4].numpy().view(np.uint32)))
      with self.subTest(depth=depth,rows=rows,tile=tile):
        for legacy,static in zip(*outputs):np.testing.assert_array_equal(legacy,static)
        self.assertTrue(np.isnan(outputs[1][1][-3:].view(np.float32)).all())
        for first in range(0,rows,tile):
          count=min(tile,rows-first)
          offset=first*depth+first//tile*round_up(tile*4,64)
          self.assertFalse(outputs[1][0][offset+count*4:offset+round_up(count*4,64)].any())

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
      self.assertGreater(len(run.captured.native.alias_copies),0,
                         f'geometry={(rows,depth,columns)}, calls={run.captured.native.labels}')
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
