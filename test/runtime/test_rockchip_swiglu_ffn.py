"""Raw projection SwiGLU packing preserves independent per-K-group quantization."""
import os,unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn
from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_swiglu_projection,payload_layout,RockchipInt8SwiGLUFeedForward
from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles,RockchipInt8Linear,RockchipInt8SplitLinear,quantize
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate
from test.runtime.test_rockchip_gelu_ffn import fixture


def reference(projected,active,hidden,depths):
  joint=projected[:active].reshape(1,active,hidden*2).realize()
  sigmoid=sigmoid_gate(joint).realize()
  activated=((joint[:,:,:hidden]*sigmoid)*joint[:,:,hidden:]).reshape(active,hidden).pad(((0,projected.shape[0]-active),(0,0)))
  payloads=[]
  scales=[]
  column=0
  for depth,(_,padded,tile,_) in zip(depths,payload_layout(projected.shape[0],depths)):
    for first in range(0,projected.shape[0],tile):
      payload,scale=quantize(activated[first:first+tile,column:column+depth].contiguous(),padded)
      payloads.append(payload)
      scales.append(scale)
    column+=depth
  return Tensor.cat(*payloads),Tensor.cat(*scales)


class TestSwiGLUFFNPack(unittest.TestCase):
  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_biased_down_matches_split_reference_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(6144)
    sources=[]
    for inputs,outputs in ((512,6144),(512,6144),(6144,512)):
      layer=nn.Linear(inputs,outputs,bias=False)
      layer.weight=Tensor(rng.normal(0,.02,(outputs,inputs)).astype(np.float16),device='CPU').realize()
      sources.append(layer)
    sources[2].bias=Tensor(rng.normal(0,.2,512).astype(np.float16),device='CPU').realize()
    group=RockchipProjections(tuple(RockchipInt8Linear(layer,'on','on',24384) for layer in sources[:2]),24384,share_packed=True)
    down=RockchipInt8SplitLinear(sources[2],'on','on',24384)
    down.prepare()
    fused=RockchipInt8SwiGLUFeedForward(group,down,decode=True)
    def baseline(x):
      gate,up=group(x)
      projected=gate.cat(up,dim=-1).contiguous().realize()
      return down((gate*sigmoid_gate(projected))*up).realize()
    for rows in (1,17,128):
      group.layers[0].phase='decode' if rows==1 else 'prefill'
      control,candidate=TinyJit(baseline),TinyJit(lambda x:fused(x).realize())
      for step in range(7):
        # In-place bias changes exercise replay's CPU read dependencies too.
        down.bias.assign(Tensor(rng.normal(0,.2,512).astype(np.float16),device='CPU')).realize()
        x=Tensor(rng.normal(0,.2,(1,rows,512)).astype(np.float32),device='CPU').realize()
        self.assertTrue(fused.enabled(x))
        a,b=control(x).numpy().copy(),candidate(x).numpy().copy()
        self.assertEqual(a.tobytes(),b.tobytes())
        if step==2:
          control.captured=RockchipCapturedJit(control.captured)
          candidate.captured=RockchipCapturedJit(candidate.captured)
      self.assertIsNone(candidate.captured.reason)
      self.assertGreaterEqual(candidate.captured.native_runs,2)
    from tinygrad.helpers import Context
    from tinygrad.llm.kernels.rockchip_padding import capacity_input
    group.layers[0].phase='prefill'
    active=UOp.variable('biased_ffn_rows',1,128)
    control=TinyJit(lambda x:capacity_input(baseline(x)).realize())
    candidate=TinyJit(lambda x:capacity_input(fused(x)).realize())
    with Context(ONPU_ZERO_PAD_VIEWS=1):
      for step,rows in enumerate((17,128,1,65,128,3,64)):
        storage=Tensor(rng.normal(0,.2,(1,128,512)).astype(np.float32),device='CPU').realize()
        x=storage[:,:active.bind(rows)]
        a,b=control(x).numpy().copy(),candidate(x).numpy().copy()
        self.assertEqual(a.tobytes(),b.tobytes())
        self.assertTrue(np.all(b[:,rows:]==0),'Bias must not leak into padded rows')
        if step==2:
          control.captured=RockchipCapturedJit(control.captured)
          candidate.captured=RockchipCapturedJit(candidate.captured)
    self.assertIsNone(candidate.captured.reason)
    self.assertGreaterEqual(candidate.captured.native_runs,2)

  def test_lfm_sdk_layout_keeps_ordinary_packets_scales_and_active_tails(self):
    from tinygrad.helpers import Context,ceildiv,round_up
    for invalid in (False,True):
      raw,sa,sw,tiles,_=fixture(128,4608,128,seed=9216)
      if invalid:
        weights=sw.numpy().copy()
        weights[0]=np.inf
        sw=Tensor(weights,device='CPU').realize()
      blocked=[]
      for value,tile in zip(raw,tiles):
        codes=value.numpy()
        width=round_up(ceildiv(tile.columns,3),32)
        blocked.append(Tensor(np.concatenate([codes[start*tile.rows:(start+min(width,tile.columns-start))*tile.rows]
          .reshape(tile.rows,min(width,tile.columns-start)//4,4).transpose(1,0,2).ravel()
          for start in range(0,tile.columns,width)]),device='CPU').realize())
      for active in (1,17,65,128):
        with Context(ONPU_EXP2_FAST_PATH=1,ONPU_EXP2_SCALE_BITS=1), \
             patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=4):
          a=pack_swiglu_projection(raw,sa,sw,tiles,128,4608,(4096,512),active)
          b=pack_swiglu_projection(blocked,sa,sw,[t._replace(sdk=True) for t in tiles],128,4608,(4096,512),active)
        for expected,actual in zip(a,b):self.assertEqual(expected.numpy().tobytes(),actual.numpy().tobytes())

  def test_channel_scratch_preserves_groups_layouts_tails_and_nonfinite_bits(self):
    from tinygrad.helpers import Context,ceildiv,round_up
    for rows,active,hidden,tile,batch,sdk,invalid in ((128,128,4864,128,4,True,False),
        (128,17,4864,64,4,True,False),(128,1,4864,128,4,True,True),
        (128,65,4864,128,4,True,False),(65,65,5632,64,2,False,False),(17,1,3072,128,4,False,True)):
      raw,sa,sw,tiles,_=fixture(rows,hidden,tile,seed=733)
      if sdk:
        blocked=[]
        for value,t in zip(raw,tiles):
          codes=value.numpy()
          width=round_up(ceildiv(t.columns,3),32)
          parts=[(start,min(width,t.columns-start)) for start in range(0,t.columns,width)]
          if len(parts)!=3:parts=[(0,t.columns)]
          blocked.append(Tensor(np.concatenate([codes[start*t.rows:(start+count)*t.rows]
            .reshape(t.rows,count//4,4).transpose(1,0,2).ravel() for start,count in parts]),device='CPU').realize())
        raw,tiles=blocked,[t._replace(sdk=True) for t in tiles]
      if invalid:
        weights=sw.numpy()
        weights[0]=np.inf
        sw=Tensor(weights,device='CPU').realize()
      depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
      values=[]
      for enabled,prefetch in ((0,0),(1,0),(1,16)):
        with Context(ONPU_SWIGLU_CHANNEL_SCRATCH=enabled,ONPU_SWIGLU_C4_PREFETCH=prefetch), \
             patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=batch):
          values.append([t.numpy().copy() for t in pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)])
      for variant in values[1:]:
        for actual,expected in zip(variant,values[0]):np.testing.assert_array_equal(actual.view(np.uint8),expected.view(np.uint8))

  def test_sdk_minicpm4_legacy_pack_preserves_full_rows_and_active_tails(self):
    from tinygrad.helpers import ceildiv,round_up
    from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_legacy_swiglu_projection
    raw,sa,sw,tiles,_=fixture(128,4096,128,seed=1792)
    blocked=[]
    for value,tile in zip(raw,tiles):
      codes=value.numpy()
      width=round_up(ceildiv(tile.columns,3),32)
      parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
      blocked.append(Tensor(np.concatenate([codes[start*tile.rows:(start+count)*tile.rows]
        .reshape(tile.rows,count//4,4).transpose(1,0,2).ravel() for start,count in parts]),device='CPU').realize())
    for active in (128,17):
      expected=pack_legacy_swiglu_projection(raw,sa,sw,tiles,128,4096,(4096,),active)
      actual=pack_legacy_swiglu_projection(blocked,sa,sw,[t._replace(sdk=True) for t in tiles],128,4096,(4096,),active)
      for a,b in zip(actual,expected):np.testing.assert_array_equal(a.numpy().view(np.uint8),b.numpy().view(np.uint8))

  def test_sdk_qwen3_pack_preserves_legacy_division_payload_and_scale_bits(self):
    from tinygrad.llm.kernels.rockchip_int8 import ffn_pack
    from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_legacy_swiglu_projection
    raw,sa,sw,tiles,_=fixture(128,3072,128,seed=91)
    self.assertEqual(len(raw),1)
    expected=ffn_pack(raw[0],sa[0],sw,128,3072,64)
    codes=raw[0].numpy()
    blocked=Tensor(np.concatenate([codes[first*128:(first+2048)*128].reshape(128,512,4).transpose(1,0,2).ravel()
                                   for first in (0,2048,4096)]),device='CPU').realize()
    for batch in (1,2,4):
      with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=batch):
        actual=pack_legacy_swiglu_projection([blocked],sa,sw,[t._replace(sdk=True) for t in tiles],128,3072,(3072,),128)
        for a,b in zip(expected,actual):np.testing.assert_array_equal(a.numpy().view(np.uint8),b.numpy().view(np.uint8))

  def test_sdk_blocked_projection_matches_payloads_scales_and_active_tails(self):
    from tinygrad.helpers import ceildiv,round_up
    for rows,active,input_tile in ((17,17,128),(128,128,128),(128,17,64),(65,1,64)):
      raw,sa,sw,tiles,_=fixture(rows,4864,input_tile,seed=83)
      blocked=[]
      for value,tile in zip(raw,tiles):
        codes=value.numpy()
        width=round_up(ceildiv(tile.columns,3),32)
        parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
        if len(parts)!=3:parts=[(0,tile.columns)]
        pieces=[codes[start*tile.rows:(start+count)*tile.rows].reshape(tile.rows,count//4,4).transpose(1,0,2).ravel()
                for start,count in parts]
        blocked.append(Tensor(np.concatenate(pieces),device='CPU').realize())
      for batch in (1,2,4):
        with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=batch):
          ref=pack_swiglu_projection(raw,sa,sw,tiles,rows,4864,(4096,768),active)
          got=pack_swiglu_projection(blocked,sa,sw,[t._replace(sdk=True) for t in tiles],rows,4864,(4096,768),active)
          for a,b in zip(ref,got):np.testing.assert_array_equal(a.numpy().view(np.uint8),b.numpy().view(np.uint8))

  def test_payload_and_scale_bits_match_dense_reference(self):
    for rows,hidden,tile,active in ((17,4864,128,17),(128,5632,64,128),(128,5632,64,113),
                                    (65,8960,64,17),(65,8960,64,65)):
      raw,sa,sw,tiles,projected=fixture(rows,hidden,tile)
      depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
      expected,expected_scale=reference(projected,active,hidden,depths)
      actual,scale=pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())
      np.testing.assert_array_equal(scale.numpy().view(np.uint32),expected_scale.numpy().view(np.uint32))

  def test_dynamic_active_rows_rebind_with_realized_outputs(self):
    rows,hidden=128,4864
    raw,sa,sw,tiles,projected=fixture(rows,hidden,64,seed=72)
    count=UOp.variable('swiglu_pack_rows',1,rows)
    run=TinyJit(lambda *args:pack_swiglu_projection(list(args[:2]),list(args[2:4]),args[4],tiles,rows,hidden,
                                                    tuple(min(4096,hidden-start) for start in range(0,hidden,4096)),args[5]))
    for active in (17,128,1,65,17):
      actual,scale=run(*raw,*sa,sw,count.bind(active))
      expected,expected_scale=reference(projected,active,hidden,tuple(min(4096,hidden-start) for start in range(0,hidden,4096)))
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())
      np.testing.assert_array_equal(scale.numpy().view(np.uint32),expected_scale.numpy().view(np.uint32))

  def test_nonfinite_scale_or_weight_matches_invalid_headers_and_zero_codes(self):
    raw,sa,sw,tiles,_=fixture(17,4864,128,seed=73)
    bad_values=sw.numpy()
    bad_values[0]=np.inf
    bad_sw=Tensor(bad_values,device='CPU').realize()
    bad_scale_values=sa[0].numpy()
    bad_scale_values[0]=np.inf
    bad_sa=[Tensor(bad_scale_values,device='CPU').realize()]
    for scales,weights in ((sa,bad_sw),(bad_sa,sw)):
      projected=dequantize_tiles(raw,scales,weights,[17],9728)
      actual,headers=pack_swiglu_projection(raw,scales,weights,tiles,17,4864,(4096,768),17)
      expected,expected_headers=reference(projected,17,4864,(4096,768))
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())
      np.testing.assert_array_equal(headers.numpy().view(np.uint32),expected_headers.numpy().view(np.uint32))

  def test_invalid_coverage_and_active_rows_are_rejected(self):
    raw,sa,sw,tiles,_=fixture(17,4864,128)
    for wrong in ([],[tiles[0]._replace(first_column=32)],tiles+tiles):
      with self.assertRaises(ValueError):pack_swiglu_projection(raw,sa,sw,wrong,17,4864,(4096,768),17)
    for active in (0,18):
      with self.assertRaises(ValueError):pack_swiglu_projection(raw,sa,sw,tiles,17,4864,(4096,768),active)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_sdk_qwen2_ffn_matches_row_major_across_native_replay(self):
    from tinygrad.llm.kernels import rockchip_gelu_ffn as ffn
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(20261031)
    sources=[]
    for inputs,outputs in ((896,4864),(896,4864),(4864,896)):
      layer=nn.Linear(inputs,outputs,bias=False)
      layer.weight=Tensor(rng.normal(0,.02,(outputs,inputs)).astype(np.float16),device='CPU').realize()
      sources.append(layer)
    group=RockchipProjections(tuple(RockchipInt8Linear(layer,'on','on',24576) for layer in sources[:2]),24576,share_packed=True)
    down=RockchipInt8SplitLinear(sources[2],'on','on',24576)
    down.prepare()
    pool=attention_pool()
    state=SharedHostPool(pool)
    fused=RockchipInt8SwiGLUFeedForward(group,down,pool=pool)
    old_getenv=ffn.getenv
    try:
      for rows in (17,128):
        runs=[TinyJit(lambda x:fused(x).realize()) for _ in range(2)]
        for step in range(7):
          x=Tensor(rng.normal(0,.2,(1,rows,896)).astype(np.float32),device='CPU').realize()
          values=[]
          for variant,run in enumerate(runs):
            def flags(key,default=0):
              if key=='ONPU_INT8_SDK_PREFILL':return variant
              if key=='ONPU_FFN_PACKED_UPLOAD':return variant
              if key=='ONPU_SWIGLU_PREFILL_ROWS':return 4 if variant else 2
              return old_getenv(key,default)
            with patch.object(ffn,'getenv',flags),state.scope():values.append(run(x).numpy().copy())
            if step==2:run.captured=RockchipCapturedJit(run.captured)
          np.testing.assert_array_equal(values[0].view(np.uint32),values[1].view(np.uint32))
        for run in runs:
          self.assertIsNone(run.captured.reason)
          self.assertGreaterEqual(run.captured.native_runs,2)
        if os.environ.get('ONPU_REPLAY_BORROW_OUTPUTS')=='1':
          native=runs[1].captured.native
          sizes={key[2] for (index,key) in native.aliases if native.labels[index]=='CPU:rockchip_int8_swiglu_pack'}
          self.assertIn(rows*9728*4,sizes)
          self.assertIn(sum(item[3] for item in payload_layout(rows,(4096,768))),sizes)
    finally:
      state.restore()
      pool.close()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_fused_ffn_matches_existing_path_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(741)
    sources=[]
    for inputs,outputs in ((512,4864),(512,4864),(4864,512)):
      layer=nn.Linear(inputs,outputs,bias=False)
      layer.weight=Tensor(rng.normal(0,.02,(outputs,inputs)).astype(np.float16),device='CPU').realize()
      sources.append(layer)
    group=RockchipProjections(tuple(RockchipInt8Linear(layer,'on','on',24576) for layer in sources[:2]),24576,share_packed=True)
    down=RockchipInt8SplitLinear(sources[2],'on','on',24576)
    down.prepare()
    fused=RockchipInt8SwiGLUFeedForward(group,down)
    def baseline(x):
      gate,up=group(x)
      projected=gate.cat(up,dim=-1).contiguous().realize()
      return down((gate*sigmoid_gate(projected))*up).realize()
    for rows in (17,128):
      control=TinyJit(baseline)
      candidate=TinyJit(lambda x:fused(x).realize())
      for step in range(7):
        x=Tensor(rng.normal(0,.2,(1,rows,512)).astype(np.float32),device='CPU').realize()
        self.assertTrue(fused.enabled(x))
        a,b=control(x).numpy(),candidate(x).numpy()
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
        if step==2:
          control.captured=RockchipCapturedJit(control.captured)
          candidate.captured=RockchipCapturedJit(candidate.captured)
      self.assertIsNone(candidate.captured.reason)
      self.assertGreaterEqual(candidate.captured.native_runs,2)

  def _sdk_blocked(self, raw, tiles):
    from tinygrad.helpers import ceildiv,round_up
    blocked=[]
    for value,tile in zip(raw,tiles):
      source=value.numpy()
      width=round_up(ceildiv(tile.columns,3),32)
      parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
      if len(parts)!=3:parts=[(0,tile.columns)]
      matrix=np.empty((tile.rows,tile.columns),dtype=np.int32)
      for start,count in parts:
        matrix[:,start:start+count]=source[start*tile.rows:(start+count)*tile.rows].reshape(tile.rows,count)
      blocked.append(Tensor(np.concatenate([matrix[:,start:start+count].reshape(tile.rows,count//4,4)
        .transpose(1,0,2).ravel() for start,count in parts]),device='CPU').realize())
    return blocked,[tile._replace(sdk=True) for tile in tiles]

  def _qwen2_sdk_pack_inputs(self, *, edge_cases=False, bad_weight=False):
    raw,sa,sw,tiles,_=fixture(128,4864,128,seed=20261101)
    from tinygrad.helpers import ceildiv,round_up
    tile=tiles[0]
    width=round_up(ceildiv(tile.columns,3),32)
    parts=[(start,min(width,tile.columns-start)) for start in range(0,tile.columns,width)]
    if len(parts)!=3:parts=[(0,tile.columns)]
    packed=raw[0].numpy()
    codes=np.empty((tile.rows,tile.columns),dtype=np.int32)
    for start,count in parts:
      codes[:,start:start+count]=packed[start*tile.rows:(start+count)*tile.rows].reshape(tile.rows,count)
    scales=sa[0].numpy().copy()
    weights=sw.numpy().copy()
    if edge_cases:
      codes[:]=0
      scales[:]=1
      weights[:]=1
      # Gate values straddle |exp2 input|=125 and include both signed tails.
      threshold=np.float32(125.0/1.4426950216293335)
      below=np.nextafter(threshold,np.float32(0.0))
      above=np.nextafter(threshold,np.float32(np.inf))
      gates=(below,above,-below,-above,np.float32(125),np.float32(-125),
             np.float32(1e-40),np.float32(1e-12))
      for row,gate in enumerate(gates):
        codes[row,0]=1 if gate>=0 else -1
        codes[row,4864]=1
        scales[row]=abs(gate)
      # Maximal signed INT32 codes are paired in the same gate/up channel.
      codes[8,0]=np.iinfo(np.int32).max
      codes[8,4864]=np.iinfo(np.int32).max
      codes[9,0]=np.iinfo(np.int32).min
      codes[9,4864]=np.iinfo(np.int32).max
      scales[8:10]=np.float32(1e-8)
      weights[4864]=np.float32(1e-9)
      # One invalid activation row must get NaN scale headers and zero codes.
      codes[10,0]=1
      codes[10,4864]=1
      scales[10]=np.inf
      sa=[Tensor(scales,device='CPU').realize()]
      sw=Tensor(weights,device='CPU').realize()
    if bad_weight:
      weights=sw.numpy().copy()
      weights[0]=np.inf
      sw=Tensor(weights,device='CPU').realize()
    blocked=np.concatenate([codes[:,start:start+count].reshape(tile.rows,count//4,4)
      .transpose(1,0,2).ravel() for start,count in parts])
    raw=[Tensor(blocked,device='CPU').realize()]
    tiles=[tile._replace(sdk=True)]
    return raw,[Tensor(scales,device='CPU').realize()],sw,tiles

  def _batched_pack_variant(self, enabled, raw, sa, sw, tiles, rows, hidden, depths, active, *, scale_bits=1, legacy=False):
    from tinygrad.helpers import Context
    from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_legacy_swiglu_projection
    pack=pack_legacy_swiglu_projection if legacy else pack_swiglu_projection
    with Context(ONPU_SWIGLU_BATCHED_PACK=int(enabled),ONPU_SWIGLU_CHANNEL_SCRATCH=1,ONPU_SWIGLU_C4_PREFETCH=0,
                 ONPU_EXP2_FAST_PATH=1,ONPU_EXP2_SCALE_BITS=scale_bits), \
         patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=4):
      payload,scales=pack(raw,sa,sw,tiles,rows,hidden,depths,active)
      return payload.numpy().copy(),scales.numpy().view(np.uint32).copy()

  def test_batched_qwen2_pack_matches_scalar_for_dynamic_active_and_exp2_edges(self):
    from tinygrad.llm.kernels.rockchip_gelu_ffn import batched_pack_stats
    raw,sa,sw,tiles=self._qwen2_sdk_pack_inputs(edge_cases=True)
    count=UOp.variable('batched_swiglu_pack_rows',1,128)
    before=batched_pack_stats()['traces']
    outputs={}
    for enabled in (False,True):
      runs=[]
      for active in (1,17,65,128):
        runs.append(self._batched_pack_variant(enabled,raw,sa,sw,tiles,128,4864,(4096,768),count.bind(active)))
      outputs[enabled]=runs
    self.assertGreater(batched_pack_stats()['traces'],before)
    for scalar,batched in zip(outputs[False],outputs[True]):
      np.testing.assert_array_equal(scalar[0],batched[0])
      np.testing.assert_array_equal(scalar[1],batched[1])

    payload,scale_bits=outputs[True][-1]
    scales=scale_bits.view(np.float32)
    self.assertTrue(np.isnan(scales[10]) and np.isnan(scales[128+10]))
    self.assertFalse(payload[256+10*4096:256+11*4096].any())
    self.assertFalse(payload[524800+512+10*768:524800+512+11*768].any())

  def test_batched_qwen2_pack_falls_back_when_shape_activation_or_scale_mode_differs(self):
    from tinygrad.llm.kernels.rockchip_gelu_ffn import batched_pack_stats
    cases=[]
    # Decode-sized M=1 and a non-Qwen2 K-group split must stay scalar.
    for rows,hidden in ((1,4864),(128,4608)):
      raw,sa,sw,tiles,_=fixture(rows,hidden,128,seed=880+rows+hidden)
      blocked,tiles=self._sdk_blocked(raw,tiles)
      cases.append((blocked,sa,sw,tiles,rows,hidden,
                    tuple(min(4096,hidden-first) for first in range(0,hidden,4096)),rows,False,1))
    # The target shape falls back for legacy division or when SCALE_BITS is off.
    raw,sa,sw,tiles=self._qwen2_sdk_pack_inputs()
    cases.append((raw,sa,sw,tiles,128,4864,(4096,768),128,True,1))
    cases.append((raw,sa,sw,tiles,128,4864,(4096,768),128,False,0))
    for raw,sa,sw,tiles,rows,hidden,depths,active,legacy,scale_bits in cases:
      before=batched_pack_stats()['traces']
      scalar=self._batched_pack_variant(False,raw,sa,sw,tiles,rows,hidden,depths,active,scale_bits=scale_bits,legacy=legacy)
      flagged=self._batched_pack_variant(True,raw,sa,sw,tiles,rows,hidden,depths,active,scale_bits=scale_bits,legacy=legacy)
      np.testing.assert_array_equal(scalar[0],flagged[0])
      np.testing.assert_array_equal(scalar[1],flagged[1])
      self.assertEqual(batched_pack_stats()['traces'],before)

  def test_batched_qwen2_pack_invalid_weight_rows_match_scalar_fallback(self):
    raw,sa,sw,tiles=self._qwen2_sdk_pack_inputs(bad_weight=True)
    scalar=self._batched_pack_variant(False,raw,sa,sw,tiles,128,4864,(4096,768),128)
    batched=self._batched_pack_variant(True,raw,sa,sw,tiles,128,4864,(4096,768),128)
    np.testing.assert_array_equal(scalar[0],batched[0])
    np.testing.assert_array_equal(scalar[1],batched[1])


class TestLegacySwiGLUPack(unittest.TestCase):
  def test_narrow_payloads_keep_direct_division_rounding_and_scale_bits(self):
    from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_legacy_swiglu_projection
    from tinygrad.llm.kernels.rockchip_int8 import _ffn_pack
    for rows in (1,17,128):
      hidden=3584
      raw,sa,sw,tiles,_=fixture(rows,hidden,128,seed=3584+rows)
      self.assertEqual(len(raw),1)
      layout=payload_layout(rows,(hidden,))
      expected=Tensor.empty(sum(x[3] for x in layout),device='CPU',dtype='uint8').custom_kernel(
        raw[0],sa[0],sw,Tensor.empty(rows,device='CPU',dtype='float32'),
        fxn=lambda o,r,a,w,s:_ffn_pack(o,r,a,w,s,rows,hidden,layout[0][2],static_layout=True))
      for batch in (1,2,4):
        with patch.dict(os.environ,ONPU_SWIGLU_PREFILL_ROWS=str(batch)):
          actual,scales=pack_legacy_swiglu_projection(raw,sa,sw,tiles,rows,hidden,(hidden,),rows)
          np.testing.assert_array_equal(actual.numpy(),expected[0].numpy())
          np.testing.assert_array_equal(scales.numpy().view(np.uint32),expected[-1].numpy().view(np.uint32))
