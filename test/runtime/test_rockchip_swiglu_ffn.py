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
