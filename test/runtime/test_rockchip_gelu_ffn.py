"""Raw projection GELU packing preserves the existing per-K-group payloads."""
import os,unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn
from tinygrad.helpers import ceildiv,round_up
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
from tinygrad.llm.kernels.rockchip_gelu_ffn import ProjectionTile,pack_gelu_projection,payload_layout,RockchipInt8GeluFeedForward
from tinygrad.llm.kernels.rockchip_int8 import quantize,dequantize_tiles,RockchipInt8Linear,RockchipInt8SplitLinear
from tinygrad.llm.kernels.rockchip import RockchipProjections


def fixture(rows,hidden,input_tile,seed=71):
  rng=np.random.default_rng(seed)
  raw=[]
  scales=[]
  tiles=[]
  row_values=[]
  sw=Tensor(rng.uniform(.001,.03,hidden*2).astype(np.float32),device='CPU').realize()
  for first in range(0,rows,input_tile):
    count=min(input_tile,rows-first)
    sa=Tensor(rng.uniform(.001,.01,count).astype(np.float32),device='CPU').realize()
    scale_index=len(scales)
    scales.append(sa)
    columns=[]
    limit=24576 if count==1 else 12288
    for column in range(0,hidden*2,limit):
      width=min(limit,hidden*2-column)
      values=rng.integers(-100000,100000,(count,width),dtype=np.int32)
      core=round_up(ceildiv(width,3),32)
      parts=[(start,min(core,width-start)) for start in range(0,width,core)]
      if len(parts)!=3:parts=[(0,width)]
      packed=np.concatenate([values[:,start:start+size].ravel() for start,size in parts])
      raw.append(Tensor(packed,device='CPU').realize())
      tiles.append(ProjectionTile(first,count,column,width,scale_index))
      columns.append(dequantize_tiles([raw[-1]],[sa],sw[column:column+width],[count],width))
    row_values.append(Tensor.cat(*columns,dim=1))
  return raw,scales,sw,tiles,Tensor.cat(*row_values,dim=0).realize()


def reference(projected,active,hidden,depths):
  activated=gated_gelu(projected[:active,:hidden],projected[:active,hidden:]).pad(((0,projected.shape[0]-active),(0,0)))
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


class TestGeluFFNPack(unittest.TestCase):
  def test_static_layout_matches_original_fragmented_payloads_bit_for_bit(self):
    for rows,hidden,tile,active in ((128,9216,64,113),(33,4928,128,1)):
      raw,sa,sw,tiles,_=fixture(rows,hidden,tile,seed=75)
      depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
      outputs=[]
      for variant in (0,1):
        with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=variant):
          packed,scales=pack_gelu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)
          outputs.append((packed.numpy(),scales.numpy().view(np.uint32)))
      for control,candidate in zip(*outputs):np.testing.assert_array_equal(control,candidate)

  def test_payload_and_scale_bits_match_existing_multi_stage_path(self):
    for rows,hidden,input_tile,active in ((17,512,128,17),(65,9216,64,17),(128,9216,64,128),
                                          (33,4928,128,1),(128,6400,64,113)):
      raw,sa,sw,tiles,projected=fixture(rows,hidden,input_tile)
      depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
      expected,scale=reference(projected,active,hidden,depths)
      actual,actual_scale=pack_gelu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())
      np.testing.assert_array_equal(actual_scale.numpy().view(np.uint32),scale.numpy().view(np.uint32))

  def test_dynamic_active_rows_rebind_without_reading_poisoned_padding(self):
    rows,hidden=128,512
    raw,sa,sw,tiles,projected=fixture(rows,hidden,64,seed=72)
    count=UOp.variable('gelu_pack_rows',1,rows)
    run=TinyJit(lambda *args:pack_gelu_projection(list(args[:2]),list(args[2:4]),args[4],tiles,rows,hidden,(hidden,),args[5]))
    for active in (17,128,1,65,17):
      actual,scale=run(*raw,*sa,sw,count.bind(active))
      expected,expected_scale=reference(projected,active,hidden,(hidden,))
      np.testing.assert_array_equal(actual.numpy(),expected.numpy())
      np.testing.assert_array_equal(scale.numpy().view(np.uint32),expected_scale.numpy().view(np.uint32))

  def test_nonfinite_projection_rows_use_invalid_headers_and_zero_codes(self):
    raw,sa,sw,tiles,projected=fixture(17,512,128,seed=73)
    values=sw.numpy()
    values[0]=np.inf
    bad_sw=Tensor(values,device='CPU').realize()
    bad_projected=dequantize_tiles(raw,sa,bad_sw,[17],1024)
    actual,scales=pack_gelu_projection(raw,sa,bad_sw,tiles,17,512,(512,),17)
    expected,expected_scales=reference(bad_projected,17,512,(512,))
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    np.testing.assert_array_equal(scales.numpy().view(np.uint32),expected_scales.numpy().view(np.uint32))

  def test_invalid_projection_coverage_and_active_rows_are_rejected(self):
    raw,sa,sw,tiles,_=fixture(17,512,128)
    for wrong in ([],[tiles[0]._replace(first_column=32)],tiles+tiles):
      with self.assertRaises(ValueError):pack_gelu_projection(raw,sa,sw,wrong,17,512,(512,),17)
    for active in (0,18):
      with self.assertRaises(ValueError):pack_gelu_projection(raw,sa,sw,tiles,17,512,(512,),active)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit exclusive NPU test')
  def test_fused_ffn_matches_existing_path_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng=np.random.default_rng(740)
    sources=[]
    for inputs,outputs in ((512,9216),(512,9216),(9216,512)):
      layer=nn.Linear(inputs,outputs,bias=False)
      layer.weight=Tensor(rng.normal(0,.02,(outputs,inputs)).astype(np.float16),device='CPU').realize()
      sources.append(layer)
    members=tuple(RockchipInt8Linear(layer,'on','on',24576) for layer in sources[:2])
    group=RockchipProjections(members,24576,share_packed=True)
    down=RockchipInt8SplitLinear(sources[2],'on','on',24576)
    down.prepare()
    fused=RockchipInt8GeluFeedForward(group,down)
    def baseline(x):
      gate,up=group(x)
      return down(gated_gelu(gate,up)).realize()
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
