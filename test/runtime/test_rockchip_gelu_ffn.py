"""Raw projection GELU packing preserves the existing per-K-group payloads."""
import os,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn,dtypes
from tinygrad.helpers import Context,ceildiv,round_up
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
from tinygrad.llm.kernels.rockchip_gelu_ffn import (ProjectionTile,pack_gelu_projection,payload_layout,
                                                   grouped_down_payload,RockchipInt8GeluFeedForward,RockchipInt8SwiGLUFeedForward)
from tinygrad.llm.kernels.rockchip_int8 import quantize,dequantize_tiles,RockchipInt8Linear,RockchipInt8SplitLinear
from tinygrad.llm.kernels.rockchip_grouped_int8 import dequantize as grouped_dequantize
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.runtime.support.rockchip_int8 import padded_depth


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


class TestGroupedDecodeDown(unittest.TestCase):
  DEPTHS=(4096,4096,4096,1408)

  def test_payload_reuses_each_part_codes_and_scales(self):
    for rows in (1,17,64,128):
      raw,sa,sw,tiles,_=fixture(rows,13696,rows,seed=76)
      packed,scales=pack_gelu_projection(raw,sa,sw,tiles,rows,13696,self.DEPTHS,rows)
      payload,activation_scales=grouped_down_payload(packed,scales,rows,self.DEPTHS,3)
      layout=payload_layout(rows,self.DEPTHS)
      header=round_up(3*rows*4,64)
      expected_scales=Tensor.cat(scales[:rows],scales[rows:2*rows],scales[2*rows:3*rows]).contiguous().realize()
      np.testing.assert_array_equal(activation_scales.numpy().view(np.uint32),expected_scales.numpy().view(np.uint32))
      self.assertEqual(payload.numel(),header+3*rows*4096)
      actual=payload.numpy()
      expected_codes=[]
      for offset,padded,tile,_ in layout[:3]:
        for first in range(0,rows,tile):
          count=min(tile,rows-first)
          tile_header=round_up(count*4,64)
          begin=offset+first*padded+(first//tile)*tile_header
          expected_codes.append(packed.numpy()[begin+tile_header:begin+tile_header+count*padded])
      np.testing.assert_array_equal(actual[header:],np.concatenate(expected_codes))
      np.testing.assert_array_equal(actual[:3*rows*4],expected_scales.numpy().view(np.uint8))
      self.assertFalse(actual[3*rows*4:header].any())

  def test_grouped_dequantization_matches_per_part_sum_order(self):
    rng=np.random.default_rng(77)
    def part_major(values):
      width=round_up(ceildiv(4096,3),32)
      bounds=[(start,min(width,4096-start)) for start in range(0,4096,width)]
      if len(bounds)!=3:bounds=[(0,4096)]
      return np.concatenate([values[:,start:start+count].reshape(-1) for start,count in bounds])
    for rows in (1,17,64):
      weight_scales=Tensor(rng.uniform(.001,.03,(3,4096)).astype(np.float32),device='CPU').realize()
      activation_scales=Tensor(rng.uniform(.001,.01,3*rows).astype(np.float32),device='CPU').realize()
      values=rng.integers(-2**30,2**30,(3,rows,4096),dtype=np.int32)
      raw=Tensor(values,device='CPU').realize()
      grouped=grouped_dequantize(raw,activation_scales,weight_scales,3,rows,4096,4096)
      parts=[dequantize_tiles([Tensor(part_major(values[index]),device='CPU').realize()],
                              [activation_scales[index*rows:(index+1)*rows]],weight_scales[index],[rows],4096)
             for index in range(3)]
      expected=parts[0]
      for value in parts[1:]:expected=expected+value
      actual=grouped[0]
      for value in grouped[1:]:actual=actual+value
      np.testing.assert_array_equal(actual.numpy().view(np.uint32),expected.numpy().view(np.uint32))

  def test_decode_ffn_uses_one_grouped_down_submission_and_matches_per_part_path(self):
    rng=np.random.default_rng(78)
    def part_raw(payload,weight,rows,columns,column_offset=0,**kwargs):
      k=weight.shape[1]*32
      header=round_up(rows*4,64)
      codes=payload.numpy()[header:header+rows*k].view(np.int8).reshape(rows,k)
      full=weight.numpy().transpose(0,2,1,3).reshape(weight.shape[0]*32,k)
      dots=codes.astype(np.int32)@full.astype(np.int32).T
      dots=dots[:,column_offset:column_offset+columns]
      return Tensor(dots.ravel(),device='CPU',dtype=dtypes.int32).realize()
    def grouped_raw(payload,weight,rows,columns,cores=3):
      groups,k=weight.shape[0],weight.shape[2]*32
      header=round_up(groups*rows*4,64)
      codes=payload.numpy()[header:header+groups*rows*k].view(np.int8).reshape(groups,rows,k)
      full=weight.numpy().transpose(0,1,3,2,4).reshape(groups,weight.shape[1]*32,k)
      dots=np.einsum('grk,gnk->grn',codes.astype(np.int32),full.astype(np.int32),optimize=True)
      return Tensor(dots[:,:,:columns].ravel(),device='CPU',dtype=dtypes.int32).realize()
    joint=RockchipInt8Linear.__new__(RockchipInt8Linear)
    joint.depth,joint.columns,joint.tile_columns=4096,27392,24576
    joint.input_scale=joint.bias=None
    joint.packed=[Tensor(rng.integers(-120,121,(768,128,32,32),dtype=np.int8),device='CPU').realize(),
                  Tensor(rng.integers(-120,121,(88,128,32,32),dtype=np.int8),device='CPU').realize()]
    joint.scales=[Tensor(rng.uniform(.001,.03,24576).astype(np.float32),device='CPU').realize(),
                  Tensor(rng.uniform(.001,.03,2816).astype(np.float32),device='CPU').realize()]
    parts=[]
    for depth in self.DEPTHS:
      k=padded_depth(depth)
      parts.append(SimpleNamespace(depth=depth,columns=4096,input_scale=None,bias=None,
        packed=[Tensor(rng.integers(-120,121,(128,k//32,32,32),dtype=np.int8),device='CPU').realize()],
        scales=[Tensor(rng.uniform(.001,.03,4096).astype(np.float32),device='CPU').realize()]))
    down=RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
    down.depth,down.columns,down.input_scale,down.bias=13696,4096,None,None
    down.parts=parts
    down.grouped_weight=Tensor.stack(*(part.packed[0] for part in parts[:3]),dim=0).contiguous().realize()
    down.grouped_scale=Tensor.stack(*(part.scales[0] for part in parts[:3]),dim=0).contiguous().realize()
    down.grouped_count=3
    fused=RockchipInt8SwiGLUFeedForward.__new__(RockchipInt8SwiGLUFeedForward)
    fused.projections=SimpleNamespace(joint=joint,layers=[SimpleNamespace(phase='decode')])
    fused.down,fused.pool=down,None
    fused.activation='silu'
    fused.combined_scales=Tensor.cat(joint.scales[0],joint.scales[1]).contiguous().realize()
    x=Tensor(rng.normal(0,.2,(1,1,4096)).astype(np.float32),device='CPU').realize()
    calls=[]
    def part_count(*args,**kwargs):
      calls.append(('part',args[2],args[3]))
      return part_raw(*args,**kwargs)
    def grouped_count(*args,**kwargs):
      calls.append(('grouped',args[2],args[3]))
      return grouped_raw(*args,**kwargs)
    with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.raw_int8_matmul',part_count), \
         patch('tinygrad.llm.kernels.rockchip_gelu_ffn.grouped_raw_matmul',grouped_count), \
         Context(ONPU_GROUPED_DECODE_DOWN=0):
      control=fused(x).numpy()
    control_calls=tuple(kind for kind,*_ in calls)
    self.assertEqual(control_calls,('part','part','part','part','part','part'))
    calls.clear()
    with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.raw_int8_matmul',part_count), \
         patch('tinygrad.llm.kernels.rockchip_gelu_ffn.grouped_raw_matmul',grouped_count), \
         Context(ONPU_GROUPED_DECODE_DOWN=1):
      candidate=fused(x).numpy()
    self.assertEqual(tuple(kind for kind,*_ in calls),('part','part','grouped','part'))
    self.assertEqual([width for kind,_,width in calls if kind=='grouped'],[4096])
    self.assertEqual([width for kind,_,width in calls if kind=='part' and width==4096],[4096])
    np.testing.assert_array_equal(control.view(np.uint32),candidate.view(np.uint32))
    bad=x.numpy()
    bad.flat[0],bad.flat[1]=np.nan,np.inf
    bad_x=Tensor(bad,device='CPU').realize()
    with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.raw_int8_matmul',part_count), \
         patch('tinygrad.llm.kernels.rockchip_gelu_ffn.grouped_raw_matmul',grouped_count), \
         Context(ONPU_GROUPED_DECODE_DOWN=0):
      bad_control=fused(bad_x).numpy()
    with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.raw_int8_matmul',part_count), \
         patch('tinygrad.llm.kernels.rockchip_gelu_ffn.grouped_raw_matmul',grouped_count), \
         Context(ONPU_GROUPED_DECODE_DOWN=1):
      bad_candidate=fused(bad_x).numpy()
    np.testing.assert_array_equal(bad_control.view(np.uint32),bad_candidate.view(np.uint32))


class TestSDKTanhGeluPack(unittest.TestCase):
  def test_sdk_tiles_match_original_quantization_with_four_row_tails(self):
    from tinygrad.helpers import Context
    from tinygrad.llm.kernels import rockchip_gelu_ffn as f
    for hidden in (6144,12288):
      rng=np.random.default_rng(hidden)
      scales=[Tensor(rng.uniform(.001,.01,128).astype(np.float32),device='CPU').realize()]
      sw=Tensor(rng.uniform(.001,.03,hidden*2).astype(np.float32),device='CPU').realize()
      raw,tiles,projected=[],[],[]
      for first in range(0,hidden*2,12288):
        value=rng.integers(-100000,100000,(128,12288),dtype=np.int32)
        packed=np.concatenate([value[:,start:start+4096].reshape(128,1024,4).transpose(1,0,2).ravel()
                               for start in (0,4096,8192)])
        raw.append(Tensor(packed,device='CPU').realize())
        tiles.append(f.ProjectionTile(0,128,first,12288,0,True))
        projected.append(dequantize_tiles([raw[-1]],scales,sw[first:first+12288],[128],12288,sdk_tiles=(True,)))
      values=Tensor.cat(*projected,dim=1).realize()
      depths=tuple(min(4096,hidden-first) for first in range(0,hidden,4096))
      with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',return_value=4), \
           Context(ONPU_EXP2_FAST_PATH=1,ONPU_EXP2_SCALE_BITS=1,ONPU_SWIGLU_CHANNEL_SCRATCH=1):
        for active in (1,17,127,128):
          count=UOp.variable('sdk_gelu_rows',1,128).bind(active)
          expected,scale=reference(values,active,hidden,depths)
          for batched,prefetch in ((0,0),(1,16)):
            with Context(ONPU_SWIGLU_BATCHED_PACK=batched,ONPU_SWIGLU_C4_PREFETCH=prefetch):
              actual,actual_scale=f.pack_gelu_projection(raw,scales,sw,tiles,128,hidden,depths,count)
              self.assertEqual(actual.numpy().tobytes(),expected.numpy().tobytes())
              self.assertEqual(actual_scale.numpy().tobytes(),scale.numpy().tobytes())

  def test_row_batch_admission_rejects_ordinary_and_unknown_tile_geometry(self):
    from tinygrad.llm.kernels.rockchip_gelu_ffn import _row_batch_supported
    good=(ProjectionTile(0,128,0,12288,0,True),)
    self.assertTrue(_row_batch_supported('gelu_tanh',good))
    for tiles in ((),(good[0]._replace(sdk=False),),(good[0]._replace(rows=64),),
                  (good[0]._replace(columns=8192),),(good[0]._replace(scale_index=1),)):
      self.assertFalse(_row_batch_supported('gelu_tanh',tiles))

  def test_public_sdk_frontend_keeps_both_projection_tiles_on_sdk_layout(self):
    from types import SimpleNamespace
    from tinygrad import dtypes,Device
    from tinygrad.llm.kernels import rockchip_gelu_ffn as f
    seen=[]
    def raw(payload,weight,rows,columns,column_offset=0,*,sdk_prefill=False,**kwargs):
      seen.append((rows,columns,weight.shape[1]*32,column_offset,sdk_prefill))
      return Tensor.zeros(rows*columns,device='CPU',dtype=dtypes.int32).realize()
    def env(key,default=0):
      return 1 if key=='ONPU_INT8_SDK_PREFILL' else 4 if key=='ONPU_SWIGLU_PREFILL_ROWS' else default
    for hidden in (6144,12288):
      joint=RockchipInt8Linear.__new__(RockchipInt8Linear)
      joint.depth,joint.columns,joint.input_scale,joint.bias=1536,hidden*2,None,None
      joint.packed=[Tensor.empty(hidden*2//32,48,32,32,device='CPU',dtype=dtypes.int8)]
      joint.scales=[Tensor.ones(hidden*2,device='CPU')]
      down=RockchipInt8SplitLinear.__new__(RockchipInt8SplitLinear)
      down.depth,down.columns,down.input_scale,down.bias=hidden,1536,None,None
      down.parts=[]
      for first in range(0,hidden,4096):
        depth=min(4096,hidden-first)
        down.parts.append(SimpleNamespace(packed=[Tensor.empty(48,depth//32,32,32,device='CPU',dtype=dtypes.int8)],
                          scales=[Tensor.ones(1536,device='CPU')],depth=depth,columns=1536))
      group=SimpleNamespace(joint=joint)
      fused=f.RockchipInt8GeluFeedForward(group,down,sdk_prefill=True)
      seen.clear()
      with patch('tinygrad.llm.kernels.rockchip_gelu_ffn.raw_int8_matmul',raw), \
           patch('tinygrad.llm.kernels.rockchip_gelu_ffn.getenv',env):
        output=fused(Tensor.zeros(1,128,1536,device='CPU')).numpy()
      projections=[call for call in seen if call[2]==1536]
      self.assertEqual(projections,[(128,12288,1536,first,True) for first in range(0,hidden*2,12288)])
      self.assertEqual(output.shape,(1,128,1536))
      self.assertFalse(output.any())
    self.assertFalse({'CL','ROCKCHIP'}.intersection(Device._opened_devices))
