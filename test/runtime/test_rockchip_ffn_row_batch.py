"""CPU contracts for direct SwiGLU activation/quantization row batching."""
import numpy as np
import pytest

from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels import rockchip_gelu_ffn as fused_ffn
from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_gelu_projection, pack_swiglu_projection, payload_layout
from tinygrad.llm.kernels.rockchip_int8 import quantize
from test.runtime.test_rockchip_gelu_ffn import fixture


def _reference(projected,active,hidden,depths):
  joint=projected[:active].reshape(1,active,hidden*2).contiguous().realize()
  gate,up=joint[:,:,:hidden],joint[:,:,hidden:]
  sigmoid=gate.sigmoid().contiguous().realize()
  activated=((gate*sigmoid)*up).reshape(active,hidden).pad(((0,projected.shape[0]-active),(0,0))).contiguous().realize()
  payloads,scales=[],[]
  column=0
  for depth,(_,padded,tile,_) in zip(depths,payload_layout(projected.shape[0],depths)):
    for first in range(0,projected.shape[0],tile):
      payload,scale=quantize(activated[first:first+tile,column:column+depth].contiguous(),padded)
      payloads.append(payload)
      scales.append(scale)
    column+=depth
  result=Tensor.cat(*payloads),Tensor.cat(*scales)
  result[0].realize(result[1])
  return result[0].numpy().copy(),result[1].numpy().view(np.uint32).copy()


def _poison_inactive_scales(scales,first_row,active):
  for values in scales:
    count=values.shape[0]
    if first_row>=active: local=0
    elif first_row+count>active: local=active-first_row
    else: continue
    if local<count:
      host=values.numpy().copy()
      host[local:]=np.nan
      values.replace(Tensor(host,device='CPU').realize())


@pytest.mark.parametrize('rows,hidden,input_tile,active', [
  (17,4608,17,17), (128,4864,128,128), (126,4864,64,65), (65,8960,64,17), (33,4928,33,1),
])
def test_direct_swiglu_row_batch_1_2_4_matches_sigmoid_quantize_reference(monkeypatch,rows,hidden,input_tile,active):
  raw,sa,sw,tiles,projected=fixture(rows,hidden,input_tile,seed=20261000+hidden+rows)
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  # Inactive per-row scales must not be read by any direct row batch.
  for first in range(0,rows,input_tile):
    _poison_inactive_scales(sa[first//input_tile:first//input_tile+1],first,active)
  expected=_reference(projected,active,hidden,depths)
  expected_packet_bytes=sum(layout[3] for layout in payload_layout(rows,depths))
  selected={'rows':1}
  monkeypatch.setattr(fused_ffn,'getenv',lambda key,default=0:selected['rows'] if key=='ONPU_SWIGLU_PREFILL_ROWS' else default)
  for batch in (1,2,4):
    selected['rows']=batch
    actual=pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)
    assert actual[0].shape==(expected_packet_bytes,) and actual[1].shape==(len(depths)*rows,)
    actual[0].realize(actual[1])
    np.testing.assert_array_equal(actual[0].numpy(),expected[0])
    actual_scale_bits=actual[1].numpy().view(np.uint32)
    np.testing.assert_array_equal(actual_scale_bits,expected[1])
    scale_values=actual_scale_bits.view(np.float32).reshape(len(depths),rows)
    for group in range(len(depths)):
      np.testing.assert_array_equal(scale_values[group,active:],np.full(rows-active,np.float32(1e-8),dtype=np.float32))


def test_direct_swiglu_row_batch_dynamic_rows_and_rebound_inputs(monkeypatch):
  rows,hidden,input_tile=128,4864,64
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  raw,sa,sw,tiles,projected=fixture(rows,hidden,input_tile,seed=901)
  active=UOp.variable('direct_swiglu_active',1,rows)
  monkeypatch.setattr(fused_ffn,'getenv',lambda key,default=0:4 if key=='ONPU_SWIGLU_PREFILL_ROWS' else default)
  def run(*args):
    nraw,nsa=len(raw),len(sa)
    result=pack_swiglu_projection(list(args[:nraw]),list(args[nraw:nraw+nsa]),args[nraw+nsa],tiles,rows,hidden,
                                 depths,args[-1])
    result[0].realize(result[1])
    return result
  jit=TinyJit(run)
  for step,count in enumerate((17,128,1,65,17)):
    if step:
      next_raw,next_sa,next_sw,next_tiles,projected=fixture(rows,hidden,input_tile,seed=901+step)
      assert next_tiles==tiles
    else: next_raw,next_sa,next_sw=raw,sa,sw
    expected=_reference(projected,count,hidden,depths)
    expected_packet_bytes=sum(layout[3] for layout in payload_layout(rows,depths))
    for first in range(0,rows,input_tile):
      _poison_inactive_scales(next_sa[first//input_tile:first//input_tile+1],first,count)
    actual=jit(*next_raw,*next_sa,next_sw,active.bind(count))
    assert actual[0].shape==(expected_packet_bytes,) and actual[1].shape==(len(depths)*rows,)
    np.testing.assert_array_equal(actual[0].numpy(),expected[0])
    np.testing.assert_array_equal(actual[1].numpy().view(np.uint32),expected[1])
  assert jit.captured is not None


def test_unsupported_batch_only_rejected_for_eligible_swiglu_prefill(monkeypatch):
  selected={'rows':3}
  monkeypatch.setattr(fused_ffn,'getenv',lambda key,default=0:selected['rows'] if key=='ONPU_SWIGLU_PREFILL_ROWS' else default)
  raw,sa,sw,tiles,_=fixture(17,512,17,seed=711)
  with pytest.raises(ValueError):pack_swiglu_projection(raw,sa,sw,tiles,17,512,(512,),17)

  # Decode-sized SwiGLU ignores the prefill batching setting and retains batch one.
  raw,sa,sw,tiles,_=fixture(1,512,1,seed=712)
  selected['rows']=1
  control=pack_swiglu_projection(raw,sa,sw,tiles,1,512,(512,),1)
  selected['rows']=3
  candidate=pack_swiglu_projection(raw,sa,sw,tiles,1,512,(512,),1)
  control[0].realize(control[1])
  candidate[0].realize(candidate[1])
  np.testing.assert_array_equal(candidate[0].numpy(),control[0].numpy())
  np.testing.assert_array_equal(candidate[1].numpy().view(np.uint32),control[1].numpy().view(np.uint32))

  # GELU does not consult the SwiGLU-only environment selector.
  raw,sa,sw,tiles,_=fixture(17,512,17,seed=713)
  selected['rows']=1
  control=pack_gelu_projection(raw,sa,sw,tiles,17,512,(512,),17)
  selected['rows']=3
  candidate=pack_gelu_projection(raw,sa,sw,tiles,17,512,(512,),17)
  control[0].realize(control[1])
  candidate[0].realize(candidate[1])
  np.testing.assert_array_equal(candidate[0].numpy(),control[0].numpy())
  np.testing.assert_array_equal(candidate[1].numpy().view(np.uint32),control[1].numpy().view(np.uint32))
