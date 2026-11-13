"""CPU packet qualification for the MiniCPM3 split-K SwiGLU layout."""
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.model import gated_activation
from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_swiglu_projection, payload_layout
from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles, quantize
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate
from test.runtime.test_rockchip_gelu_ffn import fixture


@pytest.mark.parametrize('rows,active',[(1,1),(17,17),(128,1),(128,17),(128,65),(128,128)])
@pytest.mark.parametrize('case',['ordinary','zeros','weight_nan','scale_inf'])
def test_minicpm3_raw_swiglu_packet_and_scale_bits(rows,active,case):
  hidden,depths=6400,(4096,2304)
  raw,scales,weights,tiles,_=fixture(rows,hidden,min(rows,64),seed=6400+active)
  if case=='zeros':
    raw=[Tensor.zeros_like(t).realize() for t in raw]
  if case=='weight_nan':
    values=weights.numpy().copy()
    values[0]=np.nan
    weights=Tensor(values,device='CPU').realize()
  if case=='scale_inf':
    values=scales[0].numpy().copy()
    values[0]=np.inf
    scales[0]=Tensor(values,device='CPU').realize()
  # Poison inactive rows in each input tile. Neither path may leak them into
  # the explicitly zero-padded activation consumed by down projections.
  for index,scale in enumerate(scales):
    first=index*min(rows,64)
    count=max(0,min(scale.shape[0],active-first))
    values=scale.numpy().copy()
    values[count:]=np.nan
    scales[index]=Tensor(values,device='CPU').realize()
  blocks=[]
  for first in sorted({t.first_row for t in tiles}):
    fragments=[dequantize_tiles([raw[i]],[scales[t.scale_index]],weights[t.first_column:t.first_column+t.columns],
                                 [t.rows],t.columns) for i,t in enumerate(tiles) if t.first_row==first]
    blocks.append(Tensor.cat(*fragments,dim=1))
  projected=Tensor.cat(*blocks,dim=0).realize()[:active].reshape(1,active,hidden*2)
  gate,up=projected[:,:,:hidden],projected[:,:,hidden:]
  # Decode uses the ordinary gated activation; wide prefill materializes
  # sigmoid separately. Test the selected contract for each phase.
  activated=(gated_activation(gate,up) if rows==1 else (gate*sigmoid_gate(projected))*up)
  activated=activated.reshape(active,hidden).pad(((0,rows-active),(0,0)))
  packets,headers=[],[]
  column=0
  for depth,(_,padded,tile,_) in zip(depths,payload_layout(rows,depths)):
    for first in range(0,rows,tile):
      packet,scale=quantize(activated[first:first+tile,column:column+depth].contiguous(),padded)
      packets.append(packet)
      headers.append(scale)
    column+=depth
  expected=(Tensor.cat(*packets),Tensor.cat(*headers))
  actual=pack_swiglu_projection(raw,scales,weights,tiles,rows,hidden,depths,active)
  for a,b in zip(actual,expected):
    assert a.numpy().tobytes()==b.numpy().tobytes()
  assert len(raw)==(1 if rows==1 else 2 if rows==17 else 4)
