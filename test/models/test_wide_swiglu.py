"""Opt-in two-weight-tile SwiGLU geometry and exact per-K packet contract."""
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear, quantize
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8SwiGLUFeedForward, pack_swiglu_projection, payload_layout
from test.runtime.test_rockchip_gelu_ffn import fixture


def layer(depth, columns, split=False):
  obj=object.__new__(RockchipInt8SplitLinear if split else RockchipInt8Linear)
  obj.depth,obj.columns,obj.tile_columns=depth,columns,24576
  obj.bias=obj.input_scale=None
  obj.prefill=obj.decode='on'
  obj.phase='prefill'
  widths=[24576,2816] if columns==27392 else [columns]
  obj.packed=[Tensor.empty(n//32,(depth+31)//32,32,32,device='CPU',dtype=dtypes.int8) for n in widths]
  obj.scales=[Tensor.ones(n,device='CPU') for n in widths]
  return obj


def test_wide_geometry_is_opt_in_and_preserves_decode_fallback():
  joint,down=layer(4096,27392),layer(13696,4096,True)
  down.parts=[layer(d,4096) for d in (4096,4096,4096,1408)]
  group=SimpleNamespace(joint=joint,layers=[joint])
  x=Tensor.empty(1,128,4096,device='CPU')
  with Context(ONPU_WIDE_RAW_SWIGLU=0):
    kernel=RockchipInt8SwiGLUFeedForward(group,down)
    assert kernel.combined_scales is None and not kernel.enabled(x)
  with Context(ONPU_WIDE_RAW_SWIGLU=1):
    kernel=RockchipInt8SwiGLUFeedForward(group,down,decode=True)
    assert kernel.enabled(x) and kernel.combined_scales.shape==(27392,)
    joint.phase='decode'
    assert not kernel.enabled(Tensor.empty(1,1,4096,device='CPU'))
    joint.phase='prefill'
    joint.tile_columns=12288
    assert not kernel.enabled(x)


def test_wide_packets_match_independent_multistage_quantization():
  hidden,rows,depths=13696,128,(4096,4096,4096,1408)
  raw,sa,sw,tiles,projected=fixture(rows,hidden,64,seed=629)
  for active in (1,65,128):
    gate,up=projected[:active,:hidden],projected[:active,hidden:]
    sigmoid=gate.sigmoid().contiguous().realize()
    values=((gate*sigmoid)*up).pad(((0,rows-active),(0,0))).contiguous().realize()
    packets,scales=[],[]
    column=0
    for depth,(_,padded,tile,_) in zip(depths,payload_layout(rows,depths)):
      for first in range(0,rows,tile):
        p,s=quantize(values[first:first+tile,column:column+depth].contiguous(),padded)
        packets.append(p)
        scales.append(s)
      column+=depth
    actual,scale=pack_swiglu_projection(raw,sa,sw,tiles,rows,hidden,depths,active)
    np.testing.assert_array_equal(actual.numpy(),Tensor.cat(*packets).numpy())
    np.testing.assert_array_equal(scale.numpy().view(np.uint32),Tensor.cat(*scales).numpy().view(np.uint32))
