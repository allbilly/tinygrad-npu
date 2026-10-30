"""CPU INT8 quantization must preserve sliced activation inputs under TinyJit."""
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels.rockchip_int8 import quantize


def _quantized_slices(x):
  activation=(x*2.0+1.0).contiguous().realize()
  payload0,scales0=quantize(activation[:,:1024],1024)
  payload1,scales1=quantize(activation[:,1024:],1024)
  payload=Tensor.cat(payload0,payload1).realize()
  scales=Tensor.cat(scales0,scales1).realize()
  return payload,scales


def test_tinyjit_quantized_slices_do_not_overwrite_activation_views():
  run=TinyJit(_quantized_slices)
  before={name for name in Device._opened_devices if name.split(':')[0] in
          ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP')}
  for seed in (701,1701,2701):
    for call_index in range(4):
      rng=np.random.default_rng(seed+call_index)
      values=rng.normal(0,.35,(1,2048)).astype(np.float32)
      values[0,:8]=np.array([0.,-0.,.5,-.5,1.,-1.,2.5,-2.5],dtype=np.float32)
      expected_payload,expected_scales=_quantized_slices(Tensor(values.copy(),device='CPU').realize())
      expected_payload=expected_payload.numpy().copy()
      expected_scale_bits=expected_scales.numpy().view(np.uint32).copy()

      x=Tensor(values.copy(),device='CPU').realize()
      actual_payload,actual_scales=run(x)
      actual_payload.realize(actual_scales)
      actual_payload=actual_payload.numpy().copy()
      actual_scale_bits=actual_scales.numpy().view(np.uint32).copy()
      np.testing.assert_array_equal(actual_payload,expected_payload)
      np.testing.assert_array_equal(actual_scale_bits,expected_scale_bits)
      assert np.count_nonzero(actual_payload[64:64+1024])>0
      assert np.count_nonzero(actual_payload[64+1024+64:])>0
  assert run.captured is not None
  after={name for name in Device._opened_devices if name.split(':')[0] in
         ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP')}
  assert after==before
