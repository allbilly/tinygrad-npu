import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context, ceildiv, round_up
from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv, prepare_qkv_int8

class TestRockchipBiasedQKV(unittest.TestCase):
  heads,kv_heads,dim,context,start=12,2,128,512,11
  columns=(heads+2*kv_heads)*dim
  capacity=128

  def make_case(self,seed,active,sizes,bias_dtype,normalize=True):
    rng=np.random.default_rng(seed)
    raw_values=rng.integers(-100000,100000,(self.capacity,self.columns),dtype=np.int32)
    activation_scales=rng.uniform(.001,.003,self.capacity).astype(np.float32)
    weight_scales=rng.uniform(.001,.003,self.columns).astype(np.float32)
    padded_columns=round_up(self.columns,32)
    width=round_up(ceildiv(padded_columns,3),32)
    raw_parts=[]
    scale_parts=[]
    offset=0
    for size in sizes:
      part=raw_values[offset:offset+size]
      padded=np.zeros((size,padded_columns),np.int32)
      padded[:,:self.columns]=part
      raw_parts.append(Tensor(np.concatenate([padded[:,column:column+width].ravel()
        for column in range(0,padded_columns,width)]),device='CPU').realize())
      scale_parts.append(Tensor(activation_scales[offset:offset+size],device='CPU').realize())
      offset+=size
    self.assertEqual(offset,self.capacity)
    bias_values=None if bias_dtype is None else rng.normal(0,.02,self.columns).astype(bias_dtype)
    bias=None if bias_values is None else Tensor(bias_values,device='CPU').realize()
    # Preserve the API's FP32 operation order: raw*activation scale, then
    # weight scale, then the optional float32 bias, all in separate roundings.
    dequantized=np.multiply(raw_values.astype(np.float32),activation_scales[:,None],dtype=np.float32)
    dequantized=np.multiply(dequantized,weight_scales[None,:],dtype=np.float32)
    if bias_values is not None:
      dequantized=np.add(dequantized,bias_values.astype(np.float32)[None,:],dtype=np.float32)
    projected=Tensor(dequantized[None],device='CPU').realize()
    freqs=Tensor(rng.normal(0,.4,(self.context,self.dim)).astype(np.float32),device='CPU').realize()
    qnorm=Tensor(rng.uniform(.6,1.4,self.dim).astype(np.float16),device='CPU').realize() if normalize else None
    knorm=Tensor(rng.uniform(.6,1.4,self.dim).astype(np.float32),device='CPU').realize() if normalize else None
    cache_values=rng.normal(0,.2,(2,1,self.kv_heads,self.context,self.dim)).astype(np.float16)
    cache_expected=Tensor(cache_values.copy(),device='CPU').realize()
    cache_actual=Tensor(cache_values.copy(),device='CPU').realize()
    rows=UOp.variable(f'active_{seed}',1,self.capacity).bind(active)
    expected=prepare_qkv(projected[:,:rows],cache_expected,qnorm,knorm,freqs,self.start,self.heads,1e-6)
    actual=prepare_qkv_int8(raw_parts,scale_parts,Tensor(weight_scales,device='CPU').realize(),list(sizes),
      cache_actual,qnorm,knorm,freqs,self.start,self.heads,1e-6,rows,bias=bias)
    return expected,actual

  def test_fp16_fp32_and_absent_bias_match_float_qkv_bits(self):
    for normalize in (False,True):
      for active,sizes in ((17,(128,)),(65,(64,64)),(128,(128,))):
        for index,bias_dtype in enumerate((np.float16,np.float32,None)):
          expected,actual=self.make_case(901+active+index,active,sizes,bias_dtype,normalize)
          actual_q=actual[0].pad_to((1,self.heads,self.capacity,self.dim)).numpy()[:,:,:active]
          expected_q=expected[0].pad_to((1,self.heads,self.capacity,self.dim)).numpy()[:,:,:active]
          np.testing.assert_array_equal(actual_q.view(np.uint32),expected_q.view(np.uint32))
          np.testing.assert_array_equal(actual[1].numpy().view(np.uint16),expected[1].numpy().view(np.uint16))

  def test_balanced_nonuniform_head_ranges_preserve_queries_and_cache(self):
    results=[]
    for enabled in (0,1):
      with Context(ONPU_PREFILL_CPU_TUNING=enabled):
        _,actual=self.make_case(905,65,(64,64),np.float16,False)
        q=actual[0].pad_to((1,self.heads,self.capacity,self.dim)).numpy().copy()
        cache=actual[1].numpy().copy()
        results.append((q,cache))
    np.testing.assert_array_equal(results[0][0].view(np.uint32),results[1][0].view(np.uint32))
    np.testing.assert_array_equal(results[0][1].view(np.uint16),results[1][1].view(np.uint16))

  def test_bad_bias_shape_device_and_dtype_are_rejected(self):
    from types import SimpleNamespace
    active=17
    sizes=(128,)
    padded_columns=round_up(self.columns,32)
    width=round_up(ceildiv(padded_columns,3),32)
    raw_parts=[]
    for size in sizes:
      padded=np.zeros((size,padded_columns),np.int32)
      raw_parts.append(Tensor(np.concatenate([padded[:,column:column+width].ravel()
        for column in range(0,padded_columns,width)]),device='CPU').realize())
    sa=[Tensor.ones(self.capacity,device='CPU').realize()]
    sw=Tensor.ones(self.columns,device='CPU').realize()
    cache=Tensor.zeros(2,1,self.kv_heads,self.context,self.dim,device='CPU',dtype=dtypes.half).realize()
    freqs=Tensor.zeros(self.context,self.dim,device='CPU').realize()
    qnorm=Tensor.ones(self.dim,device='CPU').realize()
    knorm=Tensor.ones(self.dim,device='CPU').realize()
    rows=UOp.variable('bad_bias_rows',1,self.capacity).bind(active)
    bad_biases=(Tensor.zeros(self.columns-1,device='CPU').realize(),
      SimpleNamespace(device='CL',dtype=dtypes.float,shape=(self.columns,)),
      Tensor.zeros(self.columns,device='CPU',dtype=dtypes.int).realize())
    for bias in bad_biases:
      with self.assertRaises(ValueError):
        prepare_qkv_int8(raw_parts,sa,sw,list(sizes),cache,qnorm,knorm,freqs,self.start,self.heads,1e-6,rows,bias=bias)

if __name__=='__main__':unittest.main()
