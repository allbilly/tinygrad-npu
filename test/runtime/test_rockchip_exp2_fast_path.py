import unittest
import numpy as np
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source


class TestExp2NormalRange(unittest.TestCase):
  def test_all_half_encodings_and_random_float_bits_match_via_actual_cpu_loader(self):
    rng=np.random.default_rng(493)
    with np.errstate(all='ignore'):
      half=np.arange(65536,dtype=np.uint16).view(np.float16).astype(np.float32)
    edges=np.arange(-124.5,125,.5,dtype=np.float32)
    edges=np.concatenate((edges,np.nextafter(edges,np.float32(np.inf)),np.nextafter(edges,np.float32(-np.inf)),
                          np.array([-125,125],dtype=np.float32),np.nextafter(np.array([-125,125],dtype=np.float32),np.float32(0))))
    values=np.concatenate((half,rng.integers(0,2**32,262144,dtype=np.uint32).view(np.float32),
                           rng.uniform(-125,125,1048576).astype(np.float32),edges))
    values=np.pad(values,(0,(-len(values))%4))
    results=[]
    programs=[]
    for fast,scale in ((0,0),(1,0),(1,1)):
      with Context(ONPU_EXP2_FAST_PATH=fast,ONPU_EXP2_SCALE_BITS=scale):
        source=_vector_exp2_source()+'''void test_exp2(float*out,const float*x,int count){
          for(int i=0;i<count;i+=4)*(float4*)(out+i)=rockchip_exp2_vector(*(const float4*)(x+i));}'''
      renderer=_renderer()
      program=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'test_exp2',renderer.target,()))
      programs.append(program)
      out=np.empty_like(values)
      program(out.ctypes.data,values.ctypes.data,values.size)
      results.append(out)
    for result in results[1:]:np.testing.assert_array_equal(results[0].view(np.uint32),result.view(np.uint32))
    self.assertNotIn('ROCKCHIP',Device._opened_devices)

  def test_activation_quantization_payloads_and_scales_remain_exact(self):
    from test.runtime.test_rockchip_shared_pool import _make_case, _run
    for nonfinite in (False,True):
      outputs=[]
      for fast,scale in ((0,0),(1,0),(1,1)):
        with Context(ONPU_EXP2_FAST_PATH=fast,ONPU_EXP2_SCALE_BITS=scale):
          outputs.append(_run(_make_case('ffn_pack',128,102,depth=3072,nonfinite=nonfinite)))
      for candidate in outputs[1:]:
        for expected,actual in zip(outputs[0],candidate):
          np.testing.assert_array_equal(np.ascontiguousarray(expected).view(np.uint8),np.ascontiguousarray(actual).view(np.uint8))


if __name__=='__main__':unittest.main()
