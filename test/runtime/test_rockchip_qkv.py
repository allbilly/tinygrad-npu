import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, function
from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv
from tinygrad.llm.model import apply_rope, precompute_freqs_cis


class TestRockchipQKV(unittest.TestCase):
  def test_normalization_rope_cache_prefix_and_rebinding(self):
    rng = np.random.default_rng(41)
    for heads,kv_heads,dim,capacity,context in [(4,2,64,32,64),(16,8,128,128,512)]:
      freqs = precompute_freqs_cis(dim,context,device='CPU').realize()
      for dtype in (np.float16,np.float32):
        qn = Tensor(rng.uniform(.5,1.5,dim).astype(dtype),device='CPU').realize()
        kn = Tensor(rng.uniform(.5,1.5,dim).astype(dtype),device='CPU').realize()
        @function(precompile=True)
        def prepare(p,c,qw,kw,f,start): return prepare_qkv(p,c,qw,kw,f,start,heads,1e-6)
        def calculate(p,c,start,rows):
          q,k = prepare(p[:,:rows],c,qn,kn,freqs,start)
          Tensor.realize(q,k)
          return q,k
        run = TinyJit(calculate)
        for count,start in [(capacity,0),(capacity,context-capacity),(17,7),(7,3),(capacity,0)]:
          data = rng.normal(0,.8,(1,capacity,(heads+2*kv_heads)*dim)).astype(np.float32)
          cache = rng.normal(0,.2,(2,1,kv_heads,context,dim)).astype(np.float16)
          q,kv = run(Tensor(data,device='CPU').realize(),Tensor(cache.copy(),device='CPU').realize(),
                     UOp.variable('start',0,context-1).bind(start),UOp.variable('rows',1,capacity).bind(count))
          got_q = q.pad_to((1,heads,capacity,dim)).numpy()[:,:,:count]
          got_kv = kv.numpy()
          projected = Tensor(data[:,:count].copy(),device='CPU')
          original_q = projected[:,:,:heads*dim].reshape(1,count,heads,dim).transpose(1,2)
          original_k = projected[:,:,heads*dim:(heads+kv_heads)*dim].reshape(1,count,kv_heads,dim).transpose(1,2)
          original_v = data[:,:count,(heads+kv_heads)*dim:].reshape(1,count,kv_heads,dim).transpose(0,2,1,3).astype(np.float16)
          original_q = apply_rope(original_q.rmsnorm(eps=1e-6)*qn,freqs[start:start+count]).numpy()
          original_k = apply_rope(original_k.rmsnorm(eps=1e-6)*kn,freqs[start:start+count]).half().numpy()
          np.testing.assert_allclose(got_q,original_q,atol=2e-6,rtol=2e-5)
          np.testing.assert_allclose(got_kv[0,:,:,start:start+count],original_k,atol=2e-3,rtol=2e-3)
          np.testing.assert_array_equal(got_kv[1,:,:,start:start+count],original_v)
          np.testing.assert_array_equal(got_kv[:,:,:,:start].view(np.uint16),cache[:,:,:,:start].view(np.uint16))
          np.testing.assert_array_equal(got_kv[:,:,:,start+count:].view(np.uint16),cache[:,:,:,start+count:].view(np.uint16))


if __name__=='__main__': unittest.main()
