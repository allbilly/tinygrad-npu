import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, function
from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv, prepare_qkv_int8
from tinygrad.llm.model import apply_rope
from tinygrad.helpers import round_up


class TestRockchipQKVWithoutNorm(unittest.TestCase):
  def test_dense_rope_cache_and_rebinding_without_norm(self):
    rng = np.random.default_rng(20261028)
    for heads,kv,dim,capacity,context in ((14,2,64,17,64),(16,2,64,32,64),(32,4,64,32,128)):
      freqs = Tensor(rng.normal(0,.5,(context,dim)).astype(np.float32),device='CPU').realize()
      @function(precompile=True)
      def prepare(projected,cache,freqs,start):
        return prepare_qkv(projected,cache,None,None,freqs,start,heads,1e-6)
      @TinyJit
      def run(projected,cache,start,rows):
        result = prepare(projected[:,:rows],cache,freqs,start)
        Tensor.realize(*result)
        return result
      for count,start in ((capacity,0),(min(7,capacity),5),(1,context-1),(capacity,2)):
        data = rng.normal(0,.8,(1,capacity,(heads+2*kv)*dim)).astype(np.float32)
        initial = rng.normal(0,.2,(2,1,kv,context,dim)).astype(np.float16)
        projected = Tensor(data,device='CPU').realize()
        cache = Tensor(initial.copy(),device='CPU').realize()
        q,got_cache = run(projected,cache,UOp.variable('start',0,context-1).bind(start),
                          UOp.variable('rows',1,capacity).bind(count))
        rawq = projected[:,:count,:heads*dim].reshape(1,count,heads,dim).transpose(1,2)
        rawk = projected[:,:,heads*dim:(heads+kv)*dim][:,:count].reshape(1,count,kv,dim).transpose(1,2)
        rawv = data[:,:count,(heads+kv)*dim:].reshape(1,count,kv,dim).transpose(0,2,1,3).astype(np.float16)
        expected_q = apply_rope(rawq,freqs[start:start+count]).numpy()
        expected_k = apply_rope(rawk,freqs[start:start+count]).half().numpy()
        actual_q = q.pad_to((1,heads,capacity,dim)).numpy()[:,:,:count]
        np.testing.assert_allclose(actual_q,expected_q,atol=2e-6,rtol=2e-5)
        actual = got_cache.numpy()
        np.testing.assert_allclose(actual[0,:,:,start:start+count],expected_k,atol=2e-3,rtol=2e-3)
        np.testing.assert_array_equal(actual[1,:,:,start:start+count],rawv)
        np.testing.assert_array_equal(actual[:,:,:,:start].view(np.uint16),initial[:,:,:,:start].view(np.uint16))
        np.testing.assert_array_equal(actual[:,:,:,start+count:].view(np.uint16),initial[:,:,:,start+count:].view(np.uint16))

  def test_int8_without_norm_matches_dense_for_full_and_partial_tiles(self):
    rng = np.random.default_rng(20261029)
    heads,kv,dim,context = 14,2,64,256
    columns = (heads+2*kv)*dim
    freqs = Tensor(rng.normal(0,.4,(context,dim)).astype(np.float32),device='CPU').realize()
    sw = rng.uniform(.001,.003,columns).astype(np.float32)
    for active,sizes in ((17,(64,17)),(81,(64,17)),(128,(64,64))):
      values = rng.integers(-120000,120000,(sum(sizes),columns),dtype=np.int32)
      sa = rng.uniform(.001,.003,sum(sizes)).astype(np.float32)
      padded_columns = round_up(columns,32)
      width = round_up((padded_columns+2)//3,32)
      raw,scales=[],[]
      offset=0
      for size in sizes:
        part=values[offset:offset+size]
        padded=np.zeros((size,padded_columns),dtype=np.int32)
        padded[:,:columns]=part
        raw.append(Tensor(np.concatenate([padded[:,c:c+width].ravel() for c in range(0,padded_columns,width)]),device='CPU'))
        scales.append(Tensor(sa[offset:offset+size],device='CPU'))
        offset+=size
      dense = Tensor((values[:active].astype(np.float32)*sa[:active,None]*sw[None,:])[None],device='CPU')
      initial = rng.normal(0,.2,(2,1,kv,context,dim)).astype(np.float16)
      cache_a,cache_b=(Tensor(initial.copy(),device='CPU').realize() for _ in range(2))
      expected=prepare_qkv(dense,cache_a,None,None,freqs,11,heads,1e-6)
      actual=prepare_qkv_int8(raw,scales,Tensor(sw,device='CPU'),list(sizes),cache_b,None,None,freqs,11,heads,1e-6,active)
      np.testing.assert_array_equal(actual[0].numpy(),expected[0].numpy())
      np.testing.assert_array_equal(actual[1].numpy(),expected[1].numpy())

  def test_one_missing_norm_is_rejected_by_both_apis(self):
    heads,kv,dim,rows,context=4,2,64,2,8
    columns=(heads+2*kv)*dim
    projected=Tensor.zeros(1,rows,columns,device='CPU').realize()
    cache=Tensor.zeros(2,1,kv,context,dim,device='CPU',dtype=dtypes.half).realize()
    norm=Tensor.ones(dim,device='CPU').realize()
    freqs=Tensor.zeros(context,dim,device='CPU').realize()
    for qn,kn in ((None,norm),(norm,None)):
      with self.assertRaises(ValueError):prepare_qkv(projected,cache,qn,kn,freqs,0,heads,1e-6)
      raw=[Tensor.zeros(rows*round_up(columns,32),device='CPU',dtype=dtypes.int32).realize()]
      scales=[Tensor.ones(rows,device='CPU').realize()]
      with self.assertRaises(ValueError):prepare_qkv_int8(raw,scales,Tensor.ones(columns,device='CPU'),[rows],cache,qn,kn,freqs,0,heads,1e-6,rows)


if __name__=='__main__': unittest.main()
