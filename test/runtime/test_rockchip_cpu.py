import functools, os, platform, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, function
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention, swiglu_pack, swiglu_pack_tiles, unpack_tiles, rms_norm
from tinygrad.llm.kernels import rockchip_cpu


class TestRockchipCPUAttention(unittest.TestCase):
  def test_prefill_pool_preserves_bits_affinity_and_parks_between_calls(self):
    if platform.machine().lower() not in ('aarch64','arm64') or not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
      self.skipTest('AArch64 attention pool needs CPU4-7')
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    pool = AttentionPool()
    previous = os.sched_getaffinity(0)
    rng = np.random.default_rng(197)
    try:
      with pool.prefill():
        self.assertEqual(os.sched_getaffinity(0),previous)
        cases = ((16,8,128,512,(1,7,16,17,64,127,128)),(9,3,64,8192,(7,17,128)))
        for heads,kv_heads,dim,context,counts in cases:
          for fast in (False,True):
            for count in counts:
              values = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
              end = count+31
              values[:,:,:,end:] = np.nan
              query = Tensor(rng.normal(0,.3,(1,heads,128,dim)).astype(np.float32),device='CPU').realize()
              rows = UOp.variable('rows',1,128).bind(count)
              cache = Tensor(values,device='CPU').realize()
              expected = prefill_attention(query[:,:,:rows],cache,end,fast_softmax=fast).pad_to((1,heads,128,dim)).numpy()[:,:,:count]
              calls = pool.lib.pool_calls()
              actual = prefill_attention(query[:,:,:rows],cache,end,fast_softmax=fast,pool=pool)
              result = actual.pad_to((1,heads,128,dim)).numpy()[:,:,:count]
              np.testing.assert_array_equal(result.view(np.uint32),expected.view(np.uint32))
              self.assertTrue(np.isfinite(result).all())
              self.assertEqual(pool.lib.pool_calls()-calls,int(count>=16))
              self.assertTrue(pool.lib.pool_parked())
              self.assertEqual(os.sched_getaffinity(0),previous)
      self.assertEqual(os.sched_getaffinity(0),previous)
    finally: pool.close()

  def test_long_two_and_three_kv_decode_pool_matches_serial_and_masks_future(self):
    if platform.machine().lower() not in ('aarch64','arm64') or not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
      self.skipTest('AArch64 attention pool needs CPU4-7')
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    pool = AttentionPool()
    rng = np.random.default_rng(89)
    context,dim = 8192,64
    try:
      with pool:
        for kv_heads in (2,3):
          heads = kv_heads*3
          query = Tensor(rng.normal(0,.3,(1,heads,1,dim)).astype(np.float32),device='CPU').realize()
          for end in (1,15,1023,1024,8191,8192):
            values = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
            values[:,:,:,end:] = np.nan
            cache = Tensor(values,device='CPU').realize()
            length = UOp.variable('end',1,context).bind(end)
            expected = decode_attention(query,cache,length,fast_softmax=True).numpy()
            calls = pool.lib.pool_calls()
            actual = decode_attention(query,cache,length,pool=pool,fast_softmax=True).numpy()
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(np.isfinite(actual).all())
            self.assertEqual(pool.lib.pool_calls()-calls,int(end>=1024))
    finally: pool.close()

  def test_attention_near_8192_capacity_and_masked_future(self):
    rng = np.random.default_rng(75)
    context,end,rows,dim = 8192,8191,17,16
    query = rng.normal(0,.3,(1,4,rows,dim)).astype(np.float32)
    values = rng.normal(0,.3,(2,1,2,context,dim)).astype(np.float16)
    values[:,:,:,end:] = np.nan
    actual = prefill_attention(Tensor(query,device='CPU'),Tensor(values,device='CPU'),end).numpy()
    expected = np.empty_like(query)
    mask = np.arange(end)[None,:]<=np.arange(end-rows,end)[:,None]
    for head in range(4):
      score = query[0,head]@values[0,0,head//2,:end].astype(np.float32).T/np.sqrt(dim)
      score = np.where(mask,score,-np.inf)
      probability = np.exp(score-score.max(-1,keepdims=True))
      probability /= probability.sum(-1,keepdims=True)
      expected[0,head] = probability@values[1,0,head//2,:end].astype(np.float32)
    self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=3e-5)
    fast = prefill_attention(Tensor(query,device='CPU'),Tensor(values,device='CPU'),end,fast_softmax=True).numpy()
    np.testing.assert_allclose(fast,expected,atol=2e-6,rtol=3e-5)

  def test_long_64_channel_prefill_rebinds_full_and_partial_eight_row_groups(self):
    rng = np.random.default_rng(85)
    context,capacity,heads,kv_heads,dim = 8192,128,9,3,64
    for fast in (False,True):
      run = TinyJit(lambda q,kv,end,rows:prefill_attention(q[:,:,:rows],kv,end,fast_softmax=fast).realize())
      for index,(rows,end) in enumerate(((128,8192),(65,8190),(7,8184),(127,8192))):
        query = rng.normal(0,.3,(1,heads,capacity,dim)).astype(np.float32)
        if index==1:query *= 100
        values = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
        values[:,:,:,end:] = np.nan
        expected = np.empty((1,heads,rows,dim),np.float32)
        mask = np.arange(end)[None,:]<=np.arange(end-rows,end)[:,None]
        for head in range(heads):
          score = query[0,head,:rows]@values[0,0,head//3,:end].astype(np.float32).T/8
          score = np.where(mask,score,-np.inf)
          probability = np.exp(score-score.max(-1,keepdims=True))
          probability /= probability.sum(-1,keepdims=True)
          expected[0,head] = probability@values[1,0,head//3,:end].astype(np.float32)
        actual = run(Tensor(query,device='CPU').realize(),Tensor(values,device='CPU').realize(),
                     UOp.variable('end',1,context).bind(end),UOp.variable('rows',1,capacity).bind(rows))
        result = actual.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows]
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result,expected,atol=4e-6,rtol=3e-4)

  def test_parallel_rms_norm_symbolic_rows_and_weight_types(self):
    rng = np.random.default_rng(45)
    @function(precompile=True)
    def norm(x,w): return rms_norm(x,w,1e-6)
    for capacity,width in [(32,576),(256,1024),(64,4096)]:
      for dtype in (np.float16,np.float32):
        run = TinyJit(lambda x,w,rows:norm(x[:,:rows],w).realize())
        for count in (capacity,17,1,capacity//2):
          data = rng.normal(0,.8,(1,capacity,width)).astype(np.float32)
          data[0,0] = 0
          gamma = rng.uniform(.5,1.5,width).astype(dtype)
          actual = run(Tensor(data,device='CPU').realize(),Tensor(gamma,device='CPU').realize(),
                       UOp.variable('rows',1,capacity).bind(count)).pad_to((1,capacity,width)).numpy()[:,:count]
          expected = (Tensor(data[:,:count].copy(),device='CPU').rmsnorm(eps=1e-6)*Tensor(gamma,device='CPU')).numpy()
          np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)

  def test_native_output_tiles_and_split_sums(self):
    rng = np.random.default_rng(39)
    for split in (False,True):
      for rows,tile,columns in [(1,1,96),(17,16,64),(128,64,1024)]:
        raw = rng.normal(0,20,(3 if split else 1,rows,columns)).astype(np.float32)
        parts = []
        for start in range(0,rows,tile):
          block = raw[:,start:start+tile]
          native = block.reshape(raw.shape[0],block.shape[1],columns//4,4).transpose(0,2,1,3).copy()
          parts.append(Tensor(native if split else native[0],device='CPU').realize())
        actual = unpack_tiles(parts,split).numpy()
        expected = ((raw[0]+raw[1])+raw[2]) if split else raw[0]
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_prefill_reused_kv_loads_match_scalar_fp32_sums(self):
    rng = np.random.default_rng(36)
    for count,start in [(128,0),(128,128),(17,45)]:
      query = Tensor(rng.normal(0,.8,(1,16,count,128)).astype(np.float32),device='CPU').realize()
      cache = Tensor(rng.normal(0,.8,(2,1,8,512,128)).astype(np.float16),device='CPU').realize()
      actual = prefill_attention(query,cache,start+count).numpy()
      with patch.object(rockchip_cpu,'_attention',functools.partial(rockchip_cpu._attention,blocked_rows=False)):
        expected = prefill_attention(query,cache,start+count).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      for block,tile in [(4,4),(8,1),(8,2),(16,1)]:
        with patch.object(rockchip_cpu,'_attention',functools.partial(rockchip_cpu._attention,query_rows=block,channel_tile=tile)):
          tiled = prefill_attention(query,cache,start+count).numpy()
        np.testing.assert_array_equal(tiled.view(np.uint32),expected.view(np.uint32))

  def test_causal_rows_do_not_read_nonfinite_future_values(self):
    rng = np.random.default_rng(40)
    query = Tensor(rng.normal(0,.3,(1,4,16,64)).astype(np.float32),device='CPU').realize()
    values = rng.normal(0,.3,(2,1,2,32,64)).astype(np.float16)
    values[1,0,:,15,0] = np.nan
    cache = Tensor(values,device='CPU').realize()
    with patch.object(rockchip_cpu,'_attention',functools.partial(rockchip_cpu._attention,blocked_rows=False)):
      expected = prefill_attention(query,cache,16).numpy()
    for block in (4,8,16):
      with patch.object(rockchip_cpu,'_attention',functools.partial(rockchip_cpu._attention,query_rows=block)):
        actual = prefill_attention(query,cache,16).numpy()
      np.testing.assert_array_equal(actual[:,:,:15].view(np.uint32),expected[:,:,:15].view(np.uint32))
      self.assertTrue(np.isfinite(actual[:,:,:15]).all())
      self.assertTrue(np.isnan(actual[:,:,15,0]).all())

  def test_swiglu_packing_matches_cpu_rounding_and_layout(self):
    rng = np.random.default_rng(25)
    for rows,depth in [(1,64),(17,128),(128,3072),(193,3072),(256,3072)]:
      tile = min(64,65536//depth)
      run = TinyJit(lambda x:swiglu_pack(x,depth,tile).realize())
      for _ in range(4):
        data = rng.normal(0,3,(1,rows,depth*2)).astype(np.float32)
        data.flat[:6] = [-0.0,0.0,-80.,80.,-1.,1.]
        gate,up = Tensor(data[:,:,:depth].copy(),device='CPU'),Tensor(data[:,:,depth:].copy(),device='CPU')
        values = (gate*gate.sigmoid().contiguous()*up).half().numpy().reshape(rows,depth)
        expected = np.concatenate([values[s:s+tile].reshape(-1,depth//8,8).transpose(1,0,2).flatten()
                                   for s in range(0,rows,tile)])
        actual = run(Tensor(data,device='CPU').realize()).numpy()
        np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
        input_tile = min(rows,64)
        native = np.concatenate([data.reshape(rows,depth*2)[s:s+input_tile].reshape(
          -1,depth//2,4).transpose(1,0,2).flatten() for s in range(0,rows,input_tile)])
        raw = Tensor(native,device='CPU').realize()
        actual = swiglu_pack(raw,depth,tile,input_tile).numpy()
        np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
        parts = [Tensor(data.reshape(rows,depth*2)[s:s+input_tile].reshape(-1,depth//2,4).transpose(
          1,0,2).copy().flatten(),device='CPU').realize() for s in range(0,rows,input_tile)]
        direct = swiglu_pack_tiles(parts,depth,tile).numpy()
        np.testing.assert_array_equal(direct.view(np.uint16),expected.view(np.uint16))
        if depth%96==0:
          actual = swiglu_pack(raw,depth,64,input_tile,split_k=True).numpy()
          split_expected = np.concatenate([values[s:s+64].reshape(-1,depth//8,8).transpose(1,0,2).flatten()
                                           for s in range(0,rows,64)])
          np.testing.assert_array_equal(actual.view(np.uint16),split_expected.view(np.uint16))
          direct = swiglu_pack_tiles(parts,depth,64,split_k=True).numpy()
          np.testing.assert_array_equal(direct.view(np.uint16),split_expected.view(np.uint16))

  def test_causal_prefill_symbolic_rows_and_cached_prefix(self):
    rng = np.random.default_rng(20261019)
    @function(precompile=True)
    def attention(q,c,length): return prefill_attention(q,c,length)
    for capacity,context,cases in [(8,32,[(1,0),(3,7),(8,24),(4,10),(1,31)]),
                                    (32,64,[(1,0),(16,8),(32,32),(24,12),(2,57)])]:
      run = TinyJit(lambda q,c,length,rows:attention(q[:,:,:rows],c,length).realize())
      for count,start in cases:
        query = rng.normal(0,.3,(1,4,capacity,64)).astype(np.float32)
        cache = rng.normal(0,.3,(2,1,2,context,64)).astype(np.float16)
        output = run(Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize(),
                     UOp.variable('length',1,context).bind(start+count),UOp.variable('rows',1,capacity).bind(count))
        result = output.pad_to((1,4,capacity,64)).numpy()
        keys,values = (np.repeat(part[:,:,:start+count].astype(np.float32),2,axis=1) for part in cache)
        score = query[:,:,:count]@keys.transpose(0,1,3,2)*np.float32(64**-.5)
        causal = np.arange(start+count)[None,:]<=start+np.arange(count)[:,None]
        score = np.where(causal,score,-np.inf)
        probability = np.exp(score-score.max(-1,keepdims=True))
        probability /= probability.sum(-1,keepdims=True)
        expected = np.pad(probability@values,((0,0),(0,0),(0,capacity-count),(0,0)))
        np.testing.assert_allclose(result,expected,atol=2e-6,rtol=2e-5)

  def test_grouped_attention_symbolic_length_and_rebinding(self):
    rng = np.random.default_rng(20261018)
    for heads,kv_heads,dim,context in [(16,8,128,512),(4,1,64,32),(9,3,64,512)]:
      run = TinyJit(lambda q,c,length:decode_attention(q,c,length).realize())
      for length in [1,7,context,context//2,1]:
        query = rng.normal(0,.3,(1,heads,1,dim)).astype(np.float32)
        cache = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
        q,c = Tensor(query,device='CPU').realize(),Tensor(cache,device='CPU').realize()
        result = run(q,c,UOp.variable('length',1,context).bind(length)).numpy()
        with patch.object(rockchip_cpu,'_attention',functools.partial(rockchip_cpu._attention,decode_grouped=False)):
          scalar = decode_attention(q,c,length).numpy()
        np.testing.assert_array_equal(result.view(np.uint32),scalar.view(np.uint32))
        keys,values = (np.repeat(part[:,:,:length].astype(np.float32),heads//kv_heads,axis=1) for part in cache)
        scores = (query@keys.transpose(0,1,3,2))*np.float32(dim**-.5)
        probability = np.exp(scores-scores.max(-1,keepdims=True))
        probability /= probability.sum(-1,keepdims=True)
        np.testing.assert_allclose(result,probability@values,atol=2e-6,rtol=2e-5)

  def test_reject_invalid_layout_and_length(self):
    query = Tensor.empty(1,4,1,64,device='CPU')
    cache = Tensor.empty(2,1,1,32,64,device='CPU').half()
    for length in [0,33]:
      with self.assertRaises(ValueError): decode_attention(query,cache,length)
    with self.assertRaises(ValueError): decode_attention(query.half(),cache,1)

  def test_precompiled_model_function(self):
    @function(precompile=True)
    def run(q,c,length): return decode_attention(q,c,length)
    query = Tensor(np.ones((1,4,1,64),dtype=np.float32),device='CPU').realize()
    cache = Tensor(np.ones((2,1,1,32,64),dtype=np.float16),device='CPU').realize()
    result = run(query,cache,UOp.variable('length',1,32).bind(17)).numpy()
    np.testing.assert_allclose(result,np.ones((1,4,1,64)),atol=1e-6)


if __name__=='__main__': unittest.main()
