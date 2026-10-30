import os
import numpy as np
import pytest
from tinygrad import Context, Tensor
from tinygrad.llm.kernels.rockchip_cpu import decode_attention
from tinygrad.llm.kernels.rockchip_pool import AttentionPool


@pytest.mark.parametrize('heads,dim', ((12,128),(14,64)))
def test_two_kv_decode_pool_is_opt_in_and_bitwise_with_poisoned_future_cache(heads,dim):
  if not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
    pytest.skip('CPU attention pool needs caller affinity on CPU4-7')
  pool=AttentionPool()
  previous_affinity=os.sched_getaffinity(0)
  try:
    with pool:
      assert pool.active
      for seed,total in ((101,16),(202,128),(303,512)):
        rng=np.random.default_rng(seed)
        query_data=rng.normal(0,.3,(1,heads,1,dim)).astype(np.float32)
        cache_data=rng.normal(0,.3,(2,1,2,512,dim)).astype(np.float16)
        cache_data[:,:,:,total:]=np.nan
        query=Tensor(query_data,device='CPU').realize()
        cache=Tensor(cache_data,device='CPU').realize()
        calls_before=pool.lib.pool_calls()
        with Context(ONPU_CPU_DECODE_POOL=0):
          baseline=decode_attention(query,cache,total,pool=pool,fast_softmax=True).numpy().copy()
        assert pool.lib.pool_calls()==calls_before
        with Context(ONPU_CPU_DECODE_POOL=1):
          actual=decode_attention(query,cache,total,pool=pool,fast_softmax=True).numpy().copy()
        np.testing.assert_array_equal(actual.view(np.uint32),baseline.view(np.uint32))
        assert np.isfinite(actual).all()
        assert pool.lib.pool_calls()-calls_before==1
        assert os.sched_getaffinity(0)=={4}
  finally:
    pool.close()
  assert os.sched_getaffinity(0)==previous_affinity
  assert not pool.started


@pytest.mark.parametrize('heads,dim', ((12,128),(14,64)))
def test_small_kv_pool_ignores_subthreshold_and_rebinds_changed_inputs(heads,dim):
  if not {4,5,6,7}.issubset(os.sched_getaffinity(0)):
    pytest.skip('CPU attention pool needs caller affinity on CPU4-7')
  pool=AttentionPool()
  previous_affinity=os.sched_getaffinity(0)
  try:
    with pool:
      for index,total in enumerate((1,15,16,128,512)):
        rng=np.random.default_rng(700+index)
        query=Tensor(rng.normal(0,.25,(1,heads,1,dim)).astype(np.float32),device='CPU').realize()
        values=rng.normal(0,.2,(2,1,2,512,dim)).astype(np.float16)
        values[:,:,:,total:]=np.nan
        cache=Tensor(values,device='CPU').realize()
        calls_before=pool.lib.pool_calls()
        with Context(ONPU_CPU_DECODE_POOL=0):
          expected=decode_attention(query,cache,total,pool=pool,fast_softmax=True).numpy().copy()
        assert pool.lib.pool_calls()==calls_before
        with Context(ONPU_CPU_DECODE_POOL=1):
          actual=decode_attention(query,cache,total,pool=pool,fast_softmax=True).numpy().copy()
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        assert np.isfinite(actual).all()
        assert pool.lib.pool_calls()-calls_before==(1 if total>=16 else 0)
  finally:
    pool.close()
  assert os.sched_getaffinity(0)==previous_affinity
  assert not pool.started
