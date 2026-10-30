import os
import numpy as np
import pytest
from tinygrad import Context,Tensor,TinyJit,UOp
from tinygrad.llm.kernels.rockchip_cpu import prefill_attention
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from tinygrad.llm.kernels.rockchip_padding import capacity_input,mark_zero_tail


def test_gqa_prefill_preserves_bits_dynamic_tails_and_future_cache_poison():
  if not {4,5,6,7}.issubset(os.sched_getaffinity(0)):pytest.skip('CPU worker pool requires CPU4-7')
  pool=AttentionPool()
  rng=np.random.default_rng(7642)
  runs=[]
  def forward(query,cache,total,rows):
    active=mark_zero_tail(query[:,:,:rows,:],query,'gqa_test')
    return capacity_input(prefill_attention(active,cache,total,True,pool)).realize()
  runs=[TinyJit(forward),TinyJit(forward)]
  try:
    for active,total in ((128,128),(17,190),(65,512),(1,128),(128,512)):
      q=rng.normal(0,.25,(1,14,128,64)).astype(np.float32)
      q[:,:,active:]=np.nan
      kv=rng.normal(0,.2,(2,1,2,512,64)).astype(np.float16)
      kv[:,:,:,total:]=np.nan
      query,cache=Tensor(q,device='CPU').realize(),Tensor(kv,device='CPU').realize()
      values=[]
      for enabled,run in enumerate(runs):
        with Context(ONPU_CPU_GQA_PREFILL=enabled,ONPU_CPU_PREFILL_LAYOUT=1),pool.prefill():
          values.append(run(query,cache,UOp.variable('gqa_total',1,512).bind(total),UOp.variable('gqa_rows',1,128).bind(active)).numpy().copy())
      np.testing.assert_array_equal(values[0].view(np.uint32),values[1].view(np.uint32))
      assert np.isfinite(values[1][:,:,:active]).all()
      assert not values[1][:,:,active:].any()
  finally:pool.close()


def test_qkv_token_major_layout_preserves_logical_queries_and_all_cache_bytes():
  from test.runtime.test_rockchip_shared_pool import _make_case,_run
  for rows in (17,65,128):
    results=[]
    for enabled in (0,1):
      with Context(ONPU_CPU_GQA_PREFILL=enabled):results.append(_run(_make_case('qkv',rows,811)))
    for a,b in zip(*results):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
