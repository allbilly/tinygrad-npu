"""A decoder pinned to CPU4 can use existing workers for large logit soft caps."""
import os
import numpy as np
import pytest
from tinygrad import Tensor, Device
from tinygrad.llm.kernels.rockchip_softcap import softcap
from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock


@pytest.mark.parametrize('count',[1024,1025,131072,256000,256003])
def test_pinned_softcap_preserves_bits_input_and_tail_with_shared_workers(count):
  _needs_pool()
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
  rng=np.random.default_rng(count)
  values=rng.normal(0,30,count).astype(np.float32)
  values[:8]=[-0.,0.,np.nan,np.inf,-np.inf,-1e30,1e30,1.]
  with _pool_test_lock():
    pool=AttentionPool()
    state=SharedHostPool(pool)
    previous=os.sched_getaffinity(0)
    try:
      os.sched_setaffinity(0,{4})
      x=Tensor(values.copy(),device='CPU').realize()
      expected=softcap(x,30.).numpy().copy()
      with state.scope():
        assert os.sched_getaffinity(0)=={4}
        before=len(state.changes)
        actual=softcap(x,30.).numpy().copy()
        assert len(state.changes)==before+int(count>=131072)
        if count>=131072:
          assert state.changes[-1]['thread_creates_replaced_per_call']==3
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      np.testing.assert_array_equal(x.numpy().view(np.uint32),values.view(np.uint32))
    finally:
      state.restore()
      pool.close()
      os.sched_setaffinity(0,previous)
  assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)
