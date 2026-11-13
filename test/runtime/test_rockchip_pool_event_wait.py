"""Event waits retain worker correctness and park all workers outside a request."""
import ctypes,time
import pytest
from tinygrad import Device
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock

@pytest.mark.parametrize('enabled',(False,True))
def test_event_wait_dispatch_park_stop_and_restart(enabled,monkeypatch):
  _needs_pool()
  monkeypatch.setenv('ONPU_HOST_POOL_WFE','1' if enabled else '0')
  with _pool_test_lock():
    for _ in range(3):
      pool=AttentionPool()
      values=[0]*4
      callback_type=ctypes.CFUNCTYPE(None,ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.c_int)
      @callback_type
      def callback(args,position,group):values[group]+=position+group
      args=(ctypes.c_void_p*4)()
      expected=[0]*4
      try:
        assert pool.event_wait is enabled
        for function in (pool.lib.parallel,pool.lib.parallel_parked):
          function.argtypes=[callback_type,ctypes.POINTER(ctypes.c_void_p),ctypes.c_int]
        for i in range(200):
          position=i%7-3
          function=pool.lib.parallel_parked if i%11==0 else pool.lib.parallel
          assert function(callback,args,position)==0
          expected=[v+position+group for group,v in enumerate(expected)]
          assert values==expected
          if i%17==0:
            assert pool.lib.pause_pool()==0 and pool.lib.pool_parked()==1
            if enabled:assert pool.lib.pool_sleepers()==3
            time.sleep(.001)
        assert pool.lib.pause_pool()==0
        if enabled:assert pool.lib.pool_sleepers()==3
      finally:pool.close()
      assert not pool.started
  assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)


def test_scoped_wait_mode_restores_after_exception_and_rejects_active_changes(monkeypatch):
  _needs_pool()
  monkeypatch.setenv('ONPU_HOST_POOL_WFE','0')
  with _pool_test_lock():
    pool=AttentionPool()
    try:
      with pytest.raises(RuntimeError,match='active owner'):
        with pool.wait_policy(True):pass
      with pool:
        assert pool.lib.pause_pool()==0 and pool.lib.pool_wait_mode()==0
        with pytest.raises(ValueError):
          with pool.wait_policy(1):pass
        with pytest.raises(RuntimeError,match='test failure'):
          with pool.wait_policy(True):
            assert pool.lib.pool_wait_mode()==1
            raise RuntimeError('test failure')
        assert pool.lib.pool_wait_mode()==0 and pool.lib.pool_parked()==1
        with pool.wait_policy(True):assert pool.lib.pool_wait_mode()==1
        assert pool.lib.pool_wait_mode()==0
    finally:pool.close()
  assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)
