"""IRQ stop requests cancel work immediately; final close joins the worker."""
import pytest
from tinygrad import Device
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip_idle import IdleHelper
from test.runtime.test_rockchip_shared_pool import _needs_pool, _pool_test_lock


@pytest.mark.parametrize('asynchronous',(False,True))
def test_rapid_wake_pause_exception_and_restart(asynchronous):
  _needs_pool()
  with _pool_test_lock(),Context(ONPU_ASYNC_IRQ_WAKE=int(asynchronous)):
    for _ in range(3):
      helper=IdleHelper()
      try:
        for _ in range(300):
          with helper:assert helper.lib.spinner_active()==1
          assert helper.lib.spinner_active()==0
          if not asynchronous:assert helper.lib.spinner_busy()==0
        with pytest.raises(RuntimeError,match='scope failure'):
          with helper:raise RuntimeError('scope failure')
        assert helper.lib.spinner_active()==0
        if not asynchronous:assert helper.lib.spinner_busy()==0
      finally:helper.close()
      assert not helper.started
      assert helper.lib.spinner_busy()==0
  assert not {'ROCKCHIP','CL'}.intersection(Device._opened_devices)


def test_switching_back_to_synchronous_wait_drains_a_pending_stop():
  _needs_pool()
  with _pool_test_lock():
    helper=IdleHelper()
    try:
      for _ in range(100):
        with Context(ONPU_ASYNC_IRQ_WAKE=1):
          with helper:pass
        with Context(ONPU_ASYNC_IRQ_WAKE=0):
          with helper:pass
          assert helper.lib.spinner_busy()==0 and helper.lib.spinner_active()==0
    finally:helper.close()
    assert helper.lib.spinner_busy()==0
