"""Aliased decode inputs still require independent, owned snapshots."""
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.llm.kernels.rockchip_replay import _copy_scalar_input
from tinygrad.engine.jit import _copy_input


@pytest.mark.parametrize('dtype,np_dtype',[(dtypes.int32,np.int32),(dtypes.int64,np.int64)])
def test_scalar_snapshot_copies_exact_bits_and_owns_independent_storage(dtype,np_dtype):
  source=Tensor([0],device='CPU',dtype=dtype).realize().uop.base
  snapshots=[]
  with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=1), \
       patch('tinygrad.llm.kernels.rockchip_replay._copy_input',side_effect=AssertionError('generic copy must be bypassed')):
    for value in (0,151935,-1,np.iinfo(np_dtype).min,np.iinfo(np_dtype).max):
      source.buffer.as_memoryview(allow_zero_copy=True)[:]=np.array([value],dtype=np_dtype).tobytes()
      copied=_copy_scalar_input(source)
      assert copied.buffer._storage is not source.buffer._storage
      snapshots.append((copied,value))
    source.buffer.as_memoryview(allow_zero_copy=True)[:]=np.array([17],dtype=np_dtype).tobytes()
    for copied,value in snapshots:
      assert bytes(copied.buffer.as_memoryview(allow_zero_copy=True))==np.array([value],dtype=np_dtype).tobytes()
  assert 'ROCKCHIP' not in Device._opened_devices and 'CL' not in Device._opened_devices


def test_scalar_copy_keeps_generic_fallback_for_disabled_or_ineligible_inputs():
  for values,dtype,enabled in (([3],dtypes.int32,0),([3,4],dtypes.int32,1),([3.5],dtypes.float,1)):
    source=Tensor(values,device='CPU',dtype=dtype).realize().uop.base
    with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=enabled), \
         patch('tinygrad.llm.kernels.rockchip_replay._copy_input',wraps=_copy_input) as fallback:
      copied=_copy_scalar_input(source)
      fallback.assert_called_once_with(source)
      assert bytes(copied.buffer.as_memoryview(allow_zero_copy=True))==bytes(source.buffer.as_memoryview(allow_zero_copy=True))
