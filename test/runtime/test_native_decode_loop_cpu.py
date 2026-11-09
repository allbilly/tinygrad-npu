"""Real C-loop execution with an owned CPU replay callback; no NPU device."""
import ctypes,contextlib,threading
from types import SimpleNamespace
import pytest
from tinygrad import Tensor,dtypes
from tinygrad.llm.kernels.rockchip_gpt2_decode import DecodeLoop


def phase(fail_at=None):
  result=Tensor([0],device='CPU',dtype=dtypes.int).realize()
  output=ctypes.c_int.from_address(result._buffer().ensure_allocated().get_buf('CPU'))
  visits=[]
  @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong,ctypes.POINTER(ctypes.c_uint64))
  def replay(position,token,stats):
    visits.append((position,token))
    if len(visits)==fail_at:return 123
    stats[0],stats[1]=1,3
    output.value=(token+1)%97
    return 0
  metadata=SimpleNamespace(version=0,cpu_clean_version=0)
  dev=SimpleNamespace(lock=threading.Lock(),check_healthy=lambda:None,poisoned=False,submit_count=0,task_count=0)
  native=SimpleNamespace(program=SimpleNamespace(fxn=replay),profiling=False,calls=[],valid=lambda inputs:True,
                         dev=dev,writes={0:metadata},write_counts={0:1},estimates=SimpleNamespace(ops=0,mem=0))
  captured=SimpleNamespace(native=native,ret=result,native_runs=0,_symbolic_ret=[])
  owner=SimpleNamespace(captured=captured,jit=SimpleNamespace(cnt=0),scope=contextlib.nullcontext)
  return owner,visits,metadata


def test_eos_uses_actual_step_count_for_tokens_versions_and_submission_counters():
  owner,visits,metadata=phase()
  loop=DecodeLoop(owner,32,97,(8,))
  assert loop(5,3,8)==[6,7,8]
  assert visits==[(3,5),(4,6),(5,7)]
  assert loop.tokens==owner.jit.cnt==owner.captured.native_runs==3
  assert metadata.version==metadata.cpu_clean_version==3
  assert owner.captured.native.dev.submit_count==3 and owner.captured.native.dev.task_count==9
  assert not owner.captured.native.dev.poisoned


def test_default_loop_keeps_fixed_count_and_failed_replay_poisoning():
  owner,_,metadata=phase()
  loop=DecodeLoop(owner,32,97)
  assert loop(5,3,4)==[6,7,8,9]
  assert metadata.version==4
  owner,_,metadata=phase(fail_at=2)
  loop=DecodeLoop(owner,32,97)
  with pytest.raises(RuntimeError,match='token 1'):loop(5,3,4)
  assert owner.captured.native.dev.poisoned and metadata.version==0


@pytest.mark.parametrize('stops',[(True,),(-1,),(97,),tuple(range(17))])
def test_invalid_stop_tokens_rejected_before_compilation(stops):
  owner,_,_=phase()
  with pytest.raises(ValueError,match='stop tokens'):DecodeLoop(owner,32,97,stops)
