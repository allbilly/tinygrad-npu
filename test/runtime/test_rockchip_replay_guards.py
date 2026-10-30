"""Deduplicated checks retain every lifetime and coherency invalidation."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.kernels.rockchip_replay import NativeReplay
from tinygrad.runtime.ops_rockchip import RockchipBuffer


class Meta:
  def __init__(self):self.version,self.clean,self.reads=3,3,0
  @property
  def cpu_clean_version(self):
    self.reads+=1
    return self.clean


def fixture(gpu=False):
  token=Tensor([7],device='CPU',dtype=dtypes.int32).realize().uop.base
  replay=NativeReplay.__new__(NativeReplay)
  replay.roots=[token.buffer]
  replay.input_sizes=[4]
  guard=Buffer('CPU',4)
  replay.guards={guard:None}
  weight=SimpleNamespace(version=8)
  replay.weight_guards={1:(weight,8)}
  replay.attention_programs=[SimpleNamespace(closed=False,owned=[object()])]
  metas=[Meta() for _ in range(3 if gpu else 1)]
  buffers=tuple(RockchipBuffer(i*64,i*64,64,metas[0 if i<4 else i-3]) for i in range(6 if gpu else 3))
  replay.plans=[(SimpleNamespace(owned=[object()],bufs=buffers),buffers) for _ in range(175)]
  return replay,token,metas,guard,weight


@pytest.mark.parametrize('gpu',(False,True))
def test_shared_output_metadata_checked_once_and_dirty_outputs_rejected(gpu):
  replay,token,metas,_,_=fixture(gpu)
  with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=1):
    assert replay.valid([token])
    assert [m.reads for m in metas]==[1]*len(metas)
    for meta in metas:
      meta.clean-=1
      assert not replay.valid([token])
      meta.clean+=1
    assert replay.valid([token])
  assert 'ROCKCHIP' not in Device._opened_devices and 'CL' not in Device._opened_devices


@pytest.mark.parametrize('mutation',('input_count','input_size','storage','weight','closed','released_attention','released_plan','reprepared'))
def test_changed_input_storage_weight_or_prepared_resource_invalidates(mutation):
  replay,token,_,guard,weight=fixture()
  inputs=[token]
  with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=1):
    assert replay.valid(inputs)
    if mutation=='input_count':inputs=[]
    elif mutation=='input_size':inputs=[Tensor([7,8],device='CPU',dtype=dtypes.int32).realize().uop.base]
    elif mutation=='storage':replay.guards[guard]=object()
    elif mutation=='weight':weight.version+=1
    elif mutation=='closed':replay.attention_programs[0].closed=True
    elif mutation=='released_attention':replay.attention_programs[0].owned=[]
    elif mutation=='released_plan':replay.plans[0][0].owned=[]
    else:
      plan,buffers=replay.plans[0]
      plan.bufs=tuple(list(buffers))
      assert plan.bufs==buffers and plan.bufs is not buffers
    assert not replay.valid(inputs)
