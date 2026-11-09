"""The tanh producer gets no alias privilege without both explicit upload flags."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from tinygrad import UOp,dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops,KernelInfo,ProgramInfo
from tinygrad.llm.kernels import rockchip_replay as replay


@pytest.mark.parametrize('packed,alias,dtype',[(p,a,d) for p in (0,1) for a in (0,1) for d in (dtypes.uint8,dtypes.half)])
def test_tanh_upload_requires_exact_byte_producer_and_two_optins(monkeypatch,packed,alias,dtype):
  flags={'ONPU_FFN_PACKED_UPLOAD':packed,'ONPU_REPLAY_TANH_PACK_ALIAS':alias}
  monkeypatch.setattr(replay,'getenv',lambda name,default=0:flags.get(name,default))
  params=(UOp.param(0,dtype,64),UOp.param(1,dtype,64))
  sink=UOp.sink(*params,arg=KernelInfo('rockchip_int8_gelu_pack'))
  info=replace(ProgramInfo.from_sink(sink,Target('CPU')),outs=(0,),ins=(1,))
  body=UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=params)))
  call=SimpleNamespace(body=body,device='CPU')
  size=64*dtype.itemsize
  assert replay._full_write(call,0,size)==bool(packed and alias and dtype is dtypes.uint8)
  assert not replay._full_write(call,0,size+64)
  assert not replay._full_write(call,1,size)
  assert not replay._full_write(SimpleNamespace(body=body.replace(arg=replace(info,ins=(0,1))),device='CPU'),0,size)
  assert not replay._full_write(SimpleNamespace(body=body,device='CL'),0,size)
