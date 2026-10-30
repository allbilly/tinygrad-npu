"""Opt-in Mali activation packing between existing INT8 NPU FFN projections."""
import functools, struct
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.runtime.support.rockchip_gpu import GPU_FFN_MAGIC, GPU_FFN_SOURCE
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


@functools.cache
def _program(out:UOp, raw:UOp, payload:UOp, weights:UOp, scales:UOp, second:UOp) -> UOp:
  inputs = (out,raw,payload,weights,scales,second)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_gpu_ffn_128_3072'))
  info = replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,4,5),ins=(1,2,3))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=GPU_FFN_SOURCE),
                                    UOp(Ops.BINARY,arg=struct.pack('<4I',128,3072,64,GPU_FFN_MAGIC))))


def gpu_ffn_pack(raw:Tensor, payload:Tensor, weight_scales:Tensor, device:str) -> tuple[tuple[Tensor,Tensor],Tensor]:
  if (raw.dtype is not dtypes.int32 or raw.numel()!=128*6144 or not isinstance(raw.device,str)
      or raw.device.split(':')[0]!='ROCKCHIP' or payload.dtype is not dtypes.uint8 or payload.numel()<512
      or not isinstance(payload.device,str) or payload.device.split(':')[0]!='ROCKCHIP'
      or weight_scales.device!='CPU' or weight_scales.dtype is not dtypes.float or weight_scales.shape!=(6144,)
      or device.split(':')[0]!='ROCKCHIP'):raise ValueError('Unqualified GPU FFN operands')
  result = Tensor.empty(196864,device=device,dtype=dtypes.uint8).custom_kernel(
    raw,payload,weight_scales.to(device),Tensor.empty(128,device=device,dtype=dtypes.float),
    Tensor.empty(196864,device=device,dtype=dtypes.uint8),fxn=_program)
  return (result[0],result[5]),result[4].to('CPU')
