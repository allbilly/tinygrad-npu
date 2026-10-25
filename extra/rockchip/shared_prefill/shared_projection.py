"""Experimental imperative GPU/NPU projection; not installed in model serving.

Only data packing/unpacking uses imported GEMs. NPU submission retains the
runtime's finite checks and blocking protocol, including an explicit CPU-cache
invalidation before its finite scan. No activation memcpy through the host.
"""
from __future__ import annotations
import ctypes,struct
from typing import cast
from tinygrad import Device,Tensor,dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import Target,round_up
from tinygrad.llm.kernels.rockchip_softmax import SharedNPUKernel
from tinygrad.runtime.autogen import opencl as cl,rockchip as rk
from tinygrad.runtime.ops_cl import CLDevice,check,checked
from tinygrad.runtime.ops_rockchip import RockchipDevice,RockchipMatmul


def source(rows:int,columns:int,depth:int,tile:int)->str:
  return f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void shared_pack(__global half *packed,__global const float *raw,__global uint *invalid,
                          __global const float *input,int start) {{
  int i=get_global_id(0);if(i>={tile*depth})return;
  int r=i/{depth},k=i%{depth};float value=input[(start+r)*{depth}+k];
  half converted=convert_half_rte(value);
  if(!isfinite(value)||!isfinite(converted))atomic_or(invalid,1u);
  packed[(k/8)*{tile*8}+r*8+k%8]=converted;
}}
__kernel void shared_unpack(__global const half *packed,__global const float *raw,__global uint *invalid,
                            __global float *output,int start) {{
  int i=get_global_id(0);if(i>={tile*columns})return;
  int r=i/{columns},c=i%{columns};
  output[(start+r)*{columns}+c]=raw[(c/4)*{tile*4}+r*4+c%4];
}}
'''


class SharedProjection:
  def __init__(self,rows:int,columns:int,depth:int,packed_weight:Tensor):
    if rows not in (32,64) or columns%32 or depth%32 or not 32<=depth<=4096:
      raise ValueError('Unsupported experimental shared projection geometry')
    tile=min(rows,64,65536//depth)
    if tile<1 or rows%tile or not 0<columns<=12288:raise ValueError('Unaligned NPU input tiles')
    if (packed_weight.device!='ROCKCHIP' or packed_weight.dtype is not dtypes.half or not packed_weight.uop.is_realized
        or packed_weight.shape!=(columns//16,depth//32,16,32)):
      raise ValueError('Expected exact static packed NPU FP16 weight')
    try:weight_buffer=packed_weight.uop.buffer
    except RuntimeError as exc:raise ValueError('Packed weights must be a materialized base allocation') from exc
    if weight_buffer.offset or weight_buffer.nbytes!=columns*depth*2 or weight_buffer.base.nbytes!=columns*depth*2:
      raise ValueError('Packed weight views are unsupported by this prototype')
    self.rows,self.columns,self.depth,self.tile=rows,columns,depth,tile
    self.weight=packed_weight
    self.npu=cast(RockchipDevice,Device['ROCKCHIP']);self.gpu=cast(CLDevice,Device['CL'])
    self.owned=[];self.shared=None;self.plan=None;self.unpack=None;self.output=None;self.closed=False
    try:
      self.lhs=self.npu.allocate(tile*depth*2);self.owned.append(self.lhs)
      self.raw=self.npu.allocate(tile*columns*4);self.owned.append(self.raw)
      self.shared=SharedNPUKernel(self.npu,(self.lhs,self.raw),source(rows,columns,depth,tile),'shared_pack',tile*depth)
      self.unpack=checked(cl.clCreateKernel(self.shared.program,b'shared_unpack',status:=ctypes.c_int32()),status)
      for i,mem in enumerate(self.shared.mem):check(cl.clSetKernelArg(self.unpack,i,ctypes.sizeof(mem),ctypes.byref(mem)))
      self.plan=RockchipMatmul(self.npu,TinyELF(struct.pack('<4I',tile,columns,depth,3),'shared_projection',Target('ROCKCHIP'),()))
      self.weight_buffer=weight_buffer._buf
      self.plan.prepare(self.raw,self.lhs,self.weight_buffer)
      self.output=Tensor.empty(rows,columns,device='CL',dtype=dtypes.float).realize()
    except BaseException:
      self.close();raise

  def __call__(self,x:Tensor)->Tensor:
    if self.closed or self.output is None or self.shared is None or self.plan is None or self.unpack is None:
      raise RuntimeError('Experimental shared projection is closed')
    if x.device!='CL' or x.dtype is not dtypes.float or x.shape!=(self.rows,self.depth) or not x.uop.is_realized:
      raise ValueError('Expected a realized contiguous GPU FP32 input')
    # The probe supplies materialized base matrices; views would need offset args.
    if not x.uop.has_buffer_identity(after_ok=True):
      raise ValueError('GPU input must own a materialized full-size buffer')
    try:input_buffer=x.uop.buffer
    except RuntimeError as exc:raise ValueError('GPU input views are unsupported by this prototype') from exc
    if input_buffer.offset or input_buffer.nbytes!=self.rows*self.depth*4 or input_buffer.base.nbytes!=self.rows*self.depth*4:
      raise ValueError('GPU input must own a materialized full-size buffer')
    src=input_buffer._buf;dst=self.output.uop.buffer._buf
    check(cl.clSetKernelArg(self.shared.kernel,3,ctypes.sizeof(src),ctypes.byref(src)))
    check(cl.clSetKernelArg(self.unpack,3,ctypes.sizeof(dst),ctypes.byref(dst)))
    assert self.shared.flag is not None
    for start in range(0,self.rows,self.tile):
      ctypes.c_uint32.from_address(self.shared.flag.addr).value=0
      self.shared.flag.meta.version+=1
      self.npu.sync(self.shared.flag,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      offset=ctypes.c_int32(start)
      check(cl.clSetKernelArg(self.shared.kernel,4,ctypes.sizeof(offset),ctypes.byref(offset)))
      check(cl.clEnqueueNDRangeKernel(self.gpu.queue,self.shared.kernel,1,None,
            (ctypes.c_size_t*1)(round_up(self.tile*self.depth,64)),(ctypes.c_size_t*1)(64),0,None,None))
      # This also completes the previous tile's unpack before NPU overwrites raw.
      self.gpu.synchronize()
      self.npu.sync(self.shared.flag,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      if ctypes.c_uint32.from_address(self.shared.flag.addr).value:
        raise ValueError('Shared projection requires finite FP16-representable activations')
      # Retain the runtime's finite scan without copying or touching these data
      # through a host Tensor. Invalidate its CPU alias before that read-only scan.
      self.npu.sync(self.lhs,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      self.plan(self.raw,self.lhs,self.weight_buffer)
      check(cl.clSetKernelArg(self.unpack,4,ctypes.sizeof(offset),ctypes.byref(offset)))
      check(cl.clEnqueueNDRangeKernel(self.gpu.queue,self.unpack,1,None,
            (ctypes.c_size_t*1)(round_up(self.tile*self.columns,64)),(ctypes.c_size_t*1)(64),0,None,None))
    self.gpu.synchronize()
    return self.output

  def close(self):
    if self.closed:return
    self.closed=True
    if self.shared is not None:self.gpu.synchronize()
    if self.plan is not None:self.plan.close();self.plan=None
    if self.unpack is not None:check(cl.clReleaseKernel(self.unpack));self.unpack=None
    if self.shared is not None:self.shared.close();self.shared=None
    for buf in reversed(self.owned):self.npu.free(buf)
    self.owned.clear();self.output=None
