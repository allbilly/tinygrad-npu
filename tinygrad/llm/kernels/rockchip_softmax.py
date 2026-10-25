"""GPU softmax over shared NPU GEM buffers with explicit blocking ownership transfers.

NPU submissions finish before GPU reads; clFinish completes GPU writes before PV.
CPU aliases use the existing RKNPU sync operations. Neither host nor device writes
an imported allocation while the other device owns it. Host consistency is managed
explicitly; OpenCL does not perform redundant whole-buffer CPU cache maintenance.
"""
from __future__ import annotations
import ctypes, fcntl, os, struct
from typing import cast
from tinygrad import Device
from tinygrad.helpers import to_char_p_p
from tinygrad.runtime.autogen import opencl as cl, rockchip as rk
from tinygrad.runtime.ops_cl import CLDevice, check, checked, BP_CB
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipDevice

SOURCE = '''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void shared_softmax(__global const float *scores, __global half *prob, __global uint *invalid) {
  const int index=get_global_id(0)/4, lane=get_global_id(0)%4, slot=get_local_id(0);
  const int head=index/1024, row=index%1024, base=head*1048576+(row/64)*65536;
  __local float scratch[64];
  float maximum=-INFINITY;
  for(int c=lane;c<1024;c+=4) {
    float value=scores[base+(c/4)*256+(row%64)*4+c%4]*.125f;
    if(!isfinite(value)) { atomic_or(invalid,1u);value=0; }
    maximum=fmax(maximum,value);
  }
  scratch[slot]=maximum;barrier(CLK_LOCAL_MEM_FENCE);
  int start=slot-lane;
  maximum=fmax(fmax(scratch[start],scratch[start+1]),fmax(scratch[start+2],scratch[start+3]));
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum=0;
  for(int c=lane;c<1024;c+=4) {
    float delta=scores[base+(c/4)*256+(row%64)*4+c%4]*.125f-maximum;
    sum+=bounded_exp(delta);
  }
  scratch[slot]=sum;barrier(CLK_LOCAL_MEM_FENCE);
  float inverse=1.f/(((scratch[start]+scratch[start+1])+scratch[start+2])+scratch[start+3]);
  for(int c=lane;c<1024;c+=4) {
    float delta=scores[base+(c/4)*256+(row%64)*4+c%4]*.125f-maximum;
    prob[base+(c/8)*512+(row%64)*8+c%8]=convert_half_rte(bounded_exp(delta)*inverse);
  }
}
'''
SOURCE = SOURCE.replace('__kernel void', '''static float bounded_exp(float delta) {
  if(delta< -20.f || !isfinite(delta))return 0.f;
  float x=delta*1.4426950216293335f;int q=convert_int_rtz(x-.5f);float s=x-convert_float(q);
  float p=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
    0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.f);
  return p*as_float((q+127)<<23);
}
__kernel void''')


class SharedNPUKernel:
  def __init__(self, npu:RockchipDevice, buffers:tuple[RockchipBuffer,...], source:str, name:str, work_items:int):
    self.npu, self.gpu = npu, cast(CLDevice,Device['CL'])
    self.work_items = work_items
    self.mem:list[cl.cl_mem] = []
    self.fds:list[int] = []
    self.program:cl.cl_program|None = None
    self.kernel:cl.cl_kernel|None = None
    self.flag:RockchipBuffer|None = None
    try:
      self.flag = self.npu.allocate(4096)
      self.initialize(buffers,self.flag,source,name)
    except BaseException:
      self.close()
      raise

  def initialize(self, buffers:tuple[RockchipBuffer,...], flag:RockchipBuffer, source:str, name:str):
    if 'cl_arm_import_memory_dma_buf' not in self.gpu.device_exts: raise RuntimeError('DMA-BUF import unavailable')
    self.npu.sync(flag,rk.RKNPU_MEM_SYNC_TO_DEVICE)
    address = cl.clGetExtensionFunctionAddress(b'clImportMemoryARM')
    if not address: raise RuntimeError('clImportMemoryARM unavailable')
    importer = ctypes.CFUNCTYPE(cl.cl_mem,cl.cl_context,cl.cl_mem_flags,ctypes.POINTER(ctypes.c_int64),
                               ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32))(address)
    for buf in (*buffers,flag):
      self.npu.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      args = bytearray(struct.pack('IIi',buf.meta.handle,os.O_CLOEXEC|os.O_RDWR,-1))
      fcntl.ioctl(self.npu.fd.fd,(3<<30)|(12<<16)|(ord('d')<<8)|0x2d,args,True)
      fd = ctypes.c_int32.from_buffer(args,8)
      self.fds.append(fd.value)
      properties = (ctypes.c_int64*5)(0x40b2,0x40b4,0x41e3,0,0)
      self.mem.append(checked(importer(self.gpu.context,cl.CL_MEM_READ_WRITE,properties,ctypes.byref(fd),
                                      buf.size,status:=ctypes.c_int32()),status))
    self.program = checked(cl.clCreateProgramWithSource(self.gpu.context,1,to_char_p_p([source.encode()]),None,
                                                       status:=ctypes.c_int32()),status)
    result = cl.clBuildProgram(self.program,1,self.gpu.cl_dev,None,BP_CB(),None)
    if result:
      log = ctypes.create_string_buffer(16384)
      cl.clGetProgramBuildInfo(self.program,self.gpu.cl_dev,cl.CL_PROGRAM_BUILD_LOG,len(log),log,None)
      raise RuntimeError(log.value.decode())
    self.kernel = checked(cl.clCreateKernel(self.program,name.encode(),status:=ctypes.c_int32()),status)
    for i,mem in enumerate(self.mem): check(cl.clSetKernelArg(self.kernel,i,ctypes.sizeof(mem),ctypes.byref(mem)))

  def bridge(self, name:str):
    assert self.flag is not None and self.kernel is not None
    def address(value): return ctypes.cast(value,ctypes.c_void_p).value
    enqueue, finish = address(cl.dll.clEnqueueNDRangeKernel),address(cl.dll.clFinish)
    queue, kernel = address(self.gpu.queue),address(self.kernel)
    sync_request = (3<<30)|(ctypes.sizeof(rk.struct_rknpu_mem_sync)<<16)|(ord('d')<<8)|0x45
    return f'''static int {name}(void){{
      *(unsigned*){self.flag.addr}ULL=0;
      if(sync_mem({self.flag.meta.obj_addr}ULL,{self.flag.size}ULL,1))return 1;
      unsigned long global[1]={{{self.work_items}}},local[1]={{64}};
      int result=((int(*)(void*,void*,unsigned,const void*,const void*,const void*,unsigned,void*,void*)){enqueue}ULL)
        ((void*){queue}ULL,(void*){kernel}ULL,1,0,global,local,0,0,0);
      if(!result)result=((int(*)(void*)){finish}ULL)((void*){queue}ULL);
      if(result){{status[2]=(U64)result;return 1;}}
      U64 args[4]={{2,{self.flag.meta.obj_addr}ULL,0,{self.flag.size}ULL}};
      if(request({sync_request}UL,args))return 1;
      if(*(unsigned*){self.flag.addr}ULL){{status[2]=1;return 1;}}
      return 0;
    }}'''

  def close(self):
    if self.kernel is not None:
      check(cl.clReleaseKernel(self.kernel))
      self.kernel=None
    if self.program is not None:
      check(cl.clReleaseProgram(self.program))
      self.program=None
    for mem in reversed(self.mem): check(cl.clReleaseMemObject(mem))
    self.mem.clear()
    for fd in self.fds: os.close(fd)
    self.fds.clear()
    if self.flag is not None:
      self.npu.free(self.flag)
      self.flag=None


class SharedNPUVisionSoftmax(SharedNPUKernel):
  def __init__(self, npu:RockchipDevice, scores:RockchipBuffer, probabilities:RockchipBuffer):
    self.heads = scores.size//(1024*1024*4)
    if self.heads not in (3,12) or scores.size!=self.heads*1024*1024*4 or probabilities.size!=self.heads*1024*1024*2:
      raise ValueError('Shared softmax requires 3 or 12 1024-token attention heads')
    super().__init__(npu,(scores,probabilities),SOURCE,'shared_softmax',self.heads*1024*4)

  def bridge(self, name:str='shared_softmax'): return super().bridge(name)
