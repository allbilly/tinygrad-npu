"""Prepared zero-copy Mali FFN between existing Rockchip matrix projections."""

from __future__ import annotations
import contextlib, ctypes, fcntl, os, struct, time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast
from tinygrad import Device
from tinygrad.device import Program, TinyELF
from tinygrad.helpers import getenv, unwrap
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.ops_cl import CLDevice, CLProgram, check
from tinygrad.runtime.ops_rockchip import RockchipBuffer

if TYPE_CHECKING:
  from tinygrad.runtime.ops_rockchip import RockchipDevice

GPU_FFN_MAGIC = 0x4750464E
GPU_FFN_SOURCE = r"""
#pragma OPENCL FP_CONTRACT OFF
#define DEPTH 3072
#define LS 256
float4 poly_exp2(float4 x) {

 int4 nan=isnan(x);
float4 safe=select(x,
(float4)0,
isinf(x)|nan);

 int4 q=convert_int4_rtz(safe+select((float4).5f,
(float4)-.5f,
safe<0)),
q1=q>>1;

 float4 s=safe-convert_float4(q);

 float4 y=fma((float4)0.00015359208919107914f,
s,
(float4)0.0013392627006396651f);

 y=fma(y,
s,
(float4)0.009618384763598442f);
y=fma(y,
s,
(float4)0.055503472685813904f);

 y=fma(y,
s,
(float4)0.24022644758224487f);
y=fma(y,
s,
(float4)0.6931471824645996f);
y=fma(y,
s,
(float4)1.0f);

 y=y*as_float4((q1+127)<<23);
y=y*as_float4((q-q1+127)<<23);

 y=select(y,
(float4)INFINITY,
x>=128);
y=select(y,
(float4)0,
x<-150);
return select(y,
(float4)NAN,
nan);


}
float4 approximate_div(float4 n,
float4 d) {

 if(any((n==0)|isinf(n)|isnan(n)|isinf(d)|isnan(d)|(d>0x1p100f)))return n/d;

 float4 r=1.0f/d;
r=fma(r,
fma(-d,
r,
(float4)1.0f),
r);

 float4 q=n*r;
return fma(fma(-q,
d,
n),
r,
q);


}
// The device divide can leave the initial estimate two ULPs away. Resolve
// residuals against each adjacent-float midpoint before quantizing the FFN.
float4 corrected_div(float4 numerator, float4 denominator) {
 float4 magnitude=fabs(numerator),q=approximate_div(magnitude,denominator);
 for(int iteration=0;iteration<3;iteration++) {
  uint4 bits=as_uint4(q),exponent=bits&(uint4)0x7f800000u;
  int4 safe=(exponent>(uint4)(24u<<23))&(exponent<(uint4)(253u<<23))&isfinite(denominator);
  float4 half_ulp=as_float4(exponent-(uint4)(24u<<23));
  float4 upper=half_ulp*denominator;
  float4 lower=select(upper,upper*.5f,(bits&(uint4)0x7fffffu)==0);
  float4 residual=fma(-q,denominator,magnitude);
  int4 odd=(bits&(uint4)1u)!=0;
  int4 increase=safe&((residual>upper)|((residual==upper)&odd));
  int4 decrease=safe&((residual< -lower)|((residual== -lower)&odd));
  int4 upper_tie=safe&(residual==upper),lower_tie=safe&(residual== -lower);
  if(any(upper_tie|lower_tie)) {
   // A rounded FMA residual can equal a midpoint while its exact value lies
   // on either side. Compare integer significands only in this rare case.
   uint4 nb=as_uint4(magnitude),db=as_uint4(denominator);
   uint4 nm=(nb&(uint4)0x7fffffu)|(uint4)0x800000u;
   uint4 dm=(db&(uint4)0x7fffffu)|(uint4)0x800000u;
   uint4 qm=(bits&(uint4)0x7fffffu)|(uint4)0x800000u;
   uint4 shift=convert_uint4(convert_int4(nb>>23)-convert_int4(bits>>23)-convert_int4(db>>23)+151);
   ulong4 lhs=convert_ulong4(nm)<<convert_ulong4(shift&63u);
   ulong4 bound=convert_ulong4(qm*2u+1u)*convert_ulong4(dm);
   int4 above=convert_int4((lhs>bound)|((lhs==bound)&convert_long4(odd)));
   increase=select(increase,above,upper_tie);
   int4 boundary=(qm==(uint4)0x800000u);
   ulong4 lower_lhs=select(lhs,lhs<<1,convert_long4(boundary));
   uint4 lower_m=select(qm*2u-1u,qm*4u-1u,boundary);
   bound=convert_ulong4(lower_m)*convert_ulong4(dm);
   int4 below=convert_int4((lower_lhs<bound)|((lower_lhs==bound)&convert_long4(odd)));
   decrease=select(decrease,below,lower_tie);
  }
  bits+=convert_uint4(increase&1);bits-=convert_uint4(decrease&1);
  q=as_float4(bits);
  if(!any(increase|decrease))break;
 }
 return as_float4(as_uint4(q)|(as_uint4(numerator)&(uint4)0x80000000u));
}
float exact_scale(float maximum) {

 if(maximum<0x1p-20f)return 1e-8f;

 uint bits=as_uint(maximum),
m=(bits&0x7fffff)|0x800000,
shift=m>=16646144?6:7;

 uint numerator=m<<shift,
q=numerator/127,
r=numerator-q*127;
q+=(r>=64);

 float result=as_float((((bits>>23)-shift)<<23)|(q-0x800000));
return max(result,
1e-8f);


}
float exact_recip(float scale) {

 uint bits=as_uint(scale),
m=(bits&0x7fffff)|0x800000,
exponent=bits>>23;

 if(m==0x800000)return as_float((254-exponent)<<23);

 ulong numerator=((ulong)1)<<47;

 uint q=convert_uint_rtz((1.0f/(float)m)*0x1p47f);
ulong product=(ulong)q*m;

 while(product>numerator){
q--;
product-=m;

}
 while(numerator-product>=m){
q++;
product+=m;

}
 uint r=(uint)(numerator-product);

 q+=(r*2>m)||((r*2==m)&&(q&1));
return as_float(((253-exponent)<<23)|(q-0x800000));


}
float4 project(__global const int *raw,
__global const float *sa,
__global const float *sw,
int row,
int col) {

 int plane=col/2048,
within=col%2048,
index=plane*128*2048+row*2048+within;

 return (convert_float4(vload4(0,
raw+index))*(float4)sa[row])*vload4(0,
sw+col);


}
__kernel void ffn_gpu(__global uchar *out_base,
__global const int *raw_base,
__global const float *sa_base,
__global const float *sw_base,
__global float *scale_out_base,
__global uchar *second_base,
ulong out_offset,
ulong raw_offset,
ulong sa_offset,
ulong sw_offset,
ulong scale_out_offset,
ulong second_offset) {

 __global uchar *out=get_group_id(0)<64?out_base+out_offset:second_base+second_offset;

 __global const int *raw=raw_base+raw_offset/4;

 __global const float *sa=sa_base+sa_offset/4;
 __global const float *sw=sw_base+sw_offset/4;
 __global float *scale_out=scale_out_base+scale_out_offset/4;

 int row=get_group_id(0),
lid=get_local_id(0);

 __local float values[DEPTH];
__local uint maxima[LS],
invalid[LS];
__local float scale,
inverse;
__local uint bad;

 uint maximum=0,
bad_lane=0;

 for(int col=lid*4;
col<DEPTH;
col+=LS*4) {

  float4 gate=project(raw,
sa,
sw,
row,
col),
up=project(raw,
sa,
sw,
row,
col+DEPTH);

  float4 value=corrected_div(gate,
1.0f+poly_exp2(gate*-1.4426950216293335f))*up;

  vstore4(value,
col/4,
values);
uint4 b=as_uint4(value)&(uint4)0x7fffffff;

  bad_lane|=any(b>=(uint4)0x7f800000);
maximum=max(maximum,
max(max(b.s0,
b.s1),
max(b.s2,
b.s3)));


}
 maxima[lid]=maximum;
invalid[lid]=bad_lane;
barrier(CLK_LOCAL_MEM_FENCE);

 for(int step=LS/2;
step>0;
step/=2) {
if(lid<step){
maxima[lid]=max(maxima[lid],
maxima[lid+step]);
invalid[lid]|=invalid[lid+step];

}barrier(CLK_LOCAL_MEM_FENCE);

}
 int base=0,
offset=row%64;

 if(lid==0) {
bad=invalid[0];
scale=bad?as_float(0x7fc00000u):exact_scale(as_float(maxima[0]));
inverse=bad?0:exact_recip(scale);
scale_out[row]=scale;
*(__global float*)(out+base+offset*4)=scale;

}
 barrier(CLK_LOCAL_MEM_FENCE);

 for(int col=lid*4;
col<DEPTH;
col+=LS*4) {

  float4 value=bad?(float4)0:vload4(col/4,
values);
float4 q=clamp(rint(value*inverse),
(float4)-127,
(float4)127);

  vstore4(convert_char4(q),
0,
(__global char*)(out+base+256+offset*DEPTH+col));


}

}
"""


def _ptr(value: Any) -> int:
  return unwrap(ctypes.cast(value, ctypes.c_void_p).value)


def _api(name: str, signature: str) -> str:
  return f"(({signature}){_ptr(getattr(cl.dll, name))}ULL)"


def _view(buf: RockchipBuffer, count: int) -> int:
  from tinygrad.runtime.autogen import rockchip as rk

  flags = rk.RKNPU_MEM_CACHEABLE | rk.RKNPU_MEM_NON_CONTIGUOUS | rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
  offset = buf.dma - int(buf.meta.dma_addr)
  if (
    count <= 0 or buf.meta.flags != flags or not 0 <= offset <= buf.meta.size - count or count > buf.size or (buf.addr - offset) & 4095 or offset & 3
  ):
    raise ValueError("Unsupported GPU FFN mapping or view")
  return offset


class GPUFFNPlan:
  def __init__(self, program: RockchipGPUProgram,
               bufs: tuple[RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer]):
    self.dev, self.program = program.dev, program
    self.bufs: tuple[RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer, RockchipBuffer] | tuple[()] = bufs
    self.owned: list[Any] = []
    self.fds: list[int] = []
    out, raw, payload, weights, scales, second = bufs
    self.offsets = tuple(_view(buf,size) for buf,size in zip(bufs,(196864,3145728,512,24576,512,196864)))
    outputs,inputs = ((out,196864),(scales,512),(second,196864)),((raw,3145728),(payload,512),(weights,24576))
    def overlaps(x:RockchipBuffer, xn:int, y:RockchipBuffer, yn:int) -> bool:
      return x.meta is y.meta and x.addr<y.addr+yn and y.addr<x.addr+xn
    if (any(overlaps(x,xn,y,yn) for x,xn in outputs for y,yn in inputs)
        or any(overlaps(x,xn,y,yn) for i,(x,xn) in enumerate(outputs) for y,yn in outputs[i+1:])):
      raise ValueError("GPU FFN outputs overlap inputs")
    try:
      self.mem = [self.import_dma(buf) for buf in bufs]
      self.dev.programs.add(self)
    except BaseException:
      self.close()
      raise

  def import_dma(self, buf: RockchipBuffer):
    request = (3 << 30) | (12 << 16) | (ord("d") << 8) | 0x2D
    data = bytearray(struct.pack("IIi", buf.meta.handle, os.O_CLOEXEC | os.O_RDWR, -1))
    fcntl.ioctl(self.dev.fd.fd, request, data, True)
    fd = struct.unpack("IIi", data)[2]
    if fd < 0:
      raise OSError("NPU DMA-buffer export failed")
    self.fds.append(fd)
    properties = (ctypes.c_ssize_t * 5)(0x40B2, 0x40B4, 0x41E3, 0, 0)
    status = ctypes.c_int32()
    value = ctypes.c_int32(fd)
    handle = self.program.importer(
      ctypes.cast(self.program.gpu.context, ctypes.c_void_p),
      cl.CL_MEM_READ_WRITE | cl.CL_MEM_HOST_NO_ACCESS,
      properties,
      ctypes.byref(value),
      buf.meta.size,
      ctypes.byref(status),
    )
    check(status.value)
    mem = ctypes.cast(handle, cl.cl_mem)
    self.owned.append(mem)
    return mem

  def close(self):
    if not self.owned and not self.fds:
      return
    check(cl.clFinish(self.program.gpu.queue))
    for mem in self.owned:
      check(cl.clReleaseMemObject(mem))
    self.owned.clear()
    for fd in self.fds:
      os.close(fd)
    self.fds.clear()
    self.bufs = ()

  def __del__(self):
    with contextlib.suppress(Exception):
      self.close()

  def invoke(self):
    if len(self.bufs) != 6:
      raise RuntimeError("GPU FFN plan is closed")
    out, raw, _, _, scales, second = self.bufs
    for buf in (out,scales,second):
      if buf.meta.cpu_clean_version != buf.meta.version:
        self.dev.sync(buf, self.program.rk_to)
    self.dev.sync(raw, self.program.rk_from)
    kernel = self.program.kernel.kernel
    queue = self.program.gpu.queue
    for index, mem in enumerate(self.mem):
      check(cl.clSetKernelArg(kernel, index, ctypes.sizeof(mem), ctypes.byref(mem)))
    for index, offset in enumerate(self.offsets, 6):
      value = ctypes.c_uint64(offset)
      check(cl.clSetKernelArg(kernel, index, 8, ctypes.byref(value)))
    check(cl.clEnqueueNDRangeKernel(queue, kernel, 1, None, (ctypes.c_size_t * 1)(32768), (ctypes.c_size_t * 1)(256), 0, None, None))
    check(cl.clFinish(queue))
    for buf in (out,scales,second):
      buf.meta.version += 1
      self.dev.sync(buf, self.program.rk_from)
      buf.meta.cpu_clean_version = buf.meta.version

  def emit(self, error: int) -> str:
    queue, kernel = _ptr(self.program.gpu.queue), _ptr(self.program.kernel.kernel)
    setarg = _api("clSetKernelArg", "int(*)(void*,unsigned,unsigned long,const void*)")
    enqueue = _api("clEnqueueNDRangeKernel", "int(*)(void*,void*,unsigned,const void*,const void*,const void*,unsigned,void*,void*)")
    finish = _api("clFinish", "int(*)(void*)")
    declarations = []
    checks = []
    for i, mem in enumerate(self.mem):
      declarations.append(f"void *gpu_arg{i}=(void*){_ptr(mem)}ULL;")
      checks.append(f"{setarg}((void*){kernel}ULL,{i}U,sizeof(void*),&gpu_arg{i})")
    for i, offset in enumerate(self.offsets, 6):
      declarations.append(f"unsigned long gpu_offset{i}={offset}UL;")
      checks.append(f"{setarg}((void*){kernel}ULL,{i}U,8UL,&gpu_offset{i})")
    declarations.append("unsigned long gpu_global[1]={32768UL},gpu_local[1]={256UL};")
    checks += [
      f"{enqueue}((void*){queue}ULL,(void*){kernel}ULL,1U,0,gpu_global,gpu_local,0U,0,0)",
      f"{finish}((void*){queue}ULL)",
    ]
    return "{" + "".join(declarations) + "if(" + "||".join(checks) + f")return {error};" + "}"


class RockchipGPUProgram(Program["RockchipDevice"]):
  def __init__(self, dev: RockchipDevice, obj: TinyELF):
    if obj.lib != struct.pack("<4I", 128, 3072, 64, GPU_FFN_MAGIC):
      raise ValueError("Unqualified GPU FFN descriptor")
    from tinygrad.runtime.autogen import rockchip as rk

    self.dev, self.obj = dev, obj
    self.rk_to, self.rk_from = rk.RKNPU_MEM_SYNC_TO_DEVICE, rk.RKNPU_MEM_SYNC_FROM_DEVICE
    self.gpu = cast(CLDevice, Device["CL"])
    if "cl_arm_import_memory_dma_buf" not in self.gpu.device_exts:
      raise RuntimeError("Mali DMA-buffer import is unavailable")
    self.importer = ctypes.CFUNCTYPE(
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)
    )(_ptr(cl.clGetExtensionFunctionAddress(b"clImportMemoryARM")))
    self.kernel = CLProgram(self.gpu, TinyELF(GPU_FFN_SOURCE.encode(), "ffn_gpu", self.gpu.renderer.target, ()))
    self.bindings: OrderedDict[tuple, GPUFFNPlan] = OrderedDict()
    self.binding_limit = getenv("ROCKCHIP_BINDING_CACHE", 256)
    if not 1 <= self.binding_limit <= 8192:
      raise ValueError("Invalid GPU FFN binding limit")

  @staticmethod
  def key(args: tuple) -> tuple:
    return tuple((id(x.meta), x.addr, x.size) if isinstance(x, RockchipBuffer) else int(x) for x in args)

  def __call__(self, *args: RockchipBuffer, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=False, timeout=None):
    if len(args) != 6 or not all(isinstance(x, RockchipBuffer) for x in args):
      raise ValueError("Invalid GPU FFN operands")
    bufs = args
    key = self.key(bufs)
    with self.dev.lock:
      self.dev.check_healthy()
      if key not in self.bindings or not self.bindings[key].owned:
        self.bindings[key] = GPUFFNPlan(self, bufs)
      self.bindings.move_to_end(key)
      if len(self.bindings) > self.binding_limit:
        self.bindings.popitem(last=False)
      begin = time.perf_counter()
      self.bindings[key].invoke()
      return time.perf_counter() - begin if wait else None
