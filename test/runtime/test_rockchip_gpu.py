import ctypes, os, struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.llm.kernels.rockchip_gpu import gpu_ffn_pack
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipDevice
from tinygrad.runtime.support import rockchip_gpu as gpu


class TestRockchipGPUViews(unittest.TestCase):
  @staticmethod
  def buffer(*, offset=0, count=64, size=128, meta_size=128, flags=None, addr=None, dma_offset=None):
    base_flags=(rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    if dma_offset is None:dma_offset=offset
    meta=SimpleNamespace(dma_addr=0x20000,flags=base_flags if flags is None else flags,size=meta_size)
    base_addr=0x40000
    buf=RockchipBuffer(base_addr+offset if addr is None else addr,meta.dma_addr+dma_offset,size,meta)
    return buf,count

  def test_view_accepts_valid_aligned_base_and_subviews(self):
    for offset in (0,4,64):
      buf,count=self.buffer(offset=offset,count=64,size=128,meta_size=128)
      with self.subTest(offset=offset):self.assertEqual(gpu._view(buf,count),offset)

  def test_view_rejects_nonpositive_count(self):
    buf,_=self.buffer(offset=0,size=128,meta_size=128)
    for count in (0,-1):
      with self.subTest(count=count),self.assertRaises(ValueError):gpu._view(buf,count)

  def test_view_rejects_exact_flag_mismatch(self):
    flags=(rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_NON_CONTIGUOUS|
           rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT|rk.RKNPU_MEM_KERNEL_MAPPING)
    buf,count=self.buffer(flags=flags)
    with self.assertRaises(ValueError):gpu._view(buf,count)

  def test_view_rejects_bad_offset_and_bounds(self):
    cases=(self.buffer(offset=-4,dma_offset=-4),
           self.buffer(offset=4,dma_offset=8),
           self.buffer(offset=68,count=64,size=128,meta_size=128),
           self.buffer(offset=4,count=132,size=128,meta_size=256))
    for buf,count in cases:
      with self.subTest(buf=buf,count=count),self.assertRaises(ValueError):gpu._view(buf,count)

  def test_view_rejects_unaligned_base_or_offset(self):
    unaligned_base,count=self.buffer(offset=0,addr=0x40004,dma_offset=0)
    unaligned_offset,count2=self.buffer(offset=1,dma_offset=1)
    for buf,size in ((unaligned_base,count),(unaligned_offset,count2)):
      with self.subTest(buf=buf),self.assertRaises(ValueError):gpu._view(buf,size)


class TestRockchipGPUDispatch(unittest.TestCase):
  @staticmethod
  def descriptor(rows=128,depth=3072,tile=64,magic=gpu.GPU_FFN_MAGIC):
    return TinyELF(struct.pack('<4I',rows,depth,tile,magic),'gpu_ffn',Target('ROCKCHIP'),())

  def test_runtime_routes_gpu_descriptor_without_device_initialization(self):
    dev=RockchipDevice.__new__(RockchipDevice)
    expected=object()
    with patch('tinygrad.runtime.support.rockchip_gpu.RockchipGPUProgram',return_value=expected) as factory:
      self.assertIs(dev.runtime(self.descriptor()),expected)
      factory.assert_called_once_with(dev,self.descriptor())

  def test_gpu_program_rejects_unqualified_descriptors_before_opencl(self):
    invalid=(self.descriptor(rows=127),self.descriptor(rows=129),self.descriptor(depth=3071),
             self.descriptor(tile=32),self.descriptor(magic=0),
             TinyELF(self.descriptor().lib+b'\0','bad_length',Target('ROCKCHIP'),()))
    with patch('tinygrad.runtime.support.rockchip_gpu.Device',side_effect=AssertionError('OpenCL initialized')):
      for obj in invalid:
        with self.subTest(lib=obj.lib),self.assertRaises(ValueError):
          gpu.RockchipGPUProgram.__new__(gpu.RockchipGPUProgram).__init__(None,obj)

  def test_gpu_ffn_pack_rejects_bad_operands_before_allocation(self):
    good_shape=128*6144
    raw_dtype=Tensor.empty(good_shape,device='CPU',dtype=dtypes.float)
    raw_count=Tensor.empty(1,device='CPU',dtype=dtypes.int32)
    raw_device=Tensor.empty(good_shape,device='CPU',dtype=dtypes.int32)
    raw_good=Tensor.empty(good_shape,device='CPU',dtype=dtypes.int32)
    payload_dtype=Tensor.empty(512,device='CPU',dtype=dtypes.int32)
    payload_small=Tensor.empty(1,device='CPU',dtype=dtypes.uint8)
    payload_good=Tensor.empty(512,device='CPU',dtype=dtypes.uint8)
    scales_shape=Tensor.empty(6143,device='CPU',dtype=dtypes.float)
    scales_dtype=Tensor.empty(6144,device='CPU',dtype=dtypes.half)
    scales_good=Tensor.empty(6144,device='CPU',dtype=dtypes.float)
    cases=((raw_dtype,payload_good,scales_good,'ROCKCHIP'),(raw_count,payload_good,scales_good,'ROCKCHIP'),
           (raw_device,payload_good,scales_good,'ROCKCHIP'),(raw_good,payload_dtype,scales_good,'ROCKCHIP'),
           (raw_good,payload_small,scales_good,'ROCKCHIP'),(raw_good,payload_good,scales_shape,'ROCKCHIP'),
           (raw_good,payload_good,scales_dtype,'ROCKCHIP'),(raw_good,payload_good,scales_good,'CPU'))
    with patch.object(Tensor,'empty',side_effect=AssertionError('output allocation attempted')):
      for raw,payload,scales,device in cases:
        with self.subTest(raw=(raw.dtype,raw.numel(),raw.device),payload=(payload.dtype,payload.numel()),device=device), \
             self.assertRaises(ValueError):gpu_ffn_pack(raw,payload,scales,device)


class TestGPUFFNEmission(unittest.TestCase):
  def test_emit_sets_all_dma_buffers_and_offsets_without_host_transfers(self):
    plan=gpu.GPUFFNPlan.__new__(gpu.GPUFFNPlan)
    plan.program=SimpleNamespace(gpu=SimpleNamespace(queue=ctypes.c_void_p(101)),
                                 kernel=SimpleNamespace(kernel=ctypes.c_void_p(202)))
    plan.mem=[ctypes.c_void_p(i+1) for i in range(6)]
    plan.offsets=(0,64,128,192,256,320)
    names=('clSetKernelArg','clEnqueueNDRangeKernel','clFinish')
    with patch.object(gpu,'_api',side_effect=lambda name,signature:f'{name}_MOCK') as api:
      code=plan.emit(4000042)
    self.assertEqual(api.call_count,len(names))
    self.assertEqual({call.args[0] for call in api.call_args_list},set(names))
    self.assertIn('((void*)101ULL)',code)
    self.assertIn('clSetKernelArg_MOCK((void*)202ULL,',code)
    for i,offset in enumerate(plan.offsets,6):self.assertIn(f'gpu_offset{i}={offset}UL',code)
    self.assertNotIn('clEnqueueWriteBuffer',code)
    self.assertNotIn('clEnqueueReadBuffer',code)
    sequence=('clSetKernelArg_MOCK','clEnqueueNDRangeKernel_MOCK','clFinish_MOCK')
    positions=[code.index(name) for name in sequence]
    self.assertEqual(positions,sorted(positions))
    self.assertIn('return 4000042;',code)


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestGPUFFNChain(unittest.TestCase):
  def test_changed_inputs_and_partial_rows_match_cpu_packing_and_integer_down_projection(self):
    import numpy as np
    from tinygrad import TinyJit
    from tinygrad.llm.kernels.rockchip_int8 import ffn_pack, raw_int8_matmul
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng = np.random.default_rng(20261028)
    # Sparse weights make the independent integer reference cheap while exercising
    # every channel of the real, full-size three-core matrix commands.
    indices = np.arange(6144)%1024
    signs = np.where(np.arange(6144)%3,1,-1).astype(np.int8)
    up = np.zeros((6144,1024),dtype=np.int8)
    up[np.arange(6144),indices] = signs
    down_indices = np.arange(1024)*3%3072
    down_signs = np.where(np.arange(1024)%2,1,-1).astype(np.int8)
    down = np.zeros((1024,3072),dtype=np.int8)
    down[np.arange(1024),down_indices] = down_signs
    def packed_weight(values):
      n,k = values.shape
      return Tensor(values.reshape(n//32,32,k//32,32).transpose(0,2,1,3).copy(),device='ROCKCHIP').realize()
    up_weight,down_weight = packed_weight(up),packed_weight(down)
    weight_scales = Tensor(rng.uniform(.01,.03,6144).astype(np.float32),device='CPU').realize()
    def chain(payload):
      npu_payload = payload.to('ROCKCHIP')
      raw = raw_int8_matmul(npu_payload,up_weight,128,6144,cpu_output=False)
      packed,scales = gpu_ffn_pack(raw,npu_payload,weight_scales,'ROCKCHIP')
      outputs = [raw_int8_matmul(tile,down_weight,64,1024) for tile in packed]
      results = (*(tile.to('CPU') for tile in packed),scales,*outputs)
      Tensor.realize(*results)
      return results
    run = TinyJit(chain)
    for step,active in enumerate((128,128,17,128,17,128,128,17)):
      values = rng.integers(-127,128,(128,1024),dtype=np.int8)
      values[active:] = 0
      if step==5:values[:] = 0
      sa = rng.uniform(.01,.03,128).astype(np.float32)
      payload = np.concatenate((sa.view(np.uint8),values.ravel().view(np.uint8)))
      projected = values.astype(np.int32)[:,indices]*signs.astype(np.int32)
      projected = np.concatenate([projected[:,first:first+2048].ravel() for first in (0,2048,4096)])
      expected,expected_scales = ffn_pack(Tensor(projected,device='CPU'),Tensor(sa,device='CPU'),weight_scales,128,3072)
      expected_bytes = expected.numpy().copy()
      expected_scales_array = expected_scales.numpy().copy()
      actual = run(Tensor(payload,device='CPU').realize())
      for tile in range(2):np.testing.assert_array_equal(actual[tile].numpy(),expected_bytes[tile*196864:(tile+1)*196864])
      np.testing.assert_array_equal(actual[2].numpy().view(np.uint32),expected_scales_array.view(np.uint32))
      for tile,out in enumerate(actual[3:]):
        integers = expected_bytes[tile*196864+256:(tile+1)*196864].view(np.int8).reshape(64,3072)
        matrix = integers.astype(np.int32)[:,down_indices]*down_signs.astype(np.int32)
        # Each core starts at the next multiple of 32 channels; the last core
        # handles the remaining 320 columns without padding the output.
        reference = np.concatenate([matrix[:,first:first+352].ravel() for first in (0,352,704)])
        np.testing.assert_array_equal(out.numpy(),reference)
      if step==1:run.captured = RockchipCapturedJit(run.captured)
    self.assertIsNone(run.captured.reason)
    self.assertGreater(run.captured.native_runs,0)
    self.assertEqual(run.captured.native.gpu_calls,1)


if __name__=='__main__':unittest.main()
