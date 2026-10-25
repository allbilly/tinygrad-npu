import gc, mmap, os, tempfile, unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_shard


class TestGGUFStream(unittest.TestCase):
  @unittest.skipUnless(hasattr(mmap.mmap,'madvise') and hasattr(mmap,'MADV_DONTNEED'),'file mapping advice is unavailable')
  def test_completed_file_pages_are_released_without_changing_values_or_file(self):
    advised = []
    class ObservedMap(mmap.mmap):
      def madvise(self,advice,*args):
        if advice==mmap.MADV_DONTNEED:advised.append(advice)
        return super().madvise(advice,*args)
    quant = (np.float16(.5).tobytes()+bytes(range(32)))*5
    native = np.array([-1.5,0,2.25],dtype=np.float16)
    payload = quant+native.tobytes()
    with tempfile.TemporaryDirectory() as directory,patch('tinygrad.runtime.ops_disk.mmap.mmap',ObservedMap):
      path = Path(directory)/'weights.bin'
      path.write_bytes(payload)
      source = Tensor(path)
      entries = {'q':(source[:len(quant)],(5,32),8),'native':(source[len(quant):],native.shape,1)}
      result = gguf_shard(entries,('CPU',),stream=True,cast_dtypes={'q':dtypes.half},chunk_size=64)
      self.assertEqual(advised,[mmap.MADV_DONTNEED]*2)
      np.testing.assert_array_equal(result['q'].numpy(),np.tile(np.arange(32,dtype=np.float16)*.5,(5,1)))
      np.testing.assert_array_equal(result['native'].numpy(),native)
      self.assertEqual(path.read_bytes(),payload)

  def test_disk_q8_chunks_tail_and_native_dtype_preservation(self):
    scales = np.array([.125,.25,.5,1.,2.],dtype=np.float16)
    values = np.arange(-80,80,dtype=np.int8).reshape(5,32)
    packed = b''.join(scale.tobytes()+row.tobytes() for scale,row in zip(scales,values))
    norm = np.array([1.1234567,2.3456789,3.8765432],dtype=np.float32)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'weights.bin'
      path.write_bytes(packed+norm.tobytes())
      source = Tensor(path)
      entries = {'q':(source[:len(packed)],(5,32),8),'rope_factors_short.weight':(source[len(packed):],(3,),0)}
      with patch('tinygrad.llm.gguf.ggml_data_to_tensor',wraps=ggml_data_to_tensor) as decode:
        streamed = gguf_shard(entries,('CPU',),stream=True,cast_dtypes={'q':dtypes.half},chunk_size=64)
      expected = (scales.astype(np.float32)[:,None]*values).astype(np.float16)
      np.testing.assert_array_equal(streamed['q'].numpy(),expected)
      np.testing.assert_array_equal(streamed['rope_factors_short.weight'].numpy(),norm)
      self.assertEqual(streamed['q'].dtype,dtypes.half)
      self.assertEqual(streamed['rope_factors_short.weight'].dtype,dtypes.float32)
      q_calls = [call.args for call in decode.call_args_list if call.args[2]==8]
      self.assertEqual([args[1] for args in q_calls],[64,64,32])
      self.assertTrue(all(args[0].numel()<=68 for args in q_calls))
      ordinary = gguf_shard(entries,('CPU',),cast_dtypes={'q':dtypes.half})
      for name in entries:np.testing.assert_array_equal(streamed[name].numpy(),ordinary[name].numpy())

  def test_native_half_and_bfloat16_chunks_keep_exact_values(self):
    half = np.linspace(-3,3,195,dtype=np.float16)
    values = np.linspace(-2,2,133,dtype=np.float32)
    bf16 = (values.view(np.uint32)>>16).astype(np.uint16)
    expected = (bf16.astype(np.uint32)<<16).view(np.float32)
    entries = {'half':(Tensor(half.view(np.uint8)),half.shape,1),'bf16':(Tensor(bf16.view(np.uint8)),bf16.shape,30)}
    result = gguf_shard(entries,('CPU',),stream=True,cast_dtypes={'bf16':dtypes.half},chunk_size=64)
    np.testing.assert_array_equal(result['half'].numpy(),half)
    np.testing.assert_array_equal(result['bf16'].numpy(),expected.astype(np.float16))

  def test_stream_refuses_invalid_storage_and_tensor_parallel(self):
    tensor = Tensor.zeros(34,dtype=dtypes.uint8,device='CPU')
    with self.assertRaisesRegex(ValueError,'one device'):gguf_shard({'q':(tensor,(32,),8)},('CPU','CPU:1'),stream=True)
    with self.assertRaisesRegex(ValueError,'positive'):gguf_shard({},('CPU',),stream=True,chunk_size=0)
    for shape in ((31,),(64,)):
      with self.subTest(shape=shape),self.assertRaisesRegex(ValueError,'packed tensor length'):
        gguf_shard({'q':(tensor,shape,8)},('CPU',),stream=True)

  def test_asynchronous_upload_staging_is_drained_at_chunk_boundaries(self):
    pending = SimpleNamespace(bytes=0,peak=0)
    def synchronize():pending.bytes = 0
    def release_cache():self.assertEqual(pending.bytes,0)
    device = SimpleNamespace(synchronize=synchronize,allocator=SimpleNamespace(free_cache=release_cache))
    def observe(data,count,kind):
      pending.bytes+=data.numel()
      pending.peak=max(pending.peak,pending.bytes)
      return ggml_data_to_tensor(data,count,kind)
    payload = (np.float16(1).tobytes()+bytes(range(32)))*5
    entries = {'q':(Tensor(list(payload),device='CPU',dtype=dtypes.uint8),(5,32),8),
               'norm':(Tensor(np.ones(3,dtype=np.float32).view(np.uint8),device='CPU'),(3,),0)}
    # Plain replacements avoid retaining the input chunks in a mock call history.
    with patch('tinygrad.llm.gguf.Device',new={'CPU':device}),patch('tinygrad.llm.gguf.ggml_data_to_tensor',new=observe):
      result = gguf_shard(entries,('CPU',),stream=True,cast_dtypes={'q':dtypes.half},chunk_size=64)
    self.assertEqual(pending.bytes,0)
    self.assertLessEqual(pending.peak,68)
    np.testing.assert_array_equal(result['q'].numpy(),np.tile(np.arange(32,dtype=np.float16),(5,1)))

  def test_live_allocations_are_bounded_by_resident_weights_and_chunk_scratch(self):
    count,weights,chunk = 131072,5,4096
    packed = (np.float16(1).tobytes()+bytes(range(32)))*(count//32)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'weights.bin'
      path.write_bytes(packed*weights)
      source = Tensor(path)
      entries = {str(i):(source[i*len(packed):(i+1)*len(packed)],(count,),8) for i in range(weights)}
      # Other tests can leave cyclic Tensor references whose later collection
      # would subtract unrelated allocations from this process-wide counter.
      gc.collect()
      baseline,peaks = GlobalCounters.mem_used,[]
      def observe(*args):
        peaks.append(GlobalCounters.mem_used-baseline)
        return ggml_data_to_tensor(*args)
      # A Mock's call history would retain the uploaded chunks and invalidate the measurement.
      with patch('tinygrad.llm.gguf.ggml_data_to_tensor',new=observe):
        result = gguf_shard(entries,('CPU',),stream=True,cast_dtypes=dict.fromkeys(entries,dtypes.half),chunk_size=chunk)
      resident = count*weights*2
      gc.collect()
      self.assertGreaterEqual(GlobalCounters.mem_used-baseline,resident)
      self.assertLessEqual(max(peaks),resident+chunk*16)
      for value in result.values():np.testing.assert_array_equal(value.numpy()[:32],np.arange(32,dtype=np.float16))

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_q8_chunks_and_matmul_match_cpu_weights(self):
    values = np.arange(-80,80,dtype=np.int8).reshape(5,32)
    scales = np.array([.125,.25,.5,1.,2.],dtype=np.float16)
    packed = b''.join(scale.tobytes()+row.tobytes() for scale,row in zip(scales,values))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'weights.bin'
      path.write_bytes(packed)
      entries = {'q':(Tensor(path),(5,32),8)}
      weights = gguf_shard(entries,('CL',),stream=True,cast_dtypes={'q':dtypes.half},chunk_size=64)['q']
      expected = (scales.astype(np.float32)[:,None]*values).astype(np.float16)
      np.testing.assert_array_equal(weights.numpy(),expected)
      x = np.linspace(-1,1,32,dtype=np.float32)
      np.testing.assert_allclose((weights.float()@Tensor(x,device='CL')).numpy(),expected.astype(np.float32)@x,rtol=1e-5,atol=1e-4)
