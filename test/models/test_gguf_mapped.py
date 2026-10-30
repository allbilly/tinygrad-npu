import gc, tempfile, unittest, weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import gguf_shard


class TestGGUFMappedCPU(unittest.TestCase):
  def test_mapped_q8_and_native_chunks_alias_file_and_survive_source_release(self):
    scales = np.array([.125,.25,.5,1.,2.],dtype=np.float16)
    qvalues = np.arange(-80,80,dtype=np.int8).reshape(5,32)
    qbytes = b''.join(scale.tobytes()+row.tobytes() for scale,row in zip(scales,qvalues))
    native = np.array([-3.5,-1.25,0.,.5,2.75,4.125,8.],dtype=np.float16)
    prefix, qtail, gap, ntail = bytes(range(19)), b'xyz', b'offset!', b'END'
    qstart = len(prefix)
    nstart = qstart+len(qbytes)+len(qtail)+len(gap)
    payload = prefix+qbytes+qtail+gap+native.tobytes()+ntail
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'weights.bin'
      path.write_bytes(payload)
      source = Tensor(path)
      entries = {
        'q':(source[qstart:qstart+len(qbytes)+len(qtail)],(5,32),8),
        'native':(source[nstart:nstart+native.nbytes+len(ntail)],native.shape,1),
      }
      expected_q = (scales.astype(np.float32)[:,None]*qvalues).astype(np.float16)
      copied = gguf_shard(entries,('CPU',),cast_dtypes={'q':dtypes.half})
      expected = {name:value.numpy().copy() for name,value in copied.items()}
      del copied

      sources = {name:data.uop.buffer for name,(data,_,_) in entries.items()}
      expected_sizes = {'q':len(qbytes),'native':native.nbytes}
      seen = []
      from tinygrad.llm.gguf import ggml_data_to_tensor
      def observe(data,count,kind):
        buf = data.uop.buffer
        seen.append((kind,data.device,data.dtype,buf.nbytes,buf.host.addr,buf.meta,buf.options.external_ptr))
        return ggml_data_to_tensor(data,count,kind)
      file_before = path.read_bytes()
      with patch('tinygrad.llm.gguf.ggml_data_to_tensor',new=observe):
        mapped = gguf_shard(entries,('CPU',),cast_dtypes={'q':dtypes.half},mapped_cpu=True)
      self.assertEqual([(kind,device,dtype) for kind,device,dtype,*_ in seen],
                       [(8,'CPU',dtypes.uint8),(1,'CPU',dtypes.uint8)])
      for kind,device,dtype,size,address,mapping,external in seen:
        self.assertEqual(address,mapping.address+(qstart if kind==8 else nstart))
        self.assertEqual(address,external)
        self.assertGreaterEqual(size,expected_sizes['q' if kind==8 else 'native'])
      self.assertIs(seen[0][5],seen[1][5])
      retained_mapping = weakref.ref(mapping)
      np.testing.assert_array_equal(mapped['q'].numpy(),expected_q)
      np.testing.assert_array_equal(mapped['q'].numpy(),expected['q'])
      np.testing.assert_array_equal(mapped['native'].numpy(),native)
      np.testing.assert_array_equal(mapped['native'].numpy(),expected['native'])
      self.assertEqual(path.read_bytes(),file_before)

      # Output tensors must own the mapped source lifetime, independently of the
      # original GGUF Tensor and entry views.
      seen.clear()
      del source, entries, sources, mapping
      gc.collect()
      self.assertIsNotNone(retained_mapping())
      np.testing.assert_array_equal(mapped['q'].numpy(),expected_q)
      np.testing.assert_array_equal(mapped['native'].numpy(),native)
      self.assertEqual(path.read_bytes(),file_before)
      del mapped
      gc.collect()

  def test_native_weight_writes_do_not_modify_checkpoint(self):
    original = np.array([1.,2.,3.,4.],dtype=np.float16)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'native.bin'
      path.write_bytes(original.tobytes())
      result = gguf_shard({'weight':(Tensor(path),original.shape,1)},('CPU',),mapped_cpu=True)['weight']
      storage = result.contiguous().realize().uop.buffer
      replacement = np.full(original.shape,5.,dtype=np.float16)
      storage.as_memoryview(allow_zero_copy=True)[:] = replacement.tobytes()
      np.testing.assert_array_equal(result.numpy(),replacement)
      self.assertEqual(path.read_bytes(),original.tobytes())
      result.assign(Tensor.full(original.shape,7.,device='CPU',dtype=dtypes.half)).realize()
      np.testing.assert_array_equal(result.numpy(),np.full(original.shape,7.,dtype=np.float16))
      self.assertEqual(path.read_bytes(),original.tobytes())

  def test_mapped_decode_keeps_exact_int8_scales_and_packed_codes(self):
    from tinygrad.llm.kernels.rockchip_int8 import quantize
    rng = np.random.default_rng(68)
    codes = rng.integers(-128,128,(64,16,32),dtype=np.int8)
    scales = rng.choice(np.array([-.25,.125,.5],dtype=np.float16),(64,16))
    payload = b''.join(scale.tobytes()+block.tobytes() for scale,block in zip(scales.flat,codes.reshape(-1,32)))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'packed.bin'
      path.write_bytes(payload)
      entries = {'weight':(Tensor(path),(64,512),8)}
      copied = gguf_shard(entries,('CPU',),cast_dtypes={'weight':dtypes.half})['weight']
      mapped = gguf_shard(entries,('CPU',),cast_dtypes={'weight':dtypes.half},mapped_cpu=True)['weight']
      results = []
      for weight in (copied,mapped):
        raw,row_scales = quantize(weight,512)
        packed = raw[256:].bitcast(dtypes.int8).reshape(2,32,16,32).permute(0,2,1,3).contiguous()
        results.append((packed.numpy(),row_scales.numpy()))
      np.testing.assert_array_equal(results[0][0],results[1][0])
      np.testing.assert_array_equal(results[0][1].view(np.uint32),results[1][1].view(np.uint32))
      self.assertEqual(path.read_bytes(),payload)

  def test_mapped_cpu_rejects_unsupported_stream_devices_and_storage(self):
    packed = (np.float16(.5).tobytes()+bytes(range(32)))*2
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'packed.bin'
      path.write_bytes(packed)
      disk_u8 = Tensor(path)
      entries = {'q':(disk_u8,(2,32),8)}
      with self.assertRaises(ValueError):gguf_shard(entries,('CPU',),stream=True,mapped_cpu=True)
      with self.assertRaises(ValueError):gguf_shard(entries,('CPU','CPU'),mapped_cpu=True)

      cpu_u8 = Tensor(np.frombuffer(packed,dtype=np.uint8).copy(),device='CPU')
      with self.assertRaises(ValueError):gguf_shard({'q':(cpu_u8,(2,32),8)},('CPU',),mapped_cpu=True)

      fp16_path = Path(directory)/'native-half.bin'
      fp16_path.write_bytes(np.array([1.,2.,3.,4.],dtype=np.float16).tobytes())
      disk_fp16 = Tensor(fp16_path).bitcast(dtypes.float16)
      with self.assertRaises(ValueError):gguf_shard({'half':(disk_fp16,(4,),1)},('CPU',),mapped_cpu=True)

  def test_noncontiguous_file_inputs_are_rejected_without_touching_file(self):
    payload = bytes(range(128))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'strided.bin'
      path.write_bytes(payload)
      source = Tensor(path)
      for data in (source[::2],source.flip(0)):
        with self.assertRaisesRegex(ValueError,'contiguous'):
          gguf_shard({'weight':(data,(data.numel(),),24)},('CPU',),mapped_cpu=True)
        self.assertEqual(path.read_bytes(),payload)


if __name__=='__main__': unittest.main()
