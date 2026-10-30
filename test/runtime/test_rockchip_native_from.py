import os, unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import getenv, round_up
from tinygrad.llm.kernels.rockchip_replay import NativeReplay
from tinygrad.runtime.ops_rockchip import RockchipBuffer
from tinygrad.runtime.autogen import rockchip as rk


class TestNativeFromInvalidation(unittest.TestCase):
  @staticmethod
  def buffer(*, flags=None, meta_size=128, size=64, offset=64, dma_offset=64,
             cpu_clean_version=1, version=1):
    base_flags = (rk.RKNPU_MEM_CACHEABLE | rk.RKNPU_MEM_NON_CONTIGUOUS |
                  rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    meta = SimpleNamespace(
      dma_addr=0x20000, flags=base_flags if flags is None else flags, size=meta_size,
      obj_addr=7, version=version, cpu_clean_version=cpu_clean_version,
    )
    base_addr = 0x10000
    return RockchipBuffer(base_addr + offset, meta.dma_addr + dma_offset, size, meta)

  @staticmethod
  def replay(raw, *, invalidator=1234, previous=True):
    replay = NativeReplay.__new__(NativeReplay)
    replay.invalidator = invalidator
    replay.invalidate_sites = 0
    replay.driver_from_sites = 0
    replay.cleaner = None
    replay.clean_sites = 0
    replay.ioctl = 'mock_ioctl'
    replay.dev = SimpleNamespace(fd=SimpleNamespace(fd=9))
    replay.npu_outputs = {0: (id(raw.meta), raw.addr, raw.size)} if previous else {}
    return replay

  def test_immediate_npu_output_uses_native_invalidator(self):
    base_flags = (rk.RKNPU_MEM_CACHEABLE | rk.RKNPU_MEM_NON_CONTIGUOUS |
                  rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    for flags in (base_flags,):
      raw = self.buffer(flags=flags)
      replay = self.replay(raw)
      code = replay.sync(raw, rk.RKNPU_MEM_SYNC_FROM_DEVICE, 1)
      with self.subTest(flags=flags):
        self.assertIn('1234ULL', code)
        self.assertIn(f'{raw.addr}ULL,{raw.size}UL', code)
        self.assertEqual(replay.invalidate_sites, 1)
        self.assertEqual(replay.driver_from_sites, 0)

  def test_missing_invalidator_or_non_immediate_output_uses_driver(self):
    raw = self.buffer()
    cases = ((self.replay(raw, invalidator=None), 1),
             (self.replay(raw, previous=False), 1),
             (self.replay(raw), 2))
    for replay, index in cases:
      with self.subTest(index=index, invalidator=replay.invalidator):
        code = replay.sync(raw, rk.RKNPU_MEM_SYNC_FROM_DEVICE, index)
        self.assertIn('mock_ioctl', code)
        self.assertEqual(replay.invalidate_sites, 0)
        self.assertEqual(replay.driver_from_sites, 1)

  def test_batch_invalidation_requires_an_exact_pending_clean_output(self):
    for enabled,pending,dirty in ((True,True,False),(False,True,False),(True,False,False),(True,True,True)):
      with self.subTest(enabled=enabled,pending=pending,dirty=dirty):
        raw = self.buffer(cpu_clean_version=0 if dirty else 1)
        replay = self.replay(raw,previous=False)
        replay.borrow_batch = enabled
        replay.pending_npu_outputs = {(id(raw.meta),raw.addr,raw.size)} if pending else set()
        code = replay.sync(raw,rk.RKNPU_MEM_SYNC_FROM_DEVICE,3)
        native = enabled and pending and not dirty
        self.assertEqual('1234ULL' in code,native)
        self.assertEqual(replay.invalidate_sites,int(native))

  def test_cpu_upload_forgets_overlapping_pending_npu_output(self):
    from collections import Counter
    from tinygrad.llm.kernels import rockchip_replay
    for offset in (0,64):
      with self.subTest(offset=offset):
        raw = self.buffer(offset=0,dma_offset=0)
        replay = self.replay(raw)
        replay.borrow_batch = True
        replay.pending_npu_outputs = {(id(raw.meta),raw.addr,raw.size)}
        dest = RockchipBuffer(raw.addr+offset,raw.dma+offset,raw.size,raw.meta)
        replay.statements,replay.alias_copies,replay.writes,replay.write_counts,replay.mutated = [],set(),{},Counter(),set()
        replay.memcpy = 'mock_memcpy'
        buf = SimpleNamespace(nbytes=64)
        dst,src = object(),object()
        def buffer(node):return (buf,dest,'target') if node is dst else (buf,123456,'source')
        with patch.object(rockchip_replay,'get_call_arg_uops',return_value=(dst,src)),patch.object(replay,'buffer',side_effect=buffer):
          replay.copy(None,2)
        self.assertEqual(bool(replay.pending_npu_outputs),offset==64)

  def test_output_identity_and_size_mismatches_use_driver(self):
    for mismatch in ('identity', 'size'):
      raw = self.buffer()
      replay = self.replay(raw)
      if mismatch == 'identity':
        other = self.buffer()
        replay.npu_outputs[0] = (id(other.meta), raw.addr, raw.size)
      else:
        replay.npu_outputs[0] = (id(raw.meta), raw.addr, raw.size-1)
      with self.subTest(mismatch=mismatch):
        code = replay.sync(raw, rk.RKNPU_MEM_SYNC_FROM_DEVICE, 1)
        self.assertIn('mock_ioctl', code)
        self.assertEqual(replay.invalidate_sites, 0)
        self.assertEqual(replay.driver_from_sites, 1)

  def test_index_zero_does_not_match_previous_output_minus_one(self):
    raw = self.buffer()
    replay = self.replay(raw, previous=False)
    replay.npu_outputs[-1] = (id(raw.meta), raw.addr, raw.size)
    code = replay.sync(raw, rk.RKNPU_MEM_SYNC_FROM_DEVICE, 0)
    self.assertIn('mock_ioctl', code)
    self.assertEqual(replay.invalidate_sites, 0)
    self.assertEqual(replay.driver_from_sites, 1)

  def test_unqualified_views_use_driver(self):
    base_flags = (rk.RKNPU_MEM_CACHEABLE | rk.RKNPU_MEM_NON_CONTIGUOUS |
                  rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)
    cases = (
      self.buffer(cpu_clean_version=0, version=1),
      self.buffer(offset=1, dma_offset=1),  # unaligned address
      self.buffer(size=65, meta_size=128),  # unaligned view size
      self.buffer(meta_size=4*1024*1024+64, size=4*1024*1024+64, offset=0, dma_offset=0),
      self.buffer(flags=base_flags & ~rk.RKNPU_MEM_CACHEABLE),
      self.buffer(flags=base_flags | rk.RKNPU_MEM_KERNEL_MAPPING),
      self.buffer(size=0),
      self.buffer(meta_size=128, size=128, offset=64, dma_offset=64),  # view exceeds allocation
    )
    for raw in cases:
      with self.subTest(raw=raw):
        replay = self.replay(raw)
        code = replay.sync(raw, rk.RKNPU_MEM_SYNC_FROM_DEVICE, 1)
        self.assertIn('mock_ioctl', code)
        self.assertEqual(replay.invalidate_sites, 0)
        self.assertEqual(replay.driver_from_sites, 1)

  def test_large_invalidation_is_opt_in_and_keeps_range_and_cleanliness_guards(self):
    for enabled in (False,True):
      for root_size,size,dirty,expected in ((6*1024**2,6*1024**2,False,enabled),
          (16*1024**2,6*1024**2,False,enabled),(16*1024**2,16*1024**2,False,enabled),
          (16*1024**2+64,64,False,False),(16*1024**2,6*1024**2,True,False)):
        raw=self.buffer(meta_size=root_size,size=size,offset=0,dma_offset=0,cpu_clean_version=0 if dirty else 1)
        replay=self.replay(raw)
        with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=int(enabled)):
          code=replay.sync(raw,rk.RKNPU_MEM_SYNC_FROM_DEVICE,1)
        self.assertEqual('1234ULL' in code,expected)
        self.assertEqual(replay.invalidate_sites,int(expected))

  def test_to_device_keeps_cleaner_path_and_never_invalidates(self):
    raw = self.buffer()
    replay = self.replay(raw)
    replay.cleaner = 5678
    code = replay.sync(raw, rk.RKNPU_MEM_SYNC_TO_DEVICE, 1)
    self.assertIn('5678ULL', code)
    self.assertEqual(replay.clean_sites, 1)
    self.assertEqual(replay.invalidate_sites, 0)
    self.assertEqual(replay.driver_from_sites, 0)
    too_large = self.buffer(meta_size=1024*1024+64,size=1024*1024+64,offset=0,dma_offset=0)
    replay = self.replay(too_large)
    replay.cleaner = 5678
    code = replay.sync(too_large,rk.RKNPU_MEM_SYNC_TO_DEVICE,1)
    self.assertIn('mock_ioctl',code)
    self.assertEqual(replay.clean_sites,0)
    self.assertEqual(replay.invalidate_sites,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_changed_integer_outputs_survive_native_invalidation_and_rebinding(self):
    from tinygrad.llm.kernels.rockchip_int8 import raw_int8_matmul
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
    rng = np.random.default_rng(20261027)
    for rows,columns,depth in ((1,3072,1024),(17,3072,1024),(128,6144,1024)):
      values = rng.integers(-128,128,(columns,depth),dtype=np.int8)
      packed = values.reshape(columns//32,32,depth//32,32).transpose(0,2,1,3).copy()
      weight = Tensor(packed,device='ROCKCHIP').realize()
      inputs = [rng.integers(-128,128,(rows,depth),dtype=np.int8) for _ in range(8)]
      inputs[3][:] = 127
      inputs[4][:,::2],inputs[4][:,1::2] = -128,127
      inputs[5][:] = 0
      reference = []
      for enabled in ('0','1'):
        with patch.dict(os.environ,ONPU_REPLAY_NATIVE_FROM=enabled):
          getenv.cache_clear()
          run = TinyJit(lambda payload:raw_int8_matmul(payload,weight,rows,columns).realize())
          for step,data in enumerate(inputs):
            header = round_up(rows*4,64)
            payload = np.zeros(header+rows*depth,dtype=np.uint8)
            payload[:rows*4] = np.ones(rows,dtype=np.float32).view(np.uint8)
            payload[header:] = data.ravel().view(np.uint8)
            actual = run(Tensor(payload,device='CPU').realize()).numpy().copy()
            matrix = data.astype(np.int32)@values.astype(np.int32).T
            expected = np.concatenate([matrix[:,first:first+columns//3].ravel() for first in range(0,columns,columns//3)])
            np.testing.assert_array_equal(actual,expected)
            if enabled=='0': reference.append(actual)
            else: np.testing.assert_array_equal(actual,reference[step])
            if step==1: run.captured = RockchipCapturedJit(run.captured)
          self.assertGreater(run.captured.native_runs,0)
          self.assertIsNone(run.captured.reason)
          native = run.captured.native
          self.assertEqual(native.invalidate_sites,int(enabled=='1'))
          self.assertEqual(native.driver_from_sites,int(enabled=='0'))
      getenv.cache_clear()


if __name__ == '__main__': unittest.main()
