import ctypes, os, subprocess, sys, unittest, weakref
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, UOp, nn
from tinygrad.llm.kernels.rockchip import (RockchipLinear, configure, pack_weight, packed_matmul, pack_split_weight,
                                         packed_activation_matmul)
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import _matmul_body, _finite_program, RockchipBuffer, RockchipDevice, RockchipProgram, RockchipAllocator
from tinygrad.runtime.autogen import rockchip as rk


class TestRockchipBounds(unittest.TestCase):
  def test_released_device_rejects_copyin_before_touching_memory(self):
    def released(*args): raise RuntimeError('This worker released its NPU ownership')
    allocator = SimpleNamespace(dev=SimpleNamespace(check_healthy=released,sync=released))
    dest = SimpleNamespace(addr=1,meta=SimpleNamespace(version=0))
    with patch('tinygrad.runtime.ops_rockchip.ctypes.memmove') as move:
      with self.assertRaises(RuntimeError): RockchipAllocator._copyin(allocator,dest,memoryview(bytearray(b'x')))
      move.assert_not_called()
    self.assertEqual(dest.meta.version,0)

  def test_domain_uapi_layout(self):
    self.assertEqual(ctypes.sizeof(rk.struct_rknpu_mem_create),48)
    self.assertEqual(rk.struct_rknpu_mem_create.iommu_domain_id.offset,40)
    self.assertEqual(ctypes.sizeof(rk.struct_rknpu_submit),104)
    self.assertEqual(rk.struct_rknpu_submit.iommu_domain_id.offset,32)
    self.assertEqual(rk.struct_rknpu_submit.subcore_task.offset,64)

  def test_cross_domain_bindings_rejected(self):
    dev = SimpleNamespace(domain=1)
    def buf(domain): return RockchipBuffer(0,0,4096,SimpleNamespace(iommu_domain_id=domain))
    RockchipDevice.check_domain(dev,buf(1),buf(1),buf(1))
    with self.assertRaises(ValueError): RockchipDevice.check_domain(dev,buf(1),buf(0),buf(1))
    with self.assertRaises(ValueError): RockchipDevice('ROCKCHIP:16')

  def test_weight_domain_spill_and_default(self):
    from tinygrad.runtime.ops_rockchip import int8_weight_device
    hw = SimpleNamespace(weight_domain=0)
    first,second = SimpleNamespace(hw=hw,mapped_bytes=4096),SimpleNamespace(mapped_bytes=0)
    options = {'ONPU_NPU_DOMAINS':2,'ONPU_NPU_DOMAIN_BYTES':4096}
    with patch('tinygrad.runtime.ops_rockchip.Device',{'ROCKCHIP':first,'ROCKCHIP:1':second}), \
         patch('tinygrad.runtime.ops_rockchip.getenv',side_effect=lambda key,default:options.get(key,default)):
      self.assertEqual(int8_weight_device(4096),'ROCKCHIP:1')
      second.mapped_bytes = 4096
      with self.assertRaises(MemoryError): int8_weight_device(4096)
      options['ONPU_NPU_DOMAINS'] = 1
      self.assertEqual(int8_weight_device(4096),'ROCKCHIP')

  def test_binding_eviction_preserves_live_replay_owner(self):
    references = []
    class Plan:
      def __init__(self,*args):
        self.owned = [object()]
        references.append(weakref.ref(self))
      def __call__(self,*args,**kwargs): return None
      def close(self): self.owned.clear()
      def __del__(self): self.close()
    with patch('tinygrad.runtime.ops_rockchip.RockchipMatmul',Plan):
      program = RockchipProgram(None,None)
      metadata = object()
      program(SimpleNamespace(meta=metadata,addr=0,size=64))
      replay_owner = references[0]()
      for i in range(1,260): program(SimpleNamespace(meta=metadata,addr=i*64,size=64))
      self.assertTrue(replay_owner.owned)
      self.assertEqual(len(program.bindings),256)
      held = references[0]
      del replay_owner
      self.assertIsNone(held())

  def test_finite_guard_all_encodings_vector_lanes_and_tails(self):
    check = _finite_program().fxn
    for bits in range(65536):
      value = ctypes.c_uint16(bits)
      self.assertEqual(bool(check(ctypes.byref(value),ctypes.c_ulong(1))),(bits&0x7c00)!=0x7c00)
    for count in (0,1,7,8,9,16,17,129):
      for offset in (0,1):
        values = np.full(count+2,0x3c00,dtype=np.uint16)
        pointer = ctypes.c_void_p(values.ctypes.data+offset*2)
        self.assertTrue(check(pointer,ctypes.c_ulong(count)))
        for position in range(count):
          for invalid in (0x7c00,0xfc00,0x7c01,0xffff):
            values[position+offset] = invalid
            self.assertFalse(check(pointer,ctypes.c_ulong(count)))
            values[position+offset] = 0x3c00

  def test_reject_unproven_geometry(self):
    for m,n,k in [(0,32,32),(1,33,32),(65,32,32),(64,32,4096),(1,32,8192)]:
      with self.assertRaises(ValueError): _matmul_body(m,n,k,0,0,0)

  def test_reject_oversized_weights_before_device_access(self):
    with self.assertRaises(ValueError): pack_weight(Tensor.empty(32,4097,device='CPU'))
    with self.assertRaises(ValueError): pack_weight(Tensor.empty(24577,32,device='CPU'))

  def test_sync_initial_clean_and_shared_buffer_ranges(self):
    dev = SimpleNamespace(check_healthy=lambda:None,fd=object())
    meta = SimpleNamespace(dma_addr=0x10000,obj_addr=1,size=32768,version=3,cpu_clean_version=-1)
    buf = RockchipBuffer(0x24000,0x14000,128,meta)
    with patch.object(rk,'DRM_IOCTL_RKNPU_MEM_SYNC') as sync:
      RockchipDevice.sync(dev,buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.assertEqual((sync.call_args.kwargs['offset'],sync.call_args.kwargs['size']),(0,32768))
      self.assertEqual(meta.cpu_clean_version,meta.version)
      for flag in (rk.RKNPU_MEM_SYNC_FROM_DEVICE,rk.RKNPU_MEM_SYNC_TO_DEVICE):
        RockchipDevice.sync(dev,buf,flag)
        self.assertEqual((sync.call_args.kwargs['offset'],sync.call_args.kwargs['size']),(16384,128))

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_matmul_replay_rebind_and_mode_transitions(self):
    rng = np.random.default_rng(18)
    for host in ['CPU','CL']:
      for m,k,n in [(1,65,97),(1,1024,3072),(16,1024,1024),(32,2048,1024),(21,3072,1024),(1,256,256),(1,1024,24563)]:
        b = rng.normal(0,.2,(n,k)).astype(np.float16)
        w = pack_weight(Tensor(b,device=host)).realize()
        fn = TinyJit(lambda x:packed_matmul(x,w,n).realize())
        for _ in range(4):
          a = rng.normal(0,.2,(m,k)).astype(np.float16)
          x = Tensor(a,device=host).realize()
          np.testing.assert_allclose(fn(x).numpy(),a.astype(np.float32)@b.astype(np.float32).T,atol=2e-4,rtol=3e-4)
    self.assertGreater(Device['ROCKCHIP'].task_count,Device['ROCKCHIP'].submit_count)
    # Identical kernel shapes in different layers must retain separate bindings.
    weights = [pack_weight(Tensor(rng.normal(0,.2,(96,64)).astype(np.float16))).realize() for _ in range(2)]
    x = Tensor(rng.normal(0,.2,(1,64)).astype(np.float16)).realize()
    fn = TinyJit(lambda a:Tensor.realize(*(packed_matmul(a,w,96) for w in weights)))
    for _ in range(4): fn(x)
    before = Device['ROCKCHIP'].allocation_count
    for _ in range(3): fn(x)
    self.assertEqual(Device['ROCKCHIP'].allocation_count,before)
    # Shared-input projections keep full-K arithmetic, rebind through JIT and preserve phase fallback.
    def linear(n):
      layer = nn.Linear(64,n,bias=False)
      layer.weight = Tensor(rng.normal(0,.2,(n,64)).astype(np.float16),device='CPU')
      return layer
    block = SimpleNamespace(attn_q=linear(64),attn_k=linear(64),attn_v=linear(64),ffn_gate=linear(128),ffn_up=linear(128))
    originals = [block.attn_q,block.attn_k,block.attn_v,block.ffn_gate,block.ffn_up]
    model = SimpleNamespace(blk=[block])
    placement = configure(model,prefill='on',decode='on',tile_columns=24576,fuse_projections=True)
    self.assertEqual(placement['fused_groups'],2)
    def fused(x):
      outputs = (*block.attn_qkv(x),*block.ffn_gate_up(x))
      Tensor.realize(*outputs)
      return outputs
    for rows in (1,17,1):
      fn = TinyJit(fused)
      for _ in range(4):
        data = rng.normal(0,.2,(1,rows,64)).astype(np.float16)
        x = Tensor(data,device='CPU').realize()
        out = fn(x)
        for got,layer in zip(out,originals):
          expected = data.astype(np.float32)@layer.weight.numpy().astype(np.float32).T
          np.testing.assert_allclose(got.numpy(),expected,atol=2e-4,rtol=1e-3)
    for layer in block.attn_qkv.layers: layer.prefill='off'
    x = Tensor(rng.normal(0,.2,(1,17,64)).astype(np.float16),device='CPU').realize()
    before = Device['ROCKCHIP'].submit_count
    for got,layer in zip(block.attn_qkv(x),originals[:3]):
      np.testing.assert_allclose(got.numpy(),layer(x).numpy(),atol=2e-4,rtol=1e-3)
    self.assertEqual(Device['ROCKCHIP'].submit_count,before)
    source = linear(5003)
    wide = RockchipLinear(source,'on','on',tile_columns=24576)
    wide.prepare()
    for rows in (17,1):
      run = TinyJit(lambda x:wide(x).realize())
      for _ in range(4):
        data = rng.normal(0,.2,(1,rows,64)).astype(np.float16)
        out = run(Tensor(data,device='CPU').realize()).numpy()
        np.testing.assert_allclose(out,data.astype(np.float32)@source.weight.numpy().astype(np.float32).T,atol=2e-4,rtol=1e-3)
    # Native replay must execute new inputs and scalar values, survive scratch release,
    # retain finite guards and fall back when immutable packed weights are changed.
    raw_weight = rng.normal(0,.2,(96,64)).astype(np.float16)
    weight = pack_weight(Tensor(raw_weight,device='CPU')).realize()
    def calculation(x,position): return (packed_matmul(x+Tensor(position)*.001,weight,96)+Tensor(position)).realize()
    replay = TinyJit(calculation)
    for position in range(12):
      data = rng.normal(0,.2,(1,64)).astype(np.float32)
      out = replay(Tensor(data,device='CPU').realize(),UOp.variable('position',0,16).bind(position)).numpy()
      expected = (data+np.float32(position*.001)).astype(np.float16).astype(np.float32)@raw_weight.astype(np.float32).T+position
      np.testing.assert_allclose(out,expected,atol=2e-4,rtol=3e-4)
      if position==2: replay.captured = RockchipCapturedJit(replay.captured)
      if position==7:
        self.assertGreater(replay.captured.native_runs,0)
        self.assertGreater(len(replay.captured.native.alias_copies),0)
        replay.captured.free_intermediates()
    self.assertIsNone(replay.captured.reason)
    self.assertGreater(replay.captured.native_runs,4)
    before = Device['ROCKCHIP'].submit_count
    with self.assertRaises(ValueError):
      replay(Tensor(np.full((1,64),np.nan,np.float32),device='CPU').realize(),UOp.variable('position',0,16).bind(13))
    self.assertEqual(Device['ROCKCHIP'].submit_count,before)
    self.assertFalse(Device['ROCKCHIP'].poisoned)
    raw_weight = rng.normal(0,.2,(96,64)).astype(np.float16)
    packed = raw_weight.reshape(6,16,2,32).transpose(0,2,1,3).copy()
    Device['ROCKCHIP'].allocator._copyin(weight._buffer()._buf,memoryview(packed).cast('B'))
    data = rng.normal(0,.2,(1,64)).astype(np.float32)
    out = replay(Tensor(data,device='CPU').realize(),UOp.variable('position',0,16).bind(14)).numpy()
    expected = (data+np.float32(.014)).astype(np.float16).astype(np.float32)@raw_weight.astype(np.float32).T+14
    np.testing.assert_allclose(out,expected,atol=2e-4,rtol=3e-4)
    self.assertEqual(replay.captured.reason,'unverified matrix weights')
    # Activation fusion must feed the exact same half tiles into every FFN-down submission.
    def projection(inputs,outputs):
      layer = nn.Linear(inputs,outputs,bias=False)
      layer.weight = Tensor(rng.normal(0,.02,(outputs,inputs)).astype(np.float16),device='CPU')
      return layer
    block = SimpleNamespace(ffn_gate=projection(64,3072),ffn_up=projection(64,3072),ffn_down=projection(3072,96),
                            config=SimpleNamespace(swiglu_limit=None))
    placement = configure(SimpleNamespace(blk=[block]),prefill='on',decode='on',tile_columns=24576,
                          fuse_projections=True,fused_ffn=True)
    self.assertEqual(placement['fused_ffn_blocks'],1)
    for rows in (1,128,256):
      def reference(x):
        gate,up = block.ffn_gate_up(x)
        return block.ffn_down(gate*gate.sigmoid().contiguous()*up).realize()
      original,fast = TinyJit(reference),TinyJit(lambda x:block.fused_ffn(x).realize())
      for _ in range(4):
        data = Tensor(rng.normal(0,.2,(1,rows,64)).astype(np.float32),device='CPU').realize()
        np.testing.assert_array_equal(fast(data).numpy(),original(data).numpy())
    # Three independent K partials use a different FP32 summation order.
    for rows in (1,21,64):
      a = rng.normal(0,.2,(rows,3072)).astype(np.float16)
      b = rng.normal(0,.2,(1024,3072)).astype(np.float16)
      lhs = Tensor(a,device='CPU').reshape(rows,384,8).permute(1,0,2).flatten().contiguous().to('ROCKCHIP')
      weight = pack_split_weight(Tensor(b,device='CPU')).realize()
      out = packed_activation_matmul(lhs,weight,rows,1024,split_k=True).numpy()
      np.testing.assert_allclose(out,a.astype(np.float32)@b.astype(np.float32).T,atol=2e-4,rtol=3e-4)
    Device['ROCKCHIP'].finalize()
    # Releasing an unused optional accelerator must let another model own it.
    result = subprocess.run([sys.executable,'-c',"from tinygrad import Device; Device['ROCKCHIP'].finalize()"],capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__': unittest.main()
