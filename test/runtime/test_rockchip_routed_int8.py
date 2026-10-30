import ctypes, os, struct, threading, unittest, weakref
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.device import TinyELF
from tinygrad.helpers import Target, round_up
from tinygrad.llm.kernels.rockchip_routed_int8 import raw_matmul
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipRoutedMatmul
from tinygrad.runtime.autogen import rockchip as rk


class FakeDevice:
  def __init__(self):
    self.programs=weakref.WeakSet()
    self.storage=[]
    self.lock=threading.Lock()
    self.fd=-1
    self.domain=0
    self.submit_count=0
    self.task_count=0
    self.poisoned=False
    self.syncs=[]
  def allocate(self,size,flags=0):
    storage=ctypes.create_string_buffer(size)
    self.storage.append(storage)
    return RockchipBuffer(ctypes.addressof(storage),0x100000+len(self.storage)*0x100000,size,
                          type('Meta',(),{'obj_addr':0x1000+len(self.storage),'version':0,'cpu_clean_version':0})())
  def free(self,*args): pass
  def sync(self,*args): self.syncs.append(args)
  def check_domain(self,*args): pass
  def check_healthy(self):
    if self.poisoned: raise RuntimeError('poisoned')


class TestRockchipRoutedInt8(unittest.TestCase):
  @staticmethod
  def descriptor(m=1,n=64,k=64,cores=3,groups=6,tile=1,marker=0x47524d4d,version=3,experts=8):
    return TinyELF(struct.pack('<9I',m,n,k,cores,groups,tile,marker,version,experts),'routed',Target('ROCKCHIP'),())

  def test_invalid_descriptors_rejected_before_device_access(self):
    class NoAccess:
      def __getattr__(self,name): raise AssertionError(f'device touched: {name}')
    for changes in ({'m':0},{'m':2},{'groups':0},{'groups':9},{'experts':0},{'experts':129},
                    {'version':2},{'tile':2},{'k':96},{'n':33},{'cores':2}):
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        RockchipRoutedMatmul(NoAccess(),self.descriptor(**changes))

  def test_buffer_bounds_and_selector_patches(self):
    dev=FakeDevice()
    plan=RockchipRoutedMatmul(dev,self.descriptor())
    out=RockchipBuffer(0,0x10000000,6*64*4,FakeDevice().allocate(8).meta)
    lhs_mem=ctypes.create_string_buffer(plan.lhs_offset+6*64)
    lhs=RockchipBuffer(ctypes.addressof(lhs_mem),0x20000000,len(lhs_mem),type('Meta',(),{'obj_addr':2,'version':0,'cpu_clean_version':0})())
    rhs=RockchipBuffer(0,0x30000000,8*64*64-1,type('Meta',(),{'obj_addr':3,'version':0,'cpu_clean_version':0})())
    with self.assertRaises(ValueError): plan.prepare(out,lhs,rhs)
    rhs=RockchipBuffer(0,0x30000000,8*64*64,type('Meta',(),{'obj_addr':3,'version':0,'cpu_clean_version':0})())
    plan.prepare(out,lhs,rhs)
    for ids in ((0,1,2,3,4,7),(2,2,1,1,0,0),(7,6,5,4,3,2)):
      for i,value in enumerate(ids): ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+i*4).value=value
      before=[ctypes.c_uint64.from_address(addr).value for addr,_ in plan.weight_patches]
      plan.update_inputs(lhs,rhs)
      after=[ctypes.c_uint64.from_address(addr).value for addr,_ in plan.weight_patches]
      self.assertNotEqual(before,after)
      for (value,(_,group)) in zip(after,plan.weight_patches):
        self.assertEqual((value >> 16) & 0xffffffff,rhs.dma+ids[group]*plan.n*plan.k)
    for bad_group,bad_id in ((0,-1),(5,8)):
      for i,value in enumerate((7,6,5,4,3,2)):
        ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+i*4).value=value
      ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+bad_group*4).value=bad_id
      before=ctypes.string_at(plan.cmd.addr,plan.cmd.size)
      with self.assertRaises(ValueError): plan.update_inputs(lhs,rhs)
      self.assertEqual(before,ctypes.string_at(plan.cmd.addr,plan.cmd.size))
    plan.close()

  def test_duplicate_routes_into_smaller_atlas(self):
    dev=FakeDevice()
    plan=RockchipRoutedMatmul(dev,self.descriptor(experts=2))
    out=dev.allocate(6*64*4)
    lhs=dev.allocate(plan.lhs_offset+6*64)
    rhs=dev.allocate(2*64*64)
    plan.prepare(out,lhs,rhs)
    ids=(0,1,0,1,1,0)
    for i,value in enumerate(ids):ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+i*4).value=value
    plan.update_inputs(lhs,rhs)
    for address,group in plan.weight_patches:
      word=ctypes.c_uint64.from_address(address).value
      self.assertEqual((word>>16)&0xffffffff,rhs.dma+ids[group]*64*64)
    plan.close()

  def test_activation_scale_guard_precedes_submit(self):
    dev=FakeDevice()
    plan=RockchipRoutedMatmul(dev,self.descriptor())
    out_mem=ctypes.create_string_buffer(6*64*4)
    lhs_mem=ctypes.create_string_buffer(plan.lhs_offset+6*64)
    rhs_mem=ctypes.create_string_buffer(8*64*64)
    def buf(mem,dma,metaid):
      return RockchipBuffer(ctypes.addressof(mem),dma,len(mem),type('Meta',(),{'obj_addr':metaid,'version':0,'cpu_clean_version':0})())
    out,lhs,rhs=buf(out_mem,0x10000000,1),buf(lhs_mem,0x20000000,2),buf(rhs_mem,0x30000000,3)
    for g in range(6):
      ctypes.c_float.from_address(lhs.addr+g*4).value=1.0
      ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+g*4).value=g%8
    plan.prepare(out,lhs,rhs)
    with patch('tinygrad.runtime.ops_rockchip._finite_program',return_value=object()), patch.object(rk,'DRM_IOCTL_RKNPU_SUBMIT') as submit:
      for scale in (0.0,-1.0,float('inf'),float('nan')):
        for index in (0,5):
          for g in range(6):ctypes.c_float.from_address(lhs.addr+g*4).value=1.0
          ctypes.c_float.from_address(lhs.addr+index*4).value=scale
          with self.assertRaises(ValueError): plan(out,lhs,rhs)
      submit.assert_not_called()
      for g in range(6): ctypes.c_float.from_address(lhs.addr+g*4).value=1.0
      plan(out,lhs,rhs)
      submit.assert_called_once()
    plan.close()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_raw_integer_goldens_and_changed_selectors_replay(self):
    rng=np.random.default_rng(271410)
    experts,n,k=8,64,64
    weights=rng.integers(-128,128,(experts,n,k),dtype=np.int8)
    weights[0,:,::2],weights[-1,:,1::2]=-128,127
    packed=weights.reshape(experts,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    groups=6
    header=round_up(groups*8,64)
    run=TinyJit(lambda payload:raw_matmul(payload,weight,groups,n).realize())
    for step,ids in enumerate(((0,1,2,3,4,7),(2,2,1,1,0,0),(7,6,5,4,3,2))*2):
      x=rng.integers(-128,128,(groups,k),dtype=np.int8)
      if step==2: x[:,::2],x[:,1::2]=-128,127
      payload=np.zeros(header+groups*k,dtype=np.uint8)
      payload[:groups*4]=np.ones(groups,dtype=np.float32).view(np.uint8)
      payload[groups*4:groups*8]=np.asarray(ids,dtype=np.int32).view(np.uint8)
      payload[header:]=x.ravel().view(np.uint8)
      expected=np.stack([x[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate(ids)])
      before=Device['ROCKCHIP'].submit_count
      actual=run(Tensor(payload,device='CPU').realize()).numpy()
      np.testing.assert_array_equal(actual,expected)
      self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
      if step==1: run.captured=RockchipCapturedJit(run.captured)
    self.assertGreater(run.captured.native_runs,0)
    payload[groups*4:groups*8]=np.asarray((0,1,2,3,4,experts),dtype=np.int32).view(np.uint8)
    before=Device['ROCKCHIP'].submit_count
    with self.assertRaises(ValueError): run(Tensor(payload,device='CPU').realize())
    self.assertEqual(Device['ROCKCHIP'].submit_count,before)
    payload[groups*4:groups*8]=np.asarray((7,6,5,4,3,2),dtype=np.int32).view(np.uint8)
    np.testing.assert_array_equal(run(Tensor(payload,device='CPU').realize()).numpy(),
      np.stack([x[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate((7,6,5,4,3,2))]))


if __name__=='__main__': unittest.main()
