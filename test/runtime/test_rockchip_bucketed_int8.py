import ctypes, os, struct, threading, unittest, weakref
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.device import TinyELF
from tinygrad.helpers import Target, round_up
from tinygrad.llm.kernels.rockchip_bucketed_int8 import raw_matmul
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipBucketedMatmul
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.rockchip_int8 import int8_body


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


class TestRockchipBucketedInt8(unittest.TestCase):
  @staticmethod
  def descriptor(capacity=17,n=64,k=64,cores=3,groups=8,tile=None,magic=0x47524d4d,version=4):
    return TinyELF(struct.pack('<8I',capacity,n,k,cores,groups,capacity if tile is None else tile,magic,version),
                   'bucketed',Target('ROCKCHIP'),())

  def test_invalid_descriptors_before_device_access(self):
    class NoAccess:
      def __getattr__(self,name): raise AssertionError(f'device touched: {name}')
    for changes in ({'capacity':0},{'capacity':129},{'n':33},{'k':96},{'tile':16},{'version':3},{'magic':0},
                    {'groups':0},{'groups':129},{'cores':2}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):
        RockchipBucketedMatmul(NoAccess(),self.descriptor(**changes))

  def _plan_and_buffers(self,groups=8,capacity=17,n=64,k=64):
    dev=FakeDevice()
    plan=RockchipBucketedMatmul(dev,self.descriptor(capacity,n,k,3,groups))
    out=dev.allocate(groups*capacity*n*4)
    lhs=dev.allocate(plan.lhs_offset+groups*capacity*k)
    rhs=dev.allocate(groups*n*k)
    return dev,plan,out,lhs,rhs

  def test_buffer_bounds_rejected_before_plan_allocation(self):
    for which in ('out','lhs','rhs'):
      dev=FakeDevice()
      plan=RockchipBucketedMatmul(dev,self.descriptor())
      sizes={'out':8*17*64*4,'lhs':plan.lhs_offset+8*17*64,'rhs':8*64*64}
      buffers={key:dev.allocate(size) for key,size in sizes.items()}
      bad=buffers[which]
      buffers[which]=RockchipBuffer(bad.addr,bad.dma,bad.size-1,bad.meta)
      with self.subTest(buffer=which),self.assertRaises(ValueError):
        plan.prepare(buffers['out'],buffers['lhs'],buffers['rhs'])
      self.assertEqual(len(dev.storage),3)

  def test_dynamic_counts_active_templates_and_ordinal_task_partition(self):
    dev,plan,out,lhs,rhs=self._plan_and_buffers()
    plan.prepare(out,lhs,rhs)
    capacity,k,n=plan.m,plan.k,plan.n
    cases=((0,17,1,0,0,0,0,0),(0,0,0,17,17,17,0,0),(0,17,0,0,17,0,1,0),(0,0,0,0,0,0,17,0))
    for counts in cases:
      for g,count in enumerate(counts): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
      plan.update_inputs(lhs,rhs)
      active=[g for g,count in enumerate(counts) if count]
      self.assertEqual(plan.tasks,len(active))
      cores=3 if len(active)>=3 else 1
      self.assertEqual(plan.core_mask,(1<<cores)-1)
      self.assertEqual(sum(count for _,count in plan.subcores),len(active))
      self.assertEqual(tuple((begin,count) for begin,count in plan.subcores if count),
                       tuple((sum((len(active)+cores-1-j)//cores for j in range(i)),
                              (len(active)+cores-1-i)//cores) for i in range(cores)))
      stride=round_up(len(int8_body(1,n,k,0,0,0,n))+4,2)
      expected_groups=[g for core in range(cores) for ordinal,g in enumerate(active) if ordinal%cores==core]
      self.assertEqual(len(expected_groups),plan.tasks)
      descs=(rk.struct_rknpu_task*plan.tasks).from_address(plan.task.addr)
      for slot,g in enumerate(expected_groups):
        desc=descs[slot]
        words=(ctypes.c_uint64*desc.regcfg_amount).from_address(plan.cmd.addr+slot*stride*8)
        regs={(word>>48,word&0xffff):(word>>16)&0xffffffff for word in words}
        body=int8_body(counts[g],n,k,lhs.dma+plan.lhs_offset+g*capacity*k,rhs.dma+g*n*k,
                        out.dma+g*capacity*n*4,n)
        self.assertEqual(tuple(words),body)
        self.assertEqual(desc.regcfg_amount,len(body))
        self.assertEqual(regs[0x201,0x1070],lhs.dma+plan.lhs_offset+g*capacity*k)
        self.assertEqual(regs[0x201,0x1110],rhs.dma+g*n*k)
        self.assertEqual(regs[0x1001,0x4020],out.dma+g*capacity*n*4)
        self.assertEqual(regs[0x201,0x1024],((k-1)<<16)|k)
      self.assertEqual(len(set(expected_groups)),len(active))

  def test_invalid_counts_leave_commands_and_dynamic_state_unchanged(self):
    dev,plan,out,lhs,rhs=self._plan_and_buffers()
    plan.prepare(out,lhs,rhs)
    for counts in ((17,0,0,0,0,0,0,0),(1,0,17,0,0,0,0,1)):
      for g,count in enumerate(counts): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
      plan.update_inputs(lhs,rhs)
    before_cmd=ctypes.string_at(plan.cmd.addr,plan.cmd.size)
    before_state=tuple(plan.dynamic_state)
    for invalid in ((0,)*8,(-1,1,0,0,0,0,0,0),(1,0,0,0,0,0,0,18)):
      for g,count in enumerate(invalid): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
      with self.assertRaises(ValueError): plan.update_inputs(lhs,rhs)
      self.assertEqual(ctypes.string_at(plan.cmd.addr,plan.cmd.size),before_cmd)
      self.assertEqual(tuple(plan.dynamic_state),before_state)
    plan.close()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_bucketed_raw_integer_dynamic_rows_capture_and_recovery(self):
    rng=np.random.default_rng(20261031)
    groups,capacity,n,k=6,17,64,64
    weights=rng.integers(-128,128,(groups,n,k),dtype=np.int8)
    weights[0,:,::2],weights[-1,:,1::2]=-128,127
    packed=weights.reshape(groups,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    header=round_up((groups*capacity+groups)*4,64)
    run=TinyJit(lambda payload:raw_matmul(payload,weight,capacity,n).realize())
    counts_cases=((0,17,1,0,0,0),(1,1,1,1,1,1),(0,0,17,0,0,0),(17,0,0,0,17,0),(0,2,0,0,0,3),(0,0,0,0,17,0))
    for step,counts in enumerate(counts_cases):
      data=rng.integers(-128,128,(groups,capacity,k),dtype=np.int8)
      if step==5: data[:,:,::2],data[:,:,1::2]=-128,127
      payload=np.zeros(header+groups*capacity*k,dtype=np.uint8)
      payload[:groups*capacity*4]=np.ones(groups*capacity,dtype=np.float32).view(np.uint8)
      count_offset=groups*capacity*4
      payload[count_offset:count_offset+groups*4]=np.asarray(counts,dtype=np.int32).view(np.uint8)
      payload[header:]=data.ravel().view(np.uint8)
      device=Device['ROCKCHIP']
      before_submit,before_tasks=device.submit_count,device.task_count
      actual=run(Tensor(payload,device='CPU').realize()).numpy()
      expected_tasks=sum(count>0 for count in counts)
      self.assertEqual(device.submit_count-before_submit,1)
      self.assertEqual(device.task_count-before_tasks,expected_tasks)
      for g,count in enumerate(counts):
        if count:
          expected=data[g,:count].astype(np.int32)@weights[g].astype(np.int32).T
          np.testing.assert_array_equal(actual[g,:count],expected)
      if step==1: run.captured=RockchipCapturedJit(run.captured)
    self.assertGreater(run.captured.native_runs,0)

    valid=(0,0,17,0,0,0)
    data=rng.integers(-128,128,(groups,capacity,k),dtype=np.int8)
    payload=np.zeros(header+groups*capacity*k,dtype=np.uint8)
    payload[:groups*capacity*4]=np.ones(groups*capacity,dtype=np.float32).view(np.uint8)
    payload[groups*capacity*4:groups*capacity*4+groups*4]=np.asarray(valid,dtype=np.int32).view(np.uint8)
    payload[header:]=data.ravel().view(np.uint8)
    for bad in ((18,0,0,0,0,0),(0,0,0,0,0,-1),(0,)*groups):
      payload[groups*capacity*4:groups*capacity*4+groups*4]=np.asarray(bad,dtype=np.int32).view(np.uint8)
      before=device.submit_count
      with self.assertRaises(ValueError): run(Tensor(payload,device='CPU').realize())
      self.assertEqual(device.submit_count,before)
      self.assertFalse(device.hw.poisoned)
    payload[groups*capacity*4:groups*capacity*4+groups*4]=np.asarray(valid,dtype=np.int32).view(np.uint8)
    run(Tensor(payload,device='CPU').realize())
    run.captured.native=None
    run.captured.warmed=False
    before=device.submit_count
    actual=run(Tensor(payload,device='CPU').realize()).numpy()
    self.assertEqual(device.submit_count-before,1)
    np.testing.assert_array_equal(actual[2,:17],data[2].astype(np.int32)@weights[2].astype(np.int32).T)


if __name__=='__main__': unittest.main()
