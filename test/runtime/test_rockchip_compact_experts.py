import ctypes, os, struct, threading, unittest, weakref
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import Target, round_up
from tinygrad.llm.kernels.rockchip_bucketed_int8 import _expert_ids, compact_matmul, matmul as grouped_matmul, raw_matmul
from tinygrad.llm.kernels.rockchip_int8 import quantize
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


class TestRockchipCompactExperts(unittest.TestCase):
  @staticmethod
  def descriptor(capacity=17,n=64,k=64,cores=3,groups=8,rows=42,version=5,magic=0x47524d4d,tile=None):
    return TinyELF(struct.pack('<9I',capacity,n,k,cores,groups,capacity if tile is None else tile,magic,version,rows),
                   'compact',Target('ROCKCHIP'),())

  def test_invalid_compact_descriptors_before_device_access(self):
    class NoAccess:
      def __getattr__(self,name): raise AssertionError(f'device touched: {name}')
    for changes in ({'rows':0},{'rows':137},{'version':4},{'magic':0},{'capacity':0},{'capacity':129},
                    {'tile':1}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):
        RockchipBucketedMatmul(NoAccess(),self.descriptor(**changes))
    with self.assertRaises(ValueError):
      RockchipBucketedMatmul(NoAccess(),TinyELF(self.descriptor().lib[:32],'abi4-bad',Target('ROCKCHIP'),()))
    old=TinyELF(struct.pack('<8I',17,64,64,3,8,17,0x47524d4d,4),'abi4',Target('ROCKCHIP'),())
    dev=FakeDevice()
    plan=RockchipBucketedMatmul(dev,old)
    self.assertEqual(plan.storage_rows,8*17)
    self.assertEqual(plan.count_offset,8*17*4)
    plan.close()

  def _plan_and_buffers(self,rows=42,groups=8,capacity=17,n=64,k=64):
    dev=FakeDevice()
    plan=RockchipBucketedMatmul(dev,self.descriptor(capacity,n,k,3,groups,rows))
    out=dev.allocate(rows*n*4)
    lhs=dev.allocate(plan.lhs_offset+rows*k)
    rhs=dev.allocate(groups*n*k)
    return dev,plan,out,lhs,rhs

  def test_compact_bounds_require_storage_rows(self):
    for which in ('out','lhs','rhs'):
      dev=FakeDevice()
      plan=RockchipBucketedMatmul(dev,self.descriptor(rows=42))
      sizes={'out':42*64*4,'lhs':plan.lhs_offset+42*64,'rhs':8*64*64}
      bufs={key:dev.allocate(size) for key,size in sizes.items()}
      bad=bufs[which]
      bufs[which]=RockchipBuffer(bad.addr,bad.dma,bad.size-1,bad.meta)
      with self.subTest(buffer=which),self.assertRaises(ValueError):
        plan.prepare(bufs['out'],bufs['lhs'],bufs['rhs'])
      self.assertEqual(len(dev.storage),3)
    # A compact 42-row payload is smaller than the former G*M slot allocation.
    dev,plan,out,lhs,rhs=self._plan_and_buffers()
    self.assertLess(lhs.size,plan.lhs_offset+plan.groups*plan.m*plan.k)
    self.assertLess(out.size,plan.groups*plan.m*plan.n*4)
    plan.prepare(out,lhs,rhs)
    plan.close()

  def test_prefix_addresses_and_active_core_partition(self):
    dev,plan,out,lhs,rhs=self._plan_and_buffers()
    plan.prepare(out,lhs,rhs)
    rows,k,n=plan.storage_rows,plan.k,plan.n
    cases=((0,17,1,0,0,17,7,0),(17,0,17,8,0,0,0,0),(0,0,0,17,0,0,17,8),
           (0,0,0,0,0,0,0,0))
    for counts in cases[:-1]:
      self.assertLessEqual(sum(counts),rows)
      for g,count in enumerate(counts): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
      plan.update_inputs(lhs,rhs)
      active=[g for g,count in enumerate(counts) if count]
      cores=3 if len(active)>=3 else 1
      self.assertEqual(plan.tasks,len(active))
      self.assertEqual(plan.core_mask,(1<<cores)-1)
      stride=round_up(len(int8_body(1,n,k,0,0,0,n))+4,2)
      expected_groups=[g for core in range(cores) for ordinal,g in enumerate(active) if ordinal%cores==core]
      self.assertEqual(len(expected_groups),plan.tasks)
      self.assertEqual(sum(count for _,count in plan.subcores),plan.tasks)
      descs=(rk.struct_rknpu_task*plan.groups).from_address(plan.task.addr)
      prefix=0
      offsets=[]
      for count in counts:
        offsets.append(prefix)
        prefix+=count
      for slot,g in enumerate(expected_groups):
        desc=descs[slot]
        words=(ctypes.c_uint64*desc.regcfg_amount).from_address(plan.cmd.addr+slot*stride*8)
        body=int8_body(counts[g],n,k,lhs.dma+plan.lhs_offset+offsets[g]*k,rhs.dma+g*n*k,
                        out.dma+offsets[g]*n*4,n)
        self.assertEqual(tuple(words),body)
        self.assertEqual(desc.regcfg_amount,len(body))
    plan.close()

  def test_invalid_counts_preserve_command_words_and_state(self):
    dev,plan,out,lhs,rhs=self._plan_and_buffers()
    plan.prepare(out,lhs,rhs)
    valid=(0,17,1,0,17,7,0,0)
    for g,count in enumerate(valid): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
    plan.update_inputs(lhs,rhs)
    words=ctypes.string_at(plan.cmd.addr,plan.cmd.size)
    state=tuple(plan.dynamic_state)
    for invalid in ((17,17,9,0,0,0,0,0),(0,0,0,0,0,0,0,18),(-1,1,0,0,0,0,0,0),(0,)*8):
      for g,count in enumerate(invalid): ctypes.c_int32.from_address(lhs.addr+plan.count_offset+g*4).value=count
      with self.assertRaises(ValueError): plan.update_inputs(lhs,rhs)
      self.assertEqual(ctypes.string_at(plan.cmd.addr,plan.cmd.size),words)
      self.assertEqual(tuple(plan.dynamic_state),state)
    plan.close()

  def test_compact_expert_ids_prefix_and_bounded_tail(self):
    counts=Tensor([2,0,1,3],dtype=dtypes.int).to('CPU').realize()
    ids=Tensor.empty(8,device='CPU',dtype=dtypes.int).custom_kernel(counts,
      fxn=lambda out,c:_expert_ids(out,c,4,8,4))[0].numpy()
    np.testing.assert_array_equal(ids,[0,0,2,3,3,3,0,0])
    for bad in ([0,5,0,0],[4,4,1,0],[-1,0,0,0]):
      invalid=Tensor(bad,dtype=dtypes.int).to('CPU').realize()
      actual=Tensor.empty(8,device='CPU',dtype=dtypes.int).custom_kernel(invalid,
        fxn=lambda out,c:_expert_ids(out,c,4,8,4))[0].numpy()
      np.testing.assert_array_equal(actual,np.zeros(8,dtype=np.int32))

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_compact_raw_native_counts_and_recovery(self):
    rng=np.random.default_rng(20261031)
    groups,capacity,rows,n,k=6,17,42,64,64
    weights=rng.integers(-128,128,(groups,n,k),dtype=np.int8)
    weights[0,:,::2],weights[-1,:,1::2]=-128,127
    packed=weights.reshape(groups,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    header=round_up((rows+groups)*4,64)
    run=TinyJit(lambda payload:raw_matmul(payload,weight,capacity,n,rows=rows).realize())
    cases=((0,17,1,0,17,7),(7,7,7,7,7,7),(0,0,0,17,0,0),(17,0,17,0,0,0),
           (0,1,0,17,7,0),(8,0,17,0,0,17))
    for step,counts in enumerate(cases):
      data=rng.integers(-128,128,(rows,k),dtype=np.int8)
      if step==5: data[:,::2],data[:,1::2]=-128,127
      payload=np.zeros(header+rows*k,dtype=np.uint8)
      payload[:rows*4]=np.ones(rows,dtype=np.float32).view(np.uint8)
      payload[rows*4:rows*4+groups*4]=np.asarray(counts,dtype=np.int32).view(np.uint8)
      payload[header:]=data.ravel().view(np.uint8)
      device=Device['ROCKCHIP']
      before_submit,before_tasks=device.submit_count,device.task_count
      actual=run(Tensor(payload,device='CPU').realize()).numpy()
      self.assertEqual(device.submit_count-before_submit,1)
      self.assertEqual(device.task_count-before_tasks,sum(count>0 for count in counts))
      offset=0
      for group,count in enumerate(counts):
        if count:
          expected=data[offset:offset+count].astype(np.int32)@weights[group].astype(np.int32).T
          np.testing.assert_array_equal(actual[offset:offset+count],expected)
          offset+=count
      self.assertEqual(offset,sum(counts))
      if step==1: run.captured=RockchipCapturedJit(run.captured)
    self.assertGreater(run.captured.native_runs,0)

    for invalid in ((17,17,9,0,0,0),(0,0,0,0,0,-1),(0,0,0,0,0,18)):
      payload[rows*4:rows*4+groups*4]=np.asarray(invalid,dtype=np.int32).view(np.uint8)
      before=device.submit_count
      with self.assertRaises(ValueError): run(Tensor(payload,device='CPU').realize())
      self.assertEqual(device.submit_count,before)
      self.assertFalse(device.hw.poisoned)
    valid=cases[-1]
    payload[rows*4:rows*4+groups*4]=np.asarray(valid,dtype=np.int32).view(np.uint8)
    run(Tensor(payload,device='CPU').realize())
    run.captured.native=None
    run.captured.warmed=False
    before=device.submit_count
    run(Tensor(payload,device='CPU').realize())
    self.assertEqual(device.submit_count-before,1)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_compact_matmul_selects_expert_scales_for_prefix_rows(self):
    rng=np.random.default_rng(20261032)
    groups,capacity,rows,n,k=4,8,12,64,64
    raw_weights=rng.integers(-128,128,(groups,n,k),dtype=np.int8)
    packed=raw_weights.reshape(groups,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    weight_scales=np.arange(1,groups*n+1,dtype=np.float32).reshape(groups,n)/100
    scales=Tensor(weight_scales,device='CPU').realize()
    counts=np.array((3,0,5,4),dtype=np.int32)
    x=rng.normal(0,.2,(rows,k)).astype(np.float32)
    codes,activation_scales=quantize(Tensor(x,device='CPU'),k)
    quantized=codes[round_up(rows*4,64):].numpy().view(np.int8).reshape(rows,k)
    counts_tensor=Tensor(counts,dtype=dtypes.int).to('CPU').realize()
    output=compact_matmul(Tensor(x,device='CPU'),counts_tensor,weight,scales,n,capacity)
    expected=[]
    offset=0
    for group,count in enumerate(counts):
      expected.append(((quantized[offset:offset+count].astype(np.int32)@raw_weights[group].astype(np.int32).T).astype(np.float32)
                       *activation_scales.numpy()[offset:offset+count,None])*weight_scales[group])
      offset+=count
    np.testing.assert_allclose(output.numpy(),np.concatenate(expected),rtol=0,atol=0)
    grouped_x=np.zeros((groups,capacity,k),dtype=np.float32)
    offset=0
    for group,count in enumerate(counts):
      grouped_x[group,:count]=x[offset:offset+count]
      offset+=count
    padded=grouped_matmul(Tensor(grouped_x,device='CPU'),counts_tensor,weight,scales,n)
    offset=0
    for group,count in enumerate(counts):
      np.testing.assert_array_equal(output.numpy()[offset:offset+count],padded.numpy()[group,:count])
      offset+=count


if __name__=='__main__': unittest.main()
