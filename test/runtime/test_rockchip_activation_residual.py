import ctypes, os, struct, threading, unittest, weakref
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import Target, round_up
from tinygrad.llm.kernels.rockchip_bucketed_int8 import compact_matmul
from tinygrad.llm.kernels.rockchip_grouped_int8 import _remainder, _dequantize_activation_residual, _dequantize_weight_residual
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_routed_int8 import matmul as routed_matmul, raw_matmul as routed_raw
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipBucketedMatmul, RockchipRoutedMatmul
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


def descriptor_routed(rows=2,n=64,k=64,cores=3,groups=6,tile=None,experts=8,version=3):
  tile=rows if tile is None else tile
  return TinyELF(struct.pack('<9I',rows,n,k,cores,groups,tile,0x47524d4d,version,experts),'routed',Target('ROCKCHIP'),())


def descriptor_bucketed(capacity=34,n=64,k=64,cores=3,groups=4,tile=17,rows=80,version=6):
  return TinyELF(struct.pack('<9I',capacity,n,k,cores,groups,tile,0x47524d4d,version,rows),'bucketed',Target('ROCKCHIP'),())


def reference_quantize(values):
  scale=np.maximum(np.max(np.abs(values),axis=-1)/np.float32(127),np.float32(1e-8)).astype(np.float32)
  codes=np.rint(values*(np.float32(1)/scale[...,None])).clip(-127,127).astype(np.int8)
  return codes,scale


def reference_residual(values):
  codes,scales=reference_quantize(values)
  residual=(values.astype(np.float64)-codes.astype(np.float64)*scales[...,None].astype(np.float64)).astype(np.float32)
  extra,extra_scales=reference_quantize(residual)
  return codes,scales,extra,extra_scales


def reference_three_products(raw,extra,scales,weights,residual):
  first=raw[:,0].astype(np.float32)*scales[:,0,None]
  second=raw[:,1].astype(np.float32)*scales[:,1,None]
  combined=(first.astype(np.float64)*weights.astype(np.float64)+(second*weights).astype(np.float64)).astype(np.float32)
  third=extra.astype(np.float32)*scales[:,0,None]
  return (third.astype(np.float64)*residual.astype(np.float64)+combined.astype(np.float64)).astype(np.float32)


class TestActivationResidualCPU(unittest.TestCase):
  def test_weight_residual_dequantization_matches_independent_fma_oracle(self):
    one=np.float32(1)
    high=np.nextafter(one,np.float32(np.inf))
    low=np.nextafter(one,np.float32(0))
    cases=[(np.array([[[0],[-1]]],dtype=np.int32),np.ones((1,1),dtype=np.int32),
            np.array([[high,one]],dtype=np.float32),np.array([[low]],dtype=np.float32),np.array([[low]],dtype=np.float32))]
    rng=np.random.default_rng(20261107)
    cases.append((rng.integers(-20000000,20000001,(7,2,19),dtype=np.int32),rng.integers(-20000000,20000001,(7,19),dtype=np.int32),
                  rng.uniform(.001,.2,(7,2)).astype(np.float32),rng.uniform(.001,.1,(7,19)).astype(np.float32),
                  rng.uniform(.00001,.001,(7,19)).astype(np.float32)))
    for index,args in enumerate(cases):
      expected=reference_three_products(*args)
      if index==0:
        separate=np.float32(np.float32(high*low)-low)
        self.assertNotEqual(expected.item(),separate)
      actual=_dequantize_weight_residual(*(Tensor(v,device='CPU').realize() for v in args)).numpy()
      np.testing.assert_array_equal(actual,expected)

  def test_weight_residual_dequantization_rejects_invalid_operands(self):
    raw=Tensor.zeros(2,2,3,device='CPU',dtype=dtypes.int)
    extra=Tensor.zeros(2,3,device='CPU',dtype=dtypes.int)
    scales=Tensor.ones(2,2,device='CPU',dtype=dtypes.float)
    weights=Tensor.ones(2,3,device='CPU',dtype=dtypes.float)
    for bad_extra,bad_residual in ((extra[:,:1],weights),(extra.float(),weights),(extra,weights[:,:1]),(extra,weights.int())):
      with self.assertRaises(ValueError):_dequantize_weight_residual(raw,bad_extra,scales,weights,bad_residual)

  def test_residual_dequantization_matches_independent_fma_oracle(self):
    one=np.float32(1)
    cases=[(np.array([[[1],[-1]]],dtype=np.int32),
            np.array([[np.nextafter(one,np.float32(np.inf)),one]],dtype=np.float32),
            np.array([[np.nextafter(one,np.float32(0))]],dtype=np.float32))]
    rng=np.random.default_rng(20261027)
    cases.append((rng.integers(-20000000,20000001,(7,2,19),dtype=np.int32),
                  rng.uniform(.001,.2,(7,2)).astype(np.float32),rng.uniform(.001,.1,(7,19)).astype(np.float32)))
    for index,(raw,scales,weights) in enumerate(cases):
      first=raw[:,0].astype(np.float32)*scales[:,0,None]
      second=raw[:,1].astype(np.float32)*scales[:,1,None]
      rounded_second=second*weights
      expected=(first.astype(np.float64)*weights.astype(np.float64)+rounded_second.astype(np.float64)).astype(np.float32)
      if index==0:self.assertFalse(np.array_equal(expected,first*weights+rounded_second))
      actual=_dequantize_activation_residual(*(Tensor(v,device='CPU').realize() for v in (raw,scales,weights))).numpy()
      np.testing.assert_array_equal(actual,expected)

  def test_residual_dequantization_rejects_invalid_operands(self):
    raw=Tensor.zeros(2,2,3,device='CPU',dtype=dtypes.int)
    scales=Tensor.ones(2,2,device='CPU',dtype=dtypes.float)
    weights=Tensor.ones(2,3,device='CPU',dtype=dtypes.float)
    for args in ((raw[:,:1],scales,weights),(raw,scales[:,:1],weights),(raw,scales,weights[:,:1]),
                 (raw.float(),scales,weights),(raw,scales.half(),weights),(raw,scales,weights.half())):
      with self.subTest(shapes=[v.shape for v in args],dtypes=[v.dtype for v in args]),self.assertRaises(ValueError):
        _dequantize_activation_residual(*args)

  def test_remainder_is_single_rounded_fma(self):
    opened=set(Device._opened_devices)
    rng=np.random.default_rng(20261104)
    for rows in (1,17):
      values=rng.normal(0,.15,(rows,64)).astype(np.float32)
      values[:,0],values[:,1],values[:,2],values[:,3]=0.0,-0.0,1e-10,-1e-10
      x=Tensor(values,device='CPU').realize()
      payload,scales=quantize(x,64)
      codes=payload[round_up(rows*4,64):].bitcast(dtypes.int8).reshape(rows,64)
      actual=_remainder(x,codes,scales).numpy()
      expected=(values.astype(np.float64)-codes.numpy().astype(np.float64)*scales.numpy()[:,None].astype(np.float64)).astype(np.float32)
      np.testing.assert_array_equal(actual,expected)
    self.assertFalse(any(device.split(':')[0]=='ROCKCHIP' for device in Device._opened_devices-opened))


class TestActivationResidualPlans(unittest.TestCase):
  def test_routed_two_row_header_and_weight_patches(self):
    dev=FakeDevice()
    plan=RockchipRoutedMatmul(dev,descriptor_routed())
    self.assertEqual(plan.selector_offset,6*2*4)
    self.assertEqual(plan.lhs_offset,round_up((6*2+6)*4,64))
    out=dev.allocate(6*2*64*4)
    lhs=dev.allocate(plan.lhs_offset+6*2*64)
    rhs=dev.allocate(8*64*64)
    plan.prepare(out,lhs,rhs)
    self.assertEqual(len(plan.weight_patches),6)
    ids=(0,7,1,1,6,0)
    for group,expert in enumerate(ids):
      ctypes.c_int32.from_address(lhs.addr+plan.selector_offset+group*4).value=expert
    plan.update_inputs(lhs,rhs)
    for address,group in plan.weight_patches:
      command=ctypes.c_uint64.from_address(address).value
      self.assertEqual((command>>16)&0xffffffff,rhs.dma+ids[group]*plan.n*plan.k)
    plan.close()

  def test_compact_chunked_abi6_prefix_tasks_and_invalid_counts(self):
    dev=FakeDevice()
    plan=RockchipBucketedMatmul(dev,descriptor_bucketed())
    self.assertEqual(plan.task_slots,4*2)
    self.assertEqual(plan.storage_rows,80)
    self.assertEqual(plan.count_offset,80*4)
    self.assertEqual(plan.lhs_offset,round_up((80+4)*4,64))
    out=dev.allocate(80*64*4)
    lhs=dev.allocate(plan.lhs_offset+80*64)
    rhs=dev.allocate(4*64*64)
    plan.prepare(out,lhs,rhs)
    cases=((34,0,1,17),(17,34,0,1),(0,0,17,1),(0,0,0,34))
    for counts in cases:
      self.assertLessEqual(sum(counts),80)
      for group,count in enumerate(counts):ctypes.c_int32.from_address(lhs.addr+plan.count_offset+group*4).value=count
      plan.update_inputs(lhs,rhs)
      chunks=[(group,first,min(17,count-first)) for group,count in enumerate(counts) for first in range(0,count,17)]
      active=3 if len(chunks)>=3 else 1
      expected=[chunk for core in range(active) for ordinal,chunk in enumerate(chunks) if ordinal%active==core]
      self.assertEqual(plan.tasks,len(chunks))
      self.assertEqual(plan.core_mask,(1<<active)-1)
      self.assertEqual(sum(count for _,count in plan.subcores),len(chunks))
      stride=round_up(len(int8_body(1,64,64,0,0,0,64))+4,2)
      prefix=0
      offsets=[]
      for count in counts:
        offsets.append(prefix)
        prefix+=count
      descs=(rk.struct_rknpu_task*plan.task_slots).from_address(plan.task.addr)
      for slot,(group,first,count) in enumerate(expected):
        desc=descs[slot]
        words=(ctypes.c_uint64*desc.regcfg_amount).from_address(plan.cmd.addr+slot*stride*8)
        expected_body=int8_body(count,64,64,lhs.dma+plan.lhs_offset+(offsets[group]+first)*64,
          rhs.dma+group*64*64,out.dma+(offsets[group]+first)*64*4,64)
        self.assertEqual(tuple(words),expected_body)
        self.assertEqual(desc.regcfg_amount,len(expected_body))
    before_commands=ctypes.string_at(plan.cmd.addr,plan.cmd.size)
    before_state=tuple(plan.dynamic_state)
    for invalid in ((34,34,17,0),(35,0,0,0),(-1,1,0,0),(0,0,0,0)):
      for group,count in enumerate(invalid):ctypes.c_int32.from_address(lhs.addr+plan.count_offset+group*4).value=count
      with self.assertRaises(ValueError):plan.update_inputs(lhs,rhs)
      self.assertEqual(ctypes.string_at(plan.cmd.addr,plan.cmd.size),before_commands)
      self.assertEqual(tuple(plan.dynamic_state),before_state)
    plan.close()


class TestActivationResidualHardware(unittest.TestCase):
  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_routed_two_row_integer_goldens_and_native(self):
    for groups in (2,6):
      for n,k in ((64,64),(1280,896),(1792,1280)):
        with self.subTest(groups=groups,n=n,k=k):self._check_routed_geometry(groups,n,k)

  def _check_routed_geometry(self,groups,n,k):
    rng=np.random.default_rng(20261105)
    experts=8
    weights=rng.integers(-128,128,(experts,n,k),dtype=np.int8)
    weights[0,:,::2],weights[-1,:,1::2]=-128,127
    packed=weights.reshape(experts,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    header=round_up((groups*2+groups)*4,64)
    run=TinyJit(lambda payload:routed_raw(payload,weight,groups,n,rows=2).realize())
    ids_cases=((0,1,2,3,4,7),(7,7,1,1,0,0),(0,6,2,2,7,3),(3,0,4,7,1,6),(2,2,2,5,5,5),(7,0,6,1,5,2))
    for step,ids in enumerate(ids_cases):
      ids=ids[:groups]
      values=rng.integers(-128,128,(groups,2,k),dtype=np.int8)
      if step==5:values[:,:,::2],values[:,:,1::2]=-128,127
      payload=np.zeros(header+groups*2*k,dtype=np.uint8)
      payload[:groups*2*4]=np.ones(groups*2,dtype=np.float32).view(np.uint8)
      payload[groups*2*4:groups*3*4]=np.asarray(ids,dtype=np.int32).view(np.uint8)
      payload[header:]=values.ravel().view(np.uint8)
      expected=np.stack([values[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate(ids)])
      device=Device['ROCKCHIP']
      before=device.submit_count
      actual=run(Tensor(payload,device='CPU').realize()).numpy()
      np.testing.assert_array_equal(actual,expected)
      self.assertEqual(device.submit_count-before,1)
      if step==1:run.captured=RockchipCapturedJit(run.captured)
    self.assertGreater(run.captured.native_runs,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_routed_and_compact_high_level_activation_residual_goldens(self):
    for full in (False,True):
      for n,k in ((64,64),(1280,896),(1792,1280)):
        with self.subTest(n=n,k=k,weight_residual=full):self._check_high_level_geometry(n,k,full)

  def _check_high_level_geometry(self,n,k,full):
    rng=np.random.default_rng(20261106)
    experts=8
    weights=rng.integers(-128,128,(experts,n,k),dtype=np.int8)
    packed=weights.reshape(experts,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight=Tensor(packed,device='ROCKCHIP').realize()
    weight_scales=rng.uniform(.005,.03,(experts,n)).astype(np.float32)
    scales=Tensor(weight_scales,device='CPU').realize()
    weights1=rng.integers(-128,128,(experts,n,k),dtype=np.int8)
    packed1=weights1.reshape(experts,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
    weight1=Tensor(packed1,device='ROCKCHIP').realize()
    weight_scales1=rng.uniform(.00001,.0001,(experts,n)).astype(np.float32)
    scales1=Tensor(weight_scales1,device='CPU').realize()
    residual=(weight1,scales1) if full else None

    def golden(values,ids):
      a0,s0,a1,s1=reference_residual(values)
      dots0=np.stack([a0[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate(ids)]).astype(np.float32)
      dots1=np.stack([a1[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate(ids)]).astype(np.float32)
      extra=np.stack([a0[g].astype(np.int32)@weights1[e].astype(np.int32).T for g,e in enumerate(ids)]).astype(np.float32)
      if full:return reference_three_products(np.stack((dots0,dots1),axis=1),extra,np.stack((s0,s1),axis=1),
                                               weight_scales[np.asarray(ids)],weight_scales1[np.asarray(ids)])
      return (dots0*s0[:,None])*weight_scales[np.asarray(ids)] + (dots1*s1[:,None])*weight_scales[np.asarray(ids)]

    ids_cases=((0,1,2,3),(2,2,1,1),(7,0,7,0),(4,4,4,3),(6,1,5,2),(7,7,0,0))
    route=TinyJit(lambda x,ids:routed_matmul(x,ids,weight,scales,n,activation_residual=True,weight_residual=residual).realize())
    for step,ids in enumerate(ids_cases):
      values=rng.normal(0,.15,(4,k)).astype(np.float32)
      if step==5:values[:,::2],values[:,1::2]=-.5,.5
      before=Device['ROCKCHIP'].submit_count
      actual=route(Tensor(values,device='CPU').realize(),Tensor(ids,dtype=dtypes.int).to('CPU').realize()).numpy()
      np.testing.assert_allclose(actual,golden(values,np.asarray(ids)),rtol=2e-6,atol=2e-6)
      self.assertEqual(Device['ROCKCHIP'].submit_count-before,2 if full else 1)
      if step==1:route.captured=RockchipCapturedJit(route.captured)
    self.assertGreater(route.captured.native_runs,0)

    groups,capacity,rows=4,65,100
    small_weight=Tensor(packed[:groups].copy(),device='ROCKCHIP').realize()
    small_scales=scales[:groups]
    small_weight1=Tensor(packed1[:groups].copy(),device='ROCKCHIP').realize()
    small_residual=(small_weight1,scales1[:groups]) if full else None
    compact=TinyJit(lambda x,c:compact_matmul(x,c,small_weight,small_scales,n,capacity,
                                            activation_residual=True,weight_residual=small_residual).realize())
    compact_cases=((65,35,0,0),(0,65,35,0),(35,0,0,65),(34,33,33,0),(50,0,20,30),(0,0,65,35))
    for step,counts in enumerate(compact_cases):
      compact_values=rng.normal(0,.15,(rows,k)).astype(np.float32)
      if step==5:compact_values[:,::2],compact_values[:,1::2]=-.25,.25
      expected=[]
      offset=0
      for group,count in enumerate(counts):
        if count:
          segment=compact_values[offset:offset+count]
          a0,s0,a1,s1=reference_residual(segment)
          dot0=(a0.astype(np.int32)@weights[group].astype(np.int32).T).astype(np.float32)
          dot1=(a1.astype(np.int32)@weights[group].astype(np.int32).T).astype(np.float32)
          if full:
            dot2=(a0.astype(np.int32)@weights1[group].astype(np.int32).T).astype(np.float32)
            expected.append(reference_three_products(np.stack((dot0,dot1),axis=1),dot2,np.stack((s0,s1),axis=1),
              np.broadcast_to(weight_scales[group],dot0.shape),np.broadcast_to(weight_scales1[group],dot0.shape)))
          else:expected.append((dot0*s0[:,None])*weight_scales[group]+(dot1*s1[:,None])*weight_scales[group])
          offset+=count
      before=Device['ROCKCHIP'].submit_count
      actual=compact(Tensor(compact_values,device='CPU').realize(),Tensor(counts,dtype=dtypes.int).to('CPU').realize()).numpy()
      np.testing.assert_allclose(actual,np.concatenate(expected),rtol=2e-6,atol=2e-6)
      self.assertEqual(Device['ROCKCHIP'].submit_count-before,2 if full else 1)
      if step==1:compact.captured=RockchipCapturedJit(compact.captured)
    self.assertGreater(compact.captured.native_runs,0)


if __name__=='__main__':unittest.main()
