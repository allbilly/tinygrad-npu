import ctypes,os,struct,threading,unittest,weakref
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Device,Tensor,TinyJit
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip_grouped_int8 import pack_weight,raw_matmul,matmul
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer,RockchipGroupedMatmul
from tinygrad.runtime.autogen import rockchip as rk


def reference_quantize(data):
  scale=np.maximum(np.max(np.abs(data),axis=-1)/np.float32(127),np.float32(1e-8))
  codes=np.rint(data*(np.float32(1)/scale[...,None])).clip(-127,127).astype(np.int8)
  return codes,scale


class TestGroupedInt8(unittest.TestCase):
  @staticmethod
  def descriptor(rows=128,n=256,k=64,groups=40,tile=128):
    return SimpleNamespace(lib=struct.pack('<8I',rows,n,k,3,groups,tile,0x47524d4d,2))

  def test_invalid_descriptors_before_device_access(self):
    for changes in ({'rows':0},{'rows':129},{'groups':0},{'groups':129},{'k':32},{'k':4160},
                    {'n':33},{'n':4128},{'tile':129},{'k':4096,'tile':128}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):RockchipGroupedMatmul(None,self.descriptor(**changes))

  def test_group_header_and_task_ranges(self):
    class FakeDevice:
      def __init__(self):
        self.programs=weakref.WeakSet()
        self.storage=[]
      def allocate(self,size,flags=0):
        storage=ctypes.create_string_buffer(size)
        self.storage.append(storage)
        return RockchipBuffer(ctypes.addressof(storage),0x100000+len(self.storage)*0x100000,size,SimpleNamespace())
      def free(self,*args):pass
      def sync(self,*args):pass
      def check_domain(self,*args):pass
    for rows,k,n,tile in ((1,64,256,128),(17,256,64,128),(128,64,256,128),(128,2560,64,64)):
      with self.subTest(rows=rows,k=k,n=n):
        dev=FakeDevice()
        plan=RockchipGroupedMatmul(dev,self.descriptor(rows=rows,k=k,n=n,tile=tile))
        header=round_up(40*rows*4,64)
        self.assertEqual(plan.lhs_offset,header)
        self.assertTrue(plan.int8)
        lhs=RockchipBuffer(0,0x20000000,header+40*rows*k,SimpleNamespace())
        rhs=RockchipBuffer(0,0x30000000,40*n*k,SimpleNamespace())
        out=RockchipBuffer(0,0x10000000,40*rows*n*4,SimpleNamespace())
        with self.assertRaises(ValueError):plan.prepare(out,RockchipBuffer(0,lhs.dma,lhs.size-1,lhs.meta),rhs)
        plan.prepare(out,lhs,rhs)
        self.assertEqual(plan.core_mask,7)
        seen=[]
        descs=(rk.struct_rknpu_task*plan.tasks).from_address(plan.task.addr)
        for core,(begin,count) in enumerate(plan.subcores[2:]):
          expected=[(g,row) for g in range(core,40,3) for row in range(0,rows,tile)]
          self.assertEqual(count,len(expected))
          for offset,(group,row) in enumerate(expected):
            desc=descs[begin+offset]
            address=plan.cmd.addr+desc.regcmd_addr-plan.cmd.dma
            words=(ctypes.c_uint64*desc.regcfg_amount).from_address(address)
            regs={(word>>48,word&65535):(word>>16)&0xffffffff for word in words}
            self.assertEqual(regs[0x201,0x1070],lhs.dma+header+(group*rows+row)*k)
            self.assertEqual(regs[0x201,0x1110],rhs.dma+group*n*k)
            self.assertEqual(regs[0x1001,0x4020],out.dma+(group*rows+row)*n*4)
            self.assertEqual(regs[0x1001,0x4010],0x80000000)
            seen.append((group,row))
        self.assertEqual(set(seen),{(g,row) for g in range(40) for row in range(0,rows,tile)})
        self.assertEqual(len(seen),len(set(seen)))
        plan.close()

  def test_final_group_scale_rejected_before_submission(self):
    storage=(ctypes.c_float*40)(*([1.]*40))
    lhs=RockchipBuffer(ctypes.addressof(storage),0,160,SimpleNamespace())
    out=rhs=RockchipBuffer(0,0,0,SimpleNamespace())
    plan=RockchipGroupedMatmul.__new__(RockchipGroupedMatmul)
    plan.dev=SimpleNamespace(lock=threading.Lock(),check_healthy=lambda:None)
    plan.bufs=(out,lhs,rhs)
    plan.groups,plan.m,plan.int8=40,1,True
    with patch('tinygrad.runtime.ops_rockchip._finite_program',return_value=SimpleNamespace()), \
         patch.object(rk,'DRM_IOCTL_RKNPU_SUBMIT') as submit:
      for bad in (0.,-1.,float('inf'),float('nan')):
        storage[-1]=bad
        with self.assertRaises(ValueError):plan(out,lhs,rhs)
      submit.assert_not_called()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_raw_integer_goldens_native_and_last_scale_guard(self):
    rng=np.random.default_rng(271107)
    for k,n in ((64,256),(256,64)):
      values=rng.integers(-128,128,(40,n,k),dtype=np.int8)
      packed=values.reshape(40,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
      weight=Tensor(packed,device='ROCKCHIP').realize()
      for rows in (1,17,128):
        with self.subTest(rows=rows,k=k,n=n):
          header=round_up(40*rows*4,64)
          run=TinyJit(lambda payload:raw_matmul(payload,weight,rows,n).realize())
          for step in range(6):
            data=rng.integers(-128,128,(40,rows,k),dtype=np.int8)
            if step==5:data[:,:,::2],data[:,:,1::2]=-128,127
            payload=np.zeros(header+data.size,dtype=np.uint8)
            payload[:40*rows*4]=np.ones(40*rows,dtype=np.float32).view(np.uint8)
            payload[header:]=data.ravel().view(np.uint8)
            expected=data.astype(np.int32)@values.astype(np.int32).swapaxes(-1,-2)
            before=Device['ROCKCHIP'].submit_count
            actual=run(Tensor(payload,device='CPU').realize()).numpy()
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
            if step==1:run.captured=RockchipCapturedJit(run.captured)
          self.assertGreater(run.captured.native_runs,0)
          payload[(40*rows-1)*4:40*rows*4]=np.array([np.inf],dtype=np.float32).view(np.uint8)
          before=Device['ROCKCHIP'].submit_count
          with self.assertRaises(ValueError):run(Tensor(payload,device='CPU').realize())
          self.assertEqual(Device['ROCKCHIP'].submit_count,before)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_prepared_projection_FP32_reference_and_native(self):
    rng=np.random.default_rng(271108)
    for k,n in ((64,256),(256,64)):
      values=rng.normal(0,.1,(40,n,k)).astype(np.float16)
      w,sw=pack_weight(Tensor(values,device='CPU'))
      w.realize()
      qw,ws=reference_quantize(values.astype(np.float32))
      for rows in (1,17,128):
        with self.subTest(rows=rows,k=k,n=n):
          run=TinyJit(lambda x:matmul(x,w,sw,n).contiguous().realize())
          for step in range(6):
            data=rng.normal(0,.2,(40,rows,k)).astype(np.float32)
            qa,ascale=reference_quantize(data)
            integer=qa.astype(np.int32)@qw.astype(np.int32).swapaxes(-1,-2)
            expected=(integer.astype(np.float32)*ascale[:,:,None])*ws[:,None,:]
            actual=run(Tensor(data,device='CPU').realize()).numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=2e-6)
            if step==1:run.captured=RockchipCapturedJit(run.captured)
          self.assertGreater(run.captured.native_runs,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_residual_three_products_native_and_one_submission(self):
    rng=np.random.default_rng(271129)
    def residual(x,codes,scales):
      return (x.astype(np.float64)-codes.astype(np.float64)*scales[...,None].astype(np.float64)).astype(np.float32)
    for k,n in ((64,256),(256,64)):
      values=rng.normal(0,.1,(40,n,k)).astype(np.float16).astype(np.float32)
      w,sw=pack_weight(Tensor(values,device='CPU'),residual=True)
      w.realize()
      w0,s0=reference_quantize(values)
      w1,s1=reference_quantize(residual(values,w0,s0))
      self.assertEqual(w.shape[0],120)
      for rows in (1,17,128):
        with self.subTest(rows=rows,k=k,n=n):
          run=TinyJit(lambda x:matmul(x,w,sw,n,residual=True).contiguous().realize())
          for step in range(6):
            data=rng.normal(0,.2,(40,rows,k)).astype(np.float32)
            if step%2:data[:,:,0]*=32
            a0,sa0=reference_quantize(data)
            a1,sa1=reference_quantize(residual(data,a0,sa0))
            parts=[]
            for a,sa,weight,scale in ((a0,sa0,w0,s0),(a1,sa1,w0,s0),(a0,sa0,w1,s1)):
              integer=a.astype(np.int32)@weight.astype(np.int32).swapaxes(-1,-2)
              parts.append((integer.astype(np.float32)*sa[:,:,None])*scale[:,None,:])
            expected=(parts[0]+parts[1])+parts[2]
            before=Device['ROCKCHIP'].submit_count
            actual=run(Tensor(data,device='CPU').realize()).numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=2e-6)
            self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
            if step==1:run.captured=RockchipCapturedJit(run.captured)
          self.assertGreater(run.captured.native_runs,0)


if __name__=='__main__':unittest.main()
