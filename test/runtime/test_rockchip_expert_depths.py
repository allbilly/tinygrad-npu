import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import round_up
from tinygrad.llm.kernels import rockchip_grouped_int8 as grouped
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_grouped_int8 import raw_matmul as grouped_raw
from tinygrad.llm.kernels.rockchip_routed_int8 import raw_matmul as routed_raw
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.support import rockchip_int8 as rk8


def commands(words):
  return {(word>>48,word&0xffff):(word>>16)&0xffffffff for word in words}


def raw_codes_reference(x):
  scale=np.maximum(np.max(np.abs(x),axis=-1)/np.float32(127),np.float32(1e-8)).astype(np.float32)
  codes=np.rint(x*(np.float32(1)/scale[...,None])).clip(-127,127).astype(np.int8)
  return codes,scale


class TestRockchipExpertDepths(unittest.TestCase):
  def test_supported_and_padded_depth_modes(self):
    self.assertEqual(rk8.EXPERT_DEPTHS,frozenset((896,1280)))
    for depth in rk8.EXPERT_DEPTHS:
      self.assertTrue(rk8.supported_depth(depth))
      with patch.object(rk8,'getenv',side_effect=lambda key,default: 0):
        self.assertEqual(rk8.padded_depth(depth),round_up(depth,512))
      with patch.object(rk8,'getenv',side_effect=lambda key,default: 1 if key=='ONPU_INT8_EXPERT_DEPTHS' else default):
        self.assertEqual(rk8.padded_depth(depth),depth)
    self.assertEqual(rk8.padded_depth(768),768)

  def test_expert_depth_register_recipes(self):
    for k in sorted(rk8.EXPERT_DEPTHS):
      for rows in (1,17,128):
        for n in (128,512):
          with self.subTest(k=k,rows=rows,n=n):
            regs=commands(rk8.int8_body(rows,n,k,0x2000,0x4000,0x6000,n))
            self.assertEqual(regs[0x201,0x1024],((k-1)<<16)|k)
            self.assertEqual(regs[0x201,0x1070],0x2000)
            self.assertEqual(regs[0x201,0x1110],0x4000)
            self.assertEqual(regs[0x1001,0x4020],0x6000)
            self.assertEqual(regs[0x201,0x1020],(1<<16)|rows)
            for address in (regs[0x201,0x1070],regs[0x201,0x1110],regs[0x1001,0x4020]):
              self.assertEqual(address%32,0)

  def test_zero_padding_removal_preserves_quantization(self):
    rng=np.random.default_rng(20261027)
    for rows,depth in ((1,896),(17,896),(1,1280),(17,1280)):
      x=rng.normal(0,.2,(rows,depth)).astype(np.float32)
      x[:,0],x[:,1],x[:,2],x[:,3]=0.0,-0.0,np.float32(1e-10),-np.float32(1e-10)
      x[:,4]=np.float32(0.5/127)*np.max(np.abs(x),axis=1)
      expected,scales=raw_codes_reference(x)
      for enabled in (0,1):
        with patch.object(rk8,'getenv',side_effect=lambda key,default,_enabled=enabled:
                          _enabled if key=='ONPU_INT8_EXPERT_DEPTHS' else default):
          physical=rk8.padded_depth(depth)
          payload,actual_scales=quantize(
            Tensor(x,device='CPU'),physical)
        header=round_up(rows*4,64)
        all_codes=payload[header:].numpy().view(np.int8).reshape(rows,physical)
        code=all_codes[:,:depth]
        np.testing.assert_array_equal(code,expected)
        np.testing.assert_array_equal(actual_scales.numpy(),scales)
        if physical>depth: np.testing.assert_array_equal(all_codes[:,depth:],0)

  def test_weight_parts_depth_policy_and_zero_tail(self):
    for depth in (896,1280):
      rng=np.random.default_rng(20261028)
      weights=rng.normal(0,.1,(3,35,depth)).astype(np.float32)
      expected,scales=raw_codes_reference(weights.reshape(-1,depth))
      before=set(Device._opened_devices)
      for enabled in (0,1):
        with patch.object(rk8,'getenv',side_effect=lambda key,default,_enabled=enabled: _enabled if key=='ONPU_INT8_EXPERT_DEPTHS' else default), \
             patch.object(grouped,'padded_depth',side_effect=lambda k,_enabled=enabled: k if _enabled else round_up(k,512)):
          packed,actual_scales,codes=grouped._weight_parts(Tensor(weights,device='CPU'))
        physical=depth if enabled else round_up(depth,512)
        self.assertEqual(codes.shape,(3,64,physical))
        np.testing.assert_array_equal(codes.numpy()[:,:35,:depth].reshape(-1,depth),expected)
        np.testing.assert_array_equal(codes.numpy()[:,35:,:],0)
        np.testing.assert_array_equal(actual_scales.numpy()[:,35:],1)
        np.testing.assert_array_equal(actual_scales.numpy()[:,:35],scales.reshape(3,35))
        if physical>depth: np.testing.assert_array_equal(codes.numpy()[:,:,depth:],0)
        self.assertEqual(packed.shape,(3,2,physical//32,32,32))
      self.assertFalse(any(d.split(':')[0]=='ROCKCHIP' for d in set(Device._opened_devices)-before))

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_grouped_expert_depth_raw_integer_native_and_ordinary(self):
    rng=np.random.default_rng(20261029)
    groups=3
    for k in (896,1280):
      for rows,n in ((17,128),(128,512)):
        weights=rng.integers(-128,128,(groups,n,k),dtype=np.int8)
        weights[0,:,::2],weights[-1,:,1::2]=-128,127
        packed=weights.reshape(groups,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
        weight=Tensor(packed,device='ROCKCHIP').realize()
        header=round_up(groups*rows*4,64)
        run=TinyJit(lambda payload:grouped_raw(payload,weight,rows,n).realize())
        for step in range(6):
          data=rng.integers(-128,128,(groups,rows,k),dtype=np.int8)
          if step==5: data[:,:,::2],data[:,:,1::2]=-128,127
          payload=np.zeros(header+data.size,dtype=np.uint8)
          payload[:groups*rows*4]=np.ones(groups*rows,dtype=np.float32).view(np.uint8)
          payload[header:]=data.ravel().view(np.uint8)
          expected=data.astype(np.int32)@weights.astype(np.int32).swapaxes(-1,-2)
          actual=run(Tensor(payload,device='CPU').realize()).numpy()
          np.testing.assert_array_equal(actual,expected)
          if step==1: run.captured=RockchipCapturedJit(run.captured)
        self.assertGreater(run.captured.native_runs,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_routed_expert_depth_raw_integer_native_and_ordinary(self):
    rng=np.random.default_rng(20261030)
    groups,experts=6,8
    for k,n in ((896,1280),(1280,1792)):
      weights=rng.integers(-128,128,(experts,n,k),dtype=np.int8)
      weights[0,:,::2],weights[-1,:,1::2]=-128,127
      packed=weights.reshape(experts,n//32,32,k//32,32).transpose(0,1,3,2,4).copy()
      weight=Tensor(packed,device='ROCKCHIP').realize()
      data_offset=round_up(groups*8,64)
      run=TinyJit(lambda payload:routed_raw(payload,weight,groups,n).realize())
      ids_cases=((0,1,2,3,4,7),(2,2,1,1,0,0),(7,6,5,4,3,2),(0,7,0,7,1,6),(3,3,3,3,3,3),(7,0,6,1,5,2))
      for step,ids in enumerate(ids_cases):
        data=rng.integers(-128,128,(groups,k),dtype=np.int8)
        if step==5: data[:,::2],data[:,1::2]=-128,127
        payload=np.zeros(data_offset+data.size,dtype=np.uint8)
        payload[:groups*4]=np.ones(groups,dtype=np.float32).view(np.uint8)
        payload[groups*4:groups*8]=np.asarray(ids,dtype=np.int32).view(np.uint8)
        payload[data_offset:]=data.ravel().view(np.uint8)
        expected=np.stack([data[g].astype(np.int32)@weights[e].astype(np.int32).T for g,e in enumerate(ids)])
        np.testing.assert_array_equal(run(Tensor(payload,device='CPU').realize()).numpy(),expected)
        if step==1: run.captured=RockchipCapturedJit(run.captured)
      self.assertGreater(run.captured.native_runs,0)


if __name__=='__main__': unittest.main()
