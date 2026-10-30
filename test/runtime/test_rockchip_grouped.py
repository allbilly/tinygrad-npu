import ctypes,os,struct,unittest,weakref
from types import SimpleNamespace
import numpy as np
from tinygrad import Device,Tensor,TinyJit,UOp
from tinygrad.llm.kernels.rockchip import grouped_matmul,pack_grouped_weight
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer,RockchipGroupedMatmul
from tinygrad.runtime.autogen import rockchip as rk


class TestGroupedMatmul(unittest.TestCase):
  @staticmethod
  def descriptor(rows=128,n=256,k=64,cores=3,groups=40,tile=64,marker=0x47524d4d,version=1):
    return SimpleNamespace(lib=struct.pack('<8I',rows,n,k,cores,groups,tile,marker,version))

  def test_invalid_descriptors_before_device_access(self):
    for changes in ({'rows':0},{'rows':129},{'groups':0},{'groups':129},{'n':33},{'n':4128},{'k':0},{'k':65},
                    {'k':4128},{'cores':2},{'tile':65},{'k':4096,'tile':64},{'marker':0},{'version':3}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):RockchipGroupedMatmul(None,self.descriptor(**changes))

  def test_invalid_weights_before_device_access(self):
    for shape in ((2,64),(129,64,64),(40,4097,64),(40,64,4097)):
      with self.subTest(shape=shape),self.assertRaises(ValueError):pack_grouped_weight(Tensor.empty(*shape,device='CPU'))

  def test_chains_cover_groups_and_stop_at_core_boundaries(self):
    class FakeDevice:
      def __init__(self):
        self.programs=weakref.WeakSet()
        self.storage=[]
      def allocate(self,size,flags=0):
        storage=ctypes.create_string_buffer(size)
        self.storage.append(storage)
        return RockchipBuffer(ctypes.addressof(storage),0x100000+len(self.storage)*0x100000,size,SimpleNamespace())
      def free(self,buf):pass
      def sync(self,*args):pass
      def check_domain(self,*args):pass
    for groups,rows,cores in ((40,128,3),(40,17,3),(40,1,1),(2,17,3),(3,65,3)):
      with self.subTest(groups=groups,rows=rows,cores=cores):
        dev=FakeDevice()
        plan=RockchipGroupedMatmul(dev,self.descriptor(rows=rows,groups=groups,cores=cores))
        out=RockchipBuffer(0,0x10000000,groups*rows*256*4,SimpleNamespace())
        lhs=RockchipBuffer(0,0x20000000,groups*rows*64*2,SimpleNamespace())
        rhs=RockchipBuffer(0,0x30000000,groups*256*64*2,SimpleNamespace())
        with self.assertRaises(ValueError):plan.prepare(out,lhs,RockchipBuffer(0,rhs.dma,rhs.size-1,rhs.meta))
        plan.prepare(out,lhs,rhs)
        active=3 if groups>=3 and cores==3 else 1
        self.assertEqual(plan.core_mask,(1<<active)-1)
        ranges=plan.subcores[2:] if active==3 else plan.subcores
        stride=plan.cmd.size//plan.tasks
        seen=[]
        for core,(start,count) in enumerate(ranges):
          expected=[(group,row) for group in range(core,groups,active) for row in range(0,rows,64)]
          self.assertEqual(count,len(expected))
          for relative,(group,row) in enumerate(expected):
            index=start+relative
            words=list((ctypes.c_uint64*(stride//8)).from_address(plan.cmd.addr+index*stride))
            regs={(word>>48,word&0xffff):(word>>16)&0xffffffff for word in words}
            self.assertEqual(regs[0x201,rk.REG_CNA_FEATURE_DATA_ADDR],lhs.dma+(group*rows+row)*64*2)
            self.assertEqual(regs[0x201,rk.REG_CNA_DCOMP_ADDR0],rhs.dma+group*256*64*2)
            self.assertEqual(regs[0x1001,rk.REG_DPU_DST_BASE_ADDR],out.dma+(group*rows+row)*256*4)
            pc=(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS)
            if relative+1<count:self.assertEqual(regs[pc],plan.cmd.dma+(index+1)*stride)
            else:self.assertNotIn(pc,regs)
            seen.append((group,row))
        self.assertEqual(set(seen),{(group,row) for group in range(groups) for row in range(0,rows,64)})
        self.assertEqual(len(seen),len(set(seen)))
        plan.close()

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_distinct_weights_changed_inputs_native_and_nonfinite_tail(self):
    rng=np.random.default_rng(271031)
    for k,n in ((64,256),(256,64)):
      values=rng.normal(0,.1,(40,n,k)).astype(np.float16)
      weight=pack_grouped_weight(Tensor(values,device='CPU')).realize()
      for rows in (1,17,128):
        with self.subTest(k=k,n=n,rows=rows):
          run=TinyJit(lambda x:grouped_matmul(x,weight,n).contiguous().realize())
          for index in range(6):
            data=rng.normal(0,.2,(40,rows,k)).astype(np.float32)
            expected=data.astype(np.float16).astype(np.float32)@values.astype(np.float32).swapaxes(-1,-2)
            before=Device['ROCKCHIP'].submit_count
            actual=run(Tensor(data,device='CPU').realize()).numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-3)
            self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
            if index==1:run.captured=RockchipCapturedJit(run.captured)
          self.assertGreater(run.captured.native_runs,0)
          for invalid in (float('nan'),float('inf')):
            data[-1,-1,-1]=invalid
            before=Device['ROCKCHIP'].submit_count
            with self.assertRaises(ValueError):run(Tensor(data,device='CPU').realize())
            self.assertEqual(Device['ROCKCHIP'].submit_count,before)
            self.assertFalse(Device['ROCKCHIP'].poisoned)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_symbolic_model_hook_padding_native_and_trained_attention_dimensions(self):
    from tinygrad.llm.kernels.rockchip_mla_projections import MLAProjections
    rng=np.random.default_rng(271103)
    key=Tensor(rng.normal(0,.1,(40,256,64)).astype(np.float16),device='CPU').realize()
    value=Tensor(rng.normal(0,.1,(40,64,256)).astype(np.float16),device='CPU').realize()
    hook=MLAProjections(key,value,prefill=True,decode=True)
    hook.prepare()
    for capacity in (128,1):
      hook.phase='prefill' if capacity==128 else 'decode'
      rowsvar=UOp.variable('grouped_hook_rows',1,capacity)
      for project,weight in ((hook.query,key),(hook.value,value)):
        n,k=weight.shape[-2:]
        run=TinyJit(lambda x,rows:project(x[:,:,:rows]).pad_to((1,40,capacity,n)).contiguous().realize())
        for index in range(6):
          rows=17 if capacity==128 and index%2 else capacity
          data=rng.normal(0,.2,(1,40,capacity,k)).astype(np.float32)
          data[:,:,rows:]=np.nan
          before=Device['ROCKCHIP'].submit_count
          actual=run(Tensor(data,device='CPU').realize(),rowsvar.bind(rows)).numpy()
          expected=data[:,:,:rows].astype(np.float16).astype(np.float32)@weight.numpy().astype(np.float32).swapaxes(-1,-2)
          np.testing.assert_allclose(actual[:,:,:rows],expected,atol=2e-4,rtol=2e-3)
          self.assertTrue(np.all(actual[:,:,rows:]==0))
          self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
          if index==1:run.captured=RockchipCapturedJit(run.captured)
        self.assertGreater(run.captured.native_runs,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_complete_model_prefill_cached_decode_and_reset(self):
    from tinygrad import nn
    from tinygrad.llm.kernels.rockchip_mla_projections import configure
    from tinygrad.llm.model import Transformer,TransformerConfig
    config=TransformerConfig(1,320,640,40,40,1e-5,64,96,10000.,32,8,max_context=512,q_lora_rank=32,kv_lora_rank=256)
    reference,model=Transformer(config),Transformer(config)
    rng=np.random.default_rng(271104)
    weights={}
    for name,param in nn.state.get_state_dict(reference).items():
      data=rng.normal(0,.02,param.shape).astype(np.float16)
      if 'norm' in name:data+=1
      weights[name]=Tensor(data,device='CPU').realize()
    for target in (reference,model):nn.state.load_state_dict(target,weights)
    hooks=configure(model,prefill=True,decode=True)
    self.assertEqual(len(hooks),1)
    count=UOp.variable('grouped_model_rows',1,128)
    start=UOp.variable('grouped_model_start',0,511)
    prefill=[TinyJit(lambda tokens,n,_model=target:_model.logits(tokens[:,:n],0).realize()) for target in (reference,model)]
    decode=[TinyJit(lambda tokens,s,_model=target:_model.logits(tokens,s).realize()) for target in (reference,model)]
    for index in range(6):
      rows=128 if index%2==0 else 17
      tokens=Tensor([[(offset+index)%63+1 for offset in range(128)]],device='CPU').realize()
      hooks[0].phase='prefill'
      expected=prefill[0](tokens,count.bind(rows)).numpy()
      before=Device['ROCKCHIP'].submit_count
      actual=prefill[1](tokens,count.bind(rows)).numpy()
      self.assertEqual(Device['ROCKCHIP'].submit_count-before,2)
      np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-3)
      self.assertLess(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),1e-3)
      token=Tensor([[index+1]],device='CPU').realize()
      hooks[0].phase='decode'
      expected=decode[0](token,start.bind(rows)).numpy()
      before=Device['ROCKCHIP'].submit_count
      actual=decode[1](token,start.bind(rows)).numpy()
      self.assertEqual(Device['ROCKCHIP'].submit_count-before,2)
      np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-3)
      self.assertLess(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),1e-3)
      if index==1:
        prefill[1].captured=RockchipCapturedJit(prefill[1].captured)
        decode[1].captured=RockchipCapturedJit(decode[1].captured)
    self.assertGreater(prefill[1].captured.native_runs,0)
    self.assertGreater(decode[1].captured.native_runs,0)


if __name__=='__main__':unittest.main()
