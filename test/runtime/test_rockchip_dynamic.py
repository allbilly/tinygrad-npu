import ctypes, os, struct, threading, unittest, weakref
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.device import TinyELF
from tinygrad.llm.kernels.rockchip import _grouped, grouped_matmul, pack_grouped_weight
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_replay import NativeReplay, RockchipCapturedJit
from tinygrad.runtime.ops_rockchip import RockchipBuffer, RockchipGroupedMatmul, RockchipProgram, _FINITE_HALF_SOURCE
from tinygrad.uop.ops import Ops


class TestDynamicGroupedMatmul(unittest.TestCase):
  @staticmethod
  def plan(dynamic):
    dev=SimpleNamespace(programs=weakref.WeakSet(),lock=threading.Lock(),check_healthy=lambda:None,
                        fd=SimpleNamespace(fd=1),domain=0,submit_count=0,task_count=0,poisoned=False)
    obj=SimpleNamespace(lib=struct.pack('<8I',1,32,32,1,1,1,0x47524d4d,7 if dynamic else 1))
    plan=RockchipGroupedMatmul(dev,obj)
    arrays=[np.zeros(32,np.float32),np.ones(32,np.float16),np.ones(1024,np.float16)]
    buffers=tuple(RockchipBuffer(a.ctypes.data,0x10000+i*4096,a.nbytes,
      SimpleNamespace(obj_addr=i+1,version=0,cpu_clean_version=0)) for i,a in enumerate(arrays))
    plan.bufs=buffers
    plan.weight_version=(3,0)
    plan.tasks,plan.core_mask,plan.subcores=1,1,((0,1),)
    plan.task=SimpleNamespace(meta=SimpleNamespace(obj_addr=9))
    return plan,arrays,buffers

  def test_explicit_descriptor_preserves_static_abi(self):
    params=[UOp.param(i,dtypes.half,1024) for i in range(3)]
    for dynamic in [False,True]:
      program=_grouped(*params,1,1,32,32,1,1,dynamic)
      binary=next(u.arg for u in program.src if u.op is Ops.BINARY)
      self.assertEqual(struct.unpack('<8I',binary)[-1],7 if dynamic else 1)
      plan,_,_=self.plan(dynamic)
      self.assertEqual(plan.dynamic_weights,dynamic)
      self.assertFalse(plan.int8)

  def test_dynamic_weights_are_revalidated_even_without_version_change(self):
    plan,arrays,buffers=self.plan(True)
    with patch('tinygrad.runtime.ops_rockchip.rk.DRM_IOCTL_RKNPU_SUBMIT') as submit:
      plan(*buffers)
      self.assertEqual(submit.call_count,1)
      for invalid in [np.nan,np.inf,-np.inf]:
        arrays[2][-1]=invalid
        with self.assertRaisesRegex(ValueError,'finite FP16 weights'):plan(*buffers)
        self.assertEqual(submit.call_count,1)
      arrays[2][-1]=2
      plan(*buffers)
      self.assertEqual(submit.call_count,2)

  def test_native_dynamic_guard_checks_last_weight_before_submission(self):
    plan,arrays,buffers=self.plan(True)
    runtime=RockchipProgram(None,None)
    runtime.bindings[tuple((id(b.meta),b.addr,b.size) for b in buffers)]=plan
    nodes=[UOp.param(i,dtypes.float if i==0 else dtypes.half,a.size) for i,a in enumerate(arrays)]
    body=SimpleNamespace(kernel_params=nodes)
    call=SimpleNamespace(device='ROCKCHIP',body=body,src=(body,*nodes))
    replay=NativeReplay.__new__(NativeReplay)
    replay.dev=plan.dev
    replay.kept,replay.aliases,replay.statements,replay.weight_guards,replay.mutated=[],{},[],{},set()
    replay.plans,replay.ioctl,replay.submits,replay.npu_outputs=[],'fake_submit',0,{}
    mapping={id(node):(SimpleNamespace(base=node,offset=0,nbytes=a.nbytes),buf,f'((void*){buf.addr}ULL)')
             for node,a,buf in zip(nodes,arrays,buffers)}
    with patch('tinygrad.llm.kernels.rockchip_replay.get_runtime',return_value=runtime), \
         patch.object(NativeReplay,'buffer',lambda _,node:mapping[id(node)]):
      replay.kernel(call,3)
    self.assertEqual(replay.weight_guards,{})
    self.assertIn(id(buffers[0].meta),replay.mutated)
    source='''typedef unsigned int U32;typedef unsigned long long U64;
    struct Submit {U32 flags,timeout,start,number,counter;int priority;U64 object,regcfg,base,user;
                   U32 mask;int fence;U32 subcores[10];} __attribute__((packed));
    static int fake_submit(int fd,unsigned long request,void *submit){return 0;}
    '''+_FINITE_HALF_SOURCE+'int probe(U64 *stats){stats[0]=stats[1]=0;'+''.join(replay.statements)+'return 0;}'
    renderer=_renderer()
    program=Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'probe',renderer.target,()))
    program.fxn.restype=ctypes.c_int
    stats=(ctypes.c_uint64*2)()
    for invalid in [np.nan,np.inf,-np.inf]:
      arrays[2][-1]=invalid
      self.assertEqual(program.fxn(stats),5000003)
      self.assertEqual(stats[0],0)
    arrays[2][:]=.5
    self.assertEqual(program.fxn(stats),0)
    self.assertEqual(stats[0],1)
    arrays[1][-1]=np.nan
    self.assertEqual(program.fxn(stats),1000003)
    self.assertEqual(stats[0],0)
    self.assertFalse(plan.dev.poisoned)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_row_major_attention_shapes_and_native_replay(self):
    rng=np.random.default_rng(291001)
    for rows in (17,128):
      for n,k in ((64,128),(128,64),(128,128),(256,128),(128,256),(512,128),(128,512)):
        run=TinyJit(lambda x,w:grouped_matmul(x,pack_grouped_weight(w),n,dynamic_weights=True,row_major=True).realize())
        for index in range(6):
          x=rng.normal(0,.2,(3,rows,k)).astype(np.float32)
          w=rng.normal(0,.2,(3,n,k)).astype(np.float16)
          before=Device['ROCKCHIP'].submit_count
          actual=run(Tensor(x,device='CPU').realize(),Tensor(w,device='CPU').realize()).numpy()
          expected=x.astype(np.float16).astype(np.float32)@w.astype(np.float32).swapaxes(-1,-2)
          np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=3e-4)
          self.assertEqual(Device['ROCKCHIP'].submit_count-before,1)
          if index==1:run.captured=RockchipCapturedJit(run.captured)
        self.assertIsNone(run.captured.reason)
        self.assertGreater(run.captured.native_runs,0)

  @unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
  def test_changed_kv_matrices_survive_native_replay(self):
    rng=np.random.default_rng(281001)
    for rows in [17,128]:
      run=TinyJit(lambda x,w:grouped_matmul(x,pack_grouped_weight(w),64,dynamic_weights=True).contiguous().realize())
      for index in range(8):
        x=rng.normal(0,.2,(3,rows,128)).astype(np.float32)
        w=rng.normal(0,.2,(3,64,128)).astype(np.float16)
        actual=run(Tensor(x,device='CPU').realize(),Tensor(w,device='CPU').realize()).numpy()
        expected=x.astype(np.float16).astype(np.float32)@w.astype(np.float32).swapaxes(-1,-2)
        np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=2e-4)
        if index==1:run.captured=RockchipCapturedJit(run.captured)
      self.assertIsNone(run.captured.reason)
      self.assertGreater(run.captured.native_runs,3)
      before=Device['ROCKCHIP'].submit_count
      w[-1,-1,-1]=np.nan
      with self.assertRaises(ValueError):run(Tensor(x,device='CPU').realize(),Tensor(w,device='CPU').realize())
      self.assertEqual(Device['ROCKCHIP'].submit_count,before)
      self.assertFalse(Device['ROCKCHIP'].poisoned)


if __name__=='__main__':unittest.main()
