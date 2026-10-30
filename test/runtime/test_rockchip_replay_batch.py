import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import UOp, dtypes
from tinygrad.llm.kernels.rockchip_replay import NativeReplay
from tinygrad.runtime.ops_rockchip import RockchipBuffer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo
from tinygrad.helpers import Target


class TestRockchipReplayBatchBorrow(unittest.TestCase):
  """Batch download alias checks without constructing a device or running kernels."""
  @staticmethod
  def node(device='CPU',size=4,base=None,offset=0):
    node = SimpleNamespace(dtype=dtypes.half,toposort=lambda:[SimpleNamespace(op=Ops.BUFFER,tag='scratch')])
    buf = SimpleNamespace(base=object() if base is None else base,offset=offset,nbytes=size*2,device=device)
    raw = RockchipBuffer(0x10000,0x20000,buf.nbytes,SimpleNamespace(flags=rk.RKNPU_MEM_CACHEABLE)) if device=='ROCKCHIP' else 0
    return node,(buf,raw,f'{device}_{id(node)}')

  @staticmethod
  def program(out_size=4,in_size=4,name='batch_reader'):
    out,inp = UOp.param(0,dtypes.half,out_size),UOp.param(1,dtypes.half,in_size)
    row = UOp.range(min(out_size,in_size),0,dtype=dtypes.int)
    sink = out[row].store(inp[row].load()).end(row).sink(arg=KernelInfo(name))
    info = replace(ProgramInfo.from_sink(sink,Target('CPU')),outs=(0,),ins=(1,))
    return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,inp))))

  @staticmethod
  def copy_call(dst,src):
    return SimpleNamespace(op=Ops.CALL,body=UOp(Ops.STORE),src=(UOp(Ops.STORE),dst,src),device='CPU',arg=SimpleNamespace(aux=None))

  @staticmethod
  def program_call(body,*nodes):
    return SimpleNamespace(op=Ops.CALL,body=body,src=(body,*nodes),device='CPU',arg=SimpleNamespace(aux=None))

  def analyze(self,calls,*bindings):
    replay = NativeReplay.__new__(NativeReplay)
    replay.calls,replay.roots,replay.borrow_outputs,replay.borrow_batch = calls,[],True,True
    buffers = {id(node):binding for node,binding in bindings}
    with patch.object(NativeReplay,'buffer',lambda _,node:buffers[id(node)]): return replay.copy_aliases()

  def setup_pair(self):
    host_a,host_b,result,source = (self.node() for _ in range(4))
    npu_a,npu_b = self.node('ROCKCHIP'),self.node('ROCKCHIP')
    return host_a,npu_a,host_b,npu_b,result,source

  def test_borrows_download_across_an_unrelated_copy(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    calls = [self.copy_call(host_a[0],npu_a[0]), self.copy_call(host_b[0],npu_b[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,uploads,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source)
    self.assertEqual(aliases,{(2,(host_a[1][0].base,0,8)):npu_a[1][2]})
    self.assertEqual((uploads,downloads),(set(),{0}))

  def test_two_downloads_can_feed_the_same_cpu_reader(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    out,a,b = (UOp.param(i,dtypes.half,4) for i in range(3))
    row = UOp.range(4,0,dtype=dtypes.int)
    sink = out[row].store(a[row].load()+b[row].load()).end(row).sink(arg=KernelInfo('batch_pair_reader'))
    info = replace(ProgramInfo.from_sink(sink,Target('CPU')),outs=(0,),ins=(1,2))
    program = UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,a,b))))
    calls = [self.copy_call(host_a[0],npu_a[0]),self.copy_call(host_b[0],npu_b[0]),
             self.program_call(program,result[0],host_a[0],host_b[0])]
    aliases,uploads,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source)
    self.assertEqual(aliases,{(2,(host_a[1][0].base,0,8)):npu_a[1][2],(2,(host_b[1][0].base,0,8)):npu_b[1][2]})
    self.assertEqual((uploads,downloads),(set(),{0,1}))

  def test_intervening_program_blocks_batch_borrow(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    calls = [self.copy_call(host_a[0],npu_a[0]), self.program_call(self.program(),result[0],source[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,_,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source)
    self.assertNotIn((2,(host_a[1][0].base,0,8)),aliases)
    self.assertNotIn(0,downloads)

  def test_intervening_copy_reading_cpu_destination_blocks_borrow(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    calls = [self.copy_call(host_a[0],npu_a[0]), self.copy_call(npu_b[0],host_a[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,_,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source)
    self.assertFalse(aliases)
    self.assertNotIn(0,downloads)

  def test_intervening_copy_writing_cpu_destination_blocks_borrow(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    calls = [self.copy_call(host_a[0],npu_a[0]), self.copy_call(host_a[0],host_b[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,_,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source)
    self.assertFalse(aliases)
    self.assertNotIn(0,downloads)

  def test_overlapping_npu_source_write_blocks_borrow(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    view_node,(view_buf,view_raw,view_expr) = self.node('ROCKCHIP',base=npu_a[1][0].base)
    view = (view_node,(view_buf,replace(view_raw,addr=npu_a[1][1].addr,dma=npu_a[1][1].dma,meta=npu_a[1][1].meta),view_expr))
    calls = [self.copy_call(host_a[0],npu_a[0]), self.copy_call(view[0],source[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,_,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source,view)
    self.assertFalse(aliases)
    self.assertNotIn(0,downloads)

  def test_nonoverlapping_npu_source_write_allows_borrow(self):
    host_a,npu_a,host_b,npu_b,result,source = self.setup_pair()
    view_node,(view_buf,view_raw,view_expr) = self.node('ROCKCHIP',base=npu_a[1][0].base,offset=8)
    view = (view_node,(view_buf,replace(view_raw,addr=npu_a[1][1].addr+8,dma=npu_a[1][1].dma+8,meta=npu_a[1][1].meta),view_expr))
    calls = [self.copy_call(host_a[0],npu_a[0]), self.copy_call(view[0],source[0]),
             self.program_call(self.program(),result[0],host_a[0])]
    aliases,_,downloads = self.analyze(calls,host_a,npu_a,host_b,npu_b,result,source,view)
    self.assertEqual(aliases,{(2,(host_a[1][0].base,0,8)):npu_a[1][2]})
    self.assertEqual(downloads,{0})


if __name__ == '__main__': unittest.main()
