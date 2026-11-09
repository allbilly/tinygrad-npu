"""Captured SDK words and PC links for bounded split-row INT8 task chains."""
import ctypes, json, struct
from pathlib import Path
from types import SimpleNamespace
import pytest
from tinygrad import Device
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipMatmul, _cmd
from tinygrad.runtime.support.rockchip_int8 import int8_sdk_bodies


@pytest.mark.parametrize('name,depth',[('rockchip_sdk_split.json',1536),('spark_k2048_sdk_prefill.json',2048)])
def test_derived_words_match_all_captured_sdk_registers_in_order(name,depth):
  fixture=json.loads((Path(__file__).parent/'fixtures'/name).read_text())
  addresses={(0x201,0x1070):0x10000000,(0x201,0x1110):0x20000000,(0x1001,0x4020):0x30000000}
  for columns,tasks in fixture['cases'].items():
    actual=int8_sdk_bodies(128,int(columns),depth,0x10000000,0x20000000,0x30000000,int(columns))
    expected=tuple(tuple(_cmd(target,reg,value+addresses.get((target,reg),0)) for target,reg,value in task) for task in tasks)
    assert actual==expected


class MemoryDevice:
  def __init__(self):self.programs,self.allocations,self.next=set(),[],0x50000000
  def check_domain(self,*buffers):pass
  def allocate(self,size,*flags):
    storage=ctypes.create_string_buffer(size)
    buf=SimpleNamespace(size=size,dma=self.next,addr=ctypes.addressof(storage),storage=storage)
    self.next+=(size+4095)//4096*4096
    self.allocations.append(buf)
    return buf
  def sync(self,*args):pass
  def free(self,*args):pass


def test_every_core_links_its_own_tasks_and_terminates_at_its_boundary():
  before=set(Device._opened_devices)
  for columns,depth,sdk,task_count in ((12288,1536,True,6),(5632,1536,True,6),(6144,1024,True,3),(12288,1536,False,3),
                                    (12288,2048,True,6),(3072,2048,True,6),(1024,2048,True,6),(12288,2048,False,3)):
    dev=MemoryDevice()
    plan=RockchipMatmul(dev,SimpleNamespace(lib=struct.pack('<9I',128,columns,depth,3,0,512,0,1,int(sdk))))
    try:
      plan.prepare(SimpleNamespace(size=128*columns*4,dma=0x30000000),
                   SimpleNamespace(size=512+128*depth,dma=0x10000000),SimpleNamespace(size=columns*depth,dma=0x20000000))
      assert plan.tasks==task_count and plan.core_mask==7
      per_core=task_count//3
      assert plan.subcores==((0,0),(0,0),(0,per_core),(per_core,per_core),(2*per_core,per_core))
      descs=(rk.struct_rknpu_task*task_count).from_address(plan.task.addr)
      words=(ctypes.c_uint64*(plan.cmd.size//8)).from_address(plan.cmd.addr)
      for i,desc in enumerate(descs):
        first=(desc.regcmd_addr-plan.cmd.dma)//8
        assert desc.regcmd_addr%16==0 and first>=0 and (first+desc.regcfg_amount+4)*8<=plan.cmd.size
        tail=words[first+desc.regcfg_amount:first+desc.regcfg_amount+4]
        values={(w>>48,w&65535):(w>>16)&0xffffffff for w in tail}
        linked=(i+1)%per_core!=0
        assert values[rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS]==((descs[i+1].regcfg_amount+1)//2+1 if linked else 0)
        if linked:assert values[rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS]==descs[i+1].regcmd_addr
        if sdk and depth in (1536,2048):
          body={(w>>48,w&65535):(w>>16)&0xffffffff for w in words[first:first+desc.regcfg_amount]}
          assert body[0x201,0x1070]==0x10000000+512+(16*(84 if depth==1536 else 64) if i%2==0 else 0)
    finally:plan.close()
  assert set(Device._opened_devices)==before


def test_split_sdk_factory_rejects_unqualified_geometry():
  for rows,columns,depth,stride in ((127,4096,1536,4096),(128,2048,1536,2048),(128,1888,1536,1856),
                                    (127,4096,2048,4096),(128,384,2048,384),(128,4096,2048,1024)):
    with pytest.raises(ValueError):int8_sdk_bodies(rows,columns,depth,0,0,0,stride)
