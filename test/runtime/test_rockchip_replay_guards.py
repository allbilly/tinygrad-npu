"""Deduplicated checks retain every lifetime and coherency invalidation."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.kernels.rockchip_replay import NativeReplay
from tinygrad.runtime.ops_rockchip import RockchipBuffer


@pytest.mark.parametrize('mutation',('none','disabled','disabled_native_clean','unproven','different_meta','different_address','different_size',
                                    'dirty','large_arena','large_view','unaligned','out_of_bounds','flags','no_cleaner'))
def test_native_cleaning_of_large_arena_requires_exact_proven_pack_view(mutation):
  from tinygrad.helpers import Context
  from tinygrad.runtime.autogen import rockchip as rk
  base_flags=rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
  meta=SimpleNamespace(flags=base_flags,size=7933952,dma_addr=0x200000,obj_addr=0x400000,version=3,cpu_clean_version=3)
  raw=RockchipBuffer(0x100000,meta.dma_addr,262656,meta)
  replay=NativeReplay.__new__(NativeReplay)
  replay.cleaner,replay.invalidator=0x1234,None
  replay.clean_sites,replay.driver_from_sites=0,0
  replay.dev,replay.ioctl=SimpleNamespace(fd=SimpleNamespace(fd=17)),'ioctl'
  replay.extra_pack_uploads={8:(id(meta),raw.addr,raw.size)}
  enabled,native_enabled=1,1
  if mutation=='disabled':enabled=0
  elif mutation=='disabled_native_clean':native_enabled=0
  elif mutation=='unproven':replay.extra_pack_uploads={}
  elif mutation=='different_meta':replay.extra_pack_uploads[8]=(id(meta)+1,raw.addr,raw.size)
  elif mutation=='different_address':replay.extra_pack_uploads[8]=(id(meta),raw.addr+64,raw.size)
  elif mutation=='different_size':replay.extra_pack_uploads[8]=(id(meta),raw.addr,raw.size+64)
  elif mutation=='dirty':meta.cpu_clean_version-=1
  elif mutation=='large_arena':meta.size=32*1024**2
  elif mutation=='large_view':
    raw=RockchipBuffer(raw.addr,raw.dma,2*1024**2,meta)
    replay.extra_pack_uploads[8]=(id(meta),raw.addr,raw.size)
  elif mutation=='unaligned':
    raw=RockchipBuffer(raw.addr+4,raw.dma+4,raw.size,meta)
    replay.extra_pack_uploads[8]=(id(meta),raw.addr,raw.size)
  elif mutation=='out_of_bounds':
    offset=meta.size-64
    raw=RockchipBuffer(0x100000+offset,meta.dma_addr+offset,raw.size,meta)
    replay.extra_pack_uploads[8]=(id(meta),raw.addr,raw.size)
  elif mutation=='flags':meta.flags|=0x80000000
  elif mutation=='no_cleaner':replay.cleaner=None
  with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=enabled,ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN=native_enabled):
    code=replay.sync(raw,rk.RKNPU_MEM_SYNC_TO_DEVICE,8)
  assert replay.clean_sites==int(mutation=='none')
  assert ('struct Sync' not in code)==(mutation=='none')


def test_disabling_extra_pack_native_clean_preserves_small_buffer_native_clean():
  from tinygrad.helpers import Context
  from tinygrad.runtime.autogen import rockchip as rk
  flags=rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT
  meta=SimpleNamespace(flags=flags,size=266240,dma_addr=0x200000,obj_addr=0x400000,version=3,cpu_clean_version=3)
  raw=RockchipBuffer(0x100000,meta.dma_addr,1088,meta)
  replay=NativeReplay.__new__(NativeReplay)
  replay.cleaner,replay.invalidator=0x1234,None
  replay.clean_sites,replay.driver_from_sites=0,0
  replay.dev,replay.ioctl=SimpleNamespace(fd=SimpleNamespace(fd=17)),'ioctl'
  with Context(ONPU_REPLAY_EXTRA_PACK_ALIAS=1,ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN=0):
    code=replay.sync(raw,rk.RKNPU_MEM_SYNC_TO_DEVICE,8)
  assert replay.clean_sites==1 and 'struct Sync' not in code


class Meta:
  def __init__(self):self.version,self.clean,self.reads=3,3,0
  @property
  def cpu_clean_version(self):
    self.reads+=1
    return self.clean


def fixture(gpu=False):
  token=Tensor([7],device='CPU',dtype=dtypes.int32).realize().uop.base
  replay=NativeReplay.__new__(NativeReplay)
  replay.roots=[token.buffer]
  replay.input_sizes=[4]
  guard=Buffer('CPU',4)
  replay.guards={guard:None}
  weight=SimpleNamespace(version=8)
  replay.weight_guards={1:(weight,8)}
  replay.attention_programs=[SimpleNamespace(closed=False,owned=[object()])]
  metas=[Meta() for _ in range(3 if gpu else 1)]
  buffers=tuple(RockchipBuffer(i*64,i*64,64,metas[0 if i<4 else i-3]) for i in range(6 if gpu else 3))
  replay.plans=[(SimpleNamespace(owned=[object()],bufs=buffers),buffers) for _ in range(175)]
  return replay,token,metas,guard,weight


@pytest.mark.parametrize('gpu',(False,True))
def test_shared_output_metadata_checked_once_and_dirty_outputs_rejected(gpu):
  replay,token,metas,_,_=fixture(gpu)
  with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=1):
    assert replay.valid([token])
    assert [m.reads for m in metas]==[1]*len(metas)
    for meta in metas:
      meta.clean-=1
      assert not replay.valid([token])
      meta.clean+=1
    assert replay.valid([token])
  assert 'ROCKCHIP' not in Device._opened_devices and 'CL' not in Device._opened_devices


@pytest.mark.parametrize('mutation',('input_count','input_size','storage','weight','closed','released_attention','released_plan','reprepared'))
def test_changed_input_storage_weight_or_prepared_resource_invalidates(mutation):
  replay,token,_,guard,weight=fixture()
  inputs=[token]
  with patch('tinygrad.llm.kernels.rockchip_replay.getenv',return_value=1):
    assert replay.valid(inputs)
    if mutation=='input_count':inputs=[]
    elif mutation=='input_size':inputs=[Tensor([7,8],device='CPU',dtype=dtypes.int32).realize().uop.base]
    elif mutation=='storage':replay.guards[guard]=object()
    elif mutation=='weight':weight.version+=1
    elif mutation=='closed':replay.attention_programs[0].closed=True
    elif mutation=='released_attention':replay.attention_programs[0].owned=[]
    elif mutation=='released_plan':replay.plans[0][0].owned=[]
    else:
      plan,buffers=replay.plans[0]
      plan.bufs=tuple(list(buffers))
      assert plan.bufs==buffers and plan.bufs is not buffers
    assert not replay.valid(inputs)
