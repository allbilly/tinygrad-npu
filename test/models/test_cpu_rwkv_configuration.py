"""CPU RWKV7 installation and chunk policy, without opening NPU or GPU devices."""
import functools
from types import SimpleNamespace
import pytest
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_recurrent import configure
from tinygrad.llm.kernels.rockchip_rwkv import rwkv_scan


def model():return Transformer(TransformerConfig(2,32,64,4,4,1e-5,96,8,10000.,0,8,max_context=24,rwkv7=(8,8,4,12)))


def test_all_hooks_install_before_chunk_selection_and_disable_restores_serial_policy():
  m=model()
  assert m.prefill_chunk_size(128)==1
  counts=[getattr(m,name).cnt for name in ('prefill_jit','rollout_jit','greedy_prefill_jit','greedy_rollout_jit')]
  assert configure(m)==2
  assert all(b.rwkv_scan is rwkv_scan for b in m.blk)
  assert m.prefill_chunk_size(128)==128 and m.prefill_chunk_size(1024)==256
  assert configure(m)==2
  assert [getattr(m,name).cnt for name in ('prefill_jit','rollout_jit','greedy_prefill_jit','greedy_rollout_jit')]==counts
  delattr(m.blk[0],'rwkv_scan')
  assert m.prefill_chunk_size(128)==1
  assert configure(m,enabled=False)==0 and all(not hasattr(b,'rwkv_scan') for b in m.blk)


def test_pool_installation_is_idempotent_and_partial_installation_is_rejected():
  m=model()
  pool=SimpleNamespace(prefill_address=1234)
  assert configure(m,pool=pool)==2
  hooks=[b.rwkv_scan for b in m.blk]
  assert configure(m,pool=pool)==2 and [b.rwkv_scan for b in m.blk]==hooks
  assert all(isinstance(h,functools.partial) and h.func is rwkv_scan for h in hooks)
  configure(m,enabled=False)
  m.blk[-1].time_mix_receptance.weight=SimpleNamespace(device='CL')
  with pytest.raises(ValueError):configure(m)
  assert all(not hasattr(b,'rwkv_scan') for b in m.blk)


def test_prepared_configuration_selects_only_explicit_rwkv_recipe(monkeypatch):
  from contextlib import nullcontext
  from tinygrad import Tensor,dtypes
  from tinygrad.helpers import Context
  from tinygrad.llm.kernels import rockchip,rockchip_pool
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  class Shared:
    def __init__(self,pool):self.pool,self.changes,self.preparation_changes=pool,[],[]
    def scope(self):return nullcontext()
  pool=SimpleNamespace(active=False,prefill_address=1234,address=4321)
  shared=Shared(pool)
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:pool)
  monkeypatch.setattr(rockchip,'shared_host_pool',lambda actual:shared)
  def prepare(layer):
    layer.packed=[Tensor.empty(1,16,32,32,device='CPU',dtype=dtypes.int8)]
    layer.scales=[Tensor.ones(layer.columns,device='CPU')]
  monkeypatch.setattr(RockchipInt8Linear,'prepare',prepare)
  m=Transformer(TransformerConfig(2,32,64,4,4,1e-5,96,8,10000.,0,8,max_context=512,rwkv7=(8,8,4,12)))
  with Context(ONPU_SHARED_HOST_POOL=1):
    result=rockchip.configure(m,prefill='on',decode='on',precision='int8',cpu_pool=True,cpu_prefill_pool=True,
                              cpu_attention=True,cpu_recurrent_scan=True,cpu_rwkv_weights=True)
  assert result['cpu_rwkv_weight_tensors']==31 and result['shared_host_pool']
  assert all(b.rwkv_scan.keywords['pool'] is pool and b.rwkv_scan.keywords['specialize'] for b in m.blk)
  for kwargs in ({'cpu_rwkv_weights':1},{'cpu_rwkv_weights':True},
                 {'cpu_rwkv_weights':True,'precision':'fp16','cpu_recurrent_scan':True}):
    with pytest.raises(ValueError):rockchip.configure(model(),**kwargs)
