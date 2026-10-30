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
