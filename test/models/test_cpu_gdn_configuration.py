"""CPU-only tests for the staged opt-in CPU GatedDeltaNet hook installer.

These tests exercise configuration and chunk policy only; they do not call a
recurrent kernel, warm up a JIT, or access a hardware device.
"""
import functools
from types import SimpleNamespace
import pytest
from tinygrad.llm.model import GatedDeltaNetBlock, SSMConfig, Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_recurrent import configure, gated_delta_scan


def _model(gdn_layers=(True,False,True,True), *, num_kv_heads=1, num_v_heads=1, context=32) -> Transformer:
  ssm=SSMConfig(conv_kernel=2,state_size=8,group_count=1,time_step_rank=num_v_heads,inner_size=8*num_v_heads)
  config=TransformerConfig(num_blocks=len(gdn_layers),dim=16,hidden_dim=32,n_heads=2,n_kv_heads=num_kv_heads,
    norm_eps=1e-5,vocab_size=32,head_dim=8,rope_theta=10000.0,rope_dim=8,v_head_dim=8,
    max_context=context,ssm=ssm,ssm_layers=tuple(gdn_layers))
  return Transformer(config)


def _jit_counts(model):
  return {name:getattr(model,name).cnt for name in ('prefill_jit','rollout_jit','greedy_prefill_jit','greedy_rollout_jit')}


def test_installs_all_cpu_gdn_hooks_idempotently_before_chunk_selection_or_warmup():
  model=_model()
  blocks=[block for block in model.blk if isinstance(block,GatedDeltaNetBlock)]
  assert len(blocks)==3 and all(block.attn_qkv.weight.device=='CPU' for block in blocks)
  assert all(block.head_k_dim==8 and block.head_v_dim==8 for block in blocks)
  assert model.prefill_chunk_size(128)==1  # no CPU scan hook: serial fallback
  jit_counts=_jit_counts(model)
  assert all(count==0 for count in jit_counts.values())

  assert configure(model)==3
  assert all(block.gated_delta_scan is gated_delta_scan for block in blocks)
  assert all(not hasattr(block,'gated_delta_scan') for block in model.blk if not isinstance(block,GatedDeltaNetBlock))
  assert model.prefill_chunk_size(128)==128
  assert model.prefill_chunk_size(512)==256
  assert _jit_counts(model)==jit_counts  # installation precedes JIT warmup

  # A second configuration call must report the same GDN count and retain the
  # same function objects without installing duplicates or touching kernels.
  hooks=tuple(block.gated_delta_scan for block in blocks)
  assert configure(model)==3
  assert tuple(block.gated_delta_scan for block in blocks)==hooks
  assert _jit_counts(model)==jit_counts
  assert configure(model,enabled=False)==0
  assert all(not hasattr(block,'gated_delta_scan') for block in blocks)
  assert model.prefill_chunk_size(128)==1
  assert _jit_counts(model)==jit_counts


def test_pool_hook_is_idempotent_replaceable_and_disable_removes_it():
  model=_model((True,False,True))
  blocks=[block for block in model.blk if isinstance(block,GatedDeltaNetBlock)]
  pool=SimpleNamespace(address=1234,prefill_address=5678)
  other_pool=SimpleNamespace(address=2345,prefill_address=6789)

  assert configure(model,pool=pool)==2
  hooks=tuple(block.gated_delta_scan for block in blocks)
  assert all(isinstance(hook,functools.partial) and hook.func is gated_delta_scan
             and hook.keywords=={'pool':pool} for hook in hooks)
  assert configure(model,pool=pool)==2
  assert tuple(block.gated_delta_scan for block in blocks)==hooks

  assert configure(model,pool=other_pool)==2
  assert all(isinstance(block.gated_delta_scan,functools.partial)
             and block.gated_delta_scan.func is gated_delta_scan
             and block.gated_delta_scan.keywords=={'pool':other_pool} for block in blocks)
  assert configure(model,enabled=False)==0
  assert all(not hasattr(block,'gated_delta_scan') for block in blocks)


def test_rejects_models_without_gdn_blocks():
  model=_model((False,False))
  assert not any(isinstance(block,GatedDeltaNetBlock) for block in model.blk)
  with pytest.raises(ValueError,match='requires GatedDeltaNet blocks'):
    configure(model)
  assert all(not hasattr(block,'gated_delta_scan') for block in model.blk)


@pytest.mark.parametrize(('attribute','bad_value'),[('head_k_dim',257),('head_v_dim',257)])
def test_rejects_later_unsupported_gdn_dimensions_before_any_partial_install(attribute,bad_value):
  model=_model((True,False,True))
  blocks=[block for block in model.blk if isinstance(block,GatedDeltaNetBlock)]
  setattr(blocks[-1],attribute,bad_value)
  with pytest.raises(ValueError,match='Unsupported CPU recurrent block geometry or device'):
    configure(model)
  assert all(not hasattr(block,'gated_delta_scan') for block in blocks)


def test_rejects_unsupported_weight_device_before_any_partial_install_without_opening_it():
  model=_model((True,False,True))
  blocks=[block for block in model.blk if isinstance(block,GatedDeltaNetBlock)]
  # The fake marker is deliberately not a Tensor, so the installer must reject
  # it at its isinstance/device gate without resolving or opening a CL device.
  blocks[-1].attn_qkv.weight=SimpleNamespace(device='CL')
  with pytest.raises(ValueError,match='Unsupported CPU recurrent block geometry or device'):
    configure(model)
  assert all(not hasattr(block,'gated_delta_scan') for block in blocks)
