"""CPU erf-GELU hook installation and benchmark CLI selection."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
from tinygrad import Device
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
from tinygrad.llm.kernels.rockchip_gelu_erf import gated_erf_gelu


def _model(activation='gelu',blocks=2):
  return Transformer(TransformerConfig(blocks,32,64,4,4,1e-5,32,8,10000.,0,8,max_context=16,
                                      ffn_activation=activation))


def test_configure_installs_clears_and_replaces_disjoint_gelu_hooks():
  opened_before={name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}
  model=_model('gelu')
  result=configure(model,cpu_gelu=True,cpu_erf_gelu=True,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==len(model.blk)
  assert result['cpu_gelu_blocks']==0
  assert all(block.cpu_gelu_activation is gated_erf_gelu for block in model.blk)

  for block in model.blk: block.config=replace(block.config,ffn_activation='gelu_tanh')
  result=configure(model,cpu_gelu=True,cpu_erf_gelu=True,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==0 and result['cpu_gelu_blocks']==len(model.blk)
  assert all(block.cpu_gelu_activation is gated_gelu for block in model.blk)

  for block in model.blk: block.config=replace(block.config,ffn_activation='gelu')
  result=configure(model,cpu_gelu=True,cpu_erf_gelu=True,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==len(model.blk) and result['cpu_gelu_blocks']==0
  assert all(block.cpu_gelu_activation is gated_erf_gelu for block in model.blk)

  result=configure(model,cpu_gelu=False,cpu_erf_gelu=False,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==0 and result['cpu_gelu_blocks']==0
  assert all(not hasattr(block,'cpu_gelu_activation') for block in model.blk)
  assert {name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}==opened_before


@pytest.mark.parametrize('field,value',[
  ('ffn_activation','silu'),('swiglu_limit',1.0),('swiglu_up_bias',.25),('num_experts',2),
])
def test_configure_excludes_unsupported_erf_gelu_configs(field,value):
  model=_model('gelu',blocks=1)
  model.blk[0].config=replace(model.blk[0].config,**{field:value})
  result=configure(model,cpu_erf_gelu=True,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==0
  assert not hasattr(model.blk[0],'cpu_gelu_activation')


def test_configure_requires_strict_bool_and_preserves_unowned_hook():
  with pytest.raises(ValueError): configure(SimpleNamespace(),cpu_erf_gelu=1)
  model=_model('gelu',blocks=1)
  def marker(gate,up): return gate*up
  model.blk[0].cpu_gelu_activation=marker
  result=configure(model,cpu_erf_gelu=False,prefill='off',decode='off')
  assert result['cpu_erf_gelu_blocks']==0
  assert model.blk[0].cpu_gelu_activation is marker


def test_cli_rejects_erf_gelu_on_non_cpu_before_reading_checkpoint_or_input(monkeypatch,tmp_path):
  from extra.rockchip import benchmark_inference
  missing_checkpoint=tmp_path/'missing.gguf'
  missing_input=tmp_path/'missing.ids'
  argv=['benchmark_inference','--checkpoint',str(missing_checkpoint),'--input',str(missing_input),
        '--report',str(tmp_path/'report.json'),'--device','CL','--cpu-erf-gelu']
  monkeypatch.setattr(sys,'argv',argv)
  with pytest.raises(ValueError,match='CPU erf GELU requires device CPU'): benchmark_inference.main()
