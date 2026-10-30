from types import SimpleNamespace
import pytest
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import RockchipLinear, configure


class _FakePool:
  def __init__(self): self.active=False
  def __enter__(self):
    self.active=True
    return self
  def __exit__(self,*args): self.active=False
  def prefill(self): return self


def _model():
  return Transformer(TransformerConfig(1,32,4864,4,2,1e-5,32,8,10000.,8,8,max_context=512))


def _patch_packed(monkeypatch):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  monkeypatch.setattr(RockchipInt8Linear,'prepare',lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU')]))
  monkeypatch.setattr(RockchipInt8SplitLinear,'prepare',
                      lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU') for _ in range((self.depth+4095)//4096)]))


def test_explicit_prefill_pool_enables_wide_int8_ffn_only_during_prefill(monkeypatch):
  from tinygrad.llm.kernels import rockchip_pool
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8SplitLinear
  pool=_FakePool()
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:pool)
  _patch_packed(monkeypatch)
  model=_model()
  observed=[]
  model.prefill_jit=lambda: observed.append((pool.active,model.blk[0].fused_ffn._float_enabled(Tensor.empty(1,16,32,device='CPU')))) or 'stub'
  result=configure(model,prefill='on',decode='off',precision='int8',fuse_projections=True,fused_ffn=True,
                   cpu_attention=True,cpu_pool=True,cpu_prefill_pool=True)
  assert result['cpu_pool'] is True and result['cpu_prefill_pool'] is True
  assert isinstance(model.blk[0].ffn_down,RockchipInt8SplitLinear)
  assert model.blk[0].fused_ffn.pool is pool
  assert model.prefill_jit()=='stub'
  assert observed==[(True,True)] and not pool.active
  assert not model.blk[0].fused_ffn._float_enabled(Tensor.empty(1,16,32,device='CPU'))


def test_pool_defaults_and_startup_failure_keep_reporting_truthful(monkeypatch):
  from tinygrad.llm.kernels import rockchip_pool
  _patch_packed(monkeypatch)
  calls=[]
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:(calls.append(1) or _FakePool()))
  result=configure(_model(),cpu_attention=True,cpu_pool=True)
  assert calls==[] and result['cpu_pool'] is False and result['cpu_prefill_pool'] is False
  def fail(): raise OSError('no workers')
  monkeypatch.setattr(rockchip_pool,'attention_pool',fail)
  result=configure(_model(),cpu_attention=True,cpu_pool=True,cpu_prefill_pool=True)
  assert result['cpu_pool'] is False and result['cpu_prefill_pool'] is False


@pytest.mark.parametrize('options',[{'cpu_pool':False,'cpu_attention':True},
                                    {'cpu_pool':True,'cpu_attention':False}])
def test_prefill_pool_requires_pool_and_cpu_attention(options):
  with pytest.raises(ValueError,match='Prefill pooling requires CPU attention'):
    configure(SimpleNamespace(),cpu_prefill_pool=True,**options)
