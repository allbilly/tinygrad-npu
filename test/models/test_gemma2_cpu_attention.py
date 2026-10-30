"""Gemma2 uses capped CPU attention while ordinary uncapped adapters stay excluded."""
import functools
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention
from test.models.test_gemma2 import TestGemma2 as Gemma2Fixture


@pytest.mark.parametrize('fast',[False,True])
def test_gemma2_capped_attention_matches_torch_prefill_decode_cached_prefill_and_reset(fast):
  fixture = Gemma2Fixture()
  generic,optimized = fixture.model(),fixture.model()
  placement = configure(optimized,prefill='off',decode='off',cpu_attention=True,cpu_fast_softmax=fast,
                        cpu_packed_attention=fast)
  assert placement['cpu_attention_blocks']==2 and placement['prepared_qkv_blocks']==placement['packed_attention_blocks']==0
  calls = []
  for i,block in enumerate(optimized.blk):
    assert block.attention_logit_softcap==block.config.attn_logit_softcap
    for name,expected in (('prefill_attention',prefill_attention),('decode_attention',decode_attention)):
      hook = getattr(block,name)
      assert isinstance(hook,functools.partial) and hook.func is expected
      assert hook.keywords['softcap']==1.25
      def counted(*args,_hook=hook,_name=name,_i=i,**kwargs):
        calls.append((_i,_name))
        return _hook(*args,**kwargs)
      setattr(block,name,counted)
  prefix = []
  for ids,start in (([7,12,4,8,2,11],0),([5],6),([6,3],7),([3,4,5],0)):
    prefix = ids.copy() if start==0 else prefix+ids
    expected = fixture.reference(generic,[prefix])
    actual = optimized.logits(Tensor([ids],device='CPU'),start).numpy()
    ordinary = generic.logits(Tensor([ids],device='CPU'),start).numpy()
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=5e-6)
    np.testing.assert_allclose(actual,ordinary,rtol=3e-5,atol=5e-6)
    assert int(actual.argmax())==int(expected.argmax())
  assert len(calls)==8 and sum(name=='decode_attention' for _,name in calls)==2


def test_changed_cap_cannot_reuse_stale_capped_adapter():
  from dataclasses import replace
  fixture = Gemma2Fixture()
  model = fixture.model()
  configure(model,prefill='off',decode='off',cpu_attention=True)
  for block in model.blk:
    block.config = replace(block.config,attn_logit_softcap=2.)
    block.prefill_attention = lambda *args:(_ for _ in ()).throw(AssertionError('Stale capped adapter used'))
  ids = [[3,4,5]]
  actual = model.logits(Tensor(ids,device='CPU'),0).numpy()
  assert np.isfinite(actual).all()
