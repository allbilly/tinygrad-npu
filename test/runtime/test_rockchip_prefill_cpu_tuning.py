"""Prefill loop scheduling must preserve payloads, scales and QKV cache bytes."""
import numpy as np
import pytest
from tinygrad.helpers import Context
from test.runtime.test_rockchip_shared_pool import _make_case, _run


@pytest.mark.parametrize('family,rows,depth', (
  ('ffn_pack',1,512), ('ffn_pack',17,1536), ('ffn_pack',65,3072), ('ffn_pack',128,3072), ('ffn_pack',128,4096),
  ('qkv',1,None), ('qkv',17,None), ('qkv',65,None), ('qkv',128,None)))
@pytest.mark.parametrize('nonfinite', (False,True))
def test_tuned_prefill_preserves_all_output_bits(family,rows,depth,nonfinite):
  with Context(ONPU_PREFILL_CPU_TUNING=0):
    reference = _run(_make_case(family,rows,101,depth=depth,nonfinite=nonfinite))
  with Context(ONPU_PREFILL_CPU_TUNING=1):
    actual = _run(_make_case(family,rows,101,depth=depth,nonfinite=nonfinite))
  assert len(actual)==len(reference)
  for a,b in zip(actual,reference):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


@pytest.mark.parametrize('heads,kv', ((12,2),(16,2),(14,2),(32,4),(16,8),(128,64)))
def test_nonuniform_worker_ranges_cover_every_head_and_balance_kv_work(heads,kv):
  from tinygrad.llm.kernels.rockchip_qkv import _balanced_head_order
  order=_balanced_head_order(heads,kv)
  total=heads+kv
  assert sorted(order)==list(range(total))
  loads=[sum(1+(head>=heads) for head in order[total*i//4:total*(i+1)//4]) for i in range(4)]
  assert max(loads)-min(loads)<=1
