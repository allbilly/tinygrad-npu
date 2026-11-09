"""Opt-in query-norm row dispatch preserves captured reductions and alias fallback."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Device,UOp
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.rockchip_exact_prepare import _QUERY_NORM_SOURCES,rewrite_exact_preparation
from tinygrad.llm.kernels.rockchip_prepare_pool import _split_independent_rows
from test.runtime.test_rockchip_exact_prepare import program,compile_cpu,invoke
from test.runtime.test_rockchip_shared_pool import _needs_pool,_pool_test_lock

FIXTURES=Path(__file__).parent/'fixtures/exact_preparation/query_norm'

@pytest.mark.parametrize('digest',list(_QUERY_NORM_SOURCES))
def test_query_norm_row_dispatch_retains_bits_and_serial_overlap_fallback(digest):
  _needs_pool()
  with _pool_test_lock():
    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    pool=AttentionPool()
    try:
      source=(FIXTURES/(digest+'.c')).read_text()
      name=_QUERY_NORM_SOURCES[digest]
      assert hashlib.sha256(source.encode()).hexdigest()==digest
      linear=UOp(Ops.LINEAR,src=(program(source,name),))
      with Context(ONPU_CPU_QUERY_NORM_ROWS=0):assert rewrite_exact_preparation(linear,pool=pool,rows=True) is linear
      with Context(ONPU_CPU_QUERY_NORM_ROWS=1):updated=rewrite_exact_preparation(linear,pool=pool,rows=True)
      assert updated is not linear
      text=next(u.arg for u in updated.src[0].src if u.op is Ops.SOURCE)
      assert 'Lidx2 = (128)*worker/4' in text
      _,args=_split_independent_rows(source,name,pool.prefill_address)
      size=int(args[1][1].rsplit('_',1)[1])
      functions=[compile_cpu(s,name) for s in (source,text)]
      rng=np.random.default_rng(20261104)
      for kind in ('finite','zeros','wide','nonfinite','tiny'):
        values=rng.normal(0,.2,size).astype(np.float32)
        if kind=='zeros':
          values[:]=0
          values[::2]=-0.
        if kind=='wide':values*=10000
        if kind=='nonfinite':
          values[::997]=np.nan
          values[1::991]=np.inf
        if kind=='tiny':values*=1e-30
        outputs=[np.empty(1024,np.float32) for _ in range(2)]
        before=pool.lib.pool_calls()
        for function,out in zip(functions,outputs):invoke(function,out,values)
        assert outputs[0].tobytes()==outputs[1].tobytes(),kind
        assert pool.lib.pool_calls()==before+1
      outputs=[]
      initial=rng.normal(0,.2,size).astype(np.float32)
      for function in functions:
        values=initial.copy()
        out=values[:1024]
        before=pool.lib.pool_calls()
        invoke(function,out,values)
        assert pool.lib.pool_calls()==before
        outputs.append(out.copy())
      assert outputs[0].tobytes()==outputs[1].tobytes()
    finally:pool.close()
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)


def test_query_norm_requires_reviewed_source_prefill_and_explicit_flag():
  digest=next(iter(_QUERY_NORM_SOURCES))
  source=(FIXTURES/(digest+'.c')).read_text()
  name=_QUERY_NORM_SOURCES[digest]
  linear=UOp(Ops.LINEAR,src=(program(source,name),))
  pool=SimpleNamespace(prefill_address=1)
  with Context(ONPU_CPU_QUERY_NORM_ROWS=1):
    assert rewrite_exact_preparation(linear,pool=pool,rows=True,phase='decode') is linear
    for text,label in ((source+' ',name),(source,'unknown')):
      unreviewed=UOp(Ops.LINEAR,src=(program(text,label),))
      assert rewrite_exact_preparation(unreviewed,pool=pool,rows=True) is unreviewed
    assert rewrite_exact_preparation(linear,pool=pool,rows=False) is linear
