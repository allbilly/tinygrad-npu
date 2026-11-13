"""Optional CPU row blocking preserves reductions, dynamic rows and input rebinding."""
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels import rockchip_static as static
from tinygrad.uop.ops import Ops


@pytest.fixture(autouse=True)
def clear_environment_cache():
  static.getenv.cache_clear()
  yield
  static.getenv.cache_clear()


def _block(monkeypatch,value):
  monkeypatch.setenv('ONPU_STATIC_GEMM_ROWS',str(value))
  static.getenv.cache_clear()


@pytest.mark.parametrize('rows', [1, 8, 17, 127, 128])
def test_eight_row_block_matches_original_bits_and_uses_distinct_source(rows, monkeypatch):
  rng=np.random.default_rng(2081+rows)
  weight=Tensor(rng.normal(0,.1,(129,17)).astype(np.float16),device='CPU').realize()
  x=Tensor(rng.normal(0,.3,(rows,17)).astype(np.float32),device='CPU').realize()
  prepared=static.prepare(weight)
  _block(monkeypatch,4)
  expected=prepared(x).numpy().copy()
  original=static._program
  sources=[]
  def record(*args,**kwargs):
    program=original(*args,**kwargs)
    sources.append(next(u.arg for u in program.src if u.op is Ops.SOURCE))
    return program
  monkeypatch.setattr(static,'_program',record)
  _block(monkeypatch,8)
  actual=prepared(x).numpy()
  assert actual.tobytes()==expected.tobytes()
  assert sources and all('for(;row+8<=rows;row+=8)' in source for source in sources)


def test_symbolic_rows_and_changed_buffers_keep_exact_original_outputs(monkeypatch):
  rng=np.random.default_rng(12817)
  weight=Tensor(rng.normal(0,.1,(129,17)).astype(np.float16),device='CPU').realize()
  prepared=static.prepare(weight)
  run=TinyJit(lambda x,n:prepared(x[:n]).realize())
  rows=UOp.variable('static_active',1,128)
  for step,active in enumerate((8,17,127,1,128,17)):
    x=Tensor(rng.normal(0,.3,(128,17)).astype(np.float32),device='CPU').realize()
    count=rows.bind(active)
    _block(monkeypatch,4)
    expected=prepared(x[:count]).pad_to((128,129)).numpy().copy()
    _block(monkeypatch,8)
    actual=run(x,count).pad_to((128,129)).numpy()
    assert actual.tobytes()==expected.tobytes(),(step,active)
    assert not actual[active:].any()


@pytest.mark.parametrize('block', ['3', '16'])
def test_unsupported_row_block_is_rejected(block,monkeypatch):
  prepared=static.prepare(Tensor.ones(129,17,device='CPU').half().realize())
  _block(monkeypatch,block)
  with pytest.raises(ValueError,match='Invalid static GEMM row block'):
    prepared(Tensor.ones(8,17,device='CPU')).realize()
