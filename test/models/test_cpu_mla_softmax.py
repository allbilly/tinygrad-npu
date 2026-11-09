"""Numerical and zero-tail lineage checks for opt-in MLA softmax integration."""
import numpy as np
import pytest
from tinygrad import Tensor,TinyJit,UOp,Context,Device
from tinygrad.llm.kernels.rockchip_mla_softmax import softmax
from tinygrad.llm.kernels.rockchip_mla_attention import MLAAttention
from tinygrad.llm.kernels.rockchip_padding import capacity_input,capacity_view_counts,mark_zero_tail


@pytest.mark.parametrize('capacity',[1,128])
@pytest.mark.parametrize('reuse',[0,1])
def test_dynamic_softmax_retains_generic_bits_and_zero_tails(capacity,reuse):
  stride=512+capacity-1
  def forward(x,start,rows):
    scores=mark_zero_tail(x[:,:,:rows,:start+rows],x,'softmax_test')
    return capacity_input(softmax(scores,start)).realize()
  def reference(x,start,rows):return x[:,:,:rows,:start+rows].softmax(-1).pad_to((1,40,capacity,stride)).realize()
  run,ref=TinyJit(forward),TinyJit(reference)
  start=UOp.variable('mla_softmax_start',0,511)
  rows=UOp.variable('mla_softmax_rows',1,capacity)
  rng=np.random.default_rng(261102)
  with Context(ONPU_ZERO_PAD_VIEWS=reuse):
    for offset,count,kind in ((0,capacity,'ordinary'),(7,1,'nan'),(384,min(65,capacity),'inf'),(511,1,'zero')):
      values=rng.normal(0,.5,(1,40,capacity,stride)).astype(np.float32)
      values[:,:,count:]=0
      values[:,:,:,offset+count:]=0
      if kind=='nan':values[:,:,:count,0]=np.nan
      if kind=='inf':values[:,:,:count,0]=np.inf
      if kind=='zero':values[:]=0
      x=Tensor(values,device='CPU').realize()
      s,n=start.bind(offset),rows.bind(count)
      a=run(x,s,n).numpy()
      b=ref(x,s,n).numpy()
      assert a.tobytes()==b.tobytes(),(capacity,reuse,offset,count,kind)
      assert not a[:,:,count:].any() and not a[:,:,:,offset+count:].any()
  # Other CPU cases in the same pytest worker can evaluate scalar constants
  # with the host interpreter. It is not an accelerator backend.
  assert set(Device._opened_devices)<={'CPU','NPY','PYTHON'}


def test_attention_pipeline_reuses_scores_probabilities_and_context_without_copies():
  with Context(ONPU_MLA_FUSED_SOFTMAX=1):fast=MLAAttention(40,512,288,256,96**-.5)
  with Context(ONPU_MLA_FUSED_SOFTMAX=0):old=MLAAttention(40,512,288,256,96**-.5)
  assert fast.fused_softmax and not old.fused_softmax
  def forward(helper,q,k,s,n):
    scores=helper.query_key(q[:,:,:n],k,s)
    probabilities=helper.softmax(scores,s)
    return capacity_input(helper.weighted_value(probabilities,k,s)).realize()
  run=TinyJit(lambda q,k,s,n:forward(fast,q,k,s,n))
  ref=TinyJit(lambda q,k,s,n:forward(old,q,k,s,n))
  rng=np.random.default_rng(261103)
  s=UOp.variable('mla_pipeline_start',0,511)
  n=UOp.variable('mla_pipeline_rows',1,128)
  before=capacity_view_counts()
  with Context(ONPU_ZERO_PAD_VIEWS=1):
    for start,rows in ((0,128),(7,17),(384,65),(511,1)):
      q=Tensor(rng.normal(0,.2,(1,40,128,288)).astype(np.float32),device='CPU').realize()
      cache=Tensor(rng.normal(0,.2,(1,1,512,288)).astype(np.float32),device='CPU').realize()
      a=run(q,cache,s.bind(start),n.bind(rows)).numpy()
      b=ref(q,cache,s.bind(start),n.bind(rows)).numpy()
      assert a.tobytes()==b.tobytes(),(start,rows)
  after=capacity_view_counts()
  assert after.get('mla_scores',0)>before.get('mla_scores',0)
  assert after.get('mla_softmax',0)>before.get('mla_softmax',0)
  assert after.get('mla_context',0)>before.get('mla_context',0)


def test_static_or_other_geometry_keeps_ordinary_softmax():
  x=Tensor.randn(1,40,17,17,device='CPU').realize()
  assert softmax(x,0).numpy().tobytes()==x.softmax(-1).numpy().tobytes()
  other=Tensor.randn(1,4,1,33,device='CPU').realize()
  assert softmax(other,32).numpy().tobytes()==other.softmax(-1).numpy().tobytes()
