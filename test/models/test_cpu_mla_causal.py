"""Causal work reduction preserves the original loop's FP32 and IEEE behavior."""
import numpy as np
import pytest
from tinygrad import Tensor,TinyJit,UOp
from tinygrad.llm.kernels import rockchip_mla_attention as mla


@pytest.mark.parametrize('case',['causal','noncausal','cache_nan','cache_inf','query_nan','overflow','signed_zero',
  'underflow','pv_overflow','prob_nan'])
def test_causal_switch_changed_positions_and_ieee_bits(monkeypatch,case):
  rng=np.random.default_rng(271151)
  helpers=[]
  for enabled in (False,True):
    monkeypatch.setattr(mla,'getenv',lambda name,default,_value=enabled:int(_value))
    helper=mla.MLAAttention(4,64,68,64,.25)
    assert helper.causal_work is enabled
    helpers.append(helper)
  capacity,context,width,values=17,64,68,64
  score_shape=(1,4,capacity,context+capacity-1)
  value_shape=(1,4,capacity,values)
  qrun=[TinyJit(lambda q,k,s,n,_h=h:_h.query_key(q[:,:,:n],k,s).pad_to(score_shape).contiguous().realize()) for h in helpers]
  vrun=[TinyJit(lambda p,k,s,n,_h=h:_h.weighted_value(p[:,:,:n,:s+n],k,s).pad_to(value_shape).contiguous().realize()) for h in helpers]
  startvar=UOp.variable('causal_test_start',0,63)
  rowvar=UOp.variable('causal_test_rows',1,capacity)
  for start,rows in ((0,17),(7,1),(0,5),(20,17)):
    total=start+rows
    q=rng.normal(0,.2,(1,4,capacity,width)).astype(np.float32)
    cache=rng.normal(0,.2,(1,1,context,width)).astype(np.float32)
    probs=rng.uniform(0,.2,score_shape).astype(np.float32)
    if case!='noncausal':
      for row in range(rows):probs[:,:,row,start+row+1:total]=0
    q[:,:,rows:]=np.nan
    cache[:,:,total:]=np.nan
    probs[:,:,rows:]=np.nan
    probs[:,:,:,total:]=np.nan
    if case=='cache_nan':cache[0,0,total-1,0]=np.nan
    if case=='cache_inf':cache[0,0,total-1,0]=np.inf
    if case=='query_nan':q[0,0,0,0]=np.nan
    if case=='overflow':
      q[:,:,:rows]=np.float32(1e30)
      cache[:,:,:total]=np.float32(1e30)
    if case=='signed_zero':cache[:,:,:total]=np.float32(-0.)
    if case=='underflow':
      cache[:,:,:total]=np.nextafter(np.float32(0),np.float32(-1))
      cache[:,:,total-1]=np.nextafter(np.float32(0),np.float32(1))
      for row in range(rows):probs[:,:,row,:start+row+1]=.5
    if case=='pv_overflow':
      cache[:,:,:total]=np.float32(1e10)
      for row in range(rows):probs[:,:,row,:start+row+1]=np.float32(1e30)
    if case=='prob_nan':probs[0,0,0,total-1]=np.nan
    tq,tk,tp=[Tensor(a,device='CPU').realize() for a in (q,cache,probs)]
    s,n=startvar.bind(start),rowvar.bind(rows)
    old,new=[run(tq,tk,s,n).numpy() for run in qrun]
    np.testing.assert_array_equal(new.view(np.uint32),old.view(np.uint32))
    old,new=[run(tp,tk,s,n).numpy() for run in vrun]
    np.testing.assert_array_equal(new.view(np.uint32),old.view(np.uint32))
