"""FP32 MLA dot, cache, padding and full-model regression tests without NPU access."""
import itertools, math, os, platform
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.llm.kernels.rockchip_mla_attention import MLAAttention, configure
import test.models.test_minicpm3 as minicpm


def reference_qk(q,cache,start,rows,scale,capacity):
  scores=(q[:,:,:rows]@cache[:,:,:start+rows].transpose(-1,-2))*scale
  if capacity!=1:scores=scores+Tensor.full((1,1,rows,start+rows),-math.inf,device='CPU').triu(start+1)
  return scores


@pytest.mark.parametrize('heads,width,values,context,capacity',[(4,12,8,32,17),(40,288,256,512,128),
  (4,288,256,33,17),(4,12,8,32,1)])
def test_dynamic_changed_inputs_and_poisoned_unused_cache_preserve_bits(heads,width,values,context,capacity):
  rng=np.random.default_rng(1801)
  scale=1/math.sqrt(96 if width==288 else 12)
  helper=MLAAttention(heads,context,width,values,scale)
  score_shape=(1,heads,capacity,context+capacity-1)
  value_shape=(1,heads,capacity,values)
  qref=TinyJit(lambda q,k,s,n:reference_qk(q,k,s,n,scale,capacity).pad_to(score_shape).contiguous().realize())
  qfast=TinyJit(lambda q,k,s,n:helper.query_key(q[:,:,:n],k,s).pad_to(score_shape).contiguous().realize())
  vref=TinyJit(lambda p,k,s,n:(p[:,:,:n,:s+n]@k[:,:,:s+n,:values]).pad_to(value_shape).contiguous().realize())
  vfast=TinyJit(lambda p,k,s,n:helper.weighted_value(p[:,:,:n,:s+n],k,s).pad_to(value_shape).contiguous().realize())
  rowvar=UOp.variable('mla_test_rows',1,capacity)
  startvar=UOp.variable('mla_test_start',0,context-1)
  for start,rows in ((0,capacity),(7,1),(0,min(17,capacity)),(context-1,1)):
    q=rng.normal(0,.2,(1,heads,capacity,width)).astype(np.float32)
    q[:,:,rows:]=np.nan
    k=rng.normal(0,.2,(1,1,context,width)).astype(np.float32)
    k[:,:,start+rows:]=np.nan
    probs=rng.uniform(0,.2,(1,heads,capacity,context+capacity-1)).astype(np.float32)
    q,k,p=[Tensor(x,device='CPU').realize() for x in (q,k,probs)]
    s,n=startvar.bind(start),rowvar.bind(rows)
    a=qref(q,k,s,n).numpy()
    b=qfast(q,k,s,n).numpy()
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    a=vref(p,k,s,n).numpy()
    b=vfast(p,k,s,n).numpy()
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))


def test_complete_model_cache_reset_generation_and_state_dict():
  fixture=minicpm.TestMiniCPM3()
  weights=fixture.fixture()
  generic,prepared=fixture.load(weights),fixture.load(weights)
  old=nn.state.get_state_dict(prepared)
  identities={k:id(v) for k,v in old.items()}
  assert configure(prepared)==2
  assert {k:id(v) for k,v in nn.state.get_state_dict(prepared).items()}==identities
  for ids,start,prefix in (([3,7,2],0,[3,7,2]),([9],3,[3,7,2,9]),([8,4],0,[8,4])):
    a=generic.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
    b=prepared.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    np.testing.assert_allclose(b,fixture.reference(weights,prefix),rtol=2e-4,atol=1e-4)
  prompt=[3,7,2]
  expected=list(itertools.islice(generic.generate(prompt.copy(),chunk_size=2),3))
  assert expected==list(itertools.islice(prepared.generate(prompt.copy(),chunk_size=2),3))
  assert configure(prepared,False)==0 and all(not hasattr(b,'cpu_mla_attention') for b in prepared.blk)


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64') or not {4,5,6,7}.issubset(os.sched_getaffinity(0)),
                    reason='CPU MLA workers require CPU4-7')
def test_pool_preserves_bits_restores_affinity_and_parks():
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  rng=np.random.default_rng(1802)
  pool=AttentionPool()
  before=os.sched_getaffinity(0)
  try:
    for capacity,start in ((17,0),(1,17)):
      q=Tensor(rng.normal(0,.2,(1,8,capacity,68)).astype(np.float32),device='CPU').realize()
      k=Tensor(rng.normal(0,.2,(1,1,32,68)).astype(np.float32),device='CPU').realize()
      serial=MLAAttention(8,32,68,64,.25)
      parallel=MLAAttention(8,32,68,64,.25,pool)
      position=UOp.variable('mla_pool_start',0,31).bind(start)
      expected=serial.query_key(q,k,position).softmax(-1).pad_to((1,8,capacity,32+capacity-1)).contiguous().realize()
      probs=expected[:,:,:,:position+capacity]
      expected_value=serial.weighted_value(probs,k,position).numpy()
      calls=pool.lib.pool_calls()
      with pool:
        actual=parallel.query_key(q,k,position).softmax(-1).pad_to((1,8,capacity,32+capacity-1)).contiguous().realize()
        value=parallel.weighted_value(actual[:,:,:,:position+capacity],k,position).numpy()
        assert bool(pool.lib.pool_parked()) is (capacity!=1)
      np.testing.assert_array_equal(expected.numpy().view(np.uint32),actual.numpy().view(np.uint32))
      np.testing.assert_array_equal(expected_value.view(np.uint32),value.view(np.uint32))
      assert pool.lib.pool_calls()==calls+2 and pool.lib.pool_parked() and os.sched_getaffinity(0)==before
  finally:pool.close()


def test_invalid_geometry_inputs_and_scalar_bounds_rejected_before_execution():
  for geometry in ((4,512,576,512),(4,4096,512,256),(0,32,12,8),(4,32,13,8)):
    with pytest.raises(ValueError):MLAAttention(*geometry,.25)
  helper=MLAAttention(4,32,12,8,.25)
  q=Tensor.ones(1,4,1,12,device='CPU').realize()
  k=Tensor.ones(1,1,32,12,device='CPU').realize()
  for start in (-1,32,True):
    with pytest.raises(ValueError):helper.query_key(q,k,start)
  with pytest.raises(ValueError):helper.query_key(q.half(),k,0)
  with pytest.raises(ValueError):helper.query_key(q,k[:,:,:16],0)
  with pytest.raises(ValueError):configure(object(),1)
  assert configure(object())==0


def test_trained_attention_dimensions_in_model_capture_preserve_logits_and_cache():
  from tinygrad.llm.model import Transformer, TransformerConfig
  config=TransformerConfig(1,320,640,40,40,1e-5,96,96,10000.,32,8,max_context=512,q_lora_rank=32,kv_lora_rank=256)
  generic,prepared=Transformer(config),Transformer(config)
  rng=np.random.default_rng(1803)
  weights={}
  for name,param in nn.state.get_state_dict(generic).items():
    value=rng.normal(0,.02,param.shape).astype(np.float16)
    if 'norm' in name:value+=1
    weights[name]=Tensor(value,device='CPU').realize()
  for model in (generic,prepared):nn.state.load_state_dict(model,weights)
  assert configure(prepared)==1
  runs=[TinyJit(lambda ids,start,_model=model:_model.logits(ids,start).realize()) for model in (generic,prepared)]
  decodes=[TinyJit(lambda ids,start,_model=model:_model.logits(ids,start).realize()) for model in (generic,prepared)]
  startvar=UOp.variable('mla_model_start',0,511)
  for index in range(4):
    ids=Tensor([list(range(1+index,18+index))],device='CPU')
    a,b=[run(ids,startvar.bind(0)).numpy() for run in runs]
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    token=Tensor([[31+index]],device='CPU')
    a,b=[run(token,startvar.bind(17)).numpy() for run in decodes]
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
