"""Bounded scalar batches publish the real state prefix, including early close."""
from itertools import islice
import pytest
from tinygrad.llm.kernels.rockchip_decode import install_text_generation


class Model:
  def __init__(self,context=20):
    self.max_context=context
    self._cached_tokens=[]
    self.state_pos=0
  def generate(self,tokens,chunk_size=None,temperature=0.0):
    while len(tokens)<self.max_context:
      self.state_pos=len(tokens)
      self._cached_tokens=tokens.copy()
      tokens.append((tokens[-1]+1)%97)
      yield tokens[-1]


class Decoder:
  def __init__(self,model,fail=False):
    self.model,self.fail,self.calls=model,fail,0
  def __call__(self,token,position,count,*,stop_tokens=()):
    assert position==self.model.state_pos
    self.calls+=1
    output=[]
    for _ in range(count):
      self.model.state_pos+=1
      if self.fail:raise RuntimeError('partial state update')
      token=(token+1)%97
      output.append(token)
      if token in stop_tokens:break
    return output


@pytest.mark.parametrize('batch',[1,4,32])
def test_fixed_budget_and_context_preserve_completed_prefix(batch):
  model=Model()
  decoder=Decoder(model)
  install_text_generation(model,decoder,batch_size=batch,limit=lambda:8)
  tokens=[1,2,3]
  assert list(model.generate(tokens))==list(range(4,12))
  assert len(tokens)==11 and len(model._cached_tokens)==model.state_pos==10
  model=Model(10)
  install_text_generation(model,Decoder(model),batch_size=batch)
  tokens=[1,2,3]
  assert list(model.generate(tokens))==list(range(4,11))
  assert len(tokens)==10 and len(model._cached_tokens)==model.state_pos==9


def test_partial_consumer_close_cannot_publish_a_shorter_recurrent_state():
  model=Model()
  install_text_generation(model,Decoder(model),batch_size=4,limit=lambda:8)
  tokens=[1,2,3]
  stream=model.generate(tokens)
  assert list(islice(stream,3))==[4,5,6]
  assert len(tokens)==6 and len(model._cached_tokens)==model.state_pos==7
  stream.close()
  assert model._cached_tokens==[1,2,3,4,5,6,7]


def test_eos_failure_cancellation_and_sampling_contracts():
  model=Model()
  install_text_generation(model,Decoder(model),batch_size=32,stop_tokens=(7,),limit=lambda:10)
  assert list(model.generate([1,2,3]))==[4,5,6,7]
  assert len(model._cached_tokens)==model.state_pos==6
  model=Model()
  install_text_generation(model,Decoder(model,fail=True),limit=lambda:8)
  stream=model.generate([1,2,3])
  assert next(stream)==4
  with pytest.raises(RuntimeError,match='partial state'):next(stream)
  assert model._cached_tokens==[]
  model=Model()
  checks=[]
  def check():
    checks.append(1)
    if len(checks)==3:raise RuntimeError('cancelled')
  install_text_generation(model,Decoder(model),limit=lambda:8,check=check)
  with pytest.raises(RuntimeError,match='cancelled'):list(model.generate([1,2,3]))
  assert model._cached_tokens==[]
  model=Model(10)
  decoder=Decoder(model)
  install_text_generation(model,decoder,limit=lambda:2)
  assert len(list(model.generate([1,2,3],temperature=.6)))==7 and decoder.calls==0


@pytest.mark.parametrize('batch',[0,33,True])
def test_invalid_batch_fails_before_generation(batch):
  model=Model()
  with pytest.raises(ValueError,match='batch size'):install_text_generation(model,Decoder(model),batch_size=batch)


def test_cold_scalar_trace_enters_the_models_shared_scope(monkeypatch):
  from contextlib import contextmanager
  from types import SimpleNamespace
  from tinygrad import Tensor
  from tinygrad.llm.kernels import rockchip_shared_pool
  from tinygrad.llm.kernels.rockchip_decode import ScalarDecoder

  pool=object()
  events=[]
  active=False

  @contextmanager
  def scope():
    nonlocal active
    assert not active
    active=True
    events.append('enter')
    try:yield
    finally:
      active=False
      events.append('exit')

  shared=SimpleNamespace(pool=pool,scope=scope)
  model=SimpleNamespace(max_context=16,config=SimpleNamespace(vocab_size=32))

  def step(token,position):
    assert active
    events.append('trace')
    return Tensor([5],device='CPU')

  model.greedy_scalar_jit=step
  monkeypatch.setattr(rockchip_shared_pool,'model_shared_host_pool',lambda m:shared)
  decoder=ScalarDecoder(model,SimpleNamespace(captured=None),pool=pool)
  assert decoder(3,1,2)==[5,5]
  assert events==['enter','trace','exit','enter','trace','exit']
