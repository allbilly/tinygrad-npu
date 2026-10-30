"""Owned CPU token storage preserves dynamic prefix and cached-generation semantics."""
import gc, itertools
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, _cpu_token_buffer


class TokenModel:
  generate = Transformer.generate
  __call__ = Transformer.__call__
  get_start_pos = Transformer.get_start_pos
  has_recurrent_block = False

  def __init__(self):
    self.max_context,self._cached_tokens,self._greedy_generation = 256,[],False
    self.token_embd = SimpleNamespace(weight=Tensor.ones(1,device='CPU').realize())
    self.blk = [SimpleNamespace(_reusable_prefix_len=lambda prefix,cached:prefix)]
    def forward(tokens,pos,*args):
      return ((tokens.sum(axis=1,keepdim=True)+Tensor(pos,device='CPU',dtype=dtypes.int32))%1009).cast(dtypes.int32).realize()
    self.greedy_prefill_jit,self.greedy_rollout_jit = TinyJit(forward),TinyJit(forward)
    self.prefill_jit,self.rollout_jit = TinyJit(forward),TinyJit(forward)

  def prefill_chunk_size(self,requested):return requested


def test_ready_prefix_view_does_not_schedule_an_input_copy():
  tokens=list(range(70))
  x=_cpu_token_buffer(tokens,256)
  start=UOp.variable('start_pos',0,255).bind(7)
  count=UOp.variable('toks',1,64).bind(17)
  view=x[:,start:start+count]
  assert view.uop.is_realized
  with patch.object(Tensor,'realize',side_effect=AssertionError('ready input scheduled')):
    bufs,values,_,_=_prepare_jit_inputs((view,start),{})
  assert values=={'start_pos':7,'toks':17} and len(bufs)==1
  del tokens,x
  gc.collect()
  concrete=Tensor(view.uop.substitute({start:UOp.const(7),count:UOp.const(17)}))
  np.testing.assert_array_equal(concrete.numpy(),np.arange(7,24,dtype=np.int32).reshape(1,17))


@pytest.mark.parametrize('temperature', (0.0,0.6))
@pytest.mark.parametrize('chunk', (32,64,128))
def test_generation_rebinding_and_cached_prefix_match(temperature,chunk):
  outputs=[]
  for enabled in (0,1):
    with Context(DEV='CPU',ONPU_CPU_TOKEN_INPUTS=enabled):
      model=TokenModel()
      runs=[]
      for n in (17,65,128,129,17):
        prompt=list(range(1,n+1))
        model._cached_tokens=[]
        result=list(itertools.islice(model.generate(prompt.copy(),chunk_size=chunk,temperature=temperature),5))
        runs.append(result)
        # Continue from a completed prefix and then test a divergent prompt.
        continued=prompt+result+[7,8]
        runs.append(list(itertools.islice(model.generate(continued,chunk_size=chunk,temperature=temperature),3)))
        model._cached_tokens=[]
        gc.collect()
      outputs.append(runs)
  assert outputs[0]==outputs[1]


def test_cpu_token_buffer_checks_extent_and_zeroes_unused_capacity():
  for tokens,size in (([],8),([1,2],1)):
    with pytest.raises(ValueError):_cpu_token_buffer(tokens,size)
  x=_cpu_token_buffer([3,7,9],8)
  np.testing.assert_array_equal(x.numpy(),[[3,7,9,0,0,0,0,0]])


def test_one_token_cached_prompt_keeps_the_decode_jit_input_contract():
  results=[]
  for enabled in (0,1):
    with Context(DEV='CPU',ONPU_CPU_TOKEN_INPUTS=enabled):
      model=TokenModel()
      prompt=list(range(1,65))
      for _ in range(4):
        model._cached_tokens=[]
        list(itertools.islice(model.generate(prompt.copy(),chunk_size=64),5))
      results.append(list(itertools.islice(model.generate(prompt.copy(),chunk_size=1),3)))
  assert results[0]==results[1]
