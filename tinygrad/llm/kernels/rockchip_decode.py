"""Scalar decoder adapter for the qualified bounded CPU/NPU generation loop."""
from __future__ import annotations
import contextlib
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.llm.kernels.rockchip_gpt2_decode import DecodeLoop
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit

def prepare_scalar_decode(model):
  def step(tokens, start_pos):
    token = Tensor.full((1,1),tokens,device='CPU',dtype=dtypes.int)
    return model.greedy_forward(token,start_pos).realize()
  jit = TinyJit(step)
  model.greedy_scalar_jit = jit
  return jit

class ScalarDecoder:
  def __init__(self, model, jit, pool=None, helper=None):
    self.model,self.jit,self.pool,self.helper = model,jit,pool,helper
    self.context,self.vocab = model.max_context,model.config.vocab_size
    self.loop:DecodeLoop|None = None

  @property
  def captured(self): return self.jit.captured

  def scope(self):
    stack = contextlib.ExitStack()
    if self.pool is not None: stack.enter_context(self.pool)
    if self.helper is not None: stack.enter_context(self.helper)
    return stack

  def __call__(self, token:int, position:int, count:int) -> list[int]:
    if not 0<=count<=self.context-position or not 0<=token<self.vocab or position<1:
      raise ValueError('Scalar decode exceeds token or context bounds')
    result = []
    while count:
      captured = self.captured
      native = captured.native if isinstance(captured,RockchipCapturedJit) else None
      if (native is not None and not native.roots and native.variables==['start_pos','tokens'] and native.valid([])
          and captured.ret.device=='CPU' and captured.ret.dtype is dtypes.int and captured.ret.numel()==1):
        if self.loop is None or not self.loop.valid(): self.loop = DecodeLoop(self,self.context,self.vocab)
        generated = self.loop(token,position,min(count,32))
      else:
        tok = Variable('tokens',0,self.vocab-1).bind(token)
        pos = Variable('start_pos',1,self.context-1).bind(position)
        generated = [int(self.model.greedy_scalar_jit(tok,pos).item())]
      result.extend(generated)
      position,count,token = position+len(generated),count-len(generated),generated[-1]
    return result
