"""Scalar decoder adapter for the qualified bounded CPU/NPU generation loop."""
from __future__ import annotations
import contextlib
from collections.abc import Callable, Iterator
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

  @contextlib.contextmanager
  def scope(self) -> Iterator[None]:
    from tinygrad.llm.kernels.rockchip_shared_pool import model_shared_host_pool
    # The C loop also invokes rewritten preparation kernels. Keep the shared
    # dispatcher active, and unwind all entered resources if a helper fails.
    with contextlib.ExitStack() as stack:
      shared=model_shared_host_pool(self.model)
      if shared is not None and shared.pool is self.pool:stack.enter_context(shared.scope())
      elif self.pool is not None:stack.enter_context(self.pool)
      from tinygrad.llm.kernels.rockchip_pool import ONPU_DECODE_HOST_WFE
      if self.pool is not None and ONPU_DECODE_HOST_WFE:stack.enter_context(self.pool.wait_policy(True))
      if self.helper is not None:stack.enter_context(self.helper)
      yield

  def __call__(self, token:int, position:int, count:int, *, stop_tokens:tuple[int,...]=()) -> list[int]:
    if not 0<=count<=self.context-position or not 0<=token<self.vocab or position<1:
      raise ValueError('Scalar decode exceeds token or context bounds')
    if len(stop_tokens)>16 or any(type(t) is not int or not 0<=t<self.vocab for t in stop_tokens):
      raise ValueError('Invalid scalar decode stop tokens')
    stop_tokens=tuple(sorted(set(stop_tokens)))
    result = []
    while count:
      captured = self.captured
      native = captured.native if isinstance(captured,RockchipCapturedJit) else None
      if (native is not None and not native.roots and native.variables==['start_pos','tokens'] and native.valid([])
          and captured.ret.device=='CPU' and captured.ret.dtype is dtypes.int and captured.ret.numel()==1):
        if self.loop is None or not self.loop.valid() or self.loop.stop_tokens!=stop_tokens:
          self.loop = DecodeLoop(self,self.context,self.vocab,stop_tokens)
        generated = self.loop(token,position,min(count,32))
      else:
        tok = Variable('tokens',0,self.vocab-1).bind(token)
        pos = Variable('start_pos',1,self.context-1).bind(position)
        # Cold tracing must see the same CPU worker scope as the later C loop.
        # Leave IRQ-helper ownership to the ordinary JIT phase wrapper here.
        from tinygrad.llm.kernels.rockchip_shared_pool import model_shared_host_pool
        shared = model_shared_host_pool(self.model)
        with shared.scope() if shared is not None and shared.pool is self.pool else contextlib.nullcontext():
          generated = [int(self.model.greedy_scalar_jit(tok,pos).item())]
      result.extend(generated)
      position,count,token = position+len(generated),count-len(generated),generated[-1]
      if token in stop_tokens:break
    return result


def install_text_generation(model, decoder:ScalarDecoder, *, batch_size:int=4, stop_tokens:tuple[int,...]=(),
                            limit:Callable[[],int|None]|None=None, batch:Callable[[],int]|None=None,
                            check:Callable[[],None]|None=None) -> Callable:
  """Keep first-token generation, then replay bounded greedy continuations.

  Publish the actual completed state prefix before yielding a batch. Early
  consumer close/cancellation cannot expose a shorter, stale recurrent cache.
  Sampling retains the original generator. Install before warmup/capture.
  """
  if type(batch_size) is not int or not 1<=batch_size<=32:raise ValueError('Invalid text decode batch size')
  original=model.generate
  stops=tuple(sorted(set(stop_tokens)))
  def generate(tokens:list[int], chunk_size:int|None=None, temperature:float=0.0) -> Iterator[int]:
    if temperature!=0:
      yield from original(tokens,chunk_size=chunk_size,temperature=temperature)
      return
    budget=limit() if limit is not None else None
    width=batch() if batch is not None else batch_size
    if budget is not None and (type(budget) is not int or budget<1):raise ValueError('Invalid text decode token budget')
    if type(width) is not int or not 1<=width<=32:raise ValueError('Invalid text decode batch size')
    try:
      if check is not None:check()
      iterator=original(tokens,chunk_size=chunk_size,temperature=temperature)
      try:first=next(iterator)
      except StopIteration:return
      finally:iterator.close()
      yield first
      if first in stops:return
      produced,position=1,len(tokens)-1
      while len(tokens)<model.max_context and (budget is None or produced<budget):
        if check is not None:check()
        count=min(width,model.max_context-len(tokens),budget-produced if budget is not None else width)
        consumed=list(tokens)
        model._cached_tokens=[]
        generated=decoder(tokens[-1],position,count,stop_tokens=stops)
        if not generated or len(generated)>count:raise RuntimeError('Invalid scalar decode result')
        model._cached_tokens=consumed+generated[:-1]
        for token in generated:
          if check is not None:check()
          tokens.append(token)
          yield token
          if token in stops:return
        position+=len(generated)
        produced+=len(generated)
    except GeneratorExit:raise
    except BaseException:
      model._cached_tokens=[]
      raise
  model.generate=generate
  return original
