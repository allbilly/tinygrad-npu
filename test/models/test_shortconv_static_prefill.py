"""Static first chunks retain short-convolution state and generation contracts."""
import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.helpers import Context
from tinygrad.llm.model import ShortConvBlock, ShortConvolution, Transformer, TransformerBlock, TransformerConfig, _static_prefill_supported


@pytest.mark.parametrize('dtype',(dtypes.float,dtypes.half))
@pytest.mark.parametrize('kernel',(3,4))
def test_literal_zero_and_dynamic_zero_preserve_outputs_and_history(dtype,kernel):
  rng=np.random.default_rng(309)
  config=SimpleNamespace(dim=8,conv_kernel=kernel,max_context=32)
  blocks=[ShortConvolution(config),ShortConvolution(config)]
  for field,shape in [('in_proj',(24,8)),('out_proj',(8,8))]:
    values=rng.normal(0,.1,shape).astype(np.float32)
    for block in blocks:getattr(block,field).weight=Tensor(values,device='CPU',dtype=dtype).realize()
  values=rng.normal(0,.1,(8,kernel)).astype(np.float32)
  for block in blocks:block.conv['weight']=Tensor(values,device='CPU',dtype=dtype).realize()
  with Context(ONPU_SHORTCONV_STATIC_PREFILL=1):
    for rows,pos in [(8,0),(1,8),(5,0),(1,5),(8,0)]:
      x=Tensor(rng.normal(0,.3,(1,rows,8)).astype(np.float32),device='CPU',dtype=dtype).realize()
      for block in blocks:block.init_state(x)
      expected=blocks[0](x,UOp.variable('shortconv_reference_pos',0,31).bind(pos)).numpy()
      actual=blocks[1](x,pos).numpy()
      assert actual.tobytes()==expected.tobytes()
      assert blocks[0].cache_conv.numpy().tobytes()==blocks[1].cache_conv.numpy().tobytes()


def test_only_explicitly_enabled_shortconv_and_ordinary_attention_can_extend_static_support():
  short=ShortConvBlock.__new__(ShortConvBlock)
  dense=TransformerBlock.__new__(TransformerBlock)
  model=SimpleNamespace(has_recurrent_block=True,blk=[short,dense])
  with Context(ONPU_SHORTCONV_STATIC_PREFILL=0):assert not _static_prefill_supported(model)
  with Context(ONPU_SHORTCONV_STATIC_PREFILL=1):
    assert _static_prefill_supported(model)
    for blocks in ([],[short,object()],[SimpleNamespace()]):
      model.blk=blocks
      assert not _static_prefill_supported(model)
  model.has_recurrent_block=False
  with Context(ONPU_SHORTCONV_STATIC_PREFILL=0):assert _static_prefill_supported(model)


def test_changed_partial_cached_and_early_closed_generation_preserve_real_state():
  def run(static):
    with Context(DEV='CPU',ONPU_CPU_TOKEN_INPUTS=1,ONPU_STATIC_PREFILL=1,ONPU_SHORTCONV_STATIC_PREFILL=static):
      Tensor.manual_seed(310)
      model=Transformer(TransformerConfig(2,8,16,2,2,1e-5,32,4,10000.,0,4,max_context=32,
                                          conv_kernel=3,conv_layers=(True,True)))
      for weight in nn.state.get_parameters(model):weight.assign(Tensor.rand(*weight.shape,device='CPU')*.2-.1).realize()
      original=model.greedy_uncached_jit
      static_calls=[]
      def first(tokens):
        static_calls.append(tokens.shape)
        return original(tokens)
      model.greedy_uncached_jit=first
      results=[]
      for prompt in ([1,2,3,4,5,6,7,8],[3,4,5,6,7,8,9,10],[1,2,3,4,5],[2,3,4,5,6,7,8,9]):
        tokens=list(prompt)
        stream=model.generate(tokens,chunk_size=8)
        generated=list(itertools.islice(stream,4))
        stream.close()
        results.append((generated,list(model._cached_tokens),[b.shortconv.cache_conv.numpy().tobytes() for b in model.blk]))
        # Continue the completed history after closing the earlier consumer.
        stream=model.generate(tokens,chunk_size=8)
        generated=list(itertools.islice(stream,2))
        stream.close()
        results.append((generated,list(model._cached_tokens),[b.shortconv.cache_conv.numpy().tobytes() for b in model.blk]))
      return results,static_calls
  expected,old_calls=run(0)
  actual,new_calls=run(1)
  assert actual==expected
  assert old_calls==[] and len(new_calls)==3
