import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestPhi3Loader(unittest.TestCase):
  def test_fused_qkv_and_gate_up_match_separate_projection_decoder(self):
    # HF Phi3 and GGUF store Q/K/V and gate/up in concatenated output rows.
    config = TransformerConfig(2,64,128,4,2,1e-5,96,16,10000.,16,16,max_context=32)
    model = Transformer(config)
    rng = np.random.default_rng(19)
    for name,p in nn.state.get_state_dict(model).items():
      values = rng.normal(0,.1,p.shape).astype(np.float16)
      if 'norm' in name:values+=1
      p.replace(Tensor(values,device='CPU').realize())
    state = {k:v.clone() for k,v in nn.state.get_state_dict(model).items()}
    for i in range(config.num_blocks):
      prefix = f'blk.{i}.'
      q,k,v = (state.pop(prefix+'attn_'+n+'.weight') for n in ('q','k','v'))
      state[prefix+'attn_qkv.weight'] = q.cat(k,v,dim=0)
      gate,up = state.pop(prefix+'ffn_gate.weight'),state.pop(prefix+'ffn_up.weight')
      state[prefix+'ffn_up.weight'] = gate.cat(up,dim=0)
    kv = {'general.architecture':'phi3','tokenizer.ggml.tokens':[str(i) for i in range(config.vocab_size)],
          'phi3.attention.head_count':4,'phi3.attention.head_count_kv':2,'phi3.block_count':2,'phi3.embedding_length':64,
          'phi3.feed_forward_length':128,'phi3.attention.layer_norm_rms_epsilon':1e-5,'phi3.context_length':32}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=32,realize=True)
    expected = nn.state.get_state_dict(model)
    for name,value in nn.state.get_state_dict(loaded).items():np.testing.assert_array_equal(value.numpy(),expected[name].numpy())
    for ids,start in (([[3,7,2]],0),([[9]],3)):
      tokens = Tensor(ids,device='CPU')
      np.testing.assert_array_equal(loaded.logits(tokens,start).numpy(),model.logits(tokens,start).numpy())
