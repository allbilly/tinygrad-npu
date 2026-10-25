"""DeepSeek-OCR text decoder: ordinary Llama MHA, dense first layer, then MoE.

The official language config leaves the MLA-only qk_rope_head_dim at zero;
its use_mla=False path still uses LlamaAttention's full-head rotary positions.
"""
import os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestDeepSeekOCR(unittest.TestCase):
  def fixture(self):
    config = TransformerConfig(2,32,16,4,4,1e-6,64,8,10000.,8,8,max_context=16,
                               num_experts=4,num_experts_per_tok=2,leading_dense_blocks=1,dense_hidden_dim=64,
                               shared_expert_dim=32,shared_expert_gate=False)
    model = Transformer(config)
    rng = np.random.default_rng(313)
    state = {}
    for name,param in nn.state.get_state_dict(model).items():
      value = rng.normal(0,.15,param.shape).astype(np.float16)
      if 'norm' in name:value+=1
      state[name] = Tensor(value,device='CPU').realize()
    return config,state

  def load(self, state, arch='deepseek2-ocr'):
    values = {'context_length':16,'block_count':2,'embedding_length':32,'feed_forward_length':64,
              'attention.head_count':4,'attention.head_count_kv':4,'attention.layer_norm_rms_epsilon':1e-6,
              'rope.dimension_count':0,'expert_count':4,'expert_used_count':2,'expert_feed_forward_length':16,
              'leading_dense_block_count':1,'expert_shared_count':2}
    kv = {'general.architecture':arch,'tokenizer.ggml.tokens':[str(i) for i in range(64)],
          **{arch+'.'+key:value for key,value in values.items()}}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})), \
         patch('tinygrad.llm.model.gguf_shard',return_value=dict(state)):
      return Transformer.from_gguf('fixture',max_context=16,realize=True)[0]

  def reference(self, state, tokens):
    w = {key:value.numpy().astype(np.float32) for key,value in state.items()}
    def norm(x,key):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w[key]
    def linear(x,key):return x@w[key+'.weight'].T
    def softmax(x):
      ex = np.exp(x-x.max(axis=-1,keepdims=True))
      return ex/ex.sum(axis=-1,keepdims=True)
    def silu(x):return x/(1+np.exp(-x))
    x = w['token_embd.weight'][tokens]
    length = len(tokens)
    phase = np.arange(length,dtype=np.float32)[:,None]*(10000.**(-np.arange(0,8,2,dtype=np.float32)/8))
    def rotary(x):
      first,second = np.split(x,2,axis=-1)
      return np.concatenate((first*np.cos(phase)-second*np.sin(phase),second*np.cos(phase)+first*np.sin(phase)),axis=-1)
    for layer in range(2):
      p = f'blk.{layer}.'
      h = norm(x,p+'attn_norm.weight')
      q,k,v = [linear(h,p+'attn_'+name).reshape(length,4,8).transpose(1,0,2) for name in ('q','k','v')]
      q,k = rotary(q),rotary(k).astype(np.float16).astype(np.float32)
      v = v.astype(np.float16).astype(np.float32)
      scores = q@k.transpose(0,2,1)/np.sqrt(8)
      scores[:,np.triu_indices(length,1)[0],np.triu_indices(length,1)[1]] = -np.inf
      h = (softmax(scores)@v).transpose(1,0,2).reshape(length,32)
      x = x+linear(h,p+'attn_output')
      h = norm(x,p+'ffn_norm.weight')
      if layer==0:ffn = linear(silu(linear(h,p+'ffn_gate'))*linear(h,p+'ffn_up'),p+'ffn_down')
      else:
        probabilities = softmax(linear(h,p+'ffn_gate_inp'))
        ffn = np.zeros_like(h)
        for position,row in enumerate(h):
          for expert in np.argsort(probabilities[position])[-2:]:
            gate = row@w[p+'ffn_gate_exps.weight'][expert].T
            up = row@w[p+'ffn_up_exps.weight'][expert].T
            ffn[position] += probabilities[position,expert]*((silu(gate)*up)@w[p+'ffn_down_exps.weight'][expert].T)
        ffn += linear(silu(linear(h,p+'ffn_gate_shexp'))*linear(h,p+'ffn_up_shexp'),p+'ffn_down_shexp')
      x = x+ffn
    return linear(norm(x[-1],'output_norm.weight'),'output')

  def test_loader_full_rotary_and_moe_match_independent_prefill_and_cached_decode(self):
    _,state = self.fixture()
    model = self.load(state)
    self.assertEqual(model.config.rope_dim,8)
    self.assertFalse(model.config.norm_topk_prob)
    self.assertFalse(model.config.shared_expert_gate)
    self.assertFalse(hasattr(model.blk[0],'ffn_gate_exps'))
    self.assertTrue(hasattr(model.blk[1],'ffn_gate_exps'))
    for ids,start in (([3,7,2],0),([9],3)):
      expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
      actual = model.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
      np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=3e-4)

  def test_zero_rotary_other_architecture_is_preserved(self):
    _,state = self.fixture()
    self.assertEqual(self.load(state,arch='deepseek2').config.rope_dim,0)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_prefill_and_cached_decode_against_numpy(self):
    from tinygrad.llm.kernels.mali import configure
    _,state = self.fixture()
    model = self.load(state)
    for param in nn.state.get_parameters(model):param.to_('CL')
    configure(model,pack_weights=False)
    for ids,start in (([3,7,2],0),([9],3)):
      expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
      actual = model.logits(Tensor([ids],device='CL'),start).numpy().ravel()
      np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=3e-4)
