"""Independent FP32 reference for MiniCPM embedding/residual/logit and LongRoPE scaling.

Formulas: openbmb/MiniCPM4-0.5B modeling_minicpm.py, revision
5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630. KV storage follows this backend's FP16 cache.
"""
import math, unittest
import numpy as np
import torch
from unittest.mock import patch
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, precompute_freqs_cis


class TestMiniCPM(unittest.TestCase):
  def make_model(self):
    config = TransformerConfig(2,32,64,4,2,1e-5,96,8,10000.,8,8,max_context=32,
                               embedding_scale=12.,residual_scale=1.4/math.sqrt(2),logit_scale=4.,
                               rope_factors=(1.0004,1.5,3.,7.),rope_mscale=1.1)
    model = Transformer(config)
    rng = np.random.default_rng(125)
    for key,p in nn.state.get_state_dict(model).items():
      data = rng.normal(0,.12,p.shape).astype(np.float32)
      if 'norm' in key:data+=1
      p.replace(Tensor(data,device='CPU').realize())
    return model

  def reference(self, model, ids):
    c = model.config
    w = {k:torch.from_numpy(v.numpy()) for k,v in nn.state.get_state_dict(model).items() if 'cache' not in k}
    def norm(x,key):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c.norm_eps)*w[key]
    x = w['token_embd.weight'][torch.tensor(ids)]*c.embedding_scale
    positions = torch.arange(x.shape[1],dtype=torch.float32)
    inv = torch.pow(torch.tensor(c.rope_theta),-torch.arange(0,c.rope_dim,2,dtype=torch.float32)/c.rope_dim)/torch.tensor(c.rope_factors)
    angles = positions[:,None]*inv[None,:]
    cos,sin = angles.cos()*c.rope_mscale,angles.sin()*c.rope_mscale
    def rotate(q):
      a,b = q.chunk(2,dim=-1)
      return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1)
    for i in range(c.num_blocks):
      prefix = f'blk.{i}.'
      h = norm(x,prefix+'attn_norm.weight')
      q = rotate((h@w[prefix+'attn_q.weight'].T).reshape(1,-1,c.n_heads,c.head_dim).transpose(1,2))
      k = rotate((h@w[prefix+'attn_k.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2)).half().float()
      v = (h@w[prefix+'attn_v.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2).half().float()
      k,v = (a.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1) for a in (k,v))
      scores = q@k.transpose(-1,-2)/math.sqrt(c.head_dim)
      mask = torch.ones(x.shape[1],x.shape[1],dtype=torch.bool).tril()
      attention = (scores.masked_fill(~mask,float('-inf')).softmax(-1)@v).transpose(1,2).reshape(1,-1,c.dim)
      x = x+(attention@w[prefix+'attn_output.weight'].T)*c.residual_scale
      h = norm(x,prefix+'ffn_norm.weight')
      gate,up = h@w[prefix+'ffn_gate.weight'].T,h@w[prefix+'ffn_up.weight'].T
      x = x+(torch.nn.functional.silu(gate)*up@w[prefix+'ffn_down.weight'].T)*c.residual_scale
    return ((norm(x[:,-1:], 'output_norm.weight')/c.logit_scale)@w['output.weight'].T)[:,-1].numpy()

  def test_scaled_full_decoder_and_cached_continuation_against_torch(self):
    model = self.make_model()
    ids = [[5,21,17,32,8,6,4]]
    np.testing.assert_allclose(model.logits(Tensor(ids,device='CPU'),0).numpy(),self.reference(model,ids),rtol=3e-5,atol=3e-6)
    for next_id in (12,9):
      start = len(ids[0])
      ids[0].append(next_id)
      out = model.logits(Tensor([[next_id]],device='CPU'),start).numpy()
      np.testing.assert_allclose(out,self.reference(model,ids),rtol=3e-5,atol=3e-6)

  def test_longrope_factors_and_attention_magnitude(self):
    factors = (1.,2.,4.,8.)
    freqs = precompute_freqs_cis(8,512,10000.,device='CPU',factors=factors,mscale=1.2).numpy()
    angles = np.arange(512,dtype=np.float32)[:,None]/(10000.**(np.arange(0,8,2,dtype=np.float32)/8)*np.array(factors,np.float32))
    np.testing.assert_allclose(freqs,np.concatenate((np.cos(angles),np.sin(angles)),axis=-1)*1.2,atol=2e-5,rtol=1e-5)
    for invalid in ((1.,2.),(1.,0.,2.,3.),(1.,math.nan,2.,3.)):
      with self.subTest(factors=invalid),self.assertRaises(ValueError):precompute_freqs_cis(8,32,factors=invalid)

  def test_gguf_loader_restores_head_layout_and_preserves_fp32_rope_factors(self):
    source = self.make_model()
    c = source.config
    weights = {k:v.clone() for k,v in nn.state.get_state_dict(source).items()}
    for i in range(c.num_blocks):
      for name,heads in (('q',c.n_heads),('k',c.n_kv_heads)):
        key = f'blk.{i}.attn_{name}.weight'
        weights[key] = weights[key].reshape(heads,2,c.head_dim//2,c.dim).permute(0,2,1,3).reshape(weights[key].shape)
    weights['rope_factors_short.weight'] = Tensor(c.rope_factors)
    weights['rope_factors_long.weight'] = Tensor(c.rope_factors)
    kv = {'general.architecture':'minicpm','tokenizer.ggml.tokens':[str(i) for i in range(c.vocab_size)],
          'minicpm.attention.head_count':c.n_heads,'minicpm.attention.head_count_kv':c.n_kv_heads,
          'minicpm.block_count':c.num_blocks,'minicpm.embedding_length':c.dim,'minicpm.feed_forward_length':c.hidden_dim,
          'minicpm.attention.layer_norm_rms_epsilon':c.norm_eps,'minicpm.context_length':32,
          'minicpm.embedding_scale':12.,'minicpm.residual_scale':c.residual_scale,'minicpm.logit_scale':4.,
          'minicpm.rope.scaling.type':'longrope'}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=weights):
      loaded,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    for before,after in zip(source.blk,loaded.blk):
      np.testing.assert_array_equal(after.attn_q.weight.numpy(),before.attn_q.weight.half().numpy())
      np.testing.assert_array_equal(after.attn_k.weight.numpy(),before.attn_k.weight.half().numpy())
    self.assertEqual((loaded.config.embedding_scale,loaded.config.logit_scale),(12.,4.))
    self.assertAlmostEqual(loaded.config.residual_scale,c.residual_scale)
    self.assertEqual(loaded.config.rope_factors,tuple(float(v) for v in np.array(c.rope_factors,np.float32)))
    self.assertNotEqual(loaded.config.rope_factors[0],float(np.float16(c.rope_factors[0])))

  def test_internlm2_converted_projections_restore_head_layout(self):
    # llama.cpp conversion/internlm.py splits the original grouped QKV and
    # applies Llama's interleaving permutation to the separate Q/K tensors.
    model = self.make_model()
    c = model.config
    state = {k:v.clone() for k,v in nn.state.get_state_dict(model).items()}
    for i in range(c.num_blocks):
      for name,heads in (('q',c.n_heads),('k',c.n_kv_heads)):
        key = f'blk.{i}.attn_{name}.weight'
        state[key] = state[key].reshape(heads,2,c.head_dim//2,c.dim).permute(0,2,1,3).reshape(state[key].shape)
    kv = {'general.architecture':'internlm2','tokenizer.ggml.tokens':[str(i) for i in range(c.vocab_size)],
          'internlm2.attention.head_count':c.n_heads,'internlm2.attention.head_count_kv':c.n_kv_heads,
          'internlm2.block_count':c.num_blocks,'internlm2.embedding_length':c.dim,'internlm2.feed_forward_length':c.hidden_dim,
          'internlm2.attention.layer_norm_rms_epsilon':c.norm_eps,'internlm2.context_length':32}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    for before,after in zip(model.blk,loaded.blk):
      np.testing.assert_array_equal(after.attn_q.weight.numpy(),before.attn_q.weight.half().numpy())
      np.testing.assert_array_equal(after.attn_k.weight.numpy(),before.attn_k.weight.half().numpy())
    self.assertEqual((loaded.config.embedding_scale,loaded.config.residual_scale,loaded.config.logit_scale),(1.,1.,1.))
