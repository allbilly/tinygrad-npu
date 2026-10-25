"""Spark-X2.5/X2.6 reference: XHToken/Spark-X2.6-4B modeling_spark.py.

The independent PyTorch decoder uses exact GELU, head-wise gates and per-layer
partial RoPE. Arithmetic is checked with FP32 caches. Separate FP16 checks cover
the production cache, where rounding-boundary differences can change one ULP.
"""
import itertools, math, os, unittest
from unittest.mock import patch
import numpy as np
import torch
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestSpark(unittest.TestCase):
  def model(self):
    c = TransformerConfig(3,48,96,4,2,1e-6,80,16,5000000.,4,16,max_context=24,
                          ffn_activation='gelu',attn_head_gate=True,sliding_window=4,
                          sliding_layers=(True,True,False),swa_rope_dim=16,swa_rope_theta=10000.)
    model = Transformer(c)
    rng = np.random.default_rng(26)
    for name,p in nn.state.get_state_dict(model).items():
      values = rng.normal(0,.13,p.shape).astype(np.float32)
      if 'norm' in name:values+=1
      p.replace(Tensor(values,device='CPU').realize())
    return model

  def reference(self, model, ids, cache_dtype=torch.float16):
    c = model.config
    w = {k:torch.from_numpy(v.numpy()).float() for k,v in nn.state.get_state_dict(model).items() if 'cache' not in k}
    def norm(x,key):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c.norm_eps)*w[key]
    x = w['token_embd.weight'][torch.tensor(ids)]
    pos = torch.arange(x.shape[1],dtype=torch.float32)
    for i in range(c.num_blocks):
      prefix = f'blk.{i}.'
      # Compute the layer policy independently of the implementation's block configs.
      window = c.sliding_window if c.sliding_layers[i] else 0
      width,theta = (c.swa_rope_dim,c.swa_rope_theta) if window else (c.rope_dim,c.rope_theta)
      angles = pos[:,None]/theta**(torch.arange(0,width,2,dtype=torch.float32)/width)[None,:]
      def rotate(a):
        first,second = a[...,:width].chunk(2,dim=-1)
        return torch.cat((first*angles.cos()-second*angles.sin(),second*angles.cos()+first*angles.sin(),a[...,width:]),dim=-1)
      h = norm(x,prefix+'attn_norm.weight')
      q = rotate((h@w[prefix+'attn_q.weight'].T).reshape(1,-1,c.n_heads,c.head_dim).transpose(1,2))
      k = rotate((h@w[prefix+'attn_k.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2)).to(cache_dtype).float()
      v = (h@w[prefix+'attn_v.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2).to(cache_dtype).float()
      k,v = (a.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1) for a in (k,v))
      visible = pos[:,None]>=pos[None,:]
      if window:visible &= pos[None,:]>pos[:,None]-window
      scores = q@k.transpose(-1,-2)/math.sqrt(c.head_dim)
      attention = scores.masked_fill(~visible,float('-inf')).softmax(-1)@v
      attention *= torch.sigmoid(h@w[prefix+'attn_gate.weight'].T).transpose(1,2).unsqueeze(-1)
      x = x+attention.transpose(1,2).reshape(1,-1,c.n_heads*c.head_dim)@w[prefix+'attn_output.weight'].T
      h = norm(x,prefix+'ffn_norm.weight')
      gate = torch.nn.functional.gelu(h@w[prefix+'ffn_gate.weight'].T,approximate='none')
      x = x+(gate*(h@w[prefix+'ffn_up.weight'].T))@w[prefix+'ffn_down.weight'].T
    return (norm(x[:,-1:],'output_norm.weight')@w['output.weight'].T)[:,-1].numpy()

  def check_cached(self, model, device, fp32_cache=False):
    if fp32_cache:
      for block in model.blk:
        block._init_state(Tensor.zeros(1,1,model.config.dim,device=model.token_embd.weight.device))
        block.cache_kv = block.cache_kv.float().realize()
    cache_dtype = torch.float32 if fp32_cache else torch.float16
    ids = [[7,12,4,8,2,11,5,19,3,9]]
    start = 0
    for end in (6,7,10):
      actual = model.logits(Tensor([ids[0][start:end]],device=device),start).numpy()
      np.testing.assert_allclose(actual,self.reference(model,[ids[0][:end]],cache_dtype),rtol=5e-5,atol=1e-5)
      start = end
    actual = model.logits(Tensor([[1,4,6]],device=device),0).numpy()
    np.testing.assert_allclose(actual,self.reference(model,[[1,4,6]],cache_dtype),rtol=5e-5,atol=1e-5)

  def test_cpu_prefill_cached_single_and_multi_token_across_window(self):
    model = self.model()
    # Existing shortcuts do not implement the new head-wise gate or GELU.
    def forbidden(*args,**kwargs):raise AssertionError('Ungated attention shortcut')
    for block in model.blk:block.prefill_attention=block.decode_attention=forbidden
    self.check_cached(model,'CPU',fp32_cache=True)

  def test_cpu_fp16_cache_error_and_greedy_tokens(self):
    model = self.model()
    ids = [7,12,4,8,2,11,5,19,3,9]
    start = 0
    for end in (6,7,10):
      actual = model.logits(Tensor([ids[start:end]],device='CPU'),start).numpy()
      expected = self.reference(model,[ids[:end]])
      # Budget is an eighth of the FP16 relative spacing, with identical greedy tokens.
      self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),np.finfo(np.float16).eps/8)
      self.assertEqual(int(actual.argmax(-1).item()),int(expected.argmax(-1).item()))
      start = end

  def test_symbolic_chunked_generation_matches_teacher(self):
    model = self.model()
    prompt = [7,12,4,8,2,11,5,19,3]
    generated = list(itertools.islice(model.generate(prompt.copy(),chunk_size=4),3))
    for index,token in enumerate(generated):
      self.assertEqual(token,int(self.reference(model,[prompt+generated[:index]]).argmax(-1).item()))

  def test_gguf_fused_qkv_and_explicit_sliding_rope_policy(self):
    model = self.model()
    c = model.config
    state = {k:v.half().clone() for k,v in nn.state.get_state_dict(model).items()}
    expected = {k:v.numpy() for k,v in state.items()}
    for i in range(c.num_blocks):
      prefix = f'blk.{i}.'
      q,k,v = (state.pop(prefix+'attn_'+name+'.weight') for name in ('q','k','v'))
      state[prefix+'attn_qkv.weight'] = q.cat(k,v,dim=0)
    values = {'attention.head_count':c.n_heads,'attention.head_count_kv':c.n_kv_heads,
              'attention.key_length':c.head_dim,'block_count':c.num_blocks,'embedding_length':c.dim,
              'feed_forward_length':c.hidden_dim,'attention.layer_norm_rms_epsilon':c.norm_eps,'context_length':c.max_context,
              'attention.sliding_window':c.sliding_window,'attention.sliding_window_pattern':list(c.sliding_layers),
              'rope.dimension_count':c.rope_dim,'rope.freq_base':c.rope_theta,
              'rope.dimension_count_swa':c.swa_rope_dim,'rope.freq_base_swa':c.swa_rope_theta}
    kv = {'general.architecture':'spark2_5','tokenizer.ggml.tokens':[str(i) for i in range(c.vocab_size)],
          **{'spark2_5.'+k:v for k,v in values.items()}}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=c.max_context,realize=True)
    for name,p in nn.state.get_state_dict(loaded).items():np.testing.assert_array_equal(p.numpy(),expected[name])
    self.assertEqual([b.config.sliding_window for b in loaded.blk],[4,4,0])
    self.assertEqual([b.config.rope_dim for b in loaded.blk],[16,16,4])
    self.assertEqual([b.config.rope_theta for b in loaded.blk],[10000.,10000.,5000000.])
    self.assertEqual((loaded.config.attn_head_gate,loaded.config.ffn_activation),(True,'gelu'))
    self.check_cached(loaded,'CPU',fp32_cache=True)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_cached_decoder_against_torch(self):
    from tinygrad.llm.kernels.mali import configure
    for pack_weights in (True,False):
      with self.subTest(pack_weights=pack_weights):
        model = self.model()
        for p in nn.state.get_parameters(model):p.replace(p.half().to('CL').realize())
        info = configure(model,fuse_projections=True,pack_weights=pack_weights)
        self.assertEqual(info['attention_blocks'],0)
        self.assertEqual(info['fused_qkv'],3)
        self.assertEqual(info['pack_weights'],pack_weights)
        for block in model.blk:
          self.assertEqual(block.attn_qkv.linear.packed_weight is not None,pack_weights)
          self.assertEqual(block.ffn_gate_up.linear.packed_weight is not None,pack_weights)
        self.check_cached(model,'CL')
        if pack_weights:
          configure(model,fuse_projections=True,pack_weights=False)
          for block in model.blk:
            self.assertIsNone(block.attn_qkv.linear.packed_weight)
            self.assertIsNone(block.ffn_gate_up.linear.packed_weight)
          self.check_cached(model,'CL')
