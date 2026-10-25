"""LFM2 short convolution, mixed attention and cache checks against independent Torch math."""
import itertools, math, os, unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, ShortConvBlock


class TestLFM2(unittest.TestCase):
  def model(self, **overrides):
    config = TransformerConfig(3,32,64,4,2,1e-5,96,8,10000.,8,8,max_context=32,qk_norm=8,
                               conv_kernel=3,conv_layers=(True,False,True))
    model = Transformer(replace(config,**overrides))
    rng = np.random.default_rng(73)
    for name,p in nn.state.get_state_dict(model).items():
      data = rng.normal(0,.1,p.shape).astype(np.float32)
      if 'norm' in name:data+=1
      p.replace(Tensor(data,device='CPU').realize())
    return model

  def reference(self, model, ids, hidden_states=None):
    c = model.config
    parameters = nn.state.get_state_dict(model)
    w = {k:torch.from_numpy(v.numpy()).float() for k,v in parameters.items() if 'cache' not in k}
    def norm(x,name):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c.norm_eps)*w[name]
    x = w['token_embd.weight'][torch.tensor(ids)]
    length = x.shape[1]
    angles = torch.arange(length,dtype=torch.float32)[:,None]/(c.rope_theta**(torch.arange(0,c.head_dim,2)/c.head_dim))[None]
    def rotate(a):
      first,second = a.chunk(2,dim=-1)
      return torch.cat((first*angles.cos()-second*angles.sin(),second*angles.cos()+first*angles.sin()),dim=-1)
    for i in range(c.num_blocks):
      prefix = f'blk.{i}.'
      h = norm(x,prefix+'attn_norm.weight')
      if c.conv_layers[i]:
        b,gate,value = (h@w[prefix+'shortconv.in_proj.weight'].T).chunk(3,dim=-1)
        conv_input = b*value
        conv = torch.nn.functional.conv1d(conv_input.transpose(1,2),w[prefix+'shortconv.conv.weight'][:,None],
                                          padding=c.conv_kernel-1,groups=c.dim)[:,:,:length].transpose(1,2)
        attention = (gate*conv)@w[prefix+'shortconv.out_proj.weight'].T
      else:
        q = (h@w[prefix+'attn_q.weight'].T).reshape(1,-1,c.n_heads,c.head_dim).transpose(1,2)
        k = (h@w[prefix+'attn_k.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2)
        q,k = rotate(norm(q,prefix+'attn_q_norm.weight')),rotate(norm(k,prefix+'attn_k_norm.weight')).half().float()
        v = (h@w[prefix+'attn_v.weight'].T).reshape(1,-1,c.n_kv_heads,c.head_dim).transpose(1,2).half().float()
        k,v = (a.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1) for a in (k,v))
        scores = (q@k.transpose(-1,-2)/math.sqrt(c.head_dim)).masked_fill(~torch.ones(length,length,dtype=torch.bool).tril(),float('-inf'))
        attention = (scores.softmax(-1)@v).transpose(1,2).reshape(1,-1,c.dim)@w[prefix+'attn_output.weight'].T
      x = x+attention
      h = norm(x,prefix+'ffn_norm.weight')
      x = x+(torch.nn.functional.silu(h@w[prefix+'ffn_gate.weight'].T)*(h@w[prefix+'ffn_up.weight'].T))@w[prefix+'ffn_down.weight'].T
      if hidden_states is not None:hidden_states.append(x.numpy().copy())
    return (norm(x[:,-1:],'output_norm.weight')@w['output.weight'].T)[:,-1].numpy()

  def compare_cache(self, model, device):
    ids = [[3,8,12,4,7,6]]
    for start,part in ((0,ids),(6,[[9]]),(7,[[11,5]])):
      if start:ids[0].extend(part[0])
      actual = model.logits(Tensor(part,device=device),start).numpy()
      np.testing.assert_allclose(actual,self.reference(model,ids),rtol=5e-5,atol=7e-6)
    restart = [[7,2,8]]
    np.testing.assert_allclose(model.logits(Tensor(restart,device=device),0).numpy(),self.reference(model,restart),rtol=5e-5,atol=7e-6)

  def test_mixed_decoder_chunks_continuation_and_reset(self):
    model = self.model()
    self.assertTrue(model.has_recurrent_block)
    self.assertEqual(model.prefill_chunk_size(16),16)
    self.compare_cache(model,'CPU')

  def test_loader_layer_list_and_nonfirst_qk_norm(self):
    source = self.model()
    state = {k:v.clone() for k,v in nn.state.get_state_dict(source).items()}
    state['token_embd_norm.weight'] = state.pop('output_norm.weight')
    state.pop('output.weight')
    kv = {'general.architecture':'lfm2','tokenizer.ggml.tokens':[str(i) for i in range(96)],'lfm2.block_count':3,
          'lfm2.attention.head_count':4,'lfm2.attention.head_count_kv':[0,2,0],'lfm2.context_length':32,
          'lfm2.embedding_length':32,'lfm2.feed_forward_length':64,'lfm2.attention.layer_norm_rms_epsilon':1e-5,
          'lfm2.shortconv.l_cache':3}
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})),patch('tinygrad.llm.model.gguf_shard',return_value=state):
      loaded,_ = Transformer.from_gguf('fixture',max_context=32,realize=True)
    self.assertIsInstance(loaded.blk[0],ShortConvBlock)
    self.assertEqual((loaded.config.n_kv_heads,loaded.config.qk_norm),(2,8))
    self.assertEqual(loaded.output.weight.uop,loaded.token_embd.weight.uop)
    np.testing.assert_array_equal(loaded.output_norm.weight.numpy(),source.output_norm.weight.half().numpy())

  def test_symbolic_chunked_generation_matches_full_prefix_teacher(self):
    model = self.model()
    prompt = [3,8,12,4,7,6]
    actual = list(itertools.islice(model.generate(prompt.copy(),chunk_size=4,temperature=0),3))
    expected = []
    for _ in range(3):
      token = int(self.reference(model,[prompt+expected]).argmax())
      expected.append(token)
    self.assertEqual(actual,expected)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_mixed_decoder_cache(self):
    from tinygrad.llm.kernels.mali import configure
    model = self.model(dim=64,hidden_dim=128,n_heads=4,n_kv_heads=2,head_dim=16,rope_dim=16,v_head_dim=16,qk_norm=16)
    for p in nn.state.get_parameters(model):p.replace(p.half().to('CL').realize())
    configure(model)
    self.compare_cache(model,'CL')
