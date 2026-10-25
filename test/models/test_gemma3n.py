"""Independent NumPy reference for Gemma3n's AltUp/LAUREL decoder equations."""
import itertools, math, os, unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.gemma3n import Gemma3nBlock, Gemma3nTransformer
from tinygrad.llm.model import TransformerConfig


class TestGemma3n(unittest.TestCase):
  def model(self):
    config = TransformerConfig(4,32,64,4,1,1e-6,40,8,1000000.,8,8,max_context=16,
                               embedding_scale=math.sqrt(32),final_logit_softcap=3.,sliding_window=2,swa_rope_theta=10000.)
    model = Gemma3nTransformer(config,(64,)*4,(True,False,True,False),2,4,32,4,0,4,(.7,-math.inf,1.1,-math.inf),7)
    rng = np.random.default_rng(30303)
    for name,param in nn.state.get_state_dict(model).items():
      values = rng.normal(1 if 'norm' in name or 'correct_scale' in name else 0,.06,param.shape).astype(np.float32)
      # BF16-exact fixture values also make later mixed BF16/FP16 loader comparisons exact.
      values = (values.view(np.uint32)&np.uint32(0xffff0000)).view(np.float32).astype(np.float16)
      param.replace(Tensor(values,device='CPU').realize())
    return model

  def weights(self,model):
    return {name:value.numpy().astype(np.float32) for name,value in nn.state.get_state_dict(model).items()
            if 'cache' not in name and 'freqs_cis' not in name}

  def norm(self,x,w=None):
    y = x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
    return y if w is None else y*w

  def modalities(self,w,p,x):return np.tanh((self.norm(x,w[p+'altup_router_norm.weight'])/32)@w[p+'altup_router.weight'].T)

  def reference(self,w,ids):
    def linear(x,name):return x@w[name+'.weight'].T
    def gelu(x):return .5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3)))
    def match(x,target):return x*target/np.sqrt(np.maximum(np.mean(x*x,axis=-1,keepdims=True),1e-5))
    def softmax(x):
      ex = np.exp(x-x.max(-1,keepdims=True))
      return ex/ex.sum(-1,keepdims=True)
    length = len(ids)
    x = w['token_embd.weight'][ids]*math.sqrt(32)
    side = self.norm((linear(x,'per_layer_model_proj')/math.sqrt(32)).reshape(length,4,4),w['per_layer_proj_norm.weight'])
    names = sorted((k for k in w if k.startswith('per_layer_token_embd.chunks.')),key=lambda k:int(k.split('.')[2]))
    side_table = np.concatenate([w[name] for name in names],axis=0)
    side_ids = np.where(np.array(ids)<32,ids,0)
    side = (side+side_table[side_ids].reshape(length,4,4)*2)/math.sqrt(2)
    target = np.sqrt(np.mean(x*x,axis=-1,keepdims=True))
    states = np.stack([x,*[match(x@weight.T,target) for weight in w['altup_proj.weight']]])
    cache = {}
    for layer,sparsity in enumerate((.7,-math.inf,1.1,-math.inf)):
      p = f'blk.{layer}.'
      coefficients = linear(self.modalities(w,p,states[0]),p+'altup_predict_coef').reshape(length,4,4)
      predicted = states+np.einsum('jtd,tij->itd',states,coefficients)
      active = predicted[0]
      normalized = self.norm(active,w[p+'attn_norm.weight'])
      laurel = normalized+self.norm(linear(linear(normalized,p+'laurel_l'),p+'laurel_r'),w[p+'laurel_post_norm.weight'])
      sliding = layer%2==0
      phase = np.arange(length,dtype=np.float32)[:,None]*(10000. if sliding else 1000000.)**(-np.arange(0,8,2,dtype=np.float32)/8)
      def rope(x):
        a,b = np.split(x,2,axis=-1)
        return np.concatenate((a*np.cos(phase)-b*np.sin(phase),b*np.cos(phase)+a*np.sin(phase)),axis=-1)
      q = rope(self.norm(linear(normalized,p+'attn_q').reshape(length,4,8).transpose(1,0,2),w[p+'attn_q_norm.weight']))
      if layer<2:
        k = rope(self.norm(linear(normalized,p+'attn_k').reshape(length,1,8).transpose(1,0,2),w[p+'attn_k_norm.weight']))
        v = self.norm(linear(normalized,p+'attn_v').reshape(length,1,8).transpose(1,0,2))
        cache[sliding] = k.astype(np.float16).astype(np.float32),v.astype(np.float16).astype(np.float32)
      k,v = cache[sliding]
      scores = q@np.repeat(k,4,axis=0).transpose(0,2,1)
      allowed = np.arange(length)[None,:]<=np.arange(length)[:,None]
      if sliding:allowed &= np.arange(length)[None,:]>np.arange(length)[:,None]-2
      attention = (softmax(np.where(allowed[None],scores,-np.inf))@np.repeat(v,4,axis=0)).transpose(1,0,2).reshape(length,32)
      residual = (active+self.norm(linear(attention,p+'attn_output'),w[p+'attn_post_norm.weight'])+laurel)/math.sqrt(2)
      normalized = self.norm(residual,w[p+'ffn_norm.weight'])
      gate,up = linear(normalized,p+'ffn_gate'),linear(normalized,p+'ffn_up')
      if math.isfinite(sparsity):gate = np.maximum(gate-gate.mean(-1,keepdims=True)-gate.std(-1,keepdims=True)*sparsity,0)
      activated = residual+self.norm(linear(gelu(gate)*up,p+'ffn_down'),w[p+'ffn_post_norm.weight'])
      coefficients = linear(self.modalities(w,p,activated),p+'altup_correct_coef')+1
      corrected = predicted+(activated-predicted[0])[None]*coefficients.T[:,:,None]
      extra = gelu(linear(corrected[0]*w[p+'altup_correct_scale.weight'],p+'inp_gate'))*side[:,layer]
      corrected[1:] += self.norm(linear(extra,p+'proj'),w[p+'post_norm.weight'])
      states = corrected
    target = np.sqrt(np.mean(states[0]**2,axis=-1,keepdims=True))
    output = np.mean(np.stack([states[0],*[match(states[i+1]@weight.T,target) for i,weight in enumerate(w['altup_unembd_proj.weight'])]]),axis=0)
    logits = linear(self.norm(output[-1],w['output_norm.weight']),'output')
    return np.tanh(logits/3)*3

  def test_altup_predict_and_correct_transpose_against_numpy(self):
    model = self.model()
    block = model.blk[0]
    assert isinstance(block,Gemma3nBlock)
    w = self.weights(model)
    states = np.random.default_rng(3).normal(0,.2,(4,1,3,32)).astype(np.float32)
    coefs = (self.modalities(w,'blk.0.',states[0])@w['blk.0.altup_predict_coef.weight'].T).reshape(1,3,4,4)
    expected = states+np.einsum('jbtd,btij->ibtd',states,coefs)
    actual = block.predict(Tensor(states,device='CPU')).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    activated = states[0]+.13
    coefs = self.modalities(w,'blk.0.',activated)@w['blk.0.altup_correct_coef.weight'].T+1
    expected += (activated-expected[0])[None]*coefs.transpose(2,0,1)[:,:,:,None]
    np.testing.assert_allclose(block.correct(Tensor(actual),Tensor(activated)).numpy(),expected,rtol=1e-5,atol=1e-6)

  def compare(self,device):
    model = self.model()
    weights = self.weights(model)
    if device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for value in nn.state.get_parameters(model):value.to_('CL')
      info = configure(model,pack_weights=False,fuse_projections=True)
      self.assertEqual(info['attention_blocks'],0)
    for ids,start in (([3,5,12,39],0),([9],4)):
      prefix = [3,5,12,39] if start==0 else [3,5,12,39,9]
      np.testing.assert_allclose(model.logits(Tensor([ids],device=device),start).numpy().ravel(),self.reference(weights,prefix),rtol=3e-4,atol=2e-4)
    self.assertEqual(sum(hasattr(block,'cache_kv') for block in model.blk),2)

  def test_cpu_sparse_decoder_cached_tokens_and_out_of_side_table_ids(self):self.compare('CPU')

  def test_chunked_greedy_generation_and_prefix_reuse(self):
    model = self.model()
    weights = self.weights(model)
    prompt,expected = [3,5,12,7],[]
    for _ in range(4):expected.append(int(self.reference(weights,prompt+expected).argmax()))
    self.assertEqual(list(itertools.islice(model.generate(prompt.copy(),chunk_size=2),4)),expected)
    extended = prompt+expected+[39]
    self.assertGreater(model.get_start_pos(extended),0)
    self.assertEqual(next(model.generate(extended.copy(),chunk_size=2)),int(self.reference(weights,extended).argmax()))

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_sparse_decoder_against_reference(self):self.compare('CL')
