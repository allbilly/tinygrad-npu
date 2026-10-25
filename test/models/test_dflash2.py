"""Independent small-array equations for the trained DFlash2 draft architecture."""
import unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2 import DFlash2, grouped_dynamic_conv
from tinygrad.llm.model import TargetOutput


def config():
  return SimpleNamespace(hidden_size=8,intermediate_size=12,num_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,
                         norm_eps=1e-6,vocab_size=16,rope_theta=10000.,layer_types=('sliding_attention','full_attention'),
                         sliding_window=2,sliding_window_non_causal=True,aux_hidden_state_layer_ids=(1,3),capture_layers=(0,2),
                         block_size=4,mask_token_id=15,conv_kernel_size=2,conv_group_size=2,sample_from_anchor=False,
                         max_position_embeddings=512)


def fixture(seed=463):
  c, rng = config(),np.random.default_rng(seed)
  shapes = {'fc.weight':(8,16),'hidden_norm.weight':(8,),'norm.weight':(8,)}
  for i in range(c.num_layers):
    p = f'layers.{i}'
    shapes.update({p+'.input_layernorm.weight':(8,),p+'.post_attention_layernorm.weight':(8,),
                   p+'.mlp.gate_proj.weight':(12,8),p+'.mlp.up_proj.weight':(12,8),p+'.mlp.down_proj.weight':(8,12),
                   p+'.self_attn.q_proj.weight':(8,8),p+'.self_attn.k_proj.weight':(4,8),
                   p+'.self_attn.v_proj.weight':(4,8),p+'.self_attn.o_proj.weight':(8,8),
                   p+'.self_attn.q_norm.weight':(4,),p+'.self_attn.k_norm.weight':(4,)})
    for conv in ('attention_conv','mlp_conv'):
      shapes[p+'.'+conv+'.base_kernel'] = (2,2,8)
      shapes[p+'.'+conv+'.kernel_projection.weight'] = (16,8)
  weights = {name:rng.normal(0,.12,shape).astype(np.float32) for name,shape in shapes.items()}
  for name,x in weights.items():
    if name.endswith('norm.weight'):x+=1
    if name.endswith('base_kernel'):x[:,0]+=1
  embedding, head = rng.normal(0,.3,(16,8)).astype(np.float32),rng.normal(0,.2,(16,8)).astype(np.float32)
  emb, out = Tensor(embedding,device='CPU').realize(),Tensor(head,device='CPU').realize()
  model = DFlash2(c,{name:Tensor(x,device='CPU').realize() for name,x in weights.items()},
                 lambda ids:emb[ids],lambda hidden:hidden@out.T,lambda _hidden:(_ for _ in ()).throw(AssertionError('verifier norm applied')))
  return model,c,weights,embedding,head,rng


def norm(x, weight, eps):
  return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*weight


def convolution(x,delta,base,block_size,group_size):
  result = np.zeros_like(x)
  for batch in range(x.shape[0]):
    for row in range(x.shape[1]):
      for channel in range(x.shape[2]):
        for tap in range(min(row%block_size+1,base.shape[0])):
          result[batch,row,channel] += x[batch,row-tap,channel]*(base[tap,channel]+delta[batch,row,tap,channel//group_size])
  return result


def reference(noise,features,position,target_start_pos,c,w,embedding,head):
  def linear(x,name):return x@w[name+'.weight'].T
  def rms(x,name):return norm(x,w[name+'.weight'],c.norm_eps)
  def prepare(x,name):
    kernels = linear(x,name+'.kernel_projection').reshape(*x.shape[:-1],2,2,4)
    return convolution(x,kernels[...,0,:,:],w[name+'.base_kernel'][0],c.block_size,c.conv_group_size),kernels[...,1,:,:]
  def finish(x,kernel,name):return convolution(x,kernel,w[name+'.base_kernel'][1],c.block_size,c.conv_group_size)
  def rope(x,positions):
    half = x.shape[-1]//2
    phase = positions.reshape(1,1,-1,1)/(c.rope_theta**(np.arange(half,dtype=np.float32)/half))
    a,b = np.split(x,2,axis=-1)
    return np.concatenate((a*np.cos(phase)-b*np.sin(phase),b*np.cos(phase)+a*np.sin(phase)),axis=-1)
  prefix = features.shape[1]
  target = rms(linear(features,'fc'),'hidden_norm')
  hidden = embedding[noise]
  for layer in range(c.num_layers):
    p = f'layers.{layer}'
    residual = hidden
    x,kernel = prepare(rms(hidden,p+'.input_layernorm'),p+'.attention_conv')
    a = p+'.self_attn'
    q = rms(linear(x,a+'.q_proj').reshape(1,c.block_size,2,4),a+'.q_norm').transpose(0,2,1,3)
    context = np.concatenate((target,x),axis=1)
    k = rms(linear(context,a+'.k_proj').reshape(1,prefix+c.block_size,1,4),a+'.k_norm').transpose(0,2,1,3)
    v = linear(context,a+'.v_proj').reshape(1,prefix+c.block_size,1,4).transpose(0,2,1,3)
    q = rope(q,np.arange(position,position+c.block_size,dtype=np.float32))
    k = rope(k,np.arange(target_start_pos,position+c.block_size,dtype=np.float32))
    k,v = np.repeat(k,2,axis=1),np.repeat(v,2,axis=1)
    scores = (q@k.transpose(0,1,3,2))/np.float32(2)
    if c.layer_types[layer]=='sliding_attention':
      for slot in range(c.block_size):
        for key in range(prefix+c.block_size):
          if key<prefix and key+target_start_pos<position-c.sliding_window:scores[:,:,slot,key]=-np.inf
          if key>=prefix and not c.sliding_window_non_causal and key>prefix+slot:scores[:,:,slot,key]=-np.inf
    probabilities = np.exp(scores-scores.max(axis=-1,keepdims=True))
    probabilities /= probabilities.sum(axis=-1,keepdims=True)
    attended = (probabilities@v).transpose(0,2,1,3).reshape(1,c.block_size,8)
    hidden = residual+finish(linear(attended,a+'.o_proj'),kernel,p+'.attention_conv')
    residual = hidden
    x,kernel = prepare(rms(hidden,p+'.post_attention_layernorm'),p+'.mlp_conv')
    gate,up = linear(x,p+'.mlp.gate_proj'),linear(x,p+'.mlp.up_proj')
    mlp = linear((gate/(1+np.exp(-gate)))*up,p+'.mlp.down_proj')
    hidden = residual+finish(mlp,kernel,p+'.mlp_conv')
  hidden = rms(hidden,'norm')
  return hidden@head.T,hidden


class TestDFlash2(unittest.TestCase):
  def test_grouped_convolution_multiple_blocks_and_batches(self):
    rng = np.random.default_rng(462)
    x = rng.normal(size=(2,12,8)).astype(np.float32)
    delta = rng.normal(0,.2,(2,12,3,4)).astype(np.float32)
    base = rng.normal(0,.2,(3,8)).astype(np.float32)
    actual = grouped_dynamic_conv(Tensor(x,device='CPU'),Tensor(delta,device='CPU'),Tensor(base,device='CPU'),block_size=4,group_size=2)
    np.testing.assert_allclose(actual.numpy(),convolution(x,delta,base,4,2),atol=1e-6,rtol=1e-6)
    for bad in (x[:,:11],x[:,:,:7]):
      with self.assertRaises(ValueError):grouped_dynamic_conv(Tensor(bad,device='CPU'),Tensor(delta,device='CPU'),
                                                             Tensor(base,device='CPU'),block_size=4,group_size=2)

  def test_full_forward_matches_independent_equations(self):
    model,c,w,embedding,head,rng = fixture()
    noise = np.array([[3,15,15,15]],np.int32)
    for prefix,start in ((0,0),(5,0),(3,11)):
      features = rng.normal(0,.4,(1,prefix,16)).astype(np.float32)
      expected = reference(noise,features,start+prefix,start,c,w,embedding,head)
      actual = model(Tensor(noise,device='CPU'),Tensor(features,device='CPU'),position=start+prefix,target_start_pos=start).realize()
      np.testing.assert_allclose(actual.logits.numpy(),expected[0],atol=2e-6,rtol=3e-5)
      np.testing.assert_allclose(actual.hidden.numpy(),expected[1],atol=2e-6,rtol=3e-5)
      np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected[0].argmax(-1))

  def test_causal_noise_option_matches_its_mask_equations(self):
    model,c,w,embedding,head,rng = fixture(464)
    c.sliding_window_non_causal=False
    noise = np.array([[3,15,15,15]],np.int32)
    features = rng.normal(0,.4,(1,5,16)).astype(np.float32)
    expected = reference(noise,features,5,0,c,w,embedding,head)
    actual = model(Tensor(noise,device='CPU'),Tensor(features,device='CPU'),position=5).realize()
    np.testing.assert_allclose(actual.logits.numpy(),expected[0],atol=2e-6,rtol=3e-5)

  def test_committed_feature_cache_replacement_and_full_block_proposal(self):
    model,c,w,_embedding,_head,rng = fixture(465)
    def output(rows):
      features = tuple(Tensor(rng.normal(0,.4,(1,rows,8)).astype(np.float32),device='CPU') for _ in range(2))
      return TargetOutput(Tensor.zeros(1,rows,16,device='CPU'),features[-1],features)
    first,second = output(5),output(2)
    model.observe(first,0)
    model.observe(second,3)
    expected = model.project_features(tuple(a[:,:3].cat(b,dim=1) for a,b in zip(first.features,second.features))).numpy()
    np.testing.assert_array_equal(model._target_hidden.numpy(),expected)
    calls = []
    class Selector:
      def choose(self,logits,hidden,anchor,max_tokens):
        calls.append((logits.shape,hidden.shape,anchor,max_tokens))
        return [6]*max_tokens
    model.candidate_selector=Selector()
    self.assertEqual(model.propose(3,position=5,max_tokens=2),(6,))
    self.assertEqual(calls,[((1,4,16),(1,4,8),3,1)])
    self.assertEqual(model.propose(3,position=5,max_tokens=1),())
    self.assertEqual(len(calls),1)
    with self.assertRaises(ValueError):model.propose(3,position=4,max_tokens=3)
    with self.assertRaises(ValueError):model.observe(output(1),6)
    np.testing.assert_array_equal(model._target_hidden.numpy(),expected)
    model.reset()
    self.assertIsNone(model._target_hidden)

  def test_shape_and_position_validation(self):
    model,c,_w,_embedding,_head,_rng = fixture(466)
    noise = Tensor([[3,15,15,15]],device='CPU',dtype=dtypes.int32)
    features = Tensor.zeros(1,3,16,device='CPU')
    with self.assertRaises(ValueError):model(noise,features,position=4)
    with self.assertRaises(ValueError):model(noise,features,position=-1)
    with self.assertRaises(ValueError):model(noise[:,:3],features,position=3)
    with self.assertRaises(ValueError):model(noise.cast(dtypes.float32),features,position=3)
    with self.assertRaises(ValueError):model.project_features((features[:,:,:8],))
    with self.assertRaises(ValueError):model.propose(3,position=0,max_tokens=2)
    self.assertIsNone(model._target_hidden)


if __name__=='__main__':unittest.main()
