"""Independent Gemma4 equations: unit attention scale, shared K/V and side inputs."""
import itertools, math, os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.gemma4 import Gemma4Transformer, SegmentedEmbedding, segment_embedding_entries
from tinygrad.llm.gguf import gguf_shard
from tinygrad.llm.model import Transformer, TransformerConfig


class TestGemma4(unittest.TestCase):
  def model(self):
    config = TransformerConfig(4,32,64,4,1,1e-6,32,16,1000000.,16,16,max_context=16,
                               embedding_scale=math.sqrt(32),final_logit_softcap=3.5,sliding_window=2,swa_rope_theta=10000.)
    model = Gemma4Transformer(config,(64,64,128,128),(True,False,True,False),8,2,4,32,(1.,1.,*([1e30]*6)),7)
    rng = np.random.default_rng(404)
    for name,param in nn.state.get_state_dict(model).items():
      values = rng.normal(1 if 'norm' in name else 0,.08,param.shape).astype(np.float16)
      if 'layer_output_scale' in name:values[:] = .85+int(name.split('.')[1])*.08
      param.replace(Tensor(values,device='CPU').realize())
    return model

  def weights(self, model):
    return {name:value.numpy().astype(np.float32) for name,value in nn.state.get_state_dict(model).items()
            if 'cache' not in name and 'freqs_cis' not in name}

  def reference(self,w,ids, *, token_identity=True):
    def norm(x,name=None):
      result = x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
      return result if name is None else result*w[name]
    def linear(x,name):return x@w[name+'.weight'].T
    def gelu(x):return .5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3)))
    def softmax(x):
      values = np.exp(x-x.max(axis=-1,keepdims=True))
      return values/values.sum(axis=-1,keepdims=True)
    length = len(ids)
    x = w['token_embd.weight'][ids]*math.sqrt(32)
    side = norm((linear(x,'per_layer_model_proj')/math.sqrt(32)).reshape(length,4,4),'per_layer_proj_norm.weight')
    if token_identity:
      names = sorted((key for key in w if key.startswith('per_layer_token_embd.chunks.')),key=lambda key:int(key.split('.')[2]))
      identity = np.concatenate([w[name] for name in names],axis=0)[ids].reshape(length,4,4)*2
      side = (side+identity)/math.sqrt(2)
    shared = {}
    for layer in range(4):
      p = f'blk.{layer}.'
      sliding = layer%2==0
      head = 8 if sliding else 16
      phase = np.arange(length,dtype=np.float32)[:,None]*(10000. if sliding else 1000000.)**(-np.arange(0,head,2,dtype=np.float32)/head)
      # Proportional RoPE keeps the full head's frequencies but rotates only a quarter.
      if not sliding:phase[:,2:]=0
      def rope(values):
        a,b = np.split(values,2,axis=-1)
        return np.concatenate((a*np.cos(phase)-b*np.sin(phase),b*np.cos(phase)+a*np.sin(phase)),axis=-1)
      h = norm(x,p+'attn_norm.weight')
      q = linear(h,p+'attn_q').reshape(length,4,head).transpose(1,0,2)
      q = rope(norm(q,p+'attn_q_norm.weight'))
      if layer<2:
        k = linear(h,p+'attn_k').reshape(length,1,head).transpose(1,0,2)
        v = linear(h,p+'attn_v').reshape(length,1,head).transpose(1,0,2)
        k = rope(norm(k,p+'attn_k_norm.weight')).astype(np.float16).astype(np.float32)
        v = norm(v).astype(np.float16).astype(np.float32)
        shared[sliding] = k,v
      else:k,v = shared[sliding]
      scores = q@np.repeat(k,4,axis=0).transpose(0,2,1)
      allowed = np.arange(length)[None,:]<=np.arange(length)[:,None]
      if sliding:allowed &= np.arange(length)[None,:]>np.arange(length)[:,None]-2
      scores = np.where(allowed[None],scores,-np.inf)
      attention = (softmax(scores)@np.repeat(v,4,axis=0)).transpose(1,0,2).reshape(length,4*head)
      x = x+norm(linear(attention,p+'attn_output'),p+'attn_post_norm.weight')
      h = norm(x,p+'ffn_norm.weight')
      x = x+norm(linear(gelu(linear(h,p+'ffn_gate'))*linear(h,p+'ffn_up'),p+'ffn_down'),p+'ffn_post_norm.weight')
      x = x+norm(linear(gelu(linear(x,p+'inp_gate'))*side[:,layer],p+'proj'),p+'post_norm.weight')
      x = x*w[p+'layer_output_scale.weight']
    logits = linear(norm(x[-1],'output_norm.weight'),'output')
    return np.tanh(logits/3.5)*3.5

  def test_segmented_embedding_boundaries_and_quantized_entry_rows(self):
    embedding = SegmentedEmbedding(23,8,7)
    values = np.arange(23*8,dtype=np.float16).reshape(23,8)/100
    for i,chunk in enumerate(embedding.chunks):chunk.weight.replace(Tensor(values[i*7:(i+1)*7],device='CPU').realize())
    ids = np.array([[0,6,7,13,14,22]],dtype=np.int32)
    np.testing.assert_array_equal(embedding(Tensor(ids,device='CPU')).numpy(),values[ids].astype(np.float32))
    packed = Tensor(np.arange(5*34,dtype=np.uint8),device='CPU')
    entries = segment_embedding_entries({'per_layer_token_embd.weight':(packed,(5,32),8)},'per_layer_token_embd.weight',2)
    self.assertEqual([value[1] for value in entries.values()],[(2,32),(2,32),(1,32)])
    self.assertEqual([value[0].numel() for value in entries.values()],[68,68,34])

  def test_prefill_and_cached_decode_match_independent_shared_attention(self):
    model = self.model()
    weights = self.weights(model)
    for ids,start in (([3,5,12,7],0),([9],4)):
      expected = self.reference(weights,[3,5,12,7] if start==0 else [3,5,12,7,9])
      actual = model.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
      np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-4)
    self.assertEqual(sum(hasattr(block,'cache_kv') for block in model.blk),2)
    self.assertFalse(hasattr(model.blk[2],'attn_k'))

  def test_chunked_generation_and_reused_prefix(self):
    model = self.model()
    weights = self.weights(model)
    prompt = [3,5,12,7]
    actual = list(itertools.islice(model.generate(prompt.copy(),chunk_size=2),4))
    expected = []
    for _ in range(4):expected.append(int(self.reference(weights,prompt+expected).argmax()))
    self.assertEqual(actual,expected)
    extended = prompt+actual+[31]
    self.assertGreater(model.get_start_pos(extended),0)
    self.assertEqual(next(model.generate(extended.copy(),chunk_size=2)),int(self.reference(weights,extended).argmax()))

  def test_embedding_input_uses_context_projection_without_token_identity(self):
    model = self.model()
    weights = self.weights(model)
    ids = [3,5,12,7]
    embeddings = model.token_embd(Tensor([ids],device='CPU')).float()*math.sqrt(32)
    actual = model.logits_from_embeddings(embeddings,0).numpy().ravel()
    np.testing.assert_allclose(actual,self.reference(weights,ids,token_identity=False),rtol=2e-4,atol=1e-4)

  def test_loader_maps_norms_drops_unused_shared_weights_and_keeps_rope_fp32(self):
    original = self.model()
    weights = self.weights(original)
    entries = {}
    for name,values in weights.items():
      if name.startswith('per_layer_token_embd.chunks.'):continue
      name = name.replace('attn_post_norm','post_attention_norm').replace('ffn_post_norm','post_ffw_norm')
      values = values.astype(np.float16)
      entries[name] = (Tensor(values.view(np.uint8).reshape(-1),device='CPU'),values.shape,1)
    side = np.concatenate([weights[f'per_layer_token_embd.chunks.{i}.weight'] for i in range(5)]).astype(np.float16)
    entries['per_layer_token_embd.weight'] = (Tensor(side.view(np.uint8).reshape(-1),device='CPU'),side.shape,1)
    factors = np.array([1.,1.,*([1e30]*6)],dtype=np.float32)
    entries['rope_freqs.weight'] = (Tensor(factors.view(np.uint8),device='CPU'),factors.shape,0)
    for layer,head in ((2,8),(3,16)):
      for name,shape in [('attn_k.weight',(head,32)),('attn_v.weight',(head,32)),('attn_k_norm.weight',(head,))]:
        entries[f'blk.{layer}.'+name] = (Tensor.zeros(math.prod(shape)*2,dtype='uint8',device='CPU'),shape,1)
    values = {'block_count':4,'embedding_length':32,'feed_forward_length':[64,64,128,128],'attention.head_count':4,
              'attention.head_count_kv':1,'attention.key_length':16,'attention.key_length_swa':8,
              'attention.layer_norm_rms_epsilon':1e-6,'attention.sliding_window':2,'attention.sliding_window_pattern':[True,False,True,False],
              'attention.shared_kv_layers':2,'embedding_length_per_layer_input':4,'context_length':16,
              'rope.freq_base':1000000.,'rope.freq_base_swa':10000.,'final_logit_softcapping':3.5}
    kv = {'general.architecture':'gemma4','tokenizer.ggml.tokens':[str(i) for i in range(32)],**{'gemma4.'+k:v for k,v in values.items()}}
    with patch('tinygrad.llm.gemma4.gguf_parse',return_value=(kv,dict(entries))), \
         patch('tinygrad.llm.model.gguf_parse',return_value=(kv,dict(entries))), \
         patch('tinygrad.llm.gemma4.gguf_shard',wraps=gguf_shard) as load:
      model,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    self.assertNotIn('blk.2.attn_k.weight',load.call_args.args[0])
    self.assertNotIn('rope_freqs.weight',load.call_args.kwargs['cast_dtypes'])
    actual = model.logits(Tensor([[3,5,12,7]],device='CPU'),0).numpy().ravel()
    np.testing.assert_allclose(actual,self.reference(weights,[3,5,12,7]),rtol=2e-4,atol=1e-4)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_decoder_against_independent_reference(self):
    from tinygrad.llm.kernels.mali import configure
    model = self.model()
    weights = self.weights(model)
    for value in nn.state.get_parameters(model):value.to_('CL')
    info = configure(model,pack_weights=False,fuse_projections=True)
    self.assertEqual(info['attention_blocks'],0)
    for ids,start in (([3,5,12,7],0),([9],4)):
      expected = self.reference(weights,[3,5,12,7] if start==0 else [3,5,12,7,9])
      np.testing.assert_allclose(model.logits(Tensor([ids],device='CL'),start).numpy().ravel(),expected,rtol=3e-4,atol=2e-4)
