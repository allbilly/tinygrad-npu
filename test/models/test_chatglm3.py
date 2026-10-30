"""Independent adjacent-pair rotary attention from the pinned ChatGLM3 decoder."""
import math, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.chatglm import unpack_chatglm
from tinygrad.llm.model import Transformer, TransformerConfig


class TestChatGLM3(unittest.TestCase):
  def fixture(self):
    rng = np.random.default_rng(363)
    state = {'token_embd.weight':rng.normal(0,.1,(64,32)),'output.weight':rng.normal(0,.1,(64,32)),
             'output_norm.weight':rng.normal(1,.1,32)}
    for layer in range(2):
      prefix = f'blk.{layer}.'
      shapes = {'attn_norm.weight':(32,),'ffn_norm.weight':(32,),'attn_qkv.weight':(64,32),'attn_qkv.bias':(64,),
                'attn_output.weight':(32,32),'ffn_up.weight':(128,32),'ffn_down.weight':(32,64)}
      for name,shape in shapes.items():state[prefix+name] = rng.normal(1 if 'norm' in name else 0,.1,shape)
    return {name:Tensor(value.astype(np.float16),device='CPU').realize() for name,value in state.items()}

  def model(self,state):
    config = TransformerConfig(2,32,64,4,2,1e-5,64,8,10000.,4,8,max_context=16,qkv_bias=True)
    model = Transformer(config)
    mapped = unpack_chatglm(dict(state),2,4,2,8,4,64)
    nn.state.load_state_dict(model,mapped,verbose=False,realize=True)
    return model,mapped

  def reference(self,state,tokens):
    w = {name:value.numpy().astype(np.float32) for name,value in state.items()}
    def norm(x,name):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*w[name]
    def softmax(x):
      ex = np.exp(x-x.max(-1,keepdims=True))
      return ex/ex.sum(-1,keepdims=True)
    length = len(tokens)
    phase = np.arange(length,dtype=np.float32)[:,None,None]*10000.**(-np.arange(0,4,2,dtype=np.float32)/4)
    def rotate(x):
      pairs = x[:,:,:4].reshape(length,x.shape[1],2,2)
      first = pairs[:,:,:,0]*np.cos(phase)-pairs[:,:,:,1]*np.sin(phase)
      second = pairs[:,:,:,1]*np.cos(phase)+pairs[:,:,:,0]*np.sin(phase)
      return np.concatenate((np.stack((first,second),axis=-1).reshape(length,x.shape[1],4),x[:,:,4:]),axis=-1)
    x = w['token_embd.weight'][tokens]
    for layer in range(2):
      p = f'blk.{layer}.'
      h = norm(x,p+'attn_norm.weight')
      packed = h@w[p+'attn_qkv.weight'].T+w[p+'attn_qkv.bias']
      q,k,v = np.split(packed,[32,48],axis=-1)
      q = rotate(q.reshape(length,4,8)).transpose(1,0,2)
      k = rotate(k.reshape(length,2,8)).transpose(1,0,2).astype(np.float16).astype(np.float32)
      v = v.reshape(length,2,8).transpose(1,0,2).astype(np.float16).astype(np.float32)
      k,v = np.repeat(k,2,axis=0),np.repeat(v,2,axis=0)
      scores = q@k.transpose(0,2,1)/math.sqrt(8)
      row,col = np.triu_indices(length,1)
      scores[:,row,col] = -np.inf
      attention = (softmax(scores)@v).transpose(1,0,2).reshape(length,32)
      x = x+attention@w[p+'attn_output.weight'].T
      h = norm(x,p+'ffn_norm.weight')
      gate,up = np.split(h@w[p+'ffn_up.weight'].T,2,axis=-1)
      x = x+(gate/(1+np.exp(-gate))*up)@w[p+'ffn_down.weight'].T
    return norm(x[-1],'output_norm.weight')@w['output.weight'].T

  def compare(self,device):
    state = self.fixture()
    model,_ = self.model(state)
    if device=='CL':
      from tinygrad.llm.kernels.mali import configure
      for param in nn.state.get_parameters(model):param.to_('CL')
      configure(model,pack_weights=False)
    for ids,start in (([3,7,2],0),([9],3)):
      expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
      actual = model.logits(Tensor([ids],device=device),start).numpy().ravel()
      np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)

  def test_cpu_prefill_and_cached_decode_against_original_layout(self):self.compare('CPU')

  def test_gguf_loader_detects_bias_and_applies_packed_projection_adapter(self):
    from tinygrad.llm.gguf import gguf_shard
    state = self.fixture()
    values = {'context_length':16,'block_count':2,'embedding_length':32,'feed_forward_length':64,
              'attention.head_count':4,'attention.head_count_kv':2,'attention.layer_norm_rms_epsilon':1e-5,
              'rope.dimension_count':4}
    kv = {'general.architecture':'chatglm','tokenizer.ggml.tokens':[str(i) for i in range(64)],
          **{'chatglm.'+key:value for key,value in values.items()}}
    entries = {name:(value.bitcast(dtypes.uint8).flatten(),value.shape,1) for name,value in state.items()}
    def load(entries,*args,**kwargs):
      for index in range(2):
        key = f'blk.{index}.ffn_gate.weight'
        self.assertEqual(entries[key][1],(64,32))
        self.assertIs(kwargs['cast_dtypes'][key],dtypes.half)
      return gguf_shard(entries,*args,**kwargs)
    with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,entries)),patch('tinygrad.llm.model.gguf_shard',new=load):
      model,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    self.assertTrue(model.config.qkv_bias)
    for ids,start in (([3,7,2],0),([9],3)):
      expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
      np.testing.assert_allclose(model.logits(Tensor([ids],device='CPU'),start).numpy().ravel(),expected,rtol=2e-4,atol=2e-5)

  def test_mapped_loader_preserves_permuted_weights_and_cached_decode(self):
    state = self.fixture()
    values = {'context_length':16,'block_count':2,'embedding_length':32,'feed_forward_length':64,
              'attention.head_count':4,'attention.head_count_kv':2,'attention.layer_norm_rms_epsilon':1e-5,
              'rope.dimension_count':4}
    kv = {'general.architecture':'chatglm','tokenizer.ggml.tokens':[str(i) for i in range(64)],
          **{'chatglm.'+key:value for key,value in values.items()}}
    data = b'prefix!'
    offsets = {}
    for name,value in state.items():
      raw = value.numpy().tobytes()
      offsets[name] = (len(data),len(raw),value.shape)
      data += raw
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'weights.bin'
      path.write_bytes(data)
      source = Tensor(path)
      entries = {name:(source[start:start+size],shape,1) for name,(start,size,shape) in offsets.items()}
      with patch('tinygrad.llm.model.gguf_parse',return_value=(kv,entries)):
        model,_ = Transformer.from_gguf('fixture',max_context=16,realize=False,mapped_cpu=True)
      self.assertTrue(model.gguf_loading['mapped_cpu'])
      for ids,start in (([3,7,2],0),([9],3)):
        expected = self.reference(state,[3,7,2] if start==0 else [3,7,2,9])
        actual = model.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
      self.assertEqual(path.read_bytes(),data)

  def test_file_entry_split_preserves_q8_rows_and_existing_separate_projections(self):
    from tinygrad.llm.chatglm import split_chatglm_ffn_entries
    from tinygrad.llm.gguf import gguf_shard
    scales = np.array([.125,.25,.5,1.,2.,4.],dtype=np.float16)
    values = np.arange(-96,96,dtype=np.int8).reshape(6,32)
    raw = b''.join(scale.tobytes()+row.tobytes() for scale,row in zip(scales,values))
    entries = {'blk.0.ffn_up.weight':(Tensor(list(raw),device='CPU',dtype=dtypes.uint8),(6,32),8)}
    split = split_chatglm_ffn_entries(entries,3)
    self.assertEqual(entries['blk.0.ffn_up.weight'][1],(6,32))
    self.assertEqual(split['blk.0.ffn_gate.weight'][1],(3,32))
    decoded = gguf_shard(split,('CPU',),stream=True,cast_dtypes=dict.fromkeys(split,dtypes.half),chunk_size=64)
    expected = (scales.astype(np.float32)[:,None]*values).astype(np.float16)
    np.testing.assert_array_equal(decoded['blk.0.ffn_gate.weight'].numpy(),expected[:3])
    np.testing.assert_array_equal(decoded['blk.0.ffn_up.weight'].numpy(),expected[3:])
    unchanged = split_chatglm_ffn_entries(split,3)
    self.assertIs(unchanged['blk.0.ffn_gate.weight'][0],split['blk.0.ffn_gate.weight'][0])
    self.assertIs(unchanged['blk.0.ffn_up.weight'][0],split['blk.0.ffn_up.weight'][0])
    with self.assertRaises(ValueError):split_chatglm_ffn_entries(entries,4)

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_streamed_loader_bounds_fused_projection_conversion_memory(self):
    import gc
    from tinygrad import Device
    from tinygrad.device import Buffer
    from tinygrad.helpers import Context, GlobalCounters
    rng = np.random.default_rng(364)
    state = {'token_embd.weight':rng.normal(0,.1,(64,64)).astype(np.float16),
             'output_norm.weight':np.ones(64,dtype=np.float16)}
    for layer in range(8):
      shapes = {'attn_norm.weight':(64,),'ffn_norm.weight':(64,),'attn_qkv.weight':(128,64),'attn_qkv.bias':(128,),
                'attn_output.weight':(64,64),'ffn_up.weight':(1024,64),'ffn_down.weight':(64,512)}
      for name,shape in shapes.items():state[f'blk.{layer}.'+name] = rng.normal(0,.1,shape).astype(np.float16)
    values = {'context_length':16,'block_count':8,'embedding_length':64,'feed_forward_length':512,
              'attention.head_count':4,'attention.head_count_kv':2,'attention.layer_norm_rms_epsilon':1e-5,'rope.dimension_count':8}
    kv = {'general.architecture':'chatglm','tokenizer.ggml.tokens':[str(i) for i in range(64)],
          **{'chatglm.'+key:value for key,value in values.items()}}
    allocator = Device['CL'].allocator
    # Match the streamed loader's completed uploads, without a Mock retaining fused tensors.
    def shard(*args,**kwargs):
      decoded = {}
      for name,value in state.items():
        decoded[name] = Tensor(value,device='CL').realize()
        Device['CL'].synchronize()
        allocator.free_cache()
      return decoded
    allocate,peaks = Buffer.allocate,[]
    gc.collect()
    Device['CL'].synchronize()
    allocator.free_cache()
    baseline = GlobalCounters.mem_used_per_device['CL']
    def retained():
      return GlobalCounters.mem_used_per_device['CL']-baseline+sum(size*len(buffers) for (size,_),buffers in allocator.cache.items())
    def observe(buffer,*args,**kwargs):
      result = allocate(buffer,*args,**kwargs)
      peaks.append(retained())
      return result
    with Context(DEV='CL'),patch('tinygrad.llm.model.gguf_parse',return_value=(kv,{})), \
         patch('tinygrad.llm.model.gguf_shard',new=shard),patch.object(Buffer,'allocate',new=observe):
      model,_ = Transformer.from_gguf('fixture',max_context=16,realize=True)
    peaks.append(retained())
    resident = sum(value.nbytes for value in state.values())
    self.assertLessEqual(max(peaks),resident+2*state['blk.0.ffn_up.weight'].nbytes)
    for index,block in enumerate(model.blk):
      gate,up = np.split(state[f'blk.{index}.ffn_up.weight'],2)
      np.testing.assert_array_equal(block.ffn_gate.weight.numpy(),gate)
      np.testing.assert_array_equal(block.ffn_up.weight.numpy(),up)

  def test_qk_bias_prefix_reordered_and_values_and_tail_preserved(self):
    state = self.fixture()
    _,mapped = self.model(state)
    for layer in range(2):
      p = f'blk.{layer}.'
      bias = state[p+'attn_qkv.bias'].numpy()
      for name,original in (('q',bias[:32].reshape(4,8)),('k',bias[32:48].reshape(2,8))):
        actual = mapped[p+'attn_'+name+'.bias'].numpy().reshape(original.shape)
        np.testing.assert_array_equal(actual[:,:4],original[:,[0,2,1,3]])
        np.testing.assert_array_equal(actual[:,4:],original[:,4:])
      np.testing.assert_array_equal(mapped[p+'attn_v.bias'].numpy(),bias[48:])
      value = mapped[p+'attn_v.weight']
      self.assertEqual(value.uop.buffer.base.nbytes,value.numel()*value.dtype.itemsize)

  def test_bad_dimensions_and_packed_width_are_rejected(self):
    state = self.fixture()
    with self.assertRaises(ValueError):unpack_chatglm(dict(state),2,4,2,8,3,64)
    with self.assertRaises(ValueError):unpack_chatglm(dict(state),2,4,2,8,12,64)
    with self.assertRaises(ValueError):unpack_chatglm(dict(state),2,4,3,8,4,64)
    state['blk.0.attn_qkv.weight'] = Tensor.zeros(63,32,device='CPU')
    with self.assertRaises(ValueError):unpack_chatglm(state,2,4,2,8,4,64)

  def test_non_grouped_projection_uses_per_head_qkv_interleaving(self):
    bias = np.arange(48,dtype=np.float16)
    state = {'blk.0.attn_qkv.bias':Tensor(bias,device='CPU')}
    result = unpack_chatglm(state,1,2,2,8,4,64)
    for part,name in enumerate(('q','k','v')):
      expected = bias.reshape(2,3,8)[:,part].copy()
      if name!='v':expected[:,:4] = expected[:,[0,2,1,3]]
      np.testing.assert_array_equal(result['blk.0.attn_'+name+'.bias'].numpy(),expected.reshape(16))

  @unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
  def test_mali_prefill_and_cached_decode_against_original_layout(self):self.compare('CL')
