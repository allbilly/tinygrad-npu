import tempfile, unittest
from pathlib import Path
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from extra.rockchip.convert_telechat2 import Mapping, mappings, metadata, source_rows, write_f16


class TestTelechatConversion(unittest.TestCase):
  def config(self):
    return {'model_type':'telechat','apply_residual_connection_post_layernorm':False,
            'hidden_size':32,'ffn_hidden_size':64,'vocab_size':64,'n_layer':1,'n_head':4,'num_key_value_heads':2,
            'layer_norm_epsilon':1e-5,'training_seqlen':8,'base_seqlen':8,'seq_length':32}

  def source(self):
    rng = np.random.default_rng(302)
    shapes = {'transformer.word_embeddings.weight':(64,32),'lm_head.weight':(64,32),'transformer.ln_f.weight':(32,)}
    prefix = 'transformer.h.0.'
    for name,shape in [('input_layernorm.weight',(32,)),('post_attention_layernorm.weight',(32,)),
                       ('self_attention.query.weight',(32,32)),('self_attention.key_value.weight',(32,32)),
                       ('self_attention.dense.weight',(32,32)),('self_attention.dense.bias',(32,)),
                       ('mlp.gate_proj.weight',(64,32)),('mlp.up_proj.weight',(64,32)),
                       ('mlp.down_proj.weight',(32,64)),('mlp.down_proj.bias',(32,))]:shapes[prefix+name] = shape
    values = {name:rng.normal(0,.1,shape).astype(np.float16) for name,shape in shapes.items()}
    for name in values:
      if 'layernorm' in name or 'ln_f' in name:values[name]+=1
    values[prefix+'mlp.down_proj.bias'] = np.linspace(.03,.09,32,dtype=np.float16)
    return values

  def test_streamed_gguf_kv_split_and_biased_ffn_against_numpy(self):
    config,data = self.config(),self.source()
    plan = mappings(config,set(data))
    kv = metadata(config,{'tokenizer.ggml.tokens':[str(i) for i in range(64)]},{'repo':'fixture/model','revision':'pinned','size':'toy'})
    def chunks(item):
      for start in range(0,item.shape[0],3):
        rows = source_rows(item,start,min(3,item.shape[0]-start))
        yield data[item.source][rows].astype('<f2').tobytes()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'model.gguf'
      write_f16(path,kv,plan,chunks)
      parsed,entries = gguf_parse(path)
      self.assertEqual(parsed['general.architecture'],'telechat2')
      self.assertEqual(parsed['telechat2.context_length'],8)
      self.assertEqual(parsed['general.source.context_length'],32)
      self.assertTrue(all(entry[2]==1 for entry in entries.values()))
      model,_ = Transformer.from_gguf(path,max_context=8,realize=True)
      block = model.blk[0]
      self.assertTrue(model.config.ffn_down_bias)
      fused = data['transformer.h.0.self_attention.key_value.weight'].reshape(2,2,8,32)
      np.testing.assert_array_equal(block.attn_k.weight.numpy(),fused[:,0].reshape(16,32))
      np.testing.assert_array_equal(block.attn_v.weight.numpy(),fused[:,1].reshape(16,32))
      x = np.random.default_rng(3).normal(0,.2,(1,3,32)).astype(np.float32)
      p = 'transformer.h.0.mlp.'
      gate = x@data[p+'gate_proj.weight'].astype(np.float32).T
      up = x@data[p+'up_proj.weight'].astype(np.float32).T
      expected = (gate/(1+np.exp(-gate))*up)@data[p+'down_proj.weight'].astype(np.float32).T+data[p+'down_proj.bias']
      np.testing.assert_allclose(block._feed_forward(Tensor(x,device='CPU')).numpy(),expected,rtol=1e-5,atol=1e-6)

  def test_writer_refuses_short_payload_and_preserves_existing_file(self):
    spec = Mapping('weight','source',(2,3),(2,3))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'model.gguf'
      with self.assertRaisesRegex(ValueError,'Incomplete'):write_f16(path,{},[spec],lambda item:[b'\0'*4])
      self.assertFalse(path.exists())
      path.write_bytes(b'user data')
      with self.assertRaises(FileExistsError):write_f16(path,{},[spec],lambda item:[b'\0'*12])
      self.assertEqual(path.read_bytes(),b'user data')

  def test_mapper_rejects_unhandled_parameters_and_post_norm_residuals(self):
    config,data = self.config(),self.source()
    with self.assertRaises(ValueError):mappings(config,set(data)|{'unhandled.weight'})
    with self.assertRaises(ValueError):mappings({**config,'apply_residual_connection_post_layernorm':True},set(data))
    with self.assertRaises(ValueError):metadata(config,{'tokenizer.ggml.tokens':[]},{'repo':'fixture','revision':'pinned','size':'toy'})
