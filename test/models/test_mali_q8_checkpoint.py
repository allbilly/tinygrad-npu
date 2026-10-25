import struct, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tinygrad import nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.mali_q8_checkpoint import Q8Checkpoint, projection_plan
from tinygrad.llm.model import Transformer, TransformerConfig


def _string(value:str) -> bytes:
  raw = value.encode()
  return struct.pack('<Q',len(raw))+raw


def _gguf(entries, architecture='qwen3', *, truncate=0):
  header = b'GGUF'+struct.pack('<iqq',3,len(entries),1)+_string('general.architecture')+struct.pack('<i',8)+_string(architecture)
  records,payload = b'',b''
  for name,shape,kind,data,offset in entries:
    if offset is None:offset = (len(payload)+31)//32*32
    records += _string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*reversed(shape))+struct.pack('<iQ',kind,offset)
    if len(payload)<offset+len(data):payload += b'\0'*(offset+len(data)-len(payload))
    payload = payload[:offset]+data+payload[offset+len(data):]
  start = (len(header)+len(records)+31)//32*32
  result = header+records+b'\0'*(start-len(header)-len(records))+payload
  return result[:-truncate] if truncate else result


class TestMaliQ8Checkpoint(unittest.TestCase):
  def checkpoint(self, entries, **kwargs):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    path = Path(directory.name)/'weights.gguf'
    path.write_bytes(_gguf(entries,**kwargs))
    return Q8Checkpoint(path)

  def model(self):
    model = Transformer(TransformerConfig(1,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=32))
    for parameter in nn.state.get_parameters(model):parameter.replace(parameter.half())
    return model

  def test_raw_payload_decodes_to_identical_fp16_operands(self):
    rng = np.random.default_rng(123)
    scale = rng.uniform(1e-7,.02,(37,2)).astype(np.float16)
    coefficients = rng.integers(-128,128,(37,2,32),dtype=np.int8)
    raw = np.empty((37,2,34),np.uint8)
    raw[:,:,:2],raw[:,:,2:] = scale.view(np.uint8).reshape(37,2,2),coefficients.view(np.uint8)
    checkpoint = self.checkpoint([('blk.0.ffn_gate.weight',(37,64),8,raw.tobytes(),None)])
    tensor = checkpoint.weight('blk.0.ffn_gate.weight',(37,64),'CPU')
    np.testing.assert_array_equal(tensor.numpy(),raw.flatten())
    expected = (scale.astype(np.float32)[:,:,None]*coefficients.astype(np.float32)).astype(np.float16).reshape(37,64)
    actual = ggml_data_to_tensor(tensor,37*64,8).half().reshape(37,64).numpy()
    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))

  def test_invalid_type_shape_and_truncated_payload(self):
    for kind,data in ((0,b'\0'*(32*32*4)),(8,b'\0'*(32*34))):
      checkpoint = self.checkpoint([('weight',(32,32),kind,data,None)])
      with self.assertRaisesRegex(ValueError,'matching Q8_0'):checkpoint.weight('weight',(32,64),'CPU')
      if kind!=8:
        with self.assertRaisesRegex(ValueError,'matching Q8_0'):checkpoint.weight('weight',(32,32),'CPU')
    with self.assertRaisesRegex(ValueError,'Truncated'):
      self.checkpoint([('weight',(32,32),8,b'\0'*(32*34),None)],truncate=1)

  def test_rejects_duplicates_overlap_unaligned_and_partial_blocks(self):
    entry = ('a',(32,32),8,b'\0'*(32*34),0)
    for second,reason in ((('a',*entry[1:]),'Duplicate'),(('b',*entry[1:]),'Overlapping'),
                          (('b',(32,32),8,entry[3],1),'alignment')):
      with self.assertRaisesRegex(ValueError,reason):self.checkpoint([entry,second])
    with self.assertRaisesRegex(ValueError,'block'):
      self.checkpoint([('a',(32,33),8,b'\0'*(33*34),None)])

  def test_plan_preserves_model_and_skips_non_q8_weights(self):
    model = self.model()
    q8 = ('blk.0.attn_q.weight',(32,32),8,b'\0'*(32*34),None)
    native = ('blk.0.attn_k.weight',(16,32),0,b'\0'*(16*32*4),None)
    original = model.blk[0].attn_q
    plan,skipped = projection_plan(model,self.checkpoint([q8,native]))
    self.assertEqual(plan,[(model.blk[0],'attn_q','blk.0.attn_q.weight')])
    self.assertIs(model.blk[0].attn_q,original)
    self.assertIn('blk.0.attn_k.weight',[item['tensor'] for item in skipped])
    bad = ('blk.0.ffn_gate.weight',(32,32),8,b'\0'*(32*34),None)
    with self.assertRaisesRegex(ValueError,'matching Q8_0'):projection_plan(model,self.checkpoint([q8,bad]))
    self.assertIs(model.blk[0].attn_q,original)

  def test_rejects_transformed_architecture_and_existing_fused_projections(self):
    model = self.model()
    entry = ('blk.0.attn_q.weight',(32,32),8,b'\0'*(32*34),None)
    with self.assertRaisesRegex(ValueError,'architecture'):projection_plan(model,self.checkpoint([entry],architecture='llama'))
    model.blk[0].prepared_qkv = SimpleNamespace()
    with self.assertRaisesRegex(ValueError,'before installing'):projection_plan(model,self.checkpoint([entry]))

  def test_rejects_previously_executed_model(self):
    model = self.model()
    checkpoint = self.checkpoint([('blk.0.attn_q.weight',(32,32),8,b'\0'*(32*34),None)])
    model.prefill_jit.cnt = 1
    with self.assertRaisesRegex(ValueError,'before running'):projection_plan(model,checkpoint)
    model.prefill_jit.cnt = 0
    model.blk[0].cache_kv = None
    with self.assertRaisesRegex(ValueError,'before running'):projection_plan(model,checkpoint)

  def test_rejects_mismatched_bias_before_replacement(self):
    model = self.model()
    original = model.blk[0].attn_q
    original.bias = model.token_embd.weight[0]
    checkpoint = self.checkpoint([('blk.0.attn_q.weight',(32,32),8,b'\0'*(32*34),None)])
    projection_plan(model,checkpoint)
    original.bias = original.bias[:16]
    with self.assertRaisesRegex(ValueError,'resident weight and bias'):projection_plan(model,checkpoint)
    self.assertIs(model.blk[0].attn_q,original)


if __name__=='__main__':unittest.main()
