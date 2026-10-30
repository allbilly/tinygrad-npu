"""CPU MLA table preparation preserves decoded values, projections and model state."""
import itertools
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip_mla import configure_weights
from test.models.test_minicpm3 import TestMiniCPM3 as MiniReference


def model(heads=4,nope=32,latent=64,value=32):
  return Transformer(TransformerConfig(1,heads*value,heads*value*2,heads,heads,1e-5,96,nope+16,
    10000.,16,value,max_context=256,q_lora_rank=32,kv_lora_rank=latent))


def payload(shape,seed):
  rng=np.random.default_rng(seed)
  codes=rng.integers(-128,128,(int(np.prod(shape))//32,32),dtype=np.int8)
  scales=rng.uniform(-.004,.004,len(codes)).astype(np.float16)
  packed=np.concatenate((scales.view(np.uint8).reshape(-1,2),codes.view(np.uint8)),axis=1).ravel()
  expected=(codes.astype(np.float32)*scales.astype(np.float32)[:,None]).astype(np.float16).reshape(shape)
  return packed,expected


def source_text(jit):
  assert jit.captured is not None
  return '\n'.join(str(s.arg) for call in jit.captured.linear.src if call.body.op.name=='PROGRAM'
    for s in call.body.src if s.op.name=='SOURCE')


@pytest.mark.parametrize('heads,nope,latent,value,capacity',[(4,32,64,32,17),(40,64,256,64,128)])
def test_q8_projection_rounding_changed_inputs_and_captured_source(heads,nope,latent,value,capacity):
  m=model(heads,nope,latent,value)
  block=m.blk[0]
  packed,expected=payload((heads,nope+value,latent),1701)
  raw=Tensor(packed,device='CPU').realize()
  fused=ggml_data_to_tensor(raw,int(np.prod(expected.shape)),8).half().reshape(expected.shape)
  block.attn_k_b['weight']=fused[:,:nope].transpose(1,2)
  block.attn_v_b['weight']=fused[:,nope:]
  before=nn.state.get_state_dict(m)
  identities={key:id(w) for key,w in before.items()}
  assert configure_weights(m)==2
  assert {key:id(w) for key,w in nn.state.get_state_dict(m).items()}==identities
  np.testing.assert_array_equal(block.attn_k_b['weight'].numpy().view(np.uint16),expected[:,:nope].transpose(0,2,1).view(np.uint16))
  np.testing.assert_array_equal(block.attn_v_b['weight'].numpy().view(np.uint16),expected[:,nope:].view(np.uint16))
  assert configure_weights(m)==2 and configure_weights(m,False)==0
  rng=np.random.default_rng(1702)
  for name,depth,weight in [('key',nope,block.attn_k_b['weight']),('value',latent,block.attn_v_b['weight'])]:
    def original(x,bytes_,rows):
      table=ggml_data_to_tensor(bytes_,int(np.prod(expected.shape)),8).half().reshape(expected.shape)
      w=table[:,:nope].transpose(1,2) if name=='key' else table[:,nope:]
      return (x[:,:,:rows]@w.transpose(-1,-2)).pad_to((1,heads,capacity,w.shape[1])).contiguous().realize()
    control=TinyJit(original)
    candidate=TinyJit(lambda x,w,rows:(x[:,:,:rows]@w.transpose(-1,-2)).pad_to((1,heads,capacity,w.shape[1])).contiguous().realize())
    variable=UOp.variable('mla_'+name+'_rows',1,capacity)
    for index,rows in enumerate((capacity,1,17,capacity)):
      inputs=Tensor(rng.normal(0,.2,(1,heads,capacity,depth)).astype(np.float32),device='CPU').realize()
      a=control(inputs,raw,variable.bind(rows)).numpy()
      b=candidate(inputs,weight,variable.bind(rows)).numpy()
      np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
      assert not a[:,:,rows:].any()
    old,new=source_text(control),source_text(candidate)
    assert old!=new and 'unsigned char' in old and 'unsigned char' not in new


def test_complete_model_prefill_cached_decode_reset_and_generation():
  fixture=MiniReference()
  state=fixture.fixture()
  control,candidate=fixture.load(state),fixture.load(state)
  assert configure_weights(candidate)==4
  for ids,start,prefix in (([3,7,2],0,[3,7,2]),([9],3,[3,7,2,9]),([8,4],0,[8,4])):
    expected=fixture.reference(state,prefix)
    a=control.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
    b=candidate.logits(Tensor([ids],device='CPU'),start).numpy().ravel()
    np.testing.assert_allclose(b,expected,rtol=2e-4,atol=1e-4)
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
  prompt=[3,7,2]
  expected_ids=list(itertools.islice(control.generate(prompt.copy(),chunk_size=2),3))
  assert expected_ids==list(itertools.islice(candidate.generate(prompt.copy(),chunk_size=2),3))


def test_unsupported_tables_are_not_partially_prepared_and_dtype_is_not_changed():
  m=model()
  block=m.blk[0]
  before=block.attn_k_b['weight'].uop
  assert configure_weights(m)==0 and block.attn_k_b['weight'].uop is before
  block.attn_k_b['weight']=block.attn_k_b['weight'].half()
  before=block.attn_k_b['weight'].uop
  assert configure_weights(m)==0 and block.attn_k_b['weight'].uop is before
  assert configure_weights(object())==0
  with pytest.raises(ValueError):configure_weights(m,1)
