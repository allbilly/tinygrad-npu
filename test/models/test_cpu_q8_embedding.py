"""Direct packed CPU rows retain GGUF rounding, views and mutation fallbacks."""
from types import SimpleNamespace
import numpy as np
import pytest
from tinygrad import Tensor,TinyJit,UOp,nn,dtypes
from tinygrad.helpers import Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels import rockchip_embedding as embedding
from tinygrad.uop.ops import Ops


def fixture():
  rng=np.random.default_rng(3267)
  vocab,width=67,64
  codes=rng.integers(-128,128,(vocab,width//32,32),dtype=np.int8)
  scales=rng.choice(np.array([0.,-.0,2**-24,.0001,.125,-.5,1.],dtype=np.float16),(vocab,width//32))
  raw=np.concatenate([scales.reshape(vocab,width//32,1).view(np.uint8),codes.view(np.uint8)],axis=-1).reshape(-1)
  packed=Tensor(raw,device='CPU').realize()
  source=nn.Embedding(vocab,width)
  source.weight=ggml_data_to_tensor(packed,vocab*width,8).reshape(vocab,width).half()
  return source,embedding.RockchipQ8Embedding(source,packed,vocab,width),raw


def test_static_boundary_invalid_and_repeated_rows_match_all_fp16_bits():
  source,lookup,_=fixture()
  for values in ([0,1,31,66],[-1,67,0,66],[31,31,1,0]):
    ids=Tensor([values],device='CPU',dtype=dtypes.int32).realize()
    expected=source(ids).numpy()
    actual=lookup(ids).numpy()
    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
  assert lookup.fast_traces==3 and lookup.fallback_traces==0


def test_symbolic_rows_replay_and_zero_tail_do_not_reuse_previous_counts():
  source,lookup,_=fixture()
  rng=np.random.default_rng(1632)
  ids=Tensor(rng.integers(0,67,(1,16),dtype=np.int32),device='CPU').realize()
  run=TinyJit(lambda x,count:lookup(x[:,:count]).realize())
  for count in (16,3,1,9,16):
    actual=run(ids,UOp.variable('cpu_q8_embedding_rows',1,16).bind(count)).pad_to((1,16,64)).numpy()
    expected=source(ids[:,:count]).numpy()
    np.testing.assert_array_equal(actual[:,:count].view(np.uint16),expected.view(np.uint16))
    assert not np.any(actual[:,count:])
  assert any(p.op is Ops.PROGRAM and p.src[0].arg.function_name=='cpu_packed_q8_embedding'
             for p in run.captured._linear.toposort(enter_calls=True))


def test_training_other_index_layouts_and_replaced_weights_fall_back():
  source,lookup,_=fixture()
  probes=[Tensor([[0,1]],device='CPU',dtype=dtypes.int64),Tensor([[0,1],[2,3]],device='CPU',dtype=dtypes.int32)]
  for ids in probes:
    np.testing.assert_array_equal(lookup(ids).numpy(),source(ids).numpy())
  ids=Tensor([[0,1]],device='CPU',dtype=dtypes.int32).realize()
  with Context(TRAINING=1):np.testing.assert_array_equal(lookup(ids).numpy(),source(ids).numpy())
  source.weight=Tensor.full((67,64),2.,device='CPU',dtype=dtypes.half).realize()
  np.testing.assert_array_equal(lookup(ids).numpy(),np.full((1,2,64),2.,dtype=np.float16))
  assert lookup.fast_traces==0 and lookup.fallback_traces==4


def test_file_mapping_configuration_lifetime_disable_and_identity(tmp_path,monkeypatch):
  source,_,raw=fixture()
  path=tmp_path/'embedding.bin'
  path.write_bytes(raw.tobytes())
  monkeypatch.setattr(embedding,'gguf_parse',lambda p:({}, {'token_embd.weight':(Tensor(path),(67,64),8)}))
  model=SimpleNamespace(token_embd=source)
  metadata=embedding.configure(model,path)
  assert metadata['enabled'] and metadata['raw_bytes']==raw.nbytes
  ids=Tensor([[0,66]],device='CPU',dtype=dtypes.int32).realize()
  np.testing.assert_array_equal(model.token_embd(ids).numpy().view(np.uint16),source(ids).numpy().view(np.uint16))
  assert embedding.configure(model,path)['enabled']
  with pytest.raises(ValueError,match='checkpoint changed'):embedding.configure(model,tmp_path/'different.bin')
  assert path.read_bytes()==raw.tobytes()
  assert embedding.configure(model,path,enabled=False)=={} and model.token_embd is source
  with pytest.raises(ValueError,match='must be a bool'):embedding.configure(model,path,enabled=1)
