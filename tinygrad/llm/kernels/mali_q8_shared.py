"""Explicit inference-only tied Q8 storage, with a real lazy FP16 weight view.

Native lookup and short FP32 projection do not realize the logical dense view.
Reading or realizing ``weight`` remains mathematically valid, but may allocate
the complete dense matrix. Install before borrowing modules or warming graphs.
"""
from __future__ import annotations
from tinygrad import Device, Tensor, dtypes
from tinygrad.engine.jit import _TinyJit
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.mali_q8 import _validate_storage, embedding_q8, gemv_q8_rows


class MaliQ8Table:
  def __init__(self, raw:Tensor, vocab_size:int, dim:int):
    _validate_storage(raw,vocab_size,dim)
    self._raw=raw
    self.vocab_size,self.dim=vocab_size,dim
    device=raw.device
    assert isinstance(device,str)
    self.device=device
    # This is the actual decoded value, not a placeholder allocation. Metadata
    # consumers can inspect its shape/device/dtype without reading its storage.
    self._weight=ggml_data_to_tensor(raw.flatten(),vocab_size*dim,8).reshape(vocab_size,dim).cast(dtypes.half)

  @property
  def raw(self) -> Tensor:return self._raw

  @property
  def weight(self) -> Tensor:return self._weight

  def metadata(self) -> dict:
    """Report storage extents and realization, without asserting process RSS."""
    return {'retained_raw_bytes':self.raw.nbytes(),'raw_realized':self.raw.uop.is_realized,
            'logical_fp16_bytes':self.weight.nbytes(),'logical_weight_realized':self.weight.uop.is_realized}


class MaliQ8Embedding:
  def __init__(self, table:MaliQ8Table):self.table=table

  @property
  def weight(self) -> Tensor:return self.table.weight

  def __call__(self, ids:Tensor) -> Tensor:
    return embedding_q8(ids.to(self.table.device),self.table.raw,self.table.vocab_size,self.table.dim)


class MaliQ8Head:
  """Bounded FP32 inference head; half-product projections remain separate."""
  def __init__(self, table:MaliQ8Table):
    self.table=table
    self.in_features,self.out_features=table.dim,table.vocab_size
    self.bias=None

  @property
  def weight(self) -> Tensor:return self.table.weight

  def __call__(self, x:Tensor) -> Tensor:
    return gemv_q8_rows(x,self.table.raw,self.out_features,self.in_features)


def replace_tied_q8_table(model, raw:Tensor) -> dict:
  """Opt into one shared raw table on an unused, consistently prepared model.

  The caller verifies checkpoint provenance and uploads the owned raw buffer.
  Metadata checks do not compare raw values with the old dense matrix. There
  must be no previously borrowed modules or captured/in-flight model work.
  """
  from tinygrad.llm.kernels.mali import MaliEmbedding, MaliLinear
  from tinygrad.llm.kernels.mali_copy import _dense_buffer
  words,head=getattr(model,'token_embd',None),getattr(model,'output',None)
  if not isinstance(words,MaliEmbedding) or not isinstance(head,MaliLinear):
    raise ValueError('Shared Q8 replacement requires prepared Mali embedding and linear output')
  weight=words.weight
  if (words.prefill_only or not head.consistent_decode or head.bias is not None or head.packed_weight is not None
      or head.q8_weight is not None or weight.dtype is not dtypes.half or weight.ndim!=2
      or head.weight.uop is not weight.uop or head.weight.shape!=weight.shape or weight.device!=raw.device
      or (head.out_features,head.in_features)!=weight.shape):
    raise ValueError('Shared Q8 replacement requires one unpacked, tied FP16 consistent-decode weight')
  vocab,dim=weight.shape
  if type(vocab) is not int or type(dim) is not int:raise ValueError('Shared Q8 table requires static dimensions')
  _validate_storage(raw,vocab,dim)
  if getattr(model,'_cached_tokens',[]) or any(isinstance(value,_TinyJit) and (value.cnt or value.captured is not None)
                                            for value in vars(model).values()):
    raise ValueError('Shared Q8 replacement must run before model generation or graph capture')
  for block in getattr(model,'blk',[]):
    if any(isinstance(getattr(block,name,None),Tensor) for name in
           ('cache_kv','cache_k','cache_kg','conv_state','recurrent_state','cache_conv')):
      raise ValueError('Shared Q8 replacement must run before model state allocation')
  if _dense_buffer(raw) is None:raise ValueError('Shared Q8 replacement requires an owned resident raw buffer')
  table=MaliQ8Table(raw,vocab,dim)
  device=Device[table.device]
  device.synchronize()
  replaced_bytes=weight.nbytes()
  model.token_embd,model.output=MaliQ8Embedding(table),MaliQ8Head(table)
  del words,head,weight
  device.allocator.free_cache()
  return {'enabled':True,'replaced_dense_logical_bytes':replaced_bytes,'head_row_capacity':8,**table.metadata()}
