"""Optional preparation of CPU MLA projection tables before JIT tracing."""
from tinygrad import Tensor, dtypes


def configure_weights(model, enabled:bool=True) -> int:
  """Realize contiguous FP16 tables once; recreate captures after changing weights."""
  from tinygrad.llm.model import MLATransformerBlock
  if type(enabled) is not bool:raise ValueError('CPU MLA weight preparation selection must be a bool')
  if not enabled:return 0
  weights:list[Tensor] = []
  for block in getattr(model,'blk',[]):
    if not isinstance(block,MLATransformerBlock):continue
    config=block.config
    expected=((config.n_heads,config.kv_lora_rank,config.head_dim-config.rope_dim),
              (config.n_heads,config.v_head_dim,config.kv_lora_rank))
    tables=(block.attn_k_b,block.attn_v_b)
    values=[table.get('weight') if isinstance(table,dict) else None for table in tables]
    if any(not isinstance(w,Tensor) or w.device!='CPU' or w.dtype is not dtypes.half or w.shape!=shape
           or not all(isinstance(n,int) and 1<=n<=4096 for n in w.shape) for w,shape in zip(values,expected)):
      return 0
    weights.extend(w for w in values if isinstance(w,Tensor))
  if not weights or sum(w.nbytes() for w in weights)>256*1024**2:return 0
  # Keep state-dict tensor identities and values. No derived copies are added
  # to model attributes, so implicit function arguments cannot retain them.
  for weight in weights:weight.replace(weight.contiguous().realize())
  return len(weights)
