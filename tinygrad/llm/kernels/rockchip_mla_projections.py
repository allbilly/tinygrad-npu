"""Optional grouped NPU MLA projections with independent phase selection."""
from __future__ import annotations
from typing import Callable
from tinygrad import Tensor,dtypes


class MLAProjections:
  def __init__(self,key:Tensor,value:Tensor,*,prefill:bool=False,decode:bool=False,precision:str='fp16',projections:str='both'):
    if precision not in ('fp16','int8','int8_residual'):raise ValueError('Unsupported grouped MLA precision')
    if projections not in ('both','query','value'):raise ValueError('Unsupported grouped MLA projection selection')
    self.projections=projections
    self.precision=precision
    self.prefill,self.decode=prefill,decode
    self.phase:str|None=None
    self.heads,self.key_columns,self.key_depth=key.shape
    _,self.value_columns,self.value_depth=value.shape
    # Closures keep derived runtime weights out of recursive state-dict walks.
    self._sources=lambda:(key,value)
    self._packed:Callable[[],tuple[Tensor|None,...]|None]=lambda:None
    self._scales:Callable[[],tuple[Tensor|None,...]|None]=lambda:None
    self.packed_bytes=0

  def prepare(self):
    from tinygrad.llm.kernels.rockchip import pack_grouped_weight
    if self._packed() is None:
      pairs:list[tuple[Tensor|None,Tensor|None]]=[]
      for i,w in enumerate(self._sources()):
        if self.projections==('value' if i==0 else 'query'):pairs.append((None,None))
        elif self.precision in ('int8','int8_residual'):
          from tinygrad.llm.kernels.rockchip_grouped_int8 import pack_weight
          pairs.append(pack_weight(w,residual=self.precision=='int8_residual'))
        else:pairs.append((pack_grouped_weight(w),None))
      packed=tuple(pair[0].realize() if pair[0] is not None else None for pair in pairs)
      scales=tuple(pair[1] for pair in pairs)
      self._scales=lambda:scales
      self._packed=lambda:packed
      self.packed_bytes=sum(w.nbytes() for w in packed if w is not None)

  def _run(self,x:Tensor,key:bool) -> Tensor:
    from tinygrad.llm.kernels.rockchip import grouped_matmul
    weight=self._sources()[0 if key else 1]
    enabled=self.prefill if self.phase=='prefill' else self.decode if self.phase=='decode' else False
    if (not enabled or self.projections==('value' if key else 'query') or x.device!='CPU'
        or x.dtype is not dtypes.float or x.ndim!=4 or x.shape[:2]!=(1,self.heads)
        or x.shape[-1]!=weight.shape[-1] or not 1<=x.max_shape[2]<=128):return x@weight.transpose(-1,-2)
    packed=self._packed()
    if packed is None:raise RuntimeError('Prepare NPU MLA projections before capture')
    selected=0 if key else 1
    prepared=packed[selected]
    if prepared is None:raise RuntimeError('Selected MLA projection was not prepared')
    rows,capacity=x.shape[2],x.max_shape[2]
    assert isinstance(capacity,int)
    columns=weight.shape[1]
    assert isinstance(columns,int)
    dense=x.pad_to((1,self.heads,capacity,weight.shape[-1])).reshape(self.heads,capacity,weight.shape[-1])
    if self.precision in ('int8','int8_residual'):
      from tinygrad.llm.kernels.rockchip_grouped_int8 import matmul
      scales=self._scales()
      scale=scales[selected] if scales is not None else None
      if scale is None:raise RuntimeError('Prepare INT8 MLA scales before capture')
      result=matmul(dense,prepared,scale,columns,residual=self.precision=='int8_residual')
    else:result=grouped_matmul(dense,prepared,columns)
    return result.reshape(1,self.heads,capacity,columns)[:,:,:rows]

  def query(self,x:Tensor) -> Tensor:return self._run(x,True)
  def value(self,x:Tensor) -> Tensor:return self._run(x,False)


def configure(model,*,prefill:bool=False,decode:bool=False,precision:str='fp16',projections:str='both') -> list[MLAProjections]:
  """Install supported CPU FP16 MLA tables; recreate captures after reconfiguration."""
  from tinygrad.llm.model import MLATransformerBlock
  if type(prefill) is not bool or type(decode) is not bool:raise ValueError('NPU MLA phases must be bools')
  if precision not in ('fp16','int8','int8_residual'):raise ValueError('Unsupported grouped MLA precision')
  if projections not in ('both','query','value'):raise ValueError('Unsupported grouped MLA projection selection')
  blocks=[b for b in getattr(model,'blk',[]) if isinstance(b,MLATransformerBlock)]
  for block in blocks:
    if isinstance(getattr(block,'npu_mla_projections',None),MLAProjections):delattr(block,'npu_mla_projections')
  if not (prefill or decode) or not blocks:return []
  hooks=[]
  for block in blocks:
    c=block.config
    weights=[table.get('weight') if isinstance(table,dict) else None for table in (block.attn_k_b,block.attn_v_b)]
    expected=((c.n_heads,c.kv_lora_rank,c.head_dim-c.rope_dim),(c.n_heads,c.v_head_dim,c.kv_lora_rank))
    if (block.indexer is not None or any(not isinstance(w,Tensor) or w.device!='CPU' or w.dtype is not dtypes.half
        or w.shape!=shape or not all(isinstance(n,int) and 1<=n<=4096 for n in w.shape) or w.shape[0]>128
        for w,shape in zip(weights,expected))):return []
    key,value=weights
    assert isinstance(key,Tensor) and isinstance(value,Tensor)
    if precision in ('int8','int8_residual') and any(w.shape[0]*w.shape[1]>24576 for w in (key,value)):return []
    if precision=='int8_residual' and key.shape[0]>42:return []
    hooks.append(MLAProjections(key,value,prefill=prefill,decode=decode,precision=precision,projections=projections))
  if sum(w.nbytes() for hook in hooks for w in hook._sources())>256*1024**2:return []
  for hook in hooks:hook.prepare()
  for block,hook in zip(blocks,hooks):setattr(block,'npu_mla_projections',hook)
  return hooks
