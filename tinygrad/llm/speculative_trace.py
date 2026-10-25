"""Bounded caller-owned native-GDN verification prefix history (experimental)."""
from __future__ import annotations
from collections.abc import Sequence
from math import prod
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import GatedDeltaNetBlock, GatedDeltaTrace, Transformer, TransformerBlock


class NativeGDNPrefixTrace:
  """One shared workspace; a verified row remains available until the next verify.

  Eligibility is deliberately narrower than target_forward's injected scan hook:
  production speculation requires the qualified native Mali scan in every GDN
  block, ordinary positional attention, one stream, and at most eight input rows.
  """
  def __init__(self, target:Transformer, capacity:int):
    from tinygrad.llm.kernels.mali_recurrent import gated_delta_scan, supported
    if type(target) is not Transformer or target.config.hc is not None or type(capacity) is not int or not 1<=capacity<=8:
      raise ValueError('Prefix state tracing requires a base single-stream target and capacity 1..8')
    if target.config.ssm is not None and target.config.ssm.kda:raise ValueError('Prefix state tracing is not qualified for KDA')
    device = target.token_embd.weight.device
    if not isinstance(device,str) or not supported(device):raise ValueError('Prefix state tracing requires qualified native Mali hardware')
    indices:list[int] = []
    for index,block in enumerate(target.blk):
      if type(block) is TransformerBlock:continue
      if (type(block) is not GatedDeltaNetBlock or getattr(block,'gated_delta_scan',None) is not gated_delta_scan
          or not 1<=block.head_k_dim<=256 or not 1<=block.head_v_dim<=256):
        raise ValueError('Prefix state tracing supports only native GatedDeltaNet and ordinary positional attention')
      indices.append(index)
    if not indices:raise ValueError('Prefix state tracing requires a native GatedDeltaNet block')
    self.target,self.capacity,self.device = target,capacity,device
    self.indices = tuple(indices)
    shapes:list[tuple[int,...]] = []
    for index in indices:
      block = target.blk[index]
      assert isinstance(block,GatedDeltaNetBlock)
      shapes.extend(((capacity,1,block.num_v_heads,block.head_v_dim,block.head_k_dim),
                     (1,block.ssm_conv_kernel-1+capacity,block.conv_channels)))
    self.shapes = tuple(shapes)
    self.nbytes = sum(prod(shape)*dtypes.float.itemsize for shape in self.shapes)
    self.state_bytes = sum(prod(shape[1:])*dtypes.float.itemsize for shape in self.shapes[::2])
    self.window_bytes = sum(prod(shape)*dtypes.float.itemsize for shape in self.shapes[1::2])
    self.buffers:tuple[Tensor,...] = ()
    self.position,self.valid_rows = -1,0

  def prepare(self) -> None:
    """Allocate once inside the timed request, before any verifier capture."""
    if not self.buffers:
      buffers = tuple(Tensor.empty(*shape,device=self.device,dtype=dtypes.float) for shape in self.shapes)
      Tensor.realize(*buffers)
      self.buffers = buffers

  def bind(self, buffers:Sequence[Tensor]|None=None) -> tuple[GatedDeltaTrace|None,...]:
    values = self.buffers if buffers is None else tuple(buffers)
    if len(values)!=len(self.shapes):raise RuntimeError('Prefix trace workspace has not been prepared')
    if any(value.shape!=shape or value.dtype is not dtypes.float or value.device!=self.device for value,shape in zip(values,self.shapes)):
      raise ValueError('Prefix trace buffers do not match their fixed workspace geometry')
    traces:list[GatedDeltaTrace|None] = [None]*len(self.target.blk)
    for slot,index in enumerate(self.indices):traces[index] = GatedDeltaTrace(values[2*slot],values[2*slot+1])
    return tuple(traces)

  def invalidate(self) -> None:self.position,self.valid_rows = -1,0

  def mark_valid(self, position:int, rows:int) -> None:
    if not self.buffers or type(position) is not int or position<0 or type(rows) is not int or not 1<=rows<=self.capacity:
      raise ValueError('Invalid verified prefix trace extent')
    self.position,self.valid_rows = position,rows

  def can_restore(self, position:int, rows:int) -> bool:
    return self.position==position and 1<=rows<=self.valid_rows

  def restore(self, position:int, rows:int) -> tuple[Tensor,...]:
    if not self.can_restore(position,rows):raise RuntimeError('Prefix trace does not cover the requested verified inputs')
    stores:list[Tensor] = []
    states:list[Tensor] = []
    for index,trace in zip(self.indices,(trace for trace in self.bind() if trace is not None)):
      block = self.target.blk[index]
      if not isinstance(block,GatedDeltaNetBlock):raise RuntimeError('Target block changed after prefix trace preparation')
      if not hasattr(block,'recurrent_state') or not hasattr(block,'conv_state'):raise RuntimeError('Target state was not initialized')
      state,conv = block.recurrent_state,block.conv_state
      stores.extend((state.assign(trace.state_history[rows-1]),conv.assign(trace.conv_window[:,rows:rows+block.ssm_conv_kernel-1])))
      states.extend((state,conv))
    Tensor.realize(*stores)
    return tuple(states)

  @property
  def restored_bytes(self) -> int:
    # Each selected state/window is read once and written once.
    return 2*sum((prod(shape[1:]) if i%2==0 else prod((shape[0],shape[1]-self.capacity,shape[2])))*dtypes.float.itemsize
                 for i,shape in enumerate(self.shapes))
