"""Validated, lazy Q8_0 GGUF byte ranges for experimental Mali projections."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import mmap, struct
from typing import Any
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, read_int32, read_int64, read_str, read_uint32, read_uint64, readers


@dataclass(frozen=True)
class Q8Entry:
  shape:tuple[int,...]
  ggml_type:int
  offset:int
  nbytes:int


class Q8Checkpoint:
  """Read metadata before creating any device buffers; reject aliased or truncated payloads.

  The initial integration deliberately supports single-file checkpoints only. Raw
  names do not reproduce architecture-specific weight permutations or splits.
  """
  def __init__(self, path:Path|str):
    self.path = Path(path).resolve()
    size = self.path.stat().st_size
    try:
      with self.path.open('rb') as file:
        if file.read(4)!=b'GGUF' or read_int32(file) not in (2,3):raise ValueError('Invalid GGUF format')
        count,metadata_count = read_int64(file),read_int64(file)
        if not 0<=count<=size//24 or not 0<=metadata_count<=size//12:raise ValueError('Invalid GGUF entry counts')
        self.metadata:dict[str,Any] = {}
        for _ in range(metadata_count):
          key,kind = read_str(file),read_int32(file)
          if key in self.metadata:raise ValueError('Duplicate GGUF metadata key')
          self.metadata[key] = readers[kind](file)
        if self.metadata.get('split.count',1)!=1:raise ValueError('Mali Q8 projections currently require a single-file GGUF')
        alignment = self.metadata.get('general.alignment',32)
        if type(alignment) is not int or alignment<=0 or alignment&(alignment-1):raise ValueError('Invalid GGUF alignment')
        records = {}
        for _ in range(count):
          name,rank = read_str(file),read_uint32(file)
          if name in records:raise ValueError('Duplicate GGUF tensor name')
          if not 1<=rank<=4:raise ValueError('Invalid GGUF tensor rank')
          dims = tuple(read_uint64(file) for _ in range(rank))
          kind,offset = read_int32(file),read_uint64(file)
          if any(dim<=0 for dim in dims) or offset%alignment:raise ValueError('Invalid GGUF tensor dimensions or alignment')
          if kind in _GGML_NATIVE:block,width = 1,_GGML_NATIVE[kind].itemsize
          elif kind in _GGML_QUANT:block,width = _GGML_QUANT[kind]
          else:raise ValueError(f'Unsupported GGUF tensor type {kind}')
          if dims[0]%block:raise ValueError('GGUF quantization block does not divide tensor rows')
          records[name] = (tuple(reversed(dims)),kind,offset,prod(dims)//block*width)
        start = round_up(file.tell(),alignment)
    except (KeyError,IndexError,struct.error,UnicodeDecodeError) as exc:
      raise ValueError('Invalid or truncated GGUF metadata') from exc
    self.entries = {name:Q8Entry(shape,kind,start+offset,nbytes) for name,(shape,kind,offset,nbytes) in records.items()}
    previous_end = start
    for entry in sorted(self.entries.values(),key=lambda entry:entry.offset):
      if entry.offset<previous_end:raise ValueError('Overlapping GGUF tensor payloads')
      if entry.offset+entry.nbytes>size:raise ValueError('Truncated GGUF tensor payload')
      previous_end = entry.offset+entry.nbytes

  def weight(self, name:str, shape:tuple[int,int], device:str) -> Tensor:
    entry = self.entries[name]
    if entry.ggml_type!=8 or entry.shape!=shape or shape[1]%32:
      raise ValueError(f'Native Q8 projection requires matching Q8_0 shape for {name}')
    # A standalone device buffer severs the DISK view before the FP16 layer is
    # replaced. Neither the original layer nor a decoded/packed weight is kept.
    source = Tensor(self.path)
    source_device = source.device
    if not isinstance(source_device,str) or not source_device.startswith('DISK:'):
      raise ValueError('Native Q8 projection requires one file-backed source device')
    raw = source[entry.offset:entry.offset+entry.nbytes].to(device).contiguous().realize()
    if raw.dtype is not dtypes.uint8 or raw.numel()!=shape[0]*(shape[1]//32)*34:raise ValueError('Invalid Q8_0 payload size')
    # Finish asynchronous uploads before evicting resident file-backed pages.
    # Otherwise conversion can retain another checkpoint-sized CPU mapping.
    Device[device].synchronize()
    mapping = getattr(Device[source_device],'mem',None)
    if mapping is not None and hasattr(mapping,'madvise') and (advice:=getattr(mmap,'MADV_DONTNEED',None)) is not None:
      mapping.madvise(advice)
    return raw


# These loaders preserve the raw GGUF row order for the named body projections.
# Other architectures may permute RoPE rows or split fused tensors, even when a
# raw tensor happens to have the same shape. Extend only with a tested mapping.
DIRECT_Q8_ARCHITECTURES = frozenset(('qwen2','qwen3','qwen35'))
BODY_PROJECTIONS = ('attn_q','attn_k','attn_v','attn_gate','attn_output','ffn_gate','ffn_up','ffn_down',
                    'attn_qkv','ssm_alpha','ssm_beta','ssm_out','ssm_g_a','ssm_g_b','ssm_f_a','ssm_f_b')


def projection_plan(model, checkpoint:Q8Checkpoint, *, recurrent:bool=False) -> tuple[list[tuple[object,str,str]],list[dict]]:
  """Validate all candidate shapes before any model mutation or Q8 allocation."""
  from tinygrad import nn
  from tinygrad.llm.model import GatedDeltaNetBlock, TransformerBlock
  from tinygrad.llm.kernels.mali_q8 import MaliQ8Linear
  if checkpoint.metadata.get('general.architecture') not in DIRECT_Q8_ARCHITECTURES:
    raise ValueError('Native Q8 body projections do not yet support this GGUF architecture')
  if (getattr(model,'_cached_tokens',[]) or any(getattr(value,'cnt',0) for name,value in vars(model).items() if name.endswith('_jit'))
      or any(any(hasattr(block,name) for name in ('cache_kv','cache_k','conv_state','recurrent_state')) for block in getattr(model,'blk',[]))):
    raise ValueError('Configure Q8 projections before running or warming the model')
  plan:list[tuple[object,str,str]] = []
  skipped:list[dict] = []
  for index,block in enumerate(getattr(model,'blk',[])):
    if not isinstance(block,TransformerBlock) and not (recurrent and isinstance(block,GatedDeltaNetBlock)):continue
    if isinstance(block,GatedDeltaNetBlock) and not (1<=block.head_k_dim<=256 and 1<=block.head_v_dim<=256):continue
    if (getattr(block,'prepared_qkv',None) is not None or getattr(block,'ffn_gate_up',None) is not None
        or isinstance(block,TransformerBlock) and getattr(block,'attn_qkv',None) is not None):
      raise ValueError('Configure Q8 projections before installing fused or prepared body projections')
    for name in BODY_PROJECTIONS:
      layer = getattr(block,name,None)
      if not isinstance(layer,(nn.Linear,MaliQ8Linear)):continue
      # An existing MaliLinear contains the original nn.Linear as `source`.
      # Replacing it also discards its optional FP16 packed GEMM buffer.
      key = f'blk.{index}.{name}.weight'
      entry = checkpoint.entries.get(key)
      if entry is None or entry.ggml_type!=8:
        skipped.append({'tensor':key,'reason':'missing Q8_0 tensor'})
        continue
      shape = (layer.out_features,layer.in_features) if isinstance(layer,MaliQ8Linear) else tuple(layer.weight.shape)
      if (len(shape)!=2 or entry.shape!=shape or shape[1]%32
          or isinstance(layer,nn.Linear) and layer.weight.dtype is not dtypes.half):
        raise ValueError(f'Native Q8 projection requires matching Q8_0 shape and FP16 operands for {key}')
      device = layer.raw.device if isinstance(layer,MaliQ8Linear) else layer.weight.device
      if device!=model.token_embd.weight.device or (layer.bias is not None and
          (layer.bias.shape!=(shape[0],) or layer.bias.device!=device or layer.bias.dtype not in (dtypes.half,dtypes.float))):
        raise ValueError(f'Native Q8 projection requires matching resident weight and bias devices for {key}')
      plan.append((block,name,key))
  if not plan:raise ValueError('No compatible Q8_0 body projections found')
  return plan,skipped
