"""Explicit FP16 Rockchip matrix kernels; surrounding operations retain their device."""
from __future__ import annotations
import contextlib, functools, math, statistics, struct, time, weakref
from collections.abc import Callable
from dataclasses import replace
from tinygrad import Device, Tensor, TinyJit, dtypes, nn
from tinygrad.engine.jit import _TinyJit
from tinygrad.helpers import Target, round_up, getenv, ContextVar
from tinygrad.llm.kernels.rockchip_shared_pool import ONPU_SHARED_HOST_POOL, SharedHostPool, shared_host_pool, erf_pool_eligible, gemma4_pool_eligible
from tinygrad.llm.kernels.rockchip_hadamard import ONPU_INT8_HADAMARD
from tinygrad.llm.kernels.rockchip_prepare_pool import (ONPU_GDN_PREP_POOL, ONPU_GDN_DECODE_PREP_POOL, ONPU_LFM_FLAT_POINTWISE,
                                                          rewrite_preparation_programs)
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp, resolve


ONPU_CPU_RMS_ALL_ROWS = ContextVar('ONPU_CPU_RMS_ALL_ROWS',0)
# `ONPU_CPU_RMS_ALL_ROWS` only routes a one-token decode norm through the dedicated CPU
# kernel when the width is 4096 (ChatGLM). Other models (MiniCPM3: width 2560) fall back
# to tinygrad's generic elementwise decomposition at decode, which costs ~5.9 us more per
# norm. This opt-in extends the dedicated kernel to every 4-aligned width; its exact decode
# arithmetic is bit-identical to the generic decomposition (verified at 2560, fp16 weights).
ONPU_CPU_RMS_ALL_WIDTHS = ContextVar('ONPU_CPU_RMS_ALL_WIDTHS',0)
# One-token decode may reuse the prepared QKV graph (one host kernel for INT8
# dequantize + rotary + KV-cache store) instead of the generic elementwise graph
# that launches a separate dequantize, query-rotary and key/value-store kernel
# per layer. Default off: it is an opt-in A/B switch, prefill is unaffected.
ONPU_DECODE_FUSED_QKV_ROPE = ContextVar('ONPU_DECODE_FUSED_QKV_ROPE',0)

@functools.cache
def _matmul(out:UOp, lhs:UOp, rhs:UOp, m:int, n:int, k:int, cores:int, column_offset:int, row_offset:int=0,
            split_k:bool=False) -> UOp:
  sink = UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_matmul_{m}_{n}_{k}'))
  info = replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
             UOp(Ops.SOURCE,arg=f'Rockchip FP16 {m}x{k}x{n} columns+{column_offset}'),
             UOp(Ops.BINARY,arg=struct.pack('<7I',m,n,k,cores,column_offset*k*2,row_offset*k*2,int(split_k)))))


def pack_weight(weight:Tensor) -> Tensor:
  n,k = weight.shape
  if not isinstance(n,int) or not isinstance(k,int) or not 0 < k <= 4096 or not 0 < n <= 24576:
    raise ValueError('Rockchip weight tiles require static K<=4096 and N<=24576')
  n,k = round_up(n,32),round_up(k,32)
  return weight.half().pad(((0,n-weight.shape[0]),(0,k-weight.shape[1]))).reshape(n//16,16,k//32,32).permute(0,2,1,3).contiguous().to('ROCKCHIP')


def pack_split_weight(weight:Tensor) -> Tensor:
  n,k = weight.shape
  if not isinstance(n,int) or not isinstance(k,int) or k%96 or not 0<k<=4096 or not 0<n<=4096:
    raise ValueError('Unsupported three-part Rockchip weight')
  n = round_up(n,32)
  return weight.half().pad(((0,n-weight.shape[0]),(0,0))).reshape(n//16,16,3,k//96,32).permute(
    2,0,3,1,4).contiguous().to('ROCKCHIP')


def packed_matmul(x:Tensor, weight:Tensor, columns:int, cores:int=3, column_offset:int=0, raw_output:bool=False) -> Tensor:
  """Return FP32 accumulation in the input device, with no host numeric fallback."""
  if cores not in (1,3): raise ValueError('Rockchip supports one or three cores')
  if x.ndim!=2 or weight.ndim!=4 or weight.shape[2:]!=(16,32) or weight.dtype is not dtypes.half or weight.device!='ROCKCHIP':
    raise ValueError('Rockchip matmul requires a matrix and an explicitly packed FP16 NPU weight')
  m,original_k = x.shape
  capacity,k = weight.shape[0]*16,weight.shape[1]*32
  n = round_up(columns,32)
  if (not isinstance(m,int) or not 1 <= m <= min(64,65536//k) or not 0 < original_k <= k or columns<1
      or column_offset<0 or column_offset%32 or column_offset+n>capacity or n>(8192 if m==1 else 4096)*cores):
    raise ValueError('Rockchip input tile exceeds the tested CBUF bounds')
  lhs = x.half().pad(((0,0),(0,k-original_k))).reshape(m,k//8,8).permute(1,0,2).contiguous().to('ROCKCHIP')
  out = Tensor.empty(n//4,m,4,device='ROCKCHIP',dtype=dtypes.float)
  out = out.custom_kernel(lhs,weight,fxn=lambda o,a,w:_matmul(o,a,w,m,n,k,cores,column_offset))[0]
  if raw_output: return out.to(x.device).flatten()
  return out.to(x.device).permute(1,0,2).reshape(m,n)[:,:columns]


def packed_activation_matmul(lhs:Tensor, weight:Tensor, rows:int, columns:int, row_offset:int=0, column_offset:int=0,
                             split_k:bool=False, raw_output:bool=False) -> Tensor:
  """Consume one tile from concatenated (K/8,M,8) input payloads, returning CPU FP32."""
  if lhs.device!='ROCKCHIP' or lhs.dtype is not dtypes.half or lhs.ndim!=1 or weight.device!='ROCKCHIP':
    raise ValueError('Invalid packed Rockchip activation')
  k,n = (weight.shape[2]*96 if split_k else weight.shape[1]*32),round_up(columns,32)
  capacity = weight.shape[1 if split_k else 0]*16
  if (not 1<=rows<=min(64,65536//(k//3 if split_k else k)) or row_offset<0 or column_offset<0 or column_offset%32
      or (row_offset+rows)*k>lhs.numel() or column_offset+n>capacity or n>(4096 if split_k else 24576 if rows==1 else 12288)):
    raise ValueError('Packed Rockchip activation exceeds its tile bounds')
  out = Tensor.empty(*((3,) if split_k else ()),n//4,rows,4,device='ROCKCHIP',dtype=dtypes.float)
  out = out.custom_kernel(lhs,weight,fxn=lambda o,a,w:_matmul(o,a,w,rows,n,k,3,column_offset,row_offset,split_k))[0]
  if raw_output: return out.to('CPU')
  if split_k: return out.to('CPU').permute(0,2,1,3).reshape(3,rows,n).sum(0)[:,:columns]
  return out.to('CPU').permute(1,0,2).reshape(rows,n)[:,:columns]

@functools.cache
def _batched(out:UOp, lhs:UOp, rhs:UOp, rows:int, columns:int, depth:int, tile:int) -> UOp:
  sink=UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_batched_{rows}_{columns}_{depth}_{tile}'))
  info=replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
    UOp(Ops.SOURCE,arg=f'Batched FP16 {rows}x{depth}x{columns}'),
    UOp(Ops.BINARY,arg=struct.pack('<10I',rows,columns,depth,3,0,0,0,0,tile,0x42544d4d))))

def packed_tiles_matmul(payload:Tensor, weight:Tensor, rows:int, columns:int, tile:int) -> list[Tensor]:
  if (payload.device not in ('CPU','ROCKCHIP') or payload.dtype is not dtypes.half or payload.ndim!=1
      or weight.device!='ROCKCHIP' or weight.dtype is not dtypes.half or weight.ndim!=4 or weight.shape[2:]!=(16,32)):
    raise ValueError('Invalid batched FP16 layout')
  k=weight.shape[1]*32
  n=round_up(columns,32)
  if (not isinstance(k,int) or not 1<=rows<=1024 or not 1<=tile<=min(64,65536//k)
      or not 0<columns<=weight.shape[0]*16 or n>12288 or payload.numel()!=rows*k):
    raise ValueError('Unsupported batched FP16 geometry')
  out=Tensor.empty(rows*n,device='ROCKCHIP',dtype=dtypes.float).custom_kernel(payload.to('ROCKCHIP'),weight,
    fxn=lambda o,a,w:_batched(o,a,w,rows,n,k,tile))[0].to('CPU')
  return [out[first*n:(first+min(tile,rows-first))*n].reshape(n//4,min(tile,rows-first),4) for first in range(0,rows,tile)]

def batched_matmul(x:Tensor, weight:Tensor, columns:int) -> list[Tensor]:
  if x.device!='CPU' or x.ndim!=2 or not isinstance(x.shape[0],int): raise ValueError('Batched matmul requires a static CPU matrix')
  rows,k=x.shape
  assert isinstance(rows,int)
  if not isinstance(k,int) or not k or k%32 or k>4096: raise ValueError('Unsupported batched matmul depth')
  tile=min(64,65536//k)
  parts=[x[first:first+tile].half().reshape(-1,k//8,8).permute(1,0,2).flatten() for first in range(0,rows,tile)]
  payload=Tensor.cat(*parts).contiguous()
  return packed_tiles_matmul(payload,weight,rows,columns,tile)


def pack_grouped_weight(weight:Tensor) -> Tensor:
  """Pack separate immutable FP16 [groups,N,K] matrices for grouped execution."""
  if weight.ndim!=3 or not all(isinstance(n,int) and n>0 for n in weight.shape):raise ValueError('Invalid grouped Rockchip weights')
  groups,n,k=weight.shape
  assert isinstance(n,int) and isinstance(k,int)
  if groups>128 or n>4096 or k>4096:raise ValueError('Grouped Rockchip weights exceed tested bounds')
  pn,pk=round_up(n,32),round_up(k,32)
  return weight.half().pad(((0,0),(0,pn-n),(0,pk-k))).reshape(groups,pn//16,16,pk//32,32).permute(
    0,1,3,2,4).contiguous().to('ROCKCHIP')


@functools.cache
def _grouped(out:UOp,lhs:UOp,rhs:UOp,groups:int,rows:int,columns:int,depth:int,tile:int,cores:int,
             dynamic_weights:bool=False,row_major:bool=False) -> UOp:
  sink=UOp.sink(out,lhs,rhs,arg=KernelInfo(f'rockchip_grouped_{groups}_{rows}_{columns}_{depth}_{cores}'+
    ('_dynamic' if dynamic_weights else '')+('_row' if row_major else '')))
  info=replace(ProgramInfo.from_sink(sink,Target('ROCKCHIP')),outs=(0,),ins=(1,2))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=(out,lhs,rhs)),
    UOp(Ops.SOURCE,arg=f'Grouped FP16 {groups}x{rows}x{depth}x{columns}'),
    UOp(Ops.BINARY,arg=struct.pack('<8I',rows,columns,depth,cores,groups,tile,0x47524d4d,8 if row_major else 7 if dynamic_weights else 1))))


def grouped_matmul(x:Tensor,weight:Tensor,columns:int,cores:int=3,*,dynamic_weights:bool=False,row_major:bool=False) -> Tensor:
  """Run per-group FP16 matrices; dynamic RHS inputs are explicitly validated on every replay."""
  if type(dynamic_weights) is not bool:raise ValueError('Dynamic grouped weights selection must be a bool')
  if type(row_major) is not bool or row_major and not dynamic_weights:raise ValueError('Row-major FP16 requires explicit dynamic weights')
  if (x.device!='CPU' or x.ndim!=3 or not all(isinstance(n,int) and n>0 for n in x.shape)
      or weight.device!='ROCKCHIP' or weight.dtype is not dtypes.half or weight.ndim!=5 or weight.shape[-2:]!=(16,32)):
    raise ValueError('Invalid grouped Rockchip operands')
  groups,rows,depth=x.shape
  k,n=weight.shape[2]*32,round_up(columns,32)
  if (cores not in (1,3) or groups!=weight.shape[0] or groups>128 or rows>128 or depth>k or k>4096
      or not 0<columns<=weight.shape[1]*16 or n!=weight.shape[1]*16 or n>4096 or row_major and (k>512 or n>512)):
    raise ValueError('Grouped Rockchip geometry exceeds tested bounds')
  tile=min(128 if row_major else 64,65536//k)
  x=x.half().pad(((0,0),(0,0),(0,k-depth)))
  payload=(x.contiguous().to('ROCKCHIP') if row_major else
    Tensor.cat(*(x[:,first:first+tile].reshape(groups,-1,k//8,8).permute(0,2,1,3).reshape(groups,-1)
      for first in range(0,rows,tile)),dim=1).contiguous().to('ROCKCHIP'))
  out=Tensor.empty(groups*rows*n,device='ROCKCHIP',dtype=dtypes.float).custom_kernel(payload,weight,
    fxn=lambda o,a,w:_grouped(o,a,w,groups,rows,n,k,tile,cores,dynamic_weights,row_major))[0].to('CPU').reshape(groups,rows*n)
  if row_major:return out.reshape(groups,rows,n)[:,:,:columns]
  return Tensor.cat(*(out[:,first*n:(first+min(tile,rows-first))*n].reshape(groups,n//4,-1,4).permute(
    0,2,1,3).reshape(groups,-1,n) for first in range(0,rows,tile)),dim=1)[:,:,:columns]


class RockchipLinear:
  """An inference-only Linear with independent prefill and decode placement.

  Auto decisions include host/device transfers and use the original layer as the
  timing and numerical reference. Model weights are immutable after preparation.
  """
  def __init__(self, source:nn.Linear, prefill:str, decode:str, tile_columns:int=4096,
               decode_tile_columns:int|None=None):
    if prefill not in ('off','auto','on') or decode not in ('off','auto','on'): raise ValueError('Invalid NPU placement')
    if not 32<=tile_columns<=24576 or tile_columns%32: raise ValueError('Invalid NPU column tile')
    if decode_tile_columns is not None and (not 32<=decode_tile_columns<=24576 or decode_tile_columns%32):
      raise ValueError('Invalid NPU decode column tile')
    self.source, self.prefill, self.decode = source, prefill, decode
    self.tile_columns = tile_columns
    # Prefill caps each core at n<=4096 (12288 across three cores); a packed width
    # above that is re-chunked per phase. Decode may therefore pack separately.
    self.decode_tile_columns = decode_tile_columns if decode_tile_columns is not None else tile_columns
    self.decode_packed:list[Tensor] = []
    self.weight, self.bias = source.weight, source.bias
    self.phase:str|None = None
    self.batch_prefill = False
    self.shared_projection:tuple[RockchipLinear,int,int]|None = None
    columns, depth = self.weight.shape
    assert isinstance(columns,int) and isinstance(depth,int)
    self.columns, self.depth = columns, depth
    self.packed:list[Tensor] = []

  def shared_npu(self, x:Tensor) -> Tensor:
    assert self.shared_projection is not None
    joint,offset,columns=self.shared_projection
    return joint.npu(x)[...,offset:offset+columns]

  def prepare(self):
    if self.shared_projection is not None:return
    if not self.packed:
      self.packed = [pack_weight(self.weight[start:start+self.tile_columns]).realize()
                     for start in range(0,self.columns,self.tile_columns)]

  def _tiles_for(self, phase:str|None) -> tuple[int,list[Tensor]]:
    """Return (tile width, packed tiles) for the phase about to run."""
    if phase=='decode' and self.decode_tile_columns!=self.tile_columns:
      if not self.decode_packed:
        self.decode_packed = [pack_weight(self.weight[start:start+self.decode_tile_columns]).realize()
                              for start in range(0,self.columns,self.decode_tile_columns)]
      return self.decode_tile_columns, self.decode_packed
    return self.tile_columns, self.packed

  def release_packed(self):
    self.packed.clear()
    self.decode_packed.clear()

  def npu(self, x:Tensor) -> Tensor:
    if self.shared_projection is not None:return self.shared_npu(x)
    from tinygrad.llm.kernels.rockchip_cpu import unpack_tiles
    shape = x.shape
    dense = x.pad_to(x.max_shape).half().reshape(-1,x.shape[-1])
    assert isinstance(self.depth,int) and isinstance(self.columns,int)
    tile_columns, packed = self._tiles_for(self.phase)
    if self.batch_prefill and x.device=='CPU' and dense.shape[0]==1024 and len(packed)==1:
      out=unpack_tiles(batched_matmul(dense,packed[0],self.columns))[:,:self.columns].reshape(*x.max_shape[:-1],self.columns)
      out=out.shrink(tuple((0,s) for s in (*shape[:-1],self.columns))).cast(x.dtype)
      return out if self.bias is None else out+self.bias
    row_tile = min(64,65536//round_up(self.depth,32))
    raw = x.device=='CPU' and dense.shape[0]>1 and len(packed)==1 and round_up(self.columns,32)<=12288 \
      and math.ceil(dense.shape[0]/row_tile)<=64
    rows = []
    for start in range(0,dense.shape[0],row_tile):
      part = dense[start:start+row_tile]
      if raw:
        rows.append(packed_matmul(part,packed[0],self.columns,raw_output=True).reshape(round_up(self.columns,32)//4,part.shape[0],4))
        continue
      columns = []
      for i,w in enumerate(packed):
        width = min(tile_columns,self.columns-i*tile_columns)
        # Three cores share the packed input and write disjoint output columns in both phases.
        for col in range(0,width,24576 if part.shape[0]==1 else 12288):
          end = min(width,col+(24576 if part.shape[0]==1 else 12288))
          columns.append(packed_matmul(part,w,end-col,column_offset=col))
      rows.append(Tensor.cat(*columns,dim=1))
    out = (unpack_tiles(rows)[:,:self.columns] if raw else Tensor.cat(*rows,dim=0)).reshape(*x.max_shape[:-1],self.weight.shape[0])
    out = out.shrink(tuple((0,s) for s in (*shape[:-1],self.weight.shape[0]))).cast(x.dtype)
    return out if self.bias is None else out+self.bias

  def __call__(self, x:Tensor) -> Tensor:
    phase = self.phase or ('decode' if resolve(math.prod(x.shape[:-1])==1,default=False) else 'prefill')
    mode = getattr(self,phase)
    return self.npu(x) if mode=='on' else self.source(x)


class RockchipProjections:
  """One shared-input projection, retaining each original output and phase policy."""
  def __init__(self, layers:tuple[RockchipLinear,...], tile_columns:int, share_packed:bool=False):
    self.layers = layers
    shared=share_packed and all(layer.prefill==layer.decode=='on' for layer in layers)
    source = nn.Linear.__new__(nn.Linear)
    source.weight = Tensor.cat(*(layer.weight for layer in layers),dim=0)
    biases = [layer.bias for layer in layers if layer.bias is not None]
    if biases and len(biases)!=len(layers): raise ValueError('Projection biases must share one layout')
    source.bias = Tensor.cat(*biases,dim=0) if biases else None
    if len({type(layer) for layer in layers})!=1: raise ValueError('Projection precisions must match')
    self.joint = type(layers[0])(source,'on','on',tile_columns)
    self.joint.batch_prefill = all(layer.batch_prefill for layer in layers)
    scales = tuple(getattr(layer,'input_scale',None) for layer in layers)
    if any(scale is not scales[0] for scale in scales): raise ValueError('Fused projections require one shared input channel scale')
    if scales[0] is not None: setattr(self.joint,'input_scale',scales[0])
    if shared:
      for layer in layers:layer.release_packed()
      Device['ROCKCHIP'].allocator.free_cache()
    try:self.joint.prepare()
    except BaseException as error:
      self.joint.release_packed()
      if shared:
        Device['ROCKCHIP'].allocator.free_cache()
        try:
          for layer in layers:layer.prepare()
        except BaseException as restore_error:
          if hasattr(error,'add_note'):error.add_note('Independent projection preparation also failed: '+str(restore_error))
      raise
    if shared:
      offset=0
      for layer in layers:
        # Retain the joint, not the group: model parameter traversal must stay acyclic.
        layer.shared_projection=(self.joint,offset,layer.columns)
        offset+=layer.columns

  def __call__(self, x:Tensor) -> tuple[Tensor,...]:
    phase = self.layers[0].phase or ('decode' if resolve(math.prod(x.shape[:-1])==1,default=False) else 'prefill')
    if not all(getattr(layer,phase)=='on' for layer in self.layers): return tuple(layer(x) for layer in self.layers)
    out = self.joint.npu(x)
    offset, parts = 0,[]
    for layer in self.layers:
      parts.append(out[...,offset:offset+layer.columns])
      offset += layer.columns
    return tuple(parts)


class RockchipRMSNorm:
  def __init__(self, source:nn.RMSNorm):
    self.source,self.weight,self.eps = source,source.weight,source.eps
    # Deferred/mapped weights arrive as lazy cast chains. Realizing the norm weight here
    # keeps that cast out of the captured decode graph, where it would otherwise run on
    # every token as a weight-only kernel at the graph head (0.155 ms/token).
    from tinygrad.llm.kernels.rockchip_cpu import ONPU_HOST_REALIZE_NORM_WEIGHTS
    if ONPU_HOST_REALIZE_NORM_WEIGHTS and self.weight is not None and self.weight.device=='CPU':
      self.weight = self.weight.contiguous().realize()

  def __call__(self, x:Tensor) -> Tensor:
    width = x.max_shape[-1]
    # Decode (one row) must be routed to the dedicated exact kernel; `ONPU_CPU_RMS_ALL_ROWS`
    # alone only covered ChatGLM's 4096, so other models ran the generic decomposition.
    decode_width_ok = (width==4096
                       or (ONPU_CPU_RMS_ALL_WIDTHS and isinstance(width,int) and 4<=width<=8192 and width%4==0))
    if (x.device=='CPU' and x.dtype is dtypes.float
        and (math.prod(x.max_shape[:-1])>=16 or ONPU_CPU_RMS_ALL_ROWS and decode_width_ok)
        and self.weight is not None and self.weight.device=='CPU' and self.weight.dtype in (dtypes.float,dtypes.half)
        and 4<=x.max_shape[-1]<=8192 and x.max_shape[-1]%4==0):
      from tinygrad.llm.kernels.rockchip_cpu import rms_norm
      return rms_norm(x,self.weight,self.eps)
    return self.source(x)


class RockchipPreparedQKV:
  def __init__(self, projections:RockchipProjections, heads:int, qnorm:Tensor|None, knorm:Tensor|None, eps:float,
               rope_dim:int|None=None, decode:bool=False):
    self.projections,self.heads,self.qnorm,self.knorm,self.eps = projections,heads,qnorm,knorm,eps
    # `rope_dim=None` keeps the original full-head rotary. `decode=True` opts the
    # one-token path into the prepared QKV graph (dequantize + rotary + cache store
    # in a single host kernel); prefill stays on the generic elementwise graph.
    self.rope_dim,self.decode = rope_dim,decode
    # The prepared kernel takes a RESIDENT contiguous bias. Under
    # --defer-weight-realization the fused INT8 QKV bias is a lazy gather over the
    # fused GGUF tensors (`Tensor.cat(half_split(q_bias), half_split(k_bias), v.clone())`),
    # so `prepare_qkv_int8`'s `bias.contiguous()` re-materialises it as a head-permuted
    # scalar-lane kernel inside EVERY replayed decode token (measured:
    # CPU:E_36_2_32_2, 28 calls x ~37 us = 1.04 ms/token, hoisted to the head of the
    # graph because the bias is weight-only). Realize it once, here, instead.
    joint = getattr(projections,'joint',None)
    bias = getattr(joint,'bias',None)
    if isinstance(bias,Tensor) and bias.device=='CPU':
      joint.bias = bias.contiguous().realize()

  def enabled(self, x:Tensor) -> bool:
    phase = self.projections.layers[0].phase or ('decode' if resolve(math.prod(x.shape[:-1])==1,default=False) else 'prefill')
    if phase=='decode':
      return self.decode and x.device=='CPU' and x.dtype is dtypes.float and x.ndim==3 and x.shape[0]==1 \
        and x.max_shape[1]==1 and all(layer.decode=='on' for layer in self.projections.layers)
    return not self.decode and x.device=='CPU' and x.dtype is dtypes.float and x.ndim==3 and x.shape[0]==1 and x.max_shape[1]>=16 \
      and all(layer.prefill=='on' for layer in self.projections.layers)

  def __call__(self, x:Tensor, start:int|UOp, cache:Tensor, freqs:Tensor) -> tuple[Tensor,Tensor]:
    from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv
    return prepare_qkv(self.projections.joint.npu(x),cache,self.qnorm,self.knorm,freqs,start,self.heads,self.eps,
                       rope_dim=self.rope_dim or 0)


class RockchipFeedForward:
  """Fused FP32 SwiGLU and raw input packing for a CPU/NPU dense FFN."""
  def __init__(self, projections:RockchipProjections, down:RockchipLinear, split_down:bool=False):
    self.projections,self.down = projections,down
    self.split_weight = pack_split_weight(down.weight).realize() if split_down and down.depth%96==0 and down.columns<=4096 else None

  def enabled(self, x:Tensor) -> bool:
    phase = self.down.phase or ('decode' if resolve(math.prod(x.shape[:-1])==1,default=False) else 'prefill')
    return x.device=='CPU' and x.dtype is dtypes.float and all(
      getattr(layer,phase)=='on' for layer in (*self.projections.layers,self.down))

  def __call__(self, x:Tensor) -> Tensor:
    from tinygrad.llm.kernels.rockchip_cpu import swiglu_pack, swiglu_pack_tiles
    joint = self.projections.joint
    input_tile = min(64,65536//round_up(joint.depth,32))
    depth = self.down.depth
    split = self.split_weight is not None
    row_tile = min(64,65536//(depth//3 if split else depth))
    if len(joint.packed)==1 and joint.columns%32==0 and joint.bias is None:
      dense = x.pad_to(x.max_shape).reshape(-1,x.shape[-1])
      parts = [packed_matmul(dense[start:start+input_tile],joint.packed[0],joint.columns,raw_output=True)
               for start in range(0,dense.shape[0],input_tile)]
      packed = swiglu_pack_tiles(parts,depth,row_tile,split)
    else:
      projected = joint.npu(x)
      packed = swiglu_pack(projected,depth,row_tile,split_k=split)
    packed = packed.to('ROCKCHIP')
    rows = []
    count = math.prod(x.max_shape[:-1])
    for start in range(0,count,row_tile):
      m = min(row_tile,count-start)
      if self.split_weight is not None:
        rows.append(packed_activation_matmul(packed,self.split_weight,m,self.down.columns,start,split_k=True,raw_output=True))
        continue
      columns = []
      for i,weight in enumerate(self.down.packed):
        width = min(self.down.tile_columns,self.down.columns-i*self.down.tile_columns)
        limit = 24576 if m==1 else 12288
        for col in range(0,width,limit):
          columns.append(packed_activation_matmul(packed,weight,m,min(limit,width-col),start,col))
      rows.append(Tensor.cat(*columns,dim=1))
    if self.split_weight is not None:
      from tinygrad.llm.kernels.rockchip_cpu import unpack_tiles
      out = unpack_tiles(rows,split=True)[:,:self.down.columns]
    else: out = Tensor.cat(*rows,dim=0)
    out = out.reshape(*x.max_shape[:-1],self.down.columns)
    out = out.shrink(tuple((0,s) for s in (*x.shape[:-1],self.down.columns))).cast(x.dtype)
    return out if self.down.bias is None else out+self.down.bias


def trim_cpu_preparation_cache() -> int:
  """Release unused CPU preparation allocations after configuration returns.

  Live model, captured graph and NPU buffers retain their owners. This is an
  opt-in startup operation for bounded-memory serving, outside request timings.
  """
  import gc
  gc.collect()
  allocator=Device['CPU'].allocator
  cached=sum(size*len(values) for (size,_),values in allocator.cache.items())
  allocator.free_cache()
  return cached


def prepare_cpu_shortconv_weights(model) -> int:
  """Prepare immutable CPU coefficients before capture, retaining their stored dtype."""
  from tinygrad.llm.model import ShortConvBlock
  blocks=[block for block in getattr(model,'blk',[]) if isinstance(block,ShortConvBlock)]
  if not blocks:raise ValueError('CPU short-convolution weights require ShortConv blocks')
  weights=[]
  for block in blocks:
    weight=block.shortconv.conv.get('weight')
    dim,kernel=block.shortconv.config.dim,block.shortconv.config.conv_kernel
    if (not isinstance(weight,Tensor) or weight.device!='CPU' or weight.dtype not in (dtypes.float,dtypes.half)
        or type(dim) is not int or type(kernel) is not int or not 4<=dim<=8192 or not 2<=kernel<=8
        or weight.shape!=(dim,kernel)):
      raise ValueError('Unsupported CPU short-convolution coefficient layout')
    weights.append(weight)
  # Realize every replacement before installing any, including failure paths.
  prepared=[weight.contiguous().realize() for weight in weights]
  for block,weight in zip(blocks,prepared):block.shortconv.conv['weight']=weight
  return len(prepared)


def configure(model, prefill:str='off', decode:str='off', tile_columns:int=4096, fuse_projections:bool=False,
              decode_tile_columns:int|None=None,
              cpu_attention:bool=False, native_replay:bool=False, fused_ffn:bool=False, split_down:bool=False, fused_qkv:bool=False,
              cpu_norm:bool=False, precision:str='fp16', cpu_pool:bool=False, cpu_fast_softmax:bool=False,
              cache_cleaning:bool=False, wake_helper:bool=False, input_scales:dict[str,Tensor]|None=None,
              batch_prefill:bool=False, prefill_rows:int=32, cpu_packed_attention:bool=False, cpu_fp16_qk:bool=False,
              cpu_int8_qk:bool=False, cpu_prefill_pool:bool=False, share_projection_weights:bool=False,
              cpu_recurrent_scan:bool=False, cpu_static_gemm:bool=False, cpu_gelu:bool=False,
              cpu_softcap:bool=False, fused_gelu_ffn:bool=False, cpu_mla_weights:bool=False, cpu_mla_attention:bool=False,
              cpu_experts:bool=False, npu_experts:bool=False,cpu_expert_prefill_rows:int=4,
              npu_expert_cpu_layers:tuple[int,...]=(),npu_expert_cpu_down_layers:tuple[int,...]=(),
              npu_expert_activation_residual:bool=False,npu_expert_weight_residual:bool=False,
              npu_mla_prefill:bool=False,npu_mla_decode:bool=False,npu_mla_precision:str='fp16',npu_mla_projections:str='both',
              fused_qkv_without_norm:bool=False,fused_swiglu_ffn:bool=False,cpu_mla_projections:bool=False,cpu_mla_pack_once:bool=False,
              fused_swiglu_decode:bool=False,cpu_erf_gelu:bool=False,
              fused_erf_gelu_ffn:bool=False,fused_erf_gelu_decode:bool=False,npu_prefill_attention:bool=False,
              fused_recurrent_projections:bool=False,cpu_head_gate:bool=False,cpu_shortconv_weights:bool=False,
              cpu_rwkv_weights:bool=False,cpu_mla_decode_pool:bool=False,cpu_exact_rows:bool=False,
              cpu_sparse_lookup:bool=False,fused_gelu_decode:bool=False,fused_gelu_sdk_prefill:bool=False) -> dict:
  """Replace only supported dense Linears; return measured placement decisions."""
  if type(share_projection_weights) is not bool:raise ValueError('share_projection_weights must be a bool')
  if type(cpu_shortconv_weights) is not bool:raise ValueError('CPU short-convolution weight selection must be a bool')
  if type(cpu_rwkv_weights) is not bool:raise ValueError('CPU RWKV7 weight selection must be a bool')
  if type(cpu_exact_rows) is not bool or type(cpu_sparse_lookup) is not bool:
    raise ValueError('Exact CPU preparation selection must be a bool')
  if (cpu_exact_rows or cpu_sparse_lookup) and not native_replay:raise ValueError('Exact CPU preparation requires native replay')
  if cpu_exact_rows and not (cpu_pool and cpu_prefill_pool):raise ValueError('Exact CPU rows require CPU prefill workers')
  if type(fused_gelu_decode) is not bool:raise ValueError('Fused GELU decode selection must be a bool')
  if type(fused_gelu_sdk_prefill) is not bool:raise ValueError('SDK GELU prefill selection must be a bool')
  if fused_gelu_sdk_prefill and (not fused_gelu_ffn or precision!='int8' or prefill=='off'):
    raise ValueError('SDK GELU prefill requires fused_gelu_ffn and active INT8 prefill')
  if fused_gelu_sdk_prefill and fused_gelu_decode:
    raise ValueError('SDK GELU prefill requires the baseline decode path')
  if fused_gelu_decode and (not fused_gelu_ffn or decode=='off'):
    raise ValueError('Fused GELU decode requires fused_gelu_ffn and an active NPU decode phase')
  if cpu_rwkv_weights and not (precision=='int8' and cpu_recurrent_scan and cpu_pool and cpu_prefill_pool and prefill==decode=='on'):
    raise ValueError('Prepared RWKV7 requires active INT8 phases and CPU recurrent/prefill workers')
  if type(fused_recurrent_projections) is not bool:raise ValueError('Fused recurrent projection selection must be a bool')
  if fused_recurrent_projections and (not fuse_projections or precision!='int8'):
    raise ValueError('Fused recurrent projections require fused INT8 projections')
  for block in getattr(model,'blk',[]):
    if isinstance(getattr(block,'recurrent_projections',None),RockchipProjections):delattr(block,'recurrent_projections')
  if type(npu_prefill_attention) is not bool:raise ValueError('NPU prefill attention selection must be a bool')
  if npu_prefill_attention and (not cpu_attention or prefill=='off'):
    raise ValueError('NPU prefill attention requires CPU attention hooks and active NPU prefill')
  # An 8192-column INT8 task can corrupt the following FP16 task; 8128 per core passes mixed replay.
  if npu_prefill_attention and precision=='int8' and tile_columns>24384:
    raise ValueError('NPU prefill attention requires column tiles <=24384 (8128 per core)')
  if type(fused_qkv_without_norm) is not bool:raise ValueError('Unnormalized QKV selection must be a bool')
  if fused_qkv_without_norm and not fused_qkv:raise ValueError('Unnormalized QKV requires fused_qkv')
  if type(cpu_static_gemm) is not bool:raise ValueError('Static CPU projection selection must be a bool')
  if type(cpu_gelu) is not bool:raise ValueError('CPU GELU selection must be a bool')
  if type(cpu_erf_gelu) is not bool:raise ValueError('CPU erf GELU selection must be a bool')
  if type(cpu_head_gate) is not bool:raise ValueError('CPU head gate selection must be a bool')
  if type(cpu_softcap) is not bool:raise ValueError('CPU soft cap selection must be a bool')
  if type(fused_gelu_ffn) is not bool:raise ValueError('Fused GELU FFN selection must be a bool')
  if type(fused_swiglu_ffn) is not bool:raise ValueError('Fused SwiGLU FFN selection must be a bool')
  if type(fused_swiglu_decode) is not bool:raise ValueError('Fused SwiGLU decode selection must be a bool')
  if type(fused_erf_gelu_ffn) is not bool:raise ValueError('Fused erf GELU selection must be a bool')
  if type(fused_erf_gelu_decode) is not bool:raise ValueError('Fused erf GELU decode selection must be a bool')
  if fused_erf_gelu_decode and (not fused_erf_gelu_ffn or decode=='off'):
    raise ValueError('Fused erf GELU decode requires fused_erf_gelu_ffn and an active NPU decode phase')
  if fused_erf_gelu_ffn and precision!='int8':raise ValueError('Fused erf GELU FFN requires INT8 projections')
  if fused_swiglu_decode and (not fused_swiglu_ffn or decode=='off'):
    raise ValueError('Fused SwiGLU decode requires fused_swiglu_ffn and an active NPU decode phase')
  if fused_swiglu_ffn and precision!='int8':raise ValueError('Fused SwiGLU FFN requires INT8 projections')
  if type(cpu_mla_attention) is not bool:raise ValueError('CPU MLA attention selection must be a bool')
  if type(cpu_mla_projections) is not bool:raise ValueError('CPU MLA projection selection must be a bool')
  if type(cpu_mla_pack_once) is not bool:raise ValueError('CPU MLA packed preparation selection must be a bool')
  if type(cpu_mla_decode_pool) is not bool:raise ValueError('CPU MLA decode pool selection must be a bool')
  if cpu_mla_decode_pool and not (cpu_mla_projections and cpu_pool):
    raise ValueError('CPU MLA decode pooling requires CPU projections and workers')
  if cpu_mla_pack_once and not (cpu_mla_weights and cpu_mla_projections):
    raise ValueError('CPU MLA packed preparation requires CPU weights and projections')
  if cpu_mla_projections and (npu_mla_prefill or npu_mla_decode):raise ValueError('Choose CPU or NPU MLA projections')
  if type(cpu_experts) is not bool:raise ValueError('CPU expert selection must be a bool')
  if type(npu_experts) is not bool:raise ValueError('NPU expert selection must be a bool')
  if npu_experts and (precision!='int8' or prefill!='on' or decode!='on'):
    raise ValueError('Experimental NPU experts require INT8 prefill and decode both on')
  if type(npu_mla_prefill) is not bool or type(npu_mla_decode) is not bool:raise ValueError('NPU MLA phases must be bools')
  if npu_mla_precision not in ('fp16','int8','int8_residual'):raise ValueError('Unsupported grouped MLA precision')
  if npu_mla_projections not in ('both','query','value'):raise ValueError('Unsupported grouped MLA projection selection')
  if fused_gelu_ffn and precision!='int8':raise ValueError('Fused GELU FFN requires INT8 projections')
  if precision not in ('fp16','int8'): raise ValueError('Unsupported Rockchip matrix precision')
  if not 1<=prefill_rows<=256: raise ValueError('Unsupported Rockchip placement query count')
  if cpu_packed_attention and not (cpu_attention and cpu_fast_softmax): raise ValueError('Packed attention requires fast CPU attention')
  if cpu_fp16_qk and not cpu_packed_attention: raise ValueError('FP16 QK requires packed CPU attention')
  if cpu_int8_qk and not cpu_packed_attention: raise ValueError('INT8 QK requires packed CPU attention')
  if cpu_int8_qk and cpu_fp16_qk: raise ValueError('Choose one CPU score-dot precision')
  if cpu_prefill_pool and not (cpu_pool and cpu_attention):raise ValueError('Prefill pooling requires CPU attention and workers')
  if input_scales and precision!='int8': raise ValueError('Input channel scales require INT8 placement')
  shortconv_weights=prepare_cpu_shortconv_weights(model) if cpu_shortconv_weights else 0
  from tinygrad.llm.kernels.rockchip_rwkv import prepare_weights as prepare_rwkv_weights
  rwkv_weights=prepare_rwkv_weights(model) if cpu_rwkv_weights else 0
  pool = None
  recurrent_pool = cpu_recurrent_scan and any(getattr(b,'num_v_heads',0)>=4 or
    getattr(getattr(b,'config',None),'rwkv7',None) and getattr(b.config,'n_heads',0)>=4 for b in getattr(model,'blk',[]))
  attention_pool_eligible = cpu_attention and any(getattr(getattr(b,'config',None),'n_kv_heads',0)>=4 or
    getattr(getattr(b,'config',None),'n_kv_heads',0)>=2 and getattr(getattr(b,'config',None),'max_context',0)>=4096
    for b in getattr(model,'blk',[]))
  from tinygrad.llm.kernels.rockchip_mla_attention import supported as mla_supported
  mla_pool = cpu_mla_attention and mla_supported(model) and any(getattr(getattr(b,'config',None),'kv_lora_rank',0)>0
    and getattr(b.config,'n_heads',0)>=4 for b in getattr(model,'blk',[]))
  expert_pool = (cpu_experts or npu_experts) and any(hasattr(b,'ffn_gate_exps') for b in getattr(model,'blk',[]))
  if cpu_pool and (cpu_prefill_pool or recurrent_pool or attention_pool_eligible or mla_pool or expert_pool):
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    try: pool = attention_pool()
    except OSError: pass
  from tinygrad.llm.kernels.rockchip_recurrent import configure as configure_recurrent, ONPU_GDN_DIRECT_DECODE
  recurrent_blocks=configure_recurrent(model,enabled=cpu_recurrent_scan,pool=pool)
  if cpu_rwkv_weights:
    from tinygrad.llm.kernels.rockchip_rwkv import configure as configure_rwkv
    configure_rwkv(model,enabled=True,pool=pool,specialize=True)
  from tinygrad.llm.kernels.rockchip_static import configure as configure_static
  static_projections = configure_static(model,enabled=cpu_static_gemm,pool=pool)
  from tinygrad.llm.kernels.rockchip_mla import configure_weights as configure_mla_weights
  mla_weights = configure_mla_weights(model,enabled=cpu_mla_weights,materialize=not cpu_mla_pack_once)
  from tinygrad.llm.kernels.rockchip_mla_attention import configure as configure_mla_attention
  mla_attention = configure_mla_attention(model,enabled=cpu_mla_attention,pool=pool)
  from tinygrad.llm.kernels.rockchip_mla_projections import configure as configure_mla_projections
  mla_projections=configure_mla_projections(model,prefill=npu_mla_prefill,decode=npu_mla_decode,
                                          precision=npu_mla_precision,projections=npu_mla_projections)
  from tinygrad.llm.kernels.rockchip_mla_cpu import configure as configure_cpu_mla_projections
  cpu_mla_projection_blocks=configure_cpu_mla_projections(model,enabled=cpu_mla_projections,
    pool=pool if cpu_mla_decode_pool or getenv('ONPU_CPU_MLA_PROJECTION_POOL',0) else None,decode_pool=cpu_mla_decode_pool)
  if cpu_mla_decode_pool and not cpu_mla_projection_blocks:
    raise ValueError('CPU MLA decode pooling requires supported projection tables')
  if cpu_mla_pack_once and (not cpu_mla_projection_blocks or mla_weights!=2*cpu_mla_projection_blocks):
    raise ValueError('CPU MLA packed preparation requires supported projection tables')
  from tinygrad.llm.kernels.rockchip_npu_experts import configure as configure_npu_experts
  npu_expert_blocks,npu_expert_bytes=configure_npu_experts(model,enabled=npu_experts,
    cpu_layers=npu_expert_cpu_layers,cpu_down_layers=npu_expert_cpu_down_layers,
    activation_residual=npu_expert_activation_residual,weight_residual=npu_expert_weight_residual)
  from tinygrad.llm.kernels.rockchip_experts import configure as configure_experts
  expert_layers=configure_experts(model,enabled=cpu_experts,pool=pool,
                                 prefill_rows=cpu_expert_prefill_rows)
  from tinygrad.llm.kernels.rockchip_softcap import softcap
  current_softcap=getattr(model,'cpu_logit_softcap',None)
  if current_softcap is softcap or isinstance(current_softcap,functools.partial) and current_softcap.func is softcap:
    delattr(model,'cpu_logit_softcap')
  softcap_enabled = cpu_softcap and .125<=getattr(getattr(model,'config',None),'final_logit_softcap',0)<=128
  if softcap_enabled:model.cpu_logit_softcap = functools.partial(softcap,pool=pool) if pool is not None else softcap
  gelu_blocks,erf_gelu_blocks = 0,0
  from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
  from tinygrad.llm.kernels.rockchip_gelu_erf import gated_erf_gelu
  for block in getattr(model,'blk',[]):
    if any(getattr(block,'cpu_gelu_activation',None) is fn for fn in (gated_gelu,gated_erf_gelu)):delattr(block,'cpu_gelu_activation')
    config = getattr(block,'config',None)
    if (cpu_gelu and config is not None and config.ffn_activation=='gelu_tanh' and not config.swiglu_limit
        and not config.swiglu_up_bias and not config.num_experts):
      block.cpu_gelu_activation = gated_gelu
      gelu_blocks += 1
    elif (cpu_erf_gelu and config is not None and config.ffn_activation=='gelu' and not config.swiglu_limit
          and not config.swiglu_up_bias and not config.num_experts):
      block.cpu_gelu_activation = gated_erf_gelu
      erf_gelu_blocks += 1
  linear_type:type[RockchipLinear] = RockchipLinear
  if precision=='int8':
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
    from tinygrad.llm.kernels.rockchip_greedy import greedy_argmax
    linear_type = RockchipInt8Linear
    model.rockchip_greedy = greedy_argmax
  layers:list[RockchipLinear] = []
  used_scales:set[str] = set()
  seen:set[int] = set()
  def visit(obj, prefix=''):
    if id(obj) in seen or isinstance(obj,Tensor): return
    seen.add(id(obj))
    items = list(enumerate(obj)) if isinstance(obj,list) else list(obj.items()) if isinstance(obj,dict) else list(vars(obj).items())
    for key,value in items:
      name = f'{prefix}.{key}' if prefix else str(key)
      if isinstance(value,nn.Linear):
        if len(value.weight.shape)!=2 or value.weight.shape[1]>(32768 if precision=='int8' else 4096): continue
        selected_type = linear_type
        if precision=='int8' and value.weight.shape[1]>4096:
          from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8SplitLinear
          selected_type = RockchipInt8SplitLinear
        layer = (selected_type(value,prefill,decode,tile_columns,decode_tile_columns)
                 if decode_tile_columns is not None else selected_type(value,prefill,decode,tile_columns))
        layer.batch_prefill = batch_prefill and precision=='fp16'
        if input_scales is not None and name in input_scales:
          setattr(layer,'input_scale',input_scales[name])
          used_scales.add(name)
        if isinstance(obj,(dict,list)): obj[key] = layer
        else: setattr(obj,str(key),layer)
        layers.append(layer)
      elif isinstance(value,(list,dict)) or hasattr(value,'__dict__') and not isinstance(value,(type,Tensor)): visit(value,name)
  visit(model)
  if input_scales and (unused:=input_scales.keys()-used_scales): raise ValueError(f'Unknown input channel scale layers: {sorted(unused)}')
  helper = None
  if native_replay and wake_helper:
    from tinygrad.llm.kernels.rockchip_idle import idle_helper
    try: helper = idle_helper()
    except (OSError,RuntimeError): pass
  timings_by_shape:dict[tuple,dict[str,float]] = {}
  decisions = []
  for layer_index,layer in enumerate(layers):
    if prefill==decode=='off': continue
    if not share_projection_weights or 'auto' in (layer.prefill,layer.decode):layer.prepare()
    for phase,rows in [('decode',1),('prefill',prefill_rows)]:
      if layer is getattr(model,'output',None): rows = 1
      mode = getattr(layer,phase)
      if mode!='auto': continue
      key = (str(layer.weight.device),layer.weight.dtype,*layer.weight.shape,phase,rows,getattr(layer,'input_scale',None) is not None)
      # Numerical accuracy depends on each layer's actual weights and channel
      # scales, even when another layer already qualified the same geometry.
      assert isinstance(layer.weight.device,str)
      x = (Tensor.rand(1,rows,layer.weight.shape[1],device=layer.weight.device)*.4-.2).realize()
      if rows>1: x = x[:,:UOp.variable('placement_rows',1,rows).bind(rows)]
      assert isinstance(x.device,str)
      original = TinyJit(lambda a:layer.source(a).realize())
      native = TinyJit(lambda a:layer.npu(a).realize())
      def invoke(fn):
        with helper if helper is not None else contextlib.nullcontext(): return fn(x).realize()
      expected_tensor,actual_tensor = invoke(original),invoke(native)
      expected = expected_tensor.pad_to(expected_tensor.max_shape).flatten().tolist()
      actual = actual_tensor.pad_to(actual_tensor.max_shape).flatten().tolist()
      assert isinstance(expected,list) and isinstance(actual,list)
      error = max(abs(a-b) for a,b in zip(actual,expected))
      valid = all(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-3+3e-3*abs(b) for a,b in zip(actual,expected))
      normalized_rmse = None
      if precision=='int8':
        finite = all(math.isfinite(a) and math.isfinite(b) for a,b in zip(actual,expected))
        normalized_rmse = math.sqrt(sum((a-b)**2 for a,b in zip(actual,expected))/max(sum(b*b for b in expected),1e-30)) if finite else math.inf
        valid = finite and normalized_rmse<=.03
      timing_reused = key in timings_by_shape
      if not timing_reused:
        # Include transfer and packing costs by realizing and synchronizing both
        # candidates. Only timings are reused for later matching geometries.
        timings = {}
        for name,fn in [('original',original),('npu',native)]:
          for _ in range(2): invoke(fn)
          if x.device.split(':')[0]=='CL':
            until = time.perf_counter()+.25
            while time.perf_counter()<until:
              invoke(fn)
              Device[x.device].synchronize()
          samples = []
          for _ in range(5):
            Device[x.device].synchronize()
            start = time.perf_counter()
            invoke(fn)
            Device[x.device].synchronize()
            samples.append(time.perf_counter()-start)
          timings[name] = statistics.median(samples)
        timings_by_shape[key] = timings
      timings = timings_by_shape[key]
      selected = valid and timings['npu']<timings['original']*.9
      decisions.append({'device':str(x.device),'shape':list(layer.weight.shape),'phase':phase,'rows':rows,'valid':valid,'max_abs_error':error,
                        'layer_index':layer_index,'timing_reused':timing_reused,'symbolic_rows':rows>1,'wake_helper':helper is not None,
                        'warmup_seconds':.25 if not timing_reused and x.device.split(':')[0]=='CL' else 0,
                        'normalized_rmse':normalized_rmse,'normalized_rmse_limit':.03 if precision=='int8' else None,
                        'original_ms':timings['original']*1000,'npu_ms':timings['npu']*1000,'selected':selected})
      setattr(layer,phase,'on' if selected else 'off')
    if layer.prefill==layer.decode=='off': layer.release_packed()
  groups = []
  if fuse_projections:
    from tinygrad.llm.model import GatedDeltaNetBlock
    for block in getattr(model,'blk',[]):
      projections = [('attn_qkv',('attn_q','attn_k','attn_v')),('ffn_gate_up',('ffn_gate','ffn_up'))]
      if fused_recurrent_projections and isinstance(block,GatedDeltaNetBlock) and not hasattr(block,'ssm_g_a'):
        projections.append(('recurrent_projections',('attn_qkv','attn_gate','ssm_beta','ssm_alpha')))
      for name,attrs in projections:
        members = tuple(member for attr in attrs if isinstance(member:=getattr(block,attr,None),RockchipLinear))
        if len(members)!=len(attrs): continue
        if not any(all(getattr(layer,phase)=='on' for layer in members) for phase in ('prefill','decode')): continue
        if len({layer.weight.shape[1] for layer in members})!=1 or len({layer.bias is None for layer in members})!=1: continue
        scales = tuple(getattr(layer,'input_scale',None) for layer in members)
        if any(scale is not scales[0] for scale in scales): continue
        group = RockchipProjections(members,tile_columns,share_packed=share_projection_weights)
        setattr(block,name,group)
        groups.append(group)
      # ONPU_BLOCK_DOMAIN_ORDER (default OFF, bit-exact): the historical placement
      # prepares every fused joint projection (qkv, gate+up) in this block loop and
      # only later prepares the standalone projections (attn_output, ffn_down) in a
      # second `layers` pass. The two passes advance the sequential iommu-domain
      # cursor separately, so a layer's four NPU submits end up in different weight
      # domains, and the blocking submit path then pays an iommu detach/attach
      # (rknpu_iommu_switch_domain) on nearly every submit. Preparing each block's
      # standalone projections right after its joints makes the cursor walk
      # block-major, so a layer stays in one domain and switches collapse to the
      # few domain boundaries per token. See scripts/probe_submit_domain.py.
      if share_projection_weights and getenv('ONPU_BLOCK_DOMAIN_ORDER',0):
        for member in getattr(block,'__dict__',{}).values():
          if (isinstance(member,RockchipLinear) and member.shared_projection is None
              and (member.prefill!='off' or member.decode!='off')):
            member.prepare()
  if share_projection_weights:
    for layer in layers:
      if layer.shared_projection is None and (layer.prefill!='off' or layer.decode!='off'):layer.prepare()
  ffn_blocks = 0
  if fused_ffn:
    for block in getattr(model,'blk',[]):
      ffn_group,down = getattr(block,'ffn_gate_up',None),getattr(block,'ffn_down',None)
      config = getattr(block,'config',None)
      if (isinstance(ffn_group,RockchipProjections) and isinstance(down,RockchipLinear) and down.weight.device=='CPU'
          and down.depth%32==0 and config is not None and config.swiglu_limit is None and getattr(config,'ffn_activation','silu')=='silu'
          and all(layer.columns==down.depth for layer in ffn_group.layers)):
        if precision=='int8':
          from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8FeedForward, RockchipInt8Linear
          assert isinstance(down,RockchipInt8Linear)
          block.fused_ffn = RockchipInt8FeedForward(ffn_group,down,pool=pool)
        else: block.fused_ffn = RockchipFeedForward(ffn_group,down,split_down)
        ffn_blocks += 1
  gelu_ffn_blocks = 0
  if fused_gelu_ffn or any(getattr(block,'fused_gelu_ffn',None) is not None for block in getattr(model,'blk',[])):
    from tinygrad.llm.model import TransformerBlock
    from tinygrad.llm.gemma3n import Gemma3nBlock
    from tinygrad.llm.gemma4 import Gemma4Block
    from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8GeluFeedForward
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
    for block in getattr(model,'blk',[]):
      if isinstance(getattr(block,'fused_gelu_ffn',None),RockchipInt8GeluFeedForward):delattr(block,'fused_gelu_ffn')
      gelu_group,down = getattr(block,'ffn_gate_up',None),getattr(block,'ffn_down',None)
      config = getattr(block,'config',None)
      standard=((type(block) is TransformerBlock or type(block) is Gemma4Block and fused_gelu_sdk_prefill)
                and config is not None and config.ffn_activation=='gelu_tanh'
                and not config.swiglu_limit and not config.swiglu_up_bias)
      nonsparse=type(block) is Gemma3nBlock and not math.isfinite(block.sparsity)
      if (fused_gelu_ffn and (standard or nonsparse) and isinstance(gelu_group,RockchipProjections)
          and isinstance(down,RockchipInt8Linear) and 512<=down.depth<=12288 and down.depth%32==0
          and gelu_group.joint.columns==down.depth*2 and len(gelu_group.joint.packed)==1 and down.input_scale is None
          and gelu_group.joint.bias is down.bias is None):
        setattr(block,'fused_gelu_ffn',RockchipInt8GeluFeedForward(gelu_group,down,pool=pool if cpu_prefill_pool else None,
                                                               decode=fused_gelu_decode, sdk_prefill=fused_gelu_sdk_prefill))
        gelu_ffn_blocks += 1
  erf_gelu_ffn_blocks,erf_gelu_decode_blocks = 0,0
  if fused_erf_gelu_ffn or any(getattr(block,'fused_erf_gelu_ffn',None) is not None for block in getattr(model,'blk',[])):
    from tinygrad.llm.model import TransformerBlock
    from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8ErfGeluFeedForward
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
    for block in getattr(model,'blk',[]):
      if isinstance(getattr(block,'fused_erf_gelu_ffn',None),RockchipInt8ErfGeluFeedForward):delattr(block,'fused_erf_gelu_ffn')
      erf_group,down = getattr(block,'ffn_gate_up',None),getattr(block,'ffn_down',None)
      config = getattr(block,'config',None)
      if (fused_erf_gelu_ffn and type(block) is TransformerBlock and config is not None and config.ffn_activation=='gelu'
          and not config.swiglu_limit and not config.swiglu_up_bias and isinstance(erf_group,RockchipProjections)
          and isinstance(down,RockchipInt8Linear) and 512<=down.depth<=12288 and down.depth%32==0
          and erf_group.joint.columns==down.depth*2 and len(erf_group.joint.packed)==1 and down.input_scale is None
          and erf_group.joint.bias is down.bias is None):
        raw_decode=fused_erf_gelu_decode and all(layer.decode=='on' for layer in (*erf_group.layers,down))
        setattr(block,'fused_erf_gelu_ffn',RockchipInt8ErfGeluFeedForward(erf_group,down,pool=pool if cpu_prefill_pool else None,
                                                                      decode=raw_decode))
        erf_gelu_ffn_blocks += 1
        erf_gelu_decode_blocks += int(raw_decode)
  swiglu_ffn_blocks,swiglu_decode_blocks = 0,0
  if fused_swiglu_ffn or any(getattr(block,'fused_swiglu_ffn',None) is not None for block in getattr(model,'blk',[])):
    from tinygrad.llm.model import FFNBlock
    from tinygrad.llm.kernels.rockchip_gelu_ffn import (RockchipInt8SwiGLUFeedForward, supported_down_bias, wide_swiglu_supported,
                                                         ONPU_WIDE_RAW_SWIGLU_DECODE)
    from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
    for block in getattr(model,'blk',[]):
      if isinstance(getattr(block,'fused_swiglu_ffn',None),RockchipInt8SwiGLUFeedForward):delattr(block,'fused_swiglu_ffn')
      swiglu_group,down = getattr(block,'ffn_gate_up',None),getattr(block,'ffn_down',None)
      config = getattr(block,'config',None)
      if (fused_swiglu_ffn and isinstance(block,FFNBlock) and config is not None and config.ffn_activation=='silu'
          and not config.swiglu_limit and not config.swiglu_up_bias and isinstance(swiglu_group,RockchipProjections)
          and isinstance(down,RockchipInt8Linear) and down.depth%32==0
          and ((512<=down.depth<=12288 and len(swiglu_group.joint.packed)==1) or wide_swiglu_supported(swiglu_group.joint,down))
          and swiglu_group.joint.columns==down.depth*2 and (down.input_scale is None or ONPU_INT8_HADAMARD)
          and swiglu_group.joint.bias is None and supported_down_bias(down)):
        # The wide (24576+2816) geometry keeps the RAW decode route available by env opt-in, so the
        # route can be A/B'd without changing the command line. Narrow geometries keep the CLI gate.
        wide_decode=bool(ONPU_WIDE_RAW_SWIGLU_DECODE) and wide_swiglu_supported(swiglu_group.joint,down)
        narrow_decode=fused_swiglu_decode and not wide_swiglu_supported(swiglu_group.joint,down)
        raw_decode=((narrow_decode or wide_decode)
                    and all(layer.decode=='on' for layer in (*swiglu_group.layers,down)))
        setattr(block,'fused_swiglu_ffn',RockchipInt8SwiGLUFeedForward(swiglu_group,down,pool=pool if cpu_prefill_pool else None,
                                                                 decode=raw_decode,legacy_division=fused_ffn and down.depth<=4096))
        swiglu_ffn_blocks += 1
        swiglu_decode_blocks += int(raw_decode)
  cleaner = None
  if native_replay and cache_cleaning:
    import ctypes, subprocess
    from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
    try: cleaner = ctypes.cast(cache_cleaner().clean_to,ctypes.c_void_p).value
    except (OSError,subprocess.TimeoutExpired): pass
  # The final vocabulary projection is one row in both phases. Select by model
  # phase, rather than treating every one-row kernel as a decode operation.
  shared_state:SharedHostPool|None = None
  exact_preparation_changes:list[dict] = []
  def in_phase(jit, phase):
    handle = getattr(jit,'_rockchip_base_jit',None)
    previous = handle() if isinstance(handle,weakref.ReferenceType) else handle
    if previous is not None:
      jit = TinyJit(previous.fxn,prune=previous.prune) if isinstance(previous,_TinyJit) and previous.fxn is not None else previous
    variants:dict[tuple,_TinyJit] = {}
    def run(*args, **kwargs):
      selected = jit
      if phase=='prefill' and native_replay and isinstance(jit,_TinyJit) and jit.fxn is not None:
        key = tuple((arg.device,arg.dtype,arg.max_shape) for arg in (*args,*kwargs.values()) if isinstance(arg,Tensor))
        if key not in variants:
          variants[key] = jit if not variants else TinyJit(jit.fxn,prune=jit.prune)
          if len(variants)>8: variants.pop(next(iter(variants)))
        selected = variants[key]
      for layer in layers: layer.phase = phase
      # Resolve current hooks per call so earlier wrappers cannot retain retired
      # packed weights when configuration changes before captures are created.
      current_projections=[p for block in getattr(model,'blk',[]) if (p:=getattr(block,'npu_mla_projections',None)) is not None]
      for projection in current_projections:projection.phase=phase
      try:
        with contextlib.ExitStack() as scope:
          if phase=='prefill' and shared_state is not None:
            scope.enter_context(shared_state.scope())
          elif pool is not None and not pool.active and (phase=='decode' or cpu_prefill_pool or recurrent_pool or mla_pool or expert_pool):
            scope.enter_context(pool if phase=='decode' else pool.prefill())
          if helper is not None and getattr(getattr(selected,'captured',None),'warmed',False): scope.enter_context(helper)
          result = selected(*args, **kwargs)
        if native_replay and getattr(selected,'captured',None) is not None:
          from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
          if not isinstance(selected.captured,RockchipCapturedJit):
            from tinygrad.llm.kernels.rockchip_prepare_pool import rewrite_rwkv_programs
            transform:Callable[[UOp],UOp]|None=(functools.partial(rewrite_rwkv_programs,state=shared_state)
                       if cpu_rwkv_weights and shared_state is not None and phase=='prefill' else
                       functools.partial(rewrite_preparation_programs,state=shared_state,phase=phase)
                       if shared_state is not None and ((recurrent_blocks and
                       (phase=='prefill' and ONPU_GDN_PREP_POOL or phase=='decode' and ONPU_GDN_DECODE_PREP_POOL))
                       or phase=='prefill' and ONPU_LFM_FLAT_POINTWISE) else None)
            if cpu_exact_rows or cpu_sparse_lookup:
              from tinygrad.llm.kernels.rockchip_exact_prepare import rewrite_exact_preparation
              base_transform=transform
              def exact_transform(linear:UOp) -> UOp:
                if base_transform is not None:linear=base_transform(linear)
                return rewrite_exact_preparation(linear,pool=pool,rows=cpu_exact_rows,sparse_lookup=cpu_sparse_lookup,
                                                 phase=phase,records=exact_preparation_changes)
              transform=exact_transform
            selected.captured = RockchipCapturedJit(selected.captured,cleaner,program_transform=transform)
        return result
      finally:
        for layer in layers: layer.phase = None
        for projection in current_projections:projection.phase=None
    # Parameter discovery traverses function attributes, but not weak handles.
    setattr(run,'_rockchip_base_jit',weakref.ref(jit))
    return run
  for phase, attr in [('prefill','prefill_jit'),('decode','rollout_jit'),
                      ('prefill','greedy_prefill_jit'),('prefill','greedy_uncached_jit'),('decode','greedy_rollout_jit'),
                      ('prefill','embeddings_prefill_jit'),
                      ('prefill','greedy_embeddings_prefill_jit'),('prefill','sampled_embeddings_prefill_jit'),
                      ('decode','greedy_scalar_jit')]:
    if hasattr(model,attr): setattr(model,attr,in_phase(getattr(model,attr),phase))
  attention_blocks,prepared_blocks,packed_blocks,npu_attention_blocks = 0,0,0,0
  if cpu_attention:
    from tinygrad.llm.kernels.rockchip_cpu import decode_attention, prefill_attention
    from tinygrad.llm.model import TransformerBlock
    for block in getattr(model,'blk',[]):
      config = getattr(block,'config',None)
      projection = getattr(block,'attn_q',None)
      attention_softcap = getattr(config,'attn_logit_softcap',0.0)
      if (isinstance(block,TransformerBlock) and config is not None and isinstance(projection,RockchipLinear)
          and projection.weight.device=='CPU' and 4<=config.head_dim<=512 and config.head_dim%4==0 and 1<=config.max_context<=8192
          and not config.attn_sinks
          and (attention_softcap==0 or not isinstance(attention_softcap,bool) and math.isfinite(attention_softcap) and .125<=attention_softcap<=128)):
        score_scale = getattr(block,'attention_score_scale',None)
        block.attention_logit_softcap = attention_softcap
        block.decode_attention = functools.partial(decode_attention,pool=pool,fast_softmax=cpu_fast_softmax,
                                                   score_scale=score_scale,window=config.sliding_window,softcap=attention_softcap)
        block.prefill_attention = functools.partial(prefill_attention,fast_softmax=cpu_fast_softmax,
                                                   pool=pool if cpu_prefill_pool else None,
                                                   score_scale=score_scale,window=config.sliding_window,softcap=attention_softcap)
        if (npu_prefill_attention and (config.n_heads,config.n_kv_heads,config.head_dim) in ((16,8,128),(32,4,64))
            and config.max_context in (64,128,256,512) and score_scale is None and not config.sliding_window and not attention_softcap):
          from tinygrad.llm.kernels.rockchip_causal_attention import prefill_attention as npu_attention
          block.prefill_attention=functools.partial(npu_attention,fast_softmax=cpu_fast_softmax,pool=pool)
          npu_attention_blocks+=1
        attention_blocks += 1
        if (cpu_packed_attention and score_scale is None and not attention_softcap
            and (not config.sliding_window or config.sliding_window>=config.max_context)
            and (config.n_heads,config.n_kv_heads,config.head_dim)==(9,3,64) and config.max_context>=4096):
          from tinygrad.llm.kernels.rockchip_packed_attention import PackedCPUAttention
          packed_attention:PackedCPUAttention
          if cpu_int8_qk:
            from tinygrad.llm.kernels.rockchip_dot_attention import Int8CPUAttention
            packed_attention = Int8CPUAttention(config.max_context)
          else: packed_attention = PackedCPUAttention(config.max_context,fp16_qk=cpu_fp16_qk)
          block.prefill_attention = packed_attention
          block.decode_attention = functools.partial(packed_attention.decode,pool=pool,fast_softmax=cpu_fast_softmax)
          packed_blocks += 1
        qkv_group = getattr(block,'attn_qkv',None)
        query_weight,key_weight = getattr(getattr(block,'attn_q_norm',None),'weight',None),getattr(getattr(block,'attn_k_norm',None),'weight',None)
        unnormalized = fused_qkv_without_norm and config.qk_norm==0 and query_weight is None and key_weight is None
        # Prefill keeps the original full-head-rotary prepared graph. One-token
        # decode can additionally opt in (default off) to a partial-rotary
        # prepared graph that fuses INT8 dequantize + rotary + KV-cache store.
        full_rotary_prefill = config.head_dim==config.rope_dim and (unnormalized or config.qk_norm==config.head_dim)
        decode_rotary = (bool(ONPU_DECODE_FUSED_QKV_ROPE) and config.qk_norm==0 and query_weight is None and key_weight is None
                         and 0<config.rope_dim<=config.head_dim and config.rope_dim%8==0)
        if (fused_qkv and not attention_softcap and getattr(block,'prepared_attention_compatible',True)
            and not config.attn_output_gate and not config.attn_head_gate
            and config.max_context<=4096 and isinstance(qkv_group,RockchipProjections)
            and (full_rotary_prefill or decode_rotary)
            and 8<=config.head_dim<=256 and config.head_dim%8==0 and 1<=config.n_kv_heads<=config.n_heads<=128
            and (full_rotary_prefill or decode_rotary or isinstance(query_weight,Tensor) and isinstance(key_weight,Tensor)
                 and all(w.device=='CPU' and w.dtype in (dtypes.half,dtypes.float) for w in (query_weight,key_weight)))):
          prepared_type:type[RockchipPreparedQKV] = RockchipPreparedQKV
          if (precision=='int8' and (len(qkv_group.joint.packed)==1 or len(getattr(qkv_group.joint,'k_parts',()))==1)
              and (qkv_group.joint.bias is None or qkv_group.joint.bias.device=='CPU'
                   and qkv_group.joint.bias.dtype in (dtypes.half,dtypes.float))):
            from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8PreparedQKV
            prepared_type = RockchipInt8PreparedQKV
          setattr(block,'prepared_qkv',prepared_type(qkv_group,config.n_heads,query_weight,key_weight,config.norm_eps,
                                                     rope_dim=config.rope_dim,decode=decode_rotary and not full_rotary_prefill))
          prepared_blocks += 1
  head_gates=0
  for block in getattr(model,'blk',[]):
    if hasattr(block,'cpu_head_gate'):delattr(block,'cpu_head_gate')
    if cpu_head_gate and getattr(getattr(block,'config',None),'attn_head_gate',False):
      from tinygrad.llm.kernels.rockchip_cpu import head_gate
      setattr(block,'cpu_head_gate',head_gate)
      head_gates+=1
  norms = 0
  if cpu_norm:
    for obj,name in [*((block,name) for block in getattr(model,'blk',[]) for name in ('attn_norm','ffn_norm')),(model,'output_norm')]:
      source = getattr(obj,name,None)
      if isinstance(source,nn.RMSNorm) and source.weight is not None and source.weight.device=='CPU':
        setattr(obj,name,RockchipRMSNorm(source))
        norms += 1
  packed_weights={id(tensor):tensor for layer in (*layers,*(group.joint for group in groups))
                  for tensor in (*layer.packed,*(part for parts in getattr(layer,'k_parts',()) for part in parts))}
  # Other graph families can retain local-thread kernels. Do not pin their
  # caller to one core until all large preparation work has shared dispatch.
  # Plain GELU graphs can keep elementwise QKV preparation (e.g. score softcap).
  # Their threaded attention/norm/dequant/softcap helpers still use this pool.
  gelu_pool = (attention_blocks>0 and gelu_ffn_blocks==attention_blocks==len(getattr(model,'blk',[]))
               and static_projections==0 and all(type(b) is TransformerBlock for b in model.blk))
  erf_pool = erf_pool_eligible(model,prefill,decode,attention_blocks,erf_gelu_blocks,erf_gelu_ffn_blocks,
                               erf_gelu_decode_blocks,head_gates,static_projections)
  if (ONPU_SHARED_HOST_POOL and pool is not None and precision=='int8' and prefill!='off'
      and attention_blocks>0
      and ((prepared_blocks==ffn_blocks==attention_blocks
            and (npu_attention_blocks==attention_blocks or cpu_prefill_pool and swiglu_ffn_blocks==attention_blocks))
           or cpu_prefill_pool and (gelu_pool or erf_pool))
      and norms>=2*attention_blocks and not (recurrent_blocks or expert_layers or npu_expert_blocks or mla_projections)):
    shared_state = shared_host_pool(pool)
  if ONPU_SHARED_HOST_POOL and pool is not None and cpu_recurrent_scan and cpu_prefill_pool and precision=='int8' and prefill==decode=='on':
    from tinygrad.llm.model import GatedDeltaNetBlock, TransformerBlock
    blocks=getattr(model,'blk',[])
    # These recurrent graphs use shared-program helpers or the same pool for
    # every threaded operation. Keep KDA, experts and other graph families out.
    if (recurrent_blocks and not (expert_layers or npu_expert_blocks or mla_projections)
        and ffn_blocks==len(blocks) and norms>=2*len(blocks) and getattr(getattr(model,'config',None),'max_context',0)==512
        and all(type(b) in (GatedDeltaNetBlock,TransformerBlock) and not hasattr(b,'ssm_g_a') for b in blocks)):
      shared_state = shared_host_pool(pool)
  if ONPU_SHARED_HOST_POOL and pool is not None and cpu_prefill_pool and precision=='int8' and prefill==decode=='on':
    from tinygrad.llm.model import ShortConvBlock, ShortConvolution, TransformerBlock
    blocks=getattr(model,'blk',[])
    # Short convolution uses single-threaded elementwise kernels. Its linear
    # projections and every threaded attention/FFN helper must be prepared.
    if (0<attention_blocks<len(blocks) and prepared_blocks==attention_blocks
        and ffn_blocks==swiglu_ffn_blocks==len(blocks) and norms>=2*len(blocks)
        and getattr(getattr(model,'config',None),'max_context',0)==512
        and not (recurrent_blocks or expert_layers or npu_expert_blocks or mla_projections)
        and all(type(b) in (ShortConvBlock,TransformerBlock) for b in blocks)
        and all(type(b.shortconv) is ShortConvolution and all(isinstance(p,RockchipInt8Linear) and p.prefill=='on'
                    for p in (b.shortconv.in_proj,b.shortconv.out_proj)) for b in blocks if type(b) is ShortConvBlock)):
      shared_state = shared_host_pool(pool)
  if (ONPU_SHARED_HOST_POOL and cpu_rwkv_weights and pool is not None and cpu_prefill_pool and precision=='int8'
      and prefill==decode=='on' and static_projections==0):
    from tinygrad.llm.rwkv import RWKV7Block
    blocks=getattr(model,'blk',[])
    if (blocks and getattr(getattr(model,'config',None),'max_context',0)==512
        and all(type(b) is RWKV7Block and isinstance(hook:=getattr(b,'rwkv_scan',None),functools.partial)
                and hook.keywords.get('pool') is pool for b in blocks)):
      shared_state=shared_host_pool(pool)
  if (ONPU_SHARED_HOST_POOL and pool is not None and cpu_prefill_pool and precision=='int8'
      and gemma4_pool_eligible(model,prefill,decode,attention_blocks,gelu_ffn_blocks if fused_gelu_sdk_prefill else 0,norms,cpu_exact_rows)):
    shared_state=shared_host_pool(pool)
  setattr(model,'_shared_host_pool',weakref.ref(shared_state) if shared_state is not None else None)
  return {'layers':len(layers),'scaled_layers':len(used_scales),'fused_groups':len(groups),'tile_columns':tile_columns,
          'cpu_shortconv_weight_tensors':shortconv_weights,'cpu_rwkv_weight_tensors':rwkv_weights,
          'fused_recurrent_projection_blocks':sum(isinstance(getattr(b,'recurrent_projections',None),RockchipProjections)
            for b in getattr(model,'blk',[])),
          'shared_host_pool':shared_state is not None,'shared_host_pool_requested':bool(ONPU_SHARED_HOST_POOL),
          'cpu_preparation_pool_requested':bool(ONPU_GDN_PREP_POOL and recurrent_blocks and shared_state is not None),
          'cpu_decode_preparation_pool_requested':bool(ONPU_GDN_DECODE_PREP_POOL and recurrent_blocks and shared_state is not None),
          'cpu_preparation_programs':shared_state.preparation_changes if shared_state is not None else [],
          'cpu_static_projections':static_projections,
          'cpu_static_row_block':getenv('ONPU_STATIC_GEMM_ROWS',4) if static_projections else None,
          'cpu_exact_rows':cpu_exact_rows,'cpu_sparse_lookup':cpu_sparse_lookup,
          'cpu_exact_preparation_sources':exact_preparation_changes,
          'cpu_mla_weights':mla_weights,
          'cpu_mla_attention_blocks':mla_attention,
          'cpu_mla_projection_blocks':cpu_mla_projection_blocks,
          'cpu_mla_pack_once':cpu_mla_pack_once,
          'cpu_mla_decode_pool':cpu_mla_decode_pool,
          'cpu_mla_projection_packed_weight_bytes':sum(getattr(block,'cpu_mla_projections').packed_bytes
            for block in getattr(model,'blk',[]) if getattr(block,'cpu_mla_projections',None) is not None),
          'cpu_expert_layers':expert_layers,
          'npu_expert_blocks':npu_expert_blocks,'npu_expert_packed_weight_bytes':npu_expert_bytes,
          'npu_mla_projection_blocks':len(mla_projections),'npu_mla_prefill':npu_mla_prefill,'npu_mla_decode':npu_mla_decode,
          'npu_mla_projection_precision':npu_mla_precision if mla_projections else None,
          'npu_mla_projection_selection':npu_mla_projections if mla_projections else None,
          'npu_mla_packed_weight_bytes':sum(p.packed_bytes for p in mla_projections),
          'cpu_softcap':softcap_enabled,
          'cpu_gelu_blocks':gelu_blocks,
          'cpu_erf_gelu_blocks':erf_gelu_blocks,
          'fused_gelu_ffn_blocks':gelu_ffn_blocks,
          'fused_gelu_sdk_prefill_blocks':gelu_ffn_blocks if fused_gelu_sdk_prefill else 0,
          'fused_gelu_decode_blocks':gelu_ffn_blocks if fused_gelu_decode else 0,
          'fused_erf_gelu_ffn_blocks':erf_gelu_ffn_blocks,
          'fused_erf_gelu_decode_blocks':erf_gelu_decode_blocks,
          'fused_swiglu_ffn_blocks':swiglu_ffn_blocks,
          'fused_swiglu_decode_blocks':swiglu_decode_blocks,
          'share_projection_weights':share_projection_weights,
          'shared_projection_layers':sum(layer.shared_projection is not None for layer in layers),
          'packed_weight_bytes':sum(tensor.nbytes() for tensor in packed_weights.values())+
                                sum(p.packed_bytes for p in mla_projections)+npu_expert_bytes,
          'k_split_layers':sum(bool(getattr(layer,'k_parts',())) for layer in layers),
          'grouped_split_layers':sum(getattr(layer,'grouped_weight',None) is not None for layer in layers),
          'batch_prefill':batch_prefill,
          'cpu_attention_blocks':attention_blocks,'cpu_head_gate_blocks':head_gates,
          'npu_prefill_attention_blocks':npu_attention_blocks,
          'cpu_recurrent_scan':recurrent_blocks>0,'cpu_recurrent_blocks':recurrent_blocks,
          'cpu_gdn_direct_decode':bool(recurrent_blocks and ONPU_GDN_DIRECT_DECODE),
          'cpu_recurrent_pool':recurrent_blocks>0 and recurrent_pool and pool is not None,
          'packed_attention_blocks':packed_blocks,
          'cpu_fp16_qk':cpu_fp16_qk and packed_blocks>0,
          'cpu_int8_qk':cpu_int8_qk and packed_blocks>0,
          'precision':precision,
          'cpu_pool':pool is not None,'host_event_wait':bool(pool and getattr(pool,'event_wait',False)),
          'cpu_prefill_pool':cpu_prefill_pool and pool is not None,
          'cpu_fast_softmax':cpu_fast_softmax,
          'cache_cleaning':cleaner is not None,'wake_helper':helper is not None,
          'native_replay':native_replay,'fused_ffn_blocks':ffn_blocks,'prepared_qkv_blocks':prepared_blocks,'cpu_norms':norms,
          'decode_layers':sum(l.decode=='on' for l in layers),
          'prefill_layers':sum(l.prefill=='on' for l in layers),'decisions':decisions}
