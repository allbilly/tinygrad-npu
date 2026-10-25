"""Staged SmolVLM vision inference: prepared NPU projections and GPU attention.

Stage boundaries are real execution boundaries. No numerical fusion barriers or
changes to the model's attention mask are needed. This is an optional encoder;
the normal CPU/OpenCL model remains the reference and fallback.
"""
from __future__ import annotations
import ctypes, json, time
from pathlib import Path
from typing import Callable
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip import RockchipLinear, RockchipProjections, pack_weight, packed_matmul
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.llm.vision import SmolVLM, VisionBlock, pixel_shuffle

def load_input_scales(path:str|Path, checkpoint_sha256:str) -> dict[str,Tensor]:
  data = json.loads(Path(path).read_text())
  if data['checkpoint_sha256']!=checkpoint_sha256: raise ValueError('Input channel calibration does not match the checkpoint')
  result = {}
  for name,values in data['scales'].items():
    scale = Tensor(values,device='CPU',dtype=dtypes.float)
    names = [name[:-4]+'.'+part for part in ('q_proj','k_proj','v_proj')] if name.endswith('.qkv') else [name]
    for key in names: result[key] = scale
  return result

class _Stage:
  __slots__ = ('jit','native','cleaner')
  def __init__(self, function:Callable, native:bool, cleaner:int|None):
    self.jit, self.native, self.cleaner = TinyJit(function), native, cleaner

  def __call__(self, *args):
    # The large encoder's generic suballocation path is not qualified yet.
    with Context(NO_MEMORY_PLANNER=1): result = self.jit(*args)
    if self.native and self.jit.captured is not None and not isinstance(self.jit.captured,RockchipCapturedJit):
      self.jit.captured = RockchipCapturedJit(self.jit.captured,self.cleaner)
    return result

class StagedVision:
  def __init__(self, model:SmolVLM, attention_device:str='CL', native:bool=True, fuse_qkv:bool=True, profile:bool=False,
               npu_connector:bool=False, fused_mlp:bool=False, gpu_softmax:bool=False, prepared_mlp:bool=False):
    if attention_device not in ('CPU','CL','ROCKCHIP'): raise ValueError('Staged vision attention requires CPU, OpenCL or Rockchip')
    if gpu_softmax and attention_device!='ROCKCHIP': raise ValueError('Shared GPU softmax requires Rockchip attention')
    if prepared_mlp and attention_device!='ROCKCHIP': raise ValueError('Prepared vision MLP requires Rockchip attention')
    self.gpu_softmax = gpu_softmax
    if model.vision.embeddings.patch_embedding.weight.device!='CPU': raise ValueError('Staged vision requires CPU projection sources')
    self.attention_device = attention_device
    self.npu_connector = npu_connector
    self.fused_mlp = 0
    if fused_mlp:
      from tinygrad.llm.kernels.rockchip_vision_mlp import FP16VisionMLP
      from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
      for block in model.vision.layers:
        fc1,fc2 = getattr(block.mlp,'fc1',None),getattr(block.mlp,'fc2',None)
        if (isinstance(fc1,RockchipLinear) and isinstance(fc2,RockchipLinear)
            and not isinstance(fc1,RockchipInt8Linear) and not isinstance(fc2,RockchipInt8Linear)):
          setattr(block,'mlp',FP16VisionMLP(fc1,fc2))
          self.fused_mlp += 1
    self.profile = profile
    self.timings:dict[str,float] = {}
    self.attention_timings:dict[str,float] = {}
    cleaner = None
    if native:
      from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
      cleaner = ctypes.cast(cache_cleaner().clean_to,ctypes.c_void_p).value
    self.embedding = _Stage(lambda pixels:model.vision.embeddings(pixels).float().realize(),native,cleaner)
    self.before, self.after = [],[]
    self.prepared_mlps = []
    self.groups:list[RockchipProjections] = []
    for block in model.vision.layers:
      members = (block.self_attn.q_proj,block.self_attn.k_proj,block.self_attn.v_proj)
      npu_members = tuple(layer for layer in members if isinstance(layer,RockchipLinear) and layer.prefill=='on')
      group = None
      scales = tuple(getattr(layer,'input_scale',None) for layer in members)
      if fuse_qkv and len(npu_members)==3 and all(scale is scales[0] for scale in scales):
        group = RockchipProjections(npu_members,4096)
        self.groups.append(group)
      def before(x:Tensor, block:VisionBlock=block, group=group):
        h = block.layer_norm1(x)
        projected = group(h) if group is not None else tuple(projection(h) for projection in
          (block.self_attn.q_proj,block.self_attn.k_proj,block.self_attn.v_proj))
        b,t,_ = x.shape
        result = tuple(value.reshape(b,t,block.self_attn.n_heads,block.self_attn.head_dim).transpose(1,2).contiguous()
                       for value in projected)
        Tensor.realize(*result)
        return result
      def after(x:Tensor, attended:Tensor, block:VisionBlock=block, prepared:bool=prepared_mlp):
        x = x + block.self_attn.out_proj(attended.transpose(1,2).reshape(*x.shape))
        if prepared:
          normalized = block.layer_norm2(x)
          Tensor.realize(x,normalized)
          return x,normalized
        return (x + block.mlp(block.layer_norm2(x))).realize()
      self.before.append(_Stage(before,native,cleaner))
      self.after.append(_Stage(after,native,cleaner))
      if prepared_mlp:
        from tinygrad.llm.kernels.rockchip_vision_prepared import PreparedVisionMLP
        fc1,fc2 = block.mlp.fc1,block.mlp.fc2
        if not isinstance(fc1,RockchipLinear) or not isinstance(fc2,RockchipLinear): raise ValueError('Prepared MLP needs NPU projections')
        self.prepared_mlps.append(PreparedVisionMLP(fc1,fc2))
    self.attention:Callable
    if attention_device=='ROCKCHIP':
      from tinygrad.llm.kernels.rockchip_attention import NPUVisionAttention
      from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
      if model.vision_config.dim//model.vision_config.n_heads!=64: raise ValueError('Prepared NPU attention requires 64-channel heads')
      if any(isinstance(member,RockchipInt8Linear) for block in model.vision.layers for member in
             (block.self_attn.q_proj,block.self_attn.out_proj,block.mlp.fc1,block.mlp.fc2)):
        raise ValueError('Prepared NPU attention currently requires FP16 vision projections')
      self.attention = NPUVisionAttention(model.vision_config.n_heads,(model.vision_config.image_size//model.vision_config.patch_size)**2,
                                          reset_context=False,profile=profile,gpu_softmax=gpu_softmax)
    else:
      def attend(q:Tensor, k:Tensor, v:Tensor):
        q,k,v = (value.to(attention_device) for value in (q,k,v))
        return q.scaled_dot_product_attention(k,v).to('CPU').contiguous().realize()
      self.attention = _Stage(attend,False,None)
    if npu_connector:
      columns,depth = model.connector.weight.shape
      if not isinstance(columns,int) or not isinstance(depth,int) or depth!=12288 or columns%32 or not 0<columns<=4096:
        raise ValueError('NPU connector requires qualified split-K geometry')
      weights = [pack_weight(model.connector.weight[:,first:first+4096]).realize() for first in range(0,12288,4096)]
      def finish(x:Tensor):
        features = pixel_shuffle(model.vision.post_layernorm(x),model.vision_config.scale_factor).contiguous()
        dense = features.reshape(-1,12288)
        outputs = []
        for row in range(0,dense.shape[0],16):
          parts = [packed_matmul(dense[row:row+16,first:first+4096],weight,columns) for first,weight in zip(range(0,12288,4096),weights)]
          outputs.append(parts[0]+parts[1]+parts[2])
        return Tensor.cat(*outputs,dim=0).reshape(*features.shape[:-1],columns).realize()
      self.finish = _Stage(finish,native,cleaner)
    else:
      connector_device = 'CL' if attention_device=='ROCKCHIP' else attention_device
      connector = model.connector.weight.to(connector_device).realize()
      def finish_gpu(x:Tensor):
        features = pixel_shuffle(model.vision.post_layernorm(x),model.vision_config.scale_factor).contiguous().to(connector_device)
        return features.linear(connector.T).float().to('CPU').contiguous().realize()
      self.finish = _Stage(finish_gpu,False,None)

  def __call__(self, pixels:Tensor) -> Tensor:
    if pixels.device!='CPU': raise ValueError('Staged vision requires CPU pixels')
    def run(name,stage,*args):
      if not self.profile: return stage(*args)
      start = time.perf_counter()
      result = stage(*args)
      self.timings[name] = (time.perf_counter()-start)*1000
      return result
    x = run('embedding',self.embedding,pixels)
    self.attention_timings = {}
    for i,(before,after) in enumerate(zip(self.before,self.after)):
      q,k,v = run(f'before-{i}',before,x)
      attended = run(f'attention-{i}',self.attention,q,k,v)
      if self.profile and hasattr(self.attention,'phase_timings'):
        for name,value in self.attention.phase_timings().items(): self.attention_timings[name] = self.attention_timings.get(name,0)+value
      result = run(f'after-{i}',after,x,attended)
      if self.prepared_mlps:
        residual,normalized = result
        x = run(f'mlp-{i}',self.prepared_mlps[i],normalized,residual)
      else:x=result
    return run('finish',self.finish,x)

  def statistics(self) -> dict:
    stages = [('embedding',self.embedding),*((f'before-{i}',stage) for i,stage in enumerate(self.before)),
              *((f'after-{i}',stage) for i,stage in enumerate(self.after)),('finish',self.finish)]
    kernels:dict[str,float] = {}
    for _,stage in stages:
      captured = stage.jit.captured
      for label,ns in getattr(getattr(captured,'native',None),'profile_ns',{}).items():
        kernels[label] = kernels.get(label,0)+ns/max(getattr(captured,'native_runs',0),1)/1e6
    return {'attention_device':self.attention_device,'gpu_softmax':self.gpu_softmax,'npu_connector':self.npu_connector,'fused_mlp':self.fused_mlp,
      'prepared_mlp':len(self.prepared_mlps),
      'fused_qkv':len(self.groups),'kernels_ms':kernels,'attention_ms':self.attention_timings,
      'stages':{name:{'native_runs':getattr(stage.jit.captured,'native_runs',0),'reason':getattr(stage.jit.captured,'reason',None)}
                for name,stage in stages}}
