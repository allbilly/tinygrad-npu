"""Prepared Mali decoder operations, retaining ordinary OpenCL fallbacks."""
from __future__ import annotations
import copy, math
from pathlib import Path
from tinygrad import Device, Tensor, nn, function, dtypes
from tinygrad.llm.model import GatedDeltaNetBlock, TransformerBlock
from tinygrad.llm.rwkv import RWKV7Block


@function(precompile=True)
def _linear(x:Tensor, weight:Tensor) -> Tensor:
  return x @ weight.T


class MaliLinear(nn.Linear):
  def __init__(self, source, prepare_gemm:bool=True, q8_weight:Tensor|None=None, *, consistent_decode:bool=False):
    if type(consistent_decode) is not bool:raise TypeError('consistent_decode must be a bool')
    if consistent_decode and q8_weight is not None:raise ValueError('Consistent decode currently requires dense projections')
    self.source, self.weight, self.bias = source, source.weight, source.bias
    columns,depth = self.weight.shape
    assert isinstance(columns,int) and isinstance(depth,int)
    self.out_features, self.in_features = columns,depth
    from tinygrad.llm.kernels.mali_linear import pack_weight
    self.packed_weight = (pack_weight(self.weight).realize() if prepare_gemm and isinstance(self.weight.device,str)
                          and self.weight.device.split(':')[0]=='CL' else None)
    self.packed_source = self.weight.uop
    self.q8_weight, self.q8_source = q8_weight, self.weight.uop
    self.consistent_decode = consistent_decode
    self.bounded_half_prefill = False

  def __call__(self, x:Tensor) -> Tensor:
    if (self.consistent_decode and x.dtype is dtypes.half and self.weight.dtype is dtypes.half
        and isinstance(x.device,str) and x.device.split(':')[0]=='CL' and x.device==self.weight.device
        and self.in_features%4==0 and 1<=math.prod(x.max_shape[:-1])<=8):
      from tinygrad.llm.kernels.mali_linear import gemv_half_rows
      result = gemv_half_rows(x,self.weight)
      return result if self.bias is None else result+self.bias
    if self.bounded_half_prefill and self.bias is None and x.dtype is dtypes.half:
      from tinygrad.llm.kernels.mali_half_prefill import bounded_linear
      if (bounded:=bounded_linear(x,self.weight)) is not None:return bounded
    if (x.dtype is dtypes.float and isinstance(x.device,str) and x.device.split(':')[0]=='CL'
        and self.weight.dtype in (dtypes.half,dtypes.float) and self.in_features%4==0):
      from tinygrad.llm.kernels.mali_linear import gemm, gemv, gemv_q8, gemv_rows
      packed = self.packed_weight if self.weight.uop is self.packed_source else None
      if math.prod(x.max_shape[:-1])==1:
        result = (gemv_q8(x,self.q8_weight,self.out_features,self.in_features)
                  if self.q8_weight is not None and self.weight.uop is self.q8_source else gemv(x,self.weight))
      elif self.consistent_decode and math.prod(x.max_shape[:-1])<=8:result = gemv_rows(x,self.weight)
      else: result = gemm(x,self.weight,packed)
      return result if self.bias is None else result+self.bias
    if all(isinstance(size,int) for size in x.shape): return self.source(x)
    # Fixed row capacity allows the normal matrix tiler to reuse weights. The
    # caller still observes the real row count and stores only valid KV entries.
    padded = x.pad_to(x.max_shape).contiguous()
    result = _linear(padded,self.weight)
    if self.bias is not None: result = result + self.bias
    return result.shrink(tuple((0,size) for size in (*x.shape[:-1],result.shape[-1])))


class MaliJoint:
  def __init__(self, layers:list[MaliLinear], prepare_gemm:bool=True):
    self.layers, self.sizes = layers,tuple(layer.out_features for layer in layers)
    source = copy.copy(layers[0].source)
    source.weight = layers[0].weight.cat(*(layer.weight for layer in layers[1:]),dim=0).contiguous().realize()
    biases = [layer.bias for layer in layers if layer.bias is not None]
    if biases:
      values = [layer.bias if layer.bias is not None else Tensor.zeros(size,device=biases[0].device,dtype=biases[0].dtype)
                for layer,size in zip(layers,self.sizes)]
      source.bias = values[0].cat(*values[1:],dim=0).contiguous().realize()
    else: source.bias = None
    offset = 0
    for layer,size in zip(layers,self.sizes):
      layer.weight.replace(source.weight[offset:offset+size])
      if layer.bias is not None:layer.bias.replace(source.bias[offset:offset+size])
      layer.packed_weight = None
      layer.packed_source = layer.q8_source = layer.weight.uop
      offset += size
    self.signature = tuple((layer.weight.uop,layer.bias.uop if layer.bias is not None else None) for layer in layers)
    self.linear = MaliLinear(source,prepare_gemm=prepare_gemm)

  def __call__(self, x:Tensor) -> tuple[Tensor,...]:
    if any(layer.weight.uop is not weight or (layer.bias.uop if layer.bias is not None else None) is not bias
           for layer,(weight,bias) in zip(self.layers,self.signature)):
      return tuple(layer(x) for layer in self.layers)
    result,offset,outputs = self.linear(x),0,[]
    for size in self.sizes:
      outputs.append(result[...,offset:offset+size])
      offset += size
    return tuple(outputs)


class MaliEmbedding(nn.Embedding):
  def __init__(self, source, prefill_only:bool=False):
    self.source,self.weight,self.prefill_only = source,source.weight,prefill_only

  def __call__(self, ids:Tensor) -> Tensor:
    if self.prefill_only and math.prod(ids.max_shape)==1:return self.source(ids)
    if ids.dtype not in (dtypes.int,dtypes.int64) or self.weight.dtype not in (dtypes.half,dtypes.float): return self.source(ids)
    from tinygrad.llm.kernels.mali_linear import embedding
    return embedding(ids,self.weight)


def _configure_output_packing(output, enabled:bool) -> tuple[bool,int]:
  # Preserve wrapper scaling and the existing MaliLinear object borrowed by a
  # draft. An unchanged weight reuses exactly one packed allocation.
  linear = output if isinstance(output,MaliLinear) else getattr(output,'source',None)
  if not isinstance(linear,MaliLinear):
    if enabled:raise ValueError('Packed output requires a supported linear output projection')
    return False,0
  if enabled:
    if (not isinstance(linear.weight.device,str) or linear.weight.device.split(':')[0]!='CL'
        or linear.weight.dtype not in (dtypes.half,dtypes.float) or linear.in_features%4):
      raise ValueError('Packed output requires an aligned FP16/FP32 Mali OpenCL projection')
    if linear.packed_weight is None or linear.packed_source is not linear.weight.uop:
      # Release a stale copy before allocating its replacement. Existing captured
      # graphs must be closed before changing configuration or checkpoint weights.
      linear.packed_weight = None
      from tinygrad.llm.kernels.mali_linear import pack_weight
      linear.packed_weight = pack_weight(linear.weight).realize()
      linear.packed_source = linear.weight.uop
  else:linear.packed_weight = None
  active = linear.packed_weight is not None and linear.packed_source is linear.weight.uop
  return active,int(linear.packed_weight.nbytes()) if active and linear.packed_weight is not None else 0


def _gdn_gate_up_candidates(model) -> list[tuple[GatedDeltaNetBlock,str]]:
  candidates:list[tuple[GatedDeltaNetBlock,str]] = []
  for block in getattr(model,'blk',[]):
    if not isinstance(block,GatedDeltaNetBlock):continue
    if not 1<=block.head_k_dim<=256 or not 1<=block.head_v_dim<=256:
      raise ValueError('Packed GDN gate/up requires a qualified recurrent scan block')
    for name in ('ffn_gate','ffn_up'):
      layer = getattr(block,name,None)
      if (not isinstance(layer,nn.Linear) or layer.weight.shape!=(6144,2048) or layer.weight.device!='CL'
          or layer.weight.dtype is not dtypes.half or layer.bias is not None):
        raise ValueError('Packed GDN gate/up requires dense FP16 CL 6144x2048 projections')
      candidates.append((block,name))
  if not candidates:raise ValueError('Packed GDN gate/up requires eligible GatedDeltaNet blocks')
  return candidates


def _configure_gdn_gate_up_packing(candidates:list[tuple[GatedDeltaNetBlock,str]]) -> tuple[int,int]:
  # The caller closes all captured users and admits combined body/head RAM first.
  # Reuse the output-packing ownership helper; never replace the original linear,
  # weight Tensor or weight UOp, and do not construct another GPU rowmajor copy.
  rebuilt:list[MaliLinear] = []
  count,total = 0,0
  try:
    for block,name in candidates:
      layer = getattr(block,name)
      if not isinstance(layer,MaliLinear):raise ValueError('Packed GDN gate/up requires prepared MaliLinear projections')
      if layer.packed_weight is None or layer.packed_source is not layer.weight.uop:rebuilt.append(layer)
      active,size = _configure_output_packing(layer,True)
      count,total = count+int(active),total+size
  except BaseException:
    # Preserve already-valid borrowed copies; release any copies built by this
    # unsuccessful request. A stale copy is deliberately not restored.
    for layer in rebuilt:layer.packed_weight = None
    raise
  return count,total


def configure(model, fused_attention:bool=True, capacities:tuple[int,...]=(), *, fuse_projections:bool=False,
              q8_head_checkpoint:Path|str|None=None, pack_weights:bool=True, recurrent_scan:bool=False,
              q8_projections_checkpoint:Path|str|None=None, pack_output:bool=False, pack_gdn_gate_up:bool=False,
              consistent_decode:bool=False, native_state_snapshots:bool=False, gated_attention:bool=False,
              bounded_half_prefill:bool=False) -> dict:
  """Configure before graph capture; rebuild captured users after reconfiguration.

  pack_output opts into one extra GPU packed output allocation, independently of
  body pack_weights. pack_gdn_gate_up separately packs only measured FP16
  GatedDeltaNet FFN gate/up projections; its default is off. The caller must admit
  combined head/body memory after model residency and before this configuration.
  Shape/model speed and numerical qualification remain separate.
  consistent_decode opts dense blocks of at most eight rows into the scalar
  GEMV reduction order. It requires separate end-to-end numerical qualification.
  native_state_snapshots opts into bit-preserving packed copies of supported
  recurrent buffers; other layouts keep the ordinary snapshot implementation.
  gated_attention opts output-gated blocks into bounded cached attention; the
  output gate stays in the block. Qualify this layout before serving it.
  consistent_decode also aligns cached attention for blocks of at most eight rows.
  bounded_half_prefill bounds qualified FP16 projection work to valid rows;
  source fingerprints keep different compiler layouts on the ordinary path.
  """
  if type(pack_gdn_gate_up) is not bool:raise TypeError('pack_gdn_gate_up must be a bool')
  if type(consistent_decode) is not bool:raise TypeError('consistent_decode must be a bool')
  if type(native_state_snapshots) is not bool:raise TypeError('native_state_snapshots must be a bool')
  if type(gated_attention) is not bool:raise TypeError('gated_attention must be a bool')
  if type(bounded_half_prefill) is not bool:raise TypeError('bounded_half_prefill must be a bool')
  if gated_attention and not fused_attention:raise ValueError('Gated attention requires fused_attention')
  if consistent_decode and (q8_head_checkpoint is not None or q8_projections_checkpoint is not None):
    raise ValueError('Consistent decode currently requires dense projections')
  weight = getattr(getattr(model,'token_embd',None),'weight',None)
  if weight is None and isinstance(model,TransformerBlock): weight = model.attn_q.weight
  if weight is None or not isinstance(weight.device,str) or weight.device.split(':')[0]!='CL':
    if q8_projections_checkpoint is not None:raise ValueError('Native Q8 body projections require a Mali OpenCL model')
    if pack_output:raise ValueError('Packed output requires a Mali OpenCL model')
    if pack_gdn_gate_up:raise ValueError('Packed GDN gate/up requires a Mali OpenCL model')
    if consistent_decode:raise ValueError('Consistent decode requires a Mali OpenCL model')
    if native_state_snapshots:raise ValueError('Native snapshots require a Mali OpenCL model')
    if gated_attention:raise ValueError('Gated attention requires a Mali OpenCL model')
    if bounded_half_prefill:raise ValueError('Bounded half prefill requires a Mali OpenCL model')
    return {'linear_layers':0}
  if not getattr(Device[weight.device],'device_name','').startswith('Mali'):
    if q8_projections_checkpoint is not None:raise ValueError('Native Q8 body projections require a Mali OpenCL model')
    if pack_output:raise ValueError('Packed output requires a Mali OpenCL model')
    if pack_gdn_gate_up:raise ValueError('Packed GDN gate/up requires a Mali OpenCL model')
    if consistent_decode:raise ValueError('Consistent decode requires a Mali OpenCL model')
    if native_state_snapshots:raise ValueError('Native snapshots require a Mali OpenCL model')
    if gated_attention:raise ValueError('Gated attention requires a Mali OpenCL model')
    if bounded_half_prefill:raise ValueError('Bounded half prefill requires a Mali OpenCL model')
    return {'linear_layers':0}
  from tinygrad.llm.kernels.mali_q8 import MaliQ8Linear
  if consistent_decode:
    for module in (model,*getattr(model,'blk',[])):
      for layer in vars(module).values():
        linear = layer if isinstance(layer,(MaliLinear,MaliQ8Linear)) else getattr(layer,'source',None)
        if isinstance(linear,MaliQ8Linear) or (isinstance(linear,MaliLinear) and linear.q8_weight is not None):
          raise ValueError('Consistent decode currently requires dense projections')
  from tinygrad.llm.kernels.mali_recurrent import gated_delta_scan, supported as recurrent_supported
  if pack_gdn_gate_up and q8_projections_checkpoint is not None:
    raise ValueError('Packed GDN gate/up cannot combine with native Q8 body replacement')
  if pack_gdn_gate_up and (not recurrent_scan or not recurrent_supported(weight.device)):
    raise ValueError('Packed GDN gate/up requires qualified recurrent_scan=True')
  gdn_candidates = _gdn_gate_up_candidates(model) if pack_gdn_gate_up else []
  q8_skipped:list[dict] = []
  if q8_projections_checkpoint is not None:
    if fuse_projections:raise ValueError('Native Q8 body projections cannot yet combine with projection fusion')
    from tinygrad.llm.kernels.mali_q8_checkpoint import Q8Checkpoint, projection_plan
    checkpoint = Q8Checkpoint(q8_projections_checkpoint)
    plan,q8_skipped = projection_plan(model,checkpoint,recurrent=recurrent_scan and recurrent_supported(weight.device))
    for block,name,key in plan:
      source = getattr(block,name)
      columns,depth = (source.out_features,source.in_features) if isinstance(source,MaliQ8Linear) else source.weight.shape
      raw = checkpoint.weight(key,(columns,depth),weight.device)
      setattr(block,name,MaliQ8Linear(raw,columns,depth,source.bias))
      del source
      # Q8 and FP16 allocations have different sizes, so allocator caching can
      # otherwise retain every replaced FP16 buffer until model destruction.
      Device[weight.device].allocator.free_cache()
  context = getattr(model,'max_context',model.config.max_context)
  chunk = min(context,64 if context>=16384 else 128)
  from tinygrad.llm.kernels.mali_attention import SharedAttention
  device = Device[weight.device]
  subgroups = (getattr(device,'device_name','')=='Mali-LODX r0p0' and getattr(device,'driver_version','')=='2.1'
               and 'cl_khr_subgroups' in getattr(device,'device_exts',()))
  attention = SharedAttention(subgroups=subgroups,consistent_decode=consistent_decode)
  half_prepared={'prepared_kernels':0}
  if bounded_half_prefill:
    from tinygrad.llm.kernels.mali_half_prefill import prepare as prepare_half_prefill
    half_prepared=prepare_half_prefill(weight.device)
  count,masks,attentions,gated_attentions = 0,0,0,0
  q8_head = None
  if q8_head_checkpoint is not None:
    from tinygrad.llm.gguf import gguf_parse
    _,entries = gguf_parse(q8_head_checkpoint)
    entry = entries.get('output.weight',entries.get('token_embd.weight'))
    if entry is None or entry[2]!=8 or tuple(model.output.weight.shape)!=entry[1] or entry[1][1]%32 or model.output.weight.dtype is not dtypes.half:
      raise ValueError('Native Q8 head requires matching Q8_0 GGUF output weights and FP16 stored operands')
    q8_head = entry[0].to(weight.device).contiguous().realize()
  qkv_count,gate_count,recurrent_count = 0,0,0
  recurrent_names = ('attn_qkv','ssm_alpha','ssm_beta','ssm_out','ssm_g_a','ssm_g_b','ssm_f_a','ssm_f_b')
  for module in (model,*getattr(model,'blk',[])):
    # RWKV's two-output, mutable-state body currently needs the ordinary graph
    # scheduler. Prepared body projections fail assignment-order verification.
    if isinstance(module,GatedDeltaNetBlock):
      if (not recurrent_scan or not recurrent_supported(weight.device)
          or not 1<=module.head_k_dim<=256 or not 1<=module.head_v_dim<=256):
        module.__dict__.pop('gated_delta_scan',None)
        for name in ('ffn_gate','ffn_up'):
          if isinstance(layer:=getattr(module,name,None),MaliLinear):layer.packed_weight = None
        for name in ('ffn_gate','ffn_up','ffn_down','attn_gate',*recurrent_names):
          if isinstance(layer:=getattr(module,name,None),MaliLinear):layer.bounded_half_prefill = False
        continue
      setattr(module,'gated_delta_scan',gated_delta_scan)
      recurrent_count += 1
    elif not isinstance(module,TransformerBlock) and module is not model: continue
    if isinstance(module,TransformerBlock):
      module.bounded_mask = True
      masks += 1
      config = module.config
      if getattr(module,'_mali_gated_attention',False):
        for name in ('prefill_attention','decode_attention'):
          if isinstance(getattr(module,name,None),SharedAttention):setattr(module,name,None)
        setattr(module,'_mali_gated_attention',False)
      if (fused_attention and getattr(module,'prepared_attention_compatible',True) and config.head_dim in (16,32,64,128,256)
          and 1<=config.n_heads//config.n_kv_heads<=8 and not config.sliding_window
          and not config.attn_sinks and (not config.attn_output_gate or gated_attention)
          and not config.attn_head_gate and not config.attn_logit_softcap):
        extra=(min(context,8),min(context,128)) if gated_attention and consistent_decode else ()
        for capacity in {1,chunk,*capacities,*extra}:
          attention.prepare(config.n_heads,config.head_dim,config.max_context,capacity,weight.device)
        module.prefill_attention = module.decode_attention = attention
        if config.attn_output_gate:
          setattr(module,'_mali_gated_attention',True)
          gated_attentions+=1
        attentions += 1
    # Recurrent blocks use the tiled row-major GEMM. Their packed copies can
    # otherwise duplicate most model weights, including FP16-only fallbacks.
    module_pack = pack_weights and not isinstance(module,GatedDeltaNetBlock)
    for name in ('attn_q','attn_k','attn_v','attn_gate','attn_output','ffn_gate','ffn_up','ffn_down','output')+(
        recurrent_names if isinstance(module,GatedDeltaNetBlock) else ()):
      source = getattr(module,name,None)
      keep_pack = (module is model and name=='output' and pack_output) or (
        pack_gdn_gate_up and isinstance(module,GatedDeltaNetBlock) and name in ('ffn_gate','ffn_up'))
      if isinstance(source,MaliQ8Linear):count += 1
      elif isinstance(source,nn.Linear) and not isinstance(source,MaliLinear):
        setattr(module,name,MaliLinear(source,prepare_gemm=module_pack and name!='output',
                                      q8_weight=q8_head if module is model and name=='output' else None))
        count += 1
      elif isinstance(source,MaliLinear):
        if not module_pack and not keep_pack:source.packed_weight = None
        if module is model and name=='output' and q8_head is not None:
          source.q8_weight,source.q8_source = q8_head,source.weight.uop
        count += 1
      elif source is not None and isinstance(getattr(source,'source',None),nn.Linear):
        if not isinstance(source.source,MaliLinear):source.source = MaliLinear(source.source,prepare_gemm=module_pack and name!='output')
        elif not module_pack and not keep_pack:source.source.packed_weight = None
        count += 1
      layer = getattr(module,name,None)
      linear = layer if isinstance(layer,MaliLinear) else getattr(layer,'source',None)
      if isinstance(linear,MaliLinear):
        linear.consistent_decode = consistent_decode
        linear.bounded_half_prefill = bounded_half_prefill and half_prepared['prepared_kernels']>0
    if not module_pack:
      for name in ('attn_qkv','ffn_gate_up'):
        if isinstance(joint:=getattr(module,name,None),MaliJoint):joint.linear.packed_weight = None
    if fuse_projections and isinstance(module,TransformerBlock):
      for name,parts in (('attn_qkv',('attn_q','attn_k','attn_v')),('ffn_gate_up',('ffn_gate','ffn_up'))):
        layers = [layer for part in parts if isinstance(layer:=getattr(module,part,None),MaliLinear)]
        if getattr(module,name,None) is None and len(layers)==len(parts):
          setattr(module,name,MaliJoint(layers,prepare_gemm=pack_weights))
        if isinstance(getattr(module,name,None),MaliJoint):
          if name=='attn_qkv':qkv_count+=1
          else:gate_count+=1
    for name in ('attn_qkv','ffn_gate_up'):
      if isinstance(joint:=getattr(module,name,None),MaliJoint):joint.linear.consistent_decode = consistent_decode
  words = getattr(model,'token_embd',None)
  if isinstance(words,nn.Embedding) and not isinstance(words,MaliEmbedding):model.token_embd = MaliEmbedding(words)
  segmented = getattr(getattr(model,'per_layer_token_embd',None),'chunks',[])
  prepared_chunks = 0
  for index,embedding in enumerate(segmented):
    if isinstance(embedding,nn.Embedding):
      if not isinstance(embedding,MaliEmbedding):segmented[index] = MaliEmbedding(embedding,prefill_only=True)
      else:embedding.prefill_only = True
      prepared_chunks += 1
  output = getattr(model,'output',None)
  packing_targets = [getattr(block,name) for block,name in gdn_candidates]
  output_linear = output if isinstance(output,MaliLinear) else getattr(output,'source',None)
  if pack_output and isinstance(output_linear,MaliLinear):packing_targets.append(output_linear)
  # Remember ownership only, never retain a stale packed Tensor for restoration.
  # Existing valid copies borrowed by other closed/prepared users survive errors.
  newly_owned_packs = [layer for layer in packing_targets if isinstance(layer,MaliLinear)
                      and (layer.packed_weight is None or layer.packed_source is not layer.weight.uop)]
  try:
    gdn_packed_layers,gdn_packed_bytes = _configure_gdn_gate_up_packing(gdn_candidates)
    packed_output,packed_output_bytes = _configure_output_packing(output,pack_output)
    active_q8 = isinstance(output,MaliLinear) and output.q8_weight is not None and output.q8_source is output.weight.uop
    if q8_head is not None and not active_q8:raise ValueError('Requested Q8 head could not attach to the active GPU output projection')
    quantized = [layer for block in getattr(model,'blk',[]) for layer in vars(block).values() if isinstance(layer,MaliQ8Linear)]
    model._mali_native_state_snapshots=native_state_snapshots
    return {'linear_layers':count,'bounded_masks':masks,'attention_blocks':attentions,'direct_embedding':words is not None,
            'prefill_chunk':chunk,'fused_qkv':qkv_count,'fused_gate_up':gate_count,'q8_output_head':active_q8,'pack_weights':pack_weights,
            'pack_output':pack_output,'packed_output_head':packed_output,'packed_output_bytes':packed_output_bytes,
            'segmented_embedding_chunks':prepared_chunks,'consistent_decode':consistent_decode,
            'native_state_snapshots':native_state_snapshots,'gated_attention_blocks':gated_attentions,
            'bounded_half_prefill':bounded_half_prefill,'half_prefill_prepared_kernels':half_prepared['prepared_kernels'],
            'fused_gated_delta_blocks':recurrent_count,
            'recurrent_pack_weights':False,'pack_gdn_gate_up':pack_gdn_gate_up,
            'gdn_gate_up_packed_layers':gdn_packed_layers,'gdn_gate_up_packed_bytes':gdn_packed_bytes,
            'gdn_gate_up_packed_scope':'ffn_gate,ffn_up' if gdn_packed_layers else 'off',
            'q8_projection_layers':len(quantized),'q8_projection_resident_bytes':sum(layer.raw.nbytes() for layer in quantized),
            'q8_projection_fp16_bytes_replaced':sum(layer.out_features*layer.in_features*2 for layer in quantized),
            'q8_projection_checkpoint':str(Path(q8_projections_checkpoint).resolve()) if q8_projections_checkpoint is not None else None,
            'q8_projection_skipped':q8_skipped,
            'generic_recurrent_blocks':sum(isinstance(block,RWKV7Block) for block in getattr(model,'blk',[]))}
  except BaseException:
    # Cover body packing, combined output packing and late attachment checks.
    for layer in newly_owned_packs:layer.packed_weight = None
    raise
