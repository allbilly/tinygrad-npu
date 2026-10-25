"""DFlash2 inference equations, without a Transformers dependency.

The checkpoint's learned input/output convolutions and attention layout follow
vllm-project/speculators at 21192d6ff51bb2aaa2ab0aa1d72e6f0b8f6a7d84.
Target features describe committed inputs; slot zero is the already selected
anchor and slots 1..block_size-1 predict tokens at their own absolute positions.

MIT License
Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple, Protocol
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.device import Buffer

if TYPE_CHECKING:
  from tinygrad.llm.dflash2_config import DFlash2Config
  from tinygrad.llm.model import TargetOutput
  from tinygrad.llm.speculative_sampling import SampledProposal


class DraftOutput(NamedTuple):
  logits: Tensor
  hidden: Tensor

  def realize(self) -> DraftOutput:
    self.logits.realize(self.hidden)
    return self


class DraftSelector(Protocol):
  # max_tokens counts followups; DFlash2.propose's public budget includes the anchor.
  def choose(self, unary_logits:Tensor, hidden:Tensor, anchor:int, max_tokens:int) -> Sequence[int]: ...


class PrefixKV(NamedTuple):
  key:Tensor
  value:Tensor
  start_pos:int
  end_pos:int


class _Projection:
  """An nn.Linear-compatible view of an existing checkpoint allocation."""
  def __init__(self, weight:Tensor):
    self.weight, self.bias = weight, None

  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose())


def grouped_dynamic_conv(hidden:Tensor, delta:Tensor, base:Tensor, *, block_size:int, group_size:int) -> Tensor:
  """Token-conditioned causal taps; a tap never crosses a draft-block boundary."""
  if hidden.ndim!=3 or group_size<=0 or hidden.shape[-1]%group_size:
    raise ValueError('DFlash2 convolution requires [batch, tokens, grouped hidden]')
  batch, rows, width = hidden.shape
  if block_size<=0 or rows%block_size or base.ndim!=2 or base.shape[1]!=width or not 0<base.shape[0]<=block_size:
    raise ValueError('Invalid DFlash2 convolution block or base kernel shape')
  kernel, groups = base.shape[0], width//group_size
  if delta.shape!=(batch,rows,kernel,groups):raise ValueError('Invalid DFlash2 convolution delta kernel shape')
  # Separate the batch and block axes before shifting; flattening across batches
  # is valid in training only because every batch contains complete blocks.
  x = hidden.reshape(batch,rows//block_size,block_size,groups,group_size)
  coefficients = delta.reshape(batch,rows//block_size,block_size,kernel,groups)
  result = Tensor.zeros(*x.shape,device=hidden.device,dtype=hidden.dtype)
  for tap in range(kernel):
    shifted = x if tap==0 else x[:,:,:block_size-tap].pad((None,None,(tap,0),None,None))
    coefficient = base[tap].cast(hidden.dtype).reshape(1,1,1,groups,group_size)+coefficients[:,:,:,tap].unsqueeze(-1)
    result = result + shifted*coefficient
  return result.reshape(hidden.shape)


class DFlash2:
  """A trained draft using resident checkpoint tensors and borrowed target heads.

  ``verifier_norm`` is borrowed for identity/compatibility with the verifier; it
  is intentionally not applied to draft outputs: the trained draft has its own
  ``norm`` before the borrowed LM head. No target module is copied or mutated.
  """
  def __init__(self, config:DFlash2Config, weights:Mapping[str,Tensor], token_embedding:Callable[[Tensor],Tensor],
               lm_head:Callable[[Tensor],Tensor], verifier_norm:Callable[[Tensor],Tensor],
               candidate_selector:DraftSelector|None=None, *, cache_prefix_kv:bool=True):
    self.config, self.weights = config, dict(weights)
    self.token_embedding, self.lm_head, self.verifier_norm = token_embedding, lm_head, verifier_norm
    self.candidate_selector = candidate_selector
    self.cache_prefix_kv = cache_prefix_kv
    self._prepared_linear:dict[str,Callable[[Tensor],Tensor]] = {}
    self._prepared_shared_head = False
    self._npu_head:_NPUDraftHead|None = None
    self._proposal_forward:_PreparedDraftForward|None = None
    self._prepared_observation:_PreparedObservation|None = None
    self.capture_layers, self.block_size = config.capture_layers, config.block_size
    self.device, self.dtype = self.weights['fc.weight'].device, self.weights['fc.weight'].dtype
    if self.dtype not in (dtypes.float32,dtypes.float16,dtypes.bfloat16):raise ValueError('DFlash2 requires floating-point weights')
    if any(x.device!=self.device for x in self.weights.values()):raise ValueError('DFlash2 weights must reside on one compute device')
    if str(self.device).startswith('DISK:'):raise ValueError('Move DFlash2 weights to a compute device before inference')
    self._cache_revision = 0
    self._retain_target_features = True
    self._observed_rows = 0
    self.reset()

  @property
  def cache_revision(self) -> int:
    """Changes on every observation attempt or reset, including equal-length replacements."""
    return self._cache_revision

  @property
  def cached_prefix_length(self) -> int:
    if not self._retain_target_features:return self._observed_rows
    return 0 if self._target_hidden is None else int(self._target_hidden.shape[1])

  def reset(self) -> None:
    if self._proposal_forward is not None:self._proposal_forward.invalidate_prefix_inputs()
    self._target_hidden:Tensor|None = None
    self._prefix_kv:tuple[PrefixKV|None,...] = (None,)*self.config.num_layers
    self._observed_rows = 0
    self._cache_revision += 1

  def prepare_mali(self, *, pack_weights:bool=False, prepare_shared_head:bool=True, pack_fc:bool=False) -> dict[str,int|bool]:
    """Opt into prepared draft projections while retaining target-owned heads.

    Row-major weights are the default to avoid a second packed checkpoint copy.
    ``pack_fc`` packs only the input feature projection; ``pack_weights`` packs
    every eligible projection. Report actual packed storage for memory admission.
    This path changes reduction order and requires device/output qualification.
    """
    if type(pack_fc) is not bool:raise TypeError('pack_fc must be a bool')
    if not isinstance(self.device,str) or self.device.split(':')[0]!='CL':raise ValueError('Prepared DFlash2 projections require Mali OpenCL')
    if not getattr(Device[self.device],'device_name','').startswith('Mali'):raise ValueError('Prepared DFlash2 projections require Mali OpenCL')
    if self.dtype not in (dtypes.float16,dtypes.float32):raise ValueError('Prepared DFlash2 projections require FP16 or FP32 weights')
    self.close_prepared_proposals()
    self.close_prepared_observation()
    from tinygrad.llm.kernels.mali import MaliLinear
    from tinygrad.llm.kernels.mali_q8_shared import MaliQ8Head
    prepared:dict[str,Callable[[Tensor],Tensor]] = {}
    packed_bytes, packed_fc = 0,False
    for name,weight in self.weights.items():
      if weight.ndim!=2 or not name.endswith('.weight') or name.startswith('candidate_selector.'):continue
      if weight.shape[1]%4:raise ValueError('Prepared DFlash2 projection depth must be divisible by four')
      layer = MaliLinear(_Projection(weight),prepare_gemm=pack_weights or (pack_fc and name=='fc.weight'))
      prepared[name[:-7]] = layer
      if layer.packed_weight is not None:
        packed_bytes += layer.packed_weight.nbytes()
        if name=='fc.weight':packed_fc = True
    self._prepared_linear = prepared
    self._prepared_shared_head = (prepare_shared_head and isinstance(self.lm_head,(MaliLinear,MaliQ8Head))
      and self.lm_head.weight.device==self.device and self.lm_head.weight.dtype in (dtypes.float16,dtypes.float32)
      and self.lm_head.weight.shape==(self.config.vocab_size,self.config.hidden_size) and self.lm_head.in_features%4==0)
    # Cached projected features/KV came from the previous arithmetic path. The
    # caller must observe the committed prefix again after changing kernels.
    self.reset()
    return {'linear_layers':len(prepared),'packed_weights':pack_weights,'prepared_shared_head':self._prepared_shared_head,
            'packed_fc':packed_fc,'packed_bytes':packed_bytes}

  def _validate_npu_head(self) -> None:
    from tinygrad.llm.kernels.mali import MaliLinear
    if (self.device!='CL' or self.dtype is not dtypes.float16 or self.block_size!=8
        or (self.config.hidden_size,self.config.vocab_size)!=(2048,248320)
        or not isinstance(self.lm_head,MaliLinear) or self.lm_head.weight.device!=self.device
        or self.lm_head.weight.dtype is not dtypes.float16 or self.lm_head.weight.shape!=(248320,2048)
        or self.lm_head.bias is not None):
      raise ValueError('NPU draft head requires the FP16 Mali Qwen3.5-2B shared head and eight trained slots')
    if not getattr(Device['CL'],'device_name','').startswith('Mali'):
      raise ValueError('NPU draft head requires Mali OpenCL')

  def prepare_npu_head(self, *, exclusive_lease:bool=False) -> dict:
    """Opt into the qualified R8/FP16/tile12288 head microcandidate only.

    The caller must explicitly reserve the NPU and about 970 MiB for one packed
    head. This never replaces the target head, moves other draft projections, or
    unloads a service. All activation copies/submissions remain in proposal time.
    Whole-model token equality and speed still require separate qualification.
    """
    if exclusive_lease is not True:raise ValueError('An explicit exclusive NPU/RAM lease is required')
    self._validate_npu_head()
    from tinygrad.llm.kernels.mali import MaliLinear
    assert isinstance(self.lm_head,MaliLinear)
    self.close_npu_head()
    self._npu_head=_NPUDraftHead(self.lm_head,rows=8,tile_columns=12288)
    return self.npu_head_preparation_metadata()

  def npu_head_preparation_metadata(self) -> dict:
    return {'enabled':False,'additional_npu_packed_bytes':0} if self._npu_head is None else self._npu_head.metadata()

  def close_npu_head(self) -> None:
    """Drop captured graphs and this draft's packed-head references.

    NPU allocator/program caches and process-global ownership remain with the
    outer lease. Finalize that device or exit its dedicated child only after all
    mixed graphs and their NPU users are closed; this method never resets it.
    """
    prepared,self._npu_head=self._npu_head,None
    errors=[]
    try:self.close_prepared_proposals()
    except BaseException as error:errors.append(error)
    if prepared is not None:
      try:prepared.close()
      except BaseException as error:errors.append(error)
    if len(errors)==1:raise errors[0]
    if errors:raise BaseExceptionGroup('NPU draft head cleanup failed',errors)

  def prepare_proposals(self, *, max_context:int=512, reuse_prefix_inputs:bool=False) -> dict[str,int|bool]:
    """Opt into one bounded draft graph; packing and selection remain in propose.

    Complete trained blocks and fixed-capacity KV inputs change reduction geometry.
    This experimental path requires real-checkpoint numerical/speed qualification.
    Reusable inputs preserve owned zero-padded storage, updating only changed
    full-attention rows; sliding windows are refreshed after each observation.
    """
    if type(reuse_prefix_inputs) is not bool:raise ValueError('reuse_prefix_inputs must be a bool')
    if not self.cache_prefix_kv:raise ValueError('Prepared proposals require the committed prefix KV cache')
    limit=self.config.max_position_embeddings
    if type(max_context) is not int or not 1<=max_context<=limit:
      raise ValueError(f'Prepared draft context must be in 1..{limit}')
    if not isinstance(self.device,str) or self.device.split(':')[0] not in ('CPU','CL'):
      raise ValueError('Prepared draft proposals support a single CPU or OpenCL device')
    self.close_prepared_proposals()
    self._proposal_forward=_PreparedDraftForward(self,max_context,reuse_prefix_inputs)
    return self.proposal_preparation_metadata()

  def proposal_preparation_metadata(self) -> dict[str,int|bool]:
    if self._proposal_forward is None:return {'enabled':False,'graphs':0,'captured_graphs':0}
    return self._proposal_forward.metadata()

  def close_prepared_proposals(self) -> None:
    prepared,self._proposal_forward=self._proposal_forward,None
    if not self._retain_target_features:self.reset()
    if prepared is not None:prepared.close()

  def prepare_observation(self, *, max_context:int=512, pad_decode_rows:bool=False, native_cache_copy:bool=False,
                          bound_prefix_memory:bool=False, retain_target_features:bool=True) -> dict:
    """Prepare FC/KV/RoPE graphs, optionally retaining only append-only prefix KV.

    Dropping projected feature history requires prepared proposals. Suffix
    replacement/deep rollback remains available in the default retaining mode.
    """
    if type(pad_decode_rows) is not bool:raise ValueError('pad_decode_rows must be a bool')
    if type(native_cache_copy) is not bool:raise ValueError('native_cache_copy must be a bool')
    if type(bound_prefix_memory) is not bool:raise ValueError('bound_prefix_memory must be a bool')
    if type(retain_target_features) is not bool:raise ValueError('retain_target_features must be a bool')
    if not self.cache_prefix_kv:raise ValueError('Prepared observation requires prefix KV caching')
    limit=self.config.max_position_embeddings
    if type(max_context) is not int or not 1<=max_context<=limit:
      raise ValueError(f'Prepared observation context must be in 1..{limit}')
    if not isinstance(self.device,str) or self.device.split(':')[0] not in ('CPU','CL'):
      raise ValueError('Prepared observation supports a single CPU or OpenCL device')
    if not retain_target_features and (self._proposal_forward is None or self._proposal_forward.closed
                                       or self._proposal_forward.capacity<max_context):
      raise ValueError('Dropping target features requires prepared proposals covering the observation context')
    if retain_target_features!=self._retain_target_features:self.reset()
    self._retain_target_features = retain_target_features
    self.close_prepared_observation()
    self._prepared_observation=_PreparedObservation(self,max_context,pad_decode_rows,native_cache_copy,bound_prefix_memory)
    return self.observation_preparation_metadata()

  def observation_preparation_metadata(self) -> dict:
    result = {'enabled':False,'graphs':0} if self._prepared_observation is None else self._prepared_observation.metadata()
    if not self._retain_target_features:
      result.update(retain_target_features=False,cached_prefix_rows=self.cached_prefix_length,retained_projected_history_bytes=0)
    return result

  def close_prepared_observation(self) -> None:
    prepared,self._prepared_observation=self._prepared_observation,None
    if prepared is not None:prepared.close()

  def _linear(self, x:Tensor, name:str) -> Tensor:
    if name in self._prepared_linear:return self._prepared_linear[name](x.float()).cast(x.dtype)
    return x.linear(self.weights[name+'.weight'].transpose())

  def _norm(self, x:Tensor, name:str) -> Tensor:
    return x.float().rmsnorm(eps=self.config.norm_eps).cast(x.dtype)*self.weights[name+'.weight']

  def _project_head(self, hidden:Tensor) -> Tensor:
    # MaliLinear's prepared GEMM expects FP32 activations. Preserve the trained
    # activation/logit dtype around that accumulation and keep the target-owned
    # head object and allocation unchanged.
    if self._npu_head is not None:return self._npu_head(hidden,self.lm_head)
    if self._prepared_shared_head:return self.lm_head(hidden.float()).cast(hidden.dtype)
    return self.lm_head(hidden)

  def project_features(self, features:Sequence[Tensor]|Tensor) -> Tensor:
    """Concatenate configured raw target taps, project once, and normalize."""
    features=self._feature_matrix(features)
    if features.shape[1]==0:return Tensor.zeros(1,0,self.config.hidden_size,device=self.device,dtype=self.dtype)
    return self._norm(self._linear(features,'fc'),'hidden_norm')

  def _feature_matrix(self, features:Sequence[Tensor]|Tensor) -> Tensor:
    if not isinstance(features,Tensor):
      if len(features)!=len(self.capture_layers):raise ValueError('DFlash2 requires every configured target feature')
      if not features:raise ValueError('DFlash2 requires target features')
      features = features[0].cat(*features[1:],dim=-1)
    if features.ndim!=3 or features.shape[0]!=1 or features.shape[-1]!=len(self.capture_layers)*self.config.hidden_size:
      raise ValueError('Invalid DFlash2 concatenated target feature shape')
    return features.to(self.device).cast(self.dtype)

  def observe(self, output:TargetOutput, start_pos:int) -> None:
    """Commit projected features and per-layer prefix KV, replacing a suffix."""
    self.observe_features(output.features,start_pos)

  def observe_features(self, features:Sequence[Tensor]|Tensor, start_pos:int) -> None:
    """Commit target feature rows without retaining logits or hidden outputs."""
    # Prepared copies can write existing buffers before a later projection fails.
    # Even unsuccessful observation attempts invalidate an external reuse claim.
    graph=self._proposal_forward
    preserved=0 if graph is None else graph.invalidate_prefix_inputs(start_pos)
    self._cache_revision += 1
    length = self.cached_prefix_length
    if type(start_pos) is not int or not 0<=start_pos<=length:raise ValueError('DFlash2 target feature cache has a gap')
    if not self._retain_target_features:
      if self._proposal_forward is None or self._proposal_forward.closed:
        raise ValueError('Append-only prefix KV requires prepared proposals')
      if start_pos!=length:raise ValueError('Append-only prefix KV cannot replace a suffix; reset the draft first')
    prepared=None if self._prepared_observation is None else self._prepared_observation(features,start_pos)
    projected = self.project_features(features).realize() if prepared is None else prepared[0]
    if not isinstance(projected.shape[1],int):raise ValueError('DFlash2 committed feature rows must be static')
    end_pos = start_pos+projected.shape[1]
    if not self._retain_target_features:
      assert self._proposal_forward is not None
      if end_pos>self._proposal_forward.capacity:raise ValueError('Append-only prefix exceeds prepared proposal capacity')
    if end_pos==0:
      self.reset()
      return
    if not self._retain_target_features:hidden=None
    elif start_pos==0:hidden=projected
    else:
      assert self._target_hidden is not None
      copied=None
      if prepared is not None and projected.shape[1]:
        assert self._prepared_observation is not None
        copied=self._prepared_observation.append_rows(self._target_hidden,projected,0,start_pos)
      hidden = copied if copied is not None else (
        self._target_hidden[:,:start_pos].cat(projected,dim=1) if projected.shape[1] else self._target_hidden[:,:start_pos]).realize()
    caches:list[PrefixKV|None] = []
    pending_copies:list[Tensor] = []
    for layer in range(self.config.num_layers):
      if not self.cache_prefix_kv:
        caches.append(None)
        continue
      prepared_copy = prepared is not None
      begin = max(0,end_pos-self.config.sliding_window) if self.config.layer_types[layer]=='sliding_attention' else 0
      previous = self._prefix_kv[layer]
      if begin>=start_pos:
        if prepared is None:
          visible = hidden[:,begin:] if hidden is not None else projected[:,begin-start_pos:]
          current=self._make_prefix_kv(visible,begin,layer)
        else:
          offset=begin-start_pos
          current=PrefixKV(prepared[1+2*layer][:,:,offset:],prepared[2+2*layer][:,:,offset:],begin,end_pos)
      elif previous is not None and previous.start_pos<=begin and previous.end_pos>=start_pos:
        preserved_key = previous.key[:,:,begin-previous.start_pos:start_pos-previous.start_pos]
        preserved_value = previous.value[:,:,begin-previous.start_pos:start_pos-previous.start_pos]
        if projected.shape[1]:
          appended = (self._make_prefix_kv(projected,start_pos,layer) if prepared is None
                      else PrefixKV(prepared[1+2*layer],prepared[2+2*layer],start_pos,end_pos))
          copied_key=copied_value=None
          if prepared is not None:
            assert self._prepared_observation is not None
            copied_key=self._prepared_observation.append_rows(previous.key,appended.key,begin-previous.start_pos,start_pos-previous.start_pos)
            copied_value=self._prepared_observation.append_rows(previous.value,appended.value,begin-previous.start_pos,start_pos-previous.start_pos)
          current = PrefixKV(copied_key if copied_key is not None else preserved_key.cat(appended.key,dim=2),
                             copied_value if copied_value is not None else preserved_value.cat(appended.value,dim=2),begin,end_pos)
        else:current=PrefixKV(preserved_key,preserved_value,begin,end_pos)
      else:
        # A deep rollback may need rows evicted from a sliding layer's window.
        if hidden is None:raise ValueError('Append-only prefix KV cannot rebuild an evicted prefix')
        current = self._make_prefix_kv(hidden[:,begin:],begin,layer)
        prepared_copy = False
      if prepared_copy:pending_copies.extend((current.key,current.value))
      else:current.key.realize(current.value)
      caches.append(current)
    # Prepared rows already own their projected KV. Batch only their pure
    # slice/concat copies; projection rebuilds retain per-layer realization.
    if pending_copies:Tensor.realize(*pending_copies)
    if self._prepared_observation is not None and self._prepared_observation.bound_prefix_memory:
      growing=[] if hidden is None else [hidden]
      for kind,cache in zip(self.config.layer_types,caches):
        if cache is not None and (kind=='full_attention' or cache.end_pos-cache.start_pos<self.config.sliding_window):
          growing.extend((cache.key,cache.value))
      for value in growing:
        if value.numel()==0:continue
        # These allocations grow at each append. Retaining every freed size
        # consumes quadratic cache space. Full sliding windows and scratch
        # buffers retain their existing allocation policy.
        buffer=value.uop.base.buffer
        assert isinstance(buffer,Buffer)
        buffer.base.options=replace(buffer.base.options,nolru=True)
    # Commit only after every required projection succeeds; exceptions retain
    # the previous matching feature and KV caches.
    self._target_hidden, self._prefix_kv = hidden, tuple(caches)
    self._observed_rows = end_pos
    if graph is not None:
      graph.preserved_prefix_rows=preserved
      graph.prefix_inputs_source=self._prefix_kv

  def _conv_prepare(self, x:Tensor, name:str) -> tuple[Tensor,Tensor]:
    c = self.config
    kernels = self._linear(x,name+'.kernel_projection').reshape(*x.shape[:-1],2,c.conv_kernel_size,c.hidden_size//c.conv_group_size)
    prepared = grouped_dynamic_conv(x,kernels[...,0,:,:],self.weights[name+'.base_kernel'][0],
                                    block_size=c.block_size,group_size=c.conv_group_size)
    return prepared,kernels[...,1,:,:]

  def _conv_finish(self, x:Tensor, kernel:Tensor, name:str) -> Tensor:
    return grouped_dynamic_conv(x,kernel,self.weights[name+'.base_kernel'][1],
                                block_size=self.block_size,group_size=self.config.conv_group_size)

  def _rope(self, x:Tensor, positions:Tensor) -> Tensor:
    half = self.config.head_dim//2
    frequencies = 1.0/(self.config.rope_theta**(Tensor.arange(half,dtype=dtypes.float32).to(self.device)/half))
    angles = positions.cast(dtypes.float32).reshape(1,1,-1,1)*frequencies.reshape(1,1,1,half)
    cosine, sine = angles.cos().cast(x.dtype), angles.sin().cast(x.dtype)
    first, second = x.chunk(2,dim=-1)
    return (first*cosine-second*sine).cat(second*cosine+first*sine,dim=-1)

  def _make_prefix_kv(self, hidden:Tensor, start_pos:int, layer:int) -> PrefixKV:
    if not isinstance(rows:=hidden.shape[1],int) or rows<1:raise ValueError('DFlash2 prefix KV requires static nonempty rows')
    positions = Tensor.arange(start_pos,start_pos+rows,dtype=dtypes.int32).to(self.device)
    key,value=self._kv_values(hidden,positions,layer)
    return PrefixKV(key,value,start_pos,start_pos+rows)

  def _kv_values(self, hidden:Tensor, positions:Tensor, layer:int) -> tuple[Tensor,Tensor]:
    c,name=self.config,f'layers.{layer}.self_attn'
    rows=hidden.shape[1]
    key = self._linear(hidden,name+'.k_proj').reshape(1,rows,c.num_key_value_heads,c.head_dim)
    key = self._norm(key,name+'.k_norm').transpose(1,2)
    key = self._rope(key,positions)
    value = self._linear(hidden,name+'.v_proj').reshape(1,rows,c.num_key_value_heads,c.head_dim).transpose(1,2)
    return key,value

  def _attention(self, x:Tensor, target_hidden:Tensor, position:int, target_start_pos:int, layer:int, cached:PrefixKV|None=None) -> Tensor:
    c, name = self.config, f'layers.{layer}.self_attn'
    rows = self.block_size
    if cached is not None:
      if cached.end_pos!=position or cached.start_pos<target_start_pos:raise ValueError('DFlash2 cached KV does not cover its prefix')
      target_start_pos=cached.start_pos
    prefix = position-target_start_pos
    q = self._linear(x,name+'.q_proj').reshape(1,rows,c.num_attention_heads,c.head_dim)
    q = self._norm(q,name+'.q_norm').transpose(1,2)
    k, v = self._linear(x,name+'.k_proj'),self._linear(x,name+'.v_proj')
    if prefix and cached is None:
      k = self._linear(target_hidden,name+'.k_proj').cat(k,dim=1)
      v = self._linear(target_hidden,name+'.v_proj').cat(v,dim=1)
    projected_rows = rows if cached is not None else prefix+rows
    k = self._norm(k.reshape(1,projected_rows,c.num_key_value_heads,c.head_dim),name+'.k_norm').transpose(1,2)
    v = v.reshape(1,projected_rows,c.num_key_value_heads,c.head_dim).transpose(1,2)
    positions = Tensor.arange(target_start_pos,position+rows,dtype=dtypes.int32).to(self.device)
    q, k = self._rope(q,positions[prefix:]),self._rope(k,positions[prefix:] if cached is not None else positions)
    if cached is not None:k,v=cached.key.cat(k,dim=2),cached.value.cat(v,dim=2)
    groups = c.num_attention_heads//c.num_key_value_heads
    if groups!=1:
      k = k.unsqueeze(2).expand(1,c.num_key_value_heads,groups,prefix+rows,c.head_dim).reshape(1,c.num_attention_heads,prefix+rows,c.head_dim)
      v = v.unsqueeze(2).expand(1,c.num_key_value_heads,groups,prefix+rows,c.head_dim).reshape(1,c.num_attention_heads,prefix+rows,c.head_dim)
    # Prefix windows are relative to the anchor, not each speculative query.
    keys = Tensor.arange(prefix+rows).to(self.device).reshape(1,1,1,-1)
    mask = keys>=0
    if c.layer_types[layer]=='sliding_attention':
      mask = (keys>=prefix) | (keys+target_start_pos>=position-c.sliding_window)
      if not c.sliding_window_non_causal:
        query = Tensor.arange(rows).to(self.device).reshape(1,1,-1,1)
        mask = mask & ((keys<prefix) | (keys<=prefix+query))
    scores = (q@k.transpose(-1,-2))*(c.head_dim**-.5)
    probabilities = mask.where(scores,float('-inf')).float().softmax(-1).cast(q.dtype)
    attended = (probabilities@v).transpose(1,2).reshape(1,rows,c.num_attention_heads*c.head_dim)
    return self._linear(attended,name+'.o_proj')

  def _forward_projected(self, noise_ids:Tensor, target_hidden:Tensor, *, position:int, target_start_pos:int=0,
                         prefix_kv:Sequence[PrefixKV|None]|None=None) -> DraftOutput:
    c = self.config
    if noise_ids.shape!=(1,self.block_size) or not dtypes.is_int(noise_ids.dtype):raise ValueError('DFlash2 requires one complete token block')
    if type(position) is not int or type(target_start_pos) is not int or not 0<=target_start_pos<=position:
      raise ValueError('Invalid DFlash2 absolute position')
    if target_hidden.shape!=(1,position-target_start_pos,c.hidden_size):raise ValueError('DFlash2 prefix must end immediately before its anchor')
    if prefix_kv is not None and len(prefix_kv)!=c.num_layers:raise ValueError('DFlash2 prefix cache layer count differs from its draft')
    return self._noise_forward(noise_ids,lambda x,layer:self._attention(x,target_hidden,position,target_start_pos,layer,
                                                                      None if prefix_kv is None else prefix_kv[layer]))

  def _noise_forward(self, noise_ids:Tensor, attention:Callable[[Tensor,int],Tensor]) -> DraftOutput:
    c=self.config
    hidden = self.token_embedding(noise_ids.to(self.device)).to(self.device).cast(self.dtype)
    if hidden.shape!=(1,self.block_size,c.hidden_size):raise ValueError('Borrowed DFlash2 embedding has an incompatible shape')
    for layer in range(c.num_layers):
      name = f'layers.{layer}'
      residual = hidden
      prepared, kernel = self._conv_prepare(self._norm(hidden,name+'.input_layernorm'),name+'.attention_conv')
      hidden = residual+self._conv_finish(attention(prepared,layer),kernel,name+'.attention_conv')
      residual = hidden
      prepared, kernel = self._conv_prepare(self._norm(hidden,name+'.post_attention_layernorm'),name+'.mlp_conv')
      gate, up = self._linear(prepared,name+'.mlp.gate_proj'),self._linear(prepared,name+'.mlp.up_proj')
      hidden = residual+self._conv_finish(self._linear(gate.silu()*up,name+'.mlp.down_proj'),kernel,name+'.mlp_conv')
    hidden = self._norm(hidden,'norm')
    logits = self._project_head(hidden)
    if logits.shape!=(1,self.block_size,c.vocab_size):raise ValueError('Borrowed DFlash2 LM head has an incompatible shape')
    return DraftOutput(logits,hidden)

  def __call__(self, noise_ids:Tensor, target_features:Sequence[Tensor]|Tensor, *, position:int, target_start_pos:int=0) -> DraftOutput:
    return self._forward_projected(noise_ids,self.project_features(target_features),position=position,target_start_pos=target_start_pos)

  def _validate_proposal(self, anchor:int, position:int, max_tokens:int) -> None:
    if type(anchor) is not int or not 0<=anchor<self.config.vocab_size:raise ValueError('Invalid DFlash2 anchor token')
    if type(position) is not int or position<0:raise ValueError('Invalid DFlash2 proposal position')
    if type(max_tokens) is not int or max_tokens<1:raise ValueError('DFlash2 proposal budget must include its anchor')
    if self.candidate_selector is None:raise ValueError('DFlash2 proposals require the trained candidate selector')
    length = self.cached_prefix_length
    if position!=length:raise ValueError('DFlash2 proposal position differs from its committed feature cache')

  def _proposal_output(self, anchor:int, position:int) -> DraftOutput:
    if not self._retain_target_features and self._proposal_forward is None:
      raise ValueError('Append-only prefix KV requires prepared proposals')
    prefix = self._target_hidden
    if prefix is None:prefix=Tensor.zeros(1,0,self.config.hidden_size,device=self.device,dtype=self.dtype)
    noise_ids = Tensor([[anchor]+[self.config.mask_token_id]*(self.block_size-1)],device=self.device,dtype=dtypes.int32)
    return (self._forward_projected(noise_ids,prefix,position=position,prefix_kv=self._prefix_kv).realize()
            if self._proposal_forward is None else self._proposal_forward(noise_ids,self._prefix_kv,position))

  def propose(self, anchor:int, *, position:int, max_tokens:int) -> Sequence[int]:
    self._validate_proposal(anchor,position,max_tokens)
    if max_tokens==1:return ()
    output=self._proposal_output(anchor,position)
    assert self.candidate_selector is not None
    result = tuple(self.candidate_selector.choose(output.logits,output.hidden,anchor,min(max_tokens-1,self.block_size-1)))
    if len(result)>min(max_tokens-1,self.block_size-1) or any(type(token) is not int or not 0<=token<self.config.vocab_size for token in result):
      raise ValueError('DFlash2 candidate selector returned invalid followup tokens')
    return result

  def propose_sampled(self, anchor:int, *, position:int, max_tokens:int, temperature:float,
                      random:Callable[[],float]) -> SampledProposal:
    from tinygrad.llm.speculative_sampling import SampledProposal
    self._validate_proposal(anchor,position,max_tokens)
    sampler=getattr(self.candidate_selector,'sample',None)
    if not callable(sampler):raise ValueError('DFlash2 sampling requires a probability-aware candidate selector')
    if max_tokens==1:return SampledProposal((),())
    output=self._proposal_output(anchor,position)
    result=sampler(output.logits,output.hidden,anchor,min(max_tokens-1,self.block_size-1),temperature=temperature,random=random)
    if (not isinstance(result,SampledProposal) or len(result.tokens)>min(max_tokens-1,self.block_size-1)
        or any(max(row.token_ids)>=self.config.vocab_size for row in result.distributions)):
      raise ValueError('DFlash2 sampled selector returned an invalid proposal')
    return result


class _NPUDraftHead:
  """Existing wide-N streaming, without a second GPU head or target mutation.

  Public activation is restricted by DFlash2._validate_npu_head to the measured
  8x2048x248320 FP16 geometry. Smaller internal fixtures test copies/lifecycle;
  they do not broaden that production eligibility or certify model quality.
  """
  def __init__(self, head:nn.Linear, *, rows:int, tile_columns:int):
    from time import perf_counter
    started=perf_counter()
    from tinygrad.llm.kernels.rockchip import RockchipLinear
    device=head.weight.device
    columns,depth=head.weight.shape
    if not isinstance(device,str) or not isinstance(columns,int) or not isinstance(depth,int):
      raise ValueError('NPU draft head requires one static matrix on one device')
    self.device,self.dtype=device,head.weight.dtype
    self.columns,self.depth=columns,depth
    self.head:Callable[[Tensor],Tensor]|None=head
    self.weight_uop:UOp|None=head.weight.uop
    self.rows,self.tile_columns=rows,tile_columns
    self.closed=False
    self.native:RockchipLinear|None=None
    # This acquires the existing process-exclusive device lock, failing rather
    # than sharing an occupied NPU or changing any serving/model session.
    Device['ROCKCHIP']
    native=RockchipLinear(head,'on','off',tile_columns=tile_columns)
    try:native.prepare()
    except BaseException as error:
      try:native.release_packed()
      except BaseException as cleanup:raise BaseExceptionGroup('NPU draft head preparation/cleanup failed',[error,cleanup]) from None
      raise
    self.native=native
    self.packed_bytes=sum(int(value.nbytes()) for value in native.packed)
    self.preparation_seconds=perf_counter()-started

  def metadata(self) -> dict:
    return {'enabled':not self.closed,'rows':self.rows,'depth':self.depth,'columns':self.columns,
            'activation_dtype':'float16','tile_columns':self.tile_columns,'preparation_seconds':self.preparation_seconds,
            'packed_buffers':0 if self.native is None else len(self.native.packed),
            'additional_npu_packed_bytes':0 if self.closed else self.packed_bytes,'gpu_weight_copies':0,
            'expected_submits_per_call':sum((min(self.tile_columns,self.columns-start)+12287)//12288
                                            for start in range(0,self.columns,self.tile_columns)),
            'route':'CL/NPU/CL; existing wide-N streaming, not one-submit batching',
            'ownership':'external exclusive lease; allocator/program cache residency ends at device finalization/process exit',
            'qualified_model_win':False,'qualified_target_quality':False}

  def __call__(self, hidden:Tensor, head:Callable[[Tensor],Tensor]) -> Tensor:
    if self.closed or self.native is None:raise RuntimeError('NPU draft head was closed')
    weight=getattr(head,'weight',None)
    if head is not self.head or not isinstance(weight,Tensor) or weight.uop is not self.weight_uop:
      raise ValueError('Borrowed NPU draft head changed; prepare again')
    if hidden.shape!=(1,self.rows,self.depth) or hidden.device!=self.device or hidden.dtype is not dtypes.float16:
      raise ValueError('NPU draft head requires its complete resident FP16 trained block')
    # Identical activation/cast/output route to the qualified R8 microprobe.
    # TinyJit may capture every bridge and submit; none is moved out of propose.
    return self.native.npu(hidden.float()).cast(hidden.dtype)

  def close(self) -> None:
    if self.closed:return
    native,self.native=self.native,None
    errors=[]
    try:
      for device in (self.device,'ROCKCHIP'):
        try:Device[device].synchronize()
        except BaseException as error:errors.append(error)
      if native is not None:
        try:native.release_packed()
        except BaseException as error:errors.append(error)
    finally:
      self.head=None
      self.weight_uop=None
      self.closed=True
    if len(errors)==1:raise errors[0]
    if errors:raise BaseExceptionGroup('NPU draft head device cleanup failed',errors)


class _PreparedDraftForward:
  """One fixed-geometry graph; output views expire on the next call.

  No selector/CPU walk is captured. Input padding, device upload, and all reset
  work are performed by __call__, within the caller's complete propose interval.
  """
  def __init__(self, draft:DFlash2, max_context:int, reuse_prefix_inputs:bool=False):
    self.draft:DFlash2|None=draft
    self.capacity,self.device,self.dtype=max_context,draft.device,draft.dtype
    self.position=UOp.variable('dflash2_proposal_position',0,max_context-1)
    self.jit=TinyJit(self._run)
    self.closed=False
    self.calls=0
    self.reuse_prefix_inputs=reuse_prefix_inputs
    self.prefix_inputs:tuple[Tensor,...]|None=None
    self.prefix_inputs_source:Sequence[PrefixKV|None]|None=None
    self.preserved_prefix_rows=0
    self.input_valid_rows:tuple[int,...]=()
    self.input_copy_bytes=self.input_zero_bytes=self.input_reused_bytes=0
    self.native_input_copy_commands=self.native_input_copy_bytes=0
    self.signature=self._signature()
    self.prefix_capacities=tuple(min(max_context,draft.config.sliding_window) if layer=='sliding_attention' else max_context
                                 for layer in draft.config.layer_types)
    self.input_bytes=2*draft.config.num_key_value_heads*sum(self.prefix_capacities)*draft.config.head_dim*draft.dtype.itemsize
    self.output_bytes=draft.block_size*(draft.config.vocab_size+draft.config.hidden_size)*draft.dtype.itemsize

  def _signature(self) -> tuple:
    assert self.draft is not None
    draft=self.draft
    modules=[]
    for module in (draft.token_embedding,draft.lm_head):
      weight=getattr(module,'weight',None)
      modules.append((id(module),id(weight.uop) if isinstance(weight,Tensor) else None))
    return (tuple((name,id(value.uop)) for name,value in sorted(draft.weights.items())),tuple(modules),
            tuple((name,id(value)) for name,value in sorted(draft._prepared_linear.items())),draft._prepared_shared_head,id(draft._npu_head))

  def metadata(self) -> dict[str,int|bool]:
    result:dict[str,int|bool]={'enabled':not self.closed,'graphs':0 if self.closed else 1,
                              'captured_graphs':int(self.jit.captured is not None),'calls':self.calls,'max_context':self.capacity,
                              'padding_input_bytes':self.input_bytes,'output_bytes':self.output_bytes,'weight_copies':0,'packed_weight_copies':0}
    if self.draft is not None and self.draft._npu_head is not None:
      result.update(packed_weight_copies=1,additional_npu_packed_bytes=self.draft._npu_head.packed_bytes)
    if self.reuse_prefix_inputs:
      result.update(reuse_prefix_inputs=True,input_copy_bytes=self.input_copy_bytes,input_zero_bytes=self.input_zero_bytes,
                    input_reused_bytes=self.input_reused_bytes,native_input_copy_commands=self.native_input_copy_commands,
                    native_input_copy_bytes=self.native_input_copy_bytes)
    return result

  def invalidate_prefix_inputs(self, start_pos:int=0) -> int:
    # Invalidate before any observation work: failed/partial observations must
    # force a complete refresh. Only a successful commit restores this boundary.
    preserved=(min(self.preserved_prefix_rows,start_pos) if type(start_pos) is int and start_pos>=0 and self.draft is not None
               and self.prefix_inputs_source is self.draft._prefix_kv else 0)
    self.preserved_prefix_rows=0
    self.prefix_inputs_source=None
    return preserved

  def _inputs(self, noise_ids:Tensor, prefix_kv:Sequence[PrefixKV|None], position:int) -> tuple[Tensor,...]:
    if self.closed or self.draft is None:raise RuntimeError('Prepared draft graph was closed')
    if self._signature()!=self.signature:raise ValueError('Prepared draft weights or borrowed modules changed; prepare again')
    draft,c=self.draft,self.draft.config
    if type(position) is not int or not 0<=position<self.capacity:raise ValueError('Prepared draft prefix exceeds its configured context')
    if noise_ids.shape!=(1,draft.block_size) or not dtypes.is_int(noise_ids.dtype):raise ValueError('Prepared draft requires a complete noise block')
    if len(prefix_kv)!=c.num_layers:raise ValueError('Prepared draft prefix cache layer count differs')
    result:list[Tensor]=[]
    for layer,cache in enumerate(prefix_kv):
      shape=(1,c.num_key_value_heads,self.prefix_capacities[layer],c.head_dim)
      valid=min(position,c.sliding_window) if c.layer_types[layer]=='sliding_attention' else position
      if cache is None:
        if position!=0:raise ValueError('Prepared draft has a missing committed prefix cache')
        if not self.reuse_prefix_inputs:result.extend(Tensor.zeros(*shape,device=self.device,dtype=self.dtype) for _ in range(2))
        continue
      if (cache.start_pos!=position-valid or cache.end_pos!=position
          or cache.key.shape!=(1,c.num_key_value_heads,valid,c.head_dim) or cache.value.shape!=cache.key.shape):
        raise ValueError('Prepared draft cache does not match its anchor-relative window')
      if any(value.device!=self.device or value.dtype!=self.dtype for value in (cache.key,cache.value)):
        raise ValueError('Prepared draft cache dtype/device differs from its resident weights')
      # A full sliding window can be an offset slice of an observation output.
      # JIT inputs need owned buffers even when padding leaves the shape unchanged.
      if not self.reuse_prefix_inputs:result.extend(value.pad_to(shape).clone() for value in (cache.key,cache.value))
    if self.reuse_prefix_inputs:return self._update_inputs(prefix_kv,position)
    Tensor.realize(*result)
    return tuple(result)

  def _update_inputs(self, prefix_kv:Sequence[PrefixKV|None], position:int) -> tuple[Tensor,...]:
    from tinygrad.llm.kernels.mali_copy import copy_rows
    assert self.draft is not None
    c=self.draft.config
    if self.prefix_inputs is None:
      inputs=tuple(Tensor.zeros(1,c.num_key_value_heads,capacity,c.head_dim,device=self.device,dtype=self.dtype).realize()
                   for capacity in self.prefix_capacities for _ in range(2))
      self.prefix_inputs=inputs
      self.input_valid_rows=(0,)*c.num_layers
      self.input_zero_bytes+=self.input_bytes
    inputs=self.prefix_inputs
    preserved=self.preserved_prefix_rows if prefix_kv is self.prefix_inputs_source else 0
    previous=self.input_valid_rows
    # If a write fails, every potentially touched tail must be cleared on retry.
    self.invalidate_prefix_inputs()
    self.input_valid_rows=self.prefix_capacities
    valid_rows=[]
    for layer,cache in enumerate(prefix_kv):
      valid=min(position,c.sliding_window) if c.layer_types[layer]=='sliding_attention' else position
      kept=min(preserved,valid) if c.layer_types[layer]=='full_attention' else 0
      row_bytes=2*c.num_key_value_heads*c.head_dim*self.dtype.itemsize
      for index in range(2):
        output=inputs[2*layer+index]
        if previous[layer]>valid:output[:,:,valid:previous[layer]].assign(0).realize()
        if valid>kept:
          assert cache is not None
          source=(cache.key,cache.value)[index]
          if copy_rows(source,output,kept,valid):
            self.native_input_copy_commands+=1
            self.native_input_copy_bytes+=(valid-kept)*row_bytes//2
          else:output[:,:,kept:valid].assign(source[:,:,kept:valid]).realize()
      self.input_copy_bytes+=(valid-kept)*row_bytes
      self.input_zero_bytes+=max(0,previous[layer]-valid)*row_bytes
      self.input_reused_bytes+=kept*row_bytes
      valid_rows.append(valid)
    self.input_valid_rows=tuple(valid_rows)
    if prefix_kv is self.draft._prefix_kv:
      self.preserved_prefix_rows=position
      self.prefix_inputs_source=prefix_kv
    return inputs

  def _attention(self, x:Tensor, inputs:tuple[Tensor,...], position:UOp, layer:int) -> Tensor:
    assert self.draft is not None
    draft,c=self.draft,self.draft.config
    name=f'layers.{layer}.self_attn'
    prefix_capacity=self.prefix_capacities[layer]
    rows,total=draft.block_size,prefix_capacity+draft.block_size
    scalar=Tensor(position,device=self.device)
    noise_positions=Tensor.arange(rows,dtype=dtypes.int32).to(self.device)+scalar
    q=draft._norm(draft._linear(x,name+'.q_proj').reshape(1,rows,c.num_attention_heads,c.head_dim),name+'.q_norm').transpose(1,2)
    k=draft._norm(draft._linear(x,name+'.k_proj').reshape(1,rows,c.num_key_value_heads,c.head_dim),name+'.k_norm').transpose(1,2)
    v=draft._linear(x,name+'.v_proj').reshape(1,rows,c.num_key_value_heads,c.head_dim).transpose(1,2)
    q,k=draft._rope(q,noise_positions),draft._rope(k,noise_positions)
    k,v=inputs[2*layer].cat(k,dim=2),inputs[2*layer+1].cat(v,dim=2)
    groups=c.num_attention_heads//c.num_key_value_heads
    if groups!=1:
      k=k.unsqueeze(2).expand(1,c.num_key_value_heads,groups,total,c.head_dim).reshape(1,c.num_attention_heads,total,c.head_dim)
      v=v.unsqueeze(2).expand(1,c.num_key_value_heads,groups,total,c.head_dim).reshape(1,c.num_attention_heads,total,c.head_dim)
    # Cached prefix slots begin at the layer's current window start, not at
    # absolute zero. Only their valid count matters because their RoPE is baked.
    valid=scalar.minimum(c.sliding_window) if c.layer_types[layer]=='sliding_attention' else scalar
    keys=Tensor.arange(total,dtype=dtypes.int32).to(self.device).reshape(1,1,1,-1)
    mask=(keys<valid) | (keys>=prefix_capacity)
    if c.layer_types[layer]=='sliding_attention' and not c.sliding_window_non_causal:
      query=Tensor.arange(rows,dtype=dtypes.int32).to(self.device).reshape(1,1,-1,1)
      mask=mask & ((keys<prefix_capacity) | (keys<=prefix_capacity+query))
    scores=(q@k.transpose(-1,-2))*(c.head_dim**-.5)
    probabilities=mask.where(scores,float('-inf')).float().softmax(-1).cast(q.dtype)
    attended=(probabilities@v).transpose(1,2).reshape(1,rows,c.num_attention_heads*c.head_dim)
    return draft._linear(attended,name+'.o_proj')

  def _run(self, noise_ids:Tensor, inputs:tuple[Tensor,...], position:UOp) -> DraftOutput:
    assert self.draft is not None
    return self.draft._noise_forward(noise_ids,lambda x,layer:self._attention(x,inputs,position,layer)).realize()

  def eager(self, noise_ids:Tensor, prefix_kv:Sequence[PrefixKV|None], position:int) -> DraftOutput:
    """Numerical reference with the same fixed padding and symbolic bounds."""
    inputs=self._inputs(noise_ids,prefix_kv,position)
    return self._run(noise_ids.contiguous().realize(),inputs,self.position.bind(position))

  def __call__(self, noise_ids:Tensor, prefix_kv:Sequence[PrefixKV|None], position:int) -> DraftOutput:
    inputs=self._inputs(noise_ids,prefix_kv,position)
    output=self.jit(noise_ids.contiguous().realize(),inputs,self.position.bind(position))
    self.calls+=1
    return output

  def close(self) -> None:
    if self.closed:return
    try:
      if isinstance(self.device,str):Device[self.device].synchronize()
      if self.jit.captured is not None:self.jit.captured.free_intermediates()
    finally:
      # Drop ownership even when device teardown fails. Reset the known JIT
      # fields directly so cleanup cannot replace the original device error.
      self.jit.cnt=0
      self.jit.captured=None
      self.jit.__dict__.pop('_linears',None)
      self.jit.fxn=None
      self.prefix_inputs=None
      self.prefix_inputs_source=None
      self.preserved_prefix_rows=0
      self.draft=None
      self.closed=True


def _observation_signature(draft:DFlash2) -> tuple:
  return (tuple((name,id(value.uop)) for name,value in sorted(draft.weights.items())),
          tuple((name,id(value)) for name,value in sorted(draft._prepared_linear.items())))


class _ObservationCall:
  def __init__(self, draft:DFlash2, rows:int, max_context:int):
    self.draft:DFlash2|None=draft
    self.rows,self.device=rows,draft.device
    self.position=UOp.variable(f'dflash2_observe_position_{rows}',0,max_context-rows)
    self.signature=_observation_signature(draft)
    self.jit=TinyJit(self._run)
    self.calls,self.closed=0,False

  def _run(self, features:Tensor, position:UOp) -> tuple[Tensor,...]:
    assert self.draft is not None
    projected=self.draft.project_features(features)
    absolute=Tensor.arange(self.rows,dtype=dtypes.int32).to(self.device)+Tensor(position,device=self.device)
    outputs=[projected]
    for layer in range(self.draft.config.num_layers):outputs.extend(self.draft._kv_values(projected,absolute,layer))
    Tensor.realize(*outputs)
    return tuple(outputs)

  def __call__(self, features:Tensor, start_pos:int, valid_rows:int|None=None) -> tuple[Tensor,...]:
    if self.closed or self.draft is None:raise RuntimeError('Prepared observation graph was closed')
    if self.signature!=_observation_signature(self.draft):raise ValueError('Observation weights or projections changed; prepare again')
    output=self.jit(features,self.position.bind(start_pos))
    self.calls+=1
    # Result buffers are reused by TinyJit. Own every new feature/KV tensor before
    # exposing it to persistent caches, including the start_pos==0 replacement.
    # Every row is independent inside this graph. Crop padded rows before
    # copying into owned outputs; no persistent cache ever sees padding.
    rows=self.rows if valid_rows is None else valid_rows
    owned=tuple((value[:,:rows] if index==0 else value[:,:,:rows]).clone() for index,value in enumerate(output))
    Tensor.realize(*owned)
    return owned

  def close(self) -> None:
    if self.closed:return
    try:
      if isinstance(self.device,str):Device[self.device].synchronize()
      if self.jit.captured is not None:self.jit.captured.free_intermediates()
    finally:
      self.jit.cnt=0
      self.jit.captured=None
      self.jit.__dict__.pop('_linears',None)
      self.jit.fxn=None
      self.draft=None
      self.closed=True


class _PreparedObservation:
  def __init__(self, draft:DFlash2, max_context:int, pad_decode_rows:bool=False, native_cache_copy:bool=False,
               bound_prefix_memory:bool=False):
    self.draft:DFlash2|None=draft
    self.max_context=max_context
    self.decode_capacity=min(8,max_context) if pad_decode_rows else None
    self.native_cache_copy=native_cache_copy
    self.bound_prefix_memory=bound_prefix_memory
    self.native_copy_appends=self.native_copy_commands=self.native_copy_bytes=self.native_copy_fallbacks=0
    self.supported_rows=tuple(rows for rows in (*range(1,9),64,128) if rows<=max_context)
    self.signature=_observation_signature(draft)
    self.calls:dict[int,_ObservationCall]={}
    self.closed=False
    c=draft.config
    self.input_bytes_per_row=len(draft.capture_layers)*c.hidden_size*draft.dtype.itemsize
    self.owned_output_bytes_per_row=(c.hidden_size+2*c.num_layers*c.num_key_value_heads*c.head_dim)*draft.dtype.itemsize

  def metadata(self) -> dict:
    return {'enabled':not self.closed,'max_context':self.max_context,'supported_rows':self.supported_rows,
            'bound_prefix_memory':self.bound_prefix_memory,
            'pad_decode_rows':self.decode_capacity is not None,'decode_capacity':self.decode_capacity,
            'native_cache_copy':self.native_cache_copy,'native_copy_appends':self.native_copy_appends,
            'native_copy_commands':self.native_copy_commands,'native_copy_bytes':self.native_copy_bytes,
            'native_copy_fallbacks':self.native_copy_fallbacks,
            'graphs':len(self.calls),'max_graphs':len(self.supported_rows),
            'captured_graphs':sum(call.jit.captured is not None for call in self.calls.values()),
            'graph_calls':{rows:call.calls for rows,call in self.calls.items()},
            'input_copy_bytes_per_row':self.input_bytes_per_row,'owned_output_copy_bytes_per_row':self.owned_output_bytes_per_row,
            'weight_copies':0,'unsupported_rows':'unchanged eager fallback'}

  def append_rows(self, previous:Tensor, appended:Tensor, begin:int, end:int) -> Tensor|None:
    if not self.native_cache_copy:return None
    if self.closed:raise RuntimeError('Prepared observation was closed')
    from tinygrad.llm.kernels.mali_copy import append_rows
    result=append_rows(previous,appended,begin,end)
    if result is None:self.native_copy_fallbacks+=1
    else:
      self.native_copy_appends+=1
      self.native_copy_commands+=1+int(end>begin)
      self.native_copy_bytes+=result.nbytes()
    return result

  def __call__(self, features:Sequence[Tensor]|Tensor, start_pos:int) -> tuple[Tensor,...]|None:
    if self.closed or self.draft is None:raise RuntimeError('Prepared observation was closed')
    if self.signature!=_observation_signature(self.draft):raise ValueError('Observation weights or projections changed; prepare again')
    first=features if isinstance(features,Tensor) else features[0] if features else None
    if first is None or first.ndim!=3 or not isinstance(rows:=first.shape[1],int):return None
    if rows not in self.supported_rows or start_pos<0 or start_pos+rows>self.max_context:return None
    # Concat/cast and a fixed-size owned input buffer stay inside observe timing.
    # The clone also normalizes target-result views from different graph arenas.
    capacity=rows
    if self.decode_capacity is not None and rows<=self.decode_capacity and start_pos+self.decode_capacity<=self.max_context:
      capacity=self.decode_capacity
    combined=self.draft._feature_matrix(features).pad_to((1,capacity,len(self.draft.capture_layers)*self.draft.config.hidden_size)).clone().realize()
    if capacity not in self.calls:self.calls[capacity]=_ObservationCall(self.draft,capacity,self.max_context)
    return self.calls[capacity](combined,start_pos,rows)

  def close(self) -> None:
    if self.closed:return
    errors=[]
    try:
      for call in self.calls.values():
        try:call.close()
        except BaseException as error:errors.append(error)
    finally:
      self.calls.clear()
      self.draft=None
      self.closed=True
    if len(errors)==1:raise errors[0]
    if errors:raise BaseExceptionGroup('Prepared observation graph cleanup failed',errors)
