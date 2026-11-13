"""Gemma3n-E2B text decoder: AltUp, LAUREL, sparse MLPs and per-layer inputs.

The reference equations are Gemma3nTextAltUp, Gemma3nTextDecoderLayer and
Gemma3nTextModel. Image/audio towers are separate from this language decoder.
"""
from __future__ import annotations
import math, pathlib
from dataclasses import replace
from tinygrad import Device, Tensor, UOp, dtypes, function, getenv, nn
from tinygrad.llm.gemma4 import Gemma4Block, Gemma4Transformer, SegmentedEmbedding, segment_embedding_entries
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.llm.model import Transformer, TransformerConfig


class Gemma3nBlock(Gemma4Block):
  def __init__(self, config:TransformerConfig, side_dim:int, shared:bool, inputs:int, active:int,
               laurel_rank:int, sparsity:float, correct_scale:bool=True):
    super().__init__(config,side_dim,shared)
    del self.layer_output_scale
    self.inputs,self.active,self.sparsity,self.correct_scale = inputs,active,sparsity,correct_scale
    self.altup_router_norm = nn.RMSNorm(config.dim,config.norm_eps)
    self.altup_router = nn.Linear(config.dim,inputs,bias=False)
    self.altup_predict_coef = nn.Linear(inputs,inputs*inputs,bias=False)
    self.altup_correct_coef = nn.Linear(inputs,inputs,bias=False)
    self.altup_correct_scale = {'weight':Tensor.ones(config.dim)}
    self.laurel_l = nn.Linear(config.dim,laurel_rank,bias=False)
    self.laurel_r = nn.Linear(laurel_rank,config.dim,bias=False)
    self.laurel_post_norm = nn.RMSNorm(config.dim,config.norm_eps)

  def _modalities(self,x:Tensor) -> Tensor:
    return self.altup_router(self.altup_router_norm(x)/self.config.dim).float().tanh()

  def predict(self,states:Tensor) -> Tensor:
    modalities = self._modalities(states[self.active])
    coefficients = self.altup_predict_coef(modalities).reshape(*modalities.shape[:-1],self.inputs,self.inputs).transpose(-1,-2)
    mixed = states.permute(1,2,3,0).matmul(coefficients).permute(3,0,1,2)
    return (mixed+states).contiguous()

  def correct(self,predictions:Tensor,activated:Tensor) -> Tensor:
    coefficients = (self.altup_correct_coef(self._modalities(activated))+1).permute(2,0,1).unsqueeze(-1)
    return (predictions+(activated-predictions[self.active]).unsqueeze(0)*coefficients).contiguous()

  def _sparse_ffn(self,x:Tensor) -> Tensor:
    fused=getattr(self,'fused_gelu_ffn',None)
    if not math.isfinite(self.sparsity) and fused is not None and fused.enabled(x):return fused(x)
    gate,up = joint(x) if (joint:=getattr(self,'ffn_gate_up',None)) is not None else (self.ffn_gate(x),self.ffn_up(x))
    if math.isfinite(self.sparsity):
      mean = gate.mean(-1,keepdim=True)
      std = (gate-mean).square().mean(-1,keepdim=True).sqrt()
      gate = (gate-(mean+std*self.sparsity)).relu()
    activation = getattr(self,'cpu_gelu_activation',None)
    activated = activation(gate,up) if activation is not None else gate.gelu(approximate='tanh')*up
    return self.ffn_down(activated)

  def forward_with_inputs(self, states:Tensor, start_pos:int|UOp, side:Tensor|None, shared:Tensor|None) -> Tensor:
    self._init_state(states[self.active])
    @function(precompile=True)
    def run(self:Gemma3nBlock, states:Tensor, start_pos:int|UOp, side:Tensor|None, shared:Tensor|None) -> Tensor:
      assert side is not None
      predictions = self.predict(states)
      active = predictions[self.active]
      normalized = self.attn_norm(active)
      laurel = normalized+self.laurel_post_norm(self.laurel_r(self.laurel_l(normalized)))
      attended = self.attn_post_norm(self._attend(normalized,start_pos,shared))
      residual = (active+attended+laurel)/math.sqrt(2)
      activated = residual+self.ffn_post_norm(self._sparse_ffn(self.ffn_norm(residual)))
      corrected = self.correct(predictions,activated)
      output = corrected[self.active]
      if self.correct_scale:output = output*self.altup_correct_scale['weight']
      gate = self.inp_gate(output)
      activation = getattr(self,'cpu_gelu_activation',None)
      activated = activation(gate,side) if activation is not None else gate.gelu(approximate='tanh')*side
      extra = self.post_norm(self.proj(activated))
      return corrected[:1].cat(corrected[1:]+extra.unsqueeze(0),dim=0).contiguous()
    return run(self,states,start_pos,side,shared)


class Gemma3nTransformer(Gemma4Transformer):
  def __init__(self, config:TransformerConfig, hiddens:tuple[int,...], sliding:tuple[bool,...], shared_layers:int,
               side_dim:int, side_vocab:int, inputs:int=4, active:int=0, laurel_rank:int=64,
               sparsity:tuple[float,...]=(), embedding_rows:int=16384):
    if inputs<2 or not 0<=active<inputs or side_dim<1 or not 0<=shared_layers<config.num_blocks:
      raise ValueError('Invalid Gemma3n AltUp or side-input configuration')
    if any(len(values)!=config.num_blocks for values in (hiddens,sliding,sparsity)):
      raise ValueError('Gemma3n layer configuration must cover all blocks')
    Transformer.__init__(self,replace(config,num_blocks=0,sliding_layers=()))
    self.config,self.side_dim,self.side_vocab,self.inputs = config,side_dim,side_vocab,inputs
    self.first_shared = config.num_blocks-shared_layers
    seen:set[bool] = set()
    for index in range(config.num_blocks):
      shared = index>=self.first_shared
      if shared and sliding[index] not in seen:raise ValueError('Shared layer has no earlier matching K/V source')
      seen.add(sliding[index])
      layer = replace(config,hidden_dim=hiddens[index],qk_norm=config.head_dim,rope_dim=config.head_dim,
                      rope_theta=config.swa_rope_theta if sliding[index] else config.rope_theta,
                      sliding_window=config.sliding_window if sliding[index] else 0,sliding_layers=(),sliding_window_pattern=0,
                      post_norm=True,ffn_activation='gelu_tanh')
      self.blk.append(Gemma3nBlock(layer,side_dim,shared,inputs,active,laurel_rank,sparsity[index]))
    self.per_layer_token_embd = SegmentedEmbedding(side_vocab,config.num_blocks*side_dim,embedding_rows)
    self.per_layer_model_proj = nn.Linear(config.dim,config.num_blocks*side_dim,bias=False)
    self.per_layer_proj_norm = nn.RMSNorm(side_dim,config.norm_eps)
    self.altup_proj = {'weight':Tensor.empty(inputs-1,config.dim,config.dim)}
    self.altup_unembd_proj = {'weight':Tensor.empty(inputs-1,config.dim,config.dim)}

  def _side_inputs(self,x:Tensor,ids:Tensor|None) -> Tensor|None:
    if ids is not None:ids = ((ids>=0)&(ids<self.side_vocab)).where(ids,0)
    return super()._side_inputs(x,ids)

  @staticmethod
  def _match_magnitude(projected:Tensor,target:Tensor) -> Tensor:
    magnitude = projected.square().mean(-1,keepdim=True).maximum(1e-5).sqrt()
    return projected*target/magnitude

  def _decode(self,x:Tensor,start_pos:int|UOp,side:Tensor|None) -> Tensor:
    target = x.square().mean(-1,keepdim=True).sqrt()
    projections = getattr(self,'altup_input_projections',None)
    branches = [x,*[self._match_magnitude(projections[index](x) if projections is not None else
                   x@self.altup_proj['weight'][index].T,target) for index in range(self.inputs-1)]]
    states = Tensor.stack(*branches,dim=0)
    caches:dict[bool,Tensor] = {}
    for index,block in enumerate(self.blk):
      assert isinstance(block,Gemma3nBlock)
      source = caches.get(bool(block.config.sliding_window)) if block.shared else None
      row = side[:,:,index,:].contiguous() if side is not None else None
      states = block.forward_with_inputs(states,start_pos,row,source)
      if not block.shared:caches[bool(block.config.sliding_window)] = Tensor(block.cache_kv.uop.after(states.uop))
    target = states[0].square().mean(-1,keepdim=True).sqrt()
    projections = getattr(self,'altup_output_projections',None)
    branches = [states[0],*[self._match_magnitude(projections[index](states[index+1]) if projections is not None else
                            states[index+1]@self.altup_unembd_proj['weight'][index].T,target)
                            for index in range(self.inputs-1)]]
    output = Tensor.stack(*branches,dim=0).mean(0)
    logits = self.output(self.output_norm(output[:,-1:]))[:,-1,:]
    return self._softcap_logits(logits)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path,max_context:int|None=None,realize:bool=bool(getenv('REALIZE',0)),
                shard:int=1,mapped_cpu:bool=False) -> tuple[Gemma3nTransformer,dict]:
    if shard!=1:raise ValueError('Gemma3n currently requires a single device')
    kv,entries = gguf_parse(gguf)
    arch = kv['general.architecture']
    if arch!='gemma3n':raise ValueError('Expected a Gemma3n GGUF')
    blocks,dim = kv[arch+'.block_count'],kv[arch+'.embedding_length']
    h = kv[arch+'.feed_forward_length']
    hiddens = tuple(h) if isinstance(h,list) else (h,)*blocks
    sliding = tuple(kv[arch+'.attention.sliding_window_pattern'])
    shared_layers = int(kv[arch+'.attention.shared_kv_layers'])
    for index in range(blocks-shared_layers,blocks):
      for name in ('attn_k.weight','attn_v.weight','attn_k_norm.weight'):entries.pop(f'blk.{index}.'+name,None)
    side_dim = kv[arch+'.embedding_length_per_layer_input']
    side_vocab = entries['per_layer_token_embd.weight'][1][0]
    laurel_rank = entries['blk.0.laurel_l.weight'][1][0]
    entries = segment_embedding_entries(entries,'per_layer_token_embd.weight',16384)
    device = Device.DEFAULT
    cast = {name:dtypes.half for name in entries} if getenv('HALF',1) else {}
    state = gguf_shard(entries,(device,),stream=realize,cast_dtypes=cast,mapped_cpu=mapped_cpu)
    state = {name.replace('post_attention_norm','attn_post_norm').replace('post_ffw_norm','ffn_post_norm'):value
             for name,value in state.items()}
    if 'output.weight' not in state:state['output.weight'] = state['token_embd.weight']
    head = kv[arch+'.attention.key_length']
    config = TransformerConfig(blocks,dim,hiddens[0],kv[arch+'.attention.head_count'],kv[arch+'.attention.head_count_kv'],
                               kv[arch+'.attention.layer_norm_rms_epsilon'],len(kv['tokenizer.ggml.tokens']),head,
                               kv.get(arch+'.rope.freq_base',1000000.),head,head,
                               max_context=min(max_context or kv[arch+'.context_length'],kv[arch+'.context_length']),
                               embedding_scale=math.sqrt(dim),final_logit_softcap=kv.get(arch+'.final_logit_softcapping',30.),
                               sliding_window=kv[arch+'.attention.sliding_window'],swa_rope_theta=kv.get(arch+'.rope.freq_base_swa',10000.))
    model = Gemma3nTransformer(config,hiddens,sliding,shared_layers,side_dim,side_vocab,
                               kv[arch+'.altup.num_inputs'],kv[arch+'.altup.active_idx'],laurel_rank,
                               tuple(kv[arch+'.activation_sparsity_scale']))
    model.gguf_loading = {'streamed':realize,'resident_fp16':bool(getenv('HALF',1)),'devices':(device,),
                          'segmented_per_layer_embeddings':True,'mapped_cpu':mapped_cpu}
    nn.state.load_state_dict(model,state,verbose=False,consume=True,realize=realize)
    return model,kv
