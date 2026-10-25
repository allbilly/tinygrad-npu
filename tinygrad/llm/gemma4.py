"""Gemma4 dense text decoder with per-layer inputs and shared K/V attention.

Equations follow the published Gemma4TextModel and Gemma4TextDecoderLayer.
The auxiliary embedding is split by vocabulary rows, avoiding a single multi-GB
allocation and keeping per-buffer indexing within ordinary integer ranges.
"""
from __future__ import annotations
import math, pathlib
from dataclasses import replace
from tinygrad import Device, Tensor, UOp, dtypes, function, getenv, nn
from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, gguf_parse, gguf_shard
from tinygrad.llm.model import Transformer, TransformerBlock, TransformerConfig, apply_rope, precompute_freqs_cis


class SegmentedEmbedding:
  def __init__(self, vocab:int, width:int, rows:int=16384):
    if min(vocab,width,rows)<1:raise ValueError('Embedding dimensions must be positive')
    self.rows,self.vocab,self.width = rows,vocab,width
    self.chunks = [nn.Embedding(min(rows,vocab-start),width) for start in range(0,vocab,rows)]

  def __call__(self, ids:Tensor) -> Tensor:
    result = None
    for index,chunk in enumerate(self.chunks):
      start,count = index*self.rows,chunk.weight.shape[0]
      assert isinstance(count,int)
      local = (ids-start).clamp(0,count-1)
      values = chunk(local).float()
      values = ((ids>=start)&(ids<start+count)).unsqueeze(-1).where(values,0)
      result = values if result is None else result+values
    assert result is not None
    return result


def segment_embedding_entries(entries:dict, name:str, rows:int) -> dict:
  data,shape,typ = entries[name]
  if len(shape)!=2 or rows<1:raise ValueError('Expected a matrix and positive embedding row chunk')
  vocab,width = shape
  if typ in _GGML_QUANT:block,nbytes = _GGML_QUANT[typ]
  elif typ in _GGML_NATIVE:block,nbytes = 1,_GGML_NATIVE[typ].itemsize
  else:raise ValueError('Unsupported embedding storage type')
  if width%block:raise ValueError('Embedding rows must align to quantization blocks')
  row_bytes = width//block*nbytes
  result = {key:value for key,value in entries.items() if key!=name}
  for index,start in enumerate(range(0,vocab,rows)):
    count = min(rows,vocab-start)
    result[name.removesuffix('.weight')+f'.chunks.{index}.weight'] = (data[start*row_bytes:(start+count)*row_bytes],(count,width),typ)
  return result


class Gemma4Block(TransformerBlock):
  prepared_attention_compatible = False  # Generic prepared attention uses a different score scale.

  def __init__(self, config:TransformerConfig, side_dim:int, shared:bool):
    super().__init__(config)
    self.shared,self.side_dim = shared,side_dim
    self.v_norm = nn.RMSNorm(config.head_dim,config.norm_eps,elementwise_affine=False)
    self.layer_output_scale = {'weight':Tensor.ones(1)}
    if side_dim:
      self.inp_gate = nn.Linear(config.dim,side_dim,bias=False)
      self.proj = nn.Linear(side_dim,config.dim,bias=False)
      self.post_norm = nn.RMSNorm(config.dim,config.norm_eps)
    if shared:
      del self.attn_k,self.attn_v,self.attn_k_norm

  def _init_state(self, x:Tensor):
    if not hasattr(self,'freqs_cis'):
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim,self.config.max_context,self.config.rope_theta,
                                            device=x.device,factors=self.config.rope_factors)
    if not self.shared and not hasattr(self,'cache_kv'):
      self.cache_kv = Tensor.zeros(2,x.shape[0],self.config.n_kv_heads,self.config.max_context,self.config.head_dim,
                                   device=x.device,dtype=dtypes.half).realize()

  def _attend(self, x:Tensor, start_pos:int|UOp, shared:Tensor|None) -> Tensor:
    batch,length,_ = x.shape
    head,heads,kv_heads = self.config.head_dim,self.config.n_heads,self.config.n_kv_heads
    if self.shared:
      query_projection,k_projection,v_projection = self.attn_q(x),None,None
    elif (joint:=getattr(self,'attn_qkv',None)) is not None:query_projection,k_projection,v_projection = joint(x)
    else:query_projection,k_projection,v_projection = self.attn_q(x),self.attn_k(x),self.attn_v(x)
    q = query_projection.reshape(batch,length,heads,head).transpose(1,2)
    q = apply_rope(self.attn_q_norm(q),self.freqs_cis[start_pos:start_pos+length])
    if self.shared:
      if shared is None:raise ValueError('Shared Gemma4 layer needs its source K/V tensor')
      assigned = shared
    else:
      assert k_projection is not None and v_projection is not None
      k = k_projection.reshape(batch,length,kv_heads,head).transpose(1,2)
      v = v_projection.reshape(batch,length,kv_heads,head).transpose(1,2)
      k = apply_rope(self.attn_k_norm(k),self.freqs_cis[start_pos:start_pos+length])
      v = self.v_norm(v)
      store = self.cache_kv[:,:,:,start_pos:start_pos+length,:].uop.store(Tensor.stack(k,v).cast(dtypes.half).uop)
      assigned = Tensor(self.cache_kv.uop.after(store))
    end = start_pos+length
    k,v = assigned[0,:,:,:end,:],assigned[1,:,:,:end,:]
    if heads!=kv_heads:k,v = k.repeat_interleave(heads//kv_heads,dim=1),v.repeat_interleave(heads//kv_heads,dim=1)
    query = Tensor.arange(x.max_shape[1],dtype=dtypes.int).to(x.device)[:length]+Tensor(start_pos,device=x.device,dtype=dtypes.int)
    key = Tensor.arange(self.config.max_context,dtype=dtypes.int).to(x.device)[:end]
    allowed = key.unsqueeze(0)<=query.unsqueeze(1)
    if self.config.sliding_window:allowed = allowed&(key.unsqueeze(0)>query.unsqueeze(1)-self.config.sliding_window)
    # Gemma4 normalizes Q, K and V and uses attention scale 1, not 1/sqrt(head).
    scores = q.matmul(k.transpose(-1,-2),dtype=dtypes.float32)+allowed.where(0,float('-inf'))
    result = scores.softmax(-1).matmul(v,dtype=dtypes.float32).transpose(1,2).reshape(batch,length,-1)
    return self.attn_output(result)

  def forward_with_inputs(self, x:Tensor, start_pos:int|UOp, side:Tensor|None, shared:Tensor|None) -> Tensor:
    self._init_state(x)
    @function(precompile=True)
    def run(self:Gemma4Block, x:Tensor, start_pos:int|UOp, side:Tensor|None, shared:Tensor|None) -> Tensor:
      h = x+self.attn_post_norm(self._attend(self.attn_norm(x),start_pos,shared))
      h = h+self.ffn_post_norm(self._feed_forward(self.ffn_norm(h)))
      if self.side_dim:
        assert side is not None
        h = h+self.post_norm(self.proj(self.inp_gate(h).gelu(approximate='tanh')*side))
      return (h*self.layer_output_scale['weight']).contiguous()
    return run(self,x,start_pos,side,shared)


class Gemma4Transformer(Transformer):
  def __init__(self, config:TransformerConfig, hiddens:tuple[int,...], sliding:tuple[bool,...], swa_head:int,
               shared_layers:int, side_dim:int, side_vocab:int, full_factors:tuple[float,...], embedding_rows:int=16384):
    if len(hiddens)!=config.num_blocks or len(sliding)!=config.num_blocks or not 0<=shared_layers<config.num_blocks:
      raise ValueError('Invalid Gemma4 layer configuration')
    if config.num_experts:raise ValueError('This Gemma4 decoder currently supports dense models')
    super().__init__(replace(config,num_blocks=0,sliding_layers=()))
    self.config,self.side_dim = config,side_dim
    self.first_shared = config.num_blocks-shared_layers
    seen:set[bool] = set()
    for index in range(config.num_blocks):
      window = config.sliding_window if sliding[index] else 0
      head = swa_head if sliding[index] else config.head_dim
      shared = index>=self.first_shared
      if shared and sliding[index] not in seen:raise ValueError('Shared layer has no earlier matching K/V source')
      seen.add(sliding[index])
      layer = replace(config,hidden_dim=hiddens[index],head_dim=head,v_head_dim=head,rope_dim=head,qk_norm=head,
                      rope_theta=config.swa_rope_theta if sliding[index] else config.rope_theta,
                      rope_factors=None if sliding[index] else full_factors,sliding_window=window,sliding_window_pattern=0,
                      sliding_layers=(),ffn_activation='gelu_tanh',post_norm=True)
      self.blk.append(Gemma4Block(layer,side_dim,shared))
    if side_dim:
      self.per_layer_token_embd = SegmentedEmbedding(side_vocab,config.num_blocks*side_dim,embedding_rows)
      self.per_layer_model_proj = nn.Linear(config.dim,config.num_blocks*side_dim,bias=False)
      self.per_layer_proj_norm = nn.RMSNorm(side_dim,config.norm_eps)

  def _side_inputs(self, x:Tensor, ids:Tensor|None) -> Tensor|None:
    if not self.side_dim:return None
    shape = (*x.shape[:-1],self.config.num_blocks,self.side_dim)
    projected = self.per_layer_proj_norm((self.per_layer_model_proj(x)/math.sqrt(self.config.dim)).reshape(shape))
    if ids is None:return projected
    identity = self.per_layer_token_embd(ids).reshape(shape)*math.sqrt(self.side_dim)
    return (projected+identity)/math.sqrt(2)

  def _decode(self, x:Tensor, start_pos:int|UOp, side:Tensor|None) -> Tensor:
    caches:dict[bool,Tensor] = {}
    for index,block in enumerate(self.blk):
      assert isinstance(block,Gemma4Block)
      is_swa = bool(block.config.sliding_window)
      source = caches.get(is_swa) if block.shared else None
      row = side[:,:,index,:].contiguous() if side is not None else None
      x = block.forward_with_inputs(x,start_pos,row,source)
      if not block.shared:
        # The source block call must complete before a sharing layer reads its cache.
        caches[is_swa] = Tensor(block.cache_kv.uop.after(x.uop))
    logits = self.output(self.output_norm(x[:,-1:]))[:,-1,:]
    cap = self.config.final_logit_softcap
    return (logits/cap).tanh()*cap if cap else logits

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    ids = tokens.to(self.token_embd.weight.device)
    x = self.token_embd(ids).float()*self.config.embedding_scale
    return self._decode(x,start_pos,self._side_inputs(x,ids)).to(tokens.device)

  def logits_from_embeddings(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    if x.ndim!=3 or x.shape[-1]!=self.config.dim:raise ValueError('Expected (batch, tokens, model dimension) embeddings')
    return self._decode(x,start_pos,self._side_inputs(x,None))

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None, realize:bool=bool(getenv('REALIZE',0)),
                shard:int=1) -> tuple[Gemma4Transformer,dict]:
    if shard!=1:raise ValueError('Gemma4 currently requires a single device')
    kv,entries = gguf_parse(gguf)
    arch = kv['general.architecture']
    if arch!='gemma4':raise ValueError('Expected a Gemma4 GGUF')
    blocks,dim = kv[arch+'.block_count'],kv[arch+'.embedding_length']
    hidden = kv[arch+'.feed_forward_length']
    hiddens = tuple(hidden) if isinstance(hidden,list) else (hidden,)*blocks
    sliding = tuple(kv[arch+'.attention.sliding_window_pattern'])
    shared_layers = int(kv.get(arch+'.attention.shared_kv_layers',0))
    for index in range(blocks-shared_layers,blocks):
      for name in ('attn_k.weight','attn_v.weight','attn_k_norm.weight'):entries.pop(f'blk.{index}.'+name,None)
    side_dim = kv.get(arch+'.embedding_length_per_layer_input',0)
    rows,side_vocab = 16384,0
    if side_dim:
      side_vocab = entries['per_layer_token_embd.weight'][1][0]
      entries = segment_embedding_entries(entries,'per_layer_token_embd.weight',rows)
    cast = {name:dtypes.half for name in entries if name!='rope_freqs.weight'} if getenv('HALF',1) else {}
    device = Device.DEFAULT
    state = gguf_shard(entries,(device,),stream=realize,cast_dtypes=cast)
    factor_values = state.pop('rope_freqs.weight').tolist()
    if not isinstance(factor_values,list):raise ValueError('Expected a Gemma4 rotary factor vector')
    factors = tuple(float(value) for value in factor_values)
    state = {name.replace('post_attention_norm','attn_post_norm').replace('post_ffw_norm','ffn_post_norm'):value
             for name,value in state.items()}
    if 'output.weight' not in state:state['output.weight'] = state['token_embd.weight']
    head = kv[arch+'.attention.key_length']
    config = TransformerConfig(blocks,dim,hiddens[0],kv[arch+'.attention.head_count'],kv[arch+'.attention.head_count_kv'],
                               kv[arch+'.attention.layer_norm_rms_epsilon'],len(kv['tokenizer.ggml.tokens']),head,
                               kv.get(arch+'.rope.freq_base',1000000.),head,head,
                               max_context=min(max_context or kv[arch+'.context_length'],kv[arch+'.context_length']),
                               num_experts=kv.get(arch+'.expert_count',0),embedding_scale=math.sqrt(dim),
                               final_logit_softcap=kv.get(arch+'.final_logit_softcapping',0.),
                               sliding_window=kv[arch+'.attention.sliding_window'],swa_rope_theta=kv.get(arch+'.rope.freq_base_swa',10000.))
    model = Gemma4Transformer(config,hiddens,sliding,kv[arch+'.attention.key_length_swa'],shared_layers,side_dim,side_vocab,factors,rows)
    model.gguf_loading = {'streamed':realize,'resident_fp16':bool(getenv('HALF',1)),'devices':(device,),
                          'segmented_per_layer_embeddings':bool(side_dim)}
    nn.state.load_state_dict(model,state,verbose=False,consume=True,realize=realize)
    return model,kv
