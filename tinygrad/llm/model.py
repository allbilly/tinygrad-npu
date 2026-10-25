from __future__ import annotations
import enum, functools, itertools, math, pathlib, re
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.uop.ops import resolve
from tinygrad.helpers import unwrap, round_up

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@dataclass(frozen=True)
class YaRNConfig:
  factor: float
  orig_ctx_len: int
  beta_fast: float
  beta_slow: float

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn:YaRNConfig|None=None,
                         factors:tuple[float,...]|None=None, mscale:float=1.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  concentration = mscale
  if factors is not None:
    if len(factors)!=dim//2 or any(not math.isfinite(v) or v<=0 for v in factors):raise ValueError('Invalid RoPE frequency factors')
    freqs = freqs / Tensor(factors)
  if yarn is not None and yarn.factor > 1.0:
    concentration = 0.1 * math.log(yarn.factor) + 1.0
    d_half = dim // 2
    low = max(0, math.floor(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_fast * 2 * math.pi)) / math.log(theta)))
    high = min(dim-1, math.ceil(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_slow * 2 * math.pi)) / math.log(theta)))
    interp = 1 - ((Tensor.arange(d_half).float() - low) / max(0.001, high-low)).clamp(0, 1)
    freqs = freqs * interp + (freqs / yarn.factor) * (1 - interp)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return (freqs.cos() * concentration).cat(freqs.sin() * concentration, dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

def gated_activation(gate:Tensor, up:Tensor, *, alpha:float=1.0, limit:float|None=None, up_bias:float=0.0,
                     activation:str='silu') -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  if activation in ('gelu','gelu_tanh'):return gate.gelu(approximate='none' if activation=='gelu' else 'tanh') * (up+up_bias)
  if activation!='silu':raise ValueError(f'Unsupported gated activation: {activation}')
  # TODO: remove the need for this contiguous
  return gate * (gate * alpha).sigmoid().contiguous() * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  gate_lower_bound: float|None = None  # kda: bounded decay gate

@dataclass(frozen=True)
class HCConfig:  # manifold-constrained hyper-connections: count residual streams, mixed by a doubly stochastic matrix
  count: int
  sinkhorn_iters: int
  eps: float

@dataclass(frozen=True)
class IndexerConfig:  # a query attends to the top_k keys picked by an indexer scoring pools of kpool keys
  n_heads: int
  head_dim: int
  top_k: int
  kpool: int
  norm_eps: float

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  yarn: YaRNConfig|None = None
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  attn_output_bias: bool = False
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  expert_proj_bias: bool = False
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  hc: HCConfig|None = None
  indexer: IndexerConfig|None = None
  embedding_scale: float = 1.0
  residual_scale: float = 1.0
  logit_scale: float = 1.0
  rope_factors: tuple[float,...]|None = None
  rope_mscale: float = 1.0
  ffn_activation: str = 'silu'
  post_norm: bool = False
  attn_logit_softcap: float = 0.0
  final_logit_softcap: float = 0.0
  conv_kernel: int = 0
  conv_layers: tuple[bool,...] = ()
  attn_head_gate: bool = False
  sliding_layers: tuple[bool,...] = ()
  swa_rope_dim: int = 0
  swa_rope_theta: float = 0.0
  rwkv7: tuple[int,...] = ()
  ffn_down_bias: bool = False

class HyperConnection:
  def __init__(self, config:TransformerConfig):
    self.count, self.eps, self.iters, self.norm_eps = (hc:=unwrap(config.hc)).count, hc.eps, hc.sinkhorn_iters, config.norm_eps
    self.fn = Linear(self.count * config.dim, (2 + self.count) * self.count, bias=False)
    self.base, self.scale = {"weight": Tensor.zeros((2 + self.count) * self.count)}, {"weight": Tensor.zeros(3)}

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], Tensor]) -> Tensor:
    """mHC: the sublayer sees a pre-weighted sum of the (B, T, hc, D) streams, its output is added back by post onto the streams mixed by comb"""
    base, scale, mix = self.base["weight"], self.scale["weight"], self.fn(xs.flatten(2).rmsnorm(eps=self.norm_eps))
    pre  = (mix[..., :self.count] * scale[0] + base[:self.count]).sigmoid() + self.eps
    post = (mix[..., self.count:2*self.count] * scale[1] + base[self.count:2*self.count]).sigmoid() * 2
    comb = (mix[..., 2*self.count:] * scale[2] + base[2*self.count:]).reshape(*mix.shape[:-1], self.count, self.count).softmax(-1) + self.eps
    # sinkhorn: normalize the columns then the rows, alternating, to make comb doubly stochastic
    for i in range(2 * self.iters - 1): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + self.eps)
    out = sublayer((xs * pre.unsqueeze(-1)).sum(2))
    return out.unsqueeze(2) * post.unsqueeze(-1) + (comb.transpose(-1, -2) @ xs)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm:nn.RMSNorm|nn.LayerNorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    if config.post_norm:
      self.attn_post_norm = nn.RMSNorm(config.dim,config.norm_eps)
      self.ffn_post_norm = nn.RMSNorm(config.dim,config.norm_eps)
    if config.hc is not None: self.hc_attn, self.hc_ffn = HyperConnection(config), HyperConnection(config)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.expert_proj_bias)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.expert_proj_bias)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim, bias=config.expert_proj_bias)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=config.ffn_down_bias)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_limit,
                             up_bias=self.config.swiglu_up_bias,activation=self.config.ffn_activation)
      x_down = self.ffn_down_exps(sel, act.contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x),limit=self.config.swiglu_limit,
                                                   activation=self.config.ffn_activation).contiguous())
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    if self.config.ffn_activation=='silu' and (fused:=getattr(self,'fused_ffn',None)) is not None and fused.enabled(x): return fused(x)
    gate,up = joint(x) if (joint:=getattr(self,'ffn_gate_up',None)) is not None else (self.ffn_gate(x),self.ffn_up(x))
    return self.ffn_down(gated_activation(gate,up,limit=self.config.swiglu_limit,activation=self.config.ffn_activation))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, *, state_trace:GatedDeltaTrace|None=None):
    if state_trace is not None and type(self) is not GatedDeltaNetBlock:raise ValueError('Only GatedDeltaNet supports prefix state tracing')
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, trace:GatedDeltaTrace|None=None):
      if self.config.hc is not None:
        xs = self.hc_attn(x, lambda h: self._attention(self.attn_norm(h), start_pos))
        return self.hc_ffn(xs, lambda h: self._feed_forward(self.ffn_norm(h))).contiguous()
      if trace is not None:
        if not isinstance(self,GatedDeltaNetBlock):raise ValueError('Only GatedDeltaNet supports prefix state tracing')
        attention = self._attention(self.attn_norm(x),start_pos,state_trace=trace)
      else:attention = self._attention(self.attn_norm(x),start_pos)
      if self.config.post_norm:attention=self.attn_post_norm(attention)
      h = x+attention*self.config.residual_scale
      feed_forward = self._feed_forward(self.ffn_norm(h))
      if self.config.post_norm:feed_forward=self.ffn_post_norm(feed_forward)
      return (h+feed_forward*self.config.residual_scale).contiguous()
    return _run(x,start_pos,state_trace) if state_trace is not None else _run(x,start_pos)

class TransformerBlock(FFNBlock):
  decode_attention:Callable[[Tensor,Tensor,int|UOp],Tensor]|None = None
  prefill_attention:Callable[[Tensor,Tensor,int|UOp],Tensor]|None = None
  bounded_mask:bool = False

  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_output_bias)
    if config.attn_head_gate: self.attn_gate = Linear(config.dim, config.n_heads, bias=config.qkv_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    if not self.config.attn_logit_softcap and not self.config.attn_head_gate and not self.config.attn_output_gate and \
       (prepared:=getattr(self,'prepared_qkv',None)) is not None and prepared.enabled(x):
      q,assigned_kv = prepared(x,start_pos,self.cache_kv,self.freqs_cis)
      attention = self.prefill_attention
      assert attention is not None
      B,T,_ = x.shape
      return self.attn_output(attention(q,assigned_kv,start_pos+T).transpose(1,2).reshape(B,T,-1))
    q,k,v = joint(x) if (joint:=getattr(self,'attn_qkv',None)) is not None else (self.attn_q(x),self.attn_k(x),self.attn_v(x))
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    if not self.config.attn_logit_softcap and not self.config.attn_head_gate and B==1 and \
       (attention:=(self.decode_attention if resolve(T==1) else self.prefill_attention)) is not None:
      attn = attention(q,assigned_kv,start_pos+T).transpose(1,2).reshape(B,T,-1)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None and not self.config.attn_head_gate:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if resolve(T != 1) or window:
      if getattr(self,'bounded_mask',False):
        # Constant-capacity index ranges avoid the symbolic arange scan's
        # quadratic temporary. Only the actual query/prefix positions are used.
        query_pos = Tensor.arange(x.max_shape[1],dtype=dtypes.int).to(x.device)[:T] + Tensor(start_pos,device=x.device,dtype=dtypes.int)
        key_pos = Tensor.arange(self.config.max_context,dtype=dtypes.int).to(x.device)[:start_pos+T]
        allowed = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        if window: allowed = allowed & (key_pos.unsqueeze(0) > query_pos.unsqueeze(1)-window)
        mask = allowed.where(0,float('-inf')).cast(x.dtype)
      else:
        mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False)
        mask = mask.triu(start_pos+1) + mask.tril(start_pos-window) if window else mask.triu(start_pos+1)
    if hasattr(self, 'attn_sinks'):
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, -1, 1, 1).expand(1, self.config.n_heads, T, 1)
      if mask is None: mask = Tensor.zeros(1, 1, T, start_pos+T, dtype=x.dtype, buffer=False)
      mask = mask.expand(1, self.config.n_heads, T, start_pos+T).cat(sink_col, dim=-1)
    if self.config.attn_logit_softcap:
      group = self.config.n_heads//self.config.n_kv_heads
      k,v = k.repeat_interleave(group,dim=1),v.repeat_interleave(group,dim=1)
      scores = q.matmul(k.transpose(-2,-1),dtype=dtypes.float32)/math.sqrt(self.config.head_dim)
      scores = (scores/self.config.attn_logit_softcap).tanh()*self.config.attn_logit_softcap
      if mask is not None:scores=scores+mask
      attn = scores.softmax(-1)@v
    else:attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    if self.config.attn_head_gate:attn = attn*self.attn_gate(x).sigmoid().transpose(1,2).unsqueeze(-1)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn, factors=self.config.rope_factors, mscale=self.config.rope_mscale)

class Indexer:
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context = (idx:=unwrap(config.indexer)), config.max_context
    self.attn_q_b, self.attn_k = Linear(config.q_lora_rank, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.proj, self.k_norm = Linear(config.dim, idx.n_heads, bias=False), nn.LayerNorm(idx.head_dim, eps=idx.norm_eps)
    self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kg"):
      self.cache_kg = Tensor.zeros(x.shape[0], round_up(self.max_context, self.idx.kpool), 2, self.idx.head_dim, dtype=dtypes.half, device=x.device)

  def __call__(self, x:Tensor, q:Tensor, start_pos:int|UOp) -> Tensor:
    (B, T, _), idx, S = x.shape, self.idx, start_pos + x.shape[1]
    # cache an indexer key and a pooling gate per token, a pool's key is the gate softmax weighted sum of its keys
    kg, iq = self.k_norm(self.attn_k(x)).stack(self.compressor_gate(x), dim=2), self.attn_q_b(q).reshape(B, T, idx.n_heads, idx.head_dim)
    kg = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(kg.half().uop))).float().reshape(B, -1, idx.kpool, 2, idx.head_dim)
    pooled_k = ((kg[..., 1, :] + self.compressor_ape["weight"]).softmax(2) * kg[..., 0, :]).sum(2).transpose(-1, -2).unsqueeze(1)  # (B, 1, D, pools)
    scores = ((iq @ pooled_k).relu() * (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5).unsqueeze(-1)).sum(2)
    # the query sees the top_k/kpool best complete pools and the tokens of its own pool while incomplete (the tail)
    q_pos, k_pos = (Tensor(start_pos) + Tensor.arange(T)).reshape(T, 1), Tensor.arange(S)
    _, sel = ((Tensor.arange(scores.shape[-1]) + 1) * idx.kpool <= q_pos + 1).where(scores, -math.inf).topk(idx.top_k // idx.kpool)
    # selected incomplete pools add only tail/future tokens, handled by the final tail and causal mask
    picked = scores.zeros_like(dtype=dtypes.bool).scatter(-1, sel, True).repeat_interleave(idx.kpool, dim=-1)[..., :S]
    return ((picked | (k_pos >= (q_pos + 1) // idx.kpool * idx.kpool)) & (k_pos <= q_pos)).where(0.0, -math.inf).reshape(B, 1, T, S).cast(x.dtype)


class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)
    self.indexer = Indexer(config) if config.indexer else None

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if (mask:=self._attn_mask(x, start_pos)) is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _attn_mask(self, x:Tensor, start_pos:int|UOp) -> Tensor|None:  # (1, 1, T, start_pos+T), None: everything is visible
    if self.indexer and self.config.max_context > self.indexer.idx.top_k: return self.indexer(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos)
    T = x.shape[1]
    return Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) if resolve(T != 1) else None

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn, factors=self.config.rope_factors, mscale=self.config.rope_mscale)
    if self.indexer: self.indexer._init_state(x)

class ShortConvolution:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.conv = {'weight':Tensor.zeros(config.dim,config.conv_kernel)}
    self.in_proj = Linear(config.dim,3*config.dim,bias=False)
    self.out_proj = Linear(config.dim,config.dim,bias=False)

  def init_state(self, x:Tensor):
    if not hasattr(self,'cache_conv'):
      self.cache_conv = Tensor.zeros(x.shape[0],self.config.conv_kernel-1,self.config.dim,
                                     dtype=x.dtype,device=x.device)

  def __call__(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    b,c,value = self.in_proj(x).chunk(3,dim=-1)
    position = start_pos if isinstance(start_pos,UOp) else UOp.variable('start_pos',0,self.config.max_context-1).bind(start_pos)
    state = Tensor(position,device=x.device).eq(0).where(0,self.cache_conv)
    rows,kernel = x.shape[1],self.config.conv_kernel
    window = state.cat(b*value,dim=1).contiguous()
    store = self.cache_conv.uop.store(window[:,rows:rows+kernel-1].uop)
    result = functools.reduce(lambda a,b:a+b,
      (window[:,i:i+rows].float()*self.conv['weight'][:,i].float() for i in range(kernel)))
    result = result.cast(x.dtype).contiguous()
    return self.out_proj(c*Tensor(result.uop.after(store)))

class ShortConvBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    self.shortconv = ShortConvolution(config)
  def _init_state(self, x:Tensor):self.shortconv.init_state(x)
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:return self.shortconv(x,start_pos)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, *, state_trace:GatedDeltaTrace|None=None) -> Tensor:
    B, T, _ = x.shape
    if state_trace is not None and getattr(self,'gated_delta_scan',None) is None:
      raise ValueError('Prefix tracing requires a qualified recurrent scan hook')
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    g, a = (alpha.float() + self.ssm_dt["bias"]).reshape(B, T, self.num_v_heads, -1), self.ssm_a.reshape(self.num_v_heads, -1)
    # the decay: log_alpha = softplus(g) * A, or bounded by gate_lower_bound: sigmoid(g * -A) * lb
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    conv_window = conv_state.cat(self.attn_qkv(x).cast(conv_state.dtype), dim=1)
    conv_window = conv_window.pad_to((B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels)).contiguous()
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
    # l2 norm x/sqrt(sum(x²)+1e-6) as in the FLA kernels, then the v heads share the k heads
    q, k = ((z.rmsnorm(eps=1e-6/self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    stores:tuple[UOp,...] = (conv_state_store,)
    if state_trace is not None:
      # This explicit external write survives the block @function boundary. A
      # Python reference to its internal conv_window would retain local params.
      trace_store = state_trace.conv_window.uop.store(conv_window.pad_to(state_trace.conv_window.shape).uop)
      stores += (trace_store,)
    state = Tensor(self.recurrent_state.uop.after(*stores))  # carry the conv write into this graph
    if (scan:=getattr(self,'gated_delta_scan',None)) is not None:
      core = (scan(q,k,v,beta,alpha,state,start_pos,T,history=state_trace.state_history) if state_trace is not None else
              scan(q,k,v,beta,alpha,state,start_pos,T)).transpose(1,2)
    elif self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-2)
      state = initial.where(0, state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()
      if getattr(self,'gated_delta_scan',None) is not None:Tensor.realize(self.conv_state,self.recurrent_state)

class GatedDeltaTrace(NamedTuple):
  """Caller-owned verification-only FP32 prefix states and raw convolution input window."""
  state_history: Tensor  # (capacity, B, H, V, K), after each valid input row
  conv_window: Tensor  # (B, kernel-1+capacity, channels), before convolution


class TargetOutput(NamedTuple):
  logits: Tensor
  hidden: Tensor
  features: tuple[Tensor,...]

  def realize(self) -> TargetOutput:
    """Materialize all outputs together so mutable decoder state is advanced once."""
    tensors:list[Tensor] = []
    for value in (self.logits,self.hidden,*self.features):
      if not any(value is other for other in tensors):tensors.append(value)
    Tensor.realize(*tensors)
    return self


class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    if config.rwkv7:
      from tinygrad.llm.rwkv import RWKV7Block
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = []
    if config.sliding_layers and len(config.sliding_layers)!=config.num_blocks:raise ValueError('Sliding layer mask must cover every block')
    for i in range(config.num_blocks):
      c = dense_config if i < config.leading_dense_blocks else config
      if config.sliding_layers:
        if not config.sliding_layers[i]:c = replace(c,sliding_window=0)
      elif config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      if c.sliding_window and c.swa_rope_dim:c = replace(c,rope_dim=c.swa_rope_dim,rope_theta=c.swa_rope_theta or c.rope_theta)
      self.blk.append(RWKV7Block(c,i) if config.rwkv7 else ShortConvBlock(c) if config.conv_kernel and config.conv_layers[i] else
                      GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.LayerNorm(config.dim,config.norm_eps) if config.rwkv7 else nn.RMSNorm(config.dim, config.norm_eps)
    if config.rwkv7:self.token_embd_norm = nn.LayerNorm(config.dim,config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.gguf_loading:dict|None = None
    self.has_serial_prefill = bool(config.rwkv7) or any(isinstance(b,GatedDeltaNetBlock) for b in self.blk)
    self.has_recurrent_block = self.has_serial_prefill or any(isinstance(b,ShortConvBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._greedy_generation = False
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.greedy_prefill_jit = TinyJit(self.greedy_forward)
    self.greedy_rollout_jit = TinyJit(self.greedy_forward)
    self.embeddings_prefill_jit = TinyJit(lambda x: self.logits_from_embeddings(x,0).realize())
    self.greedy_embeddings_prefill_jit = TinyJit(lambda x: self.greedy_from_embeddings(x,0).realize())
    self.sampled_embeddings_prefill_jit = TinyJit(lambda x, temperature: self.sample_from_embeddings(x,0,temperature).realize())

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float() * self.config.embedding_scale  # (B, T, D)
    return self.logits_from_embeddings(x, start_pos).to(tokens.device)

  def logits_from_embeddings(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """Run the decoder on (B, T, D) embeddings, including projected image tokens."""
    hidden,_ = self._hidden_from_embeddings(x,start_pos)
    # Ordinary generation projects only the last token.
    return self._project_logits(hidden[:,-1:])[:,-1,:]

  def _hidden_from_embeddings(self, x:Tensor, start_pos:int|UOp, capture_layers:tuple[int,...]=(),
                              state_trace:tuple[GatedDeltaTrace|None,...]|None=None) -> tuple[Tensor,tuple[Tensor,...]]:
    if x.ndim != 3 or x.shape[-1] != self.config.dim: raise ValueError('Expected (batch, tokens, model dimension) embeddings')
    if (len(set(capture_layers))!=len(capture_layers)
        or any(type(i) is not int or not 0<=i<len(self.blk) for i in capture_layers)):raise ValueError('Invalid target feature layer indices')
    # Direct forwards change KV/recurrent state without publishing token IDs.
    # Generation coordinators publish their valid prefix after the call succeeds.
    # Callers replaying raw JIT graphs or mutating state buffers must likewise
    # invalidate this marker themselves, since Python is not run on JIT replay.
    self._cached_tokens = []
    features:dict[int,Tensor] = {}
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    if self.config.rwkv7:
      from tinygrad.llm.rwkv import RWKV7Block
      x,v_first = self.token_embd_norm(x),None
      for index,block in enumerate(self.blk):
        assert isinstance(block,RWKV7Block)
        x,v_first = block(x,start_pos,v_first)
        if index in capture_layers:features[index]=x
    else:
      for index,block in enumerate(self.blk):
        trace = state_trace[index] if state_trace is not None else None
        x = block(x,start_pos,state_trace=trace) if trace is not None else block(x,start_pos)
        if index in capture_layers:features[index]=x
    if hc is not None: x = x.mean(2)
    return x,tuple(features[i] for i in capture_layers)

  def _project_logits(self, x:Tensor) -> Tensor:
    logits = self.output(self.output_norm(x) / self.config.logit_scale).to(x.device)
    if self.config.final_logit_softcap:logits=(logits/self.config.final_logit_softcap).tanh()*self.config.final_logit_softcap
    return logits

  def target_forward(self, tokens:Tensor, start_pos:int|UOp, *, capture_layers:tuple[int,...]=(), last_only:bool=False,
                     state_trace:tuple[GatedDeltaTrace|None,...]|None=None) -> TargetOutput:
    """Realized target logits (B,T,V), pre-final-norm hidden states and selected block outputs.

    Capture indices are zero-based decoder block outputs, before final normalization,
    and preserve the requested order. Checkpoint-specific tap indexing is adapted by
    the draft loader. Outputs stay on the decoder device. This advances decoder state;
    the speculation coordinator owns prefix accounting and rejection rollback.
    """
    if type(self) is not Transformer or self.config.hc is not None:raise NotImplementedError('Target features require the base single-stream decoder')
    if tokens.ndim!=2 or not all(tokens.max_shape):raise ValueError('Expected a nonempty target token matrix')
    if isinstance(start_pos,int) and (start_pos<0 or isinstance(tokens.shape[1],int) and start_pos+tokens.shape[1]>self.max_context):
      raise ValueError('Target token block exceeds context')
    if isinstance(start_pos,UOp) and start_pos.vmin<0:raise ValueError('Target position must be nonnegative')
    if state_trace is not None:
      if len(state_trace)!=len(self.blk):raise ValueError('Prefix trace must cover every decoder block')
      for block,trace in zip(self.blk,state_trace):
        if type(block) is TransformerBlock:
          if trace is not None:raise ValueError('Ordinary attention has no recurrent prefix trace')
          continue
        if type(block) is not GatedDeltaNetBlock or trace is None or getattr(block,'gated_delta_scan',None) is None:
          raise ValueError('Prefix trace supports only hooked GatedDeltaNet and ordinary attention blocks')
        history,window = trace
        batch,rows = tokens.max_shape
        capacity = history.max_shape[0] if history.ndim==5 else 0
        if (not 1<=rows<=capacity<=8 or history.shape!=(capacity,batch,block.num_v_heads,block.head_v_dim,block.head_k_dim)
            or window.shape!=(batch,block.ssm_conv_kernel-1+capacity,block.conv_channels)
            or any(value.dtype is not dtypes.float or value.device!=self.token_embd.weight.device for value in trace)):
          raise ValueError('Invalid FP32 GatedDeltaNet prefix trace workspace')
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()*self.config.embedding_scale
    hidden,features = self._hidden_from_embeddings(x,start_pos,capture_layers,state_trace)
    return TargetOutput(self._project_logits(hidden[:,-1:] if last_only else hidden),hidden,features).realize()

  def greedy_forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    logits = self.logits(tokens,start_pos)
    if (greedy:=getattr(self,'rockchip_greedy',None)) is not None: return greedy(logits).reshape(logits.shape[0],1)
    return logits.argmax(-1,keepdim=True)

  def greedy_from_embeddings(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    logits = self.logits_from_embeddings(x,start_pos)
    if (greedy:=getattr(self,'rockchip_greedy',None)) is not None: return greedy(logits).reshape(logits.shape[0],1)
    return logits.argmax(-1,keepdim=True)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    logits = self.logits(tokens,start_pos)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def sample_from_embeddings(self, x:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    """Capture the same Gumbel-max sampling as forward with embedding-input prefill."""
    logits = self.logits_from_embeddings(x,start_pos)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    if self._greedy_generation:
      return (self.greedy_prefill_jit if resolve(tokens.shape[1] != 1) else self.greedy_rollout_jit)(tokens.contiguous(),start_pos)
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    if arch=='gemma4':
      from tinygrad.llm.gemma4 import Gemma4Transformer
      return Gemma4Transformer.from_gguf(gguf,max_context=max_context,realize=realize,shard=shard)
    if arch=='gemma3n':
      from tinygrad.llm.gemma3n import Gemma3nTransformer
      return Gemma3nTransformer.from_gguf(gguf,max_context=max_context,realize=realize,shard=shard)
    if arch=='rwkv7':
      if shard!=1:raise ValueError('RWKV7 currently requires one device')
      n_heads = n_kv_heads = kv[f'{arch}.embedding_length']//kv[f'{arch}.wkv.head_size']
    else:n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    shard_map:dict[str, int] = {}
    if shard > 1:
      heads = n_heads if kv.get(f'{arch}.attention.kv_lora_rank') else n_kv_heads
      assert heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard MLA heads and routed experts while replicating latent projections, KV cache and shared experts
      rules = {**{w: 0 for w in ('token_embd.weight', 'output.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'ffn_gate.weight',
        'ffn_up.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight', 'attn_gate.weight')},
        **{w: 1 for w in ('attn_output.weight', 'ffn_down.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight')}, 'ffn_down_exps.weight':2}
      shard_map = {name: rules[k] for name in entries if (k:=re.sub(r"^blk\.\d+\.", "", name)) in rules}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    stream = realize and shard==1
    if stream and arch=='chatglm':
      from tinygrad.llm.chatglm import split_chatglm_ffn_entries
      entries = split_chatglm_ffn_entries(entries,kv[f'{arch}.feed_forward_length'])
    cast_dtypes = {name:dtypes.half for name in entries if not name.startswith('rope_factors_')} if stream and getenv('HALF',1) else {}
    state_dict = gguf_shard(entries, devices, shard_map,stream=stream,cast_dtypes=cast_dtypes)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) and not k.startswith('rope_factors_') else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    hc = HCConfig(kv[f'{arch}.hyper_connection.count'], kv[f'{arch}.hyper_connection.sinkhorn_iterations'],
                  kv[f'{arch}.hyper_connection.epsilon']) if f'{arch}.hyper_connection.count' in kv else None
    if hc is not None: state_dict = {k.replace('.hc_attn_', '.hc_attn.').replace('.hc_ffn_', '.hc_ffn.'):v for k,v in state_dict.items()}

    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], kv[f'{arch}.attention.indexer.kpool'],
                            kv[f'{arch}.attention.layer_norm_epsilon']) if f'{arch}.attention.indexer.kpool' in kv else None
    if indexer is not None: state_dict = {k.replace('.indexer_compressor_', '.indexer.compressor_'):v for k,v in state_dict.items()}

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    elif arch in ('kimi-linear', 'glm5-next'):
      ssm_layers = tuple(x == 0 for x in n_kv_heads)[:num_blocks]
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True,
                      gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'))
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(n_heads, 1)
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}
    if arch=='gemma2':
      state_dict = {k.replace('post_attention_norm','attn_post_norm').replace('post_ffw_norm','ffn_post_norm'):v for k,v in state_dict.items()}

    conv_kernel = 0
    conv_layers:tuple[bool,...] = ()
    if arch=='lfm2':
      conv_kernel = kv[f'{arch}.shortconv.l_cache']
      conv_layers = tuple(heads==0 for heads in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      state_dict['output_norm.weight'] = state_dict.pop('token_embd_norm.weight')

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    # OCR uses ordinary Llama MHA; its MLA-only qk_rope_head_dim metadata can be zero.
    if arch=='deepseek2-ocr' and rope_dim==0:rope_dim = head_dim
    v_head_dim = kv.get(f'{arch}.attention.value_length_mla',kv.get(f'{arch}.attention.value_length',head_dim))
    if arch=='minicpm3':
      width = state_dict['blk.0.attn_output.weight'].shape[1]
      if width%n_heads:raise ValueError('MiniCPM3 output projection does not match its head count')
      v_head_dim = width//n_heads
      for i in range(num_blocks):
        prefix = f'blk.{i}.'
        fused = state_dict.pop(prefix+'attn_kv_b.weight')
        nope = head_dim-rope_dim
        if fused.shape!=(n_heads*(nope+v_head_dim),kv_lora_rank):raise ValueError('Invalid MiniCPM3 packed K/V projection')
        fused = fused.reshape(n_heads,nope+v_head_dim,kv_lora_rank)
        state_dict[prefix+'attn_k_b.weight'] = fused[:,:nope].transpose(1,2)
        state_dict[prefix+'attn_v_b.weight'] = fused[:,nope:]
    if arch in ('phi3','spark2_5'):
      hidden = kv[f'{arch}.feed_forward_length']
      for i in range(num_blocks):
        layer_prefix = f'blk.{i}.'
        for suffix in ('weight','bias'):
          if (key:=layer_prefix+'attn_qkv.'+suffix) in state_dict:
            pieces = state_dict.pop(key).split((n_heads*head_dim,n_kv_heads*head_dim,n_kv_heads*head_dim),dim=0)
            for name,value in zip(('q','k','v'),pieces):state_dict[layer_prefix+'attn_'+name+'.'+suffix]=value
        if arch=='phi3' and layer_prefix+'ffn_gate.weight' not in state_dict and (up:=state_dict.get(layer_prefix+'ffn_up.weight')) is not None:
          if up.shape[0]!=2*hidden:raise ValueError('Phi3 fused gate/up projection has the wrong shape')
          state_dict[layer_prefix+'ffn_gate.weight'],state_dict[layer_prefix+'ffn_up.weight'] = up.chunk(2,dim=0)
    if arch=='chatglm':
      from tinygrad.llm.chatglm import unpack_chatglm
      state_dict = unpack_chatglm(state_dict,num_blocks,n_heads,n_kv_heads,head_dim,rope_dim,kv[f'{arch}.feed_forward_length'])
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    rope_factors:tuple[float,...]|None = None
    rope_mscale = 1.0
    if kv.get(f'{arch}.rope.scaling.type')=='longrope' or arch=='minicpm3' and 'rope_factors_short.weight' in state_dict:
      original = kv.get(f'{arch}.rope.scaling.original_context_length',kv[f'{arch}.context_length'])
      factor_key = 'rope_factors_'+('long' if max_context>original else 'short')+'.weight'
      if factor_key not in state_dict:raise ValueError('LongRoPE requires its per-frequency scaling tensor')
      factor_values = state_dict[factor_key].tolist()
      if not isinstance(factor_values,list):raise ValueError('LongRoPE factors must be a vector')
      rope_factors = tuple(float(v) for v in factor_values)
      scale = kv[f'{arch}.context_length']/original
      rope_mscale = math.sqrt(1+math.log(scale)/math.log(original)) if scale>1 else 1.0

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch in ('kimi-linear', 'glm5-next','minicpm3'): continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch in ('llama','minicpm','internlm2') or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch in ('llama','minicpm','internlm2') and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    sliding_pattern = kv.get(f'{arch}.attention.sliding_window_pattern',2 if arch in ('gpt-oss','gemma2') else 0)
    sliding_layers:tuple[bool,...] = ()
    if isinstance(sliding_pattern,list):
      if len(sliding_pattern)!=num_blocks or any(type(value) is not bool for value in sliding_pattern):
        raise ValueError('Sliding layer mask must contain one boolean per block')
      sliding_layers,sliding_pattern = tuple(sliding_pattern),0
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads,
      norm_eps=kv[f'{arch}.attention.'+('layer_norm_epsilon' if arch=='rwkv7' else 'layer_norm_rms_epsilon')],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_dim=rope_dim, yarn=yarn,
      v_head_dim=v_head_dim,
      max_context=max_context,
      qk_norm=next((int(value.shape[0]) for name,value in state_dict.items() if name.endswith('.attn_q_norm.weight')),0),
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func',
        ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      ffn_down_bias='blk.0.ffn_down.bias' in state_dict,
      swiglu_alpha=1.702 if arch == 'gpt-oss' else 1.0, swiglu_limit=7.0 if arch == 'gpt-oss' else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if arch == 'gpt-oss' else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=sliding_pattern,sliding_layers=sliding_layers,hc=hc,indexer=indexer,
      swa_rope_dim=kv.get(f'{arch}.rope.dimension_count_swa',0),swa_rope_theta=kv.get(f'{arch}.rope.freq_base_swa',0.0),
      attn_head_gate=arch=='spark2_5',
      embedding_scale=kv.get(f'{arch}.embedding_scale',12.0 if arch=='minicpm3' else
                             math.sqrt(kv[f'{arch}.embedding_length']) if arch=='gemma2' else 1.0),
      residual_scale=kv.get(f'{arch}.residual_scale',1.4/math.sqrt(num_blocks) if arch=='minicpm3' else 1.0),
      logit_scale=kv.get(f'{arch}.logit_scale',kv[f'{arch}.embedding_length']/256 if arch=='minicpm3' else 1.0),
      rope_factors=rope_factors,rope_mscale=rope_mscale,
      ffn_activation='gelu_tanh' if arch=='gemma2' else 'gelu' if arch=='spark2_5' else 'silu',post_norm=arch=='gemma2',
      attn_logit_softcap=kv.get(f'{arch}.attn_logit_softcapping',0.0),final_logit_softcap=kv.get(f'{arch}.final_logit_softcapping',0.0),
      conv_kernel=conv_kernel,conv_layers=conv_layers,
      rwkv7=tuple(kv[f'{arch}.attention.{name}_lora_rank'] for name in ('decay','iclr','value_residual_mix','gate')) if arch=='rwkv7' else ())
    model = Transformer(config)
    model.gguf_loading = {'streamed':stream,'resident_fp16':bool(getenv('HALF',1)),'devices':devices,'bounded_layout_conversions':stream}
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      params = nn.state.get_parameters(model)
      if stream:
        # Compact fused projection views before retaining conversions for the next
        # parameter. OpenCL cannot share CPU-style offset views of these buffers.
        for parameter in params:
          parameter.replace(parameter.contiguous()).realize()
          Device[devices[0]].synchronize()
          Device[devices[0]].allocator.free_cache()
      else:
        for parameter in params:parameter.replace(parameter.contiguous())
        Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def prefill_chunk_size(self, requested:int) -> int:
    if self.config.rwkv7:return 1  # Generic recurrent scan; bounded single-token graphs avoid a large unrolled prefill.
    if self.has_serial_prefill and not amd_custom_kernels_supported(self.token_embd.weight.device):
      return min(requested,256) if all(getattr(b,'gated_delta_scan',None) is not None
                                     for b in self.blk if isinstance(b,GatedDeltaNetBlock)) else 1
    return requested

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    chunk_size = self.prefill_chunk_size(chunk_size)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    self._cached_tokens = tokens[:start_pos]
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      previous = self._greedy_generation
      self._greedy_generation = temperature==0
      # A failed or interrupted call may already have advanced recurrent state.
      # Publish its completed prefix only after execution has succeeded.
      if self.has_recurrent_block:self._cached_tokens = []
      try: out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      finally: self._greedy_generation = previous
      start_pos += n_toks
      self._cached_tokens = tokens[:start_pos]
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
