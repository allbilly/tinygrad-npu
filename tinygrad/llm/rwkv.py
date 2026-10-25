"""RWKV-7 Goose equations: BlinkDL/RWKV-LM/RWKV-v7/rwkv_v7_demo_fast.py.

Weights use llama.cpp's RWKV7 GGUF names and linear projection orientation.
The matrix state and token-shift caches stay in FP32 on CPU and OpenCL.
"""
from __future__ import annotations
import math
from tinygrad import Tensor, nn, UOp, function
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.model import FFNBlock, GatedDeltaTrace, TransformerConfig


class RWKV7Block(FFNBlock):
  linear_names = ('time_mix_receptance','time_mix_key','time_mix_value','time_mix_output',
                  'time_mix_w1','time_mix_w2','time_mix_a1','time_mix_a2','time_mix_v1','time_mix_v2','time_mix_g1','time_mix_g2',
                  'channel_mix_key','channel_mix_value')

  def __init__(self, config:TransformerConfig, layer_id:int):
    self.config,self.layer_id = config,layer_id
    dim = config.dim
    decay_rank,iclr_rank,value_rank,gate_rank = config.rwkv7
    self.attn_norm,self.attn_norm_2 = nn.LayerNorm(dim,config.norm_eps),nn.LayerNorm(dim,config.norm_eps)
    self.time_mix_ln = nn.GroupNorm(config.n_heads,dim,eps=config.head_dim*config.norm_eps)
    self.time_mix_lerp_fused = {'weight':Tensor.zeros(6,1,1,dim)}
    self.channel_mix_lerp_k = {'weight':Tensor.zeros(dim)}
    self.time_mix_w0,self.time_mix_a0 = {'weight':Tensor.zeros(dim)},{'weight':Tensor.zeros(dim)}
    self.time_mix_k_k,self.time_mix_k_a,self.time_mix_r_k = ({'weight':Tensor.zeros(dim)} for _ in range(3))
    self.time_mix_receptance,self.time_mix_key = Linear(dim,dim,bias=False),Linear(dim,dim,bias=False)
    self.time_mix_value,self.time_mix_output = Linear(dim,dim,bias=False),Linear(dim,dim,bias=False)
    self.time_mix_w1,self.time_mix_w2 = Linear(dim,decay_rank,bias=False),Linear(decay_rank,dim,bias=False)
    self.time_mix_a1,self.time_mix_a2 = Linear(dim,iclr_rank,bias=False),Linear(iclr_rank,dim,bias=False)
    self.time_mix_g1,self.time_mix_g2 = Linear(dim,gate_rank,bias=False),Linear(gate_rank,dim,bias=False)
    if layer_id:
      self.time_mix_v0 = {'weight':Tensor.zeros(dim)}
      self.time_mix_v1,self.time_mix_v2 = Linear(dim,value_rank,bias=False),Linear(value_rank,dim,bias=False)
    self.channel_mix_key,self.channel_mix_value = Linear(dim,config.hidden_dim,bias=False),Linear(config.hidden_dim,dim,bias=False)

  def _init_state(self, x:Tensor):
    if not hasattr(self,'recurrent_state'):
      self.recurrent_state = Tensor.zeros(x.shape[0],self.config.n_heads,self.config.head_dim,self.config.head_dim,device=x.device)
      self.cache_attn_x,self.cache_ffn_x = (Tensor.zeros(x.shape[0],1,self.config.dim,device=x.device) for _ in range(2))

  def __call__(self, x:Tensor, start_pos:int|UOp, v_first:Tensor|None=None, *,
               state_trace:GatedDeltaTrace|None=None) -> tuple[Tensor,Tensor]:
    if state_trace is not None:raise ValueError('RWKV7 does not support prefix state tracing')
    self._init_state(x)
    if v_first is None:v_first = Tensor.zeros_like(x)
    @function(precompile=True,allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, v_first:Tensor) -> tuple[Tensor,Tensor]:
      B,T,C = x.shape
      heads,width,capacity = self.config.n_heads,self.config.head_dim,x.max_shape[1]
      position = start_pos if isinstance(start_pos,UOp) else UOp.variable('start_pos',0,self.config.max_context-1).bind(start_pos)
      initial = Tensor(position,device=x.device).eq(0)
      h = self.attn_norm(x)
      window = initial.where(0,self.cache_attn_x).cat(h,dim=1).contiguous()
      shift_store = self.cache_attn_x.uop.store(window[:,T:T+1].uop)
      mixed = h.unsqueeze(0)+(window[:,:T]-h).unsqueeze(0)*self.time_mix_lerp_fused['weight']
      xr,xw,xk,xv,xa,xg = (mixed[i] for i in range(6))
      r,k,v = self.time_mix_receptance(xr),self.time_mix_key(xk),self.time_mix_value(xv)
      decay = (-math.exp(-.5)*(self.time_mix_w2(self.time_mix_w1(xw).tanh())+self.time_mix_w0['weight']).sigmoid()).exp()
      a = (self.time_mix_a2(self.time_mix_a1(xa))+self.time_mix_a0['weight']).sigmoid()
      gate = self.time_mix_g2(self.time_mix_g1(xg).sigmoid())
      kk = (k*self.time_mix_k_k['weight']).reshape(B,T,heads,width)
      kk = kk/kk.square().sum(-1,keepdim=True).sqrt().maximum(1e-12)
      k = k*(1+(a-1)*self.time_mix_k_a['weight'])
      if self.layer_id:v = v+(v_first-v)*(self.time_mix_v2(self.time_mix_v1(xv))+self.time_mix_v0['weight']).sigmoid()
      else:v_first = v
      correction = ((r*k*self.time_mix_r_k['weight']).reshape(B,T,heads,width).sum(-1,keepdim=True)*v.reshape(B,T,heads,width))
      r,k,v,decay,a = (item.reshape(B,T,heads,width) for item in (r,k,v,decay,a))
      if isinstance(T,UOp):r,k,v,decay,a,kk = (item.pad_to((B,capacity,heads,width)) for item in (r,k,v,decay,a,kk))
      state = initial.where(0,self.recurrent_state)
      outputs = []
      for index in range(capacity):
        key,value = k[:,index],v[:,index]
        unit_key,rate = kk[:,index],a[:,index]
        # The rank-one correction uses the old state, before applying decay.
        update = state*decay[:,index].unsqueeze(-2)-(state*unit_key.unsqueeze(-2)).sum(-1,keepdim=True)*(unit_key*rate).unsqueeze(-2)
        update = update+value.unsqueeze(-1)*key.unsqueeze(-2)
        state = (Tensor(T,device=x.device)>index).where(update,state) if isinstance(T,UOp) else update
        outputs.append((state*r[:,index].unsqueeze(-2)).sum(-1))
      store = self.recurrent_state.uop.store(state.uop)
      core = outputs[0].stack(*outputs[1:],dim=1).contiguous()
      # Native projections must finish reading the old shift cache before its write.
      core = Tensor(core.uop.after(store,shift_store))[:,:T].reshape(B*T,C)
      attention = (self.time_mix_ln(core).reshape(B,T,heads,width)+correction).reshape(B,T,C)*gate
      x = x+self.time_mix_output(attention)
      h = self.attn_norm_2(x)
      window = initial.where(0,self.cache_ffn_x).cat(h,dim=1).contiguous()
      ffn_store = self.cache_ffn_x.uop.store(window[:,T:T+1].uop)
      h = h+(window[:,:T]-h)*self.channel_mix_lerp_k['weight']
      result = (x+self.channel_mix_value(self.channel_mix_key(h).relu().square())).contiguous()
      return Tensor(result.uop.after(ffn_store)),v_first
    return _run(x,start_pos,v_first)
