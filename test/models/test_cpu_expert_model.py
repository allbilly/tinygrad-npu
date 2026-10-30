"""Exact grouped CPU experts survive complete model/function and cached JIT boundaries."""
import numpy as np
import pytest
from tinygrad import Device,Tensor,TinyJit,UOp,nn
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip_experts import configure
from tinygrad.llm.kernels.rockchip import RockchipRMSNorm
from tinygrad.uop.ops import Ops


@pytest.mark.parametrize('cpu_norm',[False,True])
def test_complete_model_logits_decode_reset_and_state(cpu_norm):
  config=TransformerConfig(1,1280,896,10,10,1e-5,64,128,10000.,128,128,
    max_context=64,num_experts=4,num_experts_per_tok=2,norm_topk_prob=True,shared_expert_dim=128,shared_expert_gate=False)
  original,target=Transformer(config),Transformer(config)
  rng=np.random.default_rng(271161)
  weights={}
  for name,param in nn.state.get_state_dict(original).items():
    value=rng.normal(0,.02,param.shape).astype(np.float16 if '_exps.weight' in name else np.float32)
    if 'norm' in name:value+=1
    weights[name]=Tensor(value,device='CPU').realize()
  for model in (original,target):nn.state.load_state_dict(model,weights)
  if cpu_norm:
    for model in (original,target):
      model.blk[0].attn_norm=RockchipRMSNorm(model.blk[0].attn_norm)
      model.blk[0].ffn_norm=RockchipRMSNorm(model.blk[0].ffn_norm)
      model.output_norm=RockchipRMSNorm(model.output_norm)
  before={k:id(v) for k,v in nn.state.get_state_dict(target).items()}
  opened=set(Device._opened_devices)
  assert configure(target,enabled=True)==3
  assert {k:id(v) for k,v in nn.state.get_state_dict(target).items()}==before
  forwards=[TinyJit(lambda ids,start,_m=m:_m.logits(ids,start).realize()) for m in (original,target)]
  decodes=[TinyJit(lambda ids,start,_m=m:_m.logits(ids,start).realize()) for m in (original,target)]
  start=UOp.variable('expert_model_position',0,63)
  for request in range(4):
    ids=Tensor([[(i+request)%63+1 for i in range(17)]],device='CPU')
    a,b=[f(ids,start.bind(0)).numpy() for f in forwards]
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    token=Tensor([[31+request]],device='CPU')
    a,b=[f(token,start.bind(17)).numpy() for f in decodes]
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
  programs=[u for u in forwards[1].captured.linear.toposort(enter_calls=True) if u.op is Ops.PROGRAM]
  assert any(u.src[0].arg.function_name=='rockchip_cpu_expert' for u in programs)
  assert not any(d.split(':')[0]=='ROCKCHIP' for d in set(Device._opened_devices)-opened)
