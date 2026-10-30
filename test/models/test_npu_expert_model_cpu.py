"""The optional MoE hook preserves original routing/state and unmodified arithmetic."""
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn
from tinygrad.llm.model import Transformer,TransformerConfig


class OriginalExpertHook:
  def __init__(self,block):
    self._sources=lambda:(block.ffn_gate_exps,block.ffn_up_exps,block.ffn_down_exps)
    self.calls=0
  def enabled(self,selected,x):return True
  def gate_up(self,selected,x):
    self.calls+=1
    gate,up,_=self._sources()
    return gate(selected,x),up(selected,x)
  def down(self,selected,x):return self._sources()[2](selected,x)


def test_moe_hook_preserves_original_logits_cache_reset_and_state():
  config=TransformerConfig(1,128,64,4,4,1e-5,64,32,10000.,32,32,
    max_context=64,num_experts=4,num_experts_per_tok=2,norm_topk_prob=True,
    shared_expert_dim=64,shared_expert_gate=False)
  original,target=Transformer(config),Transformer(config)
  rng=np.random.default_rng(271451)
  weights={}
  for name,param in nn.state.get_state_dict(original).items():
    values=rng.normal(0,.02,param.shape).astype(np.float16 if '_exps.weight' in name else np.float32)
    if 'norm' in name:values+=1
    weights[name]=Tensor(values,device='CPU').realize()
  for model in (original,target):nn.state.load_state_dict(model,weights)
  before={k:id(v) for k,v in nn.state.get_state_dict(target).items()}
  target.blk[0].npu_experts=OriginalExpertHook(target.blk[0])
  assert {k:id(v) for k,v in nn.state.get_state_dict(target).items()}==before
  prefix=[TinyJit(lambda x,p,_m=m:_m.logits(x,p).realize()) for m in (original,target)]
  decode=[TinyJit(lambda x,p,_m=m:_m.logits(x,p).realize()) for m in (original,target)]
  position=UOp.variable('moe_hook_position',0,63)
  for request in range(4):
    ids=Tensor([[(i+request)%63+1 for i in range(17)]],device='CPU')
    a,b=[f(ids,position.bind(0)).numpy() for f in prefix]
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    for step in range(2):
      token=Tensor([[31+request+step]],device='CPU')
      a,b=[f(token,position.bind(17+step)).numpy() for f in decode]
      np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
  assert target.blk[0].npu_experts.calls>0
