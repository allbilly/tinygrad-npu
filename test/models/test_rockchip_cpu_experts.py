"""Explicit hardware qualification for grouped CPU experts beside native NPU dense layers."""
import os,unittest
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip import configure


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestRockchipCPUExperts(unittest.TestCase):
  def test_full_model_native_prefix_decode_and_reset(self):
    self._check_model(4)

  def test_full_model_native_six_row_prefill_and_reset(self):
    self._check_model(6)

  def _check_model(self,prefill_rows):
    config=TransformerConfig(1,1280,896,10,10,1e-5,64,128,10000.,128,128,
      max_context=64,num_experts=4,num_experts_per_tok=2,norm_topk_prob=True,shared_expert_dim=128,shared_expert_gate=False)
    original,target=Transformer(config),Transformer(config)
    rng=np.random.default_rng(271162)
    weights={}
    for name,param in nn.state.get_state_dict(original).items():
      value=rng.normal(0,.02,param.shape).astype(np.float16 if '_exps.weight' in name else np.float32)
      if 'norm' in name:value+=1
      weights[name]=Tensor(value,device='CPU').realize()
    raw={}
    for model in (original,target):
      nn.state.load_state_dict(model,weights)
      prefill=TinyJit(lambda ids,pos,_m=model:_m.logits(ids,pos).realize())
      decode=TinyJit(lambda ids,pos,_m=model:_m.logits(ids,pos).realize())
      model.prefill_jit,model.rollout_jit=prefill,decode
      raw[id(model)]={'prefill':prefill,'decode':decode}
      placement=configure(model,prefill='on',decode='on',precision='int8',tile_columns=24576,native_replay=True,
        fuse_projections=True,share_projection_weights=True,fused_ffn=True,split_down=True,fused_qkv=True,
        cpu_norm=True,cpu_attention=True,cpu_pool=True,cpu_prefill_pool=True,cpu_fast_softmax=True,
        cache_cleaning=True,wake_helper=True,cpu_experts=model is target,
        cpu_expert_prefill_rows=prefill_rows if model is target else 4)
      self.assertEqual(placement['cpu_expert_layers'],3 if model is target else 0)
    position=UOp.variable('expert_hardware_position',0,63)
    for request in range(4):
      ids=Tensor([[(i+request)%63+1 for i in range(17)]],device='CPU')
      a,b=[m.prefill_jit(ids,position.bind(0)).numpy() for m in (original,target)]
      np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
      for step in range(4):
        token=Tensor([[31+request+step]],device='CPU')
        a,b=[m.rollout_jit(token,position.bind(17+step)).numpy() for m in (original,target)]
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    for model in (original,target):
      for phase,jit in raw[id(model)].items():
        self.assertIsNotNone(jit.captured.native)
        self.assertIsNone(jit.captured.reason)
        self.assertGreater(jit.captured.native_runs,0)
        count=jit.captured.native.labels.count('CPU:rockchip_cpu_expert')
        self.assertEqual(count,3 if model is target else 0,(phase,count))


if __name__=='__main__':unittest.main()
