"""Whole-model NPU expert wiring against an independent CPU integer recipe."""
import os,unittest
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,nn,dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.model import Transformer,TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_grouped_int8 import _remainder
from tinygrad.runtime.support.rockchip_int8 import padded_depth


class CPUQuantizedExpert:
  def __init__(self,original,activation_residual=False,weight_residual=False):
    self.activation_residual=activation_residual
    self.weight_residual=weight_residual
    values=original.weight.numpy().astype(np.float32)
    scale=np.maximum(np.max(np.abs(values),axis=-1)/np.float32(127),np.float32(1e-8))
    codes=np.rint(values*(np.float32(1)/scale[...,None])).clip(-127,127).astype(np.int32)
    q,s=Tensor(codes,device='CPU').realize(),Tensor(scale,device='CPU').realize()
    self._prepared=lambda:(q,s)
    if weight_residual:
      remainder=(values.astype(np.float64)-codes.astype(np.float64)*scale[...,None].astype(np.float64)).astype(np.float32)
      extra_scale=np.maximum(np.max(np.abs(remainder),axis=-1)/np.float32(127),np.float32(1e-8))
      extra_code=np.rint(remainder*(np.float32(1)/extra_scale[...,None])).clip(-127,127).astype(np.int32)
      q1,s1=Tensor(extra_code,device='CPU').realize(),Tensor(extra_scale,device='CPU').realize()
      self._residual_prepared=lambda:(q1,s1)
    self.depth=values.shape[-1]
  def __call__(self,selected,x):
    rows,capacity=x.shape[1],x.max_shape[1]
    routes=x.shape[2]
    dense=x.pad_to((1,capacity,routes,self.depth)).reshape(capacity*routes,self.depth)
    physical=padded_depth(self.depth)
    payload,scales=quantize(dense,physical)
    codes=payload[round_up(capacity*routes*4,64):].bitcast(dtypes.int8).reshape(1,capacity,routes,physical)[...,:self.depth].int()
    weights,sw=self._prepared()
    ids=selected.pad_to((1,capacity,selected.shape[2]))
    integer=(codes.unsqueeze(-2)@weights[ids].transpose(-1,-2)).contiguous().squeeze(-2)
    result=(integer.float()*scales.reshape(1,capacity,routes,1))*sw[ids]
    if self.activation_residual:
      code0=payload[round_up(capacity*routes*4,64):].bitcast(dtypes.int8).reshape(capacity*routes,physical)
      extra,extra_scales=quantize(_remainder(dense,code0,scales),physical)
      code1=extra[round_up(capacity*routes*4,64):].bitcast(dtypes.int8).reshape(1,capacity,routes,physical)[...,:self.depth].int()
      integer1=(code1.unsqueeze(-2)@weights[ids].transpose(-1,-2)).contiguous().squeeze(-2)
      result=result+(integer1.float()*extra_scales.reshape(1,capacity,routes,1))*sw[ids]
    if self.weight_residual:
      residual_weights,residual_scales=self._residual_prepared()
      integer2=(codes.unsqueeze(-2)@residual_weights[ids].transpose(-1,-2)).contiguous().squeeze(-2)
      third=(integer2.float()*scales.reshape(1,capacity,routes,1)).contiguous()
      result=result.contiguous()+third*residual_scales[ids]
    return result[:,:rows].contiguous()


class CPUQuantizedExpertBlock:
  def __init__(self,block,cpu_down=False,activation_residual=False,weight_residual=False):
    layers=tuple(CPUQuantizedExpert(getattr(block,name),activation_residual,weight_residual)
                 if not (cpu_down and name=='ffn_down_exps') else block.ffn_down_exps
                 for name in ('ffn_gate_exps','ffn_up_exps','ffn_down_exps'))
    self._layers=lambda:layers
  def enabled(self,selected,x):return True
  def gate_up(self,selected,x):return self._layers()[0](selected,x),self._layers()[1](selected,x)
  def down(self,selected,x):return self._layers()[2](selected,x)


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestRockchipNPUExperts(unittest.TestCase):
  def test_full_model_integer_reference_changed_prefix_decode_reset_and_state(self):
    self._check_model(cpu_down=False)

  def test_full_model_hybrid_cpu_down_reference(self):
    self._check_model(cpu_down=True)

  def test_full_model_activation_residual_reference(self):
    self._check_model(cpu_down=False,activation_residual=True)

  def test_full_model_residual_hybrid_cpu_down_reference(self):
    self._check_model(cpu_down=True,activation_residual=True)

  def test_full_model_weight_residual_reference(self):
    self._check_model(cpu_down=False,activation_residual=True,weight_residual=True)

  def test_full_model_weight_residual_hybrid_cpu_down_reference(self):
    self._check_model(cpu_down=True,activation_residual=True,weight_residual=True)

  def _check_model(self,cpu_down,activation_residual=False,weight_residual=False):
    config=TransformerConfig(1,1280,896,10,10,1e-5,64,128,10000.,128,128,
      max_context=64,num_experts=4,num_experts_per_tok=2,norm_topk_prob=True,
      shared_expert_dim=128,shared_expert_gate=False)
    reference,target=Transformer(config),Transformer(config)
    rng=np.random.default_rng(271452)
    weights={}
    for name,param in nn.state.get_state_dict(reference).items():
      value=rng.normal(0,.02,param.shape).astype(np.float16 if '_exps.weight' in name else np.float32)
      if 'norm' in name:value+=1
      weights[name]=Tensor(value,device='CPU').realize()
    for model in (reference,target):nn.state.load_state_dict(model,weights)
    reference.blk[0].npu_experts=CPUQuantizedExpertBlock(reference.blk[0],cpu_down=cpu_down,
      activation_residual=activation_residual,weight_residual=weight_residual)
    before={k:id(v) for k,v in nn.state.get_state_dict(target).items() if '_exps.' in k}
    raw={}
    for model in (reference,target):
      model.prefill_jit=TinyJit(lambda ids,pos,_m=model:_m.logits(ids,pos).realize())
      model.rollout_jit=TinyJit(lambda ids,pos,_m=model:_m.logits(ids,pos).realize())
      raw[id(model)]={'prefill':model.prefill_jit,'decode':model.rollout_jit}
      placement=configure(model,prefill='on',decode='on',precision='int8',tile_columns=24576,native_replay=True,
        fuse_projections=True,share_projection_weights=True,fused_ffn=True,split_down=True,fused_qkv=True,
        cpu_norm=True,cpu_attention=True,cpu_pool=True,cpu_prefill_pool=True,cpu_fast_softmax=True,
        cache_cleaning=True,wake_helper=True,npu_experts=model is target,cpu_experts=model is target and cpu_down,
        npu_expert_cpu_down_layers=(0,) if model is target and cpu_down else (),
        npu_expert_activation_residual=activation_residual and model is target,
        npu_expert_weight_residual=weight_residual and model is target)
      self.assertEqual(placement['npu_expert_blocks'],int(model is target))
      self.assertEqual(placement['cpu_expert_layers'],1 if model is target and cpu_down else 0)
    state=nn.state.get_state_dict(target)
    self.assertEqual({k:id(v) for k,v in state.items() if '_exps.' in k},before)
    self.assertFalse(any('.npu_experts.' in k for k in state))
    position=UOp.variable('npu_expert_model_position',0,63)
    for request in range(4):
      ids=Tensor([[(i+request)%63+1 for i in range(17)]],device='CPU')
      a,b=[m.prefill_jit(ids,position.bind(0)).numpy() for m in (reference,target)]
      np.testing.assert_allclose(b,a,atol=2e-5,rtol=2e-5)
      for step in range(3):
        token=Tensor([[31+request+step]],device='CPU')
        a,b=[m.rollout_jit(token,position.bind(17+step)).numpy() for m in (reference,target)]
        np.testing.assert_allclose(b,a,atol=2e-5,rtol=2e-5,err_msg=f'request={request} decode_step={step}')
    for model in (reference,target):
      for phase,jit in raw[id(model)].items():
        self.assertIsNotNone(jit.captured.native)
        self.assertIsNone(jit.captured.reason)
        self.assertGreater(jit.captured.native_runs,0)
        labels=jit.captured.native.labels
        gpu_experts=sum(label.startswith('ROCKCHIP:') and label.rsplit(':',1)[-1].startswith(
          ('rockchip_bucketed_int8_','rockchip_routed_int8_')) for label in labels)
        self.assertEqual(gpu_experts,(1 if cpu_down else 2)*(2 if weight_residual else 1) if model is target else 0,(phase,labels))
        maps=labels.count('CPU:rockchip_expert_route_map')
        self.assertEqual(maps,(1 if cpu_down else 2) if model is target and phase=='prefill' else 0)
