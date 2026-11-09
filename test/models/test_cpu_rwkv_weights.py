"""RWKV7 prepared learned norms and auxiliary parameters remain CPU-only."""
from types import SimpleNamespace
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip_rwkv import prepare_weights
from tinygrad.llm.rwkv import RWKV7Block


def model(block_count:int=1):
  config=SimpleNamespace(dim=8,hidden_dim=12,n_heads=2,head_dim=4,rwkv7=(2,2,2,2),norm_eps=1e-5)
  blocks=[RWKV7Block(config,index) for index in range(block_count)]
  def norm():return SimpleNamespace(weight=Tensor.ones(8,dtype=dtypes.half),bias=Tensor.zeros(8,dtype=dtypes.half))
  return SimpleNamespace(config=config,blk=blocks,token_embd_norm=norm(),output_norm=norm())


def prepared_fields(net):
  result=[]
  for name in ('token_embd_norm','output_norm'):
    for key in ('weight','bias'):result.append((getattr(net,name),key))
  for block in net.blk:
    for name in ('attn_norm','attn_norm_2','time_mix_ln'):
      for key in ('weight','bias'):result.append((getattr(block,name),key))
    for name in ('time_mix_lerp_fused','channel_mix_lerp_k','time_mix_w0','time_mix_a0','time_mix_k_k','time_mix_k_a','time_mix_r_k','time_mix_v0'):
      if hasattr(block,name):result.append((getattr(block,name),'weight'))
  return result


def values(tensor):
  def flatten(value):return sum((flatten(x) for x in value),[]) if isinstance(value,list) else [value]
  return flatten(tensor.tolist())


class TestPrepareRWKV7Weights:
  @pytest.mark.parametrize(('block_count','expected'),[(1,17),(2,31)])
  def test_materializes_only_target_parameters_preserves_values_dtype_and_is_idempotent(self,block_count,expected):
    net=model(block_count)
    targets=prepared_fields(net)
    expected_values=[]
    for index,(owner,key) in enumerate(targets):
      old=owner[key] if isinstance(owner,dict) else getattr(owner,key)
      dtype=dtypes.float if index==0 else dtypes.half
      fill=(index+1)/16
      replacement=Tensor.full(old.shape,fill,device='CPU',dtype=dtype)
      if isinstance(owner,dict):owner[key]=replacement
      else:setattr(owner,key,replacement)
      expected_values.append((replacement.shape,dtype,fill))

    matrix=[(getattr(block,name).weight,getattr(block,name).weight.uop) for block in net.blk for name in block.linear_names
            if hasattr(block,name)]
    for block in net.blk:
      block.recurrent_state=Tensor.zeros(1,2,4,4,device='CPU',dtype=dtypes.float)
      block.cache_attn_x=Tensor.zeros(1,1,8,device='CPU',dtype=dtypes.float)
      block.cache_ffn_x=Tensor.zeros(1,1,8,device='CPU',dtype=dtypes.float)
    caches=[(getattr(block,name),getattr(block,name).uop) for block in net.blk
            for name in ('recurrent_state','cache_attn_x','cache_ffn_x')]

    assert prepare_weights(net)==expected
    for (owner,key),(shape,dtype,fill) in zip(targets,expected_values):
      value=owner[key] if isinstance(owner,dict) else getattr(owner,key)
      assert value.device=='CPU' and value.shape==shape and value.dtype is dtype
      assert value.uop.is_realized and value.uop.contiguous_view_offset()==0
      assert values(value)==[fill]*value.numel()
    assert all(value.uop is uop for value,uop in matrix)
    assert all(value.uop is uop for value,uop in caches)

    prepared=[(owner,key,(owner[key] if isinstance(owner,dict) else getattr(owner,key))) for owner,key in targets]
    assert prepare_weights(net)==expected
    assert all((owner[key] if isinstance(owner,dict) else getattr(owner,key)) is value for owner,key,value in prepared)
    assert all(value.uop is uop for value,uop in matrix)
    assert all(value.uop is uop for value,uop in caches)

  def test_invalid_late_parameter_is_rejected_before_any_field_mutates(self):
    net=model(2)
    first=net.blk[0].time_mix_lerp_fused['weight']
    assert not first.uop.is_realized
    first_uop=first.uop
    net.blk[1].time_mix_k_k['weight']=Tensor.ones(8,device='CPU',dtype=dtypes.int32)
    with pytest.raises(ValueError):prepare_weights(net)
    assert net.blk[0].time_mix_lerp_fused['weight'] is first and first.uop is first_uop and not first.uop.is_realized

  def test_wrong_device_shape_and_non_rwkv_family_are_rejected_atomically(self):
    net=model(2)
    first=net.blk[0].time_mix_lerp_fused['weight']
    first_uop=first.uop
    net.blk[1].time_mix_k_k['weight']=Tensor.ones(8,device='NPY',dtype=dtypes.half)
    with pytest.raises(ValueError):prepare_weights(net)
    assert net.blk[0].time_mix_lerp_fused['weight'] is first and first.uop is first_uop

    net=model(2)
    first=net.blk[0].time_mix_lerp_fused['weight']
    first_uop=first.uop
    net.blk[1].time_mix_k_k['weight']=Tensor.ones(7,device='CPU',dtype=dtypes.half)
    with pytest.raises(ValueError):prepare_weights(net)
    assert net.blk[0].time_mix_lerp_fused['weight'] is first and first.uop is first_uop

    net=model(1)
    net.blk[0]=SimpleNamespace(layer_id=0,config=net.config)
    with pytest.raises(ValueError):prepare_weights(net)

