"""CPU reference tests for selecting optional grouped MLA projections."""
import numpy as np
import pytest
from tinygrad import Device,Tensor,nn
from tinygrad.llm.kernels import rockchip,rockchip_grouped_int8
from tinygrad.llm.kernels.rockchip_mla_projections import configure
from test.models import test_minicpm3 as minicpm


@pytest.mark.parametrize('projection',['query','value'])
@pytest.mark.parametrize('precision',['fp16','int8_residual'])
def test_configure_prepares_only_selected_projection(monkeypatch,projection,precision):
  opened=set(Device._opened_devices)
  packed=[]
  projected=[]
  def fp16_pack(weight):
    packed.append(weight)
    return weight.contiguous().realize()
  def fp16_project(x,weight,columns,cores=3):
    projected.append(weight)
    return x.half().float()@weight.float().transpose(-1,-2)
  def int8_pack(weight,*,residual=False):
    assert residual
    packed.append(weight)
    return weight.realize(),Tensor.ones(*weight.shape[:2],device='CPU').float().realize()
  def int8_project(x,weight,scales,columns,*,residual=False):
    assert residual
    projected.append(weight)
    return x@weight.transpose(-1,-2)
  monkeypatch.setattr(rockchip,'pack_grouped_weight',fp16_pack)
  monkeypatch.setattr(rockchip,'grouped_matmul',fp16_project)
  monkeypatch.setattr(rockchip_grouped_int8,'pack_weight',int8_pack)
  monkeypatch.setattr(rockchip_grouped_int8,'matmul',int8_project)

  model=minicpm.TestMiniCPM3().load(minicpm.TestMiniCPM3().fixture())
  state=nn.state.get_state_dict(model)
  identities={name:id(weight) for name,weight in state.items()}
  state_names=set(state)
  hooks=configure(model,prefill=True,precision=precision,projections=projection)
  assert hooks
  assert {name:id(weight) for name,weight in nn.state.get_state_dict(model).items()}==identities
  assert set(nn.state.get_state_dict(model))==state_names
  assert len(packed)==len(hooks)
  for hook in hooks:
    assert hook.projections==projection
    packed_slots=hook._packed()
    scales_slots=hook._scales()
    selected=0 if projection=='query' else 1
    unused=1-selected
    assert packed_slots[unused] is None
    assert scales_slots[unused] is None
    assert packed_slots[selected] is not None
    if precision=='int8_residual': assert scales_slots[selected] is not None
    block=next(b for b in model.blk if getattr(b,'npu_mla_projections',None) is hook)
    weight=block.attn_k_b['weight'] if projection=='query' else block.attn_v_b['weight']
    x=Tensor.full((1,4,2,weight.shape[-1]),.2,device='CPU').realize()
    ref_x=x.half().float() if precision=='fp16' else x
    expected=(ref_x@weight.transpose(-1,-2)).numpy()
    hook.phase='prefill'
    fn=hook.query if projection=='query' else hook.value
    np.testing.assert_array_equal(fn(x).numpy(),expected)
    assert len(projected)==hooks.index(hook)+1
    hook.phase='decode'
    np.testing.assert_array_equal(fn(x).numpy(),(x@weight.transpose(-1,-2)).numpy())
    assert len(projected)==hooks.index(hook)+1
    other=block.attn_v_b['weight'] if projection=='query' else block.attn_k_b['weight']
    other_x=Tensor.full((1,4,2,other.shape[-1]),.2,device='CPU').realize()
    hook.phase='prefill'
    fallback=hook.value if projection=='query' else hook.query
    np.testing.assert_array_equal(fallback(other_x).numpy(),(other_x@other.transpose(-1,-2)).numpy())
    assert len(projected)==hooks.index(hook)+1
  assert all(not any(name.startswith(prefix) for name in nn.state.get_state_dict(model))
             for prefix in ('npu_mla_', '_npu_mla_'))
  assert not any(d.split(':')[0]=='ROCKCHIP' for d in set(Device._opened_devices)-opened)


@pytest.mark.parametrize('selection',['all','key','query_value',''])
def test_invalid_projection_selection_rejected_before_prepare(monkeypatch,selection):
  prepared=[]
  monkeypatch.setattr(rockchip,'pack_grouped_weight',lambda weight:prepared.append(weight))
  model=minicpm.TestMiniCPM3().load(minicpm.TestMiniCPM3().fixture())
  with pytest.raises(ValueError):
    configure(model,prefill=True,projections=selection)
  with pytest.raises(ValueError):
    rockchip.configure(model,npu_mla_prefill=True,npu_mla_projections=selection)
  assert not prepared
