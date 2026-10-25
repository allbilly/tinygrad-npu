"""CPU-only tests for staged shared projection weight ownership.

Run only after applying the companion staged patch. All packed buffers and NPU
calls are fakes on CPU; Device['ROCKCHIP'] is replaced before configure runs.
"""
from types import SimpleNamespace
import gc, weakref
import numpy as np
import pytest
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from tinygrad.llm.kernels import rockchip
from tinygrad.llm.kernels.rockchip import RockchipLinear, RockchipProjections, configure


def _linear(rng, in_features, out_features):
  layer=nn.Linear(in_features,out_features,bias=False)
  layer.weight=Tensor(rng.normal(0,0.1,(out_features,in_features)).astype(np.float32),device='CPU').realize()
  return layer


def _model():
  rng=np.random.default_rng(461)
  block=SimpleNamespace(
    attn_q=_linear(rng,32,32), attn_k=_linear(rng,32,32), attn_v=_linear(rng,32,32),
    attn_output=_linear(rng,32,32), ffn_gate=_linear(rng,32,48), ffn_up=_linear(rng,32,48),
    ffn_down=_linear(rng,48,32))
  return SimpleNamespace(blk=[block],output=_linear(rng,32,64))


def _cpu_only_backend(monkeypatch, *, fail_joint_once=False):
  prepared=[]
  frees=[]
  class FakeDevice:
    def __getitem__(self,name):
      assert name=='ROCKCHIP'
      return SimpleNamespace(allocator=SimpleNamespace(free_cache=lambda:frees.append(name)))
  monkeypatch.setattr(rockchip,'Device',FakeDevice())
  failure=RuntimeError('injected joint prepare failure')
  failed=False
  def prepare(self):
    nonlocal failed
    prepared.append(self)
    # Q/K/V concatenate to 96 output columns; this is the first shared group.
    if fail_joint_once and self.columns==96 and not failed:
      failed=True
      raise failure
    self.packed=[self.weight.clone().realize()]
  def npu(self,x):
    # Use the original CPU Linear as the fake backend. Keep the patched
    # shared_npu implementation in the call path for delegated members.
    if getattr(self,'shared_projection',None) is not None: return self.shared_npu(x)
    return self.source(x)
  monkeypatch.setattr(RockchipLinear,'prepare',prepare)
  monkeypatch.setattr(RockchipLinear,'npu',npu)
  return prepared,frees,failure


def _expected(model,x,down_x):
  block=model.blk[0]
  names=('attn_q','attn_k','attn_v','attn_output','ffn_gate','ffn_up')
  return {name:getattr(block,name)(x).numpy() for name in names} | {
    'ffn_down':block.ffn_down(down_x).numpy(),'output':model.output(x).numpy()}


def _configure_shared(model):
  return configure(model,prefill='on',decode='on',fuse_projections=True,share_projection_weights=True)


def test_opt_in_prepares_only_combined_groups_and_delegates_on_cpu(monkeypatch):
  model=_model()
  x=Tensor(np.random.default_rng(92).normal(0,0.1,(1,2,32)).astype(np.float32),device='CPU').realize()
  down_x=Tensor(np.random.default_rng(93).normal(0,0.1,(1,2,48)).astype(np.float32),device='CPU').realize()
  expected=_expected(model,x,down_x)
  prepared,frees,_=_cpu_only_backend(monkeypatch)
  result=_configure_shared(model)
  block=model.blk[0]
  qkv,gate_up=block.attn_qkv,block.ffn_gate_up
  assert isinstance(qkv,RockchipProjections) and isinstance(gate_up,RockchipProjections)
  qkv_members=qkv.layers
  gate_members=gate_up.layers
  assert all(member.shared_projection is not None and member.packed==[] for member in (*qkv_members,*gate_members))
  assert all(member.shared_projection[0] is qkv.joint for member in qkv_members)
  assert all(member.shared_projection[0] is gate_up.joint for member in gate_members)
  assert qkv.joint.packed and gate_up.joint.packed
  assert qkv.joint.packed[0].device=='CPU' and gate_up.joint.packed[0].device=='CPU'

  # The standalone attention output, FFN down and vocabulary output retain
  # individual CPU-only packs because they are not members of a fused group.
  standalone=(block.attn_output,block.ffn_down,model.output)
  assert all(isinstance(layer,RockchipLinear) and layer.shared_projection is None and layer.packed for layer in standalone)
  assert all(layer.packed[0].device=='CPU' for layer in standalone)
  assert len(prepared)==5  # two combined layers plus three standalones
  assert len(frees)==2  # one cache release before preparing each combined group

  for group,names in ((qkv,('attn_q','attn_k','attn_v')),(gate_up,('ffn_gate','ffn_up'))):
    combined=group(x)
    assert len(combined)==len(names)
    for name,value in zip(names,combined): np.testing.assert_allclose(value.numpy(),expected[name],rtol=1e-6,atol=1e-6)
    for member,name in zip(group.layers,names):
      # This call must use shared_npu -> joint.npu -> the member's column slice.
      np.testing.assert_allclose(member.npu(x).numpy(),expected[name],rtol=1e-6,atol=1e-6)

  member=block.attn_q
  member.prefill='off'
  member.phase='prefill'
  np.testing.assert_array_equal(member(x).numpy(),expected['attn_q'])  # off still calls the original CPU source
  member.phase=None
  member.prefill='on'

  # Members retain the joint owner, not the RockchipProjections group, so
  # dropping a group does not create an ownership cycle or break delegation.
  owner_ref=weakref.ref(qkv.joint)
  prepared.clear()  # Do not keep an extra strong reference to the joint in the test spy.
  del block.attn_qkv, qkv
  gc.collect()
  assert owner_ref() is not None and member.shared_projection[0] is owner_ref()
  np.testing.assert_allclose(member.npu(x).numpy(),expected['attn_q'],rtol=1e-6,atol=1e-6)
  params=get_parameters(model)
  assert params and all(isinstance(value,Tensor) for value in params)
  assert result['layers']==8
  assert result['shared_projection_layers']==5
  assert result['share_projection_weights'] is True


def test_joint_prepare_failure_restores_individual_packs_and_propagates_original(monkeypatch):
  model=_model()
  prepared,frees,failure=_cpu_only_backend(monkeypatch,fail_joint_once=True)
  with pytest.raises(RuntimeError) as raised:
    _configure_shared(model)
  assert raised.value is failure
  block=model.blk[0]
  members=(block.attn_q,block.attn_k,block.attn_v)
  assert all(member.shared_projection is None for member in members)
  assert all(member.packed and member.packed[0].device=='CPU' for member in members)
  assert all(member in prepared for member in members)
  assert not hasattr(block,'attn_qkv')
  assert frees==['ROCKCHIP','ROCKCHIP']


def test_opt_out_keeps_independent_member_and_joint_packs(monkeypatch):
  model=_model()
  prepared,_,_=_cpu_only_backend(monkeypatch)
  configure(model,prefill='on',decode='on',fuse_projections=True,share_projection_weights=False)
  block=model.blk[0]
  for group in (block.attn_qkv,block.ffn_gate_up):
    assert isinstance(group,RockchipProjections)
    assert group.joint.packed and group.joint.packed[0].device=='CPU'
    assert all(member.shared_projection is None and member.packed and member.packed[0].device=='CPU' for member in group.layers)
  assert all(layer.packed for layer in (block.attn_output,block.ffn_down,model.output))
  assert len(prepared)==10  # eight independent projections plus two joint projections


def test_int8_release_packed_clears_cpu_payloads_and_scales_without_prepare():
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  source=_linear(np.random.default_rng(8),8,4)
  layer=RockchipInt8Linear(source,'on','on')
  layer.packed=[Tensor.ones(4,8,device='CPU')]
  layer.scales=[Tensor.ones(4,device='CPU')]
  layer.release_packed()
  assert layer.packed==[] and layer.scales==[]


@pytest.mark.parametrize('split',[False,True])
def test_int8_member_delegates_and_never_prepares_independent_weights(split):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
  cls=RockchipInt8SplitLinear if split else RockchipInt8Linear
  source=_linear(np.random.default_rng(9),32,4)
  layer=cls(source,'on','on')
  result=Tensor(np.arange(18,dtype=np.float32).reshape(1,2,9),device='CPU').realize()
  joint=SimpleNamespace(npu=lambda x:result)
  layer.shared_projection=(joint,3,4)
  layer.prepare()
  assert layer.packed==[] and layer.scales==[]
  if split:assert layer.parts==[]
  x=Tensor.zeros(1,2,32,device='CPU')
  np.testing.assert_array_equal(layer.npu(x).numpy(),result.numpy()[...,3:7])


@pytest.mark.parametrize('dtype',[np.float16,np.float32])
def test_grouped_int8_weight_codes_and_scales_equal_independent_rows(dtype):
  from tinygrad.helpers import round_up
  from tinygrad.llm.kernels.rockchip_int8 import padded_depth, quantize
  rng=np.random.default_rng(813)
  matrices=[rng.normal(0,.2,(rows,64)).astype(dtype) for rows in (48,96,32)]
  depth=padded_depth(64)
  def encode(values):
    raw,scales=quantize(Tensor(values,device='CPU').realize(),depth)
    return raw.numpy()[round_up(len(values)*4,64):].view(np.int8).reshape(len(values),depth),scales.numpy()
  separate=[encode(values) for values in matrices]
  codes,scales=encode(np.concatenate(matrices,axis=0))
  np.testing.assert_array_equal(codes,np.concatenate([item[0] for item in separate],axis=0))
  np.testing.assert_array_equal(scales,np.concatenate([item[1] for item in separate],axis=0))
