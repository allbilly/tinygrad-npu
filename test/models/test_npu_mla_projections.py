"""CPU reference tests for optional grouped MLA hooks; no NPU device needed."""
import gc,weakref
import numpy as np
import pytest
from tinygrad import Tensor,TinyJit,UOp,nn
from tinygrad.llm.kernels import rockchip
from tinygrad.llm.kernels.rockchip_mla_projections import MLAProjections,configure
from test.models import test_minicpm3 as minicpm


@pytest.fixture
def emulate_npu(monkeypatch):
  calls=[]
  monkeypatch.setattr(rockchip,'pack_grouped_weight',lambda w:w.contiguous().realize())
  def run(x,w,columns,cores=3):
    calls.append(x.shape)
    return x.half().float()@w.float().transpose(-1,-2)
  monkeypatch.setattr(rockchip,'grouped_matmul',run)
  return calls


@pytest.mark.parametrize('capacity',[1,128])
def test_symbolic_rows_poisoned_padding_and_phase_fallback(capacity,emulate_npu):
  rng=np.random.default_rng(271101)
  key=Tensor(rng.normal(0,.1,(4,32,16)).astype(np.float16),device='CPU').realize()
  value=Tensor(rng.normal(0,.1,(4,8,32)).astype(np.float16),device='CPU').realize()
  hook=MLAProjections(key,value,prefill=True,decode=True)
  hook.prepare()
  hook.phase='decode' if capacity==1 else 'prefill'
  for projection,weight in ((hook.query,key),(hook.value,value)):
    columns,depth=weight.shape[-2:]
    rowvar=UOp.variable('npu_mla_rows',1,capacity)
    run=TinyJit(lambda x,n:projection(x[:,:,:n]).pad_to((1,4,capacity,columns)).contiguous().realize())
    for index in range(4):
      rows=1 if capacity==1 else (128 if index%2==0 else 17)
      data=rng.normal(0,.2,(1,4,capacity,depth)).astype(np.float32)
      data[:,:,rows:]=np.nan
      actual=run(Tensor(data,device='CPU').realize(),rowvar.bind(rows)).numpy()
      expected=data[:,:,:rows].astype(np.float16).astype(np.float32)@weight.numpy().astype(np.float32).swapaxes(-1,-2)
      np.testing.assert_allclose(actual[:,:,:rows],expected,atol=2e-5,rtol=2e-4)
      assert np.all(actual[:,:,rows:]==0)
  assert all(shape[1]==capacity for shape in emulate_npu)
  hook.phase=None
  data=Tensor(rng.normal(0,.2,(1,4,3,16)).astype(np.float32),device='CPU').realize()
  np.testing.assert_array_equal(hook.query(data).numpy(),(data@key.transpose(-1,-2)).numpy())


def test_configure_preserves_state_ids_and_rejects_unsupported_set(emulate_npu):
  fixture=minicpm.TestMiniCPM3()
  model=fixture.load(fixture.fixture())
  ids={name:id(w) for name,w in nn.state.get_state_dict(model).items()}
  hooks=configure(model,prefill=True)
  assert len(hooks)==2 and all(h.prefill and not h.decode for h in hooks)
  assert {name:id(w) for name,w in nn.state.get_state_dict(model).items()}==ids
  assert configure(model)==[] and all(not hasattr(b,'npu_mla_projections') for b in model.blk)
  model.blk[-1].attn_v_b['weight']=model.blk[-1].attn_v_b['weight'].float()
  assert configure(model,prefill=True)==[]
  assert all(not hasattr(b,'npu_mla_projections') for b in model.blk)
  with pytest.raises(ValueError):configure(model,prefill=1)


@pytest.mark.parametrize('phase',['prefill','decode'])
def test_disabled_phase_retains_float32_input_projection(phase,emulate_npu):
  key=Tensor.ones(4,32,16,device='CPU').half().realize()
  value=Tensor.ones(4,8,32,device='CPU').half().realize()
  hook=MLAProjections(key,value,prefill=phase!='prefill',decode=phase!='decode')
  hook.prepare()
  hook.phase=phase
  x=Tensor.full((1,4,1,16),.1234567,device='CPU').realize()
  np.testing.assert_array_equal(hook.query(x).numpy(),(x@key.transpose(-1,-2)).numpy())
  assert not emulate_npu


def test_phase_wrapper_selects_and_resets_projection_hook(emulate_npu):
  fixture=minicpm.TestMiniCPM3()
  model=fixture.load(fixture.fixture())
  observed=[]
  def sample(*args,**kwargs):
    observed.append(tuple(b.npu_mla_projections.phase for b in model.blk))
    if kwargs.get('fail'):raise RuntimeError('test phase reset')
  model.greedy_prefill_jit=sample
  model.greedy_rollout_jit=sample
  placement=rockchip.configure(model,prefill='off',decode='off',npu_mla_prefill=True)
  assert placement['npu_mla_projection_blocks']==2 and placement['npu_mla_packed_weight_bytes']>0
  model.greedy_prefill_jit()
  model.greedy_rollout_jit()
  assert observed==[('prefill','prefill'),('decode','decode')]
  with pytest.raises(RuntimeError):model.greedy_prefill_jit(fail=True)
  assert all(b.npu_mla_projections.phase is None for b in model.blk)
  retired=[weakref.ref(b.npu_mla_projections) for b in model.blk]
  rockchip.configure(model,prefill='off',decode='off')
  gc.collect()
  assert all(ref() is None for ref in retired)


@pytest.mark.parametrize('precision',['int8','int8_residual'])
def test_int8_selection_preserves_state_and_phase_fallback(monkeypatch,precision):
  from tinygrad.llm.kernels import rockchip_grouped_int8
  calls=[]
  def pack(weight,*,residual=False):
    assert residual==(precision=='int8_residual')
    return weight.realize(),Tensor.ones(*weight.shape[:2],device='CPU').float().realize()
  def project(x,weight,scales,columns,*,residual=False):
    assert residual==(precision=='int8_residual')
    assert scales.shape==weight.shape[:2] and columns==weight.shape[1]
    calls.append(x.shape)
    return x@weight.transpose(-1,-2)
  monkeypatch.setattr(rockchip_grouped_int8,'pack_weight',pack)
  monkeypatch.setattr(rockchip_grouped_int8,'matmul',project)
  fixture=minicpm.TestMiniCPM3()
  model=fixture.load(fixture.fixture())
  identities={name:id(w) for name,w in nn.state.get_state_dict(model).items()}
  hooks=configure(model,prefill=True,precision=precision)
  assert len(hooks)==2 and all(h.precision==precision for h in hooks)
  assert {name:id(w) for name,w in nn.state.get_state_dict(model).items()}==identities
  hook=hooks[0]
  x=Tensor.full((1,4,3,8),.2,device='CPU').realize()
  expected=(x@model.blk[0].attn_k_b['weight'].transpose(-1,-2)).numpy()
  hook.phase='prefill'
  np.testing.assert_array_equal(hook.query(x).numpy(),expected)
  assert calls==[(4,3,8)]
  hook.phase='decode'
  np.testing.assert_array_equal(hook.query(x).numpy(),expected)
  assert len(calls)==1
  assert configure(model,precision='int8')==[]
  with pytest.raises(ValueError):configure(model,prefill=True,precision='float32')
