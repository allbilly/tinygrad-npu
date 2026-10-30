"""CPU erf-GELU gate kernel exactness, fallback, and dynamic row coverage."""
import numpy as np
import pytest
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.rockchip_gelu_erf import gated_erf_gelu
from tinygrad.uop.ops import Ops


def _reference(gate,up):
  return (gate.gelu(approximate='none')*up).contiguous().realize()


def _assert_finite_bits(actual,expected):
  actual,expected=actual.numpy(),expected.numpy()
  assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
  np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def _assert_value_classes(actual,expected):
  actual,expected=actual.numpy(),expected.numpy()
  assert actual.shape==expected.shape
  np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
  np.testing.assert_array_equal(np.isposinf(actual),np.isposinf(expected))
  np.testing.assert_array_equal(np.isneginf(actual),np.isneginf(expected))
  finite=np.isfinite(expected)
  np.testing.assert_array_equal(actual[finite].view(np.uint32),expected[finite].view(np.uint32))


@pytest.mark.parametrize('rows,width',[(17,512),(128,6656),(128,10240)])
def test_finite_erf_gelu_matches_exact_tensor_gelu(rows,width):
  rng=np.random.default_rng(rows+width)
  g=rng.normal(0,.6,(1,rows,width)).astype(np.float32)
  u=rng.normal(0,.4,(1,rows,width)).astype(np.float32)
  g.reshape(-1)[:12]=[0.,-0.,1.,-1.,1e-6,-1e-6,.5,-.5,10.,-10.,100.,-100.]
  u.reshape(-1)[:12]=[-0.,1.,-1.,0.,.25,-.25,2.,-2.,-1.,1.,.01,-.01]
  gate,up=Tensor(g,device='CPU').realize(),Tensor(u,device='CPU').realize()
  actual=gated_erf_gelu(gate,up).realize()
  _assert_finite_bits(actual,_reference(gate,up))


@pytest.mark.parametrize('width',[6656,10240])
def test_erf_gelu_noncontiguous_gate_up_slices_match_generic_bits(width):
  joint=Tensor(np.random.default_rng(width).normal(0,.5,(1,17,2*width)).astype(np.float32),device='CPU').realize()
  gate,up=joint[:,:,:width],joint[:,:,width:]
  assert gate.uop.contiguous_view() is None and up.uop.contiguous_view() is None
  _assert_finite_bits(gated_erf_gelu(gate,up),_reference(gate,up))


def test_erf_gelu_nonfinite_outputs_match_nan_and_infinity_classes():
  g=np.random.default_rng(313).normal(0,.5,(1,17,512)).astype(np.float32)
  u=np.random.default_rng(314).normal(0,.5,(1,17,512)).astype(np.float32)
  g.reshape(-1)[:8]=[np.nan,np.inf,-np.inf,0.,-0.,100.,-100.,1.]
  u.reshape(-1)[:8]=[1.,1.,1.,np.inf,-np.inf,0.,-0.,np.nan]
  gate,up=Tensor(g,device='CPU').realize(),Tensor(u,device='CPU').realize()
  _assert_value_classes(gated_erf_gelu(gate,up),_reference(gate,up))


def test_erf_gelu_dynamic_rows_rebind_poisoned_tail_and_zero_padding():
  capacity,width=128,512
  count=UOp.variable('erf_gelu_rows',1,capacity)
  def make_inputs(active,seed):
    gen=np.random.default_rng(seed)
    g=gen.normal(0,.4,(1,capacity,width)).astype(np.float32)
    u=gen.normal(0,.4,(1,capacity,width)).astype(np.float32)
    g[:,active:]=np.nan
    u[:,active:]=np.nan
    return g,u,Tensor(g,device='CPU').realize(),Tensor(u,device='CPU').realize()
  run=TinyJit(lambda g,u,n:gated_erf_gelu(g[:,:n],u[:,:n]).realize())
  for step,active in enumerate((17,65,128,1,17)):
    g,u,gate,up=make_inputs(active,900+step)
    actual=run(gate,up,count.bind(active)).pad_to((1,capacity,width)).realize().numpy()
    expected=np.zeros((1,capacity,width),dtype=np.float32)
    expected[:,:active]=_reference(Tensor(g[:,:active],device='CPU'),Tensor(u[:,:active],device='CPU')).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
  assert run.captured is not None


def test_erf_gelu_fallback_preserves_generic_formula_and_dtype():
  finite_cases=(
    (Tensor(np.random.default_rng(1).normal(size=(1,512)).astype(np.float32),device='CPU'),
     Tensor(np.random.default_rng(2).normal(size=(1,512)).astype(np.float32),device='CPU')),
    (Tensor(np.random.default_rng(3).normal(size=(17,512)).astype(np.float16),device='CPU'),
     Tensor(np.random.default_rng(4).normal(size=(17,512)).astype(np.float16),device='CPU')),
    (Tensor(np.random.default_rng(5).normal(size=(17,513)).astype(np.float32),device='CPU'),
     Tensor(np.random.default_rng(6).normal(size=(17,513)).astype(np.float32),device='CPU')),
    (Tensor(np.array([[.5]],dtype=np.float32),device='CPU'),
     Tensor(np.random.default_rng(7).normal(size=(17,512)).astype(np.float32),device='CPU')),
  )
  for gate,up in finite_cases:
    gate,up=gate.realize(),up.realize()
    actual=gated_erf_gelu(gate,up).realize()
    expected=_reference(gate,up)
    assert actual.dtype is expected.dtype
    if actual.dtype is dtypes.half:
      np.testing.assert_array_equal(actual.numpy().view(np.uint16),expected.numpy().view(np.uint16))
    else: _assert_finite_bits(actual,expected)


def test_erf_gelu_program_is_serial_and_row_one_uses_generic_fallback():
  rng=np.random.default_rng(818)
  gate=Tensor(rng.normal(size=(1,17,512)).astype(np.float32),device='CPU').realize()
  up=Tensor(rng.normal(size=(1,17,512)).astype(np.float32),device='CPU').realize()
  supported=TinyJit(lambda g,u:gated_erf_gelu(g,u).realize())
  supported(gate,up)
  supported(gate,up)
  assert supported.captured is not None
  programs=[call.body for call in supported.captured.linear.src if call.body.op is Ops.PROGRAM
    and call.body.src[0].arg.function_name=='rockchip_cpu_erf_gelu_gate']
  assert len(programs)==1
  source=next(node.arg for node in programs[0].src if node.op is Ops.SOURCE)
  assert 'rockchip_exp2_vector' in source and 'pthread_create' not in source and 'pool_part' not in source
  assert programs[0].arg.outs==(0,) and programs[0].arg.ins==(1,2,3)

  gate1,up1=gate[:,:1,:].realize(),up[:,:1,:].realize()
  fallback=TinyJit(lambda g,u:gated_erf_gelu(g,u).realize())
  actual=fallback(gate1,up1)
  fallback(gate1,up1)
  assert fallback.captured is not None
  fallback_programs=[call.body for call in fallback.captured.linear.src if call.body.op is Ops.PROGRAM
    and call.body.src[0].arg.function_name=='rockchip_cpu_erf_gelu_gate']
  assert fallback_programs==[]
  _assert_finite_bits(actual,_reference(gate1,up1))


def test_erf_gelu_does_not_open_accelerators():
  before={name for name in Device._opened_devices if name.split(':')[0] in ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP')}
  gate=Tensor.ones(1,17,512,device='CPU')
  _=gated_erf_gelu(gate,gate).realize()
  after={name for name in Device._opened_devices if name.split(':')[0] in ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP')}
  assert after==before
