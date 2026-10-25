"""CPU scan qualification against a scalar reference; no NPU/OpenCL device is used."""
from types import SimpleNamespace
import platform
import numpy as np
import pytest
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.rockchip_recurrent import gated_delta_scan
from test.models.cpu_gdn_reference import FIXTURES, expected_for_call, make_fixture

# Same bounds as the existing FP32 Mali recurrence unit tests. Full-model token
# agreement and quality gates are separate and must not be inferred from these.
NUMERICAL_TOLERANCE = (2e-6,2e-4)


def _assert_numeric(actual, expected):
  atol,rtol=NUMERICAL_TOLERANCE
  np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol)


def _tensors(case):
  d=case['dims']
  b,h,c,k,v,a=(d[name] for name in ('batch','heads','capacity','key_dim','value_dim','alpha_dim'))
  q=Tensor(case['q'],device='CPU').reshape(b,h,c,k).realize()
  key=Tensor(case['k'],device='CPU').reshape(b,h,c,k).realize()
  value=Tensor(case['v'],device='CPU').reshape(b,h,c,v).realize()
  beta=Tensor(case['beta'],device='CPU').reshape(b,h,c).realize()
  alpha_shape=(b,h,c) if a==1 else (b,h,c,a)
  alpha=Tensor(case['alpha'],device='CPU').reshape(*alpha_shape).realize()
  state=Tensor(case['state_initial'],device='CPU').reshape(b,h,v,k).realize()
  return q,key,value,beta,alpha,state


def _expected_arrays(case, expected):
  d=case['dims']
  b,h,c,v,k=(d[name] for name in ('batch','heads','capacity','value_dim','key_dim'))
  out=np.asarray(expected['out_capacity_layout'],dtype=np.float32).reshape(b,h,c,v)
  state=np.asarray(expected['state_final'],dtype=np.float32).reshape(b,h,v,k)
  history=None
  if expected['history_valid_prefix_layout'] is not None:
    history=np.asarray(expected['history_valid_prefix_layout'],dtype=np.float32).reshape(expected['valid_rows'],b,h,v,k)
  return out,state,history


def _assert_output_padding_zero(actual, rows):
  assert not np.any(actual[:,:,rows:,:])


def _tensor_decode_reference(q,k,v,beta,alpha,storage,start):
  state=Tensor(start).eq(0).where(0,storage.clone())
  key,query=k[:,:,0].unsqueeze(-2),q[:,:,0].unsqueeze(-2)
  decayed=state*alpha[:,:,0].unsqueeze(-2)
  delta=(v[:,:,0].unsqueeze(-1)-(decayed*key).sum(-1,keepdim=True))*beta[:,:,0].unsqueeze(-1).unsqueeze(-1)
  updated=decayed+delta*key
  store=storage.uop.store(updated.uop)
  return Tensor((updated*query).sum(-1).unsqueeze(1).contiguous().uop.after(store)).transpose(1,2).realize()


@pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64'),reason='Exact qualified ARM CPU reduction order')
def test_arm_clang_decode_matches_tensor_output_and_state_bitwise():
  from tinygrad.renderer.cstyle import ClangRenderer
  if not isinstance(Device['CPU'].renderer,ClangRenderer):pytest.skip('Exact qualified Clang CPU reduction order')
  run=TinyJit(_tensor_decode_reference)
  for seed in (3508128,3508129,3508130):
    rng=np.random.default_rng(seed)
    values=[rng.normal(0,.1,(1,16,1,128)).astype(np.float32) for _ in range(3)]
    values += [rng.uniform(.1,.9,(1,16,1)).astype(np.float32),rng.uniform(.65,.98,(1,16,1,1)).astype(np.float32)]
    inputs=[Tensor(value,device='CPU').realize() for value in values]
    initial=rng.normal(0,.04,(1,16,128,128)).astype(np.float32)
    reference_state=Tensor(initial.copy(),device='CPU').realize()
    native_state=Tensor(initial.copy(),device='CPU').clone().realize()
    for start in (0,5,5):
      reference_state.assign(Tensor(initial.copy(),device='CPU')).realize()
      native_state.assign(Tensor(initial.copy(),device='CPU')).realize()
      pos=UOp.variable('cpu_gdn_bitwise_start',0,100).bind(start)
      expected=run(*inputs,reference_state,pos).numpy().copy()
      actual=gated_delta_scan(*inputs,native_state,pos,1).numpy().copy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      np.testing.assert_array_equal(native_state.numpy().view(np.uint32),reference_state.numpy().view(np.uint32))


@pytest.mark.parametrize('case',FIXTURES,ids=lambda case:case['name'])
def test_complete_output_state_reset_or_continuation_and_capacity_padding(case):
  q,k,v,beta,alpha,state=_tensors(case)
  expected=expected_for_call(case)
  actual=gated_delta_scan(q,k,v,beta,alpha,state,case['dims']['start'],case['dims']['rows']).numpy()
  expected_out,expected_state,_=_expected_arrays(case,expected)
  _assert_numeric(actual,expected_out)
  _assert_numeric(state.numpy(),expected_state)
  _assert_output_padding_zero(actual,case['dims']['rows'])
  assert expected['reset_applied']==(case['dims']['start']==0)
  assert expected['state_advance_steps_per_batch_head']==case['dims']['rows']


def test_scalar_alpha_prefix_history_layout_and_valid_prefix_for_multiple_batch_heads():
  # Uses B=2,H=3, odd K=5, capacity 5 and only two valid rows. The nonzero
  # initial state proves reset at position zero; history is stored [T,B,H,V,K].
  case=FIXTURES[0]
  q,k,v,beta,alpha,state=_tensors(case)
  d=case['dims']
  b,h,c,v_dim,k_dim=(d[n] for n in ('batch','heads','capacity','value_dim','key_dim'))
  history_capacity=8
  history=Tensor.full((history_capacity,b,h,v_dim,k_dim),float('nan'),device='CPU').realize()
  expected=expected_for_call(case)
  actual=gated_delta_scan(q,k,v,beta,alpha,state,0,d['rows'],history=history).numpy()
  expected_out,expected_state,expected_history=_expected_arrays(case,expected)
  _assert_numeric(actual,expected_out)
  _assert_numeric(state.numpy(),expected_state)
  _assert_numeric(history.numpy()[:d['rows']],expected_history)
  _assert_output_padding_zero(actual,d['rows'])
  assert np.isnan(history.numpy()[d['rows']:]).all()  # wrapper/C only write valid prefixes


def test_per_key_alpha_vector_case_covers_key_dim_multiple_of_four():
  case=FIXTURES[1]
  assert case['dims']['key_dim']%4==0 and case['dims']['alpha_dim']==case['dims']['key_dim']
  q,k,v,beta,alpha,state=_tensors(case)
  expected=expected_for_call(case)
  actual=gated_delta_scan(q,k,v,beta,alpha,state,case['dims']['start'],case['dims']['rows']).numpy()
  expected_out,expected_state,_=_expected_arrays(case,expected)
  _assert_numeric(actual,expected_out)
  _assert_numeric(state.numpy(),expected_state)
  _assert_output_padding_zero(actual,case['dims']['rows'])


def test_full_prefill_capacity_and_model_head_geometry_against_scalar_reference():
  case=make_fixture(name='prefill128',batch=1,heads=2,rows=127,capacity=128,
    key_dim=128,value_dim=128,alpha_dim=1,start=0,seed=9200,history=False)
  q,k,v,beta,alpha,state=_tensors(case)
  expected=expected_for_call(case)
  actual=gated_delta_scan(q,k,v,beta,alpha,state,0,127).numpy()
  expected_out,expected_state,_=_expected_arrays(case,expected)
  _assert_numeric(actual,expected_out)
  _assert_numeric(state.numpy(),expected_state)
  _assert_output_padding_zero(actual,127)


def test_chunk_boundaries_preserve_output_and_final_state_bitwise():
  case=make_fixture(name='chunk_composition',batch=2,heads=2,rows=19,capacity=19,
    key_dim=17,value_dim=9,alpha_dim=1,start=0,seed=9201,history=False)
  q,k,v,beta,alpha,whole_state=_tensors(case)
  whole=gated_delta_scan(q,k,v,beta,alpha,whole_state,0,19).numpy()
  _,_,_,_,_,chunk_state=_tensors(case)
  outputs=[]
  begin=0
  for count in (1,3,7,8):
    parts=[x[:,:,begin:begin+count].contiguous() for x in (q,k,v,beta,alpha)]
    outputs.append(gated_delta_scan(*parts,chunk_state,begin,count).numpy())
    begin+=count
  np.testing.assert_array_equal(np.concatenate(outputs,axis=2),whole)
  np.testing.assert_array_equal(chunk_state.numpy(),whole_state.numpy())


def test_tinyjit_bound_start_and_rows_repeatedly_update_state_and_prefix_history():
  # Match existing recurrent JIT convention: keep shapes/capacity fixed and bind
  # a new runtime position and valid-row count on each invocation.
  batch,heads,capacity,key_dim,value_dim=2,2,8,8,3
  state_initial=[(i%11-5)*0.007 for i in range(batch*heads*value_dim*key_dim)]
  state=Tensor(state_initial,device='CPU').reshape(batch,heads,value_dim,key_dim).realize()
  history=Tensor.full((8,batch,heads,value_dim,key_dim),float('nan'),device='CPU').realize()
  run=TinyJit(lambda q,k,v,beta,alpha,start,rows,hist:
    gated_delta_scan(q,k,v,beta,alpha,state,start,rows,history=hist).realize())
  expected_state=state_initial
  for index,(start,rows) in enumerate(((0,8),(13,5),(20,2),(0,3),(7,6))):
    case=make_fixture(name=f'jit_{index}',batch=batch,heads=heads,rows=rows,capacity=capacity,
      key_dim=key_dim,value_dim=value_dim,alpha_dim=1,start=start,seed=9100+index)
    case['state_initial']=list(expected_state)
    expected=expected_for_call(case)
    expected_out,expected_state_array,expected_history=_expected_arrays(case,expected)
    q,k,v,beta,alpha,_=_tensors(case)
    start_arg=UOp.variable('cpu_gdn_scan_start',0,64).bind(start)
    rows_arg=UOp.variable('cpu_gdn_scan_rows',1,capacity).bind(rows)
    actual=run(q,k,v,beta,alpha,start_arg,rows_arg,history).numpy()
    _assert_numeric(actual,expected_out)
    _assert_numeric(state.numpy(),expected_state_array)
    _assert_numeric(history.numpy()[:rows],expected_history)
    _assert_output_padding_zero(actual,rows)
    expected_state=expected['state_final']
  assert run.captured is not None and run.cnt>=3


def test_rejects_non_cpu_and_non_fp32_operands_before_program_creation():
  case=FIXTURES[0]
  q,k,v,beta,alpha,state=_tensors(case)
  with pytest.raises(ValueError,match='FP32 CPU operands'):
    gated_delta_scan(SimpleNamespace(device='CL'),k,v,beta,alpha,state,0,1)
  half_q=Tensor(case['q'],device='CPU',dtype=dtypes.half).reshape(2,3,5,5).realize()
  with pytest.raises(ValueError,match='FP32 CPU operands'):
    gated_delta_scan(half_q,k,v,beta,alpha,state,0,1)


def test_rejects_invalid_shapes_rows_and_storage_views():
  case=FIXTURES[0]
  q,k,v,beta,alpha,state=_tensors(case)
  state_before=state.numpy().copy()
  for invalid_rows in (0,case['dims']['capacity']+1,UOp.variable('cpu_gdn_bad_rows',0,case['dims']['capacity']).bind(0),
                       UOp.variable('cpu_gdn_too_many_rows',1,case['dims']['capacity']+1).bind(case['dims']['capacity']+1)):
    with pytest.raises(ValueError,match='row count or position'):
      gated_delta_scan(q,k,v,beta,alpha,state,0,invalid_rows)
  with pytest.raises(ValueError,match='row count or position'):
    gated_delta_scan(q,k,v,beta,alpha,state,-1,1)
  with pytest.raises(ValueError,match='row count or position'):
    gated_delta_scan(q,k,v,beta,alpha,state,UOp.variable('cpu_gdn_bad_start',-1,4).bind(-1),1)
  with pytest.raises(ValueError,match='row count or position'):
    gated_delta_scan(q,k,v,beta,alpha,state,2147483648,1)
  np.testing.assert_array_equal(state.numpy(),state_before)
  with pytest.raises(ValueError,match='geometry'):
    gated_delta_scan(q,k,v,beta,Tensor.empty(2,3,5,2,device='CPU').realize(),state,0,1)
  wrong_beta=Tensor(case['beta'],device='CPU').reshape(2,3,5)[:,:,:4].realize()
  with pytest.raises(ValueError,match='geometry'):
    gated_delta_scan(q,k,v,wrong_beta,alpha,state,0,1)
  permuted_base=Tensor.zeros(2,3,5,3,device='CPU').realize()
  state_view=permuted_base.permute(0,1,3,2)
  with pytest.raises(ValueError,match='geometry'):
    gated_delta_scan(q,k,v,beta,alpha,state_view,0,1)


def test_history_rejects_vector_alpha_bad_capacity_dtype_shape_and_view():
  case=FIXTURES[0]
  q,k,v,beta,alpha,state=_tensors(case)
  b,h,c,v_dim,k_dim=(case['dims'][n] for n in ('batch','heads','capacity','value_dim','key_dim'))
  valid_shape=(8,b,h,v_dim,k_dim)
  bad_histories=(
    Tensor.empty(4,b,h,v_dim,k_dim,device='CPU').realize(),  # less than input capacity
    Tensor.empty(9,b,h,v_dim,k_dim,device='CPU').realize(),  # wrapper bound is eight
    Tensor.empty(*valid_shape,device='CPU',dtype=dtypes.half).realize(),
    Tensor.empty(8,b,h,k_dim,v_dim,device='CPU').realize(),
    Tensor.zeros(8,b,h,k_dim,v_dim,device='CPU').realize().permute(0,1,2,4,3))
  for history in bad_histories:
    with pytest.raises(ValueError,match='prefix state history'):
      gated_delta_scan(q,k,v,beta,alpha,state,0,1,history=history)
  vector_case=FIXTURES[1]
  q,k,v,beta,alpha,state=_tensors(vector_case)
  history=Tensor.empty(8,2,2,5,8,device='CPU').realize()
  with pytest.raises(ValueError,match='prefix state history'):
    gated_delta_scan(q,k,v,beta,alpha,state,0,1,history=history)
