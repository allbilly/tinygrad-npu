"""Pure-stdlib scalar reference and deterministic fixtures for staged scan.c.

The module intentionally imports no tinygrad, NumPy, test framework, or C binding.
Tensor layouts and return-code cases mirror cpu_gated_delta_scan in scan.c. Fixtures
are rounded to binary32 on storage; expected Python-scalar results are numeric
references, not bitwise float32 oracles for the C vector reduction order.
"""
from __future__ import annotations
import copy, random, struct


def _f32(value:float) -> float:
  """Round a Python scalar to the C interface's IEEE binary32 storage value."""
  return struct.unpack('<f',struct.pack('<f',value))[0]


def _f32_values(rng:random.Random, count:int, lo:float, hi:float) -> list[float]:
  return [_f32(rng.uniform(lo,hi)) for _ in range(count)]


def make_fixture(*, name:str, batch:int, heads:int, rows:int, capacity:int,
                 key_dim:int, value_dim:int, alpha_dim:int, start:int, seed:int,
                 history:bool=True) -> dict:
  """Make flat row-major C buffers, with deliberately conspicuous padded rows."""
  rng=random.Random(seed)
  steps=batch*heads*capacity
  q=_f32_values(rng,steps*key_dim,-0.2,0.2)
  k=_f32_values(rng,steps*key_dim,-0.2,0.2)
  v=_f32_values(rng,steps*value_dim,-0.2,0.2)
  beta=_f32_values(rng,steps,0.1,0.9)
  alpha=_f32_values(rng,steps*alpha_dim,0.65,0.98)
  state=_f32_values(rng,batch*heads*value_dim*key_dim,-0.04,0.04)
  # A nonzero initial state makes start==0 reset observable. Large but finite
  # padded inputs make accidental processing of capacity tail obvious.
  for b in range(batch):
    for h in range(heads):
      for t in range(rows,capacity):
        step=(b*heads+h)*capacity+t
        q[step*key_dim:(step+1)*key_dim]=[1.25]*key_dim
        k[step*key_dim:(step+1)*key_dim]=[-1.5]*key_dim
        v[step*value_dim:(step+1)*value_dim]=[2.0]*value_dim
        beta[step]=0.95
        alpha[step*alpha_dim:(step+1)*alpha_dim]=[0.7]*alpha_dim
  return {'name':name,'dims':{'batch':batch,'heads':heads,'rows':rows,'capacity':capacity,
                              'key_dim':key_dim,'value_dim':value_dim,'alpha_dim':alpha_dim,'start':start},
          'q':q,'k':k,'v':v,'beta':beta,'alpha':alpha,'state_initial':state,
          'history_enabled':history,'seed':seed,'dtype':'float32','initial_state_is_nonzero':True}


FIXTURES = (
  make_fixture(name='scalar_alpha_odd_key_reset_padded_multi_bh',batch=2,heads=3,rows=2,capacity=5,
               key_dim=5,value_dim=3,alpha_dim=1,start=0,seed=8041),
  make_fixture(name='per_key_alpha_vector_key_existing_state_padded_multi_bh',batch=2,heads=2,rows=3,capacity=4,
               key_dim=8,value_dim=5,alpha_dim=8,start=13,seed=8042),
  make_fixture(name='per_key_alpha_odd_key_reset_padded',batch=1,heads=2,rows=2,capacity=3,
               key_dim=7,value_dim=4,alpha_dim=7,start=0,seed=8043,history=False),
)


def validate_call(*, start:int, rows:int, batch:int, heads:int, capacity:int,
                  key_dim:int, value_dim:int, alpha_dim:int,
                  null_pointers:tuple[str,...]=()) -> int:
  """Return the staged C function's status (0 valid, 1 rejected)."""
  pointer_names=('out','q','k','v','beta','alpha','state')
  if any(name not in pointer_names for name in null_pointers): raise ValueError('unknown pointer name')
  if null_pointers or start<0 or rows<1 or rows>capacity or capacity>256 or batch<1 or heads<1 \
     or key_dim<1 or key_dim>256 or value_dim<1 or value_dim>256 or alpha_dim not in (1,key_dim):
    return 1
  return 0


_VALID_CALL={'start':0,'rows':3,'batch':2,'heads':2,'capacity':5,'key_dim':8,'value_dim':4,'alpha_dim':1}
INVALID_CALLS = tuple(
  {'name':name,'overrides':overrides,'expected_status':1}
  for name,overrides in (
    ('null_out',{'null_pointers':('out',)}),('null_q',{'null_pointers':('q',)}),
    ('null_k',{'null_pointers':('k',)}),('null_v',{'null_pointers':('v',)}),
    ('null_beta',{'null_pointers':('beta',)}),('null_alpha',{'null_pointers':('alpha',)}),
    ('null_state',{'null_pointers':('state',)}),('negative_start',{'start':-1}),
    ('zero_rows',{'rows':0}),('rows_exceed_capacity',{'rows':6}),('capacity_over_limit',{'capacity':257}),
    ('zero_batch',{'batch':0}),('zero_heads',{'heads':0}),('zero_key_dim',{'key_dim':0}),
    ('key_dim_over_limit',{'key_dim':257}),('zero_value_dim',{'value_dim':0}),
    ('value_dim_over_limit',{'value_dim':257}),('invalid_alpha_dim',{'alpha_dim':2})))


def expected_for_call(case:dict) -> dict:
  """Run the scalar recurrence; output has padded capacity stride like scan.c."""
  d=case['dims']
  batch,heads,rows,capacity=(d[k] for k in ('batch','heads','rows','capacity'))
  key_dim,value_dim,alpha_dim,start=(d[k] for k in ('key_dim','value_dim','alpha_dim','start'))
  status=validate_call(start=start,rows=rows,batch=batch,heads=heads,capacity=capacity,
                       key_dim=key_dim,value_dim=value_dim,alpha_dim=alpha_dim)
  if status: raise ValueError('invalid fixture parameters')
  q,k,v,beta,alpha=(case[name] for name in ('q','k','v','beta','alpha'))
  state=copy.copy(case['state_initial'])
  state_stride=value_dim*key_dim
  bh_count=batch*heads
  if start==0: state=[0.0]*(bh_count*state_stride)
  out=[0.0]*(bh_count*capacity*value_dim)
  history=[] if case.get('history_enabled',True) else None
  history_stride=bh_count*state_stride
  if history is not None: history=[0.0]*(rows*history_stride)
  for bh in range(bh_count):
    for t in range(rows):
      step=bh*capacity+t
      alpha_offset=step*alpha_dim
      for row in range(value_dim):
        state_row=bh*state_stride+row*key_dim
        predicted=0.0
        for j in range(key_dim):
          decay=alpha[alpha_offset+(0 if alpha_dim==1 else j)]
          predicted += (state[state_row+j]*decay)*k[step*key_dim+j]
        delta=(v[step*value_dim+row]-predicted)*beta[step]
        result=0.0
        for j in range(key_dim):
          decay=alpha[alpha_offset+(0 if alpha_dim==1 else j)]
          state[state_row+j]=state[state_row+j]*decay+delta*k[step*key_dim+j]
          result += state[state_row+j]*q[step*key_dim+j]
        out[step*value_dim+row]=result
      if history is not None:
        offset=t*history_stride+bh*state_stride
        history[offset:offset+state_stride]=state[bh*state_stride:(bh+1)*state_stride]
  return {'out_capacity_layout':out,'state_final':state,'history_valid_prefix_layout':history,
          'valid_rows':rows,'padded_rows':capacity-rows,'padding_output_is_zero':True,
          'state_advance_steps_per_batch_head':rows,'reset_applied':start==0,'status':0}


def expected_fixtures() -> tuple[dict,...]:
  """Return each deterministic input case paired with its pure-Python answer."""
  return tuple({'fixture':case,'expected':expected_for_call(case)} for case in FIXTURES)


def invalid_call_results() -> tuple[dict,...]:
  """Small null-pointer and dimension/parameter rejection cases from scan.c."""
  results=[]
  for case in INVALID_CALLS:
    arguments=_VALID_CALL.copy()
    arguments.update(case['overrides'])
    results.append({'name':case['name'],'arguments':arguments,
                    'actual_expected_status':validate_call(**arguments),
                    'declared_expected_status':case['expected_status']})
  return tuple(results)
