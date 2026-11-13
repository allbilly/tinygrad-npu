"""CPU head-gate materialization preserves the existing broadcast arithmetic."""
import numpy as np
import pytest
from tinygrad import Tensor,TinyJit,UOp
from tinygrad.llm.kernels.rockchip_cpu import head_gate


def reference(attention,gate):return attention*gate.sigmoid().transpose(1,2).unsqueeze(-1)


def equal_values(a,b):
  x,y=a.numpy(),b.numpy()
  finite=np.isfinite(x)
  assert x[finite].tobytes()==y[finite].tobytes()
  for check in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(check(x),check(y))


@pytest.mark.parametrize('rows,heads,width',[(1,8,256),(17,8,256),(65,8,256),(128,8,256),(32,4,64),(128,16,256)])
@pytest.mark.parametrize('head_contiguous',[False,True])
def test_finite_and_exceptional_gate_values_preserve_output_bits(rows,heads,width,head_contiguous):
  rng=np.random.default_rng(431)
  attention=Tensor(rng.standard_normal((1,rows,heads,width),dtype=np.float32),device='CPU').transpose(1,2).realize()
  if head_contiguous:attention=attention.contiguous().realize()
  values=rng.standard_normal((1,rows,heads),dtype=np.float32)*6
  for special in (False,True):
    if special:
      cases=np.array([0.,-0.,1e-30,-1e-30,100.,-100.,1000.,-1000.,np.inf,-np.inf,np.nan,124.],dtype=np.float32)
      values.flat[:min(values.size,len(cases))]=cases[:min(values.size,len(cases))]
    gate=Tensor(values,device='CPU').realize()
    equal_values(reference(attention,gate).contiguous().realize(),head_gate(attention,gate).contiguous().realize())


def test_dynamic_rows_rebind_changed_inputs_and_ignore_poisoned_tails():
  rng=np.random.default_rng(432)
  count=UOp.variable('head_gate_rows',1,128)
  def run(a,g,n,fn):
    return fn(a[:,:n].transpose(1,2),g[:,:n]).transpose(1,2).pad_to((1,128,8,256)).contiguous().realize()
  control=TinyJit(lambda a,g,n:run(a,g,n,reference))
  candidate=TinyJit(lambda a,g,n:run(a,g,n,head_gate))
  for active in (17,128,1,65,17):
    data=rng.standard_normal((1,128,8,256),dtype=np.float32)
    gates=rng.standard_normal((1,128,8),dtype=np.float32)
    data[:,active:]=np.nan
    gates[:,active:]=np.nan
    a,g=Tensor(data,device='CPU').realize(),Tensor(gates,device='CPU').realize()
    equal_values(control(a,g,count.bind(active)),candidate(a,g,count.bind(active)))


def test_half_inputs_retain_original_fallback():
  rng=np.random.default_rng(433)
  a=Tensor(rng.standard_normal((1,8,17,64)).astype(np.float16),device='CPU').realize()
  g=Tensor(rng.standard_normal((1,17,8)).astype(np.float16),device='CPU').realize()
  equal_values(reference(a,g).contiguous().realize(),head_gate(a,g).contiguous().realize())


def test_unmaterialized_cpu_projection_keeps_original_fusion():
  rng=np.random.default_rng(434)
  attention=Tensor(rng.standard_normal((1,4,17,16),dtype=np.float32),device='CPU').realize()
  x=Tensor(rng.standard_normal((1,17,64),dtype=np.float32),device='CPU').realize()
  weight=Tensor(rng.standard_normal((64,4),dtype=np.float32),device='CPU').realize()
  gate=x@weight
  actual,expected=head_gate(attention,gate),reference(attention,gate)
  assert actual.uop is expected.uop
  equal_values(actual.contiguous().realize(),expected.contiguous().realize())


def test_unmaterialized_cpu_attention_keeps_original_fusion():
  rng=np.random.default_rng(435)
  q=Tensor(rng.standard_normal((1,4,17,32),dtype=np.float32),device='CPU').realize()
  v=Tensor(rng.standard_normal((1,4,32,16),dtype=np.float32),device='CPU').realize()
  gate=Tensor(rng.standard_normal((1,17,4),dtype=np.float32),device='CPU').realize()
  attention=q@v
  actual,expected=head_gate(attention,gate),reference(attention,gate)
  assert actual.uop is expected.uop
  equal_values(actual.contiguous().realize(),expected.contiguous().realize())
