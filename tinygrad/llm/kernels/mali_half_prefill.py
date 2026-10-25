"""Bound the row launch of qualified compiler-generated FP16 projections."""
from __future__ import annotations
import functools, hashlib, math, re
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import BEAM, Context
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.llm.kernels.mali_rows import row_kernel, valid_rows
from tinygrad.uop.ops import Ops, UOp

# These fingerprints qualify both arithmetic and row mapping. A different
# compiler layout falls back to the ordinary linear path until requalified.
_QUALIFIED = {2048:'fe60dab7febb71d0dd30fcc3fc27de965ca74ff93c580d2475d253d6a3175f30',
              6144:'f2d38f45e63b3d00e498840e53ccea239b0cce696e800c6ba32d80c17430def1'}
_sources:dict[tuple[str,int],str] = {}


def prepare(device:str) -> dict:
  """Compile empty geometry once before capture; retain source, never weights."""
  with Context(ALLOW_DEVICE_USAGE=1):
    dev=Device[device]
    if (getattr(dev,'device_name','')!='Mali-LODX r0p0' or getattr(dev,'driver_version','')!='2.1'
        or BEAM.value!=0):return {'prepared_kernels':0}
    from tinygrad.engine.realize import lower_and_compile
    for columns,expected in _QUALIFIED.items():
      key=(device,columns)
      if key in _sources:continue
      weight=Tensor.empty(columns,2048,device=device,dtype=dtypes.half)
      activation=Tensor.empty(1,128,2048,device=device,dtype=dtypes.half)
      linear=lower_and_compile((activation@weight.T).schedule_linear(),verbose=False)
      programs=[call.body for call in linear.src if call.op is Ops.CALL and call.body.op is Ops.PROGRAM]
      if len(programs)!=1:continue
      program=programs[0]
      source=next((part.arg for part in program.src if part.op is Ops.SOURCE),None)
      if not isinstance(source,str) or hashlib.sha256(source.encode()).hexdigest()!=expected:continue
      nc=4 if columns==2048 else 3
      if program.arg.launch_dims({})!=((columns//(16*nc),4,1),(8,16,1)):continue
      _sources[key]=source
  return {'prepared_kernels':sum((device,columns) in _sources for columns in _QUALIFIED)}


@functools.cache
def _kernel(out:UOp, x:UOp, weight:UOp, columns:int, source:str, count:UOp, scalars:tuple[UOp,...]) -> UOp:
  # Preserve the generated shader body. Only the launch's M bound changes;
  # unused scalar arguments carry the current row bindings through TinyJit.
  match=re.search(r'__kernel void (\w+)\(',source)
  assert match is not None
  name=f'mali_half_prefill_{columns}_valid'
  header,body=source.replace(match.group(1),name).split('{',1)
  header=header.rstrip()
  scalar_decl=''.join(f',int launch_var_{i}' for i in range(len(scalars)))
  source=header[:-1]+scalar_decl+') {\n'+body
  nc=4 if columns==2048 else 3
  return _program(name,source,(out,x,weight,*scalars),(columns//(16*nc),(count+31)//32,1),(8,16,1))


def bounded_linear(x:Tensor, weight:Tensor) -> Tensor|None:
  """Return None for unprepared devices, layouts, dtypes or row capacities."""
  device=x.device
  if (not isinstance(device,str) or device!=weight.device or x.dtype is not dtypes.half or weight.dtype is not dtypes.half
      or math.prod(x.max_shape[:-1])!=128 or x.max_shape[-1]!=2048 or weight.shape not in ((2048,2048),(6144,2048))):
    return None
  columns=weight.shape[0]
  assert isinstance(columns,int)
  source=_sources.get((device,columns))
  if source is None or (count:=valid_rows(x,32)) is None:return None
  out=Tensor.empty(*x.max_shape[:-1],columns,device=device,dtype=dtypes.half)
  result=row_kernel(out,(x.pad_to(x.max_shape).contiguous(),weight.contiguous()),count,
    lambda o,a,w,actual,scalars:_kernel(o,a,w,columns,source,actual,scalars))
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))
