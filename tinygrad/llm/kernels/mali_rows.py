"""Bind valid single-batch row extents without changing padded buffer layouts."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import replace
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import GroupOp, Ops


def valid_rows(x:Tensor, grain:int) -> UOp|None:
  # Flattened batches can have gaps between their valid rows after pad_to. Keep
  # their capacity launch; the single-batch decoder has one contiguous prefix.
  if x.ndim<2 or any(type(size) is not int or size!=1 for size in x.shape[:-2]):return None
  count = x.shape[-2]
  if not isinstance(count,UOp) or x.max_shape[-2]<=grain:return None
  if not (dtypes.is_int(count.dtype) or count.dtype is dtypes.weakint):return None
  if count.vmin<1 or count.vmax>2**31-1-grain:return None
  variables = count.variables()
  if not variables or any(not (dtypes.is_int(var.dtype) or var.dtype is dtypes.weakint) for var in variables):return None
  # Scalar OpenCL arguments are int32. Do not narrow any intermediate symbolic
  # arithmetic that requires a wider range, even if its final minimum is small.
  for node in count.toposort():
    if node.is_variable or node.op in GroupOp.ALU or node.op in (Ops.CONST,Ops.CAST):
      if not (dtypes.is_int(node.dtype) or node.dtype in (dtypes.weakint,dtypes.bool)):return None
      if node.vmin<-(2**31) or node.vmax>=2**31:return None
  return count


def row_kernel(output:Tensor, inputs:tuple[Tensor,...], count:UOp, fxn:Callable[...,UOp]) -> Tensor:
  expression,bindings = count.unbind_all()
  variables = expression.variables()
  typed = tuple(var.replace(arg=replace(var.arg,dtype=dtypes.int)) for var in variables)
  expression = expression.substitute(dict(zip(variables,typed)))
  arguments = tuple(var.bind(bindings[old]) if old in bindings else var for old,var in zip(variables,typed))
  def build(*params:UOp) -> UOp:
    buffers,scalars = params[:len(inputs)+1],params[len(inputs)+1:]
    rows = expression.substitute(dict(zip(typed,scalars)))
    return fxn(*buffers,rows,tuple(scalars))
  return Tensor(UOp.custom_kernel(*(tensor.uop for tensor in (output,*inputs)),*arguments,fxn=build)[0])
