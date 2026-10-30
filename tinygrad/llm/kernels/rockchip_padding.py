"""Reuse explicitly zero-padded backend storage without changing logical views."""
import math, weakref
from collections import Counter
from tinygrad import Tensor
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops, UOp

ONPU_ZERO_PAD_VIEWS = ContextVar('ONPU_ZERO_PAD_VIEWS', 0)
_zeroed:weakref.WeakKeyDictionary[UOp,tuple[UOp,str]] = weakref.WeakKeyDictionary()
_reused:Counter[str] = Counter()


def mark_zero_tail(logical:Tensor, full:Tensor, kind:str) -> Tensor:
  if (ONPU_ZERO_PAD_VIEWS and logical.device==full.device=='CPU' and logical.dtype==full.dtype
      and full.shape==logical.max_shape and any(not isinstance(n,int) for n in logical.shape)):
    _zeroed[logical.uop] = full.uop,kind
  return logical


def _recover(u:UOp) -> tuple[UOp,str]|None:
  if (known:=_zeroed.get(u)) is not None:return known
  if u.op not in (Ops.RESHAPE,Ops.PERMUTE):return None
  if (source:=_recover(u.src[0])) is None:return None
  full,kind = source
  if u.op is Ops.PERMUTE:return full.permute(u.marg),kind
  # Reshaping must keep the symbolic row dimension and its leading/trailing
  # products intact. Flattening a head-major shrink would compact its gaps.
  old,new = u.src[0].shape,u.shape
  old_axes,new_axes = [[i for i,n in enumerate(shape) if not isinstance(n,int)] for shape in (old,new)]
  if len(old_axes)!=1 or len(new_axes)!=1:return None
  a,b = old_axes[0],new_axes[0]
  if old[a] is not new[b] or math.prod(old[:a])!=math.prod(new[:b]) or math.prod(old[a+1:])!=math.prod(new[b+1:]):return None
  shape = tuple(n if isinstance(n,int) else int(n.vmax) for n in new)
  return full.reshape(shape),kind


def capacity_input(x:Tensor) -> Tensor:
  if ONPU_ZERO_PAD_VIEWS and x.device=='CPU' and (recovered:=_recover(x.uop)) is not None:
    full,kind = recovered
    if full.shape==x.max_shape:
      _reused[kind] += 1
      return Tensor(full,device=x.device)
  return x.pad_to(x.max_shape)


def capacity_view_counts() -> dict[str,int]:return dict(_reused)
