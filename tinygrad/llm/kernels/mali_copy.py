"""Owned inference-cache appends using the existing in-order OpenCL queue.

Appends allocate a new output; row updates require separate owned destination
storage. Retained source inputs are never modified.
Recording and profiling retain the Tensor implementation so copies remain visible.
"""
from __future__ import annotations
import ctypes, math
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import capturing
from tinygrad.helpers import DEBUG, PROFILE
from tinygrad.uop.ops import Ops


def _dense_buffer(value:Tensor) -> Buffer|None:
  if not value.uop.is_realized:return None
  # Generic contiguous_view deliberately declines all OpenCL tensors. Accept
  # only whole dense storage here; reshapes preserve its element order.
  node=value.uop
  while node.op is Ops.RESHAPE:node=node.src[0]
  if node.op is not Ops.BUFFER:return None
  buffer=node.buffer
  if (not isinstance(buffer,Buffer) or buffer.base is not buffer or buffer.offset!=0
      or not buffer.is_allocated() or buffer.nbytes!=value.numel()*value.dtype.itemsize):return None
  return buffer


def _copy_rect(device, source:Buffer, output:Buffer, source_rows:int, output_rows:int,
               begin:int, count:int, destination:int, heads:int, row_bytes:int) -> None:
  from tinygrad.runtime.autogen import opencl as cl
  from tinygrad.runtime.ops_cl import check
  source_origin=(ctypes.c_size_t*3)(begin*row_bytes,0,0)
  destination_origin=(ctypes.c_size_t*3)(destination*row_bytes,0,0)
  check(cl.clEnqueueCopyBufferRect(device.queue,source._buf,output._buf,source_origin,destination_origin,
    (ctypes.c_size_t*3)(count*row_bytes,heads,1),source_rows*row_bytes,0,output_rows*row_bytes,0,0,None,None))


def copy_rows(source:Tensor, output:Tensor, begin:int, end:int) -> bool:
  """Write dense source[...,begin:end,:] into owned output at the same rows.

  Decline unsupported storage/devices without writing. The caller owns the
  destination and invalidates its contents before attempting this operation.
  Copies use the existing in-order queue; recording keeps the Tensor fallback.
  """
  if (capturing or PROFILE or DEBUG>=2 or source.device!=output.device or not isinstance(source.device,str)
      or source.device.split(':')[0]!='CL' or source.dtype is not output.dtype
      or source.dtype not in (dtypes.float16,dtypes.float32) or source.ndim<2 or source.ndim!=output.ndim
      or source.shape[:-2]!=output.shape[:-2] or source.shape[-1]!=output.shape[-1]
      or not all(type(size) is int and size>0 for size in (*source.shape,*output.shape))):return False
  source_rows,output_rows=source.max_shape[-2],output.max_shape[-2]
  if type(begin) is not int or type(end) is not int or not 0<=begin<end<=min(source_rows,output_rows):
    raise ValueError('Invalid native cache-copy row range')
  old,destination=_dense_buffer(source),_dense_buffer(output)
  if old is None or destination is None or old is destination:return False
  device=Device[source.device]
  if getattr(device,'device_name','')!='Mali-LODX r0p0' or getattr(device,'driver_version','')!='2.1':return False
  _copy_rect(device,old,destination,source_rows,output_rows,begin,end-begin,begin,
             math.prod(source.max_shape[:-2]),source.max_shape[-1]*source.dtype.itemsize)
  return True


def append_rows(previous:Tensor, appended:Tensor, begin:int, end:int) -> Tensor|None:
  """Copy previous[...,begin:end,:] followed by appended, or decline unchanged.

  The penultimate axis is tokens; preceding axes flatten into independent heads.
  Dense original buffers are passed before slicing so their physical row pitch is
  retained. The caller owns the same synchronization boundary as Tensor.realize.
  """
  if (capturing or PROFILE or DEBUG>=2 or previous.device!=appended.device or not isinstance(previous.device,str)
      or previous.device.split(':')[0]!='CL' or previous.dtype is not appended.dtype
      or previous.dtype not in (dtypes.float16,dtypes.float32) or previous.ndim<2
      or previous.ndim!=appended.ndim or previous.shape[:-2]!=appended.shape[:-2]
      or previous.shape[-1]!=appended.shape[-1] or not all(type(size) is int and size>0 for size in (*previous.shape,*appended.shape))):
    return None
  old_rows,new_rows=previous.max_shape[-2],appended.max_shape[-2]
  if type(begin) is not int or type(end) is not int or not 0<=begin<=end<=old_rows:raise ValueError('Invalid preserved cache rows')
  old,new=_dense_buffer(previous),_dense_buffer(appended)
  if old is None or new is None:return None
  device=Device[previous.device]
  if getattr(device,'device_name','')!='Mali-LODX r0p0' or getattr(device,'driver_version','')!='2.1':return None
  kept=end-begin
  result=Tensor.empty(*previous.shape[:-2],kept+new_rows,previous.shape[-1],device=previous.device,dtype=previous.dtype).realize()
  output=result.uop.buffer
  assert isinstance(output,Buffer)
  output.ensure_allocated()
  heads,row_bytes=math.prod(previous.max_shape[:-2]),previous.max_shape[-1]*previous.dtype.itemsize
  if kept:_copy_rect(device,old,output,old_rows,kept+new_rows,begin,kept,0,heads,row_bytes)
  _copy_rect(device,new,output,new_rows,kept+new_rows,0,new_rows,kept,heads,row_bytes)
  return result
