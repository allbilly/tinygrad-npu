"""FP32 GatedDeltaNet scan on the qualified 16-lane Mali OpenCL subgroup path."""
from __future__ import annotations
import functools
from dataclasses import replace
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, ceildiv
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.uop.ops import Ops, UOp


@functools.cache
def supported(device:str|tuple[str,...]|None) -> bool:
  if not isinstance(device,str) or device.split(':')[0]!='CL':return False
  # Like the AMD capability helper, inspect the already-open device while a
  # @function body is tracing with device usage disabled.
  with Context(ALLOW_DEVICE_USAGE=1):
    dev = Device[device]
    return (getattr(dev,'device_name','')=='Mali-LODX r0p0' and getattr(dev,'driver_version','')=='2.1'
            and 'cl_khr_subgroups' in getattr(dev,'device_exts',()))


@functools.cache
def _scan(out:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, start:UOp, rows:UOp,
          batch:int, heads:int, capacity:int, key_dim:int, value_dim:int, alpha_dim:int, history:UOp|None=None) -> UOp:
  columns,tiles = ceildiv(key_dim,16),ceildiv(value_dim,4)
  initialize,loads,updates,finish = [],[],[],[]
  for j in range(columns):
    col = f'lane+{j*16}'
    loads.append(f'''float k{j}={col}<{key_dim}?k[(bh*{capacity}+t)*{key_dim}+{col}]:0.0f;
      float q{j}={col}<{key_dim}?q[(bh*{capacity}+t)*{key_dim}+{col}]:0.0f;
      float a{j}={col}<{key_dim}?alpha[(bh*{capacity}+t)*{alpha_dim}+{col if alpha_dim>1 else '0'}]:1.0f;''')
  for r in range(4):
    initialize += [f'float s{r}_{j}=row+{r}<{value_dim} && lane+{j*16}<{key_dim} && !reset?'
                   f'state[(bh*{value_dim}+row+{r})*{key_dim}+lane+{j*16}]:0.0f;' for j in range(columns)]
    body = ''.join(f'float d{j}=s{r}_{j}*a{j};predicted+=d{j}*k{j};' for j in range(columns))
    body += f'''predicted=sub_group_reduce_add(predicted);
      float delta=(v[(bh*{capacity}+t)*{value_dim}+row+{r}]-predicted)*beta[bh*{capacity}+t];float result=0.0f;'''
    body += ''.join(f's{r}_{j}=d{j}+delta*k{j};result+=s{r}_{j}*q{j};' for j in range(columns))
    body += f'''result=sub_group_reduce_add(result);
      if(lane==0)out[(bh*{capacity}+t)*{value_dim}+row+{r}]=result;'''
    if history is not None:
      body += ''.join(f'if(lane+{j*16}<{key_dim})history[((t*{batch*heads}+bh)*{value_dim}+row+{r})*{key_dim}+lane+{j*16}]=s{r}_{j};'
                      for j in range(columns))
    updates.append(f'if(row+{r}<{value_dim}){{float predicted=0.0f;{body}}}')
    finish += [f'''if(row+{r}<{value_dim} && lane+{j*16}<{key_dim})
      state[(bh*{value_dim}+row+{r})*{key_dim}+lane+{j*16}]=s{r}_{j};''' for j in range(columns)]
  source = f'''// tinygrad-cl-std=CL2.0
  #pragma OPENCL EXTENSION cl_khr_subgroups : enable
  #pragma OPENCL FP_CONTRACT OFF
  __kernel void mali_gated_delta_scan(__global float *out,__global const float *q,__global const float *k,
      __global const float *v,__global const float *beta,__global const float *alpha,__global float *state,
      __global const int *start,__global const int *rows{',__global float *history' if history is not None else ''}) {{
    int lane=get_local_id(0),bh=get_group_id(0)/{tiles},row=(get_group_id(0)%{tiles})*4;
    int reset=*start==0,count=*rows;
    {''.join(initialize)}
    for(int t=0;t<count;t++){{{''.join(loads)}{''.join(updates)}}}
    {''.join(finish)}
    for(int t=count;t<{capacity};t++)if(lane<4 && row+lane<{value_dim})out[(bh*{capacity}+t)*{value_dim}+row+lane]=0.0f;
  }}'''
  args = (out,q,k,v,beta,alpha,state,start,rows)+((history,) if history is not None else ())
  program = _program('mali_gated_delta_scan',source,args,(batch*heads*tiles,1,1),(16,1,1))
  return program.replace(arg=replace(program.arg,outs=(0,6,9) if history is not None else (0,6),ins=tuple(range(1,9))))


def gated_delta_scan(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                     start_pos:int|UOp, rows:int|UOp, *, history:Tensor|None=None) -> Tensor:
  """Scan valid rows, updating FP32 (B,H,V,K) state; position zero ignores the old state."""
  if not supported(q.device) or any(x.device!=q.device or x.dtype is not dtypes.float for x in (q,k,v,beta,alpha,state)):
    raise ValueError('Mali recurrent scan requires qualified OpenCL hardware and FP32 operands')
  if q.ndim!=4 or v.ndim!=4 or alpha.ndim not in (3,4):raise ValueError('Expected (batch, heads, capacity, channels) recurrent operands')
  batch,heads,capacity,key_dim = q.max_shape
  value_dim,alpha_dim = v.max_shape[-1],1 if alpha.ndim==3 else alpha.max_shape[-1]
  storage = state.uop
  while storage.op in (Ops.RESHAPE,Ops.AFTER):storage=storage.src[0]
  if (q.shape!=q.max_shape or k.shape!=q.shape or v.shape!=(batch,heads,capacity,value_dim)
      or beta.shape!=(batch,heads,capacity) or alpha.shape not in ((batch,heads,capacity),(batch,heads,capacity,alpha_dim))
      or state.shape!=(batch,heads,value_dim,key_dim) or alpha_dim not in (1,key_dim)
      or not 1<=key_dim<=256 or not 1<=value_dim<=256 or not 1<=capacity<=256
      or storage.op not in (Ops.BUFFER,Ops.PARAM) or storage.numel()!=state.numel()):raise ValueError('Unsupported Mali recurrent scan geometry')
  if (isinstance(rows,int) and not 1<=rows<=capacity or isinstance(rows,UOp) and (rows.vmin<1 or rows.vmax>capacity)
      or isinstance(start_pos,int) and start_pos<0 or isinstance(start_pos,UOp) and start_pos.vmin<0):
    raise ValueError('Invalid Mali recurrent scan row count or position')
  if history is not None:
    storage = history.uop
    while storage.op in (Ops.RESHAPE,Ops.AFTER):storage=storage.src[0]
    history_capacity = history.max_shape[0] if history.ndim==5 else 0
    if (alpha_dim!=1 or not capacity<=history_capacity<=8 or history.shape!=(history_capacity,batch,heads,value_dim,key_dim)
        or history.dtype is not dtypes.float or history.device!=q.device
        or storage.op not in (Ops.BUFFER,Ops.PARAM) or storage.numel()!=history.numel()):
      raise ValueError('Invalid bounded FP32 Mali prefix state history')
  start = Tensor(start_pos,dtype=dtypes.int).reshape(1).clone(device=q.device)
  count = Tensor(rows,dtype=dtypes.int).reshape(1).clone(device=q.device)
  out = Tensor.empty(batch,heads,capacity,value_dim,device=q.device,dtype=dtypes.float)
  if history is not None:
    return out.custom_kernel(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),alpha.contiguous(),state,start,count,history,
      fxn=lambda o,a,b,c,d,e,s,p,r,h:_scan(o,a,b,c,d,e,s,p,r,batch,heads,capacity,key_dim,value_dim,alpha_dim,h))[0]
  return out.custom_kernel(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),alpha.contiguous(),state,start,count,
    fxn=lambda o,a,b,c,d,e,s,p,r:_scan(o,a,b,c,d,e,s,p,r,batch,heads,capacity,key_dim,value_dim,alpha_dim))[0]
