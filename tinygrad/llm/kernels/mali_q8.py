"""Native GGUF Q8_0 Mali projections with FP16 operands and FP32 accumulation.

Only the original quantized storage is retained. Prefill decodes one 32-wide
weight tile into local memory, while decoding uses the existing Q8 GEMV reduction.
These kernels do not provide a device or quantization fallback: callers opt in
after checking their checkpoint and Mali device.
"""
from __future__ import annotations
import functools, math
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.llm.kernels.mali_linear import _gemv_q8, _small_gemm_supported
from tinygrad.llm.kernels.mali_rows import row_kernel, valid_rows
from tinygrad.uop.ops import UOp


def _validate_storage(raw:Tensor, columns:int, depth:int) -> None:
  if type(columns) is not int or type(depth) is not int or columns<1 or depth<1 or depth%32:
    raise ValueError('Mali Q8_0 requires positive static output size and input size divisible by 32')
  if raw.dtype is not dtypes.uint8 or type(raw.numel()) is not int or raw.numel()!=columns*(depth//32)*34:
    raise ValueError('Mali Q8_0 requires exactly one 34-byte block per 32 row-major weights')
  if not isinstance(raw.device,str) or raw.device.split(':')[0]!='CL':
    raise ValueError('Mali Q8_0 storage must be on one OpenCL device')


def _validate_input(x:Tensor, raw:Tensor, columns:int, depth:int) -> int:
  _validate_storage(raw,columns,depth)
  if x.ndim<1 or type(x.shape[-1]) is not int or x.shape[-1]!=depth or x.dtype not in (dtypes.half,dtypes.float) or x.device!=raw.device:
    raise ValueError('Mali Q8_0 expects FP16 or FP32 inputs on the storage device with matching input size')
  rows = math.prod(x.max_shape[:-1])
  if rows<1 or max(rows*columns,rows*depth)>2**31-1:
    raise ValueError('Mali Q8_0 input capacity must fit positive signed 32-bit indexing')
  return rows


def _gemm_geometry(rows:int, columns:int, depth:int) -> tuple[int,int,int]:
  # These complete projections were measured with exact FP32 output agreement.
  # Keep unmeasured row capacities, tails and projection sizes on the old tile.
  if rows==128 and (columns,depth) in ((3584,1024),(6144,1024),(1024,3584)):return 64,64,128
  return 32,32,64


@functools.cache
def _gemm_q8(out:UOp, x:UOp, raw:UOp, rows:int, columns:int, depth:int,
             geometry:tuple[int,int,int]=(32,32,64), launch_rows:UOp|None=None, scalars:tuple[UOp,...]=()) -> UOp:
  mr,nr,workgroup=geometry
  suffix = '_valid' if launch_rows is not None else ''
  scalar_decl = ''.join(f',int launch_var_{i}' for i in range(len(scalars)))
  column_lanes=nr//4
  rows_per_thread=mr//(workgroup//column_lanes)
  parts=workgroup//nr
  assert geometry in ((32,32,64),(64,64,128))
  stores = ''.join(f'''if(m+{i}<{rows}){{if(n+3<{columns})vstore4(sum{i},0,out+(m+{i})*{columns}+n);
    else{{if(n<{columns})out[(m+{i})*{columns}+n]=sum{i}.x;
      if(n+1<{columns})out[(m+{i})*{columns}+n+1]=sum{i}.y;
      if(n+2<{columns})out[(m+{i})*{columns}+n+2]=sum{i}.z;}}}}''' for i in range(rows_per_thread))
  sums=''.join(f'float4 sum{i}=(float4)0;' for i in range(rows_per_thread))
  arithmetic=''.join(f'if(m+{i}<{rows})sum{i}=fma((float4)a_tile[((lid/{column_lanes})*{rows_per_thread}+{i})*32+k],b,sum{i});'
                     for i in range(rows_per_thread))
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemm_q8{suffix}(__global float *out,__global const float *x,__global const uchar *raw{scalar_decl}) {{
    int lid=get_local_id(0),m=get_group_id(1)*{mr}+(lid/{column_lanes})*{rows_per_thread},
      n=get_group_id(0)*{nr}+(lid%{column_lanes})*4;
    __local float a_tile[{mr*32}];
    __local half b_tile[{nr*32}];
    {sums}
    for(int base=0;base<{depth};base+=32) {{
      for(int index=lid;index<{mr*32};index+={workgroup}) {{
        int row=index/32,k=index%32;
        a_tile[index]=get_group_id(1)*{mr}+row<{rows}?x[(get_group_id(1)*{mr}+row)*{depth}+base+k]:0.0f;
      }}
      int column=lid%{nr},part=lid/{nr},output_column=get_group_id(0)*{nr}+column;
      float scale=0.0f;
      ulong block=(((ulong)output_column)*{depth//32}+base/32)*34;
      if(output_column<{columns})scale=(float)(*(__global const half *)(raw+block));
      for(int step=0;step<{32//parts//4};step++) {{
        int k=part*{32//parts}+step*4;
        char4 quant=output_column<{columns}?vload4(0,(__global const char *)(raw+block+2+k)):(char4)0;
        half4 value=convert_half4_rte(convert_float4(quant)*(float4)scale);
        b_tile[k*{nr}+column]=value.x;b_tile[(k+1)*{nr}+column]=value.y;
        b_tile[(k+2)*{nr}+column]=value.z;b_tile[(k+3)*{nr}+column]=value.w;
      }}
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int k=0;k<32;k++) {{
        float4 b=convert_float4(vload4(0,b_tile+k*{nr}+(lid%{column_lanes})*4));
        {arithmetic}
      }}
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {stores}
  }}'''
  groups = (ceildiv(columns,nr),(launch_rows+mr-1)//mr if launch_rows is not None else ceildiv(rows,mr),1)
  return _program('mali_gemm_q8'+suffix,source,(out,x,raw,*scalars),groups,(workgroup,1,1))


def gemm_q8(x:Tensor, raw:Tensor, columns:int, depth:int) -> Tensor:
  """Project every valid input row, using its bounded capacity for the launch."""
  rows = _validate_input(x,raw,columns,depth)
  geometry=_gemm_geometry(rows,columns,depth)
  if geometry!=(32,32,64) and not _small_gemm_supported(x.device):geometry=32,32,64
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  if (count:=valid_rows(x,geometry[0])) is not None:
    result = row_kernel(output,(x.float().pad_to(x.max_shape).contiguous(),raw.flatten().contiguous()),count,
      lambda o,a,w,actual,scalars:_gemm_q8(o,a,w,rows,columns,depth,geometry,actual,scalars))
    return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))
  result = output.custom_kernel(x.float().pad_to(x.max_shape).contiguous(),raw.flatten().contiguous(),
    fxn=lambda o,a,w:_gemm_q8(o,a,w,rows,columns,depth,geometry))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


def gemv_q8(x:Tensor, raw:Tensor, columns:int, depth:int) -> Tensor:
  """Single-row projection with the same lane reduction as the FP16 Mali GEMV."""
  if _validate_input(x,raw,columns,depth)!=1: raise ValueError('Mali Q8_0 GEMV requires exactly one input row')
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  result = output.custom_kernel(x.float().pad_to(x.max_shape).contiguous(),raw.flatten().contiguous(),
    fxn=lambda o,a,w:_gemv_q8(o,a,w,columns,depth))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


@functools.cache
def _gemv_q8_rows(out:UOp, x:UOp, raw:UOp, rows:int, columns:int, depth:int,
                  group:int=16, tile:int|None=None) -> UOp:
  # Match gemv_rows' 16 K lanes, component additions and ordered lane sum.
  # Short blocks reuse each reconstructed FP16 weight without a dense copy.
  # Three accumulators avoid decoding the same weight twice for three rows;
  # larger blocks keep the two-row register footprint of the dense head path.
  if tile is None:tile=rows if rows<=3 else 2
  assert group in (2,4,8,16) and tile in (1,2,3,4,8)
  size,tiles=group*16,ceildiv(rows,tile)
  name=f'mali_gemv_q8_rows_t{tile}'+('' if group==16 else f'_g{group}')
  declarations=''.join(f'float4 sum{i}=(float4)0;' for i in range(tile))
  arithmetic=''.join(f'if(m+{i}<{rows})sum{i}=fma(vload4(0,x+(m+{i})*{depth}+k),w,sum{i});' for i in range(tile))
  partials=''.join(f'partial[{i*size}+lid]=(sum{i}.x+sum{i}.y)+(sum{i}.z+sum{i}.w);' for i in range(tile))
  stores=''.join(f'''if(m+{i}<{rows}){{float result=0;
    for(int k=0;k<16;k++)result+=partial[{i*size}+lid+k];out[(m+{i})*{columns}+column]=result;}}''' for i in range(tile))
  source=f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void {name}(__global float *out,__global const float *x,__global const uchar *raw) {{
    int lid=get_local_id(0),lane=lid%16;
    int column=(get_group_id(0)/{tiles})*{group}+lid/16,m=(get_group_id(0)%{tiles})*{tile};
    {declarations}
    if(column<{columns})for(int k=lane*4;k<{depth};k+=64) {{
      ulong block=(((ulong)column)*{depth//32}+k/32)*34;
      float scale=(float)(*(__global const half *)(raw+block));
      char4 q=vload4(0,(__global const char *)(raw+block+2+k%32));
      float4 w=convert_float4(convert_half4_rte(convert_float4(q)*(float4)scale));
      {arithmetic}
    }}
    __local float partial[{tile*size}];
    {partials}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && column<{columns}) {{{stores}}}
  }}'''
  return _program(name,source,(out,x,raw),(ceildiv(columns,group)*tiles,1,1),(size,1,1))


def gemv_q8_rows(x:Tensor, raw:Tensor, columns:int, depth:int) -> Tensor:
  """Opt-in short FP32 projection with FP16-rounded Q8 weights and GEMV order.

  Half-activation consistent decode has a different product-rounding contract;
  callers must not use this FP32 path for that operation. No dispatch is changed.
  """
  rows=_validate_input(x,raw,columns,depth)
  if x.dtype is not dtypes.float or not 1<=rows<=8:
    raise ValueError('Mali Q8 GEMV rows requires FP32 inputs with row capacity 1..8')
  tuned=rows==8 and (columns,depth)==(248320,2048) and _small_gemm_supported(x.device)
  output=Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  result=output.custom_kernel(x.pad_to(x.max_shape).contiguous(),raw.flatten().contiguous(),
    fxn=lambda o,a,w:(_gemv_q8_rows(o,a,w,rows,columns,depth,2,8) if tuned else _gemv_q8_rows(o,a,w,rows,columns,depth)))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


@functools.cache
def _embedding_q8(out:UOp, ids:UOp, raw:UOp, rows:int, vocab:int, dim:int) -> UOp:
  index_type='int' if ids.dtype is dtypes.int else 'long'
  source=f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_embedding_q8(__global float *out,__global const {index_type} *ids,__global const uchar *raw) {{
    int i=get_global_id(0);if(i>={rows*dim})return;
    {index_type} id=ids[i/{dim}];
    float value=0;
    if(id>=0 && id<{vocab}) {{
      int column=i%{dim};
      ulong block=(((ulong)id)*{dim//32}+column/32)*34;
      float scale=(float)(*(__global const half *)(raw+block));
      float quant=(float)(*((__global const char *)(raw+block+2+column%32)));
      value=(float)convert_half_rte(quant*scale);
    }}
    out[i]=value;
  }}'''
  return _program('mali_embedding_q8',source,(out,ids,raw),(ceildiv(rows*dim,128),1,1),(128,1,1))


def embedding_q8(ids:Tensor, raw:Tensor, vocab:int, dim:int) -> Tensor:
  """Gather FP16-rounded Q8 values as FP32, matching the dense Mali lookup."""
  _validate_storage(raw,vocab,dim)
  rows=math.prod(ids.max_shape)
  if ids.dtype not in (dtypes.int,dtypes.int64) or ids.device!=raw.device or not 1<=rows*dim<=2**31-1:
    raise ValueError('Mali Q8 embedding requires same-device integer IDs and positive bounded output capacity')
  output=Tensor.empty(*ids.max_shape,dim,device=raw.device,dtype=dtypes.float)
  result=output.custom_kernel(ids.pad_to(ids.max_shape).contiguous(),raw.flatten().contiguous(),
    fxn=lambda o,i,w:_embedding_q8(o,i,w,rows,vocab,dim))[0]
  return result.shrink(tuple((0,size) for size in (*ids.shape,dim)))


class MaliQ8Linear:
  """An inference-only projection retaining quantized weights and optional bias."""
  def __init__(self, raw:Tensor, out_features:int, in_features:int, bias:Tensor|None=None):
    _validate_storage(raw,out_features,in_features)
    if bias is not None and (bias.shape!=(out_features,) or bias.device!=raw.device or bias.dtype not in (dtypes.half,dtypes.float)):
      raise ValueError('Mali Q8_0 bias must match the output size, dtype, and OpenCL device')
    self.raw,self.out_features,self.in_features,self.bias = raw,out_features,in_features,bias

  def __call__(self, x:Tensor) -> Tensor:
    rows = _validate_input(x,self.raw,self.out_features,self.in_features)
    operation = gemv_q8 if rows==1 else gemm_q8
    result = operation(x,self.raw,self.out_features,self.in_features)
    return (result if self.bias is None else result+self.bias).cast(x.dtype)
