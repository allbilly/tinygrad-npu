"""Coalesced FP32-accumulating Mali GEMV and direct token lookup."""
from __future__ import annotations
import functools, math
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, ceildiv
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.llm.kernels.mali_rows import row_kernel, valid_rows
from tinygrad.uop.ops import UOp


def pack_weight(weight:Tensor) -> Tensor:
  columns,depth = weight.shape
  assert isinstance(columns,int) and isinstance(depth,int)
  tiles = ceildiv(columns,32)
  return weight.pad(((0,tiles*32-columns),(0,0))).reshape(tiles,32,depth).permute(0,2,1).contiguous()


@functools.cache
def _small_gemm_supported(device:str|tuple[str,...]|None) -> bool:
  if not isinstance(device,str) or device.split(':')[0]!='CL':return False
  # The device is already open when a prepared decoder traces with device usage
  # disabled. This is the reported CL_DRIVER_VERSION string, not a build ID.
  with Context(ALLOW_DEVICE_USAGE=1):
    dev=Device[device]
    return getattr(dev,'device_name','')=='Mali-LODX r0p0' and getattr(dev,'driver_version','')=='2.1'


def _small_gemm_geometry(rows:int, columns:int, depth:int, packed:bool) -> tuple[int,int,int]|None:
  # The actual resident full head was measured separately from output windows.
  # Its packed form and other row capacities retain the legacy kernel.
  if (columns,depth)==(248320,2048):
    if packed:return None
    return (4,64,1) if rows==3 else (16,32,2) if rows==8 else None
  # Only measured row capacities and complete projection shapes. The 8192-column
  # output window is qualified as its own shape; it does not qualify a full LM head.
  if rows not in (3,8,16) or (columns,depth) not in ((512,2048),(2048,2048),(6144,2048),(2048,6144),(8192,2048)):return None
  if columns==512:return (8,32,1)
  if rows==3:return (4,64,1)
  if rows==16 or (not packed and (columns==6144 or depth==6144 or columns==8192)):return (16,32,2)
  return (8,32,1)


@functools.cache
def _small_gemm(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int,
                packed:bool, geometry:tuple[int,int,int]) -> UOp:
  mr,nr,rpt=geometry
  assert geometry in ((8,32,1),(4,64,1),(16,32,2))
  name=f'mali_small_m{mr}_n{nr}_'+('packed' if packed else 'rowmajor')
  declarations=''.join(f'float4 sum{i}=(float4)0;' for i in range(rpt))
  stores=''.join(f'''if(m+{i}<{rows}){{
    if(n+3<{columns})vstore4(sum{i},0,out+(m+{i})*{columns}+n);
    else{{if(n<{columns})out[(m+{i})*{columns}+n]=sum{i}.x;
      if(n+1<{columns})out[(m+{i})*{columns}+n+1]=sum{i}.y;
      if(n+2<{columns})out[(m+{i})*{columns}+n+2]=sum{i}.z;}}
    }}''' for i in range(rpt))
  if packed:
    arithmetic=''.join(f'if(m+{i}<{rows})sum{i}=fma((float4)x[(m+{i})*{depth}+k],b,sum{i});' for i in range(rpt))
    body=f'''for(int k=0;k<{depth};k++) {{
      float4 b=(float4)0;
      if(n<{columns})b=convert_float4(vload4(0,weight+(((ulong)(n/32))*{depth}+k)*32+n%32));
      {arithmetic}
    }}'''
  else:
    arithmetic=''.join(f'''if(m+{i}<{rows})sum{i}=fma(
      (float4)a_tile[((lid/{nr//4})*{rpt}+{i})*32+k],b,sum{i});''' for i in range(rpt))
    body=f'''__local float a_tile[{mr*32}];
    __local half b_tile[{nr*32}];
    for(int base=0;base<{depth};base+=32) {{
      for(int index=lid;index<{mr*32};index+=64) {{
        int row=index/32,k=index%32;
        a_tile[index]=get_group_id(1)*{mr}+row<{rows} && base+k<{depth}
          ?x[(get_group_id(1)*{mr}+row)*{depth}+base+k]:0.0f;
      }}
      for(int index=lid;index<{nr*32};index+=64) {{
        int column=index/32,k=index%32;
        b_tile[k*{nr}+column]=get_group_id(0)*{nr}+column<{columns} && base+k<{depth}
          ?weight[((ulong)(get_group_id(0)*{nr}+column))*{depth}+base+k]:(half)0;
      }}
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int k=0;k<32 && base+k<{depth};k++) {{
        float4 b=convert_float4(vload4(0,b_tile+k*{nr}+(lid%{nr//4})*4));
        {arithmetic}
      }}
      barrier(CLK_LOCAL_MEM_FENCE);
    }}'''
  source=f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #pragma OPENCL FP_CONTRACT OFF
  __kernel void {name}(__global float *out,__global const float *x,__global const half *weight) {{
    int lid=get_local_id(0),m=get_group_id(1)*{mr}+(lid/{nr//4})*{rpt},n=get_group_id(0)*{nr}+(lid%{nr//4})*4;
    {declarations}
    {body}
    {stores}
  }}'''
  return _program(name,source,(out,x,weight),(ceildiv(columns,nr),ceildiv(rows,mr),1),(64,1,1))


@functools.cache
def _gemm(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int, packed:bool=True,
          launch_rows:UOp|None=None, scalars:tuple[UOp,...]=()) -> UOp:
  suffix = '_valid' if launch_rows is not None else ''
  scalar_decl = ''.join(f',int launch_var_{i}' for i in range(len(scalars)))
  groups = (ceildiv(columns,32),(launch_rows+31)//32 if launch_rows is not None else ceildiv(rows,32),1)
  typ = 'half' if weight.dtype is dtypes.half else 'float'
  load = (f'convert_float4(vload4(0,weight+(((ulong)get_group_id(0))*{depth}+k)*32+(lid%8)*4))' if packed else
          '(float4)('+','.join(f'n+{i}<{columns}?(float)weight[((ulong)n+{i})*{depth}+k]:0.0f' for i in range(4))+')')
  stores = ''.join(f'''if(m+{i}<{rows}){{if(n+3<{columns})vstore4(sum{i},0,out+(m+{i})*{columns}+n);
    else{{if(n<{columns})out[(m+{i})*{columns}+n]=sum{i}.x;
      if(n+1<{columns})out[(m+{i})*{columns}+n+1]=sum{i}.y;
      if(n+2<{columns})out[(m+{i})*{columns}+n+2]=sum{i}.z;}}}}''' for i in range(4))
  if not packed:
    source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void mali_gemm_tiled{suffix}(__global float *out,__global const float *x,__global const {typ} *weight{scalar_decl}) {{
      int lid=get_local_id(0),m=get_group_id(1)*32+(lid/8)*4,n=get_group_id(0)*32+(lid%8)*4;
      __local float a_tile[1024];
      __local {typ} b_tile[1024];
      float4 sum0=(float4)0,sum1=(float4)0,sum2=(float4)0,sum3=(float4)0;
      for(int base=0;base<{depth};base+=32) {{
        for(int index=lid;index<1024;index+=64) {{
          int row=index/32,k=index%32;
          a_tile[index]=get_group_id(1)*32+row<{rows} && base+k<{depth}
            ?x[(get_group_id(1)*32+row)*{depth}+base+k]:0.0f;
          b_tile[k*32+row]=get_group_id(0)*32+row<{columns} && base+k<{depth}
            ?weight[((ulong)(get_group_id(0)*32+row))*{depth}+base+k]:({typ})0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<32 && base+k<{depth};k++) {{
          float4 b=convert_float4(vload4(0,b_tile+k*32+(lid%8)*4));
          if(m<{rows})sum0=fma((float4)a_tile[((lid/8)*4)*32+k],b,sum0);
          if(m+1<{rows})sum1=fma((float4)a_tile[((lid/8)*4+1)*32+k],b,sum1);
          if(m+2<{rows})sum2=fma((float4)a_tile[((lid/8)*4+2)*32+k],b,sum2);
          if(m+3<{rows})sum3=fma((float4)a_tile[((lid/8)*4+3)*32+k],b,sum3);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      {stores}
    }}"""
    return _program('mali_gemm_tiled'+suffix,source,(out,x,weight,*scalars),groups,(64,1,1))
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemm{suffix}(__global float *out,__global const float *x,__global const {typ} *weight{scalar_decl}) {{
    int lid=get_local_id(0),m=get_group_id(1)*32+(lid/8)*4,n=get_group_id(0)*32+(lid%8)*4;
    float4 sum0=(float4)0,sum1=(float4)0,sum2=(float4)0,sum3=(float4)0;
    for(int k=0;k<{depth};k++) {{
      float4 b={load};
      if(m<{rows})sum0=fma((float4)x[m*{depth}+k],b,sum0);
      if(m+1<{rows})sum1=fma((float4)x[(m+1)*{depth}+k],b,sum1);
      if(m+2<{rows})sum2=fma((float4)x[(m+2)*{depth}+k],b,sum2);
      if(m+3<{rows})sum3=fma((float4)x[(m+3)*{depth}+k],b,sum3);
    }}
    {stores}
  }}'''
  return _program('mali_gemm'+suffix,source,(out,x,weight,*scalars),groups,(64,1,1))


def gemm(x:Tensor, weight:Tensor, packed:Tensor|None=None) -> Tensor:
  columns,depth = weight.shape
  assert isinstance(columns,int) and isinstance(depth,int)
  rows = math.prod(x.max_shape[:-1])
  geometry = _small_gemm_geometry(rows,columns,depth,packed is not None)
  if geometry is not None and (x.dtype is not dtypes.float or weight.dtype is not dtypes.half or x.device!=weight.device
      or (packed is not None and (packed.dtype is not dtypes.half or packed.device!=x.device
                                  or packed.shape!=(ceildiv(columns,32),depth,32))) or not _small_gemm_supported(x.device)):
    geometry = None
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  if (geometry is None and x.dtype is dtypes.float and weight.dtype in (dtypes.half,dtypes.float)
      and isinstance(x.device,str) and x.device.split(':')[0]=='CL' and x.device==weight.device
      and (count:=valid_rows(x,32)) is not None):
    result = row_kernel(output,(x.pad_to(x.max_shape).contiguous(),packed if packed is not None else weight.contiguous()),count,
      lambda o,a,w,actual,scalars:_gemm(o,a,w,rows,columns,depth,packed is not None,actual,scalars))
    return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))
  result = output.custom_kernel(x.pad_to(x.max_shape).contiguous(),packed if packed is not None else weight.contiguous(),
    fxn=lambda o,a,w:(_gemm(o,a,w,rows,columns,depth,packed is not None) if geometry is None
                     else _small_gemm(o,a,w,rows,columns,depth,packed is not None,geometry)))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


@functools.cache
def _gemv(out:UOp, x:UOp, weight:UOp, columns:int, depth:int) -> UOp:
  typ = 'half' if weight.dtype is dtypes.half else 'float'
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemv(__global float *out,__global const float *x,__global const {typ} *weight) {{
    int lane=get_local_id(0)%16,row=get_group_id(0)*16+get_local_id(0)/16;
    float4 sum=(float4)0;
    if(row<{columns})for(int k=lane*4;k<{depth};k+=64)
      sum=fma(vload4(0,x+k),convert_float4(vload4(0,weight+((ulong)row)*{depth}+k)),sum);
    __local float partial[256];
    partial[get_local_id(0)]=(sum.x+sum.y)+(sum.z+sum.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && row<{columns}) {{float result=0;
      for(int k=0;k<16;k++)result+=partial[get_local_id(0)+k];out[row]=result;}}
  }}'''
  return _program('mali_gemv',source,(out,x,weight),(ceildiv(columns,16),1,1),(256,1,1))


def gemv(x:Tensor, weight:Tensor) -> Tensor:
  columns,depth = weight.shape
  assert isinstance(columns,int) and isinstance(depth,int) and depth%4==0
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  return output.custom_kernel(x.contiguous(),weight.contiguous(),fxn=lambda o,a,w:_gemv(o,a,w,columns,depth))[0]


@functools.cache
def _gemv_rows(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int) -> UOp:
  # Share weight loads across a short verification block, retaining every
  # scalar GEMV lane, FMA, component sum and final lane reduction in its order.
  typ = 'half' if weight.dtype is dtypes.half else 'float'
  declarations = ''.join(f'float4 sum{i}=(float4)0;' for i in range(rows))
  arithmetic = ''.join(f'sum{i}=fma(vload4(0,x+{i*depth}+k),w,sum{i});' for i in range(rows))
  partials = ''.join(f'partial[{i*256}+lid]=(sum{i}.x+sum{i}.y)+(sum{i}.z+sum{i}.w);' for i in range(rows))
  stores = ''.join(f'''{{float result=0;
    for(int k=0;k<16;k++)result+=partial[{i*256}+lid+k];out[{i*columns}+column]=result;}}''' for i in range(rows))
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemv_rows(__global float *out,__global const float *x,__global const {typ} *weight) {{
    int lid=get_local_id(0),lane=lid%16,column=get_group_id(0)*16+lid/16;
    {declarations}
    if(column<{columns})for(int k=lane*4;k<{depth};k+=64) {{
      float4 w=convert_float4(vload4(0,weight+((ulong)column)*{depth}+k));
      {arithmetic}
    }}
    __local float partial[{rows*256}];
    {partials}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && column<{columns}) {{{stores}}}
  }}'''
  return _program('mali_gemv_rows',source,(out,x,weight),(ceildiv(columns,16),1,1),(256,1,1))


@functools.cache
def _gemv_rows_tiled(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int, group:int=16, tile:int=2) -> UOp:
  size = group*16
  tiles = ceildiv(rows,tile)
  typ = 'half' if weight.dtype is dtypes.half else 'float'
  name = f'mali_gemv_rows_t{tile}_g{group}'
  declarations = ''.join(f'float4 sum{i}=(float4)0;' for i in range(tile))
  arithmetic = ''.join(f'if(m+{i}<{rows})sum{i}=fma(vload4(0,x+(m+{i})*{depth}+k),w,sum{i});' for i in range(tile))
  partials = ''.join(f'partial[{i*size}+lid]=(sum{i}.x+sum{i}.y)+(sum{i}.z+sum{i}.w);' for i in range(tile))
  stores = ''.join(f'''if(m+{i}<{rows}){{float result=0;
    for(int k=0;k<16;k++)result+=partial[{i*size}+lid+k];out[(m+{i})*{columns}+column]=result;}}''' for i in range(tile))
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void {name}(__global float *out,__global const float *x,__global const {typ} *weight) {{
    int lid=get_local_id(0),lane=lid%16;
    int column=(get_group_id(0)/{tiles})*{group}+lid/16;
    int m=(get_group_id(0)%{tiles})*{tile};
    {declarations}
    if(column<{columns})for(int k=lane*4;k<{depth};k+=64) {{
      float4 w=convert_float4(vload4(0,weight+((ulong)column)*{depth}+k));
      {arithmetic}
    }}
    __local float partial[{tile*size}];
    {partials}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && column<{columns}) {{{stores}}}
  }}'''
  return _program(name,source,(out,x,weight),(ceildiv(columns,group)*tiles,1,1),(size,1,1))


def gemv_rows(x:Tensor, weight:Tensor) -> Tensor:
  """Short dense row block using the scalar GEMV accumulation order."""
  if (x.dtype is not dtypes.float or weight.dtype not in (dtypes.half,dtypes.float)
      or not isinstance(x.device,str) or x.device.split(':')[0]!='CL' or x.device!=weight.device):
    raise ValueError('GEMV rows requires FP32 OpenCL inputs and same-device FP16/FP32 weights')
  if len(weight.shape)!=2 or not x.shape or x.shape[-1]!=weight.shape[-1]:raise ValueError('GEMV rows depth mismatch')
  columns,depth = weight.shape
  rows = math.prod(x.max_shape[:-1])
  if not isinstance(columns,int) or not isinstance(depth,int) or depth%4 or not 1<=rows<=8:
    raise ValueError('GEMV rows requires static aligned weights and row capacity 1..8')
  # Keep each output's K-lane arithmetic; reduce live accumulators for the
  # qualified Mali head/down shapes and interleave tiles for weight-cache reuse.
  tiled = (weight.dtype is dtypes.half and ((rows==8 and (columns,depth)==(248320,2048))
           or (rows in (3,8) and (columns,depth)==(2048,6144))))
  if tiled:tiled = _small_gemm_supported(x.device)
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  result = output.custom_kernel(x.pad_to(x.max_shape).contiguous(),weight.contiguous(),
    fxn=lambda o,a,w:(_gemv_rows_tiled(o,a,w,rows,columns,depth) if tiled else _gemv_rows(o,a,w,rows,columns,depth)))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


@functools.cache
def _gemv_half_load_rows(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int, width:int) -> UOp:
  if width not in (4,8,16):raise ValueError(width)
  grouped=columns<=2048 and depth%16==0
  if depth%width or (grouped and (depth//16)%width):raise ValueError('Unaligned vector load')
  declarations=''.join(f'float sum{i}=0.0f;' for i in range(rows))
  fields='0123456789abcdef'[:width]
  arithmetic=''.join(f'''{{half{width} a=vload{width}(0,x+{i*depth}+k);
    '''+''.join(f'sum{i}=sum{i}+(float)(a.s{field}*w.s{field});' for field in fields)+'}' for i in range(rows))
  if grouped:
    loop=f'for(int k=lane*{depth//16};k<(lane+1)*{depth//16};k+={width})'
    partials=''.join(f'partial[{i*16}+lane]=sum{i};' for i in range(rows))
    stores=''.join(f'''{{float result=0.0f;
      for(int k=0;k<16;k++)result+=partial[{i*16}+k];out[{i*columns}+column]=(half)result;}}''' for i in range(rows))
    finish=f'__local float partial[{rows*16}];{partials}\nbarrier(CLK_LOCAL_MEM_FENCE);if(lane==0){{{stores}}}'
    coordinate='int column=get_group_id(0),lane=get_local_id(0);'
    groups,local=(columns,1,1),(16,1,1)
  else:
    loop=f'for(int k=0;k<{depth};k+={width})'
    finish=''.join(f'out[{i*columns}+column]=(half)sum{i};' for i in range(rows))
    coordinate=f'int column=get_global_id(0);if(column>={columns})return;'
    groups,local=((columns+63)//64,1,1),(64,1,1)
  name=f'mali_gemv_half_load{width}_'+('grouped' if grouped else 'sequential')
  source=f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void {name}(__global half *out,__global const half *x,__global const half *weight) {{
    {coordinate}
    {declarations}
    {loop} {{
      half{width} w=vload{width}(0,weight+((ulong)column)*{depth}+k);
      {arithmetic}
    }}
    {finish}
  }}'''
  return _program(name,source,(out,x,weight),groups,local)


@functools.cache
def _gemv_half_rows(out:UOp, x:UOp, weight:UOp, rows:int, columns:int, depth:int) -> UOp:
  # The generic scalar FP16 projection adds converted half products in K order.
  # Share each weight vector across rows without changing that expression.
  declarations = ''.join(f'float sum{i}=0.0f;' for i in range(rows))
  # Match the current scalar compiler's grouped reduction for small output
  # widths, regardless of the number of verification rows in this launch.
  if columns<=2048 and depth%16==0:
    arithmetic = ''.join(f'sum{i}=sum{i}+(float)(x[{i*depth}+k]*w);' for i in range(rows))
    partials = ''.join(f'partial[{i*16}+lane]=sum{i};' for i in range(rows))
    stores = ''.join(f'''{{float result=0.0f;
      for(int k=0;k<16;k++)result+=partial[{i*16}+k];out[{i*columns}+column]=(half)result;}}''' for i in range(rows))
    source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void mali_gemv_half_grouped_rows(__global half *out,__global const half *x,__global const half *weight) {{
      int column=get_group_id(0),lane=get_local_id(0);
      {declarations}
      for(int k=lane*{depth//16};k<(lane+1)*{depth//16};k++) {{
        half w=weight[((ulong)column)*{depth}+k];
        {arithmetic}
      }}
      __local float partial[{rows*16}];
      {partials}
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lane==0) {{{stores}}}
    }}'''
    return _program('mali_gemv_half_grouped_rows',source,(out,x,weight),(columns,1,1),(16,1,1))
  arithmetic = ''.join(f'''{{half4 a=vload4(0,x+{i*depth}+k);
    sum{i}=sum{i}+(float)(a.x*w.x)+(float)(a.y*w.y)+(float)(a.z*w.z)+(float)(a.w*w.w);}}''' for i in range(rows))
  stores = ''.join(f'out[{i*columns}+column]=(half)sum{i};' for i in range(rows))
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemv_half_rows(__global half *out,__global const half *x,__global const half *weight) {{
    int column=get_global_id(0);
    if(column>={columns})return;
    {declarations}
    for(int k=0;k<{depth};k+=4) {{
      half4 w=vload4(0,weight+((ulong)column)*{depth}+k);
      {arithmetic}
    }}
    {stores}
  }}'''
  return _program('mali_gemv_half_rows',source,(out,x,weight),(ceildiv(columns,64),1,1),(64,1,1))


def gemv_half_rows(x:Tensor, weight:Tensor) -> Tensor:
  """Short FP16 row block with the generic scalar projection's reduction order."""
  if (x.dtype is not dtypes.half or weight.dtype is not dtypes.half
      or not isinstance(x.device,str) or x.device.split(':')[0]!='CL' or x.device!=weight.device):
    raise ValueError('Half GEMV rows requires same-device FP16 OpenCL inputs and weights')
  if len(weight.shape)!=2 or not x.shape or x.shape[-1]!=weight.shape[-1]:raise ValueError('Half GEMV rows depth mismatch')
  columns,depth = weight.shape
  rows = math.prod(x.max_shape[:-1])
  if not isinstance(columns,int) or not isinstance(depth,int) or columns<1 or depth<4 or depth%4 or not 1<=rows<=8:
    raise ValueError('Half GEMV rows requires static aligned weights and row capacity 1..8')
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.half)
  # Qualify load width by complete projection shape and captured row capacity.
  # Wider loads at capacity eight regress the wide projection on this device.
  width = ({1:8,3:8,8:4}.get(rows) if (columns,depth)==(2048,2048)
           else {1:8,3:8}.get(rows) if (columns,depth)==(6144,2048) else None)
  if width is not None and not _small_gemm_supported(x.device):width = None
  result = output.custom_kernel(x.pad_to(x.max_shape).contiguous(),weight.contiguous(),
    fxn=lambda o,a,w:(_gemv_half_rows(o,a,w,rows,columns,depth) if width is None
                     else _gemv_half_load_rows(o,a,w,rows,columns,depth,width)))[0]
  return result.shrink(tuple((0,size) for size in (*x.shape[:-1],columns)))


@functools.cache
def _gemv_q8(out:UOp, x:UOp, raw:UOp, columns:int, depth:int) -> UOp:
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_gemv_q8(__global float *out,__global const float *x,__global const uchar *raw) {{
    int lane=get_local_id(0)%16,row=get_group_id(0)*16+get_local_id(0)/16;
    float4 sum=(float4)0;
    if(row<{columns})for(int k=lane*4;k<{depth};k+=64) {{
      ulong block=(((ulong)row)*{depth//32}+k/32)*34;
      float scale=(float)(*(__global const half *)(raw+block));
      char4 q=vload4(0,(__global const char *)(raw+block+2+k%32));
      float4 w=convert_float4(convert_half4_rte(convert_float4(q)*(float4)scale));
      sum=fma(vload4(0,x+k),w,sum);
    }}
    __local float partial[256];partial[get_local_id(0)]=(sum.x+sum.y)+(sum.z+sum.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lane==0 && row<{columns}) {{float result=0;
      for(int k=0;k<16;k++)result+=partial[get_local_id(0)+k];out[row]=result;}}
  }}'''
  return _program('mali_gemv_q8',source,(out,x,raw),(ceildiv(columns,16),1,1),(256,1,1))


def gemv_q8(x:Tensor, raw:Tensor, columns:int, depth:int) -> Tensor:
  assert depth%32==0 and raw.dtype is dtypes.uint8 and raw.numel()==columns*(depth//32)*34
  assert math.prod(x.max_shape[:-1])==1
  output = Tensor.empty(*x.max_shape[:-1],columns,device=x.device,dtype=dtypes.float)
  return output.custom_kernel(x.contiguous(),raw.contiguous(),fxn=lambda o,a,w:_gemv_q8(o,a,w,columns,depth))[0]


@functools.cache
def _embedding(out:UOp, ids:UOp, weight:UOp, rows:int, vocab:int, dim:int) -> UOp:
  word_type = 'half' if weight.dtype is dtypes.half else 'float'
  index_type = 'int' if ids.dtype is dtypes.int else 'long'
  source = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void mali_embedding(__global float *out,__global const {index_type} *ids,__global const {word_type} *weight) {{
    int i=get_global_id(0);if(i>={rows*dim})return;
    {index_type} id=ids[i/{dim}];
    out[i]=id>=0 && id<{vocab}?(float)weight[((ulong)id)*{dim}+i%{dim}]:0;
  }}'''
  return _program('mali_embedding',source,(out,ids,weight),(ceildiv(rows*dim,128),1,1),(128,1,1))


def embedding(ids:Tensor, weight:Tensor) -> Tensor:
  vocab,dim = weight.shape
  assert isinstance(vocab,int) and isinstance(dim,int)
  count = math.prod(ids.max_shape)
  output = Tensor.empty(*ids.max_shape,dim,device=weight.device,dtype=dtypes.float)
  result = output.custom_kernel(ids.pad_to(ids.max_shape).contiguous(),weight.contiguous(),
    fxn=lambda o,i,w:_embedding(o,i,w,count,vocab,dim))[0]
  return result.shrink(tuple((0,size) for size in (*ids.shape,dim)))
