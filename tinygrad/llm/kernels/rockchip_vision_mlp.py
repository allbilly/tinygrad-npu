"""FP16 vision MLP with fused CPU bias, FP32 GELU and down-projection packing."""
import functools
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.rockchip import RockchipLinear, packed_matmul, packed_activation_matmul, batched_matmul, packed_tiles_matmul
from tinygrad.llm.kernels.rockchip_cpu import _renderer, _vector_exp2_source, unpack_tiles
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

@functools.cache
def _gelu_pack(out:UOp, raw:tuple[UOp,...], bias:UOp, columns:int, rows:int, input_rows:int, last:int, tile_rows:int) -> UOp:
  params = ','.join(f'const float *p{i}' for i in range(len(raw)))
  pointers = ','.join(f'p{i}' for i in range(len(raw)))
  bias_type = '__fp16' if bias.dtype is dtypes.half else 'float'
  load_bias = '__builtin_convertvector(*(const half4*)(bias+c),float4)' if bias_type=='__fp16' else '*(const float4*)(bias+c)'
  parallel = rows>=64
  declarations = f'''typedef unsigned long thread_t;
  extern int pthread_create(thread_t*,const void*,void*(*)(void*),void*);
  extern int pthread_join(thread_t,void**);
  typedef struct{{__fp16 *out;const float *const *parts;const {bias_type} *bias;int begin,end;}} Work;
  static void *worker(void *opaque){{Work *w=(Work*)opaque;part(w->out,w->parts,w->bias,w->begin,w->end);return (void*)0;}}''' if parallel else ''
  dispatch = f'''Work work[4];thread_t threads[3];int created[3];
    for(int i=0;i<4;i++)work[i]=(Work){{out,parts,bias,{rows}*i/4,{rows}*(i+1)/4}};
    for(int i=0;i<3;i++){{created[i]=pthread_create(&threads[i],(void*)0,worker,&work[i])==0;
      if(!created[i])part(out,parts,bias,work[i].begin,work[i].end);}}
    part(out,parts,bias,work[3].begin,work[3].end);
    for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);''' if parallel else f'part(out,parts,bias,0,{rows});'
  source = f'''{_vector_exp2_source()}
  static __attribute__((always_inline)) float4 vision_exp2(float4 x){{
    // GELU's FP32 sigmoid/tanh expression saturates outside these bounds.
    // Inside the interval, multiplying by a single normal power of two is exact.
    int4 nan=x!=x;float4 safe=nan?(float4)0:x;
    safe=safe<(float4)-24.f?(float4)-24.f:safe;
    safe=safe>(float4)27.f?(float4)27.f:safe;
    int4 q=__builtin_convertvector(safe+(safe<0?(float4)-.5f:(float4).5f),int4);
    float4 s=safe-__builtin_convertvector(q,float4);
    float4 p=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
      0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.f);
    return nan?(float4)__builtin_nanf(""):p*__builtin_bit_cast(float4,(q+127)<<23);
  }}
  static void part(__fp16 *out,const float *const *parts,const {bias_type} *bias,int begin,int end){{
    for(int row=begin;row<end;row++){{
      int tile=row/{input_rows},local=row%{input_rows},m=tile=={len(raw)-1}?{last}:{input_rows};
      int block=row/{tile_rows},active={rows}-block*{tile_rows};if(active>{tile_rows})active={tile_rows};
      __fp16 *dst=out+block*{tile_rows*columns}+(row%{tile_rows})*8;
      for(int c=0;c<{columns};c+=4){{
        float4 value=*(const float4*)(parts[tile]+(c/4*m+local)*4)+{load_bias};
        float4 cube=(value*value)*value;
        float4 inner=(value+cube*0.044715f)*0.7978845608028654f;
        float4 sigmoid=1.0f/(1.0f+vision_exp2((inner*2.0f)*-1.4426950216293335f));
        float4 activated=(value*.5f)*(1.0f+(2.0f*sigmoid-1.0f));
        *(half4*)(dst+(c/8)*active*8+c%8)=__builtin_convertvector(activated,half4);
      }}
    }}
  }}
  {declarations}
  void rockchip_vision_gelu_pack(__fp16 *out,{params},const {bias_type} *bias){{
    const float *parts[{len(raw)}]={{{pointers}}};{dispatch}
  }}'''
  inputs = (out,*raw,bias)
  sink = UOp.sink(*inputs,arg=KernelInfo('rockchip_vision_gelu_pack'))
  renderer = _renderer()
  info = replace(ProgramInfo.from_sink(sink,renderer.target),outs=(0,),ins=tuple(range(1,len(inputs))))
  return UOp(Ops.PROGRAM,arg=info,src=(sink,UOp(Ops.LINEAR,src=inputs),UOp(Ops.SOURCE,arg=source),
             UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source))))

def gelu_pack(parts:list[Tensor], bias:Tensor, columns:int, tile_rows:int) -> Tensor:
  if not parts or len(parts)>64 or columns%32 or not 32<=columns<=4096 or not 1<=tile_rows<=min(64,65536//columns):
    raise ValueError('Unsupported vision GELU packing bounds')
  counts = [part.numel()//columns for part in parts]
  if (any(p.device!='CPU' or p.dtype is not dtypes.float or p.numel()!=count*columns for p,count in zip(parts,counts))
      or any(c!=counts[0] for c in counts[:-1]) or not 1<=counts[-1]<=counts[0]<=64
      or bias.device!='CPU' or bias.dtype not in (dtypes.half,dtypes.float) or bias.shape!=(columns,)):
    raise ValueError('Unsupported vision GELU input layout')
  rows = sum(counts)
  out = Tensor.empty(rows*columns,device='CPU',dtype=dtypes.half)
  return out.custom_kernel(*parts,bias,fxn=lambda o,*args:_gelu_pack(o,tuple(args[:-1]),args[-1],
    columns,rows,counts[0],counts[-1],tile_rows))[0]

class FP16VisionMLP:
  def __init__(self, fc1:RockchipLinear, fc2:RockchipLinear): self.fc1,self.fc2 = fc1,fc2

  def __call__(self, x:Tensor) -> Tensor:
    if (x.device!='CPU' or x.dtype is not dtypes.float or x.shape!=(1,1024,768) or self.fc1.depth!=768 or self.fc1.columns!=3072
        or self.fc2.depth!=3072 or self.fc2.columns!=768 or len(self.fc1.packed)!=1 or len(self.fc2.packed)!=1
        or self.fc1.bias is None or self.fc2.bias is None or self.fc1.prefill!='on' or self.fc2.prefill!='on'):
      return self.fc2(self.fc1(x).gelu(approximate='tanh'))
    dense = x.reshape(1024,768)
    parts = ([part.flatten() for part in batched_matmul(dense,self.fc1.packed[0],3072)] if self.fc1.batch_prefill else
             [packed_matmul(dense[start:start+64],self.fc1.packed[0],3072,raw_output=True) for start in range(0,1024,64)])
    packed = gelu_pack(parts,self.fc1.bias,3072,21).to('ROCKCHIP')
    outputs = (packed_tiles_matmul(packed,self.fc2.packed[0],1024,768,21) if self.fc2.batch_prefill else
               [packed_activation_matmul(packed,self.fc2.packed[0],min(21,1024-start),768,start,raw_output=True)
                for start in range(0,1024,21)])
    return unpack_tiles(outputs).reshape(1,1024,768)+self.fc2.bias
