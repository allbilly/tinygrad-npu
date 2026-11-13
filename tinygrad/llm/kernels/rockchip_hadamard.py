"""Opt-in fixed-sign block-Hadamard preprocessing for CPU INT8 matrices.

Activations and normalized weight transforms cancel before quantization.
This changes quantization arithmetic and requires model-quality qualification.
"""
from __future__ import annotations
import functools,re
from tinygrad import Tensor,dtypes
from tinygrad.helpers import ContextVar,round_up
from tinygrad.llm.kernels.rockchip_shared_pool import shared_program
from tinygrad.uop.ops import Ops,UOp

ONPU_INT8_HADAMARD = ContextVar('ONPU_INT8_HADAMARD',0)
B=128
# PCG64 seed1281536+depth, packed signs; no NumPy runtime dependency.
_SIGN_BITS = {
  2048: bytes.fromhex(
    'cb682a8bdde825ab09c226fdab116979a245c35c43f79611c4f05bc1a6425c65d041b719cb7584ac354c5ca99d2d7f979e841afbf86f5cfb'
    '8fed0dc6c93d50d8575992c93d090b697f69dbf3743355da653eaea5564ce2fa6406a7172fef7d0fadebe9a1b7535f36cbc0d0c8329108bf'
    '8d45168c1b9fbce411c487d4d4bac4149b3046bcd7ace03c3d372fb8231061a5028e0b99ce0f348fcd12c5f8581d6db75042d72f5c80e92a'
    '8425218e9a5acfb6f6b16ce786d0367e55f741bdcbf4117a2f993a3c0e48202f60969ad99a455ee2a909d9b850e8d945c170f20a96a48362'
    '8c89986648769a0c57ec04e3124618df38ac30208eabad9c65f71feb87bc7a14'
  ),
  3072: bytes.fromhex(
    '84fa67cc48027176442b3ec6ed8484d756acf2105a7f5b089c5c11a4648c5592425fe88045a19a7a0eb3aedf52f081ea6f43523efba03c5e'
    '951875fbfabb41a7a0d4db1d0140f8718aee07d8ec89b355367e1a7342f789177a177b436a108ef3055bbd516a88fbdafcda04f43812dfb1'
    'f37e5c00798a37f7539bf0cc95ccbd12d96574bca13fac8009f5305de49d2dac022d34fdce1379c79d091f10faa8f9d3979af5a494a1bff4'
    '2ad24db1d3fd9a1a3255c20c84d94393aac1e7106d077978273b550569954ebf03f3449553bcd87e61ce628d0abe5b2c50d3d9a72ae524f0'
    'b7cf85c2c5fd7f22745a50b6ad8917fb99b50b08ef362738017494b9ee0079a42cca2170659d8cf94a8dcc3ece3db165a128bd329c8ea8c1'
    'dd9076f00d253baf80ea284947c3fe1d8731b548df9dc81acfbc2f3fca4e199bce571aaf6a3e42960175d1e9cbcb585eeca450d19a9251fa'
    'bf9175fc4e9e058607893d7a3bac684d9733af54094ab4513ed7dc4400773dbe79803afae171c81cb843a8c2d5da3d70'
  ),
  4096: bytes.fromhex(
    '2469578933e3b0bfd664b859d6fa396d5f6c810050f5f95ca304480340e4a3022258a05867cdb93b859f4acf5198b23f57cf45f765545aac'
    '41576b53a679b1d89a1725e3dc026cc451441159ec54dad194c48f68cb31e3a91fc2838ac957af1f54f0ebf00cb12293cb85530381c14abf'
    '3052319c716a75648929623f33a86d1076959d578e5ba14724419b84e018c610c59dad84f502b649c178be340bc25c750b4c4dcd459d43b1'
    '6f9aad35affb1cb39f9abdf956cfaffcaefea263d81cb5dfc58ef0521059ebc1ec70edb0aead4a137ff71a3ffa9c931731677d853d731e6b'
    '4430e13c71add6d336fab256f47d4a73061372af072f47a24971b2d7e059793aadbfe75f7afd8e89fc909242ea5fd441e98af1a6b913bb47'
    'e02615d21bc3f17dce30de748ef71cec80fca516f0a5130287dbc57cc57f966d884de8d3b7fa03b0d73fae5025e5c35571421dd5086aa000'
    '69275b78fdc690bad2f0a4cc5980239e07eca8989f77736c708f2fe10f39ec30581b35b9e533f9e5cc983c8cb80480856efc38458a8106a7'
    '84334fb67145bb72c20d11f45c86df0727592b891f22fdd3360c0a4d490cce7cb2dc844941918b2e97879750f248fa150f9bf7a6d5dd0d3f'
    '5ee2aeb2980b185031457079b63bb29d254bd5a06e5d270406798b22debd8616cdf82d2926dd06af48155196e88a2d191cb33690e4dadfa3'
    '85456b6cc3a9e03c'
  ),
}


def supported(depth:int) -> bool:return depth in _SIGN_BITS


def enabled(depth:int) -> bool:
  if ONPU_INT8_HADAMARD.value not in (0,128):raise ValueError('Unsupported INT8 Hadamard block')
  return bool(ONPU_INT8_HADAMARD and supported(depth))


@functools.cache
def signs(depth:int) -> tuple[int,...]:
  if not supported(depth):raise ValueError('Unsupported Hadamard depth')
  bits=_SIGN_BITS[depth]
  return tuple(-1 if bits[c//8]>>(c%8)&1 else 1 for c in range(depth))


@functools.cache
def transform_source(depth:int,weight:bool=False) -> str:
 sign=','.join('1' if s>0 else '-1' for s in signs(depth))
 return f'''static const signed char sign_{depth}[{depth}]={{{sign}}};
 static __attribute__((always_inline)) void cpu_hadamard_{depth}{'_weight' if weight else ''}(float *dst,const float *src) {{
  for(int c=0;c<{depth};c++)dst[c]=src[c]*(float)sign_{depth}[c];
  for(int base=0;base<{depth};base+={B}) {{
   for(int step=1;step<{B};step*=2)for(int offset=0;offset<{B};offset+=2*step)for(int c=0;c<step;c++) {{
    float a=dst[base+offset+c],b=dst[base+offset+step+c];dst[base+offset+c]=a+b;dst[base+offset+step+c]=a-b;
   }}
  }}
  {'for(int c=0;c<'+str(depth)+';c++)dst[c]=dst[c]*0.0078125f;' if weight else ''}
 }}'''


@shared_program
@functools.cache
def _program(out:UOp,inp:UOp,scales:UOp,rows:int,depth:int,padded:int,weight:bool) -> UOp:
  from tinygrad.llm.kernels.rockchip_int8 import _quantize
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  # Cache the unpooled program; the outer decorator applies each current scope.
  p:UOp=getattr(_quantize,'__wrapped__')(out,inp,scales,rows,depth,padded,False)
  source=next(u.arg for u in p.src if u.op is Ops.SOURCE)
  old=f'const float *x=input+row*{depth};'
  if source.count(old)!=1:raise ValueError('Unexpected CPU quantizer source')
  fn=f'cpu_hadamard_{depth}'+('_weight' if weight else '')
  source=transform_source(depth,weight)+'\n'+source.replace(old,
    f'float rotated[{depth}] __attribute__((aligned(16)));{fn}(rotated,input+row*{depth});const float *x=rotated;')
  return p.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
                            UOp(Ops.BINARY,arg=_renderer().compiler.compile_cached(source)) if u.op is Ops.BINARY else u for u in p.src))


def quantize(x:Tensor,padded:int,*,weight:bool=False) -> tuple[Tensor,Tensor]:
  if x.device!='CPU' or x.ndim!=2:raise ValueError('Unsupported rotated quantization geometry')
  rows,depth=x.shape
  if (not isinstance(rows,int) or not isinstance(depth,int) or not supported(depth) or depth!=padded or not 1<=rows<=24576):
    raise ValueError('Unsupported rotated quantization geometry')
  result=Tensor.empty(round_up(rows*4,64)+rows*padded,device='CPU',dtype=dtypes.uint8).custom_kernel(
    x.float().contiguous(),Tensor.empty(rows,device='CPU',dtype=dtypes.float),
    fxn=lambda o,a,s:_program(o,a,s,rows,depth,padded,weight))
  return result[0],result[2]


def rotate_ffn_source(source:str) -> str:
  helpers:dict[int,str]={}
  def insert(match:re.Match[str]) -> str:
    depth=int(match[2])
    if not supported(depth):return match[0]
    helpers[depth]=transform_source(depth)
    return match[1]+f'''cpu_hadamard_{depth}(values,values);maximum=(float4)0;bad=(uint4)0;
   for(int h=0;h<{depth};h+=4) {{
    float4 v=*(const float4*)(values+h);uint4 bits=__builtin_bit_cast(uint4,v);
    bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;
    float4 a=v>=(float4)0?v:-v;maximum=a>maximum?a:maximum;
   }}'''+match[3]
  source,_=re.subn(r'(float values\[(\d+)\].*?)(float largest=maximum\[0\];)',insert,source,flags=re.S)
  return '\n'.join(helpers.values())+'\n'+source if helpers else source
