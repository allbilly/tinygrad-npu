"""Opt-in CPU preparation rewrites retaining reviewed arithmetic and layouts."""
from __future__ import annotations
import functools,hashlib
from typing import TYPE_CHECKING
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops,ProgramInfo,UOp
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_prepare_pool import _split_independent_rows
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_CPU_QUERY_NORM_ROWS = ContextVar('ONPU_CPU_QUERY_NORM_ROWS',0)

_QUERY_NORM_SOURCES = {
  '4091fbb06aefd2334f25855cd40d4a16578774a64c5022844d498036560cde48': 'r_8_128_64_4',
  'bfbf8f36b02cf0f12984073aee68eb0804ac2fab28ac41135006e21a3e5df62c': 'r_8_128_64_4',
  '3c05b7f5aca971c09b9c80668a6b99b88d423c0fae99d63df0b2abb403e3c1ba': 'r_8_128_128_4',
  '840bf527dba7835a3aab7b62c9478d704f516e8e14073498f51ceed60838a680': 'r_8_128_128_4',
}

_ROW_SOURCES = {
  '01f4a25a6326b3327414a00fffe0123f901a278d483c3e5e2d3f0476a0c98565': 'r_128_512_4',
  '0cf3654911173ac39178bd7ea55dbd3cb97e584b1c1eeb3507486c6526e1fdb6': 'r_128_512_4',
  '221c6e1f4bdc1979669e8b60073e018397b2d216d78b2be191cadba2212743e5': 'r_128_512_4',
  '2abfa9c634d0818104f201a46778cf68c73ed56dda495102c97147041278162f': 'E_32_512_4_4',
  '763dab993d83ac4369a9d809e2f326e043710a2f5c244e9e91bfd2b6e5b8a631': 'E_32_512_4_4',
  '916d6e4b6bc8a28748359ba47b4a8fe12406099c7a29a12c9d3361bcd74657aa': 'r_128_512_4',
  'bac9180b76f15d9f78f35cd6428cbefb3a94916fec4a4fc546a3c012e897a716': 'r_128_512_4',
  'ca7c43b39c26678c4b098da85f2db5021251080b72e020edebc61092ad7b549f': 'E_32_512_4_4',
  'cbb62c27578e5920d3f7d59a6c6b4849f00fc37e0199ecde8c0d9a1b6301ddb2': 'r_128_512_4',
  'e155c0fdcb399a4103bc924d44623139f31a293f2d86b82002cf4536b95cc20d': 'E_32_512_4_4',
  '8d64de58dccb7a46dbd2ca6726ebebfea0c097685ed7c4b747f50f3d020bfa79': 'E_32_512_4_4',
  # Gemma4: independent residual/RMS normalization rows, with original reductions.
  'e25a78cfe972b4fba3a7bfb00724fa5f794fe3771d2b1c5648b65e0a85fda825': 'E_32_512_3_4',
  'db701eb855819bcc1b50f205ac54e696ea474f699540e477534d1e3e40b459f4': 'E_32_512_3_4',
  '77150f3ad1851009aab9c3465c22eb04cc6ca2758eed2e37b7ecdf00ac17f412': 'E_32_512_3_4',
  '50bd258fd700f162d076321660cf39e212e9c9c13ef4083976fc1f7736a6f90c': 'E_32_512_3_4',
  'b17a81cf5a8a39f094126cc1177d3bbfa00ae0aaf115cff1c6e3bce16e4ab0ff': 'r_128_384_4',
  'fdc5b5b9742951b070eb25d641b57c3a30032b8b3763fe9d29b118b017036854': 'r_128_384_4',
  '0dc14ff7fb565d456b8ab8e5f0a9940fe79173887a376b5cd916815ffc7a189a': 'r_128_384_4',
}
_SPARSE_LOOKUP_SHA = 'e88e39c989f1d09637e07cb8f0ba5573763e8990e115a51add776fd559be206d'
_SPARSE_LOOKUP_SOURCE = """/* ONPU_EXACT_SPARSE_LOOKUP: retain original padded gather layout. */
void E_64_32_64_4(__fp16* restrict data0_524288,int* restrict data1_1,unsigned char* restrict data2_570425344) {
  int token=*data1_1;
  __builtin_memset(data0_524288,0,1048576UL);
  if((unsigned)token>=262144U)return;
  int group=token>>12,lane=(token>>10)&3;
  for(int block=0;block<64;block++){
    const unsigned char *p=data2_570425344+token*2176+block*34;
    unsigned short bits=(unsigned short)p[0]+((unsigned short)p[1]<<8);
    float scale=(float)__builtin_bit_cast(__fp16,bits);
    for(int j=0;j<32;j++){
      float value=scale*(float)__builtin_bit_cast(signed char,p[j+2]);
      data0_524288[(block<<13)+(j<<8)+(group<<2)+lane]=(__fp16)value;
    }
  }
}
"""


@functools.cache
def _replace(program:UOp,source:str) -> UOp:
  return program.replace(src=tuple(UOp(Ops.SOURCE,arg=source) if u.op is Ops.SOURCE else
    UOp(Ops.BINARY,arg=_renderer().compiler.compile_cached(source)) if u.op is Ops.BINARY else u for u in program.src))


def rewrite_exact_preparation(linear:UOp, *, pool:AttentionPool|None=None, rows:bool=False, sparse_lookup:bool=False,
                              phase:str='prefill', records:list[dict]|None=None) -> UOp:
  """Unknown programs retain their original implementation; overlapping rows run serially."""
  if type(rows) is not bool or type(sparse_lookup) is not bool:raise ValueError('Exact preparation selection must be bool')
  if rows and pool is None:raise ValueError('Exact row preparation requires the existing CPU workers')
  replacements={}
  for program in linear.toposort(enter_calls=True):
    if (program.op is not Ops.PROGRAM or not isinstance(program.arg,ProgramInfo) or program.arg.target.device!='CPU'
        or program.arg.outs!=(0,) or 0 in program.arg.ins):continue
    sources=[u.arg for u in program.src if u.op is Ops.SOURCE]
    if len(sources)!=1:continue
    source=sources[0]
    digest=hashlib.sha256(source.encode()).hexdigest()
    name=program.src[0].arg.function_name
    if rows and phase=='prefill' and (_ROW_SOURCES.get(digest)==name or
                                    ONPU_CPU_QUERY_NORM_ROWS and _QUERY_NORM_SOURCES.get(digest)==name):
      assert pool is not None
      address=pool.prefill_address
      assert address is not None
      source,_=_split_independent_rows(source,name,address)
      source='/* ONPU_EXACT_ROW_PARTITION */\n'+source
      kind='independent_rows'
    elif sparse_lookup and digest==_SPARSE_LOOKUP_SHA and name=='E_64_32_64_4':
      source=_SPARSE_LOOKUP_SOURCE
      kind='sparse_lookup'
    else:continue
    replacements[program]=_replace(program,source)
    record={'kind':kind,'phase':phase,'source_before_sha256':digest,'source_after_sha256':hashlib.sha256(source.encode()).hexdigest()}
    if records is not None and record not in records:records.append(record)
  return linear.substitute(replacements,walk=True,enter_calls=True) if replacements else linear
