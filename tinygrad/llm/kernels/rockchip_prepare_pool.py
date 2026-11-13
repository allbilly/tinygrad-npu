"""Parallelize reviewed, independent outer loops in CPU GDN preparation.

Separate source allowlists gate 128-row prefill preparation and one-row
decode convolution, SiLU and normalization. Each approved outer iteration
writes a disjoint output region; reduction scratch remains private to each
callback. Compiler/layout changes fall back until reviewed and qualified.
"""
from __future__ import annotations
import functools, hashlib, re
from typing import TYPE_CHECKING
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops, ProgramInfo, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool

ONPU_GDN_PREP_POOL = ContextVar('ONPU_GDN_PREP_POOL',0)
ONPU_GDN_DECODE_PREP_POOL = ContextVar('ONPU_GDN_DECODE_PREP_POOL',0)
ONPU_LFM_FLAT_POINTWISE = ContextVar('ONPU_LFM_FLAT_POINTWISE',0)
# LFM prefill pointwise programs with a 32768-row independent outer loop.
# Exactness is established by the flat-pointwise CPU checks before enabling.
_LFM_FLAT_POINTWISE_SOURCES = frozenset((
  '9f3b59d0a1a5c5e31754406ab11ade199ce2e604e3424f944346d53db523e096',
  '2e344f7d381d487513910faf49cab9db135e3ea76df453748bea3e756d402a7e',
  '9d597135e6f3bbe1cd7a5fa2e5115422d71adb5daa83ae5393cb48eb3714acaa',
  '34060e4e1f83c298b73b48e53d8c3e389ef22b14c0ae19063be05214837879f7',
))
# These exact convolution/normalization programs have a two-way repeat before
# sixteen independent head groups. Split the head axis to use all four workers.
_QUALIFIED_INNER_HEAD_SOURCES = frozenset((
  '7b3fa1e9c51ff92baf891096f2342477590a25ca95fe9b0be7482a0054d6a94b',
  'b02bea8f72ff3faf584061f59df49e553aa86d32fc3de1680eebf3ece27498a5',
))
_QUALIFIED_SOURCES = frozenset((
  # Reviewed 4B convolution/normalization layouts; CPU and full-logit evidence in recurrent_preparation.md.
  'd666c17e20abbde542d728cbcdab4d1d54f4c7fd116794231d9416865b7f2de9',
  '56ff8d8030204ba5be22587f46be32064f121fe9582360d4a8e63422f6192d54',
  '262a7a7ae61f49389eb80894d3c5925933452c7c7f1e2cfcb5977c49f4980f05',
  '69327071c639f0a96f0dcc48e427eb147cfdfdc982a422b50a94e8e813b8f4c5',
  'b11b7f2823f785119740ba5e29f62acfa383691f5debbe2a0d274dcfa8232ae4',
  'ab6e78782b3a12ce93186f7857b6fa7a21c0bcb5e4e14f2de513dfca6681a45c',
  'ee4d9430305e35adfa0880c94d8a6e61124801a54c6e7ee33bb71df2fe060051',
  '356d1b959da7d568e80d248d1b2842cc31c584688d7fb4c6e10585f29ea5ae35',
  '38e418e244f7ffebac51d2c8d44465c5660b59f26bf6867ecddaee785211a77a',
  '7c1c9b1db05fc6fc2c8bfbc2eda2ea7593ad627fcc5b4cf89d7c071f2e640aae',
)) | _QUALIFIED_INNER_HEAD_SOURCES


_QUALIFIED_DECODE_SOURCES = frozenset((
  '2a641d2045c287a2df278dfa7a042d2b4b20be939039bd6e276a9907eee8259d',
  '2c3072267574f38e388236a0b68feddd8cbcf6e6c58c650c01be6c5b58047ee4',
  '63fbaa76ac01199c8b2150ad5245ae7c7fd34f3c0b26df8410d33c82ffcd9ae9',
  '6b5bf54efab619196b8000b4ee14b909d75788cd0f34144d3750246865862a07',
  '716fa844ed862fbbfe84f4af2ce64989c91129f2ccb7e680f834a1fdbc8a8648',
  'a212554b7c8831f8315da02ac1e42df6a15e10b01675541dffdeed7071968b4c',
  'c6b09ff2e1ec313eae00c04d7d0060861289d7d003e260cf7d49d1c492657415',
  'cacad4ecf8f1922e809a393a31257cce29122e2066f9c6330b99c62f2b8b9007',
  'd3a722ae121b372736251dae76cd5b0daa6e7b171ecd29f258a54e8a6efa01ea',
  'fd8ab2036bc7985417b813ee6582e7ba7c914cc9ee93d96cd8dffaf344924612',
))


def split_preparation_source(source:str, address:int) -> tuple[str,dict]:
  """Caller must prove outer-iteration independence; source approval does that here."""
  if type(address) is not int or not 0<address<2**64:raise ValueError('Invalid CPU preparation dispatcher')
  match=re.search(r'void (\w+)\(([^)]*)\) \{',source)
  if match is None:raise ValueError('Unsupported CPU preparation function')
  name,params=match[1],match[2]
  loops=list(re.finditer(r'for \(int (\w+) = 0; \1 < (\d+); \1\+\+\) \{',source[match.end():]))
  depth=1 if hashlib.sha256(source.encode()).hexdigest() in _QUALIFIED_INNER_HEAD_SOURCES else 0
  if len(loops)<=depth or int(loops[depth][2]) not in (2,16,32,512,1024):
    raise ValueError('Unsupported CPU preparation outer loop')
  loop=loops[depth]
  index,bound=loop[1],int(loop[2])
  begin,end=match.end()+loop.start(),match.end()+loop.end()
  signature=[]
  for param in params.split(','):
    if (p:=re.fullmatch(r'(.+?)(\w+)\s*',param.strip())) is None:raise ValueError('Invalid CPU preparation parameter')
    signature.append((p[1].replace('restrict','').replace('const int','int').strip(),p[2]))
  if not signature[0][1].startswith('data0_') or len(signature)>8:raise ValueError('Unsupported CPU preparation parameters')
  declarations=';'.join(typ+' '+var for typ,var in signature)+';'
  init=','.join(var for _,var in signature)
  arguments=','.join('w->'+var for _,var in signature)
  result=source[:begin]+f'for (int {index} = {bound}*group/4; {index} < {bound}*(group+1)/4; {index}++) {{'+source[end:]
  result=result.replace(match.group(0),f'static void {name}_part({params},int group) {{',1)
  result+=f'''
 typedef struct {{{declarations}}} PoolWork;
 static void prep_callback(void **args,int unused,int group) {{(void)unused;PoolWork *w=args[0];{name}_part({arguments},group);}}
 void {name}({params}) {{PoolWork work={{{init}}};void *args[4]={{&work,0,0,0}};
  if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(prep_callback,args,0))
   for(int group=0;group<4;group++)prep_callback(args,0,group);
 }}
 '''
  return result,{'kernel':name,'outer_axis':index,'outer_bound':bound,'workers':4,
                 'source_before_sha256':hashlib.sha256(source.encode()).hexdigest(),
                 'source_after_sha256':hashlib.sha256(result.encode()).hexdigest()}


@functools.cache
def _rewrite_program(program:UOp, address:int) -> tuple[UOp,dict]:
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  source=next(u.arg for u in program.src if u.op is Ops.SOURCE)
  updated,record=split_preparation_source(source,address)
  binary=_renderer().compiler.compile_cached(updated)
  record['binary_sha256']=hashlib.sha256(binary).hexdigest()
  return program.replace(src=tuple(UOp(Ops.SOURCE,arg=updated) if u.op is Ops.SOURCE else
                                   UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in program.src)),record


def _rewrite_lfm_pointwise_program(program:UOp,address:int) -> tuple[UOp,dict]:
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  source=next(u.arg for u in program.src if u.op is Ops.SOURCE)
  name=program.src[0].arg.function_name
  if hashlib.sha256(source.encode()).hexdigest() not in _LFM_FLAT_POINTWISE_SOURCES:
    raise ValueError('Unapproved LFM pointwise source')
  updated,args=_split_independent_rows(source,name,address)
  binary=_renderer().compiler.compile_cached(updated)
  record={'kernel':name,'phase':'prefill','workers':4,'overlap_fallback':True,
          'source_before_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'source_after_sha256':hashlib.sha256(updated.encode()).hexdigest(),
          'binary_sha256':hashlib.sha256(binary).hexdigest(),'flat_pointwise':True}
  return program.replace(src=tuple(UOp(Ops.SOURCE,arg=updated) if u.op is Ops.SOURCE else
                                   UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in program.src)),record


def rewrite_preparation_programs(linear:UOp, state:SharedHostPool, phase:str='prefill') -> UOp:
  if phase not in ('prefill','decode'):raise ValueError('Invalid CPU preparation phase')
  approved=_QUALIFIED_DECODE_SOURCES if phase=='decode' else _QUALIFIED_SOURCES
  replacements={}
  for p in linear.toposort(enter_calls=True):
    if (p.op is not Ops.PROGRAM or not isinstance(p.arg,ProgramInfo) or p.arg.target.device!='CPU'
        or p.arg.target.renderer not in ('','CLANG') or p.arg.outs!=(0,) or 0 in p.arg.ins):continue
    sources=[u.arg for u in p.src if u.op is Ops.SOURCE]
    if len(sources)!=1:continue
    digest=hashlib.sha256(sources[0].encode()).hexdigest()
    if digest in approved:
      replacements[p],record=_rewrite_program(p,state.address)
      record={**record,'phase':phase}
    elif phase=='prefill' and ONPU_LFM_FLAT_POINTWISE and digest in _LFM_FLAT_POINTWISE_SOURCES:
      replacements[p],record=_rewrite_lfm_pointwise_program(p,state.address)
    else:continue
    if not any(r['source_before_sha256']==record['source_before_sha256'] and r.get('phase')==phase
               for r in state.preparation_changes):
      state.preparation_changes.append(record)
  return linear.substitute(replacements,walk=True,enter_calls=True) if replacements else linear


_RWKV_POINTWISE_SOURCES = {
  '3103806b19109147a0b6751f583273a8c69153282be69e98c1b41234e5cfa3ca': 'E_128_512_4',
  '57ff27a93802510179c96c04c2f33681c5455b877cc172d59f0cccf3b24bcd97': 'E_128_512_4',
  '5d15c768cd382ba00f582c95eb20eef5003f800bc2123c2e82e45c6f8bfc0f6a': 'r_toks_512_4',
  '61b4f073e504377f133bf568e24d248ec224e5e7e5f70b2ca2ab5c9fdc6fc5b3': 'r_toks_512_4',
  '69e488981b392278346c22741b341458ae8e65543ae2ef9398f64d7f60c8fd5a': 'E_128_512_4',
  '76543a0723b93162b1c6c09a8d4654fc54b630d3be2bc236eff086251635b866': 'E_128_512_4',
  '906af0482de4b0b9fdd3c8208ea584f1806e9e4032987bca55c93edcc58311b5': 'E_128_512_4',
  '9d1f6a5a5e2ce2400b06ee4ccfdf6aed41d6915d46b7576c8002173bc3458b58': 'E_128_512_4',
  '9df1872e1f50025f7fb530a214e45d8032d854edd113f2e74bc2b1e6bade3086': 'r_toks_512_4',
  'bdcd9d2f9f93e4e60225de56bbb352e0d53fb3c317eb9b413aea25ac26962217': 'r_toks_512_4',
  'c924cf3127884dbb8d10040647a5e0d63c10e70dc5bdcfcf30bdee240b6dc370': 'E_128_512_4',
  'ca3f74c17b726892f17312c13aa6c2dd6cbb2300c24785b3df64cbec088d51e0': 'E_128_512_4',
  '7394b984cbe6e26252da6e9f63b4711609c5a454a97223dada7e261cc2b36875': 'E_toks_512_4',
  'e9ec80601691d062a4ff1f18e955858148b1fa5f7cbf5b62dfe1ed81813eb16f': 'E_2_toks_2048_3',
  'a09017aba18c7355579adc0571b3affadcff5815344aae09f0934dbbca00e607': 'E_toks_512_4',
  'a05490ef5a1c08796af310b1d4ad312e79783f0ee88eef7ac75c633a8f4fca93': 'E_32_32_16_4_4',
  'd1defdd357786b4f9f6d99e94a876bbb5bdeac0f0bbb18957e78f9cb283b8528': 'E_32_512_4_4',
  '3c15d968fcc46fb5d811f056aaf2d6786d0162b9ad8aae652fd22c9c03fe702a': 'E_32_512_4_4',
  '5dd31c8da8b620cf80a7e96fc6eeaeee6cb9a7e69fa584383a2b71f8c6e53c73': 'E_toks_512_4',
  '9d758769d8ce04de1910fa05263d94ce97c7163c027886618dcde4e4339f733e': 'E_toks_512_4',
  'd35267c2f5bd28e7ecca63f05810d7c8ac38a8c6566d84b6b71f0dfdc2ec5bd7': 'E_32_512_4_4',
  '4e59cd235264f72e95ee088da09333fedf136eebf9c116730c895e6382758725': 'E_toks_512_4',
}


def split_rwkv_source(source:str,name:str,address:int) -> tuple[str,list[tuple[str,str]]]:
  """Split only reviewed independent loops, retaining serial execution on overlap."""
  if (type(address) is not int or not 0<address<2**64
      or _RWKV_POINTWISE_SOURCES.get(hashlib.sha256(source.encode()).hexdigest())!=name):
    raise ValueError('Unqualified RWKV7 CPU preparation source or dispatcher')
  return _split_independent_rows(source,name,address)


def _split_independent_rows(source:str,name:str,address:int) -> tuple[str,list[tuple[str,str]]]:
  if type(address) is not int or not 0<address<2**64:raise ValueError('Invalid CPU row dispatcher')
  match=re.search(r'void '+re.escape(name)+r'\(([^)]*)\) \{',source)
  if match is None:raise ValueError('Missing RWKV7 preparation signature')
  params=match[1]
  args:list[tuple[str,str]]=[]
  for param in params.split(','):
    declaration=re.fullmatch(r'(.+?)(\w+)\s*',param.strip())
    if declaration is None:raise ValueError('Invalid RWKV7 preparation parameter')
    args.append((declaration[1].replace('restrict','').strip(),declaration[2]))
  if args[0][0]!='float*' or not args[0][1].startswith('data0_'):raise ValueError('Invalid RWKV7 preparation output')
  loops=list(re.finditer(r'for \(int (Lidx\d+) = 0; \1 < (toks_|128|32|32768); \1\+\+\) \{',source))
  if not loops:raise ValueError('Unsupported RWKV7 preparation loop')
  if any(item[2]=='32768' for item in loops) and not ONPU_LFM_FLAT_POINTWISE:
    raise ValueError('Unapproved 32768-row pointwise source')
  loop=next((item for item in loops if item[2]=='toks_'),loops[0])
  index,bound=loop[1],loop[2]
  prefix,body=source[:match.start()],source[match.start():]
  serial=body.replace('void '+name+'(','static void '+name+'_serial(',1)
  part=body.replace(match[0],f'static void {name}_part({params},int worker) {{',1)
  part=part.replace(loop[0],f'for (int {index} = ({bound})*worker/4; {index} < ({bound})*(worker+1)/4; {index}++) {{',1)
  def size(typ:str,var:str) -> int:
    count=int(var.rsplit('_',1)[1])
    return count*(2 if '__fp16' in typ else 1 if 'char' in typ else 4)
  outsize=size(*args[0])
  checks=[f'overlap({args[0][1]},{outsize}UL,{var},{size(typ,var)}UL)' for typ,var in args[1:] if '*' in typ]
  fields=''.join(typ+' '+var+';' for typ,var in args)
  values=','.join(var for _,var in args)
  invoke=','.join('w->'+var for _,var in args)
  text=prefix+serial+'\n'+part+f"""
  typedef struct {{{fields}}} PoolWork;
  static void callback(void**args,int unused,int worker) {{PoolWork*w=args[0];{name}_part({invoke},worker);}}
  static int overlap(const void*a,unsigned long an,const void*b,unsigned long bn) {{
    unsigned long x=(unsigned long)a,y=(unsigned long)b;return x<=y?y-x<an:x-y<bn;
  }}
  void {name}({params}) {{
    if({'||'.join(checks)}) {{{name}_serial({values});return;}}
    PoolWork work={{{values}}};void*args[4]={{&work,0,0,0}};
    if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(callback,args,0)) {{{name}_serial({values});}}
  }}
  """
  return text,args


@functools.cache
def _rewrite_rwkv_program(program:UOp,address:int) -> tuple[UOp,dict]:
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  source=next(u.arg for u in program.src if u.op is Ops.SOURCE)
  vector_decay=hashlib.sha256(source.encode()).hexdigest()=='d1defdd357786b4f9f6d99e94a876bbb5bdeac0f0bbb18957e78f9cb283b8528'
  if vector_decay:
    from tinygrad.llm.kernels.rockchip_rwkv_activations import vector_decay_source
    updated=vector_decay_source(source,program.src[0].arg.function_name,address)
  else:updated,_=split_rwkv_source(source,program.src[0].arg.function_name,address)
  binary=_renderer().compiler.compile_cached(updated)
  record={'kernel':program.src[0].arg.function_name,'phase':'prefill','workers':4,
          'source_before_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'source_after_sha256':hashlib.sha256(updated.encode()).hexdigest(),'overlap_fallback':True,'vector_decay':vector_decay}
  return program.replace(src=tuple(UOp(Ops.SOURCE,arg=updated) if u.op is Ops.SOURCE else
                                   UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in program.src)),record


def rewrite_rwkv_programs(linear:UOp,state:SharedHostPool) -> UOp:
  replacements={}
  for program in linear.toposort(enter_calls=True):
    if (program.op is not Ops.PROGRAM or not isinstance(program.arg,ProgramInfo) or program.arg.target.device!='CPU'
        or program.arg.outs!=(0,) or 0 in program.arg.ins):continue
    sources=[u.arg for u in program.src if u.op is Ops.SOURCE]
    if len(sources)!=1 or hashlib.sha256(sources[0].encode()).hexdigest() not in _RWKV_POINTWISE_SOURCES:continue
    replacements[program],record=_rewrite_rwkv_program(program,state.address)
    if not any(r['source_before_sha256']==record['source_before_sha256'] for r in state.preparation_changes):
      state.preparation_changes.append(record)
  return linear.substitute(replacements,walk=True,enter_calls=True) if replacements else linear
