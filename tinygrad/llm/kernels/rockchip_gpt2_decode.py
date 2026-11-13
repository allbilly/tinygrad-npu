"""Bounded native greedy generation over an unchanged prepared GPT-2 step."""
from __future__ import annotations
import ctypes
from typing import Any, cast
from tinygrad import Device, Variable, dtypes
from tinygrad.device import TinyELF
from tinygrad.helpers import GlobalCounters, unwrap
from tinygrad.llm.kernels.rockchip_cpu import _renderer
from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
from tinygrad.uop.ops import sym_infer


class DecodeLoop:
  def __init__(self, phase, context:int, vocab:int, stop_tokens:tuple[int,...]=()):
    if len(stop_tokens)>16 or any(type(token) is not int or not 0<=token<vocab for token in stop_tokens):
      raise ValueError('Invalid native decode stop tokens')
    self.phase,self.context,self.vocab = phase,context,vocab
    self.stop_tokens=tuple(sorted(set(stop_tokens)))
    self.captured = phase.captured
    self.native = self.captured.native
    assert self.native is not None
    self.output = self.captured.ret._buffer().ensure_allocated()
    address = self.output.get_buf('CPU')
    replay = unwrap(ctypes.cast(self.native.program.fxn,ctypes.c_void_p).value)
    self.stat_count = 2+len(self.native.calls) if self.native.profiling else 2
    source = f'''typedef unsigned long long U64;
    int gpt2_decode_loop(int *out,int token,int position,int count,U64 *stats,U64 *done) {{
      for(int j=0;j<{self.stat_count};j++)stats[j]=0;
      *done=0;
      for(int i=0;i<count;i++) {{
        U64 current[{self.stat_count}]={{0}};
        int error=((int(*)(long long,long long,U64*)){replay}ULL)(position+i,token,current);
        for(int j=0;j<{self.stat_count};j++)stats[j]+=current[j];
        if(error)return error;
        token=*((volatile int*){address}ULL);
        if(token<0 || token>={vocab})return 3000000;
        out[i]=token;(*done)++;
        {'if('+ ' || '.join(f'token=={token}' for token in self.stop_tokens)+')break;' if self.stop_tokens else ''}
      }}
      return 0;
    }}'''
    renderer = _renderer()
    self.program = cast(Any,Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(source),'gpt2_decode_loop',renderer.target,())))
    self.program.fxn.restype = ctypes.c_int
    self.tokens = 0

  def valid(self) -> bool:
    return self.phase.captured is self.captured and self.captured.native is self.native and self.native.valid([])

  def __call__(self, token:int, position:int, count:int) -> list[int]:
    if not 0< count<=32 or not 0<=token<self.vocab or not 1<=position<=self.context-count:
      raise ValueError('GPT-2 native decode exceeds token or context bounds')
    n = self.native
    out,stats,done = (ctypes.c_int*count)(),(ctypes.c_uint64*self.stat_count)(),ctypes.c_uint64()
    with self.phase.scope():
      with n.dev.lock:
        n.dev.check_healthy()
        if not self.valid(): raise RuntimeError('GPT-2 native decode bindings changed')
        result = self.program.fxn(out,ctypes.c_int(token),ctypes.c_int(position),ctypes.c_int(count),stats,ctypes.byref(done))
        n.dev.submit_count += stats[0]
        n.dev.task_count += stats[1]
        if result:
          n.dev.poisoned = True
          raise RuntimeError(f'GPT-2 native decode failed at token {done.value}, code {result}')
        executed=int(done.value)
        if not 1<=executed<=count:
          n.dev.poisoned=True
          raise RuntimeError('Native decode returned an invalid completion count')
        for key,meta in n.writes.items():
          meta.version += n.write_counts[key]*executed
          meta.cpu_clean_version = meta.version
    generated = list(out)[:executed]
    for i in range(executed):
      values = {'start_pos':position+i,'tokens':token if i==0 else generated[i-1]}
      GlobalCounters.global_ops += sym_infer(n.estimates.ops,values)
      GlobalCounters.global_mem += sym_infer(n.estimates.mem,values)
    GlobalCounters.kernel_count += len(n.calls)*executed
    if n.profiling:
      for i,label in enumerate(n.labels):
        n.profile_ns[label] += stats[i+2]
        n.profile_counts[label] += executed
    values = {'start_pos':position+executed-1,'tokens':token if executed==1 else generated[-2]}
    for tensor,uop,bindings in self.captured._symbolic_ret:
      tensor.uop = uop.substitute({v:v.bind(values.get(v.expr,i)) for v,i in bindings.items()},walk=True)
    self.captured.native_runs += executed
    self.phase.jit.cnt += executed
    self.tokens += executed
    return generated


def decode_many(model, token:int, position:int, count:int) -> list[int]:
  """Warm through the normal executor, then replay at most32 tokens per call."""
  if not 0<=count<=model.max_context-position or not 0<=token<model.vocab_size or position<1:
    raise ValueError('GPT-2 decode exceeds token or context bounds')
  output = []
  phase = model.rockchip_gpt2['decode']
  while count:
    captured = phase.captured
    native = captured.native if isinstance(captured,RockchipCapturedJit) else None
    if (native is not None and not native.roots and native.variables==['start_pos','tokens'] and native.valid([])
        and captured.ret.device=='CPU' and captured.ret.dtype is dtypes.int and captured.ret.numel()==1):
      loop = model.rockchip_gpt2.get('loop')
      if loop is None or not loop.valid():
        loop = model.rockchip_gpt2['loop'] = DecodeLoop(phase,model.max_context,model.vocab_size)
      generated = loop(token,position,min(count,32))
    else:
      tok = Variable('tokens',0,model.vocab_size-1).bind(token)
      pos = Variable('start_pos',1,model.max_context-1).bind(position)
      generated = model(tok,pos,0.0).tolist()
    output.extend(generated)
    position,count,token = position+len(generated),count-len(generated),generated[-1]
  return output
