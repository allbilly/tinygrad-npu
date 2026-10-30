"""Opt-in shared workers for fully prepared dense INT8 prefill.

The five kernel factories retain their original arithmetic and work partitions.
Only kernels constructed in a shared prefill scope change dispatch. The bridge
parks workers outside that scope, including cached calls made by other paths.
"""
from __future__ import annotations
import contextlib, contextvars, ctypes, functools, hashlib, re, weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ParamSpec
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops, UOp
if TYPE_CHECKING:
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool

ONPU_SHARED_HOST_POOL = ContextVar('ONPU_SHARED_HOST_POOL', 0)
ONPU_PREFILL_CPU_TUNING = ContextVar('ONPU_PREFILL_CPU_TUNING', 0)
_P = ParamSpec('_P')
_current:contextvars.ContextVar[SharedHostPool|None] = contextvars.ContextVar('rockchip_shared_host_pool', default=None)


def pooled_source(source:str, address:int) -> tuple[str,dict[str,Any]]:
  if 'pthread_create(' not in source or 'Work work[4];' not in source:return source, {'changed':False}
  begin = source.index('Work work[4];')
  initialize = re.search(r'for\(int i=0;i<4;i\+\+\)work\[i\]=\(Work\)\{[^;]+;', source[begin:])
  worker = re.search(r'pthread_create\(&threads\[i\],\(void\*\)0,(\w+),&work\[i\]\)', source[begin:])
  tail = 'for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);'
  if initialize is None or worker is None or source.count(tail)!=1:
    raise ValueError('Unexpected four-worker source; refusing to rewrite')
  end, name = source.index(tail,begin)+len(tail), worker.group(1)
  dispatch = f'''Work work[4];
    {initialize.group(0)}
    void *shared_args[4]={{work,0,0,0}};
    if(((int(*)(void(*)(void**,int,int),void**,int)){address}ULL)(shared_work_callback,shared_args,0)){{
      for(int i=0;i<4;i++){name}(work+i);
    }}'''
  result = ('static void shared_work_callback(void**,int,int);\n'+source[:begin]+dispatch+source[end:]+f'''
    static void shared_work_callback(void **args,int unused,int group) {{
      (void)unused;{name}(((Work*)args[0])+group);
    }}
  ''')
  return result, {'changed':True, 'worker':name, 'work_initializer':initialize.group(0),
    'source_before_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'source_after_sha256':hashlib.sha256(result.encode()).hexdigest(),
    'thread_creates_replaced_per_call':3, 'callback_address':address}


class SharedHostPool:
  def __init__(self, pool:AttentionPool):
    from tinygrad import Device
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_cpu import CPUProgram
    from tinygrad.llm.kernels.rockchip_cpu import _renderer
    self.pool, self.active_flag, self.scope_calls = pool, ctypes.c_int(0), 0
    self.changes:list[dict[str,Any]] = []
    self.parked_address = pool.prefill_address
    self.dispatch_source = f'''
      typedef void (*work_fn)(void**,int,int);
      int shared_host_dispatch(work_fn fn,void **args,int position) {{
        volatile int *active=(volatile int*){ctypes.addressof(self.active_flag)}ULL;
        unsigned long address=*active?{pool.address}ULL:{pool.prefill_address}ULL;
        return ((int(*)(work_fn,void**,int))address)(fn,args,position);
      }}
    '''
    renderer = _renderer()
    program = Device['CPU'].runtime(TinyELF(renderer.compiler.compile_cached(self.dispatch_source),
                                         'shared_host_dispatch',renderer.target,()))
    if not isinstance(program,CPUProgram) or program.dev.remote is not None:
      raise ValueError('Shared host workers require a local CPU runtime')
    self.dispatch_program, self.address = program, program.addr
    pool.prefill_address = self.address

  @functools.cache
  def rewrite(self, program:UOp) -> UOp:
    from tinygrad.llm.kernels.rockchip_cpu import _renderer
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    updated, record = pooled_source(source,self.address)
    if not record['changed']:return program
    binary = _renderer().compiler.compile_cached(updated)
    record.update(kernel=program.src[0].arg.name,binary_sha256=hashlib.sha256(binary).hexdigest())
    self.changes.append(record)
    return program.replace(src=tuple(UOp(Ops.SOURCE,arg=updated) if u.op is Ops.SOURCE else
                                    UOp(Ops.BINARY,arg=binary) if u.op is Ops.BINARY else u for u in program.src))

  @contextlib.contextmanager
  def scope(self) -> Iterator[None]:
    if not self.pool.started:raise RuntimeError('Shared host worker pool has stopped')
    if _current.get() is self:
      yield
      return
    token, previous = _current.set(self), self.active_flag.value
    try:
      with contextlib.nullcontext() if self.pool.active else self.pool:
        self.active_flag.value = 1
        self.scope_calls += 1
        try:yield
        finally:self.active_flag.value = previous
    finally:_current.reset(token)

  def restore(self):
    self.active_flag.value = 0
    if error:=self.pool.lib.pause_pool():raise OSError(error,'Shared host workers could not park')
    if self.pool.prefill_address==self.address:self.pool.prefill_address = self.parked_address


@functools.cache
def shared_host_pool(pool:AttentionPool) -> SharedHostPool:return SharedHostPool(pool)


def model_shared_host_pool(model:Any) -> SharedHostPool|None:
  handle = getattr(model,'_shared_host_pool',None)
  return handle() if isinstance(handle,weakref.ReferenceType) else None


def shared_program(function:Callable[_P,UOp]) -> Callable[_P,UOp]:
  @functools.wraps(function)
  def generate(*args:_P.args, **kwargs:_P.kwargs) -> UOp:
    program = function(*args,**kwargs)
    return state.rewrite(program) if (state:=_current.get()) is not None else program
  for name in ('cache_info','cache_clear','cache_parameters'):
    if hasattr(function,name):setattr(generate,name,getattr(function,name))
  return generate
