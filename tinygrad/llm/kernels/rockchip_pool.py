"""Persistent per-head CPU workers, parked outside an inference call."""
import atexit, contextlib, ctypes, functools, os, threading
from pathlib import Path
from tinygrad.llm.kernels.rockchip_idle import _library


class AttentionPool:
  def __init__(self):
    self.lib = _library(Path(__file__).with_suffix('.c'))
    self.lib.start_pool.argtypes = [ctypes.c_int]
    self.lib.pool_calls.restype = ctypes.c_uint64
    self.address = ctypes.cast(self.lib.parallel,ctypes.c_void_p).value
    self.prefill_address = ctypes.cast(self.lib.parallel_parked,ctypes.c_void_p).value
    self.lock = threading.Lock()
    self.started = False
    self.owner_thread:int|None = None
    if error:=self.lib.start_pool(5): raise OSError(error,'CPU attention workers could not start on CPU5-7')
    self.started = True
    atexit.register(self.close)

  def __enter__(self):
    self.lock.acquire()
    self.previous = os.sched_getaffinity(0)
    try:
      if 4 in self.previous: os.sched_setaffinity(0,{4})
      self.owner_thread = threading.get_ident()
    except OSError:
      self.lock.release()
      raise
    return self

  def __exit__(self, *args):
    error = self.lib.pause_pool()
    self.owner_thread = None
    try: os.sched_setaffinity(0,self.previous)
    finally: self.lock.release()
    if error: raise OSError(error,'CPU attention workers could not park')

  @property
  def active(self) -> bool: return self.owner_thread==threading.get_ident()

  @contextlib.contextmanager
  def prefill(self):
    with self.lock:
      previous = os.sched_getaffinity(0)
      self.owner_thread = threading.get_ident()
      try: yield self
      finally:
        error = self.lib.pause_pool()
        self.owner_thread = None
        os.sched_setaffinity(0,previous)
        if error:raise OSError(error,'CPU prefill workers could not park')

  def close(self):
    if self.started:
      self.lib.stop_pool()
      self.started = False


@functools.cache
def attention_pool() -> AttentionPool: return AttentionPool()
