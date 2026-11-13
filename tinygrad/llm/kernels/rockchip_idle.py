"""A parked, low-priority CPU helper for the RK3588 NPU interrupt CPU."""
import atexit, ctypes, functools, hashlib, os, subprocess, tempfile, threading
from pathlib import Path
from tinygrad.helpers import ContextVar

ONPU_ASYNC_IRQ_WAKE = ContextVar('ONPU_ASYNC_IRQ_WAKE',0)


def _library(source:Path):
  directory = Path(tempfile.gettempdir())/f'tinygrad-rockchip-{os.getuid()}'
  directory.mkdir(exist_ok=True)
  binary = directory/(hashlib.sha256(source.read_bytes()).hexdigest()+'.so')
  if not binary.exists():
    with tempfile.TemporaryDirectory(dir=directory) as build:
      output = Path(build)/'helper.so'
      subprocess.run(['cc','-O3','-std=c11','-shared','-fPIC','-pthread',str(source),'-o',str(output)],check=True)
      os.replace(output,binary)
  return ctypes.CDLL(str(binary),use_errno=True)


class IdleHelper:
  def __init__(self):
    self.lib = _library(Path(__file__).with_suffix('.c'))
    self.lib.activate_spinner.argtypes = [ctypes.c_int]
    self.lib.activate_spinner_async.argtypes = [ctypes.c_int]
    self.lock = threading.Lock()
    self.runs = 0
    self.started = False
    self._asynchronous = False
    if error:=self.lib.start_spinner(): raise OSError(error,'NPU idle helper could not start')
    self.started = True
    atexit.register(self.close)
    # Qualify affinity before selecting the helper. A cgroup may exclude CPU0.
    with self: pass

  def __enter__(self):
    self.lock.acquire()
    self._asynchronous = bool(ONPU_ASYNC_IRQ_WAKE)
    activate = self.lib.activate_spinner_async if self._asynchronous else self.lib.activate_spinner
    if error:=activate(0):
      self.lock.release()
      raise OSError(error,'NPU idle helper cannot use CPU0')
    self.runs += 1
    return self

  def __exit__(self, *args):
    # The helper touches only its own atomic state. A stop request can return
    # while its low-priority thread is descheduled; close still joins it.
    pause = self.lib.pause_spinner_async if self._asynchronous else self.lib.pause_spinner
    error = pause()
    self.lock.release()
    if error: raise OSError(error,'NPU idle helper could not park')

  def close(self):
    if self.started:
      self.lib.stop_spinner()
      self.started = False


@functools.cache
def idle_helper() -> IdleHelper: return IdleHelper()
