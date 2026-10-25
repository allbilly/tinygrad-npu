"""Probed ARM cache cleaning for bounded CPU-written Rockchip input mappings."""
import ctypes, functools, platform, subprocess, sys
from pathlib import Path
from tinygrad.llm.kernels.rockchip_idle import _library


@functools.cache
def cache_cleaner():
  if platform.machine().lower() not in ('aarch64','arm64'): raise OSError('ARM cache maintenance is unavailable')
  # Probe in a child because disabled EL0 cache maintenance can raise SIGILL.
  result = subprocess.run([sys.executable,'-m','tinygrad.llm.kernels.rockchip_cache'],capture_output=True,text=True,timeout=15,check=False)
  if result.returncode or result.stdout.strip()!='64': raise OSError('EL0 cache cleaning probe failed')
  library = _library(Path(__file__).with_suffix('.c'))
  library.clean_to.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
  library.clean_to.restype = None
  return library


if __name__=='__main__':
  library = _library(Path(__file__).with_suffix('.c'))
  library.clean_to.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
  library.clean_to.restype = None
  memory = (ctypes.c_ubyte*4096)()
  library.clean_to(memory,ctypes.sizeof(memory))
  print(library.line_size())
