"""Compare output-unpacking thread counts without changing matrix arithmetic."""
import argparse, ctypes, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import UOp, dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.kernels.rockchip_cpu import _unpack_tiles


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(47)
  result = []
  for rows,columns,split in [(32,960,False),(128,4096,False),(256,4096,False),(256,1024,False),(256,1024,True)]:
    tile = min(rows,64)
    data = rng.normal(0,.8,(3 if split else 1,rows,columns)).astype(np.float32)
    pieces = [data[:,s:s+tile].reshape(data.shape[0],tile,columns//4,4).transpose(0,2,1,3).copy() for s in range(0,rows,tile)]
    if not split: pieces = [p[0] for p in pieces]
    output,golden = np.empty((rows,columns),dtype=np.float32),None
    out = UOp.param(0,dtypes.float,(rows,columns),device='CPU')
    inputs = tuple(UOp.param(i+1,dtypes.float,p.shape,device='CPU') for i,p in enumerate(pieces))
    for parallel in (False,True):
      fn = get_runtime('CPU',_unpack_tiles(out,inputs,tile,tile,columns,split,parallel)).fxn
      times = []
      for i in range(39):
        begin = time.perf_counter()
        fn(ctypes.c_void_p(output.ctypes.data),*(ctypes.c_void_p(p.ctypes.data) for p in pieces))
        if i>=8: times.append((time.perf_counter()-begin)*1000)
      if golden is None: golden = output.copy()
      np.testing.assert_array_equal(output.view(np.uint32),golden.view(np.uint32))
      result.append({'rows':rows,'columns':columns,'split':split,'parallel':parallel,
                     'median_ms':statistics.median(times),'bitwise_equal':True})
      args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
