"""Measure serial/parallel RMS normalization using identical compiled row math."""
import argparse, ctypes, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import UOp, dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.kernels.rockchip_cpu import _rms_norm


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  rng = np.random.default_rng(46)
  result = []
  for width in (576,1024,4096):
    for rows in (32,128,256):
      data = rng.normal(0,.8,(rows,width)).astype(np.float32)
      weight = rng.uniform(.5,1.5,width).astype(np.float16)
      output,golden = np.empty_like(data),None
      params = (UOp.param(0,dtypes.float,(rows,width),device='CPU'),
                UOp.param(1,dtypes.float,(rows,width),device='CPU'),UOp.param(2,dtypes.half,(width,),device='CPU'))
      for parallel in (False,True):
        fn = get_runtime('CPU',_rms_norm(*params,rows,width,1e-6,parallel)).fxn
        times = []
        for i in range(39):
          begin = time.perf_counter()
          fn(ctypes.c_void_p(output.ctypes.data),ctypes.c_void_p(data.ctypes.data),ctypes.c_void_p(weight.ctypes.data))
          if i>=8: times.append((time.perf_counter()-begin)*1000)
        if golden is None: golden = output.copy()
        np.testing.assert_array_equal(output.view(np.uint32),golden.view(np.uint32))
        result.append({'rows':rows,'width':width,'parallel':parallel,'median_ms':statistics.median(times),'bitwise_equal':True})
        args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
