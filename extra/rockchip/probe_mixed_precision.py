"""Check FP16/INT8 transitions independently of the VLM graph and its weights."""
import argparse, json, time
import numpy as np
from tinygrad import Tensor, TinyJit, nn
from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cycles',type=int,default=10)
  parser.add_argument('--int-columns',type=int,default=512)
  parser.add_argument('--int-rows',type=int,default=64)
  parser.add_argument('--native-int8',action='store_true')
  args = parser.parse_args()
  fpweight = pack_weight(Tensor.ones(768,768,device='CPU')).realize()
  fpinput = Tensor.ones(64,768,device='CPU').realize()
  source = nn.Linear(1024,args.int_columns,bias=False)
  source.weight = Tensor.ones(args.int_columns,1024,device='CPU').half().realize()
  intlayer = RockchipInt8Linear(source,'on','on')
  intlayer.prepare()
  intinput = Tensor.ones(args.int_rows,1024,device='CPU').realize()
  fp = TinyJit(lambda x:packed_matmul(x,fpweight,768).realize())
  iq = TinyJit(lambda x:intlayer.npu(x).realize())
  for phase,fn,x,expected in [('fp16-only',fp,fpinput,768.),('int8-only',iq,intinput,1024.)]:
    for index in range(4):
      start = time.perf_counter()
      actual = fn(x).numpy()
      if phase=='int8-only' and args.native_int8 and index==1:
        from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
        iq.captured = RockchipCapturedJit(iq.captured)
      np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-5)
      print(json.dumps({'phase':phase,'index':index,'ms':(time.perf_counter()-start)*1000}),flush=True)
  if args.native_int8: print(json.dumps({'native_int8_runs':iq.captured.native_runs,'reason':iq.captured.reason}),flush=True)
  for index in range(args.cycles):
    for phase,fn,x,expected in [('fp16',fp,fpinput,768.),('int8',iq,intinput,1024.)]:
      start = time.perf_counter()
      actual = fn(x).numpy()
      np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-5)
      print(json.dumps({'phase':phase,'index':index,'ms':(time.perf_counter()-start)*1000}),flush=True)

if __name__ == '__main__': main()
