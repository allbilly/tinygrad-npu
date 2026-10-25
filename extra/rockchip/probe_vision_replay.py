"""Isolate vision graph replay from decoding and mixed precision."""
import argparse, json, os, time
from pathlib import Path
import numpy as np

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--layers',type=int,default=12)
  parser.add_argument('--no-planner',action='store_true')
  args = parser.parse_args()
  os.environ['DEV'] = 'CPU'
  os.environ['ROCKCHIP_BINDING_CACHE'] = '4096'
  if args.no_planner: os.environ['NO_MEMORY_PLANNER'] = '1'
  from tinygrad import Tensor
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_vision import vision_attention
  model = SmolVLM.from_hf(args.root/'hf',512)
  model.vision.layers = model.vision.layers[:args.layers]
  for block in model.vision.layers: block.self_attn.attention = vision_attention
  model.vision.prefill_jit = model.encode_jit
  placement = configure(model.vision,prefill='on',decode='off',native_replay=True,cache_cleaning=True,wake_helper=True)
  print(json.dumps({'event':'ready','placement':placement}),flush=True)
  x = Tensor(np.load(args.root/'demo/pixels.npy')).realize()
  expected = None
  for index in range(6):
    start = time.perf_counter()
    output = model.vision.prefill_jit(x).numpy()
    elapsed = time.perf_counter()-start
    if expected is None: expected = output.copy()
    np.testing.assert_allclose(output,expected,atol=1e-5,rtol=1e-5)
    print(json.dumps({'event':'iteration','index':index,'ms':elapsed*1000}),flush=True)

if __name__ == '__main__': main()
