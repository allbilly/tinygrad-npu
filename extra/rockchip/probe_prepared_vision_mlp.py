"""Compare the retained CPU/NPU and prepared GPU/NPU vision MLP implementations."""
import ctypes, json, statistics, time
import numpy as np
from tinygrad import Device, Tensor, nn
from tinygrad.llm.kernels.rockchip import RockchipLinear
from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
from tinygrad.llm.kernels.rockchip_vision_mlp import FP16VisionMLP
from tinygrad.llm.kernels.rockchip_vision_pipeline import _Stage
from tinygrad.llm.kernels.rockchip_vision_prepared import PreparedVisionMLP

def main():
  Tensor.manual_seed(57)
  layers = [nn.Linear(768,3072),nn.Linear(3072,768)]
  for layer in layers:
    for parameter in nn.state.get_parameters(layer):parameter.replace(parameter.half().contiguous().realize())
  fc1,fc2 = [RockchipLinear(layer,'on','off') for layer in layers]
  fc1.prepare()
  fc2.prepare()
  cleaner = ctypes.cast(cache_cleaner().clean_to,ctypes.c_void_p).value
  old = FP16VisionMLP(fc1,fc2)
  reference = _Stage(lambda x,r:(r+old(x)).realize(),True,cleaner)
  fused = PreparedVisionMLP(fc1,fc2)
  try:
    for scale in (1.,.4):
      x = (Tensor.randn(1,1024,768,device='CPU')*scale).realize()
      residual = Tensor.randn(1,1024,768,device='CPU').realize()
      expected = reference(x,residual).numpy().copy()
      for name,operator in [('existing',reference),('prepared_split',fused)]:
        samples = []
        for i in range(14):
          before = Device['ROCKCHIP'].submit_count
          start = time.perf_counter()
          result = operator(x,residual)
          if i>=6:samples.append((time.perf_counter()-start)*1000)
          submits = Device['ROCKCHIP'].submit_count-before
        actual = result.numpy().copy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-3)
        print(json.dumps({'method':name,'scale':scale,'median_ms':statistics.median(samples),'all_ms':samples,
                          'submits':submits,'max_error':float(abs(actual-expected).max())}),flush=True)
    for invalid in (float('nan'),float('inf'),70000.):
      x = Tensor.full((1,1024,768),invalid,device='CPU').contiguous().realize()
      before = fused.dev.submit_count
      try:fused(x,residual)
      except ValueError:pass
      else:raise AssertionError('Invalid input accepted')
      assert fused.dev.submit_count==before and not fused.dev.poisoned
    print(json.dumps({'finite_guards':True}),flush=True)
  finally:
    fused.close()


if __name__=='__main__':main()
