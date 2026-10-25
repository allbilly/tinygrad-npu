"""Compare 16/21-row FC2 tiles using the actual guarded FP16 vision MLP."""
import ctypes, inspect, json, statistics, textwrap, time
import numpy as np
from tinygrad import Device, Tensor, nn
from tinygrad.llm.kernels import rockchip_vision_mlp as mlp
from tinygrad.llm.kernels.rockchip import RockchipLinear
from tinygrad.llm.kernels.rockchip_cache import cache_cleaner
from tinygrad.llm.kernels.rockchip_vision_pipeline import _Stage


def main():
  Tensor.manual_seed(57)
  layers = [nn.Linear(768,3072),nn.Linear(3072,768)]
  for layer in layers:
    for parameter in nn.state.get_parameters(layer):parameter.replace(parameter.half().contiguous().realize())
  fc1,fc2 = [RockchipLinear(layer,'on','off') for layer in layers]
  fc1.prepare()
  fc2.prepare()
  source = textwrap.dedent(inspect.getsource(mlp.FP16VisionMLP.__call__))
  assert source.count('21')==4
  environment = vars(mlp).copy()
  exec(source.replace('21','16'),environment)
  alternative = type('SixteenRowMLP',(mlp.FP16VisionMLP,),{'__call__':environment['__call__']})
  cleaner = ctypes.cast(cache_cleaner().clean_to,ctypes.c_void_p).value
  value = Tensor.randn(1,1024,768,device='CPU').realize()
  golden = None
  for tile,kind in [(21,mlp.FP16VisionMLP),(16,alternative)]:
    operator = kind(fc1,fc2)
    run = _Stage(lambda x:operator(x).realize(),True,cleaner)
    samples = []
    for index in range(14):
      before = Device['ROCKCHIP'].submit_count
      start = time.perf_counter()
      result = run(value)
      elapsed = (time.perf_counter()-start)*1000
      submits = Device['ROCKCHIP'].submit_count-before
      if index>=6:samples.append(elapsed)
    actual = result.numpy().copy()
    if golden is None:golden=actual
    np.testing.assert_array_equal(actual.view(np.uint32),golden.view(np.uint32))
    print(json.dumps({'tile':tile,'bitwise_equal':True,'median_ms':statistics.median(samples),
                      'all_ms':samples,'submits':submits}),flush=True)


if __name__=='__main__':main()
