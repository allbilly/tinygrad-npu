import numpy as np
from tinygrad import Tensor,nn
from tinygrad.helpers import Context
from tinygrad.llm.kernels.rockchip import RockchipRMSNorm


def test_opt_in_matches_prefill_bits_for_single_token():
  rng=np.random.default_rng(631)
  source=nn.RMSNorm(4096,1e-5)
  source.weight=Tensor(rng.uniform(.5,1.5,4096).astype(np.float16),device='CPU').realize()
  kernel=RockchipRMSNorm(source)
  x=Tensor(rng.normal(0,3,(1,1,4096)).astype(np.float32),device='CPU').realize()
  with Context(ONPU_CPU_RMS_ALL_ROWS=1):
    single=kernel(x).numpy()
    many=kernel(x.expand(1,128,4096).contiguous().realize()).numpy()[:,:1]
  np.testing.assert_array_equal(single.view(np.uint32),many.view(np.uint32))
  data=x.numpy().astype(np.float64)
  reference=data/np.sqrt((data*data).mean(-1,keepdims=True)+1e-5)*source.weight.numpy().astype(np.float64)
  np.testing.assert_allclose(single,reference,rtol=2e-6,atol=2e-6)


def test_default_and_other_widths_retain_original_dispatch():
  class Source(nn.RMSNorm):
    def __call__(self,x):return x*42
  for width in (512,4096):
    kernel=RockchipRMSNorm(Source(width,1e-5))
    x=Tensor.ones(1,1,width,device='CPU').realize()
    with Context(ONPU_CPU_RMS_ALL_ROWS=0):np.testing.assert_array_equal(kernel(x).numpy(),np.full((1,1,width),42))
    if width==512:
      with Context(ONPU_CPU_RMS_ALL_ROWS=1):np.testing.assert_array_equal(kernel(x).numpy(),np.full((1,1,width),42))
