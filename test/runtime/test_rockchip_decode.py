import os, unittest
from tinygrad import Tensor, UOp
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_decode import prepare_scalar_decode, ScalarDecoder

@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestScalarDecode(unittest.TestCase):
  def test_loop_matches_tensor_decoder_with_cache_rebinding(self):
    Tensor.manual_seed(46)
    model=Transformer(TransformerConfig(1,64,512,4,2,1e-5,96,16,10000.,16,16,max_context=64))
    jit=prepare_scalar_decode(model)
    configure(model,prefill='on',decode='on',precision='int8',fuse_projections=True,cpu_attention=True,native_replay=True,fused_ffn=True)
    decoder=ScalarDecoder(model,jit)
    prefix=Tensor([[1,2,3,4]],device='CPU').contiguous()
    token=int(model.greedy_prefill_jit(prefix,0).item())
    expected=[]
    for pos in range(4,16):
      token=int(model.greedy_rollout_jit(Tensor([[token]],device='CPU').contiguous(),UOp.variable('position',1,63).bind(pos)).item())
      expected.append(token)
    for _ in range(2):
      token=int(model.greedy_prefill_jit(prefix,0).item())
      self.assertEqual(decoder(token,4,12),expected)
    self.assertIsNotNone(decoder.loop)
    self.assertGreater(decoder.loop.tokens,0)
    with self.assertRaises(ValueError):decoder(96,4,1)
    with self.assertRaises(ValueError):decoder(0,63,2)

if __name__=='__main__':unittest.main()
