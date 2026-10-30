"""Exclusive hardware comparison for one-row fused raw SwiGLU INT8 decode."""
import os
import unittest

import numpy as np
from tinygrad import Tensor, TinyJit, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import gated_activation
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8SwiGLUFeedForward
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST') == '1', 'explicit exclusive NPU test')
class TestRawSwiGLUDecodeHardware(unittest.TestCase):
  def test_one_row_raw_decode_matches_materialized_int8_reference_and_native_replay(self):
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit

    rng = np.random.default_rng(20261028)
    for input_width, output_width, hidden in ((896,896,4864),(1024,1024,4608)):
      sources=[]
      for inputs,outputs in ((input_width,hidden),(input_width,hidden),(hidden,output_width)):
        layer=nn.Linear(inputs,outputs,bias=False)
        layer.weight=Tensor(rng.normal(0,.03,(outputs,inputs)).astype(np.float16),device='CPU').realize()
        sources.append(layer)

      members=tuple(RockchipInt8Linear(layer,'on','on',24576) for layer in sources[:2])
      projections=RockchipProjections(members,24576,share_packed=True)
      down=RockchipInt8SplitLinear(sources[2],'on','on',24576)
      down.prepare()
      for layer in (*members,down,*down.parts): layer.phase='decode'
      candidate_ffn=RockchipInt8SwiGLUFeedForward(projections,down,decode=True)
      self.assertIsNone(candidate_ffn.pool)

      def baseline(x):
        gate,up=projections(x)
        activated=gated_activation(gate,up,alpha=1.0,limit=None,up_bias=0.0,activation='silu').realize()
        return down(activated).realize()

      control=TinyJit(baseline)
      candidate=TinyJit(lambda x:candidate_ffn(x).realize())
      for step in range(7):
        if step==1: values=np.zeros((1,1,input_width),dtype=np.float32)
        elif step==2: values=np.where(np.arange(input_width)%2,.125,-.125).astype(np.float32).reshape(1,1,input_width)
        else: values=rng.normal(0,.2,(1,1,input_width)).astype(np.float32)
        x=Tensor(values,device='CPU').realize()
        self.assertTrue(candidate_ffn.enabled(x))
        expected=control(x).realize().numpy().copy()
        actual=candidate(x).realize().numpy().copy()
        np.testing.assert_array_equal(expected.view(np.uint32),actual.view(np.uint32))
        if step==2:
          control.captured=RockchipCapturedJit(control.captured)
          candidate.captured=RockchipCapturedJit(candidate.captured)

      self.assertIsNone(candidate.captured.reason)
      self.assertGreaterEqual(candidate.captured.native_runs,2)
      self.assertIsNone(control.captured.reason)
      self.assertGreaterEqual(control.captured.native_runs,2)

      def calls_with_name(jit,name):
        return [call for call in jit.captured.native.calls
                if call.body.op is Ops.PROGRAM and call.body.src[0].arg.function_name==name]
      pack_name='rockchip_int8_swiglu_pack'
      self.assertEqual(len(calls_with_name(control,pack_name)),0)
      pack_calls=calls_with_name(candidate,pack_name)
      self.assertEqual(len(pack_calls),1)
      pack_source=next(src.arg for src in pack_calls[0].body.src if src.op is Ops.SOURCE)
      self.assertNotIn('pthread_create',pack_source)
