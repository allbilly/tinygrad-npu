"""Exclusive NPU qualification for pooled native replay of full INT8 FFNs.

Run only with explicit exclusive NPU ownership.
"""
import os
import platform
import unittest

import numpy as np
from tinygrad import Tensor, TinyJit, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.rockchip import RockchipProjections
from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
from tinygrad.llm.kernels.rockchip_gelu_ffn import RockchipInt8GeluFeedForward, RockchipInt8SwiGLUFeedForward
from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8SplitLinear
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST') == '1', 'explicit exclusive NPU test')
class TestPooledFullFFNNativeReplay(unittest.TestCase):
  def test_pooled_full_ffns_match_materialized_reference_and_native_replay(self):
    if platform.machine().lower() not in ('aarch64', 'arm64'):
      self.skipTest('Rockchip full FFN requires AArch64')
    if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
      self.skipTest('CPU FFN prefill pool requires CPU4-7')

    from tinygrad.llm.kernels.rockchip_pool import AttentionPool
    from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit

    pool = AttentionPool()
    rng = np.random.default_rng(20261028)
    try:
      for activation, cls in (('swiglu', RockchipInt8SwiGLUFeedForward),
                              ('gelu', RockchipInt8GeluFeedForward)):
        sources = []
        hidden = 4864
        for inputs, outputs in ((512, hidden), (512, hidden), (hidden, 512)):
          layer = nn.Linear(inputs, outputs, bias=False)
          layer.weight = Tensor(rng.normal(0, .02, (outputs, inputs)).astype(np.float16), device='CPU').realize()
          sources.append(layer)

        projections = RockchipProjections(
          tuple(RockchipInt8Linear(layer, 'on', 'on', 24576) for layer in sources[:2]),
          24576, share_packed=True)
        down = RockchipInt8SplitLinear(sources[2], 'on', 'on', 24576)
        down.prepare()
        baseline_ffn = cls(projections, down)
        pooled_ffn = cls(projections, down, pool=pool)

        def baseline(x):
          gate, up = projections(x)
          joint = gate.cat(up, dim=-1).contiguous().realize()
          if activation == 'swiglu':
            joint = joint.reshape(1, x.shape[1], hidden*2).realize()
            sigmoid = sigmoid_gate(joint).realize()
            activated = (joint[:, :, :hidden] * sigmoid) * joint[:, :, hidden:]
          else:
            activated = gated_gelu(gate, up)
          return down(activated).realize()

        for rows in (17, 128):
          control = TinyJit(baseline)
          candidate = TinyJit(lambda x: pooled_ffn(x).realize())
          for step in range(7):
            x = Tensor(rng.normal(0, .2, (1, rows, 512)).astype(np.float32), device='CPU').realize()
            self.assertTrue(baseline_ffn.enabled(x))
            self.assertTrue(pooled_ffn.enabled(x))
            control_out = control(x).realize()
            control_array = control_out.numpy().copy()
            affinity = os.sched_getaffinity(0)
            with pool.prefill():
              self.assertTrue(pool.active)
              self.assertEqual(os.sched_getaffinity(0), affinity)
              before = pool.lib.pool_calls()
              candidate_out = candidate(x).realize()
              candidate_array = candidate_out.numpy()
              self.assertEqual(pool.lib.pool_calls() - before, 1)
              self.assertEqual(os.sched_getaffinity(0), affinity)
            self.assertTrue(pool.lib.pool_parked())
            self.assertEqual(os.sched_getaffinity(0), affinity)
            np.testing.assert_array_equal(control_array.view(np.uint32), candidate_array.view(np.uint32))

            if step == 2:
              control.captured = RockchipCapturedJit(control.captured)
              candidate.captured = RockchipCapturedJit(candidate.captured)

          self.assertIsNone(candidate.captured.reason)
          self.assertGreaterEqual(candidate.captured.native_runs, 2)
          self.assertIsNone(control.captured.reason)
          self.assertGreaterEqual(control.captured.native_runs, 2)
          # Check the captured graph selected the pooled pack kernel and baked in this
          # pool's parked-worker callback, rather than silently taking the local path.
          self.assertIsNone(control.captured.reason)
          self.assertIsNone(candidate.captured.reason)
          name='rockchip_int8_swiglu_pack' if activation=='swiglu' else 'rockchip_int8_gelu_pack'
          calls=[c for c in candidate.captured.native.calls
                 if c.body.op is Ops.PROGRAM and c.body.src[0].arg.function_name==name]
          self.assertEqual(len(calls),1)
          pack_source=next(src.arg for src in calls[0].body.src if src.op is Ops.SOURCE)
          self.assertIn('pool_part',pack_source)
          self.assertIn(str(pool.prefill_address),pack_source)
          self.assertNotIn('pthread_create',pack_source)
          batch=int(os.environ.get('ONPU_SWIGLU_PREFILL_ROWS','1'))
          if activation=='swiglu' and batch>1:
            self.assertIn(f'float batch_values[{batch}]',pack_source)
          else:
            self.assertNotIn('float batch_values',pack_source)

    finally:
      pool.close()
