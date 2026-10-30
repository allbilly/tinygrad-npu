"""CPU pool contract tests for pooled GELU/SwiGLU projection packing."""
import os
import platform

import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, UOp
from tinygrad.llm.kernels.rockchip_gelu_ffn import pack_gelu_projection, pack_swiglu_projection, payload_layout
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_sigmoid import sigmoid_gate
from test.runtime.test_rockchip_gelu_ffn import fixture


@pytest.fixture(scope='module')
def attention_pool():
  if platform.machine().lower() not in ('aarch64', 'arm64'):
    pytest.skip('CPU FFN prefill pool requires AArch64')
  if not {4, 5, 6, 7}.issubset(os.sched_getaffinity(0)):
    pytest.skip('CPU FFN prefill pool requires CPU4-7')
  from tinygrad.llm.kernels.rockchip_pool import AttentionPool
  pool = AttentionPool()
  try:
    yield pool
  finally:
    pool.close()


def _depths(hidden):
  return tuple(min(4096, hidden-start) for start in range(0, hidden, 4096))


def _reference(projected, active, hidden, depths, activation):
  if activation == 'gelu':
    from tinygrad.llm.kernels.rockchip_gelu import gated_gelu
    activated = gated_gelu(projected[:active, :hidden], projected[:active, hidden:])
  else:
    joint = projected[:active].reshape(1, active, hidden*2).realize()
    sigmoid = sigmoid_gate(joint).realize()
    activated = ((joint[:, :, :hidden] * sigmoid) * joint[:, :, hidden:]).reshape(active, hidden)
  activated = activated.pad(((0, projected.shape[0]-active), (0, 0)))
  payloads, scales = [], []
  column = 0
  for depth, (_, padded, tile, _) in zip(depths, payload_layout(projected.shape[0], depths)):
    for first in range(0, projected.shape[0], tile):
      payload, scale = quantize(activated[first:first+tile, column:column+depth].contiguous(), padded)
      payloads.append(payload)
      scales.append(scale)
    column += depth
  return Tensor.cat(*payloads), Tensor.cat(*scales)


def _arrays(outputs):
  payload, scale = outputs
  payload.realize(scale)
  return payload.numpy(), scale.numpy().view(np.uint32)


@pytest.mark.parametrize('activation,rows,hidden,input_tile,active', [
  ('gelu', 17, 4864, 128, 1),
  ('gelu', 128, 5632, 64, 113),
  ('swiglu', 17, 4864, 128, 17),
  # Two row tiles, each split into raw column fragments: exercises >4 kernel args.
  ('swiglu', 128, 8960, 64, 128),
])
def test_pooled_packer_matches_unpooled_and_independent_reference(
    attention_pool, activation, rows, hidden, input_tile, active):
  pack = pack_gelu_projection if activation == 'gelu' else pack_swiglu_projection
  raw, sa, sw, tiles, projected = fixture(rows, hidden, input_tile, seed=90210+hidden)
  depths = _depths(hidden)
  expected = _arrays(_reference(projected, active, hidden, depths, activation))
  unpooled = _arrays(pack(raw, sa, sw, tiles, rows, hidden, depths, active))
  np.testing.assert_array_equal(unpooled[0], expected[0])
  np.testing.assert_array_equal(unpooled[1], expected[1])

  affinity = os.sched_getaffinity(0)
  with attention_pool.prefill():
    assert attention_pool.active
    assert os.sched_getaffinity(0) == affinity
    before = attention_pool.lib.pool_calls()
    pooled_outputs = pack(raw, sa, sw, tiles, rows, hidden, depths, active, pool=attention_pool)
    # Realize payload and scale before observing dispatch accounting: one custom kernel
    # produces both outputs and must invoke the pool exactly once.
    pooled = _arrays(pooled_outputs)
    assert attention_pool.lib.pool_calls() - before == 1
    np.testing.assert_array_equal(pooled[0], unpooled[0])
    np.testing.assert_array_equal(pooled[1], unpooled[1])
    np.testing.assert_array_equal(pooled[0], expected[0])
    np.testing.assert_array_equal(pooled[1], expected[1])
  assert attention_pool.lib.pool_parked()
  assert os.sched_getaffinity(0) == affinity


@pytest.mark.parametrize('activation', ['gelu', 'swiglu'])
def test_pool_dispatch_threshold_and_inactive_fallback(attention_pool, activation):
  pack = pack_gelu_projection if activation == 'gelu' else pack_swiglu_projection
  rows, hidden = 17, 512
  raw, sa, sw, tiles, _ = fixture(rows, hidden, 128, seed=741)
  depths = (hidden,)
  with attention_pool.prefill():
    for active in (1, 15, 16, 17):
      before = attention_pool.lib.pool_calls()
      outputs = pack(raw, sa, sw, tiles, rows, hidden, depths, active, pool=attention_pool)
      payload, scales = _arrays(outputs)
      assert payload.size and scales.size
      # Dispatch follows capacity: inactive rows still require zero payloads
      # and scale headers, so the bounded row work covers all 17 rows.
      assert attention_pool.lib.pool_calls() - before == 1
  # The packer's static row capacity below 16 never enters the worker pool.
  small_rows = 15
  small_raw, small_sa, small_sw, small_tiles, _ = fixture(small_rows, hidden, 128, seed=742)
  with attention_pool.prefill():
    before = attention_pool.lib.pool_calls()
    _arrays(pack(small_raw, small_sa, small_sw, small_tiles, small_rows, hidden, depths, small_rows,
                 pool=attention_pool))
    assert attention_pool.lib.pool_calls() == before
  assert attention_pool.lib.pool_parked()

  # An inactive pool retains the existing local-thread fallback.
  before = attention_pool.lib.pool_calls()
  outputs = pack(raw, sa, sw, tiles, rows, hidden, depths, 17, pool=attention_pool)
  _arrays(outputs)
  assert attention_pool.lib.pool_calls() == before
  assert attention_pool.lib.pool_parked()


@pytest.mark.parametrize('activation', ['gelu', 'swiglu'])
def test_pooled_tinyjit_replay_rebinds_rows_and_changed_inputs(attention_pool, activation):
  pack = pack_gelu_projection if activation == 'gelu' else pack_swiglu_projection
  rows, hidden, input_tile = 128, 512, 64
  raw, sa, sw, tiles, projected = fixture(rows, hidden, input_tile, seed=744)
  depths = (hidden,)
  count = UOp.variable(f'{activation}_pool_rows', 1, rows)

  def run(*args):
    nraw, nsa = len(raw), len(sa)
    outputs = pack(list(args[:nraw]), list(args[nraw:nraw+nsa]), args[nraw+nsa], tiles, rows, hidden,
                   depths, args[-1], pool=attention_pool)
    # Return realized outputs so graph capture and replay observe the full custom kernel.
    outputs[0].realize(outputs[1])
    return outputs

  jit = TinyJit(run)
  affinity = os.sched_getaffinity(0)
  for step, active in enumerate((17, 128, 1, 65, 17)):
    if step:
      # Rebind every input tensor to changed storage while preserving the captured layout.
      fresh_raw, fresh_sa, fresh_sw, fresh_tiles, projected = fixture(rows, hidden, input_tile, seed=900+step)
      assert fresh_tiles == tiles
    else:
      fresh_raw, fresh_sa, fresh_sw = raw, sa, sw
    expected = _arrays(_reference(projected, active, hidden, depths, activation))
    with attention_pool.prefill():
      before = attention_pool.lib.pool_calls()
      actual = _arrays(jit(*fresh_raw, *fresh_sa, fresh_sw, count.bind(active)))
      assert attention_pool.lib.pool_calls() - before == int(rows >= 16)
      assert os.sched_getaffinity(0) == affinity
    assert attention_pool.lib.pool_parked()
    assert os.sched_getaffinity(0) == affinity
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
