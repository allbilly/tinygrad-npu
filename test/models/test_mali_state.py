"""Mali packed snapshots preserve raw state bits and independent rollback buffers."""
import os, unittest
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.mali_state import PackedStateSnapshot, supported
from tinygrad.llm.model import GatedDeltaNetBlock, Transformer
from tinygrad.llm.speculative import _StateSnapshot
from test.models.test_speculative_target import config


SPECIAL_BITS=np.array((0x00000000,0x80000000,0x7f800000,0xff800000,0x7fc00001,0x7fa12345,
  0xffc01234,0x00000001,0x80000001,0x007fffff,0x807fffff,0x3f800000),dtype=np.uint32)


def state_bits(state: Tensor) -> np.ndarray:
  return state.numpy().copy().view(np.uint32).reshape(-1)


def patterned_bits(length: int, seed: int) -> np.ndarray:
  rng=np.random.default_rng(seed)
  values=rng.bit_generator.random_raw(length).astype(np.uint32)
  values[:min(length,len(SPECIAL_BITS))]=SPECIAL_BITS[:min(length,len(SPECIAL_BITS))]
  return values


def write_bits(state: Tensor, bits: np.ndarray) -> None:
  values=np.ascontiguousarray(bits.reshape(state.shape).view(np.float32))
  state.assign(Tensor(values,device='CL',dtype=dtypes.float)).realize()


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliPackedStateSnapshot(unittest.TestCase):
  def states_with_real_gdn_shapes(self):
    model=Transformer(config(True))
    states=[]
    for block in model.blk:
      if isinstance(block,GatedDeltaNetBlock):
        block._init_state(Tensor.zeros(1,1,model.config.dim,device='CL',dtype=dtypes.float))
        states.extend((block.conv_state,block.recurrent_state))
    self.assertGreaterEqual(len(states),2)
    odd_lengths=(3,5,11)
    states.extend(Tensor.zeros(length,device='CL',dtype=dtypes.float).realize() for length in odd_lengths)
    Tensor.realize(*states)
    self.assertTrue(any(state.numel()%2 for state in states))
    return tuple(states)

  def assert_states_equal_bits(self, states, expected):
    self.assertEqual(len(states),len(expected))
    for state,bits in zip(states,expected):
      actual=state_bits(state)
      self.assertEqual(actual.shape,bits.shape)
      np.testing.assert_array_equal(actual,bits)

  def test_snapshot_restore_preserves_raw_f32_bits_and_reports_exact_bytes(self):
    states=self.states_with_real_gdn_shapes()
    original=tuple(patterned_bits(int(state.numel()),8110+index) for index,state in enumerate(states))
    for state,bits in zip(states,original):write_bits(state,bits)
    snapshot=PackedStateSnapshot(states)
    self.assertEqual(snapshot.nbytes,sum(int(state.numel())*4 for state in states))
    mutated=tuple(np.full(int(state.numel()),0x3f000000+index,dtype=np.uint32) for index,state in enumerate(states))
    for state,bits in zip(states,mutated):write_bits(state,bits)
    for state,bits,old in zip(states,mutated,original):
      np.testing.assert_array_equal(state_bits(state),bits)
      self.assertFalse(np.array_equal(bits,old))
    snapshot.restore()
    self.assert_states_equal_bits(states,original)

  def test_live_snapshots_restore_independently_through_capture_and_replay(self):
    states=self.states_with_real_gdn_shapes()
    first=tuple(patterned_bits(int(state.numel()),8210+index) for index,state in enumerate(states))
    second=tuple(patterned_bits(int(state.numel()),8310+index) for index,state in enumerate(states))
    third=tuple(np.full(int(state.numel()),0x41200000+index,dtype=np.uint32) for index,state in enumerate(states))
    for state,bits in zip(states,first):write_bits(state,bits)
    older=PackedStateSnapshot(states)
    for state,bits in zip(states,second):write_bits(state,bits)
    newer=PackedStateSnapshot(states)
    for state,bits in zip(states,third):write_bits(state,bits)
    newest=PackedStateSnapshot(states)
    for state in states:write_bits(state,np.zeros(int(state.numel()),dtype=np.uint32))
    newest.restore()
    self.assert_states_equal_bits(states,third)
    newer.restore()
    self.assert_states_equal_bits(states,second)
    for state,bits in zip(states,third):write_bits(state,bits)
    older.restore()
    self.assert_states_equal_bits(states,first)
    newest.restore()
    self.assert_states_equal_bits(states,third)

  def test_single_state_rebinds_each_independent_backup(self):
    state=Tensor.zeros(11,device='CL',dtype=dtypes.float).realize()
    patterns=[patterned_bits(11,8400+i) for i in range(3)]
    snapshots=[]
    for bits in patterns:
      write_bits(state,bits)
      snapshots.append(PackedStateSnapshot((state,)))
    for index in (2,0,1,2):
      snapshots[index].restore()
      np.testing.assert_array_equal(state_bits(state),patterns[index])

  def test_speculative_snapshot_bridge_uses_native_copy_and_preserves_bits(self):
    states=self.states_with_real_gdn_shapes()
    original=tuple(patterned_bits(int(state.numel()),8500+i) for i,state in enumerate(states))
    for state,bits in zip(states,original):write_bits(state,bits)
    target=SimpleNamespace(blk=[SimpleNamespace(conv_state=state) for state in states],_mali_native_state_snapshots=True)
    snapshot=_StateSnapshot(target)
    self.assertIsInstance(snapshot.native,PackedStateSnapshot)
    self.assertEqual(snapshot.nbytes,sum(int(state.numel())*4 for state in states))
    for state in states:write_bits(state,np.zeros(int(state.numel()),dtype=np.uint32))
    snapshot.restore()
    self.assert_states_equal_bits(states,original)

  def test_unsupported_half_states_keep_the_ordinary_snapshot_path(self):
    state=Tensor([1.,-2.,3.],device='CL',dtype=dtypes.half).realize()
    target=SimpleNamespace(blk=[SimpleNamespace(conv_state=state)],_mali_native_state_snapshots=True)
    snapshot=_StateSnapshot(target)
    self.assertIsNone(snapshot.native)
    state.assign(Tensor.zeros_like(state)).realize()
    snapshot.restore()
    np.testing.assert_array_equal(state.numpy(),np.array([1.,-2.,3.],dtype=np.float16))

  def test_rejected_layouts_do_not_write_any_input_buffer(self):
    states=self.states_with_real_gdn_shapes()
    odd=states[-1]
    baseline=tuple(state_bits(state) for state in states)
    cpu=Tensor.zeros(odd.numel(),device='CPU',dtype=dtypes.float).realize()
    half=Tensor.zeros(odd.numel(),device='CL',dtype=dtypes.half).realize()
    full_view=odd.reshape(odd.numel(),1)
    partial_view=odd[:-1]
    rejected=(
      (),
      (odd,half),
      (odd,cpu),
      (odd,odd),
      (odd,full_view),
      (odd,partial_view),
      (odd,)*97,
    )
    for case,invalid in enumerate(rejected):
      with self.subTest(case=case):
        self.assertFalse(supported(invalid))
        with self.assertRaises(ValueError):PackedStateSnapshot(invalid)
        self.assert_states_equal_bits(states,baseline)


if __name__=='__main__':unittest.main()
