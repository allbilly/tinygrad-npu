"""CPU contract tests for the optional native dense-row copy helper.

These tests exercise only eligibility, argument geometry, and error handling.
They do not create a device or claim that a native copy ran on hardware.
"""
from __future__ import annotations
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from tinygrad import dtypes
from tinygrad.llm.kernels import mali_copy


class _FakeDevice:
  device_name = 'Mali-LODX r0p0'
  driver_version = '2.1'


class _FakeDeviceLookup:
  instance = _FakeDevice()

  def __class_getitem__(cls, _key):
    return cls.instance


def _tensor(shape, *, max_shape=None, device='CL:MOCK', dtype=dtypes.float16):
  return SimpleNamespace(shape=shape, max_shape=max_shape or shape, device=device, dtype=dtype, ndim=len(shape))


class TestNativeInputCopyContracts(TestCase):
  def test_copies_only_requested_rows_with_independent_source_and_output_pitches(self):
    source = _tensor((2, 9, 16))
    output = _tensor((2, 13, 16))
    src_buf, out_buf = object(), object()
    calls = []

    with patch.object(mali_copy, '_dense_buffer', side_effect=lambda x: src_buf if x is source else out_buf), \
         patch.object(mali_copy, 'Device', _FakeDeviceLookup), \
         patch.object(mali_copy, '_copy_rect', side_effect=lambda *args: calls.append(args)), \
         patch.object(mali_copy.Tensor, 'empty', side_effect=AssertionError('copy_rows must not allocate')):
      self.assertIs(mali_copy.copy_rows(source, output, 3, 7), True)

    self.assertEqual(len(calls), 1)
    self.assertIs(calls[0][0], _FakeDeviceLookup.instance)
    self.assertEqual(calls[0][1:], (src_buf, out_buf, 9, 13, 3, 4, 3, 2, 32))

  def test_invalid_row_ranges_raise_before_dense_lookup_or_copy(self):
    source, output = _tensor((2, 9, 16)), _tensor((2, 13, 16))
    with patch.object(mali_copy, '_dense_buffer', side_effect=AssertionError('must validate first')):
      for begin, end in ((-1, 2), (3, 2), (0, 10), (True, 2), (0, 2.0)):
        with self.subTest(begin=begin, end=end), self.assertRaises(ValueError):
          mali_copy.copy_rows(source, output, begin, end)

  def test_declines_ineligible_layouts_without_copying(self):
    base = _tensor((2, 9, 16))
    cases = (
      (_tensor((2, 9, 16)), _tensor((1, 13, 16))),  # different head geometry
      (_tensor((2, 9, 16)), _tensor((2, 13, 15))),  # different row width
      (_tensor((2, 9, 16)), _tensor((2, 13, 16), dtype=dtypes.float32)),
      (_tensor((2, 9, 16), dtype=dtypes.int32), _tensor((2, 13, 16), dtype=dtypes.int32)),
      (_tensor((2, 9, 16), device='CPU'), _tensor((2, 13, 16), device='CPU')),
      (base, _tensor((2, 13, 16), device='CL:OTHER')),
      (_tensor((2, 0, 16)), _tensor((2, 13, 16))),
    )
    with patch.object(mali_copy, '_dense_buffer', return_value=object()) as dense, \
         patch.object(mali_copy, 'Device', _FakeDeviceLookup), \
         patch.object(mali_copy, '_copy_rect', side_effect=AssertionError('ineligible copy attempted')):
      for source, output in cases:
        with self.subTest(source=source, output=output):
          self.assertIs(mali_copy.copy_rows(source, output, 1, 3), False)
      self.assertEqual(dense.call_count, 0)

  def test_declines_when_native_recording_or_debug_guards_are_active(self):
    source, output = _tensor((2, 9, 16)), _tensor((2, 13, 16))
    for field, value in (('capturing', True), ('PROFILE', True), ('DEBUG', 2)):
      with self.subTest(field=field), patch.object(mali_copy, field, value), \
           patch.object(mali_copy, '_dense_buffer', side_effect=AssertionError('eligibility must decline first')):
        self.assertIs(mali_copy.copy_rows(source, output, 1, 3), False)

  def test_declines_aliasing_or_non_dense_storage(self):
    source, output = _tensor((2, 9, 16)), _tensor((2, 13, 16))
    shared = object()
    for src, dst in ((shared, shared), (None, object()), (object(), None)):
      with self.subTest(src=src, dst=dst), patch.object(
        mali_copy, '_dense_buffer', side_effect=lambda tensor: src if tensor is source else dst
      ), patch.object(mali_copy, 'Device', _FakeDeviceLookup), \
           patch.object(mali_copy, '_copy_rect', side_effect=AssertionError('declined copy attempted')):
        self.assertIs(mali_copy.copy_rows(source, output, 0, 2), False)

  def test_rejects_wrong_mali_identity_and_propagates_copy_failures(self):
    source, output = _tensor((2, 9, 16)), _tensor((2, 13, 16))
    src_buf, out_buf = object(), object()
    def get_buffer(tensor):
      return src_buf if tensor is source else out_buf

    class OtherDevice:
      device_name = 'Other Mali'
      driver_version = '2.1'

      def __class_getitem__(cls, _key):
        return cls()

    with patch.object(mali_copy, '_dense_buffer', side_effect=get_buffer), \
         patch.object(mali_copy, 'Device', OtherDevice), \
         patch.object(mali_copy, '_copy_rect', side_effect=AssertionError('wrong device copied')):
      self.assertIs(mali_copy.copy_rows(source, output, 0, 2), False)

    failure = RuntimeError('mocked enqueue failure')
    with patch.object(mali_copy, '_dense_buffer', side_effect=get_buffer), \
         patch.object(mali_copy, 'Device', _FakeDeviceLookup), \
         patch.object(mali_copy, '_copy_rect', side_effect=failure):
      with self.assertRaisesRegex(RuntimeError, 'mocked enqueue failure'):
        mali_copy.copy_rows(source, output, 0, 2)
