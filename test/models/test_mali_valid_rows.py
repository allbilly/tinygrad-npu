"""Valid-row Mali GEMM launch rows must stay dynamic and preserve legacy bits."""
from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, function
from tinygrad.llm.kernels import mali_linear
from tinygrad.uop.ops import AddrSpace


VALID_COUNTS = (128, 63, 1, 127, 31, 32, 33, 64, 65, 128)
COLUMNS, DEPTH, CAPACITY = 67, 96, 128


@contextmanager
def runtime_launches():
  from tinygrad.runtime.ops_cl import CLProgram
  initialize,invoke = CLProgram.__init__,CLProgram.__call__
  launches = []
  def init(self,device,obj):
    initialize(self,device,obj)
    self._valid_rows_name = obj.name
  def call(self,*args,**kwargs):
    if getattr(self,'_valid_rows_name','').endswith('_valid'):
      values = {name:argument for argument,(name,space,_,_) in zip(args,self.signature) if space is AddrSpace.ALU}
      launches.append((kwargs['global_size'],values))
    return invoke(self,*args,**kwargs)
  with patch.object(CLProgram,'__init__',init),patch.object(CLProgram,'__call__',call):yield launches


def legacy_gemm(x: Tensor, weight: Tensor, packed: Tensor | None, rows: int,
                columns: int = COLUMNS, depth: int = DEPTH) -> Tensor:
  """Call the unchanged capacity-row _gemm kernel for an exact bit reference."""
  output = Tensor.empty(*x.max_shape[:-1], columns, device=x.device, dtype=dtypes.float)
  source = packed if packed is not None else weight.contiguous()
  return output.custom_kernel(x.pad_to(x.max_shape).contiguous(), source,
      fxn=lambda out, activation, projection: mali_linear._gemm(
        out, activation, projection, rows, columns, depth, packed is not None))[0].realize()


def valid_launch_program(records, rows_capacity: int = CAPACITY) -> dict:
  program = next((record for record in records if record['launch_rows'] is not None), None)
  if program is None:
    raise AssertionError('expected the bounded symbolic _gemm path to carry launch_rows')
  if program['rows_capacity'] != rows_capacity:
    raise AssertionError(f'legacy rows arg changed: {program["rows_capacity"]} != {rows_capacity}')
  if not program['scalars']:
    raise AssertionError('dynamic _gemm launch must pass scalar parameters, not only a Python callback')
  return program


def assert_program_grid(test: unittest.TestCase, record: dict, bindings: dict[str, int], valid_count: int) -> None:
  launch_rows = record['launch_rows']
  variables = {variable.arg.name for variable in launch_rows.variables()}
  test.assertEqual(variables, set(bindings))
  info = record['program'].arg
  test.assertTrue(isinstance(info.global_size[1], UOp), info.global_size)
  global_size, local_size = info.launch_dims(bindings)
  test.assertEqual(global_size[1], (valid_count + 31) // 32)
  test.assertEqual(local_size, (64, 1, 1))


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware test')
class TestMaliValidRowsHardware(unittest.TestCase):
  def test_function_boundary_preserves_dynamic_runtime_launch(self):
    rng=np.random.default_rng(6204)
    weight=Tensor(rng.normal(0,.1,(COLUMNS,DEPTH)).astype(np.float16),device='CL').realize()
    layer=function(lambda x,w:mali_linear.gemm(x,w))
    run=TinyJit(lambda x,rows:layer(x[:,:rows],weight).realize())
    with runtime_launches() as launches:
      for count in (128,63,1,127,31,32,33,128):
        values=rng.normal(0,.3,(1,CAPACITY,DEPTH)).astype(np.float32)
        values[:,count:]=np.nan
        x=Tensor(values,device='CL').realize()
        actual=run(x,UOp.variable('function_rows',1,CAPACITY).bind(count)).pad_to((1,CAPACITY,COLUMNS)).numpy()[:,:count]
        expected=legacy_gemm(x,weight,None,CAPACITY).numpy()[:,:count]
        self.assert_bits_equal(actual,expected)
        self.assertTrue(launches,'function must retain the valid-row program')
        self.assertEqual(launches[-1][0][1],(count+31)//32)

  def assert_bits_equal(self, actual: np.ndarray, expected: np.ndarray) -> None:
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

  @staticmethod
  def traced_gemm(records):
    original = mali_linear._gemm
    def call(*args, **kwargs):
      result = original(*args, **kwargs)
      launch_rows = args[7] if len(args) > 7 else kwargs.get('launch_rows')
      scalars = args[8] if len(args) > 8 else kwargs.get('scalars', ())
      records.append({'rows_capacity': args[3], 'launch_rows': launch_rows,
                      'scalars': scalars, 'program': result})
      return result
    return patch.object(mali_linear, '_gemm', side_effect=call)

  def test_tinyjit_rebinds_real_buffers_and_valid_rows_match_legacy_bits(self):
    rng = np.random.default_rng(6201)
    for weight_dtype in (np.float16, np.float32):
      weights = (rng.standard_normal((COLUMNS, DEPTH), dtype=np.float32) / np.sqrt(DEPTH)).astype(weight_dtype)
      weight = Tensor(weights, device='CL').realize()
      packed_weight = mali_linear.pack_weight(weight).realize()
      for packed in (None, packed_weight):
        with self.subTest(weight_dtype=str(weight_dtype), packed=packed is not None):
          records = []
          count_name = f'mali_valid_rows_{weight_dtype.__name__}_{int(packed is not None)}'
          run = TinyJit(lambda x, count: mali_linear.gemm(x[:, :count], weight, packed).realize())
          with self.traced_gemm(records),runtime_launches() as launches:
            for count in VALID_COUNTS:
              values = rng.normal(0, 0.3, (1, CAPACITY, DEPTH)).astype(np.float32)
              values[:, count:] = np.nan
              x = Tensor(values, device='CL').realize()  # A real CL input buffer is rebound each call.
              count_var = UOp.variable(count_name, 1, CAPACITY)
              self.assertIs(count_var.dtype, dtypes.weakint)
              actual = run(x, count_var.bind(count)).pad_to((1,CAPACITY,COLUMNS)).numpy()[:,:count]
              expected = legacy_gemm(x, weight, packed, CAPACITY).numpy()[:, :count]
              self.assert_bits_equal(actual, expected)
              record = valid_launch_program(records)
              assert_program_grid(self, record, {count_name: count}, count)
              self.assertEqual(launches[-1][0][1],(count+31)//32)
              self.assertEqual(launches[-1][1],{count_name:count})

  def test_two_scalar_min_rows_rebinds_to_the_actual_minimum(self):
    rng = np.random.default_rng(6202)
    weights = (rng.standard_normal((COLUMNS, DEPTH), dtype=np.float32) / np.sqrt(DEPTH)).astype(np.float16)
    weight = Tensor(weights, device='CL').realize()
    packed_weight = mali_linear.pack_weight(weight).realize()
    for packed in (None, packed_weight):
      with self.subTest(packed=packed is not None):
        records = []
        row_name, limit_name = 'mali_actual_rows', 'mali_row_limit'
        run = TinyJit(lambda x, rows, limit: mali_linear.gemm(
          x[:, :rows.minimum(limit)], weight, packed).realize())
        with self.traced_gemm(records),runtime_launches() as launches:
          for rows, limit in ((128, 63), (63, 128), (65, 33), (33, 65)):
            valid = min(rows, limit)
            values = rng.normal(0, 0.3, (1, CAPACITY, DEPTH)).astype(np.float32)
            values[:, valid:] = np.nan
            x = Tensor(values, device='CL').realize()
            rows_var = UOp.variable(row_name, 1, CAPACITY)
            limit_var = UOp.variable(limit_name, 1, CAPACITY)
            self.assertIs(rows_var.dtype, dtypes.weakint)
            self.assertIs(limit_var.dtype, dtypes.weakint)
            actual = run(x, rows_var.bind(rows), limit_var.bind(limit)).pad_to((1,CAPACITY,COLUMNS)).numpy()[:,:valid]
            expected = legacy_gemm(x, weight, packed, CAPACITY).numpy()[:, :valid]
            self.assert_bits_equal(actual, expected)
            record = valid_launch_program(records)
            assert_program_grid(self, record, {row_name: rows, limit_name: limit}, valid)
            self.assertEqual(launches[-1][0][1],(valid+31)//32)
            self.assertEqual(launches[-1][1],{row_name:rows,limit_name:limit})

  def test_static_small_and_two_batch_symbolic_rows_keep_legacy_dispatch(self):
    rng = np.random.default_rng(6203)
    weights = (rng.standard_normal((COLUMNS, DEPTH), dtype=np.float32) / np.sqrt(DEPTH)).astype(np.float16)
    weight = Tensor(weights, device='CL').realize()
    packed_weight = mali_linear.pack_weight(weight).realize()

    small_values = rng.normal(0, 0.3, (1, 31, DEPTH)).astype(np.float32)
    small = Tensor(small_values, device='CL').realize()
    for packed in (None, packed_weight):
      records = []
      with self.subTest(path='small-static', packed=packed is not None), self.traced_gemm(records):
        actual = mali_linear.gemm(small, weight, packed).numpy()
      expected = legacy_gemm(small, weight, packed, 31).numpy()
      self.assert_bits_equal(actual, expected)
      self.assertEqual(len(records), 1)
      self.assertEqual(records[0]['rows_capacity'], 31)
      self.assertIsNone(records[0]['launch_rows'], 'static/small rows must retain the legacy launch')

    batch_values = rng.normal(0, 0.3, (2, CAPACITY, DEPTH)).astype(np.float32)
    for count in (63, 1, 127):
      values = batch_values.copy()
      values[:, count:] = np.nan
      x = Tensor(values, device='CL').realize()
      rows = UOp.variable('mali_batch_rows', 1, CAPACITY).bind(count)
      for packed in (None, packed_weight):
        records = []
        with self.subTest(path='batch2', count=count, packed=packed is not None), self.traced_gemm(records):
          actual = mali_linear.gemm(x[:, :rows], weight, packed).pad_to((2,CAPACITY,COLUMNS)).numpy()[:,:count]
        expected = legacy_gemm(x, weight, packed, 2 * CAPACITY).numpy()[:, :count]
        self.assert_bits_equal(actual, expected)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['rows_capacity'], 2 * CAPACITY)
        self.assertIsNone(records[0]['launch_rows'], 'multi-batch symbolic shapes retain the capacity-row legacy kernel')


if __name__ == '__main__':
  unittest.main()
