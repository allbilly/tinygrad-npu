"""CPU-only contracts for bounded symbolic Mali row helper metadata."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.mali_attention import _program
from tinygrad.llm.kernels.mali_rows import row_kernel, valid_rows
from tinygrad.uop.ops import AddrSpace, Ops


def launch_program(output, source, rows, scalars):
  return _program("mali_rows_contract", "__kernel void mali_rows_contract(void) {}",
                  (output, source, *scalars), (rows, 1, 1), (1, 1, 1))


def program_of(result: Tensor):
  return next(node for node in result.uop.toposort(enter_calls=True) if node.op is Ops.PROGRAM)


class TestMaliRowsPolicy(unittest.TestCase):
  def tensor_with_rows(self, count: UOp, *, batch: int = 1, capacity: int = 32):
    return Tensor.empty(batch, count, 4, device="CL", dtype=dtypes.float)

  def test_valid_rows_accepts_single_batch_contiguous_symbolic_prefix(self):
    count = UOp.variable("mali_rows_accept", 1, 32).bind(7)
    x = self.tensor_with_rows(count)
    self.assertIs(valid_rows(x, grain=4), count)

  def test_valid_rows_rejects_static_small_multi_batch_zero_min_and_wide_capacity(self):
    self.assertIsNone(valid_rows(Tensor.empty(1, 32, 4, device="CL"), grain=4))
    small = UOp.variable("mali_rows_small", 1, 4).bind(3)
    self.assertIsNone(valid_rows(self.tensor_with_rows(small), grain=4))
    rows = UOp.variable("mali_rows_batch", 1, 32).bind(7)
    self.assertIsNone(valid_rows(self.tensor_with_rows(rows, batch=2), grain=4))
    zero_min = UOp.variable("mali_rows_zero_min", 0, 32).bind(7)
    self.assertIsNone(valid_rows(self.tensor_with_rows(zero_min), grain=4))
    too_wide = UOp.variable("mali_rows_wide", 1, 2**31)
    self.assertIsNone(valid_rows(self.tensor_with_rows(too_wide, capacity=2**31), grain=4))

  def test_row_kernel_converts_default_weak_scalar_to_int32_and_keeps_capacity(self):
    rows = UOp.variable("mali_rows_weak", 1, 32).bind(7)
    self.assertIs(rows.dtype, dtypes.weakint)
    output = Tensor.empty(1, 32, 9, device="CL", dtype=dtypes.float)
    source = Tensor.empty(1, 32, 4, device="CL", dtype=dtypes.float)
    result = row_kernel(output, (source,), rows, launch_program)
    self.assertEqual(result.shape, output.shape)
    self.assertEqual(result.max_shape, (1, 32, 9))
    logical = result.shrink(((0, 1), (0, rows), (0, 9)))
    self.assertEqual(logical.shape, (1, rows, 9))

    program = program_of(result)
    self.assertEqual(program.arg.outs, (0,))
    self.assertEqual(program.arg.ins, (1,))
    self.assertEqual(program.arg.globals, (0, 1))
    scalar_params = [param for param in program.kernel_params if param.arg.addrspace is AddrSpace.ALU]
    self.assertEqual(len(scalar_params), 1)
    self.assertEqual(scalar_params[0].arg.name, "mali_rows_weak")
    self.assertIs(scalar_params[0].dtype, dtypes.int)
    self.assertEqual(scalar_params[0].arg.slot, 2)
    self.assertEqual(program.arg.launch_dims({"mali_rows_weak": 7}), ((7, 1, 1), (1, 1, 1)))

  def test_row_kernel_preserves_unbound_variable_dispatch_metadata(self):
    rows = UOp.variable("mali_rows_unbound", 1, 32)
    output = Tensor.empty(1, 32, 5, device="CL", dtype=dtypes.float)
    source = Tensor.empty(1, 32, 4, device="CL", dtype=dtypes.float)
    result = row_kernel(output, (source,), rows, launch_program)
    program = program_of(result)
    scalar_params = [param for param in program.kernel_params if param.arg.addrspace is AddrSpace.ALU]
    self.assertEqual(len(scalar_params), 1)
    self.assertEqual(scalar_params[0].arg.name, "mali_rows_unbound")
    self.assertIs(scalar_params[0].dtype, dtypes.int)
    self.assertEqual(program.arg.launch_dims({"mali_rows_unbound": 11}), ((11, 1, 1), (1, 1, 1)))

  def test_two_variable_min_expression_binds_both_int32_scalars(self):
    first = UOp.variable("mali_rows_first", 1, 32).bind(9)
    second = UOp.variable("mali_rows_second", 1, 24).bind(5)
    rows = first.minimum(second)
    self.assertIsNotNone(valid_rows(self.tensor_with_rows(rows), grain=4))
    output = Tensor.empty(1, 32, 6, device="CL", dtype=dtypes.float)
    source = Tensor.empty(1, 32, 4, device="CL", dtype=dtypes.float)
    result = row_kernel(output, (source,), rows, launch_program)
    program = program_of(result)
    scalar_params = [param for param in program.kernel_params if param.arg.addrspace is AddrSpace.ALU]
    self.assertEqual({param.arg.name for param in scalar_params}, {"mali_rows_first", "mali_rows_second"})
    self.assertEqual({param.dtype for param in scalar_params}, {dtypes.int})
    self.assertEqual({param.arg.slot for param in scalar_params}, {2, 3})
    self.assertEqual(program.arg.launch_dims({"mali_rows_first": 9, "mali_rows_second": 5}), ((5, 1, 1), (1, 1, 1)))
    self.assertEqual(program.arg.ins, (1,))
