import unittest
from tinygrad import Tensor, Context, Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.late.linearizer import do_split_ends
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.uop.ops import KernelInfo, AxisType, UOp, Ops
from tinygrad.uop.symbolic import uop_given_valid
from test.helpers import to_uops_list

class TestLinearizerRewrite(unittest.TestCase):
  def test_given_valid_preserves_range_bound(self):
    pos, count = UOp.variable("pos", 0, 28), UOp.variable("count", 1, 4)
    rng = UOp.range(pos+count, 0)
    for try_simplex in (False, True):
      self.assertIs(uop_given_valid(3<count, rng+count*8, try_simplex=try_simplex), rng+32)

  def test_given_valid_simplifies_range_index(self):
    rng = UOp.range(4, 0)
    self.assertIs(uop_given_valid(2<rng, rng%3), UOp.const(0))

  def test_range_order(self):
    types = [AxisType.DEVICE, AxisType.GLOBAL, AxisType.LOCAL, AxisType.WARP, AxisType.WEAK, AxisType.LOOP,
             AxisType.UPCAST, AxisType.PLACEHOLDER]
    # Axis type wins over numeric ids, including multi-part ids.
    ranges = [UOp.range(4, 10-i, t).replace(arg=(t, 10-i, j)) for i,t in enumerate(types) for j in (0, 1)]
    self.assertEqual(sorted(ranges[::-1], key=lambda r: r.arg), ranges)
    self.assertEqual(ranges[0].axis_id, (10, 0))
    self.assertEqual(ranges[0].axis_type, AxisType.DEVICE)
    end = do_split_ends(UOp(Ops.NOOP).end(*ranges[::-1]))
    for r in ranges:
      self.assertIs(end.src[1], r)
      end = end.src[0]
    self.assertIs(end.op, Ops.NOOP)

  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)))
      opts_to_apply.append(Opt(OptOps.SPLIT, 2, (4, AxisType.UPCAST)))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device.default.renderer)
      print(prg.src[2].arg)
      self.assertNotIn(Ops.REDUCE, [u.op for u in prg.src[1].src])
      ranges = [u for u in prg.src[1].src if u.op is Ops.RANGE]
      self.assertTrue(ranges)
      self.assertTrue(all(r.axis_type is AxisType.WEAK for r in ranges))

  def test_arange(self):
    out = Tensor.arange(32).clone("NULL")
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device.default.renderer)
      print(prg.src[2].arg)

  def test_kernel_info(self):
    out = Tensor.arange(4).clone("NULL")
    si = out.schedule_linear().src[-1]

    ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=()))
    prg = to_program(ast, Device.default.renderer)
    assert prg.src[0].arg.applied_opts == (), f"expected no opts, got {prg}"

    prg = to_program(ast.replace(arg=KernelInfo(name="custom")), Device.default.renderer)
    self.assertEqual(prg.src[0].arg.name, "custom")

  def test_dependent_loop_bound(self):
    buf, out, counts = UOp.param(0, dtypes.int, 16), UOp.param(1, dtypes.int, 4), UOp.param(2, dtypes.int, 4)
    outer = UOp.range(4, 0, AxisType.LOOP)
    inner = UOp.range(counts.index(outer).load().maximum(0).minimum(4), 1)
    store = buf.index(outer * 4 + inner).store(UOp.const(1, dtypes.int)).end(inner)
    uops = to_uops_list([out.after(store).index(outer).store(UOp.const(2, dtypes.int))])
    self.assertEqual([u.op for u in uops if u.op in (Ops.RANGE, Ops.STORE, Ops.END)],
                     [Ops.RANGE, Ops.RANGE, Ops.STORE, Ops.END, Ops.STORE, Ops.END])
    rngs, ends = [u for u in uops if u.op is Ops.RANGE], [u for u in uops if u.op is Ops.END]
    self.assertEqual([e.src[1] for e in ends], rngs[::-1])

  def test_gated_symbolic_matmul_keeps_single_reduction_loop(self):
    pos, rows = UOp.variable("pos", 0, 28).bind(0), UOp.variable("rows", 1, 4).bind(3)
    a = Tensor.empty(4, 32, device="NULL")[:rows, :pos+rows].pad(((0, 4-rows), (0, 0)))
    b = Tensor.empty(32, 8, device="NULL")[:pos+rows]
    linear, _ = (a@b).linear_with_vars()
    prg = to_program(linear.src[0].src[0], OpenCLRenderer(Target("CL")))
    ranges = [u for u in prg.src[1].src if u.op is Ops.RANGE]
    ends = [u for u in prg.src[1].src if u.op is Ops.END]
    self.assertEqual(len(ranges), 1)
    self.assertEqual(len(ends), 1)
    self.assertIs(ends[0].src[1], ranges[0])
    source = prg.src[2].arg
    self.assertEqual(source.count("{"), source.count("}"))

if __name__ == '__main__':
  unittest.main()
