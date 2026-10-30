"""Precompiled functions should not compute pure inputs their bodies ignore."""
import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit, UOp, dtypes
from tinygrad.function import function
from tinygrad.llm.gguf import ggml_data_to_tensor


_Q8_ELEMENTS = 256 * 256


def _lazy_q8_weight():
  """Small real Q8_0 payload decoded lazily by the GGUF conversion path."""
  quant = np.tile(np.arange(-16, 16, dtype=np.int8), _Q8_ELEMENTS // 32)
  blocks = np.zeros((_Q8_ELEMENTS // 32, 34), dtype=np.uint8)
  blocks[:, :2] = np.array([0x00, 0x38], dtype=np.uint8)  # little-endian FP16 scale 0.5
  blocks[:, 2:] = quant.view(np.uint8).reshape(-1, 32)
  raw = Tensor(blocks.reshape(-1), dtype=dtypes.uint8, device='CPU').realize()
  return ggml_data_to_tensor(raw, _Q8_ELEMENTS, 8).reshape(256, 256).cast(dtypes.half)


def test_precompiled_tinyjit_skips_unused_lazy_q8_weight_but_computes_used_weight():
  class IgnoresWeight:
    def __init__(self): self.weight = _lazy_q8_weight()
    @function(precompile=True)
    def __call__(self, x:Tensor) -> Tensor: return x + 3

  ignored = IgnoresWeight()
  x_values = np.arange(256,dtype=np.float32).reshape(1,256)
  assert not ignored.weight.uop.is_realized

  @TinyJit
  def repeat(value): return ignored(value).realize()

  GlobalCounters.reset()
  for shift in (0, 1, 7, 13):
    value = Tensor(x_values+shift,device='CPU').realize()
    actual = repeat(value)
    np.testing.assert_array_equal(actual.numpy(), value.numpy() + 3)
  assert not ignored.weight.uop.is_realized
  assert GlobalCounters.global_ops < _Q8_ELEMENTS

  class UsesWeight:
    def __init__(self): self.weight = _lazy_q8_weight()
    @function(precompile=True)
    def __call__(self, value:Tensor) -> Tensor: return value @ self.weight.T

  used = UsesWeight()
  value = Tensor.ones(1, 256, device='CPU').realize()
  assert not used.weight.uop.is_realized
  GlobalCounters.reset()
  actual = used(value).numpy()
  quant = np.tile(np.arange(-16, 16, dtype=np.int8), _Q8_ELEMENTS // 32)
  expected_weight = (quant.reshape(256, 256).astype(np.float32) * .5).astype(np.float16).astype(np.float32)
  np.testing.assert_allclose(actual, np.ones((1, 256), dtype=np.float32) @ expected_weight.T,
                             rtol=1e-5, atol=1e-4)
  assert GlobalCounters.global_ops >= _Q8_ELEMENTS


def test_nested_precompiled_gapped_slots_symbolic_shape_and_tuple_outputs():
  @function(precompile=True)
  def inner(ignored:Tensor, x:Tensor, n:UOp) -> tuple[Tensor, Tensor]:
    return x[:n] + 1, x[:n] * 3

  @function(precompile=True)
  def outer(ignored:Tensor, x:Tensor, another_ignored:Tensor, n:UOp) -> tuple[Tensor, Tensor]:
    return inner(ignored, x, n)

  for size in (3, 6, 2, 7):
    ignored = Tensor.full((8,), -100 - size, device='CPU').realize()
    x_values = np.arange(size, dtype=np.float32) + size
    x = Tensor(np.pad(x_values, (0, 8-size)), device='CPU').realize()
    another_ignored = Tensor.full((8,), 200 + size, device='CPU').realize()
    n = UOp.variable('dynamic_output_size', 1, 8, dtype=dtypes.int32).bind(size)
    first, second = outer(ignored, x, another_ignored, n)
    Tensor.realize(first, second)
    np.testing.assert_array_equal(first.pad_to((8,)).numpy()[:size], x_values + 1)
    np.testing.assert_array_equal(second.pad_to((8,)).numpy()[:size], x_values * 3)


def test_pruning_keeps_distinct_same_shape_inputs_after_slot_renumbering():
  @function(precompile=True)
  def combine(ignored:Tensor, a:Tensor, b:Tensor, c:Tensor) -> Tensor:
    return a+b*10+c*100
  inputs=[Tensor(np.full(8,v,np.float32),device='CPU').realize() for v in (-7,1,2,3)]
  actual=combine(*inputs).numpy()
  np.testing.assert_array_equal(actual,np.full(8,321,np.float32))


def test_pruning_preserves_device_inference_for_constant_and_scalar_outputs():
  @function(precompile=True)
  def constant(ignored:Tensor) -> Tensor:return Tensor.const(7.,dtypes.float32)
  @function(precompile=True)
  def scalar(ignored:Tensor,n:UOp) -> Tensor:return Tensor(n).float()
  # Inspect placement without allocating/opening the NULL device.
  anchor=Tensor.empty(8,device='NULL:7')
  assert constant(anchor).device=='NULL:7'
  assert scalar(anchor,UOp.variable('scalar_result',1,8).bind(3)).device=='NULL:7'


def test_precompiled_unused_effectful_argument_keeps_assignment_side_effect():
  target = Tensor.zeros(4, device='CPU').contiguous().realize()
  update = Tensor([4., 3., 2., 1.], device='CPU').realize()
  effectful = Tensor(target.uop.after(target.uop.store(update.uop)))

  @function(precompile=True)
  def ignores_state(state:Tensor, x:Tensor) -> Tensor:
    return x * 2

  x = Tensor([1., 2., 3., 4.], device='CPU').realize()
  actual = ignores_state(effectful, x)
  np.testing.assert_array_equal(actual.numpy(), [2., 4., 6., 8.])
  np.testing.assert_array_equal(target.numpy(), [4., 3., 2., 1.])


def test_precompiled_custom_gradient_preserves_original_argument_abi():
  # Match test_function.py's grad_fxn(grad, call) API. The first explicit input
  # is unused by the body, so the returned gradient tuple must keep x in slot 1.
  def grad_fxn(grad:UOp, call:UOp): return (None, grad*7)

  @function(precompile=True, grad_fxn=grad_fxn)
  def scaled(ignored:Tensor, x:Tensor) -> Tensor:
    return x * 3

  ignored = Tensor([8., 9., 10.], device='CPU').contiguous().realize()
  x = Tensor([1., 2., 3.], device='CPU').contiguous().realize()
  scaled(ignored, x).sum().backward()
  np.testing.assert_array_equal(x.grad.numpy(), [7., 7., 7.])
