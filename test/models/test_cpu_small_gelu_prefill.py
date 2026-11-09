import numpy as np
import pytest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.llm.kernels import rockchip_gelu


def _small_gelu_inputs(kind, rows, seed):
  rng = np.random.default_rng(seed)
  gate = rng.normal(0, .7, (1, 128, 256)).astype(np.float32)
  up = rng.normal(0, .7, (1, 128, 256)).astype(np.float32)
  if kind == "zero":
    gate.fill(0)
    up.fill(0)
  elif kind == "wide":
    gate *= 32
    up *= 16
  elif kind == "extreme":
    gate.flat[:8] = [-80, -40, -20, -1, 1, 20, 40, 80]
    up.flat[:8] = [80, -80, 40, -40, 20, -20, 1, -1]
  elif kind == "nonfinite":
    gate.flat[:8] = [np.nan, np.inf, -np.inf, 0, -0., 1, -1, np.nan]
    up.flat[:8] = [1, 2, 3, np.nan, np.inf, -np.inf, 0, -0.]
  return (Tensor(gate, device="CPU").realize(), Tensor(up, device="CPU").realize(),
          UOp.variable(f"small_gelu_rows_{kind}_{rows}_{seed}", 1, 128).bind(rows))


@pytest.mark.parametrize("kind", ["finite", "zero", "wide", "extreme", "nonfinite"])
@pytest.mark.parametrize("rows", [1, 17, 127, 128])
def test_small_gelu_prefill_kernel_is_bit_exact_to_existing_path(kind, rows):
  gate, up, active = _small_gelu_inputs(kind, rows, seed=20261104 + rows)
  gate, up = gate[:, :active], up[:, :active]
  with Context(ONPU_CPU_SMALL_GELU_PREFILL=0):
    control = rockchip_gelu.gated_gelu(gate, up).pad_to((1, 128, 256)).realize().numpy()[:, :rows]
  with Context(ONPU_CPU_SMALL_GELU_PREFILL=1):
    candidate = rockchip_gelu.gated_gelu(gate, up).pad_to((1, 128, 256)).realize().numpy()[:, :rows]
  np.testing.assert_array_equal(candidate.view(np.uint32), control.view(np.uint32))


def _selected_gelu_program(monkeypatch, *, enabled, shape=(1, 128, 256), dtype=dtypes.float):
  calls = []
  original = rockchip_gelu._program

  def record(*args, **kwargs):
    calls.append((args[-2], args[-1]))
    return original(*args, **kwargs)

  monkeypatch.setattr(rockchip_gelu, "_program", record)
  gate = Tensor.ones(shape, device="CPU", dtype=dtype).realize()
  up = Tensor.ones(shape, device="CPU", dtype=dtype).realize()
  with Context(ONPU_CPU_SMALL_GELU_PREFILL=int(enabled)):
    rockchip_gelu.gated_gelu(gate, up).realize()
  return calls


def test_small_gelu_program_selection_is_opt_in_and_shape_scoped(monkeypatch):
  assert _selected_gelu_program(monkeypatch, enabled=False) == []
  assert _selected_gelu_program(monkeypatch, enabled=True) == [(128, 256)]


@pytest.mark.parametrize("shape,dtype", [
  ((1, 1, 256), dtypes.float),       # decode shape
  ((1, 64, 256), dtypes.float),      # other prefill capacity
  ((1, 128, 256), dtypes.half),     # wrong dtype
])
def test_small_gelu_program_falls_back_for_decode_other_capacity_and_wrong_dtype(monkeypatch, shape, dtype):
  assert _selected_gelu_program(monkeypatch, enabled=True, shape=shape, dtype=dtype) == []
