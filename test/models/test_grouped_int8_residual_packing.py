"""CPU checks for residual grouped INT8 weight preparation."""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.llm.kernels.rockchip_grouped_int8 import pack_weight


@pytest.fixture
def cpu_only_rockchip_transfer(monkeypatch):
  original_to = Tensor.to
  transfers = []

  def to_cpu_for_rockchip(self, device, *args, **kwargs):
    if isinstance(device, str) and device.split(':')[0] == 'ROCKCHIP':
      transfers.append((device, self.shape))
      return self
    return original_to(self, device, *args, **kwargs)

  monkeypatch.setattr(Tensor, 'to', to_cpu_for_rockchip)
  before = {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'}
  yield transfers
  after = {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'}
  assert after == before


def _quantize(values):
  values = values.astype(np.float32)
  scale = np.maximum(np.max(np.abs(values), axis=-1) / np.float32(127), np.float32(1e-8))
  codes = np.clip(np.rint(values * (np.float32(1) / scale[..., None])), -127, 127).astype(np.int8)
  return codes, scale


def _untile(packed):
  groups, nt, kt, _, _ = packed.shape
  return packed.transpose(0, 1, 3, 2, 4).reshape(groups, nt * 32, kt * 32)


def _remainder(values, codes, scales):
  # FP64 represents this INT8/F32 product and close subtraction exactly;
  # one final cast is the independent oracle for the explicit F32 FMA.
  return (values.astype(np.float64)-codes.astype(np.float64)*scales[..., None].astype(np.float64)).astype(np.float32)


def test_residual_pack_has_original_then_residual_group_order_and_padding(cpu_only_rockchip_transfer):
  rng = np.random.default_rng(931)
  source = rng.normal(0, .23, (2, 35, 33)).astype(np.float32)
  source[0, 0] = 0
  weight = Tensor(source, device='CPU').realize()
  original_object = weight
  before = weight.numpy().copy()

  packed, scales = pack_weight(weight, residual=True)

  c0, s0 = _quantize(before)
  residual = _remainder(before, c0, s0)
  c1, s1 = _quantize(residual)
  padded_n, padded_k = 64, 64
  codes = np.zeros((6, padded_n, padded_k), dtype=np.int8)
  codes[0:2, :35, :33] = c0
  codes[2:4, :35, :33] = c0
  codes[4:6, :35, :33] = c1
  expected_scales = np.ones((6, padded_n), dtype=np.float32)
  expected_scales[0:2, :35] = s0
  expected_scales[2:4, :35] = s0
  expected_scales[4:6, :35] = s1
  expected_tiled = codes.reshape(6, 2, 32, 2, 32).transpose(0, 1, 3, 2, 4)

  assert packed.shape == (6, 2, 2, 32, 32)
  assert packed.dtype is dtypes.int8 and packed.device == 'CPU'
  np.testing.assert_array_equal(packed.numpy(), expected_tiled)
  np.testing.assert_array_equal(scales.numpy(), expected_scales)
  dense = _untile(packed.numpy())
  assert np.all(dense[:, 35:, :] == 0)
  assert np.all(dense[:, :, 33:] == 0)
  np.testing.assert_array_equal(weight.numpy(), before)
  assert weight is original_object
  assert cpu_only_rockchip_transfer == [('ROCKCHIP', packed.shape)]


def test_residual_pack_zero_weights_use_finite_floor_and_padding_scale_one(cpu_only_rockchip_transfer):
  weight = Tensor.zeros((1, 33, 17), device='CPU')
  packed, scales = pack_weight(weight, residual=True)
  assert packed.shape == (3, 2, 2, 32, 32)
  assert np.all(packed.numpy() == 0)
  scale = scales.numpy()
  assert np.all(scale[:, :33] == np.float32(1e-8))
  assert np.all(scale[:, 33:] == 1)
  assert np.isfinite(scale).all()
  assert cpu_only_rockchip_transfer == [('ROCKCHIP', packed.shape)]


def test_residual_pack_rejects_nonfinite_and_group_limit_before_transfer(cpu_only_rockchip_transfer):
  invalid = np.zeros((1, 2, 32), dtype=np.float32)
  invalid[0, 0, 0] = np.inf
  with pytest.raises(ValueError):
    pack_weight(Tensor(invalid, device='CPU').realize(), residual=True)
  with pytest.raises(ValueError):
    pack_weight(Tensor.zeros((43, 1, 1), device='CPU'), residual=True)
  assert cpu_only_rockchip_transfer == []


@pytest.mark.parametrize('rows', [1, 5])
def test_residual_matmul_payload_and_output_against_independent_integer_math(rows, cpu_only_rockchip_transfer, monkeypatch):
  from tinygrad.llm.kernels import rockchip_grouped_int8 as grouped
  rng = np.random.default_rng(932)
  values = rng.normal(0, .13, (2, 35, 33)).astype(np.float32)
  data = rng.normal(0, .4, (2, rows, 33)).astype(np.float32)
  data[:, :, 0] *= 24
  packed, scales = pack_weight(Tensor(values, device='CPU'), residual=True)
  w0, sw0 = _quantize(values)
  w1, sw1 = _quantize(_remainder(values, w0, sw0))
  a0, sa0 = _quantize(data)
  a1, sa1 = _quantize(_remainder(data, a0, sa0))
  expected_codes = np.pad(np.concatenate((a0, a1, a0)), ((0, 0), (0, 0), (0, 31)))
  expected_scales = np.concatenate((sa0, sa1, sa0)).reshape(-1)
  submissions = []

  def raw(payload, weight, count, columns, cores):
    header = ((6*rows*4+63)//64)*64
    contents = payload.numpy()
    np.testing.assert_array_equal(contents[:6*rows*4].view(np.float32), expected_scales)
    assert np.all(contents[6*rows*4:header] == 0)
    np.testing.assert_array_equal(contents[header:].view(np.int8).reshape(6, rows, 64), expected_codes)
    integer = expected_codes.astype(np.int32) @ _untile(weight.numpy()).astype(np.int32).swapaxes(-1, -2)
    submissions.append((count, columns, cores))
    return Tensor(integer, device='CPU')

  monkeypatch.setattr(grouped, 'raw_matmul', raw)
  actual = grouped.matmul(Tensor(data, device='CPU'), packed, scales, 35, residual=True).numpy()
  parts = []
  for a, sa, w, sw in ((a0, sa0, w0, sw0), (a1, sa1, w0, sw0), (a0, sa0, w1, sw1)):
    integer = a.astype(np.int32) @ w.astype(np.int32).swapaxes(-1, -2)
    parts.append((integer.astype(np.float32)*sa[..., None])*sw[:, None, :])
  np.testing.assert_allclose(actual, (parts[0]+parts[1])+parts[2], rtol=2e-6, atol=1e-6)
  assert submissions == [(rows, 35, 3)]
