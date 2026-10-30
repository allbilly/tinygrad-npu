"""CPU checks for grouped INT8 weight packing, without opening an NPU."""
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
      transfers.append(device)
      return self
    return original_to(self, device, *args, **kwargs)

  monkeypatch.setattr(Tensor, 'to', to_cpu_for_rockchip)
  before = {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'}
  yield transfers
  after = {name for name in Device._opened_devices if name.split(':')[0] == 'ROCKCHIP'}
  assert after == before


@pytest.mark.parametrize('shape,dtype,padded_k', [
  ((40, 256, 64), np.float32, 64),
  ((40, 64, 256), np.float32, 256),
  ((3, 33, 96), np.float32, 128),
  ((1, 65, 17), np.float16, 64),
])
def test_pack_weight_matches_independent_quantization_and_tiled_layout(shape, dtype, padded_k, cpu_only_rockchip_transfer):
  rng = np.random.default_rng(401)
  source = rng.normal(0, 0.3, shape).astype(dtype)
  source[:, 0, :] = 0
  weight = Tensor(source, device='CPU').realize()
  before = weight.numpy().copy()
  packed, scales = pack_weight(weight)

  groups, columns, depth = shape
  padded_n = (columns + 31) // 32 * 32
  reference = before.astype(np.float32)
  expected_scales = np.maximum(np.max(np.abs(reference), axis=-1) / np.float32(127), np.float32(1e-8))
  expected_codes = np.clip(np.rint(reference * (np.float32(1)/expected_scales[..., None])), -127, 127).astype(np.int8)
  expected_codes = np.pad(expected_codes, ((0, 0), (0, padded_n-columns), (0, padded_k-depth)))
  expected_tiles = expected_codes.reshape(groups, padded_n//32, 32, padded_k//32, 32).transpose(0, 1, 3, 2, 4)
  expected_scales = np.pad(expected_scales, ((0, 0), (0, padded_n-columns)), constant_values=1)

  assert packed.shape == expected_tiles.shape
  assert packed.dtype is dtypes.int8 and packed.device == 'CPU'
  np.testing.assert_array_equal(packed.numpy(), expected_tiles)
  np.testing.assert_allclose(scales.numpy(), expected_scales, rtol=1e-6, atol=0)
  assert np.all(scales.numpy()[:, columns:] == 1)
  assert np.all(scales.numpy()[:, 0] == 1e-8)
  np.testing.assert_array_equal(weight.numpy(), before)
  assert cpu_only_rockchip_transfer == ['ROCKCHIP']


@pytest.mark.parametrize('shape', [
  (129, 1, 1), (1, 4097, 1), (1, 1, 4097), (7, 4096, 1),
])
def test_pack_weight_rejects_out_of_bounds_geometry_before_transfer(shape, cpu_only_rockchip_transfer):
  weight = Tensor.zeros(shape, device='CPU')
  with pytest.raises(ValueError):
    pack_weight(weight)
  assert cpu_only_rockchip_transfer == []


def test_pack_weight_rejects_invalid_rank_and_nonfinite_values_before_transfer(cpu_only_rockchip_transfer):
  with pytest.raises(ValueError):
    pack_weight(Tensor.zeros(3, 4, device='CPU'))
  invalid = np.zeros((1, 2, 32), dtype=np.float32)
  invalid[0, 0, 0] = np.nan
  with pytest.raises(ValueError, match='scales'):
    pack_weight(Tensor(invalid, device='CPU').realize())
  assert cpu_only_rockchip_transfer == []
