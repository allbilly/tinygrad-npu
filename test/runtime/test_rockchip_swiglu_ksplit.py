from types import SimpleNamespace

import numpy as np
import pytest
from tinygrad import Tensor, Device, UOp, dtypes
from tinygrad.llm.kernels import rockchip_gelu_ffn as f
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from tinygrad.llm.kernels.rockchip_shared_pool import SharedHostPool
from tinygrad.llm.kernels.rockchip_swiglu_ksplit import pack, SIZE, combine, SplitSwiGLUPrefill
from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles


@pytest.mark.parametrize("case", ["finite", "zeros", "nan_scale", "inf_scale", "nan_weight", "inf_weight", "overflow", "signed_zero"])
@pytest.mark.parametrize('width,hidden,sdk_tail',[(896,4864,False),(1024,4608,False),(1024,4608,True)])
def test_split_payload_preserves_original_4096_scale_and_every_quantized_code(case,width,hidden,sdk_tail):
  columns,tail=hidden*2,hidden-4096
  core_widths=(3264,3264,3200) if width==896 else (3072,3072,3072)
  rng = np.random.default_rng(379)
  a = rng.integers(-100000, 100000, (128, columns), np.int32)
  s = rng.uniform(0.001, 0.01, 128).astype(np.float32)
  w = rng.uniform(0.001, 0.03, columns).astype(np.float32)
  if case == "zeros":
    a[:] = 0
  if case == "nan_scale":
    s[0] = np.nan
  if case == "inf_scale":
    s[1] = np.inf
  if case == "nan_weight":
    w[0] = np.nan
  if case == "inf_weight":
    w[0] = np.inf
  if case == "overflow":
    s[:] = 1e30
    w[:] = 1e30
  if case == "signed_zero":
    s[::2] = -0.0
    w[::2] = -0.0
  blocked = np.concatenate([
    a[:, first : first + width].reshape(128, width // 4, 4).transpose(1, 0, 2).ravel()
    for first, width in ((0,core_widths[0]),(core_widths[0],core_widths[1]),(sum(core_widths[:2]),core_widths[2]))
  ])
  raw = [Tensor(blocked, device="CPU").realize()]
  sa = [Tensor(s, device="CPU").realize()]
  sw = Tensor(w, device="CPU").realize()
  tiles = [f.ProjectionTile(0, 128, 0, columns, 0, True)]
  pool = AttentionPool()
  state = SharedHostPool(pool)
  try:
    with state.scope():
      old, oldsc = f.pack_swiglu_projection(raw, sa, sw, tiles, 128, hidden, (4096, tail), 128, pool=pool)
      old_bytes = old.numpy()
      old_scales = oldsc.numpy()
      codes = np.concatenate([old_bytes[i * 262400 + 256 : i * 262400 + 262400].reshape(64, 4096) for i in range(2)], axis=0)
      header = old_scales[:128].view(np.uint8)
      expected = np.concatenate([header, codes[:, :2048].ravel(), header, codes[:, 2048:].ravel(), old_bytes[524800:]])
      assert expected.size == (SIZE if width==896 else 591360)
      if sdk_tail:
        expected[525824:] = expected[525824:].reshape(128,tail//16,16).transpose(1,0,2).copy().ravel()
      for max_bits in (False, True):
        got, sc = pack(raw, sa, sw, tiles, pool, max_bits=max_bits,sdk_tail=sdk_tail)
        assert got.numpy().tobytes() == expected.tobytes()
        assert sc.numpy().tobytes() == old_scales.tobytes()
        assert got.numpy()[:512].tobytes() == got.numpy()[262656:263168].tobytes()
    assert set(Device._opened_devices) <= {"CPU", "NPY", "PYTHON"}
  finally:
    state.restore()
    pool.close()


@pytest.mark.parametrize("case", ["finite", "zeros", "subnormal", "nonfinite"])
@pytest.mark.parametrize('width',[896,1024])
def test_parallel_integer_add_then_fp32_dequant_preserves_bits(case,width):
  cores=((0,320),(320,320),(640,256)) if width==896 else ((0,352),(352,352),(704,320))
  rng = np.random.default_rng(380)
  a = rng.integers(-30000000, 30000000, (128, width), np.int32)
  b = rng.integers(-30000000, 30000000, (128, width), np.int32)
  sa = rng.uniform(0.001, 0.01, 128).astype(np.float32)
  sw = rng.uniform(0.001, 0.1, width).astype(np.float32)
  if case == "zeros":
    a[:] = 0
    b[:] = 0
  if case == "subnormal":
    sa[:] = np.nextafter(np.float32(0), np.float32(1))
  if case == "nonfinite":
    sa[0] = np.inf
    sw[0] = np.nan

  def pack(x):
    return np.concatenate([x[:, start : start + count].ravel() for start, count in cores])

  s = Tensor(sa, device="CPU").realize()
  w = Tensor(sw, device="CPU").realize()
  expected = dequantize_tiles([Tensor(pack(a + b), device="CPU").realize()], [s], w, [128], width).numpy()
  pool = AttentionPool()
  state = SharedHostPool(pool)
  try:
    with state.scope():
      actual = combine(Tensor(pack(a), device="CPU").realize(), Tensor(pack(b), device="CPU").realize(), s, w, pool).numpy()
    assert actual.tobytes() == expected.tobytes()
    assert set(Device._opened_devices) <= {"CPU", "NPY", "PYTHON"}
  finally:
    state.restore()
    pool.close()


@pytest.mark.parametrize('width',[896,1024])
def test_only_full_static_shape_selects_new_pipeline(width):
  obj = SimpleNamespace(pool=SimpleNamespace(active=True), phase_layer=SimpleNamespace(phase="prefill"), original=object(),width=width)
  obj.parts = [SimpleNamespace(packed=[obj.original])]
  x = Tensor.zeros(1, 128, width, device="CPU", dtype=dtypes.float)
  assert SplitSwiGLUPrefill.enabled(obj, x)
  for n in (1, 17, 127):
    assert not SplitSwiGLUPrefill.enabled(obj, x[:, :n])
  n = UOp.variable("k2_partial_rows", 1, 128).bind(128)
  assert not SplitSwiGLUPrefill.enabled(obj, x[:, :n])
  obj.pool.active = False
  assert not SplitSwiGLUPrefill.enabled(obj, x)
  obj.pool.active = True
  obj.phase_layer.phase = "decode"
  assert not SplitSwiGLUPrefill.enabled(obj, x)


@pytest.mark.parametrize('width,hidden',[(896,4864),(1024,4608)])
def test_prepared_weight_halves_preserve_signed_codes_and_rebinding_disables_path(width,hidden):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear,RockchipInt8SplitLinear
  joint=object.__new__(RockchipInt8Linear)
  joint.depth=width
  down=object.__new__(RockchipInt8SplitLinear)
  down.depth=hidden
  values=np.arange((width//32)*128*32*32,dtype=np.int32).reshape(width//32,128,32,32).astype(np.int8)
  original=Tensor(values,device='CPU').realize()
  down.parts=[SimpleNamespace(packed=[original])]
  pool=SimpleNamespace(active=True)
  ffn=SimpleNamespace(projections=SimpleNamespace(joint=joint,layers=[SimpleNamespace(phase='prefill')]),down=down,pool=pool)
  descriptor=SplitSwiGLUPrefill(ffn)
  for half,begin in zip(descriptor.weights,(0,64)):
    assert half.numpy().tobytes()==values[:,begin:begin+64].copy().tobytes()
  x=Tensor.zeros(1,128,width,device='CPU').realize()
  assert descriptor.enabled(x)
  down.parts[0].packed[0]=Tensor.zeros_like(original).realize()
  assert not descriptor.enabled(x)
