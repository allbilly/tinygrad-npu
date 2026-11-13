"""Bit-exactness of the decode RMSNorm+quantize fusion and the norm-weight hoist.

Both are decode-only optimizations behind default-OFF env flags:

* `ONPU_NORM_QUANT_DECODE=1` fuses `rockchip_cpu_rmsnorm` with the `rockchip_quantize`
  that immediately consumes its result (qkv, gate/up and lm-head inputs), building one
  single-row kernel from the exact source text of both standing kernels.
* `ONPU_HOST_REALIZE_NORM_WEIGHTS=1` materialises a deferred/mapped RMSNorm weight once,
  so the lazy GGUF cast does not re-run inside every replayed decode token.

The reference here is the REAL chain (`rms_norm` followed by `quantize`), not a numpy
reimplementation: both payload bytes and the activation scale bits must be identical.
"""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Context, round_up
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels import rockchip_norm_quant as norm_quant
from tinygrad.llm.kernels.rockchip import RockchipRMSNorm
from tinygrad.llm.kernels.rockchip_cpu import rms_norm, _rms_norm
from tinygrad.llm.kernels.rockchip_int8 import quantize, _quantize

DEPTH = 4096
EPS = 1e-6
_DECODE_FLAGS = dict(ONPU_NORM_QUANT_DECODE=1, ONPU_NORM_QUANT=0, ONPU_CPU_RMS_ALL_ROWS=1)


def _inputs(kind, half=False, seed=11):
  rng = np.random.default_rng(seed)
  if kind == 'typical': x = rng.normal(0, .3, (1, 1, DEPTH)).astype(np.float32)
  elif kind == 'zeros': x = np.zeros((1, 1, DEPTH), np.float32)
  elif kind == 'large': x = rng.normal(0, 1e4, (1, 1, DEPTH)).astype(np.float32)
  elif kind == 'nonfinite':
    x = rng.normal(0, .3, (1, 1, DEPTH)).astype(np.float32)
    x[0, 0, 7], x[0, 0, 99], x[0, 0, 123] = np.inf, -np.inf, np.nan
  elif kind == 'clamp': x = (rng.integers(-127, 128, (1, 1, DEPTH)).astype(np.float32)/np.float32(127.0))
  elif kind == 'tiny': x = rng.normal(0, 1e-30, (1, 1, DEPTH)).astype(np.float32)
  else: raise ValueError(kind)
  w = rng.uniform(.5, 1.5, DEPTH).astype(np.float16 if half else np.float32)
  return Tensor(x, device='CPU').realize(), Tensor(w, device='CPU').realize()


def _chain(x, weight, fuse):
  with Context(ONPU_NORM_QUANT_DECODE=int(fuse), ONPU_NORM_QUANT=0, ONPU_CPU_RMS_ALL_ROWS=1):
    norm_quant._decode_traces = 0
    normalized = rms_norm(x, weight, EPS)
    payload, scales = quantize(normalized.reshape(1, DEPTH), DEPTH)
    return payload.realize().numpy(), scales.realize().numpy(), norm_quant._decode_traces


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('kind', ['typical', 'zeros', 'large', 'nonfinite', 'clamp', 'tiny'])
def test_fused_decode_norm_quantize_matches_the_real_chain(kind, half):
  x, weight = _inputs(kind, half)
  reference_payload, reference_scales, reference_traces = _chain(x, weight, fuse=False)
  fused_payload, fused_scales, fused_traces = _chain(x, weight, fuse=True)
  assert reference_traces == 0 and fused_traces == 1, 'fusion must engage exactly once'
  assert np.array_equal(reference_payload, fused_payload), 'payload differs'
  assert np.array_equal(reference_scales, fused_scales, equal_nan=True), 'scale differs'


def test_quantize_call_without_a_registered_norm_stays_ordinary():
  x, weight = _inputs('typical')
  with Context(**_DECODE_FLAGS):
    norm_quant._decode_traces = 0
    plain = Tensor(np.random.default_rng(3).normal(0, .3, (1, DEPTH)).astype(np.float32), device='CPU')
    payload, scales = quantize(plain, DEPTH)
    payload.realize(); scales.realize()
  assert norm_quant._decode_traces == 0


def test_multi_row_norm_is_not_fused():
  rng = np.random.default_rng(17)
  x = Tensor(rng.normal(0, .3, (1, 16, DEPTH)).astype(np.float32), device='CPU').realize()
  weight = Tensor(rng.uniform(.5, 1.5, DEPTH).astype(np.float32), device='CPU').realize()
  with Context(**_DECODE_FLAGS):
    norm_quant._decode_traces = 0
    normalized = rms_norm(x, weight, EPS)
    payload, scales = quantize(normalized.reshape(16, DEPTH), DEPTH)
    payload.realize(); scales.realize()
  assert norm_quant._decode_traces == 0, 'prefill rows must keep the ordinary path'


def _source_of(program):
  return next(u.arg for u in program.src if u.op is Ops.SOURCE)


def test_fused_source_keeps_the_exact_text_of_both_kernels(monkeypatch):
  import re
  import tinygrad.llm.kernels.rockchip_cpu as rockchip_cpu
  import tinygrad.llm.kernels.rockchip_int8 as rockchip_int8
  captured = {}
  for module, name, key in ((norm_quant, '_decode_program', 'fused'), (rockchip_cpu, '_rms_norm', 'rms'),
                            (rockchip_int8, '_quantize', 'quant')):
    original = getattr(module, name)
    def wrapper(*args, _original=original, _key=key, **kwargs):
      program = _original(*args, **kwargs)
      captured.setdefault(_key, program)
      return program
    monkeypatch.setattr(module, name, wrapper)
  with Context(**{**_DECODE_FLAGS, 'ONPU_EXP2_FAST_PATH': 1}):
    x, weight = _inputs('typical')
    norm_quant._decode_traces = 0
    normalized = rms_norm(x, weight, EPS)
    payload, scales = quantize(normalized.reshape(1, DEPTH), DEPTH)
    payload.realize(); scales.realize()
    # the standing quantize keeps running for inputs without a registered norm
    plain = Tensor(np.random.default_rng(4).normal(0, .3, (1, DEPTH)).astype(np.float32), device='CPU')
    plain_payload, plain_scales = quantize(plain, DEPTH)
    plain_payload.realize(); plain_scales.realize()
  assert set(captured) == {'fused', 'rms', 'quant'}
  fused = re.sub(r'\s+', ' ', _source_of(captured['fused']))
  rms = re.sub(r'\s+', ' ', _source_of(captured['rms']))
  quant = re.sub(r'\s+', ' ', _source_of(captured['quant']))
  # exact decode RMSNorm arithmetic: one scalar FMA accumulator, division-form normalize
  for fragment in ['float acc=0.0f;', 'acc=__builtin_fmaf(v[0],v[0],acc)',
                   'float denominator=__builtin_sqrtf(acc*', '((value/(float4)denominator))*gamma',
                   'for(int c=0;c<4096;c+=4)']:
    assert fragment in rms, fragment
    assert fragment in fused, fragment
  # exact quantize arithmetic, same text as rockchip_quantize
  for fragment in ['float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];',
                   'int invalid=__builtin_reduce_or(bad)!=0;',
                   'float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;',
                   'invalid?__builtin_nanf(""):scale',
                   'float inverse=1.0f/scale;',
                   'rockchip_quantize_vector(value,inverse)',
                   'bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000']:
    assert fragment in quant, fragment
    assert fragment in fused, fragment


def test_norm_weight_is_materialised_once_when_opted_in():
  lazy_weight = Tensor(np.ones(DEPTH, np.float32), device='CPU').cast(dtypes.half)
  source = nn.RMSNorm(DEPTH, EPS)
  source.weight = lazy_weight
  with Context(ONPU_HOST_REALIZE_NORM_WEIGHTS=0):
    deferred = RockchipRMSNorm(source)
  assert deferred.weight.uop.base.op is not Ops.BUFFER
  with Context(ONPU_HOST_REALIZE_NORM_WEIGHTS=1):
    materialized = RockchipRMSNorm(source)
  assert materialized.weight.uop.base.op is Ops.BUFFER
  # Materialising the constant weight must not change its values.
  assert np.array_equal(materialized.weight.numpy(), deferred.weight.numpy())


def test_full_range_slice_and_reshape_views_still_recover_the_norm():
  """The projections slice a full row range before quantizing; that view must fuse too."""
  x, weight = _inputs('typical')
  reference = None
  with Context(**dict(_DECODE_FLAGS, ONPU_NORM_QUANT_DECODE=0)):
    normalized = rms_norm(x, weight, EPS)
    reference = quantize(normalized.reshape(1, DEPTH), DEPTH)
    reference_payload, reference_scales = reference[0].realize().numpy(), reference[1].realize().numpy()
  for view in ('reshape', 'reshape_then_full_slice', 'flatten'):
    with Context(**_DECODE_FLAGS):
      norm_quant._decode_traces = 0
      normalized = rms_norm(x, weight, EPS)
      if view == 'reshape': tensor = normalized.reshape(1, DEPTH)
      elif view == 'reshape_then_full_slice': tensor = normalized.reshape(-1, DEPTH)[0:1]
      else: tensor = normalized.reshape(1, DEPTH).flatten().reshape(1, DEPTH)
      payload, scales = quantize(tensor, DEPTH)
      payload, scales = payload.realize().numpy(), scales.realize().numpy()
    assert norm_quant._decode_traces == 1, f'{view} did not recover the registered norm'
    assert np.array_equal(reference_payload, payload), view
    assert np.array_equal(reference_scales, scales, equal_nan=True), view
